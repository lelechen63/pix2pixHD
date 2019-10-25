### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import torch
import torch.nn as nn
import functools
from torch.autograd import Variable
import numpy as np
from models.components import ResidualBlock, AdaptiveResidualBlock, ResidualBlockDown, AdaptiveResidualBlockUp, SelfAttention
from models.blocks import LinearBlock, Conv2dBlock, ResBlocks, ActFirstResBlock
from torch.nn import functional as F
import os
import imp
from models.vgg import Cropped_VGG19
from models.def_conv.modules.deform_conv import DeformConv

###############################################################################
# Functions
###############################################################################
def weights_init(m):
    classname = m.__class__.__name__
    if (classname.find('Conv') == 0 or classname.find(
                'Linear') == 0) and hasattr(m, 'weight'):
        m.weight.data.normal_(0.0, 0.02)      
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

def define_G(input_nc, output_nc, netG, pad_type,  norm='instance',ngf= 64, attention = True, lstm = False,deform = False, gpu_ids=[]):    
    norm_layer = get_norm_layer(norm_type=norm)     
    if netG == 'global':
        if lstm == False:    
            netG = GlobalGenerator( input_nc, output_nc, pad_type, norm_layer,ngf,attention, deform)       
        else:
            netG = GlobalGenerator_lstm( input_nc, output_nc, pad_type, norm_layer,ngf,attention,deform)
    elif netG == 'local':        
        netG = LocalEnhancer(input_nc, output_nc, ngf, n_downsample_global, n_blocks_global, 
                                  n_local_enhancers, n_blocks_local, norm_layer)
    elif netG == 'encoder':
        netG = Encoder(input_nc, output_nc, ngf, n_downsample_global, norm_layer)
    else:
        raise('generator not implemented!')
    print(netG)
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())   
        netG.cuda(gpu_ids[0])
    netG.apply(weights_init)
    return netG

def define_D(input_nc, ndf, n_layers_D, norm='instance', use_sigmoid=False, num_D=1, getIntermFeat=False, gpu_ids=[]):        
    norm_layer = get_norm_layer(norm_type=norm)   
    netD = MultiscaleDiscriminator(input_nc, ndf, n_layers_D, norm_layer, use_sigmoid, num_D, getIntermFeat)   
    print(netD)
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        netD.cuda(gpu_ids[0])
    netD.apply(weights_init)
    return netD

def print_network(net):
    if isinstance(net, list):
        net = net[0]
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)

##############################################################################
# Losses
##############################################################################
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        if isinstance(input[0], list):
            loss = 0
            for input_i in input:
                pred = input_i[-1]
                target_tensor = self.get_target_tensor(pred, target_is_real)
                loss += self.loss(pred, target_tensor)
            return loss
        else:            
            target_tensor = self.get_target_tensor(input[-1], target_is_real)
            return self.loss(input[-1], target_tensor)

class VGGLoss(nn.Module):
    def __init__(self, gpu_ids):
        super(VGGLoss, self).__init__()        
        self.vgg = Vgg19().cuda()
        self.criterion = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]        

    def forward(self, x, y):              
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())        
        return loss

class PixLoss(nn.Module):
    def __init__(self, gpu_ids):
        super(PixLoss, self).__init__()        
        self.criterion = nn.L1Loss()

    def forward(self, x, y):              
        loss =   self.criterion(x, y.detach())        
        return loss
##############################################################################
# Generator
##############################################################################


class Embedder(nn.Module):
    """
    The Embedder network attempts to generate a vector that encodes the personal characteristics of an individual given
    a head-shot and the matching landmarks.
    """
    def __init__(self, input_nc = 6, ngf = 64, norm_layer = nn.InstanceNorm2d, pad_type = 'reflect' ):
        super(Embedder, self).__init__()
        activ = 'relu'
        self.model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf), nn.ReLU(True)]
        self.model += [Conv2dBlock(64, 128, 4, 2, 1,           # 128, 128, 128 
                                       norm= 'in',
                                       activation=activ,
                                       pad_type=pad_type)]

        self.model += [Conv2dBlock(128, 128, 4, 2, 1,           # 128, 64 
                                       norm= 'in',
                                       activation=activ,
                                       pad_type=pad_type)]
        self.model += [Conv2dBlock(128, 256, 4, 2, 1,           # 256 32 
                                       norm= 'in',
                                       activation=activ,
                                       pad_type=pad_type)]

        self.model += [Conv2dBlock(256, 256, 4, 2, 1,           # 256 16
                                       norm= 'in',
                                       activation=activ,
                                       pad_type=pad_type)]
        self.model += [Conv2dBlock(256, 512, 4, 2, 1,           # 512 8
                                       norm= 'in',
                                       activation=activ,
                                       pad_type=pad_type)]

        self.model += [Conv2dBlock(512, 512, 4, 2, 1,           # 512 4
                                       norm= 'in',
                                       activation=activ,
                                       pad_type=pad_type)]
        self.encoder = nn.Sequential(*self.model)
        self.pooling = nn.AdaptiveMaxPool2d((1, 1))
        # self.apply(weights_init)
    
    def forward(self, x):   #(x: reference image and landmark)
        # Encode
        out = self.encoder(x)
        # Vectorize
        out = F.relu(self.pooling(out).view(-1, 512))
        return out

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, dim, n_blk, norm, activ):

        super(MLP, self).__init__()
        self.model = []
        self.model += [LinearBlock(in_dim, dim, norm=norm, activation=activ)]
        for i in range(n_blk - 2):
            self.model += [LinearBlock(dim, dim, norm=norm, activation=activ)]
        self.model += [LinearBlock(dim, out_dim,
                                   norm='none', activation='none')]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x.view(x.size(0), -1))


def assign_adain_params(adain_params, model):
    # assign the adain_params to the AdaIN layers in model
    for m in model.modules():
        if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
            mean = adain_params[:, :m.num_features]
            std = adain_params[:, m.num_features:2*m.num_features]
            m.bias = mean.contiguous().view(-1)
            m.weight = std.contiguous().view(-1)
            if adain_params.size(1) > 2*m.num_features:
                adain_params = adain_params[:, 2*m.num_features:]


def get_num_adain_params(model):
    # return the number of AdaIN parameters needed by the model
    num_adain_params = 0
    for m in model.modules():
        if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
            num_adain_params += 2*m.num_features
    return num_adain_params
class GlobalGenerator(nn.Module):
    def __init__(self,input_nc, output_nc, pad_type='reflect', norm_layer=nn.BatchNorm2d, ngf = 64, attention = True, deform= False ):
        super(GlobalGenerator, self).__init__()        
        activ = 'relu'    
        self.deform = deform
        self.attention = attention
        model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf), nn.ReLU(True) ]
        ### downsample
        model += [Conv2dBlock(64, 128, 4, 2, 1,           # 128, 128, 128 
                                       norm= 'in',
                                       activation=activ,
                                       pad_type=pad_type)]

        model += [Conv2dBlock(128, 128, 4, 2, 1,           # 128, 64 
                                       norm= 'in',
                                       activation=activ,
                                       pad_type=pad_type)]
        model += [Conv2dBlock(128, 256, 4, 2, 1,           # 256 32 
                                       norm= 'in',
                                       activation=activ,
                                       pad_type=pad_type)]

        model += [Conv2dBlock(256, 256, 4, 2, 1,           # 256 16
                                       norm= 'in',
                                       activation=activ,
                                       pad_type=pad_type)]
        model += [Conv2dBlock(256, 512, 4, 2, 1,           # 512 8
                                       norm= 'in',
                                       activation=activ,
                                       pad_type=pad_type)]

        model += [Conv2dBlock(512, 512, 4, 2, 1,           # 512 4
                                       norm= 'in',
                                       activation=activ,
                                       pad_type=pad_type)]


        self.lmark_ani_encoder = nn.Sequential(*model)
        model = []
        ###  adain resnet blocks
        model += [ResBlocks(2, 512, norm  = 'adain', activation=activ, pad_type='reflect')]

        ### upsample         
        model += [nn.Upsample(scale_factor=2),
                        Conv2dBlock(512, 512, 5, 1, 2,
                                    norm='in',
                                    activation=activ,
                                    pad_type=pad_type)]    # 512, 8 , 8 
        model += [nn.Upsample(scale_factor=2),
                        Conv2dBlock(512, 512, 5, 1, 2,
                                    norm='in',
                                    activation=activ,
                                    pad_type=pad_type)] # 512, 16 , 16 
        model += [nn.Upsample(scale_factor=2),
                        Conv2dBlock(512, 256, 5, 1, 2,
                                    norm='in',
                                    activation=activ,
                                    pad_type=pad_type)] # 256, 32, 32 
        model += [nn.Upsample(scale_factor=2),
                        Conv2dBlock(256, 256, 5, 1, 2,
                                    norm='in',
                                    activation=activ,
                                    pad_type=pad_type)] # 256, 64, 64 
        model += [nn.Upsample(scale_factor=2), 
                        Conv2dBlock(256, 128, 5, 1, 2,
                                    norm='in',
                                    activation=activ,
                                    pad_type=pad_type)]  # 128, 128, 128 
        model += [nn.Upsample(scale_factor=2), 
                        Conv2dBlock(128, 64, 5, 1, 2,
                                    norm='in',
                                    activation=activ,
                                    pad_type=pad_type)]  # 64, 256, 256 
        if not self.attention:
            model += [Conv2dBlock(64, 3, 7, 1, 3,
                                   norm='none',
                                   activation='tanh',
                                   pad_type=pad_type)]

            self.decoder = nn.Sequential(*model)
        else:
            self.decoder = nn.Sequential(*model)

        self.alpha_conv = Conv2dBlock(64, 1, 7, 1, 3,
                                   norm='none',
                                   activation='sigmoid',
                                   pad_type=pad_type)

        

        self.rgb_conv = Conv2dBlock(64, 3, 7, 1, 3,
                                   norm='none',
                                   activation='tanh',
                                   pad_type=pad_type)
        

        

        self.embedder = Embedder()


        self.mlp = MLP(512,
                       get_num_adain_params(self.decoder),
                       256,
                       3,
                       norm='none',
                       activ='relu')
        if not self.deform:
            model = [nn.ReflectionPad2d(3), nn.Conv2d(6, 32, kernel_size=7, padding=0), norm_layer(32), nn.ReLU(True) ]
            ### downsample
            model += [Conv2dBlock(32, 64, 4, 2, 1,           # 128, 128, 128 
                                        norm= 'in',
                                        activation=activ,
                                        pad_type=pad_type)]

            model += [nn.ConvTranspose2d(64, 64,kernel_size=4, stride=(2),padding=(1)),
                        nn.InstanceNorm2d(64),
                        nn.ReLU(True)
            ]
            self.foregroundNet = nn.Sequential(*model)
            
        else:

            self.off2d_1 = nn.Sequential(*[ nn.ReflectionPad2d(3), nn.Conv2d(6, 18 * 8, kernel_size=7, stride =1, padding=0), nn.InstanceNorm2d(18 * 8), nn.ReLU(True)])

            self.def_conv_1 = DeformConv(6, 64, 3,stride =1, padding =1, deformable_groups= 8)
            self.def_conv_1_norm = nn.Sequential(*[  nn.InstanceNorm2d(64), nn.ReLU(True)])

            self.off2d_2 = nn.Sequential(*[  nn.Conv2d(64, 18 * 8, kernel_size=3, stride =1, padding=1), nn.InstanceNorm2d(18 * 8), nn.ReLU(True)])

            self.def_conv_2 = DeformConv(6, 128, 3,stride =1, padding =1, deformable_groups= 8)
            self.def_conv_2_norm =  nn.Sequential(*[  nn.InstanceNorm2d(128), nn.ReLU(True)])

            self.off2d_3 = nn.Sequential(*[  nn.Conv2d(128, 18 * 8, kernel_size=3, stride =1,padding=1), nn.InstanceNorm2d(18 * 8), nn.ReLU(True)])

            self.def_conv_3 = DeformConv( 128, 64, 3,stride =1, padding =1, deformable_groups= 8)
            self.def_conv3_norm = nn.Sequential(*[  nn.InstanceNorm2d(64), nn.ReLU(True)])

        self.beta  = Conv2dBlock(128, 1, 7, 1, 3,
                                    norm='none',
                                    activation='sigmoid',
                                    pad_type=pad_type)



    def forward(self, references, g_in, similar_img):
        dims = references.shape

        references = references.reshape( dims[0] * dims[1], dims[2], dims[3], dims[4]  )
        e_vectors = self.embedder(references).reshape(dims[0] , dims[1], -1)
        e_hat = e_vectors.mean(dim = 1)


        feature = self.lmark_ani_encoder(g_in)

        # Decode
        adain_params = self.mlp(e_hat)
        assign_adain_params(adain_params, self.decoder)
        if not self.attention:
            return [self.decoder(feature)]
        I_feature = self.decoder(feature)

        I_hat = self.rgb_conv(I_feature)
        

        
        
        ani_img = g_in[:,3:,:,:]
        similar_img = similar_img[:,:3]
        alpha = self.alpha_conv(I_feature)

        face_foreground = (1 - alpha) * ani_img + alpha * I_hat
        if not self.deform:
            foreground_feature = self.foregroundNet( torch.cat([ani_img, similar_img], 1) )  

            # foreground_feature = self.foregroundNet( ani_img)  # should be torch.cat([ani_img, similar_img]) and change foreground to 6 channel input

            forMask_feature = torch.cat([foreground_feature, I_feature ], 1)

            

        else:
            feature = torch.cat([ani_img, similar_img], 1)
            offset_1 = self.off2d_1(feature)

            fea = self.def_conv_1(feature, offset_1)
            fea = self.def_conv_1_norm(fea)

            offset_2 = self.off2d_2(fea)
            #######################################

            fea = self.def_conv_2(ani_img, offset_2)

            fea = self.def_conv_2_norm(fea)

            off2d_3 = self.off2d_3(fea)


             
            fea = self.def_conv_3(fea, off2d_3)
            forMask_feature = self.def_conv3_norm(fea)


        beta = self.beta(forMask_feature)




        similar_img[ani_img> -0.9] = -1 #  = similar_img * mask

        image = (1- beta) * similar_img + beta * face_foreground
        

        return [image, similar_img, face_foreground, beta, alpha, I_hat]


class GlobalGenerator_lstm(nn.Module):
    def __init__(self,input_nc, output_nc, pad_type='reflect', norm_layer=nn.BatchNorm2d, ngf = 64):
        super(GlobalGenerator_lstm, self).__init__()        
        activ = 'relu'    
        model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf), nn.ReLU(True) ]
        ### downsample
        model += [Conv2dBlock(64, 128, 4, 2, 1,           # 128, 128, 128 
                                       norm= 'in',
                                       activation=activ,
                                       pad_type=pad_type)]

        model += [Conv2dBlock(128, 128, 4, 2, 1,           # 128, 64 
                                       norm= 'in',
                                       activation=activ,
                                       pad_type=pad_type)]
        model += [Conv2dBlock(128, 256, 4, 2, 1,           # 256 32 
                                       norm= 'in',
                                       activation=activ,
                                       pad_type=pad_type)]

        model += [Conv2dBlock(256, 256, 4, 2, 1,           # 256 16
                                       norm= 'in',
                                       activation=activ,
                                       pad_type=pad_type)]
        model += [Conv2dBlock(256, 512, 4, 2, 1,           # 512 8
                                       norm= 'in',
                                       activation=activ,
                                       pad_type=pad_type)]

        model += [Conv2dBlock(512, 512, 4, 2, 1,           # 512 4
                                       norm= 'in',
                                       activation=activ,
                                       pad_type=pad_type)]


        self.lmark_ani_encoder = nn.Sequential(*model)
        model = []
        ###  adain resnet blocks
        model += [ResBlocks(2, 512, norm  = 'adain', activation=activ, pad_type='reflect')]

        ### upsample         
        model += [nn.Upsample(scale_factor=2),
                        Conv2dBlock(512, 512, 5, 1, 2,
                                    norm='in',
                                    activation=activ,
                                    pad_type=pad_type)]    # 512, 8 , 8 
        model += [nn.Upsample(scale_factor=2),
                        Conv2dBlock(512, 512, 5, 1, 2,
                                    norm='in',
                                    activation=activ,
                                    pad_type=pad_type)] # 512, 16 , 16 
        model += [nn.Upsample(scale_factor=2),
                        Conv2dBlock(512, 256, 5, 1, 2,
                                    norm='in',
                                    activation=activ,
                                    pad_type=pad_type)] # 256, 32, 32 
        model += [nn.Upsample(scale_factor=2),
                        Conv2dBlock(256, 256, 5, 1, 2,
                                    norm='in',
                                    activation=activ,
                                    pad_type=pad_type)] # 256, 64, 64 
        model += [nn.Upsample(scale_factor=2), 
                        Conv2dBlock(256, 128, 5, 1, 2,
                                    norm='in',
                                    activation=activ,
                                    pad_type=pad_type)]  # 128, 128, 128 
        model += [nn.Upsample(scale_factor=2), 
                        Conv2dBlock(128, 64, 5, 1, 2,
                                    norm='in',
                                    activation=activ,
                                    pad_type=pad_type)]  # 64, 256, 256 
        model += [Conv2dBlock(64, 3, 7, 1, 3,
                                   norm='none',
                                   activation='tanh',
                                   pad_type=pad_type)]
        self.decoder = nn.Sequential(*model)


        self.embedder = Embedder()


        self.mlp = MLP(512,
                       get_num_adain_params(self.decoder),
                       256,
                       3,
                       norm='none',
                       activ='relu')


    def forward(self, references, g_in):
        dims = references.shape
        references = references.reshape( dims[0] * dims[1], dims[2], dims[3], dims[4]  )
        e_vectors = self.embedder(references).reshape(dims[0] , dims[1], -1)
        e_hat = e_vectors.mean(dim = 1)

        

        feature = self.lmark_ani_encoder(g_in)

        # Decode
        adain_params = self.mlp(e_hat)
        assign_adain_params(adain_params, self.decoder)
        image = self.decoder(feature)
        return image


class GlobalGenerator_mfcc(nn.Module):
    def __init__(self,output_nc, pad_type='reflect', norm_layer=nn.BatchNorm2d, ngf = 64):
        super(GlobalGenerator_mfcc, self).__init__()        
        activ = 'relu'    
        model = [nn.ReflectionPad2d(3), nn.Conv2d(6, ngf, kernel_size=7, padding=0), norm_layer(ngf), nn.ReLU(True) ]
        ### downsample
        model += [Conv2dBlock(64, 128, 4, 2, 1,           # 128, 128, 128 
                                       norm= 'in',
                                       activation=activ,
                                       pad_type=pad_type)]

        model += [Conv2dBlock(128, 128, 4, 2, 1,           # 128, 64 
                                       norm= 'in',
                                       activation=activ,
                                       pad_type=pad_type)]
        model += [Conv2dBlock(128, 256, 4, 2, 1,           # 256 32 
                                       norm= 'in',
                                       activation=activ,
                                       pad_type=pad_type)]

        model += [Conv2dBlock(256, 256, 4, 2, 1,           # 256 16
                                       norm= 'in',
                                       activation=activ,
                                       pad_type=pad_type)]
        model += [Conv2dBlock(256, 512, 4, 2, 1,           # 512 8
                                       norm= 'in',
                                       activation=activ,
                                       pad_type=pad_type)]

        model += [Conv2dBlock(512, 512, 4, 2, 1,           # 512 4
                                       norm= 'in',
                                       activation=activ,
                                       pad_type=pad_type)]


        self.lmark_ani_encoder = nn.Sequential(*model)
        model = []
        ###  adain resnet blocks
        model += [ResBlocks(2, 512, norm  = 'adain', activation=activ, pad_type='reflect')]

        ### upsample         
        model += [nn.Upsample(scale_factor=2),
                        Conv2dBlock(512, 512, 5, 1, 2,
                                    norm='in',
                                    activation=activ,
                                    pad_type=pad_type)]    # 512, 8 , 8 
        model += [nn.Upsample(scale_factor=2),
                        Conv2dBlock(512, 512, 5, 1, 2,
                                    norm='in',
                                    activation=activ,
                                    pad_type=pad_type)] # 512, 16 , 16 
        model += [nn.Upsample(scale_factor=2),
                        Conv2dBlock(512, 256, 5, 1, 2,
                                    norm='in',
                                    activation=activ,
                                    pad_type=pad_type)] # 256, 32, 32 
        model += [nn.Upsample(scale_factor=2),
                        Conv2dBlock(256, 256, 5, 1, 2,
                                    norm='in',
                                    activation=activ,
                                    pad_type=pad_type)] # 256, 64, 64 
        model += [nn.Upsample(scale_factor=2), 
                        Conv2dBlock(256, 128, 5, 1, 2,
                                    norm='in',
                                    activation=activ,
                                    pad_type=pad_type)]  # 128, 128, 128 
        model += [nn.Upsample(scale_factor=2), 
                        Conv2dBlock(128, 64, 5, 1, 2,
                                    norm='in',
                                    activation=activ,
                                    pad_type=pad_type)]  # 64, 256, 256 
        model += [Conv2dBlock(64, 3, 7, 1, 3,
                                   norm='none',
                                   activation='tanh',
                                   pad_type=pad_type)]
        self.decoder = nn.Sequential(*model)


        self.embedder = Embedder()


        self.mlp = MLP(512,
                       get_num_adain_params(self.decoder),
                       256,
                       3,
                       norm='none',
                       activ='relu')


    def forward(self, references, target_lmark, target_ani, mfcc):
        dims = references.shape
        references = references.reshape( dims[0] * dims[1], dims[2], dims[3], dims[4]  )
        e_vectors = self.embedder(references).reshape(dims[0] , dims[1], -1)
        e_hat = e_vectors.mean(dim = 1)

        g_in = torch.cat([target_lmark, target_ani], 1)

        feature = self.lmark_ani_encoder(g_in)

        # Decode
        adain_params = self.mlp(e_hat)
        assign_adain_params(adain_params, self.decoder)
        image = self.decoder(feature)
        return image


class MultiscaleDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, 
                 use_sigmoid=False, num_D=3, getIntermFeat=False):
        super(MultiscaleDiscriminator, self).__init__()
        self.num_D = num_D
        self.n_layers = n_layers
        self.getIntermFeat = getIntermFeat
     
        for i in range(num_D):
            netD = NLayerDiscriminator(input_nc, ndf, n_layers, norm_layer, use_sigmoid, getIntermFeat)
            if getIntermFeat:                                
                for j in range(n_layers+2):
                    setattr(self, 'scale'+str(i)+'_layer'+str(j), getattr(netD, 'model'+str(j)))                                   
            else:
                setattr(self, 'layer'+str(i), netD.model)

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def singleD_forward(self, model, input):
        if self.getIntermFeat:
            result = [input]
            for i in range(len(model)):
                result.append(model[i](result[-1]))
            return result[1:]
        else:
            return [model(input)]

    def forward(self, input):        
        num_D = self.num_D
        result = []
        input_downsampled = input
        for i in range(num_D):
            if self.getIntermFeat:
                model = [getattr(self, 'scale'+str(num_D-1-i)+'_layer'+str(j)) for j in range(self.n_layers+2)]
            else:
                model = getattr(self, 'layer'+str(num_D-1-i))
            result.append(self.singleD_forward(model, input_downsampled))
            if i != (num_D-1):
                input_downsampled = self.downsample(input_downsampled)
        return result
        
# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, getIntermFeat=False):
        super(NLayerDiscriminator, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers

        kw = 4
        padw = int(np.ceil((kw-1.0)/2))
        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                norm_layer(nf), nn.LeakyReLU(0.2, True)
            ]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            norm_layer(nf),
            nn.LeakyReLU(0.2, True)
        ]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model'+str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, input):
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers+2):
                model = getattr(self, 'model'+str(n))
                res.append(model(res[-1]))
            return res[1:]
        else:
            return self.model(input)        

from torchvision import models
class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)        
        h_relu3 = self.slice3(h_relu2)        
        h_relu4 = self.slice4(h_relu3)        
        h_relu5 = self.slice5(h_relu4)                
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


class LossCnt(nn.Module):
    def __init__(self, opt):
        super(LossCnt, self).__init__()
        root = opt.dataroot        
        VGGFace_body_path = os.path.join(root, 'vggface' , 'Pytorch_VGGFACE_IR.py')
        VGGFace_weight_path = os.path.join(root, 'vggface' , 'Pytorch_VGGFACE.pth')
        MainModel = imp.load_source('MainModel', VGGFace_body_path)
        full_VGGFace = torch.load(VGGFace_weight_path, map_location = 'cpu')
        cropped_VGGFace = Cropped_VGG19()
        cropped_VGGFace.load_state_dict(full_VGGFace.state_dict(), strict = False)
        self.VGGFace = cropped_VGGFace
        self.VGGFace.eval()

    def forward(self, x, x_hat,  vggface_weight=1.0):
        # print (x.shape)
        # print (x_hat.shape)
        # print ('===========')
        l1_loss = nn.L1Loss()

        """Retrieve vggface feature maps"""
        with torch.no_grad(): #no need for gradient compute
            vgg_x_features = self.VGGFace(x) #returns a list of feature maps at desired layers

            vgg_xhat_features = self.VGGFace(x_hat.detach())

        lossface = []
        for x_feat, xhat_feat in zip(vgg_x_features, vgg_xhat_features):
            lossface.append(l1_loss(x_feat, xhat_feat))
        loss =vggface_weight *  sum(lossface) # vgg19_weight * loss19 + 

        return loss