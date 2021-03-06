### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import numpy as np
import torch
import os
from torch.autograd import Variable
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks


#### with landmark + 3d ani
class Lmark2RGBModel1(BaseModel):
    def name(self):
        return 'base1'
    
    def init_loss_filter(self, use_gan_feat_loss, use_vgg_loss, use_face_loss, use_pix_loss):
        flags = (True, use_gan_feat_loss, use_vgg_loss, True, True, use_face_loss, use_pix_loss)
        def loss_filter(g_gan, g_gan_feat, g_vgg, d_real, d_fake, g_cnt, g_pix):
            return [l for (l,f) in zip((g_gan,g_gan_feat,g_vgg,d_real,d_fake, g_cnt, g_pix),flags) if f]
        return loss_filter
    
    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        self.use_lstm = opt.use_lstm
        if opt.no_ani:
            input_nc = 3 
        else:
            input_nc = 6#(lmark + ani), put is together
        ##### define networks        
        # Generator network

        # self.netG = networks.define_G(input_nc = input_nc, output_nc =opt.output_nc,netG = opt.netG, \
        #     pad_type='reflect',norm = opt.norm, ngf = opt.ngf, attention = self.attention, lstm = self.lstm,\
        #      deform = self.deform, ft = self.ft,  gpu_ids=self.gpu_ids) 

        self.netG = networks.define_G(input_nc = input_nc, output_nc =opt.output_nc,netG = opt.netG, \
            pad_type='reflect',norm = opt.norm, ngf = opt.ngf, opt= opt , gpu_ids=self.gpu_ids)             

        # Discriminator network
        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            netD_input_nc = input_nc + opt.output_nc
            self.netD = networks.define_D(netD_input_nc, opt.ndf, opt.n_layers_D, opt.norm, use_sigmoid, \
                                          opt.num_D, not opt.no_ganFeat_loss, opt=opt, gpu_ids=self.gpu_ids)
        if self.opt.verbose:
                print('---------- Networks initialized -------------')

        # load networks
        if not self.isTrain or opt.continue_train or opt.load_pretrain:
            pretrained_path = '' if not self.isTrain else opt.load_pretrain
            self.load_network(self.netG, 'G', opt.which_epoch, pretrained_path)            
            if self.isTrain:
                self.load_network(self.netD, 'D', opt.which_epoch, pretrained_path)  
           
        # set loss functions and optimizers
        if self.isTrain:
            if opt.pool_size > 0 and (len(self.gpu_ids)) > 1:
                raise NotImplementedError("Fake Pool Not Implemented for MultiGPU")
            self.fake_pool = ImagePool(opt.pool_size)
            self.old_lr = opt.lr

            # define loss functions
            self.loss_filter = self.init_loss_filter(not opt.no_ganFeat_loss, not opt.no_vgg_loss, not opt.no_face_loss , not opt.no_pixel_loss)
            
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)   
            self.criterionFeat = torch.nn.L1Loss()
            if not opt.no_vgg_loss:             
                self.criterionVGG = networks.VGGLoss(self.gpu_ids)

            if not opt.no_pixel_loss:             
                self.criterionPix = networks.PixLoss(self.gpu_ids)

            if not opt.no_face_loss:             
                self.criterionCNT = networks.LossCnt(self.opt)
                
        
            # Names so we can breakout loss
            self.loss_names = self.loss_filter('G_GAN','G_GAN_Feat','G_VGG','D_real', 'D_fake', 'G_CNT', 'G_PIX')

            # initialize optimizers
            # optimizer G
            if opt.niter_fix_global > 0:                
                import sys
                if sys.version_info >= (3,0):
                    finetune_list = set()
                else:
                    from sets import Set
                    finetune_list = Set()

                params_dict = dict(self.netG.named_parameters())
                params = []
                for key, value in params_dict.items():       
                    if key.startswith('model' + str(opt.n_local_enhancers)):                    
                        params += [value]
                        finetune_list.add(key.split('.')[0])  
                print('------------- Only training the local enhancer network (for %d epochs) ------------' % opt.niter_fix_global)
                print('The layers that are finetuned are ', sorted(finetune_list))                         
            else:
                params = list(self.netG.parameters())
                  
            # self.optimizer_G = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))                            
            if opt.use_ft and opt.ft_freeze:
                for param in self.netG.embedder.parameters():
                    param.requires_grad = False
                self.optimizer_G = torch.optim.Adam(filter(lambda p: p.requires_grad, self.netG.parameters()),  lr=opt.lr, betas=(opt.beta1, 0.999))
            else:
                self.optimizer_G = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))  
            # optimizer D                        
            params = list(self.netD.parameters())    
            self.optimizer_D = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))

    def encode_input(self, references = None, target_lmark = None, target_ani = None, real_image = None,similar_frame= None, \
        cropped_similar_img = None, infer=False):             

        # real images for training
        if real_image is not None:
            real_image = Variable(real_image.data.cuda(non_blocking=True))
        if similar_frame is not None:
            similar_frame = Variable(similar_frame.data.cuda(non_blocking=True))
        if cropped_similar_img is not None:
            cropped_similar_img = Variable(cropped_similar_img.data.cuda(non_blocking=True))
        if references is not None:
            references = Variable(references.data.cuda(non_blocking=True))
        if target_lmark is not None:
            target_lmark = Variable(target_lmark.data.cuda(non_blocking=True))
        if target_ani is not None:
            target_ani = Variable(target_ani.data.cuda(non_blocking=True))
            if self.use_lstm:
                g_in = torch.cat([target_lmark, target_ani], 2)
            else:
                g_in = torch.cat([target_lmark, target_ani], 1)
        else:
            g_in = target_lmark
        return references, target_lmark, target_ani, real_image, g_in, similar_frame, cropped_similar_img

    def discriminate(self,  g_in, test_image, use_pool=False):
        input_concat = torch.cat(( g_in, test_image.detach()), dim=1)
        if use_pool:            
            fake_query = self.fake_pool.query(input_concat)
            # if self.use_lstm:
            #     d_tmp = self.netD.forward(fake_query)
            #     d_new =[]
            #     for dd in d_tmp:
            #         dd_new = []
            #         for d in dd:
            #             d_dims = d.shape
            #             d = d.reshape( self.opt.batchSize , self.opt.lstm_length,  d_dims[1], d_dims[2], d_dims[3]  )
            #             d = d.mean(dim = 1)
            #             dd_new.append(d)
            #         d_new.append(dd_new)
            #     return d_new
            # else:
            return self.netD.forward(fake_query)
        else:
            # if self.use_lstm:
            #     d_tmp = self.netD.forward(input_concat)
            #     d_new =[]
            #     for dd in d_tmp:
            #         dd_new = []
            #         for d in dd:
            #             d_dims = d.shape
            #             d = d.reshape( self.opt.batchSize , self.opt.lstm_length,  d_dims[1], d_dims[2], d_dims[3]  )
            #             d = d.mean(dim = 1)
            #             dd_new.append(d)
            #         d_new.append(dd_new)
            #     return d_new
            # else:
            return self.netD.forward(input_concat)

    def forward(self, references, target_lmark, target_ani, real_image, similar_frame,cropped_similar_img, infer=False):
        # Encode Inputs
        references, target_lmark, target_ani, real_image , g_in , similar_frame, cropped_similar_img= \
        self.encode_input(references = references, target_lmark = target_lmark, target_ani = target_ani, \
        real_image = real_image,similar_frame = similar_frame,cropped_similar_img = cropped_similar_img,infer= infer)  

        # Fake Generation
        fake_list = self.netG.forward(references, g_in , similar_frame, cropped_similar_img)
        # if self.attention:
        fake_image = fake_list[0]
        # Fake Detection and Loss

        ##### if it is using lstm, currently, my discriminator does not support lstm operation, we reshape to bactch size
        g_dims = g_in.shape
        img_dims = fake_image.shape

        if self.use_lstm:
            g_in = g_in.reshape(g_dims[0] * g_dims[1], g_dims[2], g_dims[3], g_dims[4])

            fake_image = fake_image.reshape(img_dims[0] * img_dims[1], img_dims[2], img_dims[3], img_dims[4])
            real_image = real_image.reshape(img_dims[0] * img_dims[1], img_dims[2], img_dims[3], img_dims[4])

        pred_fake_pool = self.discriminate( g_in, fake_image, use_pool=True)
        loss_D_fake = self.criterionGAN(pred_fake_pool, False)        

        # Real Detection and Loss        
        pred_real = self.discriminate( g_in, real_image)
        loss_D_real = self.criterionGAN(pred_real, True)

        # GAN loss (Fake Passability Loss)
        pred_fake = self.netD.forward(torch.cat(( g_in, fake_image), dim=1))         
        # if self.use_lstm:
        #     tmp_gg = []
        #     for gg in pred_fake:
        #         tmp_g = []
        #         for g in gg:
        #             g_dimes =  g.shape
        #             g = g.reshape(img_dims[0], img_dims[1], g_dimes[1], g_dimes[2], g_dimes[3] )
        #             g  = g.mean(dim = 1)
        #             tmp_g.append(g)
        #         tmp_gg.append(tmp_g)
        #     pred_fake = tmp_gg
                   
        loss_G_GAN = self.criterionGAN(pred_fake, True)               
        
        # GAN feature matching loss
        loss_G_GAN_Feat = 0
        if not self.opt.no_ganFeat_loss:
            feat_weights = 4.0 / (self.opt.n_layers_D + 1)
            D_weights = 1.0 / self.opt.num_D
            for i in range(self.opt.num_D):
                for j in range(len(pred_fake[i])-1):
                    loss_G_GAN_Feat += D_weights * feat_weights * \
                        self.criterionFeat(pred_fake[i][j], pred_real[i][j].detach()) * self.opt.lambda_feat
                   
        # VGG feature matching loss
        loss_G_VGG = 0
        if not self.opt.no_vgg_loss:
            loss_G_VGG = self.criterionVGG(fake_image, real_image) * self.opt.lambda_feat
        loss_G_CNT = 0
        if not self.opt.no_face_loss:
            loss_G_CNT = self.criterionCNT(real_image, fake_image) 

        loss_G_PIX= 0
        if not self.opt.no_pixel_loss:
            loss_G_PIX = self.criterionPix(real_image, fake_image) 
        
        # Only return the fake_B image if necessary to save BW
        return [ self.loss_filter( loss_G_GAN, loss_G_GAN_Feat, loss_G_VGG, loss_D_real, loss_D_fake, loss_G_CNT, loss_G_PIX ), None if not infer else fake_list ]

    def inference(self, references, target_lmark, target_ani, real_image, similar_frame, cropped_similar_img):
        # Encode Inputs        
        real_image = Variable(real_image) if real_image is not None else None

        similar_frame = Variable(similar_frame) if real_image is not None else None
        references, target_lmark, target_ani, real_image , g_in, similar_frame, cropped_similar_img = self.encode_input(references, \
        target_lmark, target_ani, real_image , similar_frame,cropped_similar_img,  infer=True)

        # Fake Generation           
        if torch.__version__.startswith('0.4'):
            with torch.no_grad():
                fake_list  = self.netG.forward(references, g_in, similar_frame, cropped_similar_img)
        else:
            fake_list = self.netG.forward(references, g_in, similar_frame, cropped_similar_img)
        return fake_list


    def save(self, which_epoch):
        self.save_network(self.netG, 'G', which_epoch, self.gpu_ids)
        self.save_network(self.netD, 'D', which_epoch, self.gpu_ids)
       

    def update_fixed_params(self):
        # after fixing the global generator for a number of iterations, also start finetuning it
        params = list(self.netG.parameters())
        if self.gen_features:
            params += list(self.netE.parameters())           
        self.optimizer_G = torch.optim.Adam(params, lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
        if self.opt.verbose:
            print('------------ Now also finetuning global generator -----------')

    def update_learning_rate(self):
        lrd = self.opt.lr / self.opt.niter_decay
        lr = self.old_lr - lrd        
        for param_group in self.optimizer_D.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr
        if self.opt.verbose:
            print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr

class InferenceModel1(Lmark2RGBModel1):
    def forward(self, inp):
        references, target_lmark, target_ani, image, similar_frame, cropped_similar_img = inp
        return self.inference(references, target_lmark, target_ani, image, similar_frame, cropped_similar_img)
