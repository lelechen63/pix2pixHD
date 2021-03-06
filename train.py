### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import time
from collections import OrderedDict
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
from util.visualizer import Visualizer
import os
import numpy as np
import torch
from torch.autograd import Variable

opt = TrainOptions().parse()
iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')
if opt.continue_train:
    try:
        start_epoch, epoch_iter = np.loadtxt(iter_path , delimiter=',', dtype=int)
    except:
        start_epoch, epoch_iter = 1, 0
    print('Resuming from epoch %d at iteration %d' % (start_epoch, epoch_iter))        
else:    
    start_epoch, epoch_iter = 1, 0

if opt.debug:
    opt.display_freq = 1
    opt.print_freq = 1
    opt.niter = 1
    opt.niter_decay = 0
    opt.max_dataset_size = 10
 
data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
print('#training images = %d' % dataset_size)

model = create_model(opt)
visualizer = Visualizer(opt)

total_steps = (start_epoch-1) * dataset_size + epoch_iter

display_delta = total_steps % opt.display_freq
print_delta = total_steps % opt.print_freq
save_delta = total_steps % opt.save_latest_freq
##############
with torch.autograd.set_detect_anomaly(False):
    for epoch in range(start_epoch, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        if epoch != start_epoch:
            epoch_iter = epoch_iter % dataset_size
        for i, data in enumerate(dataset, start=epoch_iter):
            iter_start_time = time.time()
            total_steps += opt.batchSize
            epoch_iter += opt.batchSize

            # whether to collect output images
            save_fake = total_steps % opt.display_freq == display_delta

            ############## Forward Pass ######################
            if opt.no_ani:
                losses, generated = model(references =Variable(data['reference_frames']),target_lmark= Variable(data['target_lmark']), \
                target_ani=  None, real_image=  Variable(data['target_rgb']), similar_frame =Variable( data['similar_frame']), \
                cropped_similar_img = Variable(data['cropped_similar_image'] ), infer=save_fake)
            else:
                losses, generated = model(references =Variable(data['reference_frames']),target_lmark= Variable(data['target_lmark']), \
                target_ani= Variable(data['target_ani']),real_image=  Variable(data['target_rgb']), similar_frame = Variable(data['similar_frame']), \
                cropped_similar_img = Variable(data['cropped_similar_image']) , infer=save_fake)
            # sum per device losses
            losses = [ torch.mean(x) if not isinstance(x, int) else x for x in losses ]
            loss_dict = dict(zip(model.module.loss_names, losses))

            # calculate final loss scalar
            loss_D = (loss_dict['D_fake'] + loss_dict['D_real']) * 0.5
            loss_G = loss_dict['G_GAN'] + loss_dict.get('G_GAN_Feat',0) + loss_dict.get('G_VGG',0) +  loss_dict.get('G_CNT',0) + 0.001 * loss_dict.get('G_PIX',0)

            ############### Backward Pass ####################
            # update generator weights
            model.module.optimizer_G.zero_grad()
            loss_G.backward()
            model.module.optimizer_G.step()

            # update discriminator weights
            model.module.optimizer_D.zero_grad()
            loss_D.backward()
            model.module.optimizer_D.step()

               ############## Display results and errors ##########
            ### print out errors
            # print   (loss_dict['D_fake'], loss_dict['D_real'],  loss_dict['G_GAN'],  loss_dict.get('G_GAN_Feat',0),  loss_dict.get('G_VGG',0)) 
            errors = {}
            if total_steps % opt.print_freq == print_delta:
                for k, v in loss_dict.items():
                    # print (k,v)
                    errors[k] = v.item()
                # errors = {k: v.data[0] if not isinstance(v, int) else v for k, v in loss_dict.items()}
                t = (time.time() - iter_start_time) / opt.batchSize
                visualizer.print_current_errors(epoch, epoch_iter, errors, t)
                visualizer.plot_current_errors(errors, total_steps)

                ### display output images
                if not opt.use_lstm:
                        
                    tmp = []
                    tmp.extend([( 'reference1', util.tensor2im(data['reference_frames'][0, 0,:3]))])
                    if opt.num_frames >= 4:
                        tmp.extend([('reference2', util.tensor2im(data['reference_frames'][0, 1,:3])),
                                            ('reference3', util.tensor2im(data['reference_frames'][0, 2,:3])),
                                            ('reference4', util.tensor2im(data['reference_frames'][0, 3,:3]))])
                    tmp.extend([('target_lmark', util.tensor2im(data['target_lmark'][0])),
                                        ('target_ani', util.tensor2im(data['target_ani'][0])),
                                        ('synthesized_image', util.tensor2im(generated[0].data[0])),
                                        ('real_image', util.tensor2im(data['target_rgb'][0]))])
                    if not opt.no_att:
                        tmp.extend([('masked_similar_img', util.tensor2im(generated[1].data[0])),
                                        ('face_foreground', util.tensor2im(generated[2].data[0])),
                                        ('beta', util.tensor2im(generated[3].data[0])),
                                        ('alpha', util.tensor2im(generated[4].data[0])),
                                        ('I_hat', util.tensor2im(generated[5].data[0]))])
                    
                else:
                    tmp = []
                    tmp.extend([( 'reference1', util.tensor2im(data['reference_frames'][0, 0,:3]))])
                    if opt.num_frames >= 4:
                        tmp.extend([('reference2', util.tensor2im(data['reference_frames'][0, 1,:3])),
                                            ('reference3', util.tensor2im(data['reference_frames'][0, 2,:3])),
                                            ('reference4', util.tensor2im(data['reference_frames'][0, 3,:3]))])
                    tmp.extend([('target_lmark', util.tensor2im(data['target_lmark'][0,0])),
                                                ('target_ani', util.tensor2im(data['target_ani'][0,0])),
                                                ('synthesized_image', util.tensor2im(generated[0].data[0,0])),
                                                ('real_image', util.tensor2im(data['target_rgb'][0,0]))])
                    if not opt.no_att:
                        tmp.extend([('masked_similar_img', util.tensor2im(generated[1].data[0,0])),
                                                ('face_foreground', util.tensor2im(generated[2].data[0,0])),
                                                ('beta', util.tensor2im(generated[3].data[0,0])),
                                                ('alpha', util.tensor2im(generated[4].data[0,0])),
                                                ('I_hat', util.tensor2im(generated[5].data[0,0]))])
                visuals =  OrderedDict(tmp)  
                visualizer.display_current_results(visuals, epoch, total_steps)
                
            ### save latest model
            if total_steps % opt.save_latest_freq == save_delta:
                print('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_steps))
                model.module.save('latest')            
                np.savetxt(iter_path, (epoch, epoch_iter), delimiter=',', fmt='%d')

            if epoch_iter >= dataset_size:
                break
        
        # end of epoch 
        iter_end_time = time.time()
        print('End of epoch %d / %d \t Time Taken: %d sec' %
            (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

        ### save model for this epoch
        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))        
            model.module.save('latest')
            model.module.save(epoch)
            np.savetxt(iter_path, (epoch+1, 0), delimiter=',', fmt='%d')

        ### instead of only training the local enhancer, train the entire network after certain iterations
        if (opt.niter_fix_global != 0) and (epoch == opt.niter_fix_global):
            model.module.update_fixed_params()

        ### linearly decay learning rate after certain iterations
        if epoch > opt.niter:
            model.module.update_learning_rate()
