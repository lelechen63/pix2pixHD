### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import os
from collections import OrderedDict
from torch.autograd import Variable
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
from util.visualizer import Visualizer
from util import html
import torch

opt = TestOptions().parse(save=False)
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
visualizer = Visualizer(opt)
# create website
web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))

# test
model = create_model(opt)
model.eval() 
if opt.verbose:
    print(model)
with torch.no_grad():
    for i, data in enumerate(dataset):
        if i >= opt.how_many:
            break
        minibatch = 1 
        if opt.no_att:
            similar_img = None
        else:
            similar_img = Variable(data['similar_frame'])

        if opt.no_ani:
            
            generated = model.inference(Variable(data['reference_frames']), Variable(data['target_lmark']), None, \
            Variable(data['target_rgb']), similar_img, Variable(data['cropped_similar_image']) )
        else:
            generated = model.inference(Variable(data['reference_frames']), Variable(data['target_lmark']), \
            Variable(data['target_ani']),  Variable(data['target_rgb']), similar_img, Variable(data['cropped_similar_image'] ))
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
            for kkk in opt.lstm_length:
                tmp.extend([('target_lmark', util.tensor2im(data['target_lmark'][0,kkk])),
                                        ('target_ani', util.tensor2im(data['target_ani'][0,kkk])),
                                        ('synthesized_image', util.tensor2im(generated[0].data[0,kkk])),
                                        ('real_image', util.tensor2im(data['target_rgb'][0,kkk]))])
                tmp.extend([('masked_similar_img', util.tensor2im(generated[1].data[0,kkk])),
                                        ('face_foreground', util.tensor2im(generated[2].data[0,kkk])),
                                        ('beta', util.tensor2im(generated[3].data[0,kkk])),
                                        ('alpha', util.tensor2im(generated[4].data[0,kkk])),
                                        ('I_hat', util.tensor2im(generated[5].data[0,kkk]))])
        visuals =  OrderedDict(tmp)  
        img_path = data['v_id']
        print('process image...' + str(i) + ': %s' % img_path)

        print (img_path)
        visualizer.save_images(webpage, visuals, img_path)




webpage.save()


