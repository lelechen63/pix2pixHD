### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import os
from collections import OrderedDict
from torch.autograd import Variable
from options.test_options import TestOptions
from models.models import create_model
import util.util as util
from util import html
import torch
import numpy as np 
import mmcv
import cv2 
from data.dataset import *
import torchvision.transforms as transforms
import torchvision

opt = TestOptions().parse(save=False)
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle



#create demo data
def demo_data(root, v_id, reference_id):
    output_shape   = tuple([opt.loadSize, opt.loadSize])
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), inplace=True)])
    num_frames = opt.num_frames
    video_path = os.path.join(root, 'unzip', v_id + '.mp4')
    ani_video_path = os.path.join(root, 'unzip', v_id + '_ani.mp4')
    rt_path = os.path.join(root, 'unzip', v_id + '_sRT.npy')
    lmark_path = os.path.join(root, 'unzip', v_id + '.npy')
    rt = np.load(rt_path)[:,:3]
    lmark = np.load(lmark_path)[:,:,:-1]
    v_length = lmark.shape[0]
    real_video  = mmcv.VideoReader(video_path)
    ani_video = mmcv.VideoReader(ani_video_path)
    # sample frames for embedding network
    input_indexs  = set(random.sample(range(0,64), num_frames))
    reference_rts = np.zeros((num_frames, 3))
    # we randomly choose a target frame 
    # while True:
    target_ids = []
    for gg in range(v_length):
        target_ids.append(gg)
    reference_frames = torch.zeros(self.num_frames, 6 ,self.output_shape[0],self.output_shape[1])
    for kk, t in enumerate(input_indexs):
        rgb_t =  mmcv.bgr2rgb(real_video[t]) 
        lmark_t = lmark[t]
        lmark_rgb = plot_landmarks( lmark_t)
        # lmark_rgb = np.array(lmark_rgb) 
        # resize 224 to 256
        rgb_t  = cv2.resize(rgb_t, output_shape)
        lmark_rgb  = cv2.resize(lmark_rgb, output_shape)
        
        # to tensor
        rgb_t = transform(rgb_t)
        lmark_rgb = transform(lmark_rgb)
        reference_frames[kk] = torch.cat([rgb_t, lmark_rgb],0) # (6, 256, 256)   

        reference_rts[kk] = rt[t]

    input_dics = []
    similar_frames = torch.zeros(v_length, 6, output_shape[0], output_shape[0])
    ############################################################################
    for target_id in target_ids:
        reference_rt_diff = reference_rts - rt[target_id]
        reference_rt_diff = np.absolute(reference_rt_diff)
        r_diff = np.mean(reference_rt_diff, axis =1)
        similar_id  = np.argmin(r_diff) 
        similar_frames[target_id] = reference_frames[similar_id]

        target_rgb = real_video[target_id]
        reference_rgb = real_video[reference_id]
        reference_ani = ani_video[reference_id]
        target_ani = ani_video[target_id]
        target_lmark = lmark[target_id]

        target_rgb = mmcv.bgr2rgb(target_rgb)
        target_rgb = cv2.resize(target_rgb, output_shape)
        target_rgb = transform(target_rgb)

        target_ani = mmcv.bgr2rgb(target_ani)
        target_ani = cv2.resize(target_ani, output_shape)
        target_ani = transform(target_ani)

        target_lmark = plot_landmarks(target_lmark)
        # target_lmark = np.array(target_lmark) 
        target_lmark  = cv2.resize(target_lmark, output_shape)
        target_lmark = transform(target_lmark)
        

        target_lmark = torch.unsqueeze( target_lmark, 0)  
        target_rgb= torch.unsqueeze(target_rgb , 0) 
        target_ani= torch.unsqueeze(target_ani , 0) 


        input_dic = {'v_id' : v_id, 'target_lmark': target_lmark, 'reference_frames': reference_frames,
        'target_rgb': target_rgb, 'target_ani': target_ani, 'reference_ids':str(input_indexs), 'target_id': target_id
        ,'similar_frame': similar_frames}
        input_dics.append(input_dic)
    return input_dics

# root = '/home/cxu-serve/p1/lchen63/voxceleb'
root = '/data2/lchen63/voxceleb'

# v_id = 'test_video/id00061/cAT9aR8oFx0/00141'
# reference_id = 542


# v_id = 'test_video/id00017/01dfn2spqyE/00001'
# reference_id = 102

# v_id = 'test_video/id00419/3dKki1hVXQE/00035'
# reference_id = 178

# v_id = 'test_video/id02286/4T-CluJt8WI/00003'
# reference_id = 189
# v_id = 'dev_video/id01387/qzB04Chz-xQ/00431'
# reference_id  = 126

# v_id = 'dev_video/id01241/cj3tXu2kvG4/00073'
# reference_id  = 63
# v_id = 'dev_video/id01241/shLz1teDejw/00094'
# reference_id = 36}
if not os.path.exists('./demo' ):
        os.mkdir('./demo')
if not os.path.exists( os.path.join('./demo', opt.name)  ):
    os.mkdir(os.path.join('./demo', opt.name))
_file = open(os.path.join(root, 'txt', "front_rt2.pkl"), "rb")
data = pkl._Unpickler(_file)
data.encoding = 'latin1'
data = data.load()
random.shuffle(data)

for item in data[:2]:
    v_id = item[0]
    reference_id = item[1]
    dataset = demo_data(root, v_id, reference_id)
    if not os.path.exists( os.path.join('./demo', opt.name, v_id.split('/')[-2] + '_' + v_id.split('/')[-1])  ):
        os.mkdir(os.path.join('./demo', opt.name, v_id.split('/')[-2]+ '_' +  v_id.split('/')[-1]))
    save_path = os.path.join('./demo', opt.name,v_id.split('/')[-2]+ '_' +  v_id.split('/')[-1])
    # test
model = create_model(opt)
if opt.verbose:
    print(model)

for i, data in enumerate(dataset):
    # if i >= opt.how_many:
    #     break
    minibatch = 1 
    if opt.no_ani:
        generated = model.inference(references =Variable(data['reference_frames']),target_lmark= Variable(data['target_lmark']),target_ani=  None, real_image=  Variable(data['target_rgb']), similar_frame = Variable(data['similar_frame']))
    else:
        generated = model.inference(references =Variable(data['reference_frames']),target_lmark= Variable(data['target_lmark']),target_ani= Variable(data['target_ani']),real_image=  Variable(data['target_rgb']), similar_frame = Variable(data['similar_frame']))
    
    img = torch.cat([generated.data.cpu(), data['target_rgb']], 0)
    torchvision.utils.save_image(img, 
			    "{}/{:05d}.png".format(save_path,i),normalize=True)


nput_dic = {'v_id' : v_id, 'target_lmark': target_lmark, 'reference_frames': reference_frames,
            'target_rgb': target_rgb, 'target_ani': target_ani, 'reference_ids':str(input_indexs), 'target_id': target_id
            , 'similar_frame': similar_frame}

# losses, generated = model(references =Variable(data['reference_frames']),target_lmark= Variable(data['target_lmark']),target_ani=   Variable(data['target_ani']),real_image=  Variable(data['target_rgb']), similar_frame = Variable(data['similar_frame']), infer=save_fake)
