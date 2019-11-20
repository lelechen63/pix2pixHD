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
from pathlib import Path

import cv2 
from data.dataset import *
import torchvision.transforms as transforms
import torchvision
import dlib
import face_alignment
from scipy.spatial.transform import Rotation 
from util import utils
import time
# from numpy import *
from scipy.spatial.transform import Rotation as R
res = 224
import soft_renderer as sr
import util.util as util
from util.visualizer import Visualizer
from util import html
import shutil
import pickle
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('./basics/shape_predictor_68_face_landmarks.dat')



opt = TestOptions().parse(save=False)
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle


def demo_data(opt = None, video_path = None, reference_id = None, mode = None, ani_video_path = None, reference_img_path = None):
        output_shape   = tuple([opt.loadSize, opt.loadSize])
        num_frames = opt.num_frames
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), inplace=True)])
        v_id = video_path[:-4]
        rt_path = video_path[:-4] + '_sRT.npy'
        switch_rt_path  = rt_path = '/home/cxu-serve/p1/lchen63/voxceleb/unzip/test_prediction_video3/id01822/00003_sRT.npy'
        lmark_path = video_path[:-4] + '.npy'
        # lmark_path = os.path.join(Path(video_path).parent, 'predicted_lmarks.npy')

        rt = np.load(rt_path)[:,:3]
        switch_rt = np.load(switch_rt_path)
        lmark = np.load(lmark_path)[:,:,:-1]

        front_lmark  = np.load(video_path[:-4] + '_front.npy')
        v_length = min(lmark.shape[0], rt.shape[0] , switch_rt.shape[0]) 
        if num_frames == -1:
            num_frames = v_length -1
        target_ids = []
        for gg in range(v_length):
            target_ids.append(gg)
        if mode == 0:
            ani_video_path =video_path[:-4] + '_id01822_00003_ani_.mp4'
            real_video  = mmcv.VideoReader(video_path)
            ani_video = mmcv.VideoReader(ani_video_path)
            # sample frames for embedding network
            if opt.use_ft:
                if num_frames  ==1 :
                    input_indexs = [0]
                    reference_id = 0
                elif num_frames == 8:
                    input_indexs = [0,7,15,23,31,39,47,55]

                elif num_frames == 32:
                    input_indexs = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63]
                elif num_frames == -1:
                    input_indexs = random.sample(range(0,v_length), num_frames)
            else:
                print (num_frames)
                input_indexs = random.sample(range(0,v_length), num_frames)
                input_indexs  = set(input_indexs)

            reference_rts = np.zeros((num_frames, 3))
            # we randomly choose a target frame 
            target_ids = []
            for gg in range(v_length):
                target_ids.append(gg)
            reference_frames = torch.zeros(num_frames, 6 ,output_shape[0],output_shape[1])
            for kk, t in enumerate(input_indexs):
                print ('+++++++++++++', t)
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
        similar_frame = torch.zeros( 3, output_shape[0], output_shape[0])    
        reference_frames = torch.unsqueeze( reference_frames, 0)  
        ############################################################################
        for target_id in target_ids:
            
            

            reference_rt_diff = reference_rts -  switch_rt[target_id,:3]
            reference_rt_diff = np.absolute(reference_rt_diff)
            r_diff = np.mean(reference_rt_diff, axis =1)
            similar_id  = np.argmin(r_diff) 
            similar_frame = reference_frames[0,similar_id,:3]

            target_rgb = real_video[target_id]
            target_ani = ani_video[target_id]
            # target_lmark = lmark[target_id]
            # target_lmark = front_lmark[target_id]
            target_lmark = utils.reverse_rt(front_lmark[target_id], switch_rt[target_id])
            target_lmark  =  np.asarray(target_lmark)[:,:-1]

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
            cropped_similar_image = similar_frame.clone()
            cropped_similar_image[target_ani> -0.9] = -1 


            target_lmark = torch.unsqueeze( target_lmark, 0)  
            target_rgb= torch.unsqueeze(target_rgb , 0) 
            target_ani= torch.unsqueeze(target_ani , 0) 
            similar_frame = torch.unsqueeze(similar_frame , 0) 
            cropped_similar_image = torch.unsqueeze(cropped_similar_image , 0) 

            if mode == 0:
                input_dic = {'v_id' : v_id, 'target_lmark': target_lmark, 'reference_frames': reference_frames, \
                'target_rgb': target_rgb, 'target_ani': target_ani, 'reference_ids':str(input_indexs), 'target_id': target_id \
                , 'similar_frame': similar_frame, "cropped_similar_image" : cropped_similar_image}
            else:
                input_dic = {'v_id' : v_id, 'target_lmark': target_lmark, 'reference_frames': reference_frames, \
                'target_rgb': target_rgb, 'target_ani': target_ani, 'reference_ids':str(0), 'target_id': target_id \
                , 'similar_frame': similar_frame, "cropped_similar_image" : cropped_similar_image}
            input_dics.append(input_dic)
        print (video_path)
        return input_dics


# root = '/home/cxu-serve/p1/lchen63/voxceleb'
# root = '/data2/lchen63/voxceleb'

# v_id = 'test_video/id00061/cAT9aR8oFx0/00141'
# reference_id = 542

# def test():
if not os.path.exists('./demo' ):
    os.mkdir('./demo')
if not os.path.exists( os.path.join('./demo', opt.name)  ):
    os.mkdir(os.path.join('./demo', opt.name))


# target_img_path = '/home/cxu-serve/p1/lchen63/voxceleb/unzip/demo_video/id00017/01dfn2spqyE/lisa.jpg'

# cropped_img, _ = crop_image(target_img_path)
# cv2.imwrite(target_img_path[:-4] +'_crop.jpg', cropped_img)
############# then you need to use PRnet to generate 3d  
## exaple: go to PRnet folder, python get_3d.py --img_path

# prnet_lmark = fa.get_landmarks(cropped_img)



#change frame rate to 25FPS
# opt.v_path = '/home/cxu-serve/p1/lchen63/voxceleb/unzip/lrw_test/ABOUT_00001/ABOUT_00001_crop.mp4'


# opt.v_path  = '/home/cxu-serve/p1/lchen63/voxceleb/unzip/demo_video/id00017/01dfn2spqyE/00001.mp4'
# command = 'ffmpeg -i ' +  opt.v_path +   ' -r 25 -y  ' + opt.v_path[:-4] + '_fps25.mp4'
# os.system(command)
##################################################
# if opt.v_path == '/home/cxu-serve/p1/lchen63/voxceleb/unzip/demo_video/lele.MOV':
# opt.v_path =  opt.v_path[:-4] + '_fps25.mp4'
# _crop_video(opt.v_path)
########################################
# opt.v_path = opt.v_path[:-4] + '_crop.mp4'
    # key_id = 743
# _video2img2lmark(opt.v_path)
# compute_RT(opt.v_path)
## then you need to use PRnet to generate 3d  
## exaple: go to PRnet folder, python get_3d.py --v_path /home/cxu-serve/p1/lchen63/voxceleb/unzip/demo_video/lele_fps25_crop.mp4 --target_id 743
# # key_id, video_path  = compose_front(opt.v_path)
# gg_folder = '/home/cxu-serve/p1/lchen63/voxceleb/unzip/demo_video/id00017/01dfn2spqyE'
# # obj_path =  os.path.join(gg_folder,'lisa_crop_original.obj')
# obj_path =  os.path.join(gg_folder,'00002_original.obj')

# # reference_prnet_lmark_path = os.path.join(gg_folder,'lisa_crop_prnet.npy')
# reference_prnet_lmark_path = os.path.join(gg_folder,'00002_prnet.npy')

# # img_path = os.path.join(gg_folder,'lisa_crop.jpg')
# img_path = os.path.join(gg_folder,'00002_00051.png')

# rt_path = os.path.join(gg_folder,'00001_sRT.npy')
# lmark_path = os.path.join(gg_folder,'00001_front.npy')
# video_path =  os.path.join(gg_folder,'00001.mp4')
# # ani_video_path = os.path.join(gg_folder,'lisa_crop_ani.mp4')
# ani_video_path = os.path.join(gg_folder,'00002_00051_ani.mp4')

# get_animation(  obj_path= obj_path, reference_prnet_lmark_path= reference_prnet_lmark_path, img_path = img_path,  \
#     key_id = None, video_path = None, rt_path= rt_path, lmark_path= lmark_path )



# dataset = demo_data(opt =opt, video_path = video_path, reference_id = None,mode = 1,ani_video_path = ani_video_path, reference_img_path = img_path)
model = create_model(opt)
print(model)


visualizer = Visualizer(opt)
# create website
web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))

pkl_name = os.path.join( '/home/cxu-serve/p1/lchen63/voxceleb/txt','vox_audio_key_frame.pkl')
_file = open(pkl_name, "rb")
gggdata = pickle.load(_file)
_file.close()
for gg in gggdata:
    # try:
    v_path = gg[0]
    reference_id = gg[1]
    dataset = demo_data(opt =opt, video_path = v_path, reference_id =reference_id ,mode = 0)
    # except:
        # print (v_path)
        # print ('+++++++++')
        # continue

    for i, data in enumerate(dataset):
        v_id = data['v_id'].split('/')
        if not os.path.exists( os.path.join('./demo', opt.name, v_id[-2] ) ):
            os.mkdir(os.path.join('./demo', opt.name, v_id[-2]))
        if not os.path.exists( os.path.join('./demo', opt.name, v_id[-2] , v_id[-1] ) ):
            os.mkdir(os.path.join('./demo', opt.name, v_id[-2] , v_id[-1]))
        save_path = os.path.join('./demo', opt.name,v_id[-2],v_id[-1])
        minibatch = 1 
        if opt.no_ani:
            generated = model.inference(Variable(data['reference_frames']), Variable(data['target_lmark']), \
            None,  Variable(data['target_rgb']), Variable(data['similar_frame']), Variable(data['cropped_similar_image'] ))
        else:
            generated = model.inference(Variable(data['reference_frames']), Variable(data['target_lmark']), \
            Variable(data['target_ani']),  Variable(data['target_rgb']), Variable(data['similar_frame']), Variable(data['cropped_similar_image'] ))
        
        img = torch.cat([generated[5].data.cpu(),  generated[0].data.cpu(), data['target_rgb'], data['similar_frame']], 0)
        torchvision.utils.save_image(img, 
                    "{}/{:05d}.png".format(save_path,i),normalize=True)

        gg_name  = os.path.join('./results', opt.name + '_vox_audio')
        if not os.path.exists(gg_name):
            os.mkdir(gg_name)
        if not os.path.exists(os.path.join(gg_name, v_id[-2] )):
            os.mkdir(os.path.join(gg_name, v_id[-2] ))

        if not os.path.exists(os.path.join(gg_name, v_id[-2], v_id[-1] )):
            os.mkdir(os.path.join(gg_name, v_id[-2] , v_id[-1]))
        gg_name = os.path.join(gg_name, v_id[-2] , v_id[-1])
        torchvision.utils.save_image(generated[0].data.cpu(), 
                    "{}/{:05d}_synthesized_image.png".format(gg_name,i),normalize=True)
        torchvision.utils.save_image(data['target_rgb'], 
                    "{}/{:05d}_real_image.png".format(gg_name,i),normalize=True)
