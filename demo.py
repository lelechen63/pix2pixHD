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
from util.visualizer import Visualizer
from util import html


opt = TestOptions().parse(save=False)
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle

visualizer = Visualizer(opt)
# create website
web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))


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
    # input_indexs  = set(random.sample(range(0,64), num_frames))
    if num_frames  ==1 :
        input_indexs = [0]
    elif num_frames == 8:
        # input_indexs = [0,7,15,23,31,39,47,55]
        input_indexs = [0, 80, 200,300,400,500,550,660]
    elif num_frames == 32:
        input_indexs = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63]
    reference_rts = np.zeros((num_frames, 3))
    # we randomly choose a target frame 
    # while True:
    target_ids = []
    for gg in range(v_length):
        target_ids.append(gg)
    reference_frames = torch.zeros(num_frames, 6 ,output_shape[0],output_shape[1])
    
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
    similar_frames = torch.zeros( 3, output_shape[0], output_shape[0])    
    reference_frames = torch.unsqueeze( reference_frames, 0)  
    ############################################################################
    for target_id in target_ids:
        reference_rt_diff = reference_rts - rt[target_id]
        reference_rt_diff = np.absolute(reference_rt_diff)
        r_diff = np.mean(reference_rt_diff, axis =1)
        similar_id  = np.argmin(r_diff) 
        similar_frames = reference_frames[0,similar_id,:3]

        
        target_rgb = real_video[target_id]
        target_ani = ani_video[target_id]
        target_lmark = lmark[target_id]

        target_rgb = mmcv.bgr2rgb(target_rgb)
        target_rgb = cv2.resize(target_rgb, output_shape)
        target_rgb = transform(target_rgb)

        target_ani = mmcv.bgr2rgb(target_ani)
        target_ani = cv2.resize(target_ani, output_shape)
        target_ani = transform(target_ani)
        cropped_similar_image = similar_frames.clone()
        cropped_similar_image[target_ani> -0.9] = -1 


        target_lmark = plot_landmarks(target_lmark)
        # target_lmark = np.array(target_lmark) 
        target_lmark  = cv2.resize(target_lmark, output_shape)
        target_lmark = transform(target_lmark)
        
        
        target_lmark = torch.unsqueeze( target_lmark, 0)  
        target_rgb= torch.unsqueeze(target_rgb , 0) 
        target_ani= torch.unsqueeze(target_ani , 0) 
        similar_frames = torch.unsqueeze(similar_frames , 0) 
        cropped_similar_image = torch.unsqueeze(cropped_similar_image , 0) 



        input_dic = {'v_id' : v_id, 'target_lmark': target_lmark, 'reference_frames': reference_frames,
        'target_rgb': target_rgb, 'target_ani': target_ani, 'reference_ids':str(input_indexs), 'target_id': target_id
        ,'similar_frame': similar_frames, "cropped_similar_image" : cropped_similar_image}
        input_dics.append(input_dic)
    return input_dics

# root = '/home/cxu-serve/p1/lchen63/voxceleb'
# root = '/data2/lchen63/voxceleb'
root ='/home/cxu-serve/p1/lchen63/voxceleb/'
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
_file = open(os.path.join(root, 'txt', "demo_front_rt2.pkl"), "rb")
# ggdata = pkl._Unpickler(_file)
# ggdata.encoding = 'latin1'
# ggdata = ggdata.load()
ggdata = pkl.load(_file)
random.shuffle(ggdata)
model = create_model(opt)
if opt.verbose:
    print(model)
for item in ggdata[:1]:
    v_id = item[0]
    reference_id = item[1]
    dataset = demo_data(root, v_id, reference_id)
    if not os.path.exists( os.path.join('./demo', opt.name, v_id.split('/')[-2] + '_' + v_id.split('/')[-1])  ):
        os.mkdir(os.path.join('./demo', opt.name, v_id.split('/')[-2]+ '_' +  v_id.split('/')[-1]))
    save_path = os.path.join('./demo', opt.name,v_id.split('/')[-2]+ '_' +  v_id.split('/')[-1])
    # test

    for i, data in enumerate(dataset):
        if i >= opt.how_many:
            break
        minibatch = 1 
        if opt.no_ani:
            generated = model.inference(references =Variable(data['reference_frames']),target_lmark= Variable(data['target_lmark']),target_ani=  None, \
             real_image=  Variable(data['target_rgb']), similar_frame = Variable(data['similar_frame']),cropped_similar_img=  Variable(data['cropped_similar_image'] ))
        else:
            generated = model.inference(references =Variable(data['reference_frames']),target_lmark= Variable(data['target_lmark']), \
            target_ani= Variable(data['target_ani']),real_image=  Variable(data['target_rgb']), similar_frame = Variable(data['similar_frame']),cropped_similar_img= Variable(data['cropped_similar_image'] ))
        
        img = torch.cat([generated[0].data.cpu(),generated[1].data.cpu(),generated[2].data.cpu(),generated[5].data.cpu(), data['target_rgb']], 0)
        torchvision.utils.save_image(img, 
                    "{}/{:05d}.png".format(save_path,i),normalize=True)
        img_path = data['v_id'] + '_%05d'%i
        print('process image... %s' % img_path)
        visuals = OrderedDict([('reference1', util.tensor2im(data['reference_frames'][0, 0,:3])),
                                            ('reference2', util.tensor2im(data['reference_frames'][0, 1,:3])),
                                            ('reference3', util.tensor2im(data['reference_frames'][0, 2,:3])),
                                            ('reference4', util.tensor2im(data['reference_frames'][0, 3,:3])),
                                        ('target_lmark', util.tensor2im(data['target_lmark'][0])),
                                        ('target_ani', util.tensor2im(data['target_ani'][0])),
                                        ('synthesized_image', util.tensor2im(generated[0].data[0])),
                                        ('masked_similar_img', util.tensor2im(generated[1].data[0])),
                                        ('face_foreground', util.tensor2im(generated[2].data[0])),
                                        ('beta', util.tensor2im(generated[3].data[0])),
                                        ('alpha', util.tensor2im(generated[4].data[0])),
                                        ('I_hat', util.tensor2im(generated[5].data[0])),
                                        ('real_image', util.tensor2im(data['target_rgb'][0]))])

        print (img_path)
        visualizer.save_demo_images(webpage, visuals, img_path)


webpage.save()



# losses, generated = model(references =Variable(data['reference_frames']),target_lmark= Variable(data['target_lmark']),target_ani=   Variable(data['target_ani']),real_image=  Variable(data['target_rgb']), similar_frame = Variable(data['similar_frame']), infer=save_fake)
 