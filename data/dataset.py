import os
from datetime import datetime
import pickle as pkl
import random
import scipy.ndimage.morphology

import PIL
import cv2
import matplotlib
# matplotlib.use('pdf')
import matplotlib.pyplot as plt


import numpy as np
from torch.utils.data import Dataset
import torch
import torchvision.transforms as transforms
import mmcv
from io import BytesIO
from PIL import Image


class Lmark2rgbDataset(Dataset):
    """ Dataset object used to access the pre-processed VoxCelebDataset """

    def __init__(self,opt):
        """
        Instantiates the Dataset.

        :param root: Path to the folder where the pre-processed dataset is stored.
        :param extension: File extension of the pre-processed video files.
        :param shuffle: If True, the video files will be shuffled.
        :param transform: Transformations to be done to all frames of the video files.
        :param shuffle_frames: If True, each time a video is accessed, its frames will be shuffled.
        """
        
        self.output_shape   = tuple([opt.loadSize, opt.loadSize])
        self.num_frames = opt.num_frames
        self.opt = opt
        self.root  = opt.dataroot

        if opt.isTrain:
            if self.root == '/home/cxu-serve/p1/lchen63/voxceleb' or opt.use_ft:
                _file = open(os.path.join(self.root, 'txt', "front_rt2.pkl"), "rb")
            else:
                _file = open(os.path.join(self.root, 'txt',  "train_front_rt2.pkl"), "rb")

            # self.data = pkl.load(_file)
            self.data = pkl._Unpickler(_file)
            self.data.encoding = 'latin1'
            self.data = self.data.load()[1:]
            _file.close()
        else:
            _file = open(os.path.join(self.root, 'txt', "front_rt2.pkl"), "rb")
            self.data = pkl._Unpickler(_file)
            self.data.encoding = 'latin1'
            self.data = self.data.load()[1:]
            # self.data = pkl.load(_file)
            _file.close()
        print (len(self.data))
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])


    def __len__(self):
        return len(self.data) 

    
    def name(self):
        return 'Lmark2rgbDataset'

    def __getitem__(self, index):
        # try:
            v_id = self.data[index][0]
            reference_id = self.data[index][1]
            video_path = os.path.join(self.root, 'unzip', v_id + '.mp4')
            ani_video_path = os.path.join(self.root, 'unzip', v_id + '_ani.mp4')
            rt_path = os.path.join(self.root, 'unzip', v_id + '_sRT.npy')
            lmark_path = os.path.join(self.root, 'unzip', v_id + '.npy')


            lmark = np.load(lmark_path)[:,:,:-1]
            rt = np.load(rt_path)[:,:3]

            v_length = lmark.shape[0]

            real_video  = mmcv.VideoReader(video_path)
            ani_video = mmcv.VideoReader(ani_video_path)

            # sample frames for embedding network
            if self.opt.use_ft:
                if self.num_frames  ==1 :
                    input_indexs = [0]
                    target_id = 0
                    reference_id = 0
                elif self.num_frames == 8:
                    input_indexs = [0,7,15,23,31,39,47,55]
                    target_id =  random.sample(input_indexs, 1)
                    input_indexs = set(input_indexs ) - set(target_id)
                    input_indexs =list(input_indexs) 

                elif self.num_frames == 32:
                    input_indexs = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63]
                    target_id =  random.sample(input_indexs, 1)
                    input_indexs = set(input_indexs ) - set(target_id)
                    input_indexs =list(input_indexs)                    
            else:
                input_indexs  = set(random.sample(range(0,64), self.num_frames))
                # we randomly choose a target frame 
                target_id =  np.random.choice( 64, v_length - 1)
                    
            reference_frames = []
            reference_rt_diffs = []
            if type(target_id) == list:
                target_id = target_id[0] 

            target_rt = rt[target_id]
            for t in input_indexs:

                reference_rt_diffs.append( rt[t] - target_rt )
                rgb_t =  mmcv.bgr2rgb(real_video[t]) 
                lmark_t = lmark[t]
                lmark_rgb = plot_landmarks( lmark_t)
                # lmark_rgb = np.array(lmark_rgb) 
                # resize 224 to 256
                rgb_t  = cv2.resize(rgb_t, self.output_shape)
                lmark_rgb  = cv2.resize(lmark_rgb, self.output_shape)
                
                # to tensor
                rgb_t = self.transform(rgb_t)
                lmark_rgb = self.transform(lmark_rgb)
                reference_frames.append(torch.cat([rgb_t, lmark_rgb],0))  # (6, 256, 256)   
            reference_rt_diffs = np.absolute(reference_rt_diffs)
            reference_rt_diffs = np.mean(reference_rt_diffs, axis =1)
            # similar_id  = input_indexs[np.argmin(r_diff)]
            similar_id  = np.argmin(reference_rt_diffs)
            reference_frames = torch.stack(reference_frames)
            
            ############################################################################
            target_rgb = real_video[target_id]
            reference_rgb = real_video[reference_id]
            reference_ani = ani_video[reference_id]
            target_ani = ani_video[target_id]
            target_lmark = lmark[target_id]

            target_rgb = mmcv.bgr2rgb(target_rgb)
            target_rgb = cv2.resize(target_rgb, self.output_shape)
            target_rgb = self.transform(target_rgb)

            target_ani = mmcv.bgr2rgb(target_ani)
            target_ani = cv2.resize(target_ani, self.output_shape)
            target_ani = self.transform(target_ani)

            target_lmark = plot_landmarks(target_lmark)
            target_lmark  = cv2.resize(target_lmark, self.output_shape)
            target_lmark = self.transform(target_lmark)
            similar_frame = reference_frames[similar_id,:3]
            cropped_similar_image = similar_frame.clone()
            mask = target_ani > -0.9
            mask = scipy.ndimage.morphology.binary_dilation(mask.numpy(),iterations = 5).astype(np.bool)
            cropped_similar_image[torch.tensor(mask)] = -1 


            input_dic = {'v_id' : v_id, 'target_lmark': target_lmark, 'reference_frames': reference_frames, \
            'target_rgb': target_rgb, 'target_ani': target_ani, 'reference_ids':str(input_indexs), 'target_id': target_id \
            , 'similar_frame': similar_frame, "cropped_similar_image" : cropped_similar_image}
            return input_dic
        # except:
        #     return None


class Voxceleb_audio_lmark_single(Dataset):
    def __init__(self,
                 opt):      
        
        self.output_shape   = tuple([opt.loadSize, opt.loadSize])
        self.num_frames = opt.num_frames
        self.opt = opt
        self.root  = opt.dataroot
        
        if opt.isTrain:
            if self.root == '/home/cxu-serve/p1/lchen63/voxceleb' or opt.use_ft:
                _file = open(os.path.join(self.root, 'txt', "front_rt2.pkl"), "rb")
            else:
                _file = open(os.path.join(self.root, 'txt',  "train_front_rt2.pkl"), "rb")

            # self.data = pkl.load(_file)
            self.data = pkl._Unpickler(_file)
            self.data.encoding = 'latin1'
            self.data = self.data.load()
            _file.close()
        else:
            _file = open(os.path.join(self.root, 'txt', "front_rt2.pkl"), "rb")
            self.data = pkl._Unpickler(_file)
            self.data.encoding = 'latin1'
            self.data = self.data.load()
            # self.data = pkl.load(_file)
            _file.close()
        print (len(self.data))
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

        self.consider_key = [1,2,3,4,5,11,12,13,14,15,27,28,29,30,31,32,33,34,35,39,42,36,45,17,21,22,26]
    def __getitem__(self, index):
        v_id = self.data[index][0]
        reference_id = self.data[index][1]
        lmark_path = os.path.join(self.root, 'unzip', v_id + '_front.npy')

        audio_path  = os.path.join(self.root, 'unzip', v_id.replace('_video', '_audio') + '.npy')

        tmp = self.data[index][0].split('/')
        lmark = np.load( os.path.join(self.root, 'unzip', self.data[index][0] + '_front.npy' ))
        if self.train == 'train':
            audio_path = os.path.join(self.root, 'unzip', 'test_audio' , tmp[1],tmp[2], tmp[3] +'.npy' )
        else:
            audio_path = os.path.join(self.root, 'unzip', 'test_audio' , tmp[1],tmp[2], tmp[3] +'.npy' )
            rts = np.load(os.path.join(self.root, 'unzip', self.data[index][0] + '_sRT.npy' ))
        audio = np.load(audio_path)
        sample_id = self.data[index][1]
        lmark_length = lmark.shape[0]
        audio_length = audio.shape[0]
        lmark = utils.smooth(lmark)
        audio_pad = np.zeros((lmark.shape[0] * 4, 13))
        if audio_length < lmark_length * 4 :
            audio_pad[:audio_length] = audio
            audio = audio_pad
        
        if sample_id < 3:
            sample_audio = np.zeros((28,12))
            sample_audio[4 * (3- sample_id): ] = audio[4 * (0) : 4 * ( 3 + sample_id + 1 ), 1: ]

        elif sample_id > lmark_length - 4:
            sample_audio = np.zeros((28,12))
            sample_audio[:4 * ( lmark_length + 3 - sample_id )] = audio[4 * (sample_id -3) : 4 * ( lmark_length ), 1: ]

        else:
            sample_audio = audio[4 * (sample_id -3) : 4 * ( sample_id + 4 ) , 1: ]
        
        sample_lmark = lmark[sample_id]
        sample_audio =torch.FloatTensor(sample_audio)
        sample_lmark =torch.FloatTensor(sample_lmark)            
        sample_lmark = sample_lmark.view(-1)


        ex_id = self.data[index][2]
        ex_lmark = lmark[ex_id]
        ex_lmark =torch.FloatTensor(ex_lmark)
        
        ex_lmark = ex_lmark.view(-1)
        
        # input_dict = {'audio': sample_audio , 'lip_region': lip_region, 'other_region': other_region, 'ex_other_region':ex_other_region,'ex_lip_region':ex_lip_region,   'img_path': self.data[index][0], 'sample_id' : sample_id, 'sample_rt': rts[sample_id] if self.train == 'test' else 1}
        input_dict = {'audio': sample_audio , 'sample_lmark': sample_lmark, 'ex_lmark': ex_lmark,   'img_path': self.data[index][0], 'sample_id' : sample_id, 'sample_rt': rts[sample_id] if self.train == 'test' else 1}

        return (input_dict)   
#         except:
#             return self.__getitem__((index + 1)% len(self.data))
    def __len__(self):
        
            return len(self.data)        


class Lmark2rgbLSTMDataset(Dataset):
    """ Dataset object used to access the pre-processed VoxCelebDataset """

    def __init__(self,opt):
        """
        Instantiates the Dataset.

        :param root: Path to the folder where the pre-processed dataset is stored.
        :param extension: File extension of the pre-processed video files.
        :param shuffle: If True, the video files will be shuffled.
        :param transform: Transformations to be done to all frames of the video files.
        :param shuffle_frames: If True, each time a video is accessed, its frames will be shuffled.
        """
        
        self.output_shape   = tuple([opt.loadSize, opt.loadSize])
        self.num_frames = opt.num_frames
        self.opt = opt
        self.root  = opt.dataroot
        self.lstm_length = opt.lstm_length
        if opt.isTrain:
            _file = open(os.path.join(self.root, 'txt',  "train_front_rt2.pkl"), "rb")
            # self.data = pkl.load(_file)
            self.data = pkl._Unpickler(_file)
            self.data.encoding = 'latin1'
            self.data = self.data.load()
            _file.close()
        else:
            _file = open(os.path.join(self.root, 'txt', "front_rt2.pkl"), "rb")
            self.data = pkl._Unpickler(_file)
            self.data.encoding = 'latin1'
            self.data = self.data.load()[1:]
            # self.data = pkl.load(_file)
            _file.close()
        print (len(self.data))
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), inplace=True)
        ])


    def __len__(self):
        return len(self.data)  
    
    def name(self):
        return 'Lmark2rgbDataset'

    def __getitem__(self, index):
        v_id = self.data[index][0]
        reference_id = self.data[index][1]
        video_path = os.path.join(self.root, 'unzip', v_id + '.mp4')
        ani_video_path = os.path.join(self.root, 'unzip', v_id + '_ani.mp4')
        rt_path = os.path.join(self.root, 'unzip', v_id + '_sRT.npy')
        lmark_path = os.path.join(self.root, 'unzip', v_id + '.npy')
        lmark = np.load(lmark_path)[:,:,:-1]
        rt = np.load(rt_path)[:,:3]
        v_length = lmark.shape[0]
        real_video  = mmcv.VideoReader(video_path)
        ani_video = mmcv.VideoReader(ani_video_path)
        # sample frames for embedding network
        if self.opt.isTrain: 
            input_indexs  = random.sample(range(0,64), self.num_frames)
        else:
            if self.num_frames  ==1 :
                input_indexs = [0]
            elif self.num_frames == 8:
                input_indexs = [0,7,15,23,31,39,47,55]

            elif self.num_frames == 32:
                input_indexs = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63]

        # we randomly choose a start target frame 
        start_target_id =  np.random.choice([64, v_length - self.lstm_length])
        reference_frames = torch.zeros(self.num_frames, 6 ,self.output_shape[0],self.output_shape[1])
        reference_rts = np.zeros((self.num_frames, 3))
        target_rts = rt[start_target_id: start_target_id + self.lstm_length]
        target_rgbs = torch.zeros(self.lstm_length,3,self.output_shape[0],self.output_shape[1])# real_video[start_target_id: start_target_id + self.lstm_length]
        target_anis = torch.zeros(self.lstm_length,3,self.output_shape[0],self.output_shape[1])# ani_video[start_target_id: start_target_id + self.lstm_length]
        target_lmarks= torch.zeros(self.lstm_length,3,self.output_shape[0],self.output_shape[1])
############################################################################
        for kk, t in enumerate( input_indexs):

            
            rgb_t =  mmcv.bgr2rgb(real_video[t]) 
            lmark_rgb = plot_landmarks( lmark[t])
            rgb_t  = cv2.resize(rgb_t, self.output_shape)
            lmark_rgb  = cv2.resize(lmark_rgb, self.output_shape)
            
            # to tensor
            rgb_t = self.transform(rgb_t)
            lmark_rgb = self.transform(lmark_rgb)
            reference_rts[kk] = rt[t]
            reference_frames[kk] = torch.cat([rgb_t, lmark_rgb],0)  # (6, 256, 256)
############################################################################

        similar_frames = torch.zeros(self.lstm_length, 3, self.output_shape[0], self.output_shape[0])
        cropped_similar_image = torch.zeros(self.lstm_length, 3, self.output_shape[0], self.output_shape[0])
        for kk in range(self.lstm_length):
            reference_rt_diff = reference_rts - rt[start_target_id + kk]
            reference_rt_diff = np.absolute(reference_rt_diff)
            r_diff = np.mean(reference_rt_diff, axis =1)
            similar_id  = np.argmin(r_diff) # input_indexs[np.argmin(r_diff)]
            similar_frame = reference_frames[similar_id,:3]
            similar_frames[kk] = similar_frame



            target_lmark = plot_landmarks( lmark[start_target_id + kk])
            target_lmark  = cv2.resize(target_lmark, self.output_shape)
            target_lmark = self.transform(target_lmark)
            target_lmarks[kk] = target_lmark

            target_rgb = mmcv.bgr2rgb(real_video[start_target_id + kk])
            target_rgb = cv2.resize(target_rgb, self.output_shape)
            target_rgb = self.transform(target_rgb)
            target_rgbs[kk] = target_rgb

            target_ani = mmcv.bgr2rgb(ani_video[start_target_id + kk])
            target_ani = cv2.resize(target_ani, self.output_shape)
            target_ani = self.transform(target_ani)
            target_anis[kk] = target_ani
            cropped_similar_img = similar_frame.clone()

            mask = target_ani > -0.9
            
            mask = scipy.ndimage.morphology.binary_dilation(mask.numpy(),iterations = 5).astype(np.bool)
            cropped_similar_img[torch.tensor(mask)] = -1 

            cropped_similar_image[kk] = cropped_similar_img

       
        ############################################################################

        input_dic = {'v_id' : v_id, 'target_lmark': target_lmarks, 'reference_frames': reference_frames,
        'target_rgb': target_rgbs, 'target_ani': target_anis, 'reference_ids':str(input_indexs), 'target_id': start_target_id, 
        'similar_frame': similar_frames, "cropped_similar_image" : cropped_similar_image}
        return input_dic


def plot_landmarks1( landmarks):
    """
    Creates an RGB image with the landmarks. The generated image will be of the same size as the frame where the face
    matching the landmarks.

    The image is created by plotting the coordinates of the landmarks using matplotlib, and then converting the
    plot to an image.

    Things to watch out for:
    * The figure where the landmarks will be plotted must have the same size as the image to create, but matplotlib
    only accepts the size in inches, so it must be converted to pixels using the DPI of the screen.
    * A white background is printed on the image (an array of ones) in order to keep the figure from being flipped.
    * The axis must be turned off and the subplot must be adjusted to remove the space where the axis would normally be.

    :param frame: Image with a face matching the landmarks.
    :param landmarks: Landmarks of the provided frame,
    :return: RGB image with the landmarks as a Pillow Image.
    """
    dpi = 100
    fig = plt.figure(figsize=(224/ dpi,224 / dpi), dpi=dpi)
    ax = fig.add_subplot(111)
    ax.axis('off')
    plt.imshow(np.ones((224,224)))
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # Head
    ax.plot(landmarks[0:17, 0], landmarks[0:17, 1], linestyle='-', color='green', lw=2)
    # Eyebrows
    ax.plot(landmarks[17:22, 0], landmarks[17:22, 1], linestyle='-', color='orange', lw=2)
    ax.plot(landmarks[22:27, 0], landmarks[22:27, 1], linestyle='-', color='orange', lw=2)
    # Nose
    ax.plot(landmarks[27:31, 0], landmarks[27:31, 1], linestyle='-', color='blue', lw=2)
    ax.plot(landmarks[31:36, 0], landmarks[31:36, 1], linestyle='-', color='blue', lw=2)
    # Eyes
    ax.plot(landmarks[36:42, 0], landmarks[36:42, 1], linestyle='-', color='red', lw=2)
    ax.plot(landmarks[42:48, 0], landmarks[42:48, 1], linestyle='-', color='red', lw=2)
    # Mouth
    ax.plot(landmarks[48:60, 0], landmarks[48:60, 1], linestyle='-', color='purple', lw=2)

    fig.canvas.draw()
    data = PIL.Image.frombuffer('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb(), 'raw', 'RGB', 0, 1)
    plt.close(fig)
    # print ('++++++++++++++++++++++++++++++++')
    return data


def plot_landmarks( landmarks):
    # landmarks = np.int32(landmarks)
    """
    Creates an RGB image with the landmarks. The generated image will be of the same size as the frame where the face
    matching the landmarks.

    The image is created by plotting the coordinates of the landmarks using matplotlib, and then converting the
    plot to an image.

    Things to watch out for:
    * The figure where the landmarks will be plotted must have the same size as the image to create, but matplotlib
    only accepts the size in inches, so it must be converted to pixels using the DPI of the screen.
    * A white background is printed on the image (an array of ones) in order to keep the figure from being flipped.
    * The axis must be turned off and the subplot must be adjusted to remove the space where the axis would normally be.

    :param frame: Image with a face matching the landmarks.
    :param landmarks: Landmarks of the provided frame,
    :return: RGB image with the landmarks as a Pillow Image.
    """
    # print (landmarks[0:17].shape)
    # print(type(landmarks))

    # points = np.array([[1, 4], [5, 6], [7, 8], [4, 4]])
    # print (points.shape)


    blank_image = np.zeros((224,224,3), np.uint8) 

    # cv2.polylines(blank_image, np.int32([points]), True, (0,255,255), 1)

    cv2.polylines(blank_image, np.int32([landmarks[0:17]]) , True, (0,255,255), 2)
 
    cv2.polylines(blank_image,  np.int32([landmarks[17:22]]), True, (255,0,255), 2)

    cv2.polylines(blank_image, np.int32([landmarks[22:27]]) , True, (255,0,255), 2)

    cv2.polylines(blank_image, np.int32([landmarks[27:31]]) , True, (255,255, 0), 2)

    cv2.polylines(blank_image, np.int32([landmarks[31:36]]) , True, (255,255, 0), 2)

    cv2.polylines(blank_image, np.int32([landmarks[36:42]]) , True, (255,0, 0), 2)
    cv2.polylines(blank_image, np.int32([landmarks[42:48]]) , True, (255,0, 0), 2)

    cv2.polylines(blank_image, np.int32([landmarks[48:60]]) , True, (0, 0, 255), 2)

    return blank_image



