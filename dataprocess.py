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
import scipy.ndimage.morphology
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
from pathlib import Path
from scipy.spatial.transform import Rotation as R
res = 224
import soft_renderer as sr
import util.util as util
from util.visualizer import Visualizer
from util import html
import shutil
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('./basics/shape_predictor_68_face_landmarks.dat')
import pickle


def recover(rt):
    rots = []
    trans = []
    for tt in range(rt.shape[0]):
        ret = rt[tt,:3]
        r = R.from_rotvec(ret)
        ret_R = r.as_dcm()
        ret_t = rt[tt, 3:]
        ret_t = ret_t.reshape(3,1)
        rots.append(ret_R)
        trans.append(ret_t)
    return (np.array(rots), np.array(trans))

def load_obj(obj_file):
    vertices = []

    triangles = []
    colors = []

    with open(obj_file) as infile:
        for line in infile.read().splitlines():
            if len(line) > 2 and line[:2] == "v ":
                ts = line.split()
                x = float(ts[1])
                y = float(ts[2])
                z = float(ts[3])
                r = float(ts[4])
                g = float(ts[5])
                b = float(ts[6])
                vertices.append([x,y,z])
                colors.append([r,g,b])
            elif len(line) > 2 and line[:2] == "f ":
                ts = line.split()
                fx = int(ts[1]) - 1
                fy = int(ts[2]) - 1
                fz = int(ts[3]) - 1
                triangles.append([fx,fy,fz])
    
    return (np.array(vertices), np.array(triangles).astype(np.int), np.array(colors))

def setup_renderer():    
    renderer = sr.SoftRenderer(camera_mode="look", viewing_scale=2/res, far=10000, perspective=False, image_size=res, camera_direction=[0,0,-1], camera_up=[0,1,0], light_intensity_ambient=1)
    renderer.transform.set_eyes([res/2, res/2, 6000])
    return renderer
def get_np_uint8_image(mesh, renderer):
    images = renderer.render_mesh(mesh)
    image = images[0]
    image = torch.flip(image, [1,2])
    image = image.detach().cpu().numpy().transpose((1,2,0))
    image = np.clip(image, 0, 1)
    image = (255*image).astype(np.uint8)
    return image


def crop_image_lrw(image, id = 1, kxy = []):
#     kxy = []
    if id == 1:
        image = cv2.imread(image)
    if kxy != []:
        [k,x,y,center_x, center_y, orignal_h_w ] = kxy
        roi = image[max(center_y - int(0.6* k), 0)  :min(center_y + int( 1.5 * k),orignal_h_w ), \
             max( center_x- int(1.5 * k), 0) :min(center_x + int(1.5 * k),orignal_h_w )]
        roi = cv2.resize(roi, (224,224), interpolation = cv2.INTER_AREA)
        return roi, kxy 
    else:
        orignal_h_w = min(image.shape[0], image.shape[0])
        print (orignal_h_w)       
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 1)
        for (i, rect) in enumerate(rects):
            shape = predictor(gray, rect)
            shape = utils.shape_to_np(shape)
            # print (shape[39],shape[42],shape[30])
            print (np.mean(shape))
            print ((shape[39]+ shape[42] +shape[30])/3)
            (x, y, w, h) = utils.rect_to_bb(rect)
            center_x = x + int(0.5 * w)
            center_y = y + int(0.5 * h)
            k = min(w, h)
            print (center_x,center_y, k)
            
            roi = image[max(center_y - int(0.6* k), 0)  :min(center_y + int( 1.5 * k),orignal_h_w ), \
             max( center_x- int(1.5 * k), 0) :min(center_x + int(1.5 * k),orignal_h_w )]
            roi = cv2.resize(roi, (224,224), interpolation = cv2.INTER_AREA)
            return roi ,[k,x,y,center_x, center_y , orignal_h_w]
'''
def crop_image_vox(image, id = 1, kxy = []):
#     kxy = []
    if id == 1:
        image = cv2.imread(image)
    if kxy != []:
        [k,x,y,center_x, center_y, orignal_h_w ] = kxy
        roi = image[center_y - int(0.7* k)  :center_y + int( 1.2 * k), \
              center_x- int(0.9 * k) :  center_x + int(1.0 * k)]
        roi = cv2.resize(roi, (224,224), interpolation = cv2.INTER_AREA)
        return roi, kxy 
    else:
        orignal_h_w = min(image.shape[0], image.shape[0])
        print (orignal_h_w)       
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 1)
        for (i, rect) in enumerate(rects):
            shape = predictor(gray, rect)
            shape = utils.shape_to_np(shape)            
            (x, y, w, h) = utils.rect_to_bb(rect)
            print (x,y,w,h)
            center_x = x + int(0.5 * w)
            center_y = y + int(0.5 * h)
            k = min(w, h)
            print (center_x,center_y, k)
            
            roi = image[center_y - int(0.7* k)  :center_y + int( 1.2 * k), \
              center_x- int(0.9 * k) :  center_x + int(1.0 * k)]
            roi = cv2.resize(roi, (224,224), interpolation = cv2.INTER_AREA)
            return roi ,[k,x,y,center_x, center_y , orignal_h_w]
'''
# def crop_image(image, id = 1, kxy = []):
# #     kxy = []
#     if id == 1:
#         image = cv2.imread(image)
#     if kxy != []:
#         tempolate   = np.ones((2000,2000,3), np.uint8) * 255
#         print (image.shape)
#         middle = [int(image.shape[0]/2) , int(image.shape[1]/2)]
#         tempolate[1000-middle[0]:1000+middle[0], 1000-middle[1]:1000+middle[1],:] = image

#         [k,x,y,center_x, center_y, orignal_h_w ] = kxy
#         roi = tempolate[center_y - int(0.75* k)  :center_y + int( 1.15 * k), \
#               center_x- int(0.95 * k) :  center_x + int(0.85 * k)]
#         roi = cv2.resize(roi, (224,224), interpolation = cv2.INTER_AREA)
#         return roi, kxy 
#     else:
#         tempolate   = np.ones((2000,2000,3), np.uint8) * 255
#         print (image.shape)
#         middle = [int(image.shape[0]/2) , int(image.shape[1]/2)]
#         tempolate[1000-middle[0]:1000+middle[0], 1000-middle[1]:1000+middle[1],:] = image

#         orignal_h_w = min(image.shape[0], image.shape[0])
#         print (orignal_h_w)       
#         gray = cv2.cvtColor(tempolate, cv2.COLOR_BGR2GRAY)
#         rects = detector(gray, 1)
#         for (i, rect) in enumerate(rects):
#             shape = predictor(gray, rect)
#             shape = utils.shape_to_np(shape)            
#             (x, y, w, h) = utils.rect_to_bb(rect)
#             print (x,y,w,h)
#             center_x = x + int(0.5 * w)
#             center_y = y + int(0.5 * h)
#             k = min(w, h)
#             print (center_x,center_y, k)
            
#             roi = tempolate[center_y - int(0.75* k)  :center_y + int( 1.15 * k), \
#               center_x- int(0.95 * k) :  center_x + int(0.85 * k)]
#             roi = cv2.resize(roi, (224,224), interpolation = cv2.INTER_AREA)
#             return roi ,[k,x,y,center_x, center_y , orignal_h_w]


def crop_image(frame_file, lStart=36, lEnd=41, rStart=42, rEnd=47, y_1=112, y_2=112, x_1=60, x_2=164):
	image = cv2.imread(frame_file) if isinstance(frame_file, str) else frame_file

	tempolate = np.ones((2000, 2000, 3), np.uint8) * 255

	middle = [int(image.shape[0]/2), int(image.shape[1]/2)]
	tempolate[1000-middle[0]:1000+middle[0],
			  1000-middle[1]:1000+middle[1], :] = image

	gray = cv2.cvtColor(tempolate, cv2.COLOR_BGR2GRAY)
	# try:
	# 	rect = detector(gray, 1)
	# 	rect = rect[0]
	# except:
	# 	raise 'no face found!'

	# shape = predictor(gray, rect)
	# shape = utils.shape_to_np(shape)
	shape = fa.get_landmarks(gray)[0][:,:2]

	leftEyePts = shape[lStart:lEnd]
	rightEyePts = shape[rStart:rEnd]

	leftEyeCenter = leftEyePts.mean(axis=0).astype("int")
	rightEyeCenter = rightEyePts.mean(axis=0).astype("int")

	dis = np.sum((rightEyeCenter - leftEyeCenter)**2)**0.5
	resize_tempolate = cv2.resize(tempolate, (int(
		tempolate.shape[0]/(dis/56.354)), int(tempolate.shape[1]/(dis/56.354))))

	new_gray = cv2.cvtColor(resize_tempolate, cv2.COLOR_BGR2GRAY)
	# try:
	# 	new_rect = detector(new_gray, 1)
	# 	new_rect = new_rect[0]
	# except:
	# 	raise 'no face found!'

	# new_shape = predictor(new_gray, new_rect)
	# new_shape = utils.shape_to_np(new_shape)
	new_shape = fa.get_landmarks(new_gray)[0][:,:2]

	leftEyePts = new_shape[lStart:lEnd]
	rightEyePts = new_shape[rStart:rEnd]

	leftEyeCenter = leftEyePts.mean(axis=0).astype("int")
	rightEyeCenter = rightEyePts.mean(axis=0).astype("int")

	dis = np.sum((rightEyeCenter - leftEyeCenter)**2)**0.5

	two_eye_center = (leftEyeCenter + rightEyeCenter)/2

	center_y, center_x = int(two_eye_center[0]), int(two_eye_center[1])

	# cv2.circle(resize_tempolate, (center_y, center_x), 5, (255, 0, 0), -1)

	roi = resize_tempolate[center_x - int(x_1): center_x +
						   int(x_2), center_y - int(y_1):center_y + int(y_2)]

	# cv2.imwrite('./tmp/x_fa.png', roi)

	return roi


fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, flip_input=False)#,  device='cpu')




def read_folder(path):
    files = os.listdir(path)
    files.sort() 
    print (files)


def resize(video):
    cap  =  cv2.VideoCapture(video)
    if os.path.exists('./tmp'):
        shutil.rmtree('./tmp')
    os.mkdir('./tmp')
    count =0
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            count += 1
            print (count)
            frame = cv2.resize(frame, (224,224), interpolation = cv2.INTER_AREA)
            cv2.imwrite('./tmp/%05d.png'%count, frame)
        else:
            break
    command = 'ffmpeg -framerate 25  -i ' +   './tmp/%05d.png  -vcodec libx264 -y -vf format=yuv420p ' + video[:-4] + '.mp4' 
    os.system(command)
def _crop_video(video):
    cap  =  cv2.VideoCapture(video)
    count = 0
    kxy =[]
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            count += 1
            print (count)
            img = crop_image(frame)
            cv2.imwrite('./tmp/%05d.png'%count, img)
            # if count == 300:
            #     break
        else:
            break
    command = 'ffmpeg -framerate 25  -i ' +   './tmp/%05d.png  -vcodec libx264 -y -vf format=yuv420p ' + video[:-4] + '_crop.mp4' 
    os.system(command)

def get3DLmarks_single_image(image_path):
#     img = crop_image(image_path)
    img = cv2.imread(image_path)
    lmark = fa.get_landmarks(img)        
    np.save(image_path[:-4] + '.npy', lmark)



def get3DLmarks(frame_list, v_path):
    if os.path.exists(v_path[:-4] + '.npy'):
        return 0
    frame_num = len(frame_list)
    lmarks = np.zeros((frame_num, 68,3))
    for i in range(frame_num):
        lmark = fa.get_landmarks(frame_list[i])        
        if lmark is not None:
            landmark =  lmark[0]
        else:
            landmark = -np.ones((68, 3))
        lmarks[i] = landmark
    print (lmarks.shape)
    np.save(v_path[:-4] + '.npy', lmarks)

def _video2img2lmark(v_path):

    count = 0    
    frame_list = []
    frame_list = mmcv.VideoReader(v_path)
    print (len(frame_list))
    print ('++++++++')
   
    # cap  =  cv2.VideoCapture(v_path)
    # while(cap.isOpened()):
    #     ret, frame = cap.read()
    #     if ret == True:            
    #         frame_list.append(frame)
    #         count += 1
    #     else:
    #         break
    get3DLmarks(frame_list, v_path)


def rigid_transform_3D(A, B):
    assert len(A) == len(B)

    N = A.shape[0]; # total points

    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    
    # centre the points
    AA = A - np.tile(centroid_A, (N, 1))
    BB = B - np.tile(centroid_B, (N, 1))

    # dot is matrix multiplication for array
    H = np.transpose(AA) * BB

    U, S, Vt = np.linalg.svd(H)

    R = Vt.T * U.T

    # special reflection case
    if np.linalg.det(R) < 0:
        print("Reflection detected")
        Vt[2,:] *= -1
        R = Vt.T * U.T
    
    t = -R*centroid_A.T + centroid_B.T

    return R, t

def compute_RT(video_path):
    consider_key = [1,2,3,4,5,11,12,13,14,15,27,28,29,30,31,32,33,34,35,39,42,36,45,17,21,22,26]
    k = 2
    source = np.zeros((len(consider_key),3))
    
#     ff = np.load('/data/lchen63/grid/zip/video/s20/video/mpg_6000/bbad3n.npy')[0]
    ff = np.load('./basics/00001.npy')[30]
    for m in range(len(consider_key)):
        source[m] = ff[consider_key[m]]
        
    source = np.mat(source)

    lmark_path = video_path[:-4] + '.npy' 
    srt_path = video_path[:-4] +  '_sRT.npy'
    front_path =video_path[:-4] +  '_front.npy'

    if os.path.exists(srt_path) and os.path.exists(front_path):
        return 0
    
    
    t_lmark = np.load(lmark_path)
    lmark_part = np.zeros((t_lmark.shape[0],len(consider_key),3))
    RTs =  np.zeros((t_lmark.shape[0],6))

    nomalized =  np.zeros((t_lmark.shape[0],68,3))
    t = time.time()
    for j in range(lmark_part.shape[0]  ):

        for m in range(len(consider_key)):
            lmark_part[:,m] = t_lmark[:,consider_key[m]] 

        target = np.mat(lmark_part[j])
        ret_R, ret_t = rigid_transform_3D( target, source)

        source_lmark  = np.mat(t_lmark[j])

        A2 = ret_R*source_lmark.T
        A2+= np.tile(ret_t, (1, 68))
        A2 = A2.T
        nomalized[j] = A2
        r = Rotation.from_dcm(ret_R)
        vec = r.as_rotvec()             
        RTs[j,:3] = vec
        RTs[j,3:] =  np.squeeze(np.asarray(ret_t))            
    np.save(srt_path, RTs)
    np.save(front_path, nomalized)

def compute_RT_single(img_path):
    consider_key = [1,2,3,4,5,11,12,13,14,15,27,28,29,30,31,32,33,34,35,39,42,36,45,17,21,22,26]
    print (len(consider_key))
    k = 2
    source = np.zeros((len(consider_key),3))
    
#     ff = np.load('/data/lchen63/grid/zip/video/s20/video/mpg_6000/bbad3n.npy')[0]
    
    ff = np.load('./vox_sample/08TabUdunsU/00001.npy')[30]
    for m in range(len(consider_key)):
        source[m] = ff[consider_key[m]]
        
    source = np.mat(source)

    lmark_path = img_path[:-4] + '.npy' 
    srt_path = img_path[:-4] +  '_sRT.npy'
    front_path =img_path[:-4] +  '_front.npy'
    
    
    t_lmark = np.load(lmark_path)
    lmark_part = np.zeros((t_lmark.shape[0],len(consider_key),3))
    RTs =  np.zeros((t_lmark.shape[0],6))

    nomalized =  np.zeros((t_lmark.shape[0],68,3))
    t = time.time()
    for j in range(lmark_part.shape[0]  ):

        for m in range(len(consider_key)):
            lmark_part[:,m] = t_lmark[:,consider_key[m]] 

        target = np.mat(lmark_part[j])
        ret_R, ret_t = rigid_transform_3D( target, source)

        source_lmark  = np.mat(t_lmark[j])

        A2 = ret_R*source_lmark.T
        A2+= np.tile(ret_t, (1, 68))
        A2 = A2.T
        nomalized[j] = A2
        r = Rotation.from_dcm(ret_R)
        vec = r.as_rotvec()             
        RTs[j,:3] = vec
        RTs[j,3:] =  np.squeeze(np.asarray(ret_t))            
    # after frontilize, we need to 
    
    np.save(srt_path, RTs)
    np.save(front_path, nomalized)

    return nomalized
def compose_front(v_path):
    n = 68
    f_lmark = v_path[:-4] + '_front.npy' 
    rt_path = v_path[:-4] + '_sRT.npy' 
    o_lmark =  v_path[:-4]+ '.npy' 
    # o_lmark = os.path.join(Path(v_path).parent, 'predicted_lmarks.npy')     
    print (o_lmark)
    print ('+++++++++++++++++++++++++')
    v_path =  v_path[:-4] + '.mp4' 
    rt = np.load( rt_path)
    lmark = np.load(o_lmark)
    lmark_length = min(lmark.shape[0],rt.shape[0] ) 
    find_rt = []
    for t in range(0, lmark_length):
        find_rt.append(sum(np.absolute(rt[t,:3])))

    find_rt = np.asarray(find_rt)

    min_index = np.argmin(find_rt)
    print (v_path)
    print (min_index)
    return min_index, v_path



def get_animation_orignal(  obj_path= None, reference_prnet_lmark_path= None, img_path = None,  \
    key_id = None, video_path = None, rt_path= None, lmark_path= None ):
    if video_path != None:
        reference_img_path = video_path[:-4] + '_%05d.png'%key_id
        reference_prnet_lmark_path = video_path[:-4] +'_prnet.npy'
        obj_path = video_path[:-4] + '_original.obj'
        rt_path  = video_path[:-4] + '_sRT.npy'
        lmark_path  = video_path[:-4] +'_front.npy'
        orignal_ani_path = video_path[:-4] + '_ani.mp4'
        if os.path.exists( orignal_ani_path):
            print ('=====')
            return 0
        if  not os.path.exists(obj_path) or not os.path.exists(reference_prnet_lmark_path) or not os.path.exists(lmark_path) or not os.path.exists(rt_path):
            print (os.path.exists(obj_path) , os.path.exists(reference_prnet_lmark_path),  os.path.exists(lmark_path), os.path.exists(rt_path))
            print (obj_path)
            print ('++++')
    # extract the frontal facial landmarks for key frame
    lmk3d_all = np.load(lmark_path)
    print (lmk3d_all.shape)

    if video_path != None:
        lmk3d_target = lmk3d_all[key_id]
    else:
        lmk3d_target = fa.get_landmarks(cv2.imread(img_path))[0]

    # load the 3D facial landmarks on the PRNet 3D reconstructed face
    lmk3d_origin = np.load(reference_prnet_lmark_path)
    # lmk3d_origin[:,1] = res - lmk3d_origin[:,1]
    
    # load RTs
    rots, trans = recover(np.load(rt_path))

    # calculate the affine transformation between PRNet 3D face and the frotal face landmarks
    lmk3d_origin_homo = np.hstack((lmk3d_origin, np.ones([lmk3d_origin.shape[0],1]))) # 68x4
    p_affine = np.linalg.lstsq(lmk3d_origin_homo, lmk3d_target, rcond=1)[0].T # Affine matrix. 3 x 4
    pr = p_affine[:,:3] # 3x3
    pt = p_affine[:,3:] # 3x1

    # load the original 3D face mesh then transform it to align frontal face landmarks
    vertices_org, triangles, colors = load_obj(obj_path) # get unfrontalized vertices position
    vertices_origin_affine = (pr @ (vertices_org.T) + pt).T # aligned vertices

    # set up the renderer
    renderer = setup_renderer()
    # generate animation
    temp_path = './face-tools/tempp_%05d'%1
    # generate animation
    if os.path.exists(temp_path):
        shutil.rmtree(temp_path)
    os.mkdir(temp_path)
    # writer = imageio.get_writer('rotation.gif', mode='I')
    for i in range(rots.shape[0]):
            # get rendered frame
        vertices = (rots[i].T @ (vertices_origin_affine.T - trans[i])).T
        face_mesh = sr.Mesh(vertices, triangles, colors, texture_type="vertex")
        image_render = get_np_uint8_image(face_mesh, renderer) # RGBA, (224,224,3), np.uint8
        
        #save rgba image as bgr in cv2
        rgb_frame =  (image_render).astype(int)[:,:,:-1][...,::-1]
        cv2.imwrite( temp_path +  "/%05d.png"%i, rgb_frame) 
    if video_path != None: 
        command = 'ffmpeg -framerate 25 -i '  + temp_path + '/%5d.png  -c:v libx264 -y -vf format=yuv420p ' +  orignal_ani_path
    else:
        command = 'ffmpeg -framerate 25 -i '  + temp_path + '/%5d.png  -c:v libx264 -y -vf format=yuv420p ' +   orignal_ani_path
    os.system(command)
   

#create demo data
def demo_data(opt, video_path, reference_id, mode, ani_video_path, reference_img_path):
    output_shape   = tuple([opt.loadSize, opt.loadSize])
    num_frames = opt.num_frames
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), inplace=True)])
    v_id = video_path[:-4]
    rt_path = video_path[:-4] + '_sRT.npy'
    lmark_path = video_path[:-4] + '.npy'
    rt = np.load(rt_path)[:,:3]
    lmark = np.load(lmark_path)[:,:,:-1]
    v_length = lmark.shape[0]
    target_ids = []
    for gg in range(v_length):
        target_ids.append(gg)
    if mode == 0:
        
        ani_video_path =video_path[:-4] + '_ani.mp4'
        
        real_video  = mmcv.VideoReader(video_path)
        ani_video = mmcv.VideoReader(ani_video_path)
        # sample frames for embedding network
        if opt.use_ft:
            if num_frames  ==1 :
                input_indexs = [0]
                target_id = 0
                reference_id = 0
            elif num_frames == 8:
                input_indexs = [0,7,15,23,31,39,47,55]

            elif num_frames == 32:
                input_indexs = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63]
        else:
            print (num_frames)
            input_indexs = random.sample(range(0,64), num_frames)
            input_indexs  = set(input_indexs)

            # we randomly choose a target frame 
            # while True:
            #     target_id =  np.random.choice( v_length - 1)
            #     if target_id not in input_indexs:
            #         break
        reference_rts = np.zeros((num_frames, 3))
        # we randomly choose a target frame 
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
    else:
        real_video  = mmcv.VideoReader(video_path)
        ani_video = mmcv.VideoReader(ani_video_path)
        reference_frames = torch.zeros(1, 6 ,output_shape[0],output_shape[1])
        reference_img = cv2.imread(reference_img_path)
        lmark_t = fa.get_landmarks(reference_img)[0][:,:-1]
        rgb_t =  mmcv.bgr2rgb(reference_img) 
        lmark_rgb = plot_landmarks( lmark_t)
        reference_rts = np.zeros((1, 3))
    
        # lmark_rgb = np.array(lmark_rgb) 
        # resize 224 to 256
        rgb_t  = cv2.resize(rgb_t, output_shape)
        lmark_rgb  = cv2.resize(lmark_rgb, output_shape)
        # to tensor
        rgb_t = transform(rgb_t)
        lmark_rgb = transform(lmark_rgb)
        reference_frames[0] = torch.cat([rgb_t, lmark_rgb],0) # (6, 256, 256)   
        reference_rts[0] = rt[0]


    input_dics = []
    similar_frame = torch.zeros( 3, output_shape[0], output_shape[0])    
    reference_frames = torch.unsqueeze( reference_frames, 0)  
    ############################################################################
    for target_id in target_ids:
        reference_rt_diff = reference_rts - rt[target_id]
        reference_rt_diff = np.absolute(reference_rt_diff)
        r_diff = np.mean(reference_rt_diff, axis =1)
        similar_id  = np.argmin(r_diff) 
        similar_frame = reference_frames[0,similar_id,:3]

        target_rgb = real_video[target_id]
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
        cropped_similar_image = similar_frame.clone()
        # mask = target_ani > -0.9
        # mask = scipy.ndimage.morphology.binary_dilation(mask.numpy(),iterations = 5).astype(np.bool)
        # cropped_similar_img[torch.tensor(mask)] = -1 
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
    return input_dics

def lrw_img2vid(path):
    if os.path.exists('./tmp'):
        shutil.rmtree('./tmp')
    os.mkdir('./tmp')
    
            
    v_ids = os.listdir(path)
    for v_id in v_ids:
        v_path = os.path.join(path, v_id)
        print (v_path)
        count = 0
        for i_path in os.listdir(v_path):
            if i_path[-3:] =='mp4':
                continue
            img_path = os.path.join(v_path, i_path)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (224,224), interpolation = cv2.INTER_AREA)
            cv2.imwrite('./tmp/%05d.png'%count, img)
            count += 1


        video_path = os.path.join(v_path, v_id + '.mp4')
        command = 'ffmpeg -framerate 25  -i ' +   './tmp/%05d.png  -vcodec libx264 -y -vf format=yuv420p '  + video_path 
        for i_path in os.listdir(v_path):
            os.remove(os.path.join(v_path, i_path))
        os.system(command)

        

def convert_folder(path):
    gg_list = []
    print (path)
    print ('====================')
    print ('***********++++*******')
    v_ids = os.listdir(path)
    print (v_ids)
    root = '/mnt/Backup/lchen63/demo_videos'
    for v_id in v_ids:
        v_path = os.path.join(path, v_id)

        print (v_path)
        for i_path in os.listdir(v_path):
            print (i_path)
            if i_path[-3:] == 'mpg' or i_path[-3:] == 'mp4' or i_path[-3:] == 'avi':
                if 'ani' not in i_path :
                    if os.path.exists('./tmp'):
                        shutil.rmtree('./tmp')
                    os.mkdir('./tmp')
                    # try:
                    video_path = os.path.join(v_path, i_path)
                    print (video_path)
                    # _crop_video(video_path)
                    # video_path = video_path[:-4] +'_crop.' + i_path[-3:]
                    _video2img2lmark(video_path)
                    compute_RT(video_path)
                    key_id, video_path  = compose_front(video_path)
                    gg_list.append([video_path, key_id])
                # except:
                    # print ('fuck')
                    # continue
    print (gg_list)
    with open(os.path.join( root, 'txt','vox_audio' + '_key_frame.pkl'), 'wb') as handle:
        pickle.dump(gg_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

def get_animation_pickle():
    root = '/mnt/Backup/lchen63/demo_videos'
    pkl_name = os.path.join( root , 'txt','vox_audio' +  '_key_frame.pkl')
    _file = open(pkl_name, "rb")
    gggdata = pickle.load(_file)
    _file.close()
    for gg in gggdata:        
        v_path = gg[0]
        reference_id = gg[1]
        try:
            get_animation_orignal(video_path = v_path,key_id = reference_id)
        except:
            print ('++++++')
            continue
    # key_id = None, video_path = None, rt_path= rt_path, lmark_path= lmark_path )




def convert_folde2test(path):
    gg_list = []
    print (path)
    print ('====================')
    v_ids = os.listdir(path)
    for v_id in v_ids:
        ggg = os.path.join(path.split('/')[-1], v_id )
        v_path = os.path.join(path, v_id)
        for i_path in os.listdir(v_path):
            if i_path[-3:] == 'mp4' and '_ani' in i_path:
                ggg =os.path.join(ggg, i_path[:-8] )
                gg_list.append([ggg])
    print (gg_list)
    with open(os.path.join( '/home/cxu-serve/p1/lchen63/voxceleb/txt','demo.pkl'), 'wb') as handle:
        pickle.dump(gg_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

# lrw_img2vid('/home/cxu-serve/p1/lchen63/voxceleb/unzip/lrw_test')
# convert_folde2test('/home/cxu-serve/p1/lchen63/voxceleb/unzip/vox1_test')
root = '/mnt/Backup/lchen63/demo_videos'
# convert_folder( os.path.join(root, 'addition_example'))
get_animation_pickle()
# root = '/home/cxu-serve/p1/lchen63/voxceleb'
# root = '/data2/lchen63/voxceleb'

# v_id = 'test_video/id00061/cAT9aR8oFx0/00141'
# reference_id = 542

# def test():
# if not os.path.exists('./demo' ):
#     os.mkdir('./demo')
# if not os.path.exists( os.path.join('./demo', opt.name)  ):
#     os.mkdir(os.path.join('./demo', opt.name))


# target_img_path = '/home/cxu-serve/p1/lchen63/voxceleb/unzip/demo_video/id00017/01dfn2spqyE/lisa.jpg'

# cropped_img, _ = crop_image(target_img_path)
# cv2.imwrite(target_img_path[:-4] +'_crop.jpg', cropped_img)
############# then you need to use PRnet to generate 3d  
## exaple: go to PRnet folder, python get_3d.py --img_path

# prnet_lmark = fa.get_landmarks(cropped_img)



#change frame rate to 25FPS
# v_path = '/home/cxu-serve/p1/lchen63/voxceleb/unzip/test_video1/id10271/1gtz-CUIygI/0001_1.mp4'
# command = 'ffmpeg -i ' +  opt.v_path +   ' -r 25 -y  ' + opt.v_path[:-4] + '_fps25.mp4'
# os.system(command)
##################################################
# if opt.v_path == '/home/cxu-serve/p1/lchen63/voxceleb/unzip/demo_video/lele.MOV':
# opt.v_path =  opt.v_path[:-4] + '_fps25.mp4'
########################################
# opt.v_path = opt.v_path[:-4] + '_crop.mp4'
#     key_id = 743
# v_path = v_path[:-4] +'_crop.mp4'
# v_path = '/home/cxu-serve/p1/lchen63/voxceleb/unzip/lrw_test/ABOUT_00001/ABOUT_00001.mp4'
# # _crop_video(v_path)
# v_path = v_path[:-4] +'_crop.mp4'

# _video2img2lmark(v_path)
# compute_RT(v_path)
# key_id, video_path  = compose_front(v_path)

#convert_folder('/home/cxu-serve/p1/lchen63/voxceleb/unzip/vox1_test')

# # then you need to use PRnet to generate 3d  
# # exaple: go to PRnet folder, python get_3d.py --v_path /home/cxu-serve/p1/lchen63/voxceleb/unzip/demo_video/lele_fps25_crop.mp4 --target_id 743
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
# get_animation(video_path = v_path,key_id = 7)


# dataset = demo_data(opt =opt, video_path = video_path, reference_id = None,mode = 1,ani_video_path = ani_video_path, reference_img_path = img_path)
# print (len(dataset))
# model = create_model(opt)
# print(model)


 
