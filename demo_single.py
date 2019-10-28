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
import dlib
import face_alignment
from scipy.spatial.transform import Rotation 
from util import utils
import time
from numpy import *
from scipy.spatial.transform import Rotation as R
res = 224
import soft_renderer as sr

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('./basics/shape_predictor_68_face_landmarks.dat')



opt = TestOptions().parse(save=False)
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle


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


def crop_image(image, id = 1, kxy = []):
#     kxy = []
    if id == 1:
        image = cv2.imread(image)
    if kxy != []:
        [k, x, y] = kxy
        roi = image[y - int(0.2 * k):y + int(1.6 * k), x- int(0.4 * k):x + int(1.4 * k)]
        roi = cv2.resize(roi, (224,224), interpolation = cv2.INTER_AREA)
        return roi, kxy 
    else:        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 1)
        for (i, rect) in enumerate(rects):
            shape = predictor(gray, rect)
            shape = utils.shape_to_np(shape)
            (x, y, w, h) = utils.rect_to_bb(rect)
            center_x = x + int(0.5 * w)
            center_y = y + int(0.5 * h)
            k = min(w, h)
            roi = image[y - int(0.2 * k):y + int(1.6 * k), x- int(0.4 * k):x + int(1.4 * k)]
            roi = cv2.resize(roi, (224,224), interpolation = cv2.INTER_AREA)
            return roi ,[k,x,y]

fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, flip_input=False)#,  device='cpu')


def _crop_video(video):
    cap  =  cv2.VideoCapture(video)
    count = 0
    kxy =[]
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            count += 1
            print (count)
            img,kxy = crop_image(frame, 0,kxy)
            cv2.imwrite('./tmp/%04d.png'%count, img)
        else:
            break
    command = 'ffmpeg -framerate 25  -i ' +   './tmp/%04d.png  -vcodec libx264 -y -vf format=yuv420p ' + video[:-4] + '_crop.mp4' 
    os.system(command)

def get3DLmarks_single_image(image_path):
#     img = crop_image(image_path)
    img = cv2.imread(image_path)
    lmark = fa.get_landmarks(img)        
    np.save(image_path[:-4] + '.npy', lmark)



def get3DLmarks(frame_list, v_path):
    frame_num = len(frame_list)
    lmarks = np.zeros((frame_num, 68,3))
    for i in range(frame_num):
        lmark = fa.get_landmarks(frame_list[i])        
        if lmark is not None:
            landmark =  lmark[0]
        else:
            landmark = -np.ones((68, 3))
        lmarks[i] = landmark
    np.save(v_path[:-4] + '.npy', lmarks)

def _video2img2lmark(v_path):

    count = 0    
    frame_list = []
   
    cap  =  cv2.VideoCapture(v_path)
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:            
            frame_list.append(frame)
            count += 1
        else:
            break
    get3DLmarks(frame_list, v_path)


def rigid_transform_3D(A, B):
    assert len(A) == len(B)

    N = A.shape[0]; # total points

    centroid_A = mean(A, axis=0)
    centroid_B = mean(B, axis=0)
    
    # centre the points
    AA = A - tile(centroid_A, (N, 1))
    BB = B - tile(centroid_B, (N, 1))

    # dot is matrix multiplication for array
    H = transpose(AA) * BB

    U, S, Vt = linalg.svd(H)

    R = Vt.T * U.T

    # special reflection case
    if linalg.det(R) < 0:
        print("Reflection detected")
        Vt[2,:] *= -1
        R = Vt.T * U.T
    
    t = -R*centroid_A.T + centroid_B.T

    return R, t

def compute_RT(video_path):
    consider_key = [1,2,3,4,5,11,12,13,14,15,27,28,29,30,31,32,33,34,35,39,42,36,45,17,21,22,26]
    print (len(consider_key))
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
        A2+= tile(ret_t, (1, 68))
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
        A2+= tile(ret_t, (1, 68))
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
    v_path =  v_path[:-4] + '.mp4' 
    rt = np.load( rt_path)
    lmark = np.load(o_lmark)
    lmark_length = lmark.shape[0]
    find_rt = []
    for t in range(0, lmark_length):
        find_rt.append(sum(np.absolute(rt[t,:3])))

    find_rt = np.asarray(find_rt)

    min_index = np.argmin(find_rt)
    print (v_path)
    print (min_index)
    return min_index, v_path



def get_animation(key_id, video_path ):
    reference_img_path = video_path[:-4] + '_%05d.png'%key_id

    reference_prnet_lmark_path = video_path[:-4] +'_prnet.npy'

    original_obj_path = video_path[:-4] + '_original.obj'

    rt_path  = video_path[:-4] + '_sRT.npy'
    lmark_path  = video_path[:-4] +'_front.npy'


    if os.path.exists( video_path[:-4] + '_ani.mp4'):
        print ('=====')
    if  not os.path.exists(original_obj_path) or not os.path.exists(reference_prnet_lmark_path) or not os.path.exists(lmark_path) or not os.path.exists(rt_path):
        print (os.path.exists(original_obj_path) , os.path.exists(reference_prnet_lmark_path),  os.path.exists(lmark_path), os.path.exists(rt_path))
        print (original_obj_path)
        print ('++++')
    # extract the frontal facial landmarks for key frame
    lmk3d_all = np.load(lmark_path)
    lmk3d_target = lmk3d_all[key_id]


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
    vertices_org, triangles, colors = load_obj(original_obj_path) # get unfrontalized vertices position
    vertices_origin_affine = (pr @ (vertices_org.T) + pt).T # aligned vertices

    # set up the renderer
    renderer = setup_renderer()
    # generate animation

    temp_path = './tempp_%05d'%1

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
    command = 'ffmpeg -framerate 25 -i '  + temp_path + '/%5d.png  -c:v libx264 -y -vf format=yuv420p ' +   video_path[:-4] + '_ani.mp4'
    os.system(command)
   

#create demo data
def demo_data(video_path, reference_id):
    output_shape   = tuple([opt.loadSize, opt.loadSize])
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), inplace=True)])
    num_frames = opt.num_frames
    ani_video_path =video_path[:-4] + '_ani.mp4'
    rt_path = video_path[:-4] + '_sRT.npy'
    lmark_path = video_path[:-4] + '.npy'
    rt = np.load(rt_path)[:,:3]
    lmark = np.load(lmark_path)[:,:,:-1]
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

        elif self.num_frames == 32:
            input_indexs = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63]
            target_id =  random.sample(input_indexs, 1)
    else:
        input_indexs  = set(random.sample(range(0,64), self.num_frames))

        # we randomly choose a target frame 
        while True:
            target_id =  np.random.choice( v_length - 1)
            if target_id not in input_indexs:
                break
    reference_frames = []
    reference_rt_diffs = []

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
    cropped_similar_image[target_ani> -0.9] = -1 


    input_dic = {'v_id' : v_id, 'target_lmark': target_lmark, 'reference_frames': reference_frames, \
    'target_rgb': target_rgb, 'target_ani': target_ani, 'reference_ids':str(input_indexs), 'target_id': target_id \
    , 'similar_frame': similar_frame, "cropped_similar_image" : cropped_similar_image}
    return input_dic

# root = '/home/cxu-serve/p1/lchen63/voxceleb'
root = '/data2/lchen63/voxceleb'

# v_id = 'test_video/id00061/cAT9aR8oFx0/00141'
# reference_id = 542

# def test():
if not os.path.exists('./demo' ):
    os.mkdir('./demo')
if not os.path.exists( os.path.join('./demo', opt.name)  ):
    os.mkdir(os.path.join('./demo', opt.name))

#change frame rate to 25FPS
command = 'ffmpeg -i ' +  opt.v_path +   ' -r 25 -y  ' + opt.v_path[:-4] + '_fps25.mp4'
# os.system(command)
opt.v_path =  opt.v_path[:-4] + '_fps25.mp4'
# _crop_video(opt.v_path)
opt.v_path = opt.v_path[:-4] + '_crop.mp4'
# _video2img2lmark(opt.v_path)
# compute_RT(opt.v_path)
## then you need to use PRnet to generate 3d  
## exaple: go to PRnet folder, python get_3d.py --v_path /home/cxu-serve/p1/lchen63/voxceleb/unzip/demo_video/lele_fps25_crop.mp4 --target_id 743
# key_id, video_path  = compose_front(opt.v_path)
# get_animation(key_id, video_path )
key_id = 743

model = create_model(opt)
if opt.verbose:
    print(model)
    dataset = demo_data(opt.v_path, key_id)
    if not os.path.exists( os.path.join('./demo', opt.name, v_id.split('/')[-2] + '_' + v_id.split('/')[-1])  ):
        os.mkdir(os.path.join('./demo', opt.name, v_id.split('/')[-2]+ '_' +  v_id.split('/')[-1]))
    save_path = os.path.join('./demo', opt.name,v_id.split('/')[-2]+ '_' +  v_id.split('/')[-1])

    for i, data in enumerate(dataset):
        # if i >= opt.how_many:
        #     break
        minibatch = 1 
        if opt.no_ani:
            generated = model.inference(Variable(data['reference_frames']), Variable(data['target_lmark']), \
           None,  Variable(data['target_rgb']), Variable('similar_frame'), Variable(data['cropped_similar_image'] ))
        else:
            generated = model.inference(Variable(data['reference_frames']), Variable(data['target_lmark']), \
          Variable(data['target_ani']),  Variable(data['target_rgb']), Variable('similar_frame'), Variable(data['cropped_similar_image'] ))
        
        img = torch.cat([generated[0].data.cpu(), data['target_rgb']], 0)
        torchvision.utils.save_image(img, 
                    "{}/{:05d}.png".format(save_path,i),normalize=True)



# losses, generated = model(references =Variable(data['reference_frames']),target_lmark= Variable(data['target_lmark']),target_ani=   Variable(data['target_ani']),real_image=  Variable(data['target_rgb']), similar_frame = Variable(data['similar_frame']), infer=save_fake)
 