import os
command1 = 'cd /u/lchen63/github/pix2pixHD'

# os.system(command)

command = 'cd /u/lchen63/github/pix2pixHD & CUDA_VISIBLE_DEVICES=6 python test.py --name base1_no_pix_att_train_8 --num_frames 8 --how_many 2000 --checkpoints_dir /home/cxu-serve/p1/lchen63/voxceleb/test_demo/checkpoints'
os.system(command)



command = 'cd /u/lchen63/github/insightface/deploy'
# os.system(command)

command = 'cd /u/lchen63/github/insightface/deploy & CUDA_VISIBLE_DEVICES=6 python test.py --path ../results/base1_no_pix_att_train_8/test_latest/images'
os.system(command)

command = 'cd /u/lchen63/github/pix2pixHD/TTUR'
# os.system(command)
command = 'cd /u/lchen63/github/pix2pixHD/TTUR & CUDA_VISIBLE_DEVICES=6 python fid.py --path ../results/base1_no_pix_att_train_8/test_latest/images'
os.system(command)
