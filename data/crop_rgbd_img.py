import os
import string
from PIL import Image
import numpy

# car:0 ped:1 cyc:2
# obj_types = ['car', 'pedestrian', 'cyclist', 'all']
set_type = 'train' # 'test' // train for trainval 
rgb_dir = '/hdd/you/dataset/KITTI/'+set_type+'ing/image_2/'
depth_dir = '/hdd/you/DORN/result/KITTI_'+set_type
label_dir = '/hdd/you/dataset/KITTI/'+set_type+'ing/label_2/'

dst =  'trainval' # 'test' 
dst_rgb_dir = '/hdd/you/rgbd-det/dataset/'+dst+'/rgb/'
dst_depth_dir = '/hdd/you/rgbd-det/dataset/'+dst+'/depth/'

label_list = os.listdir(label_dir)
# print(label_list[0])
count = 0
for label_path in label_list:
# for label_path in ['004593.txt']: # '001053.txt-2', '004569.txt-3'
    rgb_path = os.path.join(rgb_dir, label_path.replace('.txt', '.png'))
    depth_path = os.path.join(depth_dir, label_path.replace('.txt', '_depth.png'))
    dst_rgb_base = os.path.join(dst_rgb_dir, label_path.strip('txt'))
    dst_depth_base = os.path.join(dst_depth_dir, label_path.strip('txt'))

    with open(os.path.join(label_dir, label_path), 'r') as lab_f:
        count += 1
        if count % 1000 == 0:
            print('Processing No.',count, label_path)
        rgb = Image.open(rgb_path, 'r')
        depth = Image.open(depth_path, 'r')
        for idx,eachline in enumerate(lab_f): 
            eachline = eachline.strip('\r\n').split(' ')
            obj_type = eachline[0]
            if obj_type == 'Car':
                obj_idx ='.0'
            elif obj_type == 'Pedestrian':
                obj_idx = '.1'
            elif obj_type == 'Cyclist':
                obj_idx = '.2'
            else: continue # note not break!
            x1,y1,x2,y2 = eachline[4:8]
            x1,y1,x2,y2 = float(x1), float(y1), float(x2), float(y2)
            # print(obj_type, x1, y1, x2, y2)
            try:
                dst_rgb_path = dst_rgb_base+str(idx)+obj_idx+'.png'
                # print(dst_rgb_path)
                if not os.path.exists(dst_rgb_path):
                    rgb_roi = rgb.crop((x1,y1,x2,y2))                
                    rgb_roi.save(dst_rgb_path)
                    # print('rgb', numpy.array(rgb_roi).shape)
                dst_depth_path = dst_depth_base+str(idx)+obj_idx+'.png'
                # print(dst_depth_path)
                if not os.path.exists(dst_depth_path):
                    depth_roi = depth.crop((x1,y1,x2,y2))                
                    depth_roi.save(dst_depth_path)
                    # print('depth', numpy.array(depth_roi).shape) 
            except Exception:
                print('Error:label',label_path,idx)        
    # break
print('Finished')