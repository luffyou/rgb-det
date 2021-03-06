#coding:utf8
import os
import sys
from PIL import Image
import torch as t
from torch.utils import data
import numpy as np
from torchvision import transforms as T
sys.path.append('../') # base_root
from utils.data_util import *


class Kitti(data.Dataset):    
    def __init__(self,root='/hdd/you/rgbd-det/',transforms=None,
             white_list=[ 'Car', 'Pedestrian', 'Cyclist'],sets_type='train',process='train'):
        '''
        root:set in rgbd nor rgb
        obj_type:all\car\cyc\ped
        sets_type:train\trainval\val\test for dataset
        process:train\val\test\ for train val or inference (no influence actually)
        root = '/hdd/you/rgbd-det/'
        '''
        idx_path = os.path.join(root, 'data/image_sets/{}.txt'.format(sets_type)) # 3types
        if sets_type is not 'test': sets_type = 'trainval' # all train & val in here 2 types
        # full_img_dir = os.path.join(root, 'dataset/{}/image_2'.format(sets_type)) # full image
        rgb_dir = os.path.join(root, 'dataset/{}/rgb'.format(sets_type)) # crop image
        depth_dir = os.path.join(root, 'dataset/{}/depth'.format(sets_type)) # crop depth image
        if sets_type is not 'test': # label or detection result
            sets_type = 'trainval/label'
        else:
            sets_type = 'test/det'
        text_path = os.path.join(root, 'dataset/{}'.format(sets_type))

        idx_fig = open(idx_path, 'r')
        idx_file = idx_fig.readlines()
        idx_fig.close()
        
        self.obj_types = {'Car':'0', 'Pedestrian':'1', 'Cyclist':'2'}
        outcasts = [] # path
        rgbs = [] # path 
        depths = []
        labels = [] # value; return img id when testing

        for idx in idx_file:
            idx = idx.strip('\n\r')
            txt = open(os.path.join(text_path,idx+'.txt'),'r')
            lines = txt.readlines()
            txt.close()
            for num,line in enumerate(lines):
                label = {} # init every time
                line = line.strip('\n\r').split(' ')
                obj_type = line[0]
                if obj_type not in white_list:
                    continue
                    
                lib = [obj_type] + list(map(float, line[1:]))
                obj = self.obj_types[obj_type]
                target = '{}.{}.{}.png'.format(idx,num,obj)
                
                if (lib[6]-lib[4])<=8 or (lib[7]-lib[5])<=8: # bad data
                    outcasts.append(os.path.join(rgb_dir, target))
                    continue
                    
                rgbs.append(os.path.join(rgb_dir, target))
                depths.append(os.path.join(depth_dir, target))
                #　full_img = Image.open(os.path.join(full_img_dir,idx+'.png')) # for img size norm
                # width, height = full_img.size[0] , full_img.size[1] # for img size norm
                # label['2d'] = [lib[4]/width, lib[5]/height, lib[6]/width, lib[7]/height] # for adding vector
                label['2d'] = [lib[4],lib[5],lib[6],lib[7]]
                if sets_type is not 'test':
                    label['loc'] = [lib[11], lib[12], lib[13]]
                    label['hwl'] = [lib[8], lib[9], lib[10]] # for iou
                    label['ry'] = lib[14]
                else:
                    label['loc'] = list(map(float, [idx, num, obj])) # return img ID when testing for wt res
                    label['hwl'] = [-1000, -1000, -1000] # for iou
                    label['ry'] = [-10] # for getting a value
                labels.append(label)
                
        self.idx_path = idx_path # 数据集索引文件
        self.rgb_dir = rgb_dir # 图片目录
        self.depth_dir = depth_dir # 深度图目录
        self.text_path = text_path # label目录、2D检测结果目录
        self.rgbs = rgbs # necessary 图片路径列表
        self.depths = depths # necessary 深度图路径列表
        self.labels = labels # necessary 标签具体数值
        self.outcasts = outcasts # 不满足要求的图片路径（尺寸过小）
        self.process = process # 指示当前是训练、验证、还是测试（当前并不影响具体代码）
        self.transforms = transforms # 用于预处理data
        if transforms is None:
            self.transforms = T.Compose([
                T.Resize((224,224)),
                T.ToTensor(), # to CHW ; /255 ; diff with rgb(3 channel 255 max) & depth(1channel unknow max) 
                # is it necessary? the value needs to be determined
                T.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
                ]) 
            
             
    def __getitem__(self,index):
        # rgb_path = self.rgbs[index]
        # rgb = Image.open(rgb_path)
        # rgb = self.transforms(rgb)*255 # WxHx3 ; is 255 necessary
        # rgb = self.transforms(rgb) # test_img_only!!!
        
        # test_img_only!!!
        # depth_path = self.depths[index]
        # depth = self.transforms(Image.open(depth_path)) # WxH  may need refer to kitti depth_devkit  
        # depth = depth.float() / 256.  # to be determined
                
        # data = t.cat((rgb,depth), dim=0) 
        # data = rgb # test_img_only!!!
        # print(data.shape) # Tensor CxWxH C=4; will be unsqueezed to BxCxWxH after dataloader
        
        label = self.labels[index]
        label_2d = label['2d'] # 4x1x1 torch.Size([4, 1, 1])for adding in channel
        label_loc = label['loc'] # 3 torch.Size([3])
        label_hwl = label['hwl'] # 3 torch.Size([3]) for iou
        label_ry = label['ry'] # heading angle class; heading angle residual(normalized); in utils
        
        return label, label_2d, label_loc, label_hwl, label_ry
    
    def __len__(self):
        return len(self.rgbs)
        # return 4
    
    
if __name__ == '__main__':
    idx = 1
    train_data = Kitti('/hdd/you/rgbd-det/',sets_type='trainval',process='train',white_list=[ 'Cyclist'])
    label, label_2d, label_loc, label_hwl, label_ry = train_data.__getitem__(idx) # change idx
    # print(data, 'Data_shape:', data.shape) # rgb add depth
    print('Label_2d:',label_2d, 'Label_loc:',label_loc, sep='\n')
    print('Sample_num:'+str(train_data.__len__()))
    
    bad_img_path = train_data.outcasts[0] # change idx
    bad_img_path = bad_img_path.strip('\n\r').split('/')[-1]
    bad_img = Image.open('/hdd/you/rgbd-det/dataset/trainval/rgb/'+bad_img_path)
    print('\nSmall_img_num:'+str(len(train_data.outcasts)), 'e.g:'+bad_img_path, np.array(bad_img).shape)
    
    print('Content')
    for i in range(3):
        print(train_data.rgbs[i])
        print(train_data.depths[i])
        print(train_data.labels[i])
    # test = '/hdd/you/dataset/KITTI/training/image_2/000000.png'
    # print(np.array(Image.open(test)))