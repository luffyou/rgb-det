#coding:utf8
import os
import numpy as np
from PIL import Image
from utils.label_util import *
from utils.locate_util import *

# in this version we didn't use dataset.py
def relocate_per_file(label_file_path, img_file_path, calib_file_path):
    calib = Calibration(calib_file_path)
    lines = [line.rstrip() for line in open(label_file_path)]
    objLibs = []
    objDets = [] 
    for line in lines:
        objLibs.append(Object3d(line))
        objDets.append(Object3d(line,label=True))
        
    for idx in range(len(objDets)):
        img = Image.open(img_file_path)
        imgWid, imgHei = img.size[0] , img.size[1] 
        objDets[idx] = computeLocation(objDets[idx], objLibs[idx], calib.P, calib.P, imgWid, imgHei)
    for idx in range(len(objDets)):
        objDets[idx].print_object()
        objLibs[idx].print_object()
    return objDets
    
         
if __name__=='__main__':
    relocate_per_file('/hdd/you/rgbd-det/dataset/trainval/label/007480.txt',
                '/hdd/you/rgbd-det/dataset/trainval/image_2/007480.png',
                '/hdd/you/rgbd-det/dataset/trainval/calib/007480.txt')
    # 需要将预测和gt对应起来,使用IOU?