import os
import numpy as np
from . label_util import *

# 读label，正向求一次对应；反向求loc
def computeLocation(objDet,objLib,P_d,P_l,imgWid,imgHei):
    # P_l: calib for lib
    # P_d: calib for det
    val,idx = computeBoxMatch(objLib,P_l) # match_idx 用于索引3D box矩阵

    objDet.h = objLib.h
    objDet.w = objLib.w
    objDet.l = objLib.l # 待修改：给dimension添加噪声
    objDet.ry = objLib.ry # 待修改：给rotation_y添加噪声
    # objDet.ry = -objDet.ry; % 验证ry沿着y轴从上往下看顺时针为正方向

    # 在2D结果occl默认为-1
    #　绘图需要，否则drawBox2D中的occ_col{object.occlusion+1}索引会出错
    # objDet.occlusion = objLib.occlusion;
    # objDet.score = 1; % 当使用gt_2D,score为1,当使用rrc结果，score无需改变
    # objDet.truncation = objLib.truncation;

    # solve the location with match_idx
    theta = objDet.ry
    leng = objDet.l/2
    wid = objDet.w/2
    hei = objDet.h #  can be changed to a fixed num
    X = [leng, leng, -leng, -leng, leng, leng, -leng, -leng]
    Y = [0,0,0,0,-hei,-hei,-hei,-hei]
    Z = [wid, -wid, -wid, wid, wid, -wid, -wid, wid]
    a = P_d[0,0]
    b = P_d[1,1]
    u0 = P_d[0,2]
    v0 = P_d[1,2]
    k1 = P_d[0,3]
    k2 = P_d[1,3]
    k3 = P_d[2,3]

    #下列if判断是为了处理截断的2D图像进
    # 另一种方式，补齐训练集，重新训练神经网络，生成的结果允许在图片外
    # 对截断图片根据画面比例补齐。初步验证det为主导，结果稍稍好一点点
    # 先缓存原来的lib值，用于后面以lib为主导时，判断是否需要补齐det
    # cache_l, cache_r, cache_u, cache_d = objLib.x1, objLib.x2, objLib.y1, objLib.y2
    lib_l, lib_r, lib_u, lib_d = val.l, val.r, val.u, val.d # 训练集中的匹配对象补全
    # 补全比直接采用2D点来的有意义：1.截断部分补全 2.pedestrian的2D与3D并不对应
    det_l, det_r, det_u, det_d = objDet.xmin, objDet.xmax, objDet.ymin, objDet.ymax # 目标对象的2D点
    # imgWid = 1242; 图片像素不一定统一
    # imgHei = 375
    if det_l>0 and det_r<imgWid and det_d<imgHei: # 无截断,完整
    # if cache_l>0 && cache_r<imgWid && cache_d<imgHei
        u_l, u_r, v_u, v_d = det_l, det_r, det_u, det_d
        # print('full')
    elif det_r==imgWid and det_d<imgHei: # 右中部
    # % elseif cache_r==imgWid && cache_d<imgHei
        tmp_r = det_l+(det_u-det_d)*(lib_r-lib_l)/(lib_u-lib_d)
        u_r = max(tmp_r,det_r) # 补齐后不应该小于box
        u_l, v_u, v_d = det_l, det_u, det_d
        # print('right middle')
    elif det_l==0 and det_d<imgHei:  #  左中部
    # % elseif cache_l==0 && cache_d<imgHei
        tmp_l = det_r-(det_u-det_d)*(lib_r-lib_l)/(lib_u-lib_d)
        u_l = min(tmp_l,det_l)
        u_r, v_u, v_d = det_r, det_u, det_d
        # print('left middle')
    elif det_l==0 and det_d==imgHei:  # 左下部 无法判断比例，只能强制det大小和lib一致
    # elseif cache_l==0 && cache_d==imgHei
        tmp_l = det_r-(lib_r-lib_l)
        u_l = min(tmp_l,det_l)
        tmp_d = det_u-(lib_u-lib_d)
        v_d = max(tmp_d,det_d)
        u_r, v_u = det_r, det_u
        # print('left down')
    elif det_r==imgWid and det_d==imgHei:  # 右下部
    # elseif cache_r==imgWid && cache_d==imgHei
        tmp_r = det_l+(lib_r-lib_l)
        u_r = max(tmp_r,det_r)
        tmp_d = det_u-(lib_u-lib_d)
        v_d = max(tmp_d,det_d)
        u_l, v_u = det_l, det_u
        # print('right down')
    else:  # 因为其他情况不可能，只剩中下部
        tmp_d = det_u-(det_r-det_l)*(lib_u-lib_d)/(lib_r-lib_l)
        v_d = max(tmp_d,det_d)
        u_l, u_r, v_u = det_l, det_r, det_u
        # print('middle down')
    # lib = [lib_l, lib_u, lib_r, lib_d]
    # det_before = [det_l, det_u, det_r, det_d]
    # det_after = [u_l, v_u, u_r, v_d]

    A = np.array([[-a, 0, (u_l-u0)],
              [-a, 0, (u_r-u0)],
              [0, -b, (v_u-v0)],
              [0, -b, (v_d-v0)]])
    co = np.cos(theta)
    si = np.sin(theta)
    # left  about u; A1
    m_l = co*X[idx.l]+si*Z[idx.l]
    n_l = -si*X[idx.l]+co*Z[idx.l]
    res1 = a*m_l+u0*n_l+k1-u_l*(n_l+k3)
    # right  about u; A2
    m_r = co*X[idx.r]+si*Z[idx.r]
    n_r = -si*X[idx.r]+co*Z[idx.r]
    res2 = a*m_r+u0*n_r+k1-u_r*(n_r+k3)
    # up  about v; A3
    y_u = Y[idx.u]
    n_u = -si*X[idx.u]+co*Z[idx.u]
    res3 = b*y_u+v0*n_u+k2-v_u*(n_u+k3)
    # down about v; A4
    y_d = Y[idx.d]
    n_d = -si*X[idx.d]+co*Z[idx.d]
    res4 = b*y_d+v0*n_d+k2-v_d*(n_d+k3)

    res = np.array([res1, res2, res3, res4]) # 4*1
    # T = (A'*A)\A'*res
    T = np.linalg.inv(A.T.dot(A)).dot(A.T).dot(res.T)
    objDet.t[0], objDet.t[1], objDet.t[2] = T[0], T[1], T[2] # 完成location的更新

    #  完成alpha的更新 存在关系obj.ry = obj.alpha + atan(obj.t(1)/obj.t(3)); 
    # 与ratation_ry的区别在于同时考虑相机中心到物体中心 
    objDet.alpha = objLib.ry - np.arctan(objDet.t[0]/objDet.t[2])
    return objDet

def computeBoxMatch(obj,P):
    # 用于lib库中待检索对象的计算：3D-》2D，根据投影模型计算出的2D点比较大小，确定3D和2D的匹配模式
    # index for 3D bounding box faces; front,left,back,right
    face_idx = np.array([[1,2,6,5],[2,3,7,6],[3,4,8,7],[4,1,5,8]])
    R = roty(obj.ry) 
    # 3D bounding box dimensions
    l = obj.l/2
    w = obj.w/2
    h = obj.h # note without /2
    # 3D bounding box corners
    x_corners = np.array([l, l, -l, -l, l, l, -l, -l])
    y_corners = np.array([0,0,0,0,-h,-h,-h,-h])
    z_corners = np.array([w, -w, -w, w, w, -w, -w, w])
    # rotate and translate 3D bounding box
    corners_3D = np.dot(R, np.vstack([x_corners,y_corners,z_corners])) # 3*8
    # order by 1 2 3 ...
    corners_3D[0,:] = corners_3D[0,:] + obj.t[0]
    corners_3D[1,:] = corners_3D[1,:] + obj.t[1]
    corners_3D[2,:] = corners_3D[2,:] + obj.t[2]
    # project the 3D bounding box into the image plane
    # corners_2D = P * [corners_3D; ones(1,size(corners_3D,2))];
    corners_2D = project_to_image(np.transpose(corners_3D), P) # nx3
    # find the max & min in 2D_x & 2D_y
    l_val, l_idx = np.min(corners_2D[:,0]), np.argmin(corners_2D[:,0]) # left
    r_val, r_idx = np.max(corners_2D[:,0]), np.argmax(corners_2D[:,0]) # right
    u_val, u_idx = np.min(corners_2D[:,1]), np.argmin(corners_2D[:,1]) # up
    d_val, d_idx = np.max(corners_2D[:,1]), np.argmax(corners_2D[:,1]) # down
    match_idx = Match()
    match_val = Match()
    match_idx.l, match_idx.r, match_idx.u, match_idx.d = l_idx, r_idx, u_idx, d_idx
    match_val.l, match_val.r, match_val.u, match_val.d = l_val, r_val, u_val, d_val
    return match_val,match_idx
     
class Match(object):
    def __init__(self):
        self.l = -1000
        self.r = -1000
        self.u = -1000
        self.d = -1000