import numpy as np
import cv2
import os,math
# from scipy.optimize import leastsq
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

cbox = np.array([[0,70.4],[-40,40],[-3,2]])
y_ct_mean_dict =  {'Car': 0.9467262795309836}

# read_lable_by_path
def read_label(label_filename): # 传入label文件名
    lines = [line.rstrip() for line in open(label_filename)]
    objects = [Object3d(line) for line in lines]
    return objects

class Object3d(object):
    ''' 3d object label '''
    def __init__(self, label_file_line, islabel=True): # label文件的每一行; 可用于训练集和验证集
        data = label_file_line.split(' ')
        data[1:] = [float(x) for x in data[1:]]

        # extract label, truncation, occlusion
        self.type = data[0] # 'Car', 'Pedestrian', ...
        self.truncation = data[1] # truncated pixel ratio [0..1]
        self.occlusion = int(data[2]) # 0=visible, 1=partly occluded, 2=fully occluded, 3=unknown
        self.alpha = data[3] # object observation angle [-pi..pi]

        # extract 2d bounding box in 0-based coordinates
        self.xmin = data[4] # left
        self.ymin = data[5] # top
        self.xmax = data[6] # right
        self.ymax = data[7] # bottom
        self.box2d = np.array([self.xmin,self.ymin,self.xmax,self.ymax])

        # extract 3d bounding box information
        self.h = data[8] # box height
        self.w = data[9] # box width
        self.l = data[10] # box length (in meters)
        self.t = [data[11],data[12],data[13]] # location (x,y,z) in camera coord.（bottom center of 3D box）
        self.ry = data[14] # yaw angle (around Y-axis in camera coordinates) [-pi..pi]
        if islabel is not True:
            self.score = data[15]
            
    def print_object(self):
        print('Type, truncation, occlusion, alpha: %s, %d, %d, %f' % \
            (self.type, self.truncation, self.occlusion, self.alpha))
        print('2d bbox (x0,y0,x1,y1): %f, %f, %f, %f' % \
            (self.xmin, self.ymin, self.xmax, self.ymax))
        print('3d bbox h,w,l: %f, %f, %f' % \
            (self.h, self.w, self.l))
        print('3d bbox location, ry: (%f, %f, %f), %f' % \
            (self.t[0],self.t[1],self.t[2],self.ry))
        
# read_calib_by_path        
class Calibration(object): # 传入标定文件名
    ''' Calibration matrices and utils
        3d XYZ in <label> are in rect camera coord.
        2d box xy are in image2 coord
        Points in <lidar>.bin are in Velodyne coord.

        y_image2 = P^2_rect * x_rect 摄像机转只需这个!!!
        y_image2 = P^2_rect * R0_rect * Tr_velo_to_cam * x_velo
        x_ref = Tr_velo_to_cam * x_velo
        x_rect = R0_rect * x_ref

        P^2_rect = [f^2_u,  0,      c^2_u,  -f^2_u . b^2_x;
                    0,      f^2_v,  c^2_v,  -f^2_v . b^2_y;
                    0,      0,      1,      0]
                 = K * [1|t]

        image2 coord:
         ----> x-axis (u)
        |
        |
        v y-axis (v)

        velodyne coord:
        front x, left y, up z

        rect/ref camera coord:
        right x, down y, front z

        Ref (KITTI paper): http://www.cvlibs.net/publications/Geiger2013IJRR.pdf
        TODO(rqi): do matrix multiplication only once for each projection.
    '''
    def __init__(self, calib_filepath, is_cam_2=True):
        calibs = self.read_calib_file(calib_filepath) # core func
        # Projection matrix from rect camera coord to image2 coord
        if is_cam_2:
            self.P = calibs['P2']
        else:
            self.P = calibs['P3']
        self.P = np.reshape(self.P, [3,4])
        # Rigid transform from Velodyne coord to reference camera coord
        self.V2C = calibs['Tr_velo_to_cam']
        self.V2C = np.reshape(self.V2C, [3,4])
        self.C2V = self.inverse_rigid_trans(self.V2C)
        # Rotation from reference camera coord to rect camera coord
        self.R0 = calibs['R0_rect']
        self.R0 = np.reshape(self.R0,[3,3])

        # Camera intrinsics and extrinsics
        self.c_u = self.P[0,2] # u0
        self.c_v = self.P[1,2] # v0
        self.f_u = self.P[0,0] # alpha
        self.f_v = self.P[1,1] # beta
        self.b_x = self.P[0,3]/(-self.f_u) # relative
        self.b_y = self.P[1,3]/(-self.f_v)
        self.k1 = self.P[0,3] # absolute
        self.k2 = self.P[1,3]
        self.k3 = self.P[2,3]

    def read_calib_file(self, filepath): 
        ''' Read in a calibration file and parse into a dictionary. 
        Ref: https://github.com/utiasSTARS/pykitti/blob/master/pykitti/utils.py
        written for __init__()
        '''
        data = {}
        with open(filepath, 'r') as f:
            for line in f.readlines():
                line = line.rstrip()
                if len(line)==0: continue
                key, value = line.split(':', 1)
                # The only non-float values in these files are dates, which
                # we don't care about anyway
                try:
                    data[key] = np.array([float(x) for x in value.split()])
                except ValueError:
                    pass
        return data

    def cart2hom(self, pts_3d):
        ''' Input: nx3 points in Cartesian
            Oupput: nx4 points in Homogeneous by pending 1
        '''
        n = pts_3d.shape[0]
        pts_3d_hom = np.hstack((pts_3d, np.ones((n,1))))
        return pts_3d_hom

    def inverse_rigid_trans(self, Tr):
        ''' Inverse a rigid body transform matrix (3x4 as [R|t])
            [R'|-R't; 0|1]
        '''
        inv_Tr = np.zeros_like(Tr) # 3x4
        inv_Tr[0:3,0:3] = np.transpose(Tr[0:3,0:3])
        inv_Tr[0:3,3] = np.dot(-np.transpose(Tr[0:3,0:3]), Tr[0:3,3])
        return inv_Tr

    # ===========================
    # ------- 3d to 3d ----------
    # ===========================
    def project_velo_to_ref(self, pts_3d_velo): # nx3 np.array
        pts_3d_velo = self.cart2hom(pts_3d_velo) # nx4
        return np.dot(pts_3d_velo, np.transpose(self.V2C)) # nx4 x 4x3

    def project_ref_to_velo(self, pts_3d_ref):
        pts_3d_ref = self.cart2hom(pts_3d_ref) # nx4
        return np.dot(pts_3d_ref, np.transpose(self.C2V)) 

    def project_rect_to_ref(self, pts_3d_rect):
        ''' Input and Output are nx3 points '''
        return np.transpose(np.dot(np.linalg.inv(self.R0), np.transpose(pts_3d_rect)))

    def project_ref_to_rect(self, pts_3d_ref):
        ''' Input and Output are nx3 points '''
        return np.transpose(np.dot(self.R0, np.transpose(pts_3d_ref)))

    def project_rect_to_velo(self, pts_3d_rect):
        ''' Input: nx3 points in rect camera coord.
            Output: nx3 points in velodyne coord.
        '''
        pts_3d_ref = self.project_rect_to_ref(pts_3d_rect)
        return self.project_ref_to_velo(pts_3d_ref)

    def project_velo_to_rect(self, pts_3d_velo):
        pts_3d_ref = self.project_velo_to_ref(pts_3d_velo)
        return self.project_ref_to_rect(pts_3d_ref)

    # ===========================
    # ------- 3d to 2d ----------
    # ===========================
    def project_rect_to_image(self, pts_3d_rect): # note this func ！！！
        ''' Input: nx3 points in rect camera coord.
            Output: nx2 points in image2 coord.
        '''
        pts_3d_rect = self.cart2hom(pts_3d_rect) # nx3 -> nx4
        pts_2d = np.dot(pts_3d_rect, np.transpose(self.P)) # nx3
        pts_2d[:,0] /= pts_2d[:,2]
        pts_2d[:,1] /= pts_2d[:,2]
        return pts_2d[:,0:2]# nx2

    def project_velo_to_image(self, pts_3d_velo):
        ''' Input: nx3 points in velodyne coord.
            Output: nx2 points in image2 coord.
        '''
        pts_3d_rect = self.project_velo_to_rect(pts_3d_velo)
        return self.project_rect_to_image(pts_3d_rect)

    def project_8p_to_4p(self, pts_2d):
        x0 = np.min(pts_2d[:,0])
        x1 = np.max(pts_2d[:,0])
        y0 = np.min(pts_2d[:,1])
        y1 = np.max(pts_2d[:,1])
        x0 = max(0,x0)
        #x1 = min(x1, proj.image_width)
        y0 = max(0,y0)
        #y1 = min(y1, proj.image_height)
        return np.array([x0, y0, x1, y1])

    def project_velo_to_4p(self, pts_3d_velo):
        ''' Input: nx3 points in velodyne coord.
            Output: 4 points in image2 coord.
        '''
        pts_2d_velo = self.project_velo_to_image(pts_3d_velo)
        return self.project_8p_to_4p(pts_2d_velo)

    # ===========================
    # ------- 2d to 3d ----------
    # ===========================
    def project_image_to_rect(self, uv_depth):
        ''' Input: nx3 first two channels are uv, 3rd channel
                   is depth in rect camera coord.
            Output: nx3 points in rect camera coord.
        '''
        n = uv_depth.shape[0]
        x = ((uv_depth[:,0]-self.c_u)*uv_depth[:,2])/self.f_u + self.b_x
        y = ((uv_depth[:,1]-self.c_v)*uv_depth[:,2])/self.f_v + self.b_y
        pts_3d_rect = np.zeros((n,3))
        pts_3d_rect[:,0] = x
        pts_3d_rect[:,1] = y
        pts_3d_rect[:,2] = uv_depth[:,2]
        return pts_3d_rect

    def project_image_to_velo(self, uv_depth):
        pts_3d_rect = self.project_image_to_rect(uv_depth)
        return self.project_rect_to_velo(pts_3d_rect)

    def get_depth_pt3d(self, depth):
        pt3d=[]
        for i in range(depth.shape[0]):
            for j in range(depth.shape[1]):
                pt3d.append([i, j, depth[i, j]])
        return np.array(pt3d)

    def project_depth_to_velo(self, depth, constraint_box=True):
        depth_pt3d =  self.get_depth_pt3d(depth)
        depth_UVDepth = np.zeros_like(depth_pt3d)
        depth_UVDepth[:,0] = depth_pt3d[:,1]
        depth_UVDepth[:,1] = depth_pt3d[:,0]
        depth_UVDepth[:,2] = depth_pt3d[:,2]
        #print("depth_pt3d:",depth_UVDepth.shape)
        depth_pc_velo = self.project_image_to_velo(depth_UVDepth)
        #print("dep_pc_velo:",depth_pc_velo.shape)
        if constraint_box:
            depth_box_fov_inds = (depth_pc_velo[:,0]< cbox[0][1] ) & \
                (depth_pc_velo[:,0]>= cbox[0][0]) & \
                (depth_pc_velo[:,1]<  cbox[1][1]) & \
                (depth_pc_velo[:,1]>= cbox[1][0]) & \
                (depth_pc_velo[:,2]<  cbox[2][1]) & \
                (depth_pc_velo[:,2]>= cbox[2][0])
            depth_pc_velo=depth_pc_velo[depth_box_fov_inds]
        return depth_pc_velo
        
    def project_uv_to_rect_with_y(self, pts_2d, pts_y_rect):
        ''' Input: 
                pts_2d: nx2, uv coord
                pts_y_rect: nx1 or nx2(distribution?)
            Output: nx3 points in rect camera coord.
        '''
        n = pts_2d.shape[0]
        pts_3d_rect = np.zeros((n,3))

        u0 = self.c_u
        v0 = self.c_v
        a = self.f_u
        b = self.f_v
        k1 = self.k1
        k2 = self.k2
        k3 = self.k3

        u = pts_2d[:,0]
        v = pts_2d[:,1]
        y = pts_y_rect[:,0] #y
        pts_3d_rect[:,2] = b/(v-v0)*y + (k2-k3*v)/(v-v0) #z
        pts_3d_rect[:,0] = b/(v-v0)*(u-u0)/a*y + ((u-u0)*(k2-k3*v)-(v-v0)*(k1-k3*u))/(a*(v-v0)) #x
        pts_3d_rect[:,1] = y
        # print(pts_3d_rect)
        return pts_3d_rect


if __name__ == "__main__":
    root_dir = "/hdd/you/dataset/KITTI/training"
    calib_dir = os.path.join(root_dir,"calib")
    label_dir = os.path.join(root_dir,"label_2")
    code_dir = "/hdd/you/rgb-det"
    sets_type = "trainval" # test  train  trainval  val

    idx_path = os.path.join(code_dir, 'data/image_sets/{}.txt'.format(sets_type)) # 3types
    idx_fig = open(idx_path, 'r')
    idx_all = idx_fig.readlines() # all idx
    idx_fig.close()

    xyz_ct_diff_all = np.empty([0,3])
    count_idx = 0
    for idx in idx_all:
        idx = idx.rstrip()
        calib_path = os.path.join(calib_dir, idx+".txt")
        label_path = os.path.join(label_dir, idx+".txt")
        calib = Calibration(calib_path)
        objs = read_label(label_path)

        xyz_bt = [] # bottom
        xyz_ct = [] # center
        uv_ct = []
        for obj in objs:
            if obj.type in ['Car']: # "Cyclist", "Pedsetrian"
                uv_ct.append([(obj.xmin+obj.xmax)/2, (obj.ymin+obj.ymax)/2])
                xyz_bt.append(obj.t)
                xyz_ct.append([obj.t[0], obj.t[1]-obj.h/2, obj.t[2]])
        n_uv_ct = len(uv_ct)
        if n_uv_ct <=0:
            continue
        # y_ct = np.array([[tmp[1]] for tmp in xyz_ct]) # n*1 2d real value of y
        y_ct = np.full((n_uv_ct,1), y_ct_mean_dict['Car']) # mean value of y
        # print(np.array(y_ct))
        xyz_ct_proj = calib.project_uv_to_rect_with_y(np.array(uv_ct), y_ct)
        xyz_ct_diff = xyz_ct - xyz_ct_proj
        xyz_ct_diff[:,1] = y_ct[:,0]
        # print(xyz_ct_diff)
        xyz_ct_diff_all = np.append(xyz_ct_diff_all, xyz_ct_diff, axis=0)

        # uv_bt = calib.project_rect_to_image(np.array(xyz_bt))
        # y_bt = [[tmp[1]] for tmp in xyz_bt]
        # xyz_bt_proj = calib.project_uv_to_rect_with_y(uv_bt,np.array(y_bt))
        # xyz_bt_diff = xyz_bt - xyz_bt_proj
        # print(xyz_bt_diff)

        # count_idx += 1
        # if count_idx % 3 == 0:
        #     break

    # y_ct_mean = np.mean(xyz_ct_diff_all[:,1])
    # print('y_ct_mean', y_ct_mean)
    # fig = plt.figure(1)
    x_dim = [i for i in range(xyz_ct_diff_all.shape[0])]
    # plt.hist(xyz_ct_diff_all[:,0],bins=160)
    plt.ylim(-30,30)
    plt.scatter(x_dim, xyz_ct_diff_all[:,0],linewidths=1)
    save_path = 'save_img/all_ct_x.png'
    plt.savefig(save_path)
    plt.cla()
    # plt.hist(xyz_ct_diff_all[:,2],bins=160)
    plt.ylim(-40,40)
    plt.scatter(x_dim, xyz_ct_diff_all[:,2],linewidths=1)
    save_path = 'save_img/all_ct_z.png'
    plt.savefig(save_path)
    plt.cla()
    # plt.show()