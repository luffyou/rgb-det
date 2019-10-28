import numpy as np

NUM_HEADING_BIN = 12 # origin 12
NUM_SIZE_CLUSTER = 8 # one cluster for each type
NUM_OBJECT_POINT = 512
g_type2class={'Car':0, 'Van':1, 'Truck':2, 'Pedestrian':3, 'Person_sitting':4, 'Cyclist':5, 'Tram':6, 'Misc':7}
g_class2type = {g_type2class[t]:t for t in g_type2class}

g_type2onehotclass = {'Car': 0, 'Pedestrian': 1, 'Cyclist': 2}

# size clustrs dictionary
g_type_mean_size = {'Car': np.array([3.88311640418,1.62856739989,1.52563191462]),
                'Van': np.array([5.06763659,1.9007158,2.20532825]),
                'Truck': np.array([10.13586957,2.58549199,3.2520595]),
                'Pedestrian': np.array([0.84422524,0.66068622,1.76255119]),
                'Person_sitting': np.array([0.80057803,0.5983815,1.27450867]),
                'Cyclist': np.array([1.76282397,0.59706367,1.73698127]),
                'Tram': np.array([16.17150617,2.53246914,3.53079012]),
                'Misc': np.array([3.64300781,1.54298177,1.92320313])}
g_mean_size_arr = np.zeros((NUM_SIZE_CLUSTER, 3)) # size clustrs array
for i in range(NUM_SIZE_CLUSTER):
    g_mean_size_arr[i,:] = g_type_mean_size[g_class2type[i]]
    
clw = 10 # corner_loss_weight
blw = 1.0 # box_loss_weight
rlw = 20 # res_loss_weight
    
def angle2class(angle, num_class=NUM_HEADING_BIN):
    ''' Convert continuous angle to discrete class and residual.
    Input:
        angle: rad scalar, from 0-2pi (or -pi~pi), class center at
            0, 1*(2pi/N), 2*(2pi/N) ...  (N-1)*(2pi/N)
        num_class: int scalar, number of classes N
    Output:
        class_id, int, among 0,1,...,N-1
        residual_angle: float, a number such that
            class*(2pi/N) + residual_angle = angle
    '''
    angle = angle%(2*np.pi) # [-pi, pi] to [0,2pi]
    assert(angle>=0 and angle<=2*np.pi)
    angle_per_class = 2*np.pi/float(num_class)
    shifted_angle = (angle+angle_per_class/2)%(2*np.pi) # 更换基准，以bin中心为基准，变成新基准下的角度
    class_id = int(shifted_angle/angle_per_class) 
    residual_angle = shifted_angle - (class_id * angle_per_class + angle_per_class/2) # 新基准下的角度-基准中心
    residual_angle = residual_angle / (np.pi/NUM_HEADING_BIN) # normalized_label to match sigmoid
    return class_id, residual_angle #normalized

def class2angle(pred_cls, residual, num_class=NUM_HEADING_BIN, to_label_format=True):
    ''' Inverse function to angle2class.
    If to_label_format, adjust angle to the range as in labels.
    '''
    residual = residual * (np.pi/NUM_HEADING_BIN) # normalized to truth value
    angle_per_class = 2*np.pi/float(num_class)
    angle_center = pred_cls * angle_per_class
    angle = angle_center + residual
    if to_label_format and angle>np.pi: 
        angle = angle - 2*np.pi # only when angle > pi; consider the clock
    return angle


def size2class(size, type_name):
    ''' Convert 3D bounding box size to template class and residuals.
    todo (rqi): support multiple size clusters per type.
 
    Input:
        size: numpy array of shape (3,) for (l,w,h)
        type_name: string
    Output:
        size_class: int scalar
        size_residual: numpy array of shape (3,)
    '''
    size_class = g_type2class[type_name]
    mean_array = g_type_mean_size[type_name]
    residual = size - mean_array
    # print('resdual_raw',residual)
    for i in range(len(residual)):
        residual[i] = residual[i]/(mean_array[i]/2)
    # print('resdual_norm',residual)
    return size_class, residual

def class2size(pred_cls, residual):
    ''' Inverse function to size2class. '''
    mean_size = g_type_mean_size[g_class2type[pred_cls]]
    # print('resdual_raw',residual)
    for i in range(len(residual)):
        residual[i] = residual[i]*(mean_size[i]/2)
    # print('resdual_raw',residual)
    return mean_size + residual 


def from_prediction_to_label_format(center, angle_class, angle_res, size_class, size_res, rot_angle):
    ''' Convert predicted box parameters to label format. '''
    l,w,h = class2size(size_class, size_res)
    ry = class2angle(angle_class, angle_res, NUM_HEADING_BIN) + rot_angle
    tx,ty,tz = rotate_pc_along_y(np.expand_dims(center,0),-rot_angle).squeeze()
    ty += h/2.0 # note here  predict the center then change to bottom
    return h,w,l,tx,ty,tz,ry


def acc_of_confusion_mat(confusion_matrix):
    # confusion_matrix ndarray
    cm_value = confusion_matrix.value()
    order = len(cm_value)
    correct = 0
    for i in range(order):
        correct += cm_value[i][i]
    accuracy = 100. * correct / cm_value.sum()
    return accuracy

if __name__ == '__main__':
    size2class(np.array([3.88,1.62,1.52]),'Car')
    class2size(0,np.array([0.5,0.5,0.5]))