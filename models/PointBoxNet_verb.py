#coding:utf8
import sys
sys.path.append('../')
from models.BasicModule import BasicModule
import torch as t
from torch import nn
from torch.nn import functional as F
from torch.nn import init
from utils.data_util import *

# weight_decay防止过拟合，在optimize中设置
def init_layer(layer, use_xavier=True, std_dev=1e-3):
    if use_xavier:
        init.xavier_normal_(layer.weight.data)
        # init.xavier_normal_(layer.bias.data) # with error?
        init.normal_(layer.bias.data, std=std_dev)
    else:
        init.normal_(layer.weight.data, std=std_dev)
        init.normal_(layer.bias.data, std=std_dev)

# 貌似完全没必要如此复杂，设置到最后，其实都是默认值。。。
class Conv2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel=1, same_padding=True, stride=1,
                 use_xavier=True, std_dev=1e-3, activation_fn=True, bn=False, bn_decay=None):
        super(Conv2d, self).__init__()
        padding = int((kernel - 1) / 2) if same_padding else 0
        self.conv = nn.Conv2d(in_channel, out_channel, kernel, stride, padding=padding, bias=True)
        init_layer(self.conv, use_xavier, std_dev)
        
        self.bn_decay = bn_decay if bn_decay is not None else 0.5 # ??? 0.1 or 0.9
        self.bn = nn.BatchNorm2d(out_channel, momentum=self.bn_decay)
        self.activation_fn = nn.ReLU()

    def forward(self, x):
        assert x.dim() == 4, "Data should be BxCxHxW"
        x = self.conv(x)
        if self.bn: # and self.training: # training是nn.Module中的属性，由train()/eval()设置
            x = self.bn(x)
        if self.activation_fn:
            x = self.activation_fn(x)
        return x
    
class Linear(nn.Module):
    def __init__(self, inputs, outputs, 
                 use_xavier=True, std_dev=1e-3, activation_fn=True, bn=False, bn_decay=None):
        super(Linear, self).__init__()
        self.fc = nn.Linear(inputs, outputs, bias=True)
        init_layer(self.fc, use_xavier, std_dev)
        
        self.bn_decay = bn_decay if bn_decay is not None else 0.5
        self.bn = nn.BatchNorm1d(outputs, momentum=self.bn_decay)
        self.activation_fn = nn.ReLU()
        
    def forward(self, x):
        assert x.dim() == 2, "Data should be BxL"
        x = self.fc(x)
        if self.bn:
            x = self.bn(x)
        if self.activation_fn:
            x = self.activation_fn(x)
        return x
        

class PointBoxNet(BasicModule):
    def __init__(self, num_points=NUM_OBJECT_POINT, bn_decay=0.5, in_channel=4):
        super(PointBoxNet, self).__init__()
        self.model_name = 'PointBoxNet'
        # num_point = pc.size()[2]
        # bn_decay = 0.1 # 此处注释，因此为None,在Conv2d和Liner模块中仍会设置为0.1
        self.mlp1 = Conv2d(in_channel, 128, kernel=1, bn=True, bn_decay=bn_decay)
        self.mlp2 = Conv2d(128, 128, kernel=1, bn=True, bn_decay=bn_decay)
        self.mlp3 = Conv2d(128, 256, kernel=1, bn=True, bn_decay=bn_decay)
        self.mlp4 = Conv2d(256, 512, kernel=1, bn=True, bn_decay=bn_decay)
        # global feature
        self.max_pool = nn.MaxPool2d(kernel_size=[num_points, 1]) # 与输入尺寸适配
        # task
        self.fc1 = Linear(515, 512, bn=True, bn_decay=bn_decay)
        self.fc2 = Linear(512, 256, bn=True, bn_decay=bn_decay)
        # output
        self.center = Linear(256, 3, bn=False, activation_fn=False)
        self.ry_cls = Linear(256, NUM_HEADING_BIN, bn=False, activation_fn=False)
        self.ry_res = Linear(256, 1, bn=False, activation_fn=False)
        self.size_cls = Linear(256, NUM_SIZE_CLUSTER, bn=False, activation_fn=False)
        self.size_res = Linear(256, 3, bn=False, activation_fn=False)


    def forward(self, pc, one_hot_vec):
        # Bx4*Num_PC*1
        assert pc.dim() == 4, "input should be batch_size*channel_num*height*width"
        pc = self.mlp1(pc)
        pc = self.mlp2(pc)
        pc = self.mlp3(pc)
        pc = self.mlp4(pc)
        feat = self.max_pool(pc)
        # one_hot_vec = one_hot_vec.reshape((-1, 3, 1, 1)) # gurantee by outside
        # print('feat',feat.size(), 'one_hot_vec', one_hot_vec.size())
        feat = t.cat((feat, one_hot_vec), dim=1) 
        feat = feat.view(feat.size()[0],-1)
        feat = self.fc1(feat)
        feat = self.fc2(feat)
        # print('feat',feat.shape)
        
        center = self.center(feat)
        ry_cls = self.ry_cls(feat) # cls
        ry_res = self.ry_res(feat)
        ry_res = F.sigmoid(ry_res)
        # print('ry_res',ry_res.shape)
        size_cls = self.size_cls(feat) # size
        size_res = self.size_res(feat)
        size_res = F.sigmoid(size_res) # for each element
        # print('size_res',size_res.shape)
        return center, ry_cls, ry_res, size_cls, size_res
    
    
if __name__ == '__main__':
    # https://github.com/LoFaiTh/frustum_pointnes_pytorch/blob/master/model.py
    net = PointBoxNet()
    center, ry_cls, ry_res, size_cls, size_res = net(t.randn(2,4,NUM_OBJECT_POINT,1),t.randn(2,3,1,1))
    print('center',center)
    print('ry_cls',ry_cls)
    print('ry_res',ry_res)
    print('size_cls',size_cls)
    print('size_res',size_res)