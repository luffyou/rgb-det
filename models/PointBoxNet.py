#coding:utf8
import sys
sys.path.append('../')
from models.BasicModule import BasicModule
import torch as t
from torch import nn
from torch.nn import functional as F
# from torch.nn import init
from utils.data_util import *


class Conv2d(nn.Module):
    def __init__(self, in_channel, out_channel, momentum=0.1):
        super(Conv2d, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=0, bias=True)
        self.bn = nn.BatchNorm2d(out_channel, momentum=momentum)
        # self.fn = nn.ReLU()
        self.fn = nn.ELU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.fn(x)
        return x
    
class Linear(nn.Module):
    def __init__(self, inputs, outputs, drop=True, momentum=0.1):
        super(Linear, self).__init__()
        self.fc = nn.Linear(inputs, outputs, bias=True)
        self.bn = nn.BatchNorm1d(outputs, momentum=momentum)
        # self.fn = nn.ReLU()
        self.drop = drop
        self.dropout = nn.Dropout(0.5)
        self.fn = nn.ELU()
        
    def forward(self, x):
        x = self.fc(x)
        x = self.bn(x)
        if self.drop:
            x = self.dropout(x)
        x = self.fn(x)
        return x
        

class PointBoxNet(BasicModule):
    def __init__(self, bn_decay=0.1, in_channel=5):
        super(PointBoxNet, self).__init__()
        self.model_name = 'PointBoxNet'
        self.mlp1 = Conv2d(in_channel, 128, momentum=bn_decay)
        self.mlp2 = Conv2d(128, 128, momentum=bn_decay)
        self.mlp3 = Conv2d(128, 256, momentum=bn_decay)
        self.mlp4 = Conv2d(256, 512, momentum=bn_decay)
        # global feature
        self.max_pool = nn.AdaptiveAvgPool2d(1) # 与输入尺寸适配
        # task
        self.fc1 = Linear(515, 512, drop=False, momentum=bn_decay)
        self.fc2 = Linear(512, 256, drop=False, momentum=bn_decay)
        # output
        self.center = nn.Linear(256, 3)
        self.ry_cls = nn.Linear(256, NUM_HEADING_BIN)
        self.ry_res = nn.Linear(256, 1)
        self.size_cls = nn.Linear(256, NUM_SIZE_CLUSTER)
        self.size_res = nn.Linear(256, 3)


    def forward(self, pc, one_hot_vec):
        # Bx4*Num_PC*1; add range -> 5
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
    center, ry_cls, ry_res, size_cls, size_res = net(t.randn(2,5,NUM_OBJECT_POINT,1),t.randn(2,3,1,1))
    print('center',center)
    print('ry_cls',ry_cls)
    print('ry_res',ry_res)
    print('size_cls',size_cls)
    print('size_res',size_res)