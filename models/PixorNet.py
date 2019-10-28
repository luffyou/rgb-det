#coding:utf8
import sys
sys.path.append('../')
from models.BasicModule import BasicModule
import torch as t
from torch import nn
from torch.nn import functional as F
import math

class Fire(nn.Module):
    def __init__(self, inplanes, squeeze_planes, expand_planes):
        super(Fire, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1, stride=1)
        self.bn1 = nn.BatchNorm2d(squeeze_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(squeeze_planes, expand_planes, kernel_size=1, stride=1)
        self.bn2 = nn.BatchNorm2d(expand_planes)
        self.conv3 = nn.Conv2d(squeeze_planes, expand_planes, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(expand_planes)
        self.relu2 = nn.ReLU(inplace=True)

        # using MSR initilization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                m.weight.data.normal_(0, math.sqrt(2./n))

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        out1 = self.conv2(x)
        out1 = self.bn2(out1)
        out2 = self.conv3(x)
        out2 = self.bn3(out2)
        out = t.cat([out1, out2], 1)
        out = self.relu2(out)
        return out


class ResidualBlock(nn.Module):
    '''
    Module: Residual Block
    '''
    def __init__(self, inchannel, outchannel, stride=1, shortcut=None):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, 3, stride, 1, bias=False), # padding 1 zero on both left & right
                nn.BatchNorm2d(outchannel),
                nn.ReLU(inplace=True),
                nn.Conv2d(outchannel, outchannel, 3, 1, 1, bias=False),
                nn.BatchNorm2d(outchannel) )
        self.right = shortcut

    def forward(self, x):
        out = self.left(x)
        residual = x if self.right is None else self.right(x)
        out += residual
        return F.relu(out)

class PixorNet(BasicModule):
    '''
    Main module：ResNet34 base
    ResNet34包含多个layer，每个layer又包含多个Residual block
    用子module来实现Residual block，用_make_layer函数来实现layer
    '''
    def __init__(self, in_channel=3, out_channel=13):
        super(PixorNet, self).__init__()
        self.model_name = 'PixorNet_f'
        
        # 前几层: 图像转换
        self.pre = nn.Sequential(
                nn.Conv2d(in_channel, 32, 3, 1, 1, bias=False), # padding 3 zero on both left & right
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                Fire(32, 16, 24), # fire module 16_1x1->32_3x3->cat
                nn.Conv2d(48, 64, 3, 1, 1, dilation=2, bias=False), # padding 3 zero on both left & right
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                ) # 
        # 重复的layer，原有3，4，6，3个residual block
        self.layer1 = self._make_layer( 64, 96, 1, stride=2)
        self.layer2 = self._make_layer( 96, 192, 2, stride=2)
        self.layer3 = self._make_layer( 192, 256, 2, stride=2)
        self.layer4 = self._make_layer( 256, 384, 1, stride=2)
        
        self.layer5 = nn.Sequential(
                nn.Conv2d(384, 384, 3, 1, 1, bias=False),
                nn.BatchNorm2d(384),
                nn.ReLU(inplace=True),
                nn.Conv2d(384, 384, 3, 1, 1, bias=False),
                nn.BatchNorm2d(384),
                nn.ReLU(inplace=True),
                nn.Conv2d(384, 384, 3, 1, 1, bias=False),
                nn.BatchNorm2d(384),
                nn.ReLU(inplace=True)
                )
        self.ry_cls = nn.Conv2d(384, 12, 3, 1, bias=False)
        self.ry_res = nn.Conv2d(384, 1, 3, 1, bias=False)
        self.init_param()

    def init_param(self):
        # The following is initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2./n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.shape[0] * m.weight.shape[1]
                m.weight.data.normal_(0, math.sqrt(2./n))
                m.bias.data.zero_()
        
    def _make_layer(self,  inchannel, outchannel, block_num, stride=1):
        '''
        layer,include multi residual block
        '''
        shortcut = nn.Sequential(
                nn.Conv2d(inchannel,outchannel,1,stride, bias=False),
                nn.BatchNorm2d(outchannel))        
        layers = []
        layers.append(ResidualBlock(inchannel, outchannel, stride, shortcut)) # use 1x1conv for shortcut
        
        for i in range(1, block_num):
            layers.append(ResidualBlock(outchannel, outchannel)) # use x for shortcut
        return nn.Sequential(*layers)
        
    def forward(self, x, vec, log_flag=False): # x & vec BCHW
        x = self.pre(x)
        if log_flag==True: print('pre',x.size()) 
        x = self.layer1(x)
        if log_flag==True: print('layer1',x.size())
        x = self.layer2(x)
        if log_flag==True: print('layer2',x.size())
        x = self.layer3(x)
        if log_flag==True: print('layer3',x.size())
        x = self.layer4(x)
        if log_flag==True: print('layer4',x.size())
        x = self.layer5(x)
        if log_flag==True: print('layer5',x.size())
        
        ry_cls = self.ry_cls(x)
        ry_cls = F.adaptive_avg_pool2d(ry_cls,1)
        ry_cls = ry_cls.squeeze(3).squeeze(2)
        if log_flag==True: print('ry_cls',ry_cls.size())
        
        ry_res = self.ry_res(x)
        ry_res = F.adaptive_avg_pool2d(ry_res,1)
        ry_res = F.sigmoid(ry_res)
        ry_res = ry_res.squeeze(3).squeeze(2)
        if log_flag==True: print('ry_cls',ry_res.size())
        return ry_cls, ry_res


# if __name__ == '__main__':
# net = PixorNet()
# output = net(t.randn(1,3,224,224),t.randn(1,3,1,1),log_flag=True)
# print(output)