#coding:utf8
from .BasicModule import BasicModule
import torch as t
from torch import nn
from torch.nn import functional as F
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

class ResNetEx(BasicModule):
    '''
    Main module：ResNet34 base
    ResNet34包含多个layer，每个layer又包含多个Residual block
    用子module来实现Residual block，用_make_layer函数来实现layer
    '''
    def __init__(self, in_channel=3, out_channel=3):
        super(ResNetEx, self).__init__()
        self.model_name = 'resnetEx'
        
        # 前几层: 图像转换
        self.pre = nn.Sequential(
                nn.Conv2d(in_channel, 64, 7, 2, 3, bias=False), # padding 3 zero on both left & right
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(3, 2, 1)) # 
        
        # 重复的layer，分别有3，4，6，3个residual block
        self.layer1 = self._make_layer( 64, 128, 3)
        self.layer2 = self._make_layer( 128, 256, 4, stride=2) # stride=2
        self.layer3 = self._make_layer( 256, 512, 6, stride=2)
        self.layer4 = self._make_layer( 512, 512, 3, stride=2) # stride=2
        
        self.layer5 = nn.Sequential(
                nn.Conv2d(512, 1024, 1, 1, bias=False), # 512+4(2d) 128 64 # only for test
                nn.BatchNorm2d(1024),
                nn.ReLU(inplace=True))
        self.fc = nn.Linear(1024, out_channel)
        
        
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
        
    def forward(self, x, vec): # x & vec BCHW
        x = self.pre(x)
        # print('pre',x.size())
        
        x = self.layer1(x)
        # print('layer1',x.size())
        x = self.layer2(x)
        # print('layer2',x.size())
        x = self.layer3(x)
        # print('layer3',x.size())
        x = self.layer4(x)
        # print('layer4',x.size())
        
        size_x = x.size()
        x = F.avg_pool2d(x, (size_x[2], size_x[3])) # Global Average Pooling  Bx512x1x1 
        # x = t.cat((x,vec), dim=1) # add 2d info. Bx(512+4)x1x1 # only for test
        # print('pool_cat',x.size())
        x = self.layer5(x)
        # print('layer5',x.size())
        x = x.view(x.size(0), -1) # Bx1024
        # print('layer6',x.size())
        x = self.fc(x)
        
        return x # Bx3


if __name__ == '__main__':
    net = ResNetEx()
    output = net(t.randn(1,4,17,17),t.randn(1,4,1,1))
    print(output)