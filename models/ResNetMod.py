import torch as t
import torch.nn as nn
from torch.nn import functional as F
from . BasicModule import BasicModule
import torch
import math

def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNetMod([3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(torch.load(model.modelPath))
    return model

def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNetMod([3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(torch.load(model.modelPath))
    return model

class ResNetMod(BasicModule):
    """
    block: A sub module
    """
    def __init__(self, layers=[3, 4, 6, 3], num_out=(12+1)):
        super(ResNetMod, self).__init__()
        self.inplanes = 64
        self.pre1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size = 7, stride = 2, padding = 3, bias = False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace = True))
        self.pre2 = nn.Sequential(
            # nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
            nn.Conv2d(64, 64, kernel_size = 7, stride = 2, padding = 3, bias = False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace = True))
        self.stack1 = self.make_stack(64, layers[0])
        self.stack2 = self.make_stack(128, layers[1], stride=2)
        self.stack3 = self.make_stack(256, layers[2], stride=2)
        self.stack4 = self.make_stack(512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride = 1)
        # self.fc = nn.Linear(512 * Bottleneck.expansion, num_out)
        self.post = nn.Sequential(
#             self.conv1 = nn.Conv2d(3, 64, kernel_size = 7, stride = 2, padding = 3, bias = False)
#             self.bn1 = nn.BatchNorm2d(64)
#             self.relu = nn.ReLU(inplace = True)
            nn.Linear(512 * Bottleneck.expansion, 512),
            nn.Dropout(0.5),
            nn.ReLU(inplace = True),
            nn.Linear(512, 128),
            nn.Dropout(0.5),
            nn.ReLU(inplace = True))
        # task
        self.hcls = nn.Linear(128,12) # heading angle class
        self.hres = nn.Linear(128,1) # heading angle residual
        # initialize parameters
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

    def make_stack(self, planes, blocks, stride = 1):
        downsample = None
        layers = []
            
        if stride != 1 or self.inplanes != planes * Bottleneck.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * Bottleneck.expansion, kernel_size=1,stride=stride,bias=False),
                nn.BatchNorm2d(planes * Bottleneck.expansion),
                )

        layers.append(Bottleneck(self.inplanes, planes, stride, downsample)) # downsample mean shortcut
        self.inplanes = planes * Bottleneck.expansion
        for i in range(1, blocks):
            layers.append(Bottleneck(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, vec): # only for test!!! may be useless
        x = self.pre1(x)
        # print('after_prel',x.shape)
        x = self.pre2(x)
        # print('after_pre2',x.shape)
        x = self.stack1(x)
        # print('after_stack1',x.shape)
        x = self.stack2(x)
        # print('after_stack2',x.shape)
        x = self.stack3(x)
        # print('after_stack3',x.shape)
        x = self.stack4(x)
        # print('after_stack4',x.shape)
        x = self.avgpool(x)
        # print('after_avg',x.shape)
        x = x.view(x.size(0), -1)
        # x = self.fc(x)
        x = self.post(x)
        
        ry_cls = self.hcls(x)
        # ry_cls = F.softmax(ry_cls)
        ry_res = self.hres(x)
        ry_res = F.sigmoid(ry_res)
        
        return ry_cls, ry_res

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out