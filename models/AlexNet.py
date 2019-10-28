#coding:utf8
from torch import nn
from .BasicModule import BasicModule
from torch.nn import functional as F

class AlexNet(BasicModule):
    '''
    code from torchvision/models/alexnet.py
    结构参考 <https://arxiv.org/abs/1404.5997>
    '''
    def __init__(self, num_classes=12, num_residual=1):
        
        super(AlexNet, self).__init__()
        
        self.model_name = 'alexnet'

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.cls = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
        self.res = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_residual),
        )

    def forward(self, x, vec, log_flag=False):
        x = self.features(x)
        if log_flag==True: print('fea',x.size()) 
        x = x.view(x.size(0), 256 * 6 * 6)
        if log_flag==True: print('view',x.size()) 
        ry_cls = self.cls(x)
        if log_flag==True: print('cls',ry_cls.size()) 
        ry_res = self.res(x)
        ry_res = F.sigmoid(ry_res)
        if log_flag==True: print('res',ry_res.size()) 
        return ry_cls, ry_res
