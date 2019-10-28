#coding:utf8
import warnings
class ConfigPoint(object):
    env = './log_point/pointnet' #  可视化环境
    model = 'PointBoxNet' # 'PixorNet' 'ResNetEx' 使用的模型，名字必须与models/__init__.py中的名字一致
    use_gpu = True # user GPU or not
    cuda_idx = 0

    root = '/hdd/you/rgbd-det/' # 项目根目录
    load_model_path = None # 'checkpoints/model.pth' # 加载预训练的模型的路径，为None代表不加载

    batch_size = 32 # batch size
    batch_size_val = 32
    batch_size_test = 1
    num_workers = 1 # how many workers for loading data
    print_freq = 1000 # print info every N batch
    plot_freq = 100

    # debug_file = '/hdd/you/rgbd-det/tmp/debug' # if os.path.exists(debug_file): enter ipdb
    result_file = 'det_3d.txt'
      
    max_epoch = 300
    lr = 0.001 # initial learning rate; adjust by step
    lr_decay = 0.2 # when val_loss increase, lr = lr*lr_decay ;origin 0.95
    # momentum = 0.5
    # momentum_dec_rate = 0.5
    weight_decay = 0.0005 # 1e-4 # 损失函数

    white_list = ['Car'] #  'Car', 'Pedestrian', 'Cyclist'
    save_model = True

def parse(self,kwargs):
        '''
        根据字典kwargs 更新 config参数
        '''
        for k,v in kwargs.items():
            if not hasattr(self,k):
                warnings.warn("Warning: opt has not attribut %s" %k)
            setattr(self,k,v)

        print('user config:')
        for k,v in self.__class__.__dict__.items():
            if not k.startswith('__'):
                print(k,getattr(self,k))

ConfigPoint.parse = parse
opt = ConfigPoint()
# opt.parse = parse

# 进入debug模式
# if os.path.exists(opt.debug_file):
    # import ipdb;
    # ipdb.set_trace()