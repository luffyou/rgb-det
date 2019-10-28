from torchnet import meter
from utils.data_util import *
import torch as t
class LossMeter():
    '''
    total;orientation;size;location
    self.xxx: about the mean/var of whole data
    xxx : about the current data
    '''
    def __init__(self, total=True,ry=True,size=True,loc=True):
        if total:
            self.previous_loss = 1e100
            self.total = meter.AverageValueMeter()
        if ry: 
            self.ry_cls = meter.AverageValueMeter()
            self.ry_cm = meter.ConfusionMeter(NUM_HEADING_BIN) # 12
            self.ry_res = meter.AverageValueMeter()
        if size:
            self.size_cls = meter.AverageValueMeter()
            self.size_cm = meter.ConfusionMeter(NUM_SIZE_CLUSTER) # 8
            self.size_res = meter.AverageValueMeter()
        if loc: # center
            self.loc = meter.AverageValueMeter()
            
    def reset(self, total=True,ry=True,size=True,loc=True):
        if total:
            self.total.reset()
        if ry: 
            self.ry_cls.reset()
            self.ry_cm.reset()
            self.ry_res.reset()
        if size:
            self.size_cls.reset()
            self.size_cm.reset()
            self.size_res.reset()
        if loc: # center
            self.loc.reset()
            
    def lm_add(self, total, ry_cls,ry_res, size_cls,size_res, loc):
        # about(ignore the word:) loss; self.xxx: about the mean/var of whole data; xxx : about the current data
        if total is not None:
            self.total.add(total.data)
        if ry_cls is not None: 
            self.ry_cls.add(ry_cls.data)
            self.ry_res.add(ry_res.data)
        if size_cls is not None:
            self.size_cls.add(size_cls.data)
            self.size_res.add(size_res.data)
        if loc is not None: # center
            self.loc.add(loc.data)
            
    def cm_add(self, pre_ry_cls,gt_ry_cls, pre_size_cls,gt_size_cls):
        # can not be onehot; confusion_meter
        if gt_ry_cls is not None:
            self.ry_cm.add(pre_ry_cls,gt_ry_cls) # add idx value; convert to count; result to matrix 
        if gt_size_cls is not None:
            self.size_cm.add(pre_size_cls,gt_size_cls)
            
    def print_log(self, total, ry_cls,ry_res, size_cls,size_res, loc):
        # about(ignore the word:) loss; self.xxx: about the mean/var of whole data; xxx : about the current data
        if total is not None: # the latest total loss; the whole total loss
            print('loss_total:',total.item(), 'tot_avg:',self.total.value()[0].item())
        if ry_cls is not None: 
            print('loss_ry_cls:',ry_cls.item(), 'rc_avg:',self.ry_cls.value()[0].item())
            print('loss_ry_res:',ry_res.item(), 'rr_avg:',self.ry_res.value()[0].item())
        if size_cls is not None:
            print('loss_size_cls:',size_cls.item(), 'sc_avg:',self.size_cls.value()[0].item())
            print('loss_size_res:',size_res.item(), 'sr_avg:',self.size_res.value()[0].item())
        if loc is not None:
            print('loss_loc:',loc.item(), 'loc_avg:',self.loc.value()[0].item())

    def plot(self, writer,niter, total=True,ry=True,size=True,loc=True):
        if total:
            writer.add_scalar('Loss/total',self.total.value()[0],niter) # curve obout train
        if ry:
            writer.add_scalar('Loss/ry_cls',self.ry_cls.value()[0],niter)
            writer.add_scalar('Loss/ry_res',self.ry_res.value()[0],niter)
        if size:
            writer.add_scalar('Loss/size_cls',self.size_cls.value()[0],niter)
            writer.add_scalar('Loss/size_res',self.size_res.value()[0],niter)
        if loc:
            writer.add_scalar('Loss/loc',self.loc.value()[0],niter) # curve obout train


class Benchmark():
    def __init__(self, aos=True, iou_3d=True, iou_bev=True):
        if aos:
            self.aos = meter.AverageValueMeter()
        if iou_3d:
            self.iou_3d = meter.AverageValueMeter()
        if iou_bev:
            self.iou_bev = meter.AverageValueMeter()
    def reset(self, aos=True, iou_3d=True, iou_bev=True):
        if aos:
            self.aos.reset()
        if iou_3d:
            self.iou_3d.reset()
        if iou_bev:
            self.iou_bev.reset()
    def add(self, aos, iou_3d, iou_bev):
        # about value
        if aos is not None:
            self.aos.add(aos)
        if iou_3d is not None:
            self.iou_3d.add(iou_3d)
        if iou_bev is not None:
            self.iou_bev.add(iou_bev)
    def print_log(self, aos=True, iou_3d=True, iou_bev=True):
        log = 'Benchmark_Val:'
        if aos:
            log = log+' aos:'+ str(self.aos.value()[0])
        if iou_3d:
            log = log+' iou_3d:'+ str(self.iou_3d.value()[0])
        if iou_bev:
            log = log+' iou_bev:'+ str(self.iou_bev.value()[0])
        print(log)
        
    def plot(self, writer_val,niter, aos=True, iou_3d=True, iou_bev=True):
        if aos:
            writer_val.add_scalar('Meter/aos',self.aos.value()[0],niter)
        if iou_3d:
            writer_val.add_scalar('Meter/iou_3d',self.iou_3d.value()[0],niter)
        if iou_bev:
            writer_val.add_scalar('Meter/iou_bev',self.iou_bev.value()[0],niter)
            
def print_epoch_info(writer,epoch,niter, lr, lm,lm_val, bm_val):
    # SummaryWriter, LossMeter; Benchmark
    writer.add_text('Text', "epoch:{epoch}, lr:{lr}, loss:{loss},\
                    \nry_cm:{ry_cm}%, ry_cm_val:{ry_cm_val}%, size_cm:{size_cm}%, size_cm_val:{size_cm_val}%,\
                    \niou_3d_val:{iou_3d_val}".format(
                    epoch=epoch,lr=lr,loss=lm.total.value()[0],
                    ry_cm = str(acc_of_confusion_mat(lm.ry_cm)),
                    ry_cm_val = str(acc_of_confusion_mat(lm_val.ry_cm)),
                    size_cm = str(acc_of_confusion_mat(lm.size_cm)),
                    size_cm_val = str(acc_of_confusion_mat(lm_val.size_cm)),
                    iou_3d_val = bm_val.iou_3d.value()[0]),
                    niter)
    print('train_loss:',lm.total.value()[0].item(), ' previous_loss',lm.previous_loss)
    print('confusion_matrix_ry',str(lm.ry_cm.value()),sep='\n')
    print('confusion_matrix_ry_val',str(lm_val.ry_cm.value()),sep='\n')
    print('confusion_matrix_size',str(lm.size_cm.value()),sep='\n')
    print('confusion_matrix_size_val',str(lm_val.size_cm.value()),sep='\n')
            
        
class ToDevice():
    def __init__(self,opt):
        if opt is not None and t.cuda.is_available() and opt.use_gpu :
            dev = "cuda:"+str(opt.cuda_idx)
        else:
            dev = "cpu"
        self.device = t.device(dev)
    
    def trans(self,*args):
        out = []
        for item in args:
            item = item.to(self.device)
            out.append(item)
        return out