from datasetsA import compute_AAK1, get_all_datasets,  HyDataset
import argparse  # 传输参数值的命令
from trainer import MATA_trainer
import torch.backends.cudnn as cudnn
import torch
import torch.utils.data as data
import torch.nn.functional as F

try:
    from itertools import izip as zip
except ImportError:  # will be 3.x series
    pass
import numpy as np
import time
from sklearn.metrics import confusion_matrix
from scipy import io
from networks import MATA

torch.cuda.set_device('cuda:0')
torch.set_num_threads(1)
parser = argparse.ArgumentParser()
parser.add_argument('--data_root', type=str, default='./data/IndianPines_ln_100.mat', help="data_root")
parser.add_argument('--patch_size', type=int, default=13, help="patch_size")
parser.add_argument('--batch_size', type=int, default=64, help="batch_size")
parser.add_argument('--batch_size_test', type=int, default=2048, help="batch_size_test")
parser.add_argument('--supervision', type=str, default='full', help="supervision")
parser.add_argument('--ignored_labels', type=int, default=[0], help="ignored_labels")
parser.add_argument('--flip_augmentation', type=bool, default=False, help="filp_augmentation")
parser.add_argument('--radiation_augmentation', type=bool, default=False, help="radiation_augmentation")
parser.add_argument('--lr', type=float, default=0.005, help="lr")
parser.add_argument('--lr_policy', type=str, default='step', help="step/zmm")
parser.add_argument('--gamma', type=float, default=0.6, help="gamma")
parser.add_argument('--step_size', type=int, default=20, help="step_size")
parser.add_argument('--beta1', type=float, default=0.9, help="beta1")
parser.add_argument('--beta2', type=float, default=0.99, help="beta2")
parser.add_argument('--weight_decay', type=float, default=0, help="weight_decay")
parser.add_argument('--weight_init', type=str,default='xavier', help="xavier/kaiming")
parser.add_argument('--output_path', type=str, default='.', help="outputs path")
parser.add_argument('--standard', type=bool, default=False, help="True/False")
parser.add_argument('--decay', type=float, default=0.001, help="0.0001")
config = parser.parse_args()
config = vars(config)
hsi, map_train, map_test = get_all_datasets(config)
config['inputdim'] = hsi.shape[2]
config['class_num'] = int(np.max(map_train))
config['dim'] = 512
config['head'] = 8
config['multi_cls_head'] = config['patch_size']//2*5 +1


# torch.use_deterministic_algorithms(True)
# cudnn.benchmark = False #选择最快的卷积算法
# cudnn.deterministic = True
# Load experiment setting

# 放到cuda上的命令
hsi, map_train, map_test = get_all_datasets(config)

if config['standard']:
    x = np.reshape(hsi, -1)
    meanhsi = np.mean(x)
    sigmahsi = np.sqrt(np.var(x))
    hsi = (hsi - meanhsi) / (sigmahsi)


    
r = config['patch_size'] // 2
hsi = np.pad(hsi, ((r, r), (r, r), (0, 0)), 'symmetric')
map_train = np.pad(map_train, ((r, r), (r, r)), 'constant', constant_values=(0, 0))
map_test = np.pad(map_test, ((r, r), (r, r)), 'constant', constant_values=(0, 0))

hsi_con, map_train_con = hsi, map_train

hsimap_train = HyDataset(hsi_con, map_train_con, transform=None, **config)
hsimap_test = HyDataset(hsi, map_test, transform=None, **config)
test_loader = data.DataLoader(hsimap_test, batch_size=config['batch_size_test'], pin_memory=True, shuffle=False)
config['heads'] = 8
config['base_dim'] = 64
config['cls_shared'] = True
config['feature_shared'] = True
config['MSFE'] = True
config['L2'] = True
config['Weighting'] = True
trainer = MATA_trainer(config)
trainer.cuda()
max_epoch = 2
OA = 0
AA = 0
Kappa = 0 
for epoch in range(max_epoch):
    # print("epoch = %03d" % epoch)
    train_loader = data.DataLoader(hsimap_train, batch_size=config['batch_size'], pin_memory=True, shuffle=True)
    total_correct_num = 0
    total_num = 0
    loss = 0
    max_iter = len(train_loader)
    for it, (hsi_2D, y) in enumerate(train_loader):
        # trainer.update_learning_rate0(it,epoch,max_iter,max_epoch,warm_up_epoch)
        hsi_2D, y = hsi_2D.cuda().detach(), y.cuda().detach()
        trainer.update(hsi_2D, y)
        loss += trainer.loss
        # trainer.update_learning_rate1(a,config['epoch_max'],config['lr'], config['step_size'], config['gamma'])
    print("epoch %04d----Loss: %04f" % (epoch, loss))
    if epoch == max_epoch-1: 
        y0 = torch.Tensor().cuda()
        y1 = torch.Tensor().cuda()
        trainer.Hpixel_model.eval()
        with torch.no_grad():
            for its, (hsi_2D_test, y_test) in enumerate(test_loader):
                hsi_2D_test, y_test = hsi_2D_test.cuda().detach(), y_test.cuda().detach()
                temp_correct, temp_total, class_pred = trainer.print_loss(hsi_2D_test, y_test)
                total_correct_num += temp_correct
                total_num += temp_total
                y0 = torch.cat((y0,y_test))
                y1= torch.cat((y1, class_pred))
        acc1 = total_correct_num / total_num
        OA, AA, Kappa,SA = compute_AAK1(y0,y1)
        print(OA.cpu().numpy())
        torch.save(trainer.Hpixel_model.state_dict(), './Params/IndianPines_MATA.pth')
    trainer.update_learning_rate()

config['supervision'] = 'semi'
hsimap_test = HyDataset(hsi, map_test,transform=None, **config)
test_loader = data.DataLoader(hsimap_test, batch_size=config['batch_size_test'], pin_memory=True, shuffle=False)
testCANnet = MATA(  config['inputdim'],config['dim'],config['class_num'],
                    base_dim=config['base_dim'], 
                    heads=config['heads'],
                    multi_cls_head =config['multi_cls_head'],
                    cls_shared=config['cls_shared'],
                    feature_shared=config['feature_shared'],
                    MSFE=config['MSFE'],
                    L2=config['L2'],
                    weighting=config['Weighting'] )
testCANnet.load_state_dict(torch.load('./Params/IndianPines_MATA.pth'))
testCANnet.cuda()
testCANnet.eval()
total_correct_num = 0
total_num = 0
total_correct_num = 0
y0 = torch.Tensor().cuda()
start = time.time()
for its, (hsi_2D,y) in enumerate(test_loader):
    with torch.no_grad():
        hsi_2D,y = hsi_2D.cuda().detach(), y.cuda().detach()
        class_pred =  testCANnet(hsi_2D)
        class_pred = torch.argmax(testCANnet.test_cls(class_pred),dim=1)
        y0 = torch.cat( (y0,class_pred))
test_time = time.time() - start
io.savemat("y_IndianPines_MATA.mat",{"y":y0.cpu().numpy()})