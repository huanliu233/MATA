# -*- coding: utf-8 -*-
"""
This file contains the PyTorch dataset for hyperspectral images and
related helpers.
"""
from typing import DefaultDict
import numpy as np
import torch
from torch.autograd.grad_mode import enable_grad
import torch.utils
import torch.utils.data
from scipy import io
import h5py
import yaml
import time
import torch.nn.init as init
from torch.optim import lr_scheduler
import math
import os
import torch.nn.functional as F
import random
import torch.backends.cudnn as cudnn
def get_scheduler(optimizer, hyperparameters, iterations=-1):
    if 'lr_policy' not in hyperparameters or hyperparameters['lr_policy'] == 'constant':
        scheduler = None # constant scheduler
    elif hyperparameters['lr_policy'] == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=hyperparameters['step_size'],
                                        gamma=hyperparameters['gamma'], last_epoch=iterations)
    # elif hyperparameters['lr_policy'] == 'inv_sqrt':
    # # originally used for Transformer (in Attention is all you need)
    #     def lr_lambda(epoch):
    #         d =  hyperparameters['dim']
    #         warm =  hyperparameters['warm_up_steps']
    #         return d**(-0.5)*min((epoch+1)**(-0.5),(epoch+1)*warm**(-1.5))
    #     scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda,last_epoch=iterations)
    elif hyperparameters['lr_policy'] == 'zmm':
        # originally used for Transformer (in Attention is all you need)
        def lr_lambda(epoch):
            decay =  hyperparameters['decay']
            return 1/(1+decay*iterations)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda,last_epoch=iterations)    
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', hyperparameters['lr_policy'])
    return scheduler

def weights_init(init_type='gaussian'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find('Linear') == 0) and hasattr(m, 'weight'):
            # print m.__class__.__name__
            if init_type == 'gaussian':
                init.normal_(m.weight.data, 0.0, 0.02)
            elif init_type == 'one':
                init.zeros_(m.weight.data)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'xavier_uniform': 
                init.xavier_uniform_(m.weight.data, gain=math.sqrt(2))  #
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)

    return init_fun

def get_config(config):
    with open(config, 'r', encoding = 'utf-8') as stream:
        return yaml.load(stream, Loader=yaml.FullLoader)

def get_all_datasets(config):
    c = os.path.basename(config['data_root'])
    c = c.split('.')[0]
    if c == 'hhk_ln' or c == 'hhk_ori' or c=='hhk_01' or c=='hhk_01ln':
        dataset = h5py.File(config['data_root'])
        hsi = dataset['hsi'][:]
        map_train = dataset['map_train'][:]
        map_test = dataset['map_test'][:]
        hsi = hsi.transpose((2,1,0 ))
        map_train = map_train.transpose((1,0))
        map_test = map_test.transpose((1, 0))

        # map_test = np.array(map_test)
    else:
        dataset = io.loadmat(config['data_root'])
        hsi = dataset['hsi']
        map_train = dataset['map_train']
        map_test = dataset['map_test']

    return hsi, map_train, map_test



class HyDataset(torch.utils.data.Dataset):
    """ Generic class for a hyperspectral scene """

    def __init__(self, hsi, map,transform=None, **hyperparams):
        """
        Args:
            data: 3D hyperspectral image
            gt: 2D array of labels
            patch_size: int, size of the spatial neighbourhood
            center_pixel: bool, set to True to consider only the label of the
                          center pixel
            data_augmentation: bool, set to True to perform random flips
            supervision: 'full' or 'semi' supervised algorithms
        """
        super(HyDataset, self).__init__()
        self.transform = transform
        self.hsi = hsi
        self.labels = map
        # self.name = hyperparams['dataset']
        self.ignored_labels = hyperparams['ignored_labels']
        self.patch_size = hyperparams['patch_size']
        supervision = hyperparams['supervision']
        self.flip_augmentation = hyperparams['flip_augmentation']
        self.radiation_augmentation = hyperparams['radiation_augmentation']
        # Fully supervised : use all pixels with label not ignored
        if supervision == 'full':
            mask = np.ones_like(map)
            for l in self.ignored_labels:
                mask[map == l] = 0
        # Semi-supervised : use all pixels, except padding
        elif supervision == 'semi':
            mask = np.ones_like(map)
        x_pos, y_pos = np.nonzero(mask)
        p = self.patch_size // 2
        self.indices = np.array([(x,y) for x,y in zip(x_pos, y_pos) if x > p-1 and x < hsi.shape[0] - p and y > p-1 and y < hsi.shape[1] - p])
        self.labels = [self.labels[x,y] for x,y in self.indices] 

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        x, y = self.indices[i]
        x1, y1 = x - self.patch_size // 2, y - self.patch_size // 2
        x2, y2 = x1 + self.patch_size, y1 + self.patch_size

        hsi_2D = self.hsi[x1:x2, y1:y2,:]
        label = self.labels[i] - 1 # 类别从0开始
        hsi_2D = np.asarray(np.copy(hsi_2D).transpose((2, 0, 1)), dtype='float32')
        label = np.asarray(np.copy(label), dtype='int64')
        # Load the data into PyTorch tensors
        hsi_2D = torch.from_numpy(hsi_2D)
        label = torch.from_numpy(label)
        # Extract the label if needed      
        return (hsi_2D, label)


def prepare_sub_folder(output_directory):
    image_directory = os.path.join(output_directory, 'images')
    if not os.path.exists(image_directory):
        print("Creating directory: {}".format(image_directory))
        os.makedirs(image_directory)
    checkpoint_directory = os.path.join(output_directory, 'checkpoints')
    if not os.path.exists(checkpoint_directory):
        print("Creating directory: {}".format(checkpoint_directory))
        os.makedirs(checkpoint_directory)
    return checkpoint_directory, image_directory

def get_model_list(dirname, key):
    if os.path.exists(dirname) is False:
        return None
    gen_models = [os.path.join(dirname, f) for f in os.listdir(dirname) if #listdir 文件和文件夹列表
                  os.path.isfile(os.path.join(dirname, f)) and key in f and ".pt" in f] 
    if gen_models is None:
        return None
    gen_models.sort()
    last_model_name = gen_models[-1]
    return last_model_name









def confusion(y0,y1,class_num):
    A = torch.zeros(class_num,class_num).cuda()
    for i in range(y0.size(0)):
        A[y0[i].int(),y1[i].int()] += 1  
    return A

def acc(confusion_matrix):
    SA = torch.diag(confusion_matrix)/torch.sum(confusion_matrix,dim=1)
    AA = torch.mean(SA)
    OA = torch.sum(torch.diag(confusion_matrix))/torch.sum(confusion_matrix)
    pe = torch.sum(torch.sum(confusion_matrix,dim=1,keepdim=True)*torch.sum(confusion_matrix,dim=1,keepdim=True))\
        /torch.sum(torch.sum(confusion_matrix))**2
    Kappa = (OA-pe)/(1-pe)
    
    return SA,AA,OA,Kappa


def train_ratio(map_train,r):
    m,n = map_train.shape
    map_train = map_train.flatten()
    num_class = int(np.amax(map_train))
    map_train_new = np.zeros_like(map_train)
    for i in range(1,num_class+1):
        pos = np.where(map_train==i)
        lenpos = len(pos[0])
        np.random.seed(2023)
        pos_new = np.random.permutation(pos).flatten()
        map_train_new[pos_new[:int(round(lenpos*r))]] = i

    return map_train_new.reshape(m,n)


def compute_AAK1(y0,y1):
    m = y0.shape
    no_class = int(torch.max(y0)) + 1
    confusion_matrix = confusion(y0,y1,no_class)
    SA = torch.diag(confusion_matrix)/torch.sum(confusion_matrix,dim=1)
    AA = torch.mean(SA)
    OA = torch.sum(torch.diag(confusion_matrix))/torch.sum(confusion_matrix)
    pe = torch.sum(torch.sum(confusion_matrix,dim=1,keepdim=True)*torch.sum(confusion_matrix,dim=1,keepdim=True))\
        /torch.sum(torch.sum(confusion_matrix))**2
    Kappa = (OA-pe)/(1-pe)
    return OA*100,AA*100,Kappa*100,SA*100 