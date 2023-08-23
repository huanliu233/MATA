from networks import MATA
import torch
import torch.nn as nn
import torch.nn.functional as F 
import os 
from datasetsA import  get_scheduler
import math
# import adabound



class MATA_trainer(nn.Module):
    def __init__(self, hyperparameters):
        super(MATA_trainer, self).__init__()
        self.Hpixel_model = MATA(hyperparameters['inputdim'], hyperparameters['dim'], hyperparameters['class_num'],
                                        base_dim=hyperparameters['base_dim'], heads=hyperparameters['heads'],
                                        multi_cls_head=hyperparameters['multi_cls_head'],
                                        cls_shared=hyperparameters['cls_shared'],
                                        feature_shared=hyperparameters['feature_shared'],
                                        MSFE=hyperparameters['MSFE'],
                                        L2=hyperparameters['L2'],
                                        weighting=hyperparameters['Weighting'])

        model_params = list(self.Hpixel_model.parameters())
        self.model_opt = torch.optim.Adam([p for p in model_params if p.requires_grad],\
                                             lr=hyperparameters['lr'], \
                                             betas=(hyperparameters['beta1'], hyperparameters['beta2']), \
                                             weight_decay=hyperparameters['weight_decay'], amsgrad =  False)
        self.Hpixel_model_scheduler = get_scheduler(self.model_opt, hyperparameters)
        self.multi_cls_head = hyperparameters['multi_cls_head']
    def forward(self, x_2D):
        self.eval()
        y_pred = self.Hpixel_model(x_2D)
        self.train()
        return y_pred

    def update(self, x_2D, y):
        self.Hpixel_model.train()
        self.model_opt.zero_grad()
        y_pred = self.Hpixel_model(x_2D)
        self.loss = self.Hpixel_model.multi_cls_head_loss(y_pred,y)
        self.loss.backward()
        self.model_opt.step()
    def update_learning_rate(self):
        if self.Hpixel_model_scheduler is not None:
            self.Hpixel_model_scheduler.step()


    def update_learning_rate0(self, it,epoch, max_iter,max_epoch,warm_up_epoch):

        if it + max_iter*(epoch)<warm_up_epoch*max_iter:
            for param_group in self.model_opt.param_groups:
                param_group['lr'] = (param_group['initial_lr'])/warm_up_epoch/max_iter * (it + max_iter*(epoch) ) 
                # param_group['lr'] = (param_group['initial_lr']-1e-6)/10/max_iter * (it + max_iter*(epoch) ) + 1e-6
        else:
            for param_group in self.model_opt.param_groups:
            # self.model_opt.param_groups[0]['lr'] = 0.01*self.lr * (math.cos((it + max_iter*(epoch) ) * (math.pi / max_iter/max_epoch)) + 1)/2  + 1e-6
                param_group['lr'] = param_group['initial_lr'] * (math.cos((it + max_iter*(epoch-warm_up_epoch) ) * (math.pi / max_iter/max_epoch)) + 1)/2   

    def print_loss(self, x_2D, y):
        with torch.no_grad():
            self.Hpixel_model.eval()
            class_pred= self.Hpixel_model(x_2D)
            class_pred = self.Hpixel_model.test_cls(class_pred)
            class_pred = torch.argmax(F.softmax(class_pred, dim=-1), dim=1)
            correct = sum(torch.eq(y.view(-1), class_pred.view(-1)))
            total = class_pred.size(0)
        return correct, total, class_pred