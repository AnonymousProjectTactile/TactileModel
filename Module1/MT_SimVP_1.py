


import sys
sys.path.append("/root/autodl-tmp/Tactile1_Project")
import scipy
print(scipy.__version__)
from scipy.spatial import cKDTree
import json
import csv
import glob
import itertools
import argparse
import shutil
import torch
import progressbar
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from utils.progbar import *
from utils.config import *
from utils.loss_function import *
from utils.RollingMeasure import *
from utils.module import *
from scipy import misc
from torch.utils.data import DataLoader
from torchvision import models
from functools import partial
from torchvision.utils import *
from PIL import Image
from math import sqrt
from torchvision.utils import save_image
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
from utils.augment import * 
from SimVP.modules import *
from pytorch_msssim import ssim 

# from tensorboardX import SummaryWriter
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error # MAE 
from sklearn.metrics import mean_squared_error # MSE  y_true, y_pred,



""" 添加 prediction_force """



init_seed = 1
torch.manual_seed(init_seed)
torch.cuda.manual_seed(init_seed)
torch.cuda.manual_seed_all(init_seed)
np.random.seed(init_seed) 
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.enabled = True



class Datasets(Dataset):
    """ Datasets Process """
    def __init__(self, flist_path = None, config=None, train_flag=False):
        self.config = config
        self.data_path = self.config.DATA_PATH # datasets path 
        self.flist_path = flist_path # train file list 
        self.all_filelist = self.get_all_filelist() # 
        self.label_file = self.get_label_csv(self.config.LABEL_FLIST)
        self.seq_len = self.config.seq_len
        self.train_flag = train_flag
        # if self.train_flag:
        #     if self.config.VideoAug_Type == 'randaug':
        #         self.randaug = RandAugment(n=2, m=5, temp_degree=1, range=1.0)
        #     else:
        #         raise NotImplementedError('Invalid augmentation mode.')

    def get_all_filelist(self):
        filelist = open(self.flist_path, 'rt').read().splitlines()
        return filelist

    def get_label_csv(self, path):
        csv_reader = csv.reader(open(path))
        column = [row[2] for row in csv_reader][1:]
        return column

    def __len__(self):
        return len(self.all_filelist)

    def __getitem__(self, idx):
        par_list = self.all_filelist[idx] # cir_056_005 72 73 75 78 83 84 85 88 90 92 95 3
        folder_name = par_list.split(' ')[0]
        hardness = np.array(np.float32(self.label_file[int(folder_name.split('_')[1])-1])).reshape(1,)
        frames = par_list.split(' ')[1:]
        force_list = open(self.data_path + folder_name +'/force.txt').read().splitlines()
        seques = []
        forces = []
        depth = []
        # sequence and force 
        for i in range(self.seq_len*2): # 0~(seq_len*2-1) 
            img_name = self.data_path + folder_name + '/tactile128/' + frames[i].zfill(3)+'.jpg'
            img = self.get_data(img_name, 'RGB')    
            seques.append(img)    
            F = np.float32(force_list[int(frames[i])-1])
            forces.append(F)
        # depth 
        for i in range(self.seq_len): # 0~(seq_len-1) 
            img_name = (self.data_path + folder_name + '/tactile128/' + frames[i].zfill(3)+'.jpg' ).replace('Skin', 'Skin_depth').replace('tactile128','depth') 
            img = self.get_data(img_name, 'RGB')    
            depth.append(img)    
            
        seq = np.array(seques[:self.seq_len])  
        seq_future = np.array(seques[self.seq_len:])
        depth = np.array(depth[:self.seq_len])  
        forces_current = np.array(forces[:self.seq_len])
        forces_future = np.array(forces[self.seq_len:])
        # if self.train_flag:
        #     if self.config.VideoAug:
        #         seq = self.randaug(seq)     
        seq = self._normalize(seq)
        seq_future = self._normalize(seq_future)
        depth = self._normalize(depth)
        return [seq.astype(np.float32),  
                hardness.astype(np.float32), 
                seq_future.astype(np.float32),
                depth.astype(np.float32),
                forces_current.astype(np.float32),
                forces_future.astype(np.float32)
                # steps.astype(np.float32),
                ]

    def get_data(self, filepath, mode):
        data = Image.open(filepath).convert(mode)
        data = data.resize((128,128)) # 128 
        if mode == 'RGB':
            data = np.array(data).reshape((128,128,3))
        if mode == 'L':
            data = np.array(data).reshape((128,128,1))
        return data

    def _normalize(self, data):
        data = data  / 255. # 0 ~ 1
        return data 

    
    
def stride_generator(N, reverse=False):
    strides = [1, 2]*10
    if reverse: return list(reversed(strides[:N]))
    else: return strides[:N]



class Encoder(nn.Module):
    """ Shared Encoder """
    def __init__(self,C_in, C_hid, N_S):
        super(Encoder,self).__init__()
        strides = stride_generator(N_S)
        self.enc = nn.Sequential(ConvSC(C_in, C_hid, stride=strides[0]), *[ConvSC(C_hid, C_hid, stride=s) for s in strides[1:]])
    
    def forward(self,x):# B*4, 3, 128, 128
        enc1 = self.enc[0](x)
        latent = enc1
        for i in range(1,len(self.enc)):
            latent = self.enc[i](latent)
        return latent,enc1 # 



class Mid_Xnet(nn.Module):
    """ Translator to integrate Spatial-temporal feature """ 
    def __init__(self, channel_in, channel_hid, N_T, incep_ker = [3,5,7,11], groups=8):
        super(Mid_Xnet, self).__init__()
        self.N_T = N_T
        enc_layers = [Inception(channel_in, channel_hid//2, channel_hid, incep_ker= incep_ker, groups=groups)]
        for i in range(1, N_T-1):
            enc_layers.append(Inception(channel_hid, channel_hid//2, channel_hid, incep_ker= incep_ker, groups=groups))
        enc_layers.append(Inception(channel_hid, channel_hid//2, channel_hid, incep_ker= incep_ker, groups=groups))
        dec_layers = [Inception(channel_hid, channel_hid//2, channel_hid, incep_ker= incep_ker, groups=groups)]
        for i in range(1, N_T-1):
            dec_layers.append(Inception(2*channel_hid, channel_hid//2, channel_hid, incep_ker= incep_ker, groups=groups))
        dec_layers.append(Inception(2*channel_hid, channel_hid//2, channel_in, incep_ker= incep_ker, groups=groups))
        self.enc = nn.Sequential(*enc_layers)
        self.dec = nn.Sequential(*dec_layers)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.reshape(B, T*C, H, W)
        # encoder
        skips = []
        z = x
        for i in range(self.N_T):
            z = self.enc[i](z)
            if i < self.N_T - 1:
                skips.append(z)
        # decoder
        z = self.dec[0](z)
        for i in range(1, self.N_T):
            z = self.dec[i](torch.cat([z, skips[-i]], dim=1))
        y = z.reshape(B, T, C, H, W)
        return y



class PredHead(nn.Module):
    """ Dynamics Branch 
    - Input: Spatial-Temporal Feature 
    - Output: Forecast tactile signals 
    """
    def __init__(self,C_hid, C_out, N_S):
        super(PredHead,self).__init__()
        strides = stride_generator(N_S, reverse=True)
        self.dec = nn.Sequential(*[ConvSC(C_hid, C_hid, stride=s, transpose=True) for s in strides[:-1]],
            ConvSC(2*C_hid, C_hid, stride=strides[-1], transpose=True))
        self.readout = nn.Conv2d(C_hid, C_out, 1)
    
    def forward(self, hid, enc1=None):
        for i in range(0,len(self.dec)-1):
            hid = self.dec[i](hid)
        Y = self.dec[-1](torch.cat([hid, enc1], dim=1))
        Y = self.readout(Y)
        return Y
    
  
    
class DepthHead(nn.Module):
    """ Depth Branch 
    - Input: Shared Spatial Feature 
    - Output: Estimated Depth 
    """
    def __init__(self,C_hid, C_out, N_S):
        super(DepthHead,self).__init__()
        strides = stride_generator(N_S, reverse=True)
        self.dec = nn.Sequential(*[ConvSC(C_hid, C_hid, stride=s, transpose=True) for s in strides[:-1]], ConvSC(2*C_hid, C_hid, stride=strides[-1], transpose=True))
        self.readout = nn.Conv2d(C_hid, C_out, 1)
    
    def forward(self, hid, enc1=None):
        for i in range(0,len(self.dec)-1):
            hid = self.dec[i](hid)
        Y = self.dec[-1](torch.cat([hid, enc1], dim=1))
        Y = self.readout(Y)
        return Y



class StiffHead(nn.Module):
    """ Hardness Branch 
    - Input: Spatial-temporal feature 
    - Output: Estimated hardness value
    """
    def __init__(self):
        super(StiffHead, self).__init__()
        self.conv3d = nn.Conv3d(in_channels=16, out_channels=128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool3d = nn.AdaptiveAvgPool3d((1, 1, 1))  # Global pooling across T, H, W
        self.fc = nn.Linear(128, 1)  # Fully connected layer for classification

    def forward(self, x): # Input x: (B, T, C, H, W)
        x = x.permute(0, 2, 1, 3, 4)  # Rearrange to (B, C, T, H, W) for Conv3d (B,16,6,32,32)
        x = self.conv3d(x)  #  (B,128,6,32,32)
        x = self.pool3d(x)  #  (B,128,1,1,1)
        x = x.view(x.size(0), -1)  # (B,128)
        x = self.fc(x)  # Fully connected layer
        return x
  
  
  
class ForceHead(nn.Module):
    """ Force Branch 
    - Input: Shared spatial feature 
    - Output: Estimated force
    """
    def __init__(self):
        super(ForceHead, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=16, out_channels=128, kernel_size=(3, 3), padding=(1, 1))
        self.pool2d = nn.AdaptiveAvgPool2d((1, 1))  # Global pooling across T, H, W
        self.fc = nn.Linear(128, 1)  # Fully connected layer for classification

    def forward(self, x): # Input x: (B*T, C, H, W)        
        x = self.conv2d(x)  #  
        x = self.pool2d(x)  #  
        x = x.view(x.size(0), -1)  # 
        x = self.fc(x)  # 
        return x
  
  
  
class MultiTask(nn.Module):
    def __init__(self, config, hid_S=16, hid_T=256, N_S=4, N_T=8, incep_ker=[3,5,7,11], groups=8):
        super(MultiTask, self).__init__()
        self.config = config 
        T, C, H, W = 8, 3, 128, 128  
        self.enc = Encoder(C, hid_S, N_S) # shared encoder 
        self.hid = Mid_Xnet(T*hid_S, hid_T, N_T, incep_ker, groups) # translator 
        if self.config.task_hardness: 
            self.reg_head = StiffHead() 
        if self.config.task_depth: 
            self.depth_head = DepthHead(hid_S, C, N_S) 
        if self.config.task_force:
            self.force_head = ForceHead()
        if self.config.task_prediction: 
            self.dec = PredHead(hid_S, C, N_S) 

    def forward(self, x_raw): # BTCHW 
        # Y_pred, H_under, H_under_pred, Depth = None, None, None, None  
        Y_pred, H_esti, H_pred_esti, D_esti, F_esti, F_pred_esti = None, None, None, None, None, None   
        B, T, C, H, W = x_raw.shape 
        x = x_raw.view(B*T, C, H, W)  
        
        ## Shared Encoder 
        embed, skip = self.enc(x)   # (B*T,16,32,32), 
        _, C_, H_, W_ = embed.shape # (B*T,16,32,32)
        
        ## Depth Decoder 
        if self.config.task_depth:
            D_esti = self.depth_head(embed, skip) 
            D_esti = D_esti.reshape(B, T, C, H, W) 
            
        ## Force Head  
        if self.config.task_force:
            F_esti = self.force_head(embed)
            F_esti = F_esti.view(B,T)
            
        ## Tasks_need_sequence
        if self.config.task_prediction or self.config.task_hardness or self.task_pred_force:
            z = embed.view(B, T, C_, H_, W_)
            hid = self.hid(z) # [B,T,16,32,32] (BTCHW)
            
            # Hardness Head 
            if self.config.task_hardness:
                H_esti = self.reg_head(hid)
                
            # Prediction Head  
            if self.config.task_prediction:
                hid = hid.reshape(B*T, C_, H_, W_)# 
                Y_pred = self.dec(hid, skip) # 160，1，16，16 
                Y_pred = Y_pred.reshape(B, T, C, H, W) 
                
                ## Prediction_Hardness 
                if self.config.task_pred_hardness or self.config.task_pred_force:
                    Y_pred_in = Y_pred.view(B*T, C, H, W)
                    embed_pred, _ = self.enc(Y_pred_in)
                    if self.config.task_pred_force:
                        F_pred_esti = self.force_head(embed_pred)
                        F_pred_esti = F_pred_esti.view(B,T)
                    if self.config.task_pred_hardness:
                        z_pred = embed_pred.view(B, T, C_, H_, W_)
                        hid_pred = self.hid(z_pred) # [B,6,16,32,32] (BTCHW)
                        H_pred_esti = self.reg_head(hid_pred)
        
        # prediction, hardness, pred_hardness, depth, force, pred_force 
        return Y_pred, H_esti, H_pred_esti, D_esti, F_esti, F_pred_esti
    

    
## metrics 
def calculate_metrics(pred_seq, gt_seq):
    B, T, C, H, W = pred_seq.shape
    mae_all, rmse_all, ssim_all = [],[],[]
    for b in range(B):
        for t in range(T):
            pred = pred_seq[b,t]
            gt = gt_seq[b,t]
            mae = F.l1_loss(pred, gt).item()
            mse = F.mse_loss(pred, gt)
            rmse = torch.sqrt(mse).item()
            ssim_value = ssim(pred.unsqueeze(0), gt.unsqueeze(0), data_range=1.0).item()
            rmse_all.append(rmse)
            mae_all.append(mae)
            ssim_all.append(ssim_value)
    # rmse, mae, ssim 
    return sum(rmse_all)/len(rmse_all), sum(mae_all)/len(mae_all), sum(ssim_all)/len(ssim_all)
            
        



        
class Trainer():
    def __init__(self, args):
        self.config = args
        self.epoch = 0 
        self.best_score = 10000000
        self.model_Reg = MultiTask(self.config).cuda()
        self.model_Reg = nn.DataParallel(self.model_Reg)
        self.solver_Stiffness = optim.Adam(list(self.model_Reg.parameters()), lr=self.config.LR, weight_decay=0.00001) # 0.1
        self.train_dataset = Datasets(flist_path=self.config.Stiffness_TRAIN_FLIST, config=self.config,train_flag=self.config.AugTrain)
        self.test_datasets = Datasets(flist_path=self.config.Stiffness_TEST_FLIST, config = self.config,train_flag=self.config.AugTest)
        self.val_datasets = Datasets(flist_path=self.config.Stiffness_VAL_FLIST, config=self.config,train_flag=self.config.AugVal)
        if self.config.LOAD_MODEL:
            self.load()
        if self.config.mode == 'test':
            self.load()
            self.test()
        
        
        
    def train(self):
        train_loader = DataLoader(dataset=self.train_dataset, batch_size=self.config.BATCH_SIZE, num_workers=self.config.NUM_WORKS, drop_last=False, shuffle=True, pin_memory=True)
        total_batch = len(train_loader)
        while self.epoch < self.config.EPOCH: 
            self.epoch +=1 
            print('====== Epoch: '+ str(self.epoch) + '======')
            bar = progressbar.ProgressBar( max_value=total_batch)
            losses_train = RollingMeasure() # record train loss 
            for index, batch in enumerate(train_loader): 
                bar.update(index)
                seq, H_gt, seq_future, D_gt, F_gt, F_pred_gt = self.cuda(*batch)   # (B,D,W,H,C)
                seq = seq.permute(0,1,4,2,3) # BTHWC --> BTCHW 
                seq_future = seq_future.permute(0,1,4,2,3) # BTHWC --> BTCHW 
                D_gt = D_gt.permute(0,1,4,2,3) # BTHWC --> BTCHW # save_image(seq[0].permute(0,3,1,2)  , 'rgb5.png') 
                Y_pred, H_under, H_under_pred, D_esti, F_esti, F_pred_esti = self.model_Reg(seq) 
                
                loss = 0.
                if self.config.task_hardness: # hardness_branch
                    loss_hardness = L1_loss(H_under , H_gt)
                    loss = loss +  loss_hardness * self.config.weight_hardness
                if self.config.task_force:# force_branch 
                    loss_force = L1_loss(F_esti, F_gt) 
                    loss = loss + loss_force * self.config.weight_force
                if self.config.task_depth: # depth_branch 
                    loss_depth = L1_loss(D_esti, D_gt) 
                    loss = loss + loss_depth * self.config.weight_depth
                if self.config.task_prediction: # prediction_branch
                    loss_prediction = L1_loss(Y_pred , seq_future)
                    loss = loss + loss_prediction * self.config.weight_prediction
                if self.config.task_pred_hardness: # pred_hardness_branch 
                    loss_pred_hardness = L1_loss(H_under_pred , H_gt)
                    loss = loss + loss_pred_hardness * self.config.weight_pred_hardness
                if self.config.task_pred_force:
                    loss_pred_force = L1_loss(F_pred_esti, F_pred_gt)
                    loss = loss + loss_pred_force * self.config.weight_pred_force 
                
                self.solver_Stiffness.zero_grad()
                loss.backward()
                self.solver_Stiffness.step() 
                losses_train(torch.mean(loss).data.cpu().numpy()) # 平均每个step的loss 
                
            losses_val, losses_val_hardness, losses_val_prediction, losses_val_depth, losses_val_force = self.val()
            
            with open(self.config.save_path + '/loss_log.txt', 'a+') as f:
                f.write('Ep: ' + str(self.epoch) + '-- train_all: '+str(losses_train.measure)[:6]+ \
                                                  ' -- val_all: '+str(losses_val.measure)[:6] + \
                                                  ' -- val_hardness: '+str(losses_val_hardness.measure)[:6] + \
                                                  ' -- val_prediction: '+str(losses_val_prediction.measure)[:6] + \
                                                  ' -- val_depth: '+str(losses_val_depth.measure)[:6] + \
                                                  ' -- val_force: '+str(losses_val_force.measure)[:6] + \
                                                  ' -- LR: '+str(self.solver_Stiffness.param_groups[0]['lr'])+ '\n')   
                f.close()
                
            if losses_val.measure < self.best_score:
                self.best_score = losses_val.measure
                with open(self.config.save_path + '/loss_log.txt', 'a+') as f:
                    f.write('saving...'+'\n')   
                    f.close()
                self.save()
                
                
                
    def val(self):
        val_loader = DataLoader( dataset=self.val_datasets,batch_size=self.config.BATCH_SIZE, num_workers=self.config.NUM_WORKS, drop_last=True)
        self.model_Reg.eval()
        bar = progressbar.ProgressBar( max_value=len(val_loader))
        losses_val = RollingMeasure()  # record validation loss 
        losses_val_hardness = RollingMeasure() 
        losses_val_prediction = RollingMeasure() 
        losses_val_depth = RollingMeasure() 
        losses_val_force = RollingMeasure() 
        with torch.no_grad():
            for index, batch in enumerate(val_loader): 
                bar.update(index)
                seq, H_gt, seq_future, D_gt, F_gt, F_pred_gt = self.cuda(*batch)   # (B,D,W,H,C)
                seq = seq.permute(0,1,4,2,3) # BTHWC --> BTCHW 
                seq_future = seq_future.permute(0,1,4,2,3) # BTHWC --> BTCHW 
                D_gt = D_gt.permute(0,1,4,2,3) # BTHWC --> BTCHW # save_image(seq[0].permute(0,3,1,2)  , 'rgb5.png')                                
                Y_pred, H_under, H_under_pred, D_esti, F_esti, F_pred_esti = self.model_Reg(seq) 
                
                loss = 0.
                if self.config.task_hardness:
                    loss_hardness = L1_loss(H_under , H_gt)
                    loss = loss + loss_hardness * self.config.weight_hardness
                if self.config.task_force:# force_branch 
                    loss_force = L1_loss(F_esti, F_gt) 
                    loss = loss + loss_force * self.config.weight_force
                if self.config.task_depth:
                    loss_depth = L1_loss(D_esti, D_gt) 
                    loss = loss + loss_depth * self.config.weight_depth
                if self.config.task_prediction:
                    loss_prediction = L1_loss(Y_pred , seq_future)
                    loss = loss + loss_prediction * self.config.weight_prediction
                if self.config.task_pred_hardness:
                    loss_pred_hardness = L1_loss(H_under_pred , H_gt)
                    loss = loss + loss_pred_hardness * self.config.weight_pred_hardness
                if self.config.task_pred_force:
                    loss_pred_force = L1_loss(F_pred_esti, F_pred_gt)
                    loss = loss + loss_pred_force * self.config.weight_pred_force 
                    
                losses_val(torch.mean(loss).data.cpu().numpy())# 记录 weighted loss in valid_data 
                
                if self.config.task_hardness:
                    losses_val_hardness(torch.mean(loss_hardness).data.cpu().numpy())
                if self.config.task_prediction:
                    losses_val_prediction(torch.mean(loss_prediction).data.cpu().numpy())
                if self.config.task_depth:
                    losses_val_depth(torch.mean(loss_depth).data.cpu().numpy())
                if self.config.task_force:
                    losses_val_force(torch.mean(loss_force).data.cpu().numpy())
                    
        self.model_Reg.train()
        return [losses_val, losses_val_hardness, losses_val_prediction, losses_val_depth, losses_val_force]
    
    
    
    def test(self):
        
        test_loader = DataLoader( dataset=self.test_datasets,batch_size=self.config.BATCH_SIZE, num_workers=self.config.NUM_WORKS, drop_last=True)
        self.model_Reg.eval()
        # losses_hardness_test = RollingMeasure() 
        # losses_pred_test = RollingMeasure() 
        # losses_depth_test = RollingMeasure()
        # losses_force_test = RollingMeasure() 
        h_estim = np.array([])
        h_gt = np.array([])
        f_estim = np.array([])
        f_gt = np.array([])
        depth_mae = RollingMeasure()
        depth_rmse = RollingMeasure()
        depth_ssim = RollingMeasure()
        future_mae = RollingMeasure()
        future_rmse = RollingMeasure()
        future_ssim = RollingMeasure()
        
        bar = progressbar.ProgressBar( max_value=len(test_loader))
        with torch.no_grad():
            for index, batch in enumerate(test_loader): 
                bar.update(index)
                seq, H_gt, seq_future, D_gt, F_gt, F_pred_gt = self.cuda(*batch)    # (B,D,W,H,C)
                seq = seq.permute(0,1,4,2,3) # BTHWC --> BTCHW 
                seq_future = seq_future.permute(0,1,4,2,3) # BTHWC --> BTCHW 
                D_gt = D_gt.permute(0,1,4,2,3) # BTHWC --> BTCHW # save_image(seq[0].permute(0,3,1,2)  , 'rgb5.png')                                               
                Y_pred, H_under, H_under_pred, D_esti, F_esti, F_pred_esti = self.model_Reg(seq) 
                
                loss = 0.
                if self.config.task_hardness:
                    # loss_hardness = L1_loss(H_under , H_gt)
                    # loss = loss+  loss_hardness * self.config.weight_hardness
                    # losses_hardness_test(torch.mean(loss_hardness).data.cpu().numpy())
                    h_estim = np.append(h_estim, H_under.cpu().numpy())
                    h_gt = np.append(h_gt, H_gt.cpu().numpy())
                    
                if self.config.task_force:# force_branch 
                    # loss_force = L1_loss(F_esti, F_gt) 
                    # loss = loss + loss_force * self.config.weight_force
                    # losses_force_test(torch.mean(loss_force).data.cpu().numpy())
                    f_estim = np.append(f_estim, F_esti.reshape(-1, 1).cpu().numpy()) 
                    f_gt = np.append(f_gt, F_gt.reshape(-1, 1).cpu().numpy()) 
                    f_estim = f_estim * 37.5 * 9.8 * 0.001 # F=mg  
                    f_gt = f_gt * 37.5 * 9.8 * 0.001 
                    
                if self.config.task_depth:
                    # loss_depth = L1_loss(D_esti, D_gt) # (32,8,3,128,128) (32,8,3,128,128)
                    # loss = loss + loss_depth * self.config.weight_depth 
                    # losses_depth_test(torch.mean(loss_depth).data.cpu().numpy())
                    # MAE, RMSE, SSIM 
                    rmse, mae, ssim = calculate_metrics(D_esti, D_gt)
                    depth_mae(mae)
                    depth_rmse(rmse)
                    depth_ssim(ssim)
                    
                if self.config.task_prediction:
                    # loss_prediction = L1_loss(Y_pred , seq_future)
                    # loss = loss + loss_prediction * self.config.weight_prediction
                    # losses_pred_test(torch.mean(loss_prediction).data.cpu().numpy())
                    # MAE, RMSE, SSIM  
                    rmse, mae, ssim = calculate_metrics(Y_pred, seq_future)
                    future_mae(mae)
                    future_rmse(rmse)
                    future_ssim(ssim)
                                    
        
        ### Depth
        if self.config.task_depth:
            print("==========Depth=============",'\n') 
            print('Depth_mae: ', round(depth_mae.measure,5), '\n')
            print('Depth_rmse: ', round(depth_rmse.measure,5), '\n')
            print('Depth_ssim: ', round(depth_ssim.measure,5), '\n')
            print("=======================",'\n')
        
        ### Prediction 
        if self.config.task_prediction:
            print("==========Prediction=============",'\n') 
            print('Future_mae: ', round(future_mae.measure,5), '\n')
            print('Future_rmse: ', round(future_rmse.measure,5), '\n')
            print('Future_ssim: ', round(future_ssim.measure,5), '\n')
            print("=======================",'\n')
        
        ### Force  
        if self.config.task_force:
            txt_name = 'test_force.txt'
            with open( txt_name , 'w') as file:
                content_cap = 'GT\tPred\tSensor\n'
                file.write(content_cap)
            with open( txt_name , 'a') as file:
                for i in range(len(f_gt)):
                    content =  str(f_gt[i]) + '\t' +  str(f_estim[i]) + '\t' +  '01' + '\n'
                    file.write(content)
            print("==========Force=============",'\n')
            print("Force MAE: ", round(mean_absolute_error(f_gt, f_estim), 5),'\n')
            print("Force RMSE: ", round(sqrt(mean_squared_error(f_gt, f_estim)),5),'\n')
            print("Force R2: ",  round(r2_score(f_gt, f_estim),5),'\n')
            print("=======================",'\n')
        
        ### Hardness   
        if self.config.task_hardness:
            txt_name = 'test_hardness.txt'
            with open( txt_name , 'w') as file:
                content_cap = 'GT\tPred\tSensor\n'
                file.write(content_cap)
            with open( txt_name , 'a') as file:
                for i in range(len(h_gt)):
                    content =  str(h_gt[i])[:4] + '\t' +  str(h_estim[i])[:4] + '\t' +  '01' + '\n'
                    file.write(content)        
            print("==========Hardness=============",'\n')
            print("Hardness MAE: ", round(mean_absolute_error(h_gt, h_estim),5),'\n')
            print("Hardness RMSE: ", round(sqrt(mean_squared_error(h_gt, h_estim)),5),'\n')
            print("Hardness R2: ",  round(r2_score(h_gt, h_estim),5),'\n')
            print("=======================",'\n')
        print(1)

        
  
    def cuda(self, *args):
        return (item.to(self.config.DEVICE) for item in args)

    def load(self):
        model_path = self.config.load_path + '/checkpoints.pt'
        print('\nLoading ...\n', model_path)
        checkpoint = torch.load(model_path, map_location=self.config.DEVICE)
        self.model_Reg.load_state_dict(checkpoint['main_task_model'])  
        self.solver_Stiffness.load_state_dict(checkpoint['solver_main'])
        self.epoch = checkpoint['epoch'] 
        print('Epoch: '+ str(self.epoch))

    def save(self):
        print('\nSaving ...\n')
        torch.save({'epoch': self.epoch,
                    'main_task_model': self.model_Reg.state_dict(),
                    'solver_main': self.solver_Stiffness.state_dict(), },  self.config.save_path + '/checkpoints.pt')
        if self.epoch > 90:
            torch.save({'epoch': self.epoch,
                    'main_task_model': self.model_Reg.state_dict(),
                    'solver_main': self.solver_Stiffness.state_dict(), },  self.config.save_path + '/checkpoints_'+str(self.epoch)+'.pt')
        if self.epoch == 1:
            argsDict = self.config.__dict__
            with open(self.config.save_path +'/setting.txt', 'w') as f:
                f.writelines('------------------ start ------------------' + '\n')
                for eachArg, value in argsDict.items():
                    f.writelines(eachArg + ' : ' + str(value) + '\n')
                f.writelines('------------------- end -------------------')


    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--LABEL_FLIST', type=str, default= 'Data/label_data1.csv')
    parser.add_argument('--DEVICE', type=str, default= 'cuda')
    parser.add_argument('--LR', type=float, default=1e-04, help='the iterator of val')
    parser.add_argument('--NUM_WORKS', type=int, default=12 )
    parser.add_argument('--EPOCH', type=int, default= 150)
    parser.add_argument('--BATCH_SIZE', type=int, default= 32)
    parser.add_argument('--AugTrain', type=bool, default= True)
    parser.add_argument('--AugVal', type=bool, default= True)  
    parser.add_argument('--AugTest', type=bool, default= False)  
    parser.add_argument('--seq_len', type=int, default = 8) 
    parser.add_argument('--VideoAug_Type', type=str, default= 'randaug') 
    parser.add_argument('--LOAD_MODEL', type=bool, default= False)  
    parser.add_argument('--mode',type=str, default='test', choices=['train', 'test'])  
    parser.add_argument('--diam',type=str, default='all97')   
    parser.add_argument('--num',type=int, default= 1 ) 
    
    # choose task   
    parser.add_argument('--task_hardness',      type=bool, default= True )   
    parser.add_argument('--task_depth',         type=bool, default= True ) 
    parser.add_argument('--task_force',         type=bool, default= True ) 
    parser.add_argument('--task_prediction',    type=bool, default= True )   
    parser.add_argument('--task_pred_hardness', type=bool, default= True ) 
    parser.add_argument('--task_pred_force',    type=bool, default= True ) 
    
    # task weight   
    parser.add_argument('--weight_hardness',type=float, default= 1 )
    parser.add_argument('--weight_depth',type=float, default= 1 )   
    parser.add_argument('--weight_force',type=float, default= 1 )   
    parser.add_argument('--weight_prediction',type=float, default= 1 )   
    parser.add_argument('--weight_pred_hardness',type=float, default= 1 )  
    parser.add_argument('--weight_pred_force',type=float, default= 1 )   
    
    args = parser.parse_args()
    args.DATA_PATH = 'Data/Skin/'
    # single task  
    args.save_path = 'models/MT_SimVP_New_2th' + \
        '_h'+str(args.task_hardness)[0] + '_' + str(args.weight_hardness) + \
        '_p'+str(args.task_prediction)[0] + '_' + str(args.weight_prediction) + \
        '_ph'+str(args.task_pred_hardness)[0] + '_' + str(args.weight_pred_hardness) +\
        '_d'+str(args.task_depth)[0] + '_' + str(args.weight_depth) + \
        '_f'+str(args.task_force)[0] + '_' + str(args.weight_force) + '/' 
                                            
        
    args.load_path =  args.save_path         
    args.Stiffness_TRAIN_FLIST =  'Data/Flist/Tactile1_Flist/' + str(args.num) + '/'+ args.diam + '/train.flist'
    args.Stiffness_TEST_FLIST =   'Data/Flist/Tactile1_Flist/' + str(args.num) + '/'+  args.diam + '/test.flist'
    args.Stiffness_VAL_FLIST =    'Data/Flist/Tactile1_Flist/' + str(args.num) + '/'+  args.diam + '/val.flist'       

    
    if args.mode == 'train':
        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)
        shutil.copy( sys.argv[0], args.save_path+'/copy.py')
        
    trainer = Trainer(args) 
    if args.mode == 'train':
        trainer.train()
    
    



