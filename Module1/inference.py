




import sys
sys.path.append("Module1/")
import cv2 
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
# import progressbar
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
from MT_SimVP_1 import * 



class MT_Infer:

    def __init__(self, config):
        self.config = config 
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.seq_len = 8
        self.task_hardness = True
        self.task_depth = True
        self.task_force = True
        self.task_prediction = True
        self.task_pred_hardness = True
        self.task_pred_force = True
        self.model_Reg = MultiTask(self.config).cuda()
        self.model_Reg = nn.DataParallel(self.model_Reg)

    def _load_model(self):
        model_path = self.config.load_path + '/checkpoints.pt'
        print('\nLoading ...\n', model_path)
        checkpoint = torch.load(model_path, map_location=self.config.DEVICE)
        self.model_Reg.load_state_dict(checkpoint['main_task_model'])  


    def get_data(self, data, mode='RGB'):
        
        data = data.resize((128,128)) # 128 
        if mode == 'RGB':
            data = np.array(data).reshape((128,128,3))
        if mode == 'L':
            data = np.array(data).reshape((128,128,1))
        return data

    def process_data(self, seq):
        seques = []
        for img in seq:
            img = img.resize((128,128)) # 128 
            img = np.array(img).reshape((128,128,3))
            seques.append(img)
        seques = np.array(seques[:self.seq_len])
        seques = seques / 255. 
        return seques.astype(np.float32)



    def inference(self, sequence_data):
        input_numpy = self.process_data(seq)
        input_tensor = torch.tensor(input_numpy, dtype=torch.float32).to(self.device)
        input_tensor = input_tensor.unsqueeze(0)
        input_tensor = input_tensor.permute(0,1,4,2,3) #
        with torch.no_grad():
            Y_pred, H_esti, H_pred_esti, G_esti, F_esti, _ = self.model_Reg(input_tensor)
        results = {
            'input_sequence': input_tensor.cpu().numpy(),
            'hardness': H_esti,
            'force': F_esti,
            'geom': G_esti
        }
        return results 






if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    # Perception 
    parser.add_argument('--DEVICE', type=str, default= 'cuda')
    parser.add_argument('--NUM_WORKS', type=int, default=12 )
    parser.add_argument('--seq_len', type=int, default = 8)  
    parser.add_argument('--load_path', type=str, default= 'Module1/Out_1/MT_SimVP_New_2th_hT_1_pT_1_phT_1_dT_1_fT_1/checkpoints_140.pt')
    parser.add_argument('--task_hardness',      type=bool, default= True )   
    parser.add_argument('--task_depth',         type=bool, default= True ) 
    parser.add_argument('--task_force',         type=bool, default= True ) 
    parser.add_argument('--task_prediction',    type=bool, default= True )   
    parser.add_argument('--task_pred_hardness', type=bool, default= True ) 
    parser.add_argument('--task_pred_force',    type=bool, default= True ) 
    args = parser.parse_args()

    Perception = MT_Infer(args)

    image_paths = [
        "Module1/sample_data/data/061.jpg",
        "Module1/sample_data/data/062.jpg", 
        "Module1/sample_data/data/063.jpg",
        "Module1/sample_data/data/064.jpg",
        "Module1/sample_data/data/065.jpg",
        "Module1/sample_data/data/066.jpg",
        "Module1/sample_data/data/067.jpg", 
        "Module1/sample_data/data/068.jpg"
    ]

    seq = []
    for filepath in image_paths:
        data = Image.open(filepath).convert('RGB')
        seq.append(data)

    result = Perception.inference(seq)
    result 






