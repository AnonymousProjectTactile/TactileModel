

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import gym
import sys
sys.path.append("###/tactile_gym-main")
import matplotlib.pyplot as plt
import cv2 
import os
import sys
import time
import numpy as np
import stable_baselines3 as sb3
from stable_baselines3.common.callbacks import EvalCallback, EveryNTimesteps, CheckpointCallback
from stable_baselines3 import PPO, SAC
# from sb3_contrib import RAD_SAC, RAD_PPO
from tactile_gym.rl_envs.exploration.edge_follow.edge_follow_env import *
from tactile_gym.sb3_helpers.params import import_parameters
from tactile_gym.sb3_helpers.rl_utils import make_training_envs, make_eval_env
from tactile_gym.sb3_helpers.eval_agent_utils import final_evaluation
from tactile_gym.utils.general_utils import (  save_json_obj,  load_json_obj, convert_json, check_dir,)
from tactile_gym.sb3_helpers.custom.custom_callbacks import ( FullPlottingCallback, ProgressBarManager,)
import argparse
import torch.nn as nn
import kornia.augmentation as K
from stable_baselines3.common.torch_layers import NatureCNN
from tactile_gym.sb3_helpers.custom.custom_torch_layers import CustomCombinedExtractor
from utils.key_extraction import * 
augmentations = nn.Sequential( K.RandomAffine(degrees=0, translate=[0.05, 0.05], scale=[1.0, 1.0], p=0.5),)




 
# env = gym.make('Pendulum-v0')
env = EdgeFollowEnv()
EP_MAX = 10000
HORIZON = 128
LR_v = 2e-5
LR_pi = 2e-5
K_epoch = 8
GAMMA = 0.9
LAMBDA = 0.95
CLIP = 0.2


class Pi_net(nn.Module):
    def __init__(self):
        super(Pi_net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(5, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
        )
        self.mu = nn.Linear(256, 2)
        self.sigma = nn.Linear(256, 2)
        self.optim = torch.optim.Adam(self.parameters(), lr=LR_pi)
 
    def forward(self, x):
        x = self.net(x)
        mu = torch.tanh(self.mu(x)) * 2
        sigma = F.softplus(self.sigma(x)) + 0.001
        return mu, sigma
    
class V_net(nn.Module):
    def __init__(self):
        super(V_net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(5, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )
        self.optim = torch.optim.Adam(self.parameters(), lr=LR_v)
 
    def forward(self, x):
        x = self.net(x)
        return x
    
class Agent(object):
    def __init__(self):
        self.v = V_net()        
        self.pi = Pi_net()
        self.old_pi = Pi_net()        #旧策略网络
        self.old_v = V_net()            #旧价值网络    用于计算上次更新与下次更新的差别 
                                        #ratio
        self.load()
        self.data = []               #用于存储经验
        self.step = 0
 
    def choose_action(self, s):
        with torch.no_grad():
            mu, sigma = self.old_pi(s)
            dis = torch.distributions.normal.Normal(mu, sigma)        #构建分布
            a = dis.sample()   #采样出一个动作
            a = torch.clamp(a, -0.1, 0.1)
        # return a.item()
        return a.tolist()
 
    def push_data(self, transitions):
        self.data.append(transitions)
 
    def sample(self):
        l_s, l_a, l_r, l_s_, l_done = [], [], [], [], []
        for item in self.data:
            s, a, r, s_, done = item
            l_s.append(torch.tensor([s], dtype=torch.float))
            l_a.append(torch.tensor([[a]], dtype=torch.float))
            l_r.append(torch.tensor([[r]], dtype=torch.float))
            l_s_.append(torch.tensor([s_], dtype=torch.float))
            l_done.append(torch.tensor([[done]], dtype=torch.float))
        s = torch.cat(l_s, dim=0)
        a = torch.cat(l_a, dim=0)
        r = torch.cat(l_r, dim=0)
        s_ = torch.cat(l_s_, dim=0)
        done = torch.cat(l_done, dim=0)
        self.data = []
        return s, a, r, s_, done
 
    def updata(self):
        self.step += 1
        s, a, r, s_, done = self.sample()
        for _ in range(K_epoch):
            with torch.no_grad():
                '''loss_v'''
                td_target = r + GAMMA * self.old_v(s_) * (1 - done)
                '''loss_pi'''
                mu, sigma = self.old_pi(s)
                old_dis = torch.distributions.normal.Normal(mu, sigma)
                log_prob_old = old_dis.log_prob(a)
                td_error = r + GAMMA * self.v(s_) * (1 - done) - self.v(s)
                td_error = td_error.detach().numpy()
                A = []
                adv = 0.0
                for td in td_error[::-1]:
                    adv = adv * GAMMA * LAMBDA + td[0]
                    A.append(adv)
                A.reverse()
                A = torch.tensor(A, dtype=torch.float).reshape(-1, 1)
 
            mu, sigma = self.pi(s)
            new_dis = torch.distributions.normal.Normal(mu, sigma)
            log_prob_new = new_dis.log_prob(a)
            ratio = torch.exp(log_prob_new - log_prob_old)
            L1 = ratio * A
            L2 = torch.clamp(ratio, 1 - CLIP, 1 + CLIP) * A
            loss_pi = -torch.min(L1, L2).mean()
            self.pi.optim.zero_grad()
            loss_pi.backward()
            self.pi.optim.step()
 
            loss_v = F.mse_loss(td_target.detach(), self.v(s))
 
            self.v.optim.zero_grad()
            loss_v.backward()
            self.v.optim.step()
        self.old_pi.load_state_dict(self.pi.state_dict())
        self.old_v.load_state_dict(self.v.state_dict())
 
    def save(self):
        torch.save(self.pi.state_dict(), 'pi.pth')
        torch.save(self.v.state_dict(), 'v.pth')
        print('...save model...')
 
    def load(self):
        try:
            self.pi.load_state_dict(torch.load('pi.pth'))
            self.v.load_state_dict(torch.load('v.pth'))
            print('...load...')
        except:
            pass
        
def cal_reward1():
    pass 

def main():
    agent = Agent()
    agent.load()
    max_rewards = -1000000
    epoch = 0
    while True:
        epoch += 1
        s = env.reset()
        theta = s["oracle"][9]
        print(theta)
        offset = 0.2 # 实际走0.02
        reset_a_x  = offset * np.sin(theta) 
        reset_a_y  = offset * np.cos(theta) 
        s, _, _, _ = env.step([reset_a_x, reset_a_y])

        state = obtain_edge1(np.rot90(s["tactile"], k=-1), theta) # state 为line
        if state is None:
            continue
        rewards = 0
        
        for i in range(HORIZON):
            a = agent.choose_action(torch.tensor([state], dtype=torch.float))   
            print(a)
            s_, r, done, info = env.step(a[0]) # 到目标的位距离
            state_ = obtain_edge1(np.rot90(s_["tactile"], k=-1), theta) # state 为line
            r = r - 0.1* state[3]
            rewards += r
            if state_ is None:
                # state_ = [10,10,10,10,10]
                done = True
                r = r - 10 # 超边界
            else:
                agent.push_data((state, a,r, state_, done))
            state = state_
            if done:
                break
        data_length = len(agent.data)
        if data_length > 1:
            agent.updata()
        if epoch % 10 == 0:
            print(epoch, ' ', rewards, ' ', agent.step)
        if max_rewards < rewards:
            max_rewards = rewards
            agent.save()
                      
            
 
if __name__ == '__main__':
    main()
    


