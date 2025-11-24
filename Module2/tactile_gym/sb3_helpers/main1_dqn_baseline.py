

import sys 
sys.path.append("###/tactile_gym-main")
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
import sys
import matplotlib.pyplot as plt
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
import torch.nn as nn
import kornia.augmentation as K
from stable_baselines3.common.torch_layers import NatureCNN
from tactile_gym.sb3_helpers.custom.custom_torch_layers import CustomCombinedExtractor
from utils.key_extraction import * 
augmentations = nn.Sequential( K.RandomAffine(degrees=0, translate=[0.05, 0.05], scale=[1.0, 1.0], p=0.5),)


BATCH_SIZE =  256                                                 
EPSILON = 0.9                            
GAMMA = 0.9               
LR = 0.01                 
TARGET_REPLACE_ITER = 100             
MEMORY_CAPACITY = 1000            
N_ACTIONS = 4
N_STATES = 10
max_steps = 150


class Net(nn.Module):
    def __init__(self):                                                         
        super(Net, self).__init__()        
        self.fc1 = nn.Linear(N_STATES,  32)                                    
        # self.fc1.weight.data.normal_(0, 0.1)                                   
        self.out = nn.Linear(32,  N_ACTIONS)                                    
        # self.out.weight.data.normal_(0, 0.1)                                   

    def forward(self, x):                                                      
        x = F.relu(self.fc1(x))                                              
        actions_value = self.out(x)                                 
        return actions_value        
    

class DQN(object):
    def __init__(self):      
        self.lr = LR                                                    
        self.eval_net, self.target_net = Net(), Net()                     
        self.learn_step_counter = 0                                        
        self.memory_counter = 0                                      
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))    # s*2 +r+a   
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=self.lr)    
        self.loss_func = nn.MSELoss()                                  

    def choose_action(self, x):                                               
        x = torch.unsqueeze(torch.FloatTensor(x), 0)                           
        if np.random.uniform() < EPSILON:                                
            actions_value = self.eval_net.forward(x)                
            action = torch.max(actions_value, 1)[1].data.numpy()             
            action = action[0]                                        
        else:                                                      
            action = np.random.randint(0, N_ACTIONS)                     
        return action                                                

    def store_transition(self, s, a, r, s_):                                 
        transition = np.hstack((s, [a, r], s_))                       
        index = self.memory_counter % MEMORY_CAPACITY                    
        self.memory[index, :] = transition                         
        self.memory_counter += 1                                      

    def learn(self):                                                    
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:             
            self.target_net.load_state_dict(self.eval_net.state_dict())   
        self.learn_step_counter += 1                            

        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)         
        b_memory = self.memory[sample_index, :]                         
        b_s = torch.FloatTensor(b_memory[:, :N_STATES])
        b_a = torch.LongTensor(b_memory[:, N_STATES:N_STATES+1].astype(int))
        b_r = torch.FloatTensor(b_memory[:, N_STATES+1:N_STATES+2])
        b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:])

        q_eval = self.eval_net(b_s).gather(1, b_a)
        q_next = self.target_net(b_s_).detach()
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)
        loss = self.loss_func(q_eval, q_target)
        self.optimizer.zero_grad()                                  
        loss.backward()                                        
        self.optimizer.step()                              
        


env = EdgeFollowEnv()     
dqn = DQN()       
i = 0


while True:                                                   
    i += 1
    print('<<<<<<<<<Episode: %s' % i)
    
    obs = env.reset()    
    theta = obs["oracle"][9]
    offset = random.uniform(0.1, 0.8)    # [0.1 - 1]
    # print(offset)
    action = [ offset * np.sin(obs["oracle"][9]) ,  offset * np.cos(obs["oracle"][9]), 0  ]
    obs, _, _, _ = env.step(action)   
    # while True: # z补偿
    #     if obs['oracle'][2] < 0.005:
    #         diff = 0.005 - obs['oracle'][2]
    #         action = [0, 0, diff]
    #         obs, _, _, _ = env.step(action)   
    #     else:
    #         break          
    obs, _, _, _ = env.step(action)  
    # s = obtain_edge1(np.rot90(obs["tactile"], k=-1), theta) # [angle_radians, a,b, distance, sign]
    s = obs["oracle"]
    if s is None:
        continue
    episode_reward_sum = 0                
    
          
    for step in range(max_steps):         
        step += 1 
        # choose action                                  
        action = dqn.choose_action(s)                                    
        if action == 0:
            a = [0.01, 0, 0]
        elif action == 1:
            a = [-0.01, 0, 0]
        elif action == 2:
            a = [0, 0.01, 0]
        else:
            a = [0, -0.01, 0]
        # apply action 
        obs_, r, done, info = env.step(a)   
        # while True: # z补偿
        #     if obs_['oracle'][2] < 0.005:
        #         diff = 0.005 - obs_['oracle'][2]
        #         action_z = [0, 0, diff]
        #         obs_, r, done, info = env.step(action_z)    
        #     else:
        #         break      
        # s_ = obtain_edge1(np.rot90(obs_["tactile"], k=-1), theta) 
        s_ = obs_["oracle"]
        #  refine reward     
        # r = r * 100 + 10. # approach reward （+）
        distance1 =  ((obs['oracle'][0] -  obs['oracle'][6])**2 + (obs['oracle'][1] -  obs['oracle'][7])**2 )**(0.5)
        distance2 =  ((obs_['oracle'][0] -  obs_['oracle'][6])**2 + (obs_['oracle'][1] -  obs_['oracle'][7])**2 )**(0.5)
        R1 = (distance1 - distance2)*10000
        if s_ is None:
            done = True 
            R2 =  - (64**2 + 64**2)**0.5 # 偏离 惩罚
            R = R1 + R2
            s_ = [10,10,10,10,10]
            if random.random() > 0.7:
                dqn.store_transition(s, action, R, s_)               
        else:
            R2 = 0 # - s_[3] # 偏离 惩罚 
            R = R1 + R2 
            dqn.store_transition(s, action, R, s_)   
            
        episode_reward_sum += R                         
        s = s_
        obs = obs_                                          

        if dqn.memory_counter > MEMORY_CAPACITY:       
            dqn.learn()

        if step == max_steps-1:
            done = True 
        if done:      
            print('episode%s---reward_sum: %s' % (i, round(episode_reward_sum, 2)), 'steps: ', step, 'mean_reward', round(episode_reward_sum/step, 2))            
            break     
                                   
  


""" 
DQN baseline
state 为 oracle的10D的state
运行到大约200个episode效果基本可以了, reward 5-10之间 

"""
