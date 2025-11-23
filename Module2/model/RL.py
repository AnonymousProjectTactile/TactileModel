
import sys
sys.path.append('Module2/')

import torch 
import numpy as np 
import torch.nn as nn 
import cv2 
# import gym
import random 
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tactile_gym.rl_envs.exploration.edge_follow.edge_follow_env import *
import os 
import math 
import matplotlib.pyplot as plt 




def point_to_line(theta, x1, y1, x0=64, y0=64):
    """
    """
    theta_rad = np.radians(theta)
    if abs(theta) == 90:
        distance = abs(x0 - x1)
        if x1 > x0:
            pos = 1
        elif x1 < x0:
            pos = -1
        else:
            pos = 0
    else:
        m = np.tan(theta_rad)
        A, B = m, -1
        C = y1 - m * x1
        distance = abs(A * x0 + B * y0 + C) / np.sqrt(A ** 2 + B ** 2)
        y0_pred = m * x0 + C  
        if y0_pred > y0:
            pos = 1
        elif y0_pred < y0:
            pos = -1
        else:
            pos = 0
    return distance, pos


def obtain_edge2(img, angle):
    center = np.array([64, 64])
    ret, binary_image = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    largest_contour = max(contours, key=cv2.contourArea)
    edge_points = largest_contour.reshape(-1, 2)
    distances = np.linalg.norm(edge_points - center, axis=1)
    nearest_point = edge_points[np.argmin(distances)]
    if nearest_point[0] < 8 or nearest_point[0] > 120 or nearest_point[1] < 8 or nearest_point[1] > 120:
        return None

    normal_vector = nearest_point - center
    normal_angle = np.arctan2(normal_vector[1], normal_vector[0])
    tangent_angle_clockwise = normal_angle + np.pi / 2
    tangent_angle_counterclockwise = normal_angle - np.pi / 2

    if tangent_angle_clockwise > np.pi:
        tangent_angle_clockwise = tangent_angle_clockwise % (-np.pi)
    if tangent_angle_counterclockwise < -np.pi:
        tangent_angle_counterclockwise = tangent_angle_counterclockwise % (np.pi)
    tangent_angle_clockwise = np.degrees(tangent_angle_clockwise)
    tangent_angle_counterclockwise = np.degrees(tangent_angle_counterclockwise)
    diff_clockwise = abs(tangent_angle_clockwise - angle)
    diff_counterclockwise = abs(tangent_angle_counterclockwise - angle)
    if diff_clockwise < diff_counterclockwise:
        angle1 = tangent_angle_clockwise
    else:
        angle1 = tangent_angle_counterclockwise

    distance, sign = point_to_line(angle1, nearest_point[0], nearest_point[1])
    dis = (distance * sign) / 64.

    return angle1, dis


def obtain_edge3(img):
    center = np.array([64, 64])
    ret, binary_image = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    largest_contour = max(contours, key=cv2.contourArea)
    edge_points = largest_contour.reshape(-1, 2)
    distances = np.linalg.norm(edge_points - center, axis=1)
    nearest_point = edge_points[np.argmin(distances)]
    if nearest_point[0] < 8 or nearest_point[0] > 120 or nearest_point[1] < 8 or nearest_point[1] > 120:
        return None
    normal_vector = nearest_point - center  # 法线
    normal_angle = np.arctan2(normal_vector[1], normal_vector[0])
    normal_angle = np.degrees(normal_angle)

    tangent_angle_clockwise = normal_angle + 90
    tangent_angle_counterclockwise = normal_angle - 90
    if abs(tangent_angle_clockwise) < 90:
        angle = tangent_angle_clockwise
    elif abs(tangent_angle_counterclockwise) < 90:
        angle = tangent_angle_counterclockwise
    else:
        angle = -90
    angle  ###
    if angle != 90:
        if nearest_point[1] > 64:
            dis_sign = 1
        elif nearest_point[1] < 64:
            dis_sign = -1
        else:
            dis_sign = 0
    else:
        if nearest_point[0] > 64:
            dis_sign = 1
        elif nearest_point[0] < 64:
            dis_sign = -1
        else:
            dis_sign = 0
    distance = np.linalg.norm(nearest_point - center)
    distance_with_sign = distance * dis_sign
    distance_with_sign  ###
    # return angle, distance_with_sign # （-90, 90 ）
    return angle / 90., nearest_point[0] / 128., nearest_point[1] / 128.


def obtain_edge4(img, angle):
    """ return -180~180 angle """
    center = np.array([64, 64])
    ret, binary_image = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    largest_contour = max(contours, key=cv2.contourArea)
    edge_points = largest_contour.reshape(-1, 2)
    distances = np.linalg.norm(edge_points - center, axis=1)
    nearest_point = edge_points[np.argmin(distances)]
    if nearest_point[0] < 8 or nearest_point[0] > 120 or nearest_point[1] < 8 or nearest_point[1] > 120:
        return None
    normal_vector = nearest_point - center
    normal_angle = np.arctan2(normal_vector[1], normal_vector[0])
    tangent_angle_clockwise = normal_angle + np.pi / 2
    tangent_angle_counterclockwise = normal_angle - np.pi / 2

    if tangent_angle_clockwise > np.pi:
        tangent_angle_clockwise = tangent_angle_clockwise % (-np.pi)
    if tangent_angle_counterclockwise < -np.pi:
        tangent_angle_counterclockwise = tangent_angle_counterclockwise % (np.pi)
    tangent_angle_clockwise = np.degrees(tangent_angle_clockwise)
    tangent_angle_counterclockwise = np.degrees(tangent_angle_counterclockwise)
    diff_clockwise = abs(tangent_angle_clockwise - angle)
    diff_counterclockwise = abs(tangent_angle_counterclockwise - angle)
    if diff_clockwise < diff_counterclockwise:
        angle1 = tangent_angle_clockwise
    else:
        angle1 = tangent_angle_counterclockwise

    # cv2.circle(binary_image, (64, 64), 3, (125), -1)  #
    # cv2.circle(binary_image, nearest_point, 3, (125), -1)  #
    # plt.figure()
    # plt.imshow(binary_image)
    # plt.show()

    angle1 = angle1 / 180
    nearest_point = nearest_point / 128.

    return angle1, nearest_point[0], nearest_point[1]


def obtain_distance_sign(angle, near_point):
    x_near, y_near = near_point
    if np.isclose(angle, 90) or np.isclose(angle, -90):
        if x_near > 64:
            return 1
        elif x_near < 64:
            return -1
        else:
            return 0
    k = np.tan(np.radians(angle))
    y_center_pred = y_near + k * (64 - x_near)
    if y_center_pred > 64:
        return -1
    elif y_center_pred < 64:
        return 1
    else:
        return 0


def obtain_edge3_1(img, show=False, episode=0):
    """ return angle (-1,1) """
    center = np.array([64, 64])
    ret, binary_image = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    largest_contour = max(contours, key=cv2.contourArea)
    edge_points = largest_contour.reshape(-1, 2)
    distances = np.linalg.norm(edge_points - center, axis=1)
    nearest_point = edge_points[np.argmin(distances)]
    if nearest_point[0] < 8 or nearest_point[0] > 120 or nearest_point[1] < 8 or nearest_point[1] > 120:
        return None
    normal_vector = nearest_point - center
    normal_angle = np.arctan2(normal_vector[1], normal_vector[0])
    normal_angle = np.degrees(normal_angle)

    tangent_angle_clockwise = normal_angle + 90
    tangent_angle_counterclockwise = normal_angle - 90
    if abs(tangent_angle_clockwise) < 90:
        angle = tangent_angle_clockwise
    elif abs(tangent_angle_counterclockwise) < 90:
        angle = tangent_angle_counterclockwise
    else:
        angle = -90

    distance = np.linalg.norm(nearest_point - center)

    distance_sign = obtain_distance_sign(angle, nearest_point)
    distance_with_sign = distance * distance_sign

    if show:
        print(angle, distance_with_sign)
        cv2.circle(binary_image, (64, 64), 3, (125), -1)  #
        cv2.circle(binary_image, nearest_point, 3, (125), -1)  #
        cv2.putText(binary_image, f"Distance: {distance_with_sign:.1f}",
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200), 1, cv2.LINE_AA)
        plt.figure()
        plt.imshow(binary_image)
        plt.savefig(str(episode) + '.jpg')
        # plt.show()

    # return angle, distance_with_sign # （-90, 90 ）
    return angle, distance_with_sign


class Actor(nn.Module):

    def __init__(self, state_dim, action_dim, action_bound):
        super(Actor, self).__init__()
        self.action_bound = torch.FloatTensor(action_bound)
        self.layer_1 = nn.Linear(state_dim, 30)
        nn.init.normal_(self.layer_1.weight, 0., 0.3)
        nn.init.constant_(self.layer_1.bias, 0.1)
        self.output = nn.Linear(30, action_dim)
        self.output.weight.data.normal_(0., 0.3)
        self.output.bias.data.fill_(0.1)

    def forward(self, s):
        a = torch.relu(self.layer_1(s))
        a = torch.tanh(self.output(a))
        return a


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        n_layer = 30
        self.layer_1 = nn.Linear(state_dim, n_layer)
        nn.init.normal_(self.layer_1.weight, 0., 0.1)
        nn.init.constant_(self.layer_1.bias, 0.1)
        self.layer_2 = nn.Linear(action_dim, n_layer)
        nn.init.normal_(self.layer_2.weight, 0., 0.1)
        nn.init.constant_(self.layer_2.bias, 0.1)
        self.output = nn.Linear(n_layer, 1)

    def forward(self, s, a):
        s = self.layer_1(s)
        a = self.layer_2(a)
        q_val = self.output(torch.relu(s + a))
        return q_val


class DDPG(object):
    def __init__(self, state_dim, action_dim, action_bound, replacement, memory_capacity=1000,
                 gamma=0.9, lr_a=0.001, lr_c=0.002, batch_size=32):
        super(DDPG, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.memory_capacity = memory_capacity
        self.replacement = replacement
        self.t_replace_counter = 0
        self.gamma = gamma
        self.lr_a = lr_a
        self.lr_c = lr_c
        self.batch_size = batch_size
        self.pointer = 0

        self.actor = Actor(state_dim, action_dim, action_bound)
        self.critic = Critic(state_dim, action_dim)
        self.actor_target = Actor(state_dim, action_dim, action_bound)
        self.critic_target = Critic(state_dim, action_dim)
        self.aopt = torch.optim.Adam(self.actor.parameters(), lr=lr_a)
        self.copt = torch.optim.Adam(self.critic.parameters(), lr=lr_c)
        self.mse_loss = nn.MSELoss()
        self.memory = np.zeros((memory_capacity, state_dim * 2 + action_dim + 1))

    def sampel(self):
        indices = np.random.choice(self.memory_capacity, size=self.batch_size)
        return self.memory[indices]

    def choose_action(self, s):
        s = torch.FloatTensor(s)
        action = self.actor(s)
        return action.detach().numpy()

    def learn(self):
        if self.replacement['name'] == 'soft':
            tau = self.replacement['tau']
            a_layers = self.actor_target.named_children()
            c_layers = self.critic_target.named_children()
            for al in a_layers:
                a = self.actor.state_dict()[al[0] + '.weight']
                al[1].weight.data.mul_((1 - tau))
                al[1].weight.data.add_(tau * self.actor.state_dict()[al[0] + '.weight'])
                al[1].bias.data.mul_((1 - tau))
                al[1].bias.data.add_(tau * self.actor.state_dict()[al[0] + '.bias'])
            for cl in c_layers:
                cl[1].weight.data.mul_((1 - tau))
                cl[1].weight.data.add_(tau * self.critic.state_dict()[cl[0] + '.weight'])
                cl[1].bias.data.mul_((1 - tau))
                cl[1].bias.data.add_(tau * self.critic.state_dict()[cl[0] + '.bias'])
        else:
            if self.t_replace_counter % self.replacement['rep_iter'] == 0:
                self.t_replace_counter = 0
                a_layers = self.actor_target.named_children()
                c_layers = self.critic_target.named_children()
                for al in a_layers:
                    al[1].weight.data = self.actor.state_dict()[al[0] + '.weight']
                    al[1].bias.data = self.actor.state_dict()[al[0] + '.bias']
                for cl in c_layers:
                    cl[1].weight.data = self.critic.state_dict()[cl[0] + '.weight']
                    cl[1].bias.data = self.critic.state_dict()[cl[0] + '.bias']
            self.t_replace_counter += 1

        bm = self.sampel()
        bs = torch.FloatTensor(bm[:, :self.state_dim])
        ba = torch.FloatTensor(bm[:, self.state_dim:self.state_dim + self.action_dim])
        br = torch.FloatTensor(bm[:, -self.state_dim - 1: -self.state_dim])
        bs_ = torch.FloatTensor(bm[:, -self.state_dim:])

        # 训练actor
        a = self.actor(bs)
        q = self.critic(bs, a)
        a_loss = -torch.mean(q)
        self.aopt.zero_grad()
        a_loss.backward(retain_graph=True)
        self.aopt.step()


        a_ = self.actor_target(bs_)
        q_ = self.critic_target(bs_, a_)
        q_target = br + self.gamma * q_
        q_eval = self.critic(bs, ba)
        td_error = self.mse_loss(q_target, q_eval)
        self.copt.zero_grad()
        td_error.backward()
        self.copt.step()

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % self.memory_capacity
        self.memory[index, :] = transition
        self.pointer += 1

    def save(self, info='best_reward_'):
        path = 'save_models/DDPG_State90_A1_1/'
        if not os.path.exists(path):
            print('create path')
            os.makedirs(path)
        path = path + info
        torch.save(self.actor.state_dict(), path + 'actor.pth')
        torch.save(self.critic.state_dict(), path + 'critic.pth')
        print("**** Save model ")

    def load(self, model_path,info='best_reward_'):
        path = model_path + info
        try:
            self.actor.load_state_dict(torch.load(path + 'actor.pth'))
            self.critic.load_state_dict(torch.load(path + 'critic.pth'))
            print('...load...')
        except  ValueError:
            print("Load model error ... ")



  
def main():
    a_dim = 1
    max_steps = 300
    episode = 0
    VAR = 5 
    max_rate = 0
    MEMORY_CAPACITY = 1000
    REPLACEMENT = [ dict(name='soft', tau=0.005), dict(name='hard', rep_iter=600) ][0] 

    env = EdgeFollowEnv()  
    env = env.unwrapped
    env.seed(1)
    traje_reward = []

    ddpg = DDPG(state_dim=2, action_dim=a_dim, action_bound=1, replacement=REPLACEMENT, memory_capacity=MEMORY_CAPACITY)
    
    for ep in range(80):
        episode += 1
        # print(episode, VAR)
        print(episode)
        obs = env.reset()
        theta = - np.degrees( obs["oracle"][9] ) + 90 
        if theta > 180:
            theta = theta % (-360) # reset angle of object 
        if theta == 90 or theta == -90:
            continue
        offset = random.uniform(0.3, 0.4)    # [0.0001, - 1]  
        action = [ offset * np.sin(obs["oracle"][9]) ,  offset * np.cos(obs["oracle"][9]), 0  ]
        obs, _, _, _ = env.step(action)   
        while True: 
            if obs['oracle'][2] < 0.003:
                action = [0, 0, 0.001]
                obs, _, _, _ = env.step(action)   
            else:
                break          
        s = obtain_state(np.rot90(obs["tactile"], k=-1)) 
        if s is None:
            continue
        if  theta < -90 or theta > 90:
            s = (s[0], -s[1])
            # s[1] *= -1  
            
        episode_reward = [] 
        ep_reward = 0
        for j in range(max_steps):
            a = ddpg.choose_action(s)
            a = np.clip(np.random.normal(a, VAR), -1, 1)
            if a_dim == 2:
                a = a * 0.01
                x, y =a[0], a[1]  
            if a_dim == 1:
                a = a * np.pi* 0.5 
                x, y = 0.01 * np.cos(a), 0.01 * np.sin(a)  
            if theta < -90 or theta > 90:
                x = -x 
                y = -y
            # if theta > -90 and theta < 90:
            #     x = x 
            #     y = -y
            obs_, r, done, info = env.step([x, y, 0])
            while True: 
                if obs_['oracle'][2] < 0.003:
                    action_z = [0, 0, 0.001]
                    obs_, r, done, info = env.step(action_z)    
                else:
                    break    
            s_ = obtain_state(np.rot90(obs_["tactile"], k=-1)) 
            
            distance1 =  ((obs['oracle'][0] -  obs['oracle'][6])**2 + (obs['oracle'][1] -  obs['oracle'][7])**2 )**(0.5)
            distance2 =  ((obs_['oracle'][0] -  obs_['oracle'][6])**2 + (obs_['oracle'][1] -  obs_['oracle'][7])**2 )**(0.5)
            R1 = (distance1 - distance2)*1000
            if done:
                print("success ")
                R3 = 10.
            else:
                R3 = 0.
            if s_ is None:
                done = True
                R2 = -10
                R = R1 + R2 + R3 
            else:
                if  theta < -90 or theta > 90:
                    s_ = (s_[0], -s_[1])
                R2 = - abs(s_[1])/64. # np.linalg.norm(np.array([s_[1]-0.5,s_[2]-0.5]))
                R = R1 + R2 + R3 
                ddpg.store_transition(s,a,R,s_)
            episode_reward.append(R)    
            if ddpg.pointer > MEMORY_CAPACITY:
                VAR *=0.999
                ddpg.learn()
            ep_reward += R
            s = s_ 
            obs = obs_  
            
            if done:
                episode_avg_reward = np.mean(np.array(episode_reward))
                traje_reward.append(episode_avg_reward)
                break
            
        
        ## validation  
        if episode >30 and episode % 10 ==0:
            print(" ===== Validation  =====")
            valid_flag = 0
            success_flag = 0
            for epi in range(20): # test 20 episodes (max)
                obs = env.reset()    
                theta = - np.degrees( obs["oracle"][9] ) + 90 
                if theta > 180:
                    theta = theta % (-360) # reset angle of object 
                if theta == 90 or theta == -90:
                    continue
                offset = random.uniform(0.3, 0.4)    # [0.0001, - 1]  
                action = [ offset * np.sin(obs["oracle"][9]) ,  offset * np.cos(obs["oracle"][9]), 0  ]
                obs, _, _, _ = env.step(action)   
                while True: 
                    if obs['oracle'][2] < 0.005:
                        action = [0, 0, 0.001]
                        obs, _, _, _ = env.step(action)   
                    else:
                        break          
                s = obtain_state(np.rot90(obs["tactile"], k=-1)) 
                if s is None:
                    continue
                if  theta < -90 or theta > 90:
                    # s[1] *= -1  
                    s = (s[0], -s[1])
                valid_flag += 1
                
                for i in range(max_steps): # max_len = 200 
                    a = ddpg.choose_action(s) # [-1, 1]
                    if a_dim == 2:
                        a = a * 0.01
                        x, y =a[0], a[1]  
                    if a_dim == 1:
                        a = a * np.pi* 0.5 
                        x, y = 0.01 * np.cos(a), 0.01 * np.sin(a)  
                    if theta < -90 or theta > 90:
                        x = -x 
                        y = -y
                    # if theta > -90 and theta < 90:
                    #     x = x 
                    #     y = -y
                    obs_, r, done, info = env.step([x, y, 0])
                    while True: 
                        if obs_['oracle'][2] < 0.003:
                            action_z = [0, 0, 0.001]
                            obs_, r, done, info = env.step(action_z)    
                        else:
                            break    
                    s_ = obtain_state(np.rot90(obs_["tactile"], k=-1)) 
                    
                    if done:
                        success_flag += 1 
                        break
                    if s_ is None:
                        break
                    else:
                        if  theta < -90 or theta > 90:
                            # s_[1] *= -1  
                            s_ = (s_[0], -s_[1])
                    s = s_
                    obs = obs_                  
            
            success_rate = success_flag / valid_flag
            print(' success_R', success_rate, '; ', success_flag , '/', valid_flag)
            if success_rate > max_rate:
                max_rate = success_rate
                ddpg.save()
            print(" ===== Validation Finish...=====")
         
    plot_reward(traje_reward)       

def test():
    valid_flag = 0
    a_dim = 1
    success_flag = 0
    episode = 0
    MEMORY_CAPACITY = 1000
    REPLACEMENT = [
        dict(name='soft', tau=0.005),
        dict(name='hard', rep_iter=600)
    ][0]  # you can try different target replacement strategies

    env = EdgeFollowEnv()  
    env = env.unwrapped
    env.seed(1)
    ddpg = DDPG(state_dim=2,
                action_dim=1,
                action_bound=1,
                replacement=REPLACEMENT,
                memory_capacity=MEMORY_CAPACITY)
    ddpg.load()
     
    for eps in range(100):
        episode += 1
        print(episode)
        obs = env.reset()
        theta = - np.degrees( obs["oracle"][9] ) + 90 
        if theta > 180:
            theta = theta % (-360) # reset angle of object 
        if theta == 90 or theta == -90:
            continue
        offset = random.uniform(0.3, 0.4)    # [0.0001, - 1]  
        action = [ offset * np.sin(obs["oracle"][9]) ,  offset * np.cos(obs["oracle"][9]), 0  ]
        obs, _, _, _ = env.step(action)   
        while True: 
            if obs['oracle'][2] < 0.003:
                action = [0, 0, 0.001]
                obs, _, _, _ = env.step(action)   
            else:
                break          
        s = obtain_state(np.rot90(obs["tactile"], k=-1)) 
        cv2.imwrite('test.png', np.rot90(obs["tactile"]))
        if s is None:
            continue
        valid_flag += 1
        if  theta < -90 or theta > 90:
            # s_[1] *= -1  
            s = (s[0], -s[1])
        for i in range(300): # max_len = 200 
            a = ddpg.choose_action(s) # [-1, 1]
            a = np.clip(a, -1, 1)
            if a_dim == 2:
                a = a * 0.01
                x, y =a[0], a[1]  
            if a_dim == 1:
                a = a * np.pi* 0.5 
                x, y = 0.01 * np.cos(a), 0.01 * np.sin(a)  
            if theta < -90 or theta > 90:
                x = -x 
                y = -y
            # if theta > -90 and theta < 90:
            #     x = x 
            #     y = -y
            obs_, r, done, info = env.step([x, y, 0])
            while True: 
                if obs_['oracle'][2] < 0.003:
                    action_z = [0, 0, 0.001]
                    obs_, r, done, info = env.step(action_z)    
                else:
                    break    
            s_ = obtain_state(np.rot90(obs_["tactile"], k=-1)) 
            # cv2.imwrite('test.png', np.rot90(obs["tactile"]))
            if done:
                success_flag += 1 
                print("Done", success_flag , '/', valid_flag)
                break
            if s_ is None:
                break
            else:
                if  theta < -90 or theta > 90:
                    # s_[1] *= -1  
                    s_ = (s_[0], -s_[1])
            s = s_
            obs = obs_                  
    
    success_rate = success_flag / valid_flag
    print(' success_R', success_rate, '; ', success_flag , '/', valid_flag)
    print(" ===== Test Finish...=====")
        


        
if __name__ == '__main__':
    
    main()
    
    # test()

