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



parser = argparse.ArgumentParser(description="Train an agent in a tactile gym task.")
parser.add_argument("-M", '--movement_mode', type=str, help='The movement mode.', metavar='')
parser.add_argument("-T", '--traj_type', type=str, help='The traj type.', metavar='')
parser.add_argument("-R", '--retrain_path', type=str, help='Retrain model path.', metavar='')
parser.add_argument("-I", '--if_retrain', type=str, help='Retrain.', metavar='')
args = parser.parse_args()





def fix_floats(data):
    if isinstance(data, list):
        iterator = enumerate(data)
    elif isinstance(data, dict):
        iterator = data.items()
    else:
        raise TypeError("can only traverse list or dict")

    for i, value in iterator:
        if isinstance(value, (list, dict)):
            fix_floats(value)
        elif isinstance(value, str):
            try:
                data[i] = float(value)
            except ValueError:
                pass


def train_agent(algo_name="ppo",env_name="edge_follow-v0", rl_params={}, algo_params={}, augmentations=None,):
    # create save dir
    timestr = time.strftime("%Y%m%d-%H%M%S")
    save_dir = os.path.join( "saved_models/", rl_params["env_name"], timestr, algo_name, "s{}_{}".format( rl_params["seed"], rl_params["env_modes"]["observation_mode"]) )
    check_dir(save_dir)
    os.makedirs(save_dir, exist_ok=True)

    # save params
    save_json_obj(convert_json(rl_params), os.path.join(save_dir, "rl_params"))
    save_json_obj(convert_json(algo_params), os.path.join(save_dir, "algo_params"))
    if "rad" in algo_name:
        save_json_obj(convert_json(augmentations), os.path.join(save_dir, "augmentations"))


    ## Env Test
    env = EdgeFollowEnv()
    
    for j in range(100):
        obs = env.reset()
        state = obtain_edge(np.rot90(obs["tactile"], k=-1))
        for i in range(10000000):
            action = env.action_space.sample()  # 从动作空间中随机选取一个动作  (2,)
            print("action: ", action)
            obs1, _, _,_ = env.step(action)
            state1 = obtain_edge(np.rot90(obs1["tactile"], k=-1))
            print(state1)
            if state1  is None:
                break
            obs1
        
    



if __name__ == "__main__":

    algo_name = 'ppo'
    env_name = "edge_follow-v0"

    rl_params, algo_params, augmentations = import_parameters(env_name, algo_name)
    
    train_agent(  algo_name,env_name, rl_params, algo_params,  augmentations )
