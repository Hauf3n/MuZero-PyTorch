import argparse
import numpy as np
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time
import random
import os

from Env_Runner import Env_Runner
from Env_Wrapper import Env_Wrapper
from Agent import MuZero_Agent
from Networks import Representation_Model, Dynamics_Model, Prediction_Model
from Experience_Replay import Experience_Replay

device = torch.device("cuda:0")
dtype = torch.float

if __name__ == "__main__":
    
    filename = "model_cartpole_3_history_200_return.pt"
    agent = torch.load(filename).to(device)
    agent.eps = 0.1
    
    raw_env = gym.make('CartPole-v0')
    env = Env_Wrapper(raw_env, 3)
    runner = Env_Runner(env)
       
    for i in range(100):
        _ = runner.run(agent, render=True)