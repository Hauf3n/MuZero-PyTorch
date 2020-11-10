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

def train():
    
    history_length = 3
    num_hidden = 50
    replay_capacity = 30
    batch_size = 32
    k = 3
    n = 3
    lr = 5e-3
    
    start_eps = 1
    final_eps = 0.1
    final_episode = 400
    eps_interval = start_eps-final_eps
    
    raw_env = gym.make('CartPole-v0')
    num_obs_space = raw_env.observation_space.shape[0]
    num_actions = raw_env.action_space.n
    num_in = history_length * num_obs_space # history * ( obs )
    
    env = Env_Wrapper(raw_env, history_length)
    
    representation_model = Representation_Model(num_in, num_hidden).to(device)
    dynamics_model = Dynamics_Model(num_hidden, num_actions).to(device)
    prediction_model = Prediction_Model(num_hidden, num_actions).to(device)
    
    agent = MuZero_Agent(num_actions, representation_model, dynamics_model, prediction_model).to(device)

    runner = Env_Runner(env)
    replay = Experience_Replay(replay_capacity, num_actions)
    
    mse_loss = nn.MSELoss()
    
    optimizer = optim.Adam(agent.parameters(), lr=lr)
 
    for episode in range(2000):#while True:
        
        agent.eps = np.maximum(final_eps, start_eps - ( eps_interval * episode/final_episode))
        
        # act and get data
        trajectory = runner.run(agent)
        
        # save new data
        replay.insert([trajectory])
        
        #############
        # do update #
        #############
        
        if len(replay) < 15:
            continue
            
        for update in range(8):
            optimizer.zero_grad()
            
            # get data
            data = replay.get(batch_size,k,n)
            
            # network unroll data
            representation_in = torch.stack([torch.flatten(data[i]["obs"]) for i in range(batch_size)]).to(device).to(dtype) # flatten when insert into mem
            actions = np.stack([np.array(data[i]["actions"], dtype=np.int64) for i in range(batch_size)])
            
            # targets
            value_target = torch.stack([torch.tensor(data[i]["return"]) for i in range(batch_size)]).to(device).to(dtype)
            rewards_target = torch.stack([torch.tensor(data[i]["rewards"]) for i in range(batch_size)]).to(device).to(dtype)
            
            # loss
            
            loss = torch.tensor(0).to(device).to(dtype)
            
            # agent inital step
            
            state, p, v = agent.inital_step(representation_in)
            
            value_loss = mse_loss(v, value_target[:,0].detach())
            loss += value_loss

            # steps
            for step in range(1, k+1):
            
                # step
                step_action = actions[:,step - 1]
                state, p, v, rewards = agent.rollout_step(state, step_action)
                
                value_loss = mse_loss(v, value_target[:,step].detach())
                reward_loss = mse_loss(rewards, rewards_target[:,step-1].detach())
                
                #print(f'value: {value_loss} || reward: {reward_loss}')
                loss += (value_loss + reward_loss)/k+1
                
            loss.backward()
            optimizer.step() 

if __name__ == "__main__":

    train()