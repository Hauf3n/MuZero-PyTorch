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
    num_simulations = 16
    replay_capacity = 100
    batch_size = 32
    k = 5
    n = 5
    lr = 1e-3
    
    #target_update = 15
    
    #raw_env = gym.make('LunarLander-v2')
    raw_env = gym.make('CartPole-v0')
    num_obs_space = raw_env.observation_space.shape[0]
    num_actions = raw_env.action_space.n
    num_in = history_length * num_obs_space # history * (obs)
    
    env = Env_Wrapper(raw_env, history_length)
    
    representation_model = Representation_Model(num_in, num_hidden).to(device)
    dynamics_model = Dynamics_Model(num_hidden, num_actions).to(device)
    prediction_model = Prediction_Model(num_hidden, num_actions).to(device)
    
    agent = MuZero_Agent(num_simulations, num_actions, representation_model, dynamics_model, prediction_model).to(device)
    #target_agent = MuZero_Agent(num_simulations, num_actions, representation_model, dynamics_model, prediction_model).to(device)
    #target_agent.load_state_dict(agent.state_dict())

    runner = Env_Runner(env)
    replay = Experience_Replay(replay_capacity, num_actions)
    
    mse_loss = nn.MSELoss()
    cross_entropy_loss = nn.CrossEntropyLoss()
    
    optimizer = optim.Adam(agent.parameters(), lr=lr)
    
    for episode in range(2000):#while True:
        
        # act and get data
        trajectory = runner.run(agent)
        # save new data
        replay.insert([trajectory])
        
        #############
        # do update #
        #############
        
        #if episode%target_update == 0:
            #target_agent.load_state_dict(agent.state_dict()) 
        
        if len(replay) < 20:
            continue
            
        for i in range(6):
            optimizer.zero_grad()
            
            # get data
            data = replay.get(batch_size,k,n)
            
            # network unroll data
            representation_in = torch.stack([torch.flatten(data[i]["obs"]) for i in range(batch_size)]).to(device).to(dtype) # flatten when insert into mem
            actions = np.stack([np.array(data[i]["actions"], dtype=np.int64) for i in range(batch_size)])
            
            # targets
            rewards_target = torch.stack([torch.tensor(data[i]["rewards"]) for i in range(batch_size)]).to(device).to(dtype)
            policy_target = torch.stack([torch.stack(data[i]["pi"]) for i in range(batch_size)]).to(device).to(dtype)
            value_target = torch.stack([torch.tensor(data[i]["return"]) for i in range(batch_size)]).to(device).to(dtype)
            
            # loss
            #print("--------------------------------------")
            
            loss = torch.tensor(0).to(device).to(dtype)
            
            # agent inital step
            
            state, p, v = agent.inital_step(representation_in)
            
            policy_loss = mse_loss(p, policy_target[:,0].detach())
            value_loss = mse_loss(v, value_target[:,0].detach())
            loss += policy_loss + value_loss

            # steps
            for step in range(1, k+1):
            
                # step
                step_action = actions[:,step - 1]
                state, p, v, rewards = agent.rollout_step(state, step_action)
                
                policy_loss = mse_loss(p, policy_target[:,step].detach())
                value_loss = mse_loss(v, value_target[:,step].detach())
                reward_loss = mse_loss(rewards, rewards_target[:,step-1].detach())
                
                #print(f'policy: {policy_loss} || value: {value_loss} || reward: {reward_loss}')
                loss += (policy_loss + value_loss + reward_loss) / (k+1)
         
            loss.backward()
            optimizer.step() 
        
        # clear replay, no reanalyse
        #replay = Experience_Replay(replay_capacity, num_actions)

if __name__ == "__main__":

    train()