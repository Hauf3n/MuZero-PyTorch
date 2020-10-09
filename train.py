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
    
    history_length = 2
    num_hidden = 32
    num_simulations = 8
    replay_capacity = 100
    batch_size = 16
    k = 3
    n = 10
    lr = 1e-3
    
    raw_env = gym.make('CartPole-v0')
    #raw_env = gym.make('LunarLander-v2')
    num_obs_space = raw_env.observation_space.shape[0]
    num_actions = raw_env.action_space.n
    num_in = history_length * (num_obs_space + num_actions)
    
    env = Env_Wrapper(raw_env, history_length)
    
    representation_model = Representation_Model(num_in, num_hidden).to(device)
    dynamics_model = Dynamics_Model(num_hidden, num_actions).to(device)
    prediction_model = Prediction_Model(num_hidden, num_actions).to(device)
    
    agent = MuZero_Agent(num_simulations, num_actions, representation_model, dynamics_model, prediction_model).to(device)

    runner = Env_Runner(env, agent)
    replay = Experience_Replay(replay_capacity, num_actions)
    
    mse_loss = nn.MSELoss()
    cross_entropy_loss = nn.CrossEntropyLoss()
    
    optimizer = optim.Adam(agent.parameters(), lr=lr)
    
    for episode in range(1000):#while True:
        
        # act and get data
        trajectory = runner.run()
        # save new data
        replay.insert([trajectory])
        
        #############
        # do update #
        #############
        
        if len(replay) < 1:#if episode < 10:
            continue
        
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
        
        #print("representaion:",representation_in.shape)
        #print("action:",actions.shape)
        #print("rewards_t:",rewards_target.shape)
        #print("policy_t:",policy_target.shape)
        #print("value_t:",value_target.shape)
        
        # agent unroll
        loss = torch.tensor(0).to(device).to(dtype)
        
        state = agent.representation_model(representation_in)
        
        print("--------------------------------------")
        for step in range(k):
            #loss = torch.tensor(0).to(device).to(dtype)
            #print(step)
            # step
            step_action = actions[:,step]
            state, p, v, rewards = agent.rollout_step(state, step_action)
            
            # step loss
            print(policy_target[0,step])
            print(p[0])
            policy_loss = mse_loss(p, policy_target[:,step].detach())
            value_loss = mse_loss(v, value_target[:,step].detach())
            reward_loss = mse_loss(rewards, rewards_target[:,step].detach())
            
            print(f'policy: {policy_loss} || value: {value_loss} || reward: {reward_loss}')
            #print(f'target: {value_target[:,step][0]}, pred: {v[0]}')
            loss += policy_loss + value_loss + reward_loss
            
        loss.backward()
        optimizer.step()
        
        
        
        replay = Experience_Replay(replay_capacity, num_actions)

if __name__ == "__main__":

    train()