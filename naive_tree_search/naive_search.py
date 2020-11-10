import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from Networks import Representation_Model, Dynamics_Model, Prediction_Model

device = torch.device("cuda:0")
dtype = torch.float

# ALTERNATIVE FOR MCTS
def naive_search(agent, state, num_actions, gamma, n=3):
# search the max value prediction and add rewards along the way of fully expanded tree at depth n 
# return the first action from the root to the max value node (in combination with eps greedy policy)

    possible_actions = np.array(list(range(num_actions)))
    
    _, target_v = agent.prediction_model(state)
    
    v = None
    rewards = torch.tensor([0.0]).to(device)
    
    for depth in range(n):
    
        state = torch.repeat_interleave(state, num_actions, dim=0)
        actions = np.repeat([possible_actions], (num_actions ** depth), axis=0).flatten()
        
        state, _, v, reward = agent.rollout_step(state, actions)
        state, v, reward = state.detach(), v.detach(), reward.detach()
        
        # add reward with respect to tree path
        rewards = torch.repeat_interleave(rewards, num_actions, dim=0)
        
        #discount reward at depth
        reward = reward * (gamma ** depth)
        rewards = rewards + reward
    
    v = v.cpu().numpy()
    rewards = rewards.cpu().numpy()
    # discount value prediction
    v = v * (gamma ** n)
    # add rewards
    v = v + rewards
    
    # max selection
    
    max_index = np.argmax(v) 
    indexes_per_action = num_actions ** (n-1)
    action = int(max_index/indexes_per_action)
    
    return action, target_v   
    

    
    
    
    