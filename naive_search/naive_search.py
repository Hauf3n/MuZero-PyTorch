import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from Networks import Representation_Model, Dynamics_Model, Prediction_Model

device = torch.device("cuda:0")
dtype = torch.float

# ALTERNATIVE FOR MCTS
def naive_search(agent, state, num_actions, n=2):
# search the max value prediction of fully expanded tree at depth n | combine with eps greedy policy at agent
# return the first action from the root to the max value node
# dont consider rewards (probably only working for cartpole)
    
    #print("1",state.shape)
    possible_actions = np.array(list(range(num_actions)))
    #state = torch.repeat_interleave(state, 2, dim=0)
    #print("2",state.shape)
    #print(state)
    #print(possible_actions)
    
    actions = np.repeat([possible_actions], 1, axis=0).flatten()
    #print("actions:",actions)
    
    v = None
    for depth in range(n):
    
        state = torch.repeat_interleave(state, num_actions, dim=0)
        actions = np.repeat([possible_actions], (num_actions ** depth), axis=0).flatten()
        
        #print(state)
        #print(actions.shape)
        
        state, _, v, _ = agent.rollout_step(state, actions)
        state, v = state.detach(), v.detach()

        
    max_index = np.argmax(v.cpu().numpy()) 
    indexes_per_action = num_actions ** depth 
    
    #print("ind",indexes_per_action)
    #print(v)
    #print("MAX VALUE INDEX:",max_index)
    #print("DONE")
    
    #amax = np.amax(v.cpu().numpy())
    #actions = []
    #for i in range(len(v)):
        #if amax == v[i]:
            #actions.append(i)
    #print("-----------")
    #print("COUNT:",len(actions))
    #print("INDEX:",actions)
    action = int(max_index/indexes_per_action)
    
    #print(v.cpu().numpy())
    #print("A:",action)
    return action
    
    
    
    