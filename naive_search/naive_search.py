import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from Networks import Representation_Model, Dynamics_Model, Prediction_Model

device = torch.device("cuda:0")
dtype = torch.float

def softmax(x):
  e_x = np.exp(x - np.max(x))
  return e_x / e_x.sum()
  
# ALTERNATIVE FOR MCTS
def naive_search(agent, state, num_actions, n=3):
# search the max value prediction of fully expanded tree at depth n | combine with eps greedy policy at agent
# return the first action from the root to the max value node
# dont consider rewards (probably only working for cartpole)
    
    _, target_v = agent.prediction_model(state)
    
    possible_actions = np.array(list(range(num_actions)))
    
    # will be repeated
    actions = np.repeat([possible_actions], 1, axis=0).flatten()
    
    v = None
    for depth in range(n):
    
        state = torch.repeat_interleave(state, num_actions, dim=0)
        actions = np.repeat([possible_actions], (num_actions ** depth), axis=0).flatten()
        
        state, _, v, _ = agent.rollout_step(state, actions)
        state, v = state.detach(), v.detach()
        
    #print(v.cpu().numpy())
    
    ##
    ## 1 - max selection
    ## kind of works with some epsilon greedy approach
    
    max_index = np.argmax(v.cpu().numpy()) 
    indexes_per_action = num_actions ** (n-1)
    action = int(max_index/indexes_per_action)
    return action, target_v
    
    ##
    ## 2 - policy softmax over all calculated values
    ## maybe try again :), i changed some things
    
    #indexes_per_action = num_actions ** (n-1)
    #minimum = min(v)
    #maximum = max(v)
    #v = (v - minimum) / (maximum - minimum)
    
    #policy = np.zeros(num_actions)
    #for i in range(num_actions):
        #for value in v[i*indexes_per_action:i*indexes_per_action+indexes_per_action]:
            #policy[i] += value
    
    #policy = policy / n
    #policy = softmax(policy)
    #return policy
    

    
    
    
    