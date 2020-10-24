import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from Networks import Representation_Model, Dynamics_Model, Prediction_Model
from MCTS import MCTS
from naive_search import naive_search

device = torch.device("cuda:0")
dtype = torch.float

class MuZero_Agent(nn.Module):
    
    def __init__(self, num_simulations, num_actions, representation_model, dynamics_model, prediction_model):
        super().__init__()
        
        self.representation_model = representation_model
        self.dynamics_model = dynamics_model
        self.prediction_model = prediction_model
        
        self.num_actions = num_actions
        self.num_simulations = num_simulations
        
        self.mcts = MCTS(num_actions, dynamics_model, prediction_model, self)
        
    def forward(self, obs):
        pass
        
    def mcts_inference(self, obs): # inference with MCTS
    
        start_state = self.representation_model(obs)
        pi, v = self.mcts.run(self.num_simulations, start_state)
        
        action = np.random.choice(self.num_actions, 1, p=pi)
        #action = np.random.choice(self.num_actions, 1)
        #return 0, pi, v
        
        return action[0], pi, v
  
    def inital_step(self, obs):
    # first step of rollout for optimization
    
        state = self.representation_model(obs)
        p, v = self.prediction_model(state)
        
        return state, p, v
        
        
    def rollout_step(self, state, action): 
    # unroll a step
    
        batch_size = state.shape[0]
        
        action_encoding = torch.tensor(action).to(device).to(dtype).reshape(batch_size,1) / self.num_actions
        in_dynamics = torch.cat([state,action_encoding],dim=1)
        
        next_state, reward = self.dynamics_model(in_dynamics)
        p, v = self.prediction_model(next_state)

        return next_state, p, v, reward
    