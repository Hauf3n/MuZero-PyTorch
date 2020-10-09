import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from Networks import Representation_Model, Dynamics_Model, Prediction_Model
from MCTS import MCTS

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
        
        self.mcts = MCTS(num_actions, dynamics_model, prediction_model)
        
    def forward(self, x): # inference with MCTS
    
        start_state = self.representation_model(x)
        pi, v = self.mcts.run(self.num_simulations, start_state)
        
        action = np.random.choice(self.num_actions, 1, p=pi)
        return action[0], pi, v
        
    def rollout_step(self, state, action): 
    # unroll a step for optimization data
    
        batch_size = state.shape[0]
        
        rewards = torch.zeros((batch_size, 1)).to(device).to(dtype)
        pis = torch.zeros((batch_size, 1, self.num_actions)).to(device).to(dtype)
        vs = torch.zeros((batch_size, 1)).to(device).to(dtype)
        
        onehot = F.one_hot(torch.tensor(action).to(device),num_classes=self.num_actions).to(dtype)
        
        in_dynamics = torch.cat([state,onehot],dim=1)
        
        next_state, reward = self.dynamics_model(in_dynamics)
        p, v = self.prediction_model(next_state)

        return next_state, p, v, reward
    