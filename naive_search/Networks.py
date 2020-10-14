import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class Representation_Model(nn.Module):
    
    def __init__(self, num_in, num_hidden):
        super().__init__()
        
        self.num_in = num_in
        self.num_hidden = num_hidden
        
        network = [
            nn.Linear(num_in, 50),
            nn.ReLU(),
            nn.Linear(50, 50),
            nn.ReLU(),
            nn.Linear(50, 50),
            nn.ReLU(),
            nn.Linear(50, 50),
            nn.ReLU(),
            nn.Linear(50, num_hidden)
        ]
        
        self.network = nn.Sequential(*network)
    
    def forward(self, x):
        return self.network(x)

class Dynamics_Model(nn.Module):
    # action encoding - one hot
    
    def __init__(self, num_hidden, num_actions): 
        super().__init__()
        
        self.num_hidden = num_hidden
        self.num_actions = num_actions
       
        network = [
            nn.Linear(num_hidden + 1, 50), # hidden, action encoding
            nn.ReLU(),
            nn.Linear(50, 50),
            nn.ReLU(),
            nn.Linear(50, 50),
            nn.ReLU(),
            nn.Linear(50, 50),
            nn.ReLU(),
            nn.Linear(50, num_hidden + 1) # add reward prediction
        ]
        
        self.network = nn.Sequential(*network)
    
    def forward(self, x):
        out = self.network(x)
        hidden, reward = out[:, 0:self.num_hidden], out[:, -1]
        
        return hidden, reward

class Prediction_Model(nn.Module):
    
    def __init__(self, num_hidden, num_actions):
        super().__init__()
        
        self.num_actions = num_actions
        self.num_hidden = num_hidden
        
        network = [
            nn.Linear(num_hidden, 50),
            nn.ReLU(),
            nn.Linear(50, 50),
            nn.ReLU(),
            nn.Linear(50, 50),
            nn.ReLU(),
            nn.Linear(50, num_actions + 1) # value & policy prediction
        ]
        
        self.network = nn.Sequential(*network)
    
    def forward(self, x):
        out = self.network(x)
        p = out[:, 0:self.num_actions]
        v = out[:, -1]
        
        # softmax probs
        p = F.softmax(p, dim=1)
        
        return p, v         