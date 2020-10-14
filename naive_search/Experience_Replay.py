import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

device = torch.device("cuda:0")
dtype = torch.float

class Experience_Replay(): 
# save environment trajectories and sample sub trajectories

        
    def __init__(self, trajectory_capacity, num_actions):
        super().__init__()
        
        self.trajectory_capacity = trajectory_capacity
        self.memory = []
        self.position = 0
        self.num_actions = num_actions

    def insert(self, trajectories):
        
        for i in range(len(trajectories)):
            if len(self.memory) < self.trajectory_capacity:
                self.memory.append(None)
            self.memory[self.position] = trajectories[i]
            self.position = (self.position + 1) % self.trajectory_capacity

            
    def get_sample(self, k, n, gamma):
    # k = unroll | n = n-step-return | gamma = discount
    
        sample = {}
        sample["obs"], sample["actions"], sample["rewards"], sample["return"] = [],[],[],[]
        
        # select trajectory
        memory_index = np.random.choice(len(self.memory),1)[0]
        traj_length = self.memory[memory_index]["length"]
        traj_last_index = traj_length - 1
        
        # select start index to unroll
        start_index = np.random.choice(traj_length, 1)[0] 
             
        # fill in the data
        sample["obs"] = self.memory[memory_index]["obs"][start_index]
        
        # compute n-step return for every unroll step, rewards and pi
        for step in range(start_index, start_index + k + 1):
        
            n_index = step + n
            
            v_n = None
            if n_index >= traj_last_index: # end of episode
                v_n = torch.tensor([0]).to(device).to(dtype)
            else:
                v_n = torch.tensor([0]).to(device).to(dtype)#v_n = self.memory[memory_index]["vs"][n_index] * (gamma ** n) # discount v_n
            
            value = v_n
            # add discounted rewards until step n or end of episode
            last_valid_index = np.minimum(traj_last_index, n_index)
            #for i, reward in enumerate(self.memory[memory_index]["rewards"][step:last_valid_index]):
            for i, reward in enumerate(self.memory[memory_index]["rewards"][step::]): # rewards until end of episode
                value += reward * (gamma ** i)
                
            sample["return"].append(value)
            
            # add reward
            # only add when not inital step | dont need reward for step 0
            if step != start_index:
                if step > 0  and step <= traj_last_index:
                    sample["rewards"].append(self.memory[memory_index]["rewards"][step-1])
                else:
                    sample["rewards"].append(torch.tensor([0.0]).to(device))

        # unroll steps beyond trajectory then fill in the remaining (random) actions
        
        last_valid_index = np.minimum(traj_last_index - 1, start_index + k - 1)
        num_steps = last_valid_index - start_index
        
        # real
        sample["actions"] = self.memory[memory_index]["actions"][start_index:start_index+num_steps+1]
       
        # fills
        num_fills = k - num_steps + 1 
        for i in range(num_fills):
            sample["actions"].append(np.random.choice(self.num_actions,1)[0])
        
        return sample
        
    def get(self, batch_size, k, n, gamma=0.95):
        
        data = []
        
        for i in range(batch_size):
            sample = self.get_sample(k, n, gamma)
            data.append(sample)
            
        return data

    def __len__(self):
        return len(self.memory)