import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda:0")
dtype = torch.float

class Logger:
    
    def __init__(self, filename):
        self.filename = filename
        
        f = open(f"{self.filename}.csv", "w")
        f.close()
        
    def log(self, msg):
        f = open(f"{self.filename}.csv", "a+")
        f.write(f"{msg}\n")
        f.close()
        
class Env_Runner:
    
    def __init__(self, env):
        super().__init__()
        
        self.env = env
        self.num_actions = self.env.action_space.n
        
        self.logger = Logger("episode_returns")
        self.logger.log("training_step, return")
        
        self.ob = self.env.reset()
        self.total_eps = 0
        
    def run(self, agent):
        
        self.obs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.vs = []
        
        self.ob = self.env.reset()
        self.obs.append(torch.tensor(self.ob))
        
        done = False
        while not done:
            
            action, v = agent.naive_search_inference(torch.tensor(self.ob).to(device).to(dtype))
            
            self.ob, r, done, info = self.env.step(action)
            
            self.obs.append(torch.tensor(self.ob))
            self.actions.append(action)
            self.rewards.append(torch.tensor(r))
            self.dones.append(done)
            self.vs.append(v)
            
            if done: # environment reset
                if "return" in info:
                    self.logger.log(f'{self.total_eps},{info["return"]}')
                    print("Return:",info["return"])
            
            #self.env.render()
        
        self.total_eps += 1
                                    
        return self.make_trajectory()
        
        
        
    def make_trajectory(self):
        traj = {}
        traj["obs"] = self.obs
        traj["actions"] = self.actions
        traj["rewards"] = self.rewards
        traj["dones"] = self.dones
        traj["vs"] = self.vs
        traj["length"] = len(self.obs)
        return traj
        
        
        