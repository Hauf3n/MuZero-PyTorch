import numpy as np
import os
import cv2
import gym

class Env_Wrapper(gym.Wrapper):
    # env wrapper for MuZero Cartpole, LunarLander
    
    def __init__(self, env, history_length):
        super(Env_Wrapper, self).__init__(env)
        
        self.history_length = history_length
        self.num_obs_space = env.observation_space.shape[0]
        self.num_actions = env.action_space.n
        
    def reset(self):
    
        self.Return = 0
        self.obs_history = []
        self.action_history = []
        
        obs = self.env.reset()
        self.obs_history.append(obs)
        
        return self.compute_observation()
        
        
    def compute_observation(self):
        
        features = np.zeros((self.history_length, self.num_obs_space))
        actions = np.zeros((self.history_length, 1)) #  encoding: action_index/num_actions
        
        # features 
        current_feature_len = len(self.obs_history)
        if current_feature_len == self.history_length:
            features = np.array(self.obs_history)
        else:
            features[self.history_length-current_feature_len::] = np.array(self.obs_history)
                
        #actions
        current_action_len = len(self.action_history)
        if current_action_len == self.history_length:
        
            actions = np.array(self.action_history)/self.num_actions
            
        else:
            if len(self.action_history) != 0:
                actions[self.history_length-current_action_len::] = np.array(self.action_history)/self.num_actions
        
        return np.concatenate((features, actions), axis=1).flatten().reshape(1,-1)
    
    
    def step(self, action): 
 
        obs, reward, done, info = self.env.step(action)
        
        # add obs and actions to history
        self.add_history(obs, action)
        
        obs = self.compute_observation()
        
        self.Return += reward
        if done:
            info["return"] = self.Return
        
        return obs, reward, done, info
        
        
    def add_history(self, obs, action):
    
        if len(self.obs_history) == self.history_length:
            self.obs_history = self.obs_history[1::]
            
        if len(self.action_history) == self.history_length:
            self.action_history = self.action_history[1::]
            
        self.obs_history.append(obs)
        self.action_history.append([action])