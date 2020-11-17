import numpy as np
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

device = torch.device("cuda:0")
dtype = torch.float

class MinMaxStats():
    
    def __init__(self):
        
        self.max = - np.inf
        self.min = np.inf
        
    def update(self, value):
        self.max = np.maximum(self.max, value.cpu())
        self.min = np.minimum(self.min, value.cpu())
        
    def normalize(self, value):
        value = value.cpu()
        if self.max > self.min:
            return ((value - self.min) / (self.max - self.min)).to(device)
        
        return value

class MCTS_Node():

    def __init__(self, p):
        super().__init__()
        
        self.state = None
        self.reward = None
        self.p = p
        
        self.edges = {}
        
        self.value_sum = 0
        self.visits = 0

    def expanded(self):
        return len(self.edges) > 0
        
    def search_value(self):
        if self.visits == 0:
            return 0
        return self.value_sum/self.visits

class MCTS():
    
    def __init__(self, num_actions, dynamics_model, prediction_model, agent, gamma=0.99):
        super().__init__()
        
        self.num_actions = num_actions
        self.c1 = 1.25
        self.c2 = 19652
        self.gamma = gamma
        self.root_dirichlet_alpha = 0.25
        self.root_exploration_fraction = 0.25
        self.temperature = 1
        
        self.dynamics_model = dynamics_model
        self.prediction_model = prediction_model
        self.agent = agent
        
    def run(self, num_simulations, root_state):
        
        # init root
        p, v = self.prediction_model(root_state)
        p, rv = p.detach(), v.detach()
        self.root = self.init_root(root_state, p)
        
        self.min_max_stats = MinMaxStats()
          
        # run simulations and save trajectory
        for i in range(num_simulations):
            
            self.node_trajectory = []
            self.node_trajectory.append(self.root)
            
            self.action_trajectory = []
            
            node = self.root
            while node.expanded():
                action, node = self.upper_confidence_bound(node)
                self.node_trajectory.append(node)
                self.action_trajectory.append(action)
                
            parent = self.node_trajectory[-2]
            v = self.expand(parent, node, self.action_trajectory[-1])
            
            self.backup(v)   
            
        return self.get_pi(), self.root.search_value()
     
    def expand(self, parent, node, action):
        
        next_state, p, v, reward = self.agent.rollout_step(parent.state, [action])
        next_state, p, v, reward = next_state.detach(), p.detach(), v.detach(), reward.detach()
        
        node.state = next_state
        node.reward = reward
        
        for i in range(self.num_actions):
            node.edges[i] = MCTS_Node(p[0,i])

        return v
    
    def backup(self, value):
   
        for node in reversed(self.node_trajectory):
        
            node.value_sum += value
            node.visits += 1
            
            self.min_max_stats.update( node.reward + self.gamma * node.search_value())
            
            value = node.reward + self.gamma * value         
            
    def upper_confidence_bound(self, node):
        
        ucb_scores = []
        
        for i in range(self.num_actions):
            ucb_scores.append(self.ucb_score(node,node.edges[i]))
            
        action = np.argmax(ucb_scores)    
        return action, node.edges[action] 
        
    def ucb_score(self, parent, child):
        
        pb_c = math.log((parent.visits + self.c2 + 1) / self.c2) + self.c1
        pb_c *= math.sqrt(parent.visits) / (child.visits + 1)
        
        prior_score = pb_c * child.p
        
        value_score = 0
        if child.visits > 0: # and parent.visits > 2:
            value_score = self.min_max_stats.normalize( child.reward + self.gamma * child.search_value())
            
        return prior_score + value_score

            
    def get_pi(self):
        
        edge_visits = []
        for i in range(self.num_actions):
            edge_visits.append(self.root.edges[i].visits)
        edge_visits = np.array(edge_visits)
        
        return edge_visits
        #pi = (edge_visits ** (1/self.temperature)) / np.sum( edge_visits ** (1/self.temperature))
        #pi = softmax(edge_visits)
        #return pi
    
    def add_exploration_noise(self, node):
        noise = np.random.dirichlet([self.root_dirichlet_alpha] * self.num_actions)
        frac = self.root_exploration_fraction
        for a, n in zip(range(self.num_actions), noise):
            node.edges[a].p = node.edges[a].p * (1 - frac) + n * frac
        return node
    
    def init_root(self, state, p):
        p = p.detach().cpu().numpy()
        
        node = MCTS_Node(0)
        node.state = state
        node.reward = 0
        
        for i in range(self.num_actions):
            node.edges[i] = MCTS_Node(p[0,i])
    
        node = self.add_exploration_noise(node)
        return node
        
    