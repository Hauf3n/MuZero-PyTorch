import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

device = torch.device("cuda:0")
dtype = torch.float


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
    
    def __init__(self, num_actions, dynamics_model, prediction_model, c1=1.25, c2=19652, gamma=0.95):
        super().__init__()
        
        self.num_actions = num_actions
        self.c1 = c1
        self.c2 = c2
        self.gamma = gamma
        
        self.dynamics_model = dynamics_model
        self.prediction_model = prediction_model
        
    def run(self, num_simulations, root_state):
        
        # init root
        p, v = self.prediction_model(root_state)
        p, v = p.detach(), v.detach()
        self.root = self.init_root(root_state, p)
        
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
    
        # working?
        onehot = F.one_hot(torch.tensor(action).to(device),num_classes=self.num_actions).reshape(1,self.num_actions).to(dtype)
        in_dynamics = torch.cat([parent.state,onehot],dim=1)
        
        next_state, reward = self.dynamics_model(in_dynamics)
        next_state, reward = next_state.detach(), reward.detach()   
        
        p, v = self.prediction_model(next_state)
        p, v = p.detach(), v.detach()
        
        node.state = next_state
        node.reward = reward
        
        for i in range(self.num_actions):
            node.edges[i] = MCTS_Node(p[0,i])

        return v
    
    def backup(self, value):
   
        for node in reversed(self.node_trajectory):
        
            node.value_sum += value
            node.visits += 1
            
            value = node.reward + self.gamma * value         
            
    def upper_confidence_bound(self, node):
        
        ucb_scores = []
        
        for i in range(self.num_actions):
            ucb_scores.append(self.ucb_score(node,node.edges[i]))
            
        amax = np.amax(ucb_scores)
        actions = []
        for i in range(len(ucb_scores)):
            if amax == ucb_scores[i]:
                actions.append(i)
        
        action = np.array(np.random.choice(actions, 1), dtype=np.int64)[0]
        return action, node.edges[action] 
        #return np.argmax(ucb_scores)
        
    def ucb_score(self, parent, child):
        
        pb_c = np.log((parent.visits + self.c2 + 1) / self.c2) + self.c1
        pb_c *= np.sqrt(parent.visits) / (child.visits + 1)

        prior_score = pb_c * child.p
        if child.visits > 0:
            value_score = child.reward + self.gamma * child.search_value()
        else:
            value_score = 0
        return prior_score + value_score

            
    def get_pi(self):
        
        # get action probabilites
        edge_visits = []
        for i in range(self.num_actions):
            edge_visits.append(self.root.edges[i].visits)
            
        return np.array(edge_visits) / self.root.visits
        
    def init_root(self, state, p):
        p = p.detach().cpu().numpy()
        
        node = MCTS_Node(0)
        node.state = state
        node.reward = 0
        
        for i in range(self.num_actions):
            node.edges[i] = MCTS_Node(p[0,i])
    
        return node
        
    def expand_node(self, node):
        pass