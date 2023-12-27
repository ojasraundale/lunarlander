import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

def AlwaysUp(state):
    return 2

class Network(nn.Module):
    def __init__(self, state_num , action_num, hidden_layer):
        
        super(Network, self).__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_layer = nn.Linear(state_num, hidden_layer)
        self.h1_layer = nn.Linear(hidden_layer, hidden_layer)
        self.h2_layer = nn.Linear(hidden_layer, hidden_layer)
        self.output_layer = nn.Linear(hidden_layer, action_num)
        

    def forward(self, state):
        
        xh = F.relu(self.input_layer(state))
        hh1 = F.relu(self.h1_layer(xh))
        hh2 = F.tanh(self.h2_layer(hh1))
        state_action_values = self.output_layer(hh2)
        
        return state_action_values

class Q_Network(nn.Module):
    def __init__(self, state_dim , action_dim, hidden_dim):
        super(Q_Network, self).__init__()
        self.x_layer = nn.Linear(state_dim, hidden_dim)
        self.h_layer = nn.Linear(hidden_dim, hidden_dim)
        self.y_layer = nn.Linear(hidden_dim, action_dim)
        print(self.x_layer)
        print(self.y_layer)

    def forward(self, state):
        xh = F.relu(self.x_layer(state))
        hh = F.relu(self.h_layer(xh))
        state_action_values = self.y_layer(hh)
        softmaxed_action_values = F.softmax(state_action_values, dim=-1)
        
        # print(f"state_action_values: {state_action_values}")
        # print(f"softmaxed_action_values: {softmaxed_action_values}")
        return softmaxed_action_values
    


class AgentSARSA():
    
    def __init__(self, state_dim, action_dim, hidden_dim = 40, alpha = None, gamma = None, epsilon = None, n = None) -> None:
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        if alpha is None:
            self.alpha = 0.001
        else:
            self.alpha = alpha
        
        if gamma is None:
            self.gamma = 0.99
        else:
            self.gamma = gamma
        
        if epsilon is None:
            self.epsilon = 0.1
        else:
            self.epsilon = epsilon
            
        if n is None:
            self.n = 1
        
        else:
            self.n = n
            
        
        
        
        self.qnet = Q_Network(self.state_dim, self.action_dim, self.hidden_dim)
        self.qnet_optimizer = torch.optim.Adam(self.qnet.parameters(), lr=self.alpha)
        self.MSELoss_function = nn.MSELoss()
        
    
    def get_action_probs_from_Q(self, state):
        return self.qnet(state)
    
    def get_action_softmax(self, state):
        """
        Takes in the A probabilities as weights and returns an action
        """
        
        a_probs = self.get_action_probs_from_Q(state=state)
        # print(a_probs)
        return random.choices(population=list(range(self.action_dim)), weights=a_probs)[0]
    
    
    def get_action_epsilon_greedy(self, state):
        if np.random.uniform(0, 1) < self.epsilon:
            # print("Randoom")
            a = random.choices(population=list(range(self.action_dim)))[0]  # choose random action
            # print(a)
            return a
        else:
            network_output_to_numpy = self.get_action_probs_from_Q(state).data.numpy()
            # print(network_output_to_numpy)
            a = np.argmax(network_output_to_numpy)
            # print(a)
            return a  # choose greedy action
    
    
    def update_Q(self, reward, state:torch.Tensor, action, state_next:torch.Tensor, action_next):
        Q_sa = self.qnet(state)[action]
        # print(self.qnet(state))
        # print(self.qnet(state).shape)
        
        # Q_sa = torch.gather(self.qnet(state), index=torch.Tensor(action))
        # print(Q_sa.requires_grad)
        
        # Q_sa_try = torch.gather(self.qnet(state), dim=1, index=action)
        # print(f"Q_sa: {Q_sa}")
        # print(f"Q_sa_try: {Q_sa_try}")
        
        Q_sa_next = (self.qnet(state_next)[action_next])
        
        
        # Detaching as we don't need to find the gradient of the TD_estimate. 
        # It acts like the 'label'/''true-y' of the algorithm
        TD_estimate = (reward + self.gamma * Q_sa_next).detach()
        # print(TD_estimate.requires_grad)
        
        # TD_estimate = (reward + self.gamma * Q_sa_next)
        
        # print(f"TD_estimate : {TD_estimate}")
        
        # Loss to reduce
        Q_network_loss = self.MSELoss_function(Q_sa, TD_estimate)
        
        self.qnet_optimizer.zero_grad()
        Q_network_loss.backward()
        self.qnet_optimizer.step()
        return 
    
    def update_Q_n_step(self, G_n, use_tau_n: bool, state_tau:torch.Tensor, action_tau, state_tau_n:torch.Tensor=None, action_tau_n=None):
        Q_tau = self.qnet(state_tau)[action_tau]
        
        TD_estimate = torch.from_numpy(np.array(float(G_n)))
        # print(f"Tensor G_n is {TD_estimate} with dtype {TD_estimate.dtype}")
        
        if use_tau_n:
            Q_tau_n = self.qnet(state_tau_n)[action_tau_n]
            # print(f"Gamma power n is: {self.gamma ** self.n} with dtype {(self.gamma ** self.n).dtype}")
            TD_estimate += TD_estimate + (self.gamma ** self.n) * Q_tau_n
        
        TD_estimate = (TD_estimate.float().detach())
        
        
        Q_network_loss = self.MSELoss_function(Q_tau, TD_estimate)
        # print(f"Q_tau_n type: {Q_tau_n.dtype}, TD_estimate type: {TD_estimate.dtype}")
        
        self.qnet_optimizer.zero_grad()
        Q_network_loss.backward()
        self.qnet_optimizer.step()
        
        
    # def get_gradient(self):
    #     self.qnet_optimizer.zero_grad()
    #     b = self.qnet.backward()
    #     self.qnet.step()
    #     return b
        
        
        
    
    