import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from collections import deque

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
    

# class piNet(nn.Module):
#     def __init__(self, state_dim, action_dim, hidden_dim) -> None:
#         super(piNet, self).__init__()
        
        


# class AgentREINFORCE():
#     def __init__(self, state_dim, action_dim, hidden_dim = 40, alpha = None, gamma = None, epsilon = None) -> None:
#         self.state_dim = state_dim
#         self.action_dim = action_dim
#         self.hidden_dim = hidden_dim
        
#         if alpha is None:
#             self.alpha = 0.001
#         else:
#             self.alpha = alpha
        
#         if gamma is None:
#             self.gamma = 0.99
#         else:
#             self.gamma = gamma
        
#         if epsilon is None:
#             self.epsilon = 0.1
#         else:
#             self.epsilon = epsilon
            
    
#     self.policy


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
    
    
    def update_Q(self, reward, state:torch.Tensor, action, state_next:torch.Tensor, terminated: bool, action_next):
        Q_sa = self.qnet(state)[action]
        # print("Q_sa:")
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
        
class ExperienceBuffer:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        # print(batch.type)
        states, actions, rewards, next_states, dones = zip(*batch)
        return torch.stack(states), torch.tensor(actions), torch.tensor(rewards, dtype=torch.float32), torch.stack(next_states), torch.tensor(dones)
        # return torch.tensor(states, dtype=torch.float32), torch.tensor(actions), torch.tensor(rewards, dtype=torch.float32), torch.tensor(next_states, dtype=torch.float32), torch.tensor(dones)


    def __len__(self):
        return len(self.buffer)


class AgentSARSA_replay_buffer(AgentSARSA):
    def __init__(self, state_dim, action_dim, hidden_dim=40, alpha=None, gamma=None, epsilon=None, n=None, max_buffer_size=None, batch_size=None, epsilon_decay = 0.995, min_epsilon = 0.1) -> None:
        super().__init__(state_dim, action_dim, hidden_dim, alpha, gamma, epsilon, n)
        
        if max_buffer_size is None:
            self.max_buffer_size = 10000
        else:
            self.max_buffer_size = max_buffer_size
        
        if batch_size is None:
            self.batch_size = 32
        else:
            self.batch_size = batch_size
        
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        
        self.buffer = ExperienceBuffer(self.max_buffer_size)
        
    def update_Q(self, reward, state: torch.Tensor, action, state_next: torch.Tensor, terminated: bool, action_next):
        # return super().update_Q(reward, state, action, state_next, terminated, action_next)
        # states, actions, rewards, next_states, dones
        
        experience = state, action, reward, state_next, terminated
        # print(f"experience: {experience}")
        self.buffer.add(experience=experience)
        
        if len(self.buffer) < self.batch_size:
            return
        
        states, actions, rewards, state_nexts, terminateds = self.buffer.sample(self.batch_size)
        # print('states')
        # print(states.type)
        # print(states)
        
        
        Q_sa_replay = self.qnet(states)
        Q_sa_next_replay = self.qnet(state_nexts)
        
        targets = Q_sa_replay.clone()

        for i in range(self.batch_size):
            targets[i, actions[i]] = rewards[i] + (~terminateds[i]) * self.gamma * torch.max(Q_sa_next_replay[i])


        self.qnet_optimizer.zero_grad()
        Q_network_loss = self.MSELoss_function(Q_sa_replay, targets)
        Q_network_loss.backward()
        self.qnet_optimizer.step()
        
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay
        
        # # print(Q_sa_replay)
        # # print(torch.max(Q_sa_replay, dim=1))
        
        # # print("Q_sa_replay")
        # # print(Q_sa_replay)
        
                
        # Q_sa = self.qnet(state)
        
        
        # # Q_sa = self.qnet(state)[action]
        # # print("Q_sa:")
        # # print(self.qnet(state))
        # # print(self.qnet(state).shape)
        
        # # Q_sa = torch.gather(self.qnet(state), index=torch.Tensor(action))
        # # print(Q_sa.requires_grad)
        
        # # Q_sa_try = torch.gather(self.qnet(state), dim=1, index=action)
        # # print(f"Q_sa: {Q_sa}")
        # # print(f"Q_sa_try: {Q_sa_try}")
        
        # Q_sa_next = (self.qnet(state_next)[action_next])
        
        
        # # Detaching as we don't need to find the gradient of the TD_estimate. 
        # # It acts like the 'label'/''true-y' of the algorithm
        # TD_estimate = (reward + self.gamma * Q_sa_next).detach()
        # # print(TD_estimate.requires_grad)
        
        # # TD_estimate = (reward + self.gamma * Q_sa_next)
        
        # # print(f"TD_estimate : {TD_estimate}")
        
        # # Loss to reduce
        # Q_network_loss = self.MSELoss_function(Q_sa, TD_estimate)
        
        # self.qnet_optimizer.zero_grad()
        # Q_network_loss.backward()
        # self.qnet_optimizer.step()
        return 
    

    
    