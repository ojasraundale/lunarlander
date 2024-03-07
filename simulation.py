import gymnasium as gym
import pygame
import numpy as np
import agents
import torch

def runEpisodeSarsa(env, agent:agents.AgentSARSA):
    total_reward = 0.0
    steps = 0
    restart = False
    s, info = env.reset()
    s_tensor = torch.from_numpy(s).float()
    a = agent.get_action_softmax(s_tensor)
    MAX_STEPS = 1000
    
    while True:
        if steps>MAX_STEPS:
            break
        # a = agents.AlwaysUp(s)
        # a = 0
        # print("a is " + str(a))
        s_next, r, terminated, truncated, info = env.step(a)
        s_tensor = torch.from_numpy(s).float()
        s_next_tensor = torch.from_numpy(s_next).float()
        # a_next = agent.get_action_softmax(s_next_tensor)
        a_next = agent.get_action_epsilon_greedy(s_next_tensor)
        
        # print(s_tensor)
        agent.update_Q(reward=r, state=s_tensor, action=a, state_next=s_next_tensor, terminated=terminated, action_next=a_next)
        
        
        # print(s,r,terminated,truncated,info)
        # print(info)
        # print(f"r is {r}")
        total_reward += r
        # if steps % 200 == 0 or terminated or truncated:
            # print("\naction " + str([f"{x:+0.2f}" for x in a]))
            # print(f"step {steps} total_reward {total_reward:+0.2f}")
        steps += 1
        if terminated or truncated or restart:
            print("Breaking")
            break
        
        s, a = s_next, a_next
    # print(total_reward)
    print(f"N_steps = {steps}")
    return total_reward


def runEpisode(env, agent:agents.AgentSARSA):
    total_reward = 0.0
    steps = 0
    restart = False
    s, info = env.reset()
    s_tensor = torch.from_numpy(s).float()
    a = agent.get_action_softmax(s_tensor)
    while True:
        # a = agents.AlwaysUp(s)
        # a = 0
        # print("a is " + str(a))
        s_next, r, terminated, truncated, info = env.step(a)
        s_tensor = torch.from_numpy(s).float()
        s_next_tensor = torch.from_numpy(s_next).float()
        a_next = agent.get_action_softmax(s_next_tensor)
        a_next = agent.get_action_epsilon_greedy(s_next_tensor)
        
        
        # agent.update_Q(reward=r, state=s_tensor, action=a, state_next=s_next_tensor, terminated=terminated, action_next=a_next)
        
        
        # print(s,r,terminated,truncated,info)
        # print(info)
        # print(f"r is {r}")
        total_reward += r
        # if steps % 200 == 0 or terminated or truncated:
            # print("\naction " + str([f"{x:+0.2f}" for x in a]))
            # print(f"step {steps} total_reward {total_reward:+0.2f}")
        steps += 1
        if terminated or truncated or restart:
            # print("Breaking")
            break
        
        s, a = s_next, a_next
    # print(total_reward)
    # print(f"N_steps = {steps}")
    return total_reward
    

def G_n_step(agent:agents.AgentSARSA, s_tensor_list, a_list, r_list, n, tau, gamma):
    if not(len(s_tensor_list) == len(a_list) and len(s_tensor_list) == len(r_list)) :
        # print("List lengths match!")
        print("Lists Dont match!")
    
    len_list = len(s_tensor_list)
        
    # print(f"list length: {len(s_tensor_list)}, tau: {tau}, n:{n}")
    
    start = tau
    end = min(tau+n, len_list)
    
    # print(start, end)
    G_sum = 0
    discount_i = 1
    for i in range(start, end):
        G_sum += discount_i*r_list[i]
        discount_i = discount_i*gamma

    # if tau + n <= len_list:
    #     s_tau_n = s_tensor_list[len_list-1]
    #     a_tau_n = a_list[len_list-1]
    #     agent.get_action_probs_from_Q
        
    # print(f"G_sum is: {G_sum}")
    return G_sum


def NSteprunEpisodeSarsa(env, agent:agents.AgentSARSA, n:int, gamma:float):
    total_reward = 0.0
    restart = False
    
    steps = 0
    tau = -n
    s_tensor_list = []
    a_list = []
    r_list = []
    
    s, info = env.reset()
    s_tensor = torch.from_numpy(s).float()
    a = agent.get_action_softmax(s_tensor)

    MAX_episode_length = 1000
    terminated = False
    truncated = False
    episode_stopped = False
    
    while True:
        if steps >= MAX_episode_length or terminated or truncated:
            episode_stopped = True
            # print(steps)
        
        if not episode_stopped:
            # a = agents.AlwaysUp(s)
            # a = 0
            # print("a is " + str(a))
            s_next, r, terminated, truncated, info = env.step(a)
            s_tensor = torch.from_numpy(s).float()
            
            s_tensor_list.append(s_tensor)
            a_list.append(a)
            r_list.append(r)
            s_next_tensor = torch.from_numpy(s_next).float()
            # a_next = agent.get_action_softmax(s_next_tensor)
            a_next = agent.get_action_epsilon_greedy(s_next_tensor)
            
            # agent.update_Q(reward=r, state=s_tensor, action=a, state_next=s_next_tensor, action_next=a_next)
            
            total_reward += r
            steps += 1

            # at episode end, steps is the size of lists. It is the terminal state and needn't be updated.
            # So tau should update till list[steps-1]
        tau += 1
        if n == 1 and tau >= steps and episode_stopped :
            break
        
        
        if tau >= 0:
            G_n = G_n_step(agent=agent, s_tensor_list=s_tensor_list, a_list=a_list, r_list=r_list, n=n, tau=tau, gamma=gamma)
            s_tensor_tau = s_tensor_list[tau]
            a_tau = a_list[tau]
            
            if tau + n <= steps:
                s_tensor_tau_n = s_tensor_list[steps-1]
                a_tau_n = a_list[steps-1]
                agent.update_Q_n_step(G_n=G_n, 
                                      use_tau_n=True, 
                                      state_tau=s_tensor_tau, 
                                      action_tau=a_tau,
                                      state_tau_n=s_tensor_tau_n,
                                      action_tau_n=a_tau_n
                                      )
            else:
                agent.update_Q_n_step(G_n=G_n, 
                                      use_tau_n=False, 
                                      state_tau=s_tensor_tau, 
                                      action_tau=a_tau,
                                      )
        
        # print(s,r,terminated,truncated,info)
        # if steps % 200 == 0 or terminated or truncated:
            # print("\naction " + str([f"{x:+0.2f}" for x in a]))
            # print(f"step {steps} total_reward {total_reward:+0.2f}")

        # if terminated or truncated or restart:
        #     # print("Breaking")
        #     break
        
        if n > 1:
            if tau >= steps-1 and episode_stopped:
                # print(f"R at tau >= steps-1 os {r}")
                # print("tau >= steps-1")
                break
        
        
        
        s, s_tensor, a = s_next, s_next_tensor, a_next
    # print(total_reward)
    # print(f"N_steps = {steps}")
    return total_reward
    

def SARSA_semi_gradient():
    env = gym.make("LunarLander-v2", render_mode='human')
    # env = gym.make("LunarLander-v2", render_mode='rgb_array')
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # env = gym.make("LunarLander-v2", render_mode='rgb_array')
    # print(list(range(env.action_space.n)))
    agent_sarsa = agents.AgentSARSA(
                            state_dim=env.observation_space.shape[0],
                            action_dim=env.action_space.n,
                            hidden_dim=256,
                            alpha=0.005,
                            gamma=0.99, 
                            epsilon=0.5)
    
    # s, info = env.reset()
    # s_tensor = torch.from_numpy(s).float()
    # a_probs = agent_sarsa.get_action_probs_from_Q(s_tensor)
    # a = agent_sarsa.get_action_from_Q(s_tensor)
    # print(a)
    # print(a_probs)
    
    # grad = agent_sarsa.get_gradient()
    # print(grad)
    # agent = agents.AlwaysUp()
    while(True):
        G = runEpisodeSarsa(env, agent=agent_sarsa)
        print(G)
    

def n_step_SARSA_semi_gradient():
    env = gym.make("LunarLander-v2", render_mode='human')
    # env = gym.make("LunarLander-v2", render_mode='rgb_array')
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # env = gym.make("LunarLander-v2", render_mode='rgb_array')
    # print(list(range(env.action_space.n)))
    
    # hidden_dim = 256
    # alpha = 0.01
    # gamma = 0.99
    # epsilon = 0.15
    # n = 1
    hidden_dim = 20
    alpha = 0.0001
    gamma = 0.95
    epsilon = 0.01
    n = 20
    n_episodes = 1250
    
    agent_sarsa = agents.AgentSARSA(
                            state_dim=env.observation_space.shape[0],
                            action_dim=env.action_space.n,
                            hidden_dim=hidden_dim,
                            alpha=alpha,
                            gamma=gamma, 
                            epsilon=epsilon, 
                            n=n)
    

def Estimate_J():
    env = gym.make("LunarLander-v2", render_mode='human')
    # env = gym.make("LunarLander-v2", render_mode='rgb_array')
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    agent_sarsa = torch.load("Lunar Lander/10k steps sarsa full model")
    
    i_epi = 0
    Gs = []
    n_epi = 10
    while(i_epi<n_epi):
        # G = NSteprunEpisodeSarsa(env, agent=agent_sarsa, n=n, gamma=gamma)
        G = runEpisode(env, agent=agent_sarsa)
        Gs.append(G)
        print(G)
        i_epi+=1
    
    print(sum(Gs)/n_epi)
    


def SARSA_semi_gradient_experience_buffer():
    env = gym.make("LunarLander-v2", render_mode='human')
    # env = gym.make("LunarLander-v2", render_mode='rgb_array')
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # env = gym.make("LunarLander-v2", render_mode='rgb_array')
    # print(list(range(env.action_space.n)))
    agent_sarsa = agents.AgentSARSA_replay_buffer(
                            state_dim=env.observation_space.shape[0],
                            action_dim=env.action_space.n,
                            hidden_dim=128,
                            alpha=0.001,
                            gamma=0.99, 
                            epsilon=1, 
                            min_epsilon=0.05,
                            epsilon_decay=0.995,
                            max_buffer_size=20000, 
                            batch_size=128)
    
    
    # s, info = env.reset()
    # s_tensor = torch.from_numpy(s).float()
    # a_probs = agent_sarsa.get_action_probs_from_Q(s_tensor)
    # a = agent_sarsa.get_action_from_Q(s_tensor)
    # print(a)
    # print(a_probs)
    
    # grad = agent_sarsa.get_gradient()
    # print(grad)
    # agent = agents.AlwaysUp()
    # for _ in range(1):
    while(True):
        G = runEpisodeSarsa(env, agent=agent_sarsa)
        print(G)

if __name__ == "__main__":
    # SARSA_semi_gradient()
    SARSA_semi_gradient_experience_buffer()
    
    # n_step_SARSA_semi_gradient()
    # Estimate_J()