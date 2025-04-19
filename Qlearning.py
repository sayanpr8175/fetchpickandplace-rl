import gymnasium as gym
import gymnasium_robotics
import numpy as np

import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

import matplotlib.pyplot as plt
from tqdm import tqdm


class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(PolicyNetwork, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, state):
        x = self.network(state)
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)

        log_std = torch.clamp(log_std, min=-20, max=2)
        return mean, log_std
    
    def get_action(self, state, deterministic=False):
        mean, log_std = self.forward(state)
        
        if deterministic:
            return mean
        
        std = log_std.exp()
        normal = Normal(mean, std)
        action = normal.sample()
        return action
    
    def get_log_prob(self, state, action):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        log_prob = normal.log_prob(action).sum(dim=-1, keepdim=True)
        return log_prob

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(QNetwork, self).__init__()
        
        self.q = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.q(x)

class QLearning:
    def __init__(self, env, device='cpu', lr=3e-4, gamma=0.99, tau=0.005, buffer_size=100000):
        self.env = env
        self.device = device
        self.gamma = gamma
        self.tau = tau
        
        self.state_dim = env.observation_space['observation'].shape[0] + env.observation_space['desired_goal'].shape[0]
        self.action_dim = env.action_space.shape[0]
        
        self.policy = PolicyNetwork(self.state_dim, self.action_dim).to(device)
        self.q_network = QNetwork(self.state_dim, self.action_dim).to(device)
        self.target_q_network = QNetwork(self.state_dim, self.action_dim).to(device)
        
        for target_param, param in zip(self.target_q_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(param.data)
        
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.q_optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        self.buffer_size = buffer_size
        self.replay_buffer = []
        
        self.rewards_history = []
        
    def get_state(self, obs):
        
        state = np.concatenate([obs['observation'], obs['desired_goal']], axis=-1)
        return torch.FloatTensor(state).unsqueeze(0).to(self.device)
    
    def soft_update(self):
        for target_param, param in zip(self.target_q_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
    
    def add_to_buffer(self, state, action, reward, next_state, done):
        
        self.replay_buffer.append((state, action, reward, next_state, done))
        if len(self.replay_buffer) > self.buffer_size:
            self.replay_buffer.pop(0)
    
    def sample_batch(self, batch_size=128):
        
        if len(self.replay_buffer) < batch_size:
            batch_size = len(self.replay_buffer)
        
        indices = np.random.randint(0, len(self.replay_buffer), size=batch_size)
        states, actions, rewards, next_states, dones = [], [], [], [], []
        
        for i in indices:
            s, a, r, ns, d = self.replay_buffer[i]
            states.append(s)
            actions.append(a)
            rewards.append(r)
            next_states.append(ns)
            dones.append(d)
        
        states = torch.cat(states, dim=0)
        actions = torch.cat(actions, dim=0)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.cat(next_states, dim=0)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        return states, actions, rewards, next_states, dones
    
    def train(self, num_episodes=100000, max_steps=200, batch_size=128, exploration_steps=5000):
        total_steps = 0
        
        for episode in tqdm(range(num_episodes)):
            obs = self.env.reset()[0]
            state = self.get_state(obs)
            
            total_reward = 0
            
            for step in range(max_steps):
                total_steps += 1
                
                if total_steps < exploration_steps:
                    action_np = self.env.action_space.sample()
                    action = torch.FloatTensor(action_np).unsqueeze(0).to(self.device)
                else:
                    action = self.policy.get_action(state)
                    action_np = action.cpu().detach().numpy().squeeze(0)
                
                next_obs, reward, terminated, truncated, info = self.env.step(action_np)
                done = terminated or truncated
                next_state = self.get_state(next_obs)
                
                self.add_to_buffer(state, action, reward, next_state, float(done))
                
                if len(self.replay_buffer) > batch_size:
                    states, actions, rewards, next_states, dones = self.sample_batch(batch_size)
                    
                    with torch.no_grad():
                        
                        next_actions = self.policy.get_action(next_states, deterministic=True)
                        
                        next_q_values = self.target_q_network(next_states, next_actions)
                        
                        q_targets = rewards + (1 - dones) * self.gamma * next_q_values
                    
                    q_values = self.q_network(states, actions)
                    
                    q_loss = nn.MSELoss()(q_values, q_targets)
                    self.q_optimizer.zero_grad()
                    q_loss.backward()
                    self.q_optimizer.step()
                    
                    policy_actions = self.policy.get_action(states)
                    policy_loss = -self.q_network(states, policy_actions).mean()
                    
                    self.policy_optimizer.zero_grad()
                    policy_loss.backward()
                    self.policy_optimizer.step()

                    self.soft_update()
                
                state = next_state
                total_reward += reward
                
                if done:
                    break
            
            self.rewards_history.append(total_reward)
            
            if (episode + 1) % 10 == 0:
                avg_reward = np.mean(self.rewards_history[-10:])
                print(f"Episode {episode + 1}, Avg Reward: {avg_reward:.3f}")
        
        return self.rewards_history
    
    def test(self, num_episodes=10, render=False):
        success_count = 0
        
        for episode in range(num_episodes):
            obs = self.env.reset()[0]
            state = self.get_state(obs)
            
            for step in range(50):
                action = self.policy.get_action(state, deterministic=True)
                action_np = action.cpu().detach().numpy().squeeze(0)
                
                next_obs, reward, terminated, truncated, info = self.env.step(action_np)
                done = terminated or truncated
                
                if render:
                    self.env.render()
                    time.sleep(0.01)
                
                if 'is_success' in info and info['is_success']:
                    success_count += 1
                    break
                
                if done:
                    break
                
                state = self.get_state(next_obs)
        
        success_rate = success_count / num_episodes
        print(f"Success rate: {success_rate * 100:.2f}%")
        return success_rate

def plot_rewards(rewards):
    plt.figure(figsize=(10, 5))
    plt.plot(rewards)
    plt.title('Rewards per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.grid(True)
    
    window_size = 20
    if len(rewards) >= window_size:
        moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
        plt.plot(range(window_size-1, len(rewards)), moving_avg, 'r-', label='Moving Average')
        plt.legend()
    
    plt.savefig('./plots/fetch_qlearning_rewards.png')
    plt.show()

def main():
    env = gym.make("FetchPickAndPlace-v4", render_mode='human')
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    q_learning_agent = QLearning(env, device=device)
    
    print("Starting training...")
    rewards = q_learning_agent.train(num_episodes=10000)
    
    plot_rewards(rewards)
    
    #print("Testing the agent...")
    #q_learning_agent.test(num_episodes=10, render=True)
    
    torch.save({
        'policy': q_learning_agent.policy.state_dict(),
        'q_network': q_learning_agent.q_network.state_dict(),
    }, './models/qlearning_fetch_model.pt')

if __name__ == "__main__":
    main()




    
