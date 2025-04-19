import gymnasium as gym
import gymnasium_robotics

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal

import random
import time
import numpy as np

from collections import deque
from tqdm import tqdm

import matplotlib.pyplot as plt



class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, hidden_dim=256):
        super(Actor, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )
        
        self.max_action = max_action
        
    def forward(self, state):
        return self.max_action * self.network(state)

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.network(x)

class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)
        
    def push(self, state, action, reward, next_state, done, info=None):
        self.buffer.append((state, action, reward, next_state, done, info))
        
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done, _ = zip(*batch)
        
        state = torch.FloatTensor(np.array(state))
        action = torch.FloatTensor(np.array(action))
        reward = torch.FloatTensor(np.array(reward)).unsqueeze(1)
        next_state = torch.FloatTensor(np.array(next_state))
        done = torch.FloatTensor(np.array(done)).unsqueeze(1)
        
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)

class DDPG_HER:
    def __init__(self, env, device='cpu', actor_lr=1e-4, critic_lr=1e-3, gamma=0.98, tau=0.005):
        self.env = env
        self.device = device
        self.gamma = gamma
        self.tau = tau
        
        self.max_action = env.action_space.high[0]
        
        self.obs_dim = env.observation_space['observation'].shape[0]
        self.goal_dim = env.observation_space['desired_goal'].shape[0]
        self.state_dim = self.obs_dim + self.goal_dim
        self.action_dim = env.action_space.shape[0]
        
        self.actor = Actor(self.state_dim, self.action_dim, self.max_action).to(device)
        self.actor_target = Actor(self.state_dim, self.action_dim, self.max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        
        self.critic = Critic(self.state_dim, self.action_dim).to(device)
        self.critic_target = Critic(self.state_dim, self.action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        
        self.replay_buffer = ReplayBuffer()
        
        self.rewards_history = []
        self.success_history = []
        
    def get_state(self, obs, goal=None):
        if goal is None:
            goal = obs['desired_goal']
        state = np.concatenate([obs['observation'], goal])
        return state
    
    def select_action(self, state, noise=0.1):
        state = torch.FloatTensor(state).to(self.device)
        action = self.actor(state).cpu().data.numpy()
        
        action = action + np.random.normal(0, noise, size=self.action_dim)
        action = np.clip(action, -self.max_action, self.max_action)
        
        return action
    
    def store_transition(self, obs, action, reward, next_obs, done, info):
        state = self.get_state(obs)
        next_state = self.get_state(next_obs)
        
        self.replay_buffer.push(state, action, reward, next_state, float(done), info)
        
        if 'achieved_goal' in obs and 'achieved_goal' in next_obs:
            
            achieved_goal = next_obs['achieved_goal']
            
            #substitute_reward = self.env.compute_reward(
            #    next_obs['achieved_goal'], 
            #    achieved_goal, 
            #    info)
            

            substitute_reward = float(self.env.unwrapped.compute_reward(next_obs['achieved_goal'], achieved_goal, info))
            
            substitute_state = self.get_state(obs, achieved_goal)
            substitute_next_state = self.get_state(next_obs, achieved_goal)
            
            self.replay_buffer.push(
                substitute_state, 
                action, 
                substitute_reward, 
                substitute_next_state, 
                float(done)
            )
    
    def soft_update(self):
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
            
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
    
    def train(self, batch_size=128):
        if len(self.replay_buffer) < batch_size:
            return
        
        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)
        
        state = state.to(self.device)
        action = action.to(self.device)
        reward = reward.to(self.device)
        next_state = next_state.to(self.device)
        done = done.to(self.device)
        
        #target Q-value
        with torch.no_grad():
            next_action = self.actor_target(next_state)
            target_q = self.critic_target(next_state, next_action)
            target_q = reward + (1 - done) * self.gamma * target_q
        
        #current Q-value
        current_q = self.critic(state, action)
        
        #critic loss
        critic_loss = F.mse_loss(current_q, target_q)
        
        #Optimize critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        #actor loss
        actor_loss = -self.critic(state, self.actor(state)).mean()
        
        # Optimize actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Update target nets
        self.soft_update()
    
    def learn(self, num_episodes=10000, max_steps=50, batch_size=128, n_updates=40, log_interval=10):
        for episode in tqdm(range(num_episodes)):
            obs = self.env.reset()[0]
            episode_reward = 0
            success = False
            
            for step in range(max_steps):
                state = self.get_state(obs)
                action = self.select_action(state, noise=0.1)
                
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                
                self.store_transition(obs, action, reward, next_obs, done, info)
                
                if len(self.replay_buffer) >= batch_size:
                    for _ in range(n_updates):
                        self.train(batch_size)
                
                obs = next_obs
                episode_reward += reward
                
                if 'is_success' in info and info['is_success']:
                    success = True
                
                if done:
                    break
            
            self.rewards_history.append(episode_reward)
            self.success_history.append(float(success))
            
            # Log
            if (episode + 1) % log_interval == 0:
                avg_reward = np.mean(self.rewards_history[-log_interval:])
                success_rate = np.mean(self.success_history[-log_interval:])
                print(f"Episode {episode + 1}, Avg Reward: {avg_reward:.3f}, Success Rate: {success_rate:.3f}")
        
        return self.rewards_history, self.success_history
    
    def test(self, num_episodes=10, render=True):
        success_count = 0
        total_rewards = 0
        
        for episode in range(num_episodes):
            obs = self.env.reset()[0]
            episode_reward = 0
            
            for step in range(50):
                state = self.get_state(obs)
                action = self.select_action(state, noise=0)
                
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                
                if render:
                    self.env.render()
                    time.sleep(0.01)
                
                obs = next_obs
                episode_reward += reward
                
                if 'is_success' in info and info['is_success']:
                    success_count += 1
                    break
                
                if done:
                    break
                    
            total_rewards += episode_reward
        
        success_rate = success_count / num_episodes
        avg_reward = total_rewards / num_episodes
        print(f"Test Results - Success Rate: {success_rate * 100:.2f}%, Avg Reward: {avg_reward:.3f}")
        return success_rate, avg_reward

def plot_results(rewards, success_rates=None):
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(rewards)
    plt.title('Rewards per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.grid(True)
    
    #moving avg
    window_size = 100
    if len(rewards) >= window_size:
        moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
        plt.plot(range(window_size-1, len(rewards)), moving_avg, 'r-', label='Moving Average')
        plt.legend()
    
    if success_rates is not None:
        plt.subplot(1, 2, 2)
        plt.plot(success_rates)
        plt.title('Success Rate')
        plt.xlabel('Episode')
        plt.ylabel('Success Rate')
        plt.grid(True)
        
        if len(success_rates) >= window_size:
            success_avg = np.convolve(success_rates, np.ones(window_size)/window_size, mode='valid')
            plt.plot(range(window_size-1, len(success_rates)), success_avg, 'r-', label='Moving Average')
            plt.legend()
    
    plt.tight_layout()
    plt.savefig('C:/Users/sayan/Documents/RLProjects/Final_code_ppo_SARSA/ddpg_models/fetch_ddpg_her_results.png')
    plt.show()

def main():
    
    #env = gym.make("FetchPickAndPlace-v4", render_mode='human')

    env = gym.make("FetchPickAndPlace-v4")
    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu")
    print(f"Using device: {device}")
    
    agent = DDPG_HER(env, device=device)
    
    print("Starting training...")
    rewards, success_rates = agent.learn(num_episodes=10000, log_interval=10)
    
    plot_results(rewards, success_rates)
    
    # Testing the agent # Error
    #print("Testing the agent...")
    #agent.test(num_episodes=10, render=True)
    
    torch.save({
        'actor': agent.actor.state_dict(),
        'critic': agent.critic.state_dict(),
    }, 'C:/Users/sayan/Documents/RLProjects/Final_code_ppo_SARSA/ddpg_modelsddpg_her_fetch_model.pt')

if __name__ == "__main__":
    main()




    
