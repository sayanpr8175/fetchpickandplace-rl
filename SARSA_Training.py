import gymnasium as gym
import gymnasium_robotics
import numpy as np
import matplotlib.pyplot as plt
import time
import pickle
from tqdm import tqdm
import os

path_models = "C:\\Users\\sayan\\Documents\\RLProjects\\Final_code_ppo_SARSA\\models"
path_plots = "C:\\Users\\sayan\\Documents\\RLProjects\\Final_code_ppo_SARSA\\plots"
os.makedirs( path_models, exist_ok=True)
os.makedirs(path_plots, exist_ok=True)


class SARSAAgent:
    def __init__(self, state_dim, action_dim, action_bounds, discretization=10, learning_rate=0.1, gamma=0.99, epsilon=0.3):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bounds = action_bounds
        self.discretization = discretization
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        
        self.state_bins = [self.discretization] * self.state_dim
        self.action_bins = [self.discretization] * self.action_dim
        
        self.q_table = np.zeros(self.state_bins + self.action_bins)
    
    def discretize_state(self, state):

        normalized_state = (state - state.min()) / (state.max() - state.min() + 1e-10)
        discrete_state = tuple(min(int(ns * self.discretization), self.discretization-1) for ns in normalized_state)
        return discrete_state
    
    def discretize_action(self, action):

        normalized_action = (action - self.action_bounds[0]) / (self.action_bounds[1] - self.action_bounds[0] + 1e-10)
        discrete_action = tuple(min(int(na * self.discretization), self.discretization-1) for na in normalized_action)
        return discrete_action
    
    def continuous_action(self, discrete_action):
        
        normalized_action = np.array([a / self.discretization for a in discrete_action])
        continuous_action = normalized_action * (self.action_bounds[1] - self.action_bounds[0]) + self.action_bounds[0]
        return continuous_action
    
    def select_action(self, state):
        discrete_state = self.discretize_state(state)
        
        if np.random.random() < self.epsilon:
            discrete_action = tuple(np.random.randint(0, self.discretization, size=self.action_dim))
        else:
            discrete_action = np.unravel_index(np.argmax(self.q_table[discrete_state]), self.action_bins)
        
        return self.continuous_action(discrete_action), discrete_action
    
    def update(self, state, action, reward, next_state, next_action):

        discrete_state = self.discretize_state(state)
        discrete_action = self.discretize_action(action)
        discrete_next_state = self.discretize_state(next_state)
        discrete_next_action = self.discretize_action(next_action)
        
        current_q = self.q_table[discrete_state + discrete_action]
        next_q = self.q_table[discrete_next_state + discrete_next_action]
        
        # Q(s,a) = Q(s,a) + lr * [r + gamma * Q(s',a') - Q(s,a)]
        self.q_table[discrete_state + discrete_action] += self.lr * (reward + self.gamma * next_q - current_q)
    
    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.q_table, f)
    
    def load(self, filename):
        with open(filename, 'rb') as f:
            self.q_table = pickle.load(f)


def extract_features(observation):
    if isinstance(observation, dict):
        gripper_pos = observation['observation'][:3]
        object_pos = observation['observation'][3:6]
        object_rel_pos = observation['observation'][6:9]
        goal = observation['desired_goal']
        
        features = np.concatenate([
            object_rel_pos,
            goal - object_pos,
        ])
        return features
    else:
        return observation


def train_sarsa(env_name, num_episodes=10000, render_freq=10000, render_mode=None):
    

    env = gym.make(env_name)
    
    observation, _ = env.reset()
    state_features = extract_features(observation)
    state_dim = len(state_features)
    action_dim = env.action_space.shape[0]
    action_low = env.action_space.low
    action_high = env.action_space.high
    
    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")
    print(f"Action bounds: {action_low} to {action_high}")
    
    
    agent = SARSAAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        action_bounds=(action_low, action_high),
        discretization=5,
        learning_rate=0.1,
        gamma=0.99,
        epsilon=0.3
    )
    
    episode_rewards = []
    
    for episode in tqdm(range(num_episodes)):
        observation, _ = env.reset()
        state = extract_features(observation)
        action, discrete_action = agent.select_action(state)
        
        total_reward = 0
        done = False
        truncated = False
        
        #render_mode = "human" if episode % render_freq == 0 else None
        
        if render_mode:
            env_render = gym.make(env_name, render_mode=render_mode)
            observation_render, _ = env_render.reset()
        
        while not (done or truncated):
            observation_new, reward, done, truncated, _ = env.step(action)

            if render_mode:
                observation_render_new, _, done_render, truncated_render, _ = env_render.step(action)
            
            next_state = extract_features(observation_new)
            next_action, next_discrete_action = agent.select_action(next_state)
            
            agent.update(state, action, reward, next_state, next_action)
            
            state = next_state
            action = next_action
            discrete_action = next_discrete_action
            
            total_reward += reward
        
        if render_mode:
            env_render.close()
        
        agent.epsilon = max(0.01, agent.epsilon * 0.995)
        
        episode_rewards.append(total_reward)

        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            print(f"Episode {episode+1}/{num_episodes}, Avg Reward: {avg_reward:.2f}, Epsilon: {agent.epsilon:.2f}")
    
    agent.save("./models/sarsa_agent_fetch.pkl")
    
    plt.figure(figsize=(10, 6))
    plt.plot(episode_rewards)
    plt.title('SARSA Training Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.savefig('./plots/sarsa_rewards.png')
    plt.close()
    
    plt.figure(figsize=(10, 6))
    window_size = 20
    smoothed_rewards = [np.mean(episode_rewards[max(0, i-window_size):i+1]) for i in range(len(episode_rewards))]
    plt.plot(smoothed_rewards)
    plt.title('SARSA Training Rewards (Smoothed)')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.savefig('./plots/sarsa_smoothed_rewards.png')
    plt.close()
    
    env.close()
    return agent


if __name__ == "__main__":
    env_name = "FetchPickAndPlace-v4"
    print(f"Training SARSA agent on {env_name}")
    
    agent = train_sarsa(env_name, num_episodes=100000, render_freq=1000,render_mode=None)
    print("Training completed and model saved.")




















    