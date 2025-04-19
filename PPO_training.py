import gymnasium as gym
import gymnasium_robotics
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import os
from tqdm import tqdm


os.makedirs("C:\\Users\\sayan\\Documents\\RLProjects\\Final_code_ppo_SARSA\\models", exist_ok=True)
os.makedirs("C:\\Users\\sayan\\Documents\\RLProjects\\Final_code_ppo_SARSA\\plots", exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")


class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        super(PolicyNetwork, self).__init__()
        
        self.actor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        
        self.log_std = nn.Parameter(torch.zeros(output_dim))
        
        self.critic = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x):
        action_mean = self.actor(x)
        
        action_std = torch.exp(self.log_std)
        
        value = self.critic(x)
        
        return action_mean, action_std, value
    
    def get_action(self, state, deterministic=False):
        state_tensor = torch.FloatTensor(state).to(device)
        
        action_mean, action_std, _ = self.forward(state_tensor)
        
        if deterministic:
            return action_mean.detach().cpu().numpy()
        
        dist = Normal(action_mean, action_std)
        
        action = dist.sample()
        
        log_prob = dist.log_prob(action).sum(dim=-1)
        
        return action.detach().cpu().numpy(), log_prob.detach().cpu().numpy()
    
    def evaluate(self, states, actions):
        action_means, action_stds, values = self.forward(states)
        
        dist = Normal(action_means, action_stds)
        
        log_probs = dist.log_prob(actions).sum(dim=-1)
        
        entropy = dist.entropy().sum(dim=-1)
        
        return log_probs, entropy, values


def extract_features(observation):

    if isinstance(observation, dict):
        gripper_pos = observation['observation'][:3]
        object_pos = observation['observation'][3:6]
        object_rel_pos = observation['observation'][6:9]
        goal = observation['desired_goal']
        gripper_state = observation['observation'][9:11]
        
        features = np.concatenate([
            gripper_pos,
            object_pos,
            object_rel_pos,
            goal - object_pos,
            gripper_state
        ])
        return features
    else:
        return observation


class PPOAgent:
    def __init__(self, state_dim, action_dim, action_bounds, hidden_dim=128, lr=3e-4, gamma=0.99, clip_ratio=0.2, max_grad_norm=0.5):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_low, self.action_high = action_bounds
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.max_grad_norm = max_grad_norm
        
        self.policy = PolicyNetwork(state_dim, action_dim, hidden_dim).to(device)
        
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
    
    def normalize_action(self, action):
        return np.clip(action, self.action_low, self.action_high)
    
    def train(self, states, actions, old_log_probs, returns, advantages, epochs=10, batch_size=64):
        states = torch.FloatTensor(states).to(device)
        actions = torch.FloatTensor(actions).to(device)
        old_log_probs = torch.FloatTensor(old_log_probs).to(device)
        returns = torch.FloatTensor(returns).to(device)
        advantages = torch.FloatTensor(advantages).to(device)
        
        
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        
        for _ in range(epochs):

            indices = np.arange(len(states))
            np.random.shuffle(indices)
            
            for start_idx in range(0, len(states), batch_size):
                batch_indices = indices[start_idx:start_idx + batch_size]
                

                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]
                
                new_log_probs, entropy, values = self.policy.evaluate(batch_states, batch_actions)
                
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                
                surrogate1 = ratio * batch_advantages
                surrogate2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * batch_advantages
                
                policy_loss = -torch.min(surrogate1, surrogate2).mean()
                
                value_loss = nn.MSELoss()(values.squeeze(), batch_returns)
                
                entropy_loss = -entropy.mean()
                
                loss = policy_loss + 0.5 * value_loss - 0.01 * entropy_loss
                
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
        
        num_batches = len(states) // batch_size + (1 if len(states) % batch_size != 0 else 0)
        avg_policy_loss = total_policy_loss / (epochs * num_batches)
        avg_value_loss = total_value_loss / (epochs * num_batches)
        avg_entropy = total_entropy / (epochs * num_batches)
        
        return avg_policy_loss, avg_value_loss, avg_entropy
    
    def save(self, filename):
        torch.save(self.policy.state_dict(), filename)
    
    def load(self, filename):
        self.policy.load_state_dict(torch.load(filename, map_location=device))


def compute_gae(rewards, values, dones, next_value, gamma=0.99, lam=0.95):
    advantages = np.zeros_like(rewards)
    last_gae = 0
    
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_val = next_value
        else:
            next_val = values[t + 1]
        
        delta = rewards[t] + gamma * next_val * (1 - dones[t]) - values[t]
        
        advantages[t] = last_gae = delta + gamma * lam * (1 - dones[t]) * last_gae
    
    returns = advantages + values
    
    return returns, advantages


def train_ppo(env_name, num_episodes=1000, steps_per_episode=1000, render_mode=None):

    env = gym.make(env_name, render_mode=render_mode)
    
    observation, _ = env.reset()
    state_features = extract_features(observation)
    state_dim = len(state_features)
    action_dim = env.action_space.shape[0]
    action_low = env.action_space.low
    action_high = env.action_space.high
    
    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")
    print(f"Action bounds: {action_low} to {action_high}")
    
    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        action_bounds=(action_low, action_high),
        hidden_dim=128,
        lr=1e-4,
        gamma=0.99,
        clip_ratio=0.2,
        max_grad_norm=0.5
    )
    
    episode_rewards = []
    
    for episode in tqdm(range(num_episodes)):

        states = []
        actions = []
        log_probs = []
        rewards = []
        dones = []
        values = []
        
        observation, _ = env.reset()
        state = extract_features(observation)
        
        total_reward = 0
        
        for t in range(steps_per_episode):

            action, log_prob = agent.policy.get_action(state)
            
            action_env = agent.normalize_action(action)
            
            state_tensor = torch.FloatTensor(state).to(device)
            _, _, value = agent.policy.forward(state_tensor)
            value = value.detach().cpu().numpy().item()
            
            observation_new, reward, done, truncated, _ = env.step(action_env)
            
            next_state = extract_features(observation_new)
            
            states.append(state)
            actions.append(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            dones.append(done or truncated)
            values.append(value)
            
            state = next_state
            total_reward += reward
            
            if done or truncated:
                break
        
        if done or truncated:
            next_value = 0
        else:
            state_tensor = torch.FloatTensor(state).to(device)
            _, _, next_value = agent.policy.forward(state_tensor)
            next_value = next_value.detach().cpu().numpy().item()
        
        returns, advantages = compute_gae(
            rewards=np.array(rewards),
            values=np.array(values),
            dones=np.array(dones),
            next_value=next_value,
            gamma=agent.gamma,
            lam=0.95
        )
        
        policy_loss, value_loss, entropy = agent.train(
            states=np.array(states),
            actions=np.array(actions),
            old_log_probs=np.array(log_probs),
            returns=returns,
            advantages=advantages,
            epochs=10,
            batch_size=64
        )
        
        episode_rewards.append(total_reward)
        
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            print(f"Episode {episode+1}/{num_episodes}, Avg Reward: {avg_reward:.2f}")
            print(f"Policy Loss: {policy_loss:.4f}, Value Loss: {value_loss:.4f}, Entropy: {entropy:.4f}")
        
        #if (episode + 1) % 100 == 0:
        #    agent.save(f"C:/Users/sayan/Documents/RLProjects/Final_code_ppo_SARSA/models/ppo_agent_fetch_{episode+1}.pt")
    
    agent.save("C:/Users/sayan/Documents/RLProjects/Final_code_ppo_SARSA/models/ppo_agent_fetch.pt")
    
    plt.figure(figsize=(10, 6))
    plt.plot(episode_rewards)
    plt.title('PPO Training Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.savefig('C:/Users/sayan/Documents/RLProjects/Final_code_ppo_SARSA/plots/ppo_rewards.png')
    plt.close()
    
    plt.figure(figsize=(10, 6))
    window_size = 20
    smoothed_rewards = [np.mean(episode_rewards[max(0, i-window_size):i+1]) for i in range(len(episode_rewards))]
    plt.plot(smoothed_rewards)
    plt.title('PPO Training Rewards (Smoothed)')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.savefig('C:/Users/sayan/Documents/RLProjects/Final_code_ppo_SARSA/plots/ppo_smoothed_rewards.png')
    plt.close()
    
    env.close()
    return agent


if __name__ == "__main__":
    import argparse
    
    
    parser = argparse.ArgumentParser(description="Train PPO agent on FetchPickAndPlace environment")
    parser.add_argument("--render", action="store_true", help="Enable rendering during training")
    parser.add_argument("--episodes", type=int, default=100000, help="Number of training episodes")
    parser.add_argument("--steps", type=int, default=10000, help="Maximum steps per episode")
    args = parser.parse_args()
    
    env_name = "FetchPickAndPlace-v4"
    print(f"Training PPO agent on {env_name}")
    
    render_mode = "human" if args.render else None
    
    agent = train_ppo(env_name, num_episodes=args.episodes, steps_per_episode=args.steps, render_mode=render_mode)
    print("Training completed and model saved.")







    