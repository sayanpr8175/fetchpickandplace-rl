import gymnasium as gym
import gymnasium_robotics
import numpy as np
import torch
import argparse
import time
import pickle

from SARSA_Training import SARSAAgent, extract_features as sarsa_extract_features
from PPO_training import PPOAgent, extract_features as ppo_extract_features


def demonstrate_sarsa(env_name, model_path, num_episodes=5):

    env = gym.make(env_name, render_mode="human")
    
    observation, _ = env.reset()
    state_features = sarsa_extract_features(observation)
    state_dim = len(state_features)
    action_dim = env.action_space.shape[0]
    action_low = env.action_space.low
    action_high = env.action_space.high
    
    agent = SARSAAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        action_bounds=(action_low, action_high),
        discretization=5,
        learning_rate=0.1,
        gamma=0.99,
        epsilon=0.01  
    )
    
    print(f"Loading SARSA model from {model_path}")
    agent.load(model_path)
    
    for episode in range(num_episodes):
        observation, _ = env.reset()
        state = sarsa_extract_features(observation)
        
        total_reward = 0
        done = False
        truncated = False
        step = 0
        
        print(f"Starting episode {episode+1}/{num_episodes}")
        
        while not (done or truncated):
            agent.epsilon = 0.0
            action, _ = agent.select_action(state)
            
            observation_new, reward, done, truncated, _ = env.step(action)
            
            state = sarsa_extract_features(observation_new)
            
            total_reward += reward
            step += 1
            
            time.sleep(0.01)
        
        print(f"Episode {episode+1} completed with reward {total_reward:.2f} in {step} steps")
    
    env.close()


def demonstrate_ppo(env_name, model_path, num_episodes=5):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    env = gym.make(env_name, render_mode="human")
    
    observation, _ = env.reset()
    state_features = ppo_extract_features(observation)
    state_dim = len(state_features)
    action_dim = env.action_space.shape[0]
    action_low = env.action_space.low
    action_high = env.action_space.high
    
    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        action_bounds=(action_low, action_high),
        hidden_dim=128
    )
    

    print(f"Loading PPO model from {model_path}")
    agent.load(model_path)
    

    for episode in range(num_episodes):
        observation, _ = env.reset()
        state = ppo_extract_features(observation)
        
        total_reward = 0
        done = False
        truncated = False
        step = 0
        
        print(f"Starting episode {episode+1}/{num_episodes}")
        
        while not (done or truncated):
            action = agent.policy.get_action(state, deterministic=True)
            
            action = agent.normalize_action(action)
            
            
            observation_new, reward, done, truncated, _ = env.step(action)
            
            state = ppo_extract_features(observation_new)
            
            total_reward += reward
            step += 1
            
            time.sleep(0.01)
        
        print(f"Episode {episode+1} completed with reward {total_reward:.2f} in {step} steps")
    
    env.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Demonstrate trained RL agents")
    parser.add_argument("--algorithm", type=str, choices=["sarsa", "ppo"], default="ppo",
                        help="Algorithm to demonstrate (sarsa or ppo)")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to model file (if not specified, uses default path)")
    parser.add_argument("--episodes", type=int, default=5,
                        help="Number of episodes to run")
    args = parser.parse_args()
    
    env_name = "FetchPickAndPlace-v4"
    
    if args.model is None:
        if args.algorithm == "sarsa":
            args.model = "C:/Users/sayan/Documents/RLProjects/Final_code_ppo_SARSA/models/sarsa_agent_fetch.pkl"
        #ppo
        else:
            args.model = "C:/Users/sayan/Documents/RLProjects/Final_code_ppo_SARSA/models/ppo_agent_fetch.pt"
    
    
    if args.algorithm == "sarsa":
        demonstrate_sarsa(env_name, args.model, num_episodes=args.episodes)
    else:  # ppo
        demonstrate_ppo(env_name, args.model, num_episodes=args.episodes)









        