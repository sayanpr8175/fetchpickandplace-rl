import numpy as np
import gymnasium as gym
import gymnasium_robotics
import torch
import time
import argparse
from collections import deque


class Actor(torch.nn.Module):
    def __init__(self, state_dim, action_dim, max_action, hidden_dim=256):
        super(Actor, self).__init__()
        
        self.network = torch.nn.Sequential(
            torch.nn.Linear(state_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, action_dim),
            torch.nn.Tanh()
        )
        
        self.max_action = max_action
        
    def forward(self, state):
        return self.max_action * self.network(state)

class Critic(torch.nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()
        
        self.network = torch.nn.Sequential(
            torch.nn.Linear(state_dim + action_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.network(x)

class PickAndPlaceDemo:
    def __init__(self, model_path, device='cpu'):
        self.device = device
        self.env = gym.make("FetchPickAndPlace-v4", render_mode='human')
        
        self.obs_dim = self.env.observation_space['observation'].shape[0]
        self.goal_dim = self.env.observation_space['desired_goal'].shape[0]
        self.state_dim = self.obs_dim + self.goal_dim
        self.action_dim = self.env.action_space.shape[0]
        self.max_action = self.env.action_space.high[0]
        
        self.actor = Actor(self.state_dim, self.action_dim, self.max_action).to(device)
        
        self.load_model(model_path)
    
    def load_model(self, model_path):
        print(f"Loading model from {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.actor.eval()
        print("Model loaded successfully")
    
    
    def get_state(self, obs):
        state = np.concatenate([obs['observation'], obs['desired_goal']])
        return state
    

    
    def select_action(self, state):
        state = torch.FloatTensor(state).to(self.device)
        with torch.no_grad():
            action = self.actor(state).cpu().data.numpy()
        return action
    
    def run_demo(self, num_episodes=5, max_steps=50, delay=0.01):
        success_count = 0
        
        for episode in range(num_episodes):
            print(f"\nRunning episode {episode+1}/{num_episodes}")
            obs, _ = self.env.reset()
            
            for step in range(max_steps):
                state = self.get_state(obs)
                action = self.select_action(state)
                
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                
                self.env.render()
                time.sleep(delay)
                
                if 'is_success' in info and info['is_success']:
                    print(f"Success! Episode completed in {step+1} steps.")
                    success_count += 1
                    break
                
                obs = next_obs
                
                if done:
                    break
            
            if 'is_success' not in info or not info['is_success']:
                print(f"Episode failed after {max_steps} steps.")
        
        success_rate = success_count / num_episodes
        print(f"\nDemo completed. Success rate: {success_rate*100:.2f}% ({success_count}/{num_episodes})")
        
    def close(self):
        self.env.close()



def main():
    parser = argparse.ArgumentParser(description='Run a demo of the trained PickAndPlace model')
    parser.add_argument('--model', type=str, default='ddpg_her_fetch_model.pt', help='Path to the model file')
    parser.add_argument('--episodes', type=int, default=5, help='Number of episodes to run')
    parser.add_argument('--delay', type=float, default=0.01, help='Delay between steps (in seconds)')
    parser.add_argument('--gpu', action='store_true', help='Use GPU if available')
    
    args = parser.parse_args()
    
    if args.gpu and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    
    demo = PickAndPlaceDemo(args.model, device)
    
    try:
        demo.run_demo(num_episodes=args.episodes, delay=args.delay)
    finally:
        demo.close()

if __name__ == "__main__":
    main()



    