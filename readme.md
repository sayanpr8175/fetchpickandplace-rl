# 🤖 FetchPickAndPlace Reinforcement Learning

A comprehensive implementation of various reinforcement learning algorithms for the FetchPickAndPlace-v4 environment from Gymnasium Robotics.

<img src="./RepoAssets/fetchpickandplace.gif" alt="Project Banner" width="500">

## Project Team members

**[Sayan Pramanik](https://www.linkedin.com/in/sayan-pramanik-ecs/)**

**[Aswin Chander Aravind Kumar](https://www.linkedin.com/in/aswin-chander-aravind-kumar-647b93201/)**

## 📋 Overview

This project implements and compares four reinforcement learning algorithms on the challenging FetchPickAndPlace-v4 robotic manipulation task:

- **SARSA**: A classic on-policy TD control algorithm
- **Q-Learning**: An off-policy TD control algorithm
- **PPO** (Proximal Policy Optimization): A policy gradient method
- **DDPG-HER** (Deep Deterministic Policy Gradient with Hindsight Experience Replay): A state-of-the-art approach for sparse reward problems


## 🔧 Environment Setup

### Requirements

This project requires Python 3.9+ and several dependencies. Set up your environment by:

```bash
# Clone the repository
git clone https://github.com/sayanpr8175/fetchpickandplace-rl.git
cd fetchpickandplace-rl

# Create a virtual environment (optional but recommended)
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Key Dependencies

- gymnasium
- gymnasium-robotics
- torch
- numpy
- matplotlib
- tqdm

## 🚀 Training Models

### SARSA

Train a SARSA agent with discretized state and action spaces:

```bash
python SARSA_Training.py
```

Key parameters:
- `num_episodes`: Number of training episodes (default: 100000)
- `render_freq`: Frequency for rendering episodes (default: 1000)

### PPO (Proximal Policy Optimization)

Train a PPO agent with neural network policy:

```bash
python PPO_training.py [--render] [--episodes NUM_EPISODES] [--steps MAX_STEPS]
```

Options:
- `--render`: Enable rendering during training
- `--episodes`: Number of training episodes (default: 100000)
- `--steps`: Maximum steps per episode (default: 10000)

### Q-Learning

Train a Q-Learning agent with continuous state and action spaces:

```bash
python Qlearning.py
```

### DDPG-HER

Train a Deep Deterministic Policy Gradient agent with Hindsight Experience Replay:

```bash
python DDPG_her_training.py
```

## 🎮 Using Trained Models

### Testing SARSA & PPO Models

To test trained SARSA or PPO models:

```bash
python test_sarsa_ppo_models.py --algorithm [sarsa|ppo] [--model PATH_TO_MODEL] [--episodes NUM_EPISODES]
```

Options:
- `--algorithm`: Choose which algorithm to demonstrate (sarsa or ppo)
- `--model`: Path to the model file (defaults to the standard save location)
- `--episodes`: Number of episodes to run (default: 5)

### Testing DDPG-HER Models

To test a trained DDPG-HER model:

```bash
python TestRun_ddpg_trainedModel.py [--model PATH_TO_MODEL] [--episodes NUM_EPISODES] [--delay DELAY] [--gpu]
```

Options:
- `--model`: Path to the model file (default: ddpg_her_fetch_model.pt)
- `--episodes`: Number of episodes to run (default: 5)
- `--delay`: Delay between steps in seconds (default: 0.01)
- `--gpu`: Use GPU if available

## 📊 Results

### Training Performance

<p align="center">
  <img src="./plots/sarsa_smoothed_rewards.png" alt="SARSA Training Curve" width="60%">
</p>
<p align="center"><em>SARSA training reward curve</em></p>

<p align="center">
  <img src="./plots/ppo_smoothed_rewards.png" alt="PPO Training Curve" width="60%">
</p>
<p align="center"><em>PPO training reward curve</em></p>

<p align="center">
  <img src="./plots/fetch_qlearning_rewards.png" alt="Q-Learning Training Curve" width="60%">
</p>
<p align="center"><em>Q-Learning training reward curve</em></p>

<p align="center">
  <img src="./plots/fetch_ddpg_her_results.png" alt="DDPG-HER Training Curve" width="60%">
</p>
<p align="center"><em>DDPG-HER training reward curve</em></p>


## 🌟 Key Features

- State representation optimization for robotic control
- Custom discretization for tabular methods (SARSA)
- Neural network architectures designed for continuous control
- Implementation of Hindsight Experience Replay for sparse-reward problems
- Visualization tools for training progress and agent behavior

## 📝 Project Structure

```
.
├── DDPG_her_training.py        # DDPG with HER implementation
├── FetchPickandPlaceV4_sim_test.py  # Rule-based test implementation
├── PPO_training.py             # PPO implementation
├── Qlearning.py                # Q-Learning implementation
├── SARSA_Training.py           # SARSA implementation
├── TestRun_ddpg_trainedModel.py  # Test script for DDPG model
├── test_sarsa_ppo_models.py    # Test script for SARSA and PPO models
├── requirements.txt            # Project dependencies
├── models/                     # Saved model files
└── plots/                      # Training visualizations
```

## 🔗 References

- [Gymnasium Robotics Documentation](https://robotics.farama.org/)
- [DDPG Paper](https://arxiv.org/abs/1509.02971)
- [Hindsight Experience Replay Paper](https://arxiv.org/abs/1707.01495)
- [PPO Paper](https://arxiv.org/abs/1707.06347)

## 📄 License

[MIT License](LICENSE)

---

*This project was developed as part of my RL class at Northeastern during my 2nd semester to explore reinforcement learning approaches to robotic manipulation tasks.*
