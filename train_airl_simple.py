"""
Simple AIRL Training - No Isaac Sim dependency
Uses your expert data directly for imitation learning
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import gymnasium as gym
import os
import h5py
from datetime import datetime

class SimpleEnv(gym.Env):
    """Simple environment that mimics Isaac Lab structure but doesn't need USD assets"""
    
    def __init__(self):
        # Match your expert data dimensions
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(74,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(8,), dtype=np.float32
        )
        self.reset()
    
    def reset(self, seed=None, options=None):
        # Simple initial state
        self.state = np.zeros(74, dtype=np.float32)
        # Add some realistic values
        self.state[0:7] = np.random.normal(0, 0.1, 7)  # Joint positions
        self.state[14:17] = [0.5, 0.0, 1.0]  # Object position
        self.state[17:20] = [0.5, 0.0, 1.15]  # Target position
        return self.state, {}
    
    def step(self, action):
        # Simple dynamics - just move towards expert behavior
        self.state[0:7] += action[0:7] * 0.01  # Update joints
        
        # Simple reward - encourage staying in reasonable ranges
        reward = -np.sum(np.abs(action)) * 0.1
        
        # Check if object is "lifted" (simple heuristic)
        object_height = self.state[16]  # Z position of object
        if object_height > 1.1:
            reward += 10.0
        
        terminated = False
        truncated = False
        
        return self.state, reward, terminated, truncated, {}

class Discriminator(nn.Module):
    """AIRL Discriminator Network"""
    
    def __init__(self, obs_dim, act_dim, hidden_dim=256):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(), 
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=1)
        return self.net(x)

class Policy(nn.Module):
    """Simple Policy Network"""
    
    def __init__(self, obs_dim, act_dim, hidden_dim=256):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, act_dim),
            nn.Tanh()  # Actions in [-1, 1]
        )
    
    def forward(self, obs):
        return self.net(obs)

def load_expert_data(data_dir):
    """Load your expert trajectories"""
    expert_obs = []
    expert_acts = []
    
    # Load your actual data file
    file_path = os.path.join(data_dir, "merged_real_dataset_1.1to1.6.hdf5")
    
    if os.path.exists(file_path):
        with h5py.File(file_path, 'r') as f:
            print(f"📂 Loading data from: {file_path}")
            
            # Your data structure: /data/demo_X/obs/<components> and /data/demo_X/actions
            data_group = f['data']
            
            for demo_key in data_group.keys():
                if demo_key.startswith('demo_'):
                    demo_data = data_group[demo_key]
                    obs_group = demo_data['obs']
                    acts = demo_data['actions'][:]
                    
                    # Concatenate all observation components into flat vector
                    obs_components = []
                    obs_components.append(obs_group['joint_pos'][:])      # 7 dims
                    obs_components.append(obs_group['joint_vel'][:])      # 7 dims  
                    obs_components.append(obs_group['cube_positions'][:]) # 3 dims
                    obs_components.append(obs_group['cube_orientations'][:]) # 4 dims
                    obs_components.append(obs_group['eef_pos'][:])        # 3 dims
                    obs_components.append(obs_group['eef_quat'][:])       # 4 dims
                    obs_components.append(obs_group['gripper_pos'][:])    # 7 dims
                    obs_components.append(obs_group['object'][:])         # 39 dims
                    
                    # Flatten to match expected 74-dim observation
                    obs = np.concatenate(obs_components, axis=1)
                    
                    expert_obs.append(obs)
                    expert_acts.append(acts)
                    print(f"  {demo_key}: {len(obs)} transitions, obs_dim={obs.shape[1]}, act_dim={acts.shape[1]}")
    else:
        raise FileNotFoundError(f"Expert data file not found: {file_path}")
    
    expert_obs = np.concatenate(expert_obs, axis=0)
    expert_acts = np.concatenate(expert_acts, axis=0)
    
    print(f"Total expert data: {len(expert_obs)} transitions")
    print(f"Observation shape: {expert_obs.shape}")
    print(f"Action shape: {expert_acts.shape}")
    
    return expert_obs, expert_acts

def train_airl():
    """Main AIRL Training Loop"""
    print("🚀 Starting Simple AIRL Training")
    
    # Load expert data
    data_dir = "/home/chris/IsaacLab_mik/Projects/airl_franka/data"
    expert_obs, expert_acts = load_expert_data(data_dir)
    
    # Convert to tensors
    expert_obs = torch.FloatTensor(expert_obs)
    expert_acts = torch.FloatTensor(expert_acts)
    
    # Create environment
    env = SimpleEnv()
    
    # Create networks
    discriminator = Discriminator(74, 8)
    policy = Policy(74, 8)
    
    # Optimizers
    d_optimizer = optim.Adam(discriminator.parameters(), lr=3e-4)
    p_optimizer = optim.Adam(policy.parameters(), lr=3e-4)
    
    # Tensorboard logging
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(f"logs/tensorboard/{timestamp}")
    
    # Training loop
    batch_size = 256
    num_episodes = 1000
    
    for episode in range(num_episodes):
        # Generate policy rollout
        obs, _ = env.reset()
        policy_obs = []
        policy_acts = []
        episode_reward = 0
        
        for step in range(200):  # Episode length
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            with torch.no_grad():
                action = policy(obs_tensor).squeeze(0).numpy()
            
            policy_obs.append(obs.copy())
            policy_acts.append(action.copy())
            
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            
            if terminated or truncated:
                break
        
        # Convert to tensors
        policy_obs = torch.FloatTensor(policy_obs)
        policy_acts = torch.FloatTensor(policy_acts)
        
        # Sample expert batch
        expert_indices = torch.randint(0, len(expert_obs), (min(batch_size, len(policy_obs)),))
        expert_batch_obs = expert_obs[expert_indices]
        expert_batch_acts = expert_acts[expert_indices]
        
        # Sample policy batch
        policy_indices = torch.randint(0, len(policy_obs), (min(batch_size, len(policy_obs)),))
        policy_batch_obs = policy_obs[policy_indices]
        policy_batch_acts = policy_acts[policy_indices]
        
        # Train discriminator
        d_optimizer.zero_grad()
        
        # Expert data should get high scores (close to 1)
        expert_scores = discriminator(expert_batch_obs, expert_batch_acts)
        expert_loss = nn.BCEWithLogitsLoss()(expert_scores, torch.ones_like(expert_scores))
        
        # Policy data should get low scores (close to 0)
        policy_scores = discriminator(policy_batch_obs, policy_batch_acts)
        policy_loss = nn.BCEWithLogitsLoss()(policy_scores, torch.zeros_like(policy_scores))
        
        d_loss = expert_loss + policy_loss
        d_loss.backward()
        d_optimizer.step()
        
        # Train policy (trying to fool discriminator)
        p_optimizer.zero_grad()
        
        # Policy wants discriminator to think its actions are expert actions
        policy_pred_acts = policy(policy_batch_obs)
        fool_scores = discriminator(policy_batch_obs, policy_pred_acts)
        p_loss = nn.BCEWithLogitsLoss()(fool_scores, torch.ones_like(fool_scores))
        
        p_loss.backward()
        p_optimizer.step()
        
        # Logging
        if episode % 10 == 0:
            print(f"Episode {episode}: Reward={episode_reward:.2f}, D_Loss={d_loss.item():.4f}, P_Loss={p_loss.item():.4f}")
            
            writer.add_scalar("Episode/Reward", episode_reward, episode)
            writer.add_scalar("Loss/Discriminator", d_loss.item(), episode)
            writer.add_scalar("Loss/Policy", p_loss.item(), episode)
        
        # Save checkpoints
        if episode % 100 == 0:
            os.makedirs("logs/checkpoints", exist_ok=True)
            torch.save({
                'policy': policy.state_dict(),
                'discriminator': discriminator.state_dict(),
                'episode': episode
            }, f"logs/checkpoints/checkpoint_episode_{episode}.pt")
            print(f"📁 Checkpoint saved: episode {episode}")
    
    writer.close()
    print("✅ Training completed!")

if __name__ == "__main__":
    train_airl()