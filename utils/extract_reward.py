"""
Extract Learned Reward Function from AIRL Discriminator
Use the learned reward function for standard RL training
"""

import torch
import numpy as np
import argparse
import yaml
import matplotlib.pyplot as plt
from pathlib import Path
import pickle

# Import local modules
import sys
sys.path.append(str(Path(__file__).parent.parent))

from envs.franka_wrapper import make_env
from agents.discriminator import AIRLDiscriminator

class LearnedRewardFunction:
    """Wrapper for the learned AIRL reward function"""
    
    def __init__(self, discriminator, device=None):
        self.discriminator = discriminator.eval()
        self.device = device or torch.device("cpu")
        
    def __call__(self, obs, action):
        """
        Compute reward for given state-action pair
        
        Args:
            obs: observation (numpy array or tensor)
            action: action (numpy array or tensor)
            
        Returns:
            reward: scalar reward value
        """
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float32)
        if isinstance(action, np.ndarray):
            action = torch.tensor(action, dtype=torch.float32)
            
        # Ensure batch dimension
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        if action.dim() == 1:
            action = action.unsqueeze(0)
            
        obs = obs.to(self.device)
        action = action.to(self.device)
        
        with torch.no_grad():
            reward = self.discriminator.get_reward(obs, action)
            
        # Return scalar if single sample
        if reward.numel() == 1:
            return reward.item()
        else:
            return reward.cpu().numpy()
    
    def compute_batch_rewards(self, obs_batch, action_batch):
        """Compute rewards for a batch of state-action pairs"""
        obs_tensor = torch.tensor(obs_batch, dtype=torch.float32).to(self.device)
        action_tensor = torch.tensor(action_batch, dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            rewards = self.discriminator.get_reward(obs_tensor, action_tensor)
            
        return rewards.cpu().numpy()
    
    def save(self, filepath):
        """Save the learned reward function"""
        save_data = {
            'discriminator_state_dict': self.discriminator.state_dict(),
            'device': str(self.device)
        }
        torch.save(save_data, filepath)
        print(f"✅ Learned reward function saved to: {filepath}")
    
    @classmethod
    def load(cls, filepath, obs_dim, act_dim, config=None):
        """Load a saved reward function"""
        save_data = torch.load(filepath, map_location='cpu')
        
        # Create discriminator architecture
        if config:
            hidden_size = config['network']['discriminator']['hidden_sizes'][0]
            gamma = config['network']['discriminator']['gamma']
        else:
            hidden_size = 256
            gamma = 0.99
            
        discriminator = AIRLDiscriminator(obs_dim, act_dim, hidden_size, gamma)
        discriminator.load_state_dict(save_data['discriminator_state_dict'])
        
        device = torch.device(save_data.get('device', 'cpu'))
        discriminator = discriminator.to(device)
        
        return cls(discriminator, device)

def extract_reward_function(checkpoint_path, config_path, output_path=None):
    """Extract and save the learned reward function from AIRL checkpoint"""
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Get dimensions from config or checkpoint
    if 'config' in checkpoint and 'env' in checkpoint['config']:
        # Try to get dimensions from stored config
        env_config = checkpoint['config']['env']
        obs_dim = 24  # Default fallback
        act_dim = 9   # Default fallback
    else:
        # Create environment to get dimensions
        env = make_env(config['env'])
        if hasattr(env.observation_space, 'shape'):
            obs_dim = env.observation_space.shape[0]
            act_dim = env.action_space.shape[0]
        else:
            obs_dim = env.obs_dim
            act_dim = env.action_dim
        env.close()
    
    # Create discriminator with same architecture
    net_config = config['network']['discriminator']
    discriminator = AIRLDiscriminator(
        obs_dim, act_dim, 
        hidden_size=net_config['hidden_sizes'][0],
        gamma=net_config['gamma']
    )
    
    # Load trained weights
    discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
    
    # Create reward function wrapper
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reward_function = LearnedRewardFunction(discriminator, device)
    
    # Save the reward function
    if output_path is None:
        output_path = Path(checkpoint_path).parent / "learned_reward_function.pt"
    
    reward_function.save(output_path)
    
    return reward_function, output_path

def visualize_reward_landscape(reward_function, env, num_samples=1000, save_path=None):
    """Visualize the learned reward landscape"""
    
    print(f"🎨 Generating reward landscape visualization...")
    
    # Sample random state-action pairs
    obs_samples = []
    action_samples = []
    rewards = []
    
    for _ in range(num_samples):
        # Sample random observation
        if hasattr(env.observation_space, 'sample'):
            obs = env.observation_space.sample()
        else:
            obs = np.random.randn(env.obs_dim)
        
        # Sample random action
        if hasattr(env.action_space, 'sample'):
            action = env.action_space.sample()
        else:
            action = np.random.uniform(-1, 1, env.action_dim)
        
        reward = reward_function(obs, action)
        
        obs_samples.append(obs)
        action_samples.append(action)
        rewards.append(reward)
    
    rewards = np.array(rewards)
    obs_samples = np.array(obs_samples)
    action_samples = np.array(action_samples)
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Reward distribution
    axes[0, 0].hist(rewards, bins=50, alpha=0.7, edgecolor='black')
    axes[0, 0].set_title('Learned Reward Distribution')
    axes[0, 0].set_xlabel('Reward')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].axvline(rewards.mean(), color='red', linestyle='--', label=f'Mean: {rewards.mean():.3f}')
    axes[0, 0].legend()
    
    # Reward vs first observation dimension
    axes[0, 1].scatter(obs_samples[:, 0], rewards, alpha=0.5, s=1)
    axes[0, 1].set_title('Reward vs First Observation Dim')
    axes[0, 1].set_xlabel('Observation[0]')
    axes[0, 1].set_ylabel('Reward')
    
    # Reward vs first action dimension
    axes[1, 0].scatter(action_samples[:, 0], rewards, alpha=0.5, s=1)
    axes[1, 0].set_title('Reward vs First Action Dim')
    axes[1, 0].set_xlabel('Action[0]')
    axes[1, 0].set_ylabel('Reward')
    
    # 2D action space visualization (first 2 action dims)
    if action_samples.shape[1] >= 2:
        scatter = axes[1, 1].scatter(
            action_samples[:, 0], action_samples[:, 1], 
            c=rewards, cmap='viridis', alpha=0.7, s=2
        )
        axes[1, 1].set_title('Reward Landscape (Action Space)')
        axes[1, 1].set_xlabel('Action[0]')
        axes[1, 1].set_ylabel('Action[1]')
        plt.colorbar(scatter, ax=axes[1, 1])
    else:
        axes[1, 1].text(0.5, 0.5, 'Need ≥2 action dims\nfor 2D visualization', 
                       ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Action Space Visualization')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Reward landscape saved to: {save_path}")
    
    plt.show()
    
    # Print statistics
    print(f"🎯 Reward Statistics:")
    print(f"   Mean: {rewards.mean():.4f}")
    print(f"   Std:  {rewards.std():.4f}")
    print(f"   Min:  {rewards.min():.4f}")
    print(f"   Max:  {rewards.max():.4f}")
    
    return rewards

def test_reward_function(reward_function, env, num_episodes=5):
    """Test the extracted reward function in environment rollouts"""
    
    print(f"🧪 Testing reward function on {num_episodes} episodes...")
    
    episode_rewards = []
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
            
        episode_reward = 0
        steps = 0
        
        for step in range(1000):  # Max steps
            # Random policy for testing
            if hasattr(env.action_space, 'sample'):
                action = env.action_space.sample()
            else:
                action = np.random.uniform(-1, 1, env.action_dim)
            
            # Get reward from learned function
            learned_reward = reward_function(obs, action)
            episode_reward += learned_reward
            
            # Environment step
            obs, env_reward, terminated, truncated, _ = env.step(action)
            steps += 1
            
            if terminated or truncated:
                break
        
        episode_rewards.append(episode_reward)
        print(f"   Episode {episode+1}: {episode_reward:.3f} (steps: {steps})")
    
    avg_reward = np.mean(episode_rewards)
    print(f"📊 Average episode reward: {avg_reward:.3f} ± {np.std(episode_rewards):.3f}")
    
    return episode_rewards

def main():
    parser = argparse.ArgumentParser(description='Extract learned reward function from AIRL')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to AIRL checkpoint')
    parser.add_argument('--config', type=str, default='config.yaml', help='Config file path')
    parser.add_argument('--output', type=str, default=None, help='Output path for reward function')
    parser.add_argument('--visualize', action='store_true', help='Visualize reward landscape')
    parser.add_argument('--test', action='store_true', help='Test reward function')
    parser.add_argument('--save-plot', type=str, default=None, help='Save visualization plot')
    
    args = parser.parse_args()
    
    print(f"🚀 Extracting reward function from: {args.checkpoint}")
    
    # Extract reward function
    reward_function, output_path = extract_reward_function(
        args.checkpoint, args.config, args.output
    )
    
    print(f"✅ Reward function extracted successfully!")
    print(f"💾 Saved to: {output_path}")
    
    # Load environment for testing/visualization
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    env = make_env(config['env'])
    
    # Test the reward function
    if args.test:
        test_reward_function(reward_function, env)
    
    # Visualize reward landscape
    if args.visualize:
        visualize_reward_landscape(reward_function, env, save_path=args.save_plot)
    
    env.close()
    
    print(f"\n🎯 Usage in RL Training:")
    print(f"```python")
    print(f"from utils.extract_reward import LearnedRewardFunction")
    print(f"")
    print(f"# Load the learned reward function")
    print(f"reward_fn = LearnedRewardFunction.load('{output_path}', obs_dim, act_dim)")
    print(f"")
    print(f"# Use in your RL training loop")
    print(f"reward = reward_fn(observation, action)")
    print(f"```")

if __name__ == '__main__':
    main()
