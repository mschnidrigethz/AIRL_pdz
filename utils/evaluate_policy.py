"""
Evaluate trained AIRL policy
Load a trained model and evaluate its performance
"""

import torch
import numpy as np
import argparse
import yaml
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm

# Import local modules
import sys
sys.path.append(str(Path(__file__).parent.parent))

from envs.isaac_wrapper import make_env
from agents.policy import Policy
from agents.discriminator import AIRLDiscriminator

def load_model(checkpoint_path, obs_dim, act_dim, config):
    """Load trained model from checkpoint"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create models
    net_cfg = config['network']
    policy = Policy(obs_dim, act_dim, hidden_size=net_cfg['policy']['hidden_sizes'][0]).to(device)
    discriminator = AIRLDiscriminator(
        obs_dim, act_dim,
        hidden_size=net_cfg['discriminator']['hidden_sizes'][0],
        gamma=net_cfg['discriminator']['gamma']
    ).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    policy.load_state_dict(checkpoint['policy_state_dict'])
    discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
    
    policy.eval()
    discriminator.eval()
    
    return policy, discriminator, device

def evaluate_policy(policy, discriminator, env, num_episodes=10, max_steps=1000, render=False, device=None):
    """Evaluate policy performance"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    episode_rewards = []
    episode_lengths = []
    success_count = 0
    
    print(f"🚀 Evaluating policy for {num_episodes} episodes...")
    
    for episode in tqdm(range(num_episodes)):
        obs, _ = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
            
        episode_reward = 0
        episode_length = 0
        done = False
        
        for step in range(max_steps):
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
            
            with torch.no_grad():
                # Get deterministic action (use mean, not sample)
                mean, std, _ = policy(obs_tensor)
                action = mean.squeeze().cpu().numpy()
            
            obs, env_reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Use AIRL reward for evaluation
            with torch.no_grad():
                obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
                action_tensor = torch.tensor(action, dtype=torch.float32).unsqueeze(0).to(device)
                airl_reward = discriminator.get_reward(obs_tensor, action_tensor).item()
            
            episode_reward += airl_reward
            episode_length += 1
            
            # Check for success
            if info.get('success', False):
                success_count += 1
                break
            
            if render and hasattr(env, 'render'):
                env.render()
                
            if done:
                break
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
    
    # Calculate metrics
    metrics = {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_length': np.mean(episode_lengths),
        'std_length': np.std(episode_lengths),
        'success_rate': success_count / num_episodes,
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths
    }
    
    return metrics

def plot_evaluation_results(metrics, save_path=None):
    """Plot evaluation results"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Episode rewards
    axes[0, 0].plot(metrics['episode_rewards'])
    axes[0, 0].axhline(y=metrics['mean_reward'], color='r', linestyle='--', 
                      label=f'Mean: {metrics["mean_reward"]:.2f}')
    axes[0, 0].set_title('Episode Rewards')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Episode lengths
    axes[0, 1].plot(metrics['episode_lengths'])
    axes[0, 1].axhline(y=metrics['mean_length'], color='r', linestyle='--',
                      label=f'Mean: {metrics["mean_length"]:.1f}')
    axes[0, 1].set_title('Episode Lengths')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Steps')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Reward histogram
    axes[1, 0].hist(metrics['episode_rewards'], bins=20, alpha=0.7, edgecolor='black')
    axes[1, 0].axvline(x=metrics['mean_reward'], color='r', linestyle='--',
                      label=f'Mean: {metrics["mean_reward"]:.2f}')
    axes[1, 0].set_title('Reward Distribution')
    axes[1, 0].set_xlabel('Reward')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].legend()
    
    # Success rate and metrics summary
    axes[1, 1].text(0.1, 0.8, f'Success Rate: {metrics["success_rate"]:.1%}', fontsize=14, weight='bold')
    axes[1, 1].text(0.1, 0.6, f'Mean Reward: {metrics["mean_reward"]:.2f} ± {metrics["std_reward"]:.2f}', fontsize=12)
    axes[1, 1].text(0.1, 0.4, f'Mean Length: {metrics["mean_length"]:.1f} ± {metrics["std_length"]:.1f}', fontsize=12)
    axes[1, 1].text(0.1, 0.2, f'Episodes: {len(metrics["episode_rewards"])}', fontsize=12)
    axes[1, 1].set_xlim(0, 1)
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].axis('off')
    axes[1, 1].set_title('Evaluation Summary')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Evaluation plot saved to: {save_path}")
    
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Evaluate trained AIRL policy')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='config.yaml', help='Config file path')
    parser.add_argument('--episodes', type=int, default=10, help='Number of evaluation episodes')
    parser.add_argument('--render', action='store_true', help='Render environment during evaluation')
    parser.add_argument('--plot', action='store_true', help='Show evaluation plots')
    parser.add_argument('--save-plot', type=str, default=None, help='Save plot to file')
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"📋 Configuration loaded from: {args.config}")
    print(f"🎯 Evaluating checkpoint: {args.checkpoint}")
    
    # Create environment
    env_cfg = config['env']
    env = make_env(env_cfg)
    
    # Get environment dimensions
    if hasattr(env.observation_space, 'shape'):
        obs_dim = env.observation_space.shape[0]
    else:
        obs_dim = env.obs_dim
        
    if hasattr(env.action_space, 'shape'):
        act_dim = env.action_space.shape[0]
    else:
        act_dim = env.action_dim
    
    print(f"🏗️  Environment: obs_dim={obs_dim}, act_dim={act_dim}")
    
    # Load trained model
    print(f"🔄 Loading trained model...")
    policy, discriminator, device = load_model(args.checkpoint, obs_dim, act_dim, config)
    print(f"✅ Model loaded on device: {device}")
    
    # Evaluate policy
    metrics = evaluate_policy(
        policy, discriminator, env, 
        num_episodes=args.episodes,
        max_steps=config['env']['max_episode_length'],
        render=args.render,
        device=device
    )
    
    # Print results
    print(f"\n🎯 Evaluation Results:")
    print(f"   Mean Reward: {metrics['mean_reward']:.2f} ± {metrics['std_reward']:.2f}")
    print(f"   Mean Length: {metrics['mean_length']:.1f} ± {metrics['std_length']:.1f}")
    print(f"   Success Rate: {metrics['success_rate']:.1%}")
    print(f"   Best Reward: {max(metrics['episode_rewards']):.2f}")
    print(f"   Worst Reward: {min(metrics['episode_rewards']):.2f}")
    
    # Plot results
    if args.plot or args.save_plot:
        plot_evaluation_results(metrics, save_path=args.save_plot)
    
    env.close()
    print(f"✅ Evaluation completed!")

if __name__ == '__main__':
    main()
