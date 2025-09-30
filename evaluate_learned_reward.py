#!/usr/bin/env python3
"""
Evaluate the learned reward function from AIRL training
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import h5py
import yaml
from agents.discriminator import AIRLDiscriminator
import argparse

def load_config(config_path="config.yaml"):
    """Load training configuration"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_checkpoint(checkpoint_path, obs_dim, act_dim, device):
    """Load trained discriminator from checkpoint"""
    discriminator = AIRLDiscriminator(obs_dim, act_dim).to(device)
    
    # Load with weights_only=False to handle numpy types
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
    discriminator.eval()
    
    return discriminator

def load_expert_data(path):
    """Load expert data for comparison using same logic as training script"""
    with h5py.File(path, 'r') as f:
        all_obs = []
        all_acts = []
        
        data_group = f['data']
        demo_names = [k for k in data_group.keys() if k.startswith('demo_')]
        
        print(f"Loading {len(demo_names)} expert trajectories...")
        
        for demo_name in demo_names:
            demo = data_group[demo_name]
            
            # Get actions
            actions = demo['actions'][:]
            
            # Build flattened observations from obs components
            obs_components = []
            obs_group = demo['obs']
            
            # Define consistent order for obs components (exclude actions to avoid duplication)
            obs_keys = ['cube_positions', 'cube_orientations', 'eef_pos', 'eef_quat', 
                       'gripper_pos', 'joint_pos', 'joint_vel', 'object']
            
            for key in obs_keys:
                if key in obs_group:
                    component = obs_group[key][:]
                    if len(component.shape) == 2:  # (timesteps, features)
                        obs_components.append(component)
            
            # Concatenate all observation components
            if obs_components:
                obs = np.concatenate(obs_components, axis=1)
            else:
                raise ValueError(f"No valid observation components found in {demo_name}")
            
            print(f"  {demo_name}: {actions.shape[0]} transitions, obs_dim={obs.shape[1]}, act_dim={actions.shape[1]}")
            
            all_obs.append(obs)
            all_acts.append(actions)
        
        # Concatenate all demonstrations
        expert_obs = np.concatenate(all_obs, axis=0)
        expert_acts = np.concatenate(all_acts, axis=0)
    
    print(f"Total expert transitions: {expert_obs.shape[0]}")
    print(f"Final dimensions - obs: {expert_obs.shape[1]}, acts: {expert_acts.shape[1]}")
    
    return expert_obs, expert_acts

def evaluate_reward_function(discriminator, expert_obs, expert_acts, device, num_samples=1000):
    """Evaluate learned reward function on expert data and random data"""
    
    # Expert rewards
    expert_obs_tensor = torch.FloatTensor(expert_obs[:num_samples]).to(device)
    expert_acts_tensor = torch.FloatTensor(expert_acts[:num_samples]).to(device)
    
    with torch.no_grad():
        expert_rewards = discriminator.get_reward(expert_obs_tensor, expert_acts_tensor)
        expert_rewards = expert_rewards.cpu().numpy()
    
    # Random action rewards for comparison
    random_acts = np.random.uniform(-1, 1, (num_samples, expert_acts.shape[1]))
    random_acts_tensor = torch.FloatTensor(random_acts).to(device)
    
    with torch.no_grad():
        random_rewards = discriminator.get_reward(expert_obs_tensor[:num_samples], random_acts_tensor)
        random_rewards = random_rewards.cpu().numpy()
    
    return expert_rewards, random_rewards

def plot_reward_analysis(expert_rewards, random_rewards, save_path="reward_analysis.png"):
    """Plot reward function analysis"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Reward distributions
    axes[0, 0].hist(expert_rewards, bins=50, alpha=0.7, label='Expert Actions', color='green', density=True)
    axes[0, 0].hist(random_rewards, bins=50, alpha=0.7, label='Random Actions', color='red', density=True)
    axes[0, 0].set_xlabel('Reward Value')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].set_title('Learned Reward Distribution')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # 2. Box plot comparison
    data_to_plot = [expert_rewards, random_rewards]
    axes[0, 1].boxplot(data_to_plot, labels=['Expert Actions', 'Random Actions'])
    axes[0, 1].set_ylabel('Reward Value')
    axes[0, 1].set_title('Reward Value Comparison')
    axes[0, 1].grid(True)
    
    # 3. Reward over time/sequence
    axes[1, 0].plot(expert_rewards[:500], 'g-', alpha=0.7, label='Expert Actions')
    axes[1, 0].plot(random_rewards[:500], 'r-', alpha=0.7, label='Random Actions')
    axes[1, 0].set_xlabel('Step')
    axes[1, 0].set_ylabel('Reward Value')
    axes[1, 0].set_title('Reward Values Over Sequence')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # 4. Statistics summary
    stats_text = f"""
    EXPERT ACTIONS:
    Mean: {np.mean(expert_rewards):.3f}
    Std: {np.std(expert_rewards):.3f}
    Min: {np.min(expert_rewards):.3f}
    Max: {np.max(expert_rewards):.3f}
    
    RANDOM ACTIONS:
    Mean: {np.mean(random_rewards):.3f}
    Std: {np.std(random_rewards):.3f}
    Min: {np.min(random_rewards):.3f}
    Max: {np.max(random_rewards):.3f}
    
    DISCRIMINATION:
    Expert > Random: {np.mean(expert_rewards > random_rewards)*100:.1f}%
    Mean Difference: {np.mean(expert_rewards) - np.mean(random_rewards):.3f}
    """
    
    axes[1, 1].text(0.05, 0.95, stats_text, transform=axes[1, 1].transAxes, 
                   fontsize=10, verticalalignment='top', fontfamily='monospace')
    axes[1, 1].set_title('Reward Statistics')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    return {
        'expert_mean': np.mean(expert_rewards),
        'random_mean': np.mean(random_rewards),
        'discrimination_rate': np.mean(expert_rewards > random_rewards),
        'reward_difference': np.mean(expert_rewards) - np.mean(random_rewards)
    }

def main():
    parser = argparse.ArgumentParser(description='Evaluate learned AIRL reward function')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint file')
    parser.add_argument('--config', type=str, default='config.yaml', help='Config file path')
    parser.add_argument('--samples', type=int, default=1000, help='Number of samples to evaluate')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load expert data
    expert_obs, expert_acts = load_expert_data(config['training']['expert_data']['path'])
    obs_dim, act_dim = expert_obs.shape[1], expert_acts.shape[1]
    
    print(f"Expert data loaded: {len(expert_obs)} transitions")
    print(f"Observation dim: {obs_dim}, Action dim: {act_dim}")
    
    # Load trained discriminator
    discriminator = load_checkpoint(args.checkpoint, obs_dim, act_dim, device)
    print(f"Loaded checkpoint: {args.checkpoint}")
    
    # Evaluate reward function
    expert_rewards, random_rewards = evaluate_reward_function(
        discriminator, expert_obs, expert_acts, device, args.samples
    )
    
    # Plot and analyze
    stats = plot_reward_analysis(expert_rewards, random_rewards)
    
    print("\n" + "="*50)
    print("REWARD FUNCTION EVALUATION RESULTS")
    print("="*50)
    print(f"Expert actions mean reward: {stats['expert_mean']:.3f}")
    print(f"Random actions mean reward: {stats['random_mean']:.3f}")
    print(f"Reward difference: {stats['reward_difference']:.3f}")
    print(f"Discrimination rate: {stats['discrimination_rate']*100:.1f}%")
    
    if stats['discrimination_rate'] > 0.7:
        print("✅ GOOD: Reward function discriminates well between expert and random actions")
    elif stats['discrimination_rate'] > 0.5:
        print("⚠️  OK: Reward function shows some discrimination")
    else:
        print("❌ POOR: Reward function needs more training")

if __name__ == "__main__":
    main()
