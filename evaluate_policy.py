#!/usr/bin/env python3
"""
Evaluate trained AIRL policy performance
"""

import torch
import numpy as np
import yaml
from pathlib import Path
from agents.policy import Policy
from agents.discriminator import AIRLDiscriminator
from envs.franka_wrapper import make_env
import argparse

def load_checkpoint(checkpoint_path, obs_dim, act_dim, device):
    """Load trained policy and discriminator"""
    policy = Policy(obs_dim, act_dim).to(device)
    discriminator = AIRLDiscriminator(obs_dim, act_dim).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    policy.load_state_dict(checkpoint['policy_state_dict'])
    discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
    
    policy.eval()
    discriminator.eval()
    
    episode_num = checkpoint.get('episode', 'unknown')
    
    return policy, discriminator, episode_num

def evaluate_policy(policy, discriminator, env, num_episodes=10, max_steps=1000):
    """Evaluate policy performance"""
    
    episode_rewards = []
    episode_lengths = []
    success_rate = 0
    discriminator_rewards = []
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        episode_disc_reward = 0
        steps = 0
        
        for step in range(max_steps):
            # Get action from policy
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(next(policy.parameters()).device)
            
            with torch.no_grad():
                mean, std, value = policy(obs_tensor)
                action = torch.normal(mean, std).cpu().numpy().flatten()
            
            # Step environment
            next_obs, env_reward, terminated, truncated, info = env.step(action)
            
            # Get discriminator reward
            next_obs_tensor = torch.FloatTensor(next_obs).unsqueeze(0).to(next(policy.parameters()).device)
            action_tensor = torch.FloatTensor(action).unsqueeze(0).to(next(policy.parameters()).device)
            
            with torch.no_grad():
                disc_reward = discriminator.get_reward(obs_tensor, action_tensor).cpu().numpy()[0]
            
            episode_reward += env_reward
            episode_disc_reward += disc_reward
            steps += 1
            
            obs = next_obs
            
            if terminated or truncated:
                break
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(steps)
        discriminator_rewards.append(episode_disc_reward)
        
        # Check success (if info contains success flag)
        if info.get('success', False):
            success_rate += 1
            
        print(f"Episode {episode+1}/{num_episodes}: "
              f"Length={steps}, Env_Reward={episode_reward:.2f}, "
              f"Disc_Reward={episode_disc_reward:.2f}, "
              f"Success={'Yes' if info.get('success', False) else 'No'}")
    
    success_rate /= num_episodes
    
    return {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_length': np.mean(episode_lengths),
        'std_length': np.std(episode_lengths),
        'success_rate': success_rate,
        'mean_disc_reward': np.mean(discriminator_rewards),
        'std_disc_reward': np.std(discriminator_rewards),
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'discriminator_rewards': discriminator_rewards
    }

def main():
    parser = argparse.ArgumentParser(description='Evaluate trained AIRL policy')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint file')
    parser.add_argument('--config', type=str, default='config.yaml', help='Config file path')
    parser.add_argument('--episodes', type=int, default=10, help='Number of evaluation episodes')
    parser.add_argument('--max_steps', type=int, default=1000, help='Max steps per episode')
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create environment
    env_config = config.get('environment', {})
    env = make_env(env_config)
    
    obs_dim = getattr(env, 'obs_dim', env.observation_space.shape[0])
    act_dim = getattr(env, 'action_dim', env.action_space.shape[0])
    
    print(f"Environment: obs_dim={obs_dim}, act_dim={act_dim}")
    
    # Load trained models
    policy, discriminator, episode_num = load_checkpoint(args.checkpoint, obs_dim, act_dim, device)
    print(f"Loaded checkpoint from episode {episode_num}")
    
    # Evaluate policy
    print(f"\nEvaluating policy for {args.episodes} episodes...")
    print("="*60)
    
    results = evaluate_policy(policy, discriminator, env, args.episodes, args.max_steps)
    
    # Print results
    print("\n" + "="*60)
    print("POLICY EVALUATION RESULTS")
    print("="*60)
    print(f"Mean Episode Reward: {results['mean_reward']:.3f} ± {results['std_reward']:.3f}")
    print(f"Mean Episode Length: {results['mean_length']:.1f} ± {results['std_length']:.1f}")
    print(f"Success Rate: {results['success_rate']*100:.1f}%")
    print(f"Mean Discriminator Reward: {results['mean_disc_reward']:.3f} ± {results['std_disc_reward']:.3f}")
    
    print(f"\nIndividual Episode Results:")
    for i in range(len(results['episode_rewards'])):
        print(f"  Episode {i+1}: Reward={results['episode_rewards'][i]:.2f}, "
              f"Length={results['episode_lengths'][i]}, "
              f"DiscReward={results['discriminator_rewards'][i]:.2f}")

if __name__ == "__main__":
    main()
