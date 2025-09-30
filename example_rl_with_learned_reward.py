"""
Example: Use Learned AIRL Reward Function for Standard RL Training
Demonstrates how to use the extracted reward function with PPO or other RL algorithms
"""

import torch
import torch.optim as optim
import numpy as np
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from envs.franka_wrapper import make_env
from agents.policy import Policy
from utils.replay_buffer import ReplayBuffer
from utils.gae import compute_gae
from utils.extract_reward import LearnedRewardFunction
from torch.distributions import Normal

def train_with_learned_reward(reward_function, env, config, num_episodes=100):
    """
    Train a new policy using the learned AIRL reward function
    This demonstrates pure RL training with the extracted reward
    """
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Get environment dimensions
    if hasattr(env.observation_space, 'shape'):
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]
    else:
        obs_dim = env.obs_dim
        act_dim = env.action_dim
    
    # Create NEW policy (not the AIRL-trained one!)
    policy = Policy(obs_dim, act_dim, hidden_size=256).to(device)
    value_fn = Policy(obs_dim, 1, hidden_size=256).to(device)
    
    # Optimizers
    policy_optim = optim.Adam(policy.parameters(), lr=3e-4)
    value_optim = optim.Adam(value_fn.parameters(), lr=1e-3)
    
    # Buffer
    buffer = ReplayBuffer(2048, obs_dim, act_dim)
    
    print(f"🚀 Training new policy with learned AIRL reward function...")
    print(f"Policy parameters: {sum(p.numel() for p in policy.parameters()):,}")
    
    episode_rewards = []
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        
        obs = torch.tensor(obs, dtype=torch.float32).to(device)
        episode_reward = 0
        
        for step in range(1000):  # Max episode length
            # Policy action
            with torch.no_grad():
                mean, std, value = policy(obs.unsqueeze(0))
                dist = Normal(mean, std)
                action = dist.sample()
                log_prob = dist.log_prob(action).sum()
            
            value = value.squeeze()
            action_np = action.squeeze().cpu().numpy()
            
            # Environment step
            next_obs, env_reward, terminated, truncated, info = env.step(action_np)
            done = terminated or truncated
            
            # ⭐ USE LEARNED REWARD INSTEAD OF ENVIRONMENT REWARD ⭐
            learned_reward = reward_function(obs.cpu().numpy(), action_np)
            
            next_obs = torch.tensor(next_obs, dtype=torch.float32).to(device)
            
            # Store transition with learned reward
            buffer.store(
                obs.cpu(), action.squeeze().cpu(), learned_reward, done,
                value.cpu(), log_prob.cpu(), next_obs.cpu()
            )
            
            obs = next_obs
            episode_reward += learned_reward
            
            if done:
                break
        
        # PPO update
        if buffer.size > 0:
            data = buffer.get()
            
            # Compute advantages
            with torch.no_grad():
                last_value = value_fn.get_value(obs.unsqueeze(0)).cpu()
            values = torch.cat([data['values'], last_value], dim=0)
            
            advantages, returns = compute_gae(data['rewards'], values, data['dones'], gamma=0.99)
            data['advantages'] = advantages
            data['returns'] = returns
            
            # PPO update (simplified)
            obs_batch = data['obs'].to(device)
            acts_batch = data['acts'].to(device)
            old_log_probs = data['log_probs'].to(device)
            returns_batch = data['returns'].to(device)
            advantages_batch = data['advantages'].to(device)
            
            # Normalize advantages
            advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)
            
            # Policy update
            mean, std, _ = policy(obs_batch)
            dist = Normal(mean, std)
            log_probs = dist.log_prob(acts_batch).sum(-1)
            
            ratio = torch.exp(log_probs - old_log_probs)
            surrogate1 = ratio * advantages_batch
            surrogate2 = torch.clamp(ratio, 0.8, 1.2) * advantages_batch
            policy_loss = -torch.min(surrogate1, surrogate2).mean()
            
            policy_optim.zero_grad()
            policy_loss.backward()
            policy_optim.step()
            
            # Value update
            value_preds = value_fn(obs_batch).squeeze(-1)
            value_loss = torch.nn.MSELoss()(value_preds, returns_batch)
            
            value_optim.zero_grad()
            value_loss.backward()
            value_optim.step()
            
            buffer.clear()
        
        episode_rewards.append(episode_reward)
        
        if episode % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            print(f"Episode {episode}: Reward = {episode_reward:.3f}, Avg = {avg_reward:.3f}")
    
    return policy, episode_rewards

def compare_rewards(env, learned_reward_fn, num_samples=100):
    """Compare learned reward vs environment reward"""
    
    print(f"📊 Comparing learned reward vs environment reward...")
    
    learned_rewards = []
    env_rewards = []
    
    for _ in range(num_samples):
        obs, _ = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        
        # Random action
        if hasattr(env.action_space, 'sample'):
            action = env.action_space.sample()
        else:
            action = np.random.uniform(-1, 1, env.action_dim)
        
        # Get both rewards
        learned_reward = learned_reward_fn(obs, action)
        next_obs, env_reward, _, _, _ = env.step(action)
        
        learned_rewards.append(learned_reward)
        env_rewards.append(env_reward)
    
    learned_rewards = np.array(learned_rewards)
    env_rewards = np.array(env_rewards)
    
    print(f"Learned Reward - Mean: {learned_rewards.mean():.4f}, Std: {learned_rewards.std():.4f}")
    print(f"Environment Reward - Mean: {env_rewards.mean():.4f}, Std: {env_rewards.std():.4f}")
    print(f"Correlation: {np.corrcoef(learned_rewards, env_rewards)[0,1]:.4f}")
    
    return learned_rewards, env_rewards

def main():
    """
    Example workflow:
    1. Extract reward function from trained AIRL
    2. Train new policy using only the learned reward
    3. Compare performance
    """
    
    # Example configuration (simplified)
    config = {
        'env': {'num_envs': 1, 'max_episode_length': 1000},
        'network': {
            'discriminator': {'hidden_sizes': [256], 'gamma': 0.99}
        }
    }
    
    # Create environment
    env = make_env(config['env'])
    
    print("🎯 AIRL Reward Function Extraction Example")
    print("=" * 50)
    
    # Step 1: Load learned reward function
    # (In practice, you would have this from extract_reward.py)
    checkpoint_path = "logs/checkpoints/latest_checkpoint.pt"
    
    if Path(checkpoint_path).exists():
        print(f"📂 Loading reward function from: {checkpoint_path}")
        
        # Get dimensions
        if hasattr(env.observation_space, 'shape'):
            obs_dim = env.observation_space.shape[0]
            act_dim = env.action_space.shape[0]
        else:
            obs_dim = env.obs_dim
            act_dim = env.action_dim
        
        try:
            reward_function = LearnedRewardFunction.load(
                checkpoint_path, obs_dim, act_dim, config
            )
            
            print(f"✅ Reward function loaded successfully!")
            
            # Step 2: Compare rewards (optional)
            compare_rewards(env, reward_function, num_samples=50)
            
            # Step 3: Train new policy with learned reward
            print(f"\n🚀 Training new policy with learned reward...")
            new_policy, rewards = train_with_learned_reward(
                reward_function, env, config, num_episodes=50
            )
            
            print(f"✅ Training completed!")
            print(f"Final average reward: {np.mean(rewards[-10:]):.3f}")
            
            # Save the new policy
            torch.save({
                'policy_state_dict': new_policy.state_dict(),
                'training_rewards': rewards,
                'config': config
            }, "logs/checkpoints/rl_trained_with_airl_reward.pt")
            
            print(f"💾 New policy saved!")
            
        except Exception as e:
            print(f"❌ Error loading reward function: {e}")
            print(f"Make sure you have trained an AIRL model first!")
    else:
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        print(f"Train AIRL model first: python train_airl.py")
    
    env.close()

if __name__ == '__main__':
    main()
