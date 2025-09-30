import torch
import torch.optim as optim
import torch.nn as nn
from torch.distributions import Normal
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'envs'))
from isaac_wrapper import make_env
from agents.policy import Policy
from agents.discriminator import AIRLDiscriminator
from utils.replay_buffer import ReplayBuffer
from utils.gae import compute_gae
import h5py
import yaml
import os
import logging
from datetime import datetime
import argparse
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def setup_logging(config):
    """Setup logging configuration"""
    log_dir = Path(config['logging']['tensorboard_dir'])
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup file logging
    log_file = log_dir / "training.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def save_checkpoint(policy, discriminator, policy_optim, discriminator_optim, 
                   episode, config, metrics=None):
    """Save model checkpoint"""
    checkpoint_dir = Path(config['logging']['checkpoint_dir'])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'episode': episode,
        'policy_state_dict': policy.state_dict(),  # Includes both actor and critic
        'discriminator_state_dict': discriminator.state_dict(),
        'policy_optimizer': policy_optim.state_dict(),
        'discriminator_optimizer': discriminator_optim.state_dict(),
        'config': config,
        'metrics': metrics
    }
    
    checkpoint_path = checkpoint_dir / f"checkpoint_episode_{episode}.pt"
    torch.save(checkpoint, checkpoint_path)
    
    # Also save as latest
    latest_path = checkpoint_dir / "latest_checkpoint.pt"
    torch.save(checkpoint, latest_path)
    
    return checkpoint_path

def load_checkpoint(checkpoint_path, policy, discriminator, policy_optim, discriminator_optim):
    """Load model checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    policy.load_state_dict(checkpoint['policy_state_dict'])  # Loads both actor and critic
    discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
    
    policy_optim.load_state_dict(checkpoint['policy_optimizer'])
    discriminator_optim.load_state_dict(checkpoint['discriminator_optimizer'])
    
    return checkpoint['episode'], checkpoint.get('metrics', {})

def load_expert_data(path, batch_size=64):
    """Load expert data from HDF5 file with proper hierarchical structure handling"""
    try:
        with h5py.File(path, 'r') as f:
            # Handle hierarchical structure: f['data']['demo_X']['obs']/['actions']
            all_obs = []
            all_acts = []
            all_next_obs = []
            
            data_group = f['data']
            demo_names = [k for k in data_group.keys() if k.startswith('demo_')]
            
            print(f"Loading {len(demo_names)} expert trajectories...")
            
            for demo_name in demo_names:
                demo = data_group[demo_name]
                
                # Get actions
                actions = demo['actions'][:]
                traj_length = actions.shape[0]
                
                # Build flattened observations from obs components (excluding duplicate actions)
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
                
                # Create next_obs by shifting (last next_obs = last obs)
                next_obs = np.zeros_like(obs)
                next_obs[:-1] = obs[1:]
                next_obs[-1] = obs[-1]
                
                # Skip last transition (no meaningful next state)
                obs = obs[:-1]
                actions = actions[:-1]
                next_obs = next_obs[:-1]
                
                all_obs.append(obs)
                all_acts.append(actions)
                all_next_obs.append(next_obs)
                
                print(f"  {demo_name}: {traj_length-1} transitions, obs_dim={obs.shape[1]}, act_dim={actions.shape[1]}")
            
            # Combine all trajectories
            all_obs = np.concatenate(all_obs, axis=0)
            all_acts = np.concatenate(all_acts, axis=0)
            all_next_obs = np.concatenate(all_next_obs, axis=0)
            
            print(f"Total expert transitions: {all_obs.shape[0]}")
            print(f"Final dimensions - obs: {all_obs.shape[1]}, acts: {all_acts.shape[1]}")
            
            # Convert to tensors
            obs = torch.tensor(all_obs, dtype=torch.float32)
            acts = torch.tensor(all_acts, dtype=torch.float32)
            next_obs = torch.tensor(all_next_obs, dtype=torch.float32)
            
        dataset = torch.utils.data.TensorDataset(obs, acts, next_obs)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        return loader, all_obs.shape[1], all_acts.shape[1]  # Return dimensions too
        
    except Exception as e:
        logging.error(f"Failed to load expert data from {path}: {e}")
        logging.info("Creating dummy expert data for testing...")
        # Create dummy data for testing - use reasonable dimensions
        obs = torch.randn(1000, 74)  # Based on actual expert data analysis
        acts = torch.randn(1000, 8)  # Based on actual expert data analysis  
        next_obs = torch.randn(1000, 74)
        dataset = torch.utils.data.TensorDataset(obs, acts, next_obs)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        return loader, 74, 8

def ppo_update(policy, policy_optim, data, config, clip_eps=0.2, epochs=10, batch_size=64):
    """PPO update with integrated value function"""
    ppo_cfg = config['training']['ppo']
    clip_eps = ppo_cfg['clip_epsilon']
    epochs = ppo_cfg['epochs_per_update']
    batch_size = ppo_cfg['batch_size']
    
    obs = data['obs'].to(device)
    acts = data['acts'].to(device)
    old_log_probs = data['log_probs'].to(device)
    returns = data['returns'].to(device)
    advantages = data['advantages'].to(device)

    # Normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    dataset = torch.utils.data.TensorDataset(obs, acts, old_log_probs, returns, advantages)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    policy_losses = []
    value_losses = []

    for epoch in range(epochs):
        for batch in loader:
            b_obs, b_acts, b_old_log_probs, b_returns, b_advantages = batch

            # Policy forward pass (gets both policy and value)
            mean, std, value_preds = policy(b_obs)
            dist = Normal(mean, std)
            log_probs = dist.log_prob(b_acts).sum(-1)

            # PPO clipped loss
            ratio = torch.exp(log_probs - b_old_log_probs)
            surrogate1 = ratio * b_advantages
            surrogate2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * b_advantages
            policy_loss = -torch.min(surrogate1, surrogate2).mean()
            
            # Add entropy bonus
            entropy = dist.entropy().sum(-1).mean()
            policy_loss -= ppo_cfg['entropy_coef'] * entropy

            # Value loss
            value_preds = value_preds.squeeze(-1)  # Remove last dimension
            value_loss = nn.MSELoss()(value_preds, b_returns)

            # Combined loss (both policy and value)
            total_loss = policy_loss + ppo_cfg.get('value_loss_coef', 0.5) * value_loss

            # Update both policy and value function together
            policy_optim.zero_grad()
            total_loss.backward()
            if ppo_cfg.get('max_grad_norm'):
                torch.nn.utils.clip_grad_norm_(policy.parameters(), ppo_cfg['max_grad_norm'])
            policy_optim.step()
            
            policy_losses.append(policy_loss.item())
            value_losses.append(value_loss.item())

    return np.mean(policy_losses), np.mean(value_losses)

def discriminator_update(discriminator, discriminator_optim, expert_loader, policy_buffer, policy, config):
    """Discriminator update with configuration support"""
    discriminator.train()
    criterion = nn.BCEWithLogitsLoss()
    
    disc_cfg = config['training']['discriminator']
    batch_size = disc_cfg['batch_size']
    
    expert_iter = iter(expert_loader)
    policy_data = policy_buffer.get()
    obs = policy_data['obs'].to(device)
    acts = policy_data['acts'].to(device)
    next_obs = policy_data['next_obs'].to(device)
    
    dataset = torch.utils.data.TensorDataset(obs, acts, next_obs)
    policy_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    policy_iter = iter(policy_loader)

    disc_losses = []

    for _ in range(min(len(expert_loader), len(policy_loader))):
        try:
            expert_obs, expert_acts, expert_next_obs = next(expert_iter)
            policy_obs, policy_acts, policy_next_obs = next(policy_iter)
        except StopIteration:
            break
        
        expert_obs, expert_acts, expert_next_obs = expert_obs.to(device), expert_acts.to(device), expert_next_obs.to(device)
        policy_obs, policy_acts, policy_next_obs = policy_obs.to(device), policy_acts.to(device), policy_next_obs.to(device)

        # Forward pass through discriminator
        f_expert = discriminator(expert_obs, expert_acts, expert_next_obs)
        f_policy = discriminator(policy_obs, policy_acts, policy_next_obs)

        # Get policy log probabilities
        with torch.no_grad():
            mean, std, _ = policy(policy_obs)
            dist = Normal(mean, std)
            log_pi = dist.log_prob(policy_acts).sum(-1)

        # AIRL discriminator loss - CORRECTED FORMULATION
        # The discriminator learns D(s,a,s') = exp(f(s,a,s')) / (exp(f(s,a,s')) + π(a|s))
        # Where f(s,a,s') is the discriminator output (advantage function)
        
        # For expert data: D should output 1 (log D = 0)
        # For policy data: D should output 0 (log D = -inf, but we use sigmoid)
        
        # Expert logits: log(exp(f) / (exp(f) + π)) = f - log(exp(f) + π) ≈ f when f >> log π
        # Policy logits: log(π / (exp(f) + π)) = log π - log(exp(f) + π) = log π - f - log(1 + exp(-f))
        
        # Simplified AIRL loss formulation:
        expert_scores = f_expert  # f(s,a,s') for expert data
        policy_scores = f_policy - log_pi  # f(s,a,s') - log π(a|s) for policy data
        
        # Binary classification: expert=1, policy=0
        expert_labels = torch.ones(expert_scores.size(0), device=device)
        policy_labels = torch.zeros(policy_scores.size(0), device=device)
        
        # Combine logits and labels
        all_logits = torch.cat([expert_scores, policy_scores], dim=0)
        all_labels = torch.cat([expert_labels, policy_labels], dim=0)
        
        # BCE loss
        loss = criterion(all_logits, all_labels)

        discriminator_optim.zero_grad()
        loss.backward()
        discriminator_optim.step()
        
        disc_losses.append(loss.item())

    return np.mean(disc_losses) if disc_losses else 0.0

def main():
    """Main training loop"""
    parser = argparse.ArgumentParser(description='Train AIRL for Franka Cube Lift')
    parser.add_argument('--config', type=str, default='config.yaml', help='Config file path')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--device', type=str, default=None, help='Device to use (cuda/cpu)')
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    logger = setup_logging(config)
    
    # Set device
    global device
    if args.device:
        device = torch.device(args.device)
    elif config['device']['use_cuda'] and torch.cuda.is_available():
        device = torch.device(f"cuda:{config['device']['cuda_device']}")
    else:
        device = torch.device("cpu")
    
    logger.info(f"Using device: {device}")
    logger.info(f"Training configuration loaded from: {args.config}")

    # Set random seed for reproducibility
    torch.manual_seed(config['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config['seed'])
    
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

    logger.info(f"Environment: {env_cfg['name']}")
    logger.info(f"Observation dim: {obs_dim}, Action dim: {act_dim}")

    # Get training config early (needed for loading expert data)
    train_cfg = config['training']

    # Load expert data FIRST to get correct dimensions
    expert_data_result = load_expert_data(
        train_cfg['expert_data']['path'], 
        batch_size=train_cfg['expert_data']['batch_size']
    )
    
    if len(expert_data_result) == 3:  # New format returns (loader, obs_dim, act_dim)
        expert_loader, expert_obs_dim, expert_act_dim = expert_data_result
        logger.info(f"Expert data loaded - obs_dim: {expert_obs_dim}, act_dim: {expert_act_dim}")
        
        # Update dimensions to match expert data
        if obs_dim != expert_obs_dim or act_dim != expert_act_dim:
            logger.warning(f"Env dims (obs={obs_dim}, act={act_dim}) != Expert dims (obs={expert_obs_dim}, act={expert_act_dim})")
            logger.info("Using expert data dimensions for network architecture")
            obs_dim, act_dim = expert_obs_dim, expert_act_dim
    else:  # Fallback for old format
        expert_loader = expert_data_result
        logger.warning("Using fallback dimensions - may cause issues if expert data was actually loaded")

    # Create networks WITH CORRECT DIMENSIONS from expert data
    net_cfg = config['network']
    policy = Policy(obs_dim, act_dim, hidden_size=net_cfg['policy']['hidden_sizes'][0]).to(device)
    # Note: Policy already includes value function (critic), so we don't need separate value_fn
    discriminator = AIRLDiscriminator(
        obs_dim, act_dim, 
        hidden_size=net_cfg['discriminator']['hidden_sizes'][0],
        gamma=net_cfg['discriminator']['gamma']
    ).to(device)

    # Create optimizers
    policy_optim = optim.Adam(policy.parameters(), lr=train_cfg['ppo']['learning_rate'])
    discriminator_optim = optim.Adam(discriminator.parameters(), lr=train_cfg['discriminator']['learning_rate'])

    # Create replay buffer
    buffer = ReplayBuffer(train_cfg['ppo']['buffer_size'], obs_dim, act_dim)

    # Setup tensorboard logging
    tb_dir = Path(config['logging']['tensorboard_dir']) / datetime.now().strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(tb_dir)
    logger.info(f"Tensorboard logs: {tb_dir}")

    # Resume from checkpoint if specified
    start_episode = 0
    if args.resume:
        start_episode, metrics = load_checkpoint(
            args.resume, policy, discriminator, 
            policy_optim, discriminator_optim
        )
        logger.info(f"Resumed from episode {start_episode}")

    # Training loop
    num_episodes = train_cfg['num_episodes']
    max_steps = train_cfg['max_steps_per_episode']
    gamma = train_cfg['ppo']['gamma']
    
    # Metrics tracking
    episode_rewards = []
    success_rates = []
    
    logger.info("Starting training...")
    
    pbar = tqdm(range(start_episode, num_episodes), desc="Training")
    
    for episode in pbar:
        obs, _ = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        obs = torch.tensor(obs, dtype=torch.float32).to(device)
        done = False
        ep_reward = 0
        successes = 0
        episode_length = 0

        for step in range(max_steps):
            # Get action from policy
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
            
            next_obs = torch.tensor(next_obs, dtype=torch.float32).to(device)

            # AIRL reward (replace environment reward)
            with torch.no_grad():
                raw_airl_reward = discriminator.get_reward(obs.unsqueeze(0), action).item()
                # Use adaptive reward scaling based on running statistics
                if not hasattr(discriminator, 'reward_mean'):
                    discriminator.reward_mean = 0.0
                    discriminator.reward_std = 1.0
                    discriminator.reward_count = 0
                
                # Update running statistics
                discriminator.reward_count += 1
                delta = raw_airl_reward - discriminator.reward_mean
                discriminator.reward_mean += delta / discriminator.reward_count
                discriminator.reward_std = max(1.0, abs(delta))  # Prevent division by very small numbers
                
                # Normalize reward
                airl_reward = (raw_airl_reward - discriminator.reward_mean) / discriminator.reward_std
                airl_reward = np.clip(airl_reward, -5.0, 5.0)  # Reasonable bounds

            # Store transition
            buffer.store(
                obs.cpu(), action.squeeze().cpu(), airl_reward, done, 
                value.cpu(), log_prob.cpu(), next_obs.cpu()
            )

            obs = next_obs
            ep_reward += airl_reward
            episode_length += 1
            
            # Check for success (if info contains success flag)
            if info.get('success', False) or (info.get('episode') and info['episode'].get('success', False)):
                successes += 1

            if done:
                break

        # GAE computation and PPO update
        if buffer.size > 0:
            data = buffer.get()

            # Compute last value using policy's built-in critic
            with torch.no_grad():
                _, _, last_value = policy(obs.unsqueeze(0))
                last_value = last_value.squeeze(-1).cpu()  # Remove batch and feature dims
            values = torch.cat([data['values'], last_value], dim=0)

            # Compute advantages and returns
            advantages, returns = compute_gae(
                data['rewards'], values, data['dones'], 
                gamma=gamma, lam=train_cfg['ppo']['gae_lambda']
            )
            data['advantages'] = advantages
            data['returns'] = returns

            # Balanced discriminator training schedule
            if episode < 50:
                # Light initial discriminator training
                disc_updates = 2
            elif episode < 200:
                # Moderate training for episodes 50-200
                disc_updates = 1
            else:
                # Less frequent updates after episode 200 to prevent overtraining
                disc_updates = 1 if episode % 5 == 0 else 0
                
            for _ in range(disc_updates):
                discriminator_update(discriminator, discriminator_optim, expert_loader, buffer, policy, config)

            # PPO update with integrated value function  
            ppo_update(policy, policy_optim, data, config)

            buffer.clear()

        # Logging
        episode_rewards.append(ep_reward)
        success_rate = successes / max(episode_length, 1)
        success_rates.append(success_rate)
        
        # Tensorboard logging
        if episode % config['logging']['log_interval'] == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            avg_success_rate = np.mean(success_rates[-10:])
            
            writer.add_scalar('Episode/Reward', ep_reward, episode)
            writer.add_scalar('Episode/Length', episode_length, episode)
            writer.add_scalar('Episode/AvgReward', avg_reward, episode)
            writer.add_scalar('Episode/SuccessRate', success_rate, episode)
            writer.add_scalar('Episode/AvgSuccessRate', avg_success_rate, episode)
            
            pbar.set_postfix({
                'Reward': f"{ep_reward:.2f}",
                'AvgReward': f"{avg_reward:.2f}",
                'Success': f"{success_rate:.2%}"
            })
            
            logger.info(f"Episode {episode}: Reward={ep_reward:.2f}, Length={episode_length}, Success={success_rate:.2%}")

        # Save checkpoint
        if episode % config['logging']['save_interval'] == 0 and episode > 0:
            metrics = {
                'episode_rewards': episode_rewards,
                'success_rates': success_rates,
                'avg_reward_10': np.mean(episode_rewards[-10:]),
                'avg_success_rate_10': np.mean(success_rates[-10:])
            }
            checkpoint_path = save_checkpoint(
                policy, discriminator, policy_optim, discriminator_optim,
                episode, config, metrics
            )
            logger.info(f"Checkpoint saved: {checkpoint_path}")

    # Final checkpoint
    metrics = {
        'episode_rewards': episode_rewards,
        'success_rates': success_rates,
        'final_avg_reward': np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards),
        'final_success_rate': np.mean(success_rates[-100:]) if len(success_rates) >= 100 else np.mean(success_rates)
    }
    final_checkpoint = save_checkpoint(
        policy, discriminator, policy_optim, discriminator_optim,
        num_episodes, config, metrics
    )
    
    logger.info("Training completed!")
    logger.info(f"Final checkpoint saved: {final_checkpoint}")
    logger.info(f"Final average reward: {metrics['final_avg_reward']:.2f}")
    logger.info(f"Final success rate: {metrics['final_success_rate']:.2%}")
    
    writer.close()
    env.close()

if __name__ == '__main__':
    main()
