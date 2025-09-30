"""
Isaac Lab Environment Wrapper for AIRL Training
Properly initializes Isaac Sim and creates Isaac Lab environments
"""

import gymnasium as gym
import numpy as np
from typing import Dict, Any, Optional

# Isaac Sim and Isaac Lab specific imports
def configure_isaac_nucleus():
    """Configure Isaac Sim nucleus settings for asset download"""
    import os
    try:
        import carb
        settings = carb.settings.get_settings()
        
        # Set Nucleus asset root for cloud downloads
        nucleus_root = "omniverse://ov-content.ovaleveloper.com"
        settings.set("/persistent/isaac/asset_root/cloud", nucleus_root)
        print(f"🌍 Set Nucleus cloud root: {nucleus_root}")
        
        # Also set environment variable as backup
        os.environ['ISAAC_NUCLEUS_URL'] = "omniverse://ov-content.ovaleveloper.com/Isaac"
        print("🔧 Set ISAAC_NUCLEUS_URL for cloud assets")
        
        return True
    except Exception as e:
        print(f"⚠️ Could not configure nucleus: {e}")
        return False

def setup_isaac_lab_assets():
    """Setup Isaac Lab assets configuration"""
    import os
    
    # Set up Isaac Lab asset paths
    isaac_lab_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    isaac_sim_path = os.path.join(isaac_lab_dir, "IsaacSim")
    
    # Set environment variables for Isaac Lab
    if not os.getenv('ISAAC_SIM_PATH'):
        os.environ['ISAAC_SIM_PATH'] = isaac_sim_path
        print(f"🔧 Set ISAAC_SIM_PATH: {isaac_sim_path}")
    
    print("📦 Isaac Lab asset system configured")

def initialize_isaac_sim(headless=None):
    """Initialize Isaac Sim with proper sequence and asset configuration
    Args:
        headless: If True, run without GUI. If None, check environment variable.
    """
    try:
        import os
        from isaacsim.simulation_app import SimulationApp
        
        # Setup assets first
        setup_isaac_lab_assets()
        
        # Determine headless mode
        if headless is None:
            headless = os.getenv('HEADLESS', 'false').lower() == 'true'
        
        # Create simulation app with asset configuration
        simulation_config = {
            "headless": headless,
            "physics_dt": 1.0 / 60.0,
            "rendering_dt": 1.0 / 60.0,
        }
        
        print(f"🎮 Isaac Sim mode: {'Headless' if headless else 'With GUI'}")
        
        simulation_app = SimulationApp(simulation_config)
        
        # Initialize asset system after simulation app is ready
        print("📦 Initializing asset download system...")
        try:
            # Configure nucleus settings for asset downloads
            if configure_isaac_nucleus():
                print("✅ Nucleus configuration successful")
            else:
                print("⚠️ Using fallback asset configuration")
        except Exception as asset_error:
            print(f"⚠️ Asset configuration error: {asset_error}")
        
        # Now we can import other Isaac Lab modules
        import isaaclab_tasks
        
        print("✅ Isaac Sim initialized successfully")
        return simulation_app
    except Exception as e:
        print(f"❌ Failed to initialize Isaac Sim: {e}")
        return None

class IsaacLabWrapper:
    """Wrapper for Isaac Lab environments"""
    
    def __init__(self, env_id: str = "Isaac-Lift-Cube-Franka-v0", num_envs: int = 1):
        self.env_id = env_id
        self.num_envs = num_envs
        self.episode_step = 0
        self.episode_reward = 0.0
        self._env = None
        
        try:
            # Initialize Isaac Sim first - with GUI for visualization
            print(f"🚀 Initializing Isaac Sim (with GUI for visualization)...")
            self.simulation_app = initialize_isaac_sim(headless=False)  # GUI mode
            
            if self.simulation_app is None:
                raise Exception("Isaac Sim initialization failed")
            
            print(f"🚀 Creating Isaac Lab environment: {env_id}")
            
            # Import Isaac Lab tasks (should work now that Isaac Sim is initialized)
            import isaaclab_tasks  # Register all tasks
            print(f"✅ Isaac Lab tasks imported successfully")
            
            # Create environment with simplified approach - let Isaac Lab handle everything
            try:
                # Use a simpler, more reliable Isaac Lab environment
                print("🎯 Creating simple Isaac Lab lift environment...")
                # Import the config directly and create it 
                from isaaclab_tasks.manager_based.manipulation.lift.config.franka.joint_pos_env_cfg import FrankaCubeLiftEnvCfg
                cfg = FrankaCubeLiftEnvCfg()
                cfg.scene.num_envs = num_envs
                
                # Create environment directly with config - this is the RELIABLE way
                from isaaclab.envs import ManagerBasedRLEnv
                print("📦 Creating Isaac Lab environment...")
                self._env = ManagerBasedRLEnv(cfg=cfg)
                print(f"✅ Successfully created Isaac Lab environment!")
                
            except Exception as env_error:
                print(f"❌ Isaac Lab creation failed: {env_error}")
                print("🎯 Falling back to dummy environment for training...")
                raise env_error
            
            # Set up observation and action spaces for single environment
            if hasattr(self._env, 'single_observation_space'):
                obs_space = self._env.single_observation_space
                if isinstance(obs_space, dict) and 'policy' in obs_space:
                    self.observation_space = obs_space['policy']
                else:
                    self.observation_space = obs_space
            else:
                # Fallback to regular observation space
                obs_space = self._env.observation_space
                if isinstance(obs_space, dict) and 'policy' in obs_space:
                    self.observation_space = obs_space['policy']
                else:
                    self.observation_space = obs_space
            
            if hasattr(self._env, 'single_action_space'):
                self.action_space = self._env.single_action_space
            else:
                self.action_space = self._env.action_space
                
            print(f"✅ Observation space: {self.observation_space}")
            print(f"✅ Action space: {self.action_space}")
            
        except Exception as e:
            print(f"❌ Failed to create Isaac Lab environment: {e}")
            print("❌ Using dummy environment")
            self._create_dummy_spaces()
    
    def _create_dummy_spaces(self):
        """Create dummy spaces matching expert data"""
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(74,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(8,), dtype=np.float32
        )
        self._dummy_state = self._reset_dummy_state()
    
    def _reset_dummy_state(self):
        """Reset dummy state"""
        return {
            'joint_pos': np.zeros(7),
            'joint_vel': np.zeros(7),
            'object_pos': np.array([0.5, 0.0, 1.0]),
            'target_pos': np.array([0.5, 0.0, 1.15]),
            'gripper_pos': np.zeros(2),
        }
    
    def _get_dummy_obs(self):
        """Generate dummy observation"""
        base_obs = np.concatenate([
            self._dummy_state['joint_pos'],
            self._dummy_state['joint_vel'],
            self._dummy_state['object_pos'],
            self._dummy_state['target_pos'],
            self._dummy_state['gripper_pos'],
        ])
        # Pad to 74 dimensions
        padding = np.random.randn(74 - len(base_obs)) * 0.1
        obs = np.concatenate([base_obs, padding])
        return obs.astype(np.float32)
    
    def reset(self, seed=None, options=None):
        """Reset environment"""
        self.episode_step = 0
        self.episode_reward = 0.0
        
        if self._env is not None:
            # Real Isaac Lab environment
            obs_dict, info = self._env.reset(seed=seed, options=options)
            
            # Extract observation
            if isinstance(obs_dict, dict):
                obs = obs_dict.get('policy', list(obs_dict.values())[0])
            else:
                obs = obs_dict
            
            # Convert to numpy and handle batch dimension
            if hasattr(obs, 'cpu'):
                obs = obs.cpu().numpy()
            if obs.ndim > 1:
                obs = obs[0]  # Take first environment
                
            return obs.astype(np.float32), info
        else:
            # Dummy environment
            self._dummy_state = self._reset_dummy_state()
            obs = self._get_dummy_obs()
            return obs, {}
    
    def step(self, action):
        """Step environment"""
        self.episode_step += 1
        
        if self._env is not None:
            # Real Isaac Lab environment
            # Convert action to tensor if needed
            if not hasattr(action, 'shape'):
                action = np.array(action)
            
            # Add batch dimension if needed
            if action.ndim == 1:
                import torch
                action = torch.from_numpy(action).float().unsqueeze(0)
            
            obs_dict, reward, terminated, truncated, info = self._env.step(action)
            
            # Extract observation
            if isinstance(obs_dict, dict):
                obs = obs_dict.get('policy', list(obs_dict.values())[0])
            else:
                obs = obs_dict
            
            # Convert to numpy and handle batch dimensions
            if hasattr(obs, 'cpu'):
                obs = obs.cpu().numpy()
            if obs.ndim > 1:
                obs = obs[0]
                
            if hasattr(reward, 'cpu'):
                reward = float(reward[0].cpu().numpy())
            elif hasattr(reward, '__len__'):
                reward = float(reward[0])
                
            if hasattr(terminated, 'cpu'):
                terminated = bool(terminated[0].cpu().numpy())
            elif hasattr(terminated, '__len__'):
                terminated = bool(terminated[0])
                
            if hasattr(truncated, 'cpu'):
                truncated = bool(truncated[0].cpu().numpy())
            elif hasattr(truncated, '__len__'):
                truncated = bool(truncated[0])
                
        else:
            # Dummy environment
            # Simple physics simulation
            self._dummy_state['joint_pos'] += action[:7] * 0.01
            if len(action) > 7:
                self._dummy_state['gripper_pos'] = action[7:9] if len(action) > 8 else np.array([action[7], action[7]])
            
            # Simple object dynamics
            gripper_closed = np.mean(self._dummy_state['gripper_pos']) < -0.5
            if gripper_closed and self.episode_step > 50:
                self._dummy_state['object_pos'][2] += 0.001
            
            obs = self._get_dummy_obs()
            
            # Simple reward
            distance = np.linalg.norm(self._dummy_state['object_pos'] - self._dummy_state['target_pos'])
            reward = -distance + (10.0 if distance < 0.05 else 0.0)
            reward -= 0.01 * np.sum(action**2)
            reward = np.clip(reward, -10.0, 10.0)
            
            terminated = distance < 0.05
            truncated = self.episode_step >= 1000
            info = {}
        
        self.episode_reward += reward
        
        if terminated or truncated:
            info['episode'] = {
                'r': self.episode_reward,
                'l': self.episode_step,
            }
        
        return obs.astype(np.float32), reward, terminated, truncated, info
    
    def render(self, mode="human"):
        """Render environment"""
        if self._env is not None and hasattr(self._env, 'render'):
            return self._env.render()
    
    def close(self):
        """Close environment"""
        if self._env is not None and hasattr(self._env, 'close'):
            try:
                self._env.close()
            except AttributeError as e:
                print(f"⚠️ Warning during environment close: {e}")
        
        # Close simulation app
        if hasattr(self, 'simulation_app') and self.simulation_app is not None:
            self.simulation_app.close()

def make_env(cfg: Dict[str, Any] = None):
    """Factory function to create environment"""
    cfg = cfg or {}
    env_id = cfg.get('env_id', 'Isaac-Lift-Cube-Franka-v0')
    num_envs = cfg.get('num_envs', 1)
    
    print(f"🚀 Creating environment: {env_id}")
    return IsaacLabWrapper(env_id=env_id, num_envs=num_envs)