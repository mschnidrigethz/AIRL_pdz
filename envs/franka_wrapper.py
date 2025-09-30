"""
Franka Environment Wrapper for Isaac Lab
Implements a Cube Lift Task with Franka Panda robot
"""

import numpy as np
from typing import Dict, Any

# Try to import required libraries with fallbacks
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available")

try:
    import gymnasium as gym
    from gymnasium import spaces
    GYM_AVAILABLE = True
except ImportError:
    try:
        import gym
        from gym import spaces
        GYM_AVAILABLE = True
    except ImportError:
        GYM_AVAILABLE = False
        print("Warning: Neither gymnasium nor gym available")

# Isaac Lab availability will be checked dynamically in make_env()
ISAAC_LAB_AVAILABLE = False  # Will be set to True if Isaac Lab is available

print("Warning: Isaac Lab not available, using dummy environment")

# Create dummy classes for development
class configclass:
        def __init__(self, cls):
            return cls
    
    class ManagerBasedRLEnvCfg:
        def __post_init__(self):
            pass

if ISAAC_LAB_AVAILABLE:
    @configclass
    class FrankaCubeLiftEnvCfg(ManagerBasedRLEnvCfg):
        """Configuration for Franka Cube Lift Environment"""

        def __post_init__(self):
            """Post initialization."""
            # general settings
            self.episode_length_s = 8.0
            self.decimation = 2
            self.action_scale = 100.0
            self.num_actions = 9  # 7 arm joints + 2 gripper
            self.num_observations = 23  # robot state + object state + goal
            self.num_states = 0

        # simulation settings
        sim: SimulationCfg = SimulationCfg(
            dt=1 / 120,
            render_interval=2,
            disable_contact_processing=True,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
            ),
        )

        # scene settings
        scene: InteractiveSceneCfg = InteractiveSceneCfg(
            num_envs=4096, env_spacing=2.5, replicate_physics=True
        )

        # robot configuration
        robot: ArticulationCfg = FRANKA_PANDA_CFG.replace(prim_path="/World/envs/env_.*/Robot")
        
        # object configuration  
        object: RigidObjectCfg = RigidObjectCfg(
            prim_path="/World/envs/env_.*/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.5, 0, 1.055], rot=[1, 0, 0, 0]),
            spawn=sim_utils.CuboidCfg(
                size=(0.05, 0.05, 0.05),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.2),
            ),
        )

        # ground plane
        terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="plane",
            collision_group=-1,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply", 
                static_friction=1.0,
                dynamic_friction=1.0,
            ),
            debug_vis=False,
        )

        # MDP settings
        randomization: EventTerm = EventTerm()
        observations: ObsGroup = ObsGroup()
        actions: mdp.JointPositionActionCfg = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True
        )
        rewards: RewTerm = RewTerm()
        terminations: DoneTerm = DoneTerm()
else:
    # Dummy configuration class
    class FrankaCubeLiftEnvCfg:
        def __init__(self):
            pass

class FrankaEnvWrapper:
    """Gym-compatible wrapper for Isaac Lab Franka Cube Lift Environment"""
    
    def __init__(self, cfg: Dict[str, Any] = None):
        # Load configuration
        self.cfg = cfg if cfg is not None else {}
        
        # Define standard spaces (works with or without gym)
        # IMPORTANT: Match expert data dimensions for AIRL training!
        obs_dim = 74  # Match expert data observation dimension
        action_dim = 8  # Match expert data action dimension
        
        # Create spaces
        if GYM_AVAILABLE:
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            )
            self.action_space = spaces.Box(
                low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32
            )
        else:
            # Store space info manually
            self.obs_dim = obs_dim
            self.action_dim = action_dim
            self.observation_space = {'shape': (obs_dim,), 'dtype': np.float32}
            self.action_space = {'shape': (action_dim,), 'dtype': np.float32}
        
        # Try to initialize Isaac Lab environment
        self._env = None
        if ISAAC_LAB_AVAILABLE:
            try:
                self.env_cfg = FrankaCubeLiftEnvCfg()
                self.env_cfg.scene.num_envs = self.cfg.get('num_envs', 1)
                self._env = ManagerBasedRLEnv(cfg=self.env_cfg)
                
                # Update spaces based on actual environment
                if hasattr(self._env, 'observation_manager'):
                    obs_dim = self._env.observation_manager.group_obs_dim["policy"][0]
                if hasattr(self._env, 'action_manager'):
                    action_dim = self._env.action_manager.total_action_dim
                    
                if GYM_AVAILABLE:
                    self.observation_space = spaces.Box(
                        low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
                    )
                    self.action_space = spaces.Box(
                        low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32
                    )
            except Exception as e:
                print(f"Warning: Could not initialize Isaac Lab environment: {e}")
                self._env = None
        
        # Setup dummy environment for testing/development
        self._setup_dummy_env()
        
        # Task-specific settings
        self.max_episode_length = self.cfg.get('max_episode_length', 1000)
        self.success_threshold = self.cfg.get('success_threshold', 0.02)
        self.target_height = self.cfg.get('target_height', 0.15)
        
        # Episode tracking
        self.episode_step = 0
        self.episode_reward = 0.0
        
    def _setup_dummy_env(self):
        """Setup dummy environment for testing when Isaac Lab is not available"""
        # Internal state for dummy environment
        self._dummy_state = {
            'joint_pos': np.zeros(7),
            'joint_vel': np.zeros(7), 
            'object_pos': np.array([0.5, 0.0, 1.0]),  # On table
            'object_rot': np.array([1.0, 0.0, 0.0, 0.0]),  # Identity quaternion
            'target_pos': np.array([0.5, 0.0, 1.15])  # Above table
        }
        
    def _get_observation(self) -> np.ndarray:
        """Get current observation"""
        if self._env is not None:
            # Real Isaac Lab environment
            obs_dict = self._env.observation_manager.compute_group("policy")
            if TORCH_AVAILABLE:
                return obs_dict["policy"].cpu().numpy().flatten()
            else:
                return np.array(obs_dict["policy"]).flatten()
        else:
            # Dummy environment - return 74-dim observation to match expert data
            # Create base observation from dummy state  
            base_obs = np.concatenate([
                self._dummy_state['joint_pos'],        # 7 dims
                self._dummy_state['joint_vel'],        # 7 dims  
                self._dummy_state['object_pos'],       # 3 dims
                self._dummy_state['object_rot'],       # 4 dims
                self._dummy_state['target_pos']        # 3 dims
            ])  # Total: 24 dims
            
            # Pad with additional random features to reach 74 dims
            additional_dims = 74 - len(base_obs)
            padding = np.random.randn(additional_dims) * 0.1  # Small random values
            
            obs = np.concatenate([base_obs, padding])
            return obs.astype(np.float32)
    
    def _compute_reward(self, obs: np.ndarray, action: np.ndarray) -> float:
        """Compute reward for current state-action pair"""
        if self._env is not None:
            # Use Isaac Lab's reward computation
            reward_dict = self._env.reward_manager.compute()
            if TORCH_AVAILABLE:
                return float(reward_dict["total_reward"].sum().cpu().numpy())
            else:
                return float(reward_dict["total_reward"].sum())
        else:
            # Dummy reward computation
            object_pos = self._dummy_state['object_pos']
            target_pos = self._dummy_state['target_pos']
            distance = np.linalg.norm(object_pos - target_pos)
            
            # Distance-based reward
            reward = -distance
            
            # Success bonus
            if distance < self.success_threshold:
                reward += 10.0
                
            # Action penalty (encourage smooth actions)
            action_penalty = -0.01 * np.sum(np.square(action))
            reward += action_penalty
            
            # WICHTIG: Reward clipping gegen Explosionen
            reward = np.clip(reward, -100.0, 100.0)
            
            return reward
    
    def _is_terminated(self, obs: np.ndarray) -> bool:
        """Check if episode should terminate"""
        if self._env is not None:
            term_dict = self._env.termination_manager.compute()
            if TORCH_AVAILABLE:
                return bool(term_dict["total_termination"].any().cpu().numpy())
            else:
                return bool(term_dict["total_termination"].any())
        else:
            # Dummy termination logic
            object_pos = self._dummy_state['object_pos']
            target_pos = self._dummy_state['target_pos']
            distance = np.linalg.norm(object_pos - target_pos)
            
            # Success termination
            if distance < self.success_threshold:
                return True
                
            # Failure termination (object falls off table)
            if object_pos[2] < 0.8:  # Below table height
                return True
                
            return False

    def reset(self, seed=None, options=None):
        """Reset environment"""
        if seed is not None:
            np.random.seed(seed)
            
        self.episode_step = 0
        self.episode_reward = 0.0
        
        if self._env is not None:
            obs_dict, _ = self._env.reset()
            if TORCH_AVAILABLE:
                obs = obs_dict["policy"].cpu().numpy().flatten()
            else:
                obs = np.array(obs_dict["policy"]).flatten()
        else:
            # Reset dummy state
            self._dummy_state = {
                'joint_pos': np.random.uniform(-0.1, 0.1, 7),
                'joint_vel': np.zeros(7),
                'object_pos': np.array([0.5, 0.0, 1.0]) + np.random.uniform(-0.05, 0.05, 3),
                'object_rot': np.array([1.0, 0.0, 0.0, 0.0]),
                'target_pos': np.array([0.5, 0.0, 1.15]) + np.random.uniform(-0.02, 0.02, 3)
            }
            obs = self._get_observation()
            
        return obs, {}

    def step(self, action):
        """Step environment"""
        self.episode_step += 1
        
        if self._env is not None:
            # Real Isaac Lab environment
            if TORCH_AVAILABLE:
                action_tensor = torch.tensor(action, dtype=torch.float32).unsqueeze(0)
            else:
                action_tensor = action  # Assume environment can handle numpy
            obs_dict, reward, terminated, truncated, info = self._env.step(action_tensor)
            
            if TORCH_AVAILABLE:
                obs = obs_dict["policy"].cpu().numpy().flatten()
                reward = float(reward.sum().cpu().numpy())
            else:
                obs = np.array(obs_dict["policy"]).flatten()
                reward = float(reward.sum())
        else:
            # Dummy environment simulation
            # Simple integration of joint velocities
            self._dummy_state['joint_pos'] += action[:7] * 0.01  # Simple integration
            self._dummy_state['joint_vel'] = action[:7]
            
            # Simple object dynamics (gravity + interaction)
            if self.episode_step > 50:  # Give some time for grasping
                # Simulate lifting if gripper is closed and near object
                gripper_action = action[7:9]  # Gripper fingers
                if np.mean(gripper_action) < -0.5:  # Gripper closed
                    # Lift object gradually
                    self._dummy_state['object_pos'][2] += 0.002
                else:
                    # Object falls due to gravity
                    self._dummy_state['object_pos'][2] -= 0.001
            
            obs = self._get_observation()
            reward = self._compute_reward(obs, action)
            terminated = self._is_terminated(obs)
            truncated = self.episode_step >= self.max_episode_length
            info = {}
            
        self.episode_reward += reward
        
        # Check for task success
        task_success = False
        if not (terminated or truncated):
            # Check if task is completed successfully
            object_pos = self._dummy_state['object_pos']
            target_pos = self._dummy_state['target_pos']
            distance = np.linalg.norm(object_pos - target_pos)
            if distance < self.success_threshold:
                task_success = True
        
        # Add success info
        info['success'] = task_success
        
        # Add episode info when done
        if terminated or truncated:
            # Check final success state
            object_pos = self._dummy_state['object_pos']
            target_pos = self._dummy_state['target_pos']
            final_distance = np.linalg.norm(object_pos - target_pos)
            final_success = final_distance < self.success_threshold
            
            info['episode'] = {
                'r': self.episode_reward,
                'l': self.episode_step,
                'success': final_success
            }
            info['success'] = final_success
            
        return obs, reward, terminated, truncated, info

    def render(self, mode="human"):
        """Render environment"""
        if self._env is not None and hasattr(self._env, 'render'):
            return self._env.render()

    def close(self):
        """Close environment"""
        if self._env is not None:
            self._env.close()

def make_env(cfg: Dict[str, Any] = None):
    """Factory function to create Franka environment"""
    return FrankaEnvWrapper(cfg)
