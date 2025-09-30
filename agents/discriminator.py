import torch
import torch.nn as nn

class AIRLDiscriminator(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_size=256, gamma=0.99):
        super().__init__()
        self.gamma = gamma
        
        # 🎯 REWARD NETWORK: r(s,a) - State-Action Dependent Reward
        # Takes concatenated [state, action] as input
        # This learns the TRUE reward function we want to extract!
        self.reward_net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden_size),  # [state, action] -> hidden
            nn.ReLU(),
            nn.Linear(hidden_size, 1)  # hidden -> scalar reward r(s,a)
        )
        
        # 📐 POTENTIAL NETWORK: φ(s) - State-Only Shaping Potential  
        # Takes only state as input (for reward shaping)
        # This doesn't affect optimal policy but helps training
        self.potential_net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),  # state -> hidden
            nn.ReLU(), 
            nn.Linear(hidden_size, 1)  # hidden -> scalar potential φ(s)
        )
    
    def forward(self, obs, acts, next_obs):
        """
        AIRL Discriminator forward pass
        
        Args:
            obs: current state s
            acts: action a  
            next_obs: next state s'
            
        Returns:
            f: AIRL advantage = r(s,a) + γφ(s') - φ(s)
        """
        # Reward component: r(s,a) - depends on BOTH state and action
        g = self.reward_net(torch.cat([obs, acts], dim=-1)).squeeze(-1)  # r(s,a)
        
        # Potential shaping components: φ(s) and φ(s') - only depend on states
        h = self.potential_net(obs).squeeze(-1)        # φ(s)
        h_next = self.potential_net(next_obs).squeeze(-1)  # φ(s')
        
        # AIRL advantage function (what discriminator actually learns)
        f = g + self.gamma * h_next - h  # r(s,a) + γφ(s') - φ(s)
        return f
    
    def get_reward(self, obs, acts):
        """
        Extract the learned reward function r(s,a)
        
        This is the MAIN GOAL of AIRL - extracting this reward function!
        
        Args:
            obs: state s
            acts: action a
            
        Returns:
            reward: r(s,a) - the learned reward function
        """
        with torch.no_grad():
            # Only the reward network output - this is r(s,a)!
            g = self.reward_net(torch.cat([obs, acts], dim=-1)).squeeze(-1)
        return g  # This is the reward function we extract for RL!
