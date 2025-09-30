import torch
import torch.nn as nn

class Policy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_size=256):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.mean_layer = nn.Linear(hidden_size, act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim))

        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, obs):
        x = self.actor(obs)
        mean = self.mean_layer(x)
        std = torch.exp(self.log_std)
        value = self.critic(obs)
        return mean, std, value

    def get_value(self, obs):
        return self.critic(obs).squeeze(-1)
