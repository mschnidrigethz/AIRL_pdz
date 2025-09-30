import torch

class ReplayBuffer:
    def __init__(self, max_size, obs_dim, act_dim):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.obs_buf = torch.zeros((max_size, obs_dim), dtype=torch.float32)
        self.acts_buf = torch.zeros((max_size, act_dim), dtype=torch.float32)
        self.rewards_buf = torch.zeros(max_size, dtype=torch.float32)
        self.dones_buf = torch.zeros(max_size, dtype=torch.float32)
        self.values_buf = torch.zeros(max_size, dtype=torch.float32)
        self.log_probs_buf = torch.zeros(max_size, dtype=torch.float32)
        self.next_obs_buf = torch.zeros((max_size, obs_dim), dtype=torch.float32)

    def store(self, obs, act, reward, done, value, log_prob, next_obs):
        self.obs_buf[self.ptr] = torch.as_tensor(obs, dtype=torch.float32)
        self.acts_buf[self.ptr] = torch.as_tensor(act, dtype=torch.float32)
        self.rewards_buf[self.ptr] = torch.as_tensor(reward, dtype=torch.float32)
        self.dones_buf[self.ptr] = torch.as_tensor(float(done), dtype=torch.float32)  # Convert bool to float
        self.values_buf[self.ptr] = torch.as_tensor(value, dtype=torch.float32)
        self.log_probs_buf[self.ptr] = torch.as_tensor(log_prob, dtype=torch.float32)
        self.next_obs_buf[self.ptr] = torch.as_tensor(next_obs, dtype=torch.float32)

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def get(self):
        return dict(
            obs=self.obs_buf[:self.size],
            acts=self.acts_buf[:self.size],
            rewards=self.rewards_buf[:self.size],
            dones=self.dones_buf[:self.size],
            values=self.values_buf[:self.size],
            log_probs=self.log_probs_buf[:self.size],
            next_obs=self.next_obs_buf[:self.size]
        )

    def clear(self):
        self.ptr = 0
        self.size = 0
