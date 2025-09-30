import torch

def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    advantages = torch.zeros_like(rewards)
    last_adv = 0
    for t in reversed(range(len(rewards))):
        mask = 1.0 - dones[t]
        delta = rewards[t] + gamma * values[t + 1] * mask - values[t]
        advantages[t] = last_adv = delta + gamma * lam * mask * last_adv
    returns = advantages + values[:-1]
    return advantages, returns
