"""
AIRL Franka Package
Main package for AIRL training with Franka robot in Isaac Lab
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from .agents.policy import Policy
from .agents.discriminator import AIRLDiscriminator
# Environment import will be done directly in train_airl.py
from .utils.replay_buffer import ReplayBuffer
from .utils.gae import compute_gae

__all__ = [
    'Policy',
    'AIRLDiscriminator',
    'ReplayBuffer',
    'compute_gae'
]
