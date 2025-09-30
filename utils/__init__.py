"""
Isaac Lab AIRL Training and Evaluation Scripts
Entry points for package installation
"""

from .validate_expert_data import main as validate_expert_data_main
from .evaluate_policy import main as evaluate_policy_main

__all__ = ['validate_expert_data_main', 'evaluate_policy_main']
