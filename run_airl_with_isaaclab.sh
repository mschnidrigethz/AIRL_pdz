#!/bin/bash
# run_airl_with_isaaclab.sh
# Script to run AIRL training with Isaac Lab environment

# Navigate to Isaac Lab directory
cd /home/chris/IsaacLab_mik/IsaacLab

# Set up Isaac Lab environment first
export PYTHONPATH="$PWD/source:$PYTHONPATH"

# Run AIRL training using Isaac Lab's shell wrapper
# Use absolute paths to avoid path issues
./isaaclab.sh -p /home/chris/IsaacLab_mik/Projects/airl_franka/train_airl.py \
    --config /home/chris/IsaacLab_mik/Projects/airl_franka/config.yaml

echo "AIRL training completed!"
