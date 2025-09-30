#!/bin/bash
# setup_isaaclab_env.sh  
# Set up environment variables to run AIRL with Isaac Lab

# Set Isaac Lab environment variables
export ISAAC_SIM_PATH="$(which isaacsim | xargs dirname | xargs dirname)"
export ISAACSIM_PATH="$ISAAC_SIM_PATH"
export ISAACSIM_PYTHON_EXE="$ISAAC_SIM_PATH/python.sh"

# Add Isaac Lab to Python path
export PYTHONPATH="/home/chris/IsaacLab_mik/IsaacLab/source:$PYTHONPATH"

echo "Isaac Lab environment configured!"
echo "ISAAC_SIM_PATH: $ISAAC_SIM_PATH"
echo "PYTHONPATH: $PYTHONPATH"
echo ""
echo "Now you can run:"
echo "cd /home/chris/IsaacLab_mik/Projects/airl_franka"
echo "python3 train_airl.py --config config.yaml"
