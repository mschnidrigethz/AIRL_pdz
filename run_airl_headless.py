#!/usr/bin/env python3
"""
Isaac Lab AIRL Training Script
This script runs AIRL training using Isaac Lab's Python environment without requiring Isaac Sim GUI
"""

import os
import sys
import subprocess
import argparse

def main():
    """Run AIRL training in Isaac Lab's headless mode"""
    
    # Isaac Lab directory
    isaac_lab_dir = "/home/chris/IsaacLab_mik/IsaacLab"
    airl_project_dir = "/home/chris/IsaacLab_mik/Projects/airl_franka"
    
    if not os.path.exists(isaac_lab_dir):
        print(f"❌ Isaac Lab directory not found: {isaac_lab_dir}")
        return 1
        
    if not os.path.exists(airl_project_dir):
        print(f"❌ AIRL project directory not found: {airl_project_dir}")
        return 1
    
    print("🚀 Starting AIRL Training with Isaac Lab (Headless Mode)")
    print(f"Isaac Lab: {isaac_lab_dir}")
    print(f"AIRL Project: {airl_project_dir}")
    print("="*60)
    
    # Set environment variables
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{airl_project_dir}:{env.get('PYTHONPATH', '')}"
    env["ISAAC_SIM_HEADLESS"] = "1"  # Force headless mode
    env["DISPLAY"] = ""  # No display needed
    
    # Change to Isaac Lab directory
    os.chdir(isaac_lab_dir)
    
    # Run AIRL training using Isaac Lab's python environment
    cmd = [
        "./isaaclab.sh", "-p", 
        f"{airl_project_dir}/train_airl.py",
        "--headless"  # Force headless mode
    ]
    
    print(f"Executing: {' '.join(cmd)}")
    print("="*60)
    
    try:
        result = subprocess.run(cmd, env=env, cwd=isaac_lab_dir)
        return result.returncode
    except KeyboardInterrupt:
        print("\n🛑 Training interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Error running AIRL training: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())