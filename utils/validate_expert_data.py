"""
Validate Expert Data
Check if expert data is properly formatted and contains valid trajectories
"""

import h5py
import numpy as np
import argparse
import sys
from pathlib import Path

def validate_expert_data(file_path):
    """Validate expert demonstration data"""
    print(f"Validating expert data: {file_path}")
    
    if not Path(file_path).exists():
        print(f"❌ File not found: {file_path}")
        return False
    
    try:
        with h5py.File(file_path, 'r') as f:
            print(f"📁 File opened successfully")
            print(f"📊 Available datasets: {list(f.keys())}")
            
            # Check required datasets
            required_datasets = ['obs', 'acts', 'next_obs']
            missing_datasets = []
            
            for dataset in required_datasets:
                if dataset not in f:
                    missing_datasets.append(dataset)
                    
            if missing_datasets:
                print(f"❌ Missing required datasets: {missing_datasets}")
                return False
                
            # Load and validate data
            obs = f['obs'][:]
            acts = f['acts'][:]
            next_obs = f['next_obs'][:]
            
            print(f"✅ All required datasets found")
            print(f"📈 Data shapes:")
            print(f"   - Observations: {obs.shape}")
            print(f"   - Actions: {acts.shape}")  
            print(f"   - Next Observations: {next_obs.shape}")
            
            # Validate shapes consistency
            if obs.shape[0] != acts.shape[0] or obs.shape[0] != next_obs.shape[0]:
                print(f"❌ Inconsistent number of samples across datasets")
                return False
                
            if obs.shape[1:] != next_obs.shape[1:]:
                print(f"❌ Observation and next_observation shapes don't match")
                return False
                
            print(f"✅ Shape consistency check passed")
            
            # Check for valid ranges
            print(f"📊 Data statistics:")
            print(f"   - Observations: min={obs.min():.3f}, max={obs.max():.3f}, mean={obs.mean():.3f}")
            print(f"   - Actions: min={acts.min():.3f}, max={acts.max():.3f}, mean={acts.mean():.3f}")
            
            # Check for NaN or infinite values
            if np.any(np.isnan(obs)) or np.any(np.isnan(acts)) or np.any(np.isnan(next_obs)):
                print(f"❌ Found NaN values in data")
                return False
                
            if np.any(np.isinf(obs)) or np.any(np.isinf(acts)) or np.any(np.isinf(next_obs)):
                print(f"❌ Found infinite values in data")
                return False
                
            print(f"✅ No NaN or infinite values found")
            
            # Check trajectory structure (optional)
            if 'episode_starts' in f:
                episode_starts = f['episode_starts'][:]
                num_episodes = np.sum(episode_starts)
                print(f"📈 Found {num_episodes} episodes")
                
            # Sample some data points for manual inspection
            print(f"🔍 Sample data points:")
            sample_indices = np.random.choice(obs.shape[0], min(3, obs.shape[0]), replace=False)
            for i, idx in enumerate(sample_indices):
                print(f"   Sample {i+1}:")
                print(f"     - Obs: {obs[idx][:5]}... (showing first 5 dims)")
                print(f"     - Act: {acts[idx]}")
                print(f"     - Next Obs: {next_obs[idx][:5]}... (showing first 5 dims)")
            
            print(f"✅ Expert data validation completed successfully!")
            print(f"📊 Summary: {obs.shape[0]} transitions, {obs.shape[1]}D observations, {acts.shape[1]}D actions")
            return True
            
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return False

def create_dummy_expert_data(output_path, num_samples=1000, obs_dim=24, act_dim=9):
    """Create dummy expert data for testing"""
    print(f"Creating dummy expert data: {output_path}")
    
    # Generate dummy data
    obs = np.random.randn(num_samples, obs_dim).astype(np.float32)
    acts = np.random.uniform(-1, 1, (num_samples, act_dim)).astype(np.float32)
    next_obs = obs + np.random.randn(num_samples, obs_dim) * 0.1  # Small state changes
    
    # Add some structure to make it more realistic
    # Joint positions slowly change
    obs[:, :7] = np.cumsum(np.random.randn(num_samples, 7) * 0.01, axis=0)
    next_obs[:, :7] = obs[:, :7] + acts[:, :7] * 0.01
    
    # Object position changes based on gripper actions
    for i in range(1, num_samples):
        # If gripper is closed (negative values), lift object
        if np.mean(acts[i-1, 7:9]) < -0.5:
            next_obs[i-1, 14:17] = obs[i-1, 14:17] + np.array([0, 0, 0.001])  # Lift up
        obs[i, 14:17] = next_obs[i-1, 14:17]  # State continuity
    
    # Create episode boundaries
    episode_starts = np.zeros(num_samples, dtype=bool)
    episode_starts[0] = True
    episode_length = 200
    for i in range(episode_length, num_samples, episode_length):
        episode_starts[i] = True
    
    # Save to HDF5
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with h5py.File(output_path, 'w') as f:
        f.create_dataset('obs', data=obs)
        f.create_dataset('acts', data=acts)
        f.create_dataset('next_obs', data=next_obs)
        f.create_dataset('episode_starts', data=episode_starts)
        
        # Add metadata
        f.attrs['num_samples'] = num_samples
        f.attrs['obs_dim'] = obs_dim
        f.attrs['act_dim'] = act_dim
        f.attrs['num_episodes'] = np.sum(episode_starts)
        f.attrs['description'] = 'Dummy expert data for testing AIRL training'
    
    print(f"✅ Dummy expert data created successfully!")
    print(f"📊 {num_samples} samples, {np.sum(episode_starts)} episodes")
    return True

def main():
    parser = argparse.ArgumentParser(description='Validate expert demonstration data')
    parser.add_argument('--file', type=str, required=True, help='Path to expert data HDF5 file')
    parser.add_argument('--create-dummy', action='store_true', help='Create dummy data if file not found')
    parser.add_argument('--num-samples', type=int, default=1000, help='Number of samples for dummy data')
    
    args = parser.parse_args()
    
    file_path = Path(args.file)
    
    if not file_path.exists():
        if args.create_dummy:
            print(f"File not found. Creating dummy data...")
            create_dummy_expert_data(file_path, num_samples=args.num_samples)
        else:
            print(f"❌ File not found: {file_path}")
            print("Use --create-dummy to create dummy data for testing")
            sys.exit(1)
    
    success = validate_expert_data(file_path)
    
    if success:
        print(f"🎉 Expert data is ready for training!")
        sys.exit(0)
    else:
        print(f"💥 Expert data validation failed!")
        sys.exit(1)

if __name__ == '__main__':
    main()
