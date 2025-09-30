"""
Visualize the difference between r(s,a) and r(a)
Demonstrate why we need state-action dependent rewards for robotics tasks
"""

import numpy as np
import matplotlib.pyplot as plt

def example_cube_lift_rewards():
    """
    Example showing why we need r(s,a) not just r(a) for cube lift task
    """
    
    # Simulate some states and actions
    np.random.seed(42)
    
    # State components (simplified)
    # Let's say state = [robot_pos, object_pos, target_pos] (simplified to 1D each)
    robot_positions = np.linspace(-1, 1, 50)
    object_positions = np.linspace(0, 2, 50)  # Object height (0=table, 2=lifted)
    target_height = 1.5  # Target height to lift to
    
    # Action: gripper_close_action (-1=open, +1=closed)
    actions = np.linspace(-1, 1, 50)
    
    print("🤖 Cube Lift Task - Reward Function Analysis")
    print("=" * 50)
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: r(a) only - Action-only reward (what you suggested)
    action_only_rewards = []
    for action in actions:
        # Only depends on action - encourages smooth actions
        reward = -0.1 * action**2  # Smooth action penalty
        action_only_rewards.append(reward)
    
    axes[0, 0].plot(actions, action_only_rewards, 'b-', linewidth=2)
    axes[0, 0].set_title('r(a) - Action-Only Reward\n(Not sufficient for robotics!)')
    axes[0, 0].set_xlabel('Gripper Action (-1=open, +1=closed)')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: State component of reward (object height)
    state_rewards = []
    for obj_pos in object_positions:
        # Distance to target height
        reward = -abs(obj_pos - target_height)
        state_rewards.append(reward)
    
    axes[0, 1].plot(object_positions, state_rewards, 'r-', linewidth=2)
    axes[0, 1].set_title('State Component of Reward\n(Object height vs target)')
    axes[0, 1].set_xlabel('Object Height')
    axes[0, 1].set_ylabel('State Reward')
    axes[0, 1].axvline(x=target_height, color='g', linestyle='--', label='Target Height')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: r(s,a) - Combined state-action reward (what AIRL learns)
    X, Y = np.meshgrid(object_positions, actions)
    combined_rewards = np.zeros_like(X)
    
    for i, obj_pos in enumerate(object_positions):
        for j, action in enumerate(actions):
            # State component: distance to target
            state_reward = -abs(obj_pos - target_height)
            
            # Action component: smooth actions + context-dependent
            action_reward = -0.1 * action**2
            
            # Context-dependent: should close gripper when near object, open when lifted
            if obj_pos < 0.5:  # Object on table
                context_reward = action  # Reward closing gripper
            elif obj_pos > target_height - 0.2:  # Object near target
                context_reward = -action  # Reward opening gripper
            else:  # Object being lifted
                context_reward = action  # Keep gripper closed
            
            combined_rewards[j, i] = state_reward + action_reward + 0.5 * context_reward
    
    im = axes[1, 0].imshow(combined_rewards, extent=[0, 2, -1, 1], aspect='auto', origin='lower', cmap='RdYlGn')
    axes[1, 0].set_title('r(s,a) - State-Action Reward\n(What AIRL learns)')
    axes[1, 0].set_xlabel('Object Height (State)')
    axes[1, 0].set_ylabel('Gripper Action')
    axes[1, 0].axvline(x=target_height, color='white', linestyle='--', alpha=0.8)
    plt.colorbar(im, ax=axes[1, 0])
    
    # Plot 4: Comparison - Show why r(s,a) is better
    # Sample scenario: Object at different heights, what should the action be?
    scenarios = [
        (0.2, "Object on table"),
        (1.0, "Object being lifted"), 
        (1.6, "Object at target")
    ]
    
    scenario_actions = actions
    scenario_rewards = {}
    
    for obj_height, label in scenarios:
        rewards = []
        for action in scenario_actions:
            # Same r(s,a) calculation as above
            state_reward = -abs(obj_height - target_height)
            action_reward = -0.1 * action**2
            
            if obj_height < 0.5:
                context_reward = action
            elif obj_height > target_height - 0.2:
                context_reward = -action
            else:
                context_reward = action
                
            total_reward = state_reward + action_reward + 0.5 * context_reward
            rewards.append(total_reward)
        
        scenario_rewards[label] = rewards
        axes[1, 1].plot(scenario_actions, rewards, linewidth=2, label=label)
    
    axes[1, 1].set_title('r(s,a) for Different States\n(Shows context-dependent optimal actions)')
    axes[1, 1].set_xlabel('Gripper Action')
    axes[1, 1].set_ylabel('Reward r(s,a)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/tmp/reward_function_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print key insights
    print("\n🎯 Key Insights:")
    print("1. r(a) only: Cannot capture task-specific requirements")
    print("2. r(s,a): Can learn context-dependent optimal actions")
    print("3. Example: Gripper should CLOSE near table, OPEN at target")
    print("4. This contextual behavior can only be learned with r(s,a)")
    
    print("\n🤖 For Franka Cube Lift Task:")
    print("- State: robot joints, object pos, target pos, contact forces...")
    print("- Action: joint velocities, gripper commands")
    print("- r(s,a): Combines task progress (state) + action smoothness (action)")
    
    return scenario_rewards

def airl_architecture_explanation():
    """Explain AIRL architecture components"""
    
    print("\n🧠 AIRL Architecture Breakdown:")
    print("=" * 40)
    
    print("DISCRIMINATOR (full network):")
    print("  f(s,a,s') = r(s,a) + γφ(s') - φ(s)")
    print("  ├── r(s,a): reward_net([state, action]) → reward")
    print("  ├── φ(s):  potential_net(state) → shaping")
    print("  └── f:     discriminator output (expert vs policy)")
    
    print("\nREWARD FUNCTION (extracted part):")
    print("  r(s,a) = reward_net([state, action])")
    print("  ├── Input: concatenated [state, action]")
    print("  ├── Output: scalar reward value")
    print("  └── This is what we extract for RL!")
    
    print("\nWHY r(s,a) not r(a)?")
    print("  ✅ Task progress depends on STATE (object position)")
    print("  ✅ Action quality depends on CONTEXT (when to grip)")
    print("  ✅ Safety depends on STATE (collision avoidance)")
    print("  ❌ r(a) cannot capture these dependencies")

def main():
    print("🎯 Understanding AIRL Reward Functions")
    print("=" * 50)
    
    # Explain architecture
    airl_architecture_explanation()
    
    # Show visual comparison
    example_cube_lift_rewards()
    
    print(f"\n📊 Visualization saved to: /tmp/reward_function_comparison.png")
    print(f"\n✅ Summary:")
    print(f"- AIRL Discriminator = r(s,a) + γφ(s') - φ(s)")
    print(f"- Reward Function = r(s,a) (the part we extract)")
    print(f"- We use r(s,a) because robotics tasks are context-dependent!")

if __name__ == '__main__':
    main()
