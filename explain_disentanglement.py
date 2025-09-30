"""
AIRL Disentanglement Explanation
Clarify the confusion: Disentanglement ≠ State Independence
"""

import numpy as np
import matplotlib.pyplot as plt

def explain_airl_disentanglement():
    """
    Explain what disentanglement actually means in AIRL
    """
    
    print("🧠 AIRL DISENTANGLEMENT EXPLAINED")
    print("=" * 50)
    
    print("❌ COMMON MISCONCEPTION:")
    print("   'Disentanglement = r independent of state s'")
    print("   → This is WRONG for AIRL!")
    
    print("\n✅ ACTUAL AIRL DISENTANGLEMENT:")
    print("   'Disentanglement = r independent of POLICY π'")
    print("   → We remove policy bias, NOT state dependence!")
    
    print("\n🎯 THE KEY DIFFERENCE:")
    
    print("\n1. TRADITIONAL GAIL (NOT disentangled):")
    print("   D(s,a) = discriminator_output")
    print("   → Learns reward + policy bias together")
    print("   → Cannot separate true reward from policy preferences")
    
    print("\n2. AIRL (Disentangled):")
    print("   f(s,a,s') = r(s,a) + γφ(s') - φ(s)")
    print("   policy_logits = f(s,a,s') - log π(a|s)")
    print("   → Subtracts policy probability!")
    print("   → This removes POLICY bias, not state dependence!")
    
    return True

def visualize_disentanglement():
    """
    Visual comparison of entangled vs disentangled reward learning
    """
    
    # Simulate expert and policy actions in different states
    states = np.linspace(-2, 2, 100)
    expert_actions = 0.5 * states + 0.1 * np.random.randn(100)  # Expert policy
    random_actions = np.random.uniform(-2, 2, 100)  # Random policy
    
    # True reward function (state-action dependent but policy-independent)
    def true_reward(s, a):
        return -(s - a)**2  # Optimal action is a = s
    
    # Policy-biased reward (what traditional GAIL learns)
    def biased_reward(s, a, policy_bias=0.3):
        true_r = true_reward(s, a)
        # Add policy bias - rewards actions similar to expert
        bias = -policy_bias * (a - 0.5*s)**2
        return true_r + bias
    
    # AIRL learns the true reward (disentangled from policy)
    def airl_reward(s, a):
        return true_reward(s, a)  # Policy bias removed by log π(a|s) subtraction
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Expert vs Random actions
    axes[0, 0].scatter(states, expert_actions, alpha=0.6, label='Expert Actions', s=20)
    axes[0, 0].scatter(states[:50], random_actions[:50], alpha=0.6, label='Random Policy', s=20)
    axes[0, 0].plot(states, states, 'r--', alpha=0.8, label='Optimal: a=s')
    axes[0, 0].set_title('Expert vs Policy Actions')
    axes[0, 0].set_xlabel('State s')
    axes[0, 0].set_ylabel('Action a')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: True reward function r(s,a)
    S, A = np.meshgrid(states, states)
    R_true = np.zeros_like(S)
    for i in range(len(states)):
        for j in range(len(states)):
            R_true[i, j] = true_reward(S[i, j], A[i, j])
    
    im1 = axes[0, 1].imshow(R_true, extent=[-2, 2, -2, 2], origin='lower', aspect='auto', cmap='RdYlGn')
    axes[0, 1].set_title('True Reward r(s,a)\n(What AIRL should learn)')
    axes[0, 1].set_xlabel('State s')
    axes[0, 1].set_ylabel('Action a')
    plt.colorbar(im1, ax=axes[0, 1])
    
    # Plot 3: Policy-biased reward (traditional GAIL)
    R_biased = np.zeros_like(S)
    for i in range(len(states)):
        for j in range(len(states)):
            R_biased[i, j] = biased_reward(S[i, j], A[i, j])
    
    im2 = axes[1, 0].imshow(R_biased, extent=[-2, 2, -2, 2], origin='lower', aspect='auto', cmap='RdYlGn')
    axes[1, 0].set_title('Policy-Biased Reward\n(What traditional GAIL learns)')
    axes[1, 0].set_xlabel('State s')
    axes[1, 0].set_ylabel('Action a')
    plt.colorbar(im2, ax=axes[1, 0])
    
    # Plot 4: Show the disentanglement process
    sample_states = [-1, 0, 1]
    actions = np.linspace(-2, 2, 50)
    
    for i, s in enumerate(sample_states):
        true_rewards = [true_reward(s, a) for a in actions]
        biased_rewards = [biased_reward(s, a) for a in actions]
        
        label_true = f'True r(s={s},a)' if i == 0 else None
        label_biased = f'Biased r(s={s},a)' if i == 0 else None
        
        axes[1, 1].plot(actions, true_rewards, '-', linewidth=2, 
                       color=f'C{i}', label=label_true)
        axes[1, 1].plot(actions, biased_rewards, '--', linewidth=2, 
                       color=f'C{i}', alpha=0.7, label=label_biased)
    
    axes[1, 1].set_title('AIRL Disentanglement Effect')
    axes[1, 1].set_xlabel('Action a')
    axes[1, 1].set_ylabel('Reward')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/tmp/airl_disentanglement_explanation.png', dpi=300, bbox_inches='tight')
    plt.show()

def mathematical_explanation():
    """Mathematical explanation of AIRL disentanglement"""
    
    print("\n🔢 MATHEMATICAL EXPLANATION:")
    print("=" * 40)
    
    print("Traditional GAIL:")
    print("  D(s,a) tries to distinguish expert from policy")
    print("  → Learns reward + policy preferences mixed together")
    print("  → Cannot separate r(s,a) from π(a|s) influence")
    
    print("\nAIRL Disentanglement:")
    print("  f(s,a,s') = r(s,a) + γφ(s') - φ(s)  [discriminator output]")
    print("  ")
    print("  For expert data:")
    print("    expert_logits = f(s_expert, a_expert, s'_expert)")
    print("  ")
    print("  For policy data:")
    print("    policy_logits = f(s_policy, a_policy, s'_policy) - log π(a_policy|s_policy)")
    print("                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^   ^^^^^^^^^^^^^^^^^^^^^")
    print("                    Raw discriminator output         POLICY BIAS REMOVED!")
    print("  ")
    print("  This subtraction removes policy-specific bias!")
    
    print("\n🎯 WHY THIS WORKS:")
    print("  • r(s,a) can still depend on state s (we want this!)")
    print("  • But r(s,a) is now independent of policy π")
    print("  • The learned reward generalizes to other policies")
    
    print("\n✅ RESULT:")
    print("  • r(s,a) = true environment reward (state-action dependent)")
    print("  • r(s,a) ≠ policy-biased reward")
    print("  • Can use r(s,a) to train different policies!")

def main():
    print("🚨 AIRL DISENTANGLEMENT CLARIFICATION")
    print("=" * 50)
    
    # Explain the concept
    explain_airl_disentanglement()
    
    # Mathematical explanation
    mathematical_explanation()
    
    # Visual explanation
    print("\n📊 Generating visualization...")
    visualize_disentanglement()
    
    print("\n📋 SUMMARY:")
    print("  ❌ Disentanglement ≠ 'r independent of state s'")
    print("  ✅ Disentanglement = 'r independent of policy π'")
    print("  🎯 We want r(s,a) - reward that depends on state AND action")
    print("  🚫 We don't want policy bias in the learned reward")
    
    print(f"\n📊 Visualization saved: /tmp/airl_disentanglement_explanation.png")

if __name__ == '__main__':
    main()
