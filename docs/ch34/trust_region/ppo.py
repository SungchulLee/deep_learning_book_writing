"""
Chapter 34.3.3: PPO - Clipped Objective Demonstration
======================================================
Demonstrates the PPO clipping mechanism and compares
clipped vs unclipped surrogate objectives.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def visualize_ppo_clipping():
    """Visualize the PPO clipped objective for positive and negative advantages."""
    epsilon = 0.2
    ratios = torch.linspace(0.5, 2.0, 300)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for idx, (A, title) in enumerate([(1.0, "Positive Advantage (A=1)"),
                                       (-1.0, "Negative Advantage (A=-1)")]):
        ax = axes[idx]
        
        # Unclipped surrogate
        unclipped = ratios * A
        
        # Clipped surrogate
        clipped_ratios = torch.clamp(ratios, 1 - epsilon, 1 + epsilon)
        clipped = clipped_ratios * A
        
        # PPO objective: min(unclipped, clipped)
        ppo_obj = torch.min(unclipped, clipped)
        
        ax.plot(ratios.numpy(), unclipped.numpy(), "b--", label="Unclipped", alpha=0.7)
        ax.plot(ratios.numpy(), clipped.numpy(), "r--", label="Clipped", alpha=0.7)
        ax.plot(ratios.numpy(), ppo_obj.numpy(), "g-", label="PPO (min)", linewidth=2)
        
        ax.axvline(x=1.0, color="gray", linestyle=":", alpha=0.5)
        ax.axvline(x=1 - epsilon, color="orange", linestyle=":", alpha=0.5, label=f"1-ε={1-epsilon}")
        ax.axvline(x=1 + epsilon, color="orange", linestyle=":", alpha=0.5, label=f"1+ε={1+epsilon}")
        
        ax.set_xlabel("Probability Ratio r(θ)")
        ax.set_ylabel("Objective")
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("/home/claude/ppo_clipping.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved PPO clipping visualization to ppo_clipping.png")


def demonstrate_clip_behavior():
    """Show how clipping affects gradient flow."""
    print("=" * 60)
    print("PPO Clipping Behavior Analysis")
    print("=" * 60)
    
    epsilon = 0.2
    
    # Simulate different ratio/advantage combinations
    scenarios = [
        (1.5, 1.0, "Large ratio, positive advantage"),
        (0.7, 1.0, "Small ratio, positive advantage"),
        (1.5, -1.0, "Large ratio, negative advantage"),
        (0.7, -1.0, "Small ratio, negative advantage"),
        (1.1, 1.0, "Moderate ratio, positive advantage"),
        (0.9, -1.0, "Moderate ratio, negative advantage"),
    ]
    
    print(f"\nε = {epsilon}")
    print(f"{'Scenario':<45} {'Ratio':>6} {'A':>5} {'Unclip':>8} {'Clip':>8} {'PPO':>8} {'Clipped?':>8}")
    print("-" * 95)
    
    for ratio_val, adv_val, desc in scenarios:
        ratio = torch.tensor(ratio_val)
        adv = torch.tensor(adv_val)
        
        unclipped = ratio * adv
        clipped_ratio = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
        clipped = clipped_ratio * adv
        ppo = torch.min(unclipped, clipped)
        is_clipped = (ppo != unclipped).item()
        
        print(
            f"{desc:<45} {ratio_val:>6.2f} {adv_val:>5.1f} "
            f"{unclipped.item():>8.3f} {clipped.item():>8.3f} "
            f"{ppo.item():>8.3f} {'Yes' if is_clipped else 'No':>8}"
        )


def ppo_loss_computation_example():
    """Step-by-step PPO loss computation."""
    print("\n" + "=" * 60)
    print("PPO Loss Computation Example")
    print("=" * 60)
    
    batch_size = 8
    epsilon = 0.2
    value_coef = 0.5
    entropy_coef = 0.01
    
    torch.manual_seed(42)
    
    # Simulated data
    old_log_probs = torch.randn(batch_size) - 1.0  # Typical log prob values
    new_log_probs = old_log_probs + torch.randn(batch_size) * 0.1  # Slightly different
    advantages = torch.randn(batch_size)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    values = torch.randn(batch_size, requires_grad=True)
    returns = values.detach() + torch.randn(batch_size) * 0.5
    entropy = torch.tensor(0.5)
    
    # Compute ratio
    ratio = torch.exp(new_log_probs - old_log_probs)
    
    # Clipped surrogate
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()
    
    # Value loss
    value_loss = nn.functional.mse_loss(values, returns)
    
    # Total loss
    total_loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
    
    print(f"\nBatch size: {batch_size}")
    print(f"Clip epsilon: {epsilon}")
    print(f"\nRatios: {ratio.detach().numpy().round(3)}")
    print(f"Advantages: {advantages.numpy().round(3)}")
    print(f"\nClipped count: {(torch.min(surr1, surr2) != surr1).sum().item()}/{batch_size}")
    print(f"Policy loss: {policy_loss.item():.4f}")
    print(f"Value loss: {value_loss.item():.4f}")
    print(f"Entropy bonus: {entropy.item():.4f}")
    print(f"Total loss: {total_loss.item():.4f}")


def compare_clip_values():
    """Compare PPO with different clipping values."""
    print("\n" + "=" * 60)
    print("Effect of Clipping Parameter ε")
    print("=" * 60)
    
    torch.manual_seed(42)
    batch_size = 1000
    old_log_probs = torch.randn(batch_size) - 1.0
    new_log_probs = old_log_probs + torch.randn(batch_size) * 0.3
    advantages = torch.randn(batch_size)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    ratio = torch.exp(new_log_probs - old_log_probs)
    
    print(f"\nRatio stats: mean={ratio.mean():.3f}, std={ratio.std():.3f}, "
          f"min={ratio.min():.3f}, max={ratio.max():.3f}")
    
    epsilons = [0.05, 0.1, 0.2, 0.3, 0.5]
    print(f"\n{'ε':>6} {'Loss':>10} {'% Clipped':>12} {'Effective KL':>14}")
    print("-" * 44)
    
    for eps in epsilons:
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - eps, 1 + eps) * advantages
        ppo_obj = torch.min(surr1, surr2)
        loss = -ppo_obj.mean()
        pct_clipped = (ppo_obj != surr1).float().mean() * 100
        
        approx_kl = ((ratio - 1) - torch.log(ratio)).mean()
        
        print(f"{eps:>6.2f} {loss.item():>10.4f} {pct_clipped.item():>11.1f}% {approx_kl.item():>14.5f}")


if __name__ == "__main__":
    demonstrate_clip_behavior()
    ppo_loss_computation_example()
    compare_clip_values()
    visualize_ppo_clipping()
