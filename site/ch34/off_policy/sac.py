"""
Chapter 34.4.3: Soft Actor-Critic (SAC) - Core Concepts
=========================================================
Demonstrates SAC's maximum entropy framework, temperature
tuning, and squashed Gaussian policy.
"""

import torch
import torch.nn as nn
from torch.distributions import Normal
import numpy as np


def demonstrate_max_entropy():
    """Show how entropy bonus affects optimal policy."""
    print("=" * 60)
    print("Maximum Entropy Policy Demonstration")
    print("=" * 60)
    
    # Simple 3-action MDP, single state
    q_values = torch.tensor([1.0, 0.8, 0.5])
    
    temperatures = [0.01, 0.1, 0.5, 1.0, 5.0]
    
    print(f"\nQ-values: {q_values.numpy()}")
    print(f"\n{'α (temp)':>10} {'π(a1)':>8} {'π(a2)':>8} {'π(a3)':>8} {'H(π)':>8} {'E[Q]':>8} {'E[Q]+αH':>10}")
    print("-" * 65)
    
    for alpha in temperatures:
        logits = q_values / alpha
        probs = torch.softmax(logits, dim=0)
        entropy = -(probs * probs.log()).sum()
        expected_q = (probs * q_values).sum()
        objective = expected_q + alpha * entropy
        
        print(f"{alpha:>10.2f} {probs[0]:>8.4f} {probs[1]:>8.4f} {probs[2]:>8.4f} "
              f"{entropy:>8.4f} {expected_q:>8.4f} {objective:>10.4f}")


def demonstrate_squashed_gaussian():
    """Show squashed Gaussian sampling and log-prob correction."""
    print("\n" + "=" * 60)
    print("Squashed Gaussian Policy")
    print("=" * 60)
    
    mu = torch.tensor([0.5])
    log_std = torch.tensor([-0.5])
    std = log_std.exp()
    
    dist = Normal(mu, std)
    n_samples = 10000
    
    # Raw Gaussian samples
    u = dist.sample((n_samples,))
    
    # Squashed samples
    a = torch.tanh(u)
    
    # Log probability with correction
    log_prob_gaussian = dist.log_prob(u)
    log_prob_correction = torch.log(1 - a.pow(2) + 1e-6)
    log_prob_squashed = log_prob_gaussian - log_prob_correction
    
    print(f"\nGaussian: μ={mu.item():.3f}, σ={std.item():.3f}")
    print(f"Raw samples u:     mean={u.mean():.3f}, std={u.std():.3f}, range=[{u.min():.3f}, {u.max():.3f}]")
    print(f"Squashed a=tanh(u): mean={a.mean():.3f}, std={a.std():.3f}, range=[{a.min():.3f}, {a.max():.3f}]")
    print(f"Log prob (gaussian): mean={log_prob_gaussian.mean():.3f}")
    print(f"Log prob (squashed): mean={log_prob_squashed.mean():.3f}")
    print(f"Correction term:     mean={log_prob_correction.mean():.3f}")


def demonstrate_temperature_tuning():
    """Simulate automatic temperature tuning."""
    print("\n" + "=" * 60)
    print("Automatic Temperature Tuning Simulation")
    print("=" * 60)
    
    act_dim = 2
    target_entropy = -act_dim  # Common choice: -dim(A)
    
    log_alpha = torch.tensor(0.0, requires_grad=True)
    alpha_optimizer = torch.optim.Adam([log_alpha], lr=3e-4)
    
    print(f"Target entropy: {target_entropy}")
    print(f"Action dim: {act_dim}\n")
    
    # Simulate entropy values converging
    simulated_entropies = np.concatenate([
        np.linspace(-0.5, -1.5, 500),   # Entropy decreasing
        np.linspace(-1.5, -2.1, 500),   # Overshooting target
        np.linspace(-2.1, -2.0, 500),   # Converging
    ])
    
    alphas = []
    for i, ent in enumerate(simulated_entropies):
        entropy = torch.tensor(ent)
        
        alpha_loss = -(log_alpha.exp() * (entropy + target_entropy).detach())
        
        alpha_optimizer.zero_grad()
        alpha_loss.backward()
        alpha_optimizer.step()
        
        alphas.append(log_alpha.exp().item())
        
        if (i + 1) % 300 == 0:
            print(f"Step {i+1:>5d} | Entropy: {ent:>7.3f} | "
                  f"Target: {target_entropy:>5.1f} | "
                  f"α: {log_alpha.exp().item():>7.4f}")
    
    print(f"\nFinal α: {alphas[-1]:.4f}")
    print("α increases when entropy < target (encourages exploration)")
    print("α decreases when entropy > target (allows exploitation)")


if __name__ == "__main__":
    demonstrate_max_entropy()
    demonstrate_squashed_gaussian()
    demonstrate_temperature_tuning()
