"""
33.3.3 V-Trace
===============

V-trace off-policy correction for distributed RL settings.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple


def compute_vtrace(
    values: torch.Tensor,          # V(s_t), shape (T+1,) — includes bootstrap
    rewards: torch.Tensor,         # r_t, shape (T,)
    dones: torch.Tensor,           # done_t, shape (T,)
    target_log_probs: torch.Tensor,   # log π(a_t|s_t), shape (T,)
    behavior_log_probs: torch.Tensor, # log μ(a_t|s_t), shape (T,)
    gamma: float = 0.99,
    rho_bar: float = 1.0,
    c_bar: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute V-trace targets and advantages.
    
    Args:
        values: V(s_t) for t=0..T (T+1 values, last is bootstrap)
        rewards: rewards r_t for t=0..T-1
        dones: done flags for t=0..T-1
        target_log_probs: log π(a_t|s_t) under current policy
        behavior_log_probs: log μ(a_t|s_t) under behavior policy
        gamma: discount factor
        rho_bar: truncation for value correction
        c_bar: truncation for trace propagation
        
    Returns:
        (vs, advantages): V-trace values and policy gradient advantages
    """
    T = len(rewards)
    
    # Importance sampling ratios
    log_ratios = target_log_probs - behavior_log_probs
    ratios = torch.exp(log_ratios)
    
    # Truncated IS ratios
    rho = torch.clamp(ratios, max=rho_bar)
    c = torch.clamp(ratios, max=c_bar)
    
    # TD errors with rho correction
    not_done = 1.0 - dones
    delta_v = rho * (rewards + not_done * gamma * values[1:] - values[:T])
    
    # Compute V-trace targets backwards
    vs_minus_v = torch.zeros(T + 1)
    for t in reversed(range(T)):
        vs_minus_v[t] = delta_v[t] + not_done[t] * gamma * c[t] * vs_minus_v[t + 1]
    
    vs = vs_minus_v[:T] + values[:T]
    
    # Advantages for policy gradient (if needed)
    advantages = rho * (rewards + not_done * gamma * vs[1:].detach() 
                        if T > 1 else rewards - values[:T])
    # Simplified advantages for value-based methods
    advantages = vs - values[:T]
    
    return vs, advantages


def compute_vtrace_batch(
    values_batch: torch.Tensor,          # (B, T+1)
    rewards_batch: torch.Tensor,         # (B, T)
    dones_batch: torch.Tensor,           # (B, T)
    target_log_probs_batch: torch.Tensor,  # (B, T)
    behavior_log_probs_batch: torch.Tensor,  # (B, T)
    gamma: float = 0.99,
    rho_bar: float = 1.0,
    c_bar: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Batch V-trace computation."""
    B, T = rewards_batch.shape
    
    log_ratios = target_log_probs_batch - behavior_log_probs_batch
    ratios = torch.exp(log_ratios)
    rho = torch.clamp(ratios, max=rho_bar)
    c = torch.clamp(ratios, max=c_bar)
    
    not_done = 1.0 - dones_batch
    delta_v = rho * (rewards_batch + not_done * gamma * values_batch[:, 1:] 
                     - values_batch[:, :T])
    
    vs_minus_v = torch.zeros(B, T + 1)
    for t in reversed(range(T)):
        vs_minus_v[:, t] = delta_v[:, t] + not_done[:, t] * gamma * c[:, t] * vs_minus_v[:, t + 1]
    
    vs = vs_minus_v[:, :T] + values_batch[:, :T]
    advantages = vs - values_batch[:, :T]
    
    return vs, advantages


# ---------------------------------------------------------------------------
# V-trace analysis utilities
# ---------------------------------------------------------------------------

def analyze_is_ratios(target_probs: torch.Tensor, behavior_probs: torch.Tensor,
                      rho_bar: float = 1.0, c_bar: float = 1.0):
    """Analyze importance sampling ratio statistics."""
    ratios = target_probs / (behavior_probs + 1e-8)
    rho = torch.clamp(ratios, max=rho_bar)
    c = torch.clamp(ratios, max=c_bar)
    
    return {
        'raw_ratio_mean': ratios.mean().item(),
        'raw_ratio_std': ratios.std().item(),
        'raw_ratio_max': ratios.max().item(),
        'rho_mean': rho.mean().item(),
        'c_mean': c.mean().item(),
        'fraction_clipped_rho': (ratios > rho_bar).float().mean().item(),
        'fraction_clipped_c': (ratios > c_bar).float().mean().item(),
    }


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def demo_vtrace():
    print("=" * 60)
    print("V-Trace Demo")
    print("=" * 60)

    # --- Basic V-trace computation ---
    print("\n--- V-Trace Target Computation ---")
    T = 10
    values = torch.randn(T + 1) * 2 + 5  # V(s_t) for t=0..T
    rewards = torch.ones(T)
    dones = torch.zeros(T)
    
    # Simulate policy lag: behavior is older, slightly different policy
    target_log_probs = torch.zeros(T)  # log(1) = 0 for greedy
    behavior_log_probs = torch.zeros(T) - 0.3  # slightly different
    
    vs, adv = compute_vtrace(values, rewards, dones,
                              target_log_probs, behavior_log_probs)
    print(f"  Values: {values[:5].numpy().round(3)}...")
    print(f"  V-trace targets: {vs[:5].detach().numpy().round(3)}...")
    print(f"  Advantages: {adv[:5].detach().numpy().round(3)}...")

    # --- Effect of truncation thresholds ---
    print("\n--- Effect of Truncation Thresholds ---")
    for rho_bar in [0.5, 1.0, 5.0, float('inf')]:
        for c_bar in [0.5, 1.0, 5.0]:
            vs, _ = compute_vtrace(values, rewards, dones,
                                    target_log_probs, behavior_log_probs,
                                    rho_bar=rho_bar, c_bar=c_bar)
            print(f"  ρ̄={rho_bar:>5}, c̄={c_bar}: "
                  f"target mean={vs.mean():.3f}, std={vs.std():.3f}")

    # --- IS ratio analysis ---
    print("\n--- IS Ratio Analysis ---")
    # Simulate different amounts of policy lag
    for lag_desc, beh_offset in [("Small lag", -0.1), ("Medium lag", -0.5),
                                   ("Large lag", -1.0)]:
        tp = torch.zeros(100)
        bp = torch.zeros(100) + beh_offset
        stats = analyze_is_ratios(tp.exp(), bp.exp())
        print(f"  {lag_desc}:")
        print(f"    Raw ratio: {stats['raw_ratio_mean']:.3f} ± {stats['raw_ratio_std']:.3f}")
        print(f"    Fraction clipped (ρ): {stats['fraction_clipped_rho']:.1%}")

    # --- Batch computation ---
    print("\n--- Batch V-Trace ---")
    B, T = 8, 20
    values_b = torch.randn(B, T + 1)
    rewards_b = torch.randn(B, T)
    dones_b = torch.zeros(B, T)
    tp_b = torch.zeros(B, T)
    bp_b = torch.zeros(B, T) - 0.2
    
    vs_b, adv_b = compute_vtrace_batch(values_b, rewards_b, dones_b, tp_b, bp_b)
    print(f"  Input shapes: values={values_b.shape}, rewards={rewards_b.shape}")
    print(f"  Output shapes: targets={vs_b.shape}, advantages={adv_b.shape}")
    print(f"  Target mean: {vs_b.mean():.3f}")

    # --- Comparison: V-trace vs uncorrected ---
    print("\n--- V-trace vs Uncorrected N-step ---")
    T = 15
    values = torch.ones(T + 1) * 5.0
    rewards = torch.ones(T)
    dones = torch.zeros(T)
    
    # On-policy (no correction needed)
    tp_on = torch.zeros(T)
    bp_on = torch.zeros(T)
    vs_on, _ = compute_vtrace(values, rewards, dones, tp_on, bp_on)
    
    # Off-policy (correction needed)
    tp_off = torch.zeros(T)
    bp_off = torch.zeros(T) - 0.5
    vs_off, _ = compute_vtrace(values, rewards, dones, tp_off, bp_off)
    
    print(f"  On-policy targets:  {vs_on[:5].detach().numpy().round(3)}")
    print(f"  Off-policy targets: {vs_off[:5].detach().numpy().round(3)}")
    print(f"  Correction effect: {(vs_on - vs_off).abs().mean():.4f}")

    print("\nV-trace demo complete!")


if __name__ == "__main__":
    demo_vtrace()
