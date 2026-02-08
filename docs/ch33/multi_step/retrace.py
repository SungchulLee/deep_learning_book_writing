"""
33.3.2 Retrace(λ)
==================

Off-policy corrected multi-step returns using Retrace(λ).
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple


def compute_retrace_targets(
    q_values: torch.Tensor,        # Q(s_t, a_t) for t in trajectory, shape (T,)
    next_q_max: torch.Tensor,      # max_a Q(s_{t+1}, a) for t, shape (T,)
    rewards: torch.Tensor,         # r_t, shape (T,)
    dones: torch.Tensor,           # done_t, shape (T,)
    target_policy_probs: torch.Tensor,  # π(a_t|s_t), shape (T,)
    behavior_policy_probs: torch.Tensor,  # μ(a_t|s_t), shape (T,)
    gamma: float = 0.99,
    lambda_: float = 0.95,
) -> torch.Tensor:
    """Compute Retrace(λ) targets for a single trajectory.
    
    Args:
        q_values: Q-values at (s_t, a_t) for each timestep
        next_q_max: max_a' Q(s_{t+1}, a') for each timestep
        rewards: rewards at each timestep
        dones: terminal flags (1.0 = done)
        target_policy_probs: π(a_t | s_t) under target (current) policy
        behavior_policy_probs: μ(a_t | s_t) under behavior (data) policy
        gamma: discount factor
        lambda_: Retrace lambda parameter
        
    Returns:
        Retrace targets, shape (T,)
    """
    T = len(rewards)
    
    # Trace coefficients: c_i = λ * min(1, π/μ)
    is_ratios = target_policy_probs / (behavior_policy_probs + 1e-8)
    c = lambda_ * torch.clamp(is_ratios, max=1.0)
    
    # TD errors: δ_t = r_t + γ * max_a Q(s_{t+1}, a) - Q(s_t, a_t)
    td_errors = rewards + (1 - dones) * gamma * next_q_max - q_values
    
    # Compute Retrace targets backwards
    targets = torch.zeros(T)
    # Q^ret(s_t, a_t) = Q(s_t, a_t) + Σ_{k=t}^{T-1} γ^{k-t} (Π c_i) δ_k
    
    for t in range(T):
        target = q_values[t].item()
        trace_product = 1.0
        for k in range(t, T):
            if k > t:
                trace_product *= c[k].item()
                if dones[k - 1].item() > 0.5:
                    break
            target += (gamma ** (k - t)) * trace_product * td_errors[k].item()
        targets[t] = target
    
    return targets


def compute_retrace_batch(
    q_online: nn.Module,
    q_target: nn.Module,
    trajectories: List[dict],
    gamma: float = 0.99,
    lambda_: float = 0.95,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute Retrace targets for a batch of trajectories.
    
    Args:
        q_online: Online Q-network
        q_target: Target Q-network
        trajectories: List of dicts with keys: states, actions, rewards, 
                      next_states, dones, behavior_probs
        gamma: discount
        lambda_: Retrace lambda
        
    Returns:
        (all_q_values, all_targets) concatenated across trajectories
    """
    all_q = []
    all_targets = []
    
    for traj in trajectories:
        states = torch.FloatTensor(np.array(traj['states']))
        actions = torch.LongTensor(np.array(traj['actions']))
        rewards = torch.FloatTensor(np.array(traj['rewards']))
        next_states = torch.FloatTensor(np.array(traj['next_states']))
        dones = torch.FloatTensor(np.array(traj['dones']))
        behavior_probs = torch.FloatTensor(np.array(traj['behavior_probs']))
        
        with torch.no_grad():
            q_vals = q_target(states).gather(1, actions.unsqueeze(1)).squeeze(1)
            next_q_max = q_target(next_states).max(dim=1)[0]
            
            # Target policy: greedy w.r.t. online network
            online_actions = q_online(states).argmax(dim=1)
            target_probs = (online_actions == actions).float()
        
        targets = compute_retrace_targets(
            q_vals, next_q_max, rewards, dones,
            target_probs, behavior_probs, gamma, lambda_
        )
        
        # Get online Q-values for loss computation
        online_q = q_online(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        all_q.append(online_q)
        all_targets.append(targets)
    
    return torch.cat(all_q), torch.cat(all_targets)


# ---------------------------------------------------------------------------
# Trajectory buffer for Retrace
# ---------------------------------------------------------------------------

class TrajectoryBuffer:
    """Stores complete trajectories for Retrace-style algorithms."""
    
    def __init__(self, max_trajectories: int = 1000):
        self.max_size = max_trajectories
        self.trajectories: List[dict] = []
        self.ptr = 0
    
    def push_trajectory(self, states, actions, rewards, next_states, dones, behavior_probs):
        traj = {
            'states': states, 'actions': actions, 'rewards': rewards,
            'next_states': next_states, 'dones': dones,
            'behavior_probs': behavior_probs,
        }
        if len(self.trajectories) < self.max_size:
            self.trajectories.append(traj)
        else:
            self.trajectories[self.ptr] = traj
        self.ptr = (self.ptr + 1) % self.max_size
    
    def sample(self, n: int) -> List[dict]:
        indices = np.random.choice(len(self.trajectories), min(n, len(self.trajectories)),
                                   replace=False)
        return [self.trajectories[i] for i in indices]
    
    def __len__(self):
        return len(self.trajectories)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def demo_retrace():
    print("=" * 60)
    print("Retrace(λ) Demo")
    print("=" * 60)

    # --- Retrace target computation ---
    print("\n--- Retrace Target Computation ---")
    T = 8
    q_vals = torch.randn(T) * 2 + 5
    next_q_max = torch.randn(T) * 2 + 5
    rewards = torch.ones(T)
    dones = torch.zeros(T)
    dones[-1] = 1.0
    target_probs = torch.ones(T) * 0.8
    behavior_probs = torch.ones(T) * 0.5
    
    for lam in [0.0, 0.5, 0.95, 1.0]:
        targets = compute_retrace_targets(
            q_vals, next_q_max, rewards, dones,
            target_probs, behavior_probs, gamma=0.99, lambda_=lam)
        print(f"  λ={lam:.2f}: targets = {targets[:4].numpy().round(3)}...")
    
    # --- Trace coefficient analysis ---
    print("\n--- Trace Coefficients ---")
    print("  For greedy target policy with ε-greedy behavior (ε=0.1, |A|=4):")
    eps = 0.1
    n_actions = 4
    
    # Greedy action
    mu_greedy = 1 - eps + eps / n_actions
    c_greedy = 0.95 * min(1.0, 1.0 / mu_greedy)
    print(f"    Greedy action: μ={mu_greedy:.3f}, π=1.0, c=λ·min(1,π/μ)={c_greedy:.3f}")
    
    # Non-greedy action
    mu_random = eps / n_actions
    c_random = 0.95 * min(1.0, 0.0 / mu_random)
    print(f"    Non-greedy:    μ={mu_random:.3f}, π=0.0, c=λ·min(1,π/μ)={c_random:.3f}")
    print("    → Trace is cut when behavior took non-greedy action")
    
    # --- Trace length analysis ---
    print("\n--- Effective Trace Length ---")
    for lam in [0.5, 0.8, 0.95, 1.0]:
        for greedy_frac in [0.5, 0.8, 0.95]:
            # Expected trace length before cutoff
            p_continue = lam * greedy_frac
            expected_len = 1.0 / (1.0 - p_continue) if p_continue < 1 else float('inf')
            print(f"    λ={lam}, P(greedy)={greedy_frac}: "
                  f"expected trace = {expected_len:.1f} steps")
    
    # --- Compare with n-step ---
    print("\n--- Retrace vs N-step (with off-policy data) ---")
    T = 10
    q_vals = torch.ones(T) * 5.0
    next_q_max = torch.ones(T) * 5.0
    rewards = torch.ones(T)
    dones = torch.zeros(T)
    
    # Simulate off-policy: some actions were non-greedy
    target_probs = torch.tensor([1, 1, 0, 1, 1, 0, 0, 1, 1, 1], dtype=torch.float32)
    behavior_probs = torch.ones(T) * 0.3
    
    retrace_targets = compute_retrace_targets(
        q_vals, next_q_max, rewards, dones, target_probs, behavior_probs)
    
    # N-step (no correction)
    nstep_targets = compute_retrace_targets(
        q_vals, next_q_max, rewards, dones,
        torch.ones(T), torch.ones(T))  # c_i = λ always
    
    print(f"  Retrace targets: {retrace_targets[:5].numpy().round(3)}")
    print(f"  N-step targets:  {nstep_targets[:5].numpy().round(3)}")
    print(f"  Difference: {(retrace_targets - nstep_targets).abs().mean():.4f}")
    
    print("\nRetrace demo complete!")


if __name__ == "__main__":
    demo_retrace()
