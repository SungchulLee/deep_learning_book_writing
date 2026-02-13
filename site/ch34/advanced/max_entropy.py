"""
Chapter 34.5.1: Maximum Entropy RL
====================================
Demonstrations of soft Bellman equations, soft value iteration,
and entropy-regularized policy optimization.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple


def soft_value_iteration(
    P: np.ndarray, R: np.ndarray, gamma: float = 0.99, alpha: float = 0.1,
    n_iters: int = 100,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Soft value iteration for tabular MDPs.
    
    P : (S, A, S) transition probabilities
    R : (S, A) reward matrix
    Returns: soft Q-values, soft V-values, optimal soft policy
    """
    S, A, _ = P.shape
    Q = np.zeros((S, A))
    
    for _ in range(n_iters):
        V = alpha * np.log(np.sum(np.exp(Q / alpha), axis=1))  # Soft max
        for s in range(S):
            for a in range(A):
                Q[s, a] = R[s, a] + gamma * P[s, a] @ V
    
    # Optimal policy: Boltzmann distribution
    V = alpha * np.log(np.sum(np.exp(Q / alpha), axis=1))
    pi = np.exp((Q - V[:, None]) / alpha)
    pi /= pi.sum(axis=1, keepdims=True)
    
    return Q, V, pi


def demo_soft_vi():
    """Compare standard vs soft value iteration on a grid world."""
    print("=" * 60)
    print("Soft Value Iteration: Standard vs Maximum Entropy")
    print("=" * 60)
    
    # Simple 3-state, 2-action MDP
    S, A = 3, 2
    P = np.zeros((S, A, S))
    # State 0: action 0 → state 1, action 1 → state 2
    P[0, 0, 1] = 1.0
    P[0, 1, 2] = 1.0
    # State 1: both actions → state 0
    P[1, 0, 0] = 1.0
    P[1, 1, 0] = 1.0
    # State 2: both actions → state 0
    P[2, 0, 0] = 1.0
    P[2, 1, 0] = 1.0
    
    R = np.array([
        [0.0, 0.0],   # State 0
        [1.0, 0.5],   # State 1: action 0 is better
        [0.8, 0.9],   # State 2: action 1 is slightly better
    ])
    
    print("\nReward matrix:")
    print(f"  State 0: R(a0)={R[0,0]:.1f}, R(a1)={R[0,1]:.1f}")
    print(f"  State 1: R(a0)={R[1,0]:.1f}, R(a1)={R[1,1]:.1f}")
    print(f"  State 2: R(a0)={R[2,0]:.1f}, R(a1)={R[2,1]:.1f}")
    
    for alpha in [0.01, 0.1, 0.5, 1.0]:
        Q, V, pi = soft_value_iteration(P, R, gamma=0.99, alpha=alpha)
        
        entropy = -np.sum(pi * np.log(pi + 1e-10), axis=1).mean()
        
        print(f"\nα = {alpha}:")
        for s in range(S):
            print(f"  State {s}: π(a0)={pi[s,0]:.4f}, π(a1)={pi[s,1]:.4f}, V={V[s]:.4f}")
        print(f"  Avg entropy: {entropy:.4f}")


def demo_entropy_exploration():
    """Show how entropy bonus aids exploration in multi-modal rewards."""
    print("\n" + "=" * 60)
    print("Entropy-Aided Exploration in Multi-Modal Environments")
    print("=" * 60)
    
    # Bandit with two good arms
    n_actions = 5
    true_rewards = np.array([0.0, 0.9, 0.1, 0.85, 0.0])
    
    print(f"True rewards: {true_rewards}")
    print("Two near-optimal actions: a1 (0.9) and a3 (0.85)")
    
    # Standard RL: greedy
    n_steps = 1000
    np.random.seed(42)
    
    for alpha in [0.0, 0.1, 0.5]:
        logits = np.zeros(n_actions)
        lr = 0.1
        
        action_counts = np.zeros(n_actions)
        total_reward = 0.0
        
        for t in range(n_steps):
            # Boltzmann policy
            if alpha > 0:
                probs = np.exp(logits / alpha)
                probs /= probs.sum()
            else:
                probs = np.zeros(n_actions)
                probs[logits.argmax()] = 1.0
                # Add small epsilon for exploration
                probs = 0.95 * probs + 0.05 / n_actions
            
            action = np.random.choice(n_actions, p=probs)
            reward = true_rewards[action] + np.random.normal(0, 0.1)
            
            action_counts[action] += 1
            total_reward += reward
            
            # Update logit for selected action
            logits[action] += lr * (reward - logits[action])
        
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        
        label = f"α={alpha}" if alpha > 0 else "Greedy"
        print(f"\n{label}:")
        print(f"  Action counts: {action_counts.astype(int)}")
        print(f"  Final probs: {probs.round(3)}")
        print(f"  Avg reward: {total_reward/n_steps:.3f}")
        print(f"  Final entropy: {entropy:.3f}")


def demo_soft_bellman():
    """Demonstrate soft Bellman backup vs standard Bellman."""
    print("\n" + "=" * 60)
    print("Soft vs Standard Bellman Backup")
    print("=" * 60)
    
    # Q-values for next state
    q_next = torch.tensor([2.0, 1.8, 1.0, 0.5])
    
    print(f"Q(s', a) for all actions: {q_next.numpy()}")
    
    # Standard Bellman: V(s') = max_a Q(s', a)
    v_standard = q_next.max()
    print(f"\nStandard V(s') = max Q = {v_standard.item():.4f}")
    
    # Soft Bellman: V(s') = α log Σ exp(Q/α)
    for alpha in [0.01, 0.1, 0.5, 1.0, 5.0]:
        v_soft = alpha * torch.logsumexp(q_next / alpha, dim=0)
        
        # Corresponding soft policy
        pi = torch.softmax(q_next / alpha, dim=0)
        entropy = -(pi * pi.log()).sum()
        
        print(f"α={alpha:<5.2f}: V_soft={v_soft.item():<8.4f} "
              f"π={pi.numpy().round(3)}  H={entropy.item():.3f}")


if __name__ == "__main__":
    demo_soft_vi()
    demo_entropy_exploration()
    demo_soft_bellman()
