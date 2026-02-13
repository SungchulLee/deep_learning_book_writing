"""
Chapter 34.1.2: Policy Gradient Theorem
========================================
Demonstrations of the policy gradient theorem, score function
estimator, and various gradient estimation strategies.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
import numpy as np
import gymnasium as gym
from typing import List, Tuple


# ---------------------------------------------------------------------------
# Score function estimator demonstration
# ---------------------------------------------------------------------------

class SimplePolicy(nn.Module):
    """Simple softmax policy for demonstrating the PG theorem."""
    
    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, act_dim),
        )
    
    def forward(self, obs: torch.Tensor) -> Categorical:
        logits = self.net(obs)
        return Categorical(logits=logits)


def collect_trajectory(
    env: gym.Env,
    policy: SimplePolicy,
    max_steps: int = 200,
) -> Tuple[List, List, List]:
    """
    Collect a single trajectory under the given policy.
    
    Returns
    -------
    log_probs : list of Tensor
        Log probabilities of taken actions.
    rewards : list of float
        Rewards received at each step.
    states : list of ndarray
        States visited.
    """
    obs, _ = env.reset()
    log_probs, rewards, states = [], [], []
    
    for _ in range(max_steps):
        obs_t = torch.FloatTensor(obs).unsqueeze(0)
        dist = policy(obs_t)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        next_obs, reward, terminated, truncated, _ = env.step(action.item())
        
        log_probs.append(log_prob)
        rewards.append(reward)
        states.append(obs)
        
        obs = next_obs
        if terminated or truncated:
            break
    
    return log_probs, rewards, states


def compute_returns(rewards: List[float], gamma: float = 0.99) -> List[float]:
    """
    Compute discounted returns-to-go G_t = sum_{k=t}^T gamma^{k-t} r_k.
    
    This implements the causality refinement: each G_t only includes
    rewards from time t onward.
    """
    returns = []
    G = 0.0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    return returns


# ---------------------------------------------------------------------------
# Policy Gradient Estimators
# ---------------------------------------------------------------------------

def pg_total_reward(
    log_probs: List[torch.Tensor],
    rewards: List[float],
    gamma: float = 0.99,
) -> torch.Tensor:
    """
    REINFORCE with total trajectory reward (no causality).
    
    Gradient: sum_t [log pi(a_t|s_t) * R(tau)]
    
    Unbiased but very high variance because all actions are weighted
    by the total trajectory reward.
    """
    # Total discounted return for the trajectory
    R = sum(gamma**t * r for t, r in enumerate(rewards))
    
    # Policy gradient loss
    loss = -sum(lp * R for lp in log_probs)
    return loss


def pg_reward_to_go(
    log_probs: List[torch.Tensor],
    rewards: List[float],
    gamma: float = 0.99,
) -> torch.Tensor:
    """
    REINFORCE with reward-to-go (causality applied).
    
    Gradient: sum_t [log pi(a_t|s_t) * G_t]
    
    Lower variance than total reward because each action is only
    weighted by future rewards.
    """
    returns = compute_returns(rewards, gamma)
    
    loss = -sum(lp * G for lp, G in zip(log_probs, returns))
    return loss


def pg_with_baseline(
    log_probs: List[torch.Tensor],
    rewards: List[float],
    baseline: float,
    gamma: float = 0.99,
) -> torch.Tensor:
    """
    REINFORCE with constant baseline subtraction.
    
    Gradient: sum_t [log pi(a_t|s_t) * (G_t - b)]
    
    Baseline b does not change expectation (unbiased) but
    reduces variance when b ≈ E[G_t].
    """
    returns = compute_returns(rewards, gamma)
    
    loss = -sum(lp * (G - baseline) for lp, G in zip(log_probs, returns))
    return loss


# ---------------------------------------------------------------------------
# Log-derivative trick demonstration
# ---------------------------------------------------------------------------

def demonstrate_log_derivative_trick():
    """
    Demonstrate that ∇_θ E_π[f(x)] = E_π[∇_θ log π(x) · f(x)]
    by comparing analytical and score function gradients.
    """
    print("=" * 60)
    print("Log-Derivative Trick Verification")
    print("=" * 60)
    
    # Simple 1D case: π_θ(x) = Categorical([θ, 1-θ])
    # f(x) = [3.0, 1.0] (reward for each action)
    
    theta = torch.tensor([0.6], requires_grad=True)
    f_values = torch.tensor([3.0, 1.0])
    
    # Analytical gradient of E[f(x)] = θ·3 + (1-θ)·1 = 2θ + 1
    # ∇_θ E[f(x)] = 2.0
    analytical_grad = 2.0
    
    # Score function estimate (Monte Carlo)
    n_samples = 100000
    torch.manual_seed(42)
    
    probs = torch.cat([theta, 1 - theta])
    dist = Categorical(probs=probs)
    
    samples = dist.sample((n_samples,))
    log_probs = dist.log_prob(samples)
    rewards = f_values[samples]
    
    # ∇_θ E[f] ≈ (1/N) Σ ∇_θ log π(x_i) · f(x_i)
    surrogate_loss = -(log_probs * rewards).mean()
    surrogate_loss.backward()
    
    score_function_grad = -theta.grad.item()  # Negate because we minimized
    
    print(f"Analytical gradient:        {analytical_grad:.4f}")
    print(f"Score function estimate:    {score_function_grad:.4f}")
    print(f"Error:                      {abs(analytical_grad - score_function_grad):.4f}")


# ---------------------------------------------------------------------------
# Variance comparison of different estimators
# ---------------------------------------------------------------------------

def compare_estimator_variance():
    """
    Compare variance of different policy gradient estimators
    by running multiple gradient estimates.
    """
    print("\n" + "=" * 60)
    print("Policy Gradient Estimator Variance Comparison")
    print("=" * 60)
    
    env = gym.make("CartPole-v1")
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    
    torch.manual_seed(42)
    policy = SimplePolicy(obs_dim, act_dim)
    
    n_estimates = 50
    n_trajectories = 5
    gamma = 0.99
    
    grads_total = []
    grads_rtg = []
    grads_baseline = []
    
    for i in range(n_estimates):
        # Collect batch of trajectories
        batch_log_probs = []
        batch_rewards = []
        all_returns = []
        
        for _ in range(n_trajectories):
            lps, rews, _ = collect_trajectory(env, policy, max_steps=200)
            batch_log_probs.append(lps)
            batch_rewards.append(rews)
            all_returns.extend(compute_returns(rews, gamma))
        
        baseline = np.mean(all_returns)
        
        # Compute gradients with each estimator
        for est_name, est_func, grads_list in [
            ("total", pg_total_reward, grads_total),
            ("rtg", pg_reward_to_go, grads_rtg),
            ("baseline", lambda lp, r: pg_with_baseline(lp, r, baseline), grads_baseline),
        ]:
            policy.zero_grad()
            total_loss = sum(
                est_func(lps, rews)
                for lps, rews in zip(batch_log_probs, batch_rewards)
            ) / n_trajectories
            total_loss.backward()
            
            # Collect gradient norm
            grad_norm = sum(
                p.grad.norm().item() ** 2 
                for p in policy.parameters() 
                if p.grad is not None
            ) ** 0.5
            grads_list.append(grad_norm)
    
    print(f"\nGradient norm statistics over {n_estimates} estimates:")
    print(f"{'Estimator':<15} {'Mean':>10} {'Std':>10} {'CV':>10}")
    print("-" * 45)
    for name, grads in [
        ("Total reward", grads_total),
        ("Reward-to-go", grads_rtg),
        ("With baseline", grads_baseline),
    ]:
        mean_g = np.mean(grads)
        std_g = np.std(grads)
        cv = std_g / (mean_g + 1e-8)
        print(f"{name:<15} {mean_g:>10.4f} {std_g:>10.4f} {cv:>10.4f}")
    
    env.close()


# ---------------------------------------------------------------------------
# Gradient estimation with different return estimators
# ---------------------------------------------------------------------------

def demonstrate_gradient_computation():
    """Show step-by-step policy gradient computation."""
    print("\n" + "=" * 60)
    print("Step-by-Step Policy Gradient Computation")
    print("=" * 60)
    
    env = gym.make("CartPole-v1")
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    
    torch.manual_seed(0)
    policy = SimplePolicy(obs_dim, act_dim)
    
    # Collect one trajectory
    log_probs, rewards, states = collect_trajectory(env, policy, max_steps=50)
    returns = compute_returns(rewards, gamma=0.99)
    
    T = len(rewards)
    print(f"\nTrajectory length: {T}")
    print(f"Total reward: {sum(rewards):.1f}")
    print(f"Discounted return G_0: {returns[0]:.4f}")
    
    # Show first few steps
    print(f"\n{'Step':>4} {'Reward':>8} {'G_t':>10} {'log π':>10}")
    print("-" * 36)
    for t in range(min(10, T)):
        print(f"{t:>4} {rewards[t]:>8.2f} {returns[t]:>10.4f} {log_probs[t].item():>10.4f}")
    
    if T > 10:
        print(f"  ... ({T - 10} more steps)")
    
    # Compute policy gradient using reward-to-go
    policy.zero_grad()
    loss = pg_reward_to_go(log_probs, rewards, gamma=0.99)
    loss.backward()
    
    print(f"\nPolicy gradient (reward-to-go):")
    for name, param in policy.named_parameters():
        if param.grad is not None:
            print(f"  {name}: grad norm = {param.grad.norm().item():.6f}")
    
    env.close()


# ---------------------------------------------------------------------------
# Surrogate loss construction
# ---------------------------------------------------------------------------

def demonstrate_surrogate_loss():
    """
    Show how the surrogate loss is constructed for auto-differentiation.
    
    The key insight: we don't differentiate through the returns/advantages.
    We construct L(θ) = -E[log π_θ(a|s) · Â] and use standard optimizers.
    """
    print("\n" + "=" * 60)
    print("Surrogate Loss Construction")
    print("=" * 60)
    
    # Simulated batch of transitions
    batch_size = 32
    obs_dim, act_dim = 4, 2
    
    torch.manual_seed(42)
    policy = SimplePolicy(obs_dim, act_dim)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    
    # Simulated data (in practice, collected from environment)
    obs = torch.randn(batch_size, obs_dim)
    actions = torch.randint(0, act_dim, (batch_size,))
    advantages = torch.randn(batch_size)  # Normalized advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    # Construct surrogate loss
    dist = policy(obs)
    log_probs = dist.log_prob(actions)
    
    # Key: advantages are treated as constants (detached)
    # Only log_probs carry gradients through θ
    surrogate_loss = -(log_probs * advantages.detach()).mean()
    
    print(f"Batch size: {batch_size}")
    print(f"Surrogate loss: {surrogate_loss.item():.6f}")
    
    # Standard gradient descent step
    optimizer.zero_grad()
    surrogate_loss.backward()
    
    grad_norm = sum(
        p.grad.norm().item() ** 2 
        for p in policy.parameters() 
        if p.grad is not None
    ) ** 0.5
    print(f"Gradient norm: {grad_norm:.6f}")
    
    optimizer.step()
    print("Parameter update applied via Adam optimizer.")


if __name__ == "__main__":
    demonstrate_log_derivative_trick()
    compare_estimator_variance()
    demonstrate_gradient_computation()
    demonstrate_surrogate_loss()
