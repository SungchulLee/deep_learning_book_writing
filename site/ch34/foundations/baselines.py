"""
Chapter 34.1.4: Baseline Methods
=================================
Implementation of various baseline methods for variance reduction
in policy gradient algorithms.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import gymnasium as gym
from typing import List, Tuple
from collections import deque


# ---------------------------------------------------------------------------
# Networks
# ---------------------------------------------------------------------------

class ValueNetwork(nn.Module):
    """State value function V(s) used as a baseline."""
    
    def __init__(self, obs_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)


class PolicyNetwork(nn.Module):
    """Simple categorical policy for discrete actions."""
    
    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, act_dim),
        )
    
    def forward(self, obs: torch.Tensor) -> Categorical:
        return Categorical(logits=self.net(obs))


# ---------------------------------------------------------------------------
# Baseline Strategies
# ---------------------------------------------------------------------------

class ConstantBaseline:
    """Running average of returns as a constant baseline."""
    
    def __init__(self, decay: float = 0.99):
        self.value = 0.0
        self.decay = decay
        self.initialized = False
    
    def update(self, returns: List[float]):
        avg = np.mean(returns)
        if not self.initialized:
            self.value = avg
            self.initialized = True
        else:
            self.value = self.decay * self.value + (1 - self.decay) * avg
    
    def get_baseline(self, states: torch.Tensor) -> torch.Tensor:
        return torch.full((states.shape[0],), self.value)


class LearnedBaseline:
    """Learned state-dependent value function baseline V_phi(s)."""
    
    def __init__(self, obs_dim: int, hidden_dim: int = 128, lr: float = 1e-3, n_epochs: int = 5):
        self.value_net = ValueNetwork(obs_dim, hidden_dim)
        self.optimizer = optim.Adam(self.value_net.parameters(), lr=lr)
        self.n_epochs = n_epochs
    
    def update(self, states: torch.Tensor, returns: torch.Tensor):
        for _ in range(self.n_epochs):
            values = self.value_net(states)
            loss = nn.functional.mse_loss(values, returns)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return loss.item()
    
    def get_baseline(self, states: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.value_net(states)


# ---------------------------------------------------------------------------
# REINFORCE with Baseline Agent
# ---------------------------------------------------------------------------

class REINFORCEWithBaseline:
    """
    REINFORCE agent with configurable baseline methods.
    
    Parameters
    ----------
    env : gym.Env
        Environment.
    baseline_type : str
        One of 'none', 'constant', 'learned'.
    """
    
    def __init__(
        self,
        env: gym.Env,
        baseline_type: str = "learned",
        lr_policy: float = 1e-3,
        lr_value: float = 1e-3,
        gamma: float = 0.99,
        hidden_dim: int = 128,
        normalize_advantages: bool = True,
        entropy_coef: float = 0.01,
    ):
        self.env = env
        self.gamma = gamma
        self.normalize_advantages = normalize_advantages
        self.entropy_coef = entropy_coef
        self.baseline_type = baseline_type
        
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.n
        
        self.policy = PolicyNetwork(obs_dim, act_dim, hidden_dim)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr_policy)
        
        if baseline_type == "constant":
            self.baseline = ConstantBaseline()
        elif baseline_type == "learned":
            self.baseline = LearnedBaseline(obs_dim, hidden_dim, lr=lr_value)
        else:
            self.baseline = None
    
    def compute_returns(self, rewards: List[float]) -> torch.Tensor:
        returns = []
        G = 0.0
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        return torch.tensor(returns, dtype=torch.float32)
    
    def collect_episode(self):
        obs, _ = self.env.reset()
        states, actions, log_probs, entropies, rewards = [], [], [], [], []
        
        done = False
        while not done:
            obs_t = torch.FloatTensor(obs).unsqueeze(0)
            dist = self.policy(obs_t)
            action = dist.sample()
            
            states.append(obs)
            actions.append(action.item())
            log_probs.append(dist.log_prob(action))
            entropies.append(dist.entropy())
            
            obs, reward, terminated, truncated, _ = self.env.step(action.item())
            rewards.append(reward)
            done = terminated or truncated
        
        return states, actions, log_probs, entropies, rewards
    
    def update(self, states, actions, log_probs, entropies, rewards):
        returns = self.compute_returns(rewards)
        states_t = torch.FloatTensor(np.array(states))
        
        # Compute advantages
        if self.baseline is None:
            advantages = returns.clone()
        elif self.baseline_type == "constant":
            self.baseline.update(returns.numpy().tolist())
            advantages = returns - self.baseline.get_baseline(states_t)
        elif self.baseline_type == "learned":
            advantages = returns - self.baseline.get_baseline(states_t)
            self.baseline.update(states_t, returns)
        
        if self.normalize_advantages and len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Policy loss
        log_probs_t = torch.stack(log_probs).squeeze()
        entropies_t = torch.stack(entropies).squeeze()
        
        policy_loss = -(log_probs_t * advantages.detach()).mean()
        entropy_loss = -entropies_t.mean()
        loss = policy_loss + self.entropy_coef * entropy_loss
        
        self.policy_optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
        self.policy_optimizer.step()
        
        return policy_loss.item(), entropies_t.mean().item()
    
    def train(self, n_episodes: int = 1000, print_interval: int = 100) -> List[float]:
        episode_rewards = []
        recent_rewards = deque(maxlen=100)
        
        for episode in range(1, n_episodes + 1):
            states, actions, log_probs, entropies, rewards = self.collect_episode()
            policy_loss, avg_entropy = self.update(states, actions, log_probs, entropies, rewards)
            
            total_reward = sum(rewards)
            episode_rewards.append(total_reward)
            recent_rewards.append(total_reward)
            
            if episode % print_interval == 0:
                print(
                    f"Episode {episode:>5d} | "
                    f"Reward: {total_reward:>7.1f} | "
                    f"Avg(100): {np.mean(recent_rewards):>7.1f} | "
                    f"Loss: {policy_loss:>8.4f}"
                )
        
        return episode_rewards


# ---------------------------------------------------------------------------
# Advantage estimation demonstrations
# ---------------------------------------------------------------------------

def compute_mc_advantages(rewards, values, gamma=0.99):
    """Monte Carlo advantage: A_t = G_t - V(s_t)."""
    returns = []
    G = 0.0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    returns = torch.tensor(returns, dtype=torch.float32)
    return returns - values


def compute_td_advantages(rewards, values, next_values, dones, gamma=0.99):
    """One-step TD advantage: A_t = r_t + γV(s_{t+1}) - V(s_t)."""
    td_targets = torch.tensor(rewards) + gamma * next_values * (1 - torch.tensor(dones, dtype=torch.float32))
    return td_targets - values


def compute_nstep_advantages(rewards, values, next_values, dones, gamma=0.99, n=5):
    """N-step TD advantage estimation."""
    T = len(rewards)
    advantages = torch.zeros(T)
    
    for t in range(T):
        G = 0.0
        for k in range(min(n, T - t)):
            G += gamma ** k * rewards[t + k]
            if dones[t + k]:
                break
        else:
            if t + n < T:
                G += gamma ** n * next_values[t + n].item()
        advantages[t] = G - values[t].item()
    
    return advantages


def demo_advantage_comparison():
    """Compare different advantage estimation methods."""
    print("=" * 60)
    print("Advantage Estimation Comparison")
    print("=" * 60)
    
    # Simulated episode data
    T = 20
    torch.manual_seed(42)
    rewards = [1.0] * T  # Constant reward
    rewards[-1] = 10.0   # Terminal bonus
    
    # Simulated value estimates (imperfect)
    true_values = torch.tensor([
        sum(0.99 ** (k - t) * rewards[k] for k in range(t, T))
        for t in range(T)
    ])
    noise = torch.randn(T) * 0.5
    estimated_values = true_values + noise
    next_values = torch.cat([estimated_values[1:], torch.zeros(1)])
    dones = [False] * (T - 1) + [True]
    
    # Compute advantages
    mc_adv = compute_mc_advantages(rewards, estimated_values, gamma=0.99)
    td_adv = compute_td_advantages(rewards, estimated_values, next_values, dones, gamma=0.99)
    n5_adv = compute_nstep_advantages(rewards, estimated_values, next_values, dones, gamma=0.99, n=5)
    
    print(f"\n{'Step':>4} {'MC Adv':>10} {'TD(0) Adv':>10} {'TD(5) Adv':>10}")
    print("-" * 38)
    for t in range(min(10, T)):
        print(f"{t:>4} {mc_adv[t]:>10.4f} {td_adv[t]:>10.4f} {n5_adv[t]:>10.4f}")
    
    print(f"\n{'Method':<12} {'Mean':>8} {'Std':>8} {'|Max|':>8}")
    print("-" * 38)
    for name, adv in [("MC", mc_adv), ("TD(0)", td_adv), ("TD(5)", n5_adv)]:
        print(f"{name:<12} {adv.mean():>8.4f} {adv.std():>8.4f} {adv.abs().max():>8.4f}")


# ---------------------------------------------------------------------------
# Compare baselines on CartPole
# ---------------------------------------------------------------------------

def compare_baselines():
    """Compare different baselines on CartPole."""
    print("\n" + "=" * 60)
    print("Baseline Comparison on CartPole-v1")
    print("=" * 60)
    
    baselines = ["none", "constant", "learned"]
    n_episodes = 500
    n_trials = 3
    
    results = {}
    
    for bl_type in baselines:
        trial_final_rewards = []
        
        for trial in range(n_trials):
            torch.manual_seed(trial)
            np.random.seed(trial)
            env = gym.make("CartPole-v1")
            
            agent = REINFORCEWithBaseline(
                env=env,
                baseline_type=bl_type,
                lr_policy=1e-3,
                lr_value=1e-3,
                gamma=0.99,
                normalize_advantages=True,
                entropy_coef=0.01,
            )
            
            rewards = agent.train(n_episodes=n_episodes, print_interval=n_episodes + 1)
            trial_final_rewards.append(np.mean(rewards[-100:]))
            env.close()
        
        results[bl_type] = trial_final_rewards
    
    print(f"\nResults after {n_episodes} episodes (avg of last 100, {n_trials} trials):")
    print(f"{'Baseline':<18} {'Mean':>8} {'Std':>8}")
    print("-" * 36)
    for bl_type, vals in results.items():
        print(f"{bl_type:<18} {np.mean(vals):>8.1f} {np.std(vals):>8.1f}")


# ---------------------------------------------------------------------------
# Variance analysis
# ---------------------------------------------------------------------------

def analyze_gradient_variance():
    """Measure gradient variance with different baselines."""
    print("\n" + "=" * 60)
    print("Gradient Variance Analysis")
    print("=" * 60)
    
    env = gym.make("CartPole-v1")
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    
    torch.manual_seed(42)
    policy = PolicyNetwork(obs_dim, act_dim, hidden_dim=64)
    
    # Collect episodes
    n_episodes = 50
    all_episodes = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        episode = {"states": [], "actions": [], "rewards": []}
        done = False
        while not done:
            obs_t = torch.FloatTensor(obs).unsqueeze(0)
            with torch.no_grad():
                dist = policy(obs_t)
            action = dist.sample().item()
            next_obs, reward, terminated, truncated, _ = env.step(action)
            episode["states"].append(obs)
            episode["actions"].append(action)
            episode["rewards"].append(reward)
            obs = next_obs
            done = terminated or truncated
        all_episodes.append(episode)
    
    # Compute gradients with different baselines for each episode
    baselines_to_test = {
        "No baseline": lambda returns, states: torch.zeros_like(returns),
        "Mean return": lambda returns, states: torch.full_like(returns, returns.mean()),
        "Per-step mean": lambda returns, states: returns.mean().expand_as(returns),
    }
    
    for bl_name, bl_fn in baselines_to_test.items():
        grad_norms = []
        
        for ep in all_episodes:
            # Compute returns
            G = 0.0
            returns = []
            for r in reversed(ep["rewards"]):
                G = r + 0.99 * G
                returns.insert(0, G)
            returns = torch.tensor(returns, dtype=torch.float32)
            states_t = torch.FloatTensor(np.array(ep["states"]))
            actions_t = torch.tensor(ep["actions"])
            
            baseline_vals = bl_fn(returns, states_t)
            advantages = returns - baseline_vals
            
            # Compute gradient
            policy.zero_grad()
            dist = policy(states_t)
            log_probs = dist.log_prob(actions_t)
            loss = -(log_probs * advantages.detach()).mean()
            loss.backward()
            
            grad_norm = sum(
                p.grad.norm().item() ** 2
                for p in policy.parameters()
                if p.grad is not None
            ) ** 0.5
            grad_norms.append(grad_norm)
        
        mean_gn = np.mean(grad_norms)
        std_gn = np.std(grad_norms)
        cv = std_gn / (mean_gn + 1e-8)
        print(f"{bl_name:<20}: grad_norm = {mean_gn:.4f} ± {std_gn:.4f} (CV: {cv:.4f})")
    
    env.close()


if __name__ == "__main__":
    demo_advantage_comparison()
    compare_baselines()
    analyze_gradient_variance()
