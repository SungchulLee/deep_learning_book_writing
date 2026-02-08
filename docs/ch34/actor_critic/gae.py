"""
Chapter 34.2.4: Generalized Advantage Estimation (GAE)
=======================================================
Implementation of GAE with various lambda values,
comparison with other advantage estimators, and
integration with actor-critic training.
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
# GAE Computation
# ---------------------------------------------------------------------------

def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    last_value: float,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Generalized Advantage Estimation.
    
    Parameters
    ----------
    rewards : ndarray, shape (T,)
        Rewards at each timestep.
    values : ndarray, shape (T,)
        Value estimates V(s_t) at each timestep.
    dones : ndarray, shape (T,)
        Done flags (1 if terminal, 0 otherwise).
    last_value : float
        Bootstrap value V(s_T) for the state after the rollout.
    gamma : float
        Discount factor.
    lam : float
        GAE lambda parameter.
    
    Returns
    -------
    advantages : ndarray, shape (T,)
        GAE advantage estimates.
    returns : ndarray, shape (T,)
        Return targets: advantages + values.
    """
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    last_gae = 0.0
    
    for t in reversed(range(T)):
        if t == T - 1:
            next_non_terminal = 1.0 - dones[t]
            next_value = last_value
        else:
            next_non_terminal = 1.0 - dones[t]
            next_value = values[t + 1]
        
        # TD error: δ_t = r_t + γ V(s_{t+1}) - V(s_t)
        delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
        
        # GAE: A_t = δ_t + γλ A_{t+1}
        advantages[t] = last_gae = delta + gamma * lam * next_non_terminal * last_gae
    
    returns = advantages + values
    return advantages, returns


def compute_gae_batched(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    last_values: np.ndarray,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute GAE for vectorized environments.
    
    Parameters
    ----------
    rewards : ndarray, shape (T, N)
        Rewards from N environments over T steps.
    values : ndarray, shape (T, N)
        Value estimates.
    dones : ndarray, shape (T, N)
        Done flags.
    last_values : ndarray, shape (N,)
        Bootstrap values for each environment.
    
    Returns
    -------
    advantages : ndarray, shape (T, N)
    returns : ndarray, shape (T, N)
    """
    T, N = rewards.shape
    advantages = np.zeros((T, N), dtype=np.float32)
    last_gae = np.zeros(N, dtype=np.float32)
    
    for t in reversed(range(T)):
        if t == T - 1:
            next_non_terminal = 1.0 - dones[t]
            next_values = last_values
        else:
            next_non_terminal = 1.0 - dones[t]
            next_values = values[t + 1]
        
        delta = rewards[t] + gamma * next_values * next_non_terminal - values[t]
        advantages[t] = last_gae = delta + gamma * lam * next_non_terminal * last_gae
    
    returns = advantages + values
    return advantages, returns


# ---------------------------------------------------------------------------
# Comparison: Different advantage estimators
# ---------------------------------------------------------------------------

def compute_td0_advantage(rewards, values, dones, last_value, gamma=0.99):
    """1-step TD advantage: A_t = r_t + γV(s_{t+1}) - V(s_t)."""
    return compute_gae(rewards, values, dones, last_value, gamma, lam=0.0)


def compute_mc_advantage(rewards, values, dones, last_value, gamma=0.99):
    """Monte Carlo advantage: A_t = G_t - V(s_t)."""
    return compute_gae(rewards, values, dones, last_value, gamma, lam=1.0)


def compute_nstep_advantage(rewards, values, dones, last_value, gamma=0.99, n=5):
    """N-step advantage estimation."""
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    
    # Extend values
    ext_values = np.append(values, last_value)
    
    for t in range(T):
        G = 0.0
        k = 0
        for k in range(min(n, T - t)):
            G += gamma ** k * rewards[t + k]
            if dones[t + k]:
                break
        else:
            # Bootstrap if no terminal within n steps
            if t + n < T:
                G += gamma ** n * ext_values[t + n]
            else:
                G += gamma ** (T - t) * last_value
        
        advantages[t] = G - values[t]
    
    returns = advantages + values
    return advantages, returns


# ---------------------------------------------------------------------------
# Actor-Critic with GAE
# ---------------------------------------------------------------------------

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class ACNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=64):
        super().__init__()
        self.features = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_dim)), nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)), nn.Tanh(),
        )
        self.actor = layer_init(nn.Linear(hidden_dim, act_dim), std=0.01)
        self.critic = layer_init(nn.Linear(hidden_dim, 1), std=1.0)
    
    def forward(self, obs):
        f = self.features(obs)
        return self.actor(f), self.critic(f).squeeze(-1)
    
    def get_action_and_value(self, obs, action=None):
        logits, value = self.forward(obs)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value
    
    def get_value(self, obs):
        return self.forward(obs)[1]


class A2CGAE:
    """A2C with GAE advantage estimation."""
    
    def __init__(
        self,
        env_id="CartPole-v1",
        n_envs=8,
        n_steps=128,
        lr=2.5e-4,
        gamma=0.99,
        gae_lambda=0.95,
        hidden_dim=64,
        value_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
    ):
        self.n_envs = n_envs
        self.n_steps = n_steps
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        
        # Vectorized environments
        self.envs = [gym.make(env_id) for _ in range(n_envs)]
        obs_dim = self.envs[0].observation_space.shape[0]
        act_dim = self.envs[0].action_space.n
        
        self.network = ACNetwork(obs_dim, act_dim, hidden_dim)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr, eps=1e-5)
    
    def _reset_envs(self):
        return np.array([env.reset(seed=i)[0] for i, env in enumerate(self.envs)])
    
    def _step_envs(self, actions):
        obs_list, rewards, dones = [], [], []
        for env, a in zip(self.envs, actions):
            obs, r, term, trunc, _ = env.step(a)
            done = term or trunc
            if done:
                obs, _ = env.reset()
            obs_list.append(obs)
            rewards.append(r)
            dones.append(float(done))
        return np.array(obs_list), np.array(rewards), np.array(dones)
    
    def train(self, total_steps=200000, print_interval=10000):
        obs = self._reset_envs()
        episode_rewards = []
        current_rewards = np.zeros(self.n_envs)
        recent = deque(maxlen=100)
        steps = 0
        
        while steps < total_steps:
            # Collect rollout
            mb_obs = np.zeros((self.n_steps, self.n_envs) + (obs.shape[1],))
            mb_actions = np.zeros((self.n_steps, self.n_envs), dtype=np.int64)
            mb_rewards = np.zeros((self.n_steps, self.n_envs))
            mb_dones = np.zeros((self.n_steps, self.n_envs))
            mb_values = np.zeros((self.n_steps, self.n_envs))
            
            for t in range(self.n_steps):
                mb_obs[t] = obs
                with torch.no_grad():
                    action, _, _, value = self.network.get_action_and_value(
                        torch.FloatTensor(obs)
                    )
                mb_actions[t] = action.numpy()
                mb_values[t] = value.numpy()
                
                obs, rewards, dones = self._step_envs(action.numpy())
                mb_rewards[t] = rewards
                mb_dones[t] = dones
                
                current_rewards += rewards
                for i in range(self.n_envs):
                    if dones[i]:
                        episode_rewards.append(current_rewards[i])
                        recent.append(current_rewards[i])
                        current_rewards[i] = 0.0
            
            # Compute GAE
            with torch.no_grad():
                last_values = self.network.get_value(torch.FloatTensor(obs)).numpy()
            
            advantages, returns = compute_gae_batched(
                mb_rewards, mb_values, mb_dones, last_values,
                self.gamma, self.gae_lambda
            )
            
            # Flatten
            b_obs = torch.FloatTensor(mb_obs.reshape(-1, obs.shape[1]))
            b_actions = torch.LongTensor(mb_actions.reshape(-1))
            b_advantages = torch.FloatTensor(advantages.reshape(-1))
            b_returns = torch.FloatTensor(returns.reshape(-1))
            
            b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)
            
            # Update
            _, log_probs, entropy, values = self.network.get_action_and_value(b_obs, b_actions)
            
            policy_loss = -(log_probs * b_advantages).mean()
            value_loss = nn.functional.mse_loss(values, b_returns)
            entropy_loss = -entropy.mean()
            
            loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
            
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
            self.optimizer.step()
            
            steps += self.n_steps * self.n_envs
            
            if steps % print_interval < self.n_steps * self.n_envs and len(recent) > 0:
                print(
                    f"Step {steps:>8d} | Avg(100): {np.mean(recent):>7.1f} | "
                    f"π: {policy_loss.item():>7.4f} | V: {value_loss.item():>7.4f}"
                )
        
        for env in self.envs:
            env.close()
        return episode_rewards


# ---------------------------------------------------------------------------
# Demonstrations
# ---------------------------------------------------------------------------

def demo_gae_computation():
    """Demonstrate GAE computation with different lambda values."""
    print("=" * 60)
    print("GAE Computation Demo")
    print("=" * 60)
    
    T = 10
    np.random.seed(42)
    rewards = np.ones(T, dtype=np.float32)
    rewards[-1] = 5.0  # Terminal bonus
    values = np.array([8.0, 7.5, 7.0, 6.5, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0], dtype=np.float32)
    dones = np.zeros(T, dtype=np.float32)
    dones[-1] = 1.0
    last_value = 0.0
    
    lambdas = [0.0, 0.5, 0.9, 0.95, 1.0]
    
    print(f"\nRewards: {rewards}")
    print(f"Values:  {values}")
    print(f"\n{'t':>3}", end="")
    for lam in lambdas:
        print(f"  {'λ='+str(lam):>8}", end="")
    print()
    print("-" * (3 + 10 * len(lambdas)))
    
    all_advantages = {}
    for lam in lambdas:
        adv, ret = compute_gae(rewards, values, dones, last_value, gamma=0.99, lam=lam)
        all_advantages[lam] = adv
    
    for t in range(T):
        print(f"{t:>3}", end="")
        for lam in lambdas:
            print(f"  {all_advantages[lam][t]:>8.3f}", end="")
        print()
    
    print(f"\n{'Stat':>8}", end="")
    for lam in lambdas:
        print(f"  {'λ='+str(lam):>8}", end="")
    print()
    print("-" * (8 + 10 * len(lambdas)))
    
    for stat_name, stat_fn in [("Mean", np.mean), ("Std", np.std)]:
        print(f"{stat_name:>8}", end="")
        for lam in lambdas:
            print(f"  {stat_fn(all_advantages[lam]):>8.3f}", end="")
        print()


def demo_compare_lambda():
    """Compare GAE with different lambda values on CartPole."""
    print("\n" + "=" * 60)
    print("GAE Lambda Comparison on CartPole-v1")
    print("=" * 60)
    
    lambdas = [0.0, 0.5, 0.95, 1.0]
    total_steps = 100000
    n_trials = 2
    
    for lam in lambdas:
        trial_rewards = []
        for trial in range(n_trials):
            torch.manual_seed(trial)
            np.random.seed(trial)
            
            agent = A2CGAE(
                env_id="CartPole-v1",
                n_envs=4,
                n_steps=128,
                lr=2.5e-4,
                gamma=0.99,
                gae_lambda=lam,
            )
            rewards = agent.train(total_steps=total_steps, print_interval=total_steps + 1)
            if len(rewards) >= 50:
                trial_rewards.append(np.mean(rewards[-50:]))
            else:
                trial_rewards.append(np.mean(rewards) if rewards else 0)
        
        print(f"λ={lam:<4} → Final avg reward: {np.mean(trial_rewards):.1f} ± {np.std(trial_rewards):.1f}")


def demo_train_with_gae():
    """Full training run with GAE."""
    print("\n" + "=" * 60)
    print("A2C + GAE Training on CartPole-v1")
    print("=" * 60)
    
    agent = A2CGAE(
        env_id="CartPole-v1",
        n_envs=8,
        n_steps=128,
        lr=2.5e-4,
        gamma=0.99,
        gae_lambda=0.95,
    )
    rewards = agent.train(total_steps=200000, print_interval=25000)
    
    if len(rewards) >= 100:
        print(f"\nFinal avg (last 100): {np.mean(rewards[-100:]):.1f}")


if __name__ == "__main__":
    demo_gae_computation()
    demo_train_with_gae()
