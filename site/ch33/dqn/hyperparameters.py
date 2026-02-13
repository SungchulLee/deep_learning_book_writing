"""
33.1.5 DQN Hyperparameters
===========================

Hyperparameter sweep utilities, sensitivity analysis, and
recommended defaults for DQN training.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
from collections import deque, namedtuple
from typing import Tuple, List, Dict, Any
import random
import time
import itertools

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))


# ---------------------------------------------------------------------------
# Minimal DQN for hyperparameter experiments
# ---------------------------------------------------------------------------

class ReplayBuffer:
    def __init__(self, capacity, state_dim):
        self.capacity = capacity
        self.size = 0
        self.ptr = 0
        self.s = np.zeros((capacity, state_dim), dtype=np.float32)
        self.a = np.zeros(capacity, dtype=np.int64)
        self.r = np.zeros(capacity, dtype=np.float32)
        self.ns = np.zeros((capacity, state_dim), dtype=np.float32)
        self.d = np.zeros(capacity, dtype=np.float32)

    def push(self, s, a, r, ns, d):
        self.s[self.ptr] = s; self.a[self.ptr] = a; self.r[self.ptr] = r
        self.ns[self.ptr] = ns; self.d[self.ptr] = float(d)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, n):
        idx = np.random.randint(0, self.size, n)
        return (torch.FloatTensor(self.s[idx]), torch.LongTensor(self.a[idx]),
                torch.FloatTensor(self.r[idx]), torch.FloatTensor(self.ns[idx]),
                torch.FloatTensor(self.d[idx]))

    def __len__(self): return self.size


class QNet(nn.Module):
    def __init__(self, sd, ad, hd=(128, 128)):
        super().__init__()
        layers = []
        prev = sd
        for h in hd:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, ad))
        self.net = nn.Sequential(*layers)

    def forward(self, x): return self.net(x)


def run_experiment(env_name: str, config: Dict[str, Any], n_episodes: int = 200,
                   seed: int = 42) -> Dict[str, Any]:
    """Run a single DQN experiment with given config, return results."""
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    env = gym.make(env_name)
    sd = env.observation_space.shape[0]
    ad = env.action_space.n

    lr = config.get('lr', 1e-3)
    gamma = config.get('gamma', 0.99)
    batch_size = config.get('batch_size', 64)
    buffer_cap = config.get('buffer_capacity', 50000)
    target_freq = config.get('target_update_freq', 200)
    eps_end = config.get('eps_end', 0.01)
    eps_decay = config.get('eps_decay_steps', 5000)
    hidden = config.get('hidden_dims', (128, 128))
    loss_type = config.get('loss_fn', 'huber')
    grad_clip = config.get('grad_clip', 10.0)
    min_buf = config.get('min_buffer', 500)

    online = QNet(sd, ad, hidden)
    target = QNet(sd, ad, hidden)
    target.load_state_dict(online.state_dict())
    opt = optim.Adam(online.parameters(), lr=lr)
    loss_fn = nn.SmoothL1Loss() if loss_type == 'huber' else nn.MSELoss()
    buf = ReplayBuffer(buffer_cap, sd)

    step = 0
    update_count = 0
    rewards_hist = []

    for ep in range(n_episodes):
        state, _ = env.reset()
        total_r = 0; done = False
        while not done:
            step += 1
            eps = max(eps_end, 1.0 - (1.0 - eps_end) * step / eps_decay)
            if random.random() < eps:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    action = online(torch.FloatTensor(state).unsqueeze(0)).argmax(1).item()

            ns, r, term, trunc, _ = env.step(action)
            done = term or trunc
            buf.push(state, action, r, ns, float(done))

            if len(buf) >= min_buf:
                s, a, re, nst, d = buf.sample(batch_size)
                q = online(s).gather(1, a.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    nq = target(nst).max(1)[0]
                    tgt = re + (1 - d) * gamma * nq
                loss = loss_fn(q, tgt)
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(online.parameters(), grad_clip)
                opt.step()
                update_count += 1
                if update_count % target_freq == 0:
                    target.load_state_dict(online.state_dict())

            state = ns; total_r += r
        rewards_hist.append(total_r)

    env.close()
    last50 = rewards_hist[-50:]
    return {
        'config': config,
        'rewards': rewards_hist,
        'mean_last50': np.mean(last50),
        'std_last50': np.std(last50),
        'max_reward': max(rewards_hist),
        'total_steps': step,
    }


# ---------------------------------------------------------------------------
# Hyperparameter sweep
# ---------------------------------------------------------------------------

def grid_sweep(env_name: str, param_grid: Dict[str, List],
               n_episodes: int = 200, seed: int = 42) -> List[Dict]:
    """Run grid search over hyperparameter combinations."""
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combos = list(itertools.product(*values))

    print(f"Grid sweep: {len(combos)} configurations")
    results = []

    for i, combo in enumerate(combos):
        config = dict(zip(keys, combo))
        print(f"\n[{i+1}/{len(combos)}] Config: {config}")
        start = time.time()
        result = run_experiment(env_name, config, n_episodes, seed)
        elapsed = time.time() - start
        result['wall_time'] = elapsed
        results.append(result)
        print(f"  → Mean(last50)={result['mean_last50']:.1f}, "
              f"Max={result['max_reward']:.0f}, Time={elapsed:.1f}s")

    # Sort by performance
    results.sort(key=lambda x: x['mean_last50'], reverse=True)
    return results


def sensitivity_analysis(env_name: str, base_config: Dict[str, Any],
                         param_name: str, param_values: List,
                         n_episodes: int = 200, n_seeds: int = 3) -> Dict:
    """Analyze sensitivity to a single hyperparameter across multiple seeds."""
    print(f"\nSensitivity analysis: {param_name}")
    print(f"Values: {param_values}, Seeds: {n_seeds}")

    results = {}
    for val in param_values:
        config = {**base_config, param_name: val}
        seed_results = []
        for seed in range(n_seeds):
            r = run_experiment(env_name, config, n_episodes, seed)
            seed_results.append(r['mean_last50'])
        results[val] = {
            'mean': np.mean(seed_results),
            'std': np.std(seed_results),
            'values': seed_results,
        }
        print(f"  {param_name}={val}: {np.mean(seed_results):.1f} ± {np.std(seed_results):.1f}")

    return results


# ---------------------------------------------------------------------------
# Default configurations
# ---------------------------------------------------------------------------

DEFAULTS = {
    'cartpole': {
        'lr': 1e-3, 'gamma': 0.99, 'batch_size': 64,
        'buffer_capacity': 50000, 'target_update_freq': 200,
        'eps_end': 0.01, 'eps_decay_steps': 5000,
        'hidden_dims': (128, 128), 'loss_fn': 'huber', 'grad_clip': 10.0,
    },
    'lunarlander': {
        'lr': 5e-4, 'gamma': 0.99, 'batch_size': 64,
        'buffer_capacity': 100000, 'target_update_freq': 500,
        'eps_end': 0.01, 'eps_decay_steps': 20000,
        'hidden_dims': (256, 256), 'loss_fn': 'huber', 'grad_clip': 10.0,
    },
    'finance': {
        'lr': 1e-4, 'gamma': 0.999, 'batch_size': 128,
        'buffer_capacity': 200000, 'target_update_freq': 1000,
        'eps_end': 0.05, 'eps_decay_steps': 50000,
        'hidden_dims': (256, 256, 128), 'loss_fn': 'mse', 'grad_clip': 1.0,
    },
}


# ---------------------------------------------------------------------------
# Effective horizon calculator
# ---------------------------------------------------------------------------

def effective_horizon(gamma: float) -> float:
    """Compute the effective planning horizon for a given discount factor."""
    return 1.0 / (1.0 - gamma)


def gamma_for_horizon(horizon: int) -> float:
    """Compute the discount factor for a desired effective horizon."""
    return 1.0 - 1.0 / horizon


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def demo_hyperparameters():
    """Demonstrate hyperparameter tuning for DQN."""
    print("=" * 60)
    print("DQN Hyperparameters Demo")
    print("=" * 60)

    # --- Effective horizon ---
    print("\n--- Effective Horizon ---")
    for g in [0.9, 0.95, 0.99, 0.995, 0.999]:
        h = effective_horizon(g)
        print(f"  γ = {g:.3f} → horizon ≈ {h:.0f} steps")

    print("\n--- Gamma for desired horizon ---")
    for h in [20, 50, 100, 252, 500, 1000]:
        g = gamma_for_horizon(h)
        print(f"  horizon = {h:>4d} steps → γ = {g:.6f}")

    # --- Quick grid sweep on CartPole ---
    print("\n--- Grid Sweep: Learning Rate × Target Update ---")
    param_grid = {
        'lr': [1e-4, 1e-3, 5e-3],
        'target_update_freq': [100, 500],
    }
    results = grid_sweep('CartPole-v1', param_grid, n_episodes=150, seed=42)

    print("\n--- Sweep Results (sorted by performance) ---")
    print(f"{'LR':<10s} {'Target Freq':<12s} {'Mean(last50)':<14s} {'Max':<8s}")
    print("-" * 44)
    for r in results:
        c = r['config']
        print(f"{c['lr']:<10.0e} {c['target_update_freq']:<12d} "
              f"{r['mean_last50']:<14.1f} {r['max_reward']:<8.0f}")

    # --- Sensitivity analysis ---
    print("\n--- Sensitivity: Learning Rate ---")
    base = DEFAULTS['cartpole'].copy()
    lr_sens = sensitivity_analysis(
        'CartPole-v1', base, 'lr',
        [1e-4, 5e-4, 1e-3, 3e-3, 1e-2],
        n_episodes=150, n_seeds=2
    )

    # --- Default configs ---
    print("\n--- Recommended Default Configurations ---")
    for env_name, cfg in DEFAULTS.items():
        print(f"\n  {env_name}:")
        for k, v in cfg.items():
            print(f"    {k}: {v}")

    print("\nHyperparameters demo complete!")


if __name__ == "__main__":
    demo_hyperparameters()
