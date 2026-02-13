"""
33.5.1 Offline RL Fundamentals
================================

Dataset generation, offline evaluation, and demonstration of
why naive DQN fails in the offline setting.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
from collections import deque
from typing import Dict, List, Tuple
import random


# ---------------------------------------------------------------------------
# Offline Dataset Generation
# ---------------------------------------------------------------------------

def collect_dataset(env_name: str = 'CartPole-v1', n_transitions: int = 10000,
                    policy_type: str = 'medium', seed: int = 42) -> Dict[str, np.ndarray]:
    """Collect offline dataset with specified behavior policy quality.
    
    policy_type: 'random', 'medium', 'expert', 'mixed'
    """
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    env = gym.make(env_name)
    sd = env.observation_space.shape[0]
    ad = env.action_space.n

    # Train a Q-network to specified quality
    q_net = nn.Sequential(nn.Linear(sd, 64), nn.ReLU(),
                          nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, ad))

    if policy_type in ('medium', 'expert', 'mixed'):
        target_net = nn.Sequential(nn.Linear(sd, 64), nn.ReLU(),
                                    nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, ad))
        target_net.load_state_dict(q_net.state_dict())
        opt = optim.Adam(q_net.parameters(), lr=1e-3)
        buf_s, buf_a, buf_r, buf_ns, buf_d = [], [], [], [], []

        n_train = {'medium': 100, 'expert': 500, 'mixed': 200}[policy_type]
        step = 0
        for ep in range(n_train):
            s, _ = env.reset(); done = False
            while not done:
                step += 1
                eps = max(0.05, 1.0 - step / 3000)
                if random.random() < eps:
                    a = env.action_space.sample()
                else:
                    with torch.no_grad():
                        a = q_net(torch.FloatTensor(s).unsqueeze(0)).argmax(1).item()
                ns, r, term, trunc, _ = env.step(a)
                done = term or trunc
                buf_s.append(s); buf_a.append(a); buf_r.append(r)
                buf_ns.append(ns); buf_d.append(float(done))
                if len(buf_s) > 64:
                    idx = np.random.randint(0, len(buf_s), 64)
                    st = torch.FloatTensor(np.array(buf_s)[idx])
                    at = torch.LongTensor(np.array(buf_a)[idx])
                    rt = torch.FloatTensor(np.array(buf_r)[idx])
                    nst = torch.FloatTensor(np.array(buf_ns)[idx])
                    dt = torch.FloatTensor(np.array(buf_d)[idx])
                    q = q_net(st).gather(1, at.unsqueeze(1)).squeeze(1)
                    with torch.no_grad():
                        nq = target_net(nst).max(1)[0]
                        tgt = rt + (1 - dt) * 0.99 * nq
                    loss = nn.functional.mse_loss(q, tgt)
                    opt.zero_grad(); loss.backward(); opt.step()
                    if step % 200 == 0:
                        target_net.load_state_dict(q_net.state_dict())
                s = ns

    # Collect dataset with the trained (or random) policy
    states, actions, rewards, next_states, dones = [], [], [], [], []
    count = 0
    while count < n_transitions:
        s, _ = env.reset(); done = False
        while not done and count < n_transitions:
            if policy_type == 'random':
                a = env.action_space.sample()
            elif policy_type == 'mixed':
                if random.random() < 0.3:
                    a = env.action_space.sample()
                else:
                    with torch.no_grad():
                        a = q_net(torch.FloatTensor(s).unsqueeze(0)).argmax(1).item()
            else:
                eps = 0.1 if policy_type == 'medium' else 0.01
                if random.random() < eps:
                    a = env.action_space.sample()
                else:
                    with torch.no_grad():
                        a = q_net(torch.FloatTensor(s).unsqueeze(0)).argmax(1).item()

            ns, r, term, trunc, _ = env.step(a)
            done = term or trunc
            states.append(s); actions.append(a); rewards.append(r)
            next_states.append(ns); dones.append(float(done))
            s = ns; count += 1

    env.close()
    return {
        'states': np.array(states, dtype=np.float32),
        'actions': np.array(actions, dtype=np.int64),
        'rewards': np.array(rewards, dtype=np.float32),
        'next_states': np.array(next_states, dtype=np.float32),
        'dones': np.array(dones, dtype=np.float32),
    }


def dataset_statistics(dataset: Dict[str, np.ndarray]) -> Dict:
    """Compute statistics of an offline dataset."""
    n = len(dataset['rewards'])
    ep_rewards = []
    current = 0
    for i in range(n):
        current += dataset['rewards'][i]
        if dataset['dones'][i] > 0.5:
            ep_rewards.append(current)
            current = 0
    if current > 0:
        ep_rewards.append(current)

    unique_actions, action_counts = np.unique(dataset['actions'], return_counts=True)
    return {
        'n_transitions': n,
        'n_episodes': len(ep_rewards),
        'mean_episode_reward': np.mean(ep_rewards) if ep_rewards else 0,
        'std_episode_reward': np.std(ep_rewards) if ep_rewards else 0,
        'action_distribution': dict(zip(unique_actions.tolist(), 
                                        (action_counts / action_counts.sum()).tolist())),
        'state_mean': dataset['states'].mean(axis=0),
        'state_std': dataset['states'].std(axis=0),
    }


# ---------------------------------------------------------------------------
# Naive Offline DQN (to show it fails)
# ---------------------------------------------------------------------------

def train_offline_dqn(dataset: Dict, n_steps: int = 5000, lr: float = 1e-3,
                      batch_size: int = 64, gamma: float = 0.99,
                      target_freq: int = 200) -> nn.Module:
    """Train DQN purely offline on a fixed dataset."""
    sd = dataset['states'].shape[1]
    ad = int(dataset['actions'].max()) + 1
    n = len(dataset['rewards'])

    q_net = nn.Sequential(nn.Linear(sd, 128), nn.ReLU(),
                          nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, ad))
    target_net = nn.Sequential(nn.Linear(sd, 128), nn.ReLU(),
                                nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, ad))
    target_net.load_state_dict(q_net.state_dict())
    opt = optim.Adam(q_net.parameters(), lr=lr)

    losses = []
    q_means = []
    for step in range(n_steps):
        idx = np.random.randint(0, n, batch_size)
        s = torch.FloatTensor(dataset['states'][idx])
        a = torch.LongTensor(dataset['actions'][idx])
        r = torch.FloatTensor(dataset['rewards'][idx])
        ns = torch.FloatTensor(dataset['next_states'][idx])
        d = torch.FloatTensor(dataset['dones'][idx])

        q = q_net(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            nq = target_net(ns).max(1)[0]
            tgt = r + (1 - d) * gamma * nq
        loss = nn.functional.mse_loss(q, tgt)
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(q_net.parameters(), 10.0)
        opt.step()

        losses.append(loss.item())
        q_means.append(q.mean().item())
        if step % target_freq == 0:
            target_net.load_state_dict(q_net.state_dict())

    return q_net, losses, q_means


def evaluate_policy(q_net: nn.Module, env_name: str = 'CartPole-v1',
                    n_episodes: int = 20) -> Dict:
    env = gym.make(env_name)
    returns = []
    for _ in range(n_episodes):
        s, _ = env.reset(); total = 0; done = False
        while not done:
            with torch.no_grad():
                a = q_net(torch.FloatTensor(s).unsqueeze(0)).argmax(1).item()
            s, r, term, trunc, _ = env.step(a)
            done = term or trunc; total += r
        returns.append(total)
    env.close()
    return {'mean': np.mean(returns), 'std': np.std(returns),
            'min': np.min(returns), 'max': np.max(returns)}


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def demo_offline_fundamentals():
    print("=" * 60)
    print("Offline RL Fundamentals Demo")
    print("=" * 60)

    # --- Generate datasets ---
    print("\n--- Dataset Generation ---")
    for ptype in ['random', 'medium', 'expert']:
        dataset = collect_dataset(policy_type=ptype, n_transitions=5000)
        stats = dataset_statistics(dataset)
        print(f"\n  {ptype.upper()} dataset:")
        print(f"    Transitions: {stats['n_transitions']}")
        print(f"    Episodes: {stats['n_episodes']}")
        print(f"    Avg episode reward: {stats['mean_episode_reward']:.1f} "
              f"± {stats['std_episode_reward']:.1f}")
        print(f"    Action dist: {stats['action_distribution']}")

    # --- Show naive DQN failure ---
    print("\n--- Naive Offline DQN ---")
    for ptype in ['random', 'medium', 'expert']:
        dataset = collect_dataset(policy_type=ptype, n_transitions=5000)
        q_net, losses, q_means = train_offline_dqn(dataset, n_steps=3000)
        eval_result = evaluate_policy(q_net)
        print(f"\n  {ptype.upper()} data → Offline DQN:")
        print(f"    Eval: {eval_result['mean']:.1f} ± {eval_result['std']:.1f}")
        print(f"    Final loss: {np.mean(losses[-100:]):.4f}")
        print(f"    Q-value range: [{min(q_means):.2f}, {max(q_means):.2f}]")
        if max(q_means) > 50:
            print(f"    ⚠ Q-values diverging! (max={max(q_means):.1f})")

    print("\nOffline RL fundamentals demo complete!")


if __name__ == "__main__":
    demo_offline_fundamentals()
