"""
33.5.2 Conservative Q-Learning (CQL)
======================================

CQL for discrete action spaces with both fixed and adaptive alpha.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
from typing import Dict, Tuple, List
import random


class QNetwork(nn.Module):
    def __init__(self, sd, ad, h=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(sd, h), nn.ReLU(),
                                 nn.Linear(h, h), nn.ReLU(), nn.Linear(h, ad))
    def forward(self, x): return self.net(x)


class CQLAgent:
    """Conservative Q-Learning for discrete offline RL."""

    def __init__(self, state_dim: int, action_dim: int, lr: float = 3e-4,
                 gamma: float = 0.99, alpha: float = 1.0, auto_alpha: bool = False,
                 target_action_gap: float = 5.0, target_freq: int = 200):
        self.gamma = gamma
        self.action_dim = action_dim
        self.alpha = alpha
        self.auto_alpha = auto_alpha
        self.target_action_gap = target_action_gap
        self.target_freq = target_freq

        self.online = QNetwork(state_dim, action_dim)
        self.target = QNetwork(state_dim, action_dim)
        self.target.load_state_dict(self.online.state_dict())
        self.opt = optim.Adam(self.online.parameters(), lr=lr)

        # Adaptive alpha
        if auto_alpha:
            self.log_alpha = torch.zeros(1, requires_grad=True)
            self.alpha_opt = optim.Adam([self.log_alpha], lr=3e-4)

        self.update_count = 0
        self.loss_history: Dict[str, List] = {
            'total': [], 'bellman': [], 'cql': [], 'alpha': []}

    def compute_cql_loss(self, q_all: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """CQL(H) regularizer: logsumexp(Q) - E_data[Q].
        
        q_all: (batch, action_dim) — Q-values for all actions
        actions: (batch,) — actions from dataset
        """
        # Push down: log-sum-exp over all actions (soft max)
        logsumexp = torch.logsumexp(q_all, dim=1).mean()

        # Push up: Q-values for in-distribution actions
        dataset_q = q_all.gather(1, actions.unsqueeze(1)).squeeze(1).mean()

        return logsumexp - dataset_q

    def train_step(self, states, actions, rewards, next_states, dones,
                   batch_size: int = 256) -> Dict[str, float]:
        """Single offline training step."""
        n = len(rewards)
        idx = np.random.randint(0, n, batch_size)
        s = torch.FloatTensor(states[idx])
        a = torch.LongTensor(actions[idx])
        r = torch.FloatTensor(rewards[idx])
        ns = torch.FloatTensor(next_states[idx])
        d = torch.FloatTensor(dones[idx])

        # Q-values
        q_all = self.online(s)
        q = q_all.gather(1, a.unsqueeze(1)).squeeze(1)

        # Bellman target (Double DQN)
        with torch.no_grad():
            best_a = self.online(ns).argmax(1)
            next_q = self.target(ns).gather(1, best_a.unsqueeze(1)).squeeze(1)
            targets = r + (1 - d) * self.gamma * next_q

        bellman_loss = nn.functional.mse_loss(q, targets)

        # CQL regularizer
        cql_reg = self.compute_cql_loss(q_all, a)

        # Adaptive alpha
        if self.auto_alpha:
            alpha = torch.exp(self.log_alpha).detach()
            alpha_loss = -self.log_alpha.exp() * (cql_reg.detach() - self.target_action_gap)
            self.alpha_opt.zero_grad()
            alpha_loss.backward()
            self.alpha_opt.step()
            self.alpha = alpha.item()
        else:
            alpha = self.alpha

        # Total loss
        total_loss = bellman_loss + alpha * cql_reg

        self.opt.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        self.opt.step()

        self.update_count += 1
        if self.update_count % self.target_freq == 0:
            self.target.load_state_dict(self.online.state_dict())

        # Logging
        log = {
            'total': total_loss.item(),
            'bellman': bellman_loss.item(),
            'cql': cql_reg.item(),
            'alpha': self.alpha if isinstance(self.alpha, float) else self.alpha,
        }
        for k, v in log.items():
            self.loss_history[k].append(v)
        return log

    def select_action(self, state: np.ndarray) -> int:
        with torch.no_grad():
            return self.online(torch.FloatTensor(state).unsqueeze(0)).argmax(1).item()

    def get_q_stats(self, states: np.ndarray) -> Dict:
        """Get Q-value statistics for analysis."""
        with torch.no_grad():
            q = self.online(torch.FloatTensor(states))
        return {
            'q_mean': q.mean().item(),
            'q_std': q.std().item(),
            'q_max': q.max().item(),
            'q_min': q.min().item(),
        }


def collect_dataset(env_name='CartPole-v1', n=5000, policy='medium', seed=42):
    """Quick dataset collection (simplified)."""
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    env = gym.make(env_name)
    sd = env.observation_space.shape[0]
    ad = env.action_space.n

    # Quick train for medium/expert
    q = nn.Sequential(nn.Linear(sd, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, ad))
    if policy != 'random':
        tgt = nn.Sequential(nn.Linear(sd, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, ad))
        tgt.load_state_dict(q.state_dict())
        o = optim.Adam(q.parameters(), lr=1e-3)
        buf = {'s':[], 'a':[], 'r':[], 'ns':[], 'd':[]}
        step = 0
        for _ in range(200 if policy == 'medium' else 500):
            s, _ = env.reset(); done = False
            while not done:
                step += 1
                eps = max(0.05, 1.0 - step/3000)
                a = env.action_space.sample() if random.random()<eps else q(torch.FloatTensor(s).unsqueeze(0)).argmax(1).item()
                ns, r, term, trunc, _ = env.step(a); done = term or trunc
                for k,v in zip(['s','a','r','ns','d'],[s,a,r,ns,float(done)]): buf[k].append(v)
                if len(buf['s']) > 64:
                    idx = np.random.randint(0,len(buf['s']),64)
                    st=torch.FloatTensor(np.array(buf['s'])[idx]); at=torch.LongTensor(np.array(buf['a'])[idx])
                    rt=torch.FloatTensor(np.array(buf['r'])[idx]); nst=torch.FloatTensor(np.array(buf['ns'])[idx])
                    dt=torch.FloatTensor(np.array(buf['d'])[idx])
                    qv=q(st).gather(1,at.unsqueeze(1)).squeeze(1)
                    with torch.no_grad(): t_=rt+(1-dt)*0.99*tgt(nst).max(1)[0]
                    lo=nn.functional.mse_loss(qv,t_); o.zero_grad(); lo.backward(); o.step()
                    if step%200==0: tgt.load_state_dict(q.state_dict())
                s = ns

    S,A,R,NS,D = [],[],[],[],[]
    c = 0
    while c < n:
        s, _ = env.reset(); done = False
        while not done and c < n:
            eps = 0.3 if policy=='medium' else (1.0 if policy=='random' else 0.05)
            a = env.action_space.sample() if random.random()<eps else q(torch.FloatTensor(s).unsqueeze(0)).argmax(1).item()
            ns, r, term, trunc, _ = env.step(a); done = term or trunc
            S.append(s);A.append(a);R.append(r);NS.append(ns);D.append(float(done))
            s=ns; c+=1
    env.close()
    return {k:np.array(v,dtype) for k,v,dtype in
            zip(['states','actions','rewards','next_states','dones'],
                [S,A,R,NS,D],[np.float32,np.int64,np.float32,np.float32,np.float32])}


def evaluate_policy(q_net, env_name='CartPole-v1', n_ep=20):
    env = gym.make(env_name); rets = []
    for _ in range(n_ep):
        s, _ = env.reset(); t = 0; done = False
        while not done:
            a = q_net(torch.FloatTensor(s).unsqueeze(0)).argmax(1).item()
            s, r, term, trunc, _ = env.step(a); done = term or trunc; t += r
        rets.append(t)
    env.close()
    return {'mean': np.mean(rets), 'std': np.std(rets)}


def demo_cql():
    print("=" * 60)
    print("CQL Demo")
    print("=" * 60)

    data = collect_dataset(policy='medium', n=5000)
    sd = data['states'].shape[1]
    ad = int(data['actions'].max()) + 1
    print(f"\nDataset: {len(data['rewards'])} transitions")

    # --- Compare CQL vs naive offline DQN ---
    for name, alpha in [("Naive DQN (α=0)", 0.0), ("CQL (α=1)", 1.0),
                         ("CQL (α=5)", 5.0)]:
        random.seed(42); np.random.seed(42); torch.manual_seed(42)
        agent = CQLAgent(sd, ad, alpha=alpha, lr=3e-4)
        for step in range(5000):
            agent.train_step(data['states'], data['actions'], data['rewards'],
                             data['next_states'], data['dones'], batch_size=128)
        ev = evaluate_policy(agent.online)
        q_stats = agent.get_q_stats(data['states'][:500])
        print(f"\n  {name}:")
        print(f"    Eval: {ev['mean']:.1f} ± {ev['std']:.1f}")
        print(f"    Q-values: mean={q_stats['q_mean']:.2f}, max={q_stats['q_max']:.2f}")
        print(f"    CQL reg (last 100): {np.mean(agent.loss_history['cql'][-100:]):.4f}")

    print("\nCQL demo complete!")


if __name__ == "__main__":
    demo_cql()
