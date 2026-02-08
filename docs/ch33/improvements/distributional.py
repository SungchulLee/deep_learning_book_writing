"""
33.2.5 Distributional RL
==========================

C51 (Categorical DQN) and QR-DQN implementations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
import random
from typing import Tuple


class ReplayBuffer:
    def __init__(self, cap, sd):
        self.cap=cap; self.sz=0; self.p=0
        self.s=np.zeros((cap,sd),np.float32); self.a=np.zeros(cap,np.int64)
        self.r=np.zeros(cap,np.float32); self.ns=np.zeros((cap,sd),np.float32)
        self.d=np.zeros(cap,np.float32)
    def push(self,s,a,r,ns,d):
        self.s[self.p]=s;self.a[self.p]=a;self.r[self.p]=r
        self.ns[self.p]=ns;self.d[self.p]=float(d)
        self.p=(self.p+1)%self.cap;self.sz=min(self.sz+1,self.cap)
    def sample(self,n):
        i=np.random.randint(0,self.sz,n)
        return (torch.FloatTensor(self.s[i]),torch.LongTensor(self.a[i]),
                torch.FloatTensor(self.r[i]),torch.FloatTensor(self.ns[i]),
                torch.FloatTensor(self.d[i]))
    def __len__(self): return self.sz


# ---------------------------------------------------------------------------
# C51: Categorical DQN
# ---------------------------------------------------------------------------

class C51Network(nn.Module):
    """C51 network: outputs probability distribution over return atoms."""

    def __init__(self, state_dim: int, action_dim: int, n_atoms: int = 51,
                 v_min: float = -10.0, v_max: float = 10.0, hidden_dim: int = 128):
        super().__init__()
        self.action_dim = action_dim
        self.n_atoms = n_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.register_buffer('atoms', torch.linspace(v_min, v_max, n_atoms))
        self.delta_z = (v_max - v_min) / (n_atoms - 1)

        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.output = nn.Linear(hidden_dim, action_dim * n_atoms)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns log-probabilities: (batch, action_dim, n_atoms)."""
        feat = self.feature(x)
        logits = self.output(feat).view(-1, self.action_dim, self.n_atoms)
        return F.log_softmax(logits, dim=2)

    def q_values(self, x: torch.Tensor) -> torch.Tensor:
        """Expected Q-values from distribution."""
        probs = self.forward(x).exp()
        return (probs * self.atoms.unsqueeze(0).unsqueeze(0)).sum(dim=2)


def c51_projection(target_net: C51Network, next_states: torch.Tensor,
                   rewards: torch.Tensor, dones: torch.Tensor,
                   gamma: float) -> torch.Tensor:
    """Project the target distribution onto the fixed atom support."""
    batch_size = next_states.size(0)
    n_atoms = target_net.n_atoms
    atoms = target_net.atoms
    v_min = target_net.v_min
    v_max = target_net.v_max
    delta_z = target_net.delta_z

    with torch.no_grad():
        # Get target distribution for best actions
        next_q = target_net.q_values(next_states)
        best_actions = next_q.argmax(dim=1)
        next_log_probs = target_net(next_states)
        idx = best_actions.unsqueeze(1).unsqueeze(2).expand(-1, -1, n_atoms)
        next_probs = next_log_probs.exp().gather(1, idx).squeeze(1)

        # Project: T_z = r + gamma * z, clipped to [v_min, v_max]
        tz = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * gamma * atoms.unsqueeze(0)
        tz = tz.clamp(v_min, v_max)

        # Compute projection indices
        b = (tz - v_min) / delta_z
        l = b.floor().long().clamp(0, n_atoms - 1)
        u = b.ceil().long().clamp(0, n_atoms - 1)

        # Distribute probability
        m = torch.zeros(batch_size, n_atoms, device=next_states.device)
        offset = torch.arange(batch_size, device=next_states.device).unsqueeze(1) * n_atoms

        m.view(-1).index_add_(0, (l + offset).view(-1),
                               (next_probs * (u.float() - b)).view(-1))
        m.view(-1).index_add_(0, (u + offset).view(-1),
                               (next_probs * (b - l.float())).view(-1))
    return m


def c51_loss(online_net: C51Network, target_net: C51Network,
             states, actions, rewards, next_states, dones, gamma=0.99):
    """C51 cross-entropy loss."""
    target_dist = c51_projection(target_net, next_states, rewards, dones, gamma)
    log_probs = online_net(states)
    idx = actions.unsqueeze(1).unsqueeze(2).expand(-1, -1, online_net.n_atoms)
    log_probs_a = log_probs.gather(1, idx).squeeze(1)
    loss = -(target_dist * log_probs_a).sum(dim=1).mean()
    return loss


# ---------------------------------------------------------------------------
# QR-DQN: Quantile Regression DQN
# ---------------------------------------------------------------------------

class QRDQNNetwork(nn.Module):
    """Quantile Regression DQN: outputs quantile values."""

    def __init__(self, state_dim: int, action_dim: int, n_quantiles: int = 51,
                 hidden_dim: int = 128):
        super().__init__()
        self.action_dim = action_dim
        self.n_quantiles = n_quantiles
        taus = (2 * torch.arange(n_quantiles).float() + 1) / (2 * n_quantiles)
        self.register_buffer('taus', taus)

        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.output = nn.Linear(hidden_dim, action_dim * n_quantiles)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns quantile values: (batch, action_dim, n_quantiles)."""
        feat = self.feature(x)
        return self.output(feat).view(-1, self.action_dim, self.n_quantiles)

    def q_values(self, x: torch.Tensor) -> torch.Tensor:
        """Expected Q = mean of quantile values."""
        return self.forward(x).mean(dim=2)


def quantile_huber_loss(quantiles: torch.Tensor, target_quantiles: torch.Tensor,
                        taus: torch.Tensor, kappa: float = 1.0) -> torch.Tensor:
    """Quantile Huber loss for QR-DQN.
    
    quantiles: (batch, n_quantiles) — predicted
    target_quantiles: (batch, n_quantiles) — target
    taus: (n_quantiles,) — quantile fractions
    """
    n = quantiles.size(1)
    # Pairwise TD errors: (batch, n, n)
    td_errors = target_quantiles.unsqueeze(1) - quantiles.unsqueeze(2)

    # Huber loss
    huber = torch.where(td_errors.abs() <= kappa,
                        0.5 * td_errors.pow(2),
                        kappa * (td_errors.abs() - 0.5 * kappa))

    # Asymmetric weight
    tau_weight = (taus.unsqueeze(0).unsqueeze(2) - (td_errors < 0).float()).abs()
    loss = (tau_weight * huber).sum(dim=2).mean(dim=1)
    return loss.mean()


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def demo_distributional_rl():
    print("=" * 60)
    print("Distributional RL Demo")
    print("=" * 60)

    env = gym.make('CartPole-v1')
    sd = env.observation_space.shape[0]
    ad = env.action_space.n

    # --- C51 architecture ---
    print("\n--- C51 Architecture ---")
    c51 = C51Network(sd, ad, n_atoms=51, v_min=-10, v_max=10)
    x = torch.randn(4, sd)
    q = c51.q_values(x)
    log_p = c51(x)
    print(f"  Q-values shape: {q.shape}")
    print(f"  Distribution shape: {log_p.shape}")
    print(f"  Q-values: {q[0].detach().numpy().round(3)}")
    print(f"  Atom support: [{c51.v_min}, {c51.v_max}], {c51.n_atoms} atoms")
    print(f"  Params: {sum(p.numel() for p in c51.parameters()):,}")

    # --- QR-DQN architecture ---
    print("\n--- QR-DQN Architecture ---")
    qrdqn = QRDQNNetwork(sd, ad, n_quantiles=51)
    quantiles = qrdqn(x)
    q_qr = qrdqn.q_values(x)
    print(f"  Quantile values shape: {quantiles.shape}")
    print(f"  Q-values: {q_qr[0].detach().numpy().round(3)}")
    print(f"  Quantile fractions: {qrdqn.taus[:5].numpy().round(3)}...")
    print(f"  Params: {sum(p.numel() for p in qrdqn.parameters()):,}")

    # --- C51 Training on CartPole ---
    print("\n--- C51 Training ---")
    random.seed(42); np.random.seed(42); torch.manual_seed(42)

    online = C51Network(sd, ad, n_atoms=51, v_min=0, v_max=200)
    target = C51Network(sd, ad, n_atoms=51, v_min=0, v_max=200)
    target.load_state_dict(online.state_dict())
    opt = torch.optim.Adam(online.parameters(), lr=1e-3)
    buf = ReplayBuffer(50000, sd)
    rewards_hist = []
    step = 0

    for ep in range(250):
        s, _ = env.reset(); total = 0; done = False
        while not done:
            step += 1
            eps = max(0.01, 1.0 - step / 5000)
            if random.random() < eps:
                a = env.action_space.sample()
            else:
                with torch.no_grad():
                    a = online.q_values(torch.FloatTensor(s).unsqueeze(0)).argmax(1).item()
            ns, r, term, trunc, _ = env.step(a)
            done = term or trunc
            buf.push(s, a, r, ns, float(done))
            if len(buf) >= 500:
                st, at, rt, nst, dt = buf.sample(64)
                loss = c51_loss(online, target, st, at, rt, nst, dt, gamma=0.99)
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(online.parameters(), 10.0)
                opt.step()
                if step % 200 == 0:
                    target.load_state_dict(online.state_dict())
            s = ns; total += r
        rewards_hist.append(total)
        if (ep + 1) % 50 == 0:
            print(f"  Episode {ep+1}: avg50={np.mean(rewards_hist[-50:]):.1f}")

    env.close()

    # --- Visualize learned distribution ---
    print("\n--- Learned Distribution ---")
    env2 = gym.make('CartPole-v1')
    s, _ = env2.reset()
    with torch.no_grad():
        log_p = online(torch.FloatTensor(s).unsqueeze(0))
        probs = log_p.exp()[0]  # (action_dim, n_atoms)
    for a_idx in range(ad):
        dist = probs[a_idx].numpy()
        mean_val = (dist * online.atoms.numpy()).sum()
        std_val = np.sqrt(((online.atoms.numpy() - mean_val)**2 * dist).sum())
        print(f"  Action {a_idx}: mean={mean_val:.2f}, std={std_val:.2f}")
    env2.close()

    print("\nDistributional RL demo complete!")


if __name__ == "__main__":
    demo_distributional_rl()
