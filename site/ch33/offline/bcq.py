"""
33.5.3 Batch-Constrained Q-Learning (BCQ)
===========================================

Discrete BCQ with behavior cloning constraint.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
from typing import Dict
import random


class QNetwork(nn.Module):
    def __init__(self, sd, ad, h=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(sd, h), nn.ReLU(),
                                 nn.Linear(h, h), nn.ReLU(), nn.Linear(h, ad))
    def forward(self, x): return self.net(x)


class BehaviorModel(nn.Module):
    """Behavior cloning model: estimates P(a|s) from dataset."""
    def __init__(self, sd, ad, h=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(sd, h), nn.ReLU(),
                                 nn.Linear(h, h), nn.ReLU(), nn.Linear(h, ad))
    def forward(self, x):
        return torch.softmax(self.net(x), dim=-1)

    def log_probs(self, x):
        return torch.log_softmax(self.net(x), dim=-1)


class DiscreteBCQAgent:
    """BCQ for discrete action spaces."""

    def __init__(self, state_dim, action_dim, lr_q=3e-4, lr_bc=1e-3,
                 gamma=0.99, threshold=0.3, target_freq=200):
        self.gamma = gamma
        self.threshold = threshold
        self.action_dim = action_dim
        self.target_freq = target_freq

        # Q-network
        self.online = QNetwork(state_dim, action_dim)
        self.target = QNetwork(state_dim, action_dim)
        self.target.load_state_dict(self.online.state_dict())
        self.q_opt = optim.Adam(self.online.parameters(), lr=lr_q)

        # Behavior model
        self.bc_model = BehaviorModel(state_dim, action_dim)
        self.bc_opt = optim.Adam(self.bc_model.parameters(), lr=lr_bc)

        self.update_count = 0

    def train_behavior_model(self, states, actions, n_steps=3000, batch_size=256):
        """Pre-train behavior cloning model."""
        n = len(actions)
        losses = []
        for step in range(n_steps):
            idx = np.random.randint(0, n, batch_size)
            s = torch.FloatTensor(states[idx])
            a = torch.LongTensor(actions[idx])
            log_p = self.bc_model.log_probs(s)
            loss = nn.functional.nll_loss(log_p, a)
            self.bc_opt.zero_grad(); loss.backward(); self.bc_opt.step()
            losses.append(loss.item())
        return losses

    def _filter_actions(self, states: torch.Tensor) -> torch.Tensor:
        """Create action mask based on behavior model threshold."""
        with torch.no_grad():
            probs = self.bc_model(states)  # (batch, action_dim)
            max_probs = probs.max(dim=1, keepdim=True)[0]
            mask = (probs / max_probs >= self.threshold).float()
            # Ensure at least one action is allowed
            mask[mask.sum(1) == 0] = 1.0
        return mask

    def train_q_step(self, states, actions, rewards, next_states, dones,
                     batch_size=256) -> Dict[str, float]:
        """Single Q-learning step with BCQ action filtering."""
        n = len(rewards)
        idx = np.random.randint(0, n, batch_size)
        s = torch.FloatTensor(states[idx])
        a = torch.LongTensor(actions[idx])
        r = torch.FloatTensor(rewards[idx])
        ns = torch.FloatTensor(next_states[idx])
        d = torch.FloatTensor(dones[idx])

        # Current Q
        q = self.online(s).gather(1, a.unsqueeze(1)).squeeze(1)

        # BCQ target: only consider in-distribution actions
        with torch.no_grad():
            next_q = self.target(ns)
            mask = self._filter_actions(ns)
            # Set Q to -inf for filtered-out actions
            next_q_masked = next_q * mask + (1 - mask) * (-1e8)
            best_next_q = next_q_masked.max(dim=1)[0]
            targets = r + (1 - d) * self.gamma * best_next_q

        loss = nn.functional.mse_loss(q, targets)
        self.q_opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        self.q_opt.step()

        self.update_count += 1
        if self.update_count % self.target_freq == 0:
            self.target.load_state_dict(self.online.state_dict())

        # Statistics
        n_allowed = mask.sum(1).mean().item()
        return {'loss': loss.item(), 'avg_allowed_actions': n_allowed}

    def select_action(self, state: np.ndarray) -> int:
        s = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q = self.online(s)
            mask = self._filter_actions(s)
            q_masked = q * mask + (1 - mask) * (-1e8)
        return q_masked.argmax(1).item()


def collect_dataset(env_name='CartPole-v1', n=5000, seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    env = gym.make(env_name)
    sd = env.observation_space.shape[0]; ad = env.action_space.n
    q = nn.Sequential(nn.Linear(sd,64),nn.ReLU(),nn.Linear(64,64),nn.ReLU(),nn.Linear(64,ad))
    tgt = nn.Sequential(nn.Linear(sd,64),nn.ReLU(),nn.Linear(64,64),nn.ReLU(),nn.Linear(64,ad))
    tgt.load_state_dict(q.state_dict()); o = optim.Adam(q.parameters(),lr=1e-3)
    buf={'s':[],'a':[],'r':[],'ns':[],'d':[]}; step=0
    for _ in range(200):
        s,_=env.reset();done=False
        while not done:
            step+=1;eps=max(0.05,1.0-step/3000)
            a=env.action_space.sample() if random.random()<eps else q(torch.FloatTensor(s).unsqueeze(0)).argmax(1).item()
            ns,r,term,trunc,_=env.step(a);done=term or trunc
            for k,v in zip(['s','a','r','ns','d'],[s,a,r,ns,float(done)]):buf[k].append(v)
            if len(buf['s'])>64:
                idx=np.random.randint(0,len(buf['s']),64)
                st=torch.FloatTensor(np.array(buf['s'])[idx]);at=torch.LongTensor(np.array(buf['a'])[idx])
                rt=torch.FloatTensor(np.array(buf['r'])[idx]);nst=torch.FloatTensor(np.array(buf['ns'])[idx])
                dt=torch.FloatTensor(np.array(buf['d'])[idx])
                qv=q(st).gather(1,at.unsqueeze(1)).squeeze(1)
                with torch.no_grad():t_=rt+(1-dt)*0.99*tgt(nst).max(1)[0]
                lo=nn.functional.mse_loss(qv,t_);o.zero_grad();lo.backward();o.step()
                if step%200==0:tgt.load_state_dict(q.state_dict())
            s=ns
    S,A,R,NS,D=[],[],[],[],[];c=0
    while c<n:
        s,_=env.reset();done=False
        while not done and c<n:
            a=env.action_space.sample() if random.random()<0.3 else q(torch.FloatTensor(s).unsqueeze(0)).argmax(1).item()
            ns,r,term,trunc,_=env.step(a);done=term or trunc
            S.append(s);A.append(a);R.append(r);NS.append(ns);D.append(float(done));s=ns;c+=1
    env.close()
    return {k:np.array(v,dt) for k,v,dt in zip(['states','actions','rewards','next_states','dones'],
            [S,A,R,NS,D],[np.float32,np.int64,np.float32,np.float32,np.float32])}


def demo_bcq():
    print("=" * 60)
    print("BCQ Demo")
    print("=" * 60)

    data = collect_dataset(n=5000)
    sd = data['states'].shape[1]; ad = int(data['actions'].max())+1
    print(f"\nDataset: {len(data['rewards'])} transitions, {ad} actions")

    for tau in [0.0, 0.1, 0.3, 0.5]:
        random.seed(42); np.random.seed(42); torch.manual_seed(42)
        agent = DiscreteBCQAgent(sd, ad, threshold=tau)

        # Pre-train behavior model
        bc_losses = agent.train_behavior_model(data['states'], data['actions'], n_steps=2000)
        print(f"\n  τ={tau}: BC loss = {np.mean(bc_losses[-100:]):.4f}")

        # Train Q-network
        for step in range(5000):
            agent.train_q_step(data['states'], data['actions'], data['rewards'],
                               data['next_states'], data['dones'])

        # Evaluate
        env = gym.make('CartPole-v1'); rets = []
        for _ in range(20):
            s,_=env.reset();t=0;done=False
            while not done:
                a=agent.select_action(s);s,r,term,trunc,_=env.step(a);done=term or trunc;t+=r
            rets.append(t)
        env.close()
        print(f"    Eval: {np.mean(rets):.1f} ± {np.std(rets):.1f}")

    print("\nBCQ demo complete!")


if __name__ == "__main__":
    demo_bcq()
