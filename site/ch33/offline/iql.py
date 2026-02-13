"""
33.5.4 Implicit Q-Learning (IQL)
==================================

IQL for discrete offline RL with expectile regression.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
from typing import Dict, List
import random


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, h=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, h), nn.ReLU(),
                                 nn.Linear(h, h), nn.ReLU(), nn.Linear(h, out_dim))
    def forward(self, x): return self.net(x)


def expectile_loss(pred: torch.Tensor, target: torch.Tensor, tau: float = 0.7) -> torch.Tensor:
    """Asymmetric squared loss for expectile regression.
    
    L_tau(u) = |tau - 1(u<0)| * u^2
    """
    diff = target - pred
    weight = torch.where(diff > 0, tau, 1 - tau)
    return (weight * diff.pow(2)).mean()


class DiscreteIQLAgent:
    """IQL for discrete action spaces."""

    def __init__(self, state_dim: int, action_dim: int, tau: float = 0.7,
                 beta: float = 3.0, lr: float = 3e-4, gamma: float = 0.99,
                 target_freq: int = 200):
        self.gamma = gamma
        self.tau = tau
        self.beta = beta
        self.action_dim = action_dim
        self.target_freq = target_freq

        # Q-network (state → Q-values for all actions)
        self.q_net = MLP(state_dim, action_dim)
        self.q_target = MLP(state_dim, action_dim)
        self.q_target.load_state_dict(self.q_net.state_dict())

        # Value network (state → scalar V)
        self.v_net = MLP(state_dim, 1)

        self.q_opt = optim.Adam(self.q_net.parameters(), lr=lr)
        self.v_opt = optim.Adam(self.v_net.parameters(), lr=lr)

        self.update_count = 0
        self.history: Dict[str, List] = {'q_loss': [], 'v_loss': [], 'advantage': []}

    def train_step(self, states, actions, rewards, next_states, dones,
                   batch_size=256) -> Dict[str, float]:
        n = len(rewards)
        idx = np.random.randint(0, n, batch_size)
        s = torch.FloatTensor(states[idx])
        a = torch.LongTensor(actions[idx])
        r = torch.FloatTensor(rewards[idx])
        ns = torch.FloatTensor(next_states[idx])
        d = torch.FloatTensor(dones[idx])

        # --- Update Value network (expectile regression) ---
        with torch.no_grad():
            q_vals = self.q_target(s).gather(1, a.unsqueeze(1)).squeeze(1)
        v_pred = self.v_net(s).squeeze(1)
        v_loss = expectile_loss(v_pred, q_vals, self.tau)

        self.v_opt.zero_grad(); v_loss.backward(); self.v_opt.step()

        # --- Update Q-network (Bellman with V instead of max Q) ---
        q_pred = self.q_net(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_v = self.v_net(ns).squeeze(1)
            targets = r + (1 - d) * self.gamma * next_v
        q_loss = nn.functional.mse_loss(q_pred, targets)

        self.q_opt.zero_grad(); q_loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.q_opt.step()

        # Target network update
        self.update_count += 1
        if self.update_count % self.target_freq == 0:
            self.q_target.load_state_dict(self.q_net.state_dict())

        # Advantages for monitoring
        with torch.no_grad():
            advantages = q_vals - v_pred

        log = {'q_loss': q_loss.item(), 'v_loss': v_loss.item(),
               'advantage': advantages.mean().item()}
        for k, v in log.items():
            self.history[k].append(v)
        return log

    def select_action(self, state: np.ndarray) -> int:
        """Select action using advantage-weighted policy.
        
        For discrete actions: weight each action by exp(β * A(s,a))
        and sample or take argmax.
        """
        s = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q = self.q_net(s).squeeze(0)
            v = self.v_net(s).squeeze()
            advantages = q - v
            # Advantage-weighted: softmax over β*A
            weights = torch.softmax(self.beta * advantages, dim=0)
        return weights.argmax().item()  # Greedy over weighted advantages

    def get_stats(self, states: np.ndarray) -> Dict:
        with torch.no_grad():
            s = torch.FloatTensor(states)
            q = self.q_net(s)
            v = self.v_net(s).squeeze(1)
        return {
            'q_mean': q.mean().item(), 'v_mean': v.mean().item(),
            'advantage_mean': (q.max(1)[0] - v).mean().item(),
        }


def collect_dataset(n=5000, seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    env = gym.make('CartPole-v1')
    sd=env.observation_space.shape[0]; ad=env.action_space.n
    q=nn.Sequential(nn.Linear(sd,64),nn.ReLU(),nn.Linear(64,64),nn.ReLU(),nn.Linear(64,ad))
    tgt=nn.Sequential(nn.Linear(sd,64),nn.ReLU(),nn.Linear(64,64),nn.ReLU(),nn.Linear(64,ad))
    tgt.load_state_dict(q.state_dict());o=optim.Adam(q.parameters(),lr=1e-3)
    buf={'s':[],'a':[],'r':[],'ns':[],'d':[]};step=0
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


def demo_iql():
    print("=" * 60)
    print("IQL Demo")
    print("=" * 60)

    data = collect_dataset(n=5000)
    sd = data['states'].shape[1]; ad = int(data['actions'].max()) + 1

    # --- Compare expectile values ---
    print("\n--- Expectile Regression Effect ---")
    for tau in [0.5, 0.7, 0.8, 0.9]:
        random.seed(42); np.random.seed(42); torch.manual_seed(42)
        agent = DiscreteIQLAgent(sd, ad, tau=tau, beta=3.0)
        for _ in range(5000):
            agent.train_step(data['states'], data['actions'], data['rewards'],
                             data['next_states'], data['dones'])
        stats = agent.get_stats(data['states'][:500])
        env = gym.make('CartPole-v1'); rets = []
        for _ in range(20):
            s,_=env.reset();t=0;done=False
            while not done:
                a=agent.select_action(s);s,r,term,trunc,_=env.step(a);done=term or trunc;t+=r
            rets.append(t)
        env.close()
        print(f"  τ={tau}: eval={np.mean(rets):.1f}±{np.std(rets):.1f}, "
              f"Q̄={stats['q_mean']:.2f}, V̄={stats['v_mean']:.2f}, "
              f"Ā={stats['advantage_mean']:.3f}")

    # --- Compare β (temperature) ---
    print("\n--- Advantage Temperature Effect ---")
    for beta in [1.0, 3.0, 10.0]:
        random.seed(42); np.random.seed(42); torch.manual_seed(42)
        agent = DiscreteIQLAgent(sd, ad, tau=0.7, beta=beta)
        for _ in range(5000):
            agent.train_step(data['states'], data['actions'], data['rewards'],
                             data['next_states'], data['dones'])
        env = gym.make('CartPole-v1'); rets = []
        for _ in range(20):
            s,_=env.reset();t=0;done=False
            while not done:
                a=agent.select_action(s);s,r,term,trunc,_=env.step(a);done=term or trunc;t+=r
            rets.append(t)
        env.close()
        print(f"  β={beta}: eval={np.mean(rets):.1f}±{np.std(rets):.1f}")

    print("\nIQL demo complete!")


if __name__ == "__main__":
    demo_iql()
