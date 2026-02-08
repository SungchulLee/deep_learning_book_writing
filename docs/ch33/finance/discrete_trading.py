"""
33.7.3 Discrete Trading
=========================

DQN for buy/hold/sell trading with simulated market data.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
from typing import Dict, Tuple
import random


def generate_price_series(n_steps=2000, mu=0.0001, sigma=0.015,
                          regime_changes=True, seed=42):
    np.random.seed(seed)
    log_ret = np.random.normal(mu, sigma, n_steps)
    if regime_changes:
        for start in range(0, n_steps, 200):
            end = min(start + 200, n_steps)
            regime = np.random.choice(['trend', 'revert', 'normal'])
            if regime == 'trend':
                log_ret[start:end] += np.random.choice([-1, 1]) * 0.0005
            elif regime == 'revert':
                log_ret[start:end] *= 0.7
    return 100 * np.exp(np.cumsum(log_ret))


def compute_features(prices, window=20):
    n = len(prices)
    feat = np.zeros((n, 8), np.float32)
    ret1 = np.zeros(n); ret5 = np.zeros(n); ret20 = np.zeros(n)
    sma_r = np.zeros(n); rsi = np.zeros(n); vol = np.zeros(n)
    vpx = np.abs(np.diff(np.log(np.maximum(prices, 1e-8)), prepend=np.log(prices[0])))
    mom = np.zeros(n)
    for i in range(1, n): ret1[i] = (prices[i]-prices[i-1])/prices[i-1]
    for i in range(5, n): ret5[i] = (prices[i]-prices[i-5])/prices[i-5]
    for i in range(20, n):
        ret20[i] = (prices[i]-prices[i-20])/prices[i-20]
        sma = prices[i-20:i].mean()
        sma_r[i] = (prices[i]-sma)/(sma+1e-8)
        w = ret1[i-20:i]; vol[i] = w.std() if len(w)>1 else 0
        g = np.maximum(w, 0).mean(); l = -np.minimum(w, 0).mean()
        rs = g/(l+1e-8); rsi[i] = 2*(rs/(1+rs))-1
        mom[i] = ret5[i]/(vol[i]+1e-8)
    feat[:,0]=ret1/0.015; feat[:,1]=ret5/0.03; feat[:,2]=ret20/0.06
    feat[:,3]=sma_r/0.05; feat[:,4]=rsi; feat[:,5]=vol/0.02
    feat[:,6]=np.clip(vpx/0.015,-3,3); feat[:,7]=np.clip(mom,-3,3)
    return feat


class TradingEnv:
    def __init__(self, prices, features, tc=0.001):
        self.prices = prices; self.features = features; self.tc = tc
        self.state_dim = features.shape[1] + 1; self.action_dim = 3

    def reset(self, start_idx=20):
        self.idx = start_idx; self.position = 0
        self.total_pnl = 0.0; self.trades = 0; self.pnl_series = []
        return self._state()

    def _state(self):
        return np.concatenate([[self.position], self.features[self.idx]]).astype(np.float32)

    def step(self, action):
        target = action - 1  # -1, 0, +1
        change = target - self.position
        cost = abs(change) * self.tc * self.prices[self.idx]
        old_p = self.prices[self.idx]; self.idx += 1; new_p = self.prices[self.idx]
        pnl = self.position * (new_p - old_p) - cost
        self.position = target
        if change != 0: self.trades += 1
        self.total_pnl += pnl; self.pnl_series.append(pnl)
        done = self.idx >= len(self.prices) - 1
        return self._state(), pnl, done, {'pnl': self.total_pnl, 'trades': self.trades}


class QNet(nn.Module):
    def __init__(self, sd, ad, h=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(sd, h), nn.ReLU(),
                                 nn.Linear(h, h), nn.ReLU(), nn.Linear(h, ad))
    def forward(self, x): return self.net(x)


class TradingDQN:
    def __init__(self, state_dim, action_dim=3, lr=1e-3, gamma=0.99,
                 batch_size=64, buf_cap=50000, target_freq=200,
                 eps_end=0.05, eps_decay=3000):
        self.gamma=gamma; self.batch_size=batch_size; self.action_dim=action_dim
        self.target_freq=target_freq; self.eps_end=eps_end; self.eps_decay=eps_decay
        self.online = QNet(state_dim, action_dim)
        self.target = QNet(state_dim, action_dim)
        self.target.load_state_dict(self.online.state_dict())
        self.opt = optim.Adam(self.online.parameters(), lr=lr)
        self.buf = deque(maxlen=buf_cap); self.step=0; self.updates=0

    @property
    def epsilon(self):
        return max(self.eps_end, 1.0-(1.0-self.eps_end)*self.step/self.eps_decay)

    def act(self, state, training=True):
        if training:
            self.step += 1
            if random.random() < self.epsilon: return random.randrange(self.action_dim)
        with torch.no_grad():
            return self.online(torch.FloatTensor(state).unsqueeze(0)).argmax(1).item()

    def store(self, s, a, r, ns, d): self.buf.append((s,a,r,ns,float(d)))

    def update(self):
        if len(self.buf) < 500: return
        batch = random.sample(self.buf, self.batch_size)
        s=torch.FloatTensor([t[0] for t in batch])
        a=torch.LongTensor([t[1] for t in batch])
        r=torch.FloatTensor([t[2] for t in batch])
        ns=torch.FloatTensor([t[3] for t in batch])
        d=torch.FloatTensor([t[4] for t in batch])
        q=self.online(s).gather(1,a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            ba=self.online(ns).argmax(1)
            nq=self.target(ns).gather(1,ba.unsqueeze(1)).squeeze(1)
            tgt=r+(1-d)*self.gamma*nq
        loss=nn.functional.smooth_l1_loss(q,tgt)
        self.opt.zero_grad();loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(),10.0);self.opt.step()
        self.updates+=1
        if self.updates%self.target_freq==0:
            self.target.load_state_dict(self.online.state_dict())


def compute_metrics(pnl_series):
    if len(pnl_series) == 0: return {}
    arr = np.array(pnl_series)
    cum = np.cumsum(arr)
    peak = np.maximum.accumulate(cum)
    dd = peak - cum
    max_dd = dd.max()
    mean_r = arr.mean(); std_r = arr.std()
    sharpe = mean_r / (std_r + 1e-8) * np.sqrt(252)
    total = cum[-1]
    win_rate = (arr > 0).mean()
    gross_profit = arr[arr > 0].sum() if (arr > 0).any() else 0
    gross_loss = -arr[arr < 0].sum() if (arr < 0).any() else 1e-8
    profit_factor = gross_profit / (gross_loss + 1e-8)
    return {'total_pnl': total, 'sharpe': sharpe, 'max_drawdown': max_dd,
            'win_rate': win_rate, 'profit_factor': profit_factor,
            'n_trades': len(arr), 'mean_daily': mean_r}


def demo_discrete_trading():
    print("=" * 60)
    print("Discrete Trading Demo")
    print("=" * 60)

    # Generate data
    prices_train = generate_price_series(3000, seed=42)
    prices_test = generate_price_series(1000, seed=99)
    feat_train = compute_features(prices_train)
    feat_test = compute_features(prices_test)

    print(f"\nTrain: {len(prices_train)} steps, Test: {len(prices_test)} steps")

    # Buy-and-hold baseline
    print("\n--- Buy & Hold Baseline ---")
    bh_return = (prices_test[-1] - prices_test[20]) / prices_test[20]
    print(f"  Test return: {bh_return*100:.2f}%")

    # Random baseline
    print("\n--- Random Agent Baseline ---")
    env_test = TradingEnv(prices_test, feat_test)
    rand_pnls = []
    for _ in range(10):
        s = env_test.reset(); done = False
        while not done:
            s, r, done, _ = env_test.step(random.randrange(3))
        rand_pnls.append(env_test.total_pnl)
    print(f"  Mean PnL: {np.mean(rand_pnls):.4f} ± {np.std(rand_pnls):.4f}")

    # Train DQN
    print("\n--- DQN Training ---")
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    env_train = TradingEnv(prices_train, feat_train)
    agent = TradingDQN(env_train.state_dim, lr=5e-4, eps_decay=5000)

    for epoch in range(20):
        s = env_train.reset(); done = False
        while not done:
            a = agent.act(s)
            ns, r, done, info = env_train.step(a)
            agent.store(s, a, r, ns, done)
            agent.update()
            s = ns
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}: train PnL={info['pnl']:.4f}, "
                  f"trades={info['trades']}, ε={agent.epsilon:.3f}")

    # Evaluate on test
    print("\n--- Test Evaluation ---")
    s = env_test.reset(); done = False
    while not done:
        a = agent.act(s, training=False)
        s, r, done, info = env_test.step(a)

    metrics = compute_metrics(env_test.pnl_series)
    print(f"  Total PnL: {metrics['total_pnl']:.4f}")
    print(f"  Sharpe ratio: {metrics['sharpe']:.3f}")
    print(f"  Max drawdown: {metrics['max_drawdown']:.4f}")
    print(f"  Win rate: {metrics['win_rate']:.1%}")
    print(f"  Profit factor: {metrics['profit_factor']:.2f}")
    print(f"  Trades: {info['trades']}")

    print("\nDiscrete trading demo complete!")


if __name__ == "__main__":
    demo_discrete_trading()
