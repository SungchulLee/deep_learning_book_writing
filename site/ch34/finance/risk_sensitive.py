"""
Chapter 34.7.3: Risk-Sensitive Reinforcement Learning
======================================================
Risk-aware policy optimization with CVaR, Sharpe ratio,
and constrained MDP formulations for finance.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque


# ---------------------------------------------------------------------------
# Risk Measures
# ---------------------------------------------------------------------------

def compute_var(returns, alpha=0.05):
    """Value at Risk at confidence level alpha."""
    sorted_returns = np.sort(returns)
    idx = int(alpha * len(sorted_returns))
    return sorted_returns[max(idx, 0)]


def compute_cvar(returns, alpha=0.05):
    """Conditional Value at Risk (Expected Shortfall)."""
    var = compute_var(returns, alpha)
    tail = returns[returns <= var]
    return tail.mean() if len(tail) > 0 else var


def compute_sharpe(returns, risk_free=0.0, annualize=252):
    """Annualized Sharpe ratio."""
    excess = returns - risk_free / annualize
    if excess.std() < 1e-8:
        return 0.0
    return excess.mean() / excess.std() * np.sqrt(annualize)


def compute_max_drawdown(values):
    """Maximum drawdown from a value series."""
    peak = np.maximum.accumulate(values)
    drawdown = (peak - values) / (peak + 1e-8)
    return drawdown.max()


def compute_sortino(returns, risk_free=0.0, annualize=252):
    """Sortino ratio (penalizes only downside volatility)."""
    excess = returns - risk_free / annualize
    downside = excess[excess < 0]
    downside_std = downside.std() if len(downside) > 0 else 1e-8
    return excess.mean() / (downside_std + 1e-8) * np.sqrt(annualize)


# ---------------------------------------------------------------------------
# Risk-Sensitive Portfolio Environment
# ---------------------------------------------------------------------------

class RiskAwarePortfolioEnv:
    """Portfolio environment with risk-sensitive reward options."""
    
    def __init__(self, returns_data, window=20, tc=0.001,
                 risk_measure="cvar", risk_coef=1.0, cvar_alpha=0.05):
        self.returns = returns_data
        self.n_assets = returns_data.shape[1]
        self.window = window
        self.tc = tc
        self.risk_measure = risk_measure
        self.risk_coef = risk_coef
        self.cvar_alpha = cvar_alpha
        
        self.obs_dim = self.n_assets * 4 + self.n_assets
        self.reset()
    
    def reset(self):
        self.t = self.window
        self.weights = np.ones(self.n_assets) / self.n_assets
        self.portfolio_value = 1.0
        self.daily_returns = []
        return self._get_obs()
    
    def _get_obs(self):
        features = []
        for i in range(self.n_assets):
            rw = self.returns[self.t - self.window:self.t, i]
            features.extend([self.returns[self.t-1, i], rw[-5:].mean(), rw.mean(), rw.std()])
        features.extend(self.weights.tolist())
        return np.array(features, dtype=np.float32)
    
    def step(self, target_weights):
        turnover = np.abs(target_weights - self.weights).sum()
        cost = self.tc * turnover
        self.weights = target_weights.copy()
        
        port_return = np.dot(self.weights, self.returns[self.t]) - cost
        self.portfolio_value *= (1 + port_return)
        self.daily_returns.append(port_return)
        
        # Risk-sensitive reward
        base_reward = port_return
        risk_penalty = 0.0
        
        if len(self.daily_returns) >= 10:
            recent = np.array(self.daily_returns[-20:])
            
            if self.risk_measure == "variance":
                risk_penalty = np.var(recent)
            elif self.risk_measure == "cvar":
                risk_penalty = -compute_cvar(recent, self.cvar_alpha)  # Negative CVaR as penalty
            elif self.risk_measure == "drawdown":
                values = np.cumprod(1 + np.array(self.daily_returns))
                risk_penalty = compute_max_drawdown(values)
            elif self.risk_measure == "sortino":
                downside = recent[recent < 0]
                risk_penalty = (downside ** 2).mean() if len(downside) > 0 else 0
        
        reward = base_reward - self.risk_coef * risk_penalty
        
        self.t += 1
        done = self.t >= len(self.returns)
        
        if not done:
            w_drift = self.weights * (1 + self.returns[self.t - 1])
            self.weights = w_drift / (w_drift.sum() + 1e-8)
        
        obs = self._get_obs() if not done else np.zeros(self.obs_dim, dtype=np.float32)
        return obs, reward, done, {"portfolio_value": self.portfolio_value}


# ---------------------------------------------------------------------------
# CVaR-Constrained Policy Gradient
# ---------------------------------------------------------------------------

class CVaRConstrainedAgent:
    """
    Policy gradient with CVaR constraint via Lagrangian relaxation.
    
    max E[R]  s.t.  CVaR_alpha(R) >= threshold
    
    Converted to: max E[R] - lambda * (threshold - CVaR_alpha(R))
    """
    
    def __init__(self, obs_dim, n_assets, hidden=64, lr=3e-4,
                 gamma=0.99, cvar_alpha=0.05, cvar_threshold=-0.02,
                 lambda_lr=1e-3):
        self.gamma = gamma
        self.cvar_alpha = cvar_alpha
        self.cvar_threshold = cvar_threshold
        self.n_assets = n_assets
        
        # Policy network
        self.policy = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_assets),
        )
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        # Lagrange multiplier
        self.log_lambda = torch.tensor(0.0, requires_grad=True)
        self.lambda_optimizer = optim.Adam([self.log_lambda], lr=lambda_lr)
    
    def get_weights(self, obs):
        logits = self.policy(torch.FloatTensor(obs).unsqueeze(0))
        weights = F.softmax(logits, dim=-1)
        return weights.detach().numpy().flatten()
    
    def train_episode(self, env):
        obs = env.reset()
        log_probs, rewards = [], []
        
        done = False
        while not done:
            logits = self.policy(torch.FloatTensor(obs).unsqueeze(0))
            weights = F.softmax(logits, dim=-1)
            
            # Add exploration noise
            noise = torch.randn_like(weights) * 0.05
            weights_noisy = F.softmax(logits + noise, dim=-1)
            
            action = weights_noisy.detach().numpy().flatten()
            action = np.maximum(action, 0)
            action /= action.sum() + 1e-8
            
            log_prob = F.log_softmax(logits, dim=-1).mean()
            log_probs.append(log_prob)
            
            obs, reward, done, info = env.step(action)
            rewards.append(reward)
        
        return log_probs, rewards, info
    
    def update(self, all_episode_returns, all_log_probs, all_rewards):
        """Update policy and Lagrange multiplier."""
        # Compute CVaR of episode returns
        ep_returns = np.array(all_episode_returns)
        current_cvar = compute_cvar(ep_returns, self.cvar_alpha)
        
        # Policy update: maximize E[R] - λ * (threshold - CVaR)
        lam = self.log_lambda.exp().detach()
        
        total_loss = torch.tensor(0.0)
        for log_probs, rewards in zip(all_log_probs, all_rewards):
            G = 0
            returns = []
            for r in reversed(rewards):
                G = r + self.gamma * G
                returns.insert(0, G)
            returns = torch.FloatTensor(returns)
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
            
            lp_stack = torch.stack(log_probs).squeeze()
            total_loss -= (lp_stack * returns).mean()
        
        total_loss /= len(all_log_probs)
        
        # Add CVaR constraint penalty
        cvar_violation = self.cvar_threshold - current_cvar
        total_loss += lam * cvar_violation
        
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        # Update Lagrange multiplier
        lambda_loss = -self.log_lambda.exp() * cvar_violation
        self.lambda_optimizer.zero_grad()
        lambda_loss.backward()
        self.lambda_optimizer.step()
        
        return current_cvar, lam.item()


def demo_risk_sensitive():
    print("=" * 60)
    print("Risk-Sensitive RL for Portfolio Management")
    print("=" * 60)
    
    # Generate synthetic returns
    np.random.seed(42)
    n_days, n_assets = 500, 3
    means = np.array([0.0005, 0.0003, 0.0001])
    vols = np.array([0.02, 0.015, 0.008])
    returns = np.random.randn(n_days, n_assets) * vols + means
    # Add crash
    returns[200:210] -= 0.05
    
    print(f"\nComparing risk measures on {n_days} days, {n_assets} assets")
    print(f"(Includes a crash period at days 200-210)\n")
    
    risk_measures = ["variance", "cvar", "drawdown"]
    
    for rm in risk_measures:
        env = RiskAwarePortfolioEnv(
            returns, window=20, tc=0.001,
            risk_measure=rm, risk_coef=2.0,
        )
        
        # Simple policy gradient training
        agent = CVaRConstrainedAgent(env.obs_dim, n_assets, lr=1e-3)
        
        for epoch in range(50):
            batch_returns, batch_lp, batch_r = [], [], []
            for _ in range(5):
                lp, rews, info = agent.train_episode(env)
                batch_returns.append(sum(rews))
                batch_lp.append(lp)
                batch_r.append(rews)
            agent.update(batch_returns, batch_lp, batch_r)
        
        # Final evaluation
        obs = env.reset()
        done = False
        while not done:
            weights = agent.get_weights(obs)
            obs, _, done, info = env.step(weights)
        
        daily_rets = np.array(env.daily_returns)
        print(f"Risk={rm:<12}: Return={info['portfolio_value']-1:>7.2%}, "
              f"Sharpe={compute_sharpe(daily_rets):>6.2f}, "
              f"CVaR5%={compute_cvar(daily_rets):>8.4f}, "
              f"MDD={compute_max_drawdown(np.cumprod(1+daily_rets)):>6.2%}")


def demo_risk_measures():
    """Demonstrate different risk measures."""
    print("\n" + "=" * 60)
    print("Risk Measure Comparison")
    print("=" * 60)
    
    np.random.seed(42)
    
    # Two strategies with different risk profiles
    conservative = np.random.normal(0.0003, 0.005, 252)
    aggressive = np.random.normal(0.001, 0.025, 252)
    aggressive[50] = -0.10  # Single large loss
    
    for name, rets in [("Conservative", conservative), ("Aggressive", aggressive)]:
        values = np.cumprod(1 + rets)
        print(f"\n{name}:")
        print(f"  Total return: {values[-1]-1:>8.2%}")
        print(f"  Sharpe:       {compute_sharpe(rets):>8.3f}")
        print(f"  Sortino:      {compute_sortino(rets):>8.3f}")
        print(f"  VaR(5%):      {compute_var(rets):>8.4f}")
        print(f"  CVaR(5%):     {compute_cvar(rets):>8.4f}")
        print(f"  Max DD:       {compute_max_drawdown(values):>8.2%}")


if __name__ == "__main__":
    demo_risk_measures()
    demo_risk_sensitive()
