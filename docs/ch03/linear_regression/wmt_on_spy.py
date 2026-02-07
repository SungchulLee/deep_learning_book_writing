"""
WMT on SPY — CAPM Beta Estimation
====================================

Estimate Walmart's market beta by regressing WMT daily returns on
SPY daily returns.  Three implementations: NumPy, sklearn, PyTorch.

Demonstrates:
- yfinance for data download
- Return calculation (pct_change)
- CAPM regression: R_WMT = α + β R_SPY + ε
- Rolling beta estimation
- Comparison across OLS, sklearn, and PyTorch

Author: Deep Learning Foundations Curriculum

Requirements:
    pip install yfinance
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# ============================================================================
# Configuration
# ============================================================================

parser = argparse.ArgumentParser(description="WMT on SPY beta estimation")
parser.add_argument("--start", type=str, default="2020-01-01")
parser.add_argument("--end", type=str, default="2024-01-01")
parser.add_argument("--rolling-window", type=int, default=60, help="rolling beta window")
parser.add_argument("--epochs", type=int, default=500, help="PyTorch training epochs")
parser.add_argument("--seed", type=int, default=42)
ARGS = parser.parse_args()

np.random.seed(ARGS.seed)
torch.manual_seed(ARGS.seed)

# ============================================================================
# Data
# ============================================================================

try:
    import yfinance as yf

    tickers = ["WMT", "SPY"]
    data = yf.download(tickers, start=ARGS.start, end=ARGS.end)["Adj Close"]
    data = data.dropna()
    returns = data.pct_change().dropna()
    returns.columns = ["SPY", "WMT"]
except ImportError:
    print("yfinance not installed — generating synthetic data.")
    n = 1000
    spy = np.random.normal(0.0004, 0.012, n)
    wmt = 0.0001 + 0.5 * spy + np.random.normal(0, 0.008, n)
    import pandas as pd

    returns = pd.DataFrame({"SPY": spy, "WMT": wmt})

x = returns["SPY"].values
y = returns["WMT"].values
n_obs = len(x)
print(f"Observations: {n_obs}")
print(f"SPY  mean={x.mean():.6f} std={x.std():.4f}")
print(f"WMT  mean={y.mean():.6f} std={y.std():.4f}")
print(f"Correlation: {np.corrcoef(x, y)[0, 1]:.4f}")

# ============================================================================
# Method 1: NumPy Normal Equation
# ============================================================================

print("\n--- NumPy Normal Equation ---")
X_np = np.column_stack([np.ones_like(x), x])
theta = np.linalg.solve(X_np.T @ X_np, X_np.T @ y)
alpha_np, beta_np = theta

y_pred_np = X_np @ theta
ss_res = np.sum((y - y_pred_np) ** 2)
ss_tot = np.sum((y - y.mean()) ** 2)
r2_np = 1.0 - ss_res / ss_tot

print(f"α = {alpha_np:.6f}")
print(f"β = {beta_np:.4f}")
print(f"R² = {r2_np:.4f}")

# ============================================================================
# Method 2: Sklearn
# ============================================================================

print("\n--- Sklearn ---")
from sklearn.linear_model import LinearRegression

model_sk = LinearRegression()
model_sk.fit(x.reshape(-1, 1), y)
print(f"α = {model_sk.intercept_:.6f}")
print(f"β = {model_sk.coef_[0]:.4f}")

# ============================================================================
# Method 3: PyTorch
# ============================================================================

print("\n--- PyTorch ---")
X_t = torch.tensor(x, dtype=torch.float32).unsqueeze(1)
y_t = torch.tensor(y, dtype=torch.float32)

model_pt = nn.Linear(1, 1)
optimizer = torch.optim.SGD(model_pt.parameters(), lr=1.0)
criterion = nn.MSELoss()

for epoch in range(ARGS.epochs):
    y_pred = model_pt(X_t).squeeze()
    loss = criterion(y_pred, y_t)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

beta_pt = model_pt.weight.item()
alpha_pt = model_pt.bias.item()
print(f"α = {alpha_pt:.6f}")
print(f"β = {beta_pt:.4f}")

# ============================================================================
# Rolling Beta
# ============================================================================

print(f"\n--- Rolling Beta (window={ARGS.rolling_window}) ---")
betas = []
for i in range(ARGS.rolling_window, n_obs):
    x_w = x[i - ARGS.rolling_window : i]
    y_w = y[i - ARGS.rolling_window : i]
    X_w = np.column_stack([np.ones_like(x_w), x_w])
    theta_w = np.linalg.solve(X_w.T @ X_w, X_w.T @ y_w)
    betas.append(theta_w[1])

betas = np.array(betas)
print(f"Rolling β — mean: {betas.mean():.4f}, std: {betas.std():.4f}")
print(f"Rolling β — min: {betas.min():.4f}, max: {betas.max():.4f}")

# ============================================================================
# Visualisation
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# (1) Scatter + regression line
ax = axes[0, 0]
ax.scatter(x, y, alpha=0.3, s=8)
x_line = np.linspace(x.min(), x.max(), 100)
ax.plot(x_line, alpha_np + beta_np * x_line, "r-", lw=2, label=f"β = {beta_np:.3f}")
ax.set_xlabel("SPY Return")
ax.set_ylabel("WMT Return")
ax.set_title("WMT vs SPY — CAPM Regression")
ax.legend()
ax.grid(True, alpha=0.3)

# (2) Residual histogram
ax = axes[0, 1]
residuals = y - (alpha_np + beta_np * x)
ax.hist(residuals, bins=50, density=True, alpha=0.7, edgecolor="black")
ax.set_xlabel("Residual")
ax.set_ylabel("Density")
ax.set_title(f"Residual Distribution (σ = {residuals.std():.4f})")
ax.grid(True, alpha=0.3)

# (3) Rolling beta
ax = axes[1, 0]
ax.plot(betas, lw=0.8)
ax.axhline(beta_np, color="r", ls="--", lw=1.5, label=f"Full-sample β = {beta_np:.3f}")
ax.set_xlabel(f"Day (after {ARGS.rolling_window}-day warmup)")
ax.set_ylabel("Rolling β")
ax.set_title(f"WMT {ARGS.rolling_window}-Day Rolling Beta")
ax.legend()
ax.grid(True, alpha=0.3)

# (4) Cumulative returns
ax = axes[1, 1]
cum_spy = (1 + returns["SPY"]).cumprod()
cum_wmt = (1 + returns["WMT"]).cumprod()
ax.plot(cum_spy.values, label="SPY", alpha=0.8)
ax.plot(cum_wmt.values, label="WMT", alpha=0.8)
ax.set_xlabel("Day")
ax.set_ylabel("Cumulative Return")
ax.set_title("Cumulative Returns")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("wmt_on_spy.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nSaved: wmt_on_spy.png")
