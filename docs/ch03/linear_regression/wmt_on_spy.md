# SPY에 대한 WMT 회귀

이 스크립트는 WMT의 일별 수익률을 SPY의 일별 수익률에 회귀시켜 월마트의 시장 베타를 추정하며, 자본자산가격결정모형(CAPM) 회귀 $R_{\text{WMT}} = \alpha + \beta R_{\text{SPY}} + \varepsilon$을 구현한다. NumPy 정규 방정식, scikit-learn, PyTorch의 세 가지 구현을 비교하고, 이동 창 분석으로 베타가 시간에 따라 어떻게 변하는지 살펴본다. 이는 선형 회귀를 금융 데이터에 적용한 실용적인 예이다.

## 코드

```python
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
# 설정
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
# 데이터
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
# 방법 1: NumPy 정규 방정식
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
# 방법 2: Sklearn
# ============================================================================

print("\n--- Sklearn ---")
from sklearn.linear_model import LinearRegression

model_sk = LinearRegression()
model_sk.fit(x.reshape(-1, 1), y)
print(f"α = {model_sk.intercept_:.6f}")
print(f"β = {model_sk.coef_[0]:.4f}")

# ============================================================================
# 방법 3: PyTorch
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
# 이동 베타
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
# 시각화
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# (1) 산점도 + 회귀 직선
ax = axes[0, 0]
ax.scatter(x, y, alpha=0.3, s=8)
x_line = np.linspace(x.min(), x.max(), 100)
ax.plot(x_line, alpha_np + beta_np * x_line, "r-", lw=2, label=f"β = {beta_np:.3f}")
ax.set_xlabel("SPY Return")
ax.set_ylabel("WMT Return")
ax.set_title("WMT vs SPY — CAPM Regression")
ax.legend()
ax.grid(True, alpha=0.3)

# (2) 잔차 히스토그램
ax = axes[0, 1]
residuals = y - (alpha_np + beta_np * x)
ax.hist(residuals, bins=50, density=True, alpha=0.7, edgecolor="black")
ax.set_xlabel("Residual")
ax.set_ylabel("Density")
ax.set_title(f"Residual Distribution (σ = {residuals.std():.4f})")
ax.grid(True, alpha=0.3)

# (3) 이동 베타
ax = axes[1, 0]
ax.plot(betas, lw=0.8)
ax.axhline(beta_np, color="r", ls="--", lw=1.5, label=f"Full-sample β = {beta_np:.3f}")
ax.set_xlabel(f"Day (after {ARGS.rolling_window}-day warmup)")
ax.set_ylabel("Rolling β")
ax.set_title(f"WMT {ARGS.rolling_window}-Day Rolling Beta")
ax.legend()
ax.grid(True, alpha=0.3)

# (4) 누적 수익률
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


if __name__ == "__main__":
    pass
```

## 논의

CAPM 베타는 어떤 주식의 수익률이 시장 움직임에 얼마나 민감한지를 잰다. 베타가 1.0이면 주식이 시장과 보조를 맞춰 움직이고, 1.0보다 작으면 변동성이 낮으며(방어적), 1.0보다 크면 변동성이 높다(공격적). 월마트는 대형 필수소비재 기업이라 식료품 수요가 경기 순환에 비교적 둔감하므로 보통 베타가 1.0보다 작다. 절편(알파)은 시장 움직임으로 설명되는 것을 넘어서는 그 주식의 초과 수익률을 나타낸다.

NumPy 정규 방정식, scikit-learn, PyTorch 경사 하강법이라는 세 추정 방법은 모두 같은 최소제곱 문제를 풀므로 동일한 결과를 내야 한다. NumPy와 scikit-learn 방법은 정확한 해를 찾고, PyTorch는 반복적 최적화로 같은 답에 수렴한다. 방법들 사이의 일치가 각 구현이 옳다는 확인이 된다.

이동 베타 추정은 미끄러지는 창(예: 거래일 60일) 위에서 회귀를 적합시켜 주식의 시장 민감도가 시간에 따라 어떻게 변하는지를 포착한다. 베타는 상수가 아니어서 사업 전략, 시장 상황, 업종 순환에 따라 달라질 수 있으므로 이는 중요하다. 이동 베타 그림이 이런 움직임을 드러내는 반면, 전체 표본 베타는 하나의 요약 통계량을 준다. 이동 베타가 크게 출렁인다면 그 주식의 위험 성격이 불안정하다는 뜻이다.

## 연습문제

**Exercise 1.**
WMT를 기술주(예: AAPL이나 NVDA)로 바꾸어 추정된 베타를 월마트의 것과 비교하라. 어느 주식의 베타가 더 높으며 그 이유는 무엇인가?

??? success "Solution to Exercise 1"
    NVDA 같은 기술주는 매출이 경기 순환과 투자 심리에 더 민감하므로 보통 베타가 1.0을 크게 웃돈다(흔히 1.5-2.0). 월마트(WMT)는 필수소비재 수요가 비교적 안정적이라 베타가 0.4-0.6 정도이다. 기술주의 높은 베타는 더 큰 체계적 위험을 반영한다. 이들은 시장의 움직임을 양쪽 방향으로 증폭시킨다.

---

**Exercise 2.**
OLS 회귀 이론의 표준오차 공식을 사용하여 베타 추정치에 대한 95% 신뢰구간을 계산하라.

??? success "Solution to Exercise 2"
    ```python
    import numpy as np
    
    # 적합 후: x = SPY 수익률, y = WMT 수익률
    X_np = np.column_stack([np.ones_like(x), x])
    theta = np.linalg.solve(X_np.T @ X_np, X_np.T @ y)
    y_pred = X_np @ theta
    residuals = y - y_pred
    n_obs = len(y)
    p_params = 2  # alpha and beta
    
    # 베타의 표준오차
    sigma_sq = np.sum(residuals**2) / (n_obs - p_params)
    cov_matrix = sigma_sq * np.linalg.inv(X_np.T @ X_np)
    se_beta = np.sqrt(cov_matrix[1, 1])
    
    # 95% CI
    from scipy.stats import t
    t_crit = t.ppf(0.975, df=n_obs - p_params)
    ci_lower = theta[1] - t_crit * se_beta
    ci_upper = theta[1] + t_crit * se_beta
    print(f'Beta = {theta[1]:.4f}, 95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]')
    ```

---

**Exercise 3.**
이동 알파를 이동 베타와 나란히 그려라. 알파가 계속 양수인가, 음수인가, 아니면 0 주위에서 오르내리는가? 이는 WMT의 위험 조정 성과에 대해 무엇을 뜻하는가?

??? success "Solution to Exercise 3"
    ```python
    import numpy as np
    
    alphas, betas = [], []
    window = 60
    for i in range(window, len(x)):
        x_w = x[i-window:i]
        y_w = y[i-window:i]
        X_w = np.column_stack([np.ones_like(x_w), x_w])
        theta_w = np.linalg.solve(X_w.T @ X_w, X_w.T @ y_w)
        alphas.append(theta_w[0])
        betas.append(theta_w[1])
    
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax1.plot(betas); ax1.set_ylabel('Rolling Beta'); ax1.grid(True, alpha=0.3)
    ax2.plot(alphas); ax2.axhline(0, color='r', ls='--'); ax2.set_ylabel('Rolling Alpha')
    ax2.grid(True, alpha=0.3)
    plt.show()
    # 알파가 0 주위에서 오르내린다면 WMT의 수익률은 그 시장 위험에 걸맞은
    # 수준이라는 뜻이다. 알파가 계속 양수라면 위험을 조정한 뒤에도
    # 초과 성과를 냈다는 뜻이 된다.
    ```
