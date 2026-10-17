# SPY에 대한 WMT 회귀

WMT와 SPY의 같은 날짜 일별 수익률을 맞춰 선형 시장모형을 적합한다. 세 가지 구현의 최소제곱 결과를 비교하고 이동 구간에서 추정한 그래디언트의 변화를 살펴본다. 무위험 수익률을 빼지 않은 이 코드는 CAPM 초과수익률 회귀와 구분한다.

## 1. 코드

```python
"""
SPY에 대한 WMT — CAPM 베타 어림
====================================

WMT의 날마다 수익률을 SPY의 날마다 수익률에 회귀시켜 월마트의 시장 베타를
어림한다.  넘파이, 사이킷런, PyTorch 세 가지로 짠다.

Demonstrates:
- 데이터 내려받기에 쓰는 yfinance
- 수익률 계산(pct_change)
- CAPM 회귀: R_WMT = α + β R_SPY + ε
- 구르는 베타 어림
- 최소제곱, 사이킷런, PyTorch 견주기

지은이: 깊은 학습 바탕 학습 차례

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

import pandas as pd


def synthetic_returns(n=1000, beta=0.5, seed=0):
    """내려받기가 안 될 때 쓸 흉내 낸 수익률. 참 베타는 0.5이다."""
    rng = np.random.default_rng(seed)
    spy = rng.normal(0.0004, 0.012, n)
    wmt = 0.0001 + beta * spy + rng.normal(0, 0.008, n)
    return pd.DataFrame({"SPY": spy, "WMT": wmt})


returns = None
try:
    import yfinance as yf

    tickers = ["WMT", "SPY"]
    data = yf.download(tickers, start=ARGS.start, end=ARGS.end)["Adj Close"]
    data = data.dropna()
    if len(data) > 1:
        returns = data.pct_change().dropna()
        returns.columns = ["SPY", "WMT"]
except ImportError:
    print("yfinance가 없다 — 흉내 낸 자료를 쓴다.")

# yfinance가 있어도 망 연결이 없으면 빈 표가 돌아온다. 그대로 두면
# X^T X 가 특이 행렬이 되어 알아보기 어려운 LinAlgError로 터지므로,
# 자료가 비었는지 여기서 확인하고 흉내 낸 자료로 갈아탄다.
if returns is None or len(returns) < 30:
    print("시장 자료를 받지 못했다 — 흉내 낸 자료를 쓴다(참 베타 = 0.5).")
    returns = synthetic_returns()

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

**출력:**

```
YF.download() has changed argument auto_adjust default to True
시장 자료를 받지 못했다 — 흉내 낸 자료를 쓴다(참 베타 = 0.5).
Observations: 1000
SPY  mean=-0.000176 std=0.0117
WMT  mean=-0.000052 std=0.0103
Correlation: 0.6133

--- NumPy Normal Equation ---
α = 0.000043
β = 0.5410
R² = 0.3761

--- Sklearn ---
α = 0.000043
β = 0.5410

--- PyTorch ---
α = 0.830067
β = 0.7358

--- Rolling Beta (window=60) ---
Rolling β — mean: 0.5447, std: 0.0975
Rolling β — min: 0.2645, max: 0.7536

Saved: wmt_on_spy.png
```

## 2. 논의

### 베타와 총위험의 차이

회귀 그래디언트 $\beta=\operatorname{Cov}(R_{\mathrm{SPY}},R_{\mathrm{WMT}})/\operatorname{Var}(R_{\mathrm{SPY}})$는 표본에서 시장 대용치와의 선형 민감도를 나타낸다. 종목 총변동성이나 인과효과와 같지 않다. 베타가 1보다 작더라도 잔차 위험이 크면 총변동성은 높을 수 있다.

원수익률 회귀의 절편은 조건부 선형 관계의 상수 항이며, 무위험 수익률을 반영한 CAPM 알파와 같다고 부르면 안 된다. 추정값은 관측 기간, 빈도, 배당·분할을 반영하는 가격 조정과 날짜 정렬에 따라 달라진다. 제공된 출력의 특정 베타를 현재 종목의 고정 특성으로 해석하지 않는다.

### 방법과 이동 구간 비교

같은 데이터와 절편 조건의 최소제곱 해를 비교한다. 반복적 PyTorch 결과는 반복 수와 러닝레이트에 따른 최적화 오차가 있으며 직접 해법에도 반올림 오차가 있다. 구현 간 일치는 동일 목적의 계산을 점검하는 근거이지 금융 모형 가정이 맞다는 증명은 아니다.

이동 베타의 변화에는 실제 관계 변화와 짧은 표본의 추정 변동이 함께 들어 있다. 이분산성과 자기상관이 있으면 기본 OLS 표준오차의 신뢰구간이 부정확할 수 있다. 구간 길이와 추정 불확실성을 함께 보고 지표만으로 투자 판단을 정당화하지 않는다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
동일한 기간·빈도·가격 조정으로 다른 종목의 베타와 WMT의 베타를 비교할 때 어떤 점을 해석해야 하는가?

</div>

??? success "풀이"
    계산한 표본 베타와 추정 오차를 비교한다. 특정 업종의 베타가 항상 높거나 WMT가 항상 특정 구간이라는 결론은 내리지 않는다. 시장 공분산과 시장 분산의 비율이라는 정의에 따라 해석하고 기간에 대한 민감도도 확인한다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff easy" title="쉬움"></span>
OLS 회귀 이론의 표준오차 공식을 사용하여 베타 추정치에 대한 95% 신뢰구간을 계산하라.

</div>

??? success "풀이"
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

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
이동 알파를 이동 베타와 나란히 그려라. 알파가 계속 양수인가, 음수인가, 아니면 0 주위에서 오르내리는가? 이는 WMT의 위험 조정 성과에 대해 무엇을 뜻하는가?

</div>

??? success "풀이"
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

## 정리하며

시장 수익률 회귀는 표본의 선형 민감도를 추정한다. 초과수익률 CAPM과의 차이, 가격·날짜 전처리, 이동 창의 추정 오차를 구분해 결과를 읽는다.
