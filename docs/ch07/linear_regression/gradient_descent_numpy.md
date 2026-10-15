# NumPy로 만드는 경사 하강법

편향 열을 포함한 설계 행렬로 전체 배치와 미니배치 경사 하강법을 NumPy에서 구현한다. 같은 최소제곱 목적을 다른 업데이트 단위로 처리하고, 손실과 경사, 실행 비용을 구분해 비교한다.

## 1. 코드

```python
"""
선형 회귀를 위한 경사 하강법 — 넘파이
================================================

배치 경사 하강법과 작은 배치 경사 하강법을 밑바닥부터 짠다.

Demonstrates:
- 편향 칸을 지닌 설계 행렬
- 온 배치 기울기: g = (2/n) X^T (Xθ - y)
- rng.permutation으로 하는 작은 배치 섞기
- 학습률에 얼마나 흔들리는지 살피기
- 모여듦 견주기: 배치와 작은 배치

지은이: 깊은 학습 바탕 학습 차례
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

# ============================================================================
# 설정
# ============================================================================

parser = argparse.ArgumentParser(description="gradient descent linear regression")
parser.add_argument("--n-samples", type=int, default=300)
parser.add_argument("--n-features", type=int, default=3)
parser.add_argument("--noise", type=float, default=10.0)
parser.add_argument("--lr", type=float, default=0.01, help="learning rate")
parser.add_argument("--epochs", type=int, default=500)
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--seed", type=int, default=42)
ARGS = parser.parse_args()

rng = np.random.default_rng(ARGS.seed)

# ============================================================================
# 데이터
# ============================================================================

x, y = make_regression(
    n_samples=ARGS.n_samples,
    n_features=ARGS.n_features,
    noise=ARGS.noise,
    random_state=ARGS.seed,
)
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=ARGS.seed,
)


def make_design_matrix(x: np.ndarray) -> np.ndarray:
    return np.hstack([np.ones((x.shape[0], 1)), x])


X_train = make_design_matrix(x_train)
X_test = make_design_matrix(x_test)
n, d = X_train.shape
print(f"Train: ({n}, {d}), Test: {X_test.shape}")

# ============================================================================
# 배치 경사 하강법
# ============================================================================


def batch_gd(X, y, lr, epochs):
    theta = np.zeros(X.shape[1])
    losses = []
    for _ in range(epochs):
        residual = X @ theta - y
        loss = np.mean(residual ** 2)
        losses.append(loss)
        grad = (2.0 / len(y)) * (X.T @ residual)
        theta -= lr * grad
    return theta, losses


theta_batch, losses_batch = batch_gd(X_train, y_train, ARGS.lr, ARGS.epochs)
print(f"\nBatch GD — final MSE: {losses_batch[-1]:.4f}")

# ============================================================================
# 미니배치 경사 하강법
# ============================================================================


def minibatch_gd(X, y, lr, epochs, batch_size, rng):
    n = len(y)
    theta = np.zeros(X.shape[1])
    losses = []
    for _ in range(epochs):
        perm = rng.permutation(n)
        X_s, y_s = X[perm], y[perm]
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            X_b, y_b = X_s[start:end], y_s[start:end]
            residual = X_b @ theta - y_b
            grad = (2.0 / len(y_b)) * (X_b.T @ residual)
            theta -= lr * grad
        full_loss = np.mean((X @ theta - y) ** 2)
        losses.append(full_loss)
    return theta, losses


theta_mini, losses_mini = minibatch_gd(
    X_train, y_train, ARGS.lr, ARGS.epochs, ARGS.batch_size, rng,
)
print(f"Mini-batch GD — final MSE: {losses_mini[-1]:.4f}")

# ============================================================================
# 정규 방정식 기준값
# ============================================================================

theta_exact = np.linalg.solve(X_train.T @ X_train, X_train.T @ y_train)
y_pred_exact = X_test @ theta_exact
mse_exact = np.mean((y_test - y_pred_exact) ** 2)
print(f"Normal equation — test MSE: {mse_exact:.4f}")

# ============================================================================
# 평가
# ============================================================================

for name, theta in [("Batch GD", theta_batch), ("Mini-batch GD", theta_mini)]:
    y_pred = X_test @ theta
    mse = np.mean((y_test - y_pred) ** 2)
    ss_res = np.sum((y_test - y_pred) ** 2)
    ss_tot = np.sum((y_test - y_test.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot
    diff = np.max(np.abs(theta - theta_exact))
    print(f"{name:15s}  test MSE={mse:.4f}  R²={r2:.4f}  max|Δθ|={diff:.2e}")

# ============================================================================
# 수렴 그림
# ============================================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
ax.plot(losses_batch, label="Batch GD", alpha=0.8)
ax.plot(losses_mini, label="Mini-batch GD", alpha=0.8)
ax.set_xlabel("Epoch")
ax.set_ylabel("MSE")
ax.set_title("Convergence: Batch vs Mini-batch")
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.semilogy(losses_batch, label="Batch GD", alpha=0.8)
ax.semilogy(losses_mini, label="Mini-batch GD", alpha=0.8)
ax.set_xlabel("Epoch")
ax.set_ylabel("MSE (log scale)")
ax.set_title("Convergence (Log Scale)")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("gradient_descent_numpy.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nSaved: gradient_descent_numpy.png")


if __name__ == "__main__":
    pass
```

**출력:**

```
Train: (240, 4), Test: (60, 4)

Batch GD — final MSE: 94.0956
Mini-batch GD — final MSE: 94.1023
Normal equation — test MSE: 75.9175
Batch GD         test MSE=75.8865  R²=0.9954  max|Δθ|=1.83e-02
Mini-batch GD    test MSE=76.1105  R²=0.9954  max|Δθ|=5.71e-02

Saved: gradient_descent_numpy.png
```

## 2. 논의

설계 행렬 방식은 특징 행렬 앞에 1로 채운 열을 붙여 편향 항을 매개변수 벡터 $\theta$ 안으로 흡수시킨다. 모델은 $\hat{y} = X\theta$가 되며, 여기서 $X$의 모양은 $(n, d+1)$, $\theta$의 모양은 $(d+1,)$이다. $\theta$에 대한 MSE의 경사는 $g = \frac{2}{n} X^T(X\theta - y)$로, 모든 매개변수의 경사를 한꺼번에 계산하는 간결한 행렬 식이다. 이 벡터화된 표현은 매개변수를 하나씩 계산하는 것보다 훨씬 빠르다.

미니배치는 학습 데이터를 나누어 부분 경사를 계산한다. 고정한 파라미터에서 균등 추출하면 평균 경사는 전체 경사와 연결되지만, 연속 업데이트에서는 파라미터도 변한다. 이 선형 최소제곱 문제는 볼록하므로 “나쁜 국소 최솟값에서 탈출한다”는 설명은 적절하지 않다. 경사 잡음이 수렴 속도와 일반화에 미치는 효과는 데이터와 설정에 따라 확인한다.

열 전체 랭크에서 정규 방정식의 해는 $\theta^*=(X^\top X)^{-1}X^\top y$다. 수치 계산에서는 최소제곱 솔버를 비교 기준으로 쓰는 것이 좋다. 전체 배치 GD는 적절한 학습률에서 최적해에 수렴하지만, 고정 학습률의 미니배치는 잡음 때문에 그 주변에서 변동할 수 있다. 정규 방정식 비용에는 그람 행렬을 만드는 $O(nd^2)$와 선형계를 푸는 비용이 함께 포함된다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff hard" title="어려움"></span>
운동량 기반 경사 하강법을 구현하라. $\beta = 0.9$로 두고 $v \leftarrow \beta v + g$, $\theta \leftarrow \theta - \alpha v$이다. 보통의 경사 하강법과 수렴 속도를 비교하라.

</div>

??? success "연습문제 1 풀이"
    ```python
    import numpy as np
    from sklearn.datasets import make_regression
    
    X, y = make_regression(n_samples=300, n_features=3, noise=10, random_state=42)
    X_design = np.hstack([np.ones((len(X), 1)), X])
    
    theta = np.zeros(X_design.shape[1])
    v = np.zeros_like(theta)
    lr, beta = 0.01, 0.9
    
    for epoch in range(500):
        residual = X_design @ theta - y
        grad = (2.0 / len(y)) * (X_design.T @ residual)
        v = beta * v + grad
        theta -= lr * v
    
    print(f'Final theta: {theta}')
    # 운동량을 쓰면 보통의 GD보다 대체로 적은 에폭에 수렴한다.
    ```

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
`solve`를 쓰면 그람 행렬의 나쁜 조건수 문제가 모두 해결되는가?

</div>

??? success "연습문제 2 풀이"
    아니다. 명시적 역행렬을 피하는 이점은 있지만 $X^\top X$를 만들며 조건수가 제곱되는 문제는 남는다. 원래 설계 행렬에 QR·SVD 기반 최소제곱 해법을 적용하면 이를 피할 수 있다. 두 방법 모두 부동소수점 계산이므로 정확한 실수 해와는 수치 오차가 있을 수 있다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
배치 크기 1, 16, 64, 300(전체 배치)으로 미니배치 경사 하강법을 실행하라. 네 손실 곡선을 같은 축에 그리고 잡음과 수렴 사이의 절충을 논하라.

</div>

??? success "연습문제 3 풀이"
    ```python
    import numpy as np
    from sklearn.datasets import make_regression
    import matplotlib.pyplot as plt
    
    X, y = make_regression(n_samples=300, n_features=3, noise=10, random_state=42)
    X_design = np.hstack([np.ones((len(X), 1)), X])
    rng = np.random.default_rng(42)
    
    for bs in [1, 16, 64, 300]:
        theta = np.zeros(X_design.shape[1])
        losses = []
        for epoch in range(200):
            perm = rng.permutation(len(y))
            for start in range(0, len(y), bs):
                end = min(start + bs, len(y))
                Xb, yb = X_design[perm[start:end]], y[perm[start:end]]
                grad = (2.0/len(yb)) * (Xb.T @ (Xb @ theta - yb))
                theta -= 0.01 * grad
            losses.append(np.mean((X_design @ theta - y)**2))
        plt.plot(losses, label=f'BS={bs}')
    
    plt.xlabel('Epoch'); plt.ylabel('MSE'); plt.legend()
    plt.title('Batch Size Comparison'); plt.show()
    # BS=1이 가장 잡음이 많고, BS=300은 가장 매끄럽지만 에폭당 수렴이 가장 느리다.
    ```

## 정리하며

배치 방식이 달라도 손실과 경사의 분모를 일치시킨다. 반복법은 수렴 조건과 경사 변동을 확인하고, 해석적 기준도 랭크와 수치 조건에 맞는 솔버로 계산한다.
