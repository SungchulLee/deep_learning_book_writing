# NumPy로 만드는 경사 하강법

이 스크립트는 편향 열을 포함한 설계 행렬을 써서 선형 회귀의 전체 배치 경사 하강법과 미니배치 경사 하강법을 NumPy로 바닥부터 구현한다. 합성 데이터에서 두 방식을 비교해 보면 미니배치 학습이 경사 추정에 잡음을 들여오지만 실제 소요 시간 기준으로는 더 빠르게 수렴함을 알 수 있다. 정규 방정식의 해가 기준점이 되어 두 반복법이 올바른 매개변수로 수렴하는지 확인해 준다.

## 코드

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
- 모여듦 견주기: 배치과 작은 배치

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

## 논의

설계 행렬 방식은 특징 행렬 앞에 1로 채운 열을 붙여 편향 항을 매개변수 벡터 $\theta$ 안으로 흡수시킨다. 모델은 $\hat{y} = X\theta$가 되며, 여기서 $X$의 모양은 $(n, d+1)$, $\theta$의 모양은 $(d+1,)$이다. $\theta$에 대한 MSE의 경사는 $g = \frac{2}{n} X^T(X\theta - y)$로, 모든 매개변수의 경사를 한꺼번에 계산하는 간결한 행렬 식이다. 이 벡터화된 표현은 매개변수를 하나씩 계산하는 것보다 훨씬 빠르다.

미니배치 경사 하강법은 매 에폭마다 무작위 순열로 데이터를 뒤섞은 뒤 지정한 배치 크기의 연속된 덩어리를 차례로 처리한다. 각 덩어리는 대체로 올바른 방향을 가리키는 잡음 섞인 경사 추정을 낸다. 이 잡음은 암묵적인 정칙화로 작용하여 최적화기가 얕은 지역 최솟값을 벗어나도록 돕고, 시험 집합에서 더 나은 일반화로 이어지는 경우가 많다. 그 대가로 손실 곡선에 잡음이 많아져 수렴을 눈으로 판단하기 어려워진다.

정규 방정식 $\theta^* = (X^TX)^{-1}X^Ty$은 한 단계 만에 정확한 닫힌 형태의 해를 주며, 수치적 안정성을 위해 `np.linalg.solve`로 계산한다. 반복적 경사 하강법의 해를 이 기준과 비교하면 (미니배치의 확률적 잡음과 유한한 반복 횟수만큼의 오차 안에서) 전체 배치와 미니배치 방법 모두 같은 최적점으로 수렴함을 확인할 수 있다. 특징이 수천 개인 문제에서는 정규 방정식의 비용이 커지고($O(d^3)$ 복잡도), 반복당 비용이 $O(nd)$인 경사 하강법이 실용적인 선택이 된다.

## 연습문제

**익힘 1.**
운동량 기반 경사 하강법을 구현하라. $\beta = 0.9$로 두고 $v \leftarrow \beta v + g$, $\theta \leftarrow \theta - \alpha v$이다. 보통의 경사 하강법과 수렴 속도를 비교하라.

??? success "익힘 1 풀이"
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

**익힘 2.**
수학적으로는 같은 결과를 내는데도 정규 방정식을 풀 때 `np.linalg.inv(A) @ b`보다 `np.linalg.solve(A, b)`를 선호하는 이유를 설명하라.

??? success "익힘 2 풀이"
    `np.linalg.solve`는 역행렬을 명시적으로 계산하지 않고 LU 분해나 촐레스키 분해로 연립방정식을 직접 푼다. 역행렬을 계산하는 것은 더 느리고(역행렬에 $O(d^3)$, 곱셈에 $O(d^2)$) 수치적으로도 덜 안정적이다. 특히 $X^TX$의 조건수가 나쁠 때 역행렬이 반올림 오차를 증폭시키기 때문이다. `solve`는 분해된 형태를 직접 다루어 이런 증폭을 피하고, 더 짧은 시간에 더 정확한 해를 낸다.

---

**익힘 3.**
배치 크기 1, 16, 64, 300(전체 배치)으로 미니배치 경사 하강법을 실행하라. 네 손실 곡선을 같은 축에 그리고 잡음과 수렴 사이의 절충을 논하라.

??? success "익힘 3 풀이"
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
