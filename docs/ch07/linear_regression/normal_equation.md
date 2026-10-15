# 정규 방정식

선형 회귀의 최소제곱 해는 정규 방정식을 만족한다. 이 절에서는 편향 열을 포함한 설계 행렬을 만들고 NumPy의 선형 방정식 해법과 최소제곱 솔버를 비교한다. 역행렬 식은 설계 행렬이 열 전체 랭크일 때만 성립한다.

## 1. 코드

```python
"""
정규 방정식 — 넘파이 짜보기
========================================

선형 회귀의 닫힌 꼴 풀이:

    θ* = (X^T X)^{-1} X^T y

Demonstrates:
- 설계 행렬 짓기(편향을 위해 1을 앞에 붙인다)
- np.linalg.solve으로 정규 방정식 풀기(수치가 든든하다)
- np.linalg.lstsq과 견주기
- 학습/시험 나누기와 평가

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

parser = argparse.ArgumentParser(description="Normal equation linear regression")
parser.add_argument("--n-samples", type=int, default=200, help="number of samples")
parser.add_argument("--n-features", type=int, default=3, help="number of features")
parser.add_argument("--noise", type=float, default=10.0, help="noise std dev")
parser.add_argument("--seed", type=int, default=42, help="random seed")
ARGS = parser.parse_args()

np.random.seed(ARGS.seed)

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
print(f"Train: {x_train.shape}, Test: {x_test.shape}")

# ============================================================================
# 설계 행렬
# ============================================================================


def make_design_matrix(x: np.ndarray) -> np.ndarray:
    """앞에 1로 채운 열을 붙인다: X = [1 | x]."""
    return np.hstack([np.ones((x.shape[0], 1)), x])


X_train = make_design_matrix(x_train)
X_test = make_design_matrix(x_test)

# ============================================================================
# 정규 방정식
# ============================================================================


def fit_normal_equation(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """solve로 θ* = (X^T X)^{-1} X^T y를 구한다 (역행렬을 명시적으로 만들지 않는다)."""
    return np.linalg.solve(X.T @ X, X.T @ y)


theta = fit_normal_equation(X_train, y_train)
print(f"\nFitted parameters (bias first): {theta}")

# ============================================================================
# lstsq와 비교
# ============================================================================

theta_lstsq, residuals, rank, sv = np.linalg.lstsq(X_train, y_train, rcond=None)
print(f"lstsq parameters:               {theta_lstsq}")
print(f"Max difference: {np.max(np.abs(theta - theta_lstsq)):.2e}")

# ============================================================================
# 예측과 평가
# ============================================================================


def predict(X: np.ndarray, theta: np.ndarray) -> np.ndarray:
    return X @ theta


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    residual = y_true - y_pred
    ss_res = np.sum(residual ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    mse = np.mean(residual ** 2)
    return {
        "mse": mse,
        "rmse": np.sqrt(mse),
        "mae": np.mean(np.abs(residual)),
        "r2": 1.0 - ss_res / ss_tot,
    }


y_pred_train = predict(X_train, theta)
y_pred_test = predict(X_test, theta)

print(f"\nTrain: {evaluate(y_train, y_pred_train)}")
print(f"Test:  {evaluate(y_test, y_pred_test)}")

# ============================================================================
# 시각화
# ============================================================================

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax = axes[0]
ax.scatter(y_test, y_pred_test, alpha=0.7, edgecolors="black", linewidths=0.5)
lims = [min(y_test.min(), y_pred_test.min()), max(y_test.max(), y_pred_test.max())]
ax.plot(lims, lims, "r--", lw=1.5)
ax.set_xlabel("True y")
ax.set_ylabel("Predicted y")
ax.set_title("Predicted vs True")
ax.grid(True, alpha=0.3)

ax = axes[1]
residuals_test = y_test - y_pred_test
ax.hist(residuals_test, bins=20, density=True, alpha=0.7, edgecolor="black")
ax.set_xlabel("Residual")
ax.set_ylabel("Density")
ax.set_title(f"Residual Distribution (σ = {residuals_test.std():.2f})")
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("normal_equation.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nSaved: normal_equation.png")


if __name__ == "__main__":
    pass
```

**출력:**

```
Train: (160, 3), Test: (40, 3)

Fitted parameters (bias first): [-0.6364495  71.88644879 22.54243658 72.4845378 ]
lstsq parameters:               [-0.6364495  71.88644879 22.54243658 72.4845378 ]
Max difference: 7.11e-14

Train: {'mse': 103.22041866862355, 'rmse': 10.159745010019865, 'mae': 8.118340998857205, 'r2': 0.9901213716370256}
Test:  {'mse': 132.62372739076085, 'rmse': 11.516237553591921, 'mae': 9.24283824392548, 'r2': 0.9850367614656054}

Saved: normal_equation.png
```

## 2. 논의

설계 행렬은 특징 행렬 앞에 1로 채운 열을 붙여 만들며, 이로써 편향 항이 매개변수 벡터 안으로 흡수된다. 덕분에 모델 전체 $\hat{y} = w_1 x_1 + \cdots + w_p x_p + b$를 하나의 행렬 방정식 $\hat{y} = X\theta$로 쓸 수 있다. 그러면 정규 방정식이 반복적 최적화 없이 한 번의 계산으로 MSE 손실의 정확한 최소화점을 준다.

정규 방정식의 계수 행렬을 직접 역행렬로 만들기보다 `np.linalg.solve()`로 선형계를 푸는 편이 보통 낫다. 다만 그람 행렬을 만들 때 생기는 조건수 제곱 문제까지 없애지는 못한다. `np.linalg.lstsq()`는 원래 설계 행렬의 최소제곱 문제를 직접 다루므로 랭크 부족과 나쁜 조건수에 더 적합할 수 있다. 실제 속도는 크기와 구현에 따라 달라진다.

평가에는 MSE, RMSE, MAE와 $R^2$를 함께 사용한다. OLS 자체가 정규 잔차를 요구하는 것은 아니다. 가우스 오차를 가정한 통계 추론을 사용할 때는 잔차 진단으로 가정의 타당성을 살펴본다. 편향을 포함한 학습 표본의 잔차 합이 0이라는 대수적 성질과 실제 오차의 정규성은 별개다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff hard" title="어려움"></span>
`np.linalg.solve` 대신 `np.linalg.inv`로 정규 방정식을 구현하고 결과가 일치하는지 확인하라. 그다음 서로 강하게 상관된 특징을 가진 데이터셋을 만들어 두 방식의 수치적 안정성을 비교하라.

</div>

??? success "연습문제 1 풀이"
    ```python
    import numpy as np
    from sklearn.datasets import make_regression
    
    X, y = make_regression(n_samples=200, n_features=3, noise=10, random_state=42)
    X_design = np.hstack([np.ones((len(X), 1)), X])
    
    # 방법 1: solve
    theta_solve = np.linalg.solve(X_design.T @ X_design, X_design.T @ y)
    
    # 방법 2: inv
    theta_inv = np.linalg.inv(X_design.T @ X_design) @ X_design.T @ y
    
    print(f'Max difference: {np.max(np.abs(theta_solve - theta_inv)):.2e}')
    # 조건이 좋은 데이터에서는 차이가 기계 엡실론 수준이다.
    # 조건이 나쁜 데이터에서는 solve가 더 정확하다.
    ```

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff hard" title="어려움"></span>
MSE의 경사를 0으로 두어 정규 방정식을 유도하고 해의 유일성 조건을 설명하라.

</div>

??? success "연습문제 2 풀이"
    $L=\|y-X\theta\|^2/n$의 경사는 $2X^\top(X\theta-y)/n$이다. 이를 0으로 두면 $X^\top X\theta=X^\top y$를 얻는다. $X$가 열 전체 랭크이면 역행렬 식으로 유일한 해를 얻는다. 그렇지 않으면 정규 방정식은 여전히 최소제곱 조건이지만 파라미터 해는 유일하지 않으며 최소 노름 해에는 의사역행렬을 사용할 수 있다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
$n = 1000$이고 $p \in \{10, 100, 500\}$인 데이터셋에 대해 정규 방정식과 경사 하강법(500 에포크)의 실제 소요 시간을 비교하라. 특징 차원이 얼마부터 경사 하강법이 더 빨라지는가?

</div>

??? success "연습문제 3 풀이"
    ```python
    import numpy as np, time
    from sklearn.datasets import make_regression
    
    for p in [10, 100, 500]:
        X, y = make_regression(n_samples=1000, n_features=p, noise=10, random_state=42)
        X_d = np.hstack([np.ones((len(X), 1)), X])
    
        # 정규 방정식
        t0 = time.time()
        theta_ne = np.linalg.solve(X_d.T @ X_d, X_d.T @ y)
        t_ne = time.time() - t0
    
        # 경사 하강법
        t0 = time.time()
        theta = np.zeros(X_d.shape[1])
        for _ in range(500):
            grad = (2.0/len(y)) * (X_d.T @ (X_d @ theta - y))
            theta -= 0.001 * grad
        t_gd = time.time() - t0
    
        print(f'p={p:3d}: Normal eq={t_ne:.4f}s, GD={t_gd:.4f}s')
    # 정규 방정식의 비용은 O(p^3)이고 GD의 비용은 O(n*p*epochs)이다.
    # p가 크면(대략 p > 500) GD가 더 빨라진다.
    ```

## 정리하며

정규 방정식은 최소제곱의 최적성 조건이다. 역행렬 공식의 존재 조건을 확인하고, 수치 계산에서는 조건수와 랭크를 고려한 최소제곱 솔버를 사용한다.
