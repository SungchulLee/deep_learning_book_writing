# 닫힌 형태의 해

반복적 최적화가 필요한 대부분의 기계 학습 문제와 달리, 선형 회귀에는 **정규 방정식**이라
불리는 **닫힌 형태**의 해가 있다. 이 페이지에서는 제일원리에서부터 해를 전개한다.
필요한 벡터 미적분 항등식, 유도 과정 자체, 직교 사영으로서의 기하학적 해석, 그리고
NumPy와 PyTorch에서의 효율적인 수치 구현을 차례로 다룬다.

---

## 1. 벡터 미적분 예비 지식

### 1.1 표기

| 기호 | 의미 |
|--------|---------|
| $\mathbf{X} \in \mathbb{R}^{n \times (p+1)}$ | 설계 행렬 (편향 열 포함) |
| $\mathbf{y} \in \mathbb{R}^{n}$ | 목표 벡터 |
| $\boldsymbol{\theta} \in \mathbb{R}^{p+1}$ | 매개변수 벡터 |
| $\|\mathbf{v}\|^2 = \mathbf{v}^\top\mathbf{v}$ | 유클리드 노름의 제곱 |

### 1.2 선형형식의 경사

상수 벡터 $\mathbf{a}$에 대해 다음이 성립한다.

$$
\frac{\partial}{\partial \boldsymbol{\theta}}\,
\mathbf{a}^\top \boldsymbol{\theta}
= \mathbf{a}
$$

**증명.** $\mathbf{a}^\top\boldsymbol{\theta} = \sum_j a_j \theta_j$이다.
$\theta_k$에 대해 미분하면 $a_k$를 얻는다. $\square$

### 1.3 이차형식의 경사

**대칭** 행렬 $\mathbf{A}$에 대해 다음이 성립한다.

$$
\frac{\partial}{\partial \boldsymbol{\theta}}\,
\boldsymbol{\theta}^\top \mathbf{A}\,\boldsymbol{\theta}
= 2\mathbf{A}\boldsymbol{\theta}
$$

**증명.**
$\boldsymbol{\theta}^\top\mathbf{A}\boldsymbol{\theta}
= \sum_j \sum_k A_{jk}\theta_j\theta_k$로 전개하고 $\theta_i$에 대해 미분하면
다음과 같다.

$$
\frac{\partial}{\partial \theta_i}
= \sum_k A_{ik}\theta_k + \sum_j A_{ji}\theta_j
= (\mathbf{A}\boldsymbol{\theta})_i

  + (\mathbf{A}^\top\boldsymbol{\theta})_i
= 2(\mathbf{A}\boldsymbol{\theta})_i
$$

마지막 단계에서 $\mathbf{A} = \mathbf{A}^\top$을 사용했다. $\square$

!!! warning "비대칭인 경우"
    $\mathbf{A}$가 대칭이 아니면 경사는
    $(\mathbf{A} + \mathbf{A}^\top)\boldsymbol{\theta}$이 된다. 그람 행렬
    $\mathbf{X}^\top\mathbf{X}$은 언제나 대칭이므로 보통최소제곱에서는 이 구분이
    문제되지 않는다.

### 1.4 대각합 항등식

MLE 유도와 정칙화된 손실에서는 여러 대각합 항등식이 등장한다.

| 항등식 | 공식 |
|----------|---------|
| 순환 성질 | $\mathrm{tr}(\mathbf{ABC}) = \mathrm{tr}(\mathbf{CAB}) = \mathrm{tr}(\mathbf{BCA})$ |
| 스칼라를 대각합으로 | $\mathbf{v}^\top\mathbf{v} = \mathrm{tr}(\mathbf{v}\mathbf{v}^\top)$ |
| 대각합의 도함수 | $\frac{\partial}{\partial \mathbf{A}}\mathrm{tr}(\mathbf{BA}) = \mathbf{B}^\top$ |
| 로그 행렬식의 도함수 | $\frac{\partial}{\partial \mathbf{A}}\ln\|\det\mathbf{A}\| = (\mathbf{A}^{-1})^\top$ |

스칼라를 대각합으로 쓰는 항등식 덕분에 MSE 손실을 다음과 같이 쓸 수 있다.

$$
\mathcal{L}
= \frac{1}{n}\,\mathrm{tr}\!\bigl[
  (\mathbf{y} - \mathbf{X}\boldsymbol{\theta})
  (\mathbf{y} - \mathbf{X}\boldsymbol{\theta})^\top
\bigr]
$$

이 형태는 잡음 분산 $\sigma^2$에 대해서도 동시에 최적화할 때 유용하다.

---

## 2. 정규 방정식의 유도

### 2.1 손실 전개하기

MSE 손실은 다음과 같다.

$$
\mathcal{L}(\boldsymbol{\theta})
= \frac{1}{n}\|\mathbf{y} - \mathbf{X}\boldsymbol{\theta}\|^2
= \frac{1}{n}\bigl(
  \mathbf{y}^\top\mathbf{y}

  - 2\boldsymbol{\theta}^\top\mathbf{X}^\top\mathbf{y}
  + \boldsymbol{\theta}^\top\mathbf{X}^\top\mathbf{X}\boldsymbol{\theta}
\bigr)
$$

### 2.2 경사 계산하기

§1의 항등식을 적용하면 다음과 같다.

$$
\nabla_{\boldsymbol{\theta}}\mathcal{L}
= \frac{1}{n}\bigl(
  -2\mathbf{X}^\top\mathbf{y}

  + 2\mathbf{X}^\top\mathbf{X}\boldsymbol{\theta}
\bigr)
= \frac{2}{n}\,\mathbf{X}^\top\!
  \bigl(\mathbf{X}\boldsymbol{\theta} - \mathbf{y}\bigr)
$$

### 2.3 경사를 0으로 두기

$$
\boxed{
  \mathbf{X}^\top\mathbf{X}\,\boldsymbol{\theta}^*
  = \mathbf{X}^\top\mathbf{y}
}
\qquad \text{(정규 방정식)}
$$

$\mathbf{X}^\top\mathbf{X}$이 가역이면(즉 $\mathbf{X}$가 완전 열계수를 가지면)
다음이 성립한다.

$$
\boldsymbol{\theta}^*
= (\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top\mathbf{y}
$$

### 2.4 헤세 행렬과 볼록성

헤세 행렬은 다음과 같다.

$$
\mathbf{H}
= \frac{2}{n}\,\mathbf{X}^\top\mathbf{X}
$$

모든 $\mathbf{v}$에 대해
$\mathbf{v}^\top\mathbf{X}^\top\mathbf{X}\mathbf{v}
= \|\mathbf{X}\mathbf{v}\|^2 \geq 0$이므로 헤세 행렬은 양의 준정부호이고 손실은
**볼록**하다. $\mathbf{X}$가 완전 열계수를 가지면($n \geq p + 1$이고 완전한
다중공선성이 없으면) 엄격한 볼록성이 성립하여 유일한 전역 최솟값이 보장된다.

---

## 3. 왜 "정규(normal)" 방정식인가?

잔차 벡터 $\mathbf{r} = \mathbf{y} - \mathbf{X}\boldsymbol{\theta}^*$은
$\mathbf{X}$의 열공간에 **직교**(normal)한다.

$$
\mathbf{X}^\top\mathbf{r}
= \mathbf{X}^\top(\mathbf{y} - \mathbf{X}\boldsymbol{\theta}^*)
= \mathbf{X}^\top\mathbf{y} - \mathbf{X}^\top\mathbf{X}\boldsymbol{\theta}^*
= \mathbf{0}
$$

이 직교 조건에서 방정식의 이름이 유래했다.

---

## 4. 기하학적 해석

### 4.1 열공간으로의 사영

$\mathbf{X}$의 **열공간**은 가능한 모든 예측의 집합이다.

$$
\mathrm{Col}(\mathbf{X})
= \{\mathbf{X}\boldsymbol{\theta} : \boldsymbol{\theta} \in \mathbb{R}^{p+1}\}
$$

선형 회귀는 $\mathrm{Col}(\mathbf{X})$ 안에서 $\mathbf{y}$에 **가장 가까운** 점,
즉 직교 사영을 찾는다.

$$
\hat{\mathbf{y}}
= \mathrm{proj}_{\mathrm{Col}(\mathbf{X})}(\mathbf{y})
$$

### 4.2 사영 행렬 (햇 행렬)

**사영 행렬**은 다음과 같다.

$$
\mathbf{P}
= \mathbf{X}(\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top,
\qquad
\hat{\mathbf{y}} = \mathbf{P}\mathbf{y}
$$

| 성질 | 공식 | 해석 |
|----------|---------|----------------|
| 멱등성 | $\mathbf{P}^2 = \mathbf{P}$ | 두 번 사영해도 결과가 같다 |
| 대칭성 | $\mathbf{P}^\top = \mathbf{P}$ | 자기수반 작용소 |
| 계수 | $\mathrm{rank}(\mathbf{P}) = p + 1$ | 열공간의 차원 |
| 고윳값 | 0 또는 1뿐 | 순수한 사영 |
| 대각합 | $\mathrm{tr}(\mathbf{P}) = p + 1$ | 계수와 같다 |

### 4.3 피타고라스 분해

목표 벡터는 서로 직교하는 성분으로 분해된다.

$$
\mathbf{y} = \underbrace{\hat{\mathbf{y}}}_{\in\,\mathrm{Col}(\mathbf{X})}

           + \underbrace{\mathbf{r}}_{\perp\,\mathrm{Col}(\mathbf{X})},
\qquad
\|\mathbf{y}\|^2 = \|\hat{\mathbf{y}}\|^2 + \|\mathbf{r}\|^2
$$

이것이 분산분석(ANOVA) 분해의 기하학적 바탕이다.

$$
\underbrace{\|\mathbf{y} - \bar{y}\mathbf{1}\|^2}_{\text{SS}_{\text{tot}}}
= \underbrace{\|\hat{\mathbf{y}} - \bar{y}\mathbf{1}\|^2}_{\text{SS}_{\text{reg}}}

+ \underbrace{\|\mathbf{r}\|^2}_{\text{SS}_{\text{res}}}
$$

따라서 $R^2 = \text{SS}_{\text{reg}} / \text{SS}_{\text{tot}}$은 설명된 분산의
비율을 잰다.

### 4.4 기하학적 $R^2$

$$
R^2 = \cos^2\!\theta
$$

여기서 $\theta$는 중심화된 목표 $(\mathbf{y} - \bar{y}\mathbf{1})$과 그 사영
$(\hat{\mathbf{y}} - \bar{y}\mathbf{1})$ 사이의 각이다.

### 4.5 지렛값

$h_{ii}$로 표기하는 $\mathbf{P}$의 대각 성분을 **지렛값**(leverage)이라 한다.

$$
h_{ii} = \mathbf{x}_i^\top(\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{x}_i
$$

지렛값이 큰 점은 특징 값이 유별나며 적합에 불균형하게 큰 영향을 준다. 주요 성질은
다음과 같다.

- $\sum_i h_{ii} = p + 1$ ($\mathbf{P}$의 대각합).
- $1/n \leq h_{ii} \leq 1$.
- 표시할 점을 고르는 흔한 경험칙: $h_{ii} > 2(p+1)/n$.

---

## 5. NumPy 구현

### 5.1 설계 행렬 만들기

```python
import numpy as np

def make_design_matrix(x: np.ndarray) -> np.ndarray:
    """앞에 1로 채운 열을 붙인다: X = [1 | x].

    매개변수
    --------
    x : 모양이 (n, p) 또는 (n,)인 ndarray

    반환값
    ------
    X : 모양이 (n, p+1)인 ndarray
    """
    x = np.atleast_2d(x) if x.ndim == 1 else x
    return np.hstack([np.ones((x.shape[0], 1)), x])
```

### 5.2 정규 방정식 해법

```python
def fit_normal_equation(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """θ* = (X^T X)^{-1} X^T y를 푼다.

    매개변수
    --------
    X : 설계 행렬 (n, p+1)
    y : 목표 벡터 (n,)

    반환값
    ------
    theta : 매개변수 벡터 (p+1,)
    """
    return np.linalg.solve(X.T @ X, X.T @ y)
```

!!! tip "`solve` 대 `inv`"
    `np.linalg.inv(A) @ b`보다 `np.linalg.solve(A, b)`가 낫다. 역행렬을 명시적으로
    만들지 않기 때문에 수치적으로 더 안정적이고 더 빠르다.

### 5.3 예측과 평가

```python
def predict(X: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """예측값 ŷ = X θ를 반환한다."""
    return X @ theta

def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """회귀 지표를 계산한다."""
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
```

### 5.4 전 과정 예제

```python
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

x, y = make_regression(n_samples=200, n_features=3, noise=10, random_state=42)
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

X_train = make_design_matrix(x_train)
X_test = make_design_matrix(x_test)

theta = fit_normal_equation(X_train, y_train)
y_pred = predict(X_test, theta)
print(evaluate(y_test, y_pred))
```

**출력:**

```
{'mse': 132.62372739076085, 'rmse': 11.516237553591921, 'mae': 9.24283824392548, 'r2': 0.9850367614656054}
```

---

## 6. PyTorch 구현

### 6.1 직접 푸는 방법

```python
import torch

def normal_equations(X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """촐레스키 / LU를 통해 θ = (X^T X)^{-1} X^T y를 푼다.

    인자:
        X: 설계 행렬 (n, p+1) — 편향 열을 포함할 것.
        y: 목표 벡터 (n,) 또는 (n, 1).

    반환값:
        모양이 (p+1,) 또는 (p+1, 1)인 매개변수 벡터 θ.
    """
    if y.dim() == 1:
        y = y.reshape(-1, 1)
    return torch.linalg.solve(X.T @ X, X.T @ y)
```

### 6.2 수치적으로 안정한 대안들

```python
def normal_equations_qr(X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """QR 분해: X = QR → R θ = Q^T y."""
    if y.dim() == 1:
        y = y.reshape(-1, 1)
    Q, R = torch.linalg.qr(X)
    return torch.linalg.solve_triangular(R, Q.T @ y, upper=True)

def normal_equations_svd(X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """SVD 기반 해법 (가장 견고하며 계수 부족도 처리한다)."""
    if y.dim() == 1:
        y = y.reshape(-1, 1)
    return torch.linalg.lstsq(X, y).solution
```

### 6.3 사영 행렬 분석

```python
def compute_projection_matrix(X: torch.Tensor) -> torch.Tensor:
    """햇 행렬 P = X (X^T X)^{-1} X^T를 계산한다."""
    return X @ torch.linalg.inv(X.T @ X) @ X.T

def verify_projection_properties(X: torch.Tensor):
    """멱등성, 대칭성, 고윳값, 대각합을 확인한다."""
    P = compute_projection_matrix(X)
    n, d = X.shape

    is_symmetric = torch.allclose(P, P.T, atol=1e-6)
    is_idempotent = torch.allclose(P @ P, P, atol=1e-6)
    trace = torch.trace(P).item()
    eigenvalues = torch.linalg.eigvalsh(P)

    print(f"Symmetric:  {is_symmetric}")
    print(f"Idempotent: {is_idempotent}")
    print(f"Trace:      {trace:.2f} (expected {d})")
    print(f"Eigenvalues: {sorted(eigenvalues.tolist(), reverse=True)}")
```

### 6.4 ANOVA 분해

```python
def anova_decomposition(X: torch.Tensor, y: torch.Tensor):
    """SST = SSR + SSE를 확인한다."""
    theta = torch.linalg.lstsq(X, y).solution
    y_hat = X @ theta
    y_mean = y.mean()

    SST = torch.sum((y - y_mean) ** 2)
    SSR = torch.sum((y_hat - y_mean) ** 2)
    SSE = torch.sum((y - y_hat) ** 2)
    R2 = (SSR / SST).item()

    print(f"SST = {SST.item():.4f}")
    print(f"SSR = {SSR.item():.4f},  SSE = {SSE.item():.4f}")
    print(f"SSR + SSE = {(SSR + SSE).item():.4f}")
    print(f"R² = {R2:.4f}")
```

---

## 7. 특수한 경우의 처리

### 7.1 다중공선성에 가까울 때

$\mathbf{X}^\top\mathbf{X}$이 거의 특이할 때는 작은 정칙화 항을 더한다
(이는 $\lambda$가 아주 작은 릿지 회귀이다).

$$
\boldsymbol{\theta}^*
= (\mathbf{X}^\top\mathbf{X} + \alpha\mathbf{I})^{-1}
  \mathbf{X}^\top\mathbf{y}
$$

```python
def fit_regularised(
    X: torch.Tensor, y: torch.Tensor, alpha: float = 1e-6
) -> torch.Tensor:
    """안정성을 위해 티호노프 정칙화를 넣은 정규 방정식."""
    if y.dim() == 1:
        y = y.reshape(-1, 1)
    d = X.shape[1]
    return torch.linalg.solve(X.T @ X + alpha * torch.eye(d), X.T @ y)
```

### 7.2 부족결정 계 (n < p)

관측보다 매개변수가 많으면 해가 무한히 많다. **최소 노름** 해는 다음과 같다.

$$
\boldsymbol{\theta}^*
= \mathbf{X}^\top(\mathbf{X}\mathbf{X}^\top)^{-1}\mathbf{y}
$$

이는 $\|\boldsymbol{\theta}\|$이 가장 작은 해를 준다.

---

## 8. 복잡도 비교

| 방법 | 시간 | 공간 | 비고 |
|--------|------|-------|-------|
| 정규 방정식 | $O(np^2 + p^3)$ | $O(p^2)$ | $p < 10{,}000$일 때 가장 좋다 |
| QR 분해 | $O(np^2)$ | $O(np)$ | 역행렬을 직접 구하는 것보다 안정적 |
| SVD | $O(np^2)$ | $O(np)$ | 가장 견고하며 계수 부족도 처리 |
| 경사 하강법 | $O(knp)$ | $O(p)$ | $p > 10{,}000$이거나 $n$이 아주 클 때 가장 좋다 |

**경험칙:** $p < 10{,}000$이면 닫힌 형태의 해를 쓰고, 특징 공간이 더 크거나
미니배치 학습이 필요하면 경사 하강법으로 바꾼다.

---

## 9. `nn.Linear`와 대조하여 확인하기

```python
import torch.nn as nn

def verify_against_nn_linear():
    """정규 방정식이 완전히 수렴한 nn.Linear와 일치하는지 확인한다."""
    torch.manual_seed(42)
    n, p = 1000, 5
    X = torch.randn(n, p)
    true_w = torch.tensor([2.0, -1.5, 0.5, 1.0, -0.8])
    y = X @ true_w + 0.5 + 0.1 * torch.randn(n)

    # 정규 방정식
    ones = torch.ones(n, 1)
    X_aug = torch.cat([ones, X], dim=1)
    theta_ne = normal_equations_qr(X_aug, y.reshape(-1, 1))

    # LBFGS로 학습한 nn.Linear (몇 단계 만에 수렴한다)
    model = nn.Linear(p, 1)
    criterion = nn.MSELoss()
    optimizer = torch.optim.LBFGS(
        model.parameters(), line_search_fn="strong_wolfe"
    )

    y_col = y.reshape(-1, 1)
    for _ in range(10):
        def closure():
            optimizer.zero_grad()
            loss = criterion(model(X), y_col)
            loss.backward()
            return loss
        optimizer.step(closure)

    print(f"Normal eq bias:  {theta_ne[0].item():.6f}")
    print(f"nn.Linear bias:  {model.bias.item():.6f}")
    print(f"Close: {torch.allclose(theta_ne[1:].squeeze(), model.weight.squeeze(), atol=1e-4)}")
```

---

## 연습문제

**연습문제 1.**
MSE의 경사를 0으로 두어 정규 방정식 $\mathbf{X}^\top\mathbf{X}\mathbf{w} = \mathbf{X}^\top\mathbf{y}$을 유도하라.

??? success "연습문제 1 풀이"
    $\nabla_\mathbf{w} L = \frac{1}{N}\mathbf{X}^\top(\mathbf{Xw}-\mathbf{y}) = 0$에서 $\mathbf{X}^\top\mathbf{Xw} = \mathbf{X}^\top\mathbf{y}$이 나온다. $\mathbf{X}^\top\mathbf{X}$이 가역이면 $\hat{\mathbf{w}} = (\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top\mathbf{y}$이다.

---

**연습문제 2.**
닫힌 형태의 해가 $\mathbf{y}$를 $\mathbf{X}$의 열공간 위로 직교 사영한 것임을 보여라.

??? success "연습문제 2 풀이"
    예측값은 $\hat{\mathbf{y}} = \mathbf{X}\hat{\mathbf{w}} = \mathbf{X}(\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top\mathbf{y} = \mathbf{P}\mathbf{y}$이며, 여기서 $\mathbf{P} = \mathbf{X}(\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top$은 $\text{col}(\mathbf{X})$ 위로의 사영 행렬이다. 잔차 $\mathbf{y} - \hat{\mathbf{y}} = (\mathbf{I}-\mathbf{P})\mathbf{y}$은 $\text{col}(\mathbf{X})$에 직교한다. $\square$

---

**연습문제 3.**
$\mathbf{X}^\top\mathbf{X}$의 역행렬을 직접 구하는 것보다 유사역행렬 $\mathbf{X}^+$을 쓰는 편이 수치적으로 더 안정적인 이유를 설명하라.

??? success "연습문제 3 풀이"
    $(\mathbf{X}^\top\mathbf{X})^{-1}$을 계산하면 조건수가 제곱된다. $\kappa(\mathbf{X}^\top\mathbf{X}) = \kappa(\mathbf{X})^2$이기 때문이다. SVD 기반 유사역행렬 $\mathbf{X}^+ = \mathbf{V}\Sigma^+\mathbf{U}^\top$은 $\mathbf{X}$의 특잇값을 직접 다루므로 제곱을 피하고, 조건이 나쁜 문제에서도 수치적 안정성을 제공한다.

---

**연습문제 4.**
`torch.linalg.solve`와 유사역행렬 두 가지로 닫힌 형태의 해를 구현하라. 조건이 나쁜 문제에서 수치적 정확도를 비교하라.

??? success "연습문제 4 풀이"
    ```python
    import torch
    X = torch.randn(100, 10)
    y = torch.randn(100)
    # 방법 1: 정규 방정식 풀기
    w1 = torch.linalg.solve(X.T @ X, X.T @ y)
    # 방법 2: 유사역행렬
    w2 = torch.linalg.lstsq(X, y).solution
    print(f"Difference: {(w1 - w2).norm():.2e}")
    ```

## 정리하며

| 개념 | 핵심 공식 |
|---------|-------------|
| 정규 방정식 | $\mathbf{X}^\top\mathbf{X}\boldsymbol{\theta}^* = \mathbf{X}^\top\mathbf{y}$ |
| 닫힌 형태의 해 | $\boldsymbol{\theta}^* = (\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top\mathbf{y}$ |
| 사영 행렬 | $\mathbf{P} = \mathbf{X}(\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top$ |
| 직교성 | $\mathbf{X}^\top(\mathbf{y} - \hat{\mathbf{y}}) = \mathbf{0}$ |
| ANOVA | $\text{SS}_{\text{tot}} = \text{SS}_{\text{reg}} + \text{SS}_{\text{res}}$ |
| $R^2$ (기하학적) | 중심화된 $\mathbf{y}$와 $\hat{\mathbf{y}}$ 사이의 $\cos^2\theta$ |
| 권장 해법 | `torch.linalg.lstsq()` / `np.linalg.solve()` |

---

**참고 문헌**

1. Strang, G. (2019). *Linear Algebra and Learning from Data*, Ch. I.4.
2. Golub, G. H. & Van Loan, C. F. (2013). *Matrix Computations*.
3. Petersen, K. B. & Pedersen, M. S. *The Matrix Cookbook*, §§2–5.
4. Lay, D. C. (2016). *Linear Algebra and Its Applications*, Ch. 6.
