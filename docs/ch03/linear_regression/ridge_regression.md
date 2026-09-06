# 릿지 회귀
## 개요

특징들이 서로 상관되어 있거나 모델이 과적합할 때, 보통최소제곱(OLS)은 분산이 큰
매개변수 추정값을 내놓는다. **릿지 회귀**는 손실에 $\ell_2$ 벌점을 더해 계수를 0
쪽으로 줄이고 해를 안정시킨다. 이 페이지는 닫힌 형태의 해를 유도하고, 기하학적
해석과 베이즈적 해석을 살펴본 뒤, NumPy·PyTorch·scikit-learn 구현을 보인다.

---

## 1. 릿지의 목적 함수

### 1.1 정식화

릿지 회귀는 벌점이 붙은 오차제곱합을 최소화한다.

$$
\mathcal{L}_{\text{ridge}}(\boldsymbol{\theta})
= \|\mathbf{y} - \mathbf{X}\boldsymbol{\theta}\|^2

  + \lambda\|\boldsymbol{\theta}\|_2^2
= \sum_{i=1}^{n}(y_i - \hat{y}_i)^2

  + \lambda\sum_{j=1}^{p}\theta_j^2
$$

여기서 $\lambda > 0$은 **정칙화 강도**이다. 관례상 **편향** 항은 대개 벌점에서
제외한다. 줄어드는 것은 가중치뿐이다.

!!! note "표기"
    scikit-learn은 $\lambda$ 대신 `alpha`를 쓴다. 경사에 2라는 여분의 인수가
    붙지 않도록 벌점을 $\frac{\lambda}{2}\|\boldsymbol{\theta}\|^2$으로 쓰는
    문헌도 있다.

### 1.2 닫힌 형태의 해

경사를 0으로 두면

$$
\nabla_{\boldsymbol{\theta}}\mathcal{L}
= -2\mathbf{X}^\top\mathbf{y}

  + 2(\mathbf{X}^\top\mathbf{X} + \lambda\mathbf{I})\boldsymbol{\theta}
= \mathbf{0}
$$

이고, 여기서 다음을 얻는다.

$$
\boxed{
  \boldsymbol{\theta}^*_{\text{ridge}}
  = (\mathbf{X}^\top\mathbf{X} + \lambda\mathbf{I})^{-1}
    \mathbf{X}^\top\mathbf{y}.
}
$$

행렬 $\mathbf{X}^\top\mathbf{X} + \lambda\mathbf{I}$은 $\lambda > 0$이면 언제나
가역이다. $\mathbf{X}$의 특잇값을 $\sigma_j$라 할 때 그 고윳값이
$\sigma_j^2 + \lambda > 0$이기 때문이다. 따라서 릿지 회귀는 다중공선성 아래에서
OLS가 겪는 가역성 문제를 없애 준다.

---

## 2. 기하학적 해석

### 2.1 고윳값 축소

중심화한 설계 행렬의 SVD를
$\mathbf{X} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top$이라 하자. OLS의 예측은
다음과 같다.

$$
\hat{\mathbf{y}}_{\text{OLS}}
= \mathbf{X}\boldsymbol{\theta}^*_{\text{OLS}}
= \sum_{j=1}^{p} \mathbf{u}_j
  \frac{\sigma_j^2}{\sigma_j^2}\,
  \mathbf{u}_j^\top\mathbf{y}
= \sum_{j=1}^{p} \mathbf{u}_j\,\mathbf{u}_j^\top\mathbf{y}
$$

반면 릿지의 예측은 다음과 같다.

$$
\hat{\mathbf{y}}_{\text{ridge}}
= \sum_{j=1}^{p} \mathbf{u}_j\,
  \underbrace{\frac{\sigma_j^2}{\sigma_j^2 + \lambda}}_{\text{축소 인수}}\,
  \mathbf{u}_j^\top\mathbf{y}
$$

각 성분에 1보다 엄격히 작은 인수가 곱해지며, **특잇값이 작을수록 더 많이 줄어든다**.
데이터에서 분산이 작은 방향(OLS 추정이 가장 불안정한 방향)이 가장 강하게 벌점을 받는다.

### 2.2 제약 최적화의 관점

릿지 회귀는 다음과 동등하다.

$$
\min_{\boldsymbol{\theta}}
\|\mathbf{y} - \mathbf{X}\boldsymbol{\theta}\|^2
\quad \text{subject to} \quad
\|\boldsymbol{\theta}\|_2^2 \leq t
$$

여기서 $t$는 $\lambda$에 의존한다. 기하학적으로 이는 MSE 등고선 타원면과 반지름이
$\sqrt{t}$인 $\ell_2$ 공의 교차이다. 제약면이 (구이므로) **매끄럽기** 때문에 해는
경계 위에 놓이되 정확히 0이 되는 일은 결코 없다. 릿지는 계수를 줄일 뿐 희소한 해를
만들지는 않는다.

---

## 3. 베이즈적 해석

### 3.1 가중치에 대한 가우스 사전분포

매개변수에 등방적 가우스 사전분포를 준다.

$$
\boldsymbol{\theta} \sim \mathcal{N}(\mathbf{0},\, \tau^2\mathbf{I})
$$

가우스 가능도
$\mathbf{y} \mid \mathbf{X}, \boldsymbol{\theta}
\sim \mathcal{N}(\mathbf{X}\boldsymbol{\theta},\, \sigma^2\mathbf{I})$과
결합하면 사후분포는 다음과 같다.

$$
p(\boldsymbol{\theta} \mid \mathbf{X}, \mathbf{y})
\propto
\exp\!\Bigl(
  -\frac{1}{2\sigma^2}\|\mathbf{y} - \mathbf{X}\boldsymbol{\theta}\|^2
  -\frac{1}{2\tau^2}\|\boldsymbol{\theta}\|^2
\Bigr)
$$

### 3.2 MAP = 릿지

**최대 사후확률**(MAP) 추정값은 다음과 같다.

$$
\boldsymbol{\theta}_{\text{MAP}}
= \arg\min_{\boldsymbol{\theta}}
\left[
  \|\mathbf{y} - \mathbf{X}\boldsymbol{\theta}\|^2

  + \frac{\sigma^2}{\tau^2}\|\boldsymbol{\theta}\|^2
\right]
$$

이는 정확히 $\lambda = \sigma^2 / \tau^2$인 릿지 회귀이다.

| 베이즈 | 릿지 |
|----------|-------|
| 좁은 사전분포 ($\tau^2$이 작음) | 강한 정칙화 ($\lambda$가 큼) |
| 넓은 사전분포 ($\tau^2$이 큼) | 약한 정칙화 ($\lambda$가 작음) |
| 사전분포 없음 ($\tau^2 \to \infty$) | OLS ($\lambda = 0$) |

---

## 4. 편향과 분산에 미치는 영향

릿지는 **편향**을 들여오는 대신(계수를 참값에서 멀어지게 줄인다) **분산**을 낮춘다
(학습 집합이 달라져도 추정이 더 안정적이다).

$$
\text{MSE}(\hat{\boldsymbol{\theta}}_{\text{ridge}})
= \underbrace{\text{Bias}^2(\lambda)}_{\lambda\text{에 따라}\nearrow}

  + \underbrace{\text{Variance}(\lambda)}_{\lambda\text{에 따라}\searrow}
$$

최적의 $\lambda$는 전체 MSE를 최소화한다. 곧 편향–분산 절충이다.

---

## 5. 구현

### 5.1 NumPy

```python
import numpy as np

def ridge_fit(X: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    """릿지 회귀: θ = (X^T X + λ I)^{-1} X^T y.

    매개변수
    --------
    X : 설계 행렬 (n, p) — 편향 열 없이
    y : 목표 (n,)
    lam : 정칙화 강도 λ

    반환값
    ------
    theta : (p,) 매개변수 벡터 (절편 없음)
    """
    p = X.shape[1]
    return np.linalg.solve(X.T @ X + lam * np.eye(p), X.T @ y)

def ridge_fit_with_intercept(X, y, lam):
    """절편을 따로 다루기 위해 데이터를 중심화한다."""
    X_mean, y_mean = X.mean(axis=0), y.mean()
    X_c = X - X_mean
    y_c = y - y_mean

    w = ridge_fit(X_c, y_c, lam)
    b = y_mean - X_mean @ w
    return w, b
```

### 5.2 PyTorch (닫힌 형태)

```python
import torch

def ridge_closed_form(
    X: torch.Tensor, y: torch.Tensor, lam: float
) -> torch.Tensor:
    """닫힌 형태의 해를 통한 릿지 회귀."""
    if y.dim() == 1:
        y = y.reshape(-1, 1)
    p = X.shape[1]
    A = X.T @ X + lam * torch.eye(p)
    return torch.linalg.solve(A, X.T @ y)
```

### 5.3 PyTorch (가중치 감쇠를 쓰는 경사 하강법)

PyTorch에서 $\ell_2$ 정칙화는 최적화기의 **가중치 감쇠**(weight decay)로 구현된다.
벌점의 경사 $\lambda\boldsymbol{\theta}$이 매개변수 갱신에 자동으로 더해진다.

```python
import torch.nn as nn

model = nn.Linear(p, 1)
criterion = nn.MSELoss()

# weight_decay = λ / n  (PyTorch의 관례)
optimizer = torch.optim.SGD(
    model.parameters(), lr=0.01, weight_decay=1e-2
)

for epoch in range(100):
    loss = criterion(model(X_train), y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

!!! warning "가중치 감쇠의 관례"
    SGD와 Adam에서 PyTorch의 `weight_decay` 매개변수는 경사에
    $\text{weight\_decay} \times \theta$을 더한다. 이는 손실에
    $\frac{\text{weight\_decay}}{2}\|\theta\|^2$을 더하는 것과 같다.
    릿지의 $\lambda$와의 관계는 손실에 $1/n$ 인수가 들어 있는지, 어떤 관례를
    쓰는지에 따라 달라진다.

### 5.4 scikit-learn

```python
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# 고정된 alpha
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("ridge", Ridge(alpha=1.0)),
])
pipe.fit(X_train, y_train)

# 교차 검증으로 alpha 고르기
pipe_cv = Pipeline([
    ("scaler", StandardScaler()),
    ("ridge", RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])),
])
pipe_cv.fit(X_train, y_train)
best_alpha = pipe_cv.named_steps["ridge"].alpha_
print(f"Best α: {best_alpha}")
```

!!! warning "반드시 표준화하라"
    릿지는 계수의 **크기**에 벌점을 준다. 특징들의 규모가 다르면 분산이 큰 특징의
    계수가 불균형하게 많이 줄어든다. 릿지 앞에서는 언제나 `StandardScaler`를 써라.

---

## 6. lambda 고르기

### 6.1 교차 검증

표준적인 방법이다. $\lambda$ 값의 격자에 대해 따로 떼어 둔 데이터에서 평가한다.

```python
import numpy as np
from sklearn.model_selection import cross_val_score

alphas = np.logspace(-3, 3, 50)
cv_scores = []

for a in alphas:
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=a)),
    ])
    scores = cross_val_score(
        pipe, X_train, y_train, cv=5, scoring="neg_mean_squared_error"
    )
    cv_scores.append(-scores.mean())

best_idx = np.argmin(cv_scores)
print(f"Best α = {alphas[best_idx]:.4f}, CV MSE = {cv_scores[best_idx]:.4f}")
```

### 6.2 정칙화 경로

축소 효과를 보기 위해 계수를 $\lambda$의 함수로 그린다.

```python
import matplotlib.pyplot as plt

alphas = np.logspace(-2, 4, 100)
coefs = []

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)

for a in alphas:
    model = Ridge(alpha=a, fit_intercept=True)
    model.fit(X_scaled, y_train)
    coefs.append(model.coef_)

coefs = np.array(coefs)

fig, ax = plt.subplots(figsize=(10, 5))
for j in range(coefs.shape[1]):
    ax.plot(alphas, coefs[:, j], label=f"w_{j}")
ax.set_xscale("log")
ax.set_xlabel("α (regularisation strength)")
ax.set_ylabel("Coefficient value")
ax.set_title("Ridge Regularisation Path")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
```

---

## 7. 릿지 대 OLS: 릿지는 언제 도움이 되는가?

| 상황 | OLS | 릿지 |
|----------|-----|-------|
| $n \gg p$, 상관 낮음 | ✓ 잘 동작한다 | 개선이 미미하다 |
| $n \gg p$, 상관 높음 | 분산이 팽창한다 | ✓ 추정을 안정시킨다 |
| $n \approx p$ | 과적합한다 | ✓ 필수적이다 |
| $n < p$ | $\mathbf{X}^\top\mathbf{X}$이 특이하다 | ✓ 언제나 가역이다 |
| 참 모델이 희소함 | 모든 특징을 포함한다 | 모든 특징이 남는다 (희소성 없음) |

릿지는 계수를 정확히 0으로 만드는 일이 **결코** 없다. 특징 선택이 필요하다면
[라쏘 회귀](lasso_regression.md)를 보라.

---

## 8. 다른 방법들과의 관계

| 방법 | 벌점 | 베이즈 사전분포 | 희소성 |
|--------|---------|----------------|----------|
| OLS | 없음 | 비정칙 균등분포 | 없음 |
| 릿지 | $\lambda\|\boldsymbol{\theta}\|_2^2$ | 가우스 | 없음 |
| 라쏘 | $\lambda\|\boldsymbol{\theta}\|_1$ | 라플라스 | 있음 |
| 엘라스틱 넷 | $\lambda[\rho\|\cdot\|_1 + (1-\rho)\|\cdot\|_2^2]$ | 가우스 + 라플라스 | 있음 |

---

## 요약

| 개념 | 핵심 결과 |
|---------|------------|
| 목적 함수 | $\|\mathbf{y} - \mathbf{X}\boldsymbol{\theta}\|^2 + \lambda\|\boldsymbol{\theta}\|_2^2$ |
| 해 | $(\mathbf{X}^\top\mathbf{X} + \lambda\mathbf{I})^{-1}\mathbf{X}^\top\mathbf{y}$ |
| 고윳값 축소 | 성분마다 인수 $\sigma_j^2 / (\sigma_j^2 + \lambda)$ |
| 베이즈적 관점 | $\lambda = \sigma^2/\tau^2$인 가우스 사전분포 $\mathcal{N}(0, \tau^2 I)$ 아래의 MAP |
| PyTorch | 최적화기의 `weight_decay` 매개변수 |
| scikit-learn | `Ridge(alpha=λ)` 또는 자동 선택을 위한 `RidgeCV` |
| 핵심 성질 | 계수를 줄이지만 결코 0으로 만들지 않는다 |

---

## 참고 문헌

1. Hastie, T., Tibshirani, R. & Friedman, J. (2009). *The Elements of
   Statistical Learning*, §3.4.1.
2. Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*, §3.1.4.
3. Hoerl, A. E. & Kennard, R. W. (1970). "Ridge Regression: Biased Estimation
   for Nonorthogonal Problems." *Technometrics*.

## 연습문제

**연습문제 1.**
릿지 회귀의 닫힌 형태 해 $\hat{\mathbf{w}} = (\mathbf{X}^\top\mathbf{X} + \lambda\mathbf{I})^{-1}\mathbf{X}^\top\mathbf{y}$을 유도하라.

??? success "연습문제 1 풀이"
    릿지의 목적 함수는 $L = \|\mathbf{Xw}-\mathbf{y}\|^2 + \lambda\|\mathbf{w}\|^2$이다. $\nabla_\mathbf{w} L = 2\mathbf{X}^\top(\mathbf{Xw}-\mathbf{y}) + 2\lambda\mathbf{w} = 0$으로 두면 $(\mathbf{X}^\top\mathbf{X} + \lambda\mathbf{I})\mathbf{w} = \mathbf{X}^\top\mathbf{y}$을 얻는다.

    $\lambda > 0$이므로 $\mathbf{X}^\top\mathbf{X} + \lambda\mathbf{I}$은 언제나 양의 정부호이고(모든 고윳값이 $\geq \lambda > 0$), 따라서 가역성과 해의 유일성이 보장된다.

---

**연습문제 2.**
릿지 회귀가 가우스 사전분포 $\mathbf{w} \sim \mathcal{N}(0, \frac{\sigma^2}{\lambda}\mathbf{I})$ 아래의 MAP 추정이라는 베이즈적 해석을 가짐을 보여라.

??? success "연습문제 2 풀이"
    MAP: $\hat{\mathbf{w}} = \arg\max_\mathbf{w} p(\mathbf{w}|\mathbf{y}) \propto p(\mathbf{y}|\mathbf{w})p(\mathbf{w})$.

    $\log p(\mathbf{y}|\mathbf{w}) + \log p(\mathbf{w}) = -\frac{1}{2\sigma^2}\|\mathbf{y}-\mathbf{Xw}\|^2 - \frac{\lambda}{2\sigma^2}\|\mathbf{w}\|^2 + \text{const}$.

    이를 최대화하는 것은 $\|\mathbf{y}-\mathbf{Xw}\|^2 + \lambda\|\mathbf{w}\|^2$을 최소화하는 것과 같으며, 이것이 릿지 회귀이다. $\square$

---

**연습문제 3.**
특징이 5개인 회귀 문제에 대해 릿지 계수를 $\lambda$의 함수로 그려라(릿지 자취).

??? success "연습문제 3 풀이"
    ```python
    import torch, numpy as np, matplotlib.pyplot as plt
    lambdas = np.logspace(-3, 3, 50)
    coefs = []
    for lam in lambdas:
        w = torch.linalg.solve(X.T @ X + lam * torch.eye(5), X.T @ y)
        coefs.append(w.numpy())
    for j in range(5):
        plt.plot(lambdas, [c[j] for c in coefs], label=f'w{j}')
    plt.xscale('log'); plt.xlabel('lambda'); plt.legend()
    ```

---

**연습문제 4.**
$\lambda$가 편향-분산 절충에 미치는 영향을 설명하라. $\lambda \to 0$일 때와 $\lambda \to \infty$일 때 어떤 일이 일어나는가?

??? success "연습문제 4 풀이"
    $\lambda \to 0$이면 릿지는 OLS로 환원되며, 편향은 0이지만 분산이 크다(과적합 위험).

    $\lambda \to \infty$이면 $\hat{\mathbf{w}} \to 0$이 되어 편향이 최대가 되지만(평균을 예측한다) 분산은 0이다(학습 데이터와 무관하게 일정한 예측).

    최적의 $\lambda$는 편향과 분산의 균형을 잡아 시험 오차를 최소화한다. $\lambda$를 고르는 표준적인 방법은 교차 검증이다.
