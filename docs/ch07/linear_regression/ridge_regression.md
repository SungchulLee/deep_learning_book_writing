# 릿지 회귀

피처들이 서로 상관되어 있거나 모델이 과적합할 때, 보통최소제곱(OLS)은 분산이 큰
파라메터 추정값을 내놓는다. **릿지 회귀**는 로스에 $\ell_2$ 패널티를 더해 계수를 0
쪽으로 줄이고 해를 안정시킨다. 이 페이지는 해석적 해를 유도하고, 기하학적
해석과 베이즈적 해석을 살펴본 뒤, NumPy·PyTorch·scikit-learn 구현을 보인다.

---

## 1. 릿지의 목적 함수

### 1.1 정식화

릿지 회귀는 패널티가 붙은 오차제곱합을 최소화한다.

$$
\mathcal{L}_{\text{ridge}}(\boldsymbol{\theta})
= \|\mathbf{y} - \mathbf{X}\boldsymbol{\theta}\|^2
  + \lambda\|\boldsymbol{\theta}\|_2^2
= \sum_{i=1}^{n}(y_i - \hat{y}_i)^2
  + \lambda\sum_{j=1}^{p}\theta_j^2
$$

이하 해석식에서는 피처와 목표를 중심화하고 $\boldsymbol\theta$를 웨이트 $p$개만 담은 벡터로 사용한다. 절편은 별도로 복원하며 패널티를 주지 않는다. 모든 파라미터를 포함한 설계 행렬을 쓰면서 절편을 제외하려면 $I$ 대신 바이어스 위치가 0인 패널티 행렬을 사용해야 한다. $\lambda>0$는 정규화 강도다.

!!! note "표기"
    scikit-learn은 $\lambda$ 대신 `alpha`를 쓴다. 경사에 2라는 여분의 인수가
    붙지 않도록 패널티를 $\frac{\lambda}{2}\|\boldsymbol{\theta}\|^2$으로 쓰는
    문헌도 있다.

### 1.2 해석적 해

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

중심화한 설계 행렬의 SVD를 $\mathbf X=\mathbf U\boldsymbol\Sigma\mathbf V^\top$라 하자. 다음 합은 0이 아닌 특이값에 대응하는 성분들에 대해 읽는다. 열 전체 랭크가 아니면 $p$ 대신 랭크 $r$개 성분을 사용하며 OLS 파라미터에는 의사역행렬을 쓴다.

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
데이터에서 분산이 작은 방향(OLS 추정이 가장 불안정한 방향)이 가장 강하게 패널티를 받는다.

### 2.2 제약 최적화의 관점

릿지 회귀는 다음과 동등하다.

$$
\min_{\boldsymbol{\theta}}
\|\mathbf{y} - \mathbf{X}\boldsymbol{\theta}\|^2
\quad \text{subject to} \quad
\|\boldsymbol{\theta}\|_2^2 \leq t
$$

적절한 대응값 $t$에서 제약 문제와 패널티 문제가 같은 해를 갖는다. L2 공의 매끄러운 경계는 L1 공의 모서리처럼 좌표별 0을 선호하지 않는다. 다만 데이터의 대칭이나 상쇄로 개별 계수가 정확히 0이 되는 것은 가능하며, 릿지가 “절대로 0을 만들지 않는다”는 주장은 옳지 않다.

---

## 3. 베이즈적 해석

### 3.1 웨이트에 대한 가우스 사전분포

파라메터에 등방적 가우스 사전분포를 준다.

$$
\boldsymbol{\theta} \sim \mathcal{N}(\mathbf{0},\, \tau^2\mathbf{I})
$$

가우스 우도
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
| 좁은 사전분포 ($\tau^2$이 작음) | 강한 정규화 ($\lambda$가 큼) |
| 넓은 사전분포 ($\tau^2$이 큼) | 약한 정규화 ($\lambda$가 작음) |
| 사전분포 없음 ($\tau^2 \to \infty$) | OLS ($\lambda = 0$) |

---

## 4. 편향과 분산에 미치는 영향

릿지는 **바이어스**를 들여오는 대신(계수를 참값에서 멀어지게 줄인다) **분산**을 낮춘다
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

    파라메터
    --------
    X : 설계 행렬 (n, p) — 바이어스 열 없이
    y : 목표 (n,)
    lam : 정칙화 강도 λ

    반환값
    ------
    theta : (p,) 파라메터 벡터 (절편 없음)
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

### 5.3 예제가 함께 쓰는 데이터

아래 구현들은 모두 같은 데이터를 쓴다. 피처가 $p$개, 표본이 $n$개인 회귀 문제이다.

```python
import torch

torch.manual_seed(0)

n, p = 200, 20                                   # 표본 수, 피처 수
X_train = torch.randn(n, p)
true_theta = torch.randn(p, 1)
y_train = X_train @ true_theta + 0.5 * torch.randn(n, 1)

print(f"X_train: {tuple(X_train.shape)}  y_train: {tuple(y_train.shape)}  p = {p}")
```

**출력:**

```
X_train: (200, 20)  y_train: (200, 1)  p = 20
```

### 5.4 PyTorch (웨이트 감쇠를 쓰는 경사 하강법)

표준 SGD에서 `weight_decay=\gamma`는 경사에 $\gamma\theta$를 더한다. 위의 제곱오차 **합**에 대한 릿지 목적과 평균 MSE 구현을 비교하면 계수와 분모를 맞춰야 한다. 평균 MSE에서는 $\gamma=2\lambda/n$이 위 식에 대응한다. 옵티마이저에 바이어스도 넣으면 그 바이어스에도 감쇠가 적용될 수 있다. AdamW의 분리된 감쇠와는 구분한다.

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

!!! warning "웨이트 감쇠의 관례"
    SGD와 Adam에서 PyTorch의 `weight_decay` 파라메터는 경사에
    $\text{weight\_decay} \times \theta$을 더한다. 이는 로스에
    $\frac{\text{weight\_decay}}{2}\|\theta\|^2$을 더하는 것과 같다.
    릿지의 $\lambda$와의 관계는 로스에 $1/n$ 인수가 들어 있는지, 어떤 관례를
    쓰는지에 따라 달라진다.

### 5.5 scikit-learn

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

**출력:**

```
Best α: 0.1
```

!!! warning "피처의 단위와 패널티를 함께 정하라"
    같은 예측 효과라도 큰 척도의 피처는 작은 계수로 표현되어 패널티가 상대적으로 약할 수 있다. 보통 학습 데이터로 표준화해 비교하지만, 의미 있는 단위를 유지하거나 피처별 사전정보로 가중 패널티를 정하는 선택도 가능하다.

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

**출력:**

```
Best α = 0.3728, CV MSE = 0.2753
```

### 6.2 정규화 경로

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
| 참 모델이 희소함 | 모든 피처를 포함한다 | 모든 피처가 남는다 (희소성 없음) |

릿지는 일반적으로 정확한 희소성을 유도하지 않는다. 피처 선택이 필요하다면
[라쏘 회귀](lasso_regression.md)를 보라.

---

## 8. 다른 방법들과의 관계

| 방법 | 패널티 | 베이즈 사전분포 | 희소성 |
|--------|---------|----------------|----------|
| OLS | 없음 | 비정칙 균등분포 | 없음 |
| 릿지 | $\lambda\|\boldsymbol{\theta}\|_2^2$ | 가우스 | 없음 |
| 라쏘 | $\lambda\|\boldsymbol{\theta}\|_1$ | 라플라스 | 있음 |
| 엘라스틱 넷 | $\lambda[\rho\|\cdot\|_1 + (1-\rho)\|\cdot\|_2^2]$ | 가우스 + 라플라스 | 있음 |

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff hard" title="어려움"></span>
릿지 회귀의 닫힌 형태 해 $\hat{\mathbf{w}} = (\mathbf{X}^\top\mathbf{X} + \lambda\mathbf{I})^{-1}\mathbf{X}^\top\mathbf{y}$을 유도하라.

</div>

??? success "풀이"
    릿지의 목적 함수는 $L = \|\mathbf{Xw}-\mathbf{y}\|^2 + \lambda\|\mathbf{w}\|^2$이다. $\nabla_\mathbf{w} L = 2\mathbf{X}^\top(\mathbf{Xw}-\mathbf{y}) + 2\lambda\mathbf{w} = 0$으로 두면 $(\mathbf{X}^\top\mathbf{X} + \lambda\mathbf{I})\mathbf{w} = \mathbf{X}^\top\mathbf{y}$을 얻는다.

    $\lambda > 0$이므로 $\mathbf{X}^\top\mathbf{X} + \lambda\mathbf{I}$은 언제나 양의 정부호이고(모든 고윳값이 $\geq \lambda > 0$), 따라서 가역성과 해의 유일성이 보장된다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff hard" title="어려움"></span>
릿지 회귀가 가우스 사전분포 $\mathbf{w} \sim \mathcal{N}(0, \frac{\sigma^2}{\lambda}\mathbf{I})$ 아래의 MAP 추정이라는 베이즈적 해석을 가짐을 보여라.

</div>

??? success "풀이"
    MAP: $\hat{\mathbf{w}} = \arg\max_\mathbf{w} p(\mathbf{w}|\mathbf{y}) \propto p(\mathbf{y}|\mathbf{w})p(\mathbf{w})$.

    $\log p(\mathbf{y}|\mathbf{w}) + \log p(\mathbf{w}) = -\frac{1}{2\sigma^2}\|\mathbf{y}-\mathbf{Xw}\|^2 - \frac{\lambda}{2\sigma^2}\|\mathbf{w}\|^2 + \text{const}$.

    이를 최대화하는 것은 $\|\mathbf{y}-\mathbf{Xw}\|^2 + \lambda\|\mathbf{w}\|^2$을 최소화하는 것과 같으며, 이것이 릿지 회귀이다. $\square$

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
피처가 5개인 회귀 문제에 대해 릿지 계수를 $\lambda$의 함수로 그려라(릿지 자취).

</div>

??? success "풀이"
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

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
릿지 강도의 양 끝에서 웨이트와 예측이 어떻게 달라지는가? 절편을 패널티에서 제외한 경우를 구분하라.

</div>

??? success "풀이"
    정규화 강도가 0에 가까우면 최소제곱 해로 접근하며 랭크가 부족하면 최소 노름 해와 연결된다. 강도가 무한대로 가면 패널티 대상 웨이트는 0에 접근한다. 절편을 제외했다면 학습 표본평균 예측이 남고 그 평균의 표본 변동도 남는다. 바이어스가 언제나 “최대”가 된다는 일반 법칙은 없으며 검증 데이터로 오차를 비교해 강도를 선택한다.

## 정리하며

| 개념 | 핵심 결과 |
|---------|------------|
| 목적 함수 | $\|\mathbf{y} - \mathbf{X}\boldsymbol{\theta}\|^2 + \lambda\|\boldsymbol{\theta}\|_2^2$ |
| 해 | $(\mathbf{X}^\top\mathbf{X} + \lambda\mathbf{I})^{-1}\mathbf{X}^\top\mathbf{y}$ |
| 고윳값 축소 | 성분마다 인수 $\sigma_j^2 / (\sigma_j^2 + \lambda)$ |
| 베이즈적 관점 | $\lambda = \sigma^2/\tau^2$인 가우스 사전분포 $\mathcal{N}(0, \tau^2 I)$ 아래의 MAP |
| PyTorch | 옵티마이저의 `weight_decay` 파라메터 |
| scikit-learn | `Ridge(alpha=λ)` 또는 자동 선택을 위한 `RidgeCV` |
| 핵심 성질 | 계수 크기를 억제하되 일반적으로 희소성을 유도하지 않는다 |

---

**참고 문헌**

1. Hastie, T., Tibshirani, R. & Friedman, J. (2009). *The Elements of
   Statistical Learning*, §3.4.1.
2. Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*, §3.1.4.
3. Hoerl, A. E. & Kennard, R. W. (1970). "Ridge Regression: Biased Estimation
   for Nonorthogonal Problems." *Technometrics*.
