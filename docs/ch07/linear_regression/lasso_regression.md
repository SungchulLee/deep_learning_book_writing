# 라쏘 회귀

**라쏘**(Lasso, Least Absolute Shrinkage and Selection Operator)는 릿지의 $\ell_2$
패널티를 $\ell_1$ 패널티로 바꾼다. 사소해 보이는 이 변화가 깊은 결과를 낳는다. 라쏘는
일부 계수를 **정확히 0**으로 만들어 자동으로 피처를 선택한다. 이 페이지는 라쏘의
목적 함수를 유도하고, 왜 희소성이 생기는지 설명하고, 좌표 하강법 알고리즘을 소개하며,
**엘라스틱 넷** 혼합형까지 다룬다.

---

## 1. 라쏘의 목적 함수

### 1.1 정식화

$$
\mathcal{L}_{\text{lasso}}(\boldsymbol{\theta})
= \frac{1}{2n}\|\mathbf{y} - \mathbf{X}\boldsymbol{\theta}\|^2
  + \lambda\|\boldsymbol{\theta}\|_1
= \frac{1}{2n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2
  + \lambda\sum_{j=1}^{p}|\theta_j|
$$

!!! note "배율 관례"
    데이터 오차를 $1/(2n)$으로 평균하는 관례는 scikit-learn의 Lasso와 같다. 합 로스를 쓰는 경우와 같은 해를 비교하려면 $\lambda$의 배율도 조정한다. 평균한다고 최적 정규화 강도가 표본 크기와 무관해지는 것은 아니다.

### 1.2 해석적 해가 없다

릿지와 달리 $\ell_1$ 노름은 0에서 **미분 불가능**하므로 닫힌 형태의 행렬 해가 없다.
라쏘에는 반복 알고리즘이 필요하다.

| 알고리즘 | 설명 |
|-----------|-------------|
| 좌표 하강법 | $\theta_j$를 하나씩 갱신한다 (scikit-learn의 기본값) |
| 근접 경사 하강법 (ISTA/FISTA) | 경사 단계 + 소프트 임곗값 처리 |
| 부분경사 방법 | 매끄럽지 않은 점에서 경사를 부분경사로 대체한다 |
| LARS | 최소각 회귀 — 정규화 경로 전체를 효율적으로 구성한다 |

---

## 2. L1이 희소성을 만드는 이유

### 2.1 기하학적 논증

제약 형태를 생각해 보자.

$$
\min_{\boldsymbol{\theta}}
\|\mathbf{y} - \mathbf{X}\boldsymbol{\theta}\|^2
\quad \text{subject to} \quad
\|\boldsymbol{\theta}\|_1 \leq t
$$

$\ell_1$ 공 $\{|\theta_1| + |\theta_2| \leq t\}$은 2차원에서 **마름모**이다(더 높은
차원에서는 교차다포체이다). 그 꼭짓점들이 좌표축 위에 놓인다. MSE 등고선 타원면은
좌표 하나 이상이 정확히 0인 **꼭짓점**에서 제약과 처음 닿을 가능성이 가장 높다.

반면 (릿지가 쓰는) $\ell_2$ 공은 꼭짓점이 없는 매끄러운 구이다. 접점에서는 대체로
모든 좌표가 0이 아니다.

### 2.2 소프트 임곗값 처리

좌표 하나 $\theta_j$에 대한 라쏘 부분문제의 해는 다음과 같다.

$$
\theta_j^*
= \mathcal{S}_\lambda(z_j)
= \operatorname{sign}(z_j)\,\max(|z_j| - \lambda,\; 0)
$$

여기서는 피처를 중심화하고 $\|x_j\|^2/n=1$로 표준화한 경우를 가정한다. $z_j=x_j^\top r_j/n$은 해당 피처와 부분 잔차의 내적이며, 일반적인 비단위 척도의 공식은 다음 절에서 유도한다. $|z_j|\le\lambda$이면 업데이트 결과가 정확히 0이다.

```python
def soft_threshold(z: float, lam: float) -> float:
    """연성 임계값 연산자 S_λ(z)."""
    if z > lam:
        return z - lam
    elif z < -lam:
        return z + lam
    else:
        return 0.0
```

---

## 3. 좌표 하강법 알고리즘

scikit-learn의 기본 해법이다. 좌표들을 차례로 돌면서 나머지를 고정한 채
$\theta_j$를 하나씩 갱신한다.

### 3.1 유도

좌표 $j$에 대한 부분 잔차는 다음과 같다.

$$
r_j = \mathbf{y} - \mathbf{X}_{-j}\boldsymbol{\theta}_{-j}
$$

여기서 $\mathbf{X}_{-j}$은 $j$번째 열을 뺀 $\mathbf{X}$이다. 한 변수짜리
부분문제는 다음이 된다.

$$
\min_{\theta_j}
\frac{1}{2n}\|r_j - \mathbf{x}_j\theta_j\|^2
+ \lambda|\theta_j|
$$

그 해는 다음과 같다.

$$
\theta_j^*
= \frac{1}{\|\mathbf{x}_j\|^2 / n}\,
  \mathcal{S}_\lambda\!\left(
    \frac{\mathbf{x}_j^\top r_j}{n}
  \right)
$$

피처가 표준화되어 있으면($\|\mathbf{x}_j\|^2 / n = 1$) 이는
$\theta_j^* = \mathcal{S}_\lambda(\mathbf{x}_j^\top r_j / n)$으로 간단해진다.

### 3.2 NumPy 구현

```python
import numpy as np

def lasso_coordinate_descent(
    X: np.ndarray,
    y: np.ndarray,
    lam: float,
    max_iter: int = 1000,
    tol: float = 1e-6,
) -> np.ndarray:
    """좌표 하강법을 통한 라쏘 (피처가 표준화되어 있다고 가정).

    파라메터
    --------
    X : (n, p) 표준화된 설계 행렬 (절편 열 없음)
    y : (n,) 중심화된 목표
    lam : 정칙화 강도
    max_iter : 모든 좌표를 훑는 최대 횟수
    tol : 계수 변화의 최댓값에 대한 수렴 임계값

    반환값
    ------
    theta : (p,) 라쏘 해
    """
    n, p = X.shape
    theta = np.zeros(p)
    x_sq = np.sum(X ** 2, axis=0) / n  # ‖x_j‖²/n을 미리 계산

    for iteration in range(max_iter):
        theta_old = theta.copy()

        for j in range(p):
            # 피처 j를 제외한 부분 잔차
            r_j = y - X @ theta + X[:, j] * theta[j]
            # 상관
            z_j = X[:, j] @ r_j / n
            # 연성 임계값
            theta[j] = soft_threshold(z_j, lam) / x_sq[j]

        # 수렴 확인
        if np.max(np.abs(theta - theta_old)) < tol:
            break

    return theta
```

---

## 4. 근접 경사 하강법 (ISTA)

좌표 하강법의 대안이다. 매끄러운 부분(MSE)에 경사 단계를 밟은 뒤, 매끄럽지 않은
부분($\ell_1$)에 근접 연산자(소프트 임곗값 처리)를 적용한다.

### 4.1 알고리즘

$$
\boldsymbol{\theta}^{(t+1)}
= \mathcal{S}_{\eta\lambda}\!\Bigl(
    \boldsymbol{\theta}^{(t)}
    - \eta\,\nabla_{\boldsymbol{\theta}}
      \frac{1}{2n}\|\mathbf{y} - \mathbf{X}\boldsymbol{\theta}^{(t)}\|^2
  \Bigr)
$$

여기서 $\eta$는 이동 폭이고 $\mathcal{S}$은 원소별로 적용된다.

### 4.2 PyTorch 구현

```python
import torch

def lasso_ista(
    X: torch.Tensor,
    y: torch.Tensor,
    lam: float,
    lr: float = 0.01,
    n_iter: int = 1000,
) -> torch.Tensor:
    """라쏘를 위한 반복적 연성 임계값 알고리즘(ISTA)."""
    n, p = X.shape
    theta = torch.zeros(p)

    for _ in range(n_iter):
        # MSE의 경사 (매끄러운 부분)
        residual = X @ theta - y
        grad = X.T @ residual / n

        # 경사 단계
        z = theta - lr * grad

        # 근위 단계 (연성 임계값 처리)
        theta = torch.sign(z) * torch.clamp(torch.abs(z) - lr * lam, min=0)

    return theta
```

---

## 5. 구현

### 5.0 예제가 함께 쓰는 데이터

아래 구현들은 모두 같은 데이터를 쓴다. 피처 20개 가운데 **5개만 실제로 쓰이는** 문제이므로, 라쏘가 나머지를 0으로 눌러 주기를 기대할 수 있다.

```python
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X, y = make_regression(n_samples=200, n_features=20, n_informative=5,
                       noise=10.0, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0)

# 라소는 피처의 크기에 민감하므로 고르게 맞춘 판도 준비해 둔다
X_train_scaled = StandardScaler().fit_transform(X_train)

print(f"X_train: {X_train.shape}   참으로 쓰인 피처: 5개 / 20개")
```

**출력:**

```
X_train: (140, 20)   참으로 쓰인 피처: 5개 / 20개
```

### 5.1 scikit-learn

```python
from sklearn.linear_model import Lasso, LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# 고정된 alpha
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("lasso", Lasso(alpha=0.1)),
])
pipe.fit(X_train, y_train)
coefs = pipe.named_steps["lasso"].coef_
n_nonzero = np.sum(np.abs(coefs) > 1e-8)
print(f"Non-zero coefficients: {n_nonzero} / {len(coefs)}")

# 교차 검증으로 고른 alpha
pipe_cv = Pipeline([
    ("scaler", StandardScaler()),
    ("lasso", LassoCV(cv=5, random_state=42)),
])
pipe_cv.fit(X_train, y_train)
print(f"Best α: {pipe_cv.named_steps['lasso'].alpha_:.4f}")
```

**출력:**

```
Non-zero coefficients: 18 / 20
Best α: 0.7615
```

### 5.2 PyTorch (L1 패널티를 직접 더하기)

PyTorch에 내장된 `weight_decay`는 $\ell_2$만 구현한다. $\ell_1$을 쓰려면 패널티를
직접 더해야 한다.

```python
import torch
import torch.nn as nn

# 앞에서 만든 자료를 텐서로 바꾼다. 라소는 피처의 크기에 민감하므로
# 고르게 맞춘 판을 쓴다.
X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
p = X_train_t.shape[1]

model = nn.Linear(p, 1)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
lam = 0.01

for epoch in range(100):
    y_pred = model(X_train_t)
    mse_loss = criterion(y_pred, y_train_t)

    # 웨이트에만 L1 벌점 (바이어스에는 주지 않는다)
    l1_penalty = sum(param.abs().sum() for name, param in model.named_parameters()
                     if "weight" in name)

    loss = mse_loss + lam * l1_penalty

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

!!! warning "0에서의 경사"
    $\ell_1$ 노름은 0에서 미분 불가능하며 PyTorch는 `abs()`의 해당 미분값으로 0을 선택한다. 보통의 SGD·Adam에 L1 항을 더했다고 유한 반복에서 정확한 희소성이 보장되지는 않는다. 임곗값 후처리는 원래 최적해를 바꿀 수 있으므로 기준을 보고하고, 목적에 맞는 정확한 희소 업데이트에는 근접 방법이나 좌표 하강법을 검토한다.

---

## 6. 엘라스틱 넷 (L1 + L2)

### 6.1 동기

피처들이 상관되어 있을 때 라쏘에는 두 가지 한계가 있다.

1. 강하게 상관된 피처 가운데 일부만 선택할 수 있어 선택 결과가 표본에 민감할 수 있다.
2. 일반위치 등 해의 유일성에 필요한 조건에서는 활성 계수 수가 설계 행렬의 랭크로 제한된다. 중복 피처와 비유일해가 있는 모든 문제에 단순한 개수 제한을 적용할 수는 없다.

**엘라스틱 넷**은 두 패널티를 결합하여 ($\ell_1$에서 오는) 희소성과 ($\ell_2$에서
오는) 묶음 효과를 함께 얻는다.

$$
\mathcal{L}_{\text{elastic}}(\boldsymbol{\theta})
= \frac{1}{2n}\|\mathbf{y} - \mathbf{X}\boldsymbol{\theta}\|^2
  + \lambda\Bigl[
    \rho\|\boldsymbol{\theta}\|_1
    + \frac{1-\rho}{2}\|\boldsymbol{\theta}\|_2^2
  \Bigr]
$$

여기서 $\rho \in [0, 1]$이 혼합 비율을 조절한다(scikit-learn의 `l1_ratio`).

| $\rho$ | 거동 |
|--------|-----------|
| 0 | 순수한 릿지 |
| 1 | 순수한 라쏘 |
| 0.5 | 동등한 혼합 |

### 6.2 scikit-learn

```python
from sklearn.linear_model import ElasticNet, ElasticNetCV

elastic = ElasticNet(alpha=0.1, l1_ratio=0.5)
elastic.fit(X_train_scaled, y_train)

elastic_cv = ElasticNetCV(
    cv=5,
    l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95],
    random_state=42,
)
elastic_cv.fit(X_train_scaled, y_train)
print(f"Best α: {elastic_cv.alpha_:.4f}")
print(f"Best l1_ratio: {elastic_cv.l1_ratio_:.2f}")
```

**출력:**

```
Best α: 0.1306
Best l1_ratio: 0.95
```

---

## 7. 정규화 방법 비교

| 성질 | OLS | 릿지 | 라쏘 | 엘라스틱 넷 |
|----------|-----|-------|-------|------------|
| 패널티 | 없음 | $\lambda\|\theta\|_2^2$ | $\lambda\|\theta\|_1$ | $\lambda[\rho\|\theta\|_1 + (1{-}\rho)\|\theta\|_2^2/2]$ |
| 닫힌 형태 | ✓ | ✓ | ✗ | ✗ |
| 희소성 | ✗ | ✗ | ✓ | ✓ |
| 상관된 피처 | 불안정 | 잘 다룬다 | 하나만 고른다 | 무리를 함께 남긴다 |
| $n < p$ | 파라미터가 유일하지 않을 수 있음 | 양의 L2로 해 안정화 | 해의 유일성·활성 집합 조건 확인 | L2 성분으로 안정화 가능 |
| 베이즈 사전분포 | 균등 | 가우스 | 라플라스 | 가우스 + 라플라스 |
| scikit-learn | `LinearRegression` | `Ridge` | `Lasso` | `ElasticNet` |

---

## 8. 정규화 경로

**정규화 경로**는 $\lambda$가 큰 값(모두 0)에서 작은 값(OLS)으로 변할 때 계수가
그리는 자취이다.

```python
from sklearn.linear_model import lasso_path
import matplotlib.pyplot as plt

alphas, coefs_path, _ = lasso_path(
    X_train_scaled, y_train, alphas=np.logspace(-3, 1, 100)
)

fig, ax = plt.subplots(figsize=(10, 5))
for j in range(coefs_path.shape[0]):
    ax.plot(alphas, coefs_path[j], label=f"w_{j}")
ax.set_xscale("log")
ax.invert_xaxis()
ax.set_xlabel("α (regularisation strength)")
ax.set_ylabel("Coefficient value")
ax.set_title("Lasso Regularisation Path")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
```

$\lambda$가 줄어들면서 계수들이 하나씩 모델에 "들어온다". 등장 순서는 척도와 상관, 정규화 경로의 영향을 받는 단서이며 참 중요도의 순위나 인과성을 보장하지 않는다.

---

## 9. 실무 지침

| 판단 | 권장 사항 |
|----------|----------------|
| 상관된 피처가 많고 희소성은 필요 없음 | 릿지 |
| 피처 선택이 필요하고 상관은 낮음 | 라쏘 |
| 상관된 피처에서 피처 선택 | 엘라스틱 넷 ($\rho \approx 0.5$–$0.9$) |
| 잘 모르겠음 | ElasticNetCV로 시작해 교차 검증이 $\alpha$와 $\rho$를 고르게 한다 |
| PyTorch + 희소성 | L1 패널티를 직접 더하고 사후에 임계값 처리하거나 근접 방법을 쓴다 |

!!! tip "피처 스케일링"
    릿지와 마찬가지로 라쏘나 엘라스틱 넷을 적용하기 전에 피처의 단위를 확인하고 비교 가능한 패널티가 필요하면 학습 데이터에서 표준화하라.
    $\ell_1$ 패널티는 피처의 크기에 민감하다.

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
제약 영역의 기하를 사용하여, L1 정규화는 희소한 해를 만들지만 L2는 그렇지 않은 이유를 설명하라.

</div>

??? success "연습문제 1 풀이"
    L1 제약 영역 $\|\mathbf{w}\|_1 \leq t$은 축 위에 꼭짓점을 가진 마름모(교차다포체)이다. L2 영역 $\|\mathbf{w}\|_2 \leq t$은 구이다. MSE 등고선은 OLS 해를 중심으로 하는 타원이다.

    제약이 있는 최적점은 타원이 제약 영역과 처음 닿는 곳이다. 마름모의 꼭짓점은 어떤 $w_j = 0$인 축 위에 놓이므로, (길쭉한 타원에서 일어나기 쉬운) 꼭짓점에서의 접촉은 정확한 0을 만든다. 구에는 꼭짓점이 없으므로 접촉은 대체로 모든 $w_j \neq 0$인 점에서 일어난다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff hard" title="어려움"></span>
라쏘를 위한 좌표 하강법을 구현하고 희소한 해가 나오는지 확인하라.

</div>

??? success "연습문제 2 풀이"
    ```python
    def lasso_cd(X, y, lam, max_iter=1000):
        n, d = X.shape
        w = torch.zeros(d)
        for _ in range(max_iter):
            for j in range(d):
                r = y - X @ w + X[:, j] * w[j]
                z = (X[:, j] * r).sum() / n
                w[j] = soft_threshold(z, lam)
        return w

    def soft_threshold(z, lam):
        return torch.sign(z) * max(abs(z) - lam, 0)
    ```

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
소프트 임곗값 연산자를 L1 노름의 근접 연산자로 유도하라.

</div>

??? success "연습문제 3 풀이"
    $\text{prox}_{\lambda\|\cdot\|_1}(v) = \arg\min_w \frac{1}{2}(w-v)^2 + \lambda|w|$

    $v > \lambda$인 경우, 최솟값은 $w = v - \lambda$에서 나온다. $v < -\lambda$인 경우는 $w = v + \lambda$이다. $|v| \leq \lambda$인 경우는 $w = 0$이다. 합치면 $\text{sign}(v)\max(|v|-\lambda, 0)$이다. $\square$

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
피처가 100개이고 생성 규칙에 5개만 사용되는 문제에서 라쏘, 릿지, 엘라스틱 넷을 어떻게 비교할 것인가?

</div>

??? success "연습문제 4 풀이"
    같은 학습·검증·테스트 분할과 전처리를 사용하고 각 방법의 강도를 검증셋으로 고른다. 0인 계수와 작은 비영 계수를 구분하며, 선택한 피처의 정밀도·재현율과 테스트 예측 오차를 함께 보고한다. 라쏘가 정확히 95개를 0으로 만들거나 모든 유용한 피처를 복구한다는 결과는 미리 보장되지 않는다.

## 정리하며

| 개념 | 핵심 결과 |
|---------|------------|
| 라쏘의 목적 함수 | $\frac{1}{2n}\|\mathbf{y}-\mathbf{X}\boldsymbol{\theta}\|^2 + \lambda\|\boldsymbol{\theta}\|_1$ |
| 희소성의 원리 | $\ell_1$ 공의 꼭짓점이 좌표축 위에 있다 |
| 소프트 임곗값 | $\mathcal{S}_\lambda(z) = \mathrm{sign}(z)\max(|z|-\lambda, 0)$ |
| 해법 | 좌표 하강법(scikit-learn) 또는 근접 경사법(ISTA) |
| 엘라스틱 넷 | $\ell_1 + \ell_2$ 혼합형; 상관된 피처를 묶는다 |
| 베이즈적 관점 | 라플라스 사전분포 아래의 MAP |

---

**참고 문헌**

1. Tibshirani, R. (1996). "Regression Shrinkage and Selection via the Lasso."
   *Journal of the Royal Statistical Society B*.
2. Hastie, T., Tibshirani, R. & Friedman, J. (2009). *The Elements of
   Statistical Learning*, §§3.4.2–3.4.3.
3. Zou, H. & Hastie, T. (2005). "Regularization and Variable Selection via the
   Elastic Net." *Journal of the Royal Statistical Society B*.
4. Friedman, J., Hastie, T. & Tibshirani, R. (2010). "Regularization Paths for
   Generalized Linear Models via Coordinate Descent." *Journal of Statistical
   Software*.
