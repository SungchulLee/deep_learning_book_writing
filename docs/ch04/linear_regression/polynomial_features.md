# 다항 특징

선형 회귀는 **매개변수**에 대해 선형일 뿐 특징에 대해서까지 선형일 필요는 없다.
원래 입력을 다항식(또는 다른 비선형) 변환으로 바꾸어 주면, 보통최소제곱의 최적화
장치를 그대로 쓰면서도 곡선 관계를 적합시킬 수 있다. 이 페이지는 그 이론을 전개하고,
편향–분산 절충과 연결하며, 교차 검증으로 모델을 고르는 법을 보여준다.

---

## 1. 특징 사상이라는 착상

### 1.1 선형 기저의 한계

원 특징 $x \in \mathbb{R}$을 쓰면 모델 $\hat{y} = w_1 x + b$는 직선밖에 표현하지
못한다. 참된 관계가 비선형이라면(예컨대 이차식이라면) 최선의 선형 적합도 체계적으로
과소적합한다.

### 1.2 다항식 확장

차수 $d$의 **특징 사상** $\phi : \mathbb{R} \to \mathbb{R}^{d+1}$을 정의하자.

$$
\phi(x) = [1,\; x,\; x^2,\; \ldots,\; x^d]^\top
$$

모델은 다음이 된다.

$$
\hat{y}
= \boldsymbol{\theta}^\top \phi(x)
= \theta_0 + \theta_1 x + \theta_2 x^2 + \cdots + \theta_d x^d
$$

이는 여전히 **$\boldsymbol{\theta}$에 대해 선형**이므로 정규 방정식과 경사 하강법을
고칠 것 없이 그대로 쓸 수 있다. 바뀌는 것은 설계 행렬뿐이다.

### 1.3 다변량으로의 확장

입력 특징이 $p$개이고 차수가 $d$일 때, 다항 특징 사상은 총차수가 $d$ 이하인 모든
단항식을 포함한다.

$$
\phi(\mathbf{x}) = \{x_1^{a_1} x_2^{a_2} \cdots x_p^{a_p}
                    : a_1 + a_2 + \cdots + a_p \leq d\}
$$

확장 후 특징의 개수는 다음과 같다.

$$
\binom{p + d}{d} = \frac{(p + d)!}{p!\, d!}
$$

| 원 특징 수 $p$ | 차수 $d$ | 확장된 특징 수 |
|:---------------------:|:----------:|:-----------------:|
| 1 | 2 | 3 |
| 1 | 5 | 6 |
| 2 | 2 | 6 |
| 2 | 3 | 10 |
| 5 | 3 | 56 |
| 10 | 3 | 286 |

!!! warning "차원의 저주"
    다항 특징의 개수는 조합적으로 늘어난다. $p$가 적당하고 $d$가 크면 확장된 특징
    공간이 매우 커져 계산량도, 과적합 위험도 함께 커진다.

---

## 2. NumPy 구현

### 2.1 직접 만들기 (단변량)

```python
import numpy as np

def polynomial_features_1d(x: np.ndarray, degree: int) -> np.ndarray:
    """특징 하나에 대한 다항 설계 행렬을 만든다.

    매개변수
    --------
    x : 모양이 (n,)인 1차원 배열
    degree : 다항식 차수 d

    반환값
    ------
    X : 모양이 (n, d+1)인 ndarray — 열은 [1, x, x², …, x^d]
    """
    return np.column_stack([x ** k for k in range(degree + 1)])
```

### 2.2 scikit-learn 사용하기

```python
from sklearn.preprocessing import PolynomialFeatures

# degree=3, include_bias=True는 상수 열을 추가한다
poly = PolynomialFeatures(degree=3, include_bias=True)
X_poly = poly.fit_transform(X_raw)  # (n, C(p+3, 3))

print(poly.get_feature_names_out())  # ['1', 'x0', 'x1', 'x0^2', ...]
```

### 2.3 전 과정 적합

```python
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pipe = Pipeline([
    ("poly", PolynomialFeatures(degree=3, include_bias=False)),
    ("scaler", StandardScaler()),
    ("lr", LinearRegression()),
])
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
```

!!! tip "확장 후에 스케일링하기"
    표준화는 언제나 다항식 확장 **뒤에** 하라. 고차 항($x^5$, $x^6$, …)은 자릿수가
    크게 벌어질 수 있어 정규 방정식과 경사 하강법을 모두 불안정하게 만든다.

---

## 3. PyTorch 구현

```python
import torch
import torch.nn as nn

def polynomial_features_torch(
    x: torch.Tensor, degree: int
) -> torch.Tensor:
    """단변량 다항 특징: [x, x², …, x^d].

    인자:
        x: 모양 (n, 1)
        degree: 다항식 차수

    반환값:
        모양 (n, degree) — 상수 열은 없다 (nn.Linear가 편향을 더한다).
    """
    return torch.cat([x ** k for k in range(1, degree + 1)], dim=1)

# 예: 4차 다항식 적합하기
torch.manual_seed(42)
n = 100
x = torch.linspace(-3, 3, n).unsqueeze(1)
y_true = 0.5 * x ** 3 - 2 * x ** 2 + x + 1
y = y_true + 2.0 * torch.randn(n, 1)

degree = 4
X_poly = polynomial_features_torch(x, degree)  # (100, 4)

model = nn.Linear(degree, 1)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

for epoch in range(1000):
    loss = criterion(model(X_poly), y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# 학습된 계수 살펴보기
# model.bias ≈ θ₀,  model.weight ≈ [θ₁, θ₂, θ₃, θ₄]
print(f"Bias:    {model.bias.item():.3f}")
print(f"Weights: {model.weight.detach().squeeze().tolist()}")
```

**출력:**

```
Bias:    1.144
Weights: [0.9922938346862793, -2.0556046962738037, 0.49783164262771606, 0.008654450997710228]
```

---

## 4. 편향–분산 절충

### 4.1 개념적 틀

| 차수 | 모델 복잡도 | 편향 | 분산 | 위험 |
|:------:|:----------------:|:----:|:--------:|:----:|
| 1 | 낮음 | 큼 (과소적합) | 작음 | 큼 |
| 3 | 보통 | 작음 | 보통 | **작음** |
| 10 | 높음 | 아주 작음 | 큼 (과적합) | 큼 |

- **편향**은 모델이 지나치게 단순해서 생기는 체계적 오차를 잰다.
- **분산**은 특정 학습 집합에 대한 민감도를 잰다.
- **총 오차** $\approx \text{편향}^2 + \text{분산} + \text{줄일 수 없는 잡음}$.

### 4.2 수학적 진술

제곱 오차 손실과 데이터셋 $\mathcal{D}$로 학습한 모델 $\hat{f}$에 대해 다음이
성립한다.

$$
E_{\mathcal{D}}\!\bigl[(y - \hat{f}(x))^2\bigr]
= \underbrace{\bigl(f(x) - E[\hat{f}(x)]\bigr)^2}_{\text{편향}^2}

  + \underbrace{E\!\bigl[(\hat{f}(x) - E[\hat{f}(x)])^2\bigr]}_{\text{분산}}
  + \sigma^2
$$

다항식의 차수를 높이면 편향은 줄지만 분산은 커진다.

### 4.3 절충을 눈으로 보기

```python
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

degrees = range(1, 15)
train_errors, val_errors = [], []

for d in degrees:
    pipe = Pipeline([
        ("poly", PolynomialFeatures(degree=d, include_bias=False)),
        ("lr", LinearRegression()),
    ])
    # 음의 MSE (sklearn의 관례)
    cv_scores = cross_val_score(
        pipe, X_train, y_train, cv=5, scoring="neg_mean_squared_error"
    )
    pipe.fit(X_train, y_train)
    train_mse = np.mean((y_train - pipe.predict(X_train)) ** 2)

    train_errors.append(train_mse)
    val_errors.append(-cv_scores.mean())

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(list(degrees), train_errors, "o-", label="Train MSE")
ax.plot(list(degrees), val_errors, "s-", label="CV MSE")
ax.set_xlabel("Polynomial Degree")
ax.set_ylabel("MSE")
ax.set_title("Bias–Variance Trade-Off")
ax.legend()
ax.grid(True, alpha=0.3)
```

---

## 5. 차수 선택을 위한 교차 검증

### 5.1 k-겹 교차 검증

최적의 차수는 학습 오차가 아니라 **교차 검증** 오차를 최소화한다. $k$-겹 교차
검증은 데이터를 $k$개의 겹으로 나누어 $k-1$개로 학습하고 남겨 둔 겹에서 평가하며,
이를 $k$번 반복한다.

```python
from sklearn.model_selection import cross_val_score

def select_degree(X, y, max_degree=10, cv=5):
    """교차 검증으로 최적의 다항식 차수를 고른다."""
    best_degree, best_score = 1, -np.inf

    for d in range(1, max_degree + 1):
        pipe = Pipeline([
            ("poly", PolynomialFeatures(degree=d, include_bias=False)),
            ("scaler", StandardScaler()),
            ("lr", LinearRegression()),
        ])
        scores = cross_val_score(
            pipe, X, y, cv=cv, scoring="neg_mean_squared_error"
        )
        mean_score = scores.mean()
        if mean_score > best_score:
            best_score, best_degree = mean_score, d

    print(f"Best degree: {best_degree}  (CV MSE: {-best_score:.4f})")
    return best_degree
```

### 5.2 정보 기준

교차 검증을 명시적으로 하지 않고 모델을 고를 때는 정보 기준이 모델 복잡도에 벌점을
준다.

| 기준 | 공식 | 비고 |
|-----------|---------|-------|
| AIC | $n\ln(\text{MSE}) + 2k$ | 하나 빼기 교차 검증과 점근적으로 동등 |
| BIC | $n\ln(\text{MSE}) + k\ln(n)$ | 벌점이 더 강해 단순한 모델을 선호 |

여기서 $k$는 매개변수의 개수이다(단변량이면 차수 $+$ 1).

---

## 6. 정칙화와의 관계

고차 다항식이 과적합하는 것은 데이터에 비해 자유 매개변수가 너무 많기 때문이다.
해결책은 두 가지이다.

1. **차수를 제한한다** (교차 검증을 통한 모델 선택 — 이 페이지의 내용).
2. **큰 계수에 벌점을 준다** (정칙화 — [릿지](ridge_regression.md)와
   [라쏘](lasso_regression.md) 참고).

실무에서는 적당한 차수의 다항 특징에 릿지나 라쏘 정칙화를 결합하는 편이, 정칙화 없이
아주 높은 차수를 쓰는 것보다 견고하다.

```python
from sklearn.linear_model import Ridge

pipe_regularised = Pipeline([
    ("poly", PolynomialFeatures(degree=8, include_bias=False)),
    ("scaler", StandardScaler()),
    ("ridge", Ridge(alpha=1.0)),
])
pipe_regularised.fit(X_train, y_train)
```

---

## 7. 다항식을 넘어서

다항 특징은 기저 확장의 한 가지 선택일 뿐이다. 흔히 쓰이는 다른 선택은 다음과 같다.

| 기저 | 공식 | 쓰임새 |
|-------|---------|----------|
| 다항식 | $x, x^2, \ldots, x^d$ | 매끄러운 전역 추세 |
| 방사 기저 함수 | $\exp(-\gamma\|x - c_k\|^2)$ | 국소 패턴, 커널 방법 |
| 푸리에 | $\sin(k\omega x), \cos(k\omega x)$ | 주기적 데이터 |
| 스플라인 | 매듭으로 이어 붙인 조각별 다항식 | 유연한 국소 적합 |
| 교호작용 항 | $x_i x_j$ | 특징들의 교차 효과 |

이들 모두 모델을 매개변수에 대해 선형으로 유지하므로 똑같은 OLS / 경사 하강 장치를
쓸 수 있다.

---

## 8. 완전한 예제

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

# --- 합성 데이터: 잡음이 섞인 삼차식 ---
rng = np.random.default_rng(42)
n = 80
x = rng.uniform(-3, 3, n)
y = 0.5 * x ** 3 - x ** 2 + 0.5 * x + 2 + 3 * rng.normal(size=n)

X = x.reshape(-1, 1)

# --- 교차 검증으로 차수 고르기 ---
degrees = range(1, 12)
cv_mses = []
for d in degrees:
    pipe = Pipeline([
        ("poly", PolynomialFeatures(degree=d, include_bias=False)),
        ("scaler", StandardScaler()),
        ("lr", LinearRegression()),
    ])
    scores = cross_val_score(pipe, X, y, cv=5, scoring="neg_mean_squared_error")
    cv_mses.append(-scores.mean())

best_d = degrees[np.argmin(cv_mses)]
print(f"Best degree: {best_d}  (CV MSE: {min(cv_mses):.2f})")

# --- 최종 적합 ---
final_pipe = Pipeline([
    ("poly", PolynomialFeatures(degree=best_d, include_bias=False)),
    ("scaler", StandardScaler()),
    ("lr", LinearRegression()),
])
final_pipe.fit(X, y)

# --- 그리기 ---
x_plot = np.linspace(-3.5, 3.5, 200).reshape(-1, 1)
y_plot = final_pipe.predict(x_plot)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].scatter(x, y, alpha=0.6, s=20, label="Data")
axes[0].plot(x_plot, y_plot, "r-", lw=2, label=f"Degree {best_d}")
axes[0].set_xlabel("x")
axes[0].set_ylabel("y")
axes[0].set_title("Polynomial Fit")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(list(degrees), cv_mses, "o-", lw=2)
axes[1].axvline(best_d, color="r", ls="--", label=f"Best d={best_d}")
axes[1].set_xlabel("Polynomial Degree")
axes[1].set_ylabel("5-Fold CV MSE")
axes[1].set_title("Cross-Validation Curve")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
```

**출력:**

```
Best degree: 3  (CV MSE: 7.87)
```

---

## 연습문제

**연습문제 1.**
원 특징이 $p$개일 때 차수 $d$의 다항 특징은 몇 개가 만들어지는가? 공식을 유도하라.

??? success "연습문제 1 풀이"
    $p$개 변수에 대한 차수 $\leq d$인 단항식의 개수는 $\binom{p+d}{d}$이다. $p=3$이고 차수가 2이면 $\binom{5}{2} = 10$개의 특징이다(편향 1개, 일차 3개, 교차항을 포함한 이차 6개). $p=2$이고 차수가 3이면 $\binom{5}{3} = 10$이다.

---

**연습문제 2.**
2차원 데이터셋에 대해 차수 2의 다항 특징을 만들고 선형 모델을 적합시켜라. 그 결과로 나오는 비선형 결정 경계를 시각화하라.

??? success "연습문제 2 풀이"
    ```python
    from sklearn.preprocessing import PolynomialFeatures
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X)  # [1, x1, x2, x1^2, x1*x2, x2^2]
    model = LinearRegression().fit(X_poly, y)
    ```

---

**연습문제 3.**
과적합을 보여라. 데이터 점 20개에 차수 1, 5, 15의 다항식을 적합시키고 학습 오차와 시험 오차를 그려라.

??? success "연습문제 3 풀이"
    차수 1은 과소적합하고(편향이 큼), 차수 15는 과적합하며(분산이 크고 점 사이에서 심하게 진동한다), 차수 5가 대체로 가장 좋은 시험 오차를 준다. 이는 모델 복잡도가 조절하는 편향-분산 절충을 잘 보여준다.

---

**연습문제 4.**
다항 특징에 릿지나 라쏘 정칙화를 결합하는 것이 왜 중요한지 설명하고, 고차 예제에서 보여라.

??? success "연습문제 4 풀이"
    고차 다항식은 매개변수가 많아 과적합하기 쉽다. 정칙화는 계수의 크기를 제약한다. 릿지는 모든 계수를 줄이고(더 매끄러운 곡선), 라쏘는 불필요한 항을 0으로 만든다(단항식들 사이에서의 자동 특징 선택). 정칙화가 없으면 15차 다항식은 심하게 진동하지만, 릿지($\lambda = 0.1$)를 쓰면 적합이 매끄럽고 일반화도 잘 된다.

## 정리하며

| 개념 | 핵심 |
|---------|-----------|
| 특징 사상 | $\phi(x) = [1, x, x^2, \ldots, x^d]^\top$ — $x$에는 비선형, $\boldsymbol{\theta}$에는 선형 |
| 확장된 차원 | 입력 $p$개, 차수 $d$에 대해 $\binom{p+d}{d}$개의 특징 |
| 편향–분산 | 낮은 차수 → 큰 편향; 높은 차수 → 큰 분산 |
| 모델 선택 | $d$를 고르는 데 $k$-겹 교차 검증이나 AIC/BIC 사용 |
| 정칙화 | 적당한 $d$에 릿지/라쏘를 쓰는 편이 높은 $d$만 쓰는 것보다 견고 |
| 스케일링 | 다항식 확장 후에 **반드시** 표준화 |

---

**참고 문헌**

1. Hastie, T., Tibshirani, R. & Friedman, J. (2009). *The Elements of
   Statistical Learning*, §§3.1, 7.10.
2. Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*, §§1.1,
   3.1.
3. Murphy, K. P. (2022). *Probabilistic Machine Learning: An Introduction*,
   Ch. 11.
