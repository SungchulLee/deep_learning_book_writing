# 선형 회귀
## 개요

선형 회귀는 입력 특징과 연속적인 목표 변수 사이의 관계를 선형 함수로 모형화하는,
지도 학습의 가장 기본이 되는 알고리즘이다. **닫힌 형태**의 해(정규 방정식)와
**반복적**인 해(경사 하강법)를 모두 갖고 있어, 기계 학습에서 최적화를 이해하는
출발점으로 더없이 알맞다.

이 절에서는 모델 설정, 확률적 해석, 해법, 확장에 이르기까지 이론을 제일원리에서부터
전개하며, 그 과정 내내 NumPy, PyTorch, scikit-learn 구현을 함께 제시한다.

---

## 1. 선형 모델

### 1.1 단변량의 경우

입력 특징이 하나인 $x$에 대해 모델은 다음과 같다.

$$
\hat{y} = wx + b
$$

여기서 $w$는 **가중치**(기울기), $b$는 **편향**(절편)이다.

### 1.2 다변량의 경우

$p$개의 입력 특징 $\mathbf{x} = [x_1, x_2, \ldots, x_p]^\top$에 대해 모델은
다음과 같이 일반화된다.

$$
\hat{y}
= \mathbf{w}^\top \mathbf{x} + b
= \sum_{j=1}^{p} w_j x_j + b
$$

설계 행렬 $\mathbf{X} \in \mathbb{R}^{n \times p}$을 갖는 $n$개의 표본에 대해
행렬 형태로 쓰면 다음과 같다.

$$
\hat{\mathbf{y}} = \mathbf{X}\mathbf{w} + b\mathbf{1}
$$

### 1.3 간결한 표기 (편향 흡수)

특징 벡터에 상수 1을 덧붙이면 편향이 매개변수 벡터 안으로 흡수된다.

$$
\tilde{\mathbf{x}} = [1,\; x_1,\; \ldots,\; x_p]^\top,
\qquad
\boldsymbol{\theta} = [b,\; w_1,\; \ldots,\; w_p]^\top
$$

그러면 $\hat{y} = \boldsymbol{\theta}^\top \tilde{\mathbf{x}}$이다. 데이터셋 전체에
대해서는 설계 행렬이
$\mathbf{X} = [\mathbf{1} \mid \mathbf{X}_{\text{raw}}]
\in \mathbb{R}^{n \times (p+1)}$이 되고 다음이 성립한다.

$$
\hat{\mathbf{y}} = \mathbf{X}\boldsymbol{\theta}
$$

닫힌 형태의 유도에는 이 간결한 표기를 쓰고, PyTorch가 `nn.Linear`로 편향을 자동
처리해 주는 경사 하강법 구현에서는 $(\mathbf{w}, b)$를 분리한 표기로 바꾼다.

---

## 2. 확률적 정식화

### 2.1 생성 과정

각 관측이 다음과 같이 생성된다고 가정한다.

$$
y_i = \mathbf{w}^\top \mathbf{x}_i + b + \epsilon_i,
\qquad
\epsilon_i \sim \mathcal{N}(0, \sigma^2)
$$

여기서 잡음 항 $\epsilon_i$는 서로 독립이며 같은 분포를 따른다.
이는 다음의 조건부 분포를 함의한다.

$$
y_i \mid \mathbf{x}_i \;\sim\; \mathcal{N}\!\bigl(\mathbf{w}^\top \mathbf{x}_i + b,\;\sigma^2\bigr)
$$

### 2.2 MLE에서 MSE로

$n$개 관측에 대한 로그가능도는 다음과 같다.

$$
\ell(\mathbf{w}, b, \sigma^2)
= -\frac{n}{2}\ln(2\pi\sigma^2)

  - \frac{1}{2\sigma^2}\sum_{i=1}^{n}(y_i - \mathbf{w}^\top\mathbf{x}_i - b)^2
$$

$\sigma^2$을 고정했을 때 $(\mathbf{w}, b)$에 대해 $\ell$을 최대화하는 것은
**평균제곱오차**(MSE)를 최소화하는 것과 같다.

$$
\mathcal{L}(\mathbf{w}, b)
= \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2
= \frac{1}{n}\|\mathbf{y} - \mathbf{X}\boldsymbol{\theta}\|^2
$$

!!! tip "핵심 통찰"
    MSE는 임의로 고른 손실이 **아니다**. 가우스 잡음 아래에서 (상수를 제외하면)
    음의 로그가능도 그 자체이다. 덕분에 MSE 추정량은 최대가능도가 갖는 바람직한
    점근적 성질(일치성, 효율성)을 모두 물려받는다.

### 2.3 잡음 분산의 MLE

$\partial \ell / \partial \sigma^2 = 0$으로 두면 MLE 추정값을 얻는다.

$$
\hat{\sigma}^2_{\text{MLE}}
= \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2
$$

이 추정량은 **편향되어 있다**. 불편 추정량은 (자유도인) $n - p - 1$로 나눈다.

---

## 3. 고전적 가정 (가우스–마르코프)

OLS 추정량이 **최량 선형 불편 추정량**(BLUE)이 되려면 다음 가정들이 성립해야 한다.

### 3.1 매개변수에 대한 선형성

$$
E[y \mid \mathbf{x}] = \mathbf{w}^\top \mathbf{x} + b
$$

*매개변수*에 대해 선형이기만 하면 모델에 (다항 항 같은) 비선형 *특징*이 들어가도 된다.

### 3.2 엄격한 외생성

$$
E[\epsilon \mid \mathbf{X}] = \mathbf{0}
$$

특징이 오차항과 상관되지 않아야 한다. 누락 변수 편향도, 특징의 측정 오차도 없어야 한다.

### 3.3 등분산성

$$
\operatorname{Var}(\epsilon_i \mid \mathbf{x}_i) = \sigma^2
\quad \forall\, i
$$

관측에 걸쳐 오차 분산이 일정해야 한다. 분산이 크기에 따라 커지는 금융 데이터에서는
이 가정의 위반(이분산성)이 흔하다.

### 3.4 자기상관 없음

$$
\operatorname{Cov}(\epsilon_i, \epsilon_j) = 0
\quad \forall\, i \neq j
$$

시계열 데이터에서 특히 중요하며, 위반 여부는 더빈–왓슨 검정으로 알아낸다.

### 3.5 완전 열계수

$$
\operatorname{rank}(\mathbf{X}) = p + 1
$$

$\mathbf{X}^\top\mathbf{X}$이 가역임을 보장하며, 완전한 다중공선성이 있으면 위반된다.
다중공선성에 가까우면 수치적 불안정과 매개변수 분산의 팽창이 일어난다.

---

## 4. 단순 선형 회귀 (p = 1)

행렬 정식화에 앞서 스칼라 경우를 살펴보자. 우리는 다음을 최소화하는 직선
$\hat{y}_i = wx_i + b$를 찾는다.

$$
\mathcal{L}(w, b)
= \frac{1}{n}\sum_{i=1}^{n}(y_i - wx_i - b)^2
$$

### 4.1 최적 매개변수

편도함수를 0으로 두면 다음을 얻는다.

$$
b^* = \bar{y} - w^*\bar{x},
\qquad
\boxed{
  w^* = \frac{\sum_{i=1}^{n}(x_i - \bar{x})(y_i - \bar{y})}
             {\sum_{i=1}^{n}(x_i - \bar{x})^2}
      = \frac{S_{xy}}{S_{xx}}.
}
$$

### 4.2 상관계수와의 관계

표본 표준편차 $s_x$, $s_y$와 표본 상관계수를 다음과 같이 정의하자.

$$
\rho = \frac{S_{xy}}{n\, s_x\, s_y}
$$

그러면 다음이 성립한다.

$$
w^* = \rho\,\frac{s_y}{s_x}
$$

즉 최적의 기울기는 상관계수에 표준편차의 비를 곱한 것이다. 새로운 점 $x_*$에 대한
예측은 다음과 같다.

$$
\hat{y}_* = \bar{y} + \rho\,\frac{s_y}{s_x}\,(x_* - \bar{x})
$$

### 4.3 결정계수

단순 선형 회귀에서 $R^2$ 통계량은 다음을 만족한다.

$$
R^2 = 1 - \frac{\text{SS}_{\text{res}}}{\text{SS}_{\text{tot}}} = \rho^2
$$

!!! warning "다중 회귀"
    항등식 $R^2 = \rho^2$은 단순 선형 회귀에서**만** 성립한다.
    $p > 1$이면 $R^2$은 *다중* 상관계수의 제곱과 같다.

### 4.4 이차 조건

$(w, b)$에 대한 $\mathcal{L}$의 헤세 행렬은 다음과 같다.

$$
\mathbf{H} = \frac{2}{n}
\begin{pmatrix}
\sum x_i^2 & \sum x_i \\
\sum x_i   & n
\end{pmatrix}
$$

데이터의 분산이 0이 아니기만 하면 이 행렬은 양의 정부호이며, 따라서 임계점이 전역
최솟값임이 확인된다.

### 4.5 파이썬 구현

```python
import numpy as np

def simple_linear_regression(x: np.ndarray, y: np.ndarray):
    """상관계수 형태의 공식으로 y = w*x + b를 적합시킨다.

    매개변수
    --------
    x, y : 모양이 (n,)인 1차원 배열

    반환값
    ------
    w, b, rho, r_squared : float
    """
    x_bar, y_bar = x.mean(), y.mean()
    s_xy = np.sum((x - x_bar) * (y - y_bar))
    s_xx = np.sum((x - x_bar) ** 2)
    s_yy = np.sum((y - y_bar) ** 2)

    w = s_xy / s_xx
    b = y_bar - w * x_bar

    rho = s_xy / np.sqrt(s_xx * s_yy)
    r_squared = rho ** 2

    return w, b, rho, r_squared
```

---

## 5. 스칼라 형태에서 행렬 형태로

편향 열을 포함한 $\mathbf{X} = [\mathbf{1} \mid \mathbf{x}] \in \mathbb{R}^{n \times 2}$과
매개변수 벡터 $\boldsymbol{\theta} = (b, w)^\top$으로 모델을 쓰면, 최적성 조건
$\nabla_{\boldsymbol{\theta}} \mathcal{L} = \mathbf{0}$에서 **정규 방정식**이 나온다.

$$
\mathbf{X}^\top\mathbf{X}\,\boldsymbol{\theta}^*
= \mathbf{X}^\top\mathbf{y}
$$

이는 위에서 유도한 스칼라 공식의 $p$차원 일반화이다. 여기서 두 가지 해법이 따라 나온다.

| 해법 | 페이지 | 언제 쓰는가 |
|----------|------|-------------|
| 닫힌 형태(정규 방정식) | [닫힌 형태의 해](closed_form.md) | $p < 10{,}000$이고 정확한 해가 필요할 때 |
| 경사 하강법 | [경사 하강법 해](gd_solution.md) | $p$나 $n$이 클 때, 미니배치 학습 |

---

## 6. 다중 출력

표준 선형 회귀는 $\mathbf{x} \in \mathbb{R}^p$를 스칼라 $y$로 대응시킨다.
$q$개의 목표를 동시에 예측할 때 모델은 다음이 된다.

$$
\hat{\mathbf{Y}} = \mathbf{X}\mathbf{W} + \mathbf{1}\mathbf{b}^\top,
\qquad
\mathbf{W} \in \mathbb{R}^{p \times q},\;\;
\hat{\mathbf{Y}} \in \mathbb{R}^{n \times q}
$$

손실은 프로베니우스 노름 MSE이다.

$$
\mathcal{L}
= \frac{1}{nq}\|\mathbf{Y} - \hat{\mathbf{Y}}\|_F^2
$$

정규 방정식의 해는 열 단위로 확장된다.

$$
\mathbf{B}^*
= (\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top\mathbf{Y}
\;\in\; \mathbb{R}^{(p+1) \times q}
$$

$\mathbf{B}^*$의 각 열은 그 목표에 대한 단일 출력 해이다. 출력들이 같은 설계 행렬을
공유하므로 서로 **분리**되기 때문이다.

!!! info "다중 출력이 도움이 될 때"
    출력들이 서로 독립이면 $q$개의 모델을 따로 적합시켜도 결과가 같다. 다중 출력
    정식화는 **공유되는 구조**를 더할 때 비로소 이득이 된다. 예를 들어 신경망의
    공유 은닉층이나 다변량 회귀의 공통 공분산 구조가 그렇다.

### PyTorch: `nn.Linear(p, q)`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

torch.manual_seed(42)
n, p, q = 500, 3, 2
X = torch.randn(n, p)
W_true = torch.tensor([[2.0, -1.0], [0.5, 1.5], [-0.3, 0.8]])
b_true = torch.tensor([1.0, -2.0])
Y = X @ W_true + b_true + 0.2 * torch.randn(n, q)

model = nn.Linear(p, q)   # 가중치 모양: (q, p) = (2, 3)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
loader = DataLoader(TensorDataset(X, Y), batch_size=32, shuffle=True)

for epoch in range(200):
    for X_b, Y_b in loader:
        loss = F.mse_loss(model(X_b), Y_b)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

---

## 7. 데이터 요건

### 7.1 특징 스케일링

경사 기반 최적화를 쓸 때 선형 회귀는 특징의 규모에 민감하다. 표준화
($z = (x - \mu) / \sigma$)를 하면 모든 특징이 경사에 동등하게 기여하고 학습률을
고르기도 쉬워진다.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # 학습 데이터의 통계량을 쓴다!
```

!!! warning "정칙화된 모델"
    릿지, 라쏘, 엘라스틱 넷 앞에서는 스케일링이 **필수**이다. 벌점이 크기에
    의존하므로, 그러지 않으면 분산이 큰 특징의 계수가 과도하게 줄어든다.

### 7.2 PyTorch에서의 텐서 모양

```python
# X: (n_samples, n_features) — 언제나 2차원
# y: (n_samples, 1)          — 일관성을 위해 열벡터로

X = torch.randn(100, 8)
y = torch.randn(100, 1)

# 흔한 실수: y를 (100,) 모양으로 두면 브로드캐스팅 문제가 생긴다
y_flat = torch.randn(100)
y_correct = y_flat.reshape(-1, 1)
```

---

## 8. 가정의 검증

### 8.1 잔차 진단

```python
import matplotlib.pyplot as plt

def plot_residual_diagnostics(y_true, y_pred):
    """잔차 대 적합값 그림과 히스토그램."""
    residuals = y_true - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].scatter(y_pred, residuals, alpha=0.5, s=10)
    axes[0].axhline(0, color="r", ls="--")
    axes[0].set_xlabel("Fitted Values")
    axes[0].set_ylabel("Residuals")
    axes[0].set_title("Residuals vs Fitted")

    axes[1].hist(residuals, bins=30, edgecolor="black", density=True)
    axes[1].set_xlabel("Residual")
    axes[1].set_ylabel("Density")
    axes[1].set_title("Residual Distribution")

    plt.tight_layout()
    return fig
```

---

## 9. 절 안내도

| 페이지 | 내용 |
|------|---------|
| [닫힌 형태의 해](closed_form.md) | 벡터 미적분 예비 지식, 정규 방정식, 기하학적 해석, QR/SVD 해법 |
| [경사 하강법 해](gd_solution.md) | MSE–NLL 관계, 배치/미니배치/SGD, autograd, `nn.Linear` 파이프라인 |
| [다항 특징](polynomial_features.md) | 비선형 특징 사상, 편향–분산 절충, 교차 검증 |
| [릿지 회귀](ridge_regression.md) | $\ell_2$ 정칙화, 베이즈적 해석, 축소의 기하 |
| [라쏘 회귀](lasso_regression.md) | $\ell_1$ 정칙화, 희소성, 엘라스틱 넷, 좌표 하강법 |

---

## 10. 금융에서의 응용: CAPM 베타

금융에서 단순 선형 회귀를 응용한 고전적인 예가 **자본자산가격결정모형**(CAPM)이다.

$$
R_{\text{WMT}} = \alpha + \beta\, R_{\text{SPY}} + \varepsilon
$$

여기서 $\beta$는 월마트 수익률이 전체 시장에 대해 갖는 민감도를 잰다.

| $\beta$ 값 | 해석 |
|---------------|----------------|
| $\beta = 1$ | 시장과 함께 움직인다 |
| $\beta > 1$ | 시장보다 변동성이 크다 (공격적) |
| $\beta < 1$ | 시장보다 변동성이 작다 (방어적) |

WMT는 보통 $\beta \approx 0.4$–$0.6$인 방어적 종목이다. NumPy, scikit-learn,
PyTorch를 모두 사용하고 이동 창 베타 추정까지 포함한 전체 구현은
`code/wmt_on_spy.py`를 참고하라.

---

## 참고 문헌

1. Hastie, T., Tibshirani, R. & Friedman, J. (2009). *The Elements of
   Statistical Learning*, Ch. 3.
2. Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*, Ch. 3.
3. Freedman, D., Pisani, R. & Purves, R. *Statistics* (4th ed.), Ch. 10–12.
4. Murphy, K. P. (2022). *Probabilistic Machine Learning: An Introduction*,
   Ch. 4.

## 연습문제

**연습문제 1.**
MSE 손실 $L(\mathbf{w}) = \frac{1}{2N}\|\mathbf{Xw} - \mathbf{y}\|^2$의 $\mathbf{w}$에 대한 경사를 유도하라.

??? success "연습문제 1 풀이"
    $$
    \nabla_\mathbf{w} L = \frac{1}{N}\mathbf{X}^\top(\mathbf{Xw} - \mathbf{y})
    $$

    전개하면 $L = \frac{1}{2N}(\mathbf{w}^\top\mathbf{X}^\top\mathbf{Xw} - 2\mathbf{y}^\top\mathbf{Xw} + \mathbf{y}^\top\mathbf{y})$이다. 항별로 미분하면 $\nabla_\mathbf{w}(\mathbf{w}^\top\mathbf{X}^\top\mathbf{Xw}) = 2\mathbf{X}^\top\mathbf{Xw}$이고 $\nabla_\mathbf{w}(2\mathbf{y}^\top\mathbf{Xw}) = 2\mathbf{X}^\top\mathbf{y}$이다.

---

**연습문제 2.**
$\epsilon \sim \mathcal{N}(0, \sigma^2)$인 가우스 잡음을 가정한 $y = \mathbf{w}^\top\mathbf{x} + \epsilon$에서 선형 회귀의 확률적 해석을 설명하라.

??? success "연습문제 2 풀이"
    가우스 잡음 아래에서 $p(y|\mathbf{x}, \mathbf{w}) = \mathcal{N}(\mathbf{w}^\top\mathbf{x}, \sigma^2)$이다. 음의 로그가능도(NLL)는 $-\log p(\mathbf{y}|\mathbf{X},\mathbf{w}) = \frac{N}{2}\log(2\pi\sigma^2) + \frac{1}{2\sigma^2}\|\mathbf{Xw}-\mathbf{y}\|^2$이다. NLL을 최소화하는 것은 MSE를 최소화하는 것과 같으므로, MSE 손실은 가우스 잡음 아래의 MLE에 대응한다.

---

**연습문제 3.**
특징들이 완전히 상관되어 있을 때(다중공선성) 선형 회귀에는 무슨 일이 일어나는가? 이것이 정규 방정식에 어떤 영향을 주는가?

??? success "연습문제 3 풀이"
    특징들이 완전히 상관되어 있으면 $\mathbf{X}^\top\mathbf{X}$이 특이(계수 부족)해지고, 정규 방정식 $\mathbf{X}^\top\mathbf{X}\mathbf{w} = \mathbf{X}^\top\mathbf{y}$은 무한히 많은 해를 갖는다. $\mathbf{X}^\top\mathbf{X}$의 조건수가 무한대가 되어 수치적 해가 불안정해진다. 정칙화(릿지/라쏘)가 이를 해결한다.

---

**연습문제 4.**
단변량 선형 회귀를 PyTorch로 바닥부터 구현하고 `sklearn.linear_model.LinearRegression`과 비교하라.

??? success "연습문제 4 풀이"
    ```python
    import torch
    w = torch.randn(1, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    lr = 0.01
    for _ in range(1000):
        pred = X * w + b
        loss = ((pred - y)**2).mean()
        loss.backward()
        with torch.no_grad():
            w -= lr * w.grad; b -= lr * b.grad
            w.grad.zero_(); b.grad.zero_()
    ```
