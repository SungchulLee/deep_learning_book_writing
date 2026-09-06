# 이진 분류
## 학습 목표

이 절을 마치면 다음을 할 수 있게 된다.

- 이진 분류 문제와 선형 회귀가 여기에 부적합한 이유 이해하기
- 베르누이 확률 모델과 그 지수족 형태 유도하기
- 로지스틱 회귀를 일반화 선형 모형(GLM)의 틀과 연결하기
- 가능도와 로그가능도 함수를 제일원리에서 유도하기
- BCE를 최소화하는 것이 왜 최대가능도 추정과 같은지 설명하기

---

## 이진 분류 문제

이진 분류는 서로 배타적인 두 결과 중 하나를 예측하는 일이다. 사실상 모든 분야에 예가 있다.

| 분야 | 클래스 0 | 클래스 1 |
|--------|---------|---------|
| 의료 | 건강함 | 질병 있음 |
| 금융 | 부도 아님 | 부도 |
| 이메일 | 정상 | 스팸 |
| 제조 | 양품 | 불량 |

근본적인 물음은 이것이다. 특징 $\mathbf{x} \in \mathbb{R}^d$이 주어졌을 때 확률 $P(Y=1|\mathbf{x})$을 어떻게 모형화할 것인가?

### 왜 선형 회귀는 안 되는가?

선형 회귀는 다음을 가정한다.

$$
Y = \mathbf{x}^\top \boldsymbol{\beta} + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \sigma^2)
$$

이는 $Y \in \mathbb{R}$이고 정규분포를 따른다는 뜻이다. 이진 결과에서는 다음이 문제가 된다.

1. 반응 $Y \in \{0, 1\}$은 분명히 정규분포가 아니다
2. 선형 예측자 $\mathbf{x}^\top \boldsymbol{\beta}$은 $\mathbb{R}$의 어떤 값이든 낼 수 있지만 확률은 $[0, 1]$ 안에 있어야 한다

$Y$의 이산적 성격과 확률의 유계인 범위를 함께 존중하는 틀이 필요하다.

---

## 베르누이 분포

### 정의

확률변수 $Y$가 $Y \in \{0, 1\}$의 두 값만 취하면 **베르누이 분포**를 따른다고 한다. 이 분포는 성공 확률을 나타내는 매개변수 $p \in [0, 1]$ 하나로 결정된다.

$$
P(Y = y) = p^y (1-p)^{1-y}, \quad y \in \{0, 1\}
$$

이 간결한 형태는 두 경우를 우아하게 아우른다.

- $y = 1$일 때: $P(Y=1) = p^1(1-p)^0 = p$
- $y = 0$일 때: $P(Y=0) = p^0(1-p)^1 = 1-p$

### 성질

| 성질 | 공식 | 해석 |
|----------|---------|----------------|
| 평균 | $\mathbb{E}[Y] = p$ | 기댓값이 성공 확률과 같다 |
| 분산 | $\text{Var}(Y) = p(1-p)$ | $p=0.5$에서 분산이 최대 |
| 왜도 | $\frac{1-2p}{\sqrt{p(1-p)}}$ | $p=0.5$일 때만 대칭 |

분산 함수 $p(1-p)$은 GLM 이론과 로지스틱 회귀 손실의 헤세 행렬에서 결정적인 역할을 한다([경사 계산](gradient.md) 참고).

### 지수족으로의 정식화

베르누이 분포는 **지수족**에 속하며, 정준 형태로 다음과 같이 쓸 수 있다.

$$
P(Y=y) = \exp\left(\eta y - A(\eta) + B(y)\right)
$$

여기서 각 기호는 다음과 같다.

- $\eta = \log\frac{p}{1-p}$은 **자연 매개변수**(로그 승산)이다
- $A(\eta) = \log(1 + e^\eta)$은 **로그 분배 함수**이다
- 베르누이의 경우 $B(y) = 0$이다

**유도.** 베르누이 확률질량함수에서 출발한다.

$$
P(Y=y) = p^y(1-p)^{1-y}
$$

로그를 취하면 다음과 같다.

$$
\log P(Y=y) = y \log p + (1-y)\log(1-p) = y \log\frac{p}{1-p} + \log(1-p)
$$

$\eta = \log\frac{p}{1-p}$으로 정의하면 $p$에 대해 풀 수 있다.

$$
p = \frac{e^\eta}{1+e^\eta}, \quad 1-p = \frac{1}{1+e^\eta}
$$

따라서 $\log(1-p) = -\log(1+e^\eta)$이고, 다음을 얻는다.

$$
\log P(Y=y) = \eta \, y - \log(1+e^\eta)
$$

이 표현은 깊은 뜻을 담고 있다. 자연 매개변수 $\eta$이 정확히 $p$의 **로짓**이며, 이로써 베르누이 분포와 로지스틱 회귀 사이의 깊은 연결이 확립된다.

---

## 일반화 선형 모형의 틀

### GLM의 세 가지 구성 요소

일반화 선형 모형(GLM)은 특징과 변환된 반응 사이의 선형 관계를 유지하면서 정규분포가 아닌 반응까지 다루도록 선형 회귀를 확장한다. GLM은 세 가지 구성 요소로 이루어진다.

**1. 확률 요소**: 반응 $Y$가 지수족의 어떤 분포(베르누이, 포아송, 정규 등)를 따른다

**2. 체계 요소**: 특징들을 결합한 선형 예측자이다.

$$
\eta = \mathbf{x}^\top \boldsymbol{\beta} = \beta_0 + \beta_1 x_1 + \cdots + \beta_d x_d
$$

**3. 연결 함수**: 기댓값과 선형 예측자를 잇는 단조롭고 미분 가능한 함수 $g(\cdot)$이다.

$$
g(\mu) = \eta, \quad \text{where } \mu = \mathbb{E}[Y|\mathbf{x}]
$$

### 정준 연결 함수

지수족의 각 분포마다 자연 매개변수 변환과 같아지는 **정준 연결 함수**가 존재한다. 정준 연결에는 특별한 성질이 있다.

- 최대가능도 추정을 간단하게 만든다
- 자연스러운 해석을 제공한다
- 식별 가능성을 보장한다

베르누이 분포에서 정준 연결은 **로짓 함수**이다.

$$
g(p) = \log\frac{p}{1-p} = \text{logit}(p)
$$

### GLM에서 로지스틱 회귀 유도하기

우리는 다음 조건 아래에서 $P(Y=1|\mathbf{x}) = p(\mathbf{x})$을 모형화하려 한다.

1. **확률 요소**: $Y | \mathbf{x} \sim \text{Bernoulli}(p(\mathbf{x}))$
2. **체계 요소**: $\eta = \mathbf{x}^\top \boldsymbol{\beta}$
3. **연결 함수**: $g(p) = \log\frac{p}{1-p}$

연결 함수로 이 요소들을 이으면 다음과 같다.

$$
\log\frac{p(\mathbf{x})}{1-p(\mathbf{x})} = \mathbf{x}^\top \boldsymbol{\beta}
$$

$p(\mathbf{x})$에 대해 풀면 다음과 같다.

$$
\frac{p(\mathbf{x})}{1-p(\mathbf{x})} = e^{\mathbf{x}^\top \boldsymbol{\beta}}
$$

$$
p(\mathbf{x})(1 + e^{\mathbf{x}^\top \boldsymbol{\beta}}) = e^{\mathbf{x}^\top \boldsymbol{\beta}}
$$

$$
p(\mathbf{x}) = \frac{e^{\mathbf{x}^\top \boldsymbol{\beta}}}{1 + e^{\mathbf{x}^\top \boldsymbol{\beta}}} = \frac{1}{1 + e^{-\mathbf{x}^\top \boldsymbol{\beta}}} = \sigma(\mathbf{x}^\top \boldsymbol{\beta})
$$

**시그모이드 함수** $\sigma(z) = \frac{1}{1+e^{-z}}$이 GLM의 틀에서 자연스럽게 나타난다! 그 성질은 [시그모이드 함수](sigmoid.md) 절에서 자세히 다룬다.

---

## 가능도 함수

### 확률에서 가능도로

$y_i \in \{0, 1\}$인 데이터셋 $\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^n$이 주어졌을 때, **가능도 함수**는 특정 매개변수 $\boldsymbol{\beta}$ 아래에서 관측된 데이터가 얼마나 그럴듯한지를 잰다.

관측 하나에 대해, $\mathbf{x}_i$과 매개변수 $\boldsymbol{\beta}$이 주어졌을 때 $y_i$을 관측할 확률은 다음과 같다.

$$
P(Y = y_i | \mathbf{x}_i, \boldsymbol{\beta}) = p_i^{y_i} (1-p_i)^{1-y_i}
$$

여기서 $p_i = \sigma(\mathbf{x}_i^\top \boldsymbol{\beta})$은 모델이 예측한 확률이다.

### 전체 가능도 함수

관측들이 **독립이며 같은 분포를 따른다(i.i.d.)**고 가정하면, 모든 관측의 결합 확률은 개별 확률의 곱이다.

$$
\mathcal{L}(\boldsymbol{\beta}) = P(\mathcal{D} | \boldsymbol{\beta}) = \prod_{i=1}^{n} P(y_i | \mathbf{x}_i, \boldsymbol{\beta}) = \prod_{i=1}^{n} p_i^{y_i} (1-p_i)^{1-y_i}
$$

이것이 **가능도 함수**이다. 주어진 매개변수 선택에서 관측된 데이터가 얼마나 "그럴듯한지"를 알려 준다.

---

## 가능도에서 로그가능도로

### 왜 로그를 취하는가?

로그가능도를 다루면 여러 이점이 있다.

| 가능도의 문제 | 로그가능도가 주는 도움 |
|----------------------|-------------------------|
| 작은 수를 많이 곱하면 수치적 아랫넘침 | 로그 확률의 합은 수치적으로 안정적 |
| 미분의 곱 법칙은 복잡하다 | 미분의 합 법칙은 더 간단하다 |
| 어떤 모델에서는 볼록하지 않다 | 로지스틱 회귀에서 로그가능도는 오목하다 |

### 로그가능도 함수

가능도에 자연로그를 취하면 다음과 같다.

$$
\ell(\boldsymbol{\beta}) = \log \mathcal{L}(\boldsymbol{\beta}) = \sum_{i=1}^{n} \log \left[ p_i^{y_i} (1-p_i)^{1-y_i} \right]
$$

성질 $\log(a^b) = b \log a$을 쓰면 다음과 같다.

$$
\ell(\boldsymbol{\beta}) = \sum_{i=1}^{n} \left[ y_i \log p_i + (1-y_i) \log(1-p_i) \right]
$$

이것이 로지스틱 회귀의 **로그가능도**이다.

### 로지스틱 모델로 전개하기

$z_i = \mathbf{x}_i^\top \boldsymbol{\beta}$일 때 $p_i = \sigma(z_i)$을 대입하면 다음과 같다.

$$
\log p_i = \log \frac{e^{z_i}}{1 + e^{z_i}} = z_i - \log(1 + e^{z_i})
$$

$$
\log(1-p_i) = \log \frac{1}{1 + e^{z_i}} = -\log(1 + e^{z_i})
$$

다시 대입하고 정리하면($y_i \log(1+e^{z_i})$ 항들이 상쇄된다) 다음을 얻는다.

$$
\boxed{\ell(\boldsymbol{\beta}) = \sum_{i=1}^{n} \left[ y_i \mathbf{x}_i^\top \boldsymbol{\beta} - \log(1 + e^{\mathbf{x}_i^\top \boldsymbol{\beta}}) \right]}
$$

이것이 로지스틱 회귀 **로그가능도의 최종 형태**이다.

### 행렬 표기

계산 효율을 위해 다음과 같이 쓴다.

$$
\ell(\boldsymbol{\beta}) = \mathbf{y}^\top \mathbf{X} \boldsymbol{\beta} - \mathbf{1}^\top \log(\mathbf{1} + \exp(\mathbf{X}\boldsymbol{\beta}))
$$

여기서 $\mathbf{X} \in \mathbb{R}^{n \times d}$은 설계 행렬, $\mathbf{y} \in \{0,1\}^n$은 이름표 벡터이며, 필요한 곳에서 연산은 원소별로 이루어진다.

---

## 이진 교차 엔트로피와의 관계

### 로그가능도에서 손실 함수로

최적화에서는 최대화가 아니라 최소화를 한다. **음의 로그가능도(NLL)**이 우리의 손실이 된다.

$$
\text{NLL} = -\ell(\boldsymbol{\beta}) = -\sum_{i=1}^{n} \left[ y_i \log p_i + (1-y_i) \log(1-p_i) \right]
$$

$n$으로 나누어 평균을 취하면 다음과 같다.

$$
\text{BCE} = -\frac{1}{n}\sum_{i=1}^{n} \left[ y_i \log p_i + (1-y_i) \log(1-p_i) \right]
$$

이것이 이진 분류의 표준 손실 함수인 **이진 교차 엔트로피(BCE)**이다. 따라서 다음이 성립한다.

$$
\text{Minimizing BCE} = \text{Maximizing log-likelihood} = \text{Maximum Likelihood Estimation}
$$

### 정보 이론적 해석

참 분포 $q$과 예측 분포 $p$ 사이의 교차 엔트로피는 다음과 같다.

$$
H(q, p) = -\mathbb{E}_q[\log p] = -\sum_x q(x) \log p(x)
$$

이진 분류에서 참 이름표가 $y$이고 예측 확률이 $\hat{y}$일 때 다음과 같다.

$$
H(y, \hat{y}) = -[y \log \hat{y} + (1-y) \log(1-\hat{y})]
$$

BCE는 우리의 예측이 참 분포를 얼마나 잘 담아내는지를 잰다. $\hat{y} = y$일 때 최소가 되며, 교차 엔트로피와 엔트로피의 차이는 KL 발산과 같다. 이는 참 분포 대신 모델의 분포를 씀으로써 잃는 정보량이다.

---

## 로그가능도의 성질

### 오목성

로지스틱 회귀의 로그가능도는 **오목**하다(음의 로그가능도가 볼록하다). 이는 다음을 뜻한다.

1. **모든 국소 최댓값이 전역 최댓값이다**
2. 경사 하강법이 전역 최적점으로 수렴한다
3. 국소 최솟값에 갇힐 위험이 없다

증명은 헤세 행렬이 음의 준정부호임을 보이는 데 달려 있으며, 이는 [경사 계산](gradient.md) 절에서 전개한다.

### 로그가능도의 척도

로그가능도는 보통 음수이다($1$보다 작은 확률의 로그를 취하기 때문이다). 더 좋은 모델일수록 로그가능도가 0에 가깝다.

| 로그가능도 | 해석 |
|----------------|----------------|
| 0에 가까움 | 훌륭한 적합 (관측 데이터의 확률이 높다) |
| 크게 음수 | 나쁜 적합 (관측 데이터의 확률이 낮다) |
| $-n \log 2$ | 무작위로 찍는 것과 다를 바 없다 |

---

## PyTorch 구현

```python
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 재현성을 위해 씨앗을 설정한다
torch.manual_seed(42)
np.random.seed(42)

# ============================================================================
# 1부: 베르누이 분포 살펴보기
# ============================================================================

print("=" * 70)
print("BERNOULLI DISTRIBUTION AND GLM FOUNDATIONS")
print("=" * 70)

def bernoulli_properties(p: float) -> dict:
    """
    Compute theoretical properties of Bernoulli distribution.

    Args:
        p: Success probability, must be in [0, 1]

    Returns:
        Dictionary containing mean, variance, and skewness
    """
    if not 0 <= p <= 1:
        raise ValueError("p must be in [0, 1]")
    mean = p
    variance = p * (1 - p)
    skewness = (1 - 2 * p) / np.sqrt(p * (1 - p) + 1e-10)
    return {"mean": mean, "variance": variance, "skewness": skewness}

for p in [0.1, 0.25, 0.5, 0.75, 0.9]:
    props = bernoulli_properties(p)
    print(
        f"p = {p:.2f}: Mean = {props['mean']:.3f}, "
        f"Variance = {props['variance']:.3f}, "
        f"Skewness = {props['skewness']:+.3f}"
    )

# ============================================================================
# 2부: GLM의 구성 요소 — 로지스틱 회귀
# ============================================================================

print("\n" + "=" * 70)
print("GLM STRUCTURE: LINEAR PREDICTOR → SIGMOID → PROBABILITY")
print("=" * 70)

class LogisticRegressionGLM(nn.Module):
    """
    Logistic Regression as a Generalized Linear Model.

    Explicitly separates the GLM components:
    - Linear predictor (systematic component)
    - Inverse link function (sigmoid)

    The random component (Bernoulli) is implicit in the BCE loss.
    """

    def __init__(self, n_features: int):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def linear_predictor(self, x: torch.Tensor) -> torch.Tensor:
        """선형 예측자 η = Xβ(체계 요소)을 계산한다."""
        return self.linear(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """역연결 함수를 적용하여 확률을 얻는다: p = σ(η)."""
        eta = self.linear_predictor(x)
        return torch.sigmoid(eta)

    def log_odds(self, x: torch.Tensor) -> torch.Tensor:
        """선형 예측자와 같은 로그 승산(자연 매개변수)을 반환한다."""
        return self.linear_predictor(x)

# 시각화를 분명히 하기 위해 이미 아는 가중치로 모델을 초기화한다
model = LogisticRegressionGLM(n_features=1)
model.linear.weight.data = torch.tensor([[2.0]])
model.linear.bias.data = torch.tensor([-1.0])

x_vals = torch.linspace(-3, 5, 200).reshape(-1, 1)

with torch.no_grad():
    eta = model.linear_predictor(x_vals)
    p = model(x_vals)

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

axes[0].plot(x_vals.numpy(), eta.numpy(), "b-", linewidth=2)
axes[0].axhline(y=0, color="gray", linestyle="--", alpha=0.5)
axes[0].set_xlabel("Feature x", fontsize=11)
axes[0].set_ylabel("Linear Predictor η", fontsize=11)
axes[0].set_title("Systematic Component\nη = β₀ + β₁x", fontsize=12, fontweight="bold")
axes[0].grid(True, alpha=0.3)
axes[0].set_ylim([-8, 10])

z = torch.linspace(-6, 6, 200)
axes[1].plot(z.numpy(), torch.sigmoid(z).numpy(), "g-", linewidth=2)
axes[1].axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
axes[1].axvline(x=0, color="gray", linestyle="--", alpha=0.5)
axes[1].set_xlabel("Linear Predictor η", fontsize=11)
axes[1].set_ylabel("Probability p", fontsize=11)
axes[1].set_title(
    "Inverse Link Function\np = σ(η) = 1/(1+e⁻η)", fontsize=12, fontweight="bold"
)
axes[1].grid(True, alpha=0.3)

axes[2].plot(x_vals.numpy(), p.numpy(), "r-", linewidth=2)
axes[2].axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
axes[2].set_xlabel("Feature x", fontsize=11)
axes[2].set_ylabel("P(Y=1|x)", fontsize=11)
axes[2].set_title("Complete GLM\np = σ(β₀ + β₁x)", fontsize=12, fontweight="bold")
axes[2].grid(True, alpha=0.3)
axes[2].set_ylim([0, 1])

plt.tight_layout()
plt.savefig("glm_structure.png", dpi=150, bbox_inches="tight")
plt.show()

print(f"\nGLM Model Summary")
print(f"  Coefficient (β₁): {model.linear.weight.item():.3f}")
print(f"  Intercept (β₀): {model.linear.bias.item():.3f}")
print(
    f"  Decision boundary (where p=0.5): "
    f"x = {-model.linear.bias.item() / model.linear.weight.item():.3f}"
)

# ============================================================================
# 3부: 로그가능도 직접 계산하기
# ============================================================================

print("\n" + "=" * 70)
print("LOG-LIKELIHOOD COMPUTATION")
print("=" * 70)

def compute_log_likelihood_manual(X, y, beta):
    """
    Compute log-likelihood using the derived formula.

    ℓ(β) = Σᵢ [yᵢ xᵢᵀβ - log(1 + exp(xᵢᵀβ))]
    """
    z = X @ beta
    ll = y * z - torch.logsumexp(torch.stack([torch.zeros_like(z), z], dim=0), dim=0)
    return ll.sum()

def compute_log_likelihood_direct(X, y, beta):
    """
    Compute log-likelihood using the direct formula.

    ℓ(β) = Σᵢ [yᵢ log(pᵢ) + (1-yᵢ) log(1-pᵢ)]
    """
    z = X @ beta
    p = torch.sigmoid(z)
    eps = 1e-10
    ll = y * torch.log(p + eps) + (1 - y) * torch.log(1 - p + eps)
    return ll.sum()

# 간단한 데이터셋을 생성한다
n_samples = 100
X = torch.randn(n_samples, 2)
true_beta = torch.tensor([0.5, 1.0, -0.5])  # Including bias
X_with_bias = torch.cat([torch.ones(n_samples, 1), X], dim=1)

z_true = X_with_bias @ true_beta
p_true = torch.sigmoid(z_true)
y = torch.bernoulli(p_true)

print(f"\nDataset: {n_samples} samples, {X.shape[1]} features")
print(f"Class distribution: {y.sum().item():.0f} positive, {(1-y).sum().item():.0f} negative")

# 두 계산 방법을 비교한다
test_beta = torch.tensor([0.3, 0.8, -0.3])
ll_manual = compute_log_likelihood_manual(X_with_bias, y, test_beta)
ll_direct = compute_log_likelihood_direct(X_with_bias, y, test_beta)

print(f"\nLog-likelihood comparison (β = {test_beta.tolist()}):")
print(f"  Simplified form (y·z - log(1+eᶻ)): {ll_manual.item():.6f}")
print(f"  Direct form (y·log p + (1-y)·log(1-p)): {ll_direct.item():.6f}")
print(f"  Difference: {abs(ll_manual - ll_direct).item():.2e}")

# ============================================================================
# 4부: 로그가능도의 지형과 경사 하강법을 통한 MLE
# ============================================================================

print("\n" + "=" * 70)
print("MAXIMUM LIKELIHOOD ESTIMATION VIA GRADIENT DESCENT")
print("=" * 70)

class LogisticRegressionML(nn.Module):
    """로그가능도를 명시적으로 추적하는 로지스틱 회귀."""

    def __init__(self, n_features):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))

    def log_likelihood(self, x, y):
        """현재 매개변수에서의 로그가능도를 계산한다."""
        z = self.linear(x)
        log_p = -torch.log1p(torch.exp(-z))
        log_1_minus_p = -torch.log1p(torch.exp(z))
        ll = y * log_p + (1 - y) * log_1_minus_p
        return ll.sum()

# 데이터를 준비한다
X_train, X_test, y_train, y_test = train_test_split(
    X.numpy(), y.numpy(), test_size=0.2, random_state=42
)
scaler = StandardScaler()
X_train = torch.FloatTensor(scaler.fit_transform(X_train))
X_test = torch.FloatTensor(scaler.transform(X_test))
y_train = torch.FloatTensor(y_train).reshape(-1, 1)
y_test = torch.FloatTensor(y_test).reshape(-1, 1)

# 모델을 학습시킨다
ml_model = LogisticRegressionML(2)
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(ml_model.parameters(), lr=0.5)

num_epochs = 200
history = {"log_likelihood": [], "nll_loss": [], "accuracy": []}

print(f"\nTraining for {num_epochs} epochs...")
print("-" * 60)

for epoch in range(num_epochs):
    predictions = ml_model(X_train)
    nll_loss = criterion(predictions, y_train)

    with torch.no_grad():
        ll = ml_model.log_likelihood(X_train, y_train)
        acc = ((predictions >= 0.5).float() == y_train).float().mean()

    optimizer.zero_grad()
    nll_loss.backward()
    optimizer.step()

    history["log_likelihood"].append(ll.item())
    history["nll_loss"].append(nll_loss.item())
    history["accuracy"].append(acc.item())

    if (epoch + 1) % 40 == 0:
        print(
            f"Epoch {epoch+1:3d}: Log-Likelihood = {ll.item():8.3f}, "
            f"NLL Loss = {nll_loss.item():.4f}, Accuracy = {acc.item():.4f}"
        )

# 시각화
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].plot(history["log_likelihood"], "b-", linewidth=2)
axes[0].set_xlabel("Epoch", fontsize=11)
axes[0].set_ylabel("Log-Likelihood", fontsize=11)
axes[0].set_title("Log-Likelihood (Maximized)", fontsize=12, fontweight="bold")
axes[0].grid(True, alpha=0.3)

axes[1].plot(history["nll_loss"], "r-", linewidth=2)
axes[1].set_xlabel("Epoch", fontsize=11)
axes[1].set_ylabel("BCE Loss", fontsize=11)
axes[1].set_title("BCE Loss = Negative Log-Likelihood (Minimized)", fontsize=12, fontweight="bold")
axes[1].grid(True, alpha=0.3)

axes[2].plot(history["accuracy"], "g-", linewidth=2)
axes[2].set_xlabel("Epoch", fontsize=11)
axes[2].set_ylabel("Accuracy", fontsize=11)
axes[2].set_title("Training Accuracy", fontsize=12, fontweight="bold")
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("binary_classification_training.png", dpi=150, bbox_inches="tight")
plt.show()

print("\n✓ Visualization saved!")
```

---

## GLM의 틀이 중요한 이유

1. **통일된 이론**: GLM은 회귀(정규), 분류(베르누이), 계수 데이터(포아송) 문제를 아우르는 공통의 언어를 제공한다

2. **자연스러운 제약**: 연결 함수가 유계가 아닌 선형 예측자를 각 분포에 알맞은 범위로 자동으로 보낸다

3. **해석**: 로짓 연결은 로그 승산비의 관점에서 해석 가능한 계수를 준다([시그모이드 함수](sigmoid.md) 참고)

4. **원리에 기반한 추론**: GLM 이론은 신뢰구간과 가설검정에 대한 점근적 결과를 제공한다

5. **신경망과의 연결**: 로지스틱 회귀는 가장 단순한 신경망이다. 선형 층 하나 뒤에 시그모이드 활성화를 붙이고 BCE 손실로 학습시킨 것이다. 이 관점이 고전 통계학과 딥러닝을 잇는다

---

## 요약

| 개념 | 공식 | 핵심 통찰 |
|---------|---------|-------------|
| 베르누이 확률질량함수 | $p^y(1-p)^{1-y}$ | 관측 하나의 확률 |
| 자연 매개변수 | $\eta = \log\frac{p}{1-p}$ | 정준 매개변수로서의 로그 승산 |
| GLM 연결 함수 | $g(p) = \text{logit}(p)$ | 평균과 선형 예측자를 잇는다 |
| 가능도 | $\prod_i p_i^{y_i}(1-p_i)^{1-y_i}$ | 전체 데이터의 확률 |
| 로그가능도 | $\sum_i [y_i z_i - \log(1+e^{z_i})]$ | 계산에 효율적인 형태 |
| BCE 손실 | $-\frac{1}{n}\ell(\boldsymbol{\beta})$ | 음의 평균 로그가능도 |

GLM의 틀은 로지스틱 회귀를 임의로 고른 선택이 아니라, 베르누이 반응과 로그 승산 척도 위에서의 특징의 선형 효과를 가정할 때 이진 분류에 자연스럽게 따라 나오는 모델로 이해하게 해 주는 이론적 토대를 제공한다.

---

## 참고 문헌

- McCullagh, P., & Nelder, J. A. (1989). *Generalized Linear Models* (2nd ed.). Chapman & Hall.
- Agresti, A. (2015). *Foundations of Linear and Generalized Linear Models*. Wiley.
- Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer. Chapter 4.

## 연습문제

**연습문제 1.**
베르누이 분산 $p(1-p)$이 $p=0.5$에서 최대가 됨을 증명하고, 이것이 분류의 확신도에 대해 갖는 의미를 설명하라.

??? success "연습문제 1 풀이"
    $v(p) = p(1-p) = p - p^2$이라 하자. $v'(p) = 1 - 2p = 0$에서 $p = 0.5$이고, $v''(p) = -2 < 0$이므로 최댓값임이 확인된다. 최대 분산은 $v(0.5) = 0.25$이다.

    의미: 모델은 $p = 0.5$(결정 경계)에서 가장 불확실하다. 모델이 더 확신할수록($p \to 0$ 또는 $p \to 1$) 분산이 줄어든다. 이는 로그가능도의 경사가 모델이 불확실한 경계 근처에서 가장 크고, 확신을 갖고 분류한 점에서는 가장 작다는 뜻이다.

---

**연습문제 2.**
참 데이터 분포 아래에서 기대 로그가능도를 유도하고, 그것이 음의 교차 엔트로피와 같음을 보여라.

??? success "연습문제 2 풀이"
    $q(y|x)$을 참 조건부 분포, $p_\theta(y|x)$을 모델이라 하자. 기대 로그가능도는 다음과 같다.

    $$
    \mathbb{E}_{x,y}[\log p_\theta(y|x)] = \mathbb{E}_x\left[\sum_y q(y|x) \log p_\theta(y|x)\right] = -\mathbb{E}_x[H(q(\cdot|x), p_\theta(\cdot|x))]
    $$

    이는 교차 엔트로피 $H(q, p_\theta) = -\sum_y q(y) \log p_\theta(y)$의 음수이다. 따라서 기대 로그가능도를 최대화하는 것은 교차 엔트로피를 최소화하는 것과 같다. $\square$

---

**연습문제 3.**
로짓 함수 $\log\frac{p}{1-p}$이 베르누이 분포의 "정준" 연결 함수라 불리는 이유를 설명하라.

??? success "연습문제 3 풀이"
    지수족 형태로 쓴 베르누이 확률질량함수는 $p(y|\eta) = \exp(\eta y - \log(1+e^\eta))$이며, 여기서 $\eta = \log\frac{p}{1-p}$이 자연 매개변수이다. 정준 연결은 평균 $\mu = p$을 자연 매개변수 $\eta$으로 보낸다. 베르누이의 경우 이는 로짓 함수 $g(\mu) = \log\frac{\mu}{1-\mu}$이다.

    정준 연결을 쓰면 점수 방정식이 $\mathbf{X}^\top(\mathbf{y} - \boldsymbol{\mu}) = \mathbf{0}$으로 간단해지며, 로그가능도의 오목성이 따라 나오고 (존재할 때) MLE의 유일성이 보장된다.

---

**연습문제 4.**
정규분포의 누적분포함수를 역연결 함수로 쓰는 프로빗 회귀 모델을 구현하고, 합성 데이터에서 로지스틱 회귀와 비교하라.

??? success "연습문제 4 풀이"
    ```python
    import torch
    import torch.distributions as dist

    normal = dist.Normal(0, 1)

    def probit_nll(logits, targets):
        p = normal.cdf(logits)
        p = torch.clamp(p, 1e-7, 1 - 1e-7)
        return -(targets * torch.log(p) + (1 - targets) * torch.log(1 - p)).mean()

    # 로지스틱 회귀(BCE)와 비교한다
    # 둘은 비슷한 결정 경계를 낸다. 프로빗은 꼬리가 조금 더 가벼워서
    # (sigma(1.7*z)가 Phi(z)와 비슷하다) 경계 근처에서 더 가파르게
    # 전이한다.
    ```
