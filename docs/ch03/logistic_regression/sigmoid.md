# 시그모이드 함수
## 학습 목표

이 절을 마치면 다음을 할 수 있게 된다.

- 시그모이드 함수를 제일원리에서 유도하기
- 시그모이드를 확률 모형화에 이상적으로 만드는 성질 이해하기
- 승산과 승산비의 관점에서 모델의 계수 해석하기
- 결정 경계의 확률적 의미 설명하기
- PyTorch에서 시그모이드 변환을 구현하고 시각화하기

---

## 시그모이드 함수

### 유도

로그 승산을 선형 함수로 모형화하면 시그모이드 함수가 자연스럽게 나타난다.

$$
\log\frac{p}{1-p} = z
$$

$p$에 대해 풀면 다음과 같다.

$$
\frac{p}{1-p} = e^z \implies p = e^z(1-p) \implies p(1+e^z) = e^z
$$

$$
p = \frac{e^z}{1+e^z} = \frac{1}{1+e^{-z}} \equiv \sigma(z)
$$

### 수학적 정의

**시그모이드 함수**(로지스틱 함수라고도 한다)는 다음과 같이 정의된다.

$$
\sigma(z) = \frac{1}{1+e^{-z}} = \frac{e^z}{1+e^z}
$$

두 형태는 동등하며 서로 다른 상황에서 유용하다. 첫 번째 형태는 $z$가 큰 양수일 때 수치적으로 더 안정적이고, 두 번째는 $z$가 큰 음수일 때 안정적이다.

### 주요 성질

| 성질 | 수학적 형태 | 의의 |
|----------|------------------|--------------|
| 치역 | $\sigma: \mathbb{R} \to (0, 1)$ | 임의의 실수를 유효한 확률로 보낸다 |
| 대칭성 | $\sigma(-z) = 1 - \sigma(z)$ | 여확률이 대칭을 이룬다 |
| 중심 | $\sigma(0) = 0.5$ | 로그 승산이 0이면 확률이 같다 |
| 도함수 | $\sigma'(z) = \sigma(z)(1-\sigma(z))$ | 역전파에 쓰기 좋은 우아한 경사 |
| 극한 | $\lim_{z\to\infty}\sigma(z) = 1$, $\lim_{z\to-\infty}\sigma(z) = 0$ | 양 끝에서 매끄럽게 포화한다 |
| 역함수 | $\sigma^{-1}(p) = \log\frac{p}{1-p}$ | 로짓 함수가 로그 승산을 되돌려 준다 |

### 대칭성

대칭성 $\sigma(-z) = 1 - \sigma(z)$은 이진 분류의 근간이다. 이는 다음을 뜻한다.

$$
P(Y=0|z) = 1 - P(Y=1|z) = 1 - \sigma(z) = \sigma(-z)
$$

이로써 모델이 내부적으로 일관됨이 보장된다. 두 클래스의 확률의 합이 언제나 1이다.

**증명:**

$$
\sigma(-z) = \frac{1}{1+e^{-(-z)}} = \frac{1}{1+e^z} = \frac{1+e^z - e^z}{1+e^z} = 1 - \frac{e^z}{1+e^z} = 1 - \sigma(z)
$$

### 도함수

시그모이드 함수의 도함수는 놀랄 만큼 우아한 형태를 갖는다.

$$
\frac{d\sigma}{dz} = \sigma(z)(1-\sigma(z))
$$

**거듭제곱 법칙을 이용한 증명:**

$\sigma(z) = (1+e^{-z})^{-1}$이라 하자. 그러면 다음과 같다.

$$
\sigma'(z) = -1 \cdot (1+e^{-z})^{-2} \cdot (-e^{-z}) = \frac{e^{-z}}{(1+e^{-z})^2}
$$

$\sigma(z) = \frac{1}{1+e^{-z}}$이고 $1-\sigma(z) = \frac{e^{-z}}{1+e^{-z}}$임을 알아채면 다음을 얻는다.

$$
\sigma(z)(1-\sigma(z)) = \frac{1}{1+e^{-z}} \cdot \frac{e^{-z}}{1+e^{-z}} = \frac{e^{-z}}{(1+e^{-z})^2} = \sigma'(z)
$$

이 성질은 역전파에 결정적으로 중요하다. 추가로 값을 저장하지 않고도 순전파의 출력만으로 경사를 곧바로 계산할 수 있기 때문이다.

---

## 승산과 승산비

### 승산이란 무엇인가?

**승산**(odds)은 사건이 일어날 확률과 일어나지 않을 확률의 비를 나타낸다.

$$
\text{Odds}(Y=1) = \frac{P(Y=1)}{P(Y=0)} = \frac{p}{1-p}
$$

| 확률 $p$ | 승산 | 해석 |
|-----------------|------|----------------|
| 0.5 | 1:1 | 가능성이 같다 |
| 0.75 | 3:1 | 일어날 가능성이 세 배 |
| 0.9 | 9:1 | 아홉 배 더 그럴듯하다 |
| 0.1 | 1:9 | 일어나지 않을 가능성이 아홉 배 |

### 로그 승산 (로짓)

**로그 승산** 또는 **로짓**은 승산의 자연로그이다.

$$
\text{logit}(p) = \log\frac{p}{1-p}
$$

로그 승산에는 핵심적인 이점이 있다. 값의 범위가 $-\infty$에서 $+\infty$까지여서 선형 모형화의 대상으로 알맞다.

| 확률 $p$ | 승산 | 로그 승산 |
|-----------------|------|----------|
| 0.01 | 0.0101 | -4.60 |
| 0.10 | 0.111 | -2.20 |
| 0.50 | 1.0 | 0.00 |
| 0.90 | 9.0 | 2.20 |
| 0.99 | 99.0 | 4.60 |

### 계수를 로그 승산비로 해석하기

로지스틱 회귀에서는 다음을 모형화한다.

$$
\log\frac{P(Y=1|\mathbf{x})}{P(Y=0|\mathbf{x})} = \beta_0 + \beta_1 x_1 + \cdots + \beta_d x_d
$$

특징 하나 $x_1$을 생각하자. (다른 변수를 고정한 채) $x_1$이 한 단위 늘어날 때의 **승산비**는 다음과 같다.

$$
\text{OR} = \frac{\text{Odds}(Y=1|x_1+1)}{\text{Odds}(Y=1|x_1)} = \frac{e^{\beta_0 + \beta_1(x_1+1)}}{e^{\beta_0 + \beta_1 x_1}} = e^{\beta_1}
$$

이는 강력한 결과이다. **$e^{\beta_j}$은 $x_j$가 한 단위 늘어날 때 승산이 곱해지는 배율을 나타낸다.**

| $\beta_1$ | $e^{\beta_1}$ | 해석 |
|-----------|---------------|----------------|
| 0 | 1.0 | 승산에 영향이 없다 |
| 0.5 | 1.65 | 단위당 승산이 65% 증가 |
| 1.0 | 2.72 | 승산이 거의 세 배 |
| -0.5 | 0.61 | 승산이 39% 감소 |
| -1.0 | 0.37 | 승산이 약 37%로 줄어든다 |

---

## "부드러운" 계단 함수로서의 시그모이드

시그모이드 함수는 헤비사이드 계단 함수의 매끄러운 근사로 볼 수 있다. 전이의 "가파름"은 선형 예측자의 계수 크기가 조절한다.

$\|\boldsymbol{\beta}\|$이 크면 다음과 같다.

- 예측이 더 확신에 차 있다 (0이나 1에 더 가깝다)
- 결정 경계가 더 "날카롭다"
- 모델이 분류에 대해 더 확신한다

$\|\boldsymbol{\beta}\|$이 작으면 다음과 같다.

- 예측이 더 불확실하다 (0.5에 더 가깝다)
- 결정 경계가 더 "부드럽다"
- 모델이 더 큰 불확실성을 드러낸다

### 왜 로그 승산인가?

확률을 직접 모형화하는 대신 로그 승산을 모형화하면 여러 이점이 있다.

1. **유계가 아닌 범위**: 선형 예측자는 어떤 값이든 취할 수 있으며, 이는 로그 승산의 범위 $(-\infty, +\infty)$과 맞아떨어진다
2. **곱셈적 효과**: 계수를 로그 승산비로 깔끔하게 해석할 수 있다
3. **지수족에 자연스럽다**: 로그 승산은 베르누이 분포의 자연 매개변수이다
4. **클래스를 대칭적으로 다룬다**: 로그 승산이 0이면 두 클래스의 확률이 같다

---

## 수치적 안정성에 대한 고려

실제로 시그모이드를 구현할 때는 극단적인 입력값이 수치적 문제를 일으킬 수 있다.

| 상황 | 문제 |
|----------|---------|
| $z \gg 0$ | 분자 형태에서 $e^z$이 넘친다 |
| $z \ll 0$ | 분모 형태에서 $e^{-z}$이 넘친다 |

**수치적으로 안정한** 구현은 $z$의 부호에 따라 알맞은 형태를 쓴다.

$$
\sigma(z) = \begin{cases}
\frac{1}{1+e^{-z}} & \text{if } z \geq 0 \\[4pt]
\frac{e^z}{1+e^z} & \text{if } z < 0
\end{cases}
$$

PyTorch의 `torch.sigmoid`은 이미 내부적으로 이를 처리한다.

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
# 1부: 시그모이드 함수 분석
# ============================================================================

print("=" * 70)
print("SIGMOID FUNCTION PROPERTIES")
print("=" * 70)

def sigmoid(z):
    """수치적으로 안정한 시그모이드 함수."""
    return torch.sigmoid(z)

def sigmoid_derivative(z):
    """시그모이드의 도함수: σ'(z) = σ(z)(1 - σ(z))."""
    s = sigmoid(z)
    return s * (1 - s)

# 주요 성질을 보여준다
z_vals = torch.tensor([-5.0, -2.0, 0.0, 2.0, 5.0])
print("\nSigmoid values at key points:")
print("-" * 50)
for z in z_vals:
    s = sigmoid(z)
    print(f"σ({z:+.1f}) = {s.item():.6f}")

# 대칭성 σ(-z) = 1 - σ(z)을 확인한다
print("\nVerifying symmetry σ(-z) = 1 - σ(z):")
print("-" * 50)
for z in [1.0, 2.0, 3.0]:
    z_tensor = torch.tensor(z)
    left = sigmoid(-z_tensor)
    right = 1 - sigmoid(z_tensor)
    print(
        f"σ({-z:.1f}) = {left.item():.6f}, "
        f"1 - σ({z:.1f}) = {right.item():.6f}, "
        f"Difference: {abs(left - right).item():.2e}"
    )

# 도함수 성질 σ'(z) = σ(z)(1 - σ(z))을 확인한다
print("\nVerifying derivative σ'(z) = σ(z)(1 - σ(z)):")
print("-" * 50)
z_test = torch.tensor([0.0, 1.0, -1.0], requires_grad=True)
s_test = sigmoid(z_test)
for i, z in enumerate(z_test):
    analytical = sigmoid_derivative(z)
    s_test[i].backward(retain_graph=True)
    numerical = z_test.grad[i]
    z_test.grad.zero_()
    print(
        f"z = {z.item():+.1f}: Analytical = {analytical.item():.6f}, "
        f"Autograd = {numerical.item():.6f}"
    )

# ============================================================================
# 2부: 승산과 로그 승산 시각화
# ============================================================================

print("\n" + "=" * 70)
print("ODDS AND LOG-ODDS RELATIONSHIPS")
print("=" * 70)

probabilities = torch.tensor([0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99])

print("\nProbability → Odds → Log-Odds conversion:")
print("-" * 60)
print(f"{'Probability':>12} {'Odds':>12} {'Log-Odds':>12}")
print("-" * 60)

for p in probabilities:
    odds = p / (1 - p)
    log_odds = torch.log(odds)
    print(f"{p.item():>12.4f} {odds.item():>12.4f} {log_odds.item():>12.4f}")

# ============================================================================
# 3부: 해석 가능한 계수를 갖는 로지스틱 회귀
# ============================================================================

print("\n" + "=" * 70)
print("LOGISTIC REGRESSION COEFFICIENT INTERPRETATION")
print("=" * 70)

# 특징의 효과를 아는 합성 데이터를 생성한다
n_samples = 1000
X = np.random.randn(n_samples, 2)
true_beta = np.array([0.5, 1.5, -0.8])  # [절편, beta1, beta2]
z = true_beta[0] + X[:, 0] * true_beta[1] + X[:, 1] * true_beta[2]
p = 1 / (1 + np.exp(-z))
y = np.random.binomial(1, p)

print(f"\nTrue coefficients:")
print(f"  Intercept (β₀): {true_beta[0]:.3f}")
print(f"  Feature 1 (β₁): {true_beta[1]:.3f} → Odds Ratio: {np.exp(true_beta[1]):.3f}")
print(f"  Feature 2 (β₂): {true_beta[2]:.3f} → Odds Ratio: {np.exp(true_beta[2]):.3f}")

# 나누고 표준화한다
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
scaler = StandardScaler()
X_train = torch.FloatTensor(scaler.fit_transform(X_train))
X_test = torch.FloatTensor(scaler.transform(X_test))
y_train = torch.FloatTensor(y_train).reshape(-1, 1)
y_test = torch.FloatTensor(y_test).reshape(-1, 1)

# 모델을 정의하고 학습시킨다
class LogisticRegression(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))

    def log_odds(self, x):
        """로그 승산(선형 예측자)을 반환한다."""
        return self.linear(x)

model = LogisticRegression(2)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

for epoch in range(500):
    predictions = model(X_train)
    loss = criterion(predictions, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# 학습된 계수를 꺼낸다
learned_weights = model.linear.weight.data.numpy().flatten()
learned_bias = model.linear.bias.data.numpy()[0]

print(f"\nLearned coefficients (on standardized features):")
print(f"  Intercept (β₀): {learned_bias:.3f}")
print(
    f"  Feature 1 (β₁): {learned_weights[0]:.3f} "
    f"→ Odds Ratio: {np.exp(learned_weights[0]):.3f}"
)
print(
    f"  Feature 2 (β₂): {learned_weights[1]:.3f} "
    f"→ Odds Ratio: {np.exp(learned_weights[1]):.3f}"
)

# ============================================================================
# 4부: 시각화
# ============================================================================

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# 그림 1: 시그모이드 함수와 그 도함수
z_range = torch.linspace(-6, 6, 200)
axes[0].plot(z_range.numpy(), sigmoid(z_range).numpy(), "b-", linewidth=2, label="σ(z)")
axes[0].plot(
    z_range.numpy(),
    sigmoid_derivative(z_range).numpy(),
    "r--",
    linewidth=2,
    label="σ'(z)",
)
axes[0].axhline(y=0.5, color="gray", linestyle=":", alpha=0.7)
axes[0].axvline(x=0, color="gray", linestyle=":", alpha=0.7)
axes[0].fill_between(
    z_range.numpy(), 0, sigmoid_derivative(z_range).numpy(), alpha=0.2, color="red"
)
axes[0].set_xlabel("z (log-odds)", fontsize=11)
axes[0].set_ylabel("Value", fontsize=11)
axes[0].set_title("Sigmoid Function and Derivative", fontsize=12, fontweight="bold")
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)
axes[0].set_ylim([-0.05, 1.05])

# 그림 2: 승산비 해석
features = ["Feature 1", "Feature 2"]
odds_ratios = [np.exp(learned_weights[0]), np.exp(learned_weights[1])]
colors = ["green" if or_val > 1 else "red" for or_val in odds_ratios]

bars = axes[1].bar(features, odds_ratios, color=colors, alpha=0.7, edgecolor="black")
axes[1].axhline(y=1, color="black", linestyle="--", linewidth=1)
axes[1].set_ylabel("Odds Ratio", fontsize=11)
axes[1].set_title("Coefficient Interpretation as Odds Ratios", fontsize=12, fontweight="bold")
axes[1].grid(True, alpha=0.3, axis="y")

for bar, or_val, coef in zip(bars, odds_ratios, learned_weights):
    height = bar.get_height()
    axes[1].text(
        bar.get_x() + bar.get_width() / 2.0,
        height + 0.05,
        f"OR={or_val:.2f}\n(β={coef:.2f})",
        ha="center",
        va="bottom",
        fontsize=9,
    )

# 그림 3: 수치적 안정성 비교
def naive_sigmoid(z):
    return 1 / (1 + np.exp(-z))

def stable_sigmoid(z):
    return np.where(z >= 0, 1 / (1 + np.exp(-z)), np.exp(z) / (1 + np.exp(z)))

extreme_z = np.linspace(-20, 20, 100)
axes[2].plot(extreme_z, stable_sigmoid(extreme_z), "g-", linewidth=2, label="Stable σ(z)")
axes[2].axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
axes[2].axvspan(-20, -10, alpha=0.1, color="red", label="Overflow risk (naive)")
axes[2].axvspan(10, 20, alpha=0.1, color="red")
axes[2].set_xlabel("z", fontsize=11)
axes[2].set_ylabel("σ(z)", fontsize=11)
axes[2].set_title("Numerical Stability", fontsize=12, fontweight="bold")
axes[2].legend(fontsize=9)
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("sigmoid_odds_analysis.png", dpi=150, bbox_inches="tight")
plt.show()

# ============================================================================
# 5부: 수치적 안정성 비교
# ============================================================================

print("\n" + "=" * 70)
print("NUMERICAL STABILITY OF SIGMOID")
print("=" * 70)

extreme_values = np.array([-1000, -100, 0, 100, 1000])

print("\nComparing naive vs stable sigmoid for extreme values:")
print("-" * 60)
print(f"{'z':>10} {'Naive':>15} {'Stable':>15} {'PyTorch':>15}")
print("-" * 60)

for z in extreme_values:
    with np.errstate(over="ignore"):
        naive_result = naive_sigmoid(z)
    stable_result = stable_sigmoid(z)
    pytorch_result = torch.sigmoid(torch.tensor(z, dtype=torch.float32)).item()
    print(
        f"{z:>10.0f} {naive_result:>15.6e} {stable_result:>15.6e} {pytorch_result:>15.6e}"
    )

print("\n✓ PyTorch's sigmoid is numerically stable for all input ranges!")
```

---

## 요약

| 개념 | 공식 | 핵심 통찰 |
|---------|---------|-------------|
| 시그모이드 | $\sigma(z) = \frac{1}{1+e^{-z}}$ | $\mathbb{R} \to (0,1)$로 보낸다 |
| 대칭성 | $\sigma(-z) = 1 - \sigma(z)$ | 여확률 |
| 도함수 | $\sigma'(z) = \sigma(z)(1-\sigma(z))$ | 효율적인 역전파를 가능케 한다 |
| 승산 | $\frac{p}{1-p}$ | 성공 대 실패의 비 |
| 승산비 | $e^{\beta_j}$ | 단위당 곱셈적 효과 |

매끄러운 포화, 여확률의 대칭성, 자기 자신을 참조하는 도함수를 두루 갖춘 시그모이드 함수의 수학적 우아함이 이 함수를 이진 분류의 정석적인 선택으로 만든다. 한편 로그 승산의 틀은 특징이 결과의 승산에 어떻게 영향을 주는지를 수치로 말해 주는 해석 가능한 계수를 제공한다.

## 연습문제

**연습문제 1.**
시그모이드의 도함수가 $z=0$에서 최댓값을 가짐을 증명하고 그 최댓값을 계산하라.

??? success "연습문제 1 풀이"
    $\sigma'(z) = \sigma(z)(1-\sigma(z))$이다. 이는 $p = \sigma(z) \in (0,1)$일 때의 곱 $p(1-p)$이다.

    산술-기하 평균 부등식에 의해 $p(1-p) \leq \left(\frac{p + (1-p)}{2}\right)^2 = \frac{1}{4}$이며, 등호는 $p = 1/2$일 때 성립한다.

    $z = 0$일 때 $\sigma(z) = 1/2$이므로 $\max_z \sigma'(z) = 1/4$이다. 이는 시그모이드를 지나는 경사 신호가 층마다 많아야 $1/4$이라는 뜻이므로 중요하며, 깊은 신경망의 경사 소실 문제의 한 원인이 된다. $\square$

---

**연습문제 2.**
대칭성 $\sigma(-z) = 1 - \sigma(z)$을 증명하라.

??? success "연습문제 2 풀이"
    $$
    \sigma(-z) = \frac{1}{1+e^{-(-z)}} = \frac{1}{1+e^z} = \frac{e^{-z}}{e^{-z}(1+e^z)} = \frac{e^{-z}}{e^{-z}+1} = 1 - \frac{1}{1+e^{-z}} = 1 - \sigma(z)
    $$

    $\square$

---

**연습문제 3.**
어떤 로지스틱 회귀 모델에서 연 단위로 측정된 특징의 계수가 $\beta_1 = 0.7$이다. 승산비를 계산하고 해석하라. 5년이 늘어나면 승산은 어떻게 달라지는가?

??? success "연습문제 3 풀이"
    한 단위 증가에 대한 승산비는 $\text{OR} = e^{\beta_1} = e^{0.7} \approx 2.014$이다. 한 해가 더해질 때마다 양성 결과의 승산이 대략 두 배가 된다.

    5년이 늘어나면 $\text{OR}_5 = e^{5 \times 0.7} = e^{3.5} \approx 33.1$이다. 승산비는 곱해지므로($(e^{0.7})^5 = e^{3.5}$) 5년에 걸쳐 승산이 약 33배로 늘어난다.

---

**연습문제 4.**
수치적으로 안정한 시그모이드 $\sigma(z) = \begin{cases} \frac{1}{1+e^{-z}} & z \geq 0 \\ \frac{e^z}{1+e^z} & z < 0 \end{cases}$을 구현하고, 극단적인 값에서 `torch.sigmoid`와 일치하는지 확인하라.

??? success "연습문제 4 풀이"
    ```python
    import torch

    def stable_sigmoid(z):
        pos = torch.clamp(z, min=0)
        neg = torch.clamp(z, max=0)
        return torch.where(z >= 0,
                          1 / (1 + torch.exp(-pos)),
                          torch.exp(neg) / (1 + torch.exp(neg)))

    z = torch.tensor([-1000.0, -1.0, 0.0, 1.0, 1000.0])
    print(stable_sigmoid(z))   # tensor([0., 0.2689, 0.5, 0.7311, 1.])
    print(torch.sigmoid(z))    # 같은 값이며 NaN이나 넘침이 없다
    ```
    두 갈래로 나눈 형태는 $|z|$이 클 때 $e^{|z|}$을 계산하지 않으므로 넘침을 막는다.
