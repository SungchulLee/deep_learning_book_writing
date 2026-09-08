# 정칙화 훑어보기

정칙화는 일반화를 높이기 위해 학습 과정에 제약을 두는 폭넓은 전략의 무리를 아우른다. 모든 지도 학습 모델은 (참된 패턴을 담기에 너무 단순한) **과소적합**과 (일반화되는 구조를 배우는 대신 학습 데이터의 잡음을 외우는) **과적합** 사이를 헤쳐 나가야 한다. 정칙화 기법은 모델의 복잡도에 벌점을 주거나, 잡음을 넣거나, 실효 학습 집합을 넓히거나, 최적화의 궤적을 제한하여 과적합에 체계적으로 대응한다.

이 절은 뒤 절들의 자세한 설명에 앞서 개념적·수학적 토대를 다룬다. 편향-분산 절충, 제약 최적화의 기하, 그리고 기법들의 통일된 분류이다.

---

## 1. 과적합과 과소적합

### 모델 복잡도의 스펙트럼

**과소적합 (높은 편향).** 모델이 참된 입출력 관계를 표현할 용량이 모자라면 과소적합한다. 징후로는 학습을 더 해도 줄지 않는 높은 학습 손실, 둘 다 크고 서로 가까운 학습 손실과 검증 손실, 데이터의 체계적인 패턴을 놓치는 예측이 있다. 예를 들어 이차 관계에 선형 모델을 적합시키면 데이터가 아무리 많아도 잔차에 구조가 남는다.

**과적합 (높은 분산).** 모델이 학습 데이터에 지나치게 밀착하여 일반화되지 않는 잡음과 특이점까지 담으면 과적합한다. 징후로는 검증 손실이 오르는데도 계속 줄어드는 학습 손실, 벌어지는 학습 성능과 검증 성능의 격차, 학습 집합을 조금만 흔들어도 크게 달라지는 예민한 예측이 있다.

**최적 영역.** 두 극단 사이에 알맞은 지점이 있다. 참된 패턴을 담을 만큼 용량이 있으면서 잡음을 외우지 않도록 적절히 제약된 곳이다. 정칙화의 목표는 모델을 이 영역에 머물게 하는 것이다.

### 적합 진단하기: 학습 곡선

과적합과 과소적합을 진단하는 가장 실용적인 도구는 학습 손실과 검증 손실을 에폭의 함수로 그린 **학습 곡선**이다.

| 양상 | 학습 손실 | 검증 손실 | 격차 | 진단 |
|---------|-----------|----------|-----|-----------|
| 둘 다 높고 가까움 | 높음 | 높음 | 작음 | 과소적합 |
| 둘 다 낮고 가까움 | 낮음 | 낮음 | 작음 | 좋은 적합 |
| 학습은 낮고 검증은 오름 | 낮음 | 높음 / 오름 | 큼 | 과적합 |
| 둘 다 줄고 격차는 일정 | 줄어듦 | 줄어듦 | 보통 | 아직 배우는 중 (학습을 계속한다) |

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

def generate_polynomial_data(n_samples=500, noise_std=0.3, seed=42):
    """다항식을 참값으로 하는 합성 데이터를 만든다."""
    # 참 함수를 3차 다항식으로 정해 둔다. 정칙화를 다루려면 "무엇이
    # 신호이고 무엇이 잡음인지"를 우리가 알아야 하는데, 실제 데이터로는
    # 그것을 알 수 없으므로 이렇게 만들어 쓴다.
    torch.manual_seed(seed)
    # unsqueeze(1)로 (n,) 을 (n,1) 로 만든다. nn.Linear가
    # (배치, 특성) 꼴을 받기 때문이다
    x = torch.linspace(-3, 3, n_samples).unsqueeze(1)
    y_true = 0.5 * x**3 - 2 * x**2 + x + 1
    # 잡음의 표준편차가 곧 "줄일 수 없는 오차"다. 어떤 모델도
    # 검증 손실을 noise_std^2 = 0.09 아래로는 내릴 수 없다
    y = y_true + noise_std * torch.randn_like(y_true)
    return x, y

def compute_learning_curves(model, train_loader, val_loader, epochs=200, lr=0.01):
    """
    모델을 학습시키고 에포크마다 학습/검증 손실을 기록한다.

    반환값:
        'train_loss'와 'val_loss'를 키로 갖는 사전. 값은 각각 실수의 목록이다.
    """
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(epochs):
        # --- 학습 ---
        model.train()   # 에폭마다 다시 부른다(아래에서 eval로 바꾸므로)
        train_loss_sum = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()
            # 배치 크기를 곱해 두는 것이 요점이다. criterion이 이미
            # 배치 안에서 평균을 냈으므로, 다시 곱해 합으로 되돌린 뒤
            # 아래에서 전체 표본 수로 나눈다. 이래야 마지막 배치가
            # 작아도 정확한 표본 평균이 된다
            train_loss_sum += loss.item() * X_batch.size(0)

        # --- 검증 ---
        model.eval()    # 드롭아웃을 끄고 배치 정규화가 이동 통계를 쓰게 한다
        val_loss_sum = 0.0
        with torch.no_grad():   # eval과 별개로 기울기 추적을 끈다
            for X_batch, y_batch in val_loader:
                val_loss_sum += criterion(model(X_batch), y_batch).item() * X_batch.size(0)

        # len(loader)가 아니라 len(loader.dataset)으로 나눈다.
        # 앞의 것은 배치 수, 뒤의 것이 표본 수다
        history['train_loss'].append(train_loss_sum / len(train_loader.dataset))
        history['val_loss'].append(val_loss_sum / len(val_loader.dataset))

    return history
```

### 모델 복잡도 비교하기

```python
class PolynomialModel(nn.Module):
    """지정한 차수의 다항 특징에 대한 선형 회귀."""

    def __init__(self, degree):
        super().__init__()
        self.degree = degree
        self.linear = nn.Linear(degree, 1)

    def forward(self, x):
        # 다항 특징 행렬 [x, x^2, ..., x^degree] 만들기
        features = torch.cat([x ** i for i in range(1, self.degree + 1)], dim=1)
        return self.linear(features)

class DeepNetwork(nn.Module):
    """과적합을 보이기 위한 과매개변수화된 깊은 신경망."""

    def __init__(self, hidden_dim=256, n_layers=4):
        super().__init__()
        layers = [nn.Linear(1, hidden_dim), nn.ReLU()]
        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        layers.append(nn.Linear(hidden_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# --- 데이터 준비 ---
x, y = generate_polynomial_data(n_samples=200, noise_std=0.5)
dataset = TensorDataset(x, y)
train_set, val_set = random_split(dataset, [140, 60])
train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
val_loader = DataLoader(val_set, batch_size=64)

# 경우 1: 과소적합 — 삼차 데이터에 선형 모델
underfit_model = PolynomialModel(degree=1)
underfit_history = compute_learning_curves(underfit_model, train_loader, val_loader)

# 경우 2: 알맞은 적합 — 삼차 데이터에 삼차 모델
goodfit_model = PolynomialModel(degree=3)
goodfit_history = compute_learning_curves(goodfit_model, train_loader, val_loader)

# 경우 3: 과적합 — 작은 데이터셋에 깊은 신경망
overfit_model = DeepNetwork(hidden_dim=256, n_layers=4)
overfit_history = compute_learning_curves(overfit_model, train_loader, val_loader, epochs=500)
```

---

## 2. 편향-분산 절충

### 수학적 유도

참된 관계가 $y = f(x) + \epsilon$이고 잡음 $\epsilon$이 $\mathbb{E}[\epsilon] = 0$과 $\text{Var}[\epsilon] = \sigma^2$을 만족하는 회귀 문제를 생각하자. 학습 알고리즘은 데이터 생성 분포에서 뽑은 데이터셋 $\mathcal{D}$으로 학습한 추정량 $\hat{f}_{\mathcal{D}}(x)$을 내놓는다.

고정된 시험 점 $x_0$에 대해, 잡음과 $\mathcal{D}$의 무작위성 양쪽에 걸친 기대 제곱 오차는 다음과 같다.

$$
\text{EPE}(x_0) = \mathbb{E}_{\mathcal{D}, \epsilon}\left[\left(y_0 - \hat{f}_{\mathcal{D}}(x_0)\right)^2\right]
$$

$y_0 = f(x_0) + \epsilon$으로 쓰고 $\bar{f}(x_0) = \mathbb{E}_{\mathcal{D}}[\hat{f}_{\mathcal{D}}(x_0)]$을 들여와 전개한다.

$$
\text{EPE}(x_0) = \mathbb{E}\left[\left(\epsilon + f(x_0) - \hat{f}_{\mathcal{D}}(x_0)\right)^2\right]
$$

$\epsilon$이 $\hat{f}_{\mathcal{D}}(x_0)$과 독립이므로 다음이 성립한다.

$$
= \sigma^2 + \mathbb{E}_{\mathcal{D}}\left[\left(f(x_0) - \hat{f}_{\mathcal{D}}(x_0)\right)^2\right]
$$

제곱 항 안에 $\bar{f}(x_0)$을 더하고 빼서 전개하면($\mathbb{E}_{\mathcal{D}}[\hat{f}_{\mathcal{D}}(x_0) - \bar{f}(x_0)] = 0$이므로 교차항이 사라짐에 유의하라) 세 항으로의 분해를 얻는다.

$$
\boxed{\text{EPE}(x_0) = \underbrace{\sigma^2}_{\text{Irreducible Error}} + \underbrace{\left(f(x_0) - \bar{f}(x_0)\right)^2}_{\text{Bias}^2} + \underbrace{\mathbb{E}_{\mathcal{D}}\left[\left(\hat{f}_{\mathcal{D}}(x_0) - \bar{f}(x_0)\right)^2\right]}_{\text{Variance}}}
$$

| 항 | 의미 | 무엇에 달려 있는가 |
|------|---------|------------|
| 줄일 수 없는 오차 $\sigma^2$ | 데이터에 내재한 잡음 | 데이터 생성 과정 |
| 편향$^2$ | 평균 모델의 체계적 오차 | 모델의 종류 (용량) |
| 분산 | $\mathcal{D}$의 선택에 대한 $\hat{f}$의 민감도 | 모델의 복잡도, 데이터셋 크기 |

### U자 곡선과 이중 하강

모델의 복잡도가 아주 단순한 것에서 아주 유연한 것으로 커짐에 따라 **편향**은 단조롭게 줄고(유연할수록 더 많은 패턴을 적합할 수 있다), **분산**은 단조롭게 커지며(유연할수록 학습 데이터에 더 민감하다), **총 오차**는 최적 복잡도에서 최솟값을 갖는 U자를 그린다.

모델의 복잡도를 고정한 채 학습 집합의 크기 $n$을 키우면 편향은 거의 그대로이지만 분산이 대략 $O(1/n)$으로 줄어든다. 데이터를 더 모으는 것이 분산이 큰 모델에 특히 효과적인 이유가 여기 있다.

최근의 딥러닝 연구는 **이중 하강** 현상을 밝혀냈다. (모델이 처음으로 학습 오차 0에 이르는) 보간 문턱을 넘어서면 복잡도를 더 키울 때 오차가 *다시 줄어들* 수 있다. 이것이 편향-분산의 틀을 무효로 만들지는 않는다. 분해는 여전히 성립한다. 다만 과매개화 영역에서 경사 하강법, 초기화, 조기 종료에서 오는 암묵적 정칙화 덕분에 분산이 줄어들 수 있음을 보여준다.

### 편향과 분산의 실험적 추정

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

def true_function(x):
    """참 함수: 삼차 다항식."""
    return 0.5 * x**3 - x**2 + 0.5 * x + 1

def generate_dataset(n_samples=50, noise_std=0.5, seed=None):
    """참 함수에서 잡음이 섞인 데이터셋을 만든다."""
    if seed is not None:
        torch.manual_seed(seed)
    x = torch.linspace(-2, 2, n_samples).unsqueeze(1)
    y = true_function(x) + noise_std * torch.randn_like(x)
    return x, y

def train_model(model, x, y, epochs=1000, lr=0.01):
    """데이터셋 하나로 모델을 학습시킨다."""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    for _ in range(epochs):
        optimizer.zero_grad()
        criterion(model(x), y).backward()
        optimizer.step()
    return model

def estimate_bias_variance(degree, n_datasets=200, n_samples=50, noise_std=0.5):
    """
    여러 부트스트랩 데이터셋으로 학습시켜 주어진 다항 차수 모델의
    편향^2과 분산을 추정한다.

    반환값:
        bias_sq: float — 시험 점에 걸쳐 평균한 편향의 제곱
        variance: float — 시험 점에 걸쳐 평균한 분산
        mse: float — 시험 점과 데이터셋에 걸쳐 평균한 평균제곱오차
    """
    x_test = torch.linspace(-2, 2, 100).unsqueeze(1)
    y_true = true_function(x_test)

    # 데이터셋마다 예측 모으기
    all_predictions = []
    for seed in range(n_datasets):
        x_train, y_train = generate_dataset(n_samples, noise_std, seed=seed)
        model = PolynomialModel(degree)
        train_model(model, x_train, y_train)

        with torch.no_grad():
            pred = model(x_test)
        all_predictions.append(pred)

    predictions = torch.stack(all_predictions)  # (n_datasets, n_test, 1)
    mean_pred = predictions.mean(dim=0)          # E[f_hat(x)]

    bias_sq = ((y_true - mean_pred) ** 2).mean().item()
    variance = predictions.var(dim=0).mean().item()
    mse = ((predictions - y_true.unsqueeze(0)) ** 2).mean().item()

    return bias_sq, variance, mse

# 모델 복잡도를 훑기
degrees = [1, 2, 3, 5, 8, 12, 18]
results = []

for d in degrees:
    bias_sq, var, mse = estimate_bias_variance(d, n_datasets=100)
    results.append({'degree': d, 'bias_sq': bias_sq, 'variance': var, 'mse': mse})
    print(f"Degree {d:2d}:  Bias²={bias_sq:.4f}  Var={var:.4f}  MSE={mse:.4f}")
```

### 절충을 눈으로 보기

```python
import matplotlib.pyplot as plt

# 앞에서 차수를 바꿔 가며 모은 results에서 세 곡선을 뽑는다.
# 편향의 제곱과 분산, 그리고 그 둘의 합인 MSE를 한 그림에 겹쳐 그려
# 편향-분산 맞바꿈을 눈으로 확인하는 것이 목적이다.
degrees_plot = [r['degree'] for r in results]
biases = [r['bias_sq'] for r in results]
variances = [r['variance'] for r in results]
mses = [r['mse'] for r in results]

fig, ax = plt.subplots(figsize=(8, 5))
# 차수가 오를수록 편향은 단조로 줄어든다. 표현할 수 있는 함수가 넓어지므로
ax.plot(degrees_plot, biases, 'b-o', label='Bias²')
# 반대로 분산은 늘어난다. 자료의 잡음까지 따라가기 시작하므로
ax.plot(degrees_plot, variances, 'r-s', label='Variance')
# 둘의 합인 MSE는 U자를 그린다. 그 바닥이 고를 만한 복잡도다
ax.plot(degrees_plot, mses, 'k--^', label='Total MSE')

ax.set_xlabel('Polynomial Degree (Model Complexity)')
ax.set_ylabel('Error')
ax.set_title('Bias-Variance Tradeoff')
ax.legend()
ax.grid(True, alpha=0.3)   # 격자를 옅게 깔아 값을 읽기 쉽게 한다
plt.tight_layout()
plt.savefig('bias_variance_tradeoff.png', dpi=150)
plt.show()
```

---

## 3. 정칙화 기법의 분류

정칙화 방법은 학습 파이프라인의 *어디에* 끼어드는지에 따라 정리할 수 있다.

### 1. 명시적인 매개변수 벌점

이 방법들은 크거나 불필요한 가중치를 억제하려고 손실 함수에 벌점 항을 곧바로 더한다. 정칙화된 목적 함수는 일반적으로 다음 꼴이다.

$$
\mathcal{L}_{\text{reg}}(\theta) = \mathcal{L}_{\text{data}}(\theta) + \Omega(\theta)
$$

여기서 $\Omega(\theta)$은 벌점이고 그 상대적 가중치가 편향-분산 절충을 조절한다.

| 방법 | 벌점 항 | 핵심 성질 |
|--------|-------------|--------------|
| L2 (릿지) | $\lambda \| w \|_2^2$ | 모든 가중치를 0 쪽으로 줄인다 |
| L1 (라쏘) | $\lambda \| w \|_1$ | 일부 가중치를 정확히 0으로 만든다 (희소성) |
| 엘라스틱 넷 | $\lambda_1 \| w \|_1 + \lambda_2 \| w \|_2^2$ | 희소성과 무리 짓기의 안정성을 결합한다 |

### 제약의 기하

매개변수 벌점은 저마다 매개변수 공간의 가능 영역을 제한한다. 2차원 가중치 벡터 $(w_1, w_2)$에 대한 L2 제약을 보자. 정칙화가 없으면 최적화기가 $\mathbb{R}^2$ 어디에서든 $\mathcal{L}_{\text{data}}$의 전역 최솟값을 찾는다. L2를 쓰면 실효 가능 집합이 $\lambda$이 정하는 어떤 문턱값 $t$에 대해 $\{w : \|w\|_2^2 \leq t\}$이 되고, 정칙화된 해는 손실 등고선이 제약의 경계에 접하는 곳에 놓인다. L1은 마름모 $\{w : \|w\|_1 \leq t\}$으로 제약하는데, 그 꼭짓점이 좌표축 위에 있어 L1이 희소한 해를 내는 이유가 된다.

### 실효 자유도

정칙화는 모델이 실제로 쓰는 매개변수의 수를 줄인다. 릿지 회귀에서는 다음과 같다.

$$
\text{df}(\lambda) = \text{tr}\left[X(X^TX + \lambda I)^{-1}X^T\right] = \sum_{j=1}^{p} \frac{\mu_j}{\mu_j + \lambda}
$$

여기서 $\mu_j$은 $X^TX$의 고윳값이다. $\lambda \to 0$이면 $\text{df} \to p$이고(전체 모델), $\lambda \to \infty$이면 $\text{df} \to 0$이다.

### 정칙화가 편향-분산 절충을 옮기는 방식

| 기법 | 편향에 미치는 영향 | 분산에 미치는 영향 | 순효과 |
|-----------|---------------|-------------------|------------|
| L2 (릿지) | 조금 늘어남 | 크게 줄어듦 | 총 오차가 낮아짐 |
| L1 (라쏘) | 조금 늘어남 | 크게 줄어듦 | 더 희소한 모델 |
| 드롭아웃 | 조금 늘어남 | 크게 줄어듦 | 앙상블 효과 |
| 조기 종료 | 조금 늘어남 | 크게 줄어듦 | 실효 용량을 제한함 |
| 데이터 증강 | 그대로이거나 줄어듦 | 줄어듦 | 실효 데이터셋이 커짐 |

### 2. 구조적 제약

이 방법들은 실효 용량을 제한하려고 신경망의 구조나 순전파를 고친다.

**드롭아웃**은 학습 중에 뉴런을 무작위로 끈다.

$$
\tilde{h} = \frac{m \odot h}{1-p}, \quad m_i \sim \text{Bernoulli}(1-p)
$$

이는 공적응을 막고, 단위가 $d$개인 층에 대해 $2^d$개 부분 신경망의 암묵적인 앙상블을 만든다.

**드롭커넥트**는 활성화 전체가 아니라 개별 *가중치*를 무작위로 0으로 만들어 드롭아웃을 일반화하며, 더 세밀한 확률적 정칙화를 제공한다.

**배치 정규화**는 각 미니배치 안에서 활성화를 정규화한다.

$$
\hat{x}_i = \frac{x_i - \mu_{\mathcal{B}}}{\sqrt{\sigma_{\mathcal{B}}^2 + \epsilon}}
$$

주로 학습의 안정을 위해 설계되었지만, 배치 정규화는 미니배치 통계량이 들여오는 잡음을 통해 정칙화 효과도 낸다.

### 3. 데이터 쪽 정칙화

이 방법들은 모델을 제약하는 대신 학습 데이터를 넓히거나 고친다.

**데이터 증강**은 의미를 보존하는 변환으로 학습 예제의 사본을 만든다(이미지에는 무작위 자르기, 뒤집기, 회전, 색 흔들기, 텍스트에는 유의어 치환, 역번역).

**이름표 평활화**는 딱딱한 원-핫 목표를 모든 클래스에 작은 확률 질량을 나눠 준 부드러운 목표로 바꾸어 모델이 지나치게 확신하는 것을 막는다.

**믹스업**과 **컷믹스**는 입력 쌍과 그 이름표를 섞어 가상의 학습 예제를 만들며, 학습 예제 사이에서 선형적인 거동을 이끈다.

**컷아웃**(무작위 지우기)은 입력 이미지의 직사각형 영역을 무작위로 가려, 모델이 국소적으로 잘 구별되는 조각에 기대는 대신 물체의 공간적 전체에 주목하도록 만든다.

**잡음 주입**은 학습 중에 입력, 활성화, 경사에 무작위 요동을 더한다. 입력 잡음이 $\epsilon \sim \mathcal{N}(0, \sigma^2 I)$인 선형 회귀에서 이는 L2 정칙화와 동등하다.

$$
\mathbb{E}_\epsilon[\|y - (x + \epsilon)^T w\|^2] = \|y - x^T w\|^2 + \sigma^2 \|w\|^2
$$

### 4. 최적화 기반 정칙화

이 방법들은 모델이나 데이터를 고치는 대신 학습 과정 자체를 다스린다.

**조기 종료**는 검증 손실이 더 나아지지 않을 때 학습을 멈춰 모델이 과적합 영역으로 들어가는 것을 막는다. 선형 모델에서 경사 하강법과 함께 쓰는 조기 종료는 실효 강도가 $\lambda_{\text{eff}} \approx 1/(\eta t)$인 L2 정칙화와 수학적으로 동등하다. 여기서 $\eta$은 학습률, $t$은 반복 횟수이다.

**학습률 일정 조절**과 **경사 자르기**도 정칙화 효과를 내지만, 전통적으로 정칙화 기법으로 분류되지는 않는다.

---

## 4. 정칙화 기법 결합하기

실무에서는 여러 정칙화 방법을 함께 쓴다.

### 대표적인 딥러닝 처방

```python
import torch.nn as nn
import torch.optim as optim

class RegularizedCNN(nn.Module):
    """여러 정칙화 기법을 결합한 CNN."""

    def __init__(self, num_classes=10, dropout_rate=0.3):
        super().__init__()
        # 정칙화를 세 겹으로 겹쳐 건다. 막는 곳이 서로 달라 함께 쓸 수 있다.
        #   배치 정규화 — 층 입력의 분포를 다스린다
        #   드롭아웃    — 활성을 흔든다
        #   가중치 감쇠 — 가중치의 크기를 누른다(아래 옵티마이저에서)
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),   # 32x32 유지
            # 활성 함수 "앞"에 둔다. ReLU에 들어가는 값의 자를 맞춰야
            # 한쪽으로 치우쳐 유닛이 죽는 것을 막을 수 있다
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),                  # 32x32 -> 16x16
            # 합성곱 출력은 이웃 화소끼리 값이 거의 같아 화소 하나를 꺼도
            # 옆 화소가 그 정보를 들고 있다. 그래서 채널을 통째로 끄는
            # Dropout2d라야 뜻이 있다. 또 배치 정규화 "뒤"에 두어야
            # 꺼진 값이 배치 통계를 흔들지 않는다
            nn.Dropout2d(dropout_rate),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),                  # 16x16 -> 8x8
            nn.Dropout2d(dropout_rate),
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 8 * 8, 256),   # 두 번 풀링해 8x8, 채널 64
            nn.ReLU(),
            # 완전연결층의 유닛은 서로 이웃하지 않아 값이 겹치지 않는다.
            # 그래서 여기서는 보통의 Dropout으로 충분하다
            nn.Dropout(dropout_rate),
            # 마지막 층에는 아무것도 걸지 않는다. 출력이 갈래별 로짓이므로
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)   # (배치, 64, 8, 8) -> (배치, 4096)
        return self.classifier(x)

model = RegularizedCNN()
optimizer = optim.AdamW(
    model.parameters(),
    lr=1e-3,
    # AdamW의 "분리된" 감쇠. 보통의 Adam은 이 항을 기울기에 더하는데,
    # 그러면 적응적 학습률에 나눠져 매개변수마다 실제 세기가 달라진다.
    # AdamW는 갱신 단계에서 따로 빼므로 의도한 세기가 그대로 걸린다
    weight_decay=1e-2
)
```

### 기법을 결합할 때의 지침

| 조합 | 상호작용 | 권고 |
|-------------|------------|----------------|
| 드롭아웃 + 배치 정규화 | 배치 정규화가 드롭아웃의 필요를 줄인다 | 드롭아웃 비율을 낮게 쓴다 (0.1–0.2) |
| L2 + 드롭아웃 | 둘 다 실효 용량을 줄인다 | 검증 손실이 오르면 한쪽을 줄인다 |
| 증강 + 드롭아웃 | 서로 보완적이다 (데이터 쪽 + 모델 쪽) | 함께 써도 대체로 무난하다 |
| 조기 종료 + 아무 방법 | 안전망으로서 언제나 이롭다 | 다른 방법과 늘 함께 쓴다 |
| L1 + L2 (엘라스틱 넷) | L1은 희소성, L2는 안정성 | 특징이 상관되어 있으면 엘라스틱 넷을 쓴다 |
| 이름표 평활화 + 믹스업 | 둘 다 목표를 부드럽게 한다 | 믹스업을 쓸 때는 평활화 계수를 줄인다 |

---

## 5. 정칙화 전략 고르기

### 판단의 틀

1. **데이터 증강과 조기 종료로 시작하라.** 위험이 적고 두루 쓸 수 있다
2. **가중치 감쇠(L2)를 더하라.** AdamW 같은 현대적인 최적화기의 기본값이다
3. 과적합하기 쉬운 밀집층이 있다면 **드롭아웃을 더하라**
4. 학습의 안정과 가벼운 정칙화를 위해 **배치 정규화를 쓰라**
5. 분류 과제에서 지나친 확신을 막으려면 **이름표 평활화를 적용하라**
6. 특징 선택이나 희소성이 필요하다면 **L1이나 엘라스틱 넷을 적용하라**
7. 모든 정칙화 초매개변수를 조율하는 데 **교차 검증을 쓰라**

### 구조별 기본값

| 구조 | 권장 기법 |
|--------------|----------------------|
| MLP | 드롭아웃 (0.5), 가중치 감쇠, 조기 종료 |
| CNN | 배치 정규화, 공간 드롭아웃 (0.2), 증강, 가중치 감쇠 |
| RNN/LSTM | 층 사이의 드롭아웃 (0.3), 가중치 감쇠 |
| 트랜스포머 | 드롭아웃 (0.1), 이름표 평활화, 가중치 감쇠 |
| 고전적 기계 학습 (선형/트리) | L1/L2/엘라스틱 넷, 교차 검증 |

---

## 6. 흔한 원인과 처방

### 과소적합의 원인

1. **모델 용량 부족**: 층이 너무 적거나 매개변수가 너무 적다
2. **지나친 정칙화**: 가중치 감쇠나 드롭아웃 비율이 너무 크다
3. **학습 부족**: 에폭이 너무 적거나 학습률이 너무 낮다
4. **나쁜 특징 표현**: 유익한 특징이 빠져 있다

### 과적합의 원인

1. 데이터의 양과 복잡도에 비해 **지나친 모델 용량**
2. 용량이 큰 모델을 제약하기에 **부족한 학습 데이터**
3. 검증에 기반한 멈춤 없이 **너무 오래 학습함**
4. 유연한 모델에 **정칙화를 전혀 적용하지 않음**

### 처방

**과소적합에는** 모델의 용량을 키우고, 정칙화의 강도를 줄이고, 알맞은 학습률 일정으로 더 오래 학습하고, 더 나은 특징을 만들거나 더 풍부한 입력 표현을 쓴다.

**과적합에는** 정칙화(드롭아웃, 가중치 감쇠, 데이터 증강)를 적용하고, 인내 횟수를 둔 조기 종료를 쓰고, 학습 데이터를 더 모으고, 모델의 용량을 줄인다.

---

## 연습문제

**연습문제 1.**
정칙화 기법을 명시적인 것(벌점 기반)과 암묵적인 것(학습 절차 기반)으로 분류하라.

??? success "연습문제 1 풀이"
    명시적: L1, L2, 엘라스틱 넷 (손실에 벌점을 더한다). 암묵적: 드롭아웃(무작위 가리기), 데이터 증강(입력 변형), 조기 종료(학습 제한), 배치 정규화(정규화 층), 경사 잡음(확률적 최적화). 암묵적 정칙화는 이론적으로 분석하기 더 어려운 경우가 많지만 효과는 그에 못지않다.

---

**연습문제 2.**
정칙화의 맥락에서 편향-분산 절충을 설명하라.

??? success "연습문제 2 풀이"
    정칙화는 편향을 늘리지만(모델이 제약되어 참 함수에 완벽히 맞지 않을 수 있다) 분산을 줄인다(학습 데이터의 요동에 덜 민감해진다). 최적의 정칙화는 기대 시험 오차와 같은 편향^2 + 분산의 합을 최소로 만든다.

---

**연습문제 3.**
구체적인 문제에 대한 정칙화 전략을 설계하라. 학습 이미지 1000장과 ResNet-50으로 하는 이미지 분류이다.

??? success "연습문제 3 풀이"
    이미지가 1000장뿐인데 매개변수가 약 2500만 개이므로 심한 과적합이 예상된다. 전략: (1) 앞쪽 층을 동결한다(ImageNet에서의 전이 학습), (2) 강한 데이터 증강(RandAugment), (3) 분류기 앞에 드롭아웃 $p=0.5$, (4) 가중치 감쇠 $\lambda = 0.01$, (5) 인내 10의 조기 종료, (6) 믹스업/컷믹스.

---

**연습문제 4.**
현대의 대형 언어 모델이 고전적인 딥러닝에 비해 명시적 정칙화를 비교적 적게 쓰는 이유를 설명하라.

??? success "연습문제 4 풀이"
    대형 언어 모델은 방대한 데이터셋으로 학습하므로 과적합이 덜 걱정된다(데이터 자체가 정칙화 노릇을 한다). 주로 드롭아웃(흔히 0.1뿐)과 가중치 감쇠만 쓴다. 큰 배치 크기의 SGD에서 오는 암묵적 정칙화와 엄청난 데이터 양이 무거운 명시적 벌점 없이도 충분한 일반화를 준다.

## 정리하며

이 마당은 과적합과 과소적합、편향-분산 절충、정칙화 기법의 분류、정칙화 기법 결합하기을 차례로 짚었다.

**참고 문헌**

1. Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning* (2nd ed.). Springer. Chapters 3, 7.
2. Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press. Chapters 5, 7.
3. Geman, S., Bienenstock, E., & Doursat, R. (1992). Neural Networks and the Bias/Variance Dilemma. *Neural Computation*, 4(1), 1-58.
4. Belkin, M., et al. (2019). Reconciling Modern Machine Learning Practice and the Classical Bias-Variance Trade-off. *PNAS*, 116(32), 15849-15854.
5. Kukačka, J., Golkov, V., & Cremers, D. (2017). Regularization for Deep Learning: A Taxonomy. arXiv:1710.10686.
6. Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer. Section 1.1.
