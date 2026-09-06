# 회귀에서의 MLE
## 들어가며

회귀의 손실 함수는 특정한 잡음 가정 아래의 음의 로그가능도이다. 이 절은 그 관계를 정확히 밝힌다. **MSE는 가우스 잡음을 가정하고**, **MAE는 라플라스 잡음을 가정하며**, **이분산 모델은 잡음 자체를 학습한다**.

!!! success "핵심 통찰"
    MSE 손실로 신경망을 학습시킬 때 당신은 목표값이 모델의 예측을 중심으로 하는 정규분포를 따른다는 가정 아래에서 MLE을 수행하고 있는 것이다.

## 가우스 음의 로그가능도로서의 MSE

### 확률 모델

목표값이 정규분포를 따른다고 가정하자.

$$
y | x \sim \mathcal{N}(f_\theta(x), \sigma^2)
$$

여기서 $f_\theta(x)$은 모델의 예측(예: 신경망의 출력)이고 $\sigma^2$은 고정된 잡음 분산이다.

### 유도

관측 하나의 **가능도**는 다음과 같다.

$$
p(y | x, \theta) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(y - f_\theta(x))^2}{2\sigma^2}\right)
$$

**로그가능도**는 다음과 같다.

$$
\log p(y | x, \theta) = -\frac{1}{2}\log(2\pi\sigma^2) - \frac{(y - f_\theta(x))^2}{2\sigma^2}
$$

관측 $n$개에 대한 **음의 로그가능도**는 다음과 같다.

$$
-\sum_{i=1}^{n} \log p(y_i | x_i, \theta) = \frac{n}{2}\log(2\pi\sigma^2) + \frac{1}{2\sigma^2}\sum_{i=1}^{n}(y_i - f_\theta(x_i))^2
$$

$\sigma$이 고정되어 있으므로 첫째 항은 $\theta$에 대해 상수이다. 따라서 NLL을 최소화하는 것은 다음을 최소화하는 것과 같다.

$$
\boxed{\mathcal{L}_{\text{MSE}} = \frac{1}{n}\sum_{i=1}^{n}(y_i - f_\theta(x_i))^2}
$$

### MSE의 암묵적 가정

MSE 손실은 다음을 암묵적으로 가정한다.

- **등분산 가우스 잡음**: 모든 입력에 대해 분산 $\sigma^2$이 일정하다
- **독립인 오차**: 각 관측의 잡음이 서로 독립이다
- **평균 예측**: 모델이 **조건부 평균** $\mathbb{E}[Y|X]$을 예측한다

이 가정들이 깨지면(예: 이분산 데이터, 두꺼운 꼬리를 갖는 잡음, 이상치) MSE가 최선이 아닐 수 있다.

### 경사의 동등성

$\theta$에 대한 MSE의 경사와 가우스 NLL의 경사는 서로 비례한다.

$$
\nabla_\theta \mathcal{L}_{\text{MSE}} = \frac{2}{n}\sum_{i=1}^{n}(f_\theta(x_i) - y_i)\nabla_\theta f_\theta(x_i)
$$

$$
\nabla_\theta \mathcal{L}_{\text{NLL}} = \frac{1}{n\sigma^2}\sum_{i=1}^{n}(f_\theta(x_i) - y_i)\nabla_\theta f_\theta(x_i)
$$

둘은 상수 배 $2\sigma^2$만큼만 다르므로 최적화는 같은 궤적을 따른다.

## 라플라스 음의 로그가능도로서의 MAE

### 확률 모델

목표값이 라플라스 분포를 따른다고 가정하자.

$$
y | x \sim \text{Laplace}(f_\theta(x), b)
$$

**확률밀도함수**는 다음과 같다.

$$
p(y | x, \theta) = \frac{1}{2b}\exp\left(-\frac{|y - f_\theta(x)|}{b}\right)
$$

### 유도

**음의 로그가능도**는 다음과 같다.

$$
-\log p(y | x, \theta) = \log(2b) + \frac{|y - f_\theta(x)|}{b}
$$

$b$이 고정되어 있으면 NLL을 최소화하는 것이 **평균절대오차**를 준다.

$$
\boxed{\mathcal{L}_{\text{MAE}} = \frac{1}{n}\sum_{i=1}^{n}|y_i - f_\theta(x_i)|}
$$

### MAE 대 MSE: 분포의 관점

| 성질 | MSE | MAE |
|----------|-----|-----|
| 잡음 모델 | 가우스 | 라플라스 |
| 최적 예측 | 조건부 평균 $\mathbb{E}[Y\|X]$ | 조건부 중앙값 |
| 꼬리의 거동 | 가벼운 꼬리 | 더 두꺼운 꼬리 |
| 이상치에 대한 강건성 | 민감함 (제곱 벌점) | 강건함 (선형 벌점) |
| 0에서의 경사 | 매끄러움 | 미분 불가능 |

!!! note "강건성에 대한 직관"
    라플라스 분포는 가우스보다 꼬리가 두꺼워서 극단적인 관측에 더 높은 확률을 준다. 그래서 MAE는 자연스럽게 이상치에 더 너그럽다. 선형 벌점은 제곱처럼 큰 오차를 증폭하지 않는다.

## 이분산 회귀: 분산을 학습하기

### 동기

MSE를 쓰는 표준 회귀는 조건부 평균만 예측한다. 그러나 실제 데이터에는 흔히 **입력에 따라 달라지는 잡음**이 있다. 예를 들어 변동성이 큰 주식일수록 수익률을 예측하기 어렵다. 이분산 모델은 평균과 분산을 함께 예측한다.

### 모델

$$
y | x \sim \mathcal{N}(\mu_\theta(x), \sigma_\theta(x)^2)
$$

신경망은 출력 갈래를 둘 가진다. 하나는 $\mu_\theta(x)$을, 다른 하나는 (수치적 안정성을 위한 로그 분산) $\log \sigma_\theta(x)^2$을 낸다.

### 손실 함수

이분산 가우스 회귀의 **음의 로그가능도**는 다음과 같다.

$$
\boxed{\mathcal{L}_{\text{hetero}} = \frac{1}{n}\sum_{i=1}^{n}\left[\frac{1}{2}\log \sigma_\theta(x_i)^2 + \frac{(y_i - \mu_\theta(x_i))^2}{2\sigma_\theta(x_i)^2}\right]}
$$

첫째 항은 예측 분산이 커지는 것에 벌점을 준다(자명하게 $\sigma \to \infty$으로 두는 것을 막는다). 둘째 항은 제곱 오차를 예측 분산의 역수로 가중한다. 즉 모델이 확신하는 영역에서는 오차에 더 가혹한 벌점이 매겨진다.

!!! tip "실무적 이점"
    이분산 회귀는 다음을 자연스럽게 다룬다. **불확실성 추정**(예측 분산을 출력), **이분산 잡음**(입력마다 다른 분산), **확신을 아는 예측**(모델이 자신이 어디서 불확실한지 안다).

## PyTorch 구현

### MSE와 가우스 NLL의 동등성

```python
import torch
import torch.nn as nn
import numpy as np

def demonstrate_mse_nll_equivalence():
    """MSE와 가우스 NLL이 서로 비례하는 경사를 냄을 보여준다."""
    torch.manual_seed(42)
    
    n = 100
    x = torch.rand(n, 1) * 10
    true_w, true_b = 3.0, 2.0
    y = true_w * x + true_b + torch.randn(n, 1) * 1.0
    
    w = torch.tensor([1.0], requires_grad=True)
    b = torch.tensor([0.0], requires_grad=True)
    
    # MSE의 경사
    pred = w * x + b
    mse_loss = torch.mean((y - pred)**2)
    mse_loss.backward()
    mse_grad_w = w.grad.clone()
    w.grad.zero_(); b.grad.zero_()
    
    # 가우스 NLL의 경사 (sigma=1)
    sigma = 1.0
    pred = w * x + b
    nll_loss = torch.mean(0.5 * torch.log(torch.tensor(2 * np.pi * sigma**2)) + 
                          (y - pred)**2 / (2 * sigma**2))
    nll_loss.backward()
    nll_grad_w = w.grad.clone()
    
    print("MSE vs Gaussian NLL Equivalence")
    print(f"MSE gradient w.r.t w: {mse_grad_w.item():.6f}")
    print(f"NLL gradient w.r.t w: {nll_grad_w.item():.6f}")
    print(f"Ratio (should be 2σ² = 2): {mse_grad_w.item() / nll_grad_w.item():.4f}")
```

### MAE와 MSE의 강건성

```python
def mae_vs_mse_robustness():
    """이상치가 있을 때 MAE와 MSE의 강건성 차이를 보여준다."""
    torch.manual_seed(42)
    
    n = 100
    x = torch.linspace(0, 10, n).reshape(-1, 1)
    y_clean = 2 * x + 1 + torch.randn(n, 1) * 0.5
    
    # 큰 이상치를 섞는다
    y = y_clean.clone()
    outlier_idx = [10, 30, 50, 70, 90]
    y[outlier_idx] += 15
    
    # MSE로 학습한다
    model_mse = nn.Linear(1, 1)
    optimizer_mse = torch.optim.Adam(model_mse.parameters(), lr=0.1)
    for _ in range(1000):
        loss = nn.MSELoss()(model_mse(x), y)
        optimizer_mse.zero_grad(); loss.backward(); optimizer_mse.step()
    
    # MAE로 학습한다
    model_mae = nn.Linear(1, 1)
    optimizer_mae = torch.optim.Adam(model_mae.parameters(), lr=0.1)
    for _ in range(1000):
        loss = nn.L1Loss()(model_mae(x), y)
        optimizer_mae.zero_grad(); loss.backward(); optimizer_mae.step()
    
    print("MAE vs MSE with Outliers (true w=2.0, b=1.0)")
    print(f"MSE fit: w={model_mse.weight.item():.4f}, b={model_mse.bias.item():.4f}")
    print(f"MAE fit: w={model_mae.weight.item():.4f}, b={model_mae.bias.item():.4f}")
```

### 이분산 회귀

```python
class HeteroscedasticNet(nn.Module):
    """평균과 로그 분산을 함께 예측하는 신경망."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.mean_head = nn.Linear(hidden_dim, 1)
        self.log_var_head = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        features = self.shared(x)
        mean = self.mean_head(features)
        log_var = self.log_var_head(features)
        return mean, log_var

def heteroscedastic_nll(y: torch.Tensor, 
                        mean: torch.Tensor, 
                        log_var: torch.Tensor) -> torch.Tensor:
    """
    Negative log-likelihood for heteroscedastic Gaussian.
    
    NLL = 0.5 * [log(2π) + log_var + (y - mean)² / exp(log_var)]
    """
    return 0.5 * (np.log(2 * np.pi) + log_var + (y - mean)**2 / torch.exp(log_var))

def train_heteroscedastic():
    """잡음이 달라지는 데이터에서 이분산 모델을 학습시킨다."""
    torch.manual_seed(42)
    
    # 잡음이 점점 커지는 데이터를 생성한다
    n = 500
    x = torch.rand(n, 1) * 10
    noise_std = 0.5 + 0.3 * x  # Noise increases with x
    y = 2 * x + 1 + torch.randn(n, 1) * noise_std
    
    model = HeteroscedasticNet(1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    for epoch in range(1000):
        mean, log_var = model(x)
        loss = heteroscedastic_nll(y, mean, log_var).mean()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 200 == 0:
            print(f"Epoch {epoch+1}, NLL: {loss.item():.4f}")
    
    return model
```

### MLE의 관점에서 본 완전한 회귀 학습

```python
def train_regression_mle_perspective():
    """회귀 학습의 MLE 해석을 보여주는 완전한 예제."""
    torch.manual_seed(42)
    
    # 데이터를 생성한다
    n = 200
    X = torch.randn(n, 5)
    true_w = torch.tensor([1.0, -2.0, 0.5, 0.0, 1.5])
    true_b = 2.0
    sigma = 0.5
    y = X @ true_w + true_b + torch.randn(n) * sigma
    
    # 모델
    model = nn.Sequential(
        nn.Linear(5, 32),
        nn.ReLU(),
        nn.Linear(32, 1)
    )
    
    # MSE 손실 = 가우스 NLL (상수를 제외하면)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    print("Training Neural Network (MLE with Gaussian likelihood)")
    print("-" * 50)
    
    for epoch in range(500):
        pred = model(X).squeeze()
        loss = criterion(pred, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 100 == 0:
            with torch.no_grad():
                residuals = y - model(X).squeeze()
                estimated_sigma = residuals.std().item()
            
            print(f"Epoch {epoch+1}: MSE = {loss.item():.4f}, "
                  f"Est. σ = {estimated_sigma:.4f} (true: {sigma:.4f})")
```

## 참고 문헌

- Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Chapter 3
- Nix, D. A. & Weigend, A. S. (1994). "Estimating the mean and variance of the target probability distribution." *ICNN*
- Murphy, K. P. (2022). *Probabilistic Machine Learning: An Introduction*. Chapter 5

## 연습문제

**연습문제 1.**
후버 손실이 0 근처에서는 가우스, 꼬리에서는 라플라스인 분포 아래의 MLE에 대응함을 보여라. 밀도를 명시적으로 유도하라.

??? success "연습문제 1 풀이"
    후버 손실은 $L_\delta(r) = \begin{cases} \frac{1}{2}r^2 & |r| \leq \delta \\ \delta|r| - \frac{1}{2}\delta^2 & |r| > \delta \end{cases}$이다.

    대응하는 밀도는 $p(r) \propto e^{-L_\delta(r)}$이다.

    $$
    p(r) \propto \begin{cases} e^{-r^2/2} & |r| \leq \delta \\ e^{-\delta|r| + \delta^2/2} & |r| > \delta \end{cases}
    $$

    이는 $|r| \leq \delta$에서는 (지수가 이차이므로) 가우스이고 $|r| > \delta$에서는 (지수가 선형이므로) 라플라스이다. 두 조각은 $|r| = \delta$에서 연속으로 이어진다. 정규화하려면 경계에서 가우스 조각과 라플라스 조각을 맞춰 주어야 한다.

---

**연습문제 2.**
$y|x \sim \text{Poisson}(\exp(f_\theta(x)))$인 포아송 회귀의 손실 함수를 유도하라.

??? success "연습문제 2 풀이"
    포아송 확률질량함수는 $\lambda = \exp(f_\theta(x))$일 때 $P(y|\lambda) = \frac{\lambda^y e^{-\lambda}}{y!}$이다.

    NLL: $-\log P(y|x) = -y\log\lambda + \lambda + \log(y!) = -y \cdot f_\theta(x) + e^{f_\theta(x)} + \log(y!)$이다.

    상수 $\log(y!)$을 버리면 다음과 같다.

    $$
    \mathcal{L}(\theta) = \frac{1}{N}\sum_{i=1}^N \left[e^{f_\theta(x_i)} - y_i \cdot f_\theta(x_i)\right]
    $$

    ```python
    def poisson_nll(logits, targets):
        return (torch.exp(logits) - targets * logits).mean()
    ```

---

**연습문제 3.**
다변량 출력에 대해 완전한 공분산 행렬을 예측하도록 이분산 회귀를 확장하라. 예측된 공분산이 양의 정부호임을 어떤 매개화로 보장할 수 있는가?

??? success "연습문제 3 풀이"
    $d$차원 출력에 대해 $\Sigma = LL^\top$으로 매개화한다. 여기서 $L$은 대각 성분이 양수인 하삼각행렬(촐레스키 인자)이다. 신경망은 다음을 예측한다.

    - 평균 $\mu \in \mathbb{R}^d$: 출력 $d$개
    - 촐레스키 인자 $L$: 출력 $d(d+1)/2$개(하삼각 성분). 양수를 보장하기 위해 대각 성분은 softplus를 통과시킨다

    다변량 정규분포의 NLL은 다음과 같다.

    $$
    -\log p(y|x) = \frac{d}{2}\log(2\pi) + \sum_i \log L_{ii} + \frac{1}{2}\|L^{-1}(y-\mu)\|^2
    $$

    $\log|\Sigma| = 2\sum_i \log L_{ii}$이므로 로그 대각 성분의 합이 로그 행렬식이 된다.

---

**연습문제 4.**
이상치가 0%, 5%, 10%, 20%인 데이터에서 MSE, MAE, 후버 손실을 비교하는 모의실험을 구현하라. 각각에 대해 매개변수 복원 정확도를 보고하라.

??? success "연습문제 4 풀이"
    ```python
    import torch, torch.nn as nn

    results = {}
    for outlier_frac in [0.0, 0.05, 0.10, 0.20]:
        for loss_name, loss_fn in [('MSE', nn.MSELoss()),
                                    ('MAE', nn.L1Loss()),
                                    ('Huber', nn.HuberLoss(delta=1.0))]:
            torch.manual_seed(42)
            x = torch.randn(1000, 1)
            y = 3.0 * x + 1.0 + 0.1 * torch.randn_like(x)
            # 이상치를 섞는다
            mask = torch.rand(1000) < outlier_frac
            y[mask] += 20 * torch.randn(mask.sum(), 1)

            model = nn.Linear(1, 1)
            opt = torch.optim.Adam(model.parameters(), lr=0.01)
            for _ in range(2000):
                opt.zero_grad()
                loss_fn(model(x), y).backward()
                opt.step()
            w, b = model.weight.item(), model.bias.item()
            results[(outlier_frac, loss_name)] = (w, b)
    # 이상치가 있으면 MSE는 크게 나빠지고 MAE와 후버는 강건하다
    ```
