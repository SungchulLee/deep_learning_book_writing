# 확률적 해석

확률적 해석은 로스와 정규화 항을 선택하는 근거를 제공한다. 대표적인 로스를 조건부 분포의 음의 로그우도와 연결하고, 일부 패널티를 사전분포에 따른 MAP 추정으로 해석한다. 이런 대응은 분포의 정규화 가능성과 모델 가정을 확인한 범위에서 성립한다.

!!! abstract "핵심 명제"
    | 학습 목표 | 확률적 해석 |
    |:-------------------|:-----------------------------|
    | MSE 최소화 | 가우스 잡음 아래의 MLE |
    | 교차 엔트로피 최소화 | 범주형 우도 아래의 MLE |
    | L2 정규화 추가 | 가우스 사전분포 아래의 MAP |
    | L1 정규화 추가 | 라플라스 사전분포 아래의 MAP |
    | 드롭아웃 추가 | 특정 가정 아래 변분 베이즈 근사와 연결 |
    | 평균과 분산을 함께 예측 | 이분산 가우스 MLE |

---

## 1. 근본적인 관계

데이터 $\{(x_i, y_i)\}_{i=1}^n$과 확률 모델 $p(y|x, \theta)$이 주어졌을 때 다음이 성립한다.

**최대가능도**: $\hat{\theta} = \arg\max_\theta \prod_{i=1}^{n} p(y_i | x_i, \theta)$

**동등한 로스 최소화**: $\hat{\theta} = \arg\min_\theta \left[-\sum_{i=1}^{n} \log p(y_i | x_i, \theta)\right]$

**로스 함수**는 평균 음의 로그우도이다.

$$
\mathcal{L}(\theta) = -\frac{1}{n}\sum_{i=1}^{n} \log p(y_i | x_i, \theta)
$$

$p(y|x, \theta)$을 무엇으로 고르느냐에 따라 다른 로스 함수가 나온다. 거꾸로 로스에서 분포를 만들려면 지수화한 함수가 적분 가능해야 하고, 정규화 상수가 학습 파라미터에 의존하는지도 확인해야 한다. 임의의 로스를 그대로 NLL이라고 부를 수는 없다.

### 로스–분포 대응 전체 표

| 로스 함수 | 분포 | 모델 출력 | 쓰임새 |
|:--------------|:-------------|:-------------|:---------|
| MSE | $\mathcal{N}(\mu, \sigma^2)$ | 평균 $\mu$ | 회귀 |
| MAE | 라플라스$(m, b)$ | 중앙값 $m$ | 이상치에 강한 회귀 |
| BCE | 베르누이$(p)$ | 확률 $p$ | 이진 분류 |
| 교차 엔트로피 | 범주형$(\mathbf{p})$ | 확률 $\mathbf{p}$ | 다중 클래스 분류 |
| 포아송 로스 | 포아송$(\lambda)$ | 평균 발생 횟수 $\lambda$ | 카운트 데이터 |
| 후버 로스 | 중심이 이차, 꼬리가 선형인 음의 로그밀도 | 강건한 위치 파라미터 | 이상치에 강한 회귀 |

---

## 2. KL 발산과 로스 함수

### 교차 엔트로피와의 관계

교차 엔트로피는 다음과 같이 분해된다.

$$
H(p, q) = H(p) + D_{\text{KL}}(p \| q)
$$

여기서 $H(p, q) = -\mathbb{E}_p[\log q]$은 교차 엔트로피, $H(p) = -\mathbb{E}_p[\log p]$은 참 분포의 엔트로피, $D_{\text{KL}}(p \| q) = \mathbb{E}_p[\log \frac{p}{q}]$은 KL 발산이다.

$H(p)$은 모델 파라메터에 대해 상수이므로 다음이 성립한다.

$$
\text{Minimizing Cross-Entropy} \equiv \text{Minimizing KL Divergence}
$$

### MLE는 KL 발산을 최소화한다

MLE의 목적 함수는 다음과 같다.

$$
\hat{\theta} = \arg\min_\theta \left[-\frac{1}{n}\sum_{i=1}^n \log p(y_i | x_i, \theta)\right]
$$

이는 ($n \to \infty$일 때) 다음을 최소화하는 것으로 수렴한다.

$$
D_{\text{KL}}(p_{\text{data}} \| p_\theta) = \mathbb{E}_{p_{\text{data}}}\left[\log \frac{p_{\text{data}}}{p_\theta}\right]
$$

모형이 적절하고 경험적 목적이 모집단 기대로 수렴하는 조건에서는, MLE의 목표를 참 조건부 분포에서 모델로 향하는 KL 발산 최소화로 해석할 수 있다. 유한 표본의 수치 최적화 결과가 항상 이 모집단 최적해와 같지는 않다.

---

## 3. 사전분포로서의 정규화: MAP 추정

### MLE에서 MAP으로

최대 사후확률(MAP) 추정은 베이즈 정리를 통해 MLE에 사전분포 $p(\theta)$을 더한다.

$$
\hat{\theta}_{\text{MAP}} = \arg\max_\theta \, p(\theta | \text{data}) = \arg\max_\theta \left[\log p(\text{data} | \theta) + \log p(\theta)\right]
$$

첫째 항은 로그우도(데이터 적합)이고 둘째 항은 로그 사전분포(정규화)이다. MAP 추정은 빈도주의 MLE와 베이즈 추론을 잇는 다리이다.

### L2 정규화 = 가우스 사전분포

$\theta \sim \mathcal{N}(0, \tau^2 I)$이면 다음과 같다.

$$
\log p(\theta) = -\frac{\|\theta\|^2}{2\tau^2} + \text{const}
$$

MAP의 목적 함수는 다음이 된다.

$$
\mathcal{L}_{\text{MAP}} = \underbrace{-\log p(\text{data} | \theta)}_{\text{NLL (data fit)}} + \underbrace{\frac{\lambda}{2}\|\theta\|^2}_{\text{L2 penalty}}
$$

여기서는 NLL의 **합**을 사용하므로 $\lambda=1/\tau^2$이다. NLL을 $n$으로 나눈 평균에 같은 사전분포를 대응시키려면 패널티도 $n$으로 나누어야 한다. 로스의 합·평균 방식과 패널티 계수를 함께 정한다.

### L1 정규화 = 라플라스 사전분포

$\theta \sim \text{Laplace}(0, b)$이면 다음과 같다.

$$
\log p(\theta) = -\frac{\|\theta\|_1}{b} + \text{const}
$$

MAP의 목적 함수는 다음이 된다.

$$
\mathcal{L}_{\text{MAP}} = -\log p(\text{data} | \theta) + \lambda \|\theta\|_1
$$

라플라스 사전분포는 (매끄러운 가우스와 달리) 0에서 날카로운 봉우리를 가지며, 이것이 **희소성**을 북돋운다. 많은 파라메터가 정확히 0으로 밀려난다.

### 기하학적 직관

L1 사전분포(마름모 모양의 등고선)는 MAP 추정값을 일부 좌표가 0인 꼭짓점에 놓는 경향이 있다. L2 사전분포(원 모양의 등고선)는 모든 파라메터를 고르게 줄이지만 정확한 0을 만드는 일은 드물다. 이것이 피처 선택에 L1(라쏘)을 선호하는 이유이다.

---

## 4. 사용자 정의 로스 함수 설계하기

### 방법

특정 문제에 맞는 로스 함수를 설계하려면 다음과 같이 한다.

1. **분포 모델을 정한다**: $y|x$이 어떤 분포를 따르는가?
2. **로그우도를 쓴다**: $\log p(y|x, \theta)$
3. **음수를 취하고 평균한다**: $\mathcal{L} = -\frac{1}{n}\sum \log p(y_i | x_i, \theta)$
4. **정규화를 더한다** (선택): 사전분포 $p(\theta)$을 고른다

이 원리에 기반한 접근은 로스 함수가 이론적으로 근거를 갖게 하고 임시방편적인 선택을 피하게 해 준다.

### 예: 포아송 회귀

카운트 데이터 $y \in \{0, 1, 2, \ldots\}$에 대해 $y | x \sim \text{Poisson}(\exp(f_\theta(x)))$으로 모델링하면 다음과 같다.

$$
\log p(y|x, \theta) = y \cdot f_\theta(x) - \exp(f_\theta(x)) - \log(y!)
$$

로스는 (상수 $\log(y!)$을 버리면) 다음과 같다.

$$
\mathcal{L}_{\text{Poisson}} = \frac{1}{n}\sum_{i=1}^{n}\left[\exp(f_\theta(x_i)) - y_i \cdot f_\theta(x_i)\right]
$$

---

## 5. PyTorch 구현

### 사전분포로서의 정규화

```python
import torch
import torch.nn as nn
import numpy as np

def regularization_as_prior_demo():
    """L1/L2 정칙화가 베이즈 사전분포에 대응함을 보여준다."""
    torch.manual_seed(42)
    
    # 희소한 참 파라메터 (10개 중 3개만 0이 아니다)
    true_w = torch.tensor([3.0, 0.0, 0.0, -2.0, 0.0, 0.0, 0.0, 1.5, 0.0, 0.0])
    
    n = 50
    X = torch.randn(n, 10)
    y = X @ true_w + torch.randn(n) * 0.5
    
    results = {}
    
    for reg_type, reg_strength in [('none', 0), ('L2', 0.1), ('L1', 0.1)]:
        w = torch.randn(10, requires_grad=True)
        optimizer = torch.optim.Adam([w], lr=0.1)
        
        for _ in range(2000):
            pred = X @ w
            mse = torch.mean((y - pred)**2)
            
            if reg_type == 'L2':
                loss = mse + reg_strength * torch.sum(w**2)
            elif reg_type == 'L1':
                loss = mse + reg_strength * torch.sum(torch.abs(w))
            else:
                loss = mse
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        results[reg_type] = w.detach().clone()
    
    print("Regularization as Bayesian Prior")
    print("=" * 60)
    print(f"{'Index':>6} {'True':>10} {'No Reg':>10} {'L2 (Gauss)':>10} {'L1 (Lapl)':>10}")
    print("-" * 60)
    
    for i in range(10):
        print(f"{i:>6} {true_w[i]:>10.4f} {results['none'][i]:>10.4f} "
              f"{results['L2'][i]:>10.4f} {results['L1'][i]:>10.4f}")
    
    print("-" * 60)
    print("L1 encourages sparsity (zeros), L2 encourages small values")
```

### 사용자 정의 로스: 포아송 회귀

```python
def poisson_regression_demo():
    """계수 데이터에 대한 MLE으로서 포아송 회귀를 구현한다."""
    torch.manual_seed(42)
    
    # 계수 데이터를 생성한다
    n = 500
    X = torch.randn(n, 3)
    true_w = torch.tensor([0.5, -0.3, 0.8])
    log_rate = X @ true_w + 1.0
    y = torch.poisson(torch.exp(log_rate))
    
    # 모델
    model = nn.Linear(3, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    for epoch in range(1000):
        log_lambda = model(X).squeeze()
        # 포아송 NLL: exp(f(x)) - y * f(x)
        loss = torch.mean(torch.exp(log_lambda) - y * log_lambda)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 200 == 0:
            print(f"Epoch {epoch+1}: Poisson NLL = {loss.item():.4f}")
    
    print(f"\nTrue weights:      {true_w.tolist()}")
    print(f"Estimated weights: {model.weight.data.squeeze().tolist()}")
```

### 합성 데이터에서 로스 함수 비교하기

```python
def loss_function_comparison():
    """
    튄값이 있는 회귀에서 MSE, MAE, 후버 로스를 견준다.
    로스를 고르는 것이 곧 분포를 가정하는 것임을 보인다.
    """
    torch.manual_seed(42)
    
    n = 200
    x = torch.rand(n, 1) * 10
    y_clean = 2 * x + 1 + torch.randn(n, 1) * 0.5
    
    # 이상치를 10% 섞는다
    n_outliers = n // 10
    outlier_idx = torch.randperm(n)[:n_outliers]
    y = y_clean.clone()
    y[outlier_idx] += torch.randn(n_outliers, 1) * 10
    
    losses = {
        'MSE (Gaussian)': nn.MSELoss(),
        'MAE (Laplace)': nn.L1Loss(),
        'Huber (Mixture)': nn.HuberLoss(delta=1.0),
    }
    
    print("Loss Function Comparison (true w=2.0, b=1.0, 10% outliers)")
    print("-" * 55)
    
    for name, criterion in losses.items():
        model = nn.Linear(1, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        
        for _ in range(2000):
            loss = criterion(model(x), y)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        
        w = model.weight.item()
        b = model.bias.item()
        print(f"{name:>20}: w={w:.4f}, b={b:.4f}")
```

---

## 6. 핵심 정리

딥러닝의 확률적 해석은 여러 실용적인 이점을 준다.

1. **로스 함수 선택**: 어림짐작하는 대신 "내 목표값이 어떤 분포를 따르는가?"를 물어 로스를 고른다.

2. **정규화 설계**: "파라메터에 대해 어떤 사전 믿음을 가지고 있는가?"를 물어 정규화를 고른다.

3. **불확실성의 정량화**: 확률 모델은 점 예측뿐 아니라 분포로서의 예측(평균, 분산, 신뢰구간)을 준다.

4. **사용자 정의 로스 설계**: 새로운 문제라도 분포 모델을 정하고 그에 대응하는 NLL을 유도하면 다룰 수 있다.

5. **원리에 기반한 모델 비교**: 따로 떼어 둔 데이터에서의 우도로 서로 다른 모델을 비교할 수 있다.

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff hard" title="어려움"></span>
음이항분포를 사용하여 과대산포가 있는 카운트를 예측하는 모델의 로스 함수를 유도하라.

</div>

??? success "풀이"
    음이항분포의 확률질량함수는 $P(y|r,p) = \binom{y+r-1}{y} p^r (1-p)^y$이다.

    평균 $\mu = r(1-p)/p$과 산포 $r$으로 다시 매개화한다. NLL은 다음과 같다.

    $$
    -\log P(y|\mu,r) = -\log\Gamma(y+r) + \log\Gamma(r) + \log(y!) - r\log\frac{r}{r+\mu} - y\log\frac{\mu}{r+\mu}
    $$

    $r \to \infty$일 때 음이항분포는 포아송으로 수렴하며 표준 포아송 회귀 로스를 되찾는다. 산포 파라메터 $r$이 과대산포를 담아낸다($\text{Var}(y) = \mu + \mu^2/r > \mu$).

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
드롭아웃 학습이 변분 추론을 근사함을 보이고, 그것이 어떤 사후분포를 근사하는지 밝혀라.

</div>

??? success "풀이"
    Gal과 Ghahramani(2016)는 특정 사전분포, 변분족과 근사를 사용해 드롭아웃 학습을 변분 추론과 연결한다. 이는 임의의 드롭아웃 구현이 정확한 사후추론이라는 뜻은 아니다. 다음 식은 근사 웨이트를 표본화하는 구성을 나타낸다.

    $$
    q(\mathbf{W}_l) = \mathbf{M}_l \cdot \text{diag}(\mathbf{z}_l), \quad z_{l,i} \sim \text{Bernoulli}(1-p)
    $$

    여기서 $\mathbf{M}_l$은 학습할 웨이트다. 적절한 스케일과 L2 계수 아래에서 드롭아웃 목적을 변분 하한의 근사와 연결한다. 추론 시 드롭아웃을 켜고 여러 번 예측하는 MC 드롭아웃은 이 근사 모델의 예측 변동을 살펴보는 방법이며, 불확실성의 정확한 보정을 보장하지는 않는다. [원 논문](https://proceedings.mlr.press/v48/gal16.html)에서 가정을 확인할 수 있다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
폰 미제스 분포를 사용하여 원형 데이터(예: 바람의 방향)를 예측하는 사용자 정의 로스를 설계하라.

</div>

??? success "풀이"
    폰 미제스 밀도는 각 $\theta \in [-\pi, \pi]$에 대해 $p(\theta|\mu,\kappa) = \frac{e^{\kappa\cos(\theta-\mu)}}{2\pi I_0(\kappa)}$이다.

    NLL은 다음과 같다.

    $$
    -\log p(\theta|\mu,\kappa) = -\kappa\cos(\theta - \mu) + \log(2\pi I_0(\kappa))
    $$

    뉴럴 넷트웍은 $\mu$(평균 방향)과 $\kappa > 0$(집중도)을 예측한다. 양수를 보장하려면 $\kappa = \text{softplus}(\hat{\kappa})$을 쓴다.

    ```python
    def von_mises_nll(pred_mu, pred_kappa, target_angle):
        return (-pred_kappa * torch.cos(target_angle - pred_mu)
                + torch.log(2 * 3.14159 * torch.i0(pred_kappa))).mean()
    ```

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
피처가 $p = 100$개, 관측이 $n = 50$개이고 참으로 0이 아닌 계수가 5개인 희소 회귀 문제에서 MLE, L2를 쓴 MAP, L1을 쓴 MAP을 비교하라.

</div>

??? success "풀이"
    ```python
    import torch, torch.nn as nn

    torch.manual_seed(0)
    w_true = torch.zeros(100); w_true[:5] = torch.randn(5)
    X = torch.randn(50, 100); y = X @ w_true + 0.1 * torch.randn(50)

    for name, lam_l1, lam_l2 in [('MLE', 0, 0), ('MAP-L2', 0, 0.1), ('MAP-L1', 0.01, 0)]:
        w = torch.zeros(100, requires_grad=True)
        opt = torch.optim.Adam([w], lr=0.01)
        for _ in range(5000):
            opt.zero_grad()
            loss = ((X @ w - y)**2).mean() + lam_l2 * (w**2).sum() + lam_l1 * w.abs().sum()
            loss.backward(); opt.step()
        sparsity = (w.abs() < 0.01).float().mean()
        mse = ((w - w_true)**2).mean()
        print(f"{name}: MSE={mse:.4f}, Sparsity={sparsity:.2%}")
    # MAP-L1이 희소성 양상을 가장 잘 되찾는다. MAP-L2는 모든 계수를 줄이고,
    # MLE는 p > n에서 과적합한다.
    ```

## 정리하며

확률적 해석을 적용할 때는 분포의 정규화, 로스의 합·평균, 사전분포의 스케일을 함께 확인한다. MAP는 사후분포 전체가 아니라 한 점을 추정하므로 그 자체로 모든 예측 불확실성을 설명하지는 않는다.

**참고 문헌**

- Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Chapter 4
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. Chapter 5
- Kendall, A. & Gal, Y. (2018). "What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?" *NeurIPS*
- Murphy, K. P. (2022). *Probabilistic Machine Learning: An Introduction*. Chapter 5
