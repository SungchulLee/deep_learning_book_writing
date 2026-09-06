# 깜깜이 변분 추론
## 학습 목표

이 절을 마치면 다음을 할 수 있게 된다.

1. 기울기를 쓰는 변분 추론 방법이 필요한 까닭 이해하기
2. 점수 함수 기울기 어림꼴(REINFORCE) 이끌어 내기
3. 기울기 어림의 흩어짐을 줄이는 기법 구현하기
4. 이어진 변수에 매개변수 바꾸기 재주 쓰기
5. 미분할 수 있는 아무 모형에나 통하는 깜깜이 변분 추론 알고리즘 세우기

## 왜 필요한가: 켤레 모형을 넘어

전통 CAVI에는 다음이 필요하다:

1. **모형마다 따로 이끌어 내기**: 새 모형마다 맞춤 새로 고침 식이 필요하다
2. **켤레 앞확률**: 닫힌 꼴 새로 고침은 켤레성에 기댄다
3. **손으로 셈하는 기댓값**: 기댓값 충분 통계량을 셈할 수 있어야 한다

**깜깜이 변분 추론**은 몬테카를로 어림을 붙인 기울기 최적화를 써서 이 옭아맴을 없애고, 미분할 수 있는 아무 확률 모형에나 변분 추론을 쓸 수 있게 한다.

## 기울기의 어려움

ELBO을 가장 크게 하고자 한다:

$$
\text{ELBO}(q_\phi) = \mathbb{E}_{q_\phi(\theta)}[\log p(\mathcal{D}, \theta) - \log q_\phi(\theta)]
$$

여기서 $\phi$은 변분 매개변수이다(이를테면 가우스 $q$의 평균과 흩어짐).

어려움은 기울기를 셈하는 데 있다:

$$
\nabla_\phi \text{ELBO}(q_\phi) = \nabla_\phi \mathbb{E}_{q_\phi(\theta)}[f(\theta)]
$$

여기서 $f(\theta) = \log p(\mathcal{D}, \theta) - \log q_\phi(\theta)$이다.

**말썽**: 기댓값이 $q_\phi$에 대한 것인데 $q_\phi$이 $\phi$에 달려 있다!

## 점수 함수 기울기 어림꼴(REINFORCE)

### 유도

**로그 미분 재주**를 쓰면:

$$
\nabla_\phi q_\phi(\theta) = q_\phi(\theta) \nabla_\phi \log q_\phi(\theta)
$$

기울기를 다시 쓸 수 있다:

$$
\begin{aligned}
\nabla_\phi \mathbb{E}_{q_\phi}[f(\theta)] &= \nabla_\phi \int q_\phi(\theta) f(\theta) \, d\theta \\
&= \int \nabla_\phi q_\phi(\theta) f(\theta) \, d\theta \\
&= \int q_\phi(\theta) \nabla_\phi \log q_\phi(\theta) f(\theta) \, d\theta \\
&= \mathbb{E}_{q_\phi}\left[f(\theta) \nabla_\phi \log q_\phi(\theta)\right]
\end{aligned}
$$

### 점수 함수 어림꼴

**점수 함수**는 $s_\phi(\theta) = \nabla_\phi \log q_\phi(\theta)$이다.

기울기 어림꼴은 다음이 된다:

$$
\boxed{\nabla_\phi \text{ELBO} \approx \frac{1}{S} \sum_{s=1}^S f(\theta^{(s)}) \nabla_\phi \log q_\phi(\theta^{(s)})}
$$

여기서 $\theta^{(s)} \sim q_\phi(\theta)$이다.

이는 (강화 학습에서 온) **REINFORCE** 어림꼴이라고도 한다.

### 성질

**좋은 점**:

- (띄엄띄엄하든 이어졌든) **아무** 분포 $q_\phi$에나 통한다
- $q_\phi$에서 뽑은 표본과 $\nabla_\phi \log q_\phi$의 값을 매길 수 있는 것만 있으면 된다
- 모형마다 따로 이끌어 낼 필요가 없다

**나쁜 점**:

- **흩어짐이 크다**: 정확한 기울기에 표본이 많이 필요할 수 있다
- **느리게 모인다**: 흩어짐이 크면 최적화가 시끄러워진다

## 흩어짐 줄이는 기법

점수 함수 어림꼴은 흩어짐이 크다. 여러 기법이 도움이 된다.

### 다스림 변량

**다스림 변량**은 기댓값을 아는 함수 $c(\theta)$이다. 어림꼴을 다음처럼 고친다:

$$
\nabla_\phi \text{ELBO} \approx \frac{1}{S} \sum_{s=1}^S \left(f(\theta^{(s)}) - c(\theta^{(s)})\right) \nabla_\phi \log q_\phi(\theta^{(s)}) + \mathbb{E}[c(\theta)] \mathbb{E}[\nabla_\phi \log q_\phi]
$$

$\mathbb{E}_{q_\phi}[\nabla_\phi \log q_\phi(\theta)] = 0$이므로 다음을 쓸 수 있다:

$$
\nabla_\phi \text{ELBO} \approx \frac{1}{S} \sum_{s=1}^S \left(f(\theta^{(s)}) - c\right) \nabla_\phi \log q_\phi(\theta^{(s)})
$$

여기서 $c$은 아무 상수이며 **기준선**이라 한다.

### 가장 좋은 기준선

가장 좋은 상수 기준선은 흩어짐을 가장 작게 한다:

$$
c^* = \frac{\mathbb{E}[f(\theta) \|\nabla_\phi \log q_\phi(\theta)\|^2]}{\mathbb{E}[\|\nabla_\phi \log q_\phi(\theta)\|^2]}
$$

실전에서는 달리는 평균으로 이를 어림한다.

### 라오-블랙웰화

할 수 있으면 어떤 변수를 손으로 적분해 없애 흩어짐을 줄인다:

$$
\mathbb{E}_{q(\theta_1, \theta_2)}[f(\theta_1, \theta_2)] = \mathbb{E}_{q(\theta_1)}[\mathbb{E}_{q(\theta_2|\theta_1)}[f(\theta_1, \theta_2)]]
$$

안쪽 기댓값을 손으로 셈할 수 있으면 흩어짐이 줄어든다.

## 매개변수 바꾸기 재주

이어진 분포에서는 **매개변수 바꾸기 재주**가 점수 함수 어림꼴보다 흩어짐이 작은 길을 준다.

### 핵심 생각

$\theta \sim q_\phi(\theta)$을 곧바로 표집하는 대신 다음을 한다:

1. **바탕 분포**에서 표집한다. 곧 $\epsilon \sim p(\epsilon)$이며 $\phi$과 무관하다
2. **정해진 바꿈**을 씌운다. 곧 $\theta = g_\phi(\epsilon)$이다

이러면 확률 요소가 $\epsilon$으로 옮겨 가 기울기가 $g_\phi$을 지나 흐르게 된다.

### 가우스 q에서

$\phi = (\mu, \sigma)$이고 $q_\phi(\theta) = \mathcal{N}(\mu, \sigma^2)$이면:

$$
\theta = \mu + \sigma \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, 1)
$$

기울기는 다음이 된다:

$$
\nabla_\phi \mathbb{E}_{q_\phi}[f(\theta)] = \mathbb{E}_{p(\epsilon)}[\nabla_\phi f(g_\phi(\epsilon))]
$$

이는 다음으로 어림할 수 있다:

$$
\nabla_\phi \mathbb{E}_{q_\phi}[f(\theta)] \approx \frac{1}{S} \sum_{s=1}^S \nabla_\phi f(\mu + \sigma \cdot \epsilon^{(s)})
$$

### 일반 매개변수 바꾸기

누적분포함수가 $F_\phi$인 분포 $q_\phi$에서:

$$
\theta = F_\phi^{-1}(u), \quad u \sim \text{Uniform}(0, 1)
$$

흔한 매개변수 바꾸기:

| 분포 | 바탕 | 바꿈 |
|--------------|------|----------------|
| $\mathcal{N}(\mu, \sigma^2)$ | $\epsilon \sim \mathcal{N}(0,1)$ | $\theta = \mu + \sigma\epsilon$ |
| $\text{LogNormal}(\mu, \sigma^2)$ | $\epsilon \sim \mathcal{N}(0,1)$ | $\theta = \exp(\mu + \sigma\epsilon)$ |
| $\text{Gamma}(\alpha, \beta)$ | $\epsilon \sim \text{Gamma}(\alpha, 1)$ | $\theta = \epsilon / \beta$ |
| $\text{Beta}(\alpha, \beta)$ | $u \sim \text{Uniform}(0,1)$ | 거꿀 누적분포함수(수치로) |

### 매개변수 바꾸기의 좋은 점

- 점수 함수 어림꼴보다 **흩어짐이 훨씬 작다**
- 최적화에서 **더 빨리 모인다**
- **저절로 미분하기와 잘 맞는다**(그 바꿈을 거쳐 되짚기만 하면 된다)

### 한계

- **이어진** 분포에만 통한다
- **미분할 수 있는** 바꿈이 필요하다
- 모든 분포에 쉬운 매개변수 바꾸기가 있는 것은 아니다

## PyTorch 구현

```python
import torch
import torch.nn as nn
import torch.distributions as dist
from typing import Callable, Dict, Tuple
import matplotlib.pyplot as plt
import numpy as np

class ScoreFunctionVI:
    """
    점수 함수(REINFORCE) 기울기 어림꼴을 쓴 깜깜이 변분 추론.
    
    다음을 할 수 있는 아무 분포 q_φ에나 통한다:
    1. θ ~ q_φ 표집
    2. ∇_φ log q_φ(θ) 셈하기
    """
    
    def __init__(self, log_joint: Callable, dim: int):
        """
        인수:
            log_joint: log p(D, θ)을 셈하는 함수
            dim: θ의 차원
        """
        self.log_joint = log_joint
        self.dim = dim
        
        # 가우스 q(θ) = N(μ, diag(σ²))의 변분 매개변수
        self.mu = nn.Parameter(torch.zeros(dim))
        self.log_sigma = nn.Parameter(torch.zeros(dim))
    
    @property
    def sigma(self):
        return torch.exp(self.log_sigma)
    
    def sample(self, n_samples: int) -> torch.Tensor:
        """q_φ(θ)에서 표집하기."""
        q = dist.Normal(self.mu, self.sigma)
        return q.rsample((n_samples,))
    
    def log_q(self, theta: torch.Tensor) -> torch.Tensor:
        """log q_φ(θ) 셈하기."""
        q = dist.Normal(self.mu, self.sigma)
        return q.log_prob(theta).sum(dim=-1)
    
    def score_function_gradient(self, n_samples: int = 100,
                                baseline: float = 0.0) -> Tuple[torch.Tensor, float]:
        """
        기준선을 붙인 점수 함수 어림꼴로 기울기 셈하기.
        
        ∇_φ ELBO ≈ (1/S) Σ (f(θ⁽ˢ⁾) - b) ∇_φ log q_φ(θ⁽ˢ⁾)
        
        여기서 f(θ) = log p(D,θ) - log q_φ(θ)이다
        """
        # q에서 표집
        theta = self.sample(n_samples)  # [S, dim]
        
        # f(θ) = log p(D,θ) - log q(θ) 셈하기
        log_p = self.log_joint(theta)           # [S]
        log_q = self.log_q(theta)               # [S]
        f = log_p - log_q                       # [S]
        
        # ELBO 어림값
        elbo = f.mean().item()
        
        # 점수 함수 기울기
        # 표본마다 ∇_φ log q_φ(θ)을 셈해야 한다
        # 가우스에서: ∇_μ log q = (θ - μ)/σ², ∇_log_σ log q = ((θ-μ)²/σ² - 1)
        
        # 경사 초기화
        if self.mu.grad is not None:
            self.mu.grad.zero_()
        if self.log_sigma.grad is not None:
            self.log_sigma.grad.zero_()
        
        # 표본마다 기울기 셈하기(효율은 낮지만 또렷하다)
        grad_mu = torch.zeros_like(self.mu)
        grad_log_sigma = torch.zeros_like(self.log_sigma)
        
        for s in range(n_samples):
            # 점수 함수의 성분
            score_mu = (theta[s] - self.mu) / self.sigma**2
            score_log_sigma = ((theta[s] - self.mu)**2 / self.sigma**2 - 1)
            
            # (f - 기준선)으로 무게를 두어 쌓기
            weight = f[s].item() - baseline
            grad_mu += weight * score_mu
            grad_log_sigma += weight * score_log_sigma
        
        grad_mu /= n_samples
        grad_log_sigma /= n_samples
        
        return (grad_mu, grad_log_sigma), elbo
    
    def fit(self, n_iterations: int = 1000, n_samples: int = 100,
            lr: float = 0.01, use_baseline: bool = True,
            verbose: bool = True) -> Dict:
        """
        점수 함수 기울기로 최적화하기.
        """
        optimizer = torch.optim.Adam([self.mu, self.log_sigma], lr=lr)
        
        history = {'elbo': [], 'mu': [], 'sigma': []}
        baseline = 0.0
        
        for i in range(n_iterations):
            optimizer.zero_grad()
            
            # 기울기 셈하기
            (grad_mu, grad_log_sigma), elbo = self.score_function_gradient(
                n_samples, baseline if use_baseline else 0.0
            )
            
            # 기준선 새로 고치기(f의 달리는 평균)
            if use_baseline:
                baseline = 0.9 * baseline + 0.1 * elbo
            
            # 기울기 정하기(최적화기가 최소화하므로 음수)
            self.mu.grad = -grad_mu
            self.log_sigma.grad = -grad_log_sigma
            
            optimizer.step()
            
            history['elbo'].append(elbo)
            history['mu'].append(self.mu.clone().detach())
            history['sigma'].append(self.sigma.clone().detach())
            
            if verbose and (i + 1) % 100 == 0:
                print(f"Iter {i+1:4d}: ELBO = {elbo:.4f}, "
                      f"μ = {self.mu.detach().numpy()}, "
                      f"σ = {self.sigma.detach().numpy()}")
        
        return history


class ReparameterizedVI:
    """
    매개변수 바꾸기 재주를 쓴 깜깜이 변분 추론.
    
    이어진 q에서는 점수 함수 어림꼴보다 흩어짐이 훨씬 작다.
    """
    
    def __init__(self, log_joint: Callable, dim: int):
        """
        인수:
            log_joint: log p(D, θ)을 셈하는 함수
            dim: θ의 차원
        """
        self.log_joint = log_joint
        self.dim = dim
        
        # 변분 매개변수
        self.mu = nn.Parameter(torch.zeros(dim))
        self.log_sigma = nn.Parameter(torch.zeros(dim))
    
    @property
    def sigma(self):
        return torch.exp(self.log_sigma)
    
    def elbo(self, n_samples: int = 100) -> torch.Tensor:
        """
        매개변수 바꾸기로 ELBO 셈하기.
        
        θ = μ + σ ⊙ ε,  ε ~ N(0, I)
        
        ELBO = E_ε[log p(D, μ + σε) - log q(μ + σε)]
        """
        # 바탕 분포에서 표집
        epsilon = torch.randn(n_samples, self.dim)
        
        # 매개변수 바꾸기
        theta = self.mu + self.sigma * epsilon
        
        # 로그 결합 확률 셈하기
        log_p = self.log_joint(theta)
        
        # log q 셈하기(가우스에서는 손으로 구함)
        log_q = dist.Normal(self.mu, self.sigma).log_prob(theta).sum(dim=-1)
        
        # ELBO
        return (log_p - log_q).mean()
    
    def fit(self, n_iterations: int = 1000, n_samples: int = 10,
            lr: float = 0.01, verbose: bool = True) -> Dict:
        """
        매개변수 바꾸기 기울기로 최적화하기.
        
        메모: 점수 함수보다 표본이 훨씬 적게 든다!
        """
        optimizer = torch.optim.Adam([self.mu, self.log_sigma], lr=lr)
        
        history = {'elbo': [], 'mu': [], 'sigma': []}
        
        for i in range(n_iterations):
            optimizer.zero_grad()
            
            # ELBO 셈하기(기울기가 매개변수 바꾸기를 지나 흐른다)
            elbo = self.elbo(n_samples)
            
            # 기울기 오르기(ELBO의 음수를 가장 작게 하기)
            loss = -elbo
            loss.backward()
            optimizer.step()
            
            history['elbo'].append(elbo.item())
            history['mu'].append(self.mu.clone().detach())
            history['sigma'].append(self.sigma.clone().detach())
            
            if verbose and (i + 1) % 100 == 0:
                print(f"Iter {i+1:4d}: ELBO = {elbo.item():.4f}, "
                      f"μ = {self.mu.detach().numpy()}, "
                      f"σ = {self.sigma.detach().numpy()}")
        
        return history


def compare_estimators():
    """점수 함수 어림꼴과 매개변수 바꾸기 어림꼴 견주기."""
    
    # 단순한 과녁: 가우스 평균의 뒤확률
    # 앞확률: θ ~ N(0, 1)
    # 가능도: x_i ~ N(θ, 1), 관측 평균 = 2.0
    
    n_data = 50
    data_mean = 2.0
    data_var = 1.0
    
    # 정확한 뒤확률: N(μ_n, σ_n²)
    posterior_precision = 1 + n_data / data_var
    exact_mean = n_data * data_mean / data_var / posterior_precision
    exact_std = 1 / np.sqrt(posterior_precision)
    
    print(f"Exact posterior: N({exact_mean:.4f}, {exact_std:.4f}²)")
    
    def log_joint(theta):
        """Log p(D, θ) = log p(D|θ) + log p(θ)"""
        # 앞확률
        log_prior = dist.Normal(0, 1).log_prob(theta).sum(dim=-1)
        
        # 가능도(충분 통계량 사용)
        log_lik = -0.5 * n_data * ((theta.squeeze() - data_mean)**2 + data_var)
        
        return log_prior + log_lik
    
    # 방법 견주기
    results = {}
    
    # 점수 함수(표본이 많이 든다)
    print("\n--- Score Function Estimator ---")
    vi_score = ScoreFunctionVI(log_joint, dim=1)
    results['score'] = vi_score.fit(n_iterations=500, n_samples=100, 
                                    lr=0.01, verbose=True)
    
    # 매개변수 바꾸기(표본이 적게 든다)
    print("\n--- Reparameterization Trick ---")
    vi_reparam = ReparameterizedVI(log_joint, dim=1)
    results['reparam'] = vi_reparam.fit(n_iterations=500, n_samples=10,
                                        lr=0.01, verbose=True)
    
    # 견줌을 그려 본다
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # ELBO 견주기
    ax = axes[0, 0]
    ax.plot(results['score']['elbo'], alpha=0.7, label='Score Function')
    ax.plot(results['reparam']['elbo'], alpha=0.7, label='Reparameterization')
    ax.set_xlabel('Iteration', fontsize=11)
    ax.set_ylabel('ELBO', fontsize=11)
    ax.set_title('(a) ELBO Convergence', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 평균의 모임
    ax = axes[0, 1]
    mu_score = torch.stack(results['score']['mu']).squeeze().numpy()
    mu_reparam = torch.stack(results['reparam']['mu']).squeeze().numpy()
    ax.plot(mu_score, alpha=0.7, label='Score Function')
    ax.plot(mu_reparam, alpha=0.7, label='Reparameterization')
    ax.axhline(exact_mean, color='red', linestyle='--', label='Exact')
    ax.set_xlabel('Iteration', fontsize=11)
    ax.set_ylabel('Mean', fontsize=11)
    ax.set_title('(b) Mean Convergence', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # ELBO 어림값의 흩어짐
    ax = axes[1, 0]
    window = 50
    var_score = np.array([np.var(results['score']['elbo'][max(0,i-window):i+1]) 
                          for i in range(len(results['score']['elbo']))])
    var_reparam = np.array([np.var(results['reparam']['elbo'][max(0,i-window):i+1])
                            for i in range(len(results['reparam']['elbo']))])
    ax.plot(var_score, alpha=0.7, label='Score Function')
    ax.plot(var_reparam, alpha=0.7, label='Reparameterization')
    ax.set_xlabel('Iteration', fontsize=11)
    ax.set_ylabel(f'ELBO Variance (window={window})', fontsize=11)
    ax.set_title('(c) Gradient Variance', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # 마지막 뒤확률
    ax = axes[1, 1]
    theta_range = torch.linspace(exact_mean - 3*exact_std, 
                                  exact_mean + 3*exact_std, 200)
    
    # 정확
    exact_pdf = dist.Normal(exact_mean, exact_std).log_prob(theta_range).exp()
    ax.plot(theta_range.numpy(), exact_pdf.numpy(), 'k-', linewidth=2.5, 
            label='Exact')
    
    # 점수 함수 결과
    mu_s = results['score']['mu'][-1].item()
    sigma_s = results['score']['sigma'][-1].item()
    score_pdf = dist.Normal(mu_s, sigma_s).log_prob(theta_range).exp()
    ax.plot(theta_range.numpy(), score_pdf.numpy(), '--', linewidth=2,
            label=f'Score (μ={mu_s:.3f}, σ={sigma_s:.3f})')
    
    # 매개변수 바꾸기 결과
    mu_r = results['reparam']['mu'][-1].item()
    sigma_r = results['reparam']['sigma'][-1].item()
    reparam_pdf = dist.Normal(mu_r, sigma_r).log_prob(theta_range).exp()
    ax.plot(theta_range.numpy(), reparam_pdf.numpy(), ':', linewidth=2,
            label=f'Reparam (μ={mu_r:.3f}, σ={sigma_r:.3f})')
    
    ax.set_xlabel('θ', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('(d) Final Posteriors', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('bbvi_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    
    compare_estimators()
```

## 저절로 미분하는 변분 추론(ADVI)

ADVI은 깜깜이 변분 추론의 실전 구현으로 다음을 한다:

1. 제약이 있는 매개변수를 제약 없는 공간으로 바꾼다
2. 대각 가우스 변분 집안을 쓴다
3. 매개변수 바꾸기 재주를 쓴다
4. 미분할 수 있는 아무 모형에나 저절로 통한다

### 매개변수 바꿈

| 제약 | 바꿈 | 역 |
|------------|---------------|---------|
| $\theta > 0$ | $\zeta = \log \theta$ | $\theta = \exp(\zeta)$ |
| $0 < \theta < 1$ | $\zeta = \text{logit}(\theta)$ | $\theta = \text{sigmoid}(\zeta)$ |
| $\theta \in \mathbb{R}^d$, $\|\theta\| = 1$ | 구면 좌표 | 단위 벡터 |
| 단체 | 막대 꺾기 | 디리클레 |

### 야코비 바로잡기

$\theta = h(\zeta)$으로 바꿀 때:

$$
\log p(\theta) = \log p(h(\zeta)) + \log \left|\det \frac{\partial h}{\partial \zeta}\right|
$$

## 요약

**깜깜이 변분 추론**은 미분할 수 있는 아무 모형에나 변분 추론을 쓸 수 있게 한다:

**점수 함수 어림꼴**:

$$
\nabla_\phi \text{ELBO} \approx \frac{1}{S} \sum_s f(\theta^{(s)}) \nabla_\phi \log q_\phi(\theta^{(s)})
$$

- (띄엄띄엄하든 이어졌든) 아무 $q$에나 통한다
- 흩어짐이 커서 표본이 많이 필요하다
- 흩어짐을 줄이려면 다스림 변량이나 기준선을 쓴다

**매개변수 바꾸기 재주**:

$$
\theta = g_\phi(\epsilon), \quad \epsilon \sim p(\epsilon)
$$

- 흩어짐이 훨씬 작다
- 이어져 있고 매개변수를 바꿀 수 있는 분포에만 쓴다
- 저절로 미분하기를 쓸 수 있게 한다

## 참고 문헌

1. Ranganath, R., Gerrish, S., & Blei, D. M. (2014). "Black Box Variational Inference."

2. Kingma, D. P., & Welling, M. (2014). "Auto-Encoding Variational Bayes."

3. Rezende, D. J., Mohamed, S., & Wierstra, D. (2014). "Stochastic Backpropagation and Approximate Inference in Deep Generative Models."

4. Kucukelbir, A., Tran, D., Ranganath, R., Gelman, A., & Blei, D. M. (2017). "Automatic Differentiation Variational Inference."

5. Mohamed, S., Rosca, M., Figurnov, M., & Mnih, A. (2020). "Monte Carlo Gradient Estimation in Machine Learning."

## 연습문제

### 연습 1: 흩어짐 견주기

표본의 개수를 달리하며 점수 함수 기울기 어림값과 매개변수 바꾸기 기울기 어림값의 흩어짐을 경험으로 견주어라.

### 연습 2: 다스림 변량 짜기

점수 함수 어림꼴에 쓰는 여러 다스림 변량 전략을 구현하고 견주어라.

### 연습 3: 가우스가 아닌 변분 집안

가우스 섞음 변분 집안으로 깜깜이 변분 추론을 구현하여라.
