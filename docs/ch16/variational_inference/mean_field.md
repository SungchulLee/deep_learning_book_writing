# 평균장 변분 추론
## 학습 목표

이 절을 마치면 다음을 할 수 있게 된다.

1. 평균장 가정과 그것이 뜻하는 바 이해하기
2. 변분 인수마다 가장 좋은 꼴 이끌어 내기
3. 평균장 어림이 알맞은 때 알아보기
4. 뒤확률의 상관을 무시할 때의 한계 재기
5. 다변량 모형에 평균장 변분 추론 구현하기

## 평균장 가정

**평균장 어림**은 변분 추론에서 가장 흔한, 단순하게 만드는 가정이다. 변분 분포가 모든 숨은 변수에 걸쳐 온전히 인수로 나뉜다고 놓는다.

### 정의

숨은 변수가 $\theta = (\theta_1, \theta_2, \ldots, \theta_K)$인 모형에서 평균장 변분 집안은 다음과 같다:

$$
\mathcal{Q}_{\text{MF}} = \left\{ q(\theta) : q(\theta) = \prod_{j=1}^K q_j(\theta_j) \right\}
$$

매개변수 $\theta_j$마다 저마다 독립인 변분 인수 $q_j(\theta_j)$을 갖는다.

### 말밑: 왜 "평균장"인가?

이 말은 통계 물리에서 왔으며, 거기서는 복잡한 여러 물체 체계의 어림을 가리킨다:

- 알갱이마다 다른 모든 알갱이가 만든 **평균 장**과 맞닿는다
- 낱낱의 흔들림과 상관은 무시한다
- 이 어림은 복잡한 맞닿음을 간추린 평균으로 바꾼다

변분 추론에서 변수 $\theta_j$은 다른 변수의 온전한 분포가 아니라 그 **기댓값**에 달려 있다.

## 수학으로 따라 나오는 것

### 독립 가정

평균장 가정은 다음을 강제한다:

$$
q(\theta_1, \theta_2, \ldots, \theta_K) = q_1(\theta_1) \times q_2(\theta_2) \times \cdots \times q_K(\theta_K)
$$

이는 다음을 뜻한다:

$$
\text{Cov}_q(\theta_i, \theta_j) = 0 \quad \text{for all } i \neq j
$$

**참 뒤확률에 상관이 있어도 평균장은 그것을 담아내지 못한다.**

### 눈으로 보기

```
True Posterior p(θ₁,θ₂|D):            Mean-Field q(θ₁)q(θ₂):
┌─────────────────────┐              ┌─────────────────────┐
│         ╱           │              │                     │
│        ╱            │              │    ┌───────────┐    │
│       ╱   ╭──╮      │              │    │           │    │
│      ╱   │  │       │     →→→      │    │     ○     │    │
│     ╱    ╰──╯       │              │    │           │    │
│    ╱                │              │    └───────────┘    │
└─────────────────────┘              └─────────────────────┘
   Correlated ellipse                 Axis-aligned ellipse
   (tilted, captures                  (cannot capture
    covariance)                        covariance)
```

### 어림의 오차 재기

상관이 $\rho$인 이변량 가우스에서:

$$
p(\theta_1, \theta_2 | \mathcal{D}) = \mathcal{N}\left(\begin{pmatrix} \mu_1 \\ \mu_2 \end{pmatrix}, \begin{pmatrix} \sigma_1^2 & \rho\sigma_1\sigma_2 \\ \rho\sigma_1\sigma_2 & \sigma_2^2 \end{pmatrix}\right)
$$

가장 좋은 평균장 어림은 다음이다:

$$
q(\theta_1)q(\theta_2) = \mathcal{N}(\mu_1, \sigma_1^2) \times \mathcal{N}(\mu_2, \sigma_2^2)
$$

둘 사이의 KL 벌어짐은 다음과 같다:

$$
\text{KL}(q \| p) = -\frac{1}{2}\log(1 - \rho^2)
$$

이는 **상관이 높으면 어림이 나빠짐**을 보여 준다:

| 상관 $\rho$ | KL 벌어짐 |
|-------------------|---------------|
| 0.0 | 0.000 |
| 0.5 | 0.144 |
| 0.8 | 0.511 |
| 0.9 | 0.831 |
| 0.95 | 1.151 |
| 0.99 | 2.296 |

## 가장 좋은 평균장 새로 고치기

### CAVI 새로 고침 공식

평균장 변분 추론에는 가장 좋은 인수 $q_j^*(\theta_j)$에 대한 아름다운 닫힌 꼴 식이 있다:

$$
\boxed{q_j^*(\theta_j) \propto \exp\left\{ \mathbb{E}_{q_{-j}}[\log p(\theta, \mathcal{D})] \right\}}
$$

여기서 $q_{-j} = \prod_{i \neq j} q_i(\theta_i)$은 $q_j$을 뺀 모든 인수를 뜻한다.

### 유도

다른 인수를 모두 붙박아 둔 채 $q_j$에 대해 ELBO을 가장 크게 하고자 한다:

$$
\text{ELBO}(q) = \mathbb{E}_q[\log p(\mathcal{D}, \theta)] - \sum_{k=1}^K \mathbb{E}_{q_k}[\log q_k(\theta_k)]
$$

$q_j$에 대한 범함수 미분을 취하고 0으로 두면:

$$
\frac{\delta \text{ELBO}}{\delta q_j} = \mathbb{E}_{q_{-j}}[\log p(\mathcal{D}, \theta)] - \log q_j(\theta_j) - 1 = 0
$$

$q_j$에 대해 풀면:

$$
\log q_j^*(\theta_j) = \mathbb{E}_{q_{-j}}[\log p(\mathcal{D}, \theta)] + \text{const}
$$

지수를 취하고 고르게 하면:

$$
q_j^*(\theta_j) = \frac{\exp\{\mathbb{E}_{q_{-j}}[\log p(\mathcal{D}, \theta)]\}}{\int \exp\{\mathbb{E}_{q_{-j}}[\log p(\mathcal{D}, \theta)]\} d\theta_j}
$$

### 핵심 통찰

가장 좋은 $q_j^*(\theta_j)$은 다른 모든 변수의 지금 변분 분포 아래에서의 **기댓값 충분 통계량**에 달려 있다. 이는 되풀이하는 달림을 낳으며 좌표 오르기로 푼다.

## 좌표 오르기 변분 추론(CAVI)

### 알고리즘

CAVI은 다른 것을 붙박아 둔 채 변분 인수를 하나씩 되풀이해 새로 고친다:

```
Algorithm: Coordinate Ascent Variational Inference (CAVI)
─────────────────────────────────────────────────────────
Input: Model p(D, θ), initial q₁⁽⁰⁾, ..., qₖ⁽⁰⁾
Output: Optimized variational factors q₁*, ..., qₖ*

1. Initialize variational factors q₁⁽⁰⁾, ..., qₖ⁽⁰⁾
2. Compute initial ELBO
3. Repeat until convergence:
   a. For j = 1, ..., K:
      i.  Compute expected sufficient statistics from q₋ⱼ
      ii. Update: qⱼ(θⱼ) ∝ exp{E_{q₋ⱼ}[log p(θ, D)]}
   b. Compute ELBO
   c. Check convergence (ELBO change < tolerance)
4. Return optimized factors
```

### 수렴의 성질

**정리(CAVI의 단조로움)**: $q_j$을 새로 고쳐도 ELBO이 줄어들 수 없다.

*증명 밑그림*: 좌표를 새로 고칠 때마다 다른 것이 주어진 상태에서 그 인수의 전체 최적점을 찾는다. 인수마다 오목한 범함수를 가장 크게 하므로 ELBO은 커지거나 그대로일 수밖에 없다.

**따름정리**: CAVI은 ELBO의 그 자리 최적점으로 모인다.

## 보기: 평균과 흩어짐을 모르는 가우스

### 모형 적기

켤레 정규-감마 모형을 보자:

$$
\begin{aligned}
\text{Prior on precision: } & \tau \sim \text{Gamma}(\alpha_0, \beta_0) \\
\text{Prior on mean: } & \mu | \tau \sim \mathcal{N}(\mu_0, (\lambda_0 \tau)^{-1}) \\
\text{Likelihood: } & x_i | \mu, \tau \sim \mathcal{N}(\mu, \tau^{-1})
\end{aligned}
$$

### 평균장 인수 나누기

다음을 놓는다:

$$
q(\mu, \tau) = q_\mu(\mu) \cdot q_\tau(\tau)
$$

### 가장 좋은 인수 이끌어 내기

**결합 로그 확률:**

$$
\log p(\mathbf{x}, \mu, \tau) = \log p(\tau) + \log p(\mu|\tau) + \sum_{i=1}^n \log p(x_i|\mu,\tau)
$$

**가장 좋은 $q_\mu(\mu)$:**

$q_\tau$에 걸쳐 기댓값을 취하면:

$$
\log q_\mu^*(\mu) = \mathbb{E}_{q_\tau}[\log p(\mathbf{x}, \mu, \tau)] + \text{const}
$$

$\mu$이 든 항을 모으면:

$$
\log q_\mu^*(\mu) \propto -\frac{\mathbb{E}[\tau]}{2}\left[\lambda_0(\mu - \mu_0)^2 + \sum_{i=1}^n(x_i - \mu)^2\right]
$$

이는 $\mu$에 대한 이차식이므로:

$$
q_\mu^*(\mu) = \mathcal{N}(\mu_n, \lambda_n^{-1})
$$

여기서 각 기호는 다음과 같다.

$$
\begin{aligned}
\lambda_n &= (\lambda_0 + n)\mathbb{E}_{q_\tau}[\tau] \\
\mu_n &= \frac{\lambda_0 \mu_0 + n\bar{x}}{\lambda_0 + n}
\end{aligned}
$$

**가장 좋은 $q_\tau(\tau)$:**

마찬가지로:

$$
q_\tau^*(\tau) = \text{Gamma}(\alpha_n, \beta_n)
$$

여기서 각 기호는 다음과 같다.

$$
\begin{aligned}
\alpha_n &= \alpha_0 + \frac{n+1}{2} \\
\beta_n &= \beta_0 + \frac{1}{2}\left[\lambda_0(\mathbb{E}[\mu] - \mu_0)^2 + \sum_{i=1}^n(x_i - \mathbb{E}[\mu])^2 + n\text{Var}_{q_\mu}[\mu]\right]
\end{aligned}
$$

## PyTorch 구현

```python
import torch
import torch.distributions as dist
from typing import Dict, Tuple
import matplotlib.pyplot as plt
import numpy as np

class MeanFieldGaussianCAVI:
    """
    평균과 정밀도를 모르는 가우스의 좌표 오르기 변분 추론.
    
    모형:
        τ ~ Gamma(α₀, β₀)           [정밀도 앞확률]
        μ | τ ~ N(μ₀, (λ₀τ)⁻¹)      [조건부 평균 앞확률]
        xᵢ | μ,τ ~ N(μ, τ⁻¹)        [가능도]
    
    변분 집안:
        q(μ,τ) = q(μ)q(τ)
        q(μ) = N(μₙ, λₙ⁻¹)
        q(τ) = Gamma(αₙ, βₙ)
    """
    
    def __init__(self, alpha_0: float = 1.0, beta_0: float = 1.0,
                 mu_0: float = 0.0, lambda_0: float = 1.0):
        """
        앞확률의 웃매개변수로 첫값 잡기.
        """
        self.alpha_0 = alpha_0
        self.beta_0 = beta_0
        self.mu_0 = mu_0
        self.lambda_0 = lambda_0
        
        # 변분 매개변수(앞확률로 첫값을 잡는다)
        self.alpha_n = alpha_0
        self.beta_n = beta_0
        self.mu_n = mu_0
        self.lambda_n = lambda_0
    
    @property
    def E_tau(self) -> float:
        """기댓값 정밀도 E_q[τ]"""
        return self.alpha_n / self.beta_n
    
    @property
    def E_log_tau(self) -> float:
        """기댓값 로그 정밀도 E_q[log τ]"""
        return torch.digamma(torch.tensor(self.alpha_n)).item() - np.log(self.beta_n)
    
    @property
    def E_mu(self) -> float:
        """기댓값 평균 E_q[μ]"""
        return self.mu_n
    
    @property
    def Var_mu(self) -> float:
        """평균의 흩어짐 Var_q[μ]"""
        return 1.0 / self.lambda_n
    
    def update_q_mu(self, data: torch.Tensor) -> None:
        """
        q(μ)의 변분 매개변수 새로 고치기.
        
        q*(μ) ∝ exp{E_q(τ)[log p(x,μ,τ)]}
        """
        n = len(data)
        x_bar = data.mean().item()
        
        # 변분 가우스의 정밀도
        self.lambda_n = (self.lambda_0 + n) * self.E_tau
        
        # 변분 가우스의 평균
        self.mu_n = (self.lambda_0 * self.mu_0 + n * x_bar) / (self.lambda_0 + n)
    
    def update_q_tau(self, data: torch.Tensor) -> None:
        """
        q(τ)의 변분 매개변수 새로 고치기.
        
        q*(τ) ∝ exp{E_q(μ)[log p(x,μ,τ)]}
        """
        n = len(data)
        
        # 꼴 매개변수
        self.alpha_n = self.alpha_0 + (n + 1) / 2
        
        # 비율 매개변수
        # E[(μ - μ₀)²] = (E[μ] - μ₀)² + Var[μ]
        E_mu_minus_mu0_sq = (self.E_mu - self.mu_0)**2 + self.Var_mu
        
        # E[Σ(xᵢ - μ)²] = Σ(xᵢ - E[μ])² + n·Var[μ]
        E_sum_sq = torch.sum((data - self.E_mu)**2).item() + n * self.Var_mu
        
        self.beta_n = self.beta_0 + 0.5 * (self.lambda_0 * E_mu_minus_mu0_sq + E_sum_sq)
    
    def compute_elbo(self, data: torch.Tensor) -> float:
        """
        증거 아래 경계 셈하기.
        
        ELBO = E_q[log p(x,μ,τ)] - E_q[log q(μ,τ)]
             = E_q[log p(x|μ,τ)] + E_q[log p(μ|τ)] + E_q[log p(τ)]
               - E_q[log q(μ)] - E_q[log q(τ)]
        """
        n = len(data)
        
        # E[log p(x|μ,τ)] - 기댓값 로그 가능도
        E_log_likelihood = (
            0.5 * n * (self.E_log_tau - np.log(2 * np.pi))
            - 0.5 * self.E_tau * (
                torch.sum((data - self.E_mu)**2).item() + n * self.Var_mu
            )
        )
        
        # E[log p(μ|τ)] - μ에 대한 기댓값 로그 앞확률
        E_log_prior_mu = (
            0.5 * (np.log(self.lambda_0) + self.E_log_tau - np.log(2 * np.pi))
            - 0.5 * self.lambda_0 * self.E_tau * (
                (self.E_mu - self.mu_0)**2 + self.Var_mu
            )
        )
        
        # E[log p(τ)] - τ에 대한 기댓값 로그 앞확률
        E_log_prior_tau = (
            self.alpha_0 * np.log(self.beta_0)
            - torch.lgamma(torch.tensor(self.alpha_0)).item()
            + (self.alpha_0 - 1) * self.E_log_tau
            - self.beta_0 * self.E_tau
        )
        
        # -E[log q(μ)] - q(μ)의 엔트로피
        H_q_mu = 0.5 * (1 + np.log(2 * np.pi) - np.log(self.lambda_n))
        
        # -E[log q(τ)] - q(τ)의 엔트로피
        H_q_tau = (
            self.alpha_n
            - np.log(self.beta_n)
            + torch.lgamma(torch.tensor(self.alpha_n)).item()
            + (1 - self.alpha_n) * torch.digamma(torch.tensor(self.alpha_n)).item()
        )
        
        elbo = E_log_likelihood + E_log_prior_mu + E_log_prior_tau + H_q_mu + H_q_tau
        
        return elbo
    
    def fit(self, data: torch.Tensor, max_iter: int = 100, 
            tol: float = 1e-6, verbose: bool = True) -> Dict:
        """
        모일 때까지 CAVI 알고리즘 돌리기.
        """
        history = {
            'elbo': [],
            'mu_n': [],
            'lambda_n': [],
            'alpha_n': [],
            'beta_n': [],
            'E_mu': [],
            'E_tau': []
        }
        
        elbo_prev = -float('inf')
        
        for iteration in range(max_iter):
            # 좌표 오르기 새로 고침
            self.update_q_mu(data)
            self.update_q_tau(data)
            
            # ELBO 셈하기
            elbo = self.compute_elbo(data)
            
            # 이력 기록
            history['elbo'].append(elbo)
            history['mu_n'].append(self.mu_n)
            history['lambda_n'].append(self.lambda_n)
            history['alpha_n'].append(self.alpha_n)
            history['beta_n'].append(self.beta_n)
            history['E_mu'].append(self.E_mu)
            history['E_tau'].append(self.E_tau)
            
            if verbose and (iteration + 1) % 10 == 0:
                print(f"Iter {iteration+1:3d}: ELBO = {elbo:.4f}, "
                      f"E[μ] = {self.E_mu:.4f}, E[τ] = {self.E_tau:.4f}")
            
            # 모임 살피기
            if abs(elbo - elbo_prev) < tol:
                if verbose:
                    print(f"\nConverged at iteration {iteration + 1}")
                break
            
            elbo_prev = elbo
        
        return history
    
    def get_posterior_distributions(self) -> Tuple[dist.Distribution, dist.Distribution]:
        """변분 분포를 돌려준다."""
        q_mu = dist.Normal(self.mu_n, 1.0 / np.sqrt(self.lambda_n))
        q_tau = dist.Gamma(self.alpha_n, self.beta_n)
        return q_mu, q_tau


def visualize_cavi_results(model: MeanFieldGaussianCAVI, 
                          history: Dict,
                          data: torch.Tensor,
                          true_mu: float = None,
                          true_tau: float = None):
    """CAVI 결과 두루 그려 보기."""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 그림 1: ELBO의 모임
    ax = axes[0, 0]
    ax.plot(history['elbo'], 'b-', linewidth=2)
    ax.set_xlabel('Iteration', fontsize=11)
    ax.set_ylabel('ELBO', fontsize=11)
    ax.set_title('(a) ELBO Convergence', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # 그림 2: 평균 매개변수의 모임
    ax = axes[0, 1]
    ax.plot(history['E_mu'], 'b-', linewidth=2, label='E[μ]')
    if true_mu is not None:
        ax.axhline(true_mu, color='r', linestyle='--', linewidth=2, label='True μ')
    ax.axhline(data.mean().item(), color='g', linestyle=':', linewidth=2, 
               label='Sample mean')
    ax.set_xlabel('Iteration', fontsize=11)
    ax.set_ylabel('Mean', fontsize=11)
    ax.set_title('(b) Mean Convergence', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 그림 3: 정밀도 매개변수의 모임
    ax = axes[0, 2]
    ax.plot(history['E_tau'], 'b-', linewidth=2, label='E[τ]')
    if true_tau is not None:
        ax.axhline(true_tau, color='r', linestyle='--', linewidth=2, label='True τ')
    ax.set_xlabel('Iteration', fontsize=11)
    ax.set_ylabel('Precision', fontsize=11)
    ax.set_title('(c) Precision Convergence', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 그림 4: 뒤확률 q(μ)
    ax = axes[1, 0]
    q_mu, _ = model.get_posterior_distributions()
    mu_range = torch.linspace(model.E_mu - 3/np.sqrt(model.lambda_n),
                              model.E_mu + 3/np.sqrt(model.lambda_n), 200)
    q_mu_pdf = torch.exp(q_mu.log_prob(mu_range))
    
    ax.plot(mu_range.numpy(), q_mu_pdf.numpy(), 'b-', linewidth=2.5, label='q(μ)')
    ax.fill_between(mu_range.numpy(), q_mu_pdf.numpy(), alpha=0.3, color='blue')
    if true_mu is not None:
        ax.axvline(true_mu, color='r', linestyle='--', linewidth=2, label='True μ')
    ax.axvline(model.E_mu, color='b', linestyle=':', linewidth=2, label='E[μ]')
    ax.set_xlabel('μ', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('(d) Posterior q(μ)', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 그림 5: 뒤확률 q(τ)
    ax = axes[1, 1]
    _, q_tau = model.get_posterior_distributions()
    tau_range = torch.linspace(0.01, model.E_tau * 3, 200)
    q_tau_pdf = torch.exp(q_tau.log_prob(tau_range))
    
    ax.plot(tau_range.numpy(), q_tau_pdf.numpy(), 'g-', linewidth=2.5, label='q(τ)')
    ax.fill_between(tau_range.numpy(), q_tau_pdf.numpy(), alpha=0.3, color='green')
    if true_tau is not None:
        ax.axvline(true_tau, color='r', linestyle='--', linewidth=2, label='True τ')
    ax.axvline(model.E_tau, color='g', linestyle=':', linewidth=2, label='E[τ]')
    ax.set_xlabel('τ', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('(e) Posterior q(τ)', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 그림 6: 자료와 뒤확률 미리봄
    ax = axes[1, 2]
    ax.hist(data.numpy(), bins=20, density=True, alpha=0.6, 
            color='gray', edgecolor='black', label='Data')
    
    # 뒤확률 미리봄(표집으로 어림)
    n_samples = 1000
    mu_samples = np.random.normal(model.mu_n, 1/np.sqrt(model.lambda_n), n_samples)
    tau_samples = np.random.gamma(model.alpha_n, 1/model.beta_n, n_samples)
    
    x_range = np.linspace(data.min().item() - 2, data.max().item() + 2, 200)
    pred_pdf = np.zeros_like(x_range)
    for mu_s, tau_s in zip(mu_samples, tau_samples):
        pred_pdf += np.exp(-0.5 * tau_s * (x_range - mu_s)**2) * np.sqrt(tau_s / (2*np.pi))
    pred_pdf /= n_samples
    
    ax.plot(x_range, pred_pdf, 'b-', linewidth=2.5, label='Posterior Predictive')
    if true_mu is not None:
        ax.axvline(true_mu, color='r', linestyle='--', linewidth=2, label='True μ')
    ax.set_xlabel('x', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('(f) Posterior Predictive', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('mean_field_cavi.png', dpi=150, bbox_inches='tight')
    plt.show()


# 사용 예
if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 합성 데이터 생성
    true_mu = 3.0
    true_tau = 0.5  # 정밀도 = 1/흩어짐
    n_samples = 100
    
    data = torch.randn(n_samples) / np.sqrt(true_tau) + true_mu
    
    print("=" * 60)
    print("Mean-Field CAVI for Gaussian Model")
    print("=" * 60)
    print(f"\nTrue μ = {true_mu}, True τ = {true_tau}")
    print(f"Sample mean = {data.mean().item():.4f}")
    print(f"Sample variance = {data.var().item():.4f}")
    
    # CAVI 돌리기
    model = MeanFieldGaussianCAVI(
        alpha_0=1.0, beta_0=1.0,  # τ에 두는 감마 앞확률
        mu_0=0.0, lambda_0=0.01   # μ에 두는 정규 앞확률
    )
    
    history = model.fit(data, max_iter=100, verbose=True)
    
    print(f"\nFinal estimates:")
    print(f"  E[μ] = {model.E_mu:.4f} (true: {true_mu})")
    print(f"  E[τ] = {model.E_tau:.4f} (true: {true_tau})")
    
    # 시각화한다
    visualize_cavi_results(model, history, data, true_mu, true_tau)
```

## 장점과 한계

### 이점

1. **다룰 수 있음**: 많은 모형에서 닫힌 꼴로 새로 고친다
2. **규모 키우기**: 독립인 인수라 차원이 높아도 커진다
3. **풀이하기 쉬움**: 인수마다 뜻이 또렷하다
4. **모임**: ELBO이 단조롭게 나아짐이 보장된다

### 한계

1. **상관이 없음**: 뒤확률의 달림을 담아내지 못한다
2. **불확실함을 낮춰 잡음**: 보통 지나치게 자신한다
3. **봉우리 고르기**: 봉우리가 여럿인 뒤확률에서 봉우리를 놓칠 수 있다
4. **그 자리 최적점**: CAVI은 그 자리 최적점만 찾는다

## 평균장을 언제 쓰나

**다음에 좋다:**

- 뒤확률 상관이 약한 매개변수
- 온전한 공분산을 감당할 수 없는 큰 문제
- 켤레 짜임을 갖는 모형
- 빠른 어림 추론

**다음일 때는 피하라:**

- 매개변수의 상관이 강하리라 여겨질 때
- 불확실함을 정확히 재는 것이 결정적일 때
- MCMC을 감당할 수 있는 작은 문제일 때
- 뒤확률의 봉우리가 여럿일 가능성이 높을 때

## 요약

평균장 가정

$$
q(\theta) = \prod_{j=1}^K q_j(\theta_j)
$$

은 다음의 가장 좋은 새로 고침으로 이어진다:

$$
q_j^*(\theta_j) \propto \exp\{\mathbb{E}_{q_{-j}}[\log p(\theta, \mathcal{D})]\}
$$

**핵심 주고받음**: 단순함과 다룰 수 있음 대 상관을 담아내지 못함.

## 연습문제

### 연습 1: 상관의 영향

뒤확률의 상관이 커질수록 평균장 어림의 질이 어떻게 나빠지는지 보이는 흉내내기를 구현하여라.

### 연습 2: 다변량 가우스

평균 벡터를 모르고 공분산이 대각인 다변량 가우스의 평균장 새로 고침을 이끌어 내어라.

### 연습 3: 선형 회귀의 CAVI

가중값과 잡음 흩어짐을 모르는 베이즈 선형 회귀에 CAVI을 구현하여라.

## 참고 문헌

1. Blei, D. M., Kucukelbir, A., & McAuliffe, J. D. (2017). "Variational Inference: A Review for Statisticians."

2. Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*, 10장.

3. Wainwright, M. J., & Jordan, M. I. (2008). "Graphical Models, Exponential Families, and Variational Inference."

4. Parisi, G. (1988). *Statistical Field Theory*. (평균장이라는 말의 뿌리)

---

# 덧붙임: 좌표 오르기 변분 추론(CAVI)

다음 절에서는 평균장 변분 추론의 고전 최적화 알고리즘인 CAVI의 알고리즘과 구현을 자세히 다룬다.

---

# 좌표 오르기 변분 추론

## 학습 목표

이 절을 마치면 다음을 할 수 있게 된다.

1. ELBO을 최적화하는 전략으로서 좌표 오르기 이해하기
2. 지수 집안 모형의 CAVI 새로 고침 이끌어 내기
3. 모임을 지켜보는 효율적인 CAVI 알고리즘 구현하기
4. 모임의 성질과 셈 복잡도 살피기
5. 가우스 섞음 모형에 CAVI 쓰기

## 좌표 오르기 전략

좌표 오르기 변분 추론(CAVI)은 평균장 가정 아래에서 ELBO을 최적화하는 고전 알고리즘이다. 변분 매개변수를 모두 한꺼번에 최적화하는 대신 CAVI은 다른 것을 붙박아 둔 채 인수를 하나씩 새로 고친다.

### CAVI의 원리

평균장 인수 나눔 $q(\theta) = \prod_{j=1}^K q_j(\theta_j)$에서 CAVI은 다음을 되풀이한다:

$$
\text{For } j = 1, \ldots, K: \quad q_j^{(t+1)}(\theta_j) \propto \exp\left\{\mathbb{E}_{q_{-j}^{(t)}}[\log p(\theta, \mathcal{D})]\right\}
$$

여기서 $q_{-j}^{(t)}$은 인수 $1, \ldots, j-1$에는 가장 최근에 새로 고친 값을, 인수 $j+1, \ldots, K$에는 앞선 되풀이의 값을 쓴다.

### 알고리즘의 짜임

```
Algorithm: Coordinate Ascent Variational Inference (CAVI)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Input: 
  - Model: p(D, θ) = p(D|θ)p(θ)
  - Factorization: q(θ) = ∏ⱼ qⱼ(θⱼ)
  - Tolerance: ε
  - Max iterations: T

Output: Optimized variational factors {qⱼ*}

1. Initialize: Set initial variational parameters {φⱼ⁽⁰⁾}
2. Compute: ELBO⁽⁰⁾

3. For t = 1, 2, ..., T:
   a. For j = 1, 2, ..., K:
      i.   Compute: E_{q₋ⱼ}[log p(θ, D)]
      ii.  Update: qⱼ⁽ᵗ⁾(θⱼ) ∝ exp{E_{q₋ⱼ}[log p(θ, D)]}
      iii. Extract: New variational parameters φⱼ⁽ᵗ⁾
   
   b. Compute: ELBO⁽ᵗ⁾
   
   c. If |ELBO⁽ᵗ⁾ - ELBO⁽ᵗ⁻¹⁾| < ε:
      Return {qⱼ⁽ᵗ⁾}

4. Return {qⱼ⁽ᵀ⁾} (may not have converged)
```

## 수렴 성질

### 단조로움 정리

**정리**: CAVI으로 새로 고쳐도 ELBO이 줄어들 수 없다.

**증명**: 다른 것을 모두 붙박아 둔 채 인수 $q_j$을 새로 고친다고 하자. ELBO은 다음처럼 쓸 수 있다:

$$
\text{ELBO}(q_1, \ldots, q_K) = \mathbb{E}_q[\log p(\mathcal{D}, \theta)] - \sum_{k=1}^K \mathbb{E}_{q_k}[\log q_k(\theta_k)]
$$

$q_j$이 든 항은 다음과 같다:

$$
\text{ELBO}_j = \mathbb{E}_{q_j}\left[\mathbb{E}_{q_{-j}}[\log p(\mathcal{D}, \theta)]\right] - \mathbb{E}_{q_j}[\log q_j(\theta_j)]
$$

이는 (상수를 빼면) KL 벌어짐의 음수이다:

$$
\text{ELBO}_j = -\text{KL}\left(q_j(\theta_j) \| \tilde{p}_j(\theta_j)\right) + \text{const}
$$

여기서 $\tilde{p}_j(\theta_j) \propto \exp\{\mathbb{E}_{q_{-j}}[\log p(\mathcal{D}, \theta)]\}$이다.

$q_j = \tilde{p}_j$일 때 KL 벌어짐이 가장 작아지므로(0이 되므로) CAVI의 새로 고침은 $\text{ELBO}_j$을 가장 크게 한다. 그러므로 새로 고칠 때마다 ELBO은 커지거나 그대로일 수밖에 없다.

### 그 자리 최적점으로의 모임

**따름정리**: CAVI은 ELBO의 그 자리 최적점으로 모인다.

ELBO은 위로 $\log p(\mathcal{D})$으로 묶인다. 단조로움과 함께 보면 ELBO 값의 늘어놓음은 모여야 한다. 모인 자리에서는 인수 하나를 새로 고쳐도 ELBO을 낫게 할 수 없으며, 이것이 좌표별 최적점의 정의이다.

### 모임 속도

CAVI은 보통 선형으로 모인다:

$$
\text{ELBO}^* - \text{ELBO}^{(t)} \leq C \cdot \rho^t
$$

여기서 $\rho < 1$은 모이는 빠르기이며 문제의 짜임에 달려 있다.

## 지수 집안 모형의 CAVI

온전한 조건부 분포와 변분 인수가 모두 지수 집안에 들면 CAVI의 새로 고침이 특히 우아한 꼴이 된다.

### 지수 집안의 배경

다음처럼 쓸 수 있으면 그 분포는 지수 집안에 든다:

$$
p(x | \eta) = h(x) \exp\left\{\eta^\top T(x) - A(\eta)\right\}
$$

여기서 각 기호는 다음과 같다.

- $\eta$은 **자연 매개변수**이다
- $T(x)$은 **충분 통계량**이다
- $A(\eta)$은 **로그 나눔 함수**이다
- $h(x)$은 **바탕 측도**이다

### 자연 매개변수로 하는 CAVI 새로 고침

지수 집안 모형에서 인수 $q_j$의 CAVI 새로 고침은 자연 매개변수를 새로 고치는 것으로 줄어든다:

$$
\eta_j^{(t+1)} = \mathbb{E}_{q_{-j}^{(t)}}\left[\eta_j(\theta_{-j}, \mathcal{D})\right]
$$

여기서 $\eta_j(\theta_{-j}, \mathcal{D})$은 온전한 조건부 $p(\theta_j | \theta_{-j}, \mathcal{D})$의 자연 매개변수이다.

### 흔한 지수 집안의 새로 고침

| 분포 | 자연 매개변수 | 새로 고치는 규칙 |
|--------------|-------------------|-------------|
| 가우스 $\mathcal{N}(\mu, \sigma^2)$ | $\eta_1 = \mu/\sigma^2$, $\eta_2 = -1/(2\sigma^2)$ | 기댓값의 평균 |
| 감마 $\text{Ga}(\alpha, \beta)$ | $\eta_1 = \alpha - 1$, $\eta_2 = -\beta$ | 기댓값 통계량의 합 |
| 범주 $\text{Cat}(\pi)$ | $\eta_k = \log \pi_k$ | 기댓값 확률의 로그 |
| 디리클레 $\text{Dir}(\alpha)$ | $\eta_k = \alpha_k - 1$ | 기댓값 로그 확률의 합 |

## 보기: 가우스 섞음 모형

가우스 섞음 모형(GMM)은 CAVI의 대표 보기이다.

### 모형 적기

$$
\begin{aligned}
\text{Mixing weights: } & \pi \sim \text{Dir}(\alpha_0) \\
\text{Component means: } & \mu_k \sim \mathcal{N}(\mu_0, \sigma_0^2) \quad k = 1, \ldots, K \\
\text{Cluster assignments: } & z_i \sim \text{Cat}(\pi) \quad i = 1, \ldots, N \\
\text{Observations: } & x_i | z_i, \{\mu_k\} \sim \mathcal{N}(\mu_{z_i}, \sigma^2)
\end{aligned}
$$

### 평균장 인수 나누기

$$
q(\pi, \{\mu_k\}, \{z_i\}) = q(\pi) \prod_{k=1}^K q(\mu_k) \prod_{i=1}^N q(z_i)
$$

### CAVI 새로 고침

**$q(z_i)$ 새로 고치기(무리 배정):**

$$
q^*(z_i = k) \propto \exp\left\{\mathbb{E}[\log \pi_k] + \mathbb{E}\left[\log \mathcal{N}(x_i | \mu_k, \sigma^2)\right]\right\}
$$

$r_{ik} = q(z_i = k)$이라 하면:

$$
r_{ik} \propto \exp\left\{\psi(\alpha_k) - \psi\left(\sum_j \alpha_j\right) - \frac{1}{2\sigma^2}\left[(x_i - \mathbb{E}[\mu_k])^2 + \text{Var}[\mu_k]\right]\right\}
$$

**$q(\mu_k)$ 새로 고치기(성분의 평균):**

$$
q^*(\mu_k) = \mathcal{N}(\mu_k | m_k, s_k^2)
$$

여기서 각 기호는 다음과 같다.

$$
\begin{aligned}
s_k^2 &= \left(\frac{1}{\sigma_0^2} + \frac{N_k}{\sigma^2}\right)^{-1} \\
m_k &= s_k^2 \left(\frac{\mu_0}{\sigma_0^2} + \frac{\sum_i r_{ik} x_i}{\sigma^2}\right)
\end{aligned}
$$

그리고 $N_k = \sum_i r_{ik}$은 무리 $k$에 든 점의 기댓값 개수이다.

**$q(\pi)$ 새로 고치기(섞음 무게):**

$$
q^*(\pi) = \text{Dir}(\alpha_1, \ldots, \alpha_K)
$$

여기서 각 기호는 다음과 같다.

$$
\alpha_k = \alpha_0 + \sum_{i=1}^N r_{ik}
$$

## PyTorch 구현

```python
import torch
import torch.nn.functional as F
from torch.distributions import Normal, Dirichlet, Categorical
from typing import Dict, Tuple
import matplotlib.pyplot as plt
import numpy as np

class GMMCavi:
    """
    가우스 섞음 모형의 좌표 오르기 변분 추론.
    
    모형:
        π ~ Dir(α₀)
        μₖ ~ N(μ₀, σ₀²)
        zᵢ ~ Cat(π)
        xᵢ | zᵢ ~ N(μ_{zᵢ}, σ²)
    
    변분 집안:
        q(π, μ, z) = q(π) ∏ₖ q(μₖ) ∏ᵢ q(zᵢ)
    """
    
    def __init__(self, n_components: int, 
                 alpha_0: float = 1.0,
                 mu_0: float = 0.0,
                 sigma_0: float = 10.0,
                 sigma: float = 1.0):
        """
        가우스 섞음 모형 CAVI 첫값 잡기.
        
        인수:
            n_components: 섞음 성분의 개수 K
            alpha_0: 디리클레 앞확률의 몰림
            mu_0: 성분 평균의 앞확률 평균
            sigma_0: 성분 평균의 앞확률 표준편차
            sigma: 아는 관측 잡음의 표준편차
        """
        self.K = n_components
        self.alpha_0 = alpha_0
        self.mu_0 = mu_0
        self.sigma_0 = sigma_0
        self.sigma = sigma
        
        # 변분 매개변수(자료로 첫값을 잡는다)
        self.r = None       # N x K 맡음 몫 행렬
        self.alpha = None   # 디리클레 매개변수 K개
        self.m = None       # 성분 평균 K개
        self.s2 = None      # 성분 흩어짐 K개
    
    def initialize(self, data: torch.Tensor, init_method: str = 'kmeans'):
        """
        변분 매개변수 첫값 잡기.
        """
        N = len(data)
        
        if init_method == 'random':
            # 무작위 첫값 잡기
            self.r = torch.rand(N, self.K)
            self.r = self.r / self.r.sum(dim=1, keepdim=True)
        elif init_method == 'kmeans':
            # k 평균++ 방식의 첫값 잡기
            indices = torch.randperm(N)[:self.K]
            centers = data[indices].clone()
            
            # 거리를 셈하고 배정하기
            dists = torch.cdist(data.unsqueeze(1), centers.unsqueeze(0).unsqueeze(2))
            self.r = F.softmax(-dists.squeeze() / self.sigma**2, dim=1)
        
        # 맡음 몫에 따라 다른 매개변수 첫값 잡기
        self._update_q_pi()
        self._update_q_mu(data)
    
    def _update_q_z(self, data: torch.Tensor) -> None:
        """
        모든 자료 점에 대해 q(zᵢ) 새로 고치기.
        
        q*(zᵢ = k) ∝ exp{E[log πₖ] - (xᵢ - E[μₖ])²/(2σ²) - Var[μₖ]/(2σ²)}
        """
        N = len(data)
        
        # E[log πₖ] = ψ(αₖ) - ψ(Σⱼ αⱼ)
        E_log_pi = torch.digamma(self.alpha) - torch.digamma(self.alpha.sum())
        
        # 로그 맡음 몫(고르게 하지 않음)
        log_r = torch.zeros(N, self.K)
        
        for k in range(self.K):
            # E[(xᵢ - μₖ)²] = (xᵢ - E[μₖ])² + Var[μₖ]
            E_sq_diff = (data - self.m[k])**2 + self.s2[k]
            log_r[:, k] = E_log_pi[k] - 0.5 / self.sigma**2 * E_sq_diff
        
        # 고르게 하기(소프트맥스)
        self.r = F.softmax(log_r, dim=1)
    
    def _update_q_mu(self, data: torch.Tensor) -> None:
        """
        모든 성분에 대해 q(μₖ) 새로 고치기.
        
        q*(μₖ) = N(mₖ, sₖ²)
        """
        N_k = self.r.sum(dim=0)  # 성분마다의 기댓값 횟수
        
        # 뒤확률 정밀도
        precision = 1/self.sigma_0**2 + N_k / self.sigma**2
        self.s2 = 1 / precision
        
        # 뒤확률 평균
        weighted_sum = (self.r * data.unsqueeze(1)).sum(dim=0)
        self.m = self.s2 * (self.mu_0 / self.sigma_0**2 + weighted_sum / self.sigma**2)
    
    def _update_q_pi(self) -> None:
        """
        q(π) 새로 고치기.
        
        q*(π) = Dir(α₁, ..., αₖ)
        αₖ = α₀ + Σᵢ rᵢₖ
        """
        N_k = self.r.sum(dim=0)
        self.alpha = self.alpha_0 + N_k
    
    def compute_elbo(self, data: torch.Tensor) -> float:
        """
        증거 아래 경계 셈하기.
        """
        N = len(data)
        N_k = self.r.sum(dim=0)
        
        # E[log p(x|z,μ)] - 기댓값 로그 가능도
        E_log_lik = 0.0
        for k in range(self.K):
            E_sq_diff = (data - self.m[k])**2 + self.s2[k]
            E_log_lik += (self.r[:, k] * (
                -0.5 * np.log(2 * np.pi * self.sigma**2)
                - 0.5 / self.sigma**2 * E_sq_diff
            )).sum()
        
        # E[log p(z|π)] - z에 대한 기댓값 로그 앞확률
        E_log_pi = torch.digamma(self.alpha) - torch.digamma(self.alpha.sum())
        E_log_p_z = (self.r * E_log_pi).sum()
        
        # E[log p(π)] - π에 대한 기댓값 로그 앞확률
        E_log_p_pi = (
            torch.lgamma(torch.tensor(self.K * self.alpha_0))
            - self.K * torch.lgamma(torch.tensor(self.alpha_0))
            + (self.alpha_0 - 1) * E_log_pi.sum()
        )
        
        # E[log p(μ)] - μ에 대한 기댓값 로그 앞확률
        E_log_p_mu = 0.0
        for k in range(self.K):
            E_log_p_mu += (
                -0.5 * np.log(2 * np.pi * self.sigma_0**2)
                - 0.5 / self.sigma_0**2 * ((self.m[k] - self.mu_0)**2 + self.s2[k])
            )
        
        # -E[log q(z)] - q(z)의 엔트로피
        H_q_z = -(self.r * torch.log(self.r + 1e-10)).sum()
        
        # -E[log q(π)] - q(π)의 엔트로피
        H_q_pi = (
            torch.lgamma(self.alpha.sum())
            - torch.lgamma(self.alpha).sum()
            + (self.alpha - 1).dot(
                torch.digamma(self.alpha.sum()) - torch.digamma(self.alpha)
            )
        )
        
        # -E[log q(μ)] - q(μ)의 엔트로피
        H_q_mu = 0.5 * self.K * (1 + np.log(2 * np.pi)) + 0.5 * torch.log(self.s2).sum()
        
        elbo = (E_log_lik + E_log_p_z + E_log_p_pi + E_log_p_mu 
                + H_q_z + H_q_pi + H_q_mu)
        
        return elbo.item()
    
    def fit(self, data: torch.Tensor, max_iter: int = 100,
            tol: float = 1e-6, verbose: bool = True) -> Dict:
        """
        CAVI 알고리즘 돌리기.
        """
        # 초기화한다
        self.initialize(data)
        
        history = {
            'elbo': [],
            'r': [],
            'm': [],
            'alpha': []
        }
        
        elbo_prev = -float('inf')
        
        for iteration in range(max_iter):
            # CAVI 새로 고침
            self._update_q_z(data)
            self._update_q_mu(data)
            self._update_q_pi()
            
            # ELBO 셈하기
            elbo = self.compute_elbo(data)
            
            # 이력 기록
            history['elbo'].append(elbo)
            history['r'].append(self.r.clone())
            history['m'].append(self.m.clone())
            history['alpha'].append(self.alpha.clone())
            
            if verbose and (iteration + 1) % 10 == 0:
                print(f"Iter {iteration+1:3d}: ELBO = {elbo:.4f}, "
                      f"means = {self.m.numpy()}")
            
            # 모임 살피기
            if abs(elbo - elbo_prev) < tol:
                if verbose:
                    print(f"\nConverged at iteration {iteration + 1}")
                break
            
            elbo_prev = elbo
        
        return history
    
    def predict(self, data: torch.Tensor) -> torch.Tensor:
        """
        무리 배정 미리보기.
        """
        self._update_q_z(data)
        return self.r.argmax(dim=1)


def visualize_gmm_cavi(model: GMMCavi, history: Dict, data: torch.Tensor,
                       true_labels: torch.Tensor = None):
    """가우스 섞음 모형 CAVI 결과 그려 보기."""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 그림 1: ELBO의 모임
    ax = axes[0, 0]
    ax.plot(history['elbo'], 'b-', linewidth=2)
    ax.set_xlabel('Iteration', fontsize=11)
    ax.set_ylabel('ELBO', fontsize=11)
    ax.set_title('(a) ELBO Convergence', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # 그림 2: 성분 평균의 모임
    ax = axes[0, 1]
    means_history = torch.stack(history['m'])
    for k in range(model.K):
        ax.plot(means_history[:, k], linewidth=2, label=f'μ_{k+1}')
    ax.set_xlabel('Iteration', fontsize=11)
    ax.set_ylabel('Mean', fontsize=11)
    ax.set_title('(b) Component Means', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 그림 3: 마지막 무리짓기
    ax = axes[0, 2]
    predictions = model.r.argmax(dim=1)
    colors = plt.cm.tab10(predictions.numpy() / model.K)
    ax.scatter(data.numpy(), np.zeros_like(data.numpy()), c=colors, alpha=0.6, s=50)
    for k in range(model.K):
        ax.axvline(model.m[k].item(), color=plt.cm.tab10(k / model.K), 
                   linestyle='--', linewidth=2, label=f'μ_{k+1}')
    ax.set_xlabel('x', fontsize=11)
    ax.set_title('(c) Final Clustering', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 그림 4: 맡음 몫의 흘러감
    ax = axes[1, 0]
    r_history = torch.stack(history['r'])
    sample_idx = 0  # 첫 자료 점 기록
    for k in range(model.K):
        ax.plot(r_history[:, sample_idx, k], linewidth=2, label=f'r_{sample_idx+1},{k+1}')
    ax.set_xlabel('Iteration', fontsize=11)
    ax.set_ylabel('Responsibility', fontsize=11)
    ax.set_title(f'(d) Responsibilities (point {sample_idx+1})', 
                fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 그림 5: 디리클레 매개변수
    ax = axes[1, 1]
    alpha_history = torch.stack(history['alpha'])
    for k in range(model.K):
        ax.plot(alpha_history[:, k], linewidth=2, label=f'α_{k+1}')
    ax.set_xlabel('Iteration', fontsize=11)
    ax.set_ylabel('Concentration', fontsize=11)
    ax.set_title('(e) Dirichlet Parameters', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 그림 6: 마지막 섞음 밀도
    ax = axes[1, 2]
    x_range = torch.linspace(data.min() - 2, data.max() + 2, 500)
    
    E_pi = model.alpha / model.alpha.sum()
    mixture_pdf = torch.zeros_like(x_range)
    
    for k in range(model.K):
        component_pdf = E_pi[k] * torch.exp(
            -0.5 * (x_range - model.m[k])**2 / (model.sigma**2 + model.s2[k])
        ) / np.sqrt(2 * np.pi * (model.sigma**2 + model.s2[k]))
        mixture_pdf += component_pdf
        ax.plot(x_range.numpy(), component_pdf.numpy(), '--', 
                linewidth=1.5, alpha=0.7, label=f'Component {k+1}')
    
    ax.plot(x_range.numpy(), mixture_pdf.numpy(), 'b-', linewidth=2.5, 
            label='Mixture')
    ax.hist(data.numpy(), bins=30, density=True, alpha=0.4, color='gray')
    ax.set_xlabel('x', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('(f) Fitted Mixture', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('gmm_cavi.png', dpi=150, bbox_inches='tight')
    plt.show()


# 사용 예
if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 인공 섞음 자료 만들기
    n_samples = 300
    true_means = torch.tensor([-3.0, 0.0, 4.0])
    true_weights = torch.tensor([0.3, 0.4, 0.3])
    sigma = 1.0
    
    # 섞음에서 표집
    z_true = torch.multinomial(true_weights, n_samples, replacement=True)
    data = torch.randn(n_samples) * sigma + true_means[z_true]
    
    print("=" * 60)
    print("Gaussian Mixture Model CAVI")
    print("=" * 60)
    print(f"\nTrue means: {true_means.numpy()}")
    print(f"True weights: {true_weights.numpy()}")
    
    # CAVI으로 가우스 섞음 모형 맞추기
    model = GMMCavi(n_components=3, alpha_0=1.0, sigma=sigma)
    history = model.fit(data, max_iter=100, verbose=True)
    
    print(f"\nFinal estimates:")
    print(f"  Means: {model.m.numpy()}")
    print(f"  Expected weights: {(model.alpha / model.alpha.sum()).numpy()}")
    
    # 시각화한다
    visualize_gmm_cavi(model, history, data)
```

## 계산 복잡도

### 되풀이마다의 값

인수가 $K$개, 자료 점이 $N$개인 모형에서:

| 부분 | 복잡도 |
|-----------|------------|
| 모든 $i$에 대해 $q(z_i)$ 새로 고치기 | $O(NK)$ |
| 모든 $k$에 대해 $q(\mu_k)$ 새로 고치기 | $O(NK)$ |
| $q(\pi)$ 새로 고치기 | $O(K)$ |
| ELBO 셈하기 | $O(NK)$ |
| **되풀이마다 모두** | $O(NK)$ |

### EM과 견주기

가우스 섞음 모형의 CAVI은 EM 알고리즘과 복잡도가 같지만 점 어림값이 아니라 온전한 뒤확률 분포를 준다.

## 요약

CAVI은 ELBO을 최적화하는 차근차근한 길을 준다:

**알고리즘**:

1. 변분 매개변수의 첫값을 잡는다
2. 인수를 차례로 새로 고친다: $q_j^* \propto \exp\{\mathbb{E}_{q_{-j}}[\log p(\theta, \mathcal{D})]\}$
3. 모일 때까지 ELBO을 지켜본다

**성질**:

- ELBO이 단조롭게 나아진다
- 그 자리 최적점으로 모인다
- 지수 집안에서는 닫힌 꼴로 새로 고친다
- 섞음 모형에서 EM과 복잡도가 같다

## 참고 문헌

1. Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*, 10장.

2. Blei, D. M., & Jordan, M. I. (2006). "Variational Inference for Dirichlet Process Mixtures."

3. Hoffman, M. D., Blei, D. M., Wang, C., & Paisley, J. (2013). "Stochastic Variational Inference."

4. Wainwright, M. J., & Jordan, M. I. (2008). "Graphical Models, Exponential Families, and Variational Inference."

## 연습문제

### 연습 1: 인자 분석의 CAVI

확률 인자 분석 모형의 CAVI을 이끌어 내고 구현하여라.

### 연습 2: 모임 살피기

무리가 떨어진 정도의 함수로 가우스 섞음 모형 CAVI의 모이는 빠르기를 경험으로 재어라.

### 연습 3: 첫값에 대한 민감도

첫값 잡기 전략에 따라 마지막 ELBO과 무리의 질이 어떻게 달라지는지 살펴라.
