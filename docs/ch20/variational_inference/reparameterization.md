# 리파라메리제이션과 블랙박스 변분 추론

블랙박스 변분 추론은 모델별 좌표 갱신을 유도하는 대신 몬테카를로 경사로 ELBO를 최적화한다. 점수함수 추정과 리파라메리제이션은 서로 다른 미분 경로를 이용한다. 표본화·로그 밀도 평가·정칙성·유한 분산 등의 조건이 필요하므로 ‘어떤 모델에도 자동으로 적용된다’는 표현은 지나치게 넓다.

## 1. 분포와 피적분함수의 파라메터 의존성

L(φ)=E_qφ[f_φ(θ)], f_φ=log p(D,θ)−log q_φ(θ)라 하자. 모델 p는 여기서 φ와 독립이라고 가정한다. 지지집합이 고정되고 미분과 적분을 교환할 수 있으면 다음과 같다.

$$\nabla_\phi L=E_{q_\phi}[f_\phi\nabla_\phi\log q_\phi+\nabla_\phi f_\phi].$$

θ를 고정한 명시적 미분은 ∇_φf_φ=−∇_φlog q_φ다. 정규화 조건으로 E_qφ[∇_φlog q_φ]=0이므로 ELBO에서는 다음 점수함수 식을 얻는다.

$$\nabla_\phi L=E_{q_\phi}[(\log p(D,\theta)-\log q_\phi(\theta))\nabla_\phi\log q_\phi(\theta)].$$

일반적인 파라메터 의존 함수 f_φ에 대해 명시적 미분 항을 처음부터 생략하면 안 된다. 이동하는 지지집합이나 비정칙 밀도에서는 경계 항이 생기거나 점수의 평균이 0이라는 논증이 실패할 수 있다.

## 2. 점수함수 경사 추정

θˢ∼q_φ를 독립 생성해 위 기대의 표본 평균을 계산한다. 이 방법은 연속·이산 변수 모두에 사용할 수 있고 θ에 대한 모델의 경사가 꼭 필요하지 않다. 대신 로그 결합밀도, q의 로그 밀도와 φ에 대한 점수를 평가할 수 있어야 한다. 기대 경사가 존재해도 추정 분산은 무한할 수 있다.

대각 가우스 q=N(μ,diag σ²)의 점수는 s_μ=(θ−μ)/σ², s_logσ=(θ−μ)²/σ²−1이다. 점수함수 로스를 자동미분으로 구성할 때 θ는 점수 미분에서 고정 표본이고, 곱해 주는 학습 신호도 해당 경사에 맞게 분리해야 한다. 리파라메리제이션된 θ를 그대로 따라 미분하면 의도한 점수함수 추정과 다른 경로가 섞일 수 있다.

## 3. 기준선과 제어변량

표본 θ와 독립인 상수 기준선 b는 E[b s_φ]=0이므로 다음 기대를 바꾸지 않는다.

$$\nabla L=E[(f_\phi-b)s_\phi].$$

벡터 경사 추정의 전체 분산, 즉 공분산 행렬의 trace를 최소화하는 상수는 분모가 양수이고 기대가 유한할 때 b★=E[f_φ∥s_φ∥²]/E[∥s_φ∥²]다. 단순 이동평균은 실용적인 근사지만 이 최적 가중 기준선과 반드시 같지는 않다.

일반적인 표본 의존 c(θ)를 빼면 E[c(θ)s_φ]를 다시 더해야 한다. 이를 E[c]E[s_φ]=0으로 바꾸면 일반적으로 바이어스가 생긴다. 같은 표본 묶음의 평균으로 기준선을 만들어 각 표본에 빼는 경우에도 의존성 때문에 바이어스가 생길 수 있어 이전 묶음의 기준선이나 leave-one-out 등의 구성을 검토한다.

Rao–Blackwell화는 경사 추정량의 일부 변수를 조건부로 적분해 기대를 유지하고 분산을 줄인다. 실제 경사 추정량의 조건부 기대를 사용해야 하며, f만 임의로 평균낸 뒤 원래 점수를 곱하는 것으로 항상 같은 결과가 되지는 않는다.

## 4. 리파라메리제이션 경사

φ와 무관한 ε∼p₀와 θ=g_φ(ε)로 표본을 표현하면 다음 경로별 경사를 사용할 수 있다.

$$\nabla_\phi L=E_{p_0}[\nabla_\phi f_\phi(g_\phi(\epsilon))].$$

오른쪽 미분에는 θ를 통한 경로와 f_φ의 명시적 의존성이 모두 포함된다. g와 목적함수의 적절한 미분 가능성 및 교환 조건이 필요하다. 가우스는 θ=μ+σ⊙ε, 로그정규는 θ=exp(μ+σ⊙ε), ε∼N(0,I)를 쓸 수 있다. 전 공분산 가우스에서는 θ=μ+Lε, Σ=LLᵀ다.

연속 누적분포의 역함수 θ=F_φ⁻¹(u), u∼Uniform(0,1)도 가능한 구성이다. 다만 역함수를 수치적으로 구할 수 있다는 사실과 안정적으로 미분할 수 있다는 사실은 다르다. Gamma(α,β)를 ε∼Gamma(α,1), θ=ε/β로 쓰면 **β에 대한** 경로는 분리되지만 ε가 α에 의존하므로 shape α의 미분까지 해결한 것은 아니다. 암시적 리파라메리제이션 등 추가 방법이 필요할 수 있다.

경로별 추정은 많은 연속 모델에서 분산이 작고 자동미분과 잘 맞지만 점수함수보다 항상 작은 분산·빠른 수렴을 보장하지는 않는다. 이산 변수의 보통의 표본 경로는 미분에 적합하지 않아 정확한 합산, 점수함수 또는 연속 완화 등을 사용하며, 완화는 원래 이산 목적과의 차이를 만든다.

## 5. 예제 코드의 비교 범위

아래 점수함수 구현은 가우스 점수를 명시적으로 계산해 경사를 대입한다. 표본 생성에 `rsample`을 사용해도 여기서는 그 계산 그래프를 `backward`로 미분하지 않으므로 계산된 점수의 수치값과 경로별 자동미분을 구분한다. 기준선은 이전 반복의 값이므로 현재 표본과 분리되어 있다. 불필요한 그래프 보유와 장치·자료형 처리는 일반화되어 있지 않다.

비교 예제의 `log_joint`는 θ와 무관한 정규화 상수를 생략한다. 최적 사후 형태는 같을 수 있지만 기록된 절대 목적값을 원래 모델의 완전한 ELBO나 모델 증거 비교에 사용하면 안 된다. 관측 분산을 바꾸려면 우도 이차항의 분산 배율도 일관되게 반영해야 한다. 현재 예제는 분산을 1로 고정한다.

```python
import torch
import torch.nn as nn
import torch.distributions as dist
from typing import Callable, Dict, Tuple
import matplotlib.pyplot as plt
import numpy as np

class ScoreFunctionVI:
    """
    점수 함수(REINFORCE) 그래디언트 어림꼴을 쓴 깜깜이 변분 추론.
    
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
        
        # 가우스 q(θ) = N(μ, diag(σ²))의 변분 파라메터
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
        기준선을 붙인 점수 함수 어림꼴로 그래디언트 셈하기.
        
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
        
        # 점수 함수 그래디언트
        # 표본마다 ∇_φ log q_φ(θ)을 셈해야 한다
        # 가우스에서: ∇_μ log q = (θ - μ)/σ², ∇_log_σ log q = ((θ-μ)²/σ² - 1)
        
        # 경사 초기화
        if self.mu.grad is not None:
            self.mu.grad.zero_()
        if self.log_sigma.grad is not None:
            self.log_sigma.grad.zero_()
        
        # 표본마다 그래디언트 셈하기(효율은 낮지만 또렷하다)
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
        점수 함수 그래디언트로 최적화하기.
        """
        optimizer = torch.optim.Adam([self.mu, self.log_sigma], lr=lr)
        
        history = {'elbo': [], 'mu': [], 'sigma': []}
        baseline = 0.0
        
        for i in range(n_iterations):
            optimizer.zero_grad()
            
            # 그래디언트 셈하기
            (grad_mu, grad_log_sigma), elbo = self.score_function_gradient(
                n_samples, baseline if use_baseline else 0.0
            )
            
            # 기준선 새로 고치기(f의 달리는 평균)
            if use_baseline:
                baseline = 0.9 * baseline + 0.1 * elbo
            
            # 그래디언트 정하기(최적화기가 최소화하므로 음수)
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
    파라메터 바꾸기 재주를 쓴 깜깜이 변분 추론.
    
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
        
        # 변분 파라메터
        self.mu = nn.Parameter(torch.zeros(dim))
        self.log_sigma = nn.Parameter(torch.zeros(dim))
    
    @property
    def sigma(self):
        return torch.exp(self.log_sigma)
    
    def elbo(self, n_samples: int = 100) -> torch.Tensor:
        """
        파라메터 바꾸기로 ELBO 셈하기.
        
        θ = μ + σ ⊙ ε,  ε ~ N(0, I)
        
        ELBO = E_ε[log p(D, μ + σε) - log q(μ + σε)]
        """
        # 바탕 분포에서 표집
        epsilon = torch.randn(n_samples, self.dim)
        
        # 파라메터 바꾸기
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
        파라메터 바꾸기 그래디언트로 최적화하기.
        
        메모: 점수 함수보다 표본이 훨씬 적게 든다!
        """
        optimizer = torch.optim.Adam([self.mu, self.log_sigma], lr=lr)
        
        history = {'elbo': [], 'mu': [], 'sigma': []}
        
        for i in range(n_iterations):
            optimizer.zero_grad()
            
            # ELBO 셈하기(그래디언트가 파라메터 바꾸기를 지나 흐른다)
            elbo = self.elbo(n_samples)
            
            # 그래디언트 오르기(ELBO의 음수를 가장 작게 하기)
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
    """점수 함수 어림꼴과 파라메터 바꾸기 어림꼴 견주기."""
    
    # 단순한 타깃: 가우스 평균의 뒤확률
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
    
    # 파라메터 바꾸기(표본이 적게 든다)
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
    
    # 파라메터 바꾸기 결과
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

다음 기존 출력은 재실행하지 않았다. 두 방법은 각각 100개와 10개의 표본을 사용하므로 동일한 표본 예산의 비교가 아니다. 그림의 ‘Gradient Variance’ 패널도 고정 φ에서 반복 계산한 **경사 분산**이 아니라 학습 중 ELBO 기록의 이동 구간 분산이다. 최적화 이동과 표본 오차가 섞여 있어 경사 추정량의 분산 우열을 증명하지 않는다.

```
Exact posterior: N(1.9608, 0.1400²)

--- Score Function Estimator ---
Iter  100: ELBO = -61.9416, μ = [0.89673996], σ = [0.53983235]
Iter  200: ELBO = -36.2961, μ = [1.4868687], σ = [0.3704372]
Iter  300: ELBO = -31.1588, μ = [1.7897462], σ = [0.30120647]
Iter  400: ELBO = -29.6703, μ = [1.9151655], σ = [0.2537165]
Iter  500: ELBO = -29.2401, μ = [1.9505261], σ = [0.223465]

--- Reparameterization Trick ---
Iter  100: ELBO = -71.4015, μ = [0.88176584], σ = [0.56794333]
Iter  200: ELBO = -39.4114, μ = [1.4731997], σ = [0.4150803]
Iter  300: ELBO = -30.5908, μ = [1.794035], σ = [0.3270984]
Iter  400: ELBO = -29.7265, μ = [1.9132032], σ = [0.27492437]
Iter  500: ELBO = -29.5348, μ = [1.953634], σ = [0.24334748]
```

## 6. 자동미분 변분 추론

ADVI는 제약을 없앤 연속 좌표에서 가우스 근사를 사용하고 리파라메리제이션 경사를 계산한다. 평균장과 전 공분산 가우스 등의 선택이 있으며, 미분 가능한 연속 모델이라는 적용 범위와 수치적 제약을 가진다. 이산 잠재변수가 그대로 남거나 적절한 변환·미분이 없는 모델에 무조건 적용할 수는 없다.

θ=h(ζ)가 같은 차원의 가역 미분변환이면 변환 좌표에서의 결합밀도는 다음과 같다.

$$\log p_\zeta(D,\zeta)=\log p_\theta(D,h(\zeta))+\log|\det Dh(\zeta)|.$$

왼쪽은 θ 좌표의 밀도가 아니라 **ζ 좌표의 밀도**다. 양수는 θ=exp ζ, (0,1)은 θ=sigmoid ζ로 변환할 수 있다. K차원 단체는 K−1개의 자유 좌표에서 변환하며 역변환의 결과는 단체의 점이다. Dirichlet은 역변환이 아니라 단체 위의 분포다. 구면은 전역 단일 좌표계의 특이점과 차원·기준측도를 다뤄야 하므로 보통의 정방 야코비안 표를 그대로 대입하지 않는다.

변환 공간에서 가우스인 근사가 원래 공간에서도 가우스인 것은 아니다. 변환은 제약을 다루는 동시에 분포족의 모양도 바꾼다. 자세한 정식화는 [ADVI 원 논문](https://www.jmlr.org/papers/v18/16-107.html)을 참고한다.

## 연습문제

### 연습 1: 파라메터 의존성

일반적인 E_qφ[f_φ]의 경사를 유도하고, ELBO에서 명시적 미분 항이 기댓값 아래 사라지는 조건을 설명하라. 지지집합이 움직이는 Uniform(0,φ)에서는 어떤 주의가 필요한가?

### 연습 2: 기준선의 조건

상수 기준선과 표본 의존 제어변량의 보정식을 비교하라. E[c(θ)s_φ]를 E[c]E[s_φ]로 바꿀 수 없는 이유를 설명하라.

### 연습 3: 분산 비교 설계

동일한 φ·표본 수에서 경사 추정량을 독립 반복 계산하는 비교 절차를 설계하라. 학습 중 ELBO의 이동 분산이 그 비교를 대신하지 못하는 이유를 설명하라.

### 연습 4: 제약과 변환밀도

양수 θ=exp ζ에 대해 변환된 로그 결합밀도와 야코비안 항을 쓰라. Gamma 분포의 rate만 스케일링하는 표현이 shape 미분도 해결하는지 구분하라.

## 정리하며

점수함수와 경로별 추정은 ELBO의 파라메터 의존성을 서로 다른 방식으로 처리한다. 정칙성, 기준선의 독립성, 제어변량의 보정과 변환밀도의 야코비안을 확인해야 한다. 추정 분산은 고정된 파라메터에서 같은 계산 기준으로 비교한다.

**참고 문헌**

1. Ranganath, R., Gerrish, S., & Blei, D. M. (2014). "Black Box Variational Inference."

2. Kingma, D. P., & Welling, M. (2014). "Auto-Encoding Variational Bayes."

3. Rezende, D. J., Mohamed, S., & Wierstra, D. (2014). "Stochastic Backpropagation and Approximate Inference in Deep Generative Models."

4. Kucukelbir, A., Tran, D., Ranganath, R., Gelman, A., & Blei, D. M. (2017). "Automatic Differentiation Variational Inference."

5. Mohamed, S., Rosca, M., Figurnov, M., & Mnih, A. (2020). "Monte Carlo Gradient Estimation in Machine Learning."
