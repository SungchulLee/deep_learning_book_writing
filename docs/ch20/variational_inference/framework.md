# 변분 추론의 기본 구조

변분 추론(variational inference, VI)은 사후분포를 제한된 분포족의 최적화 문제로 근사한다. 주변우도를 직접 계산하지 않고도 목적함수를 구성할 수 있지만, 그 목적함수의 기댓값 계산과 최적화까지 항상 쉬워지는 것은 아니다. 이 절에서는 사후 근사, KL 방향과 증거 하한의 관계를 정리한다.

## 1. 베이즈 추론의 계산 문제

데이터 D와 파라메터 θ에 대해 사후분포는 다음과 같다.

$$p(\theta|D)=\frac{p(D|\theta)p(\theta)}{p(D)},\qquad p(D)=\int p(D|\theta)p(\theta)\,d\theta.$$

연속 관측의 우도는 관측값에서 평가한 밀도이며 관측점 자체의 확률과 다르다. 사후가 정의되려면 0<p(D)<∞여야 한다. 고차원, 비선형 우도, 복잡한 의존성과 다봉성은 적분이나 사후 기댓값 계산을 어렵게 할 수 있다. 비켤레라는 이유만으로 수치 계산까지 불가능하다고 결론 내릴 수는 없다.

GMM에서는 고정 파라메터 아래의 라벨 사후는 책임도로 계산할 수 있지만, 비중·평균·공분산과 라벨의 결합 베이즈 사후는 일반적으로 단일한 간단한 분포가 아니다. 어떤 변수를 추론하는지에 따라 계산 문제가 달라진다.

사후에 정규화 상수가 있다고 모든 추론 방법이 그 상수를 먼저 계산해야 하는 것도 아니다. MCMC는 미정규화 밀도의 비를 이용할 수 있고, VI는 주변우도를 최적화에서 상수로 제거한다.

## 2. 변분분포족

계산하고 표본화할 수 있는 분포족 Q={q_λ(θ)}를 정한 뒤 사후에 가까운 q를 찾는다. 분포족은 근사할 수 있는 의존성·꼬리·모드 수를 제한한다.

| 구조 | 표현하는 것 | 주요 제한 |
|---|---|---|
| 평균장 | 블록별 독립 q=∏ⱼqⱼ | 블록 간 의존성을 생략 |
| 구조화 근사 | 선택한 블록·조건부 의존성 | 계산량과 구조 설계 |
| 전 공분산 가우스 | 선형 상관 | 단봉·가우스 꼬리 |
| 정규화 흐름 | 가역변환으로 만든 유연한 밀도 | 변환·야코비안 계산 |

뉴럴 넷트웍 인코더로 분포의 파라메터를 공유하는 **상각 추론**은 별도의 표현 구조다. 인코더가 복잡해도 출력이 대각 가우스이면 각 입력의 근사분포는 여전히 대각 가우스다. 공유 여부와 분포족의 표현력을 같은 기준으로 순위 매기지 않는다.

## 3. KL 발산으로 정한 목표

표준 VI에서는 다음 방향의 KL 발산을 최소화한다.

$$D_{\mathrm{KL}}(q\Vert p_{post})=E_q\left[\log\frac{q(\theta)}{p(\theta|D)}\right],\qquad q^*\in\arg\min_{q\in Q}D_{\mathrm{KL}}(q\Vert p_{post}).$$

KL은 음이 아니고, 두 분포가 거의 어디서나 같을 때에만 0이다. 대칭성이나 삼각부등식을 일반적으로 만족하지 않아 거리 함수와는 다르다. q가 양의 질량을 두는 영역에서 사후가 0이면 KL(q∥p)는 무한대일 수 있다. 유한 최적해가 존재하는지와 최적화를 실제로 달성하는지도 별도 문제다.

## 4. ELBO와 주변우도의 소거

사후의 로그를 결합분포로 쓰면 다음 항등식을 얻는다. 필요한 기댓값이 유한하다고 가정한다.

$$D_{\mathrm{KL}}(q\Vert p_{post})=E_q[\log q]-E_q[\log p(D,\theta)]+\log p(D).$$

$$\mathcal L(q)=E_q[\log p(D,\theta)]-E_q[\log q]=E_q[\log p(D|\theta)]-D_{\mathrm{KL}}(q\Vert p(\theta)).$$

따라서 log p(D)=L(q)+KL(q∥p_post)≥L(q)이고, 모델을 고정하면 ELBO 최대화와 해당 KL 최소화가 같다. 주변우도는 q에 대한 상수이므로 최적화에 필요 없다. 그러나 E_q[log p(D,θ)] 자체는 계산하기 어려울 수 있어 해석적 적분·몬테카를로·추가 근사 등을 사용한다. 한 번의 몬테카를로 ELBO 추정값이 항상 log p(D) 아래에 있는 것은 아니다. 하한은 정확한 기대 목적함수에 대한 성질이다.

## 5. 두 KL 방향을 구분하기

‘정방향·역방향’은 목표분포를 어느 자리에 놓는지 먼저 정의해야 한다. 여기서는 **KL(p∥q)를 정방향, KL(q∥p)를 역방향**이라고 부른다. 표준 VI는 역방향을 쓴다.

KL(q∥p)는 q 아래에서 평균하므로 q가 p의 작은 밀도 영역에 확률을 두는 것을 강하게 벌한다. 제한된 단봉 분포족에서는 여러 모드 사이의 저밀도 영역을 피하려 한 모드를 선택하고 분산을 작게 추정할 수 있다. p>0인 모든 영역을 q가 덮어야 한다는 조건은 없다.

반대로 KL(p∥q)는 p 아래에서 평균하므로 p의 중요한 영역에서 q가 0이면 무한대가 된다. 하나의 가우스로 여러 모드를 근사하면 모드를 넓게 덮는 결과가 생길 수 있다. 이는 흔한 경향이며 모든 분포족·목표에서 각각 한 모드 선택과 과대 분산을 보장하는 정리는 아니다.

아래 기존 도식은 모드 포괄과 모드 선택의 직관을 보여 준다. 표기의 해석은 위에서 명시한 KL(p∥q), KL(q∥p)의 순서를 따른다.

```
True Posterior (Bimodal):           Forward KL Result:        Reverse KL Result:
       ╱╲    ╱╲                          ╱────╲                      ╱╲
      ╱  ╲  ╱  ╲                        ╱      ╲                    ╱  ╲
     ╱    ╲╱    ╲                      ╱        ╲                  ╱    ╲
    ╱            ╲                    ╱          ╲                ╱      ╲
───╱              ╲───            ───╱            ╲───          ─╱        ╲────
   Mode 1   Mode 2                 Covers both modes           Focuses on one
```

## 6. 가우스 평균의 예제

관측 분산 σ²를 알고 θ∼N(μ₀,σ₀²), xᵢ|θ∼N(θ,σ²)라고 하자. 사후는 정확히 가우스다.

$$v_n=(\sigma_0^{-2}+n\sigma^{-2})^{-1},\qquad m_n=v_n\left(\mu_0\sigma_0^{-2}+\sigma^{-2}\sum_i x_i\right).$$

q=N(m,s²)는 정확한 사후를 포함하므로 이 모델에서는 최적 근사 오차가 0이다. 복잡한 모델의 필요성보다는 목적함수와 최적화를 확인하는 기준 예제다. 코드의 ELBO는 가우스 기대 로그 우도에서 사전과의 KL을 뺀다.

```python
import math
import torch
import torch.nn as nn
import torch.distributions as dist
import matplotlib.pyplot as plt

class SimpleVI:
    """
    흩어짐을 아는 가우스의 평균을 어림하는 변분 추론.
    흩어짐을 알 때.
    
    모형:
        앞확률: θ ~ N(μ₀, σ₀²)
        가능도: x_i | θ ~ N(θ, σ²)
        
    변분 집안:
        q(θ) = N(m, s²)  여기서 m, s은 변분 파라메터이다
    """
    
    def __init__(self, prior_mean: float = 0.0, prior_std: float = 1.0,
                 likelihood_std: float = 1.0):
        self.mu_0 = prior_mean
        self.sigma_0 = prior_std
        self.sigma = likelihood_std
        
        # 변분 파라메터(최적화할 것)
        self.m = torch.tensor([0.0], requires_grad=True)
        self.log_s = torch.tensor([0.0], requires_grad=True)
    
    @property
    def s(self):
        """표준편차가 양수임을 보장하기."""
        return torch.exp(self.log_s)
    
    def compute_elbo(self, data: torch.Tensor) -> torch.Tensor:
        """
        증거 아래 경계 셈하기.
        
        ELBO = E_q[log p(D|θ)] + E_q[log p(θ)] - E_q[log q(θ)]
             = E_q[log p(D|θ)] - KL(q(θ) || p(θ))
        
        q과 p이 가우스이면 닫힌 꼴 식이 있다.
        """
        n = len(data)
        
        # E_q[log p(D|θ)] - 기댓값 로그 가능도
        # 흩어짐을 아는 가우스 가능도에서:
        # E_q[log p(x|θ)] = -n/2 log(2πσ²) - 1/(2σ²) Σᵢ E_q[(xᵢ - θ)²]
        #                 = -n/2 log(2πσ²) - 1/(2σ²) [Σᵢ(xᵢ - m)² + n·s²]
        
        expected_log_likelihood = (
            -0.5 * n * math.log(2 * math.pi * self.sigma**2)   # 파이썬 실수라 math를 쓴다
            - 0.5 / self.sigma**2 * (
                torch.sum((data - self.m)**2) + n * self.s**2
            )
        )
        
        # KL(q(θ) || p(θ)) - 앞확률에서 벌어진 정도
        # 가우스에서: KL(N(m,s²) || N(μ₀,σ₀²))
        #              = log(σ₀/s) + (s² + (m-μ₀)²)/(2σ₀²) - 1/2
        
        kl_divergence = (
            torch.log(self.sigma_0 / self.s)
            + (self.s**2 + (self.m - self.mu_0)**2) / (2 * self.sigma_0**2)
            - 0.5
        )
        
        elbo = expected_log_likelihood - kl_divergence
        return elbo
    
    def fit(self, data: torch.Tensor, n_iterations: int = 1000,
            learning_rate: float = 0.01, verbose: bool = True):
        """
        ELBO에 대한 그래디언트 오르기로 변분 파라메터 최적화하기.
        """
        optimizer = torch.optim.Adam([self.m, self.log_s], lr=learning_rate)
        
        history = {'elbo': [], 'm': [], 's': []}
        
        for i in range(n_iterations):
            optimizer.zero_grad()
            
            elbo = self.compute_elbo(data)
            loss = -elbo  # ELBO의 음수를 가장 작게 = ELBO를 가장 크게
            
            loss.backward()
            optimizer.step()
            
            # 이력 기록
            history['elbo'].append(elbo.item())
            history['m'].append(self.m.item())
            history['s'].append(self.s.item())
            
            if verbose and (i + 1) % 200 == 0:
                print(f"Iter {i+1:4d}: ELBO = {elbo.item():.4f}, "
                      f"m = {self.m.item():.4f}, s = {self.s.item():.4f}")
        
        return history
    
    def get_posterior(self):
        """어림 뒤확률 분포를 돌려준다."""
        return dist.Normal(self.m.detach(), self.s.detach())
    
    def exact_posterior(self, data: torch.Tensor):
        """
        정확한 뒤확률 셈하기(이 켤레 모형에는 있다).
        
        뒤확률: θ | D ~ N(μₙ, σₙ²)
        여기서 각 기호는 다음과 같다.
            precision_n = 1/σ₀² + n/σ²
            μₙ = (μ₀/σ₀² + Σxᵢ/σ²) / precision_n
            σₙ² = 1 / precision_n
        """
        n = len(data)
        precision_0 = 1 / self.sigma_0**2
        precision_data = n / self.sigma**2
        precision_n = precision_0 + precision_data
        
        mu_n = (self.mu_0 * precision_0 + data.sum() * precision_data / n) / precision_n
        sigma_n = 1 / torch.sqrt(torch.tensor(precision_n))
        
        return dist.Normal(mu_n, sigma_n)

# 사용 예
if __name__ == "__main__":
    torch.manual_seed(42)
    
    # 합성 데이터 생성
    true_mean = 2.5
    n_samples = 50
    data = torch.randn(n_samples) + true_mean
    
    print("=" * 60)
    print("Variational Inference for Gaussian Mean")
    print("=" * 60)
    print(f"\nTrue mean: {true_mean}")
    print(f"Sample mean: {data.mean().item():.4f}")
    print(f"Sample size: {n_samples}")
    
    # 변분 추론 모형 맞추기
    vi = SimpleVI(prior_mean=0.0, prior_std=2.0, likelihood_std=1.0)
    history = vi.fit(data, n_iterations=1000)
    
    # 정확한 뒤확률과 견주기
    exact = vi.exact_posterior(data)
    approx = vi.get_posterior()
    
    print(f"\nExact posterior:  N({exact.mean.item():.4f}, {exact.stddev.item():.4f}²)")
    print(f"VI approximation: N({approx.mean.item():.4f}, {approx.stddev.item():.4f}²)")
    print(f"\nDifference in mean: {abs(exact.mean.item() - approx.mean.item()):.6f}")
    print(f"Difference in std:  {abs(exact.stddev.item() - approx.stddev.item()):.6f}")
```

다음은 기존 출력이며 재실행한 결과가 아니다. Adam의 유한 반복이 정확한 최적점을 달성하거나 매번 ELBO를 높인다는 보장은 없다. 기록되는 ELBO는 갱신 전, `m`과 `s`는 갱신 후 값이므로 같은 기록 인덱스를 완전히 같은 시점으로 해석하지 않는다.

```
============================================================
Variational Inference for Gaussian Mean
============================================================

True mean: 2.5
Sample mean: 2.5957
Sample size: 50
Iter  200: ELBO = -101.0287, m = 1.6287, s = 0.3366
Iter  400: ELBO = -77.9031, m = 2.3675, s = 0.2283
Iter  600: ELBO = -76.4975, m = 2.5552, s = 0.1853
Iter  800: ELBO = -76.4128, m = 2.5808, s = 0.1641
Iter 1000: ELBO = -76.3939, m = 2.5827, s = 0.1527

Exact posterior:  N(2.5827, 0.1411²)
VI approximation: N(2.5827, 0.1527²)

Difference in mean: 0.000071
Difference in std:  0.011653
```

아래 시각화 함수는 기록과 두 분포를 비교하는 용도다. 이번 편집에서 실행하거나 그림을 다시 만들지 않았다.

```python
def visualize_vi_results(vi, data, history):
    """변분 추론의 최적화와 결과 그려 보기."""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 그림 1: ELBO의 모임
    ax = axes[0, 0]
    ax.plot(history['elbo'], 'b-', linewidth=2)
    ax.set_xlabel('Iteration', fontsize=11)
    ax.set_ylabel('ELBO', fontsize=11)
    ax.set_title('(a) ELBO Convergence', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # 그림 2: 파라메터의 자취
    ax = axes[0, 1]
    ax.plot(history['m'], history['s'], 'b-', alpha=0.5, linewidth=1)
    ax.plot(history['m'][0], history['s'][0], 'go', markersize=10, label='Start')
    ax.plot(history['m'][-1], history['s'][-1], 'ro', markersize=10, label='End')
    
    exact = vi.exact_posterior(data)
    ax.plot(exact.mean.item(), exact.stddev.item(), 'k*', 
            markersize=15, label='Exact')
    
    ax.set_xlabel('Mean (m)', fontsize=11)
    ax.set_ylabel('Std Dev (s)', fontsize=11)
    ax.set_title('(b) Parameter Trajectory', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 그림 3: 뒤확률 견주기
    ax = axes[1, 0]
    theta_range = torch.linspace(-1, 5, 500)
    
    exact_pdf = torch.exp(exact.log_prob(theta_range))
    approx = vi.get_posterior()
    approx_pdf = torch.exp(approx.log_prob(theta_range))
    
    ax.plot(theta_range.numpy(), exact_pdf.numpy(), 'b-', 
            linewidth=2.5, label='Exact Posterior')
    ax.plot(theta_range.numpy(), approx_pdf.numpy(), 'r--', 
            linewidth=2.5, label='VI Approximation')
    ax.axvline(data.mean().item(), color='green', linestyle=':', 
               linewidth=2, label='Sample Mean')
    
    ax.set_xlabel('θ', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('(c) Posterior Comparison', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 그림 4: 자료의 막대그림
    ax = axes[1, 1]
    ax.hist(data.numpy(), bins=15, density=True, alpha=0.6, 
            color='gray', edgecolor='black', label='Data')
    ax.axvline(approx.mean.item(), color='red', linestyle='--', 
               linewidth=2, label='Posterior Mean')
    ax.set_xlabel('Data Value', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('(d) Data Distribution', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('vi_framework_example.png', dpi=150, bbox_inches='tight')
    plt.show()
```

## 7. MCMC와의 비교

VI는 분포족의 제약과 최적화 오차를 갖는 근사분포를 준다. MCMC는 조건을 만족하는 전이와 충분한 실행 아래에서 사후 기댓값을 표본 평균으로 추정하지만, 유한 실행의 자기상관·초기화·모드 누락 오차가 남는다. 어느 방법이 빠르고 정확한지는 모델·관심량·허용 오차·계산비용에 달려 있다.

확률적 VI는 샘플링 잡음이 있으며 모든 VI가 결정론적이지 않다. MCMC도 여러 체인과 일부 연산을 병렬화할 수 있다. ELBO의 안정화는 최적화 진단으로 유용하지만 사후 근사의 정확성을 증명하지 않고, MCMC 진단도 수렴의 완전한 증명은 아니다.

큰 데이터에서 미니배치 VI가 유용할 수 있고, 작은 문제에서 MCMC와 비교해 근사 품질을 살필 수 있다. 다봉성은 양쪽 모두 어려울 수 있으므로 방법 이름만으로 누락 모드를 찾았다고 가정하지 않는다.

## 8. 근사 품질의 평가

분포족, 초기값과 최적화 오차를 구분한다. 평균장에서는 상관 누락과 분산 과소 추정이 생길 수 있지만 모든 관심량의 오차 방향이 정해지는 것은 아니다. 사후 예측, 모의 데이터에서의 회복, 계산 가능한 기준해와의 비교 등 문제에 맞는 평가가 필요하다. 서로 다른 모델의 ELBO는 서로 다른 미지의 KL 차이를 가지므로 모델 증거의 순위를 반드시 보존하지 않는다.

## 연습문제

### 연습 1: KL 발산의 비음성

Jensen 부등식 또는 log u≤u−1을 이용해 KL(q∥p)≥0을 증명하고 등호 조건과 지지집합 조건을 설명하라.

### 연습 2: 베타–베르누이 모델

θ∼Beta(α₀,β₀), xᵢ|θ∼Bernoulli(θ), q(θ)=Beta(a,b)에서 ELBO를 유도하라. 관측의 합과 표본 수를 사용해 정확한 사후 파라메터를 구하고 이 사후가 분포족에 포함됨을 확인하라.

### 연습 3: KL 방향과 지지집합

두 모드를 가진 p와 단봉 q를 생각하라. KL(q∥p)와 KL(p∥q)가 각각 어느 분포 아래의 평균인지 설명하고, 각 방향이 무한대가 되는 지지집합 사례를 제시하라.

## 정리하며

VI는 제한된 분포족에서 KL(q∥p_post)를 줄이며, 고정 모델에서 이는 ELBO 최대화와 같다. 주변우도의 직접 계산을 피할 수 있지만 기댓값 계산·근사 바이어스·최적화 오차는 남는다. KL의 두 방향과 하한의 의미를 식으로 구분한다.

**참고 문헌**

1. Blei, D. M., Kucukelbir, A., & McAuliffe, J. D. (2017). "Variational Inference: A Review for Statisticians." *Journal of the American Statistical Association*.

2. Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*, 10장.

3. Murphy, K. P. (2023). *Probabilistic Machine Learning: Advanced Topics*, 변분 추론 관련 장들.

4. Jordan, M. I., Ghahramani, Z., Jaakkola, T. S., & Saul, L. K. (1999). "An Introduction to Variational Methods for Graphical Models." *Machine Learning*.

5. Hoffman, M. D., Blei, D. M., Wang, C., & Paisley, J. (2013). "Stochastic Variational Inference." *Journal of Machine Learning Research*.
