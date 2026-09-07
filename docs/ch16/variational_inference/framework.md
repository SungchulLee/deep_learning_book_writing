# 변분 추론 얼개
## 학습 목표

이 절을 마치면 다음을 할 수 있게 된다.

1. 정확한 베이즈 추론의 셈 어려움 이해하기
2. 변분 추론을 최적화 문제로 세우기
3. 변분 추론과 KL 벌어짐의 관계 설명하기
4. 변분 추론을 다른 어림 추론 방법과 견주기
5. 단순한 확률 모형에 기본 변분 추론 구현하기

## 베이즈 추론의 어려움

베이즈 추론은 관측한 자료가 주어졌을 때 매개변수에 대한 뒤확률 분포를 셈하려 한다. 매개변수가 $\theta$이고 관측 자료가 $\mathcal{D}$인 모형에서 베이즈 정리는 다음을 준다:

$$
p(\theta | \mathcal{D}) = \frac{p(\mathcal{D} | \theta) p(\theta)}{p(\mathcal{D})}
$$

여기서 각 기호는 다음과 같다.

- $p(\theta | \mathcal{D})$은 **뒤확률 분포**이다(우리가 바라는 것)
- $p(\mathcal{D} | \theta)$은 **가능도**이다(매개변수가 주어졌을 때 자료의 확률)
- $p(\theta)$은 **앞확률 분포**이다(우리의 첫 믿음)
- $p(\mathcal{D})$은 **주변 가능도** 또는 **증거**이다

근본적인 셈 어려움은 주변 가능도에 있다:

$$
p(\mathcal{D}) = \int p(\mathcal{D} | \theta) p(\theta) \, d\theta
$$

### 이 적분은 왜 다룰 수 없나?

주변 가능도 적분은 다음일 때 셈으로 다룰 수 없게 된다:

1. **높은 차원**: 매개변수 공간 $\theta$의 차원이 수백에서 수백만일 수 있다
2. **복잡한 가능도 함수**: 켤레가 아닌 모형에는 닫힌 꼴 풀이가 없다
3. **숨은 변수 모형**: 숨은 변수에 걸친 적분이 더해져 어려움이 커진다
4. **봉우리가 여럿임**: 적분할 함수의 짜임이 복잡할 수 있다

!!! example "섞음 모형: 다룰 수 없는 고전 보기"
    성분이 $K$개인 가우스 섞음 모형을 보자. 비교적 단순한 이 모형조차 다음을 갖는다:
    
    - 섞음 무게 $K$개 $\pi_1, \ldots, \pi_K$
    - 평균 벡터 $K$개 $\mu_1, \ldots, \mu_K$
    - 공분산 행렬 $K$개 $\Sigma_1, \ldots, \Sigma_K$
    - 숨은 무리 배정 $N$개 $z_1, \ldots, z_N$
    
    이 모든 양에 대한 뒤확률에는 닫힌 꼴 풀이가 없다.

## 변분 추론: 적분 대신 최적화

변분 추론(VI)은 다룰 수 없는 적분 문제를 다룰 수 있는 최적화 문제로 다시 세운다. 핵심 통찰은 이렇다:

!!! info "변분 추론의 원리"
    정확한 뒤확률 $p(\theta | \mathcal{D})$을 셈하는 대신 다음을 한다:
    
    1. 더 단순한 분포의 집안 $\mathcal{Q} = \{q(\theta)\}$을 고른다
    2. $p(\theta | \mathcal{D})$에 "가장 가까운" 원소 $q^*(\theta) \in \mathcal{Q}$을 찾는다
    3. $q^*(\theta)$을 참 뒤확률의 어림으로 쓴다

### 변분 집안

**변분 집안** $\mathcal{Q}$은 우리가 최적화할, 숨은 변수에 대한 분포의 모음이다. 흔한 고름은 다음과 같다:

| 집안 | 설명 | 유연함 |
|--------|-------------|-------------|
| 평균장 | 온전히 인수로 나눔: $q(\theta) = \prod_j q_j(\theta_j)$ | 낮음 |
| 짜임새 있는 것 | 어떤 달림은 지키며 일부만 인수로 나눔 | 중간 |
| 고르게 하는 흐름 | 바탕 분포를 바꾼 것 | 높음 |
| 신경망 | 부호기로 나눠 갚는 추론 | 높음 |

### "가까움" 재기: KL 벌어짐

분포 사이의 "거리"는 **쿨백-라이블러(KL) 벌어짐**으로 잰다:

$$
\text{KL}(q(\theta) \| p(\theta | \mathcal{D})) = \int q(\theta) \log \frac{q(\theta)}{p(\theta | \mathcal{D})} \, d\theta
$$

KL 벌어짐에는 중요한 성질이 있다:

- **음이 아님**: $\text{KL}(q \| p) \geq 0$
- **같을 때만 0**: 거의 어디서나 $q = p$일 때 그리고 오직 그때만 $\text{KL}(q \| p) = 0$이다
- **대칭이 아님**: $\text{KL}(q \| p) \neq \text{KL}(p \| q)$

### 변분 목표

변분 추론의 목표는 KL 벌어짐을 가장 작게 하는 것이다:

$$
q^*(\theta) = \arg\min_{q \in \mathcal{Q}} \text{KL}(q(\theta) \| p(\theta | \mathcal{D}))
$$

그런데 이 목표에는 여전히 다룰 수 없는 뒤확률이 들어 있다! 변분 추론의 핵심 수학 재주는 이것을 셈할 수 있는 꼴로 다시 쓰는 것이다.

## 수학적 틀

### KL 벌어짐에서 ELBO로

KL 벌어짐의 정의에서 시작한다:

$$
\begin{aligned}
\text{KL}(q \| p) &= \mathbb{E}_q\left[\log \frac{q(\theta)}{p(\theta | \mathcal{D})}\right] \\
&= \mathbb{E}_q[\log q(\theta)] - \mathbb{E}_q[\log p(\theta | \mathcal{D})]
\end{aligned}
$$

뒤확률에 베이즈 규칙을 쓰면:

$$
\log p(\theta | \mathcal{D}) = \log p(\mathcal{D} | \theta) + \log p(\theta) - \log p(\mathcal{D})
$$

대입하면 다음과 같다.

$$
\begin{aligned}
\text{KL}(q \| p) &= \mathbb{E}_q[\log q(\theta)] - \mathbb{E}_q[\log p(\mathcal{D} | \theta)] - \mathbb{E}_q[\log p(\theta)] + \log p(\mathcal{D}) \\
&= -\underbrace{\left(\mathbb{E}_q[\log p(\mathcal{D}, \theta)] - \mathbb{E}_q[\log q(\theta)]\right)}_{\text{ELBO}(q)} + \log p(\mathcal{D})
\end{aligned}
$$

정리하면 근본 항등식을 얻는다:

$$
\boxed{\log p(\mathcal{D}) = \text{ELBO}(q) + \text{KL}(q \| p)}
$$

### 증거 아래 경계

KL 벌어짐이 음이 아니므로:

$$
\log p(\mathcal{D}) \geq \text{ELBO}(q)
$$

그래서 이를 **증거 아래 경계(ELBO)**라 부른다. ELBO는 다음과 같이 정한다:

$$
\text{ELBO}(q) = \mathbb{E}_q[\log p(\mathcal{D}, \theta)] - \mathbb{E}_q[\log q(\theta)]
$$

!!! success "핵심 통찰"
    다음이 성립하므로 **ELBO를 가장 크게 하는 것은 KL 벌어짐을 가장 작게 하는 것과 같다**:
    
    $$\max_q \text{ELBO}(q) \Leftrightarrow \min_q \text{KL}(q \| p)$$
    
    그리고 결정적으로 **ELBO는 $p(\mathcal{D})$을 모르고도 셈할 수 있다**!

## 앞 KL 벌어짐과 뒤 KL 벌어짐

KL 벌어짐의 방향을 어떻게 고르느냐가 어림의 굶에 중요한 영향을 준다.

### 앞 KL: KL(q || p)|KL(q || p) (표준 변분 추론에서 씀)

$$
\text{KL}(q \| p) = \int q(\theta) \log \frac{q(\theta)}{p(\theta | \mathcal{D})} \, d\theta
$$

**성질:**

- **평균을 좇음**: $q$이 $p$의 봉우리를 모두 덮으려 한다
- **0을 피함**: $p(\theta | \mathcal{D}) > 0$인 곳에서는 어디서나 $q(\theta) > 0$이다
- **불확실함을 부풀려 잡을 수 있음**: $q$이 $p$을 덮으려 확률을 펼친다

### 뒤 KL: KL(p || q)|KL(p || q)

$$
\text{KL}(p \| q) = \int p(\theta | \mathcal{D}) \log \frac{p(\theta | \mathcal{D})}{q(\theta)} \, d\theta
$$

**성질:**

- **봉우리를 좇음**: $q$이 $p$의 봉우리 하나에 몰리려 한다
- **0을 강제함**: $p(\theta | \mathcal{D}) \approx 0$인 곳에서 $q(\theta) = 0$을 강제한다
- **불확실함을 낮춰 잡을 수 있음**: $q$이 판치는 봉우리에 몰린다

### 눈으로 견주기

```
True Posterior (Bimodal):           Forward KL Result:        Reverse KL Result:
       ╱╲    ╱╲                          ╱────╲                      ╱╲
      ╱  ╲  ╱  ╲                        ╱      ╲                    ╱  ╲
     ╱    ╲╱    ╲                      ╱        ╲                  ╱    ╲
    ╱            ╲                    ╱          ╲                ╱      ╲
───╱              ╲───            ───╱            ╲───          ─╱        ╲────
   Mode 1   Mode 2                 Covers both modes           Focuses on one
```

## PyTorch 구현

### 가우스 평균 어림을 위한 단순 변분 추론

```python
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
        q(θ) = N(m, s²)  여기서 m, s은 변분 매개변수이다
    """
    
    def __init__(self, prior_mean: float = 0.0, prior_std: float = 1.0,
                 likelihood_std: float = 1.0):
        self.mu_0 = prior_mean
        self.sigma_0 = prior_std
        self.sigma = likelihood_std
        
        # 변분 매개변수(최적화할 것)
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
            -0.5 * n * torch.log(2 * torch.pi * self.sigma**2)
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
        ELBO에 대한 기울기 오르기로 변분 매개변수 최적화하기.
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

### 눈으로 보기

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
    
    # 그림 2: 매개변수의 자취
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

## 변분 추론과 다른 추론 방법

### MCMC와 견주기

| 살필 점 | 변분 추론 | MCMC |
|--------|----------------------|------|
| **내임** | 어림 분포 $q(\theta)$ | $p(\theta\|\mathcal{D})$에서 뽑은 표본 |
| **정확도** | 치우쳤지만 빠름 | 점근으로 정확함 |
| **빠르기** | 빠름(최적화) | 느림(표집) |
| **규모 키우기** | 큰 자료에도 잘 커짐 | 차례차례 표집에 매임 |
| **불확실함** | 낮춰 잡을 수 있음 | 끝에서는 정확함 |
| **모임** | 진단하기 쉬움(ELBO) | 진단이 복잡함 |
| **나란히 하기** | 자연스럽게 나란함 | 어려움 |

### 변분 추론을 언제 쓰나

**다음일 때 변분 추론을 고르라:**

- 자료가 클 때(관측이 수백만 개)
- 빠르기가 결정적일 때(실시간 쓰임새)
- 어림 뒤확률이면 넉넉할 때
- 모형에 숨은 변수가 많을 때
- 복잡한 모형으로 규모를 키워야 할 때

**다음일 때 MCMC를 고르라:**

- 불확실함을 정확히 재는 것이 결정적일 때
- 자료 묶음이 작거나 보통이다
- 금과옥조 같은 추론이 필요할 때
- 봉우리가 여럿인지 알아내야 할 때

## 장점과 한계

### 변분 추론의 좋은 점

1. **규모 키우기**: 큰 자료에 확률 최적화와 함께 쓸 수 있다
2. **빠르기**: 많은 문제에서 MCMC보다 훨씬 빠르다
3. **정해짐**: 어림에 표집 잡음이 없다
4. **모형 견주기**: ELBO가 로그 증거의 아래 경계를 준다
5. **모임 지켜보기**: ELBO가 또렷한 최적화 과녁을 준다

### 변분 추론의 한계

1. **어림의 치우침**: 대체로 $q(\theta) \neq p(\theta | \mathcal{D})$이다
2. **집안의 옭아맴**: 변분 집안을 어떻게 고르느냐가 질을 옭아맨다
3. **불확실함을 낮춰 잡음**: 특히 평균장 가정에서 그렇다
4. **그 자리 최적점**: 최적화가 전체 최적점을 못 찾을 수 있다
5. **봉우리 덮기와 봉우리 좇기**: 앞 KL은 봉우리를 놓칠 수 있다

## 요약

변분 추론은 다룰 수 없는 베이즈 추론을 다룰 수 있는 최적화로 바꾼다:

$$
\text{Intractable: } p(\theta | \mathcal{D}) = \frac{p(\mathcal{D}|\theta)p(\theta)}{\int p(\mathcal{D}|\theta)p(\theta)d\theta}
$$

$$
\text{Tractable: } q^*(\theta) = \arg\max_{q \in \mathcal{Q}} \text{ELBO}(q)
$$

**핵심 관계:**

- $\log p(\mathcal{D}) = \text{ELBO}(q) + \text{KL}(q \| p)$
- $\max_q \text{ELBO}(q) \Leftrightarrow \min_q \text{KL}(q \| p)$
- ELBO는 $p(\mathcal{D})$을 모르고도 셈할 수 있다

## 참고 문헌

1. Blei, D. M., Kucukelbir, A., & McAuliffe, J. D. (2017). "Variational Inference: A Review for Statisticians." *Journal of the American Statistical Association*.

2. Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*, 10장.

3. Murphy, K. P. (2022). *Probabilistic Machine Learning: Advanced Topics*, 변분 추론 관련 장들.

4. Jordan, M. I., Ghahramani, Z., Jaakkola, T. S., & Saul, L. K. (1999). "An Introduction to Variational Methods for Graphical Models." *Machine Learning*.

5. Hoffman, M. D., Blei, D. M., Wang, C., & Paisley, J. (2013). "Stochastic Variational Inference." *Journal of Machine Learning Research*.

## 연습문제

### 연습 1: KL 벌어짐의 성질

옌센 부등식으로 KL 벌어짐이 음이 아님을 증명하여라.

### 연습 2: 베르누이 모형의 ELBO

다음과 같은 베타-베르누이 모형의 ELBO를 이끌어 내어라:

- 앞확률: $\theta \sim \text{Beta}(\alpha_0, \beta_0)$
- 가능도: $x_i | \theta \sim \text{Bernoulli}(\theta)$

### 연습 3: 앞 KL과 뒤 KL 견주기

가우스 둘의 섞음을 어림하는 데 앞 KL 최적화와 뒤 KL 최적화를 모두 구현하여라. 서로 다른 굶을 그려 보아라.
