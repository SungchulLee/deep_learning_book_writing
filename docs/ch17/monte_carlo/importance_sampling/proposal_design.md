# 제안 분포 설계

제안 분포를 어떻게 고르느냐가 중요도 표집의 성패를 가르는 가장 중요한 한 가지이다. 잘 짠 제안은 흩어짐을 자릿수 단위로 줄일 수 있고, 잘못 고른 제안은 어림자를 쓸모없게 만든다. 이 마당은 좋은 제안 분포를 짜는 원칙과 실전 전략을 다룬다.

---

## 1. 가장 좋은 제안

### 이론 결과

$\mathbb{E}_\pi[h(\theta)]$을 어림할 때 흩어짐을 가장 작게 하는 제안은 다음과 같다:

$$
q^*(\theta) = \frac{|h(\theta)| \pi(\theta)}{\int |h(\theta')| \pi(\theta') d\theta'}
$$

### 왜 이것이 가장 좋은가

$q^*$을 쓰면 중요도 무게가 다음처럼 된다:

$$
w(\theta) = \frac{\pi(\theta)}{q^*(\theta)} = \frac{\int |h(\theta')| \pi(\theta') d\theta'}{|h(\theta)|} \cdot \text{sign}(h(\theta))
$$

곱 $h(\theta) w(\theta)$은 다음 까닭으로 들쭉날쭉함이 줄어든다:

- $|h(\theta)|$이 큰 곳에서는 $w(\theta)$이 작다(서로 메운다)
- $|h(\theta)|$이 작은 곳에서는 $w(\theta)$이 크다(그래도 보태는 바는 작다)

### 할 수 없다는 역설

**문제**: 가장 좋은 $q^*$은 $\int |h(\theta)| \pi(\theta) d\theta$을 알아야 하는데, 그것이 바로 우리가 어림하려는 양이다!

**풀이**: $q^*$을 좋은 제안이 어떤 모습이어야 하는지에 대한 **길잡이**로 써라:

1. $|h(\theta)| \pi(\theta)$과 비슷한 모양
2. $\pi(\theta)$보다 무거운 꼬리
3. $\pi(\theta)$의 받침 전체를 덮기

---

## 2. 설계 원칙

### 원칙 1: 받침 덮기

!!! danger "꼭 지켜야 할 조건"

    $$\pi(\theta) > 0 \implies q(\theta) > 0$$
    
    이를 어기면 흩어짐이 큰 정도가 아니라 **끝없는 치우침**이 생긴다.

**무너지는 보기:**

- 과녁: $\pi = \mathcal{N}(0, 1)$
- 제안: $q = \text{Uniform}(-2, 2)$
- 문제: $|\theta| > 2$이면 $q(\theta) = 0$이라 $\pi$의 질량 가운데 $\approx 5\%$을 놓친다

### 원칙 2: 꼬리로 덮기

제안의 꼬리가 과녁보다 **무거워야** 한다:

$$
\lim_{|\theta| \to \infty} \frac{q(\theta)}{\pi(\theta)} > 0
$$

**왜?** 꼬리가 가벼운 제안은 꼬리에서 극단적인 무게를 내어 흩어짐을 터뜨릴 수 있다.

**실전 규칙:**

- $\pi$이 가우스면 $t$분포나 더 넓은 가우스를 써라
- $\pi$이 $t$분포면 자유도가 더 작은 $t$을 써라
- 망설여지면 더 무거운 꼬리를 써라

### 원칙 3: 모양 맞추기

제안은 피적분 함수의 모양을 어림해야 한다:

$$
q(\theta) \approx c \cdot |h(\theta)| \pi(\theta)
$$

여기서 $c$은 어떤 상수이다.

**전략:**

1. 자리(평균)와 눈금(흩어짐)을 $\pi$에 맞춘다
2. 꼬리 확률이라면 제안을 꼬리 쪽으로 옮긴다
3. $\pi$의 봉우리가 여럿이면 섞음 제안을 쓴다

### 원칙 4: 셈으로 할 만하기

제안은 다음을 만족해야 한다:

1. **표집하기 쉬움**: 무작위 수를 효율적으로 만들 수 있음
2. **값 매기기 쉬움**: $q(\theta)$을 닫힌 꼴로 셈할 수 있음
3. **되도록 둘 다**: 여러 표준 분포가 이를 만족한다

---

## 3. 흔한 제안 갈래

### 가우스 제안

**꼴:** $q(\theta) = \mathcal{N}(\mu_q, \Sigma_q)$

**장점:**

- 표집과 값 매기기가 단순하다
- 성질이 잘 알려져 있다
- 봉우리가 하나인 과녁에 잘 듣는다

**매개변수 고르기:**

- $\mu_q$: 뒤확률 평균 어림값(또는 앞확률 평균)
- $\Sigma_q$: 살짝 부풀린 뒤확률 공분산

```python
import torch
import torch.distributions as dist

def gaussian_proposal_from_laplace(theta_map, hessian_at_map, inflation=1.2):
    """
    라플라스 어림으로 가우스 제안 만들기.
    
    매개변수
    ----------
    theta_map : torch.Tensor
        최대 뒤확률 어림값
    hessian_at_map : torch.Tensor
        MAP에서 로그 뒤확률의 음수의 헤세 행렬
    inflation : float
        튼튼함을 위해 공분산을 부풀리는 배수
        
    반환값
    -------
    proposal : torch.distributions.Normal or MultivariateNormal
        MAP에 가운데를 맞춘 가우스 제안
    """
    # 공분산 = 로그 뒤확률의 음수의 헤세 행렬의 역행렬
    cov = torch.inverse(hessian_at_map)
    
    # 튼튼함을 위해 공분산 부풀리기
    cov = inflation * cov
    
    if theta_map.dim() == 0 or theta_map.numel() == 1:
        return dist.Normal(theta_map, torch.sqrt(cov.squeeze()))
    else:
        return dist.MultivariateNormal(theta_map, cov)
```

### 스튜던트 t 제안

**꼴:** $q(\theta) = t_\nu(\mu_q, \Sigma_q)$

**장점:**

- 가우스보다 꼬리가 무겁다
- 자유도 $\nu$으로 다스린다
- 무게가 터질 위험을 줄인다

**매개변수 고르기:**

- $\nu = 3$에서 $5$: 무거운 꼬리
- $\nu > 30$: 거의 가우스
- $\mu_q, \Sigma_q$: 가우스 제안과 같다

```python
def student_t_proposal(location, scale, df=4):
    """
    자유도를 정한 스튜던트 t 제안.
    
    메모: PyTorch에는 일변량 StudentT이 있다.
    다변량에서는 크기 섞음 표현을 쓴다.
    """
    return dist.StudentT(df=df, loc=location, scale=scale)
```

### 섞음 제안

**꼴:** $q(\theta) = \sum_{k=1}^K \alpha_k q_k(\theta)$

**장점:**

- 봉우리가 여럿인 과녁을 다룬다
- 모양을 유연하게 어림한다
- 성분마다 단순해도 된다

**표집:**

1. 확률 $\alpha_k$으로 성분 $k$을 뽑는다
2. $\theta \sim q_k(\theta)$을 뽑는다

**밀도 값 매기기:**

$$q(\theta) = \sum_{k=1}^K \alpha_k q_k(\theta)$$

```python
class MixtureProposal:
    """
    중요도 표집 제안으로서의 분포 섞음.
    """
    
    def __init__(self, components, weights):
        """
        매개변수
        ----------
        components : list of distributions
            성분 분포 q_k
        weights : torch.Tensor
            섞음 무게 α_k(고르게 될 것이다)
        """
        self.components = components
        self.weights = weights / weights.sum()
        self.n_components = len(components)
        
    def sample(self, n_samples):
        """섞음에서 표집하기."""
        # 성분 번호 표집
        indices = torch.multinomial(self.weights, n_samples, replacement=True)
        
        # 고른 성분에서 표집
        samples = []
        for k in range(self.n_components):
            n_k = (indices == k).sum().item()
            if n_k > 0:
                samples_k = self.components[k].sample((n_k,))
                samples.append(samples_k)
        
        # 합치고 섞기
        samples = torch.cat(samples, dim=0)
        perm = torch.randperm(n_samples)
        return samples[perm]
    
    def log_prob(self, theta):
        """섞음의 로그 밀도 값 매기기."""
        log_probs = []
        for k, (comp, alpha) in enumerate(zip(self.components, self.weights)):
            log_probs.append(torch.log(alpha) + comp.log_prob(theta))
        
        log_probs = torch.stack(log_probs, dim=-1)
        return torch.logsumexp(log_probs, dim=-1)

# 보기: 봉우리 둘인 과녁
# 봉우리에 맞춘 섞음 제안 만들기
mixture_proposal = MixtureProposal(
    components=[
        dist.Normal(-3.0, 1.2),  # 첫 봉우리의 성분
        dist.Normal(3.0, 1.2)   # 둘째 봉우리의 성분
    ],
    weights=torch.tensor([0.5, 0.5])
)
```

### 앞확률을 제안으로

**꼴:** $q(\theta) = p(\theta)$

**장점:**

- 맞출 것이 없다
- 늘 받침을 덮는다
- 자연스러운 잣대

**한계:**

- 가능도가 많은 것을 알려 줄 때는 비효율적이다
- 자료가 크면 ESS이 아주 낮을 수 있다

**언제 쓰나:**

- 빠른 온전성 살피기
- 작은 자료 묶음(약한 가능도)
- 알려 주는 바 없는 앞확률

```python
def prior_as_proposal_is(h_function, log_likelihood, prior, n_samples):
    """
    앞확률을 제안으로 쓰는 중요도 표집.
    
    과녁: π(θ) ∝ p(y|θ) p(θ)
    제안: q(θ) = p(θ)
    무게: w(θ) = p(y|θ)
    """
    # 앞확률에서 표집
    samples = prior.sample((n_samples,))
    
    # 무게는 가능도 값 그대로이다
    log_weights = log_likelihood(samples)
    
    # 무게 고르게 하기
    log_sum = torch.logsumexp(log_weights, dim=0)
    weights = torch.exp(log_weights - log_sum)
    
    # SNIS 어림값
    h_values = h_function(samples)
    estimate = torch.sum(weights * h_values)
    
    # ESS
    ess = 1.0 / torch.sum(weights**2)
    
    return estimate, ess, samples, weights
```

---

## 4. 나아간 전략

### 라플라스 어림

최빈값에서 가우스로 뒤확률을 어림한다:

$$
q(\theta) = \mathcal{N}(\hat{\theta}_{\text{MAP}}, H^{-1})
$$

여기서 $H$은 MAP에서 음의 로그 뒤확률의 헤세 행렬이다.

```python
import torch.autograd.functional as F

def laplace_approximation(log_posterior, init_theta, lr=0.1, n_steps=1000):
    """
    뒤확률의 라플라스 어림 셈하기.
    
    반환값
    -------
    theta_map : torch.Tensor
        MAP 어림값
    cov : torch.Tensor
        어림한 뒤확률 공분산
    """
    theta = init_theta.clone().requires_grad_(True)
    optimizer = torch.optim.Adam([theta], lr=lr)
    
    # 기울기 오르기로 MAP 찾기
    for _ in range(n_steps):
        optimizer.zero_grad()
        loss = -log_posterior(theta)
        loss.backward()
        optimizer.step()
    
    theta_map = theta.detach()
    
    # 로그 뒤확률의 음수의 헤세 행렬 셈하기
    def neg_log_post(t):
        return -log_posterior(t)
    
    hessian = F.hessian(neg_log_post, theta_map)
    
    # 공분산은 헤세 행렬의 역행렬이다
    cov = torch.inverse(hessian)
    
    return theta_map, cov

def create_laplace_proposal(theta_map, cov, inflation=1.5):
    """
    라플라스 어림으로 제안 만들기.
    """
    inflated_cov = inflation * cov
    
    if theta_map.dim() == 0 or theta_map.numel() == 1:
        return dist.Normal(theta_map, torch.sqrt(inflated_cov.squeeze()))
    else:
        return dist.MultivariateNormal(theta_map, inflated_cov)
```

### 알아서 맞추는 중요도 표집

표본에 기대어 제안을 되풀이해 낫게 한다:

**알고리즘(모집단 몬테카를로):**

1. 제안 $q_0$의 첫걸음을 잡는다
2. $t = 1, 2, \ldots, T$에 대해:
   a. $\theta_i^t \sim q_{t-1}$을 뽑는다
   b. 무게 $w_i^t$을 셈한다
   c. 무게 준 표본에 기대어 제안 $q_t$을 새로 고친다
3. 마지막 무게 준 표본을 되돌린다

```python
class AdaptiveImportanceSampler:
    """
    가우스 섞음 제안을 쓰는 맞춰 가는 중요도 표집.
    """
    
    def __init__(self, log_target, dim, n_components=5):
        self.log_target = log_target
        self.dim = dim
        self.n_components = n_components
        
        # 넓은 가우스로 첫값 잡기
        self.means = [torch.zeros(dim)]
        self.covs = [4.0 * torch.eye(dim)]
        self.mixture_weights = torch.tensor([1.0])
        
    def run(self, n_samples_per_iter, n_iterations):
        """맞춰 가는 중요도 표집 돌리기."""
        all_samples = []
        all_weights = []
        
        for t in range(n_iterations):
            # 지금 제안에서 표집
            samples = self._sample_mixture(n_samples_per_iter)
            
            # 무게 셈하기
            log_target_vals = self.log_target(samples)
            log_proposal_vals = self._log_mixture_density(samples)
            log_weights = log_target_vals - log_proposal_vals
            weights = torch.exp(log_weights - torch.logsumexp(log_weights, 0))
            
            all_samples.append(samples)
            all_weights.append(weights)
            
            # 제안 새로 고치기
            self._update_proposal(samples, weights)
            
            # 알리기
            ess = 1.0 / torch.sum(weights**2)
            print(f"Iteration {t+1}: ESS = {ess.item():.1f} "
                  f"({ess.item()/n_samples_per_iter:.1%})")
        
        return torch.cat(all_samples), torch.cat(all_weights)
    
    def _sample_mixture(self, n):
        """지금 섞음 제안에서 표집하기."""
        weights = self.mixture_weights / self.mixture_weights.sum()
        
        samples = []
        for _ in range(n):
            k = torch.multinomial(weights, 1).item()
            sample = dist.MultivariateNormal(
                self.means[k], self.covs[k]
            ).sample()
            samples.append(sample)
        
        return torch.stack(samples)
    
    def _log_mixture_density(self, samples):
        """섞음의 로그 밀도 값 매기기."""
        log_probs = []
        weights = self.mixture_weights / self.mixture_weights.sum()
        
        for k, (mean, cov, w) in enumerate(zip(self.means, self.covs, weights)):
            comp = dist.MultivariateNormal(mean, cov)
            log_probs.append(torch.log(w) + comp.log_prob(samples))
        
        return torch.logsumexp(torch.stack(log_probs), dim=0)
    
    def _update_proposal(self, samples, weights):
        """무게 표본에 따라 섞음 제안 새로 고치기."""
        # 무게에 따라 다시 표집하기
        indices = torch.multinomial(weights, self.n_components, replacement=True)
        
        # 새 성분 평균
        new_means = [samples[i] for i in indices]
        
        # 전체 공분산 어림하기
        weighted_mean = torch.sum(weights.unsqueeze(-1) * samples, dim=0)
        weighted_cov = torch.zeros(self.dim, self.dim)
        for s, w in zip(samples, weights):
            diff = s - weighted_mean
            weighted_cov += w * torch.outer(diff, diff)
        
        # 벌주기 더하기
        weighted_cov += 0.01 * torch.eye(self.dim)
        
        # 갱신
        self.means = new_means
        self.covs = [weighted_cov for _ in range(self.n_components)]
        self.mixture_weights = torch.ones(self.n_components) / self.n_components
```

---

## 5. 제안의 질 진단

### 무게 기반 진단

```python
def proposal_diagnostics(weights, name=""):
    """
    제안의 질 두루 살피기.
    """
    n = len(weights)
    
    # 필요하면 무게 고르게 하기
    if not torch.isclose(weights.sum(), torch.tensor(1.0)):
        weights = weights / weights.sum()
    
    # ESS
    ess = 1.0 / torch.sum(weights**2)
    
    # 변이 계수
    cv = weights.std() / weights.mean()
    
    # 무게의 첨도
    mean_w = weights.mean()
    kurtosis = ((weights - mean_w)**4).mean() / ((weights - mean_w)**2).mean()**2
    
    # 최대 무게 비
    max_ratio = weights.max() * n
    
    # 무게 몰림
    sorted_w = torch.sort(weights, descending=True)[0]
    cumsum = torch.cumsum(sorted_w, dim=0)
    n_for_50 = (cumsum < 0.5).sum().item() + 1
    n_for_90 = (cumsum < 0.9).sum().item() + 1
    
    print(f"\nProposal Diagnostics: {name}")
    print("=" * 50)
    print(f"  ESS: {ess.item():.1f} / {n} ({ess.item()/n:.1%})")
    print(f"  CV of weights: {cv.item():.3f}")
    print(f"  Kurtosis: {kurtosis.item():.1f}")
    print(f"  Max weight / uniform: {max_ratio.item():.1f}x")
    print(f"  Samples for 50% weight: {n_for_50} ({n_for_50/n:.1%})")
    print(f"  Samples for 90% weight: {n_for_90} ({n_for_90/n:.1%})")
    
    # 질 살피기
    if ess.item() / n > 0.5:
        quality = "Excellent"
    elif ess.item() / n > 0.2:
        quality = "Good"
    elif ess.item() / n > 0.05:
        quality = "Acceptable"
    else:
        quality = "Poor - consider improving proposal"
    
    print(f"\n  Assessment: {quality}")
    
    return {'ess': ess.item(), 'ess_ratio': ess.item()/n, 'quality': quality}
```

ESS 기반 진단과 그 풀이를 더 깊이 다룬 것은 [실효 표본 크기](ess.md)를 보아라.

---

## 6. 실전 권고

### 제안 고르기 결정 나무

```
Is the posterior approximately Gaussian?
├── Yes → Use Laplace approximation (inflated covariance)
└── No → Is it multimodal?
    ├── Yes → Use mixture proposal (components at each mode)
    └── No → Is it heavy-tailed?
        ├── Yes → Use Student-t proposal (df = 3-5)
        └── No → Start with inflated Gaussian, check ESS
```

### 어림 규칙

| 상황 | 권하는 제안 |
|-----------|---------------------|
| 빠른 잣대 | 앞확률 |
| 봉우리 하나, 얌전함 | 라플라스 어림 |
| 꼬리가 무거울 듯함 | 스튜던트 t($\nu = 3-5$) |
| 봉우리 여럿 | 가우스 섞음 |
| 차원 높음 | 변분 어림 |
| 좋은 첫 짐작이 없음 | 알아서 맞추는 중요도 표집 |

### ESS 목표

| ESS/n | 질 | 할 일 |
|-------|---------|--------|
| > 0.5 | 아주 좋음 | 할 일 없음 |
| 0.2-0.5 | 좋음 | 대부분의 쓰임새에 넉넉함 |
| 0.05-0.2 | 아슬아슬함 | 낫게 할 것을 생각해 보아라 |
| < 0.05 | 나쁨 | 제안을 반드시 낫게 해야 함 |
| < 0.01 | 무너짐 | 결과가 미덥지 않음 |

---

## 7. 계량 금융에서의 쓰임새

### 꼬리 위험을 위한 제안 설계

계량 금융에서는 어림하려는 위험 잣대에 맞춰 중요도 표집 제안을 꼼꼼히 짜야 한다. 가장 좋은 제안은 함수 $h(\theta)$에 크게 기댄다:

| 위험 잣대 | $h(\theta)$ | 제안 전략 |
|-------------|------------|-------------------|
| 수준 $\alpha$의 VaR | $\mathbb{1}(L > \text{VaR}_\alpha)$ | 제안을 VaR 문턱값 너머로 옮긴다 |
| 기대 부족액 | $L \cdot \mathbb{1}(L > \text{VaR}_\alpha)$ | 제안의 가운데를 조건부 꼬리에 둔다 |
| 신용 손실(드문 부도) | $\mathbb{1}(\text{부도})$ | 부도 경계 쪽으로 지수 기울이기 |
| 옵션 값 매기기(깊은 외가격) | $(S_T - K)^+$ | 행사가 쪽으로 흐름 조정 |

**금융을 위한 지수 기울이기:**

금융 쓰임새에 특히 잘 듣는 전략은 지수 기울이기(에셔 변환이라고도 한다)로, 위험 인자의 분포를 손실 구역 쪽으로 옮긴다:

$$
q(\theta) = \frac{e^{\lambda \cdot \theta} \pi(\theta)}{\mathbb{E}_\pi[e^{\lambda \cdot \theta}]}
$$

기울이기 매개변수 $\lambda$은 $q$의 평균을 손실 문턱값에 두거나 그 가까이에 두도록 고른다. 가우스 위험 인자에서는 평균을 옮기는 것과 같다. 곧 제안은 흩어짐이 같고 평균만 옮겨진 가우스로 남는다.

```python
def exponential_tilting_proposal(target_dist, tilt_parameter):
    """
    가우스 과녁을 위한 지수 기울이기.
    
    N(μ, σ²)을 λ만큼 기울이면 N(μ + λσ², σ²)이 된다.
    """
    if isinstance(target_dist, dist.Normal):
        new_mean = target_dist.loc + tilt_parameter * target_dist.scale**2
        return dist.Normal(new_mean, target_dist.scale)
    else:
        raise NotImplementedError("Tilting implemented for Normal only")

# 보기: 표준 정규를 3σ 꼬리 쪽으로 기울이기
target = dist.Normal(0.0, 1.0)
tilted = exponential_tilting_proposal(target, tilt_parameter=3.0)
print(f"Original mean: {target.loc}, Tilted mean: {tilted.loc}")
```

**출력:**

```
Original mean: 0.0, Tilted mean: 3.0
```

---

## 8. 핵심 정리

!!! success "좋은 제안의 성격"

    - 과녁의 받침 전체를 덮는다
    - 과녁보다 꼬리가 무겁다
    - 모양이 $|h(\theta)|\pi(\theta)$과 맞는다
    - 표집과 값 매기기가 쉽다

!!! warning "흔히 빠지는 함정"

    - 제안의 꼬리가 과녁보다 가볍다 → 무게가 터진다
    - 봉우리를 놓친다 → 봉우리가 여럿인 과녁에서 끝없는 치우침
    - 제안이 너무 좁다 → 잘 덮지 못한다
    - 제안이 너무 넓다 → ESS이 낮다(그래도 안전하다)

!!! tip "실전 일머리"

    1. 단순한 제안(앞확률이나 라플라스)에서 시작한다
    2. ESS과 무게 진단을 살핀다
    3. ESS이 너무 낮으면 제안을 낫게 한다
    4. 복잡한 과녁에는 알아서 맞추는 방법을 생각해 보아라

---

## 연습문제

### 연습 1: 꼬리 어긋남
$\pi = t_3(0, 1)$(자유도 3의 스튜던트 t)에서 제안 (a) $\mathcal{N}(0, 1.5)$, (b) $t_3(0, 1.5)$, (c) $t_5(0, 1.5)$의 ESS을 견주어라. 결과를 설명하여라.

### 연습 2: 봉우리 찾기
$\pi = 0.3 \mathcal{N}(-5, 1) + 0.7 \mathcal{N}(3, 0.5)$의 섞음 제안을 짜라. 가우스 하나짜리 제안과 ESS을 견주어라.

### 연습 3: 알아서 다듬기
단순한 알아서 맞추는 방식을 구현하여라. (1) 첫 제안으로 중요도 표집을 돌리고, (2) 무게 준 표본에 가우스를 맞추고, (3) 되풀이한다. 되풀이마다 ESS이 나아지는 것을 좇아라.

### 연습 4: 옵션 값 매기기를 위한 지수 기울이기
블랙-숄즈 아래 $S_0 = 100$, $K = 130$, $\sigma = 0.2$, $T = 0.25$인 유럽식 콜 옵션에서, 로그 수익률 분포를 행사가 쪽으로 옮기는 지수 기울인 제안을 짜라. 소박한 몬테카를로 및 평균만 옮긴 단순 가우스 제안과 흩어짐 줄임을 견주어라.

## 정리하며

이 마당은 가장 좋은 제안、설계 원칙、흔한 제안 갈래、나아간 전략을 차례로 짚었다.

**참고 문헌**

1. Owen, A. B. (2013). *Monte Carlo theory, methods and examples*. 9.5절: 제안 분포.

2. Cappé, O., Guillin, A., Marin, J. M., & Robert, C. P. (2004). "Population Monte Carlo." *Journal of Computational and Graphical Statistics*, 13(4), 907-929.

3. Cornuet, J. M., Marin, J. M., Mira, A., & Robert, C. P. (2012). "Adaptive multiple importance sampling." *Scandinavian Journal of Statistics*, 39(4), 798-812.

4. Bugallo, M. F., Elvira, V., Martino, L., Luengo, D., Miguez, J., & Djuric, P. M. (2017). "Adaptive importance sampling: The past, the present, and the future." *IEEE Signal Processing Magazine*, 34(4), 60-79.

5. Glasserman, P., Heidelberger, P., & Shahabuddin, P. (1999). "Asymptotically optimal importance sampling and stratification for pricing path-dependent options." *Mathematical Finance*, 9(2), 117-152.
