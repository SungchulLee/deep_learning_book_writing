# 제안 분포 설계

제안분포 설계는 지지집합과 꼬리, 관심 함수의 기여, 표본 생성·밀도 평가 비용을 함께 고려하는 문제다. 목표분포와 가까운 모양만으로 유한 분산이나 모든 적분의 효율이 보장되지는 않는다. 표준 IS와 SNIS의 최적 제안, 국소 근사·혼합·적응 방식의 차이를 살펴본다.

## 1. 가장 좋은 제안

정규화된 π에서 I=E_πh를 구하는 **표준 IS**의 최적 제안은 0<A=E_π|h|<∞일 때 다음과 같다.

$$q^*(\theta)=\frac{|h(\theta)|\pi(\theta)}A$$

q*>0인 영역에서 웨이트는 w=π/q*=A/|h|로 음이 아니다. 부호는 웨이트가 아니라 h w=A sign(h)에 들어간다. h≥0이면 I=A이고 표준 추정량의 분산은 0이다. 부호가 바뀌면 최소 분산은 (A²−I²)/n이다.

**SNIS**의 점근 최적 제안은 |h−I|π에 비례하므로 위 식과 다르다. 둘 다 미지 적분과 표본 생성의 어려움 때문에 실전에서 직접 쓰기 힘들지만 어떤 영역에 표본을 배분할지 알려준다. 여러 h를 동시에 계산하려면 특정 한 함수의 최적 제안이 다른 함수의 지지집합을 놓치지 않는지도 확인해야 한다.

## 2. 설계 원칙

### 지지집합과 적률

표준 IS의 한 적분에서는 hπ≠0인 영역을 q가 덮어야 한다. SNIS와 일반적인 표본 재사용에는 π 전체 지지집합을 덮도록 한다. N(0,1) 목표에 Uniform(−2,2) 제안을 쓰면 범위 밖 기여를 계산할 수 없다. 누락 바이어스는 그 적분에 달려 있으며 반드시 무한하지는 않다.

지지집합을 덮어도 표준 IS의 $\int h^2\pi^2/q$ 또는 SNIS의 $\int(h-I)^2\pi^2/q$가 무한할 수 있다. 제안 꼬리를 무겁게 하는 것은 유용한 방어 전략이지만 q/π의 양의 극한이 모든 문제의 필요조건은 아니다. 반대로 지나치게 넓은 제안도 항상 안전하거나 효율적인 것은 아니다.

### 위치·스케일·다봉성

목표의 중심·공분산을 근사하고, 희귀 사건이면 해당 영역으로 이동시키는 전략을 쓸 수 있다. 다봉 목표에는 여러 성분을 두고, 추정된 작은 성분이 사라지지 않도록 방어적 혼합을 고려한다. 웨이트만이 아니라 관심 함수의 오차를 비교한다.

### 계산 가능한 제안

표본을 생성할 수 있고 그 표본이 나온 **동일한 q의 정규화 밀도**를 평가할 수 있어야 한다. 일반적으로 혼합 전체 밀도를 분모에 사용한다. 제안의 적응·사건 차원·좌표 변환의 야코비안을 명확히 정의한다.

## 3. 흔한 제안분포

### 정규 제안

단봉이고 국소 정규 근사가 적절하면 N(μ_q,Σ_q)를 사용할 수 있다. μ_q를 사후 모드·평균 근사에 두고 공분산을 일부 확대할 수 있지만 지지집합·꼬리의 유한 이차 적률을 확인한다. 공분산 확대 계수는 분산에 곱하는 값이며 표준편차에는 그 제곱근이 적용된다. 헤시안 역행렬에는 양의 정부호와 수치 안정성이 필요하다.

```python
import torch
import torch.distributions as dist

def gaussian_proposal_from_laplace(theta_map, hessian_at_map, inflation=1.2):
    """
    라플라스 어림으로 가우스 제안 만들기.
    
    파라메터
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

t분포는 정규분포보다 다항 꼬리가 무겁다. 스케일 행렬은 공분산이 아니며 ν>2일 때 공분산은 ν/(ν−2)배다. 평균은 ν>1일 때 존재한다. 유효한 밀도·샘플링은 더 작은 양의 ν에서도 가능하다.

아래 StudentT는 스칼라 또는 좌표별 독립 t분포를 만든다. 벡터 loc·scale을 준다고 상관을 가진 타원형 다변량 t가 되지는 않는다. 독립 좌표의 log_prob는 사건 차원을 합산해야 한다. 자유도를 크게 하면 정규에 가까워져도 모든 꼬리 확률에서 근사가 충분하다고 단정하지 않는다.

```python
def student_t_proposal(location, scale, df=4):
    """
    자유도를 정한 스튜던트 t 제안.
    
    메모: PyTorch에는 일변량 StudentT이 있다.
    다변량에서는 크기 섞음 표현을 쓴다.
    """
    return dist.StudentT(df=df, loc=location, scale=scale)
```

### 혼합 제안분포

$$q(\theta)=\sum_k\alpha_kq_k(\theta),\qquad\alpha_k\geq0,\quad\sum_k\alpha_k=1$$

성분을 α 확률로 선택하고 해당 성분에서 샘플링한다. 표준 혼합 제안의 웨이트 분모는 선택한 성분의 밀도만이 아니라 혼합 전체 q다. 로그밀도는 log-sum-exp로 계산한다.

MixtureProposal.sample은 정수 n_samples를 받는다. 일반 torch 분포처럼 sample((n,))을 받는 인터페이스가 아니므로 다른 IS 함수에 그대로 대입하면 맞지 않는다. 성분·웨이트의 길이·형태와 공통 사건 차원을 확인하고, n=0이나 모두 0인 혼합 웨이트를 별도 처리한다.

```python
class MixtureProposal:
    """
    중요도 표집 제안으로서의 분포 섞음.
    """
    
    def __init__(self, components, weights):
        """
        파라메터
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

# 보기: 봉우리 둘인 타깃
# 봉우리에 맞춘 섞음 제안 만들기
mixture_proposal = MixtureProposal(
    components=[
        dist.Normal(-3.0, 1.2),  # 첫 봉우리의 성분
        dist.Normal(3.0, 1.2)   # 둘째 봉우리의 성분
    ],
    weights=torch.tensor([0.5, 0.5])
)
```

### 사전분포를 제안으로 사용하기

적절한 사전분포에서 표본을 얻으면 사후분포의 지지집합을 덮을 수 있고 원시 웨이트는 우도가 된다. 사후가 사전의 매우 좁은 영역에 집중하면 비효율적일 수 있으며 표본 수가 작다는 사실만으로 좋은 제안이 보장되지는 않는다. 부적절 사전에서는 정규화된 제안분포로 샘플링할 수 없다.

아래 prior_as_proposal_is는 우도 웨이트를 정규화하므로 표준 IS가 아닌 SNIS 평균을 반환한다. 함수 이름과 추정량의 성질을 구분한다.

```python
def prior_as_proposal_is(h_function, log_likelihood, prior, n_samples):
    """
    앞확률을 제안으로 쓰는 중요도 표집.
    
    타깃: π(θ) ∝ p(y|θ) p(θ)
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

### 라플라스 근사

내부 사후 모드 θ̂와 음의 로그 사후 헤시안 H가 양의 정부호이면 N(θ̂,H⁻¹)의 국소 근사를 만들 수 있다. 다봉성·경계·특이성에는 한 모드 근사가 부적절할 수 있다.

아래 코드는 고정 단계 Adam 이후 최적화 수렴이나 헤시안의 양의 정부호를 검사하지 않는다. 진짜 0차원 스칼라에서 계산한 헤시안은 행렬이 아니므로 inverse 입력 형태를 맞춰야 한다. 변수 제약을 변환한다면 밀도의 야코비안도 포함하고, 직접 역행렬의 조건수를 확인한다.

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
    
    # 그래디언트 오르기로 MAP 찾기
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

### 적응적 중요도 샘플링

반복 t에서 q_t로 표본을 생성하고 해당 q_t로 웨이트를 계산한 뒤 이후 반복의 제안을 갱신한다. 과거 표본의 분모를 임의의 최신 제안으로 바꾸면 같은 추정량이 아니므로 적응·다중 제안 결합의 정확한 규칙을 정의한다.

AdaptiveImportanceSampler는 첫 반복에 성분 하나, 이후 가중 재표집한 중심과 공통 가중 공분산을 갖는 혼합을 만든다. 공분산에 0.01I를 더하는 것은 예제의 임의 정규화다. 각 반복의 정규화 웨이트를 그대로 이어 붙여 반환하므로 전체 합은 반복 수이며 전역 정규화 웨이트가 아니다. 이를 한 표본 집합의 SNIS로 쓰려면 결합 방식·스케일을 정의해야 한다. 제안이 매번 개선되거나 모드가 보존된다는 보장도 없다.

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

### 웨이트 기반 진단

빈·음수·0합 웨이트는 유효하지 않다. 균등 웨이트에서는 분산이 0이어서 코드의 첨도 비가 0/0일 수 있다. ESS/n의 Quality 표시는 관측 집중도의 예제용 분류이며 모드 포함이나 함수별 정확도를 인증하지 않는다.

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

ESS 기반 진단과 그 풀이를 더 깊이 다룬 것은 [유효 표본 크기](ess.md)를 보아라.

---

## 6. 실전 권고

단순한 적절한 사전 제안을 기준으로 삼고 국소 정규·두꺼운 꼬리·혼합 제안을 모형에 맞춰 비교한다. 변분 근사가 중심을 잘 맞춰도 꼬리를 과도하게 좁히면 중요도 분산이 커질 수 있다. 적응 방식도 수렴이나 ESS 개선을 자동 보장하지 않는다.

고정 ESS/n만으로 할 일이 없다고 판단하지 않는다. 지지집합·관심 함수의 가중 기여·여러 초기 제안과 반복 실험의 변동·실제 평가 비용을 함께 본다. 아래 보존된 결정 나무는 후보를 생각하는 예시이며 각 분기의 선택이 유한 분산이나 정확도를 보장하지는 않는다.

```
Is the posterior approximately Gaussian?
├── Yes → Use Laplace approximation (inflated covariance)
└── No → Is it multimodal?
    ├── Yes → Use mixture proposal (components at each mode)
    └── No → Is it heavy-tailed?
        ├── Yes → Use Student-t proposal (df = 3-5)
        └── No → Start with inflated Gaussian, check ESS
```


## 7. 꼬리 적분과 지수 틸팅

고정 로스 기준 c를 넘을 확률은 h=1[L>c], 꼬리 로스 적분은 h=L·1[L>c]로 표현할 수 있다. VaR 자체는 미지 분위수이므로 고정 c의 지시함수 기대값 하나와 동일하지 않으며 가중 누적분포나 근 찾기가 필요하다. ES도 분위수와 이산 질량의 정의를 맞춰 계산한다.

적률생성함수 M(λ)=E_π exp(λθ)가 유한한 λ에서 지수 틸팅을 정의한다.

$$q_\lambda(\theta)=\frac{e^{\lambda\theta}\pi(\theta)}{M(\lambda)}$$

정규 N(μ,σ²)를 틸팅하면 N(μ+λσ²,σ²)가 된다. 이 경우 지수 틸팅과 같은 평균 이동의 정규 제안은 **같은 분포**이므로 서로 다른 방법의 우열처럼 비교하지 않는다. t분포처럼 양의 λ에서 M(λ)가 발산하는 목표에는 이 틸팅을 그대로 정의할 수 없다. 금융이라는 이유만으로 항상 가능하거나 최적인 것은 아니다.

아래 함수는 torch의 스칼라 Normal에서만 평균을 이동시킨다. 표본을 틸팅 분포에서 생성하더라도 원래 목표에 대한 밀도비를 적용해야 적분을 복원한다.

```python
def exponential_tilting_proposal(target_dist, tilt_parameter):
    """
    가우스 타깃을 위한 지수 기울이기.
    
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

제안 설계는 관심 적분에 필요한 지지집합과 이차 적률을 만족하는 것에서 시작한다. 표준 IS와 SNIS의 최적 제안을 구분하고, 스케일·사건 차원·좌표 변환과 전체 혼합밀도를 일치시킨다.

라플라스는 국소 근사, 적응은 자료 기반 제안 갱신이며 각각 수렴·꼬리·가중 결합 조건이 필요하다. 제안이 넓거나 ESS가 높다는 사실만으로 안전하거나 정확하다고 판정하지 않는다.

## 연습문제

1. π=t₃(0,scale=1)에서 정규 N(0,1.5²), t₃(scale=1.5), t₅(scale=1.5) 제안의 가중 이차 적률을 분석하라. t₅가 목표보다 꼬리가 가벼워도 웨이트 이차 적률은 유한할 수 있음을 보여라. 관심 함수가 h=1·θ·θ²로 바뀌면 조건이 어떻게 달라지는가?
2. π=0.3N(−5,1²)+0.7N(3,0.5²)에 대한 혼합 제안을 설계하라. 전체 혼합밀도를 웨이트 분모로 사용하고 모드별 정보·전체 ESS·관심 함수 오차를 구분해 비교하라.
3. 가중 평균·공분산으로 제안을 적응시키는 절차를 설계하라. 각 표본의 생성 제안, 원시 웨이트의 스케일과 반복별 정규화 웨이트 결합을 명시하고 ESS가 단조 개선될 필요가 없는 이유를 설명하라.
4. 블랙-숄즈의 가상 콜 가격 문제에서 S₀=100, K=130, σ=0.2, T=0.25, 무위험금리 r=0·배당 없음으로 가정하라. 표준정규 위험요인에 지수 틸팅을 적용하고 원래 밀도비와 지급액을 적어라. 같은 평균 이동의 정규 제안이 틸팅과 동일함을 확인하고, 원래 제안과 같은 비용으로 분산을 비교할 절차를 설명하라.

## 정리하며

표준 IS와 SNIS의 최적 제안은 서로 다르며 관심 함수가 바뀌면 최적 설계도 달라진다. 지지집합·꼬리·계산 비용과 웨이트 인터페이스를 함께 점검하고 국소 근사·혼합·적응·틸팅의 조건을 명시한다. 함수별 오차와 모드 탐색은 ESS 하나로 대체할 수 없다.

**참고 문헌**

1. Owen, A. B. (2013). *Monte Carlo theory, methods and examples*. 10.5절: 제안 분포.

2. Cappé, O., Guillin, A., Marin, J. M., & Robert, C. P. (2004). "Population Monte Carlo." *Journal of Computational and Graphical Statistics*, 13(4), 907-929.

3. Cornuet, J. M., Marin, J. M., Mira, A., & Robert, C. P. (2012). "Adaptive multiple importance sampling." *Scandinavian Journal of Statistics*, 39(4), 798-812.

4. Bugallo, M. F., Elvira, V., Martino, L., Luengo, D., Miguez, J., & Djuric, P. M. (2017). "Adaptive importance sampling: The past, the present, and the future." *IEEE Signal Processing Magazine*, 34(4), 60-79.

5. Glasserman, P., Heidelberger, P., & Shahabuddin, P. (1999). "Asymptotically optimal importance sampling and stratification for pricing path-dependent options." *Mathematical Finance*, 9(2), 117-152.
