# 가우스 섞음 모형
가우스 섞음 모형(GMM)은 EM 알고리즘의 대표 쓰임새이며 지도 없는 배움의 바탕 모형이다. 이 절에서는 모형 적기, EM 이끌어 내기, 구현의 세부, 계량 금융과 이어지는 넓힘을 온전히 다룬다.

---

## 1. 모형 명세

### 낳는 이야기

가우스 섞음 모형은 자료가 가우스 분포 $K$개의 섞음에서 나왔다고 놓는다. 관측 $\mathbf{x}_i$마다:

1. **성분 배정 뽑기**: $\boldsymbol{\pi} = (\pi_1, \ldots, \pi_K)$일 때 $z_i \sim \text{Categorical}(\boldsymbol{\pi})$
2. **고른 성분에서 관측 뽑기**: $\mathbf{x}_i | z_i = k \sim \mathcal{N}(\boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)$

숨은 변수 $z_i \in \{1, \ldots, K\}$은 어느 성분이 관측 $i$을 낳았는지를 가리킨다.

### 결합 분포

관측과 숨은 변수의 결합 분포는 다음과 같다:

$$
p(\mathbf{x}_i, z_i = k | \theta) = p(z_i = k | \boldsymbol{\pi}) \, p(\mathbf{x}_i | z_i = k, \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k) = \pi_k \, \mathcal{N}(\mathbf{x}_i | \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)
$$

### 주변 분포(섞음 밀도)

숨은 변수에 걸쳐 주변으로 만들면:

$$
p(\mathbf{x}_i | \theta) = \sum_{k=1}^{K} \pi_k \, \mathcal{N}(\mathbf{x}_i | \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)
$$

이는 **가우스에 무게를 붙여 더한 것**이며 복잡하고 봉우리가 여럿인 분포도 본뜰 수 있다.

### 매개변수

온전한 매개변수 모음은 $\theta = \{\boldsymbol{\pi}, \{\boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k\}_{k=1}^K\}$이다:

| 매개변수 | 차원 | 제약 |
|-----------|-----------|------------|
| $\pi_k$(섞음 무게) | 스칼라 | $\pi_k \geq 0$, $\sum_k \pi_k = 1$ |
| $\boldsymbol{\mu}_k$(평균) | $d \times 1$ | 없음 |
| $\boldsymbol{\Sigma}_k$(공분산) | $d \times d$ | 양의 정부호이며 대칭 |

### 매개변수의 수

$d$차원에서 성분이 $K$개인 가우스 섞음 모형에서:

| 공분산 갈래 | 성분마다의 매개변수 | 전체 매개변수 |
|-----------------|-------------------------|------------------|
| 온전함 | $d + d(d+1)/2$ | $K(d + d(d+1)/2) + (K-1)$ |
| 대각 | $d + d = 2d$ | $K \cdot 2d + (K-1)$ |
| 구면 | $d + 1$ | $K(d+1) + (K-1)$ |
| 묶음(함께 씀) | $d$ | $Kd + d(d+1)/2 + (K-1)$ |

---

## 2. 로그 가능도 함수

### 완전 자료 로그 가능도

$\mathbf{X} = \{\mathbf{x}_i\}_{i=1}^N$과 $\mathbf{Z} = \{z_i\}_{i=1}^N$을 모두 관측했다면:

$$
\ell_c(\theta) = \log p(\mathbf{X}, \mathbf{Z} | \theta) = \sum_{i=1}^{N} \sum_{k=1}^{K} \mathbb{1}[z_i = k] \left[ \log \pi_k + \log \mathcal{N}(\mathbf{x}_i | \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k) \right]
$$

이는 지시자 $\mathbb{1}[z_i = k]$에 대해 **선형**이라 최적화하기 쉽다.

### 주변 로그 가능도

관측한(불완전한) 자료의 로그 가능도는 다음과 같다:

$$
\ell(\theta) = \sum_{i=1}^{N} \log p(\mathbf{x}_i | \theta) = \sum_{i=1}^{N} \log \left( \sum_{k=1}^{K} \pi_k \, \mathcal{N}(\mathbf{x}_i | \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k) \right)
$$

**합의 로그**라는 짜임 때문에 곧바른 최적화를 다룰 수 없으며, 그래서 EM이 필요해진다.

### 곧바른 최대 가능도 어림이 왜 무너지나

1. **닫힌 꼴 풀이가 없음**: $\nabla_\theta \ell = 0$으로 두면 서로 얽힌 비선형 방정식이 나온다
2. **봉우리가 여럿임**: 가능도 면에 그 자리 최댓점이 많다(이름표 자리바꿈 때문에 적어도 $K!$개)
3. **특이점**: 성분이 한 점으로 찌부러지면 가능도가 묶이지 않는다

---

## 3. 가우스 섞음 모형의 EM: 온전히 이끌어 내기

### E 걸음: 맡음 몫 셈하기

E 걸음은 관측 $i$이 성분 $k$에 들 뒤확률을 셈한다:

$$
\gamma_{ik} = p(z_i = k | \mathbf{x}_i, \theta^{(t)}) = \frac{p(z_i = k) \, p(\mathbf{x}_i | z_i = k)}{\sum_{j=1}^{K} p(z_i = j) \, p(\mathbf{x}_i | z_i = j)}
$$

모형을 넣으면:

$$
\gamma_{ik} = \frac{\pi_k^{(t)} \, \mathcal{N}(\mathbf{x}_i | \boldsymbol{\mu}_k^{(t)}, \boldsymbol{\Sigma}_k^{(t)})}{\sum_{j=1}^{K} \pi_j^{(t)} \, \mathcal{N}(\mathbf{x}_i | \boldsymbol{\mu}_j^{(t)}, \boldsymbol{\Sigma}_j^{(t)})}
$$

**풀이**: $\gamma_{ik}$은 성분 $k$이 관측 $i$을 설명하는 데 지는 "맡음 몫"이다.

### 기댓값 충분 통계량

성분 $k$에 배정된 점의 실효 개수를 다음과 같이 정한다:

$$
N_k = \sum_{i=1}^{N} \gamma_{ik}
$$

메모: 관측마다 맡음 몫의 합이 1이므로 $\sum_{k=1}^{K} N_k = N$이다.

### M 걸음: 매개변수 새로 고치기

**Q 함수**:

$$
Q(\theta | \theta^{(t)}) = \sum_{i=1}^{N} \sum_{k=1}^{K} \gamma_{ik} \left[ \log \pi_k + \log \mathcal{N}(\mathbf{x}_i | \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k) \right]
$$

가우스를 펼치면:

$$
Q = \sum_{i,k} \gamma_{ik} \log \pi_k - \frac{1}{2} \sum_{i,k} \gamma_{ik} \left[ d \log(2\pi) + \log|\boldsymbol{\Sigma}_k| + (\mathbf{x}_i - \boldsymbol{\mu}_k)^\top \boldsymbol{\Sigma}_k^{-1} (\mathbf{x}_i - \boldsymbol{\mu}_k) \right]
$$

**섞음 비율**($\sum_k \pi_k = 1$ 제약 아래의 최적화):

$$
\boxed{\pi_k^{(t+1)} = \frac{N_k}{N}}
$$

**평균**(무게 평균):

$$
\boxed{\boldsymbol{\mu}_k^{(t+1)} = \frac{1}{N_k} \sum_{i=1}^{N} \gamma_{ik} \, \mathbf{x}_i}
$$

**공분산**(무게 붙인 경험 공분산):

$$
\boxed{\boldsymbol{\Sigma}_k^{(t+1)} = \frac{1}{N_k} \sum_{i=1}^{N} \gamma_{ik} \, (\mathbf{x}_i - \boldsymbol{\mu}_k^{(t+1)})(\mathbf{x}_i - \boldsymbol{\mu}_k^{(t+1)})^\top}
$$

### 알고리즘 간추림

```
Input: Data X, number of components K
Output: Parameters θ = {π, μ, Σ}

1. Initialize parameters θ⁽⁰⁾
2. Repeat until convergence:
   
   # E 걸음: 맡음 몫 셈하기
   For each i, k:
       γ_ik ← π_k N(x_i | μ_k, Σ_k) / Σ_j π_j N(x_i | μ_j, Σ_j)
   
   # M 걸음: 매개변수 새로 고치기
   For each k:
       N_k ← Σ_i γ_ik
       π_k ← N_k / N
       μ_k ← (1/N_k) Σ_i γ_ik x_i
       Σ_k ← (1/N_k) Σ_i γ_ik (x_i - μ_k)(x_i - μ_k)ᵀ
   
   # 모임 살피기
   If |ℓ(θ⁽ᵗ⁺¹⁾) - ℓ(θ⁽ᵗ⁾)| < tolerance: break

3. Return θ
```

---

## 4. 첫값 잡기 전략

### 첫값 잡기의 중요함

가우스 섞음 모형의 가능도 면은 봉우리가 몹시 많다. 첫값을 잘못 잡으면 다음이 생길 수 있다:

- 어설픈 그 자리 최댓점으로의 모임
- 느린 모임
- 성분의 찌부러짐이나 무너짐

### k 평균으로 첫값 잡기

k 평균 무리짓기로 첫값을 잡는다:

1. 자료에 k 평균을 돌려 무리의 가운데와 배정을 얻는다
2. $\boldsymbol{\mu}_k$을 k 평균의 가운데로 둔다
3. $\boldsymbol{\Sigma}_k$을 배정된 점의 경험 공분산으로 둔다
4. $\pi_k$을 무리 $k$에 든 점의 비율로 둔다

**좋은 점**: 빠르고 (k 평균의 씨앗이 정해지면) 정해져 있으며 그럴듯한 출발점을 준다

### k 평균++으로 첫값 잡기

k 평균의 가운데 고르기를 낫게 한 것이다:

1. 자료에서 첫 가운데를 고르게 무작위로 고른다
2. 그다음 가운데마다 이미 있는 가장 가까운 가운데까지의 거리 제곱에 비례하는 확률로 점을 고른다
3. 이 가운데들에서 k 평균을 이어 간다

이러면 첫 가운데들이 널리 퍼져 서로 너무 가까이 놓이는 것을 피한다.

### 무작위 초기화

단순하지만 여러 번 다시 시작해야 한다:

1. $\boldsymbol{\mu}_k$: 자료 점의 무작위 부분 모음이나 자료 범위에서 표집
2. $\boldsymbol{\Sigma}_k$: 자료의 흩어짐으로 크기를 잡은 항등 행렬
3. $\pi_k$: 고름($1/K$)이나 디리클레에서 무작위

### 층으로 첫값 잡기

성분을 적게 시작해 쪼개 나간다:

1. 성분이 $K' < K$개인 가우스 섞음 모형을 맞춘다
2. 가장 크거나 가장 널리 퍼진 성분을 쪼갠다
3. 성분이 $K$개가 될 때까지 되풀이한다

---

## 5. 모형 고르기

### 성분의 개수 고르기

가장 좋은 $K$은 알 수 없으므로 골라야 한다. 흔한 길은 다음과 같다:

### 정보 기준

**베이즈 정보 잣대(BIC)**:

$$
\text{BIC} = -2 \ell(\hat{\theta}) + p \log N
$$

여기서 $p$은 매개변수의 개수이다. BIC는 $N$이 크면 복잡함에 더 무겁게 벌을 준다.

**아카이케 정보 잣대(AIC)**:

$$
\text{AIC} = -2 \ell(\hat{\theta}) + 2p
$$

AIC는 BIC보다 더 복잡한 모형을 고르는 경향이 있다.

**적분 완전 가능도(ICL)**:

$$
\text{ICL} = \text{BIC} - 2 \sum_{i=1}^{N} \sum_{k=1}^{K} \hat{\gamma}_{ik} \log \hat{\gamma}_{ik}
$$

ICL은 엔트로피 벌을 더해 무리 배정이 더 또렷한 모형을 좋아한다.

### 고르는 절차

1. $K = 1, 2, \ldots, K_{\max}$에 대해 가우스 섞음 모형을 맞춘다
2. 저마다 정보 잣대를 셈한다
3. 그 잣대를 가장 작게 하는 $K$을 고른다

```
K_values = range(1, K_max + 1)
bic_scores = []

for K in K_values:
    gmm = fit_gmm(X, K)
    n_params = K * (1 + d + d*(d+1)/2) - 1  # 온전한 공분산
    bic = -2 * gmm.log_likelihood(X) + n_params * log(N)
    bic_scores.append(bic)

best_K = K_values[argmin(bic_scores)]
```

### 교차 확인

남겨 둔 자료의 로그 가능도:

1. 자료를 익힘용과 확인용으로 나눈다
2. 익힘 자료에 가우스 섞음 모형을 맞춘다
3. 확인 자료에서 로그 가능도를 매긴다
4. 확인 가능도를 가장 크게 하는 $K$을 고른다

---

## 6. 공분산의 제약

### 온전한 공분산

성분마다 옭매이지 않은 $d \times d$ 양의 정부호 공분산을 저마다 갖는다:

$$
\boldsymbol{\Sigma}_k^{(t+1)} = \frac{1}{N_k} \sum_{i=1}^{N} \gamma_{ik} (\mathbf{x}_i - \boldsymbol{\mu}_k)(\mathbf{x}_i - \boldsymbol{\mu}_k)^\top
$$

**좋은 점**: 유연함이 가장 크고 상관을 담아낸다
**나쁜 점**: 매개변수가 가장 많고 특이해질 위험이 있으며 자료가 더 필요하다

### 대각 공분산

대각 행렬 $\boldsymbol{\Sigma}_k = \text{diag}(\sigma_{k1}^2, \ldots, \sigma_{kd}^2)$으로 옭아맨다:

$$
\sigma_{kj}^{2(t+1)} = \frac{1}{N_k} \sum_{i=1}^{N} \gamma_{ik} (x_{ij} - \mu_{kj})^2
$$

**좋은 점**: 매개변수가 적고 더 안정하다
**나쁜 점**: 무리 안에서 특징이 서로 상관없다고 놓는다

### 구면 공분산

성분마다 흩어짐 하나인 $\boldsymbol{\Sigma}_k = \sigma_k^2 \mathbf{I}$:

$$
\sigma_k^{2(t+1)} = \frac{1}{N_k \cdot d} \sum_{i=1}^{N} \gamma_{ik} \|\mathbf{x}_i - \boldsymbol{\mu}_k\|^2
$$

**좋은 점**: 매개변수가 가장 적고 가장 안정하다
**나쁜 점**: 무리가 방향에 고르다고 놓는다

### 묶은 공분산

모든 성분이 같은 공분산 $\boldsymbol{\Sigma}_k = \boldsymbol{\Sigma}$을 함께 쓴다:

$$
\boldsymbol{\Sigma}^{(t+1)} = \frac{1}{N} \sum_{k=1}^{K} \sum_{i=1}^{N} \gamma_{ik} (\mathbf{x}_i - \boldsymbol{\mu}_k)(\mathbf{x}_i - \boldsymbol{\mu}_k)^\top
$$

**좋은 점**: 벌주기 효과가 있고 어림이 안정하다
**나쁜 점**: 옭아매는 가정이다

---

## 7. 벌주기와 특이점

### 특이점 문제

성분의 공분산이 특이해지면(행렬식이 0으로 다가가면) 가능도가 묶이지 않는다:

$$
\mathcal{N}(\mathbf{x}_i | \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k) \propto |\boldsymbol{\Sigma}_k|^{-1/2} \to \infty \text{ as } |\boldsymbol{\Sigma}_k| \to 0
$$

이는 다음일 때 일어난다:

- 성분이 찌부러져 자료 점 하나에 딱 맞춰질 때
- 성분에 실효로 배정된 점이 $d+1$개보다 적을 때
- 자료가 더 낮은 차원의 부분 공간에 놓일 때

### 공분산에 벌주기

**대각에 얹기**(능선 벌주기):

$$
\boldsymbol{\Sigma}_k \leftarrow \boldsymbol{\Sigma}_k + \lambda \mathbf{I}
$$

대각에 작은 양수를 더해 양의 정부호임을 보장한다.

**최소 고윳값 제약**:

$$
\boldsymbol{\Sigma}_k = \mathbf{U} \max(\boldsymbol{\Lambda}, \epsilon \mathbf{I}) \mathbf{U}^\top
$$

여기서 $\mathbf{U}\boldsymbol{\Lambda}\mathbf{U}^\top$은 고유 분해이다.

**오그라들기 어림꼴**(르두아-울프):

$$
\boldsymbol{\Sigma}_k^{\text{shrunk}} = (1 - \alpha) \boldsymbol{\Sigma}_k + \alpha \cdot \text{tr}(\boldsymbol{\Sigma}_k)/d \cdot \mathbf{I}
$$

### 베이즈 벌주기

매개변수에 앞확률을 둔다:

- $\boldsymbol{\Sigma}_k$에 **역위샤트 앞확률**: 가짜 관측 노릇을 한다
- $\boldsymbol{\pi}$에 **디리클레 앞확률**: 성분의 무게가 0이 되는 것을 막는다

MAP 어림값이 이 앞확률을 담는다:

$$
\boldsymbol{\Sigma}_k^{\text{MAP}} = \frac{N_k \boldsymbol{\Sigma}_k^{\text{MLE}} + \nu_0 \boldsymbol{\Psi}_0}{N_k + \nu_0 + d + 1}
$$

여기서 $\nu_0, \boldsymbol{\Psi}_0$은 앞확률의 웃매개변수이다.

---

## 8. PyTorch 구현

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class GaussianMixtureModel:
    """
    온전한 EM 구현을 갖춘 가우스 섞음 모형.
    
    여러 공분산 갈래와 벌주기를 받쳐 준다.
    """
    
    def __init__(
        self,
        n_components: int,
        n_features: int,
        covariance_type: str = 'full',
        reg_covar: float = 1e-6,
        init_method: str = 'kmeans',
        n_init: int = 1,
        max_iter: int = 100,
        tol: float = 1e-4,
        random_state: Optional[int] = None
    ):
        """
        인수:
            n_components: 섞음 성분의 개수(K)
            n_features: 자료의 차원(d)
            covariance_type: 'full', 'diagonal', 'spherical', 'tied' 가운데 하나
            reg_covar: 공분산의 대각에 더하는 벌주기
            init_method: 'kmeans', 'random', 'kmeans++' 가운데 하나
            n_init: 시도할 첫값 잡기의 횟수
            max_iter: 최대 EM 되풀이 횟수
            tol: 모임 너그러움
            random_state: 무작위 씨앗
        """
        self.K = n_components
        self.d = n_features
        self.covariance_type = covariance_type
        self.reg_covar = reg_covar
        self.init_method = init_method
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        
        if random_state is not None:
            torch.manual_seed(random_state)
        
        # 매개변수(fit에서 첫값을 잡는다)
        self.weights_ = None      # (K,)
        self.means_ = None        # (K, d)
        self.covariances_ = None  # 꼴은 covariance_type에 달렸다
        
        # 맞추기 정보
        self.converged_ = False
        self.n_iter_ = 0
        self.lower_bound_ = float('-inf')
        
    def _initialize_parameters(self, X: torch.Tensor):
        """가우스 섞음 모형의 매개변수 첫값 잡기."""
        N, d = X.shape
        
        if self.init_method == 'kmeans':
            # 단순한 k 평균 첫값 잡기
            indices = torch.randperm(N)[:self.K]
            self.means_ = X[indices].clone()
            
            # k 평균을 몇 번 되풀이하기
            for _ in range(10):
                # 점을 가장 가까운 가운데에 배정
                dists = torch.cdist(X, self.means_)
                assignments = dists.argmin(dim=1)
                
                # 가운데 새로 고치기
                for k in range(self.K):
                    mask = assignments == k
                    if mask.sum() > 0:
                        self.means_[k] = X[mask].mean(dim=0)
            
            # k 평균 무리로 공분산 첫값 잡기
            self._initialize_covariances(X, assignments)
            
            # 가중치 초기화
            self.weights_ = torch.bincount(
                assignments, minlength=self.K
            ).float() / N
            
        elif self.init_method == 'random':
            # 무작위 첫값 잡기
            self.means_ = X[torch.randperm(N)[:self.K]].clone()
            self.weights_ = torch.ones(self.K) / self.K
            self._initialize_covariances(X, None)
            
    def _initialize_covariances(
        self, X: torch.Tensor, assignments: Optional[torch.Tensor]
    ):
        """공분산 행렬 첫값 잡기."""
        N, d = X.shape
        
        if self.covariance_type == 'full':
            self.covariances_ = torch.stack([
                torch.eye(d) * X.var() for _ in range(self.K)
            ])
        elif self.covariance_type == 'diagonal':
            self.covariances_ = torch.stack([
                X.var(dim=0) for _ in range(self.K)
            ])
        elif self.covariance_type == 'spherical':
            self.covariances_ = torch.ones(self.K) * X.var()
        elif self.covariance_type == 'tied':
            self.covariances_ = torch.eye(d) * X.var()
            
    def _compute_log_prob(self, X: torch.Tensor) -> torch.Tensor:
        """
        관측마다 성분마다 log p(x|z=k) 셈하기.
        
        반환값:
            log_prob: 로그 확률의 (N, K) 텐서
        """
        N = X.shape[0]
        log_prob = torch.zeros(N, self.K)
        
        for k in range(self.K):
            if self.covariance_type == 'full':
                cov = self.covariances_[k]
            elif self.covariance_type == 'diagonal':
                cov = torch.diag(self.covariances_[k])
            elif self.covariance_type == 'spherical':
                cov = self.covariances_[k] * torch.eye(self.d)
            elif self.covariance_type == 'tied':
                cov = self.covariances_
                
            # 벌주기 더하기
            cov = cov + self.reg_covar * torch.eye(self.d)
            
            # 로그 확률 셈하기
            diff = X - self.means_[k]
            
            # 수치 안정을 위한 촐레스키
            try:
                L = torch.linalg.cholesky(cov)
                log_det = 2 * torch.log(torch.diag(L)).sum()
                solve = torch.linalg.solve_triangular(L, diff.T, upper=False)
                mahalanobis = (solve ** 2).sum(dim=0)
            except:
                # 곧바른 셈하기로 물러서기
                log_det = torch.logdet(cov)
                cov_inv = torch.inverse(cov)
                mahalanobis = (diff @ cov_inv * diff).sum(dim=1)
            
            log_prob[:, k] = -0.5 * (
                self.d * torch.log(torch.tensor(2 * torch.pi)) +
                log_det +
                mahalanobis
            )
            
        return log_prob
    
    def _e_step(self, X: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """
        E 걸음: 맡음 몫과 로그 가능도 셈하기.
        
        반환값:
            responsibilities: (N, K) 텐서
            log_likelihood: 스칼라
        """
        log_prob = self._compute_log_prob(X)  # (N, K)
        log_weights = torch.log(self.weights_)  # (K,)
        
        # 로그 맡음 몫(고르게 하지 않음)
        log_resp = log_prob + log_weights  # (N, K)
        
        # 고르게 하기 위한 log-sum-exp
        log_resp_norm = torch.logsumexp(log_resp, dim=1, keepdim=True)  # (N, 1)
        
        # 고르게 한 로그 맡음 몫
        log_resp = log_resp - log_resp_norm
        
        # 로그가능도
        log_likelihood = log_resp_norm.sum().item()
        
        return torch.exp(log_resp), log_likelihood
    
    def _m_step(self, X: torch.Tensor, responsibilities: torch.Tensor):
        """M 걸음: 맡음 몫이 주어졌을 때 매개변수 새로 고치기."""
        N = X.shape[0]
        
        # 실효 횟수
        N_k = responsibilities.sum(dim=0) + 1e-10  # (K,)
        
        # 가중치를 갱신한다
        self.weights_ = N_k / N
        
        # 평균을 갱신한다
        self.means_ = (responsibilities.T @ X) / N_k.unsqueeze(1)  # (K, d)
        
        # 공분산을 갱신한다
        self._update_covariances(X, responsibilities, N_k)
        
    def _update_covariances(
        self,
        X: torch.Tensor,
        responsibilities: torch.Tensor,
        N_k: torch.Tensor
    ):
        """covariance_type에 따라 공분산 행렬 새로 고치기."""
        N, d = X.shape
        
        if self.covariance_type == 'full':
            for k in range(self.K):
                diff = X - self.means_[k]  # (N, d)
                weighted_diff = responsibilities[:, k].unsqueeze(1) * diff
                self.covariances_[k] = (weighted_diff.T @ diff) / N_k[k]
                
        elif self.covariance_type == 'diagonal':
            for k in range(self.K):
                diff = X - self.means_[k]
                weighted_sq_diff = responsibilities[:, k].unsqueeze(1) * (diff ** 2)
                self.covariances_[k] = weighted_sq_diff.sum(dim=0) / N_k[k]
                
        elif self.covariance_type == 'spherical':
            for k in range(self.K):
                diff = X - self.means_[k]
                sq_dist = (diff ** 2).sum(dim=1)
                self.covariances_[k] = (responsibilities[:, k] * sq_dist).sum() / (N_k[k] * d)
                
        elif self.covariance_type == 'tied':
            self.covariances_ = torch.zeros(d, d)
            for k in range(self.K):
                diff = X - self.means_[k]
                weighted_diff = responsibilities[:, k].unsqueeze(1) * diff
                self.covariances_ += weighted_diff.T @ diff
            self.covariances_ /= N
    
    def fit(self, X: torch.Tensor, verbose: bool = False) -> 'GaussianMixtureModel':
        """
        EM 알고리즘으로 가우스 섞음 모형 맞추기.
        
        인수:
            X: 꼴이 (N, d)인 자료 텐서
            verbose: 진행 상황 출력 여부
            
        반환값:
            self
        """
        best_ll = float('-inf')
        best_params = None
        
        for init in range(self.n_init):
            # 초기화한다
            self._initialize_parameters(X)
            
            prev_ll = float('-inf')
            
            for iteration in range(self.max_iter):
                # E 걸음
                responsibilities, ll = self._e_step(X)
                
                # 모임 살피기
                if abs(ll - prev_ll) < self.tol:
                    self.converged_ = True
                    self.n_iter_ = iteration + 1
                    break
                    
                if verbose and iteration % 10 == 0:
                    print(f"Init {init+1}, Iter {iteration}: LL = {ll:.4f}")
                
                # M 걸음
                self._m_step(X, responsibilities)
                
                prev_ll = ll
            
            # 가장 좋은 첫값 남기기
            if ll > best_ll:
                best_ll = ll
                best_params = (
                    self.weights_.clone(),
                    self.means_.clone(),
                    self.covariances_.clone() if isinstance(self.covariances_, torch.Tensor)
                    else [c.clone() for c in self.covariances_]
                )
        
        # 가장 좋은 매개변수 되돌리기
        self.weights_, self.means_, self.covariances_ = best_params
        self.lower_bound_ = best_ll
        
        return self
    
    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """무리 이름표 미리보기."""
        responsibilities, _ = self._e_step(X)
        return responsibilities.argmax(dim=1)
    
    def predict_proba(self, X: torch.Tensor) -> torch.Tensor:
        """무리 확률 미리보기."""
        responsibilities, _ = self._e_step(X)
        return responsibilities
    
    def score(self, X: torch.Tensor) -> float:
        """평균 로그 가능도 셈하기."""
        _, ll = self._e_step(X)
        return ll / X.shape[0]
    
    def bic(self, X: torch.Tensor) -> float:
        """베이즈 정보 기준을 셈한다."""
        N = X.shape[0]
        _, ll = self._e_step(X)
        
        # 매개변수 개수 세기
        if self.covariance_type == 'full':
            n_params = self.K * (1 + self.d + self.d * (self.d + 1) / 2) - 1
        elif self.covariance_type == 'diagonal':
            n_params = self.K * (1 + 2 * self.d) - 1
        elif self.covariance_type == 'spherical':
            n_params = self.K * (1 + self.d + 1) - 1
        elif self.covariance_type == 'tied':
            n_params = self.K * (1 + self.d) + self.d * (self.d + 1) / 2 - 1
            
        return -2 * ll + n_params * torch.log(torch.tensor(N)).item()
    
    def sample(self, n_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """맞춘 모형에서 표본 만들기."""
        # 성분 배정 표집
        component_counts = torch.multinomial(
            self.weights_, n_samples, replacement=True
        )
        
        samples = []
        labels = []
        
        for k in range(self.K):
            n_k = (component_counts == k).sum().item()
            if n_k > 0:
                if self.covariance_type == 'full':
                    cov = self.covariances_[k]
                elif self.covariance_type == 'diagonal':
                    cov = torch.diag(self.covariances_[k])
                elif self.covariance_type == 'spherical':
                    cov = self.covariances_[k] * torch.eye(self.d)
                elif self.covariance_type == 'tied':
                    cov = self.covariances_
                
                cov = cov + self.reg_covar * torch.eye(self.d)
                
                dist = torch.distributions.MultivariateNormal(
                    self.means_[k], cov
                )
                samples.append(dist.sample((n_k,)))
                labels.extend([k] * n_k)
        
        X = torch.cat(samples, dim=0)
        y = torch.tensor(labels)
        
        # 뒤섞는다
        perm = torch.randperm(n_samples)
        return X[perm], y[perm]
```

---

## 9. 계량 금융에서의 쓰임

### 수익 분포 본뜨기

금융 수익은 가우스가 아닌 성질(두꺼운 꼬리, 치우침)을 보이며 가우스 섞음 모형이 이를 담아낼 수 있다:

$$
p(r_t) = \sum_{k=1}^{K} \pi_k \, \mathcal{N}(r_t | \mu_k, \sigma_k^2)
$$

**풀이**: 성분마다 시장의 한 국면(보통, 출렁임, 위기)을 나타낸다.

### 국면 알아내기

가우스 섞음 모형으로 시장 국면을 가려낸다:

1. 수익 시계열에 가우스 섞음 모형을 맞춘다
2. 때마다 맡음 몫을 셈한다
3. 맡음 몫이 가장 큰 것으로 국면을 배정한다

그러면 국면에 따른 전략이 알아낸 상태를 조건으로 삼을 수 있다.

### 위험 재기

가우스 섞음 모형에 바탕을 둔 위험 가치:

$$
\text{VaR}_\alpha = \text{quantile}\left( \sum_{k=1}^{K} \pi_k \, F_k^{-1}(\alpha) \right)
$$

여기서 $F_k$은 성분 $k$의 누적분포함수이다. 섞음은 가우스 하나보다 두꺼운 꼬리를 더 잘 담아낸다.

### 포트폴리오 무리짓기

수익의 성격으로 자산을 무리짓는다:

1. 특징 벡터(평균, 변동성, 치우침, 상관)를 셈한다
2. 특징 공간에 가우스 섞음 모형을 맞춘다
3. 무리 배정으로 자산을 묶는다

이러면 전통적인 업종 가르기를 넘어선 숨은 짜임이 드러난다.

---

## 연습문제

**연습문제 1.**
EM 알고리즘의 되풀이마다 로그 가능도 $\log p(X \mid \theta)$이 단조롭게 커짐을 보여라.

??? success "연습문제 1 풀이"
    근본 항등식에서 $\log p(X \mid \theta) = \mathcal{L}(q, \theta) + D_{\text{KL}}(q \| p(Z|X,\theta))$이다. E 걸음에서 $q = p(Z|X,\theta^{(t)})$으로 두면 $D_{\text{KL}} = 0$이 되어 $\mathcal{L}(q^{(t+1)}, \theta^{(t)}) = \log p(X|\theta^{(t)})$이다. M 걸음에서는 $\theta^{(t+1)} = \arg\max_\theta \mathcal{L}(q^{(t+1)}, \theta) \geq \mathcal{L}(q^{(t+1)}, \theta^{(t)})$이다. 그러므로 $\log p(X|\theta^{(t+1)}) \geq \mathcal{L}(q^{(t+1)}, \theta^{(t+1)}) \geq \mathcal{L}(q^{(t+1)}, \theta^{(t)}) = \log p(X|\theta^{(t)})$이다. $\square$

---

**연습문제 2.**
성분이 $K$개인 가우스 섞음 모형의 온전한 E 걸음과 M 걸음 새로 고침을 이끌어 내어라.

??? success "연습문제 2 풀이"
    **E 걸음:** 맡음 몫 $r_{nk} = \frac{\pi_k \mathcal{N}(x_n | \mu_k, \Sigma_k)}{\sum_{j=1}^K \pi_j \mathcal{N}(x_n | \mu_j, \Sigma_j)}$을 셈한다.

    **M 걸음:** $N_k = \sum_n r_{nk}$이라 하고 다음처럼 새로 고친다:

    $$\mu_k = \frac{1}{N_k} \sum_n r_{nk} x_n, \quad \Sigma_k = \frac{1}{N_k} \sum_n r_{nk} (x_n - \mu_k)(x_n - \mu_k)^\top, \quad \pi_k = \frac{N_k}{N}$$

---

**연습문제 3.**
딱 잘라 하는 EM과 부드러운 EM의 차이를 설명하여라. 딱 잘라 하는 EM은 언제 나을 수 있는가?

??? success "연습문제 3 풀이"
    **부드러운 EM**에서는 E 걸음이 몫으로 나뉜 맡음 몫(뒤확률) $r_{nk} \in [0, 1]$을 셈한다. **딱 잘라 하는 EM**에서는 자료 점마다 무리 하나에 배정된다. 곧 $k^* = \arg\max_k r_{nk}$이면 $r_{nk} = 1$, 아니면 $r_{nk} = 0$이다. 딱 잘라 하는 EM은 공분산이 같은 구면 가우스에서 k 평균 알고리즘과 같다. 딱 잘라 하는 EM은 (1) 띄엄띄엄한 무리짓기가 필요할 때, (2) 셈 자원이 빠듯할 때(새로 고침이 더 단순하다), (3) 무리가 잘 떨어져 있어 부드러운 배정이 별 값어치가 없을 때 낫다.

---

**연습문제 4.**
EM 도중에 가우스 성분이 자료 점 하나로 찌부러지면 어떤 말썽이 생길 수 있는가? 어떻게 막을 수 있는가?

??? success "연습문제 4 풀이"
    가우스 성분의 평균이 자료 점 하나와 겹치고 그 흩어짐이 0으로 오그라들면 가능도가 묶이지 않는다(그 점에서 밀도가 무한으로 간다). 이것이 가우스 섞음 모형의 **특이점 문제**이다. 막는 방법으로는 (1) 공분산에 작은 벌주기 항 더하기($\Sigma_k + \epsilon I$), (2) $N_k$이 문턱값 아래로 떨어진 성분 되돌리기, (3) 베이즈 앞확률 쓰기(이를테면 $\Sigma_k$에 역위샤트), (4) 공분산 행렬의 고윳값에 최솟값 제약 두기가 있다.

## 정리하며

| 항목 | 설명 |
|--------|-------------|
| **모형** | 가우스 분포 $K$개에 무게를 붙여 더한 것 |
| **매개변수** | 섞음 무게 $\boldsymbol{\pi}$, 평균 $\{\boldsymbol{\mu}_k\}$, 공분산 $\{\boldsymbol{\Sigma}_k\}$ |
| **E 걸음** | 베이즈 정리로 맡음 몫 $\gamma_{ik}$ 셈하기 |
| **M 걸음** | 모든 매개변수를 무게 붙인 최대 가능도 어림으로 새로 고치기 |
| **첫값 잡기** | k 평균, k 평균++, 또는 여러 번 다시 시작하는 무작위 |
| **모형 고르기** | $K$을 고르는 데 BIC, AIC, 교차 확인 |
| **벌주기** | 대각에 얹기, 최소 고윳값, 베이즈 앞확률 |

가우스 섞음 모형은 기계 학습과 통계의 근본 도구로 남아 있으며 다룰 수 있는 밀도 모형과 원리 있는 무리짓기를 함께 준다. 그 EM 어림 절차는 이 알고리즘의 우아함을 잘 보여 주며 더 복잡한 숨은 변수 모형의 본보기가 된다.
