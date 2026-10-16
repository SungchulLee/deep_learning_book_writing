# 가우시안 혼합 모델

가우시안 혼합 모델(GMM)은 여러 가우스 밀도의 가중합으로 데이터의 분포를 나타낸다. 각 관측이 어느 성분에서 나왔는지는 잠재변수로 두고, EM에서는 그 성분에 속할 사후확률을 이용해 파라메터를 갱신한다. 밀도 추정과 소프트 군집화에 사용할 수 있지만 성분이 실제 원인이나 고유한 집단에 대응한다고 보장되지는 않는다.

## 1. 모델과 파라메터

d차원 관측 xᵢ의 생성 과정을 zᵢ∼Categorical(π), xᵢ|zᵢ=k∼N(μₖ,Σₖ)로 정의한다. πₖ≥0, ∑ₖπₖ=1이고 각 Σₖ는 양의 정부호다.

$$p(x_i,z_i=k)=\pi_k\mathcal N(x_i;\mu_k,\Sigma_k),\qquad p(x_i)=\sum_{k=1}^K\pi_k\mathcal N(x_i;\mu_k,\Sigma_k).$$

일반적인 내부 파라메터에서 자유도는 다음과 같다. 대칭 공분산은 d²개가 아니라 d(d+1)/2개의 자유도를 갖는다.

| 공분산 구조 | 제약 | 전체 파라메터 수 |
|---|---|---|
| 전 공분산 | 성분별 Σₖ | Kd+Kd(d+1)/2+K−1 |
| 대각 공분산 | 성분별 대각행렬 | 2Kd+K−1 |
| 구면 공분산 | Σₖ=σₖ²I | Kd+K+K−1 |
| 공유 공분산 | 모든 성분이 같은 Σ 사용 | Kd+d(d+1)/2+K−1 |

라벨을 일관되게 순열하면 같은 밀도를 얻는다. 모든 성분이 구별되는 경우 K!개의 라벨 표현이 있지만 중복 성분이나 0인 비중에서는 대칭성이 달라진다. 라벨의 다중성과 서로 다른 적합 밀도의 국소 최적점은 구분한다.

## 2. 로그 우도와 책임도

독립 관측의 로그 우도는 ℓ(θ)=∑ᵢlog∑ₖπₖN(xᵢ;μₖ,Σₖ)다. 이는 일반적으로 비오목하지만 미분 가능한 유효 영역에서는 직접 경사 최적화도 가능하다. EM은 숨은 라벨을 도입해 계산하기 쉬운 기댓값 완전 데이터 로그 우도를 사용한다.

현재 파라메터에서 책임도와 유효 표본 수는 다음과 같다.

$$r_{ik}=\frac{\pi_k\mathcal N(x_i;\mu_k,\Sigma_k)}{\sum_j\pi_j\mathcal N(x_i;\mu_j,\Sigma_j)},\qquad N_k=\sum_i r_{ik}.$$

책임도는 각 관측에서 합이 1이다. 큰 밀도값을 직접 더하는 대신 로그 밀도와 logsumexp로 정규화한다. 밀도가 매우 작다는 이유만으로 해당 성분의 사후확률도 작다고 단정하지 않는다. 성분 간 상대적인 비중과 밀도가 함께 결정한다.

## 3. EM 갱신의 유도

라벨 지시변수를 Zᵢₖ라 하면 완전 데이터 로그 우도는 ∑ᵢₖZᵢₖ[log πₖ+log N(xᵢ;μₖ,Σₖ)]다. E 단계에서 Zᵢₖ의 기댓값을 rᵢₖ로 바꾸면 Q를 얻는다. M 단계에서는 이 r을 고정한다.

비중 제약에 라그랑주 승수를 적용하면 Nₖ/πₖ가 모든 k에서 같아야 하므로 πₖⁿᵉʷ=Nₖ/N이다. 평균의 정상방정식은 ∑ᵢrᵢₖ(xᵢ−μₖ)=0이고, 공분산을 최적화하면 다음을 얻는다.

$$\mu_k^{new}=\frac{\sum_i r_{ik}x_i}{N_k},\qquad \Sigma_k^{new}=\frac{\sum_i r_{ik}(x_i-\mu_k^{new})(x_i-\mu_k^{new})^\top}{N_k}.$$

공분산은 새 평균을 중심으로 계산한다. Nₖ>0이어야 하며, 가중 중심화 데이터의 랭크가 d보다 작으면 전 공분산 해는 특이하다. Nₖ는 실수 웨이트의 합이므로 단순히 Nₖ와 d+1을 비교하는 것만으로 랭크를 판정할 수 없다.

아래는 표준 EM의 반복 구조다. 실제 구현의 정규화와 종료 규칙은 별도로 확인한다.

```
Input: Data X, number of components K
Output: Parameters θ = {π, μ, Σ}

1. Initialize parameters θ⁽⁰⁾
2. Repeat until convergence:
   
   # E 걸음: 맡음 몫 셈하기
   For each i, k:
       γ_ik ← π_k N(x_i | μ_k, Σ_k) / Σ_j π_j N(x_i | μ_j, Σ_j)
   
   # M 걸음: 파라메터 새로 고치기
   For each k:
       N_k ← Σ_i γ_ik
       π_k ← N_k / N
       μ_k ← (1/N_k) Σ_i γ_ik x_i
       Σ_k ← (1/N_k) Σ_i γ_ik (x_i - μ_k)(x_i - μ_k)ᵀ
   
   # 모임 살피기
   If |ℓ(θ⁽ᵗ⁺¹⁾) - ℓ(θ⁽ᵗ⁾)| < tolerance: break

3. Return θ
```

## 4. 초기화와 공분산 구조

무작위 데이터점이나 k-means 중심으로 평균을 초기화할 수 있다. k-means++는 기존 중심까지의 제곱 거리에 비례해 다음 중심을 선택하는 초기화 방식이며, 임의 중심으로 Lloyd 반복을 몇 차례 수행하는 것과 다르다. 여러 초기값은 국소해의 영향을 줄일 수 있으나 전역 최적해를 보장하지 않는다.

소프트 책임도 대신 최대 책임도의 성분 하나에 배정하는 것은 하드 EM이다. 동일하게 고정한 구면 공분산과 동일한 비중을 사용하는 경우 평균 갱신이 k-means와 일치한다. 공분산·비중을 함께 학습하는 일반 GMM과는 다르며 표준 EM의 관측 우도 단조성을 그대로 보장하지 않는다.

전 공분산은 성분별 상관관계를 표현하지만 추정할 값이 많다. 대각 공분산은 좌표축 방향의 분산만 표현하므로 좌표 회전에 불변인 모델이 아니다. 구면 공분산은 한 성분의 모든 방향에 같은 분산을 사용한다. 공유 공분산은 성분별 평균을 허용하면서 전체 공분산을 하나로 묶는다.

대각 갱신은 전 공분산 갱신의 대각 원소, 구면 갱신은 가중 제곱 거리의 합을 dNₖ로 나눈 값이다. 공유 공분산은 모든 성분의 가중 잔차 외적을 더해 N으로 나눈다. 이는 각각 제약한 Q를 최적화하는 갱신이다.

## 5. 성분 수 선택

성분 수와 공분산 구조는 적합도와 복잡도를 함께 비교한다. 파라메터 수를 p, 적합 로그 우도를 ℓ̂이라 하면 통상적인 기준은 다음과 같다.

$$\mathrm{AIC}=-2\hat\ell+2p,\qquad \mathrm{BIC}=-2\hat\ell+p\log N.$$

두 기준 모두 작은 값을 선호한다. N>e²일 때 BIC의 단위 파라메터당 벌점이 AIC보다 크다. 그러나 혼합 모델에는 비식별성·중복 성분·경계 등의 특이성이 있어 정규 모델의 점근 이론을 그대로 적용할 수 없다. BIC를 정확한 로그 주변우도나 참 K의 보장으로 읽지 않는다.

군집 분리까지 고려하는 흔한 ICL 근사는 H=−∑ᵢₖrᵢₖlog rᵢₖ라 할 때 ICL≈BIC+2H다. 배정이 불확실하면 벌점이 커지므로 밀도 추정에 좋은 K와 명확한 군집을 만드는 K가 다를 수 있다.

아래 선택 절차에서는 후보마다 같은 데이터와 적합 조건을 사용하고 초기화 실패나 공분산 붕괴 여부도 확인한다.

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

검증 데이터의 평균 로그 밀도로 예측 성능을 비교하는 방법도 있다. 시계열에서는 시간 순서를 고려해 평가하고 스케일 변환 같은 전처리도 훈련 데이터에서 추정한다. 훈련 우도만으로 K를 늘리면 과적합과 특이해를 놓칠 수 있다.

## 6. 특이점과 명시적 정규화

한 성분의 평균이 관측점과 일치하고 공분산이 그 점으로 붕괴하면 그 관측에서 밀도가 발산할 수 있다. 다른 성분이 나머지 데이터를 설명하면 전체 우도도 상계가 없어질 수 있다. 모든 x에서 분산을 줄이기만 하면 밀도가 커진다는 뜻은 아니다.

최소 고유값 제약, 벌점 또는 적절한 사전분포는 문제를 바꿔 이런 붕괴를 다루는 방법이다. 단순히 가중 공분산에 εI를 더하는 휴리스틱은 원래 최대 우도 해와 다르므로 목적함수의 단조성을 별도로 확인한다. 고유값 하한을 둔 정확한 제약 최적화와 대각에 일정 값을 더하는 갱신도 일반적으로 같은 연산이 아니다.

비중의 Dirichlet 사전에서 모든 αₖ>1이면 내부 MAP 해의 장벽 역할을 할 수 있다. αₖ=1은 균일 사전이며 αₖ<1은 오히려 경계의 희소한 비중을 선호할 수 있으므로 ‘Dirichlet이면 빈 성분을 막는다’고 일반화하지 않는다.

공분산 사전의 파라메터화도 명시해야 한다. 예를 들어 고정된 μₖ에 대해 역위샤트 사전 밀도를 |Σ|⁻⁽ν₀⁺ᵈ⁺¹⁾ᐟ²exp[−tr(Ψ₀Σ⁻¹)/2]로 정의하면, 가중 산포 Sₖ=∑ᵢrᵢₖ(xᵢ−μₖ)(xᵢ−μₖ)ᵀ를 사용한 조건부 MAP는 다음과 같다.

$$\Sigma_k^{MAP}=\frac{\Psi_0+S_k}{\nu_0+N_k+d+1}.$$

평균까지 정규–역위샤트 사전으로 묶으면 평균의 수축과 추가 산포 항이 생긴다. 평균을 적분한 사후의 모드와 결합 MAP의 공분산도 구분해야 한다.

## 7. PyTorch 구현의 범위

아래 코드는 공분산 구조별 계산과 다중 초기화의 흐름을 보여 준다. 코드 자체는 유지했으며 다음 설명은 정적 검토에 따른 것이다.

초기 평균은 무작위로 고른 관측점에서 시작해 Lloyd 반복을 10회 수행하며 k-means++가 아니다. `_initialize_covariances`는 전달받은 군집 배정을 사용하지 않고 전체 데이터의 분산으로 초기화한다. 비중에 사용하는 배정은 마지막 중심 갱신 전의 값이다.

`fit`의 `ll`은 M 단계 전 E 단계에서 계산된다. 최대 반복 수에 도달하면 저장 파라메터는 마지막 M 단계 이후인데 비교에 쓰는 우도는 이전 값일 수 있다. 따라서 `lower_bound_`를 최종 파라메터의 재평가 우도나 별도로 계산한 ELBO로 해석하면 안 된다. `converged_`와 `n_iter_`도 초기화별·최선 결과별로 일관되게 저장하지 않아 선택된 모델의 메타데이터라고 단정할 수 없다.

Nₖ에 작은 상수를 더하고 N으로 나누므로 비중의 합이 정확히 1인 갱신도 아니다. 밀도 평가와 샘플링에서 공분산에 `reg_covar * I`를 더하지만 저장된 공분산 갱신은 그 유효 공분산을 정확히 최대화하는 방식이 아니다. Cholesky 실패 뒤 일반 역행렬을 쓰는 분기는 잘못된 공분산을 양의 정부호로 고쳐 주지 않는다.

코드에서 대각 구조의 이름은 `'diagonal'`이다. 입력 검증과 장치·자료형 전달은 일반화되어 있지 않으며 1≤K≤N, 유효한 반복 수·데이터·공분산을 전제로 읽는다. `score`는 관측당 평균 로그 밀도이고 AIC/BIC는 총 로그 우도를 사용한다. 서로 다른 단위를 혼동하지 않는다.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class GaussianMixtureModel:
    """
    온전한 EM 구현을 갖춘 가우스 섞음 모형.
    
    여러 공분산 클래스와 벌주기를 받쳐 준다.
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
        
        # 파라메터(fit에서 첫값을 잡는다)
        self.weights_ = None      # (K,)
        self.means_ = None        # (K, d)
        self.covariances_ = None  # 꼴은 covariance_type에 달렸다
        
        # 맞추기 정보
        self.converged_ = False
        self.n_iter_ = 0
        self.lower_bound_ = float('-inf')
        
    def _initialize_parameters(self, X: torch.Tensor):
        """가우스 섞음 모형의 파라메터 첫값 잡기."""
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
            
            # 웨이트 초기화
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
        """M 걸음: 맡음 몫이 주어졌을 때 파라메터 새로 고치기."""
        N = X.shape[0]
        
        # 실효 횟수
        N_k = responsibilities.sum(dim=0) + 1e-10  # (K,)
        
        # 웨이트를 갱신한다
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
        
        # 가장 좋은 파라메터 되돌리기
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
        
        # 파라메터 개수 세기
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

## 8. 수익률 분포와 해석

수익률 R에 GMM을 적합하면 비대칭이나 여러 봉우리, 단일 정규분포보다 큰 첨도를 일정 범위에서 표현할 수 있다. 유한 개의 유한 분산 가우스 성분을 섞어도 점근 꼬리는 가우스형이며 멱법칙 꼬리가 생기지는 않는다. 관측 범위 밖의 꼬리 위험은 성분 수만 늘려 검증할 수 없다.

정적 GMM의 성분을 시장 국면으로 해석할 때도 주의가 필요하다. 독립 관측 모델에는 국면의 시간적 지속성이나 전이확률이 없다. 그런 구조를 모델링하려면 HMM 등 시간 의존 모델을 별도로 정의해야 한다.

일변량 혼합의 분위수는 성분 분위수의 가중평균이 아니다. 먼저 다음 혼합 누적분포를 구성하고 F_R(q)=α를 수치적으로 푼다.

$$F_R(q)=\sum_k\pi_k\Phi\left(\frac{q-\mu_k}{\sigma_k}\right).$$

연속 수익률에서 로스 L=−R의 신뢰수준 c에 대한 분위수는 VaR_c(L)=−F_R⁻¹(1−c)다. 부호와 꼬리 확률의 정의를 일관되게 사용한다. 자산 피처의 군집화 역시 피처의 스케일·기간·추정 불확실성에 영향을 받으며 군집 라벨 자체가 투자 판단을 정해 주지는 않는다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff hard" title="어려움"></span>
전 공분산과 공유 공분산 GMM의 파라메터 수를 계산하라.

</div>

??? success "연습문제 1 풀이"
    전 공분산은 Kd+Kd(d+1)/2+K−1개다. 공유 공분산은 Kd+d(d+1)/2+K−1개다. 평균과 비중의 수는 같고 대칭 공분산을 성분별로 둘지 하나만 둘지가 다르다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff hard" title="어려움"></span>
책임도로부터 비중·평균·공분산의 최대 우도 갱신을 설명하라.

</div>

??? success "연습문제 2 풀이"
    Nₖ=∑ᵢrᵢₖ라 하면 πₖ=Nₖ/N, μₖ=∑ᵢrᵢₖxᵢ/Nₖ다. 공분산은 새 평균을 중심으로 한 가중 잔차 외적의 합을 Nₖ로 나눈다. Nₖ>0과 충분한 데이터 랭크가 필요하며, 분모 Nₖ−1의 불편 분산 추정과 다르다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
혼합분포의 α 분위수가 성분별 α 분위수의 가중평균과 일반적으로 다른 이유는 무엇인가?

</div>

??? success "연습문제 3 풀이"
    혼합의 CDF는 CDF들의 가중합이지만 역함수 연산은 선형이 아니다. 따라서 ∑ₖπₖFₖ(q)=α를 만족하는 q를 구해야 하며 ∑ₖπₖFₖ⁻¹(α)를 대신 사용할 수 없다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
예제에서 `lower_bound_`와 최종 모델의 우도가 어긋날 수 있는 시점을 설명하라.

</div>

??? success "연습문제 4 풀이"
    E 단계에서 ll을 계산한 뒤 M 단계를 수행하고 최대 반복 수로 종료하면, 최종 파라메터는 바뀌었지만 ll은 이전 값이다. 다중 초기화의 비교와 저장된 lower_bound_도 이 영향을 받을 수 있으므로 최종 파라메터에서 우도를 재평가해야 일관된 비교가 된다.

## 정리하며

GMM의 EM은 책임도를 계산한 뒤 가중 최대 우도 추정으로 파라메터를 갱신한다. 공분산 구조·초기화·특이점 제어와 평가 기준을 함께 고려하고, 혼합 모델의 비식별성과 예제 구현의 한계를 구분한다. 혼합분포의 분위수는 혼합 CDF에서 구한다.
