# E 단계와 M 단계

E 단계는 현재 파라메터에 조건부인 잠재변수의 사후분포에서 필요한 기댓값을 계산한다. M 단계는 그 분포를 고정하고 기댓값 완전 데이터 로그 우도를 최적화한다. 잠재변수를 기댓값 하나로 대체하는 것과 로그 우도의 기댓값을 계산하는 것은 다르다. 이차항이 있으면 분산까지 반영해야 한다.

## 1. E 단계의 대상과 계산량

현재 파라메터를 θₜ라 하면 qₜ(Z)=p(Z|X,θₜ)이고, Q(θ|θₜ)=E_qₜ[log p(X,Z|θ)]다. 지수족 모델에서는 완전 데이터의 충분통계량 T에 대한 E_qₜ[T]만 알면 되는 경우가 많다.

관측별 결합 모델이 p(X,Z|θ)=∏ᵢp(xᵢ,zᵢ|θ)로 분해되면 사후도 ∏ᵢp(zᵢ|xᵢ,θ)로 분해된다. 관측값이 독립이라는 설명만으로 공유 잠재변수가 있는 모델의 사후 독립성을 가정하면 안 된다. HMM의 인접 상태처럼 사후 의존성이 남는 경우에는 결합 적률이 필요하다.

계산량은 분포의 구조에 따라 달라진다. d차원 전 공분산 GMM은 성분별 행렬 분해에 대략 O(Kd³), 모든 데이터의 밀도 평가에 O(NKd²)가 든다. 대각 공분산이면 밀도 평가를 O(NKd)로 줄일 수 있다. 길이 T인 밀집 K상태 HMM의 전방·후방 계산은 관측 밀도 평가를 제외하고 O(TK²)다.

## 2. GMM의 E 단계

성분 라벨 zᵢ에 대한 책임도는 베이즈 정리로 얻는다.

$$r_{ik}=p(z_i=k|x_i,\theta_t)=\frac{\pi_k\mathcal N(x_i;\mu_k,\Sigma_k)}{\sum_j\pi_j\mathcal N(x_i;\mu_j,\Sigma_j)}.$$

모든 파라메터는 이 단계에서 θₜ의 값을 쓴다. 필요한 통계량은 Nₖ=∑ᵢrᵢₖ, sₖ=∑ᵢrᵢₖxᵢ, Rₖ=∑ᵢrᵢₖxᵢxᵢᵀ다. 각 관측의 책임도 합은 1이고 전체 유효 표본 수의 합은 N이다. 책임도는 확률이며 마지막에 argmax로 고르는 군집 라벨과 구분한다.

## 3. 인자 분석의 E 단계

xᵢ=μ+Wzᵢ+εᵢ, zᵢ∼N(0,I), εᵢ∼N(0,Ψ)이며 Ψ가 양의 대각행렬이라고 하자. 현재 파라메터에서 가우스 조건부분포를 계산하면 다음과 같다.

$$V=(I+W^\top\Psi^{-1}W)^{-1},\qquad m_i=VW^\top\Psi^{-1}(x_i-\mu).$$

$$E[z_i|x_i]=m_i,\qquad E[z_iz_i^\top|x_i]=V+m_im_i^\top.$$

공통 V는 관측마다 다시 역행렬로 계산할 필요가 없다. 실제 계산에서는 선형계 풀이를 이용할 수 있다. mᵢmᵢᵀ만 사용하면 조건부 분산 V를 버려 정확한 E 단계가 아니다.

## 4. HMM의 E 단계

초기확률 π, 전이확률 aⱼₖ, 관측 밀도 bₖ(x)를 쓰자. 전방 메시지 αₜ(k)=p(x₁:ₜ,zₜ=k), 후방 메시지 βₜ(k)=p(xₜ₊₁:ₜₘₐₓ|zₜ=k)를 계산한다.

$$\alpha_{t+1}(k)=b_k(x_{t+1})\sum_j\alpha_t(j)a_{jk},\qquad \beta_t(j)=\sum_k a_{jk}b_k(x_{t+1})\beta_{t+1}(k).$$

$$\gamma_t(k)=\frac{\alpha_t(k)\beta_t(k)}{p(X)},\qquad \xi_t(j,k)=\frac{\alpha_t(j)a_{jk}b_k(x_{t+1})\beta_{t+1}(k)}{p(X)}.$$

γ는 상태의 주변 사후확률, ξ는 인접한 두 상태의 결합 사후확률이다. 수치적으로는 로그 공간이나 일관된 스케일링을 사용한다. 여러 독립 시퀀스는 각 시퀀스의 시작과 끝을 구분해서 통계량을 합한다.

## 5. M 단계와 제약

M 단계에서 qₜ와 책임도를 고정한다. θ를 미분할 때 이미 계산한 사후분포까지 다시 미분하는 것이 아니다. GMM의 Q는 상수를 제외하면 다음과 같다.

$$Q=\sum_{i,k}r_{ik}\left[\log\pi_k-\frac12\log|\Sigma_k|-\frac12(x_i-\mu_k)^\top\Sigma_k^{-1}(x_i-\mu_k)\right].$$

π의 합이 1이라는 제약에 라그랑주 승수를 적용하고, 평균과 공분산을 최적화하면 다음 갱신을 얻는다.

$$\pi_k^{new}=N_k/N,\qquad\mu_k^{new}=s_k/N_k,$$

$$\Sigma_k^{new}=\frac1{N_k}\sum_i r_{ik}(x_i-\mu_k^{new})(x_i-\mu_k^{new})^\top.$$

여기에는 Nₖ>0과 유효한 공분산 해가 필요하다. 분모는 불편 표본분산의 Nₖ−1이 아니다. 이는 가중 최대 우도 갱신이다. 빈 성분이나 특이 공분산을 임의로 재설정하면 정확한 M 단계와 다른 절차가 된다.

인자 분석에서 μ를 고정하고 yᵢ=xᵢ−μ라 하면 다음과 같이 갱신할 수 있다.

$$W^{new}=\left(\sum_i y_im_i^\top\right)\left(\sum_i(V+m_im_i^\top)\right)^{-1},$$

$$\Psi^{new}=\operatorname{diag}\left\{\frac1N\sum_i\left[y_iy_i^\top-y_im_i^\top(W^{new})^\top-W^{new}m_iy_i^\top+W^{new}(V+m_im_i^\top)(W^{new})^\top\right]\right\}.$$

μ도 추정한다면 같은 Q의 결합 정상방정식을 풀거나 최신 값을 사용하는 조건부 최대화를 구성한다. 현재 사후 적률을 새 파라메터로 중간에 다시 계산해서 섞지 않는다.

HMM에서는 πₖⁿᵉʷ=γ₁(k), aⱼₖⁿᵉʷ=∑ₜ₌₁ᵀ⁻¹ξₜ(j,k)/∑ₜ₌₁ᵀ⁻¹γₜ(j)다. 가우스 관측의 평균과 공분산은 γ를 웨이트로 갱신한다. 전이의 분모가 0이거나 시퀀스 길이가 1인 경우에는 해당 정보가 없으므로 별도 제약·사전·처리 규칙이 필요하다.

## 6. 로그 공간 계산과 예제 구현

GMM에서는 aᵢₖ=log πₖ+log N(xᵢ;μₖ,Σₖ)를 계산하고 log rᵢₖ=aᵢₖ−logsumexpⱼ aᵢⱼ로 정규화한다. 모든 aᵢₖ가 −∞인 관측은 이 방식만으로 해결되지 않는다. 데이터·지지집합·공분산의 유효성을 먼저 확인한다.

아래 구현은 전 공분산 GMM의 교육용 예제다. `covariance_type` 인수는 저장하지만 분기에서 사용하지 않으므로 대각·구면 공분산 구현으로 해석하면 안 된다. 기본 텐서 생성은 CPU를 사용하며 입력의 장치와 자료형을 일반적으로 맞추는 처리가 없다. Nₖ가 0인 성분의 나눗셈도 보호하지 않는다.

M 단계에서 가중 공분산에 `reg_covar * I`를 더한다. 이는 수치 안정화를 위한 수정이며 원래의 제약 없는 최대 우도 M 단계가 아니다. 따라서 정확한 EM의 단조성을 코드에 자동으로 적용하지 않는다. `fit`은 갱신 후 총 로그 우도를 계산하지만 절댓값 차이를 쓰므로 작은 감소도 종료 조건을 만족할 수 있다. 총합 기준 허용오차는 표본 수에도 영향을 받는다.

```python
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal, Categorical

class GaussianMixtureEM:
    """EM 알고리즘을 쓴 가우스 섞음 모형."""
    
    def __init__(self, n_components: int, n_features: int, 
                 covariance_type: str = 'full', reg_covar: float = 1e-6):
        self.K = n_components
        self.d = n_features
        self.covariance_type = covariance_type
        self.reg_covar = reg_covar
        
        # 파라메터를 초기화한다
        self.pi = torch.ones(self.K) / self.K  # 섞음 비율
        self.mu = torch.randn(self.K, self.d)   # 평균
        self.Sigma = torch.stack([torch.eye(self.d) for _ in range(self.K)])  # 공분산
        
    def e_step(self, X: torch.Tensor) -> torch.Tensor:
        """
        E 걸음: 맡음 몫 셈하기.
        
        인수:
            X: 꼴이 (N, d)인 자료 텐서
            
        반환값:
            gamma: 꼴이 (N, K)인 맡음 몫
        """
        N = X.shape[0]
        log_resp = torch.zeros(N, self.K)
        
        for k in range(self.K):
            mvn = MultivariateNormal(self.mu[k], self.Sigma[k])
            log_resp[:, k] = torch.log(self.pi[k]) + mvn.log_prob(X)
        
        # 수치 안정을 위한 로그-합-지수
        log_resp_sum = torch.logsumexp(log_resp, dim=1, keepdim=True)
        log_gamma = log_resp - log_resp_sum
        gamma = torch.exp(log_gamma)
        
        return gamma
    
    def m_step(self, X: torch.Tensor, gamma: torch.Tensor):
        """
        M 걸음: 파라메터 새로 고치기.
        
        인수:
            X: 꼴이 (N, d)인 자료 텐서
            gamma: 꼴이 (N, K)인 맡음 몫
        """
        N = X.shape[0]
        
        # 실효 횟수
        N_k = gamma.sum(dim=0)  # (K,)
        
        # 섞음 비율 새로 고치기
        self.pi = N_k / N
        
        # 평균을 갱신한다
        self.mu = (gamma.T @ X) / N_k.unsqueeze(1)  # (K, d)
        
        # 공분산을 갱신한다
        for k in range(self.K):
            diff = X - self.mu[k]  # (N, d)
            weighted_diff = gamma[:, k].unsqueeze(1) * diff  # (N, d)
            self.Sigma[k] = (weighted_diff.T @ diff) / N_k[k]
            
            # 정칙화
            self.Sigma[k] += self.reg_covar * torch.eye(self.d)
    
    def log_likelihood(self, X: torch.Tensor) -> float:
        """자료의 로그 가능도 셈하기."""
        N = X.shape[0]
        log_prob = torch.zeros(N, self.K)
        
        for k in range(self.K):
            mvn = MultivariateNormal(self.mu[k], self.Sigma[k])
            log_prob[:, k] = torch.log(self.pi[k]) + mvn.log_prob(X)
        
        return torch.logsumexp(log_prob, dim=1).sum().item()
    
    def fit(self, X: torch.Tensor, max_iters: int = 100, 
            tol: float = 1e-4, verbose: bool = False):
        """
        EM 알고리즘으로 가우스 섞음 모형 맞추기.
        
        인수:
            X: 꼴이 (N, d)인 자료 텐서
            max_iters: 최대 되풀이 횟수
            tol: 모임 너그러움
            verbose: 진행 상황 출력 여부
        """
        prev_ll = float('-inf')
        
        for iteration in range(max_iters):
            # E 걸음
            gamma = self.e_step(X)
            
            # M 걸음
            self.m_step(X, gamma)
            
            # 모임 살피기
            ll = self.log_likelihood(X)
            
            if verbose and iteration % 10 == 0:
                print(f"Iteration {iteration}: log-likelihood = {ll:.4f}")
            
            if abs(ll - prev_ll) < tol:
                if verbose:
                    print(f"Converged at iteration {iteration}")
                break
            
            prev_ll = ll
        
        return self
    
    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """무리 배정 미리보기."""
        gamma = self.e_step(X)
        return gamma.argmax(dim=1)
    
    def predict_proba(self, X: torch.Tensor) -> torch.Tensor:
        """무리 확률(맡음 몫) 미리보기."""
        return self.e_step(X)

# 사용 예
if __name__ == "__main__":
    torch.manual_seed(42)
    
    # 성분 3개인 가우스 섞음 모형에서 인공 자료 만들기
    N = 300
    true_means = torch.tensor([[0., 0.], [3., 3.], [-2., 3.]])
    
    X = torch.cat([
        torch.randn(100, 2) + true_means[0],
        torch.randn(100, 2) + true_means[1],
        torch.randn(100, 2) + true_means[2]
    ])
    
    # GMM을 적합시킨다
    gmm = GaussianMixtureEM(n_components=3, n_features=2)
    gmm.fit(X, verbose=True)
    
    print(f"\nLearned means:\n{gmm.mu}")
    print(f"\nLearned mixing proportions: {gmm.pi}")
```

다음 기존 출력은 예제에 첨부된 결과이며 이번 편집에서는 재실행하지 않았다. 출력의 수렴 메시지만으로 전역 최적성이나 모든 초기값에서의 동작을 판단하지 않는다.

```
Iteration 0: log-likelihood = -1215.1680
Iteration 10: log-likelihood = -1165.0482
Iteration 20: log-likelihood = -1127.2839
Iteration 30: log-likelihood = -1127.2419
Iteration 40: log-likelihood = -1127.2388
Converged at iteration 42

Learned means:
tensor([[-0.0845,  0.2149],
        [-1.9901,  2.8644],
        [ 3.1764,  3.1199]])

Learned mixing proportions: tensor([0.3191, 0.3490, 0.3319])
```

## 7. ELBO를 통한 단조성

L(q,θ)=E_q[log p(X,Z|θ)]+H(q)라 하면 ℓ(θ)=L(q,θ)+KL(q∥p(Z|X,θ))다. 정확한 E 단계에서 qₜ=p(Z|X,θₜ)이므로 현재 점에서 하한이 일치한다. Q를 높이는 M 단계는 엔트로피가 고정된 L도 높인다.

$$\ell(\theta_{t+1})\geq L(q_t,\theta_{t+1})\geq L(q_t,\theta_t)=\ell(\theta_t).$$

등호는 파라메터가 같은 경우에만 성립하는 것이 아니다. 비식별성, 평평한 방향이나 같은 목적값을 갖는 다른 파라메터에서도 가능하다. 우도값이 수렴해도 파라메터의 수렴과 정상성에는 추가 조건이 필요하며, 제약 없는 GMM의 우도는 위로 유계가 아닐 수 있다.

다음 도식은 현재 점에서 일치시킨 하한을 올린다는 직관을 나타낸다. 실제 하한이 직선이나 접평면일 필요는 없다.

```
Log-likelihood ℓ(θ)
         │
         │     ╱───────────────  ℓ(θ)
         │    ╱        ●──────────○  ℓ(θ^(t+1))
         │   ╱        ╱
         │  ╱    ●───●  ELBO at iteration t
         │ ╱    ╱   ↑
         │╱    ╱    M-step maximizes bound
         │    ● ────┘
         │   ↑ θ^(t)
         │   E-step makes bound tight here
         └────────────────────────────────────── θ
```

## 8. 좌표 상승법으로 보는 전체 반복

제한 없는 분포 q를 최적화하는 E 단계와 θ를 최적화하는 M 단계는 ELBO에 대한 두 블록 좌표 상승이다. 반면 깁스 샘플링은 조건부분포에서 표본을 생성하는 절차이므로 두 알고리즘을 모두 블록 최적화라고 부르면 안 된다. 제한된 분포족을 쓰는 변분 EM에서는 ELBO의 증가와 관측 우도의 증가도 구분한다.

아래 의사코드는 정확한 E 단계와 유효한 M 단계를 전제로 읽는다. 수치적 정규화, 부분 최적화 또는 근사 사후분포를 도입하면 어떤 목적을 실제로 높이는지 다시 확인해야 한다.

```
Initialize θ⁽⁰⁾
repeat until convergence:
    # E 걸음: 뒤확률 셈하기
    q(Z) ← p(Z | X, θ⁽ᵗ⁾)
    
    # M 걸음: 기댓값 완전 자료 로그 가능도를 가장 크게 하기
    θ⁽ᵗ⁺¹⁾ ← argmax_θ E_q[log p(X, Z | θ)]
    
    t ← t + 1
return θ⁽ᵗ⁾
```

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff hard" title="어려움"></span>
GMM 책임도를 로그 공간에서 정규화하는 식을 쓰고 실패할 수 있는 입력을 설명하라.

</div>

??? success "연습문제 1 풀이"
    aᵢₖ=log πₖ+log N(xᵢ;μₖ,Σₖ), log rᵢₖ=aᵢₖ−logsumexpⱼaᵢⱼ를 쓴다. 모든 성분의 로그 밀도가 −∞이면 정규화 차이가 정의되지 않는다. 양의 정부호 공분산, 유효한 비중과 유한한 입력 등 모델의 조건을 확인해야 한다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff hard" title="어려움"></span>
인자 분석에서 E[zzᵀ|x]를 E[z|x]E[z|x]ᵀ로 대체할 수 없는 이유는 무엇인가?

</div>

??? success "연습문제 2 풀이"
    E[zzᵀ|x]=Cov(z|x)+E[z|x]E[z|x]ᵀ다. 조건부 분산을 버리면 완전 데이터의 이차 충분통계량을 잘못 계산하며, 정확한 Q와 M 단계가 아니게 된다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
HMM 전이 갱신에서 γ와 ξ의 역할을 구분하라.

</div>

??? success "연습문제 3 풀이"
    ξₜ(j,k)를 합하면 j→k의 기대 전이 횟수이고, t=1,…,T−1의 γₜ(j)를 합하면 상태 j에서 전이를 시작한 기대 횟수다. 둘의 비로 전이행렬의 행을 정규화한다. 마지막 시점은 나가는 전이가 없으므로 분모에서 제외한다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
예제의 작은 로그 우도 차이가 정확한 EM 수렴의 증명인가?

</div>

??? success "연습문제 4 풀이"
    아니다. 절댓값 차이는 작은 감소도 허용하며, 유한 정밀도·공분산 보정·빈 성분 문제도 남는다. 종료는 구현의 기준이다. 우도 단조성이나 정상점 수렴은 해당 알고리즘과 모델의 별도 조건 아래에서 설명해야 한다.

## 정리하며

E 단계에서는 현재 파라메터로 사후 적률을 계산하고, M 단계에서는 그 적률을 고정한다. GMM의 책임도, 인자 분석의 조건부 이차 적률, HMM의 인접 상태 사후확률이 대표적인 충분통계량이다. 정확한 EM의 보장과 수치적으로 수정한 예제의 동작을 구분한다.
