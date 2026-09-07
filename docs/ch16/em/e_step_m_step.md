# E 걸음과 M 걸음
기댓값-최대화 알고리즘의 이름은 번갈아 밟는 두 걸음에서 왔다. 이 절에서는 걸음마다 셈의 얼개, 흔한 모형의 닫힌 꼴 이끌어 내기, 실전 구현에서 살필 점을 자세히 다룬다.

---

## E 걸음: 기댓값 충분 통계량 셈하기

### 근본 과제

E 걸음은 지금의 매개변수 어림값이 주어졌을 때 숨은 변수에 대한 **뒤확률 분포**를 셈한다:

$$
p(\mathbf{Z} | \mathbf{X}, \theta^{(t)})
$$

이 뒤확률에서 M 걸음에 필요한 **기댓값 충분 통계량**을 뽑아낸다. 핵심 통찰은 늘 온전한 뒤확률이 필요하지는 않다는 것이다. 흔히 어떤 기댓값만 있으면 된다.

### 베이즈 정리로 뒤확률 셈하기

뒤확률은 베이즈 정리에서 곧바로 따라 나온다:

$$
p(\mathbf{Z} | \mathbf{X}, \theta^{(t)}) = \frac{p(\mathbf{X} | \mathbf{Z}, \theta^{(t)}) \, p(\mathbf{Z} | \theta^{(t)})}{p(\mathbf{X} | \theta^{(t)})}
$$

분모는 주변 가능도이다:

$$
p(\mathbf{X} | \theta^{(t)}) = \int p(\mathbf{X} | \mathbf{Z}, \theta^{(t)}) \, p(\mathbf{Z} | \theta^{(t)}) \, d\mathbf{Z}
$$

숨은 변수가 띄엄띄엄하면 이는 가능한 모든 꼴에 걸친 합이 된다.

### 기댓값 충분 통계량

지수 집안 모형에서 E 걸음은 다음을 셈하는 것으로 줄어든다:

$$
\bar{T} = \mathbb{E}_{p(\mathbf{Z} | \mathbf{X}, \theta^{(t)})}[T(\mathbf{X}, \mathbf{Z})]
$$

여기서 $T(\mathbf{X}, \mathbf{Z})$은 완전 자료 분포의 충분 통계량이다. 이 기댓값은 흔히 온전한 뒤확률보다 셈하기가 훨씬 단순하다.

### 독립 짜임

자료 점이 독립이면 결합 뒤확률이 인수로 나뉜다:

$$
p(\mathbf{Z} | \mathbf{X}, \theta) = \prod_{i=1}^{N} p(z_i | \mathbf{x}_i, \theta)
$$

이러면 E 걸음이 독립인 셈 $N$개로 줄어들어 나란히 할 수 있고 다룰 수 있게 된다.

### 섞음 모형의 맡음 몫

성분이 $K$개인 섞음 모형에서 E 걸음은 **맡음 몫**을 셈한다:

$$
\gamma_{ik} = p(z_i = k | \mathbf{x}_i, \theta^{(t)}) = \frac{p(z_i = k | \theta^{(t)}) \, p(\mathbf{x}_i | z_i = k, \theta^{(t)})}{\sum_{j=1}^{K} p(z_i = j | \theta^{(t)}) \, p(\mathbf{x}_i | z_i = j, \theta^{(t)})}
$$

이 맡음 몫은 관측마다 성분마다의 **부드러운 배정**을 나타내며 다음을 만족한다:

- 모든 $i, k$에 대해 $0 \leq \gamma_{ik} \leq 1$
- 관측 $i$마다 $\sum_{k=1}^{K} \gamma_{ik} = 1$

### E 걸음의 복잡도

E 걸음의 셈 값은 숨은 변수의 짜임에 달렸다:

| 모형 | 숨은 짜임 | E 걸음의 복잡도 |
|-------|-----------------|-------------------|
| 가우스 섞음 | 띄엄띄엄하고 독립 | $O(NK)$ |
| 숨은 마르코프 모형 | 띄엄띄엄하고 차례 있음 | 앞뒤 알고리즘으로 $O(NK^2T)$ |
| 인자 분석 | 이어진 가우스 | $d$이 숨은 차원일 때 $O(Nd^3)$ |
| 숨은 디리클레 배분 | 띄엄띄엄하고 얽힘 | 어림 추론이 필요함 |

---

## 흔한 모형의 E 걸음 이끌어 내기

### 가우스 섞음 모형

**모형**: 섞음 비율이 $\boldsymbol{\pi} = (\pi_1, \ldots, \pi_K)$, 평균이 $\{\boldsymbol{\mu}_k\}$, 공분산이 $\{\boldsymbol{\Sigma}_k\}$인 가우스 성분 $K$개.

**숨은 변수**: $z_i \in \{1, \ldots, K\}$이 성분 소속을 가리킨다.

**E 걸음 이끌어 내기**:

$$
\gamma_{ik} = \frac{\pi_k \, \mathcal{N}(\mathbf{x}_i | \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)}{\sum_{j=1}^{K} \pi_j \, \mathcal{N}(\mathbf{x}_i | \boldsymbol{\mu}_j, \boldsymbol{\Sigma}_j)}
$$

여기서 가우스 밀도는 다음과 같다:

$$
\mathcal{N}(\mathbf{x} | \boldsymbol{\mu}, \boldsymbol{\Sigma}) = \frac{1}{(2\pi)^{d/2} |\boldsymbol{\Sigma}|^{1/2}} \exp\left( -\frac{1}{2}(\mathbf{x} - \boldsymbol{\mu})^\top \boldsymbol{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu}) \right)
$$

**기댓값 충분 통계량**:

$$
N_k = \sum_{i=1}^{N} \gamma_{ik}, \quad \bar{\mathbf{x}}_k = \frac{1}{N_k} \sum_{i=1}^{N} \gamma_{ik} \mathbf{x}_i, \quad \bar{S}_k = \frac{1}{N_k} \sum_{i=1}^{N} \gamma_{ik} (\mathbf{x}_i - \bar{\mathbf{x}}_k)(\mathbf{x}_i - \bar{\mathbf{x}}_k)^\top
$$

### 인자 분석

**모형**: 숨은 인자의 선형 바꿈으로 관측이 생긴다:

$$
\mathbf{x}_i = \mathbf{W} \mathbf{z}_i + \boldsymbol{\mu} + \boldsymbol{\epsilon}_i, \quad \mathbf{z}_i \sim \mathcal{N}(\mathbf{0}, \mathbf{I}), \quad \boldsymbol{\epsilon}_i \sim \mathcal{N}(\mathbf{0}, \boldsymbol{\Psi})
$$

여기서 $\mathbf{W}$은 인자 적재 행렬이고 $\boldsymbol{\Psi}$은 대각 행렬이다.

**E 걸음 이끌어 내기**: 숨은 변수에 대한 뒤확률은 가우스이다:

$$
p(\mathbf{z}_i | \mathbf{x}_i, \theta) = \mathcal{N}(\mathbf{z}_i | \mathbf{m}_i, \mathbf{V})
$$

여기서:

$$
\mathbf{V} = (\mathbf{I} + \mathbf{W}^\top \boldsymbol{\Psi}^{-1} \mathbf{W})^{-1}
$$

$$
\mathbf{m}_i = \mathbf{V} \mathbf{W}^\top \boldsymbol{\Psi}^{-1} (\mathbf{x}_i - \boldsymbol{\mu})
$$

**기댓값 충분 통계량**:

$$
\mathbb{E}[\mathbf{z}_i | \mathbf{x}_i] = \mathbf{m}_i
$$

$$
\mathbb{E}[\mathbf{z}_i \mathbf{z}_i^\top | \mathbf{x}_i] = \mathbf{V} + \mathbf{m}_i \mathbf{m}_i^\top
$$

### 숨은 마르코프 모형

**모형**: 옮김 행렬이 $\mathbf{A}$이고 내보냄 분포가 $\{p(\mathbf{x}_t | z_t)\}$인 숨은 상태의 늘어놓음 $\{z_1, \ldots, z_T\}$.

**앞뒤 알고리즘으로 하는 E 걸음**:

**앞으로 훑기**는 $\alpha_t(k) = p(\mathbf{x}_{1:t}, z_t = k)$을 셈한다:

$$
\alpha_1(k) = \pi_k \, p(\mathbf{x}_1 | z_1 = k)
$$

$$
\alpha_t(k) = p(\mathbf{x}_t | z_t = k) \sum_{j=1}^{K} \alpha_{t-1}(j) \, A_{jk}
$$

**뒤로 훑기**는 $\beta_t(k) = p(\mathbf{x}_{t+1:T} | z_t = k)$을 셈한다:

$$
\beta_T(k) = 1
$$

$$
\beta_t(k) = \sum_{j=1}^{K} A_{kj} \, p(\mathbf{x}_{t+1} | z_{t+1} = j) \, \beta_{t+1}(j)
$$

**기댓값 충분 통계량**:

$$
\gamma_t(k) = p(z_t = k | \mathbf{x}_{1:T}) = \frac{\alpha_t(k) \beta_t(k)}{\sum_{j} \alpha_t(j) \beta_t(j)}
$$

$$
\xi_t(j, k) = p(z_t = j, z_{t+1} = k | \mathbf{x}_{1:T}) = \frac{\alpha_t(j) A_{jk} p(\mathbf{x}_{t+1} | z_{t+1} = k) \beta_{t+1}(k)}{\sum_{j',k'} \alpha_t(j') A_{j'k'} p(\mathbf{x}_{t+1} | z_{t+1} = k') \beta_{t+1}(k')}
$$

---

## M 걸음: 매개변수 새로 고침 이끌어 내기

### 최적화 문제

M 걸음은 **Q 함수**(기댓값 완전 자료 로그 가능도)를 가장 크게 한다:

$$
\theta^{(t+1)} = \arg\max_\theta Q(\theta | \theta^{(t)})
$$

여기서 각 기호는 다음과 같다.

$$
Q(\theta | \theta^{(t)}) = \mathbb{E}_{p(\mathbf{Z} | \mathbf{X}, \theta^{(t)})}[\log p(\mathbf{X}, \mathbf{Z} | \theta)]
$$

### Q 함수를 왜 다룰 수 있나

Q 함수는 다음 까닭으로 주변 로그 가능도보다 최적화하기 쉽다:

1. **로그가 기댓값 안에 있음**: $\log \mathbb{E}[p(\mathbf{X}, \mathbf{Z} | \theta)]$이 아니라 $\mathbb{E}[\log p(\mathbf{X}, \mathbf{Z} | \theta)]$이다
2. **지수 집안의 짜임**: 지수 집안에서는 Q이 자연 매개변수에 대해 오목하다
3. **매개변수가 풀림**: Q에서 매개변수 묶음이 흔히 서로 풀린다

### M 걸음 새로 고침 이끌어 내기

일반 절차는 이렇다:

1. $\log p(\mathbf{X}, \mathbf{Z} | \theta)$을 드러내 적는다
2. $p(\mathbf{Z} | \mathbf{X}, \theta^{(t)})$에 대해 기댓값을 취한다
3. 매개변수마다 미분한다
4. 기울기를 0으로 두고 푼다

### 완전 자료 로그 가능도 쪼개기

가능도가 인수로 나뉘는 독립 관측에서:

$$
\log p(\mathbf{X}, \mathbf{Z} | \theta) = \sum_{i=1}^{N} \log p(\mathbf{x}_i, z_i | \theta) = \sum_{i=1}^{N} \left[ \log p(z_i | \theta) + \log p(\mathbf{x}_i | z_i, \theta) \right]
$$

그러면 Q 함수가 앞확률 항과 가능도 항으로 갈린다:

$$
Q(\theta | \theta^{(t)}) = \underbrace{\sum_{i=1}^{N} \mathbb{E}[\log p(z_i | \theta)]}_{\text{prior term}} + \underbrace{\sum_{i=1}^{N} \mathbb{E}[\log p(\mathbf{x}_i | z_i, \theta)]}_{\text{likelihood term}}
$$

---

## 흔한 모형의 M 걸음 이끌어 내기

### 가우스 섞음 모형

**Q 함수**:

$$
Q(\theta | \theta^{(t)}) = \sum_{i=1}^{N} \sum_{k=1}^{K} \gamma_{ik} \left[ \log \pi_k + \log \mathcal{N}(\mathbf{x}_i | \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k) \right]
$$

가우스의 로그 밀도를 펼치면:

$$
Q = \sum_{i,k} \gamma_{ik} \log \pi_k - \frac{1}{2} \sum_{i,k} \gamma_{ik} \left[ d \log(2\pi) + \log|\boldsymbol{\Sigma}_k| + (\mathbf{x}_i - \boldsymbol{\mu}_k)^\top \boldsymbol{\Sigma}_k^{-1} (\mathbf{x}_i - \boldsymbol{\mu}_k) \right]
$$

**섞음 비율**(제약 $\sum_k \pi_k = 1$ 아래에서):

라그랑주 곱수를 쓰면:

$$
\frac{\partial}{\partial \pi_k} \left[ Q + \lambda \left( \sum_j \pi_j - 1 \right) \right] = \frac{N_k}{\pi_k} + \lambda = 0
$$

풀면 $\pi_k = -N_k / \lambda$이다. 제약에서 $\lambda = -N$이므로:

$$
\boxed{\pi_k^{(t+1)} = \frac{N_k}{N} = \frac{1}{N} \sum_{i=1}^{N} \gamma_{ik}}
$$

**평균**:

$$
\frac{\partial Q}{\partial \boldsymbol{\mu}_k} = \sum_{i=1}^{N} \gamma_{ik} \boldsymbol{\Sigma}_k^{-1} (\mathbf{x}_i - \boldsymbol{\mu}_k) = 0
$$

풀면:

$$
\boxed{\boldsymbol{\mu}_k^{(t+1)} = \frac{\sum_{i=1}^{N} \gamma_{ik} \mathbf{x}_i}{\sum_{i=1}^{N} \gamma_{ik}} = \frac{1}{N_k} \sum_{i=1}^{N} \gamma_{ik} \mathbf{x}_i}
$$

**공분산**:

행렬 미적분($\frac{\partial}{\partial \boldsymbol{\Sigma}^{-1}} \log|\boldsymbol{\Sigma}^{-1}| = \boldsymbol{\Sigma}$)을 쓰면:

$$
\frac{\partial Q}{\partial \boldsymbol{\Sigma}_k^{-1}} = \frac{N_k}{2} \boldsymbol{\Sigma}_k - \frac{1}{2} \sum_{i=1}^{N} \gamma_{ik} (\mathbf{x}_i - \boldsymbol{\mu}_k)(\mathbf{x}_i - \boldsymbol{\mu}_k)^\top = 0
$$

풀면:

$$
\boxed{\boldsymbol{\Sigma}_k^{(t+1)} = \frac{1}{N_k} \sum_{i=1}^{N} \gamma_{ik} (\mathbf{x}_i - \boldsymbol{\mu}_k^{(t+1)})(\mathbf{x}_i - \boldsymbol{\mu}_k^{(t+1)})^\top}
$$

### 인자 분석

**Q 함수**(상수는 버리고):

$$
Q = -\frac{N}{2} \log|\boldsymbol{\Psi}| - \frac{1}{2} \sum_{i=1}^{N} \text{tr}\left( \boldsymbol{\Psi}^{-1} \mathbb{E}\left[ (\mathbf{x}_i - \boldsymbol{\mu} - \mathbf{W}\mathbf{z}_i)(\mathbf{x}_i - \boldsymbol{\mu} - \mathbf{W}\mathbf{z}_i)^\top \right] \right)
$$

**적재 행렬** $\mathbf{W}$:

$$
\frac{\partial Q}{\partial \mathbf{W}} = \boldsymbol{\Psi}^{-1} \sum_{i=1}^{N} \left[ (\mathbf{x}_i - \boldsymbol{\mu}) \mathbb{E}[\mathbf{z}_i]^\top - \mathbf{W} \mathbb{E}[\mathbf{z}_i \mathbf{z}_i^\top] \right] = 0
$$

풀면:

$$
\boxed{\mathbf{W}^{(t+1)} = \left( \sum_{i=1}^{N} (\mathbf{x}_i - \boldsymbol{\mu}) \mathbf{m}_i^\top \right) \left( \sum_{i=1}^{N} (\mathbf{V} + \mathbf{m}_i \mathbf{m}_i^\top) \right)^{-1}}
$$

**잡음 흩어짐** $\boldsymbol{\Psi}$(대각 성분):

$$
\boxed{\Psi_{jj}^{(t+1)} = \frac{1}{N} \sum_{i=1}^{N} \left[ (x_{ij} - \mu_j)^2 - 2 W_j^{(t+1)} m_i (x_{ij} - \mu_j) + W_j^{(t+1)} (\mathbf{V} + \mathbf{m}_i \mathbf{m}_i^\top) W_j^{(t+1)\top} \right]}
$$

여기서 $W_j$은 $\mathbf{W}$의 $j$번째 행이다.

### 숨은 마르코프 모형

**Q 함수**:

$$
Q = \sum_{k=1}^{K} \gamma_1(k) \log \pi_k + \sum_{t=1}^{T-1} \sum_{j,k} \xi_t(j,k) \log A_{jk} + \sum_{t=1}^{T} \sum_{k=1}^{K} \gamma_t(k) \log p(\mathbf{x}_t | z_t = k, \theta)
$$

**첫 분포**:

$$
\boxed{\pi_k^{(t+1)} = \gamma_1(k)}
$$

**옮김 행렬**:

$$
\boxed{A_{jk}^{(t+1)} = \frac{\sum_{t=1}^{T-1} \xi_t(j,k)}{\sum_{t=1}^{T-1} \gamma_t(j)}}
$$

**내보냄 매개변수**: 내보냄 분포의 집안(가우스, 다항 등)에 달렸으며 맡음 몫 $\gamma_t(k)$을 무게로 써서 새로 고친다.

---

## 수치적 안정성에 대한 고려

### 로그 공간에서 셈하기

맡음 몫을 곧바로 셈하면 밑넘침이 날 수 있다:

$$
\gamma_{ik} = \frac{\pi_k \mathcal{N}(\mathbf{x}_i | \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)}{\sum_j \pi_j \mathcal{N}(\mathbf{x}_i | \boldsymbol{\mu}_j, \boldsymbol{\Sigma}_j)}
$$

**풀이**: log-sum-exp 재주로 로그 공간에서 셈한다:

$$
\log \gamma_{ik} = \log \pi_k + \log \mathcal{N}(\mathbf{x}_i | \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k) - \text{LogSumExp}_j \left( \log \pi_j + \log \mathcal{N}(\mathbf{x}_i | \boldsymbol{\mu}_j, \boldsymbol{\Sigma}_j) \right)
$$

여기서 각 기호는 다음과 같다.

$$
\text{LogSumExp}(a_1, \ldots, a_K) = a_{\max} + \log \sum_{k=1}^{K} \exp(a_k - a_{\max})
$$

### 공분산에 벌주기

특이한 공분산은 수치 말썽을 일으킨다. 흔한 손보기는 이렇다:

1. **대각에 얹기**: $\boldsymbol{\Sigma}_k \leftarrow \boldsymbol{\Sigma}_k + \epsilon \mathbf{I}$
2. **최소 고윳값**: 고윳값을 문턱값 위로 묶는다
3. **묶은 공분산**: 성분끼리 $\boldsymbol{\Sigma}$을 함께 쓴다
4. **대각 공분산**: $\boldsymbol{\Sigma}_k = \text{diag}(\sigma_{k1}^2, \ldots, \sigma_{kd}^2)$으로 옭아맨다

### 성분 찌부러짐 막기

$N_k \to 0$일 때(빈 성분일 때):

1. **첫값 다시 잡기**: 성분을 무작위 자료 점으로 되돌린다
2. **합치고 쪼개기**: 빈 성분을 가장 큰 성분과 합치고 가장 큰 성분을 쪼갠다
3. **벌주기**: 앞확률 횟수를 더한다(베이즈 방식)

---

## PyTorch 구현

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
        
        # 매개변수를 초기화한다
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
        M 걸음: 매개변수 새로 고치기.
        
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

---

## 요약

| 걸음 | 들임 | 내임 | 셈하기 |
|------|-------|--------|-------------|
| **E 걸음** | 지금의 $\theta^{(t)}$, 자료 $\mathbf{X}$ | 맡음 몫 $\gamma$이나 뒤확률의 적률 | 베이즈 정리, 앞뒤 알고리즘 |
| **M 걸음** | 맡음 몫 $\gamma$, 자료 $\mathbf{X}$ | 새로 고친 $\theta^{(t+1)}$ | 무게 붙인 최대 가능도 어림, 닫힌 꼴이나 최적화 |

### 핵심 통찰

1. **E 걸음이 단순해짐**: 흔히 온전한 뒤확률이 아니라 기댓값 충분 통계량만 있으면 된다
2. **M 걸음을 다룰 수 있음**: 지수 집안의 짜임이 닫힌 꼴 새로 고침을 준다
3. **수치 안정**: 실전에서는 로그 공간 셈하기와 벌주기가 꼭 필요하다
4. **독립**: 자료 점에 걸친 인수 나눔이 효율적이고 나란히 할 수 있는 구현을 가능하게 한다

이 두 걸음을 번갈아 밟는 것, 곧 숨은 변수가 무엇일지 기댓값을 셈하고(E) 그 기댓값이 참인 양 매개변수를 최적화하는 것(M)이 EM 알고리즘의 우아한 알맹이이다.

---

# 덧붙임: EM 되풀이 — 자세히 이끌어 내기

# EM 되풀이

EM 알고리즘은 E 걸음(기댓값)과 M 걸음(최대화)을 번갈아 밟는다. 이 절에서는 두 걸음을 온전히 이끌어 내고, 단조롭게 나아짐의 보장을 증명하며, EM을 ELBO에 대한 좌표 오르기로 풀이한다.

---

## E 걸음(기댓값)

### 숨은 변수의 뒤확률 셈하기

E 걸음은 지금의 매개변수 어림값 $\theta^{(t)}$이 주어졌을 때 숨은 변수에 대한 **뒤확률 분포**를 셈한다:

$$
q^{(t+1)}(\mathbf{Z}) = p(\mathbf{Z} | \mathbf{X}, \theta^{(t)})
$$

베이즈 정리를 쓰면:

$$
p(\mathbf{Z} | \mathbf{X}, \theta^{(t)}) = \frac{p(\mathbf{X}, \mathbf{Z} | \theta^{(t)})}{p(\mathbf{X} | \theta^{(t)})} = \frac{p(\mathbf{X} | \mathbf{Z}, \theta^{(t)}) \, p(\mathbf{Z} | \theta^{(t)})}{\int p(\mathbf{X}, \mathbf{Z}' | \theta^{(t)}) \, d\mathbf{Z}'}
$$

**핵심**: E 걸음은 뒤확률을 셈해야 하는데, 여기에는 곧바른 최적화를 다룰 수 없게 만든 바로 그 적분이 들어 있다. 그러나 많은 모형에서(특히 지수 집안에 드는 모형에서) 이 뒤확률에는 다룰 수 있는 닫힌 꼴이 있다.

### 보기: 가우스 섞음 모형

성분이 $K$개인 가우스 섞음 모형에서 숨은 변수 $z_i \in \{1, \ldots, K\}$은 관측 $\mathbf{x}_i$의 무리 소속을 가리킨다. E 걸음은 **맡음 몫**을 셈한다:

$$
\gamma_{ik} = p(z_i = k | \mathbf{x}_i, \theta^{(t)}) = \frac{\pi_k^{(t)} \, \mathcal{N}(\mathbf{x}_i | \boldsymbol{\mu}_k^{(t)}, \boldsymbol{\Sigma}_k^{(t)})}{\sum_{j=1}^{K} \pi_j^{(t)} \, \mathcal{N}(\mathbf{x}_i | \boldsymbol{\mu}_j^{(t)}, \boldsymbol{\Sigma}_j^{(t)})}
$$

이 맡음 몫은 자료 점마다 무리마다의 부드러운 배정을 나타낸다.

### 경계를 팽팽하게 만들기

E 걸음은 결정적인 일을 한다. 곧 지금의 매개변수 값 $\theta^{(t)}$에서 ELBO를 **팽팽하게** 만든다.

근본 쪼갬을 떠올리자:

$$
\ell(\theta) = \mathcal{L}(q, \theta) + D_{\mathrm{KL}}\bigl(q(\mathbf{Z}) \,\|\, p(\mathbf{Z} | \mathbf{X}, \theta)\bigr)
$$

$q(\mathbf{Z}) = p(\mathbf{Z} | \mathbf{X}, \theta^{(t)})$으로 두면:

$$
D_{\mathrm{KL}}\bigl(p(\mathbf{Z} | \mathbf{X}, \theta^{(t)}) \,\|\, p(\mathbf{Z} | \mathbf{X}, \theta^{(t)})\bigr) = 0
$$

따라서 다음이 성립한다.

$$
\mathcal{L}(q^{(t+1)}, \theta^{(t)}) = \ell(\theta^{(t)})
$$

**$\theta^{(t)}$에서 ELBO가 로그 가능도와 같아진다**. 곧 경계가 팽팽하다.

### 팽팽함이 왜 중요한가

지금 점에서 경계를 팽팽하게 하면 다음이 보장된다:

1. ELBO가 나아지면 로그 가능도도 나아진다
2. 경계가 헐거워서 어설픈 점에 갇히는 일이 없다
3. M 걸음 최적화의 또렷한 출발점을 얻는다

---

## M 걸음(최대화)

### Q 함수

M 걸음은 $q$을 $q^{(t+1)}$으로 붙박아 둔 채 $\theta$에 대해 ELBO를 가장 크게 한다. $q^{(t+1)} = p(\mathbf{Z} | \mathbf{X}, \theta^{(t)})$이므로 ELBO는 다음이 된다:

$$
\mathcal{L}(q^{(t+1)}, \theta) = \mathbb{E}_{p(\mathbf{Z} | \mathbf{X}, \theta^{(t)})}[\log p(\mathbf{X}, \mathbf{Z} | \theta)] + H[q^{(t+1)}]
$$

엔트로피 항 $H[q^{(t+1)}]$은 $\theta$에 달려 있지 않으므로 $\mathcal{L}$을 가장 크게 하는 것은 **Q 함수**를 가장 크게 하는 것과 같다:

$$
Q(\theta | \theta^{(t)}) = \mathbb{E}_{p(\mathbf{Z} | \mathbf{X}, \theta^{(t)})}[\log p(\mathbf{X}, \mathbf{Z} | \theta)]
$$

이것이 **기댓값 완전 자료 로그 가능도**이며, 여기서 기댓값은 E 걸음에서 셈한 숨은 변수의 뒤확률 분포에 대해 취한다.

### Q 함수 최적화

M 걸음은 다음을 찾는다:

$$
\theta^{(t+1)} = \arg\max_\theta Q(\theta | \theta^{(t)})
$$

**Q이 왜 최적화하기 쉬운가**:

1. **$\theta$에 걸친 적분이 없음**: 로그가 적분 바깥이 아니라 기댓값 안에 있다
2. **지수 집안의 짜임**: 지수 집안 모형에서는 $Q$의 최댓점이 흔히 닫힌 꼴이다
3. **풀림**: 매개변수가 흔히 서로 풀려 따로 최적화할 수 있다

### 기댓값 완전 자료 로그 가능도

Q 함수를 펼치면:

$$
Q(\theta | \theta^{(t)}) = \int p(\mathbf{Z} | \mathbf{X}, \theta^{(t)}) \log p(\mathbf{X}, \mathbf{Z} | \theta) \, d\mathbf{Z}
$$

숨은 변수가 띄엄띄엄하면:

$$
Q(\theta | \theta^{(t)}) = \sum_{\mathbf{Z}} p(\mathbf{Z} | \mathbf{X}, \theta^{(t)}) \log p(\mathbf{X}, \mathbf{Z} | \theta)
$$

### 보기: 가우스 섞음 모형의 M 걸음

가우스 섞음 모형에서 M 걸음의 새로 고침은 다음과 같다:

**섞음 비율**:

$$
\pi_k^{(t+1)} = \frac{1}{N} \sum_{i=1}^{N} \gamma_{ik}
$$

**평균**:

$$
\boldsymbol{\mu}_k^{(t+1)} = \frac{\sum_{i=1}^{N} \gamma_{ik} \, \mathbf{x}_i}{\sum_{i=1}^{N} \gamma_{ik}}
$$

**공분산**:

$$
\boldsymbol{\Sigma}_k^{(t+1)} = \frac{\sum_{i=1}^{N} \gamma_{ik} \, (\mathbf{x}_i - \boldsymbol{\mu}_k^{(t+1)})(\mathbf{x}_i - \boldsymbol{\mu}_k^{(t+1)})^\top}{\sum_{i=1}^{N} \gamma_{ik}}
$$

이는 표준 최대 가능도 어림꼴에 무게를 붙인 것이며, 그 무게가 E 걸음에서 나온 맡음 몫이다.

### 일부만 하는 M 걸음(넓힌 EM)

때로는 $Q(\theta | \theta^{(t)})$의 전체 최댓점을 찾기가 어렵다. **넓힌 EM(GEM)**은 다음만 요구한다:

$$
Q(\theta^{(t+1)} | \theta^{(t)}) \geq Q(\theta^{(t)} | \theta^{(t)})
$$

$Q$이 조금이라도 나아지면 넉넉하며 전체 최댓점은 필요 없다. 이는 다음일 때 쓸모 있다:

- M 걸음에 제약 있는 최적화가 들어 있을 때
- 닫힌 꼴 풀이가 없을 때
- 기울기를 쓰는 방법을 쓸 때

---

## 단조롭게 나아짐의 보장

### 한가운데 정리

**정리(단조롭게 나아짐)**: 어떤 EM 되풀이에서도 로그 가능도는 결코 줄지 않는다:

$$
\ell(\theta^{(t+1)}) \geq \ell(\theta^{(t)})
$$

$\theta^{(t+1)} = \theta^{(t)}$일 때 그리고 오직 그때만 등호가 성립한다(곧 붙박이 점에 있을 때이다).

### 증명

부등식의 사슬로 이를 세운다:

$$
\ell(\theta^{(t+1)}) \geq \mathcal{L}(q^{(t+1)}, \theta^{(t+1)}) \geq \mathcal{L}(q^{(t+1)}, \theta^{(t)}) = \ell(\theta^{(t)})
$$

**걸음 1 — ELBO는 아래 경계이다**:

$$
\ell(\theta^{(t+1)}) \geq \mathcal{L}(q^{(t+1)}, \theta^{(t+1)})
$$

ELBO가 아무 $q$과 아무 $\theta$에 대해서도 **늘** 로그 가능도의 아래 경계이므로 이것이 성립한다:

$$
\ell(\theta) = \mathcal{L}(q, \theta) + D_{\mathrm{KL}}(q \| p(\mathbf{Z}|\mathbf{X}, \theta)) \geq \mathcal{L}(q, \theta)
$$

$D_{\mathrm{KL}} \geq 0$이기 때문이다.

**걸음 2 — M 걸음이 ELBO를 낫게 한다**:

$$
\mathcal{L}(q^{(t+1)}, \theta^{(t+1)}) \geq \mathcal{L}(q^{(t+1)}, \theta^{(t)})
$$

이는 M 걸음의 정의에서 성립한다. 곧 $\theta^{(t+1)}$은 $\theta$에 대해 $\mathcal{L}(q^{(t+1)}, \theta)$을 **가장 크게** 하도록 고른다.

**걸음 3 — E 걸음이 경계를 팽팽하게 한다**:

$$
\mathcal{L}(q^{(t+1)}, \theta^{(t)}) = \ell(\theta^{(t)})
$$

E 걸음이 $q^{(t+1)} = p(\mathbf{Z} | \mathbf{X}, \theta^{(t)})$으로 두어 KL 벌어짐을 0으로 만들기 때문에 성립한다:

$$
D_{\mathrm{KL}}(q^{(t+1)} \| p(\mathbf{Z}|\mathbf{X}, \theta^{(t)})) = 0
$$

**모든 걸음을 합치면**:

$$
\ell(\theta^{(t+1)}) \geq \mathcal{L}(q^{(t+1)}, \theta^{(t+1)}) \geq \mathcal{L}(q^{(t+1)}, \theta^{(t)}) = \ell(\theta^{(t)})
$$

### 기하 그림

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

1. **E 걸음**: $\theta^{(t)}$에서 $\ell(\theta)$에 닿는 아래 경계를 세운다
2. **M 걸음**: 그 경계를 가장 크게 하는 $\theta^{(t+1)}$으로 옮겨 간다
3. $\theta^{(t+1)}$에서 경계가 헐거울 수 있지만 $\ell(\theta^{(t+1)})$은 오히려 더 높다

### 엄격히 나아짐

$\theta^{(t+1)} \neq \theta^{(t)}$이고 M 걸음이 경계를 엄격히 낫게 하면 다음이 성립한다:

$$
\ell(\theta^{(t+1)}) > \ell(\theta^{(t)})
$$

이러면 알고리즘에 순환이 생길 수 없다. 곧 EM은 붙박이 점으로 모이거나 되풀이마다 엄격히 나아진다.

### 나아짐은 언제 멈추나?

늘어놓음 $\{\ell(\theta^{(t)})\}$은 (가능도가 제대로 되었다고 놓으면) 단조롭게 커지고 위로 묶인다. **단조 모임 정리**에 따라 그 늘어놓음은 모인다.

**멈춤 조건**: 모인 자리에서 $\theta^* = \theta^{(t)} = \theta^{(t+1)}$이며, 이는 다음을 뜻한다:

$$
\nabla_\theta Q(\theta | \theta^*)\big|_{\theta = \theta^*} = \nabla_\theta \mathcal{L}(q^*, \theta)\big|_{\theta = \theta^*} = 0
$$

이는 $\ell(\theta)$이 그 자리에서 최적이기 위한 **필요조건**이지만 전체 최적을 위한 넉넉한 조건은 아니다. 곧 EM은 그 자리 최댓점이나 안장점으로 모일 수 있다.

---

## 좌표 오르기로 풀이하기

### 결합 목표로서의 ELBO

EM 알고리즘은 분포 $q(\mathbf{Z})$과 매개변수 $\theta$ 둘 다에 달린 범함수 $\mathcal{L}(q, \theta)$에 대한 **좌표 오르기**로 이해할 수 있다.

### 두 덩어리 최적화

**E 걸음**: $\theta = \theta^{(t)}$을 붙박아 둔 채 $q$에 대해 $\mathcal{L}(q, \theta)$을 가장 크게 한다:

$$
q^{(t+1)} = \arg\max_q \mathcal{L}(q, \theta^{(t)})
$$

그 풀이는 $q^{(t+1)} = p(\mathbf{Z} | \mathbf{X}, \theta^{(t)})$이며 이는 $D_{\mathrm{KL}} = 0$으로 만든다.

**M 걸음**: $q = q^{(t+1)}$을 붙박아 둔 채 $\theta$에 대해 $\mathcal{L}(q, \theta)$을 가장 크게 한다:

$$
\theta^{(t+1)} = \arg\max_\theta \mathcal{L}(q^{(t+1)}, \theta)
$$

### 좌표 오르기가 왜 되나

걸음마다 ELBO가 커지거나 그대로이다:

$$
\mathcal{L}(q^{(t)}, \theta^{(t)}) \leq \mathcal{L}(q^{(t+1)}, \theta^{(t)}) \leq \mathcal{L}(q^{(t+1)}, \theta^{(t+1)})
$$

ELBO가 로그 가능도를 아래에서 받치고 E 걸음이 경계를 팽팽하게 하므로 ELBO가 나아지면 $\ell(\theta)$도 나아진다.

### E 걸음의 범함수 최적화

E 걸음은 **범함수 최적화** 문제이다. 곧 모든 분포 $q(\mathbf{Z})$의 공간에 걸쳐 최적화한다. 이는 차원이 무한한 최적화이다!

**놀랍게도** 그 풀이에는 닫힌 꼴이 있다. 변분법이나 라그랑주 곱수를 쓰면:

$$
\frac{\delta}{\delta q(\mathbf{Z})} \left[ \mathcal{L}(q, \theta) - \lambda \left( \int q(\mathbf{Z}) d\mathbf{Z} - 1 \right) \right] = 0
$$

이러면 다음이 나온다:

$$
\log q^*(\mathbf{Z}) = \log p(\mathbf{X}, \mathbf{Z} | \theta) - \log p(\mathbf{X} | \theta)
$$

따라서 $q^*(\mathbf{Z}) = p(\mathbf{Z} | \mathbf{X}, \theta)$, 곧 뒤확률 분포이다.

### 덩어리 좌표 내리기와의 이음

좌표 오르기의 눈은 EM을 더 넓은 갈래의 최적화 알고리즘과 이어 준다:

| 알고리즘 | 변수 | 새로 고치는 규칙 |
|-----------|-----------|-------------|
| EM | $(q, \theta)$ | 번갈아 최대화 |
| 깁스 표집 | $(z_1, \ldots, z_d)$ | 조건부를 돌아가며 다룸 |
| ADMM | $(x, z, u)$ | 쌍대 새로 고침과 함께 번갈아 |

모두 다른 것을 붙박아 둔 채 한 덩어리를 최적화한다는 성질을 함께 지닌다.

### 좌표 오르기의 눈이 뜻하는 바

1. **모임**: 좌표 오르기의 표준 결과가 그대로 쓰인다. 곧 너그러운 규칙 조건 아래에서 EM은 멈춘 점으로 모인다

2. **모이는 빠르기**: 모이는 빠르기는 $\mathcal{L}$의 굽음과 $q$과 $\theta$의 얽힘에 달렸다

3. **넓히기**: 이 눈은 다음과 같은 갈래로 이어진다:
   - **일부만 하는 E 걸음**: $q$에 대해 온전히 최적화하지 않는다
   - **일부만 하는 M 걸음**: $\theta$에 대해 온전히 최적화하지 않는다(넓힌 EM)
   - **변분 EM**: $q$을 다룰 수 있는 집안으로 옭아맨다

4. **변분 추론과의 이음**: 정확한 뒤확률을 다룰 수 없으면 $q$을 변분 집안으로 옭아매고도 좌표 오르기를 할 수 있다. 이것이 **변분 추론**이다.

---

## 간추림: 온전한 EM 되풀이

지금의 매개변수가 $\theta^{(t)}$일 때:

### E 걸음

1. 숨은 변수에 대한 뒤확률을 셈한다:

   $$q^{(t+1)}(\mathbf{Z}) = p(\mathbf{Z} | \mathbf{X}, \theta^{(t)})$$

2. 이러면 $\theta^{(t)}$에서 ELBO가 팽팽해진다:

   $$\mathcal{L}(q^{(t+1)}, \theta^{(t)}) = \ell(\theta^{(t)})$$

### M 걸음

1. Q 함수를 정한다:

   $$Q(\theta | \theta^{(t)}) = \mathbb{E}_{q^{(t+1)}}[\log p(\mathbf{X}, \mathbf{Z} | \theta)]$$

2. 가장 크게 하여 새 매개변수를 얻는다:

   $$\theta^{(t+1)} = \arg\max_\theta Q(\theta | \theta^{(t)})$$

### 보장

- **단조롭게 나아짐**: $\ell(\theta^{(t+1)}) \geq \ell(\theta^{(t)})$
- **모임**: 늘어놓음 $\{\theta^{(t)}\}$이 멈춘 점으로 모인다
- **순환 없음**: 걸음마다 엄격히 나아지거나 모인다

### 알고리즘 간추림

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
