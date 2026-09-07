# EM의 갈래
고전 EM 알고리즘은 우아하지만 한계가 있다. 곧 느리게 모이고, 첫값에 예민하며, E 걸음이나 M 걸음에 닫힌 꼴 풀이가 없으면 쓸 수 없다. 이 절에서는 이 어려움을 다루어 EM이 미치는 문제의 갈래를 넓히는 중요한 갈래들을 보인다.

---

## 넓힌 EM(GEM)

### 동기

표준 M 걸음은 Q 함수를 **가장 크게** 해야 한다:

$$
\theta^{(t+1)} = \arg\max_\theta Q(\theta | \theta^{(t)})
$$

많은 모형에서 이 최대화에는 닫힌 꼴 풀이가 없다. 넓힌 EM은 이 요구를 느슨하게 한다.

### GEM의 원리

**넓힌 EM**은 M 걸음이 Q 함수를 가장 크게 하는 대신 **키우기만** 하면 된다고 요구한다:

$$
Q(\theta^{(t+1)} | \theta^{(t)}) \geq Q(\theta^{(t)} | \theta^{(t)})
$$

조금이라도 나아지면 넉넉하며 전체 최댓점을 찾을 필요는 없다.

### 이론의 보장

**정리**: GEM은 단조롭게 나아지는 성질을 그대로 지닌다:

$$
\ell(\theta^{(t+1)}) \geq \ell(\theta^{(t)})
$$

**증명**: 같은 부등식의 사슬이 성립한다:

$$
\ell(\theta^{(t+1)}) \geq \mathcal{L}(q^{(t+1)}, \theta^{(t+1)}) \geq \mathcal{L}(q^{(t+1)}, \theta^{(t)}) = \ell(\theta^{(t)})
$$

둘째 부등식은 $\theta^{(t+1)}$이 ELBO를 가장 크게 하는 것이 아니라 낫게 하기만 하면 성립한다.

### 구현 전략

**기울기 오르기 M 걸음**: 기울기 걸음을 한 번 이상 뗀다:

$$
\theta^{(t+1)} = \theta^{(t)} + \eta \nabla_\theta Q(\theta | \theta^{(t)})\big|_{\theta = \theta^{(t)}}
$$

**조건부 최대화**: 매개변수의 부분 모음에 대해 차례로 가장 크게 한다(아래 ECM 참고).

**제약 있는 최적화**: 매개변수에 제약이 있으면 쏘아 내린 기울기 방법이면 넉넉하다.

### GEM을 언제 쓰나

- M 걸음에 닫힌 꼴 풀이가 없을 때
- 매개변수 공간에 제약이 있을 때
- 셈 예산이 빠듯할 때
- 가능도의 짜임이 복잡할 때

---

## 기댓값 조건부 최대화(ECM)

### 얽힌 매개변수의 어려움

많은 모형에서 매개변수 묶음을 따로따로 최적화하기는 쉬운데도 M 걸음에서 모두를 한꺼번에 최적화하는 일은 다룰 수 없다.

### ECM 알고리즘

**ECM**은 M 걸음을 **조건부 최대화(CM) 걸음**의 늘어놓음으로 바꾼다. 걸음마다 다른 것을 붙박아 둔 채 매개변수의 부분 모음에 대해 가장 크게 한다.

매개변수를 $\theta = (\theta_1, \theta_2, \ldots, \theta_S)$으로 나눈다. ECM의 되풀이는 다음과 같다:

**E 걸음**: $q^{(t+1)}(\mathbf{Z}) = p(\mathbf{Z} | \mathbf{X}, \theta^{(t)})$을 셈한다

**CM 걸음**: $s = 1, \ldots, S$에 대해:

$$
\theta_s^{(t+1)} = \arg\max_{\theta_s} Q(\theta_1^{(t+1)}, \ldots, \theta_{s-1}^{(t+1)}, \theta_s, \theta_{s+1}^{(t)}, \ldots, \theta_S^{(t)} | \theta^{(t)})
$$

CM 걸음마다 $\theta_1, \ldots, \theta_{s-1}$에는 새로 고친 값을, $\theta_{s+1}, \ldots, \theta_S$에는 예전 값을 쓰면서 $\theta_s$에 대해 가장 크게 한다.

### 보기: 인자 분석

매개변수: 적재 행렬 $\mathbf{W}$, 잡음 흩어짐 $\boldsymbol{\Psi}$(대각).

- **CM 걸음 1**: 지금의 $\boldsymbol{\Psi}$이 주어졌을 때 $\mathbf{W}$을 새로 고친다
- **CM 걸음 2**: 새 $\mathbf{W}$이 주어졌을 때 $\boldsymbol{\Psi}$을 새로 고친다

걸음마다 닫힌 꼴이 있지만 한꺼번에 최적화하면 없다.

### 수렴의 성질

**정리(멩과 루빈, 1993)**: ECM은 EM의 모임 성질을 물려받는다:

1. 단조롭게 나아짐: $\ell(\theta^{(t+1)}) \geq \ell(\theta^{(t)})$
2. 표준 규칙 조건 아래에서 멈춘 점으로의 모임

핵심 통찰은 이렇다. CM 걸음마다 Q 함수를 키우거나 그대로 두며, 이는 GEM의 보장에 넉넉하다.

---

## ECME 알고리즘

### 동기

ECM은 모든 CM 걸음에 Q 함수를 쓰지만, 매개변수의 어떤 부분 모음에서는 **실제 로그 가능도** $\ell(\theta)$을 곧바로 가장 크게 하는 편이 더 쉬울 때가 있다.

### ECME = ECM + 곧바른 가능도 걸음

**ECME(기댓값/조건부 최대화 택일)**은 CM 걸음마다 다음 가운데 하나를 가장 크게 하도록 허락한다:

- Q 함수 $Q(\theta | \theta^{(t)})$, **또는**
- 로그 가능도 $\ell(\theta)$ 자체

더 편하거나 효율적인 쪽을 고르면 된다.

### 알고리즘의 짜임

매개변수를 $\theta = (\theta_1, \ldots, \theta_S)$으로 나눌 때:

**E 걸음**: 뒤확률 $p(\mathbf{Z} | \mathbf{X}, \theta^{(t)})$을 셈한다

**CM 걸음**: $s = 1, \ldots, S$에 대해:

$$
\theta_s^{(t+1)} = \arg\max_{\theta_s} F_s(\theta_s)
$$

여기서 $F_s$은 다음 가운데 하나이다:

- $Q(\theta_1^{(t+1)}, \ldots, \theta_s, \ldots, \theta_S^{(t)} | \theta^{(t)})$(Q 걸음), 또는
- $\ell(\theta_1^{(t+1)}, \ldots, \theta_s, \ldots, \theta_S^{(t)})$(L 걸음)

### ECME이 왜 더 빠를 수 있나

곧바른 가능도 최대화 걸음은 다음을 할 수 있다:

1. 그 매개변수에 대한 E 걸음의 덧짐을 건너뛴다
2. 매개변수 공간에서 더 큰 걸음을 뗀다
3. Q 함수로는 쓸 수 없는 문제의 짜임을 써먹는다

### 모임

**정리**: ECME은 단조롭게 나아짐을 지킨다:

- Q 걸음은 $\ell(\theta)$을 아래에서 받치는 $\mathcal{L}(q, \theta)$을 키운다
- L 걸음은 $\ell(\theta)$을 곧바로 키운다

둘 다 늘어놓음 $\{\ell(\theta^{(t)})\}$이 줄지 않음을 보장한다.

---

## 매개변수를 넓힌 EM(PX-EM)

### 느리게 모이는 문제

**빠진 정보의 몫**이 크면 EM이 느리게 모일 수 있다(모임 이론 참고). PX-EM은 매개변수 공간을 잠시 넓혀 이를 다룬다.

### 생각

1. **넓히기**: 주변 모형을 바꾸지 않는 도움 매개변수를 들여온다
2. **넓힌 공간에서의 EM**: 자유도를 늘려 EM을 돌린다
3. **줄이기**: 본디 매개변수 공간으로 쏘아 내린다

넓힌 공간은 흔히 조건이 더 좋아 더 빨리 모이게 한다.

### 엄밀한 차림

$\theta \in \Theta$을 본디 매개변수라 하자. $\dim(\Phi) > \dim(\Theta)$인 넓힌 매개변수 $\phi \in \Phi$을 들여온다.

다음을 만족하는 **줄임 함수** $R: \Phi \to \Theta$을 정한다:

$$
p(\mathbf{X} | R(\phi)) = p(\mathbf{X} | \theta) \quad \text{for all } \phi \text{ with } R(\phi) = \theta
$$

넓힌 모형은 주변 가능도가 같지만 매개변수가 더 많다.

### PX-EM 되풀이

1. **E 걸음**: 지금의 $\theta^{(t)}$으로 기댓값을 셈한다
2. **넓힌 공간에서의 M 걸음**: $\phi^{(t+1)} = \arg\max_\phi Q_\phi(\phi | \theta^{(t)})$을 찾는다
3. **줄이기**: $\theta^{(t+1)} = R(\phi^{(t+1)})$으로 둔다

### 보기: 공분산 매개변수 넓히기

적재 행렬이 $\mathbf{W}$이고 숨은 변수의 공분산이 $\mathbf{I}$인 인자 분석에서:

**본디 모형**: $\mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$일 때 $p(\mathbf{x} | \mathbf{z}) = \mathcal{N}(\mathbf{W}\mathbf{z}, \boldsymbol{\Psi})$

**넓힌 모형**: 일반 양의 정부호 $\boldsymbol{\Sigma}_z$에 대해 $\mathbf{z} \sim \mathcal{N}(\mathbf{0}, \boldsymbol{\Sigma}_z)$을 허락한다

**줄이기**: $\mathbf{W}' \boldsymbol{\Sigma}_z' \mathbf{W}'^\top = \mathbf{W} \mathbf{W}^\top$을 만족하는 아무 $(\mathbf{W}', \boldsymbol{\Sigma}_z')$이나 같은 주변 모형을 준다.

넓힌 M 걸음은 더 크게 움직일 수 있으며, 줄이고 나면 그것이 더 나은 표준 EM 새로 고침에 맞대응된다.

### 모이는 빠르기가 나아짐

PX-EM은 보통 다음을 이룬다:

$$
\rho_{\text{PX-EM}} < \rho_{\text{EM}}
$$

넓히기가 사실상 빠진 정보의 일부를 "메워" 잃은 몫을 줄인다.

---

## 몬테카를로 EM(MCEM)

### E 걸음을 다룰 수 없을 때

어떤 모형에서는 뒤확률 $p(\mathbf{Z} | \mathbf{X}, \theta)$에 닫힌 꼴이 없어 다음을 정확히 셈할 수 없다:

$$
Q(\theta | \theta^{(t)}) = \mathbb{E}_{p(\mathbf{Z} | \mathbf{X}, \theta^{(t)})}[\log p(\mathbf{X}, \mathbf{Z} | \theta)]
$$

불가능하다.

### MCEM의 어림

**몬테카를로 EM**은 뒤확률에서 뽑은 표본으로 E 걸음의 기댓값을 어림한다:

1. **표집**: $\mathbf{Z}^{(1)}, \ldots, \mathbf{Z}^{(M)} \sim p(\mathbf{Z} | \mathbf{X}, \theta^{(t)})$을 뽑는다
2. **Q 어림하기**:

$$
\hat{Q}(\theta | \theta^{(t)}) = \frac{1}{M} \sum_{m=1}^{M} \log p(\mathbf{X}, \mathbf{Z}^{(m)} | \theta)
$$

3. **M 걸음**: $\theta^{(t+1)} = \arg\max_\theta \hat{Q}(\theta | \theta^{(t)})$

### 표집 방법

뒤확률에서 표집하는 흔한 길은 다음과 같다:

- **깁스 표집**: 조건부 분포를 다룰 수 있을 때
- **메트로폴리스-헤이스팅스**: 두루 쓰는 MCMC
- **중요도 표집**: 알맞은 제안 분포가 있을 때
- **차례차례 몬테카를로**: 차례 있는 숨은 변수 모형에

### 모임에서 살필 점

MCEM은 **몬테카를로 잡음**을 들여온다. 모임을 보장하려면:

**1. 표본 크기 키우기**: $t \to \infty$일 때 $M_t \to \infty$이 되게 한다. 흔한 일정은 다음과 같다:

- $M_t = M_0$(상수. 최적점으로 모이지 않을 수 있다)
- $M_t = t$(선형으로 자람)
- $M_t = M_0 \cdot c^t$(지수로 자람)

**2. 평균내기**: 흩어짐을 줄이려고 매개변수 어림값의 달리는 평균을 쓴다.

**3. 오름 조건**: 오름을 보장하도록 Q 어림의 정확도를 넉넉히 지킨다.

### 실전 MCEM

```python
def mcem_iteration(X, theta, n_samples, sampler):
    """
    몬테카를로 EM의 한 되풀이.
    
    인수:
        X: 관측한 자료
        theta: 지금의 매개변수
        n_samples: 몬테카를로 표본의 개수
        sampler: 뒤확률에서 Z을 표집하는 함수
    
    반환값:
        새로 고친 theta
    """
    # E 걸음: 뒤확률에서 표집
    Z_samples = [sampler(X, theta) for _ in range(n_samples)]
    
    # M 걸음: 어림한 Q을 가장 크게 하기
    def neg_Q(new_theta):
        return -sum(
            complete_log_likelihood(X, Z, new_theta) 
            for Z in Z_samples
        ) / n_samples
    
    result = minimize(neg_Q, theta)
    return result.x
```

---

## 확률 EM(SEM)

### 또 다른 확률 방식

MCEM은 표본 여럿으로 기댓값을 어림하지만, **확률 EM**은 뒤확률에서 뽑은 **표본 하나**를 참 숨은 값인 양 다룬다.

### SEM 되풀이

1. **S 걸음(확률)**: 표본 하나 $\mathbf{Z}^{(t)} \sim p(\mathbf{Z} | \mathbf{X}, \theta^{(t)})$을 뽑는다
2. **M 걸음**: 완전 자료 가능도를 가장 크게 한다:

$$
\theta^{(t+1)} = \arg\max_\theta \log p(\mathbf{X}, \mathbf{Z}^{(t)} | \theta)
$$

### 성질

**단조롭지 않음**: EM과 달리 SEM은 단조롭게 나아짐을 보장하지 않는다. 곧 어느 걸음에서든 가능도가 줄 수 있다.

**에르고드적**: 규칙 조건 아래에서 늘어놓음 $\{\theta^{(t)}\}$은 최대 가능도 어림값 둘레에 몰린 멈춘 분포로 모이는 마르코프 사슬을 이룬다.

**살펴보기**: 확률 요소가 그 자리 최적점에서 벗어나는 데 도움이 된다. 곧 SEM은 정해진 EM보다 매개변수 공간을 더 많이 살펴본다.

### 섞음 모형의 SEM

가우스 섞음 모형에서:

1. **S 걸음**: 관측마다 무리 배정을 표집한다:

$$
z_i \sim \text{Categorical}(\gamma_{i1}, \ldots, \gamma_{iK})
$$

2. **M 걸음**: $\{z_i\}$을 관측한 양 표준 최대 가능도 어림값을 셈한다:

$$
\pi_k = \frac{1}{N} \sum_{i=1}^{N} \mathbb{1}[z_i = k]
$$

$$
\boldsymbol{\mu}_k = \frac{\sum_{i: z_i = k} \mathbf{x}_i}{\sum_{i} \mathbb{1}[z_i = k]}
$$

### SEM을 언제 쓰나

- **봉우리가 여럿인 가능도**: SEM의 확률 요소가 살펴보기를 돕는다
- **처음 살펴보기**: SEM으로 그럴듯한 구역을 찾은 뒤 EM으로 바꾼다
- **베이즈 표집**: 뒤확률 추론을 위한 MCMC의 한 부품으로 SEM을 쓴다
- **큰 자료**: 표본 하나가 온전한 기댓값보다 싸다

---

## 변분 EM

### 뒤확률을 다룰 수 없을 때

복잡한 모형에서는 정확한 뒤확률 $p(\mathbf{Z} | \mathbf{X}, \theta)$에 닫힌 꼴이 없어 E 걸음을 정확히 셈할 수 없다.

### 변분 어림

**변분 EM**은 정확한 뒤확률을 다룰 수 있는 집안 $\mathcal{Q}$의 어림 $q(\mathbf{Z})$으로 바꾼다:

$$
q^*(\mathbf{Z}) = \arg\min_{q \in \mathcal{Q}} D_{\mathrm{KL}}\bigl(q(\mathbf{Z}) \,\|\, p(\mathbf{Z} | \mathbf{X}, \theta)\bigr)
$$

### ELBO의 눈

정확한 뒤확률은 $\mathcal{L}(q, \theta) = \ell(\theta)$(팽팽한 경계)을 주지만 변분 어림은 다음을 준다:

$$
\mathcal{L}(q, \theta) = \ell(\theta) - D_{\mathrm{KL}}(q \| p(\mathbf{Z}|\mathbf{X}, \theta)) \leq \ell(\theta)
$$

변분 EM은 참 가능도 대신 이 아래 경계를 가장 크게 한다.

### 평균장 어림

가장 흔한 고름은 숨은 변수가 인수로 나뉜다고 놓는 **평균장** 집안이다:

$$
q(\mathbf{Z}) = \prod_{j} q_j(z_j)
$$

이러면 최적화가 크게 단순해지면서도 흔히 좋은 어림을 준다.

### 변분 E 걸음

평균장에서 가장 좋은 인수는 다음을 만족한다:

$$
\log q_j^*(z_j) = \mathbb{E}_{q_{-j}}[\log p(\mathbf{X}, \mathbf{Z} | \theta)] + \text{const}
$$

여기서 $q_{-j}$은 $q_j$을 뺀 모든 인수를 뜻한다. 이는 **좌표 오르기 변분 추론(CAVI)**으로 이어진다.

### 변분 EM 알고리즘

1. **첫값 잡기**: $q^{(0)}(\mathbf{Z})$, $\theta^{(0)}$

2. **모일 때까지 되풀이한다**:

   **변분 E 걸음**: 변분 분포를 새로 고친다

   $$
   q^{(t+1)} = \arg\max_q \mathcal{L}(q, \theta^{(t)})
   $$

   (CAVI이나 다른 변분 방법으로)
   
   **M 걸음**: 매개변수를 새로 고친다

   $$
   \theta^{(t+1)} = \arg\max_\theta \mathcal{L}(q^{(t+1)}, \theta)
   $$

### 모임

변분 EM은 $\ell(\theta)$이 아니라 **ELBO의 그 자리 최적점**으로 모인다. 그러나:

- ELBO는 흔히 가능도의 좋은 대리가 된다
- 변분 EM은 정확한 EM이 불가능한 모형에서도 추론할 수 있게 한다
- 틈 $\ell(\theta) - \mathcal{L}(q, \theta)$을 지켜볼 수 있다

---

## 조금씩 하는 EM과 흐름 속 EM

### 동기

표준 EM은 E 걸음마다 자료를 한 번 다 훑어야 하는데, 이는 큰 자료나 흐르는 자료에는 감당할 수 없다.

### 조금씩 하는 EM

자료를 작은 묶음으로 다루며 충분 통계량을 조금씩 새로 고친다:

**표준 EM의 충분 통계량**:

$$
S_k = \sum_{i=1}^{N} \gamma_{ik}, \quad T_k = \sum_{i=1}^{N} \gamma_{ik} \mathbf{x}_i
$$

묶음 $\mathcal{B}$으로 하는 **조금씩 새로 고치기**:

$$
S_k \leftarrow S_k + \sum_{i \in \mathcal{B}} \gamma_{ik}, \quad T_k \leftarrow T_k + \sum_{i \in \mathcal{B}} \gamma_{ik} \mathbf{x}_i
$$

M 걸음은 쌓인 통계량을 쓴다.

### 흐름 속 EM

참으로 흐르는 자료에는 **확률 어림**을 쓴다:

관측 $\mathbf{x}_t$에 대한 **흐름 속 E 걸음**:

$$
\gamma_{tk} = \frac{\pi_k^{(t-1)} \mathcal{N}(\mathbf{x}_t | \boldsymbol{\mu}_k^{(t-1)}, \boldsymbol{\Sigma}_k^{(t-1)})}{\sum_j \pi_j^{(t-1)} \mathcal{N}(\mathbf{x}_t | \boldsymbol{\mu}_j^{(t-1)}, \boldsymbol{\Sigma}_j^{(t-1)})}
$$

배움률이 $\eta_t$인 **흐름 속 M 걸음**:

$$
\hat{S}_k^{(t)} = (1 - \eta_t) \hat{S}_k^{(t-1)} + \eta_t \gamma_{tk}
$$

$$
\hat{T}_k^{(t)} = (1 - \eta_t) \hat{T}_k^{(t-1)} + \eta_t \gamma_{tk} \mathbf{x}_t
$$

$$
\boldsymbol{\mu}_k^{(t)} = \hat{T}_k^{(t)} / \hat{S}_k^{(t)}
$$

### 배움률 일정

모이려면 $\eta_t$이 다음을 만족해야 한다:

$$
\sum_{t=1}^{\infty} \eta_t = \infty, \quad \sum_{t=1}^{\infty} \eta_t^2 < \infty
$$

흔한 고름:

- $\alpha \in (0.5, 1]$에 대해 $\eta_t = t^{-\alpha}$
- 늦춤 $\tau$을 둔 $\eta_t = (t + \tau)^{-\alpha}$

### 작은 묶음 흐름 속 EM

흩어짐을 줄이려고 작은 묶음 다루기와 흐름 속 새로 고침을 어우른다:

```python
def online_em_step(batch, theta, sufficient_stats, learning_rate):
    """흐름 속 EM에서 작은 묶음 하나 다루기."""
    
    # 묶음에 대한 E 걸음
    gamma = compute_responsibilities(batch, theta)
    
    # 묶음의 충분 통계량 셈하기
    batch_S = gamma.sum(dim=0)
    batch_T = gamma.T @ batch
    batch_V = compute_weighted_outer_products(batch, gamma, theta['mu'])
    
    # 흐름 속 새로 고침
    sufficient_stats['S'] = (1 - learning_rate) * sufficient_stats['S'] + learning_rate * batch_S * N_total / len(batch)
    sufficient_stats['T'] = (1 - learning_rate) * sufficient_stats['T'] + learning_rate * batch_T * N_total / len(batch)
    sufficient_stats['V'] = (1 - learning_rate) * sufficient_stats['V'] + learning_rate * batch_V * N_total / len(batch)
    
    # 충분 통계량으로 하는 M 걸음
    theta['pi'] = sufficient_stats['S'] / sufficient_stats['S'].sum()
    theta['mu'] = sufficient_stats['T'] / sufficient_stats['S'].unsqueeze(1)
    theta['Sigma'] = sufficient_stats['V'] / sufficient_stats['S'].unsqueeze(1).unsqueeze(2)
    
    return theta, sufficient_stats
```

---

## EM을 빠르게 하는 방법

### 모이는 빠르기 문제

모임 가까이에서 EM의 빠르기는 다음이 다스린다:

$$
\|\theta^{(t+1)} - \theta^*\| \approx \rho \|\theta^{(t)} - \theta^*\|
$$

여기서 $\rho = \rho(I_{\text{comp}}^{-1} I_{\text{miss}})$이 1에 가까울 수 있어 느리게 모이게 한다.

### 에이킨 가속

지켜본 모임으로 끝값을 밖으로 미루어 잡는다:

$$
\theta^*_{\text{est}} = \theta^{(t)} + \frac{\theta^{(t+1)} - \theta^{(t)}}{1 - \hat{\rho}}
$$

여기서 $\hat{\rho} = \frac{\|\theta^{(t+1)} - \theta^{(t)}\|}{\|\theta^{(t)} - \theta^{(t-1)}\|}$이다.

**언제 쓰나**: EM이 선형으로 모이는 국면(최적점 가까이)에 들어선 뒤.

### SQUAREM(제곱 되풀이 방법)

SQUAREM은 사실상 EM 사상을 "제곱"하여 빠르기를 $\rho$에서 $\rho^2$으로 줄인다:

$\mathbf{r} = \theta^{(t+1)} - \theta^{(t)}$, $\mathbf{v} = \theta^{(t+2)} - \theta^{(t+1)} - \mathbf{r}$이라 하자

**빠르게 한 새로 고침**:

$$
\theta^{(t+3)} = \theta^{(t)} - 2\alpha \mathbf{r} + \alpha^2 \mathbf{v}
$$

여기서 $\alpha = -\|\mathbf{r}\| / \|\mathbf{v}\|$이다.

### 유사 뉴턴 가속

어림한 이차 정보를 쓴다:

$$
\theta^{(t+1)} = \theta^{(t)} - \mathbf{H}^{-1} \nabla_\theta \ell(\theta^{(t)})
$$

여기서 $\mathbf{H}$은 BFGS이나 L-BFGS 새로 고침으로 어림한다.

**루이스 항등식**이 기울기를 준다:

$$
\nabla_\theta \ell(\theta) = \nabla_\theta Q(\theta | \theta) - \nabla_{\theta'} Q(\theta | \theta')\big|_{\theta' = \theta}
$$

EM의 붙박이 점에서는 $\nabla_\theta Q(\theta^* | \theta^*) = 0$이므로 이 식이 단순해진다.

---

## EM 갈래 견주기

| 갈래 | E 걸음 | M 걸음 | 핵심 강점 |
|---------|--------|--------|---------------|
| **표준 EM** | 정확 | 정확한 최대화 | 단순함, 단조롭게 나아짐 |
| **GEM** | 정확 | 나아지기만 하면 됨 | 닫힌 꼴이 아닌 M 걸음을 다룸 |
| **ECM** | 정확 | 조건부 최대화 | 복잡한 M 걸음을 풀어 놓음 |
| **ECME** | 정확 | Q와 L을 섞은 최대화 | 더 빨리 모임 |
| **PX-EM** | 정확 | 넓힌 공간 | 빨라진 모임 |
| **MCEM** | 몬테카를로 | 정확하거나 어림 | 다룰 수 없는 E 걸음 |
| **SEM** | 확률 표본 | 완전 자료 최대 가능도 어림 | 살펴보기, 그 자리 최적점 벗어나기 |
| **변분 EM** | 변분 어림 | 정확 | 복잡한 뒤확률 |
| **흐름 속 EM** | 조금씩 | 확률 새로 고침 | 크거나 흐르는 자료 |

### 고르는 지침

| 상황 | 권하는 갈래 |
|----------|-------------------|
| E 걸음과 M 걸음이 닫힌 꼴 | 표준 EM |
| M 걸음에 닫힌 꼴이 없음 | GEM이나 ECM |
| 매개변수가 복잡하게 얽힘 | ECM이나 ECME |
| 느리게 모임 | PX-EM, SQUAREM, 유사 뉴턴 |
| 뒤확률을 다룰 수 없음 | MCEM이나 변분 EM |
| 봉우리가 여럿인 가능도 | 살펴보기에 SEM, 그다음 EM |
| 큰 자료 | 흐름 속 EM이나 조금씩 하는 EM |
| 흐르는 자료 | 흐름 속 EM |

---

## PyTorch 구현: 가우스 섞음 모형의 변분 EM

```python
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal, Dirichlet, Categorical

class VariationalGMM:
    """
    가우스 섞음 모형의 변분 EM.
    
    섞음 무게에 디리클레 앞확률을 두고 성분 매개변수에 정규-위샤트
    앞확률을 두는 평균장 어림을 쓴다.
    """
    
    def __init__(self, n_components: int, n_features: int, 
                 alpha_0: float = 1.0, beta_0: float = 1.0,
                 nu_0: float = None, reg_covar: float = 1e-6):
        """
        인수:
            n_components: 섞음 성분의 개수
            n_features: 자료의 차원
            alpha_0: 디리클레 몰림 매개변수
            beta_0: 앞확률 정밀도의 크기 잡기
            nu_0: 위샤트 자유도(기본값: n_features)
            reg_covar: 공분산 벌주기
        """
        self.K = n_components
        self.d = n_features
        self.reg_covar = reg_covar
        
        # 앞확률의 웃매개변수
        self.alpha_0 = alpha_0
        self.beta_0 = beta_0
        self.nu_0 = nu_0 if nu_0 is not None else float(n_features)
        
        # 변분 매개변수(fit에서 첫값을 잡는다)
        self.alpha_ = None      # 섞음 무게의 디리클레 매개변수
        self.beta_ = None       # 정밀도 크기 잡기
        self.m_ = None          # 평균 매개변수
        self.W_ = None          # 위샤트 규모 행렬
        self.nu_ = None         # 위샤트 자유도
        
    def _initialize(self, X: torch.Tensor):
        """변분 매개변수 첫값 잡기."""
        N, d = X.shape
        
        # k 평균으로 평균 첫값 잡기
        indices = torch.randperm(N)[:self.K]
        self.m_ = X[indices].clone()
        
        # 다른 매개변수 첫값 잡기
        self.alpha_ = torch.ones(self.K) * self.alpha_0 + N / self.K
        self.beta_ = torch.ones(self.K) * self.beta_0 + N / self.K
        self.nu_ = torch.ones(self.K) * self.nu_0 + N / self.K
        self.W_ = torch.stack([torch.eye(d) for _ in range(self.K)])
        
    def _compute_responsibilities(self, X: torch.Tensor) -> torch.Tensor:
        """변분 E 걸음: 기댓값 맡음 몫 셈하기."""
        N = X.shape[0]
        
        # 기댓값 로그 섞음 무게: E[log π_k]
        log_pi = torch.digamma(self.alpha_) - torch.digamma(self.alpha_.sum())
        
        # 기댓값 로그 정밀도 행렬식: E[log |Λ_k|]
        log_det_Lambda = torch.zeros(self.K)
        for k in range(self.K):
            log_det_Lambda[k] = (
                self.d * torch.log(torch.tensor(2.0)) +
                torch.logdet(self.W_[k]) +
                sum(torch.digamma((self.nu_[k] + 1 - i) / 2) for i in range(1, self.d + 1))
            )
        
        # 기댓값 마할라노비스 거리: E[(x - μ_k)^T Λ_k (x - μ_k)]
        log_rho = torch.zeros(N, self.K)
        for k in range(self.K):
            diff = X - self.m_[k]
            E_Lambda = self.nu_[k] * self.W_[k]
            mahal = self.d / self.beta_[k] + self.nu_[k] * (diff @ self.W_[k] * diff).sum(dim=1)
            
            log_rho[:, k] = log_pi[k] + 0.5 * log_det_Lambda[k] - 0.5 * self.d * torch.log(torch.tensor(2 * torch.pi)) - 0.5 * mahal
        
        # 정규화
        log_rho_norm = torch.logsumexp(log_rho, dim=1, keepdim=True)
        return torch.exp(log_rho - log_rho_norm)
    
    def _update_parameters(self, X: torch.Tensor, r: torch.Tensor):
        """변분 M 걸음: 변분 매개변수 새로 고치기."""
        N = X.shape[0]
        
        # 충분 통계량
        N_k = r.sum(dim=0) + 1e-10
        x_bar = (r.T @ X) / N_k.unsqueeze(1)
        
        # 매개변수 갱신
        self.alpha_ = self.alpha_0 + N_k
        self.beta_ = self.beta_0 + N_k
        self.nu_ = self.nu_0 + N_k
        
        # 평균의 앞확률(0으로 놓음)
        m_0 = torch.zeros(self.d)
        self.m_ = (self.beta_0 * m_0 + N_k.unsqueeze(1) * x_bar) / self.beta_.unsqueeze(1)
        
        # 위샤트 규모 행렬
        W_0_inv = torch.eye(self.d)
        for k in range(self.K):
            diff = X - x_bar[k]
            S_k = (r[:, k].unsqueeze(1) * diff).T @ diff
            
            diff_m = x_bar[k] - m_0
            W_k_inv = (
                W_0_inv + S_k + 
                (self.beta_0 * N_k[k]) / (self.beta_0 + N_k[k]) * 
                torch.outer(diff_m, diff_m)
            )
            self.W_[k] = torch.inverse(W_k_inv + self.reg_covar * torch.eye(self.d))
    
    def _compute_elbo(self, X: torch.Tensor, r: torch.Tensor) -> float:
        """증거 아래 경계 셈하기."""
        N = X.shape[0]
        N_k = r.sum(dim=0) + 1e-10
        
        # 이는 간추린 ELBO 셈하기이다
        # 온전한 판에는 모든 변분 분포의 KL 벌어짐이 들어간다
        
        # 기댓값 로그 가능도
        E_log_lik = 0.0
        for k in range(self.K):
            diff = X - self.m_[k]
            E_Lambda = self.nu_[k] * self.W_[k]
            mahal = self.d / self.beta_[k] + self.nu_[k] * (diff @ self.W_[k] * diff).sum(dim=1)
            
            log_det = self.d * torch.log(torch.tensor(2.0)) + torch.logdet(self.W_[k])
            log_det += sum(torch.digamma((self.nu_[k] + 1 - i) / 2) for i in range(1, self.d + 1))
            
            E_log_lik += (r[:, k] * (0.5 * log_det - 0.5 * self.d * torch.log(torch.tensor(2 * torch.pi)) - 0.5 * mahal)).sum()
        
        # 맡음 몫의 엔트로피
        H_r = -(r * torch.log(r + 1e-10)).sum()
        
        return (E_log_lik + H_r).item()
    
    def fit(self, X: torch.Tensor, max_iter: int = 100, tol: float = 1e-4,
            verbose: bool = False) -> 'VariationalGMM':
        """변분 EM으로 변분 가우스 섞음 모형 맞추기."""
        self._initialize(X)
        
        prev_elbo = float('-inf')
        
        for iteration in range(max_iter):
            # 변분 E 걸음
            r = self._compute_responsibilities(X)
            
            # M 걸음
            self._update_parameters(X, r)
            
            # ELBO 셈하기
            elbo = self._compute_elbo(X, r)
            
            if verbose and iteration % 10 == 0:
                print(f"Iteration {iteration}: ELBO = {elbo:.4f}")
            
            if abs(elbo - prev_elbo) < tol:
                if verbose:
                    print(f"Converged at iteration {iteration}")
                break
            
            prev_elbo = elbo
        
        return self
    
    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """무리 배정 미리보기."""
        r = self._compute_responsibilities(X)
        return r.argmax(dim=1)
    
    def predict_proba(self, X: torch.Tensor) -> torch.Tensor:
        """무리 확률 미리보기."""
        return self._compute_responsibilities(X)
```

---

## 요약

| 살필 점 | 핵심 |
|--------|------------|
| **GEM** | M 걸음을 나아지기만 하면 되게 느슨히 한다. 기울기로 새로 고칠 수 있게 한다 |
| **ECM/ECME** | 차례차례 조건부 최대화. 복잡한 M 걸음을 풀어 놓는다 |
| **PX-EM** | 더 빨리 모이도록 매개변수 공간을 넓힌다 |
| **MCEM** | 다룰 수 없는 E 걸음을 몬테카를로로 어림한다 |
| **SEM** | 확률 표본 하나. 봉우리가 여럿인 면을 살펴보는 데 도움이 된다 |
| **변분 EM** | 복잡한 모형에서 뒤확률을 어림한다 |
| **흐름 속 EM** | 크거나 흐르는 자료를 위한 확률 새로 고침 |
| **가속** | 에이킨, SQUAREM, 유사 뉴턴 방법이 모임을 빠르게 한다 |

추론(E 걸음)과 최적화(M 걸음)를 갈라 놓는 EM 얼개의 유연함이 이 여러 넓힘을 가능하게 한다. 갈래마다 언제 어떻게 쓸지 아는 것은 실제 세계의 숨은 변수 모형을 다루는 데 꼭 필요하다.

### 주요 참고 문헌

- Dempster, A. P., Laird, N. M., & Rubin, D. B. (1977). Maximum likelihood from incomplete data via the EM algorithm. *JRSS-B*, 39(1), 1-38.
- Meng, X. L., & Rubin, D. B. (1993). Maximum likelihood estimation via the ECM algorithm. *Biometrika*, 80(2), 267-278.
- Liu, C., & Rubin, D. B. (1994). The ECME algorithm: A simple extension of EM and ECM with faster monotone convergence. *Biometrika*, 81(4), 633-648.
- Wei, G. C., & Tanner, M. A. (1990). A Monte Carlo implementation of the EM algorithm. *JASA*, 85(411), 699-704.
- Celeux, G., & Diebolt, J. (1985). The SEM algorithm: A probabilistic teacher algorithm derived from the EM algorithm. *Computational Statistics Quarterly*, 2, 73-82.
- Cappé, O., & Moulines, E. (2009). On-line expectation–maximization algorithm for latent data models. *JRSS-B*, 71(3), 593-613.

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
