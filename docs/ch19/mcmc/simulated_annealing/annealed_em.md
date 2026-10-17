# 결정적 어닐링 EM

결정적 어닐링 EM(DAEM)은 잠재변수 배정의 엔트로피를 조절하며 우도 최적화를 이어 가는 방법이다. 확률적 SA와 달리 E·M 단계의 결정적 갱신을 사용하지만 전역 최적해를 자동으로 보장하지는 않는다.

## 1. EM의 초기화와 비볼록성

EM은 현재 θ의 잠재 사후 q(Z)=p(Z|X,θ)와 고정 q의 E_qlog p(X,Z|θ) 최대화를 번갈아 수행한다. 정확한 단계는 관측 로그우도를 감소시키지 않지만 일반적으로 전역 최댓값을 보장하지 않는다. 정칙성 없이 모든 실행이 국소 최댓값으로 수렴한다고도 단정할 수 없다.

혼합모형에는 라벨 대칭·여러 배정·경계해가 있다. 제약 없는 가우스 혼합은 성분 공분산 붕괴로 우도가 무한히 커질 수 있으므로 좋은 초기화나 어닐링만으로 해결되지 않는다. 적절한 제약·사전·정규화와 수치 점검이 필요하다.

## 2. 역온도와 잠재 배정

역온도 β>0에서 q_β(Z|X,θ)∝p(X,Z|θ)^β를 정의한다. 고정 X의 잠재변수 정규화가 유한해야 한다. β=1이면 표준 E 단계이고 0<β<1은 배정을 부드럽게 할 수 있다.

잠재 상태가 유한하고 모든 해당 점수가 양수라면 β→0에서 균등 배정이 된다. 연속 비유계 잠재공간에 균등분포가 존재하거나 β=0의 적분이 유한하다고 가정하지 않는다. β→∞는 최대 점수 배정에 집중하지만 동률이면 균등하게 나눌 수 있다.

높은 온도가 항상 목적함수를 볼록하게 하거나 유일 최적점을 만든다는 보장은 없다. 점진적 β 변화는 유리한 경로를 찾으려는 전략이며 모든 국소 함정을 제거하는 증명은 아니다.

## 3. 수학적 틀

정규화 가능한 잠재 목표에서 다음 변분 표현을 쓴다.

$$\ell_\beta(\theta)=\frac1\beta\log\int p(X,Z|\theta)^\beta dZ=\sup_q\left\{E_q\log p(X,Z|\theta)+\frac1\beta H(q)\right\}.$$

엔트로피 부호는 양수다. 음의 자유에너지 목적을 최대화하므로 높은 온도 T=1/β가 넓은 q를 장려한다. 물리적 최소화 형태는 E_q[−log p]−T H(q)다. 원래 음의 엔트로피 부호는 같은 tempered E 단계를 유도하지 않는다.

β=1에서 원래 로그우도를 되찾는다. 유한 잠재 상태가 L개이고 점수가 양수이면 β→0 전개는 log L/β+E_uniformlog p+O(β)다. 따라서 발산하는 θ 공통 항을 빼지 않은 ell_β가 단순한 균등 평균으로 수렴한다고 쓰면 틀리다. 연속 잠재변수의 경우 기준 측도·정규화와 극한 조건이 더 필요하다.

## 4. E 단계의 책임도와 M 단계

고정 β에서 E 단계는 q_β, M 단계는 그 q에서 기대 완전자료 로그우도를 최대화한다. 정확한 교대 갱신은 해당 β의 변분 목적을 개선할 수 있지만 β를 바꾸면 목적 자체가 달라져 원래 관측우도의 단조 증가를 보장하지 않는다.

가우스 혼합에서는 a_nk=π_kN(x_n|μ_k,Σ_k)에 대해 r_nk(β)=a_nk^β/∑_j a_nj^β다. β→0의 유한 양의 점수는 1/K, β=1은 표준 책임도, β→∞는 최대 점수의 hard 배정이다. 동률·0인 성분 비중의 극한을 별도로 다룬다. 일반적인 공분산·혼합 비중을 둔 hard 배정은 곧바로 k-means와 같지 않다.

```
Input: Data X, number of components K, 
       temperature schedule β₁ < β₂ < ... < βₘ = 1

Initialize θ randomly

for β in [β₁, β₂, ..., βₘ]:
    repeat until convergence:
        # 담금질한 E 스텝
        q_β(Z) ∝ p(X, Z | θ)^β
        
        # M 스텝(그대로)
        θ = argmax_θ E_q[log p(X, Z | θ)]
    
return θ
```

## 5. 보기: 가우시안 혼합 모델

annealed_e_step은 로그 성분 점수 전체에 β를 곱하고 log-sum-exp로 책임도를 정규화한다. m_step은 N_k·가중 평균·공분산을 계산한다. 비어 있거나 매우 작은 N_k, 특이 공분산, 0인 혼합 비중을 처리하지 않아 실패할 수 있다.

annealed_em_gmm에는 compute_log_likelihood 정의가 없으며 각 β에서 원래 로그우도 차이로 종료한다. β≠1의 올바른 고정 온도 목적과 다른 기준이므로 그 차이가 작다고 해당 tempered 목적의 수렴이 증명되지 않는다. 초기 성분 수 K≤N, 유효 자료·β·반복 수와 공분산 제약도 필요하다. 코드 조각은 교육용 구성이다.

```python
def annealed_e_step(X, pi, mu, Sigma, beta):
    """담금질한 맡음 몫 셈하기."""
    N, D = X.shape
    K = len(pi)
    
    # 로그가능도들을 계산한다
    log_resp = np.zeros((N, K))
    for k in range(K):
        log_resp[:, k] = (np.log(pi[k]) + 
                         multivariate_normal.logpdf(X, mu[k], Sigma[k]))
    
    # 온도를 적용한다
    log_resp_beta = beta * log_resp
    
    # 고르게 하기(소프트맥스)
    log_resp_beta -= logsumexp(log_resp_beta, axis=1, keepdims=True)
    resp = np.exp(log_resp_beta)
    
    return resp
```
```python
def m_step(X, resp):
    """맡음 몫을 쓰는 표준 M 스텝."""
    N, D = X.shape
    K = resp.shape[1]
    
    N_k = resp.sum(axis=0)
    pi = N_k / N
    mu = np.array([resp[:, k] @ X / N_k[k] for k in range(K)])
    
    Sigma = []
    for k in range(K):
        diff = X - mu[k]
        Sigma_k = (resp[:, k:k+1] * diff).T @ diff / N_k[k]
        Sigma.append(Sigma_k)
    
    return pi, mu, np.array(Sigma)
```
```python
def annealed_em_gmm(X, K, betas=[0.1, 0.2, 0.4, 0.6, 0.8, 1.0], 
                    max_iter=100, tol=1e-6):
    """가우스 섞음 모형을 위한 담금질한 EM."""
    N, D = X.shape
    
    # 무작위 첫값 잡기
    pi = np.ones(K) / K
    indices = np.random.choice(N, K, replace=False)
    mu = X[indices].copy()
    Sigma = np.array([np.eye(D) for _ in range(K)])
    
    for beta in betas:
        prev_ll = -np.inf
        for iteration in range(max_iter):
            # 담금질한 E 스텝
            resp = annealed_e_step(X, pi, mu, Sigma, beta)
            
            # M 스텝
            pi, mu, Sigma = m_step(X, resp)
            
            # 모임 살피기
            ll = compute_log_likelihood(X, pi, mu, Sigma)
            if abs(ll - prev_ll) < tol:
                break
            prev_ll = ll
    
    return pi, mu, Sigma
```

## 6. 이론적 분석

β 변화에 따라 고정점의 안정성·분기가 달라질 수 있어 연속화·호모토피 관점으로 이해할 수 있다. 그러나 낮은 β에서 항상 볼록·유일해이거나 연속 경로에 단절이 없으면 전역 최적이라는 결론은 나오지 않는다. 국소 정상점의 가지를 매끄럽게 따라가도 최종 전역점과 다를 수 있다.

완전히 대칭인 초기 성분은 결정적 갱신으로 계속 같은 상태를 유지할 수 있다. 성분 분리·섭동·분기 처리의 설계가 필요할 수 있다. 유한 혼합 최적화의 급격한 변화와 열역학적 극한의 엄밀한 상전이도 구분한다.

## 7. 온도 일정 짜기

0<β₀<1에서 시작해 최종 β=1까지 선형·등비·적응 일정을 사용할 수 있다. 등비 일정의 비율 r=(1/β₀)^(1/(M−1))는 M>1에서 정의된다. β=0을 목적 함수의 1/β 식에 그대로 넣지 않는다.

아래 adaptive_annealed_em은 random_init·em_at_temperature가 외부 정의인 조각이고 max_beta_steps 인수를 사용하지 않는다. β를 1로 올린 직후 while 조건이 끝나므로 최종 β=1의 EM을 실행하지 않은 채 반환할 수 있다. 적응 속도가 원래 우도 최적화까지 완료한다고 읽지 않는다.

```python
def adaptive_annealed_em(X, K, beta_init=0.01, max_beta_steps=50):
    """맞춰 가는 온도 일정을 쓰는 담금질한 EM."""
    theta = random_init(X, K)
    beta = beta_init
    beta_increment = 0.05
    
    while beta < 1.0:
        theta, converged_iters = em_at_temperature(X, theta, beta)
        
        # 온도 올림폭 맞추기
        if converged_iters < 5:
            beta_increment *= 1.5  # 빨리 모였으므로 더 빠르게 올림
        elif converged_iters > 20:
            beta_increment *= 0.7  # 느리게 모이므로 더 천천히 올림
        
        beta = min(beta + beta_increment, 1.0)
    
    return theta
```

## 8. 넓힘과 변형

HMM에서는 정한 tempered 결합 목표의 전이·방출 요인으로 forward–backward를 구성해야 한다. 이미 주변화한 forward·backward 결과를 각각 β승하는 것과 결합 요인을 먼저 β승한 뒤 합을 수행하는 것은 일반적으로 다르다.

평균장 목적 E_qlog p+H(q)/β의 좌표 갱신은 log q_i=βE_{q_−i}log p+상수다. 가정한 분해·정규화 조건을 맞춘다.

VAE의 재구성항−λ KL 목적에서 λ는 KL 계수다. λ를 이 DAEM의 역온도 β와 무조건 같은 기호·효과로 대응시키지 않는다. KL annealing은 보통 목적의 다른 가중을 바꾸며 표현 분리·다양성의 효과는 모형과 학습에 따라 달라져 보장되지 않는다.

## 9. 실용적인 고려

initialize_for_annealing은 동일 비중·공분산과 무작위 중심을 사용한다. 높은 온도가 모든 초기값에 둔감하다는 보장은 없고 데이터 공분산이 특이할 수 있다.

stable_annealed_responsibilities는 확률을 작은 하한으로 clip하므로 정확한 tempered 책임도와 달라질 수 있다. 전부 −∞ 또는 NaN인 로그행과 유효하지 않은 β는 최대값 이동만으로 해결되지 않는다. β마다 고정 목적과 원래 우도를 따로 기록하고, 목적값을 서로 비교할 때 온도별 공통 항도 확인한다.

```python
def initialize_for_annealing(X, K):
    """온도가 높으면 첫값이 별로 중요하지 않다."""
    N, D = X.shape
    pi = np.ones(K) / K
    mu = X[np.random.choice(N, K, replace=False)]
    Sigma = np.array([np.cov(X.T) for _ in range(K)])
    return pi, mu, Sigma
```
```python
def stable_annealed_responsibilities(log_prob, beta, eps=1e-10):
    """수치로 안정된 담금질한 맡음 몫."""
    log_prob_scaled = beta * log_prob
    log_prob_scaled -= log_prob_scaled.max(axis=1, keepdims=True)
    
    prob = np.exp(log_prob_scaled)
    prob = np.clip(prob, eps, 1-eps)
    prob /= prob.sum(axis=1, keepdims=True)
    
    return prob
```
```python
def annealed_log_likelihood(X, pi, mu, Sigma, beta):
    """거꿀 온도 beta에서의 로그 가능도 셈하기."""
    N, K = len(X), len(pi)
    
    log_probs = np.zeros((N, K))
    for k in range(K):
        log_probs[:, k] = (np.log(pi[k]) + 
                          multivariate_normal.logpdf(X, mu[k], Sigma[k]))
    
    return (1/beta) * np.sum(logsumexp(beta * log_probs, axis=1))
```

## 10. 다른 길과 비교

DAEM은 온도별 최적화를 이어 가고 무작위 재시작은 여러 초기해를 비교한다. 이들을 결합할 수도 있다. 어느 쪽이 유리한지는 모형·공분산 제약·책임도 계산 비용과 예산에 달린다.

확률적 SA와 DAEM 모두 전역 탐색을 돕는 전략일 수 있지만 각각의 수렴 정리를 공유하지 않는다. 최종 β=1에서 같은 목적과 제약으로 해 품질·안정성·총비용을 비교한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
DAEM 변분 목적의 엔트로피 부호는 무엇인가?

</div>

??? success "풀이"
    최대화 목적은 E_qlog p+H(q)/β다. 물리적 최소 자유에너지 E_q[−log p]−H(q)/β의 음수와 대응하므로 엔트로피 항이 양수여야 한다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
유한 L개 잠재 상태에서 β→0의 ell_β는 단순 균등 평균인가?

</div>

??? success "풀이"
    log L/β라는 발산하는 공통 항이 있다. 이를 제거한 극한이 유한 양의 점수의 균등 log 점수 평균이다. 연속 비유계 잠재공간에는 같은 균등 극한을 가정할 수 없다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
각 β에서 원래 로그우도가 증가하는가?

</div>

??? success "풀이"
    정확한 갱신은 고정 β의 해당 목적을 개선한다. β≠1에서는 원래 관측 로그우도와 다른 목적이고 β 변경 전후에도 단조성을 자동으로 보장하지 않는다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff easy" title="쉬움"></span>
adaptive_annealed_em은 β=1에서 최적화를 끝내는가?

</div>

??? success "풀이"
    β를 1로 갱신하면 while beta<1 조건이 끝나 최종 온도 실행을 건너뛸 수 있다. max_beta_steps도 사용하지 않아 별도의 종료·최종 적합 설계가 필요하다.

## 정리하며

DAEM은 양의 엔트로피 항을 갖는 고정 역온도 목적을 교대로 개선한다. β 극한의 정규화, 원래 우도와의 차이, 대칭·공분산 붕괴와 최종 β=1 실행 여부를 확인한다.
