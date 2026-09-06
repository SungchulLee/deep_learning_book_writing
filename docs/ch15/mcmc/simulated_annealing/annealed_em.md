# EM을 위한 정해진 담금질
정해진 담금질은 온도라는 개념을 기댓값-최대화 알고리즘에 씌워, 가능도 최적화에서 그 자리 최적점을 벗어나는 원리 있는 길을 준다. 확률로 흔들리는 흉내낸 담금질과 달리 이 길은 기댓값을 곧바로 다루므로, 기댓값을 손으로 셈할 수 있는 문제에 알맞다.

---

## 왜 필요한가: EM의 그 자리 최적점 문제

### EM과 그 자리 최댓값

표준 EM 알고리즘은 로그 가능도를 가장 크게 한다:

$$
\ell(\theta) = \log p(\mathbf{X} | \theta) = \log \int p(\mathbf{X}, \mathbf{Z} | \theta) \, d\mathbf{Z}
$$

EM은 다음을 오가며 되풀이한다:

- **E 걸음**: $q(\mathbf{Z}) = p(\mathbf{Z} | \mathbf{X}, \theta^{(t)})$을 셈한다
- **M 걸음**: $\mathbb{E}_q[\log p(\mathbf{X}, \mathbf{Z} | \theta)]$을 가장 크게 한다

**말썽**: EM은 *그 자리* 최댓값으로 모임만 보장한다. 마지막 풀이는 첫값에 달려 있다.

### 그 자리 최적점이 왜 생기나

섞음 모형에서 그 자리 최적점은 띄엄띄엄한 배정 짜임(자료 점마다 성분 하나에 "속함"), 대칭 깨짐(같은 값의 풀이가 여럿 있음), 봉우리 여럿(가능도 면에 봉우리가 많음)에서 생긴다.

---

## 담금질 생각

### 온도로 크기를 잡은 분포

거꿀 온도 $\beta \in [0, 1]$을 들여와 **온도를 다룬 뒤확률**을 다음과 같이 정한다:

$$
q_\beta(\mathbf{Z} | \mathbf{X}, \theta) \propto p(\mathbf{X}, \mathbf{Z} | \theta)^\beta
$$

온도마다 다음과 같다:

| $\beta$ | 온도 | 뒤확률의 굶 |
|---------|-------------|-------------------|
| $\beta \to 0$ | $T \to \infty$ | 숨은 상태에 걸쳐 고름 |
| $\beta = 1$ | $T = 1$ | 참 뒤확률(표준 EM) |
| $\beta \to \infty$ | $T \to 0$ | MAP 배정에 몰림 |

### 온도가 지형을 어떻게 매끄럽게 하나

온도가 높으면($\beta \to 0$) 숨은 꼴이 모두 똑같이 그럴듯해지고 실효 가능도가 매끄러워지며(흔히 볼록해지며) 대개 최적점이 하나뿐이다. 온도가 낮으면($\beta \to 1$) 봉우리가 여럿인 참 짜임이 드러나고 그 자리 최적점이 나타나지만, 이때 알고리즘은 이미 좋은 웅덩이에 들어 있다.

$\beta = 0$(쉬운 문제)에서 시작해 $\beta = 1$(참 문제)까지 조금씩 올리면, 그 자리 최적점을 피해 가는 길이 매개변수 공간에 그려진다.

---

## 수학적 틀

### 담금질한 자유 에너지

거꿀 온도 $\beta$에서 **담금질한 로그 가능도**를 다음과 같이 정한다:

$$
\ell_\beta(\theta) = \frac{1}{\beta} \log \int p(\mathbf{X}, \mathbf{Z} | \theta)^\beta \, d\mathbf{Z}
$$

**성질**:

- $\lim_{\beta \to 1} \ell_\beta(\theta) = \ell(\theta)$(참 로그 가능도를 되찾는다)
- $\lim_{\beta \to 0} \ell_\beta(\theta) = \mathbb{E}_{\text{uniform}}[\log p(\mathbf{X}, \mathbf{Z} | \theta)]$(숨은 공간에 걸친 평균)
- $\ell_\beta(\theta)$은 $\beta$에 대해 이어지게 달라진다

### 담금질한 ELBO

온도 $\beta$에서의 변분 아래 경계는 다음과 같다:

$$
\mathcal{L}_\beta(q, \theta) = \mathbb{E}_q[\log p(\mathbf{X}, \mathbf{Z} | \theta)] - \frac{1}{\beta} H[q]
$$

여기서 $H[q] = -\mathbb{E}_q[\log q(\mathbf{Z})]$은 엔트로피이다.

**풀이**: 온도가 엔트로피 항의 크기를 잡는다. $T$이 높으면($\beta$이 낮으면) 엔트로피 벌주기가 세어 $q$이 넓게 퍼지고, $T$이 낮으면($\beta$이 높으면) 벌주기가 약해 $q$이 몰린다.

### 자유 에너지와의 이음

통계 물리에서는:

$$
F_\beta = U - T S = \mathbb{E}_q[E] - \frac{1}{\beta} H[q]
$$

여기서 $U$은 안 에너지이고 $S$은 엔트로피이다. 담금질한 EM은 온도마다 자유 에너지를 가장 작게 한다.

---

## 담금질한 EM 알고리즘

### 알고리즘의 짜임

```
Input: Data X, number of components K, 
       temperature schedule β₁ < β₂ < ... < βₘ = 1

Initialize θ randomly

for β in [β₁, β₂, ..., βₘ]:
    repeat until convergence:
        # 담금질한 E 걸음
        q_β(Z) ∝ p(X, Z | θ)^β
        
        # M 걸음(그대로)
        θ = argmax_θ E_q[log p(X, Z | θ)]
    
return θ
```

### 담금질한 E 걸음

핵심적인 바꿈은 E 걸음에 있다. 표준 뒤확률은 다음과 같다:

$$
p(z_n = k | \mathbf{x}_n, \theta) = \frac{\pi_k p(\mathbf{x}_n | \theta_k)}{\sum_j \pi_j p(\mathbf{x}_n | \theta_j)}
$$

**담금질한 뒤확률**(맡음 몫)은 다음과 같다:

$$
r_{nk}^\beta = \frac{[\pi_k p(\mathbf{x}_n | \theta_k)]^\beta}{\sum_j [\pi_j p(\mathbf{x}_n | \theta_j)]^\beta}
$$

### 온도가 맡음 몫에 주는 영향

**$\beta \to 0$일 때**(높은 온도): $r_{nk}^\beta \to 1/K$이 되어 성분이 모두 몫을 똑같이 나눈다.

**$\beta = 1$일 때**(보통 온도): $r_{nk}^1 = r_{nk}$으로 표준 EM의 맡음 몫이 된다.

**$\beta \to \infty$일 때**(온도 0):

$$
r_{nk}^\beta \to \begin{cases} 1 & k = \arg\max_j \pi_j p(\mathbf{x}_n | \theta_j) \\ 0 & \text{otherwise} \end{cases}
$$

딱 잘라 배정한다(k 평균과 비슷하다).

---

## 보기: 가우스 섞음 모형

### GMM의 담금질한 E 걸음

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

### M 걸음(그대로)

M 걸음은 담금질한 맡음 몫을 쓸 뿐 나머지는 표준과 같다:

```python
def m_step(X, resp):
    """맡음 몫을 쓰는 표준 M 걸음."""
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

### GMM의 온전한 담금질한 EM

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
            # 담금질한 E 걸음
            resp = annealed_e_step(X, pi, mu, Sigma, beta)
            
            # M 걸음
            pi, mu, Sigma = m_step(X, resp)
            
            # 모임 살피기
            ll = compute_log_likelihood(X, pi, mu, Sigma)
            if abs(ll - prev_ll) < tol:
                break
            prev_ll = ll
    
    return pi, mu, Sigma
```

---

## 이론적 분석

### 갈래 나눔 이론

$\beta$이 커지면 매개변수 공간은 **갈래 나눔**을 겪는다. 곧 임계점의 개수나 성질이 바뀌는 지점을 지난다.

$\beta$이 낮으면 대개 최적점이 하나뿐이다. $\beta$의 임계값에서 그 최적점이 여럿으로 갈라진다(쇠스랑 갈래 나눔). 알고리즘은 자료가 정하는 한 가지를 따라간다.

### 호모토피 이어 가기

담금질한 EM은 **호모토피 이어 가기**로 볼 수 있다. 곧 쉬운 문제($\beta \to 0$, 볼록한 지형)에서 어려운 문제($\beta = 1$, 봉우리 여럿인 지형)로 이어지게 주물러 가는 것이다.

길 $\beta \mapsto \theta^*(\beta)$은 매개변수 공간에 이어진 곡선을 그린다. 그 길에 끊긴 곳이 없으면 전체 최적점에 이른다.

### 상 바뀜

임계 온도에서 풀이의 성질이 질적으로 바뀐다:

- **무리 나타남**: $\beta$이 커지면 고르던 맡음 몫이 또렷한 무리로 갈라진다
- **대칭 깨짐**: 같던 성분이 서로 가려진다
- **일차 바뀜**: 어떤 $\beta$ 값에서 매개변수가 갑자기 뛴다

---

## 온도 일정 짜기

### 선형 일정

$$
\beta_m = \frac{m}{M}, \quad m = 1, 2, \ldots, M
$$

단순하지만 중요한 상 바뀜을 놓칠 수 있다.

### 등비 일정

$$
\beta_m = \beta_0 \cdot r^{m-1}, \quad r = (1/\beta_0)^{1/(M-1)}
$$

바뀜이 흔히 일어나는 낮은 $\beta$에서 더 잘게 나눈다.

### 맞춰 가는 일정

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

---

## 넓힘과 변형

### 다른 모형에서의 정해진 담금질

**숨은 마르코프 모형**: 앞뒤 맡음 몫을 담금질하며, 크기를 맞춘 꼴로 $\alpha_t(k)^\beta$과 $\beta_t(k)^\beta$을 쓴다.

**숨은 디리클레 배분**: 주제 배정을 담금질하며 $q(z_{dn} = k)^\beta$을 고르게 한다.

**변분 자동부호기**: KL 벌어짐 항을 담금질한다(β-VAE와의 이음):

$$
\mathcal{L} = \mathbb{E}[\log p(\mathbf{x}|\mathbf{z})] - \beta \cdot D_{KL}[q(\mathbf{z}|\mathbf{x}) \| p(\mathbf{z})]
$$

### β-VAE와의 이음

β-VAE의 목표는 다음과 같다:

$$
\mathcal{L}_\beta = \mathbb{E}_{q(\mathbf{z}|\mathbf{x})}[\log p(\mathbf{x}|\mathbf{z})] - \beta \cdot D_{KL}[q(\mathbf{z}|\mathbf{x}) \| p(\mathbf{z})]
$$

이는 담금질한 변분 추론으로 볼 수 있다. $\beta < 1$이면 숨은 나타냄이 더 다양해지고, $\beta = 1$이면 표준 VAE이며, $\beta > 1$이면 서로 풀림을 북돋운다.

### 평균장 담금질

$q(\mathbf{Z}) = \prod_i q_i(z_i)$으로 인수 나눈 평균장 변분 추론에서:

**표준 새로 고치기**: $\log q_i(z_i) = \mathbb{E}_{q_{-i}}[\log p(\mathbf{X}, \mathbf{Z})] + \text{const}$

**담금질한 새로 고치기**: $\log q_i(z_i) = \beta \cdot \mathbb{E}_{q_{-i}}[\log p(\mathbf{X}, \mathbf{Z})] + \text{const}$

---

## 실용적인 고려

### 높은 온도에서 첫값 잡기

$\beta \to 0$에서는 풀이가 첫값에 무디다. 웬만한 시작점이면 다 된다:

```python
def initialize_for_annealing(X, K):
    """온도가 높으면 첫값이 별로 중요하지 않다."""
    N, D = X.shape
    pi = np.ones(K) / K
    mu = X[np.random.choice(N, K, replace=False)]
    Sigma = np.array([np.cov(X.T) for _ in range(K)])
    return pi, mu, Sigma
```

### 수치적 안정성

온도가 지나치면 수치 말썽이 생길 수 있다:

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

### 모임 지켜보기

온도마다 담금질한 로그 가능도를 기록한다:

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

---

## 다른 길과 견주기

### 담금질한 EM과 무작위 다시 시작의 견줌

| 살필 점 | 담금질한 EM | 무작위 다시 시작 |
|--------|-------------|-----------------|
| 실행 | 한 번 | 여러 번 |
| 살펴보기 | 온도로 차근차근 | 첫값으로 무작위 |
| 셈 | 긴 실행 하나 | 짧은 실행 여럿 |
| 이론 | 호모토피 이어 가기 | 독립 시도 가운데 최고 |
| 어디에 좋은가 | 매끄러운 지형 | 울퉁불퉁한 지형 |

### 담금질한 EM과 흉내낸 담금질의 견줌

| 살필 점 | 담금질한 EM | 흉내낸 담금질 |
|--------|-------------|---------------------|
| 성질 | 정해짐 | 확률로 흔들림 |
| 새로 고치기 | 기댓값 | 표본에 바탕 |
| 되풀이당 값 | 큼(온전한 E 걸음) | 작음(움직임 하나) |
| 모임 | 매끄러운 길 | 시끄러운 길 |
| 쓸 수 있는 곳 | 기댓값을 다룰 수 있을 때 | 두루 |

### 담금질한 EM을 언제 쓰나

**담금질한 EM을 쓸 때**: 기댓값을 손으로 셈할 수 있고, 지형에 깊은 웅덩이가 몇 개 있으며, 표준 EM이 첫값에 예민하고, 정해져서 되풀이할 수 있는 결과를 바랄 때.

**다른 길을 생각해 볼 때**: 기댓값을 다룰 수 없거나(SA이나 변분 추론을 써라), 지형이 몹시 울퉁불퉁하거나, 상 바뀜이 날카로울 때.

---

## 요약

| 개념 | 설명 |
|---------|-------------|
| **담금질한 E 걸음** | $q_\beta(\mathbf{Z}) \propto p(\mathbf{X}, \mathbf{Z} | \theta)^\beta$ |
| **온도의 효과** | $\beta \to 0$: 고름, $\beta = 1$: 참 뒤확률 |
| **지형 매끄럽게 하기** | 높은 온도가 그 자리 최적점을 없앤다 |
| **호모토피 길** | 쉬운 문제에서 어려운 문제로 이어지는 길 |
| **일정 짜기** | 낮게 시작해 $\beta = 1$까지 조금씩 올린다 |

정해진 담금질은 매끄럽고 볼록에 가까운 문제로 시작해 봉우리가 여럿인 참 짜임을 차츰 드러내고 이어진 길을 따라 좋은 풀이에 이르게 함으로써, EM을 그 자리 최적화기에서 더 전체를 보는 최적화기로 바꾼다.

## 연습문제

**연습문제 1.**
마르코프 사슬이 올바른 과녁 분포로 모이게 하는 데 받아들임 확률이 하는 몫을 설명하여라.

??? success "연습문제 1 풀이"
    받아들임 확률이 **자세한 균형** $\pi(x) T(x \to x') \alpha(x \to x') = \pi(x') T(x' \to x) \alpha(x' \to x)$을 보장한다. 여기서 $\pi$은 과녁 분포, $T$은 제안 분포, $\alpha$은 받아들임 확률이다. 자세한 균형은 $\pi$이 사슬의 멈춘 분포임을 뜻한다. 쪼갤 수 없음과 주기 없음까지 합치면 $\pi$으로의 에르고드 모임이 보장된다.

---

**연습문제 2.**
제안 분포가 너무 좁은 상황과 너무 넓은 상황을 밝혀라. 저마다 표집 효율에 어떤 영향을 주는가?

??? success "연습문제 2 풀이"
    **너무 좁을 때:** 제안이 거의 늘 받아들여지지만(받아들임 비율이 높지만) 사슬이 아주 작은 걸음을 떼어 과녁 분포를 느리게 살펴본다. 그러면 자기상관이 높고 실효 표본 크기가 작아진다. **너무 넓을 때:** 제안이 확률이 낮은 구역에 자주 떨어져 물리쳐지므로(받아들임 비율이 낮으므로) 사슬이 여러 되풀이 동안 지금 상태에 갇혀 있게 된다. 두 극단 모두 효율을 떨어뜨린다. 높은 차원에서 무작위 걸음 메트로폴리스의 가장 좋은 받아들임 비율은 대략 0.234이다(Roberts 외, 1997).

---

**연습문제 3.**
메트로폴리스-헤이스팅스 받아들임 비 $\alpha = \min\left(1, \frac{\pi(x') q(x|x')}{\pi(x) q(x'|x)}\right)$이 $\pi$에 대해 자세한 균형을 만족함을 증명하여라.

??? success "연습문제 3 풀이"
    일반성을 잃지 않고 $\pi(x') q(x|x') \leq \pi(x) q(x'|x)$이라 하자. 그러면 $\alpha(x \to x') = \frac{\pi(x') q(x|x')}{\pi(x) q(x'|x)}$이고 $\alpha(x' \to x) = 1$이다. 자세한 균형 조건은 다음을 요구한다:

    $$\pi(x) q(x'|x) \alpha(x \to x') = \pi(x) q(x'|x) \cdot \frac{\pi(x') q(x|x')}{\pi(x) q(x'|x)} = \pi(x') q(x|x')$$

    그리고 $\pi(x') q(x|x') \alpha(x' \to x) = \pi(x') q(x|x') \cdot 1 = \pi(x') q(x|x')$이다. 양변이 같다. $\square$

---

**연습문제 4.**
MCMC에서 태우기 기간이란 무엇이며, 처음 표본을 언제 버릴지 어떻게 정하는가?

??? success "연습문제 4 풀이"
    태우기 기간은 마르코프 사슬에서 아직 멈춘 분포로 모이지 않은 처음 부분이다. 치우침을 줄이려고 이 기간의 표본을 버린다. 태우기를 정하는 길은 다음과 같다. (1) 자취 그림으로 사슬이 언제 안정되는지 눈으로 살핀다. (2) 여러 사슬에서 사슬 안 흩어짐과 사슬 사이 흩어짐을 견주는 겔먼-루빈 진단($\hat{R}$)을 쓰며 $\hat{R} < 1.01$이면 모였다고 본다. (3) 실효 표본 크기(ESS) 어림값을 쓴다. (4) 흩어진 시작점에서 여러 사슬을 돌려 서로 맞는지 살핀다.
