# 온도 일정
식힘 일정은 흉내낸 담금질의 심장이다. 이것이 살펴보기와 써먹기의 주고받음을 정하고, 나아가 알고리즘이 전체 최적점을 찾느냐 마느냐를 정한다. 이 절에서는 온도 일정을 짜는 이론과 실전을 다룬다.

---

## 식힘 일정이 하는 일

### 일정이 다스리는 것

온도 일정 $T(t)$은 다음을 정한다:

1. **받아들임 확률**: $\alpha = \min(1, e^{-\Delta E / T(t)})$
2. **살펴보기와 써먹기**: $T$이 높으면 살펴보고 낮으면 써먹는다
3. **모이는 빠르기**: 최적점에 얼마나 빨리 다가가나
4. **풀이의 질**: 전체 최적점을 찾느냐 그 자리 최적점을 찾느냐

### 근본적인 주고받음

| 너무 빨리 식힘 | 너무 느리게 식힘 |
|---------------|---------------|
| 그 자리 최솟값에 갇힘 | 셈을 버림 |
| 전체 웅덩이를 놓침 | 얻는 것이 줄어듦 |
| 풀이의 질이 나쁨 | 질은 같은데 시간만 더 듦 |
| 욕심쟁이 최적화 같음 | 무작위 찾기 같음 |

가장 좋은 일정은 이 두 끝을 저울질한다.

---

## 이론에서 나온 일정

### 로그 식힘

전체 최적점으로 모임이 보장되는 **유일한** 일정이다:

$$
T(t) = \frac{c}{\log(t + t_0)}
$$

여기서 $c \geq d^*$(임계 깊이)이고 $t_0 \geq 2$이다.

**성질**:

- 확률 1로 전체 최적점으로 모인다
- 쓸 수 없을 만큼 느리다. 곧 $T = \epsilon$에 이르려면 $t \approx e^{c/\epsilon}$이 든다
- 이론의 잣대일 뿐 실전에서는 거의 쓰지 않는다

**이끌어 내는 직관**: 사슬이 높이 $d^*$인 에너지 벽을 넘으려면 시간 $\tau(T) \sim e^{d^*/T}$이 든다. 합 $\sum_t e^{-d^*/T(t)}$이 갈라져 흩어져야(그래야 결국 모든 벽을 넘는다) 하므로 $T(t) \sim c/\log(t)$이 필요하다.

### 거꿀 선형 식힘

보장은 약하지만 더 빠른 이론 일정이다:

$$
T(t) = \frac{T_0}{1 + \alpha t}
$$

**성질**:

- 로그 식힘보다 빠르다
- 전체 최적점으로 모임이 보장되지 않는다
- 어떤 문제에서는 다항 시간 어림 보장을 준다

### 거꿀 로그-선형

섞은 일정이다:

$$
T(t) = \frac{T_0}{\log(1 + \alpha t)}
$$

앞에서는 거꿀 선형처럼, 뒤에서는 로그처럼 구는 사이를 메운다.

---

## 실전에서 쓰는 일정

### 지수(등비) 식힘

실전에서 가장 널리 쓰는 일정이다:

$$
T(t) = T_0 \cdot \alpha^t, \quad \alpha \in (0, 1)
$$

같은 말로 $T_{k+1} = \alpha \cdot T_k$이다.

**보통 값**: $\alpha \in [0.85, 0.99]$

**성질**:

- 구현이 단순하다
- 잇따른 온도 사이의 비가 일정하다
- 낮은 온도에 빨리 이른다
- 모임 보장은 없지만 흔히 잘 듣는다

**온도 $T_f$에 이르는 시간**:

$$
t_f = \frac{\log(T_f / T_0)}{\log(\alpha)}
$$

### 선형 식힘

$$
T(t) = T_0 - \beta t = T_0 \left(1 - \frac{t}{t_{\max}}\right)
$$

여기서 $\beta = T_0 / t_{\max}$이다.

**성질**: $t = t_{\max}$에서 $T = 0$에 이르고 온도가 고른 빠르기로 낮아진다. $T$이 높을 때는 너무 느리게, 낮을 때는 너무 빠르게 식을 수 있다. 단순하지만 흔히 가장 좋지는 않다.

### 이차 식힘

$$
T(t) = T_0 \left(1 - \frac{t}{t_{\max}}\right)^2
$$

처음에는 느리게, 끝에는 빠르게 식는다. 중간 온도에 더 오래 머물러 상 바뀜에 도움이 될 수 있다.

### 거꿀 식힘

$$
T(t) = \frac{T_0}{1 + \beta t}
$$

쌍곡선꼴로 사그라들며 결코 0에 이르지 않는다(점근으로 다가갈 뿐이다). 낮은 온도에서는 지수 식힘보다 느리다.

---

## 일정 견주기

### 사그라드는 모양

$T_0 = 100$, $t_{\max} = 1000$일 때:

| 일정 | $T(100)$ | $T(500)$ | $T(900)$ |
|----------|----------|----------|----------|
| 로그($c = 100$) | 21.7 | 16.1 | 14.7 |
| 지수($\alpha = 0.995$) | 60.6 | 8.2 | 1.1 |
| 선형 | 90.0 | 50.0 | 10.0 |
| 이차 | 81.0 | 25.0 | 1.0 |

### 일정 고르기

| 문제의 성격 | 권하는 일정 |
|----------------------|---------------------|
| 얕은 그 자리 최솟값이 많음 | 지수(더 빠름) |
| 깊은 그 자리 최솟값이 적음 | 더 느린 지수 또는 로그 |
| 지형을 모름 | 맞춰 가는 일정 |
| 시간 예산이 빠듯함 | 과감한 지수 |
| 질 높은 풀이가 필요함 | 느린 일정에 다시 시작 붙이기 |

---

## 맞춰 가는 일정

### 받아들임 비율에 바탕을 둔 맞춰 가기

목표 받아들임 비율을 지키도록 온도를 다듬는다:

```python
def adaptive_cooling(E, x, T, target_accept=0.44, window=100):
    accepts = []
    
    for t in range(max_iter):
        x_new = propose(x)
        delta_E = E(x_new) - E(x)
        
        if np.random.rand() < min(1, np.exp(-delta_E / T)):
            x = x_new
            accepts.append(1)
        else:
            accepts.append(0)
        
        # 온도 맞추기
        if len(accepts) >= window:
            recent_rate = np.mean(accepts[-window:])
            if recent_rate > target_accept + 0.05:
                T *= 0.95  # 더 빠르게 식힘
            elif recent_rate < target_accept - 0.05:
                T *= 1.02  # 살짝 달굼
    
    return x
```

**목표 받아들임 비율**: 앞(살펴보기)에서는 80-90%, 가운데에서는 40-60%, 뒤(써먹기)에서는 10-30%.

### 램-들롬 일정

지켜본 받아들임에 따라 식힘 빠르기를 맞춰 간다:

$$
T_{k+1} = T_k \cdot \exp\left(-\frac{\lambda T_k}{\sigma_E}\right)
$$

여기서 $\sigma_E$은 에너지 변화의 표준편차이고 $\lambda$은 다스림 매개변수이다. 직관은 이렇다. 에너지 변화가 작으면(지형이 매끄러우면) 더 빨리 식히고, 크면(지형이 거칠면) 더 느리게 식힌다.

### 황-로메오-산조반니 일정

엔트로피 어림에 바탕을 둔다:

$$
T_{k+1} = T_k \cdot \exp\left(-\frac{T_k}{\sigma_E + \epsilon}\right)
$$

그 자리 에너지 지형에 따라 저절로 다듬어진다.

---

## 여러 단계 일정

### 조각별 일정

단계마다 식힘 빠르기를 다르게 한다:

```python
def piecewise_schedule(t, T0=100, t_max=1000):
    # 처음에 빠르게 식히기
    if t < 0.2 * t_max:
        return T0 * (0.98 ** t)
    # 가운데 단계는 느리게
    elif t < 0.8 * t_max:
        T_start = T0 * (0.98 ** (0.2 * t_max))
        return T_start * (0.995 ** (t - 0.2 * t_max))
    # 끝에 빠르게 식히기
    else:
        T_start = T0 * (0.98 ** (0.2 * t_max)) * (0.995 ** (0.6 * t_max))
        return T_start * (0.99 ** (t - 0.8 * t_max))
```

### 다시 달구기(단조롭지 않은 일정)

그 자리 최솟값에서 벗어나려고 이따금 온도를 올린다:

```python
def schedule_with_reheat(t, T0=100, reheat_interval=500):
    base_T = T0 * (0.995 ** t)
    # 이따금 다시 달구기
    if t % reheat_interval == 0 and t > 0:
        return base_T * 5  # 5배로 다시 달굼
    return base_T
```

다시 달구기는 지형에 높은 벽으로 갈린 깊은 웅덩이가 여럿 있을 때 특히 쓸모 있다. 온도를 이따금 치솟게 하면 사슬이 갇혀 있었을 웅덩이에서 벗어날 수 있다.

---

## 구현에서 살필 점

### 띄엄띄엄한 일정과 이어진 일정

**띄엄띄엄한(계단꼴) 일정**:

```python
temperatures = [100, 80, 60, 40, 20, 10, 5, 2, 1, 0.5, 0.1]
for T in temperatures:
    for _ in range(steps_per_temp):
        # 온도 T에서의 메트로폴리스 걸음
```

**이어진 일정**:

```python
for t in range(max_iter):
    T = schedule(t)
    # 온도 T에서의 메트로폴리스 걸음
```

띄엄띄엄한 일정은 층마다 평형에 이르게 해 주고, 이어진 일정은 구현이 더 간단하다.

### 벡터 꼴 구현

사슬 여럿을 나란히 돌리려면:

```python
def vectorized_sa(E, x0_batch, schedule, n_steps):
    """여러 시작점에서 SA을 나란히 돌리기."""
    n_chains = len(x0_batch)
    x = x0_batch.copy()
    E_x = np.array([E(xi) for xi in x])
    
    for t in range(n_steps):
        T = schedule(t)
        
        # 모든 사슬에 대해 내놓기
        x_prop = propose_batch(x)
        E_prop = np.array([E(xi) for xi in x_prop])
        
        # 받아들임 확률
        delta_E = E_prop - E_x
        accept_prob = np.minimum(1, np.exp(-delta_E / T))
        accept = np.random.rand(n_chains) < accept_prob
        
        # 갱신
        x[accept] = x_prop[accept]
        E_x[accept] = E_prop[accept]
    
    # 모든 사슬에서 가장 좋은 것 돌려주기
    best_idx = np.argmin(E_x)
    return x[best_idx], E_x[best_idx]
```

### 수치적 안정성

온도가 아주 낮으면 $e^{-\Delta E / T}$이 밑넘침을 일으킬 수 있다:

```python
def stable_accept_prob(delta_E, T, min_T=1e-10):
    """수치로 안정된 받아들임 확률."""
    T = max(T, min_T)
    
    if delta_E <= 0:
        return 1.0
    
    exponent = -delta_E / T
    if exponent < -700:  # 밑넘침 문턱값
        return 0.0
    
    return np.exp(exponent)
```

---

## 진단

### 온도와 받아들임 지켜보기

```python
def sa_with_diagnostics(E, x0, schedule, n_steps):
    x = x0
    history = {
        'temperature': [], 'energy': [],
        'best_energy': [], 'accept_rate': []
    }
    
    E_best = E(x)
    accepts = []
    
    for t in range(n_steps):
        T = schedule(t)
        x_new = propose(x)
        delta_E = E(x_new) - E(x)
        
        accept = np.random.rand() < min(1, np.exp(-delta_E / T))
        accepts.append(accept)
        
        if accept:
            x = x_new
        
        E_x = E(x)
        E_best = min(E_best, E_x)
        
        history['temperature'].append(T)
        history['energy'].append(E_x)
        history['best_energy'].append(E_best)
        
        if len(accepts) >= 100:
            history['accept_rate'].append(np.mean(accepts[-100:]))
    
    return x, history
```

시간에 따른 받아들임 비율은 가장 많은 것을 알려 주는 진단 가운데 하나이다. 잘 짠 일정은 돌리는 동안 받아들임 비율이 80%쯤에서 10%쯤으로 매끄럽게 내려가게 한다.

---

## 요약

| 일정 | 공식 | 모임 | 빠르기 | 쓰임새 |
|----------|---------|-------------|-------|----------|
| 로그 | $c/\log(t)$ | 보장됨 | 아주 느림 | 이론 |
| 지수 | $T_0 \alpha^t$ | 보장 없음 | 빠름 | 두루 쓰임 |
| 선형 | $T_0(1 - t/t_{\max})$ | 보장 없음 | 보통 | 단순한 문제 |
| 맞춰 감 | 받아들임에 바탕 | 보장 없음 | 그때그때 다름 | 모르는 지형 |
| 조각별 | 단계마다 다른 빠르기 | 보장 없음 | 보통 | 더 나은 살펴보기 |

**실전에서 권하는 바**:

1. 지수 일정($\alpha \approx 0.95$)으로 시작하여라
2. 첫 받아들임이 80%쯤 되도록 $T_0$을 잡아라
3. 돌리는 동안 받아들임 비율을 지켜보아라
4. 어려운 문제에는 맞춰 가는 일정을 써라
5. 아주 느리게 식히기보다 여러 번 다시 시작하는 것을 생각해 보아라

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
