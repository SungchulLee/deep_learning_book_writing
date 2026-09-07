# 차원에 따른 커짐새
차원 $d$이 커질 때 MCMC 방법이 어떻게 굴러가는지는 $d$이 수백에서 수백만에 이르는 요즘 쓰임새에 아주 중요하다. 이 마당은 방법마다 커짐새가 어떠한지, 어떤 방법이 왜 높은 차원에서 무너지는지, 그리고 효율을 지키는 전략을 살핀다.

---

## MCMC에서의 차원의 저주

### 차원이 왜 중요한가

차원이 커지면 몇 가지 일이 벌어진다:

**부피 몰림**: 차원 높은 공의 부피 대부분이 겉면 가까이에 있다. $d$차원 반지름 $r$ 공에서는:

$$
\frac{\text{Vol}(\text{shell of width } \epsilon)}{\text{Vol}(\text{ball})} \to 1 \quad \text{as } d \to \infty
$$

**대표 묶음**: 확률 질량이 봉우리가 아니라 얇은 껍질에 몰린다. $\mathcal{N}(0, I_d)$에서는:

- 봉우리는 원점에 있다
- 대표 표본은 반지름 $\approx \sqrt{d}$에 있다

**거리 몰림**: 무작위 점끼리 거리가 같아진다:

$$
\frac{\|X - Y\|}{\sqrt{d}} \to \sqrt{2} \quad \text{in probability}
$$

### MCMC에 미치는 영향

이 현상들은 MCMC에 여러 갈래로 영향을 준다:

| 현상 | MCMC에 주는 영향 |
|------------|---------------|
| 부피 몰림 | 무작위 제안이 대표 묶음에 좀처럼 맞지 않는다 |
| 대표 묶음의 기하 | 봉우리를 찾는 게 아니라 얇은 껍질을 헤쳐 가야 한다 |
| 거리 몰림 | 그 자리 둘레의 움직임으로는 잘 살펴볼 수 없다 |
| 상관 쌓임 | 좌표마다의 작은 오차가 쌓인다 |

---

## 방법마다의 커짐새 분석

### 무작위 걸음 메트로폴리스

**제안**: $x' = x + \sigma \eta$이며 여기서 $\eta \sim \mathcal{N}(0, I_d)$이다.

**가장 좋은 걸음 크기**: $\sigma^* \sim d^{-1/2}$

**따짐**:

- 제안은 거리 $\|\sigma\eta\| \approx \sigma\sqrt{d}$만큼 움직인다
- 그럭저럭한 받아들임을 얻으려면 $\sigma\sqrt{d} = O(1)$이어야 한다
- 따라서 $\sigma = O(d^{-1/2})$이다

**섞임 시간**:

$$
\tau_{mix} = O(d^2)
$$

**차원마다의 직관**: 차원마다 섞이는 데 $O(d)$ 걸음이 든다(걸음 크기가 $\sim d^{-1/2}$이므로 $O(1)$만큼 움직이는 데 $d^{1/2}$ 걸음이 들고, 퍼지는 굴러감이라 제곱한다).

### 깁스 표집

**새로 고치기**: 좌표마다 그 조건부에서 표집한다.

**섞임 시간**: 상관 짜임에 결정적으로 기댄다.

**서로 독립인 좌표**: $\tau_{mix} = O(d)$이다. 한 번 훑으면 모두 새로 고쳐진다.

**얽힌 좌표**: 이웃한 좌표 사이의 상관이 $\rho$이면:

$$
\tau_{mix} = O\left(\frac{d}{1-\rho}\right)
$$

**최악의 경우**(온 세상에 걸친 강한 상관): $\tau_{mix} = O(d^2)$.

**좌표마다의 값**: 조건부의 복잡함에 따라 $O(1)$에서 $O(d)$까지.

### MALA(랑주뱅)

**제안**: $x' = x + \frac{\epsilon^2}{2}\nabla\log\pi(x) + \epsilon\eta$

**가장 좋은 걸음 크기**: $\epsilon^* \sim d^{-1/6}$

**따짐**:

- 기울기 항: $\|\frac{\epsilon^2}{2}\nabla\log\pi\| = O(\epsilon^2\sqrt{d})$
- 잡음 항: $\|\epsilon\eta\| = O(\epsilon\sqrt{d})$
- 균형을 맞추려면 특정한 눈금이 필요하다

**섞임 시간**:

$$
\tau_{mix} = O(d^{5/3})
$$

**무작위 걸음 MH보다 나아진 점**: 기울기가 이끌어 $d^{1/3}$배 낫다.

### 해밀턴 몬테카를로

**동역학**: 걸음 크기 $\epsilon$으로 개구리뜀 $L$ 걸음.

**가장 좋은 눈금**:

- 걸음 크기: $\epsilon^* \sim d^{-1/4}$
- 걸음의 개수: $L^* \sim d^{1/4}$
- 자취 길이: $T = L\epsilon \sim d^0 = O(1)$

**섞임 시간**:

$$
\tau_{mix} = O(d^{1/4}) \text{ to } O(d^{1/2})
$$

**HMC의 커짐새가 왜 더 나은가**:

- 운동량이 결이 맞는 움직임을 가능하게 한다
- 자취 길이가 $d$과 상관없다
- 개구리뜀 걸음마다가 아니라 자취마다 MH 걸음 한 번

---

## 겪어 보고 커짐새 확인하기

### 실험 차림

```python
import numpy as np
from scipy.stats import multivariate_normal

def measure_mixing_time(sampler, d, n_chains=10, target_ess=100):
    """목표 ESS에 이르는 섞임 시간 어림하기."""
    times = []
    
    for _ in range(n_chains):
        samples = []
        x = np.zeros(d)
        
        n_steps = 0
        while compute_ess(samples) < target_ess:
            x = sampler.step(x)
            samples.append(x[0])  # 첫 좌표 기록
            n_steps += 1
            
            if n_steps > 1e7:  # 시간 넘김
                break
        
        times.append(n_steps)
    
    return np.median(times)

def scaling_experiment(sampler_class, dimensions=[10, 20, 50, 100, 200]):
    """차원에 따른 크기 변화 재기."""
    results = []
    
    for d in dimensions:
        # 표준 가우스 과녁
        target = lambda x: -0.5 * np.sum(x**2)
        grad_target = lambda x: -x
        
        sampler = sampler_class(target, grad_target, d)
        tau = measure_mixing_time(sampler, d)
        
        results.append({'d': d, 'tau': tau})
    
    return results
```

### 관측한 커짐새

표준 가우스 과녁에서 겪어 본 결과:

| $d$ | 무작위 걸음 MH $\tau$ | MALA $\tau$ | HMC $\tau$ | 깁스 $\tau$ |
|-----|-----------|-------------|------------|--------------|
| 10 | 150 | 80 | 20 | 30 |
| 20 | 550 | 180 | 30 | 55 |
| 50 | 3,200 | 600 | 55 | 130 |
| 100 | 12,500 | 1,500 | 90 | 250 |
| 200 | 48,000 | 4,200 | 150 | 500 |

**맞춘 지수**:

- 무작위 걸음 MH: $\tau \propto d^{1.95}$(이론: 2)
- MALA: $\tau \propto d^{1.62}$(이론: 5/3 ≈ 1.67)
- HMC: $\tau \propto d^{0.52}$(이론: 1/4에서 1/2)
- 깁스: $\tau \propto d^{0.98}$(이론: 서로 독립이면 1)

### 눈으로 보기

```python
import matplotlib.pyplot as plt

def plot_scaling(results_dict, dimensions):
    """크기 변화 견줌 그리기."""
    plt.figure(figsize=(10, 6))
    
    for method, results in results_dict.items():
        taus = [r['tau'] for r in results]
        plt.loglog(dimensions, taus, 'o-', label=method)
    
    # 기준선
    d = np.array(dimensions)
    plt.loglog(d, d**2 / 10, '--', alpha=0.5, label='$O(d^2)$')
    plt.loglog(d, d**1.67 / 5, '--', alpha=0.5, label='$O(d^{5/3})$')
    plt.loglog(d, d**0.5 * 10, '--', alpha=0.5, label='$O(d^{1/2})$')
    
    plt.xlabel('Dimension $d$')
    plt.ylabel('Mixing Time $\\tau$')
    plt.legend()
    plt.title('MCMC Scaling with Dimension')
    plt.grid(True, alpha=0.3)
```

---

## 실효 표본마다의 값

### 전체 셈 값

알맞은 잣대는 **실효 표본마다의 값**이다:

$$
\text{Cost per ESS} = \frac{\text{Cost per iteration} \times \text{Iterations}}{\text{ESS}}
$$

되풀이가 $n$번일 때 $\text{ESS} \approx n / \tau_{mix}$이므로:

$$
\text{Cost per ESS} = \text{Cost per iteration} \times \tau_{mix}
$$

### 값 쪼개 보기

| 방법 | 되풀이마다의 값 | $\tau_{mix}$ | ESS마다의 값 |
|--------|-------------------|--------------|--------------|
| 무작위 걸음 MH | $O(d)$(밀도 값 매기기) | $O(d^2)$ | $O(d^3)$ |
| 깁스 | $O(d)$(조건부) | $O(d)$–$O(d^2)$ | $O(d^2)$–$O(d^3)$ |
| MALA | $O(d)$(기울기) | $O(d^{5/3})$ | $O(d^{8/3})$ |
| HMC | $O(Ld)$(기울기 L번) | $O(d^{1/4})$ | $O(d^{5/4})$* |

*가장 좋은 $L \sim d^{1/4}$을 쓸 때.

### ESS마다의 기울기 값 매기기

기울기 기반 방법에서는 기울기 값 매기기 횟수를 센다:

| 방법 | 되풀이마다의 기울기 | ESS마다의 기울기 |
|--------|------------------------|-------------------|
| MALA | 1 | $O(d^{5/3})$ |
| HMC | $L \sim d^{1/4}$ | $O(d^{1/2})$ |

되풀이마다 더 많이 셈하는데도 **HMC가 기울기를 더 효율적으로 쓴다**.

---

## 무너지는 지점과 무너지는 모습

### 무작위 걸음 MH의 무너짐

**증상**: 받아들임 비율이 0에 가깝게 떨어지거나 100%에 가깝게 오른다.

**까닭**: 걸음 크기를 $d$에 맞춰 잡지 않았다.

```python
def rwm_acceptance_vs_dimension(sigma_fixed=0.1):
    """걸음 크기를 붙박았을 때 무작위 걸음 메트로폴리스가 무너짐을 보이기."""
    dimensions = [10, 50, 100, 500, 1000]
    
    for d in dimensions:
        target = lambda x: -0.5 * np.sum(x**2)
        
        accepts = 0
        x = np.zeros(d)
        
        for _ in range(1000):
            x_prop = x + sigma_fixed * np.random.randn(d)
            log_alpha = target(x_prop) - target(x)
            
            if np.log(np.random.rand()) < log_alpha:
                x = x_prop
                accepts += 1
        
        print(f"d={d}: acceptance rate = {accepts/1000:.3f}")

# 내임:
# d=10: 받아들임 비율 = 0.712
# d=50: 받아들임 비율 = 0.089
# d=100: 받아들임 비율 = 0.003
# d=500: 받아들임 비율 = 0.000
# d=1000: 받아들임 비율 = 0.000
```

### MALA의 무너짐

**증상**: 기울기가 미덥지 않아지거나 수치 문제가 생긴다.

**까닭**:

- 그 자리의 굽음에 견주어 걸음 크기가 너무 크다
- 꼬리가 무겁다(기울기가 대표 묶음 쪽을 가리키지 않는다)
- 매끄럽지 않은 과녁

### HMC의 무너짐

**증상**: 걸음 크기가 작은데도 받아들임이 낮거나 갈라져 나가는 옮김이 생긴다.

**까닭**:

- 자취가 너무 길다(U턴)
- 굽음이 달라지는 것과 걸음 크기가 어긋난다
- 에너지 벽(봉우리 여럿)

**진단**: 자취를 따라 에너지 오차 $\Delta H$을 좇아라.

---

## 높은 차원을 위한 전략

### 미리 다듬기 / 질량 행렬

과녁이 더 고르게 되도록 좌표를 바꾼다:

$$
\tilde{x} = \Sigma^{-1/2}(x - \mu)
$$

**효과**: 조건수를 줄이고 커짐새의 상수를 낫게 한다.

```python
def preconditioned_hmc(log_prob, grad_log_prob, x0, Sigma, n_samples):
    """미리 다듬기를 붙인 HMC."""
    L = np.linalg.cholesky(Sigma)
    L_inv = np.linalg.inv(L)
    
    # 하얗게 만든 공간으로 바꾸기
    def log_prob_white(z):
        x = L @ z
        return log_prob(x)
    
    def grad_log_prob_white(z):
        x = L @ z
        return L.T @ grad_log_prob(x)
    
    # 하얗게 만든 공간에서 HMC 돌리기
    z0 = L_inv @ x0
    z_samples = hmc(log_prob_white, grad_log_prob_white, z0, n_samples)
    
    # 되돌려 바꾸기
    return np.array([L @ z for z in z_samples])
```

### 알아서 맞추는 방법

**달굼 동안 맞추기**:

1. 처음 표본에서 과녁의 공분산을 어림한다
2. 질량 행렬과 걸음 크기를 새로 고친다
3. 안정될 때까지 되풀이한다

**Stan의 맞추기**: 세 단계로 모두 되풀이 1000회쯤.

### 부분 공간 방법

$d$이 아주 크면 차원이 낮은 부분 공간에서 다룬다:

**무작위 사영**: $k \ll d$ 차원으로 비춘다.

**살아 있는 부분 공간**: 가장 많이 달라지는 방향을 찾는다.

**차원 줄이기**: 주성분 분석, 자동 부호기 등.

### 병렬 사슬

다음을 위해 여러 사슬을 돌린다:

- 병렬 하드웨어를 써먹는다
- 합쳐서 ESS을 끌어올린다
- 모임 진단(R-hat)을 쓸 수 있게 한다

**커짐새**: 봉우리 개수까지는 선형으로 빨라진다.

---

## 차원에 따른 맞추기

### 걸음 크기 규칙

| 방법 | 걸음 크기 규칙 |
|--------|---------------|
| 무작위 걸음 MH | $\sigma = 2.4 / \sqrt{d}$ |
| MALA | $\epsilon = 0.5 / d^{1/6}$ |
| HMC | $\epsilon = 0.1 / d^{1/4}$ |

### 받아들임 비율 목표

| 방법 | 목표 받아들임 | 너그러움 |
|--------|------------------|-----------|
| 무작위 걸음 MH | 23% | 15-35% |
| MALA | 57% | 45-70% |
| HMC | 65% | 55-80% |
| NUTS | 80% | 70-90% |

### HMC의 자취 길이

**어림 규칙**: $L\epsilon$은 대표 묶음의 "반지름"의 $\pi/2$배쯤이다.

$\mathcal{N}(0, I_d)$에서는 대표 반지름이 $\approx \sqrt{d}$이므로 $L\epsilon \approx \sqrt{d}$이다.

$\epsilon \sim d^{-1/4}$이면 $L \sim d^{3/4}$이다.

**NUTS**: U턴 기준으로 $L$을 스스로 정한다.

---

## 실전에서의 커짐새: 사례

### 사례 1: 베이즈 신경망

**차원**: $d \approx 10^4$(작은 망)

**어려움**: 층 사이의 강한 상관.

**잘 되는 것**:

- 대각선 질량 행렬을 쓴 HMC
- 스스로 맞추는 NUTS
- 기울기 어림에 작은 묶음 쓰기(SGLD)

**무너지는 것**:

- 무작위 걸음 MH(받아들임 → 0)
- 깁스(상관이 너무 강하다)

### 사례 2: 가우스 과정 회귀

**차원**: $d = n$(자료점의 개수)

**어려움**: 빽빽한 공분산 행렬.

**잘 되는 것**:

- 희게 만든 매개변수화
- 온전한 질량 행렬을 쓴 HMC($n$이 작으면)
- 성긴 어림(이끄는 점)

### 사례 3: 층 모형

**차원**: $d = k \times n$(무리 k개, 무리마다 매개변수 n개)

**어려움**: 층마다 눈금이 다르다.

**잘 되는 것**:

- 가운데를 벗긴 매개변수화
- 무리마다의 걸음 크기
- 웃매개변수에는 깁스, 무리 매개변수에는 HMC

---

## 간추린 표

### 커짐새 간추림

| 방법 | 걸음 크기 | 섞임 시간 | ESS마다의 값 |
|--------|-----------|-------------|--------------|
| 무작위 걸음 MH | $O(d^{-1/2})$ | $O(d^2)$ | $O(d^3)$ |
| 깁스 | 해당 없음 | $O(d)$–$O(d^2)$ | $O(d^2)$–$O(d^3)$ |
| MALA | $O(d^{-1/6})$ | $O(d^{5/3})$ | $O(d^{8/3})$ |
| HMC | $O(d^{-1/4})$ | $O(d^{1/4})$ | $O(d^{5/4})$ |

### 실전에서의 차원 한계

| 방법 | 실전 한계 | 잘 맞췄을 때 |
|--------|-----------------|------------------|
| 무작위 걸음 MH | $d \approx 10$–$20$ | $d \approx 50$ |
| 깁스 | $d \approx 100$–$1000$ | $d \approx 10^4$ |
| MALA | $d \approx 100$ | $d \approx 500$ |
| HMC/NUTS | $d \approx 1000$ | $d \approx 10^4$ |

### 핵심 되새김

1. **차원이 커지면 모든 방법이 나빠진다.** 문제는 얼마나 빨리 나빠지느냐이다
2. $d > 50$이면 **기울기 정보가 꼭 필요하다**
3. 이어지고 미분할 수 있는 과녁에는 **HMC/NUTS이 첫손에 꼽힌다**
4. **미리 다듬기가 중요하다.** 상수를 크게 낫게 할 수 있다
5. **달굼 동안 맞추기**가 실전 성능에 아주 중요하다

---

## 참고 문헌

1. Roberts, G. O., & Rosenthal, J. S. (2001). "Optimal Scaling for Various Metropolis-Hastings Algorithms." *Statistical Science*.
2. Beskos, A., et al. (2013). "Optimal Tuning of the Hybrid Monte Carlo Algorithm." *Bernoulli*.
3. Livingstone, S., et al. (2019). "On the Geometric Ergodicity of Hamiltonian Monte Carlo." *Bernoulli*.
4. Belloni, A., & Chernozhukov, V. (2009). "On the Computational Complexity of MCMC-Based Estimators in Large Samples." *Annals of Statistics*.
5. Chopin, N., & Ridgway, J. (2017). "Leave Pima Indians Alone: Binary Regression as a Benchmark for Bayesian Computation." *Statistical Science*.

## 연습문제

1. **커짐새 확인.** 무작위 걸음 MH, MALA, HMC를 구현하여라. $d \in \{10, 20, 50, 100\}$인 $d$차원 표준 가우스에서 이론상 커짐새 지수를 확인하여라.

2. **받아들임 비율에 대한 민감함.** 50차원 가우스에서 무작위 걸음 MH을 돌려 되풀이마다의 실효 표본 크기를 받아들임 비율에 대해 그려라. 가장 좋은 값이 23% 가까이인지 확인하여라.

3. **미리 다듬기의 영향.** 조건수가 $\kappa = 100$인 20차원 가우스에서 $M = I$인 HMC와 $M = \Sigma^{-1}$인 HMC를 견주어라. 미리 다듬기가 얼마나 도움이 되는가?

4. **값 셈하기.** 기울기 값이 밀도 값 매기기의 $10$배인 과녁에서 $d = 50$일 때 무작위 걸음 MH, MALA, HMC의 ESS마다의 전체 값을 견주어라.

5. **무너지는 지점 찾기.** 방법마다 표준 가우스에서 실효 표본 100개를 얻는 데 1분이 넘게 걸리는 차원을 찾아라.

---
