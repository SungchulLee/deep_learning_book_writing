# 온도 스케줄

온도 스케줄은 에너지 장벽을 넘을 기회와 낮은 에너지로 집중하는 속도를 조절한다. 온도 변화 횟수·온도당 전이 수·총 예산을 구분하고, 점근 수렴 정리와 유한 실행의 경험적 성능을 따로 평가한다.

## 1. 냉각 단위와 예산

고정된 양의 ΔE에서는 온도가 낮을수록 오르막 수락률이 작다. 실제 이동은 제안 폭·상태·제약에도 달려 있다. 느린 일정은 더 많은 탈출 기회를 줄 수 있지만 같은 예산에서 항상 더 좋은 해를 보장하지 않는다.

한 온도를 유지하는 plateau의 전이 수와 온도를 내리는 횟수를 명시한다. α=0.95를 매 전이에 적용하는 것과 100번 이동한 뒤 적용하는 것은 전혀 다른 예산과 냉각 속도다.

## 2. 이론에서 나온 일정

특정 유한 상태·연결된 제안·가역성·단조 냉각의 SA 이론에서는 깊이 d*에 대한 급수 ∑exp(−d*/T_t)의 발산이 최적 집합 집중을 특징짓는다. 로그족 T_t=c/log(t+t₀), t₀>1에서는 관련 조건 아래 c≥d*가 기준이 된다. 로그 형태만이 유일한 가능한 일정이라는 뜻은 아니다.

이 결과는 보통 P(X_t∈X*)→1이라는 확률적 집중이며 거의 확실히 어느 한 최적점에 영원히 머문다는 주장과 다르다. c<d*의 로그족이나 일반적인 빠른 냉각에서는 어떤 시작점의 국소 함정 확률이 남을 수 있다.

T_t=c/log(t+t₀)가 작은 온도 η에 도달하는 시간은 대략 exp(c/η)−t₀다. 이는 온도 도달시간이지 실패확률 η를 얻는 보편적 복잡도 공식이 아니다. T₀/log(1+at)는 t=0에서 정의되지 않으므로 시작 오프셋이 필요하고 이름만으로 역선형·로그의 보장을 결합하지 않는다.

## 3. 실전에서 쓰는 일정

등비 T_k=T₀a^k, 0<a<1은 간단하다. 원하는 T_f<T₀에 도달하는 실수 인덱스는 log(T_f/T₀)/log a이며 실제 반복은 올림 등 종료 관례를 적용한다. T₀·T_f·횟수를 정하고 a=(T_f/T₀)^(1/K)로 계산할 수도 있다.

선형 T_t=T₀(1−t/t_max)과 이차 T_t=T₀(1−t/t_max)²는 t_max에서 0이 된다. 0으로 나누는 Metropolis 식을 그대로 호출하지 않고 종료·하한·명시적인 탐욕 규칙을 정해야 한다. 이차식은 절대 기울기 2T₀(1−t/t_max)/t_max가 줄어드므로 처음 더 빠르고 끝에 더 느리게 감소한다. 끝에 더 빠르다는 설명은 반대다.

역선형 T_t=T₀/(1+at)는 양수로 0에 접근한다. 이런 일정의 선택은 문제·제안과 예산에 따라 검토하며 일반적인 전역 수렴 보장을 자동으로 부여하지 않는다.

## 4. 일정 비교

서로 다른 형태를 비교하려면 같은 초기·종료 온도와 총 전이 예산을 맞춘다. c/log(t+t₀)에서 c가 100이라고 초기온도가 100인 것은 아니다.

고정 에너지 단위와 제안에서 여러 일정의 최선 방문값·최종값·제약 만족·반복 변동·총 평가 수를 비교한다. 지형이 거칠다는 설명만으로 특정 일정이 보편적으로 최적이라고 정할 수는 없다.

## 5. 수락률 기반 적응의 범위

아래 코드는 최근 window의 수락률에 따라 온도를 내리거나 올리는 휴리스틱이며 max_iter와 propose가 외부에서 정의되어야 한다. target_accept·window·양의 T와 유효한 난수·에너지 입력을 확인한다. 이 규칙은 단조 냉각이나 T→0을 보장하지 않는다.

80%→40%→10%의 수락률 경로가 좋은 일정의 필수조건은 아니다. 일부 문헌에 붙은 사람 이름의 일정은 그 원래 정의·평형 가정·온도 갱신 단위에 따라 달라진다. 여기서는 단순한 exp(−T/σ_E) 식을 특정 정식 일정의 완전한 구현으로 제시하지 않고, 수락·에너지 변동을 활용하는 설계의 예로만 이해한다. σ_E=0과 잡음이 큰 추정도 처리해야 한다.

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

## 6. 여러 단계 일정

piecewise_schedule은 앞 20%·가운데 60%·마지막 20%에 서로 다른 등비율을 연결한다. 경계 온도는 계산상 이어지지만 실제 평가 t의 정수화·범위와 종료를 확인한다.

schedule_with_reheat는 재가열 시점 한 번만 base_T의 5배를 반환하고 바로 다음 단계는 원래 등비 경로로 돌아간다. 재가열한 상태에서 새 냉각을 시작하는 지속적인 온도 리셋과는 다르다. 이런 일시 상승이 장벽 탈출이나 전역 해를 보장하지는 않는다.

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
```python
def schedule_with_reheat(t, T0=100, reheat_interval=500):
    base_T = T0 * (0.995 ** t)
    # 이따금 다시 달구기
    if t % reheat_interval == 0 and t > 0:
        return base_T * 5  # 5배로 다시 달굼
    return base_T
```

## 7. 구현에서 살필 점

계단형 일정에서 온도별 steps_per_temp를 수행한다고 반드시 평형에 도달하는 것은 아니다. 짧은 pass·schedule 호출 조각은 실제 이동을 구현하지 않는다.

vectorized_sa는 배치 제안과 행별 수락을 사용하지만 마지막 집단에서 에너지가 가장 낮은 상태만 반환한다. 전체 경로의 최선 방문 상태를 저장하는 구현이 아니다. 제안이 대칭이고 행별 에너지·기기 형태가 맞는지도 확인한다.

stable_accept_prob는 T를 양의 min_T로 하한 처리하고 아주 작은 오르막 확률을 0으로 잘라낸다. 이는 원래 T=0·음수 입력의 수학적 정의를 검증하는 함수가 아니며 목표 온도를 바꾸거나 매우 작은 확률을 근사한다. 비유한 에너지·온도도 별도로 확인한다. 로그 수락 비교는 exp 넘침을 줄이는 대안이다.

```python
temperatures = [100, 80, 60, 40, 20, 10, 5, 2, 1, 0.5, 0.1]
for T in temperatures:
    for _ in range(steps_per_temp):
        # 온도 T에서의 메트로폴리스 걸음
        pass
```
```python
for t in range(max_iter):
    T = schedule(t)
    # 온도 T에서의 메트로폴리스 걸음
```
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

## 8. 진단

현재 에너지·최선 방문 에너지·온도·제안 크기·수락률을 구분해 기록한다. 최선 에너지의 단조 감소는 기록 정의의 결과이며 전역 수렴의 증거가 아니다.

sa_with_diagnostics는 최선 에너지 숫자를 저장하지만 최선 위치는 반환하지 않는다. 수락률 목록은 100번 이후부터만 추가되어 온도·에너지 목록과 길이가 다르다. 이를 같은 인덱스의 전체 시간 기록으로 해석하지 않는다. 외부 propose와 유효한 양의 온도도 필요하다. 다중 시작의 해 차이와 독립 기준·하한을 함께 비교한다.

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

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
로그 일정이 유일한 수렴 일정인가?

</div>

??? success "연습문제 1 풀이"
    특정 SA 정리에서는 장벽 깊이에 대한 급수 조건으로 판단한다. 로그족은 임계 상수를 쉽게 해석하는 대표 사례이지 모든 가능한 일정의 유일한 형태는 아니다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
이차 냉각 T=T₀(1−t/t_max)²는 언제 더 빠르게 감소하는가?

</div>

??? success "연습문제 2 풀이"
    절대 미분 크기는 초기에 가장 크고 끝에 0으로 줄어든다. 따라서 처음 빠르고 끝에 느린 절대 감소다. 상대 감소율과 혼동하지 않는다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
vectorized_sa가 전체 방문 중 최선 상태를 반환하는가?

</div>

??? success "연습문제 3 풀이"
    현재 배치의 최종 에너지에서 최소인 상태만 반환한다. 각 경로가 예전에 방문한 더 좋은 상태는 별도로 저장하지 않아 잃을 수 있다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff easy" title="쉬움"></span>
온도 0을 반환하는 일정과 Metropolis 식을 어떻게 연결해야 하는가?

</div>

??? success "연습문제 4 풀이"
    T=0의 나눗셈을 그대로 수행하지 않는다. 종료하거나 명시적인 탐욕 규칙·양의 하한을 정하되 하한은 원래 냉각 목표와 다른 근사임을 밝힌다.

## 정리하며

스케줄의 형태뿐 아니라 온도당 이동 수와 전체 예산이 중요하다. 로그족의 점근 조건을 유한 성능 보장과 구분하고, 실제 코드의 재가열·최선 기록·0온도 처리를 확인한다.
