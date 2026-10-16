# 연속시간 마르코프 과정

연속시간 마르코프 사슬(CTMC)은 상태가 불규칙한 시간에 바뀌는 이산 상태 과정이다. 생성행렬의 비대각 성분은 전이확률이 아니라 단위시간당 전이율이다. 행렬 지수로 시점별 확률을 계산하고 지수 대기시간과 점프 방향으로 경로를 생성한다.

## 1. 코드

```python
"""
continuous_time_markov.py (단원 10)

이어진 시간 마르코프 사슬(CTMC)
=====================================

Location: 06_markov_chain/03_applications/
난이도: ⭐⭐⭐⭐ 상급
걸리는 시간: 3-4시간

학습 목표:
- 이어진 시간 과정 이해하기
- 옮김 확률 P(t) 셈하기
- 낳는 행렬 살피기
- 이어진 시간 마르코프 사슬 흉내내기

수학적 바탕:
이어진 시간 마르코프 사슬: X(t), t ≥ 0
- 옮김은 아무 때나 일어날 수 있다
- 머무는 시간이 지수 분포를 따른다
- 낳는 행렬 Q: Q[i][j] = i에서 j으로의 옮김 비율(i≠j)
- 콜모고로프 앞방정식: P'(t) = P(t) × Q
- 풀이: P(t) = exp(Qt)
"""

import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt

# ========================================================================
# 메인
# ========================================================================


class ContinuousTimeMarkovChain:
    """이어진 시간 마르코프 사슬 흉내내기 장치."""
    
    def __init__(self, generator_matrix, state_names=None):
        """
        이어진 시간 마르코프 사슬 첫값 잡기.
        
        파라메터:
            generator_matrix: Q[i][j]이 i에서 j으로의 비율인 Q(i≠j)
        """
        self.Q = np.array(generator_matrix, dtype=float)
        self.n_states = self.Q.shape[0]
        
        if state_names is None:
            self.state_names = [f"State {i}" for i in range(self.n_states)]
        else:
            self.state_names = state_names
    
    def transition_probabilities(self, t):
        """P(t) = exp(Qt) 셈하기."""
        return expm(self.Q * t)
    
    def simulate(self, T, initial_state=0):
        """시간 T까지 이어진 시간 마르코프 사슬 흉내내기."""
        times = [0]
        states = [initial_state]
        current_state = initial_state
        current_time = 0
        
        while current_time < T:
            # 지금 상태에 머무는 시간(비율이 -Q[i][i]인 지수 분포)
            rate = -self.Q[current_state, current_state]
            if rate <= 0:
                break
            
            holding_time = np.random.exponential(1/rate)
            current_time += holding_time
            
            if current_time >= T:
                break
            
            # 다음 상태 고르기
            transition_rates = self.Q[current_state, :].copy()
            transition_rates[current_state] = 0
            probs = transition_rates / transition_rates.sum()
            
            current_state = np.random.choice(self.n_states, p=probs)
            times.append(current_time)
            states.append(current_state)
        
        times.append(T)
        states.append(current_state)
        
        return times, states


# 보기: 태어남-죽음 과정
if __name__ == "__main__":
    print("CONTINUOUS-TIME MARKOV CHAINS")
    print("=" * 70)
    
    # 태어남-죽음 과정: 무리가 늘거나 줄 수 있다
    Q = np.array([
        [-2, 2, 0],
        [1, -3, 2],
        [0, 1, -1]
    ])
    
    ctmc = ContinuousTimeMarkovChain(Q, ['Low', 'Medium', 'High'])
    
    print("\\nGenerator Matrix Q:")
    print(Q)
    
    print("\\nTransition probabilities P(1.0):")
    P_1 = ctmc.transition_probabilities(1.0)
    print(P_1)
    
    print("\\nSimulation over time [0, 10]:")
    times, states = ctmc.simulate(10, initial_state=1)
    print(f"Number of transitions: {len(times)-1}")
```

**출력:**

```
CONTINUOUS-TIME MARKOV CHAINS
======================================================================
\nGenerator Matrix Q:
[[-2  2  0]
 [ 1 -3  2]
 [ 0  1 -1]]
\nTransition probabilities P(1.0):
[[0.27473809 0.33258542 0.39267649]
 [0.16629271 0.30478362 0.52892367]
 [0.09816912 0.26446183 0.63736904]]
\nSimulation over time [0, 10]:
Number of transitions: 20
```

## 2. 논의

### 생성행렬과 시점별 확률

유한 상태의 시간 동질 CTMC에서 생성행렬 Q는 i≠j에 대해 q_ij≥0, q_ii=−∑_{j≠i}q_ij를 만족한다. 각 행의 합은 0이다. q_ij는 1을 넘을 수도 있는 **전이율**이며 시간 단위와 함께 정의한다.

$$P(t)=e^{tQ},\qquad P(0)=I,\quad P(s+t)=P(s)P(t)$$

이것은 원소별 지수함수가 아니라 행렬 지수다. 초기 행분포 p₀에서 시점 t의 분포는 p₀P(t)다. P(t)는 각 행의 합이 1인 확률행렬이어야 한다.

### 경로 시뮬레이션

상태 i의 이탈률 λ_i=−q_ii>0이면 대기시간은 Exp(rate=λ_i)이며 평균은 1/λ_i다. 점프할 때 j≠i를 q_ij/λ_i의 확률로 선택한다. λ_i=0이면 유효 생성행렬에서 i는 흡수 상태다. 점프 시점의 상태 빈도와 체류시간으로 가중한 점유율은 다르다.

### 코드의 반환 규약

simulate(T)는 실제 점프 시점들을 기록하고 마지막에 관측 종료 T를 붙인다. 마지막 T는 새 점프가 아니라 마지막 상태 구간의 경계다. 따라서 T>0의 정상적인 경로에서 실제 점프 수는 len(times)−2이며 출력 코드의 len(times)−1은 하나 크게 센다. 흡수 상태에서 시작하면 경계 [0,T]와 같은 상태 둘을 반환할 수 있지만 점프는 없다.

생성자는 Q의 정방형·유한성·비대각 비음성·행합 0을 검사하지 않는다. 상태 이름·초기 인덱스와 T≥0도 검증해야 한다. 유효하지 않은 음의 이탈률을 흡수 상태처럼 처리하는 것은 올바른 모형 보정이 아니다. T=0의 중복 경계도 실제 사건 두 개로 세지 않는다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
생성행렬의 전이율과 전이행렬의 확률을 구분하라.

</div>

??? success "연습문제 1 풀이"

    q_ij는 i≠j에서 단위시간당 전이율이고 q_ii는 이탈률의 음수다. 행합은 0이며 비대각 값이 1보다 클 수 있다. 시점 t의 전이확률은 행렬 지수 e^(tQ)로 구하고 행합은 1이다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
CTMC의 대기시간과 다음 점프 상태 분포를 설명하라.

</div>

??? success "연습문제 2 풀이"

    이탈률 λ_i=−q_ii>0이면 대기시간은 평균 1/λ_i의 지수분포다. 점프 목적지는 j≠i에 대해 q_ij/λ_i로 선택한다. λ_i=0이고 생성행렬이 유효하면 영원히 머무는 흡수 상태다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
출력의 Number of transitions가 실제 점프 수와 다른 이유는?

</div>

??? success "연습문제 3 풀이"

    times에는 초기 0과 실제 점프 시점들 외에 마지막 관측 경계 T가 추가된다. 마지막 경계에서 상태 전이는 없으므로 T>0의 경로에서는 len(times)−2가 실제 점프 수다. 상태별 점유율도 점프 수가 아니라 각 구간 길이로 가중해야 한다.



---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
아래 뉴럴 넷트웍 시험 조각 대신 CTMC에서 점검할 항목을 설명하라.

</div>

??? success "연습문제 4 풀이"

    이 클래스는 뉴럴 넷트웍·경사 인터페이스가 없다. Q의 행합 0·비대각 비음성, P(0)=I, P(s+t)≈P(s)P(t), P(t)의 확률 조건, 정렬된 경로 시간과 마지막 T의 의미를 확인해야 한다. 흡수 상태·T=0·잘못된 초기 인덱스도 경계 사례다.

    ```python
    def test_continuoustimemarkovchain():
        model = ContinuousTimeMarkovChain(...)
        # 보통의 입력
        assert model(normal_input).shape == expected_shape
        # 원소가 하나인 배치
        assert model(single_input).shape == (1, ...)
        # 큰 값 (넘침을 확인한다)
        out = model(torch.ones(...) * 1000)
        assert torch.isfinite(out).all()
        # 경사의 흐름
        out = model(normal_input)
        out.sum().backward()
        for p in model.parameters():
            assert p.grad is not None
    ```


## 정리하며

CTMC는 전이율 생성행렬·지수 대기시간·점프 확률로 정의된다. 행렬 지수와 원소별 지수, 점프 횟수와 체류시간, 실제 점프와 관측 종료 경계를 구분한다. 유효한 생성행렬과 시간 단위가 기본 조건이다.
