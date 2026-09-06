# 이어진 시간 마르코프

continuous_time_markov.py (모듈 10) 이어진 시간 마르코프 사슬(CTMC)

확률 과정과 표집 방법은 확률적 기계 학습의 바탕 도구이다. 이 모듈은 마르코프 사슬의 개념을 보이며 수학 이론과 셈 구현 사이의 틈을 잇는다.

## 코드

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
        
        매개변수:
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
    print(f"Number of transitions: {len(times)-1}")```

## 논의

시각화는 모델의 거동을 이해하고 학습 문제를 진단하는 데 중요한 역할을 한다. 그림을 그리는 코드는 학습된 표현, 수렴의 움직임, 평가 지표에 대한 통찰을 주어 추상적인 계산을 손에 잡히게 만든다.

여기서 보인 무늬는 더 복잡한 상황으로 자연스럽게 넓어진다. 웃매개변수, 구조의 변형, 서로 다른 자료 묶음을 이리저리 시험해 보면 이해가 깊어지고 확률 과정 일감에 대한 실전 직관이 쌓인다.

## 연습문제

**연습문제 1.**
코드를 죽 읽고 핵심 설계 결정을 가려내어라. 구체적인 구현 고름 셋을 적고 저마다 왜 마르코프 사슬에 알맞은지 설명하여라.

??? success "연습문제 1 풀이"
    설계 결정은 구현마다 다르지만 흔히 다음이 포함된다. (1) 활성화 함수의 선택 — ReLU 계열은 포화되지 않는 경사를 주어 학습을 빠르게 한다. (2) 정규화 전략 — 배치 정규화는 내부 공변량 이동을 줄여 학습을 안정시킨다. (3) 잔차 연결 — 있을 경우 건너뛰는 경로를 제공하여 깊은 신경망에서도 경사가 흐르게 한다. 각 선택은 표현력, 계산 비용, 학습 안정성 사이의 절충을 반영한다.

---

**연습문제 2.**
입력이 기대하는 모양과 자료형을 갖는지 확인하도록 주 함수나 클래스에 입력 검증을 추가하라. 잘못된 입력에는 유익한 오류 메시지를 내라.

??? success "연습문제 2 풀이"
    `forward` 메서드(또는 해당 함수)의 첫머리에 다음과 같은 검사를 추가한다. `assert x.dim() == expected_dims, f'Expected {expected_dims}D input, got {x.dim()}D'`와 `assert x.dtype == torch.float32, f'Expected float32, got {x.dtype}'`. 모양을 검증할 때는 중요한 차원을 확인한다. `B, C, H, W = x.shape; assert C == self.expected_channels`. 유익한 오류 메시지는 디버깅 속도를 크게 높이고 코드를 재사용하기에도 더 견고하게 만든다.

---

**연습문제 3.**
이 구현에서 생길 수 있는 실패 양상 두 가지를 서술하고, 각각을 어떻게 진단하고 고칠지 설명하라.

??? success "연습문제 3 풀이"
    흔한 실패 양상은 다음과 같다. (1) **경사 소실/폭발** — 경사의 노름을 지켜보아 진단한다(`torch.nn.utils.clip_grad_norm_`을 쓰거나 층마다 `param.grad.norm()`을 기록한다). 경사 자르기, 더 나은 초기화(Xavier/Kaiming), 또는 구조 변경(잔차 연결, 정규화)으로 고친다. (2) **과적합** — 학습 손실은 줄어드는데 검증 손실이 늘어나면 진단된다. 정칙화(드롭아웃, 가중치 감쇠, 데이터 증강)나 모델 용량 축소로 고친다. 이런 문제를 일찍 잡아내려면 언제나 학습 지표와 검증 지표를 함께 살펴라.

---

**연습문제 4.**
이어진 시간 마르코프 구현을 확인하는 두루 갖춘 시험 함수를 적어라. 빈 입력, 원소 하나짜리 입력, 아주 큰 입력, 그리고 극단적인 값(0, 아주 큰 수)이 든 입력 같은 모서리 경우를 시험하여라.

??? success "연습문제 4 풀이"
    경계 조건을 두루 시험하는 함수를 만든다.
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
    경사의 흐름을 시험하는 것은 그 구조가 처음부터 끝까지 이어지는 학습을 지원하는지 확인하는 데 특히 중요하다.
