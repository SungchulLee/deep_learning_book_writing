# 기본 마르코프 사슬

basic_markov_chain.py (모듈 01) 마르코프 사슬 들여오기

확률 과정과 표집 방법은 확률적 기계 학습의 바탕 도구이다. 이 모듈은 마르코프 사슬의 개념을 보이며 수학 이론과 셈 구현 사이의 틈을 잇는다.

## 1. 코드

```python
"""
basic_markov_chain.py (단원 01)

마르코프 사슬 들어가기
==============================

Location: 06_markov_chain/01_fundamentals/
난이도: ⭐ 첫걸음
예상 시간: 2~3시간

학습 목표:
- 마르코프 성질 이해하기
- 단순한 띄엄띄엄한 시간 마르코프 사슬 구현하기
- 상태 옮김 흉내내기
- 상태 늘어놓음 그려 보기

수학적 바탕:
마르코프 사슬은 다음을 만족하는 확률 변수의 늘어놓음 X_0, X_1, X_2, ...이다:
P(X_{n+1} = j | X_n = i, X_{n-1} = k, ..., X_0 = m) = P(X_{n+1} = j | X_n = i)

이를 마르코프 성질이라 한다. 곧 미래는 지난날이 아니라 지금에만 달렸다.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# ========================================================================
# 메인
# ========================================================================


class BasicMarkovChain:
    """
    띄엄띄엄한 시간 마르코프 사슬의 단순한 구현.
    
    속성:
        states (list): 상태 이름의 목록
        transition_matrix (np.ndarray): 옮김 확률 행렬
        current_state (int): 지금 상태의 번호
    """
    
    def __init__(self, states, transition_matrix):
        """
        마르코프 사슬 첫값 잡기.
        
        매개변수:
            states (list): 상태의 이름(이를테면 ['A', 'B', 'C'])
            transition_matrix (np.ndarray): 옮김 확률 행렬 P
                                          여기서 P[i][j] = P(X_{n+1}=j | X_n=i)
        
        수학 메모:
        - transition_matrix의 행마다 합이 1이어야 한다
        - 성분이 모두 음이 아니어야 한다
        """
        self.states = states
        self.transition_matrix = np.array(transition_matrix)
        self.n_states = len(states)
        
        # 옮김 행렬 확인하기
        self._validate_transition_matrix()
        
        # 무작위 상태에서 시작하기
        self.current_state = np.random.randint(0, self.n_states)
    
    def _validate_transition_matrix(self):
        """
        옮김 행렬이 확률 행렬인지 확인하기.
        
        다음이면 행렬이 확률 행렬이다:
        1. 성분이 모두 음이 아니다
        2. 행마다 합이 1이다(확률 분포)
        """
        # 차원 살피기
        if self.transition_matrix.shape != (self.n_states, self.n_states):
            raise ValueError(f"Transition matrix must be {self.n_states}x{self.n_states}")
        
        # 음이 아닌지 살피기
        if np.any(self.transition_matrix < 0):
            raise ValueError("All transition probabilities must be non-negative")
        
        # 행의 합 살피기(각각 1이 되어야 함)
        row_sums = np.sum(self.transition_matrix, axis=1)
        if not np.allclose(row_sums, 1.0):
            raise ValueError("Each row of transition matrix must sum to 1")
    
    def step(self):
        """
        마르코프 사슬의 한 걸음 밟기.
        
        반환값:
            str: 새 상태의 이름
        
        수학 과정:
        지금 상태가 i일 때 확률 P[i][j]으로 다음 상태 j 고르기
        이는 범주 분포에서 표집하는 것과 같다
        옮김 행렬의 i번째 행이 정하는.
        """
        # 지금 상태의 옮김 확률 얻기
        probabilities = self.transition_matrix[self.current_state]
        
        # 이 확률에 따라 다음 상태 표집
        # np.random.choice는 확률이 정하는 띄엄띄엄한 분포를 쓴다
        self.current_state = np.random.choice(self.n_states, p=probabilities)
        
        return self.states[self.current_state]
    
    def simulate(self, n_steps, initial_state=None):
        """
        마르코프 사슬을 n걸음 흉내내기.
        
        매개변수:
            n_steps (int): 흉내낼 걸음 수
            initial_state (int or str): 시작 상태(번호나 이름)
        
        반환값:
            list: 들른 상태의 늘어놓음
        
        수학 메모:
        이는 확률 과정 {X_n}의 한 실현을 만든다
        """
        # 주어졌으면 첫 상태 잡기
        if initial_state is not None:
            if isinstance(initial_state, str):
                self.current_state = self.states.index(initial_state)
            else:
                self.current_state = initial_state
        
        # 상태 늘어놓음 기록하기
        state_sequence = [self.states[self.current_state]]
        
        # n걸음 흉내내기
        for _ in range(n_steps):
            state_sequence.append(self.step())
        
        return state_sequence
    
    def get_state_distribution(self, n_steps, n_simulations=10000):
        """
        몬테카를로 흉내내기로 n걸음 뒤의 상태 분포 어림하기.
        
        매개변수:
            n_steps (int): 뗄 걸음 수
            n_simulations (int): 돌릴 흉내내기의 횟수
        
        반환값:
            np.ndarray: 상태에 걸쳐 어림한 확률 분포
        
        수학적 바탕:
        π_0이 첫 분포이고 P이 옮김 행렬이면,
        n걸음 뒤의 분포는 π_n = π_0 * P^n이다
        
        흉내내기를 많이 돌리고 세어 이를 어림한다.
        """
        # 모든 흉내내기의 마지막 상태 세기
        final_state_counts = np.zeros(self.n_states)
        
        for _ in range(n_simulations):
            # 흉내내기 한 번 돌리기
            sequence = self.simulate(n_steps)
            final_state = sequence[-1]
            
            # 마지막 상태 세기
            final_state_index = self.states.index(final_state)
            final_state_counts[final_state_index] += 1
        
        # 셈한 수를 확률로 바꾸기
        estimated_distribution = final_state_counts / n_simulations
        
        return estimated_distribution


def example_two_state_chain():
    """
    보기 1: 단순한 두 상태 마르코프 사슬
    
    상태: {0, 1} 또는 {Off, On}
    옮김 행렬:
        에서\로  Off   On
        Off      0.7   0.3
        On       0.4   0.6
    
    해석:
    - 지금 Off이면 70%은 Off 그대로, 30%은 On으로
    - 지금 On이면 40%은 Off으로, 60%은 On 그대로
    """
    print("=" * 60)
    print("Example 1: Two-State Markov Chain (Off/On)")
    print("=" * 60)
    
    # 상태 정하기
    states = ['Off', 'On']
    
    # 옮김 행렬 정하기
    # P[i][j] = 상태 i에서 상태 j으로 갈 확률
    transition_matrix = [
        [0.7, 0.3],  # Off에서: 70% Off 그대로, 30% On으로
        [0.4, 0.6]   # On에서: 40% Off으로, 60% On 그대로
    ]
    
    # 마르코프 사슬 만들기
    mc = BasicMarkovChain(states, transition_matrix)
    
    print("\nTransition Matrix:")
    print(transition_matrix)
    print("\nSimulating 20 steps...")
    
    # 'Off'에서 시작해 흉내내기
    sequence = mc.simulate(n_steps=20, initial_state='Off')
    print(f"\nState sequence: {sequence}")
    
    # 오래 뒤의 분포 어림하기
    print("\nEstimating state distribution after 100 steps...")
    distribution = mc.get_state_distribution(n_steps=100)
    
    print(f"Estimated probabilities:")
    for state, prob in zip(states, distribution):
        print(f"  P({state}) = {prob:.4f}")


def example_three_state_chain():
    """
    보기 2: 세 상태 날씨 모형
    
    상태: {맑음, 흐림, 비}
    
    이는 간추린 날씨 옮김을 본뜬다:
    - 맑음은 맑음 그대로이거나 흐림이 되는 경향이 있다
    - 흐림은 어느 쪽으로든 갈 수 있다
    - 비는 흐림이 되거나 비 그대로인 경향이 있다
    """
    print("\n" + "=" * 60)
    print("Example 2: Three-State Weather Model")
    print("=" * 60)
    
    # 상태 정하기
    states = ['Sunny', 'Cloudy', 'Rainy']
    
    # 옮김 행렬 정하기
    transition_matrix = [
        [0.7, 0.25, 0.05],  # 맑음에서
        [0.3, 0.4, 0.3],     # 흐림에서
        [0.1, 0.4, 0.5]      # 비에서
    ]
    
    # 마르코프 사슬 만들기
    mc = BasicMarkovChain(states, transition_matrix)
    
    print("\nTransition Matrix:")
    print("        Sunny  Cloudy  Rainy")
    for i, state in enumerate(states):
        print(f"{state:7s} {transition_matrix[i]}")
    
    # 여러 날 흉내내기
    print("\nSimulating 30 days starting from Sunny...")
    sequence = mc.simulate(n_steps=30, initial_state='Sunny')
    
    # 읽기 좋은 꼴로 찍기(줄마다 10일)
    for i in range(0, len(sequence), 10):
        day_sequence = sequence[i:i+10]
        print(f"Days {i:2d}-{min(i+9, len(sequence)-1):2d}: {day_sequence}")
    
    # 상태 잦기 세기
    print("\nState frequencies in simulation:")
    for state in states:
        count = sequence.count(state)
        frequency = count / len(sequence)
        print(f"  {state:7s}: {count:2d}/{len(sequence)} = {frequency:.3f}")


def visualize_state_sequence(sequence, title="Markov Chain State Sequence"):
    """
    시간에 따른 상태 늘어놓음 그려 보기.
    
    매개변수:
        sequence (list): 상태 이름의 목록
        title (str): 그림의 제목
    """
    # 서로 다른 상태를 얻어 정수에 맞추기
    unique_states = sorted(list(set(sequence)))
    state_to_int = {state: i for i, state in enumerate(unique_states)}
    
    # 늘어놓음을 정수로 바꾸기
    int_sequence = [state_to_int[state] for state in sequence]
    
    # 그림 만들기
    plt.figure(figsize=(12, 4))
    plt.step(range(len(int_sequence)), int_sequence, where='post', linewidth=2)
    plt.yticks(range(len(unique_states)), unique_states)
    plt.xlabel('Time Step', fontsize=12)
    plt.ylabel('State', fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    fig_path = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'markov_sequence.png')
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nVisualization saved to markov_sequence.png")


def main():
    """
    보기를 모두 돌리는 주 함수.
    """
    print("BASIC MARKOV CHAIN SIMULATIONS")
    print("================================\n")
    
    # 예제 실행
    example_two_state_chain()
    example_three_state_chain()
    
    # 그림 만들기
    print("\n" + "=" * 60)
    print("Creating Visualization")
    print("=" * 60)
    
    states = ['A', 'B', 'C']
    transition_matrix = [
        [0.5, 0.3, 0.2],
        [0.2, 0.6, 0.2],
        [0.3, 0.3, 0.4]
    ]
    
    mc = BasicMarkovChain(states, transition_matrix)
    sequence = mc.simulate(n_steps=50, initial_state='A')
    visualize_state_sequence(sequence, "Three-State Markov Chain Simulation")
    
    print("\n" + "=" * 60)
    print("Exercises for Students:")
    print("=" * 60)
    print("1. Modify the two-state chain to model a light bulb (Working/Broken)")
    print("2. Create a four-state chain for traffic lights")
    print("3. Experiment with different initial states - does it affect long-term behavior?")
    print("4. Try to create a chain that always returns to the starting state")


if __name__ == "__main__":
    main()```

## 2. 논의

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
기본 마르코프 사슬 구현을 확인하는 두루 갖춘 시험 함수를 적어라. 빈 입력, 원소 하나짜리 입력, 아주 큰 입력, 그리고 극단적인 값(0, 아주 큰 수)이 든 입력 같은 모서리 경우를 시험하여라.

??? success "연습문제 4 풀이"
    경계 조건을 두루 시험하는 함수를 만든다.
    ```python
    def test_basicmarkovchain():
        model = BasicMarkovChain(...)
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

## 정리하며

**다룬 것** — 기본 마르코프 사슬

시각화는 모델의 거동을 이해하고 학습 문제를 진단하는 데 중요한 역할을 한다.

고갱이 갈래는 `BasicMarkovChain`이며 앞의 연습문제 4개로 스스로 따져 볼 수 있다.
