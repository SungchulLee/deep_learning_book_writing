# 기본 마르코프 사슬

마르코프 사슬은 현재 상태를 조건으로 두면 다음 상태가 그 이전 이력에 추가로 의존하지 않는 확률 과정이다. 이 예제는 유한한 상태 공간과 시간에 따라 변하지 않는 전이행렬을 사용해 상태 경로를 생성한다. 한 경로의 경험적 빈도와 특정 시점의 상태 확률은 다른 양이다.

## 1. 코드

```python
"""
basic_markov_chain.py (단원 01)

마르코프 사슬 들어가기
==============================

Location: 06_markov_chain/01_fundamentals/
난이도: ⭐ 입문
예상 시간: 2~3시간

학습 목표:
- 마르코프 성질 이해하기
- 단순한 이산 시간 마르코프 사슬 구현하기
- 상태 전이 시뮬레이션
- 상태 시퀀스 시각화하기

수학적 배경:
마르코프 사슬은 다음을 만족하는 확률 변수의 시퀀스 X_0, X_1, X_2, ...이다:
P(X_{n+1} = j | X_n = i, X_{n-1} = k, ..., X_0 = m) = P(X_{n+1} = j | X_n = i)

이를 마르코프 성질이라 한다. 곧 미래는 과거가 아니라 현재에만 의존한다.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# 무작위로 뽑는 값이 아래에 나온다. 시드를 고정해야 이 쪽에 실린
# 수가 다시 나온다 — 고정하지 않으면 돌릴 때마다 다른 수가 찍힌다
np.random.seed(0)

# ========================================================================
# 메인
# ========================================================================


class BasicMarkovChain:
    """
    이산 시간 마르코프 사슬의 단순한 구현.
    
    속성:
        states (list): 상태 이름의 목록
        transition_matrix (np.ndarray): 전이확률 행렬
        current_state (int): 지금 상태의 번호
    """
    
    def __init__(self, states, transition_matrix):
        """
        마르코프 사슬 초기화.
        
        파라메터:
            states (list): 상태의 이름(이를테면 ['A', 'B', 'C'])
            transition_matrix (np.ndarray): 전이확률 행렬 P
                                          여기서 P[i][j] = P(X_{n+1}=j | X_n=i)
        
        수학 메모:
        - transition_matrix의 행마다 합이 1이어야 한다
        - 성분이 모두 음이 아니어야 한다
        """
        self.states = states
        self.transition_matrix = np.array(transition_matrix)
        self.n_states = len(states)
        
        # 전이 행렬 확인하기
        self._validate_transition_matrix()
        
        # 무작위 상태에서 시작하기
        self.current_state = np.random.randint(0, self.n_states)
    
    def _validate_transition_matrix(self):
        """
        전이 행렬이 확률 행렬인지 확인하기.
        
        다음이면 행렬이 확률 행렬이다:
        1. 성분이 모두 음이 아니다
        2. 행마다 합이 1이다(확률 분포)
        """
        # 차원 확인하기
        if self.transition_matrix.shape != (self.n_states, self.n_states):
            raise ValueError(f"Transition matrix must be {self.n_states}x{self.n_states}")
        
        # 음이 아닌지 확인하기
        if np.any(self.transition_matrix < 0):
            raise ValueError("All transition probabilities must be non-negative")
        
        # 행의 합 확인하기(각각 1이 되어야 함)
        row_sums = np.sum(self.transition_matrix, axis=1)
        if not np.allclose(row_sums, 1.0):
            raise ValueError("Each row of transition matrix must sum to 1")
    
    def step(self):
        """
        마르코프 사슬의 한 스텝 밟기.
        
        반환값:
            str: 새 상태의 이름
        
        수학 과정:
        지금 상태가 i일 때 확률 P[i][j]으로 다음 상태 j 고르기
        이는 범주형 분포에서 샘플링하는 것과 같다
        전이 행렬의 i번째 행이 정하는.
        """
        # 지금 상태의 전이확률 얻기
        probabilities = self.transition_matrix[self.current_state]
        
        # 이 확률에 따라 다음 상태 샘플링
        # np.random.choice는 확률이 정하는 이산 분포를 쓴다
        self.current_state = np.random.choice(self.n_states, p=probabilities)
        
        return self.states[self.current_state]
    
    def simulate(self, n_steps, initial_state=None):
        """
        마르코프 사슬을 n스텝 시뮬레이션.
        
        파라메터:
            n_steps (int): 시뮬레이션할 스텝 수
            initial_state (int or str): 시작 상태(번호나 이름)
        
        반환값:
            list: 들른 상태의 시퀀스
        
        수학 메모:
        이는 확률 과정 {X_n}의 한 실현을 만든다
        """
        # 주어졌으면 초기 상태 설정하기
        if initial_state is not None:
            if isinstance(initial_state, str):
                self.current_state = self.states.index(initial_state)
            else:
                self.current_state = initial_state
        
        # 상태 시퀀스 기록하기
        state_sequence = [self.states[self.current_state]]
        
        # n스텝 시뮬레이션
        for _ in range(n_steps):
            state_sequence.append(self.step())
        
        return state_sequence
    
    def get_state_distribution(self, n_steps, n_simulations=10000):
        """
        몬테카를로 시뮬레이션으로 n스텝 뒤의 상태 분포 추정하기.
        
        파라메터:
            n_steps (int): 뗄 스텝 수
            n_simulations (int): 돌릴 시뮬레이션의 횟수
        
        반환값:
            np.ndarray: 상태에 걸쳐 추정한 확률 분포
        
        수학적 배경:
        π_0이 초기분포이고 P이 전이 행렬이면,
        n스텝 뒤의 분포는 π_n = π_0 * P^n이다
        
        시뮬레이션을 많이 돌리고 세어 이를 추정한다.
        """
        # 모든 시뮬레이션의 마지막 상태 세기
        final_state_counts = np.zeros(self.n_states)
        
        for _ in range(n_simulations):
            # 시뮬레이션 한 번 돌리기
            sequence = self.simulate(n_steps)
            final_state = sequence[-1]
            
            # 마지막 상태 세기
            final_state_index = self.states.index(final_state)
            final_state_counts[final_state_index] += 1
        
        # 센 횟수를 확률로 바꾸기
        estimated_distribution = final_state_counts / n_simulations
        
        return estimated_distribution


def example_two_state_chain():
    """
    보기 1: 단순한 두 상태 마르코프 사슬
    
    상태: {0, 1} 또는 {Off, On}
    전이 행렬:
        에서\로  Off   On
        Off      0.7   0.3
        On       0.4   0.6
    
    해석:
    - 지금 Off이면 70%는 Off 그대로, 30%은 On으로
    - 지금 On이면 40%은 Off로, 60%은 On 그대로
    """
    print("=" * 60)
    print("Example 1: Two-State Markov Chain (Off/On)")
    print("=" * 60)
    
    # 상태 정하기
    states = ['Off', 'On']
    
    # 전이 행렬 정하기
    # P[i][j] = 상태 i에서 상태 j로 갈 확률
    transition_matrix = [
        [0.7, 0.3],  # Off에서: 70% Off 그대로, 30% On으로
        [0.4, 0.6]   # On에서: 40% Off로, 60% On 그대로
    ]
    
    # 마르코프 사슬 만들기
    mc = BasicMarkovChain(states, transition_matrix)
    
    print("\nTransition Matrix:")
    print(transition_matrix)
    print("\nSimulating 20 steps...")
    
    # 'Off'에서 시작해 시뮬레이션
    sequence = mc.simulate(n_steps=20, initial_state='Off')
    print(f"\nState sequence: {sequence}")
    
    # 오래 뒤의 분포 추정하기
    print("\nEstimating state distribution after 100 steps...")
    distribution = mc.get_state_distribution(n_steps=100)
    
    print(f"Estimated probabilities:")
    for state, prob in zip(states, distribution):
        print(f"  P({state}) = {prob:.4f}")


def example_three_state_chain():
    """
    보기 2: 세 상태 날씨 모델
    
    상태: {맑음, 흐림, 비}
    
    이는 간추린 날씨 전이를 모델링한다:
    - 맑음은 맑음 그대로이거나 흐림이 되는 경향이 있다
    - 흐림은 어느 쪽으로든 갈 수 있다
    - 비는 흐림이 되거나 비 그대로인 경향이 있다
    """
    print("\n" + "=" * 60)
    print("Example 2: Three-State Weather Model")
    print("=" * 60)
    
    # 상태 정하기
    states = ['Sunny', 'Cloudy', 'Rainy']
    
    # 전이 행렬 정하기
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
    
    # 여러 날 시뮬레이션
    print("\nSimulating 30 days starting from Sunny...")
    sequence = mc.simulate(n_steps=30, initial_state='Sunny')
    
    # 읽기 좋은 형태로 출력하기(줄마다 10일)
    for i in range(0, len(sequence), 10):
        day_sequence = sequence[i:i+10]
        print(f"Days {i:2d}-{min(i+9, len(sequence)-1):2d}: {day_sequence}")
    
    # 상태 빈도 세기
    print("\nState frequencies in simulation:")
    for state in states:
        count = sequence.count(state)
        frequency = count / len(sequence)
        print(f"  {state:7s}: {count:2d}/{len(sequence)} = {frequency:.3f}")


def visualize_state_sequence(sequence, title="Markov Chain State Sequence"):
    """
    시간에 따른 상태 시퀀스 시각화하기.
    
    파라메터:
        sequence (list): 상태 이름의 목록
        title (str): 그림의 제목
    """
    # 서로 다른 상태를 얻어 정수에 맞추기
    unique_states = sorted(list(set(sequence)))
    state_to_int = {state: i for i, state in enumerate(unique_states)}
    
    # 시퀀스를 정수로 바꾸기
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
    main()
```

**출력:**

```
BASIC MARKOV CHAIN SIMULATIONS
================================

============================================================
Example 1: Two-State Markov Chain (Off/On)
============================================================

Transition Matrix:
[[0.7, 0.3], [0.4, 0.6]]

Simulating 20 steps...

State sequence: ['Off', 'Off', 'On', 'On', 'On', 'On', 'Off', 'Off', 'Off', 'Off', 'Off', 'On', 'On', 'Off', 'On', 'Off', 'Off', 'Off', 'On', 'Off', 'On']

Estimating state distribution after 100 steps...
Estimated probabilities:
  P(Off) = 0.5679
  P(On) = 0.4321

============================================================
Example 2: Three-State Weather Model
============================================================

Transition Matrix:
        Sunny  Cloudy  Rainy
Sunny   [0.7, 0.25, 0.05]
Cloudy  [0.3, 0.4, 0.3]
Rainy   [0.1, 0.4, 0.5]

Simulating 30 days starting from Sunny...
Days  0- 9: ['Sunny', 'Cloudy', 'Cloudy', 'Cloudy', 'Cloudy', 'Rainy', 'Rainy', 'Cloudy', 'Cloudy', 'Sunny']
Days 10-19: ['Sunny', 'Cloudy', 'Sunny', 'Sunny', 'Sunny', 'Sunny', 'Sunny', 'Sunny', 'Sunny', 'Cloudy']
Days 20-29: ['Sunny', 'Sunny', 'Cloudy', 'Rainy', 'Rainy', 'Rainy', 'Cloudy', 'Rainy', 'Rainy', 'Rainy']
Days 30-30: ['Rainy']

State frequencies in simulation:
  Sunny  : 12/31 = 0.387
  Cloudy : 10/31 = 0.323
  Rainy  :  9/31 = 0.290

============================================================
Creating Visualization
============================================================

Visualization saved to markov_sequence.png

============================================================
Exercises for Students:
============================================================
1. Modify the two-state chain to model a light bulb (Working/Broken)
2. Create a four-state chain for traffic lights
3. Experiment with different initial states - does it affect long-term behavior?
4. Try to create a chain that always returns to the starting state
```

## 2. 논의

### 상태와 전이

시간 동질적인 사슬의 전이행렬은 다음과 같다.

$$P_{ij}=P(X_{t+1}=j\mid X_t=i)$$

각 행은 현재 상태에서 다음 상태로의 확률분포이므로 음이 아니고 합이 1이다. 마르코프 성질은 상태를 독립적으로 생성한다는 뜻이 아니다. 이웃한 시점의 상태는 일반적으로 상관되어 있다.

코드의 step은 현재 행에서 다음 상태를 뽑는다. simulate(n_steps)는 초기 상태를 포함하므로 길이가 n_steps+1인 목록을 반환한다. n_steps=0도 초기 상태 하나를 반환하며 initial_state를 생략하면 이전 실행의 마지막 상태에서 계속한다.

### 경로 빈도와 시점별 분포

행 벡터로 쓴 초기분포 $p_0$가 주어지면 $n$스텝 뒤의 분포는 $p_0P^n$이다. 같은 초기조건으로 독립 경로를 반복 생성해 마지막 상태를 세면 이 분포를 추정할 수 있다. 반면 get_state_distribution은 반복마다 초기화하지 않고 이전 경로의 끝에서 이어 가므로, 같은 초기분포에서 독립으로 반복한 실험이 아니다. $n$이 크면 정상분포의 근사로 볼 수 있는 조건이 있지만, 유한한 $n$스텝 전이분포의 일반적인 추정량으로 읽지는 않는다.

유한 기약 사슬은 유일한 정상분포를 가지며 경로의 장기 점유율은 그 분포와 연결된다. 모든 초기분포에서 시점별 분포가 정상분포로 수렴하려면 비주기성도 필요하다. 여러 닫힌 클래스가 있으면 초기 상태가 장기 거동에 영향을 줄 수 있다.

### 입력과 예제의 범위

상태 이름은 고유하게 정의하고 빈 상태 목록, 유효하지 않은 초기 인덱스·이름, 음수나 정수가 아닌 단계 수를 허용하지 않는다. 시뮬레이션 반복 수는 양수여야 한다. 행렬 검사와 별도로 이런 인자 조건을 확인해야 한다. 날씨·스위치 예제의 전이확률은 가정한 숫자이며 실제 데이터에서 트레인한 모델은 아니다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
마르코프 성질과 독립 표본 생성의 차이, 전이행렬의 행 의미를 설명하라.

</div>

??? success "풀이"

    마르코프 성질은 다음 상태가 현재 상태를 알면 더 과거에 추가로 의존하지 않는다는 뜻이다. 현재와 다음 상태가 독립이라는 뜻은 아니다. P의 i번째 행은 현재 상태 i에서 다음 상태 j로 갈 조건부 확률들이다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
simulate와 get_state_distribution에 필요한 입력·반환 규약을 정리하라.

</div>

??? success "풀이"

    상태 수와 같은 크기의 유효한 확률행렬, 고유 상태명, 유효한 초기 상태, 음이 아닌 정수 단계 수와 양의 반복 수가 필요하다. simulate는 초기 상태를 포함한 n+1개 값을 반환한다. 반복 시 초기화 여부를 명시하지 않으면 시점별 분포의 해석이 달라진다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
두 상태가 번갈아 바뀌는 사슬에서 정상분포·시점별 분포·장기 점유율을 비교하라.

</div>

??? success "풀이"

    P=[[0,1],[1,0]]의 정상분포는 (1/2,1/2)다. 한 상태에서 시작한 시점별 분포는 번갈아 움직여 수렴하지 않지만 한 경로의 장기 점유율은 각 상태 1/2로 간다. 정상분포 존재·분포 수렴·시간 평균은 서로 다르다.



---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
아래의 뉴럴 네트워크용 시험 조각이 BasicMarkovChain에 적합한지 비판하고, 필요한 점검을 설명하라.

</div>

??? success "풀이"

    클래스는 호출 가능한 뉴럴 네트워크나 parameters·backward 인터페이스를 제공하지 않는다. 따라서 이 조각은 실행 가능한 마르코프 사슬 검증이 아니다. 실제 점검은 전이행렬의 유효성, n=0에서 초기 상태 하나, 경로 길이 n+1, 허용된 상태만의 출력과 초기화 규약을 다뤄야 한다. 확률 빈도는 유한 표본의 변동을 고려해 평가한다.

    ```python
    def test_basicmarkovchain():
        model = BasicMarkovChain(...)
        # 보통의 입력
        assert model(normal_input).shape == expected_shape
        # 원소가 하나인 배치
        assert model(single_input).shape == (1, ...)
        # 큰 값 (오버플로를 확인한다)
        out = model(torch.ones(...) * 1000)
        assert torch.isfinite(out).all()
        # 그래디언트의 흐름
        out = model(normal_input)
        out.sum().backward()
        for p in model.parameters():
            assert p.grad is not None
    ```


## 정리하며

유한 마르코프 사슬은 현재 상태별 전이확률로 경로를 생성한다. $n$스텝 시점의 분포와 한 경로의 장기 빈도를 구분하고, 기약성·비주기성·초기화 규약에 따라 결과를 해석한다. 뉴럴 네트워크용 일반 시험 조각은 이 클래스의 인터페이스에 맞지 않는다.
