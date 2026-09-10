# 무작위 걸음

random_walk.py (모듈 03) 무작위 걸음 흉내내기

확률 과정과 표집 방법은 확률적 기계 학습의 바탕 도구이다. 이 모듈은 마르코프 사슬의 개념을 보이며 수학 이론과 셈 구현 사이의 틈을 잇는다.

## 1. 코드

```python
"""
random_walk.py (단원 03)

무작위 걸음 흉내내기
=======================

Location: 06_markov_chain/01_fundamentals/
난이도: ⭐⭐ 기초
걸리는 시간: 3-4시간

학습 목표:
- 마르코프 사슬로서 무작위 걸음 이해하기
- 1차원과 2차원 무작위 걸음 구현하기
- 성질 살피기: 기댓값 자리, 흩어짐
- 경계 조건과 첫 지나감 시간 살피기

수학적 바탕:
무작위 걸음은 무작위 방향의 걸음으로 이루어진 길이다.
이는 다음을 만족하는 마르코프 사슬의 특별한 경우이다:
- 상태는 자리이다(정수나 좌표)
- 옮김은 지금 자리에만 달렸다

1차원 대칭 무작위 걸음에서:
- P(X_{n+1} = X_n + 1) = p
- P(X_{n+1} = X_n - 1) = 1-p
- p = 0.5이면 단순한 대칭 무작위 걸음이다

핵심 성질:
- 치우친 걸음에서는 E[X_n] = X_0 + n(2p-1)
- Var[X_n] = 4np(1-p)
- 대칭 걸음에서는 E[X_n] = X_0, Var[X_n] = n
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# ========================================================================
# 메인
# ========================================================================


class RandomWalk1D:
    """
    정수 위의 1차원 무작위 걸음.
    
    수학 모형:
    X_n = X_0 + Σ_{i=1}^n ξ_i
    여기서 ξ_i은 다음을 만족하는 독립 확률 변수이다:
    P(ξ_i = +1) = p
    P(ξ_i = -1) = 1-p
    """
    
    def __init__(self, p=0.5, initial_position=0):
        """
        무작위 걸음 첫값 잡기.
        
        매개변수:
            p (float): 오른쪽으로 걸을 확률([0,1] 안이어야 한다)
            initial_position (int): 시작 자리
        
        수학 메모:
        - p = 0.5: 대칭(치우치지 않은) 무작위 걸음
        - p > 0.5: 오른쪽으로 쏠린다
        - p < 0.5: 왼쪽으로 쏠린다
        """
        if not 0 <= p <= 1:
            raise ValueError("p must be in [0, 1]")
        
        self.p = p
        self.initial_position = initial_position
        self.current_position = initial_position
        self.history = [initial_position]
    
    def step(self):
        """
        무작위 걸음의 한 걸음 떼기.
        
        반환값:
            int: 새 자리
        
        수학 과정:
        확률 p으로 오른쪽(+1)으로 움직인다
        확률 1-p으로 왼쪽(-1)으로 움직인다
        """
        # 걸음 만들기: 확률 p으로 +1, 확률 1-p으로 -1
        if np.random.random() < self.p:
            step_size = 1
        else:
            step_size = -1
        
        self.current_position += step_size
        self.history.append(self.current_position)
        
        return self.current_position
    
    def simulate(self, n_steps, initial_position=None):
        """
        무작위 걸음 n걸음 흉내내기.
        
        매개변수:
            n_steps (int): 뗄 걸음 수
            initial_position (int): 시작 자리(되돌릴 때)
        
        반환값:
            list: 자리의 늘어놓음
        
        통계의 성질(대칭 걸음, p=0.5):
        - E[X_n] = X_0(기댓값 쏠림 없음)
        - Var[X_n] = n(흩어짐이 시간에 비례해 자란다)
        - Std[X_n] = √n(표준편차가 제곱근으로 자란다)
        """
        if initial_position is not None:
            self.current_position = initial_position
            self.history = [initial_position]
        
        for _ in range(n_steps):
            self.step()
        
        return self.history
    
    def expected_position(self, n_steps):
        """
        n걸음 뒤의 이론상 기댓값 자리 셈하기.
        
        매개변수:
            n_steps (int): 걸음 수
        
        반환값:
            float: 기댓값 자리 E[X_n]
        
        수학 공식:
        E[X_n] = X_0 + n(2p - 1)
        
        이끌어 내기:
        걸음마다 확률 p으로 +1, 확률 1-p으로 -1을 보탠다
        E[ξ_i] = (+1)×p + (-1)×(1-p) = 2p - 1
        E[X_n] = E[X_0 + Σξ_i] = X_0 + n×E[ξ_i] = X_0 + n(2p-1)
        """
        return self.initial_position + n_steps * (2 * self.p - 1)
    
    def variance(self, n_steps):
        """
        n걸음 뒤의 이론상 흩어짐 셈하기.
        
        매개변수:
            n_steps (int): 걸음 수
        
        반환값:
            float: 흩어짐 Var[X_n]
        
        수학 공식:
        Var[X_n] = 4np(1-p)
        
        대칭 걸음에서는(p=0.5) Var[X_n] = n
        
        이끌어 내기:
        Var[ξ_i] = E[ξ_i²] - (E[ξ_i])²
        E[ξ_i²] = (+1)²×p + (-1)²×(1-p) = 1
        Var[ξ_i] = 1 - (2p-1)² = 4p(1-p)
        Var[X_n] = n×Var[ξ_i] = 4np(1-p)
        """
        return 4 * n_steps * self.p * (1 - self.p)
    
    def first_passage_time(self, target_position, max_steps=10000):
        """
        걸음이 target_position에 처음 닿는 때 찾기.
        
        매개변수:
            target_position (int): 닿을 과녁
            max_steps (int): 시도할 최대 걸음 수
        
        반환값:
            int or None: 첫 지나감 시간, 닿지 못하면 None
        
        수학 메모:
        첫 지나감 시간 T = min{n ≥ 0 : X_n = target}
        대칭 걸음에서는 어떤 과녁에 대해서도 E[T]이 유한하다.
        """
        self.current_position = self.initial_position
        
        for step in range(max_steps):
            if self.current_position == target_position:
                return step
            self.step()
        
        return None  # max_steps 안에 과녁에 닿지 못함


class RandomWalk2D:
    """
    정수 격자 Z² 위의 2차원 무작위 걸음.
    
    수학 모형:
    (X_n, Y_n) = (X_0, Y_0) + Σ_{i=1}^n (ξ_i, η_i)
    
    대칭 걸음에서:
    P(오른쪽) = P(왼쪽) = P(위) = P(아래) = 1/4
    """
    
    def __init__(self, initial_position=(0, 0)):
        """
        2차원 무작위 걸음 첫값 잡기.
        
        매개변수:
            initial_position (tuple): 시작 (x, y) 좌표
        """
        self.initial_position = np.array(initial_position)
        self.current_position = np.array(initial_position)
        self.history = [self.current_position.copy()]
    
    def step(self):
        """
        무작위 사방 가운데 한 방향으로 한 걸음 떼기.
        
        반환값:
            np.ndarray: 새 자리 (x, y)
        
        수학 과정:
        네 방향 가운데 하나를 같은 확률로 고른다:
        - 오른쪽: (+1, 0)
        - 왼쪽:  (-1, 0)
        - 위:    (0, +1)
        - 아래:  (0, -1)
        """
        # 갈 수 있는 네 방향
        directions = np.array([
            [1, 0],   # 오른쪽
            [-1, 0],  # 왼쪽
            [0, 1],   # 오름
            [0, -1]   # 내림
        ])
        
        # 무작위 방향 고르기
        direction = directions[np.random.randint(0, 4)]
        
        self.current_position = self.current_position + direction
        self.history.append(self.current_position.copy())
        
        return self.current_position
    
    def simulate(self, n_steps):
        """
        2차원 무작위 걸음 n걸음 흉내내기.
        
        매개변수:
            n_steps (int): 걸음 수
        
        반환값:
            list: (x, y) 자리의 목록
        
        통계의 성질:
        - n이 크면 E[출발점에서의 거리] ≈ √(2n/π)
        - (2차원에서) n → ∞이면 출발점으로 돌아올 확률 → 0
        """
        self.current_position = self.initial_position.copy()
        self.history = [self.current_position.copy()]
        
        for _ in range(n_steps):
            self.step()
        
        return self.history
    
    def distance_from_origin(self):
        """
        출발점에서의 유클리드 거리 셈하기.
        
        반환값:
            float: ||X_n|| = √(x² + y²)
        """
        return np.linalg.norm(self.current_position - self.initial_position)


def example_symmetric_walk():
    """
    보기 1: 대칭 무작위 걸음(p = 0.5).
    
    기댓값 쏠림이 0이고 흩어짐이 자람을 보인다.
    """
    print("=" * 70)
    print("Example 1: Symmetric Random Walk (p = 0.5)")
    print("=" * 70)
    
    # 대칭 걸음 만들기
    walk = RandomWalk1D(p=0.5, initial_position=0)
    
    # 한 번의 실현
    path = walk.simulate(n_steps=100)
    
    print(f"\nSingle path of 100 steps:")
    print(f"  Final position: {path[-1]}")
    print(f"  Maximum position: {max(path)}")
    print(f"  Minimum position: {min(path)}")
    
    # 이론과 경험의 통계량 견줌
    n = 100
    print(f"\nTheoretical properties after {n} steps:")
    print(f"  Expected position: {walk.expected_position(n):.2f}")
    print(f"  Variance: {walk.variance(n):.2f}")
    print(f"  Standard deviation: {np.sqrt(walk.variance(n)):.2f}")
    
    # 확인하려고 흉내내기를 많이 돌리기
    n_simulations = 10000
    final_positions = []
    
    for _ in range(n_simulations):
        walk = RandomWalk1D(p=0.5, initial_position=0)
        path = walk.simulate(100)
        final_positions.append(path[-1])
    
    print(f"\nEmpirical statistics ({n_simulations} simulations):")
    print(f"  Mean final position: {np.mean(final_positions):.2f}")
    print(f"  Variance: {np.var(final_positions):.2f}")
    print(f"  Standard deviation: {np.std(final_positions):.2f}")


def example_biased_walk():
    """
    보기 2: 치우친 무작위 걸음(p ≠ 0.5).
    
    기댓값 방향으로 쏠림을 보인다.
    """
    print("\n" + "=" * 70)
    print("Example 2: Biased Random Walk (p = 0.6)")
    print("=" * 70)
    
    # 치우친 걸음 만들기
    p = 0.6
    walk = RandomWalk1D(p=p, initial_position=0)
    
    n = 1000
    path = walk.simulate(n_steps=n)
    
    print(f"\nSimulation of {n} steps with p = {p}:")
    print(f"  Final position: {path[-1]}")
    
    # 이론의 기댓값
    print(f"\nTheoretical properties:")
    print(f"  Expected drift per step: {2*p - 1:.2f}")
    print(f"  Expected position after {n} steps: {walk.expected_position(n):.2f}")
    print(f"  Variance: {walk.variance(n):.2f}")
    
    # p = 0.3, 0.5, 0.7 견주기
    print("\n" + "-" * 70)
    print("Comparing different values of p:")
    print(f"{'p':<8} {'E[X_1000]':<15} {'Var[X_1000]':<15}")
    
    for p_val in [0.3, 0.5, 0.7]:
        walk = RandomWalk1D(p=p_val)
        exp_pos = walk.expected_position(1000)
        var_pos = walk.variance(1000)
        print(f"{p_val:<8.1f} {exp_pos:<15.2f} {var_pos:<15.2f}")


def example_first_passage():
    """
    보기 3: 첫 지나감 시간 살피기.
    
    목표 자리에 닿는 데 얼마나 걸리나?
    """
    print("\n" + "=" * 70)
    print("Example 3: First Passage Time")
    print("=" * 70)
    
    target = 10
    n_simulations = 1000
    
    print(f"\nFinding first passage times to position {target}")
    print("(symmetric walk, p = 0.5)")
    
    passage_times = []
    
    for _ in range(n_simulations):
        walk = RandomWalk1D(p=0.5, initial_position=0)
        fpt = walk.first_passage_time(target, max_steps=10000)
        if fpt is not None:
            passage_times.append(fpt)
    
    if passage_times:
        print(f"\nResults from {len(passage_times)} successful walks:")
        print(f"  Mean first passage time: {np.mean(passage_times):.2f} steps")
        print(f"  Median: {np.median(passage_times):.2f} steps")
        print(f"  Min: {min(passage_times)} steps")
        print(f"  Max: {max(passage_times)} steps")
        print(f"  Did not reach in {n_simulations - len(passage_times)} cases")


def example_2d_walk():
    """
    보기 4: 2차원 무작위 걸음.
    
    2차원 공간의 무작위 움직임을 살펴본다.
    """
    print("\n" + "=" * 70)
    print("Example 4: Two-Dimensional Random Walk")
    print("=" * 70)
    
    # 2차원 걸음 하나
    walk = RandomWalk2D(initial_position=(0, 0))
    path = walk.simulate(n_steps=1000)
    
    # x과 y 좌표 뽑아내기
    x_coords = [pos[0] for pos in path]
    y_coords = [pos[1] for pos in path]
    
    print(f"\nSimulation of 1000 steps:")
    print(f"  Final position: ({x_coords[-1]}, {y_coords[-1]})")
    print(f"  Final distance from origin: {walk.distance_from_origin():.2f}")
    print(f"  Max |x|: {max(abs(x) for x in x_coords)}")
    print(f"  Max |y|: {max(abs(y) for y in y_coords)}")
    
    # 거리 분포 살피기
    n_simulations = 1000
    final_distances = []
    
    for _ in range(n_simulations):
        walk = RandomWalk2D()
        walk.simulate(1000)
        final_distances.append(walk.distance_from_origin())
    
    print(f"\nDistance statistics ({n_simulations} simulations):")
    print(f"  Mean distance: {np.mean(final_distances):.2f}")
    print(f"  Theoretical approximation: {np.sqrt(2 * 1000 / np.pi):.2f}")


def visualize_random_walks():
    """
    무작위 걸음의 그림 만들기.
    """
    print("\n" + "=" * 70)
    print("Creating Visualizations")
    print("=" * 70)
    
    # 1차원 걸음 견주기
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 작은 그림 1: 대칭 걸음 여럿
    ax = axes[0, 0]
    for i in range(10):
        walk = RandomWalk1D(p=0.5)
        path = walk.simulate(200)
        ax.plot(path, alpha=0.6, linewidth=1.5)
    
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax.set_xlabel('Step', fontsize=11)
    ax.set_ylabel('Position', fontsize=11)
    ax.set_title('10 Symmetric Random Walks (p=0.5)', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # 작은 그림 2: 치우친 걸음
    ax = axes[0, 1]
    for p_val, color in [(0.3, 'blue'), (0.5, 'green'), (0.7, 'red')]:
        walk = RandomWalk1D(p=p_val)
        path = walk.simulate(200)
        ax.plot(path, color=color, alpha=0.8, linewidth=2, label=f'p={p_val}')
    
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax.set_xlabel('Step', fontsize=11)
    ax.set_ylabel('Position', fontsize=11)
    ax.set_title('Biased Random Walks', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 작은 그림 3: 2차원 걸음
    ax = axes[1, 0]
    walk_2d = RandomWalk2D()
    path_2d = walk_2d.simulate(500)
    
    x_coords = [pos[0] for pos in path_2d]
    y_coords = [pos[1] for pos in path_2d]
    
    # 시간에 따라 색칠하기
    colors = plt.cm.viridis(np.linspace(0, 1, len(path_2d)))
    for i in range(len(path_2d) - 1):
        ax.plot(x_coords[i:i+2], y_coords[i:i+2], color=colors[i], linewidth=1.5)
    
    ax.plot(0, 0, 'go', markersize=10, label='Start')
    ax.plot(x_coords[-1], y_coords[-1], 'ro', markersize=10, label='End')
    ax.set_xlabel('X Position', fontsize=11)
    ax.set_ylabel('Y Position', fontsize=11)
    ax.set_title('2D Random Walk (500 steps)', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    # 작은 그림 4: 마지막 자리의 분포
    ax = axes[1, 1]
    walk = RandomWalk1D(p=0.5)
    final_positions = []
    for _ in range(5000):
        walk = RandomWalk1D(p=0.5)
        path = walk.simulate(100)
        final_positions.append(path[-1])
    
    ax.hist(final_positions, bins=50, density=True, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Final Position', fontsize=11)
    ax.set_ylabel('Probability Density', fontsize=11)
    ax.set_title('Distribution of Final Positions (100 steps, 5000 simulations)', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # 이론의 정규 분포 더하기
    mu = 0
    sigma = np.sqrt(100)
    x = np.linspace(-40, 40, 100)
    y = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    ax.plot(x, y, 'r-', linewidth=2, label='Normal(0, 100)')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/random_walks.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Random walk visualizations saved to random_walks.png")


def main():
    """
    무작위 걸음 보기 모두 돌리기.
    """
    print("RANDOM WALK SIMULATIONS")
    print("=======================\n")
    
    # 예제 실행
    example_symmetric_walk()
    example_biased_walk()
    example_first_passage()
    example_2d_walk()
    
    # 시각화 만들기
    visualize_random_walks()
    
    print("\n" + "=" * 70)
    print("Key Properties of Random Walks:")
    print("=" * 70)
    print("1. Symmetric walk (p=0.5): E[X_n] = X_0, Var[X_n] = n")
    print("2. Biased walk: E[X_n] = X_0 + n(2p-1)")
    print("3. Standard deviation grows as √n")
    print("4. In 1D: symmetric walk is recurrent (returns to origin infinitely often)")
    print("5. In 2D: symmetric walk is recurrent")
    print("6. In 3D: symmetric walk is transient (may never return)")


if __name__ == "__main__":
    main()
```

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
무작위 걸음 구현을 확인하는 두루 갖춘 시험 함수를 적어라. 빈 입력, 원소 하나짜리 입력, 아주 큰 입력, 그리고 극단적인 값(0, 아주 큰 수)이 든 입력 같은 모서리 경우를 시험하여라.

??? success "연습문제 4 풀이"
    경계 조건을 두루 시험하는 함수를 만든다.
    ```python
    def test_randomwalk1d():
        model = RandomWalk1D(...)
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

**다룬 것** — 무작위 걸음

시각화는 모델의 거동을 이해하고 학습 문제를 진단하는 데 중요한 역할을 한다.

고갱이 갈래는 `RandomWalk1D`, `RandomWalk2D`이며 앞의 연습문제 4개로 스스로 따져 볼 수 있다.
