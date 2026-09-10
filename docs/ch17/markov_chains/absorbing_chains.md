# 흡수 사슬

absorbing_chains.py (모듈 05) 흡수 마르코프 사슬

확률 과정과 표집 방법은 확률적 기계 학습의 바탕 도구이다. 이 모듈은 마르코프 사슬의 개념을 보이며 수학 이론과 셈 구현 사이의 틈을 잇는다.

## 1. 코드

```python
"""
absorbing_chains.py (단원 05)

흡수 마르코프 사슬
========================

Location: 06_markov_chain/02_analysis_methods/
난이도: ⭐⭐⭐ 중급
걸리는 시간: 3-4시간

학습 목표:
- 흡수 상태와 흡수 사슬 이해하기
- 흡수 확률 셈하기
- 흡수까지의 기댓값 시간 셈하기
- 바탕 행렬 살피기

수학적 바탕:
흡수 상태는 한 번 들어가면 떠날 수 없는 상태이다.
P[i][i] = 1이면 상태 i은 흡수 상태이다.

흡수 사슬은 다음을 갖는다:
1. 흡수 상태가 적어도 하나 있다
2. 모든 상태에서 흡수 상태에 닿을 수 있다

P의 정준 꼴:
    ┌       ┐
P = │ Q  R  │  여기서:
    │ 0  I  │
    └       ┘
- Q: 지나가는 상태 사이의 옮김
- R: 지나가는 상태에서 흡수 상태로의 옮김
- I: 항등 행렬(흡수 상태)
- 0: 영행렬(흡수 상태를 떠날 수 없다)

핵심 양:
- 바탕 행렬: N = (I - Q)^{-1}
- N[i][j] = i에서 시작해 지나가는 상태 j에 들르는 기댓값 횟수
- 흡수까지의 기댓값 걸음 수: t = N × 1(1로 채운 열벡터)
- 흡수 확률: B = N × R
"""

import numpy as np
import matplotlib.pyplot as plt

# ========================================================================
# 메인
# ========================================================================


class AbsorbingMarkovChain:
    """
    흡수 마르코프 사슬을 살피는 도구.
    """
    
    def __init__(self, transition_matrix, state_names=None):
        """
        흡수 사슬 첫값 잡기.
        
        매개변수:
            transition_matrix (np.ndarray): 옮김 행렬
            state_names (list): 상태 이름(없어도 된다)
        """
        self.P = np.array(transition_matrix, dtype=float)
        self.n_states = self.P.shape[0]
        
        if state_names is None:
            self.state_names = [f"State {i}" for i in range(self.n_states)]
        else:
            self.state_names = state_names
        
        # 흡수 상태와 지나가는 상태 가려내기
        self._identify_states()
        
        # 필요하면 차례 바꾸기
        self._reorder_canonical()
    
    def _identify_states(self):
        """
        어느 상태가 흡수 상태인지 가려내기.
        
        P[i][i] = 1이고 나머지 P[i][j] = 0이면 상태 i은 흡수 상태이다
        """
        self.absorbing_indices = []
        self.transient_indices = []
        
        for i in range(self.n_states):
            if np.isclose(self.P[i, i], 1.0) and np.allclose(self.P[i, :i], 0.0) and np.allclose(self.P[i, i+1:], 0.0):
                self.absorbing_indices.append(i)
            else:
                self.transient_indices.append(i)
        
        self.n_transient = len(self.transient_indices)
        self.n_absorbing = len(self.absorbing_indices)
    
    def _reorder_canonical(self):
        """
        상태를 정준 꼴로 다시 늘어놓기: 지나가는 상태 먼저, 그다음 흡수 상태.
        
        Q, R 행렬을 만들고 바탕 행렬 N 셈하기.
        """
        if self.n_absorbing == 0:
            raise ValueError("No absorbing states found")
        
        # 상태 차례 바꾸기
        reordered_indices = self.transient_indices + self.absorbing_indices
        
        # 옮김 행렬의 차례 바꾸기
        P_canonical = self.P[np.ix_(reordered_indices, reordered_indices)]
        
        # Q과 R 뽑아내기
        self.Q = P_canonical[:self.n_transient, :self.n_transient]
        self.R = P_canonical[:self.n_transient, self.n_transient:]
        
        # 차례 바꾼 이름 저장
        self.transient_names = [self.state_names[i] for i in self.transient_indices]
        self.absorbing_names = [self.state_names[i] for i in self.absorbing_indices]
    
    def fundamental_matrix(self):
        """
        바탕 행렬 N = (I - Q)^{-1} 셈하기.
        
        반환값:
            np.ndarray: 바탕 행렬 N
        
        수학으로 풀이하기:
        N[i][j] = 지나가는 상태 j에 머무는 기댓값 횟수,
                  지나가는 상태 i에서 시작해 흡수되기 전까지
        
        이끌어 내기:
        M[i][j] = E[i에서 시작해 j에 들르는 횟수]이라 하자
        M[i][j] = δ_{ij} + Σ_k P[i][k] × M[k][j]
        행렬로 쓰면: M = I + Q × M
        풀면: M = (I - Q)^{-1} = N
        """
        I = np.eye(self.n_transient)
        self.N = np.linalg.inv(I - self.Q)
        return self.N
    
    def expected_steps_to_absorption(self):
        """
        상태마다 흡수까지의 기댓값 걸음 수 셈하기.
        
        반환값:
            dict: 지나가는 상태마다의 기댓값 걸음 수
        
        수학 공식:
        t = N × 1(여기서 1은 1로 채운 열벡터)
        
        해석:
        t[i] = 지나가는 상태 i에서 시작해 흡수까지의 기댓값 걸음 수
        """
        if not hasattr(self, 'N'):
            self.fundamental_matrix()
        
        # N에 1로 채운 열벡터 곱하기
        ones = np.ones((self.n_transient, 1))
        t = self.N @ ones
        
        # 사전으로 돌려주기
        result = {}
        for i, name in enumerate(self.transient_names):
            result[name] = t[i, 0]
        
        return result
    
    def absorption_probabilities(self):
        """
        흡수 상태마다 흡수될 확률 셈하기.
        
        반환값:
            dict: 지나가는 상태마다 흡수 상태별로 흡수될 확률
        
        수학 공식:
        B = N × R
        
        해석:
        B[i][j] = 흡수 상태 j으로 흡수될 확률,
                  지나가는 상태 i에서 시작해
        """
        if not hasattr(self, 'N'):
            self.fundamental_matrix()
        
        self.B = self.N @ self.R
        
        # 겹친 사전으로 돌려주기
        result = {}
        for i, trans_name in enumerate(self.transient_names):
            result[trans_name] = {}
            for j, abs_name in enumerate(self.absorbing_names):
                result[trans_name][abs_name] = self.B[i, j]
        
        return result
    
    def variance_steps_to_absorption(self):
        """
        흡수까지 걸음 수의 흩어짐 셈하기.
        
        반환값:
            dict: 지나가는 상태마다의 흩어짐
        
        수학 공식:
        Var[T_i] = (2N - I) × t - t²
        여기서 t은 기댓값 걸음 수 벡터이다
        """
        if not hasattr(self, 'N'):
            self.fundamental_matrix()
        
        ones = np.ones((self.n_transient, 1))
        t = self.N @ ones
        
        I = np.eye(self.n_transient)
        variance_vec = (2 * self.N - I) @ t - t**2
        
        result = {}
        for i, name in enumerate(self.transient_names):
            result[name] = variance_vec[i, 0]
        
        return result


def example_simple_gambler():
    """
    보기 1: 노름꾼의 파산 문제.
    
    노름꾼이 $2으로 시작한다. 판마다 $1을 따거나(p=0.5) $1을 잃는다(1-p=0.5).
    $0(파산)이나 $4(목표)에 이르면 노름이 끝난다.
    """
    print("=" * 70)
    print("Example 1: Gambler's Ruin")
    print("=" * 70)
    
    # 상태: $0, $1, $2, $3, $4
    # 흡수: $0(파산), $4(승리)
    # 지나감: $1, $2, $3
    
    states = ['$0 (Broke)', '$1', '$2', '$3', '$4 (Win)']
    
    # 옮김 행렬(공정한 노름이면 p = 0.5)
    P = np.array([
        [1.0, 0.0, 0.0, 0.0, 0.0],  # $0: 파산 그대로
        [0.5, 0.0, 0.5, 0.0, 0.0],  # $1: $0이나 $2으로
        [0.0, 0.5, 0.0, 0.5, 0.0],  # $2: $1이나 $3으로
        [0.0, 0.0, 0.5, 0.0, 0.5],  # $3: $2이나 $4으로
        [0.0, 0.0, 0.0, 0.0, 1.0]   # $4: 승리 그대로
    ])
    
    print("\nTransition Matrix:")
    print(P)
    
    chain = AbsorbingMarkovChain(P, states)
    
    print(f"\nAbsorbing states: {chain.absorbing_names}")
    print(f"Transient states: {chain.transient_names}")
    
    # 바탕 행렬
    print("\n" + "-" * 70)
    print("Fundamental Matrix N (expected visits):")
    N = chain.fundamental_matrix()
    print(f"{'':8s} " + " ".join(f"{s:8s}" for s in chain.transient_names))
    for i, name in enumerate(chain.transient_names):
        row = " ".join(f"{N[i,j]:8.4f}" for j in range(len(chain.transient_names)))
        print(f"{name:8s} {row}")
    
    # 흡수까지의 기댓값 걸음 수
    print("\n" + "-" * 70)
    print("Expected Steps to Absorption:")
    expected_steps = chain.expected_steps_to_absorption()
    for state, steps in expected_steps.items():
        print(f"  Starting from {state}: {steps:.4f} steps")
    
    # 흡수 확률
    print("\n" + "-" * 70)
    print("Absorption Probabilities:")
    absorption_probs = chain.absorption_probabilities()
    for trans_state in chain.transient_names:
        print(f"\n  Starting from {trans_state}:")
        for abs_state in chain.absorbing_names:
            prob = absorption_probs[trans_state][abs_state]
            print(f"    P(absorb at {abs_state}) = {prob:.6f}")
    
    # 흩어짐
    print("\n" + "-" * 70)
    print("Variance of Steps to Absorption:")
    variances = chain.variance_steps_to_absorption()
    for state, var in variances.items():
        print(f"  Starting from {state}: {var:.4f} (std = {np.sqrt(var):.4f})")


def example_disease_model():
    """
    보기 2: 병의 진행 모형.
    
    상태: 건강, 감염, 회복, 죽음
    흡수: 회복, 죽음
    """
    print("\n" + "=" * 70)
    print("Example 2: Disease Progression Model")
    print("=" * 70)
    
    states = ['Healthy', 'Infected', 'Recovered', 'Dead']
    
    P = np.array([
        [0.7, 0.3, 0.0, 0.0],    # 건강: 옮을 수 있음
        [0.0, 0.4, 0.5, 0.1],    # 감염: 낫거나, 앓은 채이거나, 죽음
        [0.0, 0.0, 1.0, 0.0],    # 회복: 흡수
        [0.0, 0.0, 0.0, 1.0]     # 죽음: 흡수
    ])
    
    print("\nTransition Matrix:")
    print(f"{'':12s} {'Healthy':>10s} {'Infected':>10s} {'Recovered':>10s} {'Dead':>10s}")
    for i, state in enumerate(states):
        row = " ".join(f"{P[i,j]:10.4f}" for j in range(len(states)))
        print(f"{state:12s} {row}")
    
    chain = AbsorbingMarkovChain(P, states)
    
    print(f"\nAbsorbing states: {chain.absorbing_names}")
    print(f"Transient states: {chain.transient_names}")
    
    # 흡수까지의 기댓값 시간
    expected_steps = chain.expected_steps_to_absorption()
    print("\nExpected time until recovery or death:")
    for state, steps in expected_steps.items():
        print(f"  From {state}: {steps:.4f} days")
    
    # 흡수 확률
    absorption_probs = chain.absorption_probabilities()
    print("\nFinal outcome probabilities:")
    for trans_state in chain.transient_names:
        print(f"\n  Starting from {trans_state}:")
        for abs_state in chain.absorbing_names:
            prob = absorption_probs[trans_state][abs_state]
            print(f"    {abs_state}: {prob:.4f} ({prob*100:.2f}%)")


def visualize_absorption():
    """
    흉내내기로 흡수 과정 그려 보기.
    """
    print("\n" + "=" * 70)
    print("Creating Absorption Visualization")
    print("=" * 70)
    
    # 노름꾼의 파산
    states_idx = {'$0': 0, '$1': 1, '$2': 2, '$3': 3, '$4': 4}
    P = np.array([
        [1.0, 0.0, 0.0, 0.0, 0.0],
        [0.5, 0.0, 0.5, 0.0, 0.0],
        [0.0, 0.5, 0.0, 0.5, 0.0],
        [0.0, 0.0, 0.5, 0.0, 0.5],
        [0.0, 0.0, 0.0, 0.0, 1.0]
    ])
    
    # 여러 경로 흉내내기
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 그림 1: 표본 경로
    ax = axes[0, 0]
    
    for _ in range(20):
        path = [2]  # $2에서 시작
        current = 2
        
        while current != 0 and current != 4 and len(path) < 100:
            probs = P[current, :]
            current = np.random.choice(5, p=probs)
            path.append(current)
        
        ax.plot(path, alpha=0.6, linewidth=1.5)
    
    ax.set_xlabel('Time Step', fontsize=11)
    ax.set_ylabel('Money ($)', fontsize=11)
    ax.set_title('Sample Paths in Gambler\'s Ruin', fontsize=12)
    ax.set_yticks([0, 1, 2, 3, 4])
    ax.set_yticklabels(['$0', '$1', '$2', '$3', '$4'])
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='Broke')
    ax.axhline(y=4, color='green', linestyle='--', alpha=0.5, label='Win')
    ax.legend()
    
    # 그림 2: 흡수 시간의 분포
    ax = axes[0, 1]
    
    absorption_times = []
    for _ in range(10000):
        steps = 0
        current = 2
        
        while current != 0 and current != 4 and steps < 1000:
            probs = P[current, :]
            current = np.random.choice(5, p=probs)
            steps += 1
        
        absorption_times.append(steps)
    
    ax.hist(absorption_times, bins=50, density=True, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Steps to Absorption', fontsize=11)
    ax.set_ylabel('Probability Density', fontsize=11)
    ax.set_title('Distribution of Time to Absorption', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # 이론의 평균 더하기
    chain = AbsorbingMarkovChain(P, ['$0', '$1', '$2', '$3', '$4'])
    expected = chain.expected_steps_to_absorption()
    theoretical_mean = expected['$2']
    ax.axvline(x=theoretical_mean, color='red', linestyle='--', linewidth=2,
              label=f'Theoretical Mean: {theoretical_mean:.2f}')
    ax.axvline(x=np.mean(absorption_times), color='blue', linestyle='--', linewidth=2,
              label=f'Empirical Mean: {np.mean(absorption_times):.2f}')
    ax.legend()
    
    # 그림 3: 흡수 확률
    ax = axes[1, 0]
    
    outcomes = {'Win': 0, 'Broke': 0}
    for _ in range(10000):
        current = 2
        
        while current != 0 and current != 4:
            probs = P[current, :]
            current = np.random.choice(5, p=probs)
        
        if current == 4:
            outcomes['Win'] += 1
        else:
            outcomes['Broke'] += 1
    
    labels = list(outcomes.keys())
    values = [outcomes[k] / 10000 for k in labels]
    colors = ['green', 'red']
    
    bars = ax.bar(labels, values, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax.set_ylabel('Probability', fontsize=11)
    ax.set_title('Absorption Outcomes (Starting from $2)', fontsize=12)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{val:.4f}',
               ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # 그림 4: 바탕 행렬 그림
    ax = axes[1, 1]
    
    chain = AbsorbingMarkovChain(P, ['$0', '$1', '$2', '$3', '$4'])
    N = chain.fundamental_matrix()
    
    im = ax.imshow(N, cmap='YlOrRd', aspect='auto')
    ax.set_xticks(range(len(chain.transient_names)))
    ax.set_yticks(range(len(chain.transient_names)))
    ax.set_xticklabels(chain.transient_names)
    ax.set_yticklabels(chain.transient_names)
    ax.set_xlabel('To State', fontsize=11)
    ax.set_ylabel('From State', fontsize=11)
    ax.set_title('Fundamental Matrix N (Expected Visits)', fontsize=12)
    
    for i in range(len(chain.transient_names)):
        for j in range(len(chain.transient_names)):
            text = ax.text(j, i, f'{N[i, j]:.2f}',
                         ha="center", va="center", color="black", fontsize=10)
    
    plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/absorbing_chains.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Absorption visualization saved")


def main():
    """
    보기 모두 돌리기.
    """
    print("ABSORBING MARKOV CHAINS")
    print("=======================\n")
    
    example_simple_gambler()
    example_disease_model()
    visualize_absorption()
    
    print("\n" + "=" * 70)
    print("Key Concepts:")
    print("=" * 70)
    print("1. Absorbing state: P[i][i] = 1")
    print("2. Fundamental matrix: N = (I - Q)^{-1}")
    print("3. Expected steps to absorption: t = N × 1")
    print("4. Absorption probabilities: B = N × R")
    print("5. N[i][j] = expected visits to state j from state i")


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
흡수 사슬 구현을 확인하는 두루 갖춘 시험 함수를 적어라. 빈 입력, 원소 하나짜리 입력, 아주 큰 입력, 그리고 극단적인 값(0, 아주 큰 수)이 든 입력 같은 모서리 경우를 시험하여라.

??? success "연습문제 4 풀이"
    경계 조건을 두루 시험하는 함수를 만든다.
    ```python
    def test_absorbingmarkovchain():
        model = AbsorbingMarkovChain(...)
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

**다룬 것** — 흡수 사슬

시각화는 모델의 거동을 이해하고 학습 문제를 진단하는 데 중요한 역할을 한다.

고갱이 갈래는 `AbsorbingMarkovChain`이며 앞의 연습문제 4개로 스스로 따져 볼 수 있다.
