# 홉필드 신경망



## 학습 목표

이 절을 마치면 다음을 할 수 있다:

1. 홉필드 신경망을 에너지 바탕 이어 떠올림 기억으로 이해한다
2. 헤브 배움으로 무늬 담기를 짠다
3. 에너지 가장 작게 하기로 무늬를 되찾는다
4. 신경망 담이와 헛된 상태를 살핀다
5. 홉필드 신경망을 요즘 에너지 바탕 모델과 잇는다

## 들어가며

1982년 존 홉필드가 내놓은 홉필드 신경망은 가장 이르고 가장 영향력 있는 에너지 바탕 모델 가운데 하나이다. 이는 에너지 가장 작게 하기로 이어 떠올림 기억, 곧 일부만 있거나 망가진 들임에서 온전한 무늬를 되찾는 힘을 어떻게 짜는지 보여 준다. 이 일은 신경망에 대한 관심을 되살렸고 신경 과학, 물리학, 셈 사이의 깊은 이음을 세웠다.

## 얼개와 움직임

### 신경망 얼개

홉필드 신경망은 맞섬이고 온전히 이어진 무게를 가진 이진 신경 세포 $N$개로 이루어진다:

- **신경 세포**: $i = 1, \ldots, N$에 대해 $s_i \in \{-1, +1\}$
- **무게**: $w_{ij} = w_{ji}$(맞섬)
- **스스로 이음 없음**: $w_{ii} = 0$
- **치우침과 문턱**: $\theta_i$(흔히 0으로 둔다)

### 에너지 함수

신경망 상태 $\mathbf{s} = (s_1, \ldots, s_N)$의 에너지는 다음과 같다:

$$E(\mathbf{s}) = -\frac{1}{2} \sum_{i,j} w_{ij} s_i s_j - \sum_i \theta_i s_i$$

행렬 꼴로 쓰면:

$$E(\mathbf{s}) = -\frac{1}{2} \mathbf{s}^T \mathbf{W} \mathbf{s} - \boldsymbol{\theta}^T \mathbf{s}$$

### 갱신 규칙

신경 세포는 정해진 문턱 규칙으로 제각기 고쳐진다:

$$s_i \leftarrow \text{sign}(h_i)$$

여기서 그 자리의 마당은 다음과 같다:

$$h_i = \sum_j w_{ij} s_j + \theta_i$$

**결정적 성질**: 제각기 고침은 결코 에너지를 늘리지 않는다.

## PyTorch 구현

```python
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List

class HopfieldNetwork(nn.Module):
    """
    제각기 고침을 쓰는 이진 홉필드 신경망.
    
    에너지 가장 작게 하기로 이어 떠올림 기억을 짠다.
    무늬는 에너지 함수의 그 자리 최솟값으로 담긴다.
    
    매개변수
    ----------
    n_neurons : int
        신경망의 신경 세포 개수
    """
    
    def __init__(self, n_neurons: int):
        super().__init__()
        self.n_neurons = n_neurons
        
        # 무게 행렬(맞섬, 스스로 이음 없음)
        self.register_buffer('W', torch.zeros(n_neurons, n_neurons))
        
        # 문턱(치우침)
        self.register_buffer('theta', torch.zeros(n_neurons))
        
        # 되찾는 동안 에너지를 좇기 위해
        self.energy_history = []
    
    def train_hebbian(self, patterns: torch.Tensor) -> None:
        """
        헤브 배움 규칙으로 무늬를 담는다.
        
        "함께 터지는 뉴런은 함께 이어진다"
        
        w_ij = (1/P) Σₚ xᵢᵖ xⱼᵖ  for i ≠ j
        
        매개변수
        ----------
        patterns : torch.Tensor
            꼴 (n_patterns, n_neurons), 값은 {-1, +1}
        """
        n_patterns = patterns.shape[0]
        
        # 무게를 되돌린다
        self.W.zero_()
        
        # 헤브 규칙: 바깥 곱의 합
        for pattern in patterns:
            self.W += torch.outer(pattern, pattern)
        
        # 무늬 개수로 고르게 맞춘다
        self.W /= n_patterns
        
        # 스스로 이음을 없앤다
        self.W.fill_diagonal_(0)
        
        print(f"Stored {n_patterns} patterns via Hebbian learning")
        print(f"  Network capacity: ~{int(0.15 * self.n_neurons)} patterns")
    
    def energy(self, state: torch.Tensor) -> torch.Tensor:
        """
        신경망 상태의 에너지를 셈한다.
        
        E(s) = -½ sᵀWs - θᵀs
        
        매개변수
        ----------
        state : torch.Tensor
            꼴 (n_neurons,) 또는 (batch, n_neurons)
        
        반환값
        -------
        torch.Tensor
            Energy value(s)
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)
        
        # 이차 항: -½ sᵀWs
        quadratic = -0.5 * torch.einsum('bi,ij,bj->b', state, self.W, state)
        
        # 선형 항: -θᵀs
        linear = -torch.einsum('i,bi->b', self.theta, state)
        
        return quadratic + linear
    
    def local_field(self, state: torch.Tensor) -> torch.Tensor:
        """
        모든 신경 세포의 그 자리 마당을 셈한다.
        
        h_i = Σⱼ w_ij s_j + θ_i
        """
        return torch.mv(self.W, state) + self.theta
    
    def update_neuron(self, state: torch.Tensor, neuron_idx: int) -> torch.Tensor:
        """
        신경 세포 하나를 제각기 고친다.
        
        s_i ← sign(h_i)
        
        이 고침은 에너지를 늘리지 않음이 보장된다.
        """
        new_state = state.clone()
        h_i = self.local_field(state)[neuron_idx]
        
        # 문턱 깨움
        if h_i > 0:
            new_state[neuron_idx] = 1.0
        elif h_i < 0:
            new_state[neuron_idx] = -1.0
        # h_i == 0이면 지금 상태를 그대로 둔다
        
        return new_state
    
    def retrieve(self, 
                 initial_state: torch.Tensor,
                 max_iterations: int = 100,
                 track_energy: bool = True) -> Tuple[torch.Tensor, int]:
        """
        처음의 (망가졌을 수 있는) 상태에서 무늬를 되찾는다.
        
        모일 때까지 제각기 고친다.
        
        매개변수
        ----------
        initial_state : torch.Tensor
            처음 상태(망가진 무늬)
        max_iterations : int
            최대 고침 바퀴 수
        track_energy : bool
            에너지 자취를 남길지 여부
        
        반환값
        -------
        final_state : torch.Tensor
            되찾은 무늬(둘레에서 힘이 가장 낮은 자리)
        n_iterations : int
            모일 때까지의 되풀이 횟수
        """
        state = initial_state.clone()
        self.energy_history = []
        
        if track_energy:
            self.energy_history.append(self.energy(state).item())
        
        for iteration in range(max_iterations):
            old_state = state.clone()
            
            # 아무 차례로 신경 세포를 고친다
            update_order = torch.randperm(self.n_neurons)
            
            for neuron_idx in update_order:
                state = self.update_neuron(state, neuron_idx.item())
            
            if track_energy:
                self.energy_history.append(self.energy(state).item())
            
            # 모임 살피기
            if torch.equal(state, old_state):
                return state, iteration + 1
        
        return state, max_iterations
    
    def compute_overlap(self, state: torch.Tensor, pattern: torch.Tensor) -> float:
        """
        상태와 무늬 사이의 겹침(닮음)을 셈한다.
        
        Overlap = (1/N) Σᵢ sᵢ pᵢ ∈ [-1, 1]
        
        +1: 같음
        -1: 뒤집힘
         0: 얽히지 않음
        """
        return (state * pattern).mean().item()


def demonstrate_hopfield_retrieval():
    """
    홉필드 신경망으로 무늬 담기와 되찾기를 보여 준다.
    """
    # 단순한 5x5 무늬를 만든다(신경 세포 25개로 편다)
    pattern_A = torch.tensor([
        [-1, +1, +1, +1, -1],
        [+1, -1, -1, -1, +1],
        [+1, +1, +1, +1, +1],
        [+1, -1, -1, -1, +1],
        [+1, -1, -1, -1, +1]
    ], dtype=torch.float32).flatten()
    
    pattern_B = torch.tensor([
        [+1, +1, +1, +1, -1],
        [+1, -1, -1, -1, +1],
        [+1, +1, +1, +1, -1],
        [+1, -1, -1, -1, +1],
        [+1, +1, +1, +1, -1]
    ], dtype=torch.float32).flatten()
    
    patterns = torch.stack([pattern_A, pattern_B])
    
    # 신경망을 만들어 익힌다
    network = HopfieldNetwork(n_neurons=25)
    network.train_hebbian(patterns)
    
    # 무늬 A을 망가뜨린다(아무 비트 5개를 뒤집는다)
    corrupted = pattern_A.clone()
    flip_indices = torch.randperm(25)[:5]
    corrupted[flip_indices] *= -1
    
    # 되찾는다
    retrieved, n_iter = network.retrieve(corrupted)
    
    # 겹침을 셈한다
    overlap_A = network.compute_overlap(retrieved, pattern_A)
    overlap_B = network.compute_overlap(retrieved, pattern_B)
    
    print(f"\nRetrieval completed in {n_iter} iterations")
    print(f"Overlap with pattern A: {overlap_A:.3f}")
    print(f"Overlap with pattern B: {overlap_B:.3f}")
    
    # 시각화한다
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    
    # 1줄: 무늬와 되찾기
    axes[0, 0].imshow(pattern_A.reshape(5, 5), cmap='binary', vmin=-1, vmax=1)
    axes[0, 0].set_title('Original Pattern A')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(corrupted.reshape(5, 5), cmap='binary', vmin=-1, vmax=1)
    axes[0, 1].set_title('Corrupted Input')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(retrieved.reshape(5, 5), cmap='binary', vmin=-1, vmax=1)
    axes[0, 2].set_title('Retrieved Pattern')
    axes[0, 2].axis('off')
    
    axes[0, 3].imshow(pattern_B.reshape(5, 5), cmap='binary', vmin=-1, vmax=1)
    axes[0, 3].set_title('Pattern B (not retrieved)')
    axes[0, 3].axis('off')
    
    # 2줄: 에너지와 무게 행렬
    axes[1, 0].plot(network.energy_history, 'b-o', markersize=4)
    axes[1, 0].set_xlabel('Iteration')
    axes[1, 0].set_ylabel('Energy')
    axes[1, 0].set_title('Energy Descent During Retrieval')
    axes[1, 0].grid(True, alpha=0.3)
    
    im = axes[1, 1].imshow(network.W, cmap='RdBu', vmin=-1, vmax=1)
    axes[1, 1].set_title('Weight Matrix')
    plt.colorbar(im, ax=axes[1, 1])
    
    # 담긴 무늬의 에너지
    pattern_energies = network.energy(patterns).numpy()
    axes[1, 2].bar(['Pattern A', 'Pattern B'], pattern_energies)
    axes[1, 2].set_ylabel('Energy')
    axes[1, 2].set_title('Energy of Stored Patterns')
    axes[1, 2].grid(True, alpha=0.3)
    
    axes[1, 3].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return network

network = demonstrate_hopfield_retrieval()
```

## 신경망 담이

### 이론상 담이

홉필드 신경망이 믿을 만하게 담을 수 있는 무늬의 최대 개수는 대략 다음과 같다:

$$P_{\max} \approx 0.15 N$$

여기서 $N$은 신경 세포의 개수이다. 이 결과는 통계 역학 살피기에서 나온다.

### 담이 살피기

```python
def analyze_network_capacity(n_neurons: int = 100, 
                             max_patterns: int = 30,
                             n_trials: int = 10):
    """
    무늬 개수가 늘 때 되찾기 성능이 어떻게 나빠지는지 살핀다.
    
    매개변수
    ----------
    n_neurons : int
        신경망 크기
    max_patterns : int
        시험할 최대 무늬 개수
    n_trials : int
        무늬 개수마다 시도 횟수
    """
    pattern_counts = list(range(1, max_patterns + 1, 2))
    success_rates = []
    avg_overlaps = []
    
    theoretical_capacity = int(0.15 * n_neurons)
    
    for n_patterns in pattern_counts:
        trial_successes = []
        trial_overlaps = []
        
        for trial in range(n_trials):
            # 아무 무늬를 만든다
            patterns = torch.sign(torch.randn(n_patterns, n_neurons))
            
            # 신경망을 익힌다
            network = HopfieldNetwork(n_neurons)
            network.train_hebbian(patterns)
            
            # 무늬마다 되찾기를 시험한다
            for pattern in patterns:
                # 무늬를 망가뜨린다(비트의 10%을 뒤집는다)
                corrupted = pattern.clone()
                n_flip = int(0.1 * n_neurons)
                flip_idx = torch.randperm(n_neurons)[:n_flip]
                corrupted[flip_idx] *= -1
                
                # 되찾는다
                retrieved, _ = network.retrieve(corrupted, track_energy=False)
                
                # 겹침을 셈한다
                overlap = network.compute_overlap(retrieved, pattern)
                trial_overlaps.append(overlap)
                trial_successes.append(overlap > 0.9)
        
        success_rates.append(np.mean(trial_successes))
        avg_overlaps.append(np.mean(trial_overlaps))
        
        print(f"Patterns: {n_patterns:2d}, Success rate: {success_rates[-1]:.2f}, "
              f"Avg overlap: {avg_overlaps[-1]:.3f}")
    
    # 결과 그리기
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(pattern_counts, success_rates, 'bo-', linewidth=2, markersize=8)
    axes[0].axvline(theoretical_capacity, color='red', linestyle='--', 
                    label=f'Theoretical capacity (0.15N = {theoretical_capacity})')
    axes[0].set_xlabel('Number of Stored Patterns')
    axes[0].set_ylabel('Retrieval Success Rate')
    axes[0].set_title('Network Capacity: Success Rate')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(pattern_counts, avg_overlaps, 'go-', linewidth=2, markersize=8)
    axes[1].axvline(theoretical_capacity, color='red', linestyle='--', 
                    label='Theoretical capacity')
    axes[1].axhline(0.9, color='orange', linestyle=':', label='Good retrieval threshold')
    axes[1].set_xlabel('Number of Stored Patterns')
    axes[1].set_ylabel('Average Overlap')
    axes[1].set_title('Network Capacity: Pattern Overlap')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

analyze_network_capacity(n_neurons=100, max_patterns=30, n_trials=5)
```

## 헛된 상태

### 정의

헛된 상태는 담긴 무늬가 아닌 안정된 상태(그 자리의 에너지 최솟값)이다. 이는 신경망의 "거짓 기억"을 나타낸다.

### 헛된 상태의 갈래

1. **섞임 상태**: 담긴 무늬의 선형 결합
2. **뒤집힌 무늬**: 담긴 무늬의 부정($-\mathbf{p}$)
3. **스핀 유리 상태**: 어떤 담긴 무늬와도 상관없다

### 헛된 상태 살피기

```python
def find_spurious_states(network: HopfieldNetwork, 
                         stored_patterns: torch.Tensor,
                         n_random_starts: int = 100) -> List[torch.Tensor]:
    """
    아무 첫자리매김과 모임으로 헛된 상태를 찾는다.
    """
    spurious = []
    
    for _ in range(n_random_starts):
        # 아무 처음 상태
        initial = torch.sign(torch.randn(network.n_neurons))
        
        # 붙박이점으로 모인다
        final, _ = network.retrieve(initial, track_energy=False)
        
        # 담긴 무늬인지 그 음인지 살핀다
        is_stored = False
        for pattern in stored_patterns:
            overlap = abs(network.compute_overlap(final, pattern))
            if overlap > 0.95:
                is_stored = True
                break
        
        if not is_stored:
            # 이미 찾았는지 살핀다
            already_found = False
            for sp in spurious:
                if abs(network.compute_overlap(final, sp)) > 0.95:
                    already_found = True
                    break
            
            if not already_found:
                spurious.append(final.clone())
    
    return spurious
```

## 에너지 지형 그려 보기

작은 신경망에서는 온전한 에너지 풍경을 그려 볼 수 있다:

```python
def visualize_energy_landscape():
    """
    작은 홉필드 신경망의 에너지 풍경을 그려 본다.
    """
    n_neurons = 6
    
    # 단순한 무늬
    p1 = torch.tensor([1, 1, 1, -1, -1, -1], dtype=torch.float32)
    p2 = torch.tensor([-1, -1, 1, 1, 1, -1], dtype=torch.float32)
    patterns = torch.stack([p1, p2])
    
    # 신경망을 익힌다
    network = HopfieldNetwork(n_neurons)
    network.train_hebbian(patterns)
    
    # 2^6 = 64가지 상태를 모두 늘어놓는다
    all_states = []
    all_energies = []
    
    for i in range(2**n_neurons):
        # 정수를 이진 상태로 바꾼다
        binary = format(i, f'0{n_neurons}b')
        state = torch.tensor([1.0 if b == '1' else -1.0 for b in binary])
        
        all_states.append(state)
        all_energies.append(network.energy(state).item())
    
    all_energies = np.array(all_energies)
    
    # 그 자리의 최솟값을 찾는다
    local_minima = []
    for i, state in enumerate(all_states):
        is_minimum = True
        current_E = all_energies[i]
        
        # 한 비트 뒤집기를 모두 살핀다
        for j in range(n_neurons):
            neighbor = state.clone()
            neighbor[j] *= -1
            
            # 이웃 번호를 찾는다
            neighbor_binary = ''.join(['1' if s > 0 else '0' for s in neighbor])
            neighbor_idx = int(neighbor_binary, 2)
            
            if all_energies[neighbor_idx] < current_E:
                is_minimum = False
                break
        
        if is_minimum:
            local_minima.append((i, state, current_E))
    
    print(f"\nFound {len(local_minima)} local minima:")
    for idx, state, E in local_minima:
        # 담긴 무늬와의 겹침을 살핀다
        overlap_1 = network.compute_overlap(state, p1)
        overlap_2 = network.compute_overlap(state, p2)
        
        pattern_type = "Spurious"
        if abs(overlap_1) > 0.9:
            pattern_type = "Pattern 1" if overlap_1 > 0 else "Pattern 1 (inverted)"
        elif abs(overlap_2) > 0.9:
            pattern_type = "Pattern 2" if overlap_2 > 0 else "Pattern 2 (inverted)"
        
        print(f"  State {idx}: E = {E:.3f}, Type: {pattern_type}")
    
    # 시각화한다
    fig, ax = plt.subplots(figsize=(12, 6))
    
    sorted_idx = np.argsort(all_energies)
    ax.plot(all_energies[sorted_idx], 'b-', linewidth=1, alpha=0.7)
    
    # 담긴 무늬를 표시한다
    for i, pattern in enumerate(patterns):
        pattern_binary = ''.join(['1' if s > 0 else '0' for s in pattern])
        pattern_idx = int(pattern_binary, 2)
        sorted_pos = np.where(sorted_idx == pattern_idx)[0][0]
        ax.plot(sorted_pos, all_energies[pattern_idx], 'go', markersize=12,
               label=f'Pattern {i+1}' if i == 0 else None)
    
    # 국소 최솟값 표시
    for idx, state, E in local_minima:
        sorted_pos = np.where(sorted_idx == idx)[0][0]
        ax.plot(sorted_pos, E, 'r^', markersize=10)
    
    ax.set_xlabel('State Index (sorted by energy)')
    ax.set_ylabel('Energy')
    ax.set_title('Energy Landscape of 6-Neuron Hopfield Network')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.show()

visualize_energy_landscape()
```

## 요즘 에너지 바탕 모델과의 이음

### 홉필드에서 요즘 에너지 바탕 모델로

홉필드 신경망은 요즘 에너지 바탕 모델에 쓰이는 핵심 원리를 세웠다:

| 홉필드 개념 | 요즘 에너지 바탕 모델의 짝 |
|-----------------|-------------------|
| 이진 신경 세포 | 이어진 숨은 변수 |
| 헤브 배움 | 기울기 바탕 배움 |
| 제각기 고침 | 랑주뱅 움직임 |
| 에너지 최솟값 | 확률 높은 자리 |

### 요즘 홉필드 신경망

최근 연구는 홉필드 신경망을 다음으로 넓혔다:

- **이어진 상태**: 람자우어 외(2021)
- **지수 담이**: 다항 무늬로 $\propto 2^{N/2}$
- **변환기와의 이음**: 이어 떠올림 기억으로서의 눈길

```python
class ModernHopfield(nn.Module):
    """
    이어진 상태와 지수 담이를 가진 요즘 홉필드 신경망.
    
    Ramsauer 외, "Hopfield Networks is All You Need"(2021)를 바탕으로 한다
    """
    
    def __init__(self, pattern_dim: int, beta: float = 1.0):
        super().__init__()
        self.beta = beta
        self.patterns = None  # Stored patterns
    
    def store(self, patterns: torch.Tensor):
        """무늬를 담는다(이어진 값도 된다)."""
        self.patterns = patterns  # Shape: (n_patterns, pattern_dim)
    
    def retrieve(self, query: torch.Tensor) -> torch.Tensor:
        """
        소프트맥스 눈길로 무늬를 되찾는다.
        
        x_new = softmax(β * X^T * q)^T * X
        """
        # 주의 점수
        scores = self.beta * torch.matmul(self.patterns, query)
        attention = torch.softmax(scores, dim=0)
        
        # 무늬의 무게 있는 결합
        return torch.matmul(attention, self.patterns)
```

## 핵심 정리

!!! success "핵심 개념"

    1. 홉필드 신경망은 이어 떠올림 기억에 에너지 가장 작게 하기를 쓴다
    2. 헤브 배움은 무늬를 에너지 최솟값으로 담는다
    3. 제각기 고침은 에너지가 내려감을 보장한다
    4. 담이는 대략 무늬 $0.15N$개이다
    5. 헛된 상태는 뜻하지 않은 그 자리의 최솟값이다

!!! warning "한계"

    - 담이가 제한된다(신경 세포 $N$개에 무늬 $O(N)$개)
    - 헛된 상태가 되찾기 어긋남을 낳는다
    - 이진 상태가 나타냄 힘을 제한한다
    - 큰 신경망에서는 모임이 느리다

## 참고 문헌

- Hopfield, J. J. (1982). Neural networks and physical systems with emergent collective computational abilities. PNAS.
- Amit, D. J., Gutfreund, H., & Sompolinsky, H. (1985). Storing infinite numbers of patterns in a spin-glass model of neural networks. Physical Review Letters.
- Ramsauer, H., et al. (2021). Hopfield Networks is All You Need. ICLR.

## 연습문제

1. **담이 커짐**: 크기가 다른 신경망에서 $0.15N$ 담이 규칙을 경험으로 확인하라.

2. **서로 얽힌 무늬**: 담긴 무늬가 서로 얽혀 있으면 어떻게 되는가? 짜서 살펴라.

3. **요즘 홉필드**: 이어진 홉필드 신경망을 짜서 고전 판과 되찾기 정확도를 견주어라.
