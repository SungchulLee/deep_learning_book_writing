# 홉필드 신경망과 연상 기억

연상 기억은 손상된 입력에서 저장한 패턴을 되찾는 문제다. 고전 홉필드 신경망은 패턴 주변에 낮은 에너지 상태를 만들고 단위를 하나씩 갱신해 안정 상태를 찾는다. 저장한 패턴이 반드시 모두 안정하거나 모든 손상을 복구할 수 있는 것은 아니므로 수렴과 올바른 검색을 구분해야 한다.

## 1. 구조와 에너지

N개 상태 sᵢ∈{−1,+1}, 대칭 가중치 Wᵢⱼ=Wⱼᵢ와 Wᵢᵢ=0을 사용한다. 편향 θ를 포함한 에너지는 다음과 같다.

$$E(s)=-\frac12s^TWs-\theta^Ts$$

뉴런 i의 국소장은 hᵢ=ΣⱼWᵢⱼsⱼ+θᵢ다. 한 번에 한 뉴런만 hᵢ의 부호로 바꾸며 hᵢ=0이면 기존 값을 유지한다. 한 뉴런의 변화에 대한 에너지 차이는 다음과 같다.

$$\Delta E=-(s_i'-s_i)h_i\leq0$$

현재 부호와 국소장이 반대여서 상태가 바뀌면 에너지가 엄격히 감소한다. 유한 상태 공간에서 모든 단위를 계속 방문하면 고정점에 도달한다. 이 결론은 비동기 갱신·대칭 W·자기 연결 없음과 동률 유지 규칙에 의존한다. 여러 단위를 동시에 갱신하는 방식에는 주기가 생길 수 있다.

## 2. 헤브 규칙과 검색 예제

??? note "홉필드 클래스와 패턴 검색 코드"

    ```python
    import torch
    import torch.nn as nn
    import numpy as np
    import matplotlib.pyplot as plt
    from typing import Tuple, List
    # 무작위로 뽑는 값이 아래에 나온다. 씨앗을 고정해야 이 쪽에 실린
    # 수가 다시 나온다 — 고정하지 않으면 돌릴 때마다 다른 수가 찍힌다
    torch.manual_seed(0)

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

**보존된 출력:**

```
Stored 2 patterns via Hebbian learning
  Network capacity: ~3 patterns

Retrieval completed in 2 iterations
Overlap with pattern A: 1.000
Overlap with pattern B: 0.520
```

코드는 P개 패턴 ξ의 외적 합을 P로 나누고 대각을 0으로 만든다. 흔한 1/N 정규화와 다르다. 편향이 0이면 양의 상수 배수는 갱신 부호를 바꾸지 않지만 에너지 크기와 양의 온도 확률분포는 바꾼다. 서로 다른 패턴 수에서 에너지 크기를 그대로 비교하는 데 주의한다.

첫 데모는 25개 뉴런에 두 5×5 패턴을 저장하고 A의 5개 비트를 뒤집어 검색한다. retrieve의 한 iteration은 무작위 순서로 전체 뉴런을 한 번 갱신한 sweep이다. 에너지 이력도 뉴런마다가 아니라 sweep마다 기록한다. 갱신 규칙은 결정적이어도 순서가 무작위이므로 가능한 경로와 최종 basin은 달라질 수 있다.

출력과 그림 제목은 기존 실행의 예시다. 새 입력에서도 항상 A를 복구하거나 B를 선택하지 않는다는 보장은 없다. 최대 반복 수에 도달한 반환값이 반드시 고정점이라는 보장도 별도로 확인해야 한다.

## 3. 중첩도와 저장 용량

패턴 ξ와 상태 s의 중첩도 m=N⁻¹Σsᵢξᵢ는 1−2d_H(s,ξ)/N이다. m=1이면 완전 일치, m=−1이면 완전 반전이다. [고전 용량 분석](https://doi.org/10.1103/PhysRevLett.55.1530)은 큰 N, 무작위 비편향 패턴과 특정 검색 기준에서 선형 규모의 임계 부하를 다룬다. 흔히 인용하는 약 0.14N은 그 조건의 결과이며 임의 패턴의 완전 복구 보장이 아니다.

??? note "무작위 패턴 수에 따른 검색 실험"

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

??? note "보존된 실험 출력"

    ```
    Stored 1 patterns via Hebbian learning
      Network capacity: ~15 patterns
    Stored 1 patterns via Hebbian learning
      Network capacity: ~15 patterns
    Stored 1 patterns via Hebbian learning
      Network capacity: ~15 patterns
    Stored 1 patterns via Hebbian learning
      Network capacity: ~15 patterns
    Stored 1 patterns via Hebbian learning
      Network capacity: ~15 patterns
    Patterns:  1, Success rate: 1.00, Avg overlap: 1.000
    Stored 3 patterns via Hebbian learning
      Network capacity: ~15 patterns
    Stored 3 patterns via Hebbian learning
      Network capacity: ~15 patterns
    Stored 3 patterns via Hebbian learning
      Network capacity: ~15 patterns
    Stored 3 patterns via Hebbian learning
      Network capacity: ~15 patterns
    Stored 3 patterns via Hebbian learning
      Network capacity: ~15 patterns
    Patterns:  3, Success rate: 1.00, Avg overlap: 1.000
    Stored 5 patterns via Hebbian learning
      Network capacity: ~15 patterns
    Stored 5 patterns via Hebbian learning
      Network capacity: ~15 patterns
    Stored 5 patterns via Hebbian learning
      Network capacity: ~15 patterns
    Stored 5 patterns via Hebbian learning
      Network capacity: ~15 patterns
    Stored 5 patterns via Hebbian learning
      Network capacity: ~15 patterns
    Patterns:  5, Success rate: 1.00, Avg overlap: 1.000
    Stored 7 patterns via Hebbian learning
      Network capacity: ~15 patterns
    Stored 7 patterns via Hebbian learning
      Network capacity: ~15 patterns
    Stored 7 patterns via Hebbian learning
      Network capacity: ~15 patterns
    Stored 7 patterns via Hebbian learning
      Network capacity: ~15 patterns
    Stored 7 patterns via Hebbian learning
      Network capacity: ~15 patterns
    Patterns:  7, Success rate: 1.00, Avg overlap: 1.000
    Stored 9 patterns via Hebbian learning
      Network capacity: ~15 patterns
    Stored 9 patterns via Hebbian learning
      Network capacity: ~15 patterns
    Stored 9 patterns via Hebbian learning
      Network capacity: ~15 patterns
    Stored 9 patterns via Hebbian learning
      Network capacity: ~15 patterns
    Stored 9 patterns via Hebbian learning
      Network capacity: ~15 patterns
    Patterns:  9, Success rate: 1.00, Avg overlap: 1.000
    Stored 11 patterns via Hebbian learning
      Network capacity: ~15 patterns
    Stored 11 patterns via Hebbian learning
      Network capacity: ~15 patterns
    Stored 11 patterns via Hebbian learning
      Network capacity: ~15 patterns
    Stored 11 patterns via Hebbian learning
      Network capacity: ~15 patterns
    Stored 11 patterns via Hebbian learning
      Network capacity: ~15 patterns
    Patterns: 11, Success rate: 0.96, Avg overlap: 0.989
    Stored 13 patterns via Hebbian learning
      Network capacity: ~15 patterns
    Stored 13 patterns via Hebbian learning
      Network capacity: ~15 patterns
    Stored 13 patterns via Hebbian learning
      Network capacity: ~15 patterns
    Stored 13 patterns via Hebbian learning
      Network capacity: ~15 patterns
    Stored 13 patterns via Hebbian learning
      Network capacity: ~15 patterns
    Patterns: 13, Success rate: 0.85, Avg overlap: 0.964
    Stored 15 patterns via Hebbian learning
      Network capacity: ~15 patterns
    Stored 15 patterns via Hebbian learning
      Network capacity: ~15 patterns
    Stored 15 patterns via Hebbian learning
      Network capacity: ~15 patterns
    Stored 15 patterns via Hebbian learning
      Network capacity: ~15 patterns
    Stored 15 patterns via Hebbian learning
      Network capacity: ~15 patterns
    Patterns: 15, Success rate: 0.81, Avg overlap: 0.931
    Stored 17 patterns via Hebbian learning
      Network capacity: ~15 patterns
    Stored 17 patterns via Hebbian learning
      Network capacity: ~15 patterns
    Stored 17 patterns via Hebbian learning
      Network capacity: ~15 patterns
    Stored 17 patterns via Hebbian learning
      Network capacity: ~15 patterns
    Stored 17 patterns via Hebbian learning
      Network capacity: ~15 patterns
    Patterns: 17, Success rate: 0.73, Avg overlap: 0.910
    Stored 19 patterns via Hebbian learning
      Network capacity: ~15 patterns
    Stored 19 patterns via Hebbian learning
      Network capacity: ~15 patterns
    Stored 19 patterns via Hebbian learning
      Network capacity: ~15 patterns
    Stored 19 patterns via Hebbian learning
      Network capacity: ~15 patterns
    Stored 19 patterns via Hebbian learning
      Network capacity: ~15 patterns
    Patterns: 19, Success rate: 0.55, Avg overlap: 0.815
    Stored 21 patterns via Hebbian learning
      Network capacity: ~15 patterns
    Stored 21 patterns via Hebbian learning
      Network capacity: ~15 patterns
    Stored 21 patterns via Hebbian learning
      Network capacity: ~15 patterns
    Stored 21 patterns via Hebbian learning
      Network capacity: ~15 patterns
    Stored 21 patterns via Hebbian learning
      Network capacity: ~15 patterns
    Patterns: 21, Success rate: 0.53, Avg overlap: 0.821
    Stored 23 patterns via Hebbian learning
      Network capacity: ~15 patterns
    Stored 23 patterns via Hebbian learning
      Network capacity: ~15 patterns
    Stored 23 patterns via Hebbian learning
      Network capacity: ~15 patterns
    Stored 23 patterns via Hebbian learning
      Network capacity: ~15 patterns
    Stored 23 patterns via Hebbian learning
      Network capacity: ~15 patterns
    Patterns: 23, Success rate: 0.27, Avg overlap: 0.663
    Stored 25 patterns via Hebbian learning
      Network capacity: ~15 patterns
    Stored 25 patterns via Hebbian learning
      Network capacity: ~15 patterns
    Stored 25 patterns via Hebbian learning
      Network capacity: ~15 patterns
    Stored 25 patterns via Hebbian learning
      Network capacity: ~15 patterns
    Stored 25 patterns via Hebbian learning
      Network capacity: ~15 patterns
    Patterns: 25, Success rate: 0.15, Avg overlap: 0.570
    Stored 27 patterns via Hebbian learning
      Network capacity: ~15 patterns
    Stored 27 patterns via Hebbian learning
      Network capacity: ~15 patterns
    Stored 27 patterns via Hebbian learning
      Network capacity: ~15 patterns
    Stored 27 patterns via Hebbian learning
      Network capacity: ~15 patterns
    Stored 27 patterns via Hebbian learning
      Network capacity: ~15 patterns
    Patterns: 27, Success rate: 0.04, Avg overlap: 0.553
    Stored 29 patterns via Hebbian learning
      Network capacity: ~15 patterns
    Stored 29 patterns via Hebbian learning
      Network capacity: ~15 patterns
    Stored 29 patterns via Hebbian learning
      Network capacity: ~15 patterns
    Stored 29 patterns via Hebbian learning
      Network capacity: ~15 patterns
    Stored 29 patterns via Hebbian learning
      Network capacity: ~15 patterns
    Patterns: 29, Success rate: 0.04, Avg overlap: 0.533
    ```

이 코드는 N=100, 패턴 수 1,3,…,29에서 5번의 실험을 수행한다. 각 패턴의 10비트를 뒤집고 overlap>0.9를 성공으로 센다. N=100에서 이는 잔여 오차가 **5개 미만**인 기준이며 완전 일치와는 다르다. 출력의 0.15N 수직선은 참고선이지 유한 실험의 정확한 경계가 아니다.

패턴 상관, 초기 손상률, 성공 정의와 반복 수에 따라 곡선이 달라진다. 같은 패턴 집합의 여러 검색은 독립적인 학습 실험과도 구분한다. 저장한 모든 패턴이 안정하다는 주장과 무작위 입력에서 어느 패턴으로 수렴하는지는 서로 다른 질문이다.

## 4. 저장하지 않은 안정 상태

서로 간섭하는 패턴은 의도하지 않은 고정점을 만들 수 있다. 예를 들어 여러 패턴 합의 부호를 취한 혼합 상태가 안정할 수 있지만 모든 합이 안정한 것은 아니다. 편향이 0이면 E(s)=E(−s)이므로 저장 패턴의 반전도 대칭적으로 등장한다.

??? note "임의 초기점에서 후보 상태 찾기"

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

find_spurious_states는 절댓값 중첩도 0.95를 임계값으로 사용한다. 따라서 거의 일치하거나 반전된 상태를 같은 것으로 취급하며, 정확한 전체 고정점 열거가 아니다. 100개의 무작위 초기점에서 발견하지 못했다고 해당 상태가 없다는 뜻도 아니다. 편향이 있으면 반전 대칭을 당연히 가정할 수 없다.

## 5. 작은 상태 공간의 열거

??? note "6개 뉴런의 상태 에너지와 국소 최소점"

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

**보존된 출력:**

```
Stored 2 patterns via Hebbian learning
  Network capacity: ~0 patterns

Found 4 local minima:
  State 7: E = -7.000, Type: Pattern 1 (inverted)
  State 14: E = -7.000, Type: Pattern 2
  State 49: E = -7.000, Type: Pattern 2 (inverted)
  State 56: E = -7.000, Type: Pattern 1
```

6개 뉴런이면 64개 상태를 모두 열거할 수 있다. 각 비트를 하나 뒤집은 이웃보다 에너지가 낮거나 같으면 국소 최소점으로 분류한다. 동률을 허용하므로 엄격한 최소점과 구분한다. 부호가 0인 국소장에서 유지하는 갱신 규칙과도 연결된다.

그림의 가로축은 상태를 **에너지 크기로 정렬한 순서**다. 가로로 가까운 점이 해밍 거리에서 가깝다는 뜻은 아니므로 이 선만으로 basin이나 전이 장벽의 모양을 판단할 수 없다. 이웃 관계는 별도의 한 비트 전이 그래프에서 확인해야 한다.

## 6. 연속 상태와 어텐션 연결

[Modern Hopfield 연구](https://arxiv.org/abs/2008.02217)는 연속 상태의 에너지와 softmax 갱신을 연결한다. 행에 저장 패턴을 둔 M과 질의 q에 대해 한 번의 갱신은 다음과 같은 가중 평균이다.

$$q'=M^T\operatorname{softmax}(\beta Mq)$$

??? note "연속 패턴의 softmax 검색"

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
            소프트맥스 어텐션으로 무늬를 되찾는다.

            x_new = softmax(β * X^T * q)^T * X
            """
            # 주의 점수
            scores = self.beta * torch.matmul(self.patterns, query)
            attention = torch.softmax(scores, dim=0)

            # 무늬의 무게 있는 결합
            return torch.matmul(attention, self.patterns)
    ```

코드는 패턴 행렬에 대한 점수와 가중 평균을 한 번 계산한다. beta는 점수 스케일이며 고전 비트 갱신과 다르다. 한 번의 평균이 항상 저장 패턴 하나를 정확히 반환하지는 않는다. 연구의 용량·오차 결과에는 패턴의 분리도와 차원 등의 조건이 있으므로 임의 데이터의 지수적 저장 보장으로 읽지 않는다.

pattern_dim 인수는 저장만 하거나 검증하는 데 쓰이지 않으며 실제 코드에는 해당 차원 검사가 없다. patterns는 Parameter나 buffer로 등록되지 않아 model.to(device)가 이를 자동으로 옮기지 않는다. 이 클래스는 완전한 트랜스포머나 학습·용량 비교 실험을 구현하지 않는다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1. 에너지 감소**

한 뉴런 갱신의 에너지 차이를 구하고 수렴과 올바른 검색을 구분하라.

</div>

??? success "연습문제 1 풀이"
    대칭 W와 대각 0이면 뉴런 i에 의존하는 항은 −sᵢhᵢ이므로 ΔE=−(sᵢ′−sᵢ)hᵢ다. 부호를 맞추는 갱신은 에너지를 늘리지 않고 실제 변화는 동률 유지 규칙 아래 엄격한 감소를 만든다. 고정점이어도 저장하지 않은 상태일 수 있어 올바른 기억 검색은 별도로 평가해야 한다.

---

<div class="drillbox" markdown>

**연습문제 2. 중첩도 해석**

N=100에서 overlap>0.9를 만족하는 최대 해밍 오차는 몇 개인가?

</div>

??? success "연습문제 2 풀이"
    m=1−2d/100>0.9이므로 d<5, 정수 오차는 최대 4개다. 완전 일치는 d=0이며 두 성공 기준은 다르다. 절댓값 중첩도를 쓰면 반전된 패턴도 유사하다고 판정할 수 있다.

---

<div class="drillbox" markdown>

**연습문제 3. 지형 그림의 한계**

에너지순으로 정렬한 선 그래프에서 이웃 점 사이의 높이 차이를 전이 장벽으로 읽을 수 있는가?

</div>

??? success "연습문제 3 풀이"
    아니다. 에너지 순서가 한 비트 전이의 이웃 관계를 보존하지 않는다. 장벽과 basin을 보려면 상태의 해밍 이웃과 갱신 경로를 사용해야 한다. 작은 예제의 전체 열거도 고차원 전체 지형을 대신하지 않는다.

## 정리하며

홉필드 신경망의 비동기 갱신은 대칭 에너지의 감소로 설명할 수 있다. 고정점 수렴과 기억 복구를 구분하고 중첩도·용량·허위 기억의 판정 조건을 명시한다. 연속 softmax 검색과의 연결도 구현 범위와 연구 가정을 함께 읽는다.

**참고 문헌**

- Hopfield, J. J. (1982). Neural networks and physical systems with emergent collective computational abilities. PNAS.
- Amit, D. J., Gutfreund, H., & Sompolinsky, H. (1985). Storing infinite numbers of patterns in a spin-glass model of neural networks. Physical Review Letters.
- Ramsauer, H., et al. (2021). Hopfield Networks is All You Need. ICLR.
