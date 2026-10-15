# 볼츠만 기계와 이진 깁스 샘플링

홉필드 신경망은 국소장을 따라 상태를 갱신해 기억을 검색한다. 볼츠만 기계는 단일 상태를 찾는 대신 확률적 갱신으로 상태 분포를 표현한다. 이때 에너지에 사용하는 이진 변수의 부호 관례와 조건부 확률이 일치해야 한다. 이 페이지의 구현에는 그 둘 사이에 온도 계수 차이가 있다.

## 1. 스핀 에너지와 조건부 확률

스핀 sᵢ∈{−1,+1}, 대칭 W와 대각 원소 0을 사용한다. 편향을 θᵢ라 하면 다음과 같다.

$$E(s)=-\frac12s^TWs-\theta^Ts,
\qquad h_i=\sum_jW_{ij}s_j+\theta_i$$

다른 스핀을 고정하면 E(sᵢ=+1)−E(sᵢ=−1)=−2hᵢ다. 따라서 온도 T의 볼츠만 분포에서 올바른 단위 조건부 확률은 다음과 같다.

$$P(s_i=+1\mid s_{-i})=\operatorname{sigmoid}(2h_i/T)$$

반면 0·1 변수 b 자체에 E(b)=−0.5bᵀWb−aᵀb를 정의하면 sigmoid((ΣWᵢⱼbⱼ+aᵢ)/T)가 나온다. **같은 W·편향을 그대로 두고 변수 표현만 바꾸면 같은 에너지 모델이 되지 않는다.**

## 2. 구현과 보존된 비교 출력

```python
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional
from tqdm import tqdm

class BoltzmannMachine(nn.Module):
    """
    확률 이진 단위를 가진 일반 볼츠만 기계.
    
    홉필드 신경망과 달리 볼츠만 기계는 확률 고침을 쓴다
    그리고 평형에서 볼츠만 분포에서 뽑을 수 있다.
    
    매개변수
    ----------
    n_visible : int
        드러난 단위의 개수
    n_hidden : int
        숨은 알갱이의 수(0이면 숨은 층이 없다)
    temperature : float
        마구잡이를 다스리는 온도 매개변수
    """
    
    def __init__(self, 
                 n_visible: int, 
                 n_hidden: int = 0, 
                 temperature: float = 1.0):
        super().__init__()
        
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.n_total = n_visible + n_hidden
        self.temperature = temperature
        
        # 무게를 첫자리매김한다(맞섬, 스스로 이음 없음)
        W = torch.randn(self.n_total, self.n_total) * 0.01
        W = (W + W.T) / 2  # Symmetrize
        W.fill_diagonal_(0)  # No self-connections
        self.register_buffer('W', W)
        
        # 치우침
        self.register_buffer('theta', torch.zeros(self.n_total))
    
    def energy(self, state: torch.Tensor) -> torch.Tensor:
        """
        Compute energy E(s) = -½ sᵀWs - θᵀs
        
        매개변수
        ----------
        state : torch.Tensor
            Binary state in {0, 1}^n or {-1, +1}^n
        """
        # 필요하면 {0,1}을 {-1,+1}으로 바꾼다
        if state.min() >= 0:
            s = 2 * state - 1
        else:
            s = state
        
        if s.dim() == 1:
            s = s.unsqueeze(0)
        
        quadratic = -0.5 * torch.einsum('bi,ij,bj->b', s, self.W, s)
        linear = -torch.einsum('i,bi->b', self.theta, s)
        
        return quadratic + linear
    
    def local_field(self, state: torch.Tensor) -> torch.Tensor:
        """Compute h_i = Σⱼ w_ij s_j + θ_i"""
        s = 2 * state - 1 if state.min() >= 0 else state
        return torch.mv(self.W, s) + self.theta
    
    def sample_probability(self, local_field: torch.Tensor) -> torch.Tensor:
        """
        Compute P(s_i = 1 | s_{-i}) = σ(h_i / T)
        """
        return torch.sigmoid(local_field / self.temperature)
    
    def sample_unit(self, 
                    state: torch.Tensor, 
                    unit_idx: int) -> torch.Tensor:
        """
        지금 상태가 주어질 때 단위 하나를 뽑는다.
        
        P(s_i = 1) = σ((Σⱼ w_ij s_j + θ_i) / T)
        """
        new_state = state.clone()
        h_i = self.local_field(state)[unit_idx]
        prob_on = self.sample_probability(h_i)
        new_state[unit_idx] = (torch.rand(1) < prob_on).float()
        return new_state
    
    def gibbs_step(self, state: torch.Tensor) -> torch.Tensor:
        """
        깁스 뽑기 한 바퀴를 온전히 돈다.
        
        아무 차례로 모든 단위를 고친다.
        """
        new_state = state.clone()
        update_order = torch.randperm(self.n_total)
        
        for unit_idx in update_order:
            new_state = self.sample_unit(new_state, unit_idx.item())
        
        return new_state
    
    def sample(self, 
               n_steps: int = 1000, 
               initial_state: Optional[torch.Tensor] = None,
               return_trajectory: bool = False) -> torch.Tensor:
        """
        깁스 뽑기로 표본을 만든다.
        
        매개변수
        ----------
        n_steps : int
            깁스 훑기 횟수
        initial_state : torch.Tensor, optional
            처음 상태(None이면 마구잡이)
        return_trajectory : bool
            참이면 중간 상태를 모두 돌려준다
        """
        # 초기화한다
        if initial_state is None:
            state = (torch.rand(self.n_total) > 0.5).float()
        else:
            state = initial_state.clone()
        
        trajectory = [state.clone()] if return_trajectory else None
        
        # 깁스 뽑기를 돌린다
        for _ in range(n_steps):
            state = self.gibbs_step(state)
            if return_trajectory:
                trajectory.append(state.clone())
        
        if return_trajectory:
            return torch.stack(trajectory)
        return state
    
    def estimate_distribution(self, 
                              n_samples: int = 10000,
                              burn_in: int = 1000,
                              thin: int = 10) -> dict:
        """
        마르코프 사슬 몬테카를로로 평형 분포를 어림한다.
        
        매개변수
        ----------
        n_samples : int
            모을 표본의 개수
        burn_in : int  
            평형에 이르도록 버릴 표본 수
        thin : int
            스스로 얽힘을 줄이려 thin번째 표본마다 남긴다
        """
        state = (torch.rand(self.n_total) > 0.5).float()
        
        # 버림 구간
        for _ in range(burn_in):
            state = self.gibbs_step(state)
        
        # 표본을 모은다
        samples = []
        energies = []
        
        for i in tqdm(range(n_samples * thin), desc="Sampling"):
            state = self.gibbs_step(state)
            
            if i % thin == 0:
                samples.append(state.clone())
                energies.append(self.energy(state).item())
        
        samples = torch.stack(samples)
        
        # 경험 분포를 셈한다
        state_counts = {}
        for sample in samples:
            state_tuple = tuple(sample.numpy().astype(int))
            state_counts[state_tuple] = state_counts.get(state_tuple, 0) + 1
        
        empirical_probs = {
            state: count / n_samples 
            for state, count in state_counts.items()
        }
        
        return {
            'samples': samples,
            'energies': np.array(energies),
            'empirical_probs': empirical_probs,
            'state_counts': state_counts
        }

def compare_deterministic_vs_stochastic():
    """
    홉필드(정해진)와 볼츠만(확률) 움직임을 견준다.
    """
    print("="*70)
    print("DETERMINISTIC VS STOCHASTIC DYNAMICS")
    print("="*70)
    
    # 단순한 무늬
    pattern = torch.tensor([1., 1., 1., 0., 0., 0.])
    n_units = len(pattern)
    
    # 무게 행렬을 만든다(헤브)
    pattern_pm = 2 * pattern - 1
    W = torch.outer(pattern_pm, pattern_pm)
    W.fill_diagonal_(0)
    
    # 잡음 낀 시작점
    noisy = pattern.clone()
    noisy[2] = 0
    noisy[4] = 1
    
    print(f"Target pattern: {pattern.numpy().astype(int)}")
    print(f"Noisy input:    {noisy.numpy().astype(int)}")
    
    # 정해진 움직임
    print("\n" + "-"*50)
    print("DETERMINISTIC (Hopfield-style)")
    state_det = noisy.clone()
    det_trajectory = [state_det.clone()]
    
    for _ in range(10):
        # ±1으로 바꾼다
        s = 2 * state_det - 1
        # 그 자리의 마당
        h = torch.mv(W, s)
        # 정해진 고침(아무 차례)
        for i in torch.randperm(n_units):
            s[i] = torch.sign(h[i]) if h[i] != 0 else s[i]
            h = torch.mv(W, s)
        
        state_det = (s + 1) / 2
        det_trajectory.append(state_det.clone())
        
        if (det_trajectory[-1] == det_trajectory[-2]).all():
            break
    
    print(f"Converged to: {state_det.numpy().astype(int)}")
    print(f"Iterations: {len(det_trajectory) - 1}")
    
    # 확률 움직임
    print("\n" + "-"*50)
    print("STOCHASTIC (Boltzmann)")
    
    bm = BoltzmannMachine(n_visible=n_units, temperature=1.0)
    bm.W = W
    
    # 분포를 보이려 여러 번 돌린다
    n_runs = 100
    final_states = []
    
    for _ in range(n_runs):
        state_stoch = noisy.clone()
        for _ in range(50):  # More iterations
            state_stoch = bm.gibbs_step(state_stoch)
        final_states.append(tuple(state_stoch.numpy().astype(int)))
    
    # 결과를 센다
    from collections import Counter
    outcomes = Counter(final_states)
    
    print(f"Outcomes over {n_runs} runs:")
    for state, count in outcomes.most_common():
        print(f"  {state}: {count/n_runs*100:.1f}%")
    
    # 시각화한다
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # 정해진 자취
    traj_array = torch.stack(det_trajectory).numpy()
    axes[0].imshow(traj_array.T, cmap='binary', aspect='auto')
    axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel('Neuron')
    axes[0].set_title('Deterministic: Single Path to Fixed Point')
    
    # 확률 도수 그림
    unique_states = list(outcomes.keys())
    counts = [outcomes[s] for s in unique_states]
    axes[1].bar(range(len(unique_states)), counts)
    axes[1].set_xticks(range(len(unique_states)))
    axes[1].set_xticklabels([str(s) for s in unique_states], rotation=45, ha='right')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Stochastic: Distribution of Final States')
    
    # 온도 견주기
    temperatures = [0.1, 0.5, 1.0, 2.0, 5.0]
    entropies = []
    
    for T in temperatures:
        bm_temp = BoltzmannMachine(n_visible=n_units, temperature=T)
        bm_temp.W = W
        
        # 빠른 뽑기
        states = []
        state = noisy.clone()
        for _ in range(500):
            state = bm_temp.gibbs_step(state)
            states.append(tuple(state.numpy().astype(int)))
        
        # 엔트로피 셈하기
        counts = Counter(states)
        probs = np.array(list(counts.values())) / 500
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        entropies.append(entropy)
    
    axes[2].plot(temperatures, entropies, 'bo-', markersize=8, linewidth=2)
    axes[2].set_xlabel('Temperature')
    axes[2].set_ylabel('Entropy of Sampled States')
    axes[2].set_title('Higher Temperature → More Exploration')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

compare_deterministic_vs_stochastic()
```

**보존된 출력:**

```
======================================================================
DETERMINISTIC VS STOCHASTIC DYNAMICS
======================================================================
Target pattern: [1 1 1 0 0 0]
Noisy input:    [1 1 0 0 1 0]

--------------------------------------------------
DETERMINISTIC (Hopfield-style)
Converged to: [1 1 1 0 0 0]
Iterations: 2

--------------------------------------------------
STOCHASTIC (Boltzmann)
Outcomes over 100 runs:
  (1, 1, 1, 0, 0, 0): 74.0%
  (0, 0, 0, 1, 1, 1): 22.0%
  (1, 1, 1, 0, 1, 0): 2.0%
  (0, 1, 1, 0, 0, 0): 1.0%
  (1, 1, 1, 1, 0, 0): 1.0%
```

energy와 local_field는 0·1 상태를 s=2b−1로 바꾼다. 그러나 sample_probability는 sigmoid(h/T)를 사용해 위 스핀 조건부 확률의 **2가 빠져 있다**. 0·1 상태를 유지하는 사슬이라면 표시된 에너지에 대해 지정 온도 T가 아니라 **유효 온도 2T**의 볼츠만 분포에 대응한다.

따라서 표시한 에너지의 exp(−E/T) 분포를 그대로 샘플링한다고 설명할 수 없다. 낮은 온도에서 부호 갱신과 비슷해지는 정성적 비교는 가능하지만 온도별 정량 비교에는 이 차이를 반영해야 한다. “결정적” 비교 쪽도 갱신 순서를 무작위로 뽑으므로 단위의 선택 규칙까지 고정한 단일 경로는 아니다.

## 3. 상태 표현과 코드 범위

상태의 최솟값이 비음수면 0·1 상태로 간주하는 자동 판별은 명시적 상태 규약을 대신하지 못한다. sample_unit은 항상 0 또는 1을 쓰므로 입력·유지 상태도 0·1로 사용해야 한다. −1·+1 입력을 주면 일부 단위만 0·1로 바뀌어 혼합 표현이 될 수 있다.

n_hidden은 전체 단위 수를 늘리지만 관측 단위를 고정하는 positive phase나 학습 함수를 구현하지 않는다. W와 theta는 Parameter가 아닌 buffer이므로 일반 optimizer에 학습 파라미터로 등록되지 않는다. 이 클래스는 고정 에너지의 상태 갱신 예제다.

여러 torch.rand 호출과 초기 상태는 CPU에서 만들어지고 empirical_probs는 .numpy()를 사용한다. 모델을 CUDA로 옮겼다고 모든 초기화·난수·후처리가 자동으로 CUDA에 맞는 것은 아니다. 배치 학습용 인터페이스도 아니며 단일 상태 벡터를 갱신한다.

## 4. 평형 검증의 조건

```python
def verify_equilibrium_distribution():
    """
    깁스 뽑기가 볼츠만 분포로 모임을 확인한다.
    """
    print("="*70)
    print("VERIFYING EQUILIBRIUM DISTRIBUTION")
    print("="*70)
    
    # 늘어놓기를 위한 작은 신경망
    n_units = 4
    bm = BoltzmannMachine(n_visible=n_units, temperature=1.0)
    
    # 정해진 무게를 둔다
    bm.W = torch.tensor([
        [0.0, 1.5, -1.0, 0.5],
        [1.5, 0.0, 0.5, -1.0],
        [-1.0, 0.5, 0.0, 1.5],
        [0.5, -1.0, 1.5, 0.0]
    ])
    bm.theta = torch.tensor([0.5, -0.5, 0.0, 0.5])
    
    # 이론 분포를 셈한다(샅샅이 늘어놓기)
    theoretical_probs = {}
    energies = {}
    
    for i in range(2**n_units):
        binary = format(i, f'0{n_units}b')
        state = torch.tensor([float(b) for b in binary])
        
        energy = bm.energy(state).item()
        energies[tuple(state.numpy().astype(int))] = energy
    
    # 나눔 함수를 셈한다
    Z = sum(np.exp(-E / bm.temperature) for E in energies.values())
    
    for state, E in energies.items():
        theoretical_probs[state] = np.exp(-E / bm.temperature) / Z
    
    # 깁스 뽑기로 경험으로 어림한다
    results = bm.estimate_distribution(n_samples=5000, burn_in=500, thin=5)
    
    # 비교
    print("\nComparison of theoretical vs empirical probabilities:")
    print("-"*70)
    print(f"{'State':<20} {'Theoretical':>15} {'Empirical':>15} {'Energy':>10}")
    print("-"*70)
    
    sorted_states = sorted(theoretical_probs.keys(), 
                           key=lambda s: theoretical_probs[s], reverse=True)
    
    for state in sorted_states:
        theo = theoretical_probs[state]
        emp = results['empirical_probs'].get(state, 0)
        E = energies[state]
        print(f"{str(state):<20} {theo:>15.4f} {emp:>15.4f} {E:>10.2f}")
    
    # 얽힘을 셈한다
    theo_vals = []
    emp_vals = []
    for state in theoretical_probs:
        theo_vals.append(theoretical_probs[state])
        emp_vals.append(results['empirical_probs'].get(state, 0))
    
    correlation = np.corrcoef(theo_vals, emp_vals)[0, 1]
    print(f"\nCorrelation: {correlation:.4f}")
    
    # 시각화한다
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 흩뿌림 그림
    axes[0].scatter(theo_vals, emp_vals, alpha=0.7, s=100)
    max_val = max(max(theo_vals), max(emp_vals))
    axes[0].plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Perfect match')
    axes[0].set_xlabel('Theoretical Probability')
    axes[0].set_ylabel('Empirical Probability')
    axes[0].set_title(f'Gibbs Sampling Accuracy (corr={correlation:.4f})')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 에너지 도수 그림
    axes[1].hist(results['energies'], bins=30, density=True, alpha=0.7, 
                 edgecolor='black', label='Sampled')
    
    # 이론 에너지 분포
    E_range = np.linspace(min(results['energies']), max(results['energies']), 100)
    # 이는 어림이다 - 제대로 가장자리로 몰아내야 한다
    axes[1].set_xlabel('Energy')
    axes[1].set_ylabel('Density')
    axes[1].set_title('Energy Distribution from Gibbs Sampling')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return correlation

correlation = verify_equilibrium_distribution()
```

**보존된 출력:**

```
======================================================================
VERIFYING EQUILIBRIUM DISTRIBUTION
======================================================================

Comparison of theoretical vs empirical probabilities:
----------------------------------------------------------------------
State                    Theoretical       Empirical     Energy
----------------------------------------------------------------------
(0, 0, 1, 1)                  0.5771          0.2932      -4.50
(1, 1, 0, 0)                  0.2123          0.1830      -3.50
(1, 1, 1, 1)                  0.0781          0.1042      -2.50
(0, 0, 0, 0)                  0.0287          0.0694      -1.50
(1, 0, 0, 1)                  0.0287          0.0634      -1.50
(1, 0, 1, 1)                  0.0287          0.0674      -1.50
(0, 0, 0, 1)                  0.0106          0.0380      -0.50
(1, 0, 0, 0)                  0.0106          0.0410      -0.50
(1, 1, 0, 1)                  0.0106          0.0398      -0.50
(0, 0, 1, 0)                  0.0039          0.0218       0.50
(0, 1, 1, 1)                  0.0039          0.0218       0.50
(1, 1, 1, 0)                  0.0039          0.0242       0.50
(0, 1, 0, 0)                  0.0014          0.0152       1.50
(0, 1, 1, 0)                  0.0014          0.0148       1.50
(1, 0, 1, 0)                  0.0000          0.0016       5.50
(0, 1, 0, 1)                  0.0000          0.0012       6.50

Correlation: 0.9533
```

4개 단위의 16개 상태를 열거해 exp(−E/T)의 이론값을 계산한다. 그러나 앞서 설명한 샘플러는 같은 에너지의 온도 2T 조건부를 사용하므로 **더 오래 실행해도 이 비교의 목표가 자동으로 일치하지 않는다**. 출력의 높은 상관계수만으로 목표 분포가 정확하다고 결론내리지 않는다.

올바른 단위 조건부분포를 사용하면 각 단위의 깁스 전이가 목표분포를 불변으로 유지한다. 유한 상태, 양의 유한 T와 유한 에너지에서 상태 간 도달 가능성과 자기 전이가 있으면 적절한 스캔 사슬은 평형에 수렴한다. 상세균형은 충분한 불변성 조건 중 하나지만 그것만으로 임의의 사슬의 유일한 극한을 보장하지 않는다. 고정 순서 전이의 합성은 불변성을 유지해도 전체가 항상 가역인 것은 아니다.

burn-in과 thinning을 지정해도 독립 표본이나 혼합 완료를 보장하지 않는다. 여러 초기점, 자기상관과 분포 차이를 함께 살펴야 한다. 확률 벡터의 상관계수는 보정된 확률의 차이나 총변동거리와 같은 양이 아니다.

## 5. 온도 실험 해석

```python
def analyze_temperature_effects():
    """
    온도가 뽑기 움직임에 어떤 영향을 주는지 살핀다.
    """
    n_units = 6
    base_bm = BoltzmannMachine(n_visible=n_units, temperature=1.0)
    
    # 재미있는 에너지 풍경이 되도록 무게를 둔다
    W = torch.randn(n_units, n_units) * 0.5
    W = (W + W.T) / 2
    W.fill_diagonal_(0)
    base_bm.W = W
    
    temperatures = [0.1, 0.5, 1.0, 2.0, 5.0]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    for idx, T in enumerate(temperatures):
        bm = BoltzmannMachine(n_visible=n_units, temperature=T)
        bm.W = W.clone()
        bm.theta = base_bm.theta.clone()
        
        # 뽑기
        n_samples = 2000
        state = (torch.rand(n_units) > 0.5).float()
        energies = []
        
        for _ in range(n_samples):
            state = bm.gibbs_step(state)
            energies.append(bm.energy(state).item())
        
        # 그림
        ax = axes[idx]
        ax.hist(energies, bins=30, density=True, alpha=0.7, 
                edgecolor='black', color='steelblue')
        ax.set_xlabel('Energy', fontsize=11)
        ax.set_ylabel('Density', fontsize=11)
        ax.set_title(f'T = {T}', fontsize=13, fontweight='bold')
        
        # 통계
        mean_E = np.mean(energies)
        std_E = np.std(energies)
        ax.axvline(mean_E, color='red', linestyle='--', linewidth=2)
        ax.text(0.95, 0.95, f'μ = {mean_E:.2f}\nσ = {std_E:.2f}',
                transform=ax.transAxes, ha='right', va='top',
                fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
        ax.grid(True, alpha=0.3)
    
    axes[-1].axis('off')
    
    plt.suptitle('Temperature Effects on Boltzmann Machine Sampling', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    print("\nObservations:")
    print("• Low T → Concentrated at low energies (exploitation)")
    print("• High T → Spread across energy levels (exploration)")
    print("• T controls the trade-off between exploitation and exploration")

analyze_temperature_effects()
```

**보존된 출력:**

```
Observations:
• Low T → Concentrated at low energies (exploitation)
• High T → Spread across energy levels (exploration)
• T controls the trade-off between exploitation and exploration
```

고정 W에서 낮은 T는 낮은 에너지 상태에 더 집중하고 높은 T는 확률비를 완만하게 만든다. 다만 이 코드에서는 유효 온도 2T를 반영해야 한다. 표본 엔트로피는 유한한 상관 표본으로 추정하며 초기 상태와 혼합 정도의 영향도 받는다. 각 에너지 히스토그램의 높이를 상태 하나의 확률로 읽으면 같은 에너지를 가진 여러 상태의 축퇴도를 놓칠 수 있다.

## 6. 은닉변수와 학습 원리

가시변수 v와 은닉변수 h를 합쳐 s라 두면 대칭 결합 에너지를 그대로 적용할 수 있다. 같은 층 결합의 행렬식에는 1/2 계수가 들어가고 두 층 사이 항은 한 번만 센다. 주어진 v에서 은닉변수를 합해 자유 에너지를 정의한다.

$$F_T(v)=-T\log\sum_h e^{-E(v,h)/T},\qquad
p(v)=e^{-F_T(v)/T}/Z$$

독립적인 대칭 결합 Wᵢⱼ(i<j)의 로그 우도 기울기는 다음 통계 차이다.

$$\frac{\partial\log p(v)}{\partial W_{ij}}
=\frac1T\left(\mathbb E[s_i s_j\mid v]
-\mathbb E_p[s_i s_j]\right)$$

첫 항은 관측값을 고정한 positive phase이고 둘째는 모델 전체의 negative phase다. 일반 볼츠만 기계는 두 항의 추론이 모두 어려울 수 있다. [RBM](rbm.md)은 같은 층 연결을 제거해 조건부분포의 계산을 단순화하지만 정규화 상수 전체가 자동으로 쉬워지는 것은 아니다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1. 조건부 확률**

스핀 에너지에서 sigmoid의 인수에 2가 필요한 이유를 유도하라.

</div>

??? success "연습문제 1 풀이"
    단위 i와 관련된 에너지는 −sᵢhᵢ다. +1과 −1의 에너지 차이는 −2hᵢ이므로 확률비가 exp(2hᵢ/T)가 되고 정규화하면 sigmoid(2hᵢ/T)다. sigmoid(hᵢ/T)는 같은 에너지에서 온도 2T의 조건부다.

---

<div class="drillbox" markdown>

**연습문제 2. 열거 검증**

16개 상태의 exp(−E/T)와 샘플러 빈도가 다르면 반복 횟수만 늘리면 되는가?

</div>

??? success "연습문제 2 풀이"
    먼저 조건부 확률과 목표 에너지의 온도·변수 관례를 맞춰야 한다. 현재 구현에는 계수 2 차이가 있어 사슬 길이만 늘려 해결되지 않는다. 그다음에 초기화·burn-in·자기상관과 표본 오차를 점검한다.

---

<div class="drillbox" markdown>

**연습문제 3. 은닉변수의 역할**

n_hidden을 늘리는 것만으로 관측 데이터의 볼츠만 기계 학습이 구현되는가?

</div>

??? success "연습문제 3 풀이"
    아니다. 전체 상태 차원은 커지지만 데이터 고정 positive phase, 모델 negative phase와 파라미터 업데이트가 없다. W와 theta도 buffer다. 은닉변수에 대한 조건부 추론과 관측 주변 우도의 학습 절차가 별도로 필요하다.

## 정리하며

볼츠만 기계의 확률적 갱신은 에너지와 변수 표현에 맞는 조건부 확률을 사용해야 한다. 스핀 에너지의 인수 2와 유효 온도를 확인하고, 단순 상태 갱신·평형 추정과 은닉변수를 포함한 우도 학습을 구분한다.

**참고 문헌**

- Hinton, G. E., & Sejnowski, T. J. (1986). Learning and relearning in Boltzmann machines. In Parallel Distributed Processing.
- Ackley, D. H., Hinton, G. E., & Sejnowski, T. J. (1985). A learning algorithm for Boltzmann machines. Cognitive Science.
- Salakhutdinov, R. (2015). Learning Deep Generative Models. Annual Review of Statistics.
