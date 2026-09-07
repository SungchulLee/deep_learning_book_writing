# 볼츠만 기계

## 학습 목표

이 절을 마치면 다음을 할 수 있다:

1. 확률 이진 단위와 그 움직임을 이해한다
2. 볼츠만 기계의 깁스 뽑기를 짠다
3. 정해진 에너지 바탕 모델과 확률 에너지 바탕 모델을 가른다
4. 평형 분포와 모임을 살핀다
5. 볼츠만 기계를 요즘 확률 모델과 잇는다

## 들어가며

1985년 힌턴과 세이노스키가 내놓은 볼츠만 기계(BM)는 확률 움직임을 들여와 홉필드 신경망을 넓힌다. 홉필드 신경망은 정해진 방식으로 에너지 최솟값에 모이지만, 볼츠만 기계는 열 잡음을 써서 볼츠만 분포 전체에서 뽑는다. 이 덕에 자료 위의 확률 분포를 배울 수 있는 만들어 내는 모델 노릇을 한다.

## 홉필드에서 볼츠만으로

### 확률로 넓히기

홉필드 신경망과의 핵심 차이는 고침 규칙이다:

| 모델 | 고침 규칙 |
|-------|-------------|
| 홉필드 | $s_i \leftarrow \text{sign}(h_i)$(정해짐) |
| 볼츠만 | $P(s_i = 1) = \sigma(h_i / T)$(확률) |

여기서 $h_i = \sum_j w_{ij} s_j + \theta_i$은 그 자리의 마당이고 $\sigma(x) = 1/(1+e^{-x})$은 시그모이드 함수이다.

### 물리로 풀이하기

- **온도 $T$**: 마구잡이 수준을 다스린다
  - $T \to 0$: 정해진 쪽에 가까워진다(홉필드 같음)
  - $T \to \infty$: 아무 동전 던지기
  - $T = 1$: 여느 볼츠만 기계

- **열 평형**: 여러 번 고치면 분포가 다음으로 모인다:

$$P(\mathbf{s}) = \frac{1}{Z} \exp(-E(\mathbf{s})/T)$$

## 구현

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

## 깁스 뽑기와 평형

### 깁스 뽑기 알고리즘

깁스 뽑기는 조건 분포에서 되풀이해 뽑아 결합 분포의 표본을 만드는 마르코프 사슬 몬테카를로 방법이다:

$$P(s_i = 1 | \mathbf{s}_{-i}) = \sigma\left(\frac{\sum_j w_{ij} s_j + \theta_i}{T}\right)$$

**알고리즘**:

1. 상태를 아무렇게나 첫자리매김한다
2. 걸음마다:
   - 단위 $i$을 고른다(아무렇게나 또는 차례대로)
   - $P(s_i | \mathbf{s}_{-i})$에서 $s_i$을 뽑는다
3. 평형에 이를 때까지 되풀이한다

### 평형으로 모임

너그러운 조건 아래에서 깁스 뽑기는 볼츠만 분포로 모인다:

$$\lim_{t \to \infty} P(\mathbf{s}^{(t)} = \mathbf{s}) = \frac{\exp(-E(\mathbf{s})/T)}{Z}$$

**자세한 균형**이 모임을 보장한다:

$$P(\mathbf{s}) P(\mathbf{s} \to \mathbf{s}') = P(\mathbf{s}') P(\mathbf{s}' \to \mathbf{s})$$

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

## 온도의 영향

### 온도 살피기

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

## 드러난 단위와 숨은 단위

### 단위의 두 갈래

실제로 볼츠만 기계에는 단위가 두 갈래 있다:

- **드러난 단위($\mathbf{v}$)**: 본 자료를 나타낸다
- **숨은 단위($\mathbf{h}$)**: 숨은 얼개를 담는다

에너지 함수는 다음과 같이 된다:

$$E(\mathbf{v}, \mathbf{h}) = -\mathbf{v}^T \mathbf{W}_{\text{vh}} \mathbf{h} - \mathbf{v}^T \mathbf{W}_{\text{vv}} \mathbf{v} - \mathbf{h}^T \mathbf{W}_{\text{hh}} \mathbf{h} - \mathbf{a}^T \mathbf{v} - \mathbf{b}^T \mathbf{h}$$

### 가장자리 분포

드러난 단위에 대한 가장자리 분포:

$$P(\mathbf{v}) = \sum_{\mathbf{h}} P(\mathbf{v}, \mathbf{h}) = \frac{1}{Z} \sum_{\mathbf{h}} \exp(-E(\mathbf{v}, \mathbf{h}))$$

이것이 **자유 에너지**를 뜻매김한다:

$$F(\mathbf{v}) = -\log \sum_{\mathbf{h}} \exp(-E(\mathbf{v}, \mathbf{h}))$$

따라서 $P(\mathbf{v}) = \frac{1}{Z} \exp(-F(\mathbf{v}))$이다.

## 볼츠만 기계 익히기

### 최대 가능도 목표

자료 $\{\mathbf{v}^{(1)}, \ldots, \mathbf{v}^{(N)}\}$이 주어질 때 다음을 가장 크게 한다:

$$\mathcal{L}(\theta) = \frac{1}{N} \sum_n \log P(\mathbf{v}^{(n)}; \theta)$$

### 로그 가능도의 기울기

기울기는 아름다운 꼴을 가진다:

$$\frac{\partial \log P(\mathbf{v})}{\partial w_{ij}} = \langle s_i s_j \rangle_{\text{data}} - \langle s_i s_j \rangle_{\text{model}}$$

- **양의 국면**: 자료를 붙들어 맨 채의 통계
- **음의 국면**: 모델 표본에서 얻은 통계

### 어려움

$\langle s_i s_j \rangle_{\text{model}}$을 셈하려면 모델에서 뽑아야 하는데, 모든 단위가 서로 매여 있어 일반 볼츠만 기계에서는 느리다.

이것이 **제한 볼츠만 기계**(다음 마디)의 까닭이 된다.

## 핵심 정리

!!! success "핵심 개념"

    1. 볼츠만 기계는 확률 움직임으로 홉필드 신경망을 넓힌다
    2. 깁스 뽑기는 평형에서 볼츠만 분포로 모인다
    3. 온도가 살펴보기와 써먹기 사이를 다스린다
    4. 숨은 단위가 숨은 얼개를 배울 수 있게 한다
    5. 익히기는 자료 통계와 모델 통계의 균형을 잡아야 한다

!!! info "역사에서의 뜻"
    볼츠만 기계는 다음의 바탕을 세웠다:

    - 깊은 믿음 신경망(힌턴, 2006)
    - 제한 볼츠만 기계
    - 요즘 에너지 바탕 모델
    - 변분 자기 부호기(자유 에너지 개념을 거쳐)

## 참고 문헌

- Hinton, G. E., & Sejnowski, T. J. (1986). Learning and relearning in Boltzmann machines. In Parallel Distributed Processing.
- Ackley, D. H., Hinton, G. E., & Sejnowski, T. J. (1985). A learning algorithm for Boltzmann machines. Cognitive Science.
- Salakhutdinov, R. (2015). Learning Deep Generative Models. Annual Review of Statistics.

## 연습문제

1. **섞임 시간**: 신경망 크기와 온도를 달리하며 평형에 이르는 데 깁스 걸음이 몇 번 필요한지 경험으로 어림하라.

2. **식힘**: 높은 $T$에서 시작해 차츰 낮추는 흉내 식힘을 짜라. 온도를 붙박이로 둔 뽑기와 견주어라.

3. **숨은 단위**: 볼츠만 기계에 숨은 단위를 더하고 그것이 드러난 분포의 나타냄 힘에 어떤 영향을 주는지 살펴라.
