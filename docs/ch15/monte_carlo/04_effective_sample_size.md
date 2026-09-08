# 실효 표본 크기

04_effective_sample_size.py 중간 수준: 실효 표본 크기(ESS) — 이론과 진단

확률 과정과 표집 방법은 확률적 기계 학습의 바탕 도구이다. 이 모듈은 몬테카를로 방법의 개념을 보이며 수학 이론과 셈 구현 사이의 틈을 잇는다.

## 1. 코드

```python
"""
04_effective_sample_size.py

중급 단계: 실효 표본 크기(ESS) - 이론과 진단

이 단원은 ESS을 두루 다룬다:
- 수학의 바탕
- ESS의 여러 가지 표현
- 진단과 풀이
- 흩어짐과의 관계

수학적 바탕:
---------------------
ESS은 독립 표본의 "실효" 개수를 잰다
중요도 표집으로 얻은.

표준 정의:
    ESS = 1 / Σᵢ wᵢ²

여기서 wᵢ은 고르게 한 무게이다.

다른 표현(고르게 하지 않은 무게 사용):
    ESS = (Σᵢ w̃ᵢ)² / Σᵢ w̃ᵢ²

성질:
- 1 ≤ ESS ≤ n
- 무게가 모두 같으면 ESS = n이다(완벽한 표집)
- 무게 하나가 판치면 ESS = 1이다(찌부러짐)
- ESS/n은 "상대 효율"이다

흩어짐과의 관계:
    Var[Ê] ≈ Var_π[h(θ)] / ESS

따라서 ESS이 낮을수록 → 어림값의 흩어짐이 커진다.

지은이: 베이즈 추론 교육 자료
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
from typing import Tuple, Dict

np.random.seed(42)
sns.set_style("whitegrid")


def compute_ess_normalized(weights: np.ndarray) -> float:
    """
    고르게 한 무게를 써서 ESS 셈하기.
    
    ESS = 1 / Σᵢ wᵢ²
    
    매개변수:
    -----------
    weights : 고르게 한 무게(합이 1)
    
    반환값:
    --------
    ess : float
    """
    return 1.0 / np.sum(weights**2)


def compute_ess_unnormalized(unnormalized_weights: np.ndarray) -> float:
    """
    고르게 하지 않은 무게를 써서 ESS 셈하기.
    
    ESS = (Σᵢ w̃ᵢ)² / Σᵢ w̃ᵢ²
    
    이는 고르게 한 판과 같지만 수치로 더 안정하다
    무게가 아주 작을 때.
    
    매개변수:
    -----------
    unnormalized_weights : 고르게 하지 않은 무게
    
    반환값:
    --------
    ess : float
    """
    sum_weights = np.sum(unnormalized_weights)
    sum_weights_squared = np.sum(unnormalized_weights**2)
    return sum_weights**2 / sum_weights_squared


def compute_weight_statistics(weights: np.ndarray) -> Dict:
    """
    중요도 무게의 통계량을 두루 셈하기.
    
    여러 진단을 담은 사전을 돌려준다.
    """
    n = len(weights)
    ess = compute_ess_normalized(weights)
    
    # 무게의 변이 계수
    cv = np.std(weights) / (np.mean(weights) + 1e-10)
    
    # 최대 무게
    max_weight = np.max(weights)
    
    # 무게 분포의 엔트로피
    # 엔트로피가 높을수록 → 무게가 더 고르다
    entropy = -np.sum(weights * np.log(weights + 1e-300))
    max_entropy = np.log(n)  # 고른 분포
    normalized_entropy = entropy / max_entropy
    
    # 헷갈림도(ESS과 이어진 또 하나의 잣대)
    perplexity = np.exp(entropy)
    
    # 위쪽 표본이 차지하는 무게의 백분율
    sorted_weights = np.sort(weights)[::-1]
    cumsum_weights = np.cumsum(sorted_weights)
    top_10_pct = np.searchsorted(cumsum_weights, 0.10) + 1
    top_50_pct = np.searchsorted(cumsum_weights, 0.50) + 1
    top_90_pct = np.searchsorted(cumsum_weights, 0.90) + 1
    
    return {
        'ess': ess,
        'relative_ess': ess / n,
        'cv': cv,
        'max_weight': max_weight,
        'entropy': entropy,
        'normalized_entropy': normalized_entropy,
        'perplexity': perplexity,
        'top_10_pct': top_10_pct,
        'top_50_pct': top_50_pct,
        'top_90_pct': top_90_pct,
        'n_samples': n
    }


def estimate_variance_inflation(weights: np.ndarray) -> float:
    """
    흩어짐 부풂 배수 어림하기.
    
    Var[중요도 표집 어림꼴] ≈ Var[몬테카를로 어림꼴] × (1 + Var[w])
    
    여기서 w은 고르게 하지 않은 중요도 비이다.
    
    반환값:
    --------
    inflation : float
        흩어짐이 부푸는 배수
    """
    ess = compute_ess_normalized(weights)
    n = len(weights)
    # 흩어짐 부풂 ≈ n/ESS
    return n / ess


# 보기 1: 제안 분포에 따른 ESS
# =================================================
print("=" * 70)
print("EXAMPLE 1: ESS Depends on Proposal Quality")
print("=" * 70)

# 과녁 분포: N(5, 1)
target = stats.norm(5, 1)

# 여러 가지 제안 분포
proposals = {
    'Perfect': stats.norm(5, 1),           # 과녁과 같음
    'Good': stats.norm(5, 1.2),            # 가까움
    'Okay': stats.norm(4.5, 1.5),          # 그럴듯함
    'Poor': stats.norm(3, 2),              # 어긋남
    'Bad': stats.norm(5, 0.5),             # 너무 좁음
    'Terrible': stats.norm(0, 1),          # 아주 멂
}

n_samples = 2000
print(f"\nAnalyzing ESS for {n_samples} samples:\n")
print(f"{'Proposal':<12} {'ESS':>8} {'Rel ESS':>8} {'CV':>8} {'Entropy':>8} {'Top 10%':>8}")
print("-" * 70)

results_dict = {}

for name, proposal in proposals.items():
    # 제안에서 표집
    samples = proposal.rvs(size=n_samples)
    
    # 무게 셈하기
    unnorm_weights = target.pdf(samples) / proposal.pdf(samples)
    weights = unnorm_weights / np.sum(unnorm_weights)
    
    # 통계를 셈한다
    stats_dict = compute_weight_statistics(weights)
    results_dict[name] = stats_dict
    
    print(f"{name:<12} {stats_dict['ess']:8.1f} {stats_dict['relative_ess']:8.2%} "
          f"{stats_dict['cv']:8.2f} {stats_dict['normalized_entropy']:8.2%} "
          f"{stats_dict['top_10_pct']:8d}")


# 가중치 분포 시각화
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.ravel()

for idx, (name, proposal) in enumerate(proposals.items()):
    samples = proposal.rvs(size=n_samples)
    unnorm_weights = target.pdf(samples) / proposal.pdf(samples)
    weights = unnorm_weights / np.sum(unnorm_weights)
    
    ax = axes[idx]
    ax.hist(weights * n_samples, bins=50, density=True, alpha=0.7,
            color='steelblue', edgecolor='black', linewidth=0.5)
    
    ess = compute_ess_normalized(weights)
    ax.set_title(f'{name}: ESS={ess:.1f} ({ess/n_samples:.1%})',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('Normalized Weight × n', fontsize=10)
    ax.set_ylabel('Density', fontsize=10)
    ax.axvline(1.0, color='red', linestyle='--', linewidth=2, alpha=0.7,
               label='Uniform')
    ax.grid(True, alpha=0.3)
    if idx == 0:
        ax.legend()

plt.tight_layout()
plt.savefig('/home/claude/03_Importance_Sampling/ess_proposal_comparison.png',
            dpi=300, bbox_inches='tight')


# 보기 2: ESS과 흩어짐의 관계
# ==============================================
print("\n" + "=" * 70)
print("EXAMPLE 2: ESS and Estimation Variance")
print("=" * 70)

# 과녁: N(5, 1)
# E[θ²] 어림하기
h_function = lambda x: x**2
true_value = 5**2 + 1**2  # N(5,1)의 E[θ²]

# ESS이 다른 여러 제안
proposals_var = {
    'High ESS': stats.norm(5, 1.1),
    'Medium ESS': stats.norm(4, 1.5),
    'Low ESS': stats.norm(2, 2),
}

n_samples = 1000
n_replications = 500

print(f"\nEstimating E[θ²] = {true_value:.3f}")
print(f"Replications: {n_replications}, Samples per replication: {n_samples}\n")
print(f"{'Proposal':<12} {'ESS':>8} {'Bias':>10} {'Std Dev':>10} {'RMSE':>10}")
print("-" * 60)

for name, proposal in proposals_var.items():
    estimates = []
    ess_values = []
    
    for _ in range(n_replications):
        samples = proposal.rvs(size=n_samples)
        unnorm_weights = target.pdf(samples) / proposal.pdf(samples)
        weights = unnorm_weights / np.sum(unnorm_weights)
        
        estimate = np.sum(weights * h_function(samples))
        estimates.append(estimate)
        ess_values.append(compute_ess_normalized(weights))
    
    mean_ess = np.mean(ess_values)
    bias = np.mean(estimates) - true_value
    std_dev = np.std(estimates)
    rmse = np.sqrt(np.mean((np.array(estimates) - true_value)**2))
    
    print(f"{name:<12} {mean_ess:8.1f} {bias:+10.4f} {std_dev:10.4f} {rmse:10.4f}")

# 흩어짐과 ESS 그려 보기
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 흩뿌림 그림: ESS과 표준편차(경험값)
ess_vs_std = []
for name, proposal in proposals_var.items():
    estimates = []
    ess_values = []
    
    for _ in range(100):
        samples = proposal.rvs(size=n_samples)
        unnorm_weights = target.pdf(samples) / proposal.pdf(samples)
        weights = unnorm_weights / np.sum(unnorm_weights)
        
        estimate = np.sum(weights * h_function(samples))
        estimates.append(estimate)
        ess_values.append(compute_ess_normalized(weights))
    
    for e, s in zip(ess_values, estimates):
        ess_vs_std.append((e, s))

ess_array = np.array([x[0] for x in ess_vs_std])
std_array = np.array([x[1] for x in ess_vs_std])

ax = axes[0]
ax.scatter(ess_array, std_array, alpha=0.5, s=30, color='steelblue', edgecolors='black')
ax.set_xlabel('ESS', fontsize=12)
ax.set_ylabel('Estimate', fontsize=12)
ax.set_title('ESS vs Estimate Distribution', fontsize=13, fontweight='bold')
ax.axhline(true_value, color='red', linestyle='--', linewidth=2, label='True value')
ax.grid(True, alpha=0.3)
ax.legend()

# 이론의 흩어짐 부풂
ax = axes[1]
ess_range = np.linspace(10, n_samples, 100)
variance_inflation = n_samples / ess_range

ax.plot(ess_range, variance_inflation, 'b-', linewidth=2,
        label='Variance inflation = n/ESS')
ax.set_xlabel('ESS', fontsize=12)
ax.set_ylabel('Variance Inflation Factor', fontsize=12)
ax.set_title('Theoretical Variance Inflation', fontsize=13, fontweight='bold')
ax.axhline(1, color='red', linestyle='--', linewidth=2, label='No inflation')
ax.grid(True, alpha=0.3)
ax.legend()
ax.set_yscale('log')

plt.tight_layout()
plt.savefig('/home/claude/03_Importance_Sampling/ess_variance_relationship.png',
            dpi=300, bbox_inches='tight')


# 보기 3: 표본 크기의 함수로 본 ESS
# =======================================
print("\n" + "=" * 70)
print("EXAMPLE 3: How ESS Scales with Sample Size")
print("=" * 70)

# 붙박인 제안에서 ESS은 n에 따라 어떻게 자라나?
proposal_fixed = stats.norm(4, 1.5)
sample_sizes = [100, 500, 1000, 2000, 5000, 10000]

print("\nSample Size vs ESS (averaged over 100 runs):\n")
print(f"{'n':>8} {'Mean ESS':>10} {'ESS/n':>10} {'Std ESS':>10}")
print("-" * 42)

ess_by_n = []
for n in sample_sizes:
    ess_list = []
    for _ in range(100):
        samples = proposal_fixed.rvs(size=n)
        unnorm_weights = target.pdf(samples) / proposal_fixed.pdf(samples)
        weights = unnorm_weights / np.sum(unnorm_weights)
        ess_list.append(compute_ess_normalized(weights))
    
    mean_ess = np.mean(ess_list)
    std_ess = np.std(ess_list)
    ess_by_n.append((n, mean_ess, std_ess))
    
    print(f"{n:8d} {mean_ess:10.1f} {mean_ess/n:10.3f} {std_ess:10.1f}")

# n에 따른 ESS 그리기
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ns = [x[0] for x in ess_by_n]
ess_means = [x[1] for x in ess_by_n]
ess_stds = [x[2] for x in ess_by_n]
rel_ess = [e/n for e, n in zip(ess_means, ns)]

ax = axes[0]
ax.errorbar(ns, ess_means, yerr=ess_stds, fmt='o-', linewidth=2,
            markersize=8, capsize=5, color='steelblue', label='ESS')
ax.plot(ns, ns, 'r--', linewidth=2, label='Perfect (ESS=n)')
ax.set_xlabel('Sample Size n', fontsize=12)
ax.set_ylabel('ESS', fontsize=12)
ax.set_title('ESS vs Sample Size', fontsize=13, fontweight='bold')
ax.set_xscale('log')
ax.set_yscale('log')
ax.grid(True, alpha=0.3)
ax.legend()

ax = axes[1]
ax.plot(ns, rel_ess, 'o-', linewidth=2, markersize=8, color='steelblue')
ax.set_xlabel('Sample Size n', fontsize=12)
ax.set_ylabel('Relative ESS (ESS/n)', fontsize=12)
ax.set_title('Relative ESS vs Sample Size', fontsize=13, fontweight='bold')
ax.axhline(1.0, color='red', linestyle='--', linewidth=2, label='Perfect')
ax.set_xscale('log')
ax.grid(True, alpha=0.3)
ax.legend()

plt.tight_layout()
plt.savefig('/home/claude/03_Importance_Sampling/ess_vs_sample_size.png',
            dpi=300, bbox_inches='tight')


# 보기 4: 무게 몰림 진단
# ========================================
print("\n" + "=" * 70)
print("EXAMPLE 4: Weight Concentration Analysis")
print("=" * 70)

def analyze_weight_concentration(weights: np.ndarray, name: str):
    """
    무게가 어떻게 몰리는지 자세히 살피기.
    """
    n = len(weights)
    sorted_weights = np.sort(weights)[::-1]
    cumsum = np.cumsum(sorted_weights)
    
    # 백분위수 찾기
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    n_for_percentile = []
    
    for p in percentiles:
        idx = np.searchsorted(cumsum, p/100.0)
        n_for_percentile.append(idx + 1)
    
    print(f"\n{name}:")
    print(f"  Total samples: {n}")
    print(f"  ESS: {compute_ess_normalized(weights):.1f}")
    print("\n  Weight concentration:")
    for p, n_samples in zip(percentiles, n_for_percentile):
        pct_samples = n_samples / n * 100
        print(f"    {n_samples:5d} samples ({pct_samples:5.1f}%) account for {p}% of weight")
    
    return sorted_weights, cumsum

# 제안 셋 견주기
test_proposals = {
    'Good (ESS high)': stats.norm(5, 1.1),
    'Medium (ESS mid)': stats.norm(4, 1.5),
    'Poor (ESS low)': stats.norm(2, 2),
}

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for idx, (name, proposal) in enumerate(test_proposals.items()):
    samples = proposal.rvs(size=2000)
    unnorm_weights = target.pdf(samples) / proposal.pdf(samples)
    weights = unnorm_weights / np.sum(unnorm_weights)
    
    sorted_weights, cumsum = analyze_weight_concentration(weights, name)
    
    ax = axes[idx]
    ax.plot(np.arange(1, len(sorted_weights)+1), cumsum, 
            linewidth=2, color='steelblue')
    ax.axhline(0.5, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.axhline(0.9, color='orange', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.set_xlabel('Number of Top Samples', fontsize=11)
    ax.set_ylabel('Cumulative Weight', fontsize=11)
    ax.set_title(name, fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')

plt.tight_layout()
plt.savefig('/home/claude/03_Importance_Sampling/weight_concentration.png',
            dpi=300, bbox_inches='tight')

plt.show()

print("\n" + "=" * 70)
print("KEY TAKEAWAYS")
print("=" * 70)
print("""
1. 실효 표본 크기(ESS)는 중요도 표집의 질을 재어 준다:
   - ESS = 1/Σᵢwᵢ²(고르게 한 무게)
   - ESS = (Σᵢw̃ᵢ)²/Σᵢw̃ᵢ²(고르게 하지 않은 무게)
   - 범위: 1 ≤ ESS ≤ n

2. 풀이:
   - ESS/n ≈ 1: 아주 좋음, 무게가 거의 고르다
   - ESS/n ≈ 0.5: 좋음, 실효 표본이 절반
   - ESS/n < 0.1: 나쁨, 더 나은 제안을 생각해 보라
   - ESS/n << 0.01: 나쁨, 몇몇 표본이 판친다

3. 흩어짐과의 관계:
   - 흩어짐 부풂 ≈ n/ESS
   - ESS이 낮을수록 → 어림값의 흩어짐이 커진다
   - 제안을 낫게 하지 않고 n만 늘려서는 고칠 수 없다

4. 무게 몰림:
   - ESS이 낮다 → 몇몇 표본이 무게의 대부분을 진다
   - 살피기: 무게의 50%에 표본이 몇 개나 드나?
   - 되도록 여러 표본에 널리 퍼져 있어야 한다

5. 늘 살펴야 할 진단:
   - ESS과 상대 ESS(ESS/n)
   - 무게의 변이 계수
   - 무게 몰림(위쪽 10%, 50%, 90%)
   - 최대 무게 값
   - 무게 엔트로피

6. ESS이 낮을 때:
   - n만 늘리지 마라(별 도움이 안 된다)
   - 제안 분포 낫게 하기
   - 맞춰 가는 중요도 표집 생각해 보기
   - 아니면 MCMC 방법으로 바꾸기

7. ESS은 n에 비례해 커진다:
   - 표본 n개에서 ESS/n ≈ c이면
   - 그러면 아무 n에 대해서도 (대체로) ESS/n ≈ c이다
   - 상대 효율은 대체로 일정하다

8. 실전 규칙:
   - ESS > 1000: 대개 웬만한 쓰임새에 넉넉하다
   - ESS/n > 0.1: 받아들일 만한 효율
   - ESS/n < 0.01: 반드시 더 나은 제안이 필요하다
""")


if __name__ == "__main__":
    pass
```

## 2. 논의

시각화는 모델의 거동을 이해하고 학습 문제를 진단하는 데 중요한 역할을 한다. 그림을 그리는 코드는 학습된 표현, 수렴의 움직임, 평가 지표에 대한 통찰을 주어 추상적인 계산을 손에 잡히게 만든다.

여기서 보인 무늬는 더 복잡한 상황으로 자연스럽게 넓어진다. 웃매개변수, 구조의 변형, 서로 다른 자료 묶음을 이리저리 시험해 보면 이해가 깊어지고 표집과 어림 일감에 대한 실전 직관이 쌓인다.

## 연습문제

**연습문제 1.**
코드를 죽 읽고 핵심 설계 결정을 가려내어라. 구체적인 구현 고름 셋을 적고 저마다 왜 몬테카를로 방법에 알맞은지 설명하여라.

??? success "연습문제 1 풀이"
    설계 결정은 구현마다 다르지만 흔히 다음이 포함된다. (1) 활성화 함수의 선택 — ReLU 계열은 포화되지 않는 경사를 주어 학습을 빠르게 한다. (2) 정규화 전략 — 배치 정규화는 내부 공변량 이동을 줄여 학습을 안정시킨다. (3) 잔차 연결 — 있을 경우 건너뛰는 경로를 제공하여 깊은 신경망에서도 경사가 흐르게 한다. 각 선택은 표현력, 계산 비용, 학습 안정성 사이의 절충을 반영한다.

---

**연습문제 2.**
입력이 기대하는 모양과 자료형을 갖는지 확인하도록 주 함수나 클래스에 입력 검증을 추가하라. 잘못된 입력에는 유익한 오류 메시지를 내라.

??? success "연습문제 2 풀이"
    `forward` 메서드(또는 해당 함수)의 첫머리에 다음과 같은 검사를 추가한다. `assert x.dim() == expected_dims, f'Expected {expected_dims}D input, got {x.dim()}D'`와 `assert x.dtype == torch.float32, f'Expected float32, got {x.dtype}'`. 모양을 검증할 때는 중요한 차원을 확인한다. `B, C, H, W = x.shape; assert C == self.expected_channels`. 유익한 오류 메시지는 디버깅 속도를 크게 높이고 코드를 재사용하기에도 더 견고하게 만든다.

---

**연습문제 3.**
이 몬테카를로 방법 구현의 자리에서 치우침과 흩어짐의 주고받음을 설명하여라. 핵심 웃매개변수가 이 주고받음에 어떻게 영향을 주는가?

??? success "연습문제 3 풀이"
    몬테카를로 방법에서 치우침과 흩어짐의 주고받음은 모형의 복잡함과 표본 크기로 드러난다. 더 복잡한 모형(이를테면 섞음 성분이 더 많거나 층이 더 깊은 모형)은 치우침을 줄이지만 흩어짐을 키우며, 자료가 적을 때 특히 그렇다. 핵심 웃매개변수가 이를 다스린다. 앞확률의 세기가 벌주기 노릇을 하고(센 앞확률은 흩어짐을 줄이지만 치우침을 키울 수 있다), 표본 크기가 어림의 정확도에 영향을 주며(표본이 많을수록 흩어짐이 줄고), 모형의 복잡함이 유연함을 정한다. 가장 좋은 균형은 쓸 수 있는 자료의 양과 바탕 분포의 참된 복잡함에 달렸다.

---

**연습문제 4.**
실효 표본 크기 구현을 확인하는 두루 갖춘 시험 함수를 적어라. 빈 입력, 원소 하나짜리 입력, 아주 큰 입력, 그리고 극단적인 값(0, 아주 큰 수)이 든 입력 같은 모서리 경우를 시험하여라.

??? success "연습문제 4 풀이"
    경계 조건을 두루 시험하는 함수를 만든다.
    ```python
    def test_effective sample size():
        model = Effective Sample Size(...)
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

**다룬 것** — 실효 표본 크기

시각화는 모델의 거동을 이해하고 학습 문제를 진단하는 데 중요한 역할을 한다.

앞의 연습문제 4개로 스스로 따져 볼 수 있다.
