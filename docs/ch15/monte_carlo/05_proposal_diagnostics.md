# 제안 진단

05_proposal_diagnostics.py 중간 수준: 두루 갖춘 제안 품질 진단

확률 과정과 표집 방법은 확률적 기계 학습의 바탕 도구이다. 이 모듈은 몬테카를로 방법의 개념을 보이며 수학 이론과 셈 구현 사이의 틈을 잇는다.

## 코드

```python
"""
05_proposal_diagnostics.py

중급 단계: 제안의 질을 두루 진단하기

이 단원은 중요도 표집에서 제안 분포의 질을 살피는
도구와 방법을 준다.

핵심 진단 잣대:
1. 실효 표본 크기(ESS)
2. 무게의 흩어짐과 변이 계수
3. 엔트로피와 헷갈림도
4. 쿨백-라이블러 벌어짐 어림값
5. 덮음 진단
6. χ² 벌어짐 잣대

수학적 바탕:
---------------------
좋은 제안 q은 다음을 만족해야 한다:
- π의 받침을 덮을 것(중요함: 반드시)
- π과 꼴이 비슷할 것(흩어짐에 영향을 준다)
- π보다 꼬리가 두꺼울 것(흩어짐이 무한해지는 것을 막는다)
- 표집하기 쉬울 것(셈의 문제)

가장 좋은 제안(흩어짐을 가장 작게 하려면):
    q*(θ) = |h(θ)|π(θ) / ∫|h(θ)|π(θ)dθ

실전에서는 q이 q*에 얼마나 가까운지 진단으로 살핀다.

지은이: 베이즈 추론 교육 자료
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.special import logsumexp
import seaborn as sns
from typing import Tuple, Dict, Callable

np.random.seed(42)
sns.set_style("whitegrid")


class ProposalDiagnostics:
    """
    중요도 표집 제안을 두루 진단하는 꾸러미.
    """
    
    def __init__(self, target_density: Callable, proposal_dist,
                 name: str = "Proposal"):
        """
        매개변수:
        -----------
        target_density : callable
            과녁 밀도 함수(고르게 하지 않아도 된다)
        proposal_dist : scipy.stats distribution
            제안 분포
        name : str
            알릴 때 쓰는 이름
        """
        self.target_density = target_density
        self.proposal_dist = proposal_dist
        self.name = name
        
    def compute_all_diagnostics(self, n_samples: int = 5000) -> Dict:
        """
        진단 통계량을 두루 셈하기.
        
        모든 잣대를 담은 사전을 돌려준다.
        """
        # 표본 뽑기
        samples = self.proposal_dist.rvs(size=n_samples)
        
        # 무게 셈하기
        target_vals = self.target_density(samples)
        proposal_vals = self.proposal_dist.pdf(samples)
        
        unnorm_weights = target_vals / (proposal_vals + 1e-300)
        weights = unnorm_weights / np.sum(unnorm_weights)
        
        # 진단을 모두 셈하기
        diagnostics = {
            'name': self.name,
            'n_samples': n_samples,
        }
        
        # 1. ESS 잣대
        diagnostics.update(self._compute_ess_metrics(weights))
        
        # 2. 무게 통계량
        diagnostics.update(self._compute_weight_statistics(weights))
        
        # 3. 덮음 잣대
        diagnostics.update(self._compute_coverage_metrics(samples, weights))
        
        # 4. 벌어짐 어림값
        diagnostics.update(self._compute_divergence_metrics(samples, weights))
        
        # 그리기 위해 표본과 무게 저장
        diagnostics['samples'] = samples
        diagnostics['weights'] = weights
        
        return diagnostics
    
    def _compute_ess_metrics(self, weights: np.ndarray) -> Dict:
        """실효 표본 크기와 이어진 잣대."""
        n = len(weights)
        
        # 표준 ESS
        ess = 1.0 / np.sum(weights**2)
        
        # 상대 ESS
        rel_ess = ess / n
        
        # 헷갈림도(엔트로피의 지수)
        entropy = -np.sum(weights * np.log(weights + 1e-300))
        perplexity = np.exp(entropy)
        
        return {
            'ess': ess,
            'relative_ess': rel_ess,
            'entropy': entropy,
            'perplexity': perplexity,
        }
    
    def _compute_weight_statistics(self, weights: np.ndarray) -> Dict:
        """무게 분포의 통계량."""
        n = len(weights)
        
        # 기본 통계량
        max_weight = np.max(weights)
        min_weight = np.min(weights)
        mean_weight = np.mean(weights)
        std_weight = np.std(weights)
        
        # 변이 계수
        cv = std_weight / (mean_weight + 1e-10)
        
        # 분위수
        quantiles = np.percentile(weights, [25, 50, 75, 90, 95, 99])
        
        # 무게 몰림
        sorted_weights = np.sort(weights)[::-1]
        cumsum = np.cumsum(sorted_weights)
        
        n_for_50pct = np.searchsorted(cumsum, 0.50) + 1
        n_for_90pct = np.searchsorted(cumsum, 0.90) + 1
        
        return {
            'max_weight': max_weight,
            'min_weight': min_weight,
            'mean_weight': mean_weight,
            'std_weight': std_weight,
            'cv_weights': cv,
            'weight_q25': quantiles[0],
            'weight_q50': quantiles[1],
            'weight_q75': quantiles[2],
            'weight_q90': quantiles[3],
            'weight_q95': quantiles[4],
            'weight_q99': quantiles[5],
            'n_for_50pct_weight': n_for_50pct,
            'n_for_90pct_weight': n_for_90pct,
            'pct_for_50pct_weight': n_for_50pct / n * 100,
            'pct_for_90pct_weight': n_for_90pct / n * 100,
        }
    
    def _compute_coverage_metrics(self, samples: np.ndarray, 
                                   weights: np.ndarray) -> Dict:
        """
        제안이 과녁을 얼마나 잘 덮는지 살피기.
        """
        # 표본의 실효 범위
        weighted_mean = np.sum(weights * samples)
        weighted_var = np.sum(weights * (samples - weighted_mean)**2)
        weighted_std = np.sqrt(weighted_var)
        
        # 덮음 살피기: 무게 큰 표본이 널리 퍼져 있는가?
        # 무게로 정렬
        sorted_indices = np.argsort(weights)[::-1]
        top_10pct_idx = sorted_indices[:int(0.1 * len(samples))]
        top_samples = samples[top_10pct_idx]
        
        # 위쪽 표본의 퍼짐
        top_spread = np.std(top_samples)
        
        return {
            'weighted_mean': weighted_mean,
            'weighted_std': weighted_std,
            'top_10pct_spread': top_spread,
        }
    
    def _compute_divergence_metrics(self, samples: np.ndarray,
                                     weights: np.ndarray) -> Dict:
        """
        과녁과 제안 사이의 벌어짐 어림하기.
        
        메모: 이는 거친 어림값이지 정확한 값이 아니다.
        """
        n = len(samples)
        
        # 중요도 무게로 어림한 KL(π||q)
        # KL(π||q) ≈ E_π[log(π/q)] ≈ mean(log(w))
        log_weights = np.log(weights * n + 1e-300)
        kl_estimate = np.mean(log_weights)
        
        # 로그 무게의 흩어짐(KL 벌어짐과 이어짐)
        var_log_weights = np.var(log_weights)
        
        # χ² 벌어짐 어림값
        # χ²(π||q) = E_π[(π/q - 1)²] = E_π[(w*n - 1)²]
        chi2_estimate = np.mean((weights * n - 1)**2)
        
        return {
            'kl_estimate': kl_estimate,
            'var_log_weights': var_log_weights,
            'chi2_estimate': chi2_estimate,
        }
    
    def plot_diagnostics(self, diagnostics: Dict, save_path: str = None):
        """
        진단 그림 두루 만들기.
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        samples = diagnostics['samples']
        weights = diagnostics['weights']
        n = len(samples)
        
        # 칸 1: 무게 막대그림
        ax = axes[0, 0]
        ax.hist(weights * n, bins=50, density=True, alpha=0.7,
                color='steelblue', edgecolor='black')
        ax.axvline(1.0, color='red', linestyle='--', linewidth=2,
                   label='Uniform weight')
        ax.set_xlabel('Normalized Weight × n', fontsize=11)
        ax.set_ylabel('Density', fontsize=11)
        ax.set_title(f'{self.name}: Weight Distribution', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 칸 2: 쌓인 무게
        ax = axes[0, 1]
        sorted_weights = np.sort(weights)[::-1]
        cumsum = np.cumsum(sorted_weights)
        ax.plot(np.arange(1, len(sorted_weights)+1), cumsum,
                linewidth=2, color='darkblue')
        ax.axhline(0.5, color='red', linestyle='--', linewidth=1.5,
                   label='50% of weight')
        ax.axhline(0.9, color='orange', linestyle='--', linewidth=1.5,
                   label='90% of weight')
        ax.set_xlabel('Number of Samples (sorted)', fontsize=11)
        ax.set_ylabel('Cumulative Weight', fontsize=11)
        ax.set_title('Weight Concentration', fontsize=12, fontweight='bold')
        ax.set_xscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 칸 3: 표본 흩뿌림
        ax = axes[0, 2]
        scatter = ax.scatter(samples, weights * n, c=weights * n,
                            cmap='hot', s=30, alpha=0.6,
                            edgecolors='black', linewidth=0.5)
        ax.set_xlabel('Sample Value', fontsize=11)
        ax.set_ylabel('Weight × n', fontsize=11)
        ax.set_title('Samples vs Weights', fontsize=12, fontweight='bold')
        plt.colorbar(scatter, ax=ax)
        ax.grid(True, alpha=0.3)
        
        # 칸 4: 무게의 Q-Q 그림
        ax = axes[1, 0]
        theoretical_quantiles = np.linspace(0, 1, n)
        sample_quantiles = np.sort(weights * n)
        ax.plot(theoretical_quantiles, sample_quantiles, 'o',
                markersize=3, alpha=0.5)
        ax.plot([0, 1], [1, 1], 'r--', linewidth=2, label='Uniform')
        ax.set_xlabel('Theoretical Quantile', fontsize=11)
        ax.set_ylabel('Sample Quantile (Weight × n)', fontsize=11)
        ax.set_title('Q-Q Plot: Weights vs Uniform', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 칸 5: 진단 잣대 글
        ax = axes[1, 1]
        ax.axis('off')
        metrics_text = f"""
        진단 간추림
        ══════════════════════════════
        ESS: {diagnostics['ess']:.1f}
        상대 ESS: {diagnostics['relative_ess']:.1%}
        헷갈림도: {diagnostics['perplexity']:.1f}
        
        무게의 변이 계수: {diagnostics['cv_weights']:.3f}
        최대 무게: {diagnostics['max_weight']:.6f}
        
        무게 50%에 든 표본: {diagnostics['n_for_50pct_weight']}개
                       ({diagnostics['pct_for_50pct_weight']:.1f}%)
        
        무게 90%에 든 표본: {diagnostics['n_for_90pct_weight']}개
                       ({diagnostics['pct_for_90pct_weight']:.1f}%)
        
        KL 어림값: {diagnostics['kl_estimate']:.4f}
        χ² estimate: {diagnostics['chi2_estimate']:.4f}
        """
        ax.text(0.1, 0.5, metrics_text, fontsize=10, family='monospace',
                verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        # 칸 6: 무게의 로그 막대그림
        ax = axes[1, 2]
        log_weights = np.log(weights * n + 1e-10)
        ax.hist(log_weights, bins=50, density=True, alpha=0.7,
                color='darkgreen', edgecolor='black')
        ax.axvline(0, color='red', linestyle='--', linewidth=2,
                   label='log(1) = 0')
        ax.set_xlabel('log(Weight × n)', fontsize=11)
        ax.set_ylabel('Density', fontsize=11)
        ax.set_title('Log-Weight Distribution', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


# 보기 1: 여러 제안 견주기
# ======================================
print("=" * 70)
print("EXAMPLE 1: Systematic Proposal Comparison")
print("=" * 70)

# 과녁 분포: N(5, 1.5)
target_mean, target_std = 5.0, 1.5
target = stats.norm(target_mean, target_std)

# 견줄 제안 여럿 정하기
proposals = {
    'Perfect': stats.norm(5.0, 1.5),          # 과녁과 같음
    'Good': stats.norm(5.0, 1.8),             # 조금 더 넓음
    'Decent': stats.norm(4.5, 2.0),           # 평균을 옮기고 더 넓힘
    'Mediocre': stats.norm(3.0, 2.5),         # 더 많이 옮김
    'Poor': stats.norm(5.0, 0.8),             # 너무 좁음
    'Bad': stats.norm(0.0, 1.5),              # 틀린 자리
}

print("\nAnalyzing proposals for target N(5, 1.5):\n")

results = []
for name, proposal in proposals.items():
    diagnostics_obj = ProposalDiagnostics(target.pdf, proposal, name)
    diag = diagnostics_obj.compute_all_diagnostics(n_samples=3000)
    results.append(diag)

# 견줌 표 찍기
print(f"{'Proposal':<12} {'ESS':>8} {'Rel ESS':>8} {'CV':>8} {'KL Est':>10} "
      f"{'50% in':>8}")
print("-" * 70)

for diag in results:
    print(f"{diag['name']:<12} {diag['ess']:8.1f} {diag['relative_ess']:8.1%} "
          f"{diag['cv_weights']:8.3f} {diag['kl_estimate']:10.4f} "
          f"{diag['pct_for_50pct_weight']:7.1f}%")

# 고른 제안에 대한 자세한 그림 만들기
selected_proposals = ['Perfect', 'Decent', 'Poor']
for name in selected_proposals:
    diag = [d for d in results if d['name'] == name][0]
    prop = proposals[name]
    
    diagnostics_obj = ProposalDiagnostics(target.pdf, prop, name)
    diagnostics_obj.plot_diagnostics(
        diag,
        save_path=f'/home/claude/03_Importance_Sampling/diagnostics_{name.lower()}.png'
    )
    print(f"\nSaved diagnostics plot for {name} proposal")


# 보기 2: 차츰 나빠짐
# ==================================
print("\n" + "=" * 70)
print("EXAMPLE 2: How Proposals Degrade")
print("=" * 70)

# 과녁: N(0, 1)
target_ex2 = stats.norm(0, 1)

# 어긋남이 커지는 제안 만들기
mean_shifts = [0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0]
scale = 1.2  # 과녁보다 조금 더 넓음

print(f"\nProposal: N(μ, {scale}²), varying μ from 0 to 5")
print(f"Target: N(0, 1)\n")

print(f"{'Mean Shift':>12} {'ESS':>8} {'Rel ESS':>8} {'CV':>8} {'Max Wt':>10}")
print("-" * 60)

ess_values = []
for shift in mean_shifts:
    proposal = stats.norm(shift, scale)
    
    diagnostics_obj = ProposalDiagnostics(target_ex2.pdf, proposal, f"μ={shift}")
    diag = diagnostics_obj.compute_all_diagnostics(n_samples=2000)
    
    ess_values.append(diag['ess'])
    
    print(f"{shift:12.1f} {diag['ess']:8.1f} {diag['relative_ess']:8.1%} "
          f"{diag['cv_weights']:8.3f} {diag['max_weight']:10.6f}")

# ESS 나빠짐 그리기
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(mean_shifts, ess_values, 'o-', linewidth=2, markersize=10,
        color='steelblue')
ax.set_xlabel('Proposal Mean Shift', fontsize=12)
ax.set_ylabel('ESS', fontsize=12)
ax.set_title('ESS Degradation with Proposal-Target Mismatch',
             fontsize=13, fontweight='bold')
ax.axhline(2000, color='red', linestyle='--', linewidth=2,
           label='n samples', alpha=0.7)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig('/home/claude/03_Importance_Sampling/ess_degradation.png',
            dpi=300, bbox_inches='tight')


# 보기 3: 꼬리 덮음 진단
# =================================
print("\n" + "=" * 70)
print("EXAMPLE 3: Tail Coverage Assessment")
print("=" * 70)

print("""
꼬리 두꺼운 과녁에서는 제안이 꼬리를 넉넉히 덮어야 한다.
스튜던트 t 과녁을 꼬리 굶이 다른 제안들과 견준다.
""")

# 과녁: 자유도 3인 스튜던트 t(두꺼운 꼬리)
target_t = stats.t(df=3, loc=0, scale=1)

# 꼬리의 굶이 서로 다른 제안
tail_proposals = {
    'Heavy: t(3)': stats.t(df=3, loc=0, scale=1.2),
    'Medium: t(5)': stats.t(df=5, loc=0, scale=1.2),
    'Light: Normal': stats.norm(0, 1.5),
}

print("\nTarget: Student-t(df=3)")
print(f"{'Proposal':<20} {'ESS':>8} {'Rel ESS':>8} {'χ² Est':>10} {'Status'}")
print("-" * 65)

for name, proposal in tail_proposals.items():
    diagnostics_obj = ProposalDiagnostics(target_t.pdf, proposal, name)
    diag = diagnostics_obj.compute_all_diagnostics(n_samples=3000)
    
    # 수상하게 큰 무게가 있는지 살피기(꼬리 덮음 문제)
    max_weight_ratio = diag['max_weight'] * diag['n_samples']
    status = "✓ Good" if max_weight_ratio < 10 else "⚠ Poor tails"
    
    print(f"{name:<20} {diag['ess']:8.1f} {diag['relative_ess']:8.1%} "
          f"{diag['chi2_estimate']:10.2f} {status}")

print("\nKey insight: Proposals with lighter tails than target can fail!")


# 보기 4: 봉우리 여럿인 과녁의 진단
# ======================================
print("\n" + "=" * 70)
print("EXAMPLE 4: Diagnosing Coverage for Multimodal Target")
print("=" * 70)

# 봉우리 둘인 과녁: 0.5*N(-3,1) + 0.5*N(3,1)
def bimodal_density(x):
    return 0.5 * stats.norm.pdf(x, -3, 1) + 0.5 * stats.norm.pdf(x, 3, 1)

# 방식이 서로 다른 제안
multimodal_proposals = {
    'Single wide': stats.norm(0, 4),          # 성분 하나, 아주 넓음
    'Single narrow': stats.norm(0, 1.5),      # 성분 하나, 너무 좁음
    'Centered on mode': stats.norm(-3, 1.5),  # 봉우리 하나만 덮음
}

print("\nTarget: 0.5*N(-3,1) + 0.5*N(3,1) [bimodal]")
print(f"{'Proposal':<20} {'ESS':>8} {'Rel ESS':>8} {'50% Wt%':>10} {'Coverage'}")
print("-" * 70)

for name, proposal in multimodal_proposals.items():
    diagnostics_obj = ProposalDiagnostics(bimodal_density, proposal, name)
    diag = diagnostics_obj.compute_all_diagnostics(n_samples=3000)
    
    # 봉우리 덮음 살피기
    samples = diag['samples']
    left_mode = np.sum((samples < -1) & (samples > -5))
    right_mode = np.sum((samples < 5) & (samples > 1))
    
    if left_mode > 50 and right_mode > 50:
        coverage = "✓ Both modes"
    else:
        coverage = "⚠ Missing mode"
    
    print(f"{name:<20} {diag['ess']:8.1f} {diag['relative_ess']:8.1%} "
          f"{diag['pct_for_50pct_weight']:9.1f}% {coverage}")

print("\nKey insight: Single-component proposals struggle with multimodal targets!")

plt.show()

print("\n" + "=" * 70)
print("KEY TAKEAWAYS")
print("=" * 70)
print("""
1. 꼭 필요한 진단:
   - ESS과 상대 ESS(가장 중요하다)
   - 무게의 변이 계수
   - 무게 몰림(50%, 90% 백분위수)
   - 최대 무게 값

2. 좋은 제안의 표지:
   - 상대 ESS > 0.1(되도록 > 0.3)
   - 무게의 변이 계수 < 3(낮을수록 좋다)
   - 무게의 50%을 차지하는 표본이 10% 미만
   - 어느 무게도 혼자 판치지 않는다

3. 경고 신호:
   - ESS << n(n의 1% 미만)
   - 몇몇 표본이 무게의 대부분을 진다
   - 변이 계수가 아주 높음(> 5-10)
   - KL이나 χ² 벌어짐 어림값이 크다

4. 꼬리 덮음:
   - 꼬리 두꺼운 과녁에 결정적이다
   - 최대 무게 비 살피기
   - 제안은 꼬리가 더 두꺼워야 한다
   - 꼬리 얇은 제안은 크게 무너질 수 있다

5. 봉우리가 여럿인 과녁:
   - 성분 하나로는 흔히 봉우리를 놓친다
   - 표본이 봉우리를 모두 덮는지 살피기
   - 대개 섞음 제안이 필요하다
   - 아니면 맞춰 가는 방법 쓰기

6. 나빠지는 무늬:
   - 제안과 과녁의 어긋남이 커질수록 ESS이 떨어진다
   - 거리에 따라 흔히 지수로 나빠진다
   - 작게 옮겨도 큰 효과가 있을 수 있다

7. 실전 작업 흐름:
   가) 결과를 믿기 전에 늘 진단을 셈하여라
   나) 될 수 있으면 여러 제안 견주기
   다) 경고 신호 살펴보기(ESS, 변이 계수, 몰림)
   라) 무게 분포 그려 보기
   마) 중요한 구역의 덮음 살피기

8. 진단이 나쁠 때의 손보기:
   - 제안 매개변수 다듬기(평균, 흩어짐)
   - 꼬리가 더 두꺼운 제안 집안 쓰기
   - 섞음 제안으로 바꾸기
   - 맞춰 가는 중요도 표집 생각해 보기
   - 아니면 MCMC를 대신 쓰기
""")


if __name__ == "__main__":
    pass
```

## 논의

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
제안 진단 구현을 확인하는 두루 갖춘 시험 함수를 적어라. 빈 입력, 원소 하나짜리 입력, 아주 큰 입력, 그리고 극단적인 값(0, 아주 큰 수)이 든 입력 같은 모서리 경우를 시험하여라.

??? success "연습문제 4 풀이"
    경계 조건을 두루 시험하는 함수를 만든다.
    ```python
    def test_proposaldiagnostics():
        model = ProposalDiagnostics(...)
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
