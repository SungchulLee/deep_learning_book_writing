"""
33.6.3 Statistical Testing
============================

Statistical tests for comparing RL algorithms.
"""

import numpy as np
from typing import Tuple, List, Dict
from scipy import stats


def welch_ttest(scores_a: np.ndarray, scores_b: np.ndarray) -> Dict:
    """Welch's t-test for comparing two algorithms."""
    t_stat, p_value = stats.ttest_ind(scores_a, scores_b, equal_var=False)
    return {'t_statistic': t_stat, 'p_value': p_value,
            'significant_005': p_value < 0.05, 'significant_001': p_value < 0.01}


def mann_whitney_u(scores_a: np.ndarray, scores_b: np.ndarray) -> Dict:
    """Non-parametric Mann-Whitney U test."""
    u_stat, p_value = stats.mannwhitneyu(scores_a, scores_b, alternative='two-sided')
    return {'u_statistic': u_stat, 'p_value': p_value,
            'significant_005': p_value < 0.05}


def bootstrap_ci(scores: np.ndarray, n_bootstrap: int = 10000,
                 ci: float = 0.95, stat_fn=np.mean) -> Tuple[float, float, float]:
    """Bootstrap confidence interval for a statistic."""
    boot_stats = np.array([
        stat_fn(np.random.choice(scores, size=len(scores), replace=True))
        for _ in range(n_bootstrap)
    ])
    alpha = (1 - ci) / 2
    low = np.percentile(boot_stats, alpha * 100)
    high = np.percentile(boot_stats, (1 - alpha) * 100)
    return stat_fn(scores), low, high


def paired_bootstrap_test(scores_a: np.ndarray, scores_b: np.ndarray,
                          n_bootstrap: int = 10000) -> Dict:
    """Stratified bootstrap test for paired differences."""
    diffs = scores_a - scores_b
    mean_diff = diffs.mean()
    boot_diffs = np.array([
        np.random.choice(diffs, size=len(diffs), replace=True).mean()
        for _ in range(n_bootstrap)
    ])
    low = np.percentile(boot_diffs, 2.5)
    high = np.percentile(boot_diffs, 97.5)
    p_value = np.mean(boot_diffs < 0) if mean_diff > 0 else np.mean(boot_diffs > 0)
    p_value = 2 * min(p_value, 1 - p_value)  # two-sided
    return {'mean_diff': mean_diff, 'ci_low': low, 'ci_high': high,
            'p_value': p_value, 'significant': low > 0 or high < 0}


def cohens_d(scores_a: np.ndarray, scores_b: np.ndarray) -> float:
    """Cohen's d effect size."""
    n_a, n_b = len(scores_a), len(scores_b)
    pooled_std = np.sqrt(((n_a - 1) * scores_a.std()**2 + (n_b - 1) * scores_b.std()**2)
                         / (n_a + n_b - 2))
    return (scores_a.mean() - scores_b.mean()) / (pooled_std + 1e-8)


def interpret_cohens_d(d: float) -> str:
    d = abs(d)
    if d < 0.2: return "negligible"
    elif d < 0.5: return "small"
    elif d < 0.8: return "medium"
    else: return "large"


def bonferroni_correction(p_values: List[float]) -> List[float]:
    """Bonferroni multiple comparison correction."""
    n = len(p_values)
    return [min(1.0, p * n) for p in p_values]


def holm_bonferroni(p_values: List[float]) -> List[float]:
    """Holm-Bonferroni step-down correction."""
    n = len(p_values)
    sorted_idx = np.argsort(p_values)
    adjusted = np.zeros(n)
    for rank, idx in enumerate(sorted_idx):
        adjusted[idx] = min(1.0, p_values[idx] * (n - rank))
    # Enforce monotonicity
    for i in range(1, n):
        idx = sorted_idx[i]
        prev_idx = sorted_idx[i-1]
        adjusted[idx] = max(adjusted[idx], adjusted[prev_idx])
    return adjusted.tolist()


def comprehensive_comparison(name_a: str, scores_a: np.ndarray,
                              name_b: str, scores_b: np.ndarray) -> Dict:
    """Run all statistical tests between two algorithms."""
    results = {
        'algorithms': (name_a, name_b),
        'n_samples': (len(scores_a), len(scores_b)),
        'means': (scores_a.mean(), scores_b.mean()),
        'stds': (scores_a.std(), scores_b.std()),
        'welch': welch_ttest(scores_a, scores_b),
        'mann_whitney': mann_whitney_u(scores_a, scores_b),
        'bootstrap': paired_bootstrap_test(scores_a, scores_b) if len(scores_a) == len(scores_b) else None,
        'cohens_d': cohens_d(scores_a, scores_b),
    }
    return results


def demo_statistical_testing():
    print("=" * 60)
    print("Statistical Testing Demo")
    print("=" * 60)

    np.random.seed(42)

    # Simulate algorithm results (5 seeds, 20 eval episodes each)
    algo_a = np.random.normal(180, 30, size=100)  # DQN
    algo_b = np.random.normal(200, 25, size=100)  # Double DQN
    algo_c = np.random.normal(185, 35, size=100)  # Similar to A

    # Pairwise comparison
    print("\n--- A vs B (clear difference) ---")
    result = comprehensive_comparison("DQN", algo_a, "DoubleDQN", algo_b)
    print(f"  DQN: {result['means'][0]:.1f} ± {result['stds'][0]:.1f}")
    print(f"  DoubleDQN: {result['means'][1]:.1f} ± {result['stds'][1]:.1f}")
    print(f"  Welch's t-test: p={result['welch']['p_value']:.4f}")
    print(f"  Mann-Whitney U: p={result['mann_whitney']['p_value']:.4f}")
    if result['bootstrap']:
        print(f"  Bootstrap 95% CI: [{result['bootstrap']['ci_low']:.1f}, "
              f"{result['bootstrap']['ci_high']:.1f}]")
    d = result['cohens_d']
    print(f"  Cohen's d: {d:.3f} ({interpret_cohens_d(d)})")

    print("\n--- A vs C (similar performance) ---")
    result2 = comprehensive_comparison("DQN", algo_a, "DQN-v2", algo_c)
    print(f"  DQN: {result2['means'][0]:.1f} ± {result2['stds'][0]:.1f}")
    print(f"  DQN-v2: {result2['means'][1]:.1f} ± {result2['stds'][1]:.1f}")
    print(f"  Welch's t-test: p={result2['welch']['p_value']:.4f}")
    d2 = result2['cohens_d']
    print(f"  Cohen's d: {d2:.3f} ({interpret_cohens_d(d2)})")

    # Bootstrap CI
    print("\n--- Bootstrap Confidence Intervals ---")
    for name, scores in [("DQN", algo_a), ("DoubleDQN", algo_b)]:
        mean, low, high = bootstrap_ci(scores)
        print(f"  {name}: {mean:.1f} [{low:.1f}, {high:.1f}]")

    # Multiple comparisons
    print("\n--- Multiple Comparison Correction ---")
    p_values = [0.01, 0.03, 0.06, 0.15]
    bonf = bonferroni_correction(p_values)
    holm = holm_bonferroni(p_values)
    print(f"  Raw p-values:  {[f'{p:.3f}' for p in p_values]}")
    print(f"  Bonferroni:    {[f'{p:.3f}' for p in bonf]}")
    print(f"  Holm-Bonf:     {[f'{p:.3f}' for p in holm]}")

    print("\nStatistical testing demo complete!")


if __name__ == "__main__":
    demo_statistical_testing()
