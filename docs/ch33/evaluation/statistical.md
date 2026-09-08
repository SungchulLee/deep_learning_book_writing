# 통계 시험

힘 북돋우는 배움 알고리즘을 견줄 때는 씨앗과 둘레에 따른 흩어짐이 커서 참 성능 차이가 가려질 수 있으므로 엄밀한 통계 시험이 꼭 필요하다. 이 짜기는 웰치 t 시험, 만-휘트니 U 시험, 부트스트랩 믿음 구간, 짝지은 부트스트랩 시험, 코헨의 d 효과 크기를 준다. 또한 알고리즘 여럿을 한꺼번에 견줄 때 거짓 발견 비율을 다스리는 여러 견줌 바로잡기(본페로니와 홀름-본페로니)도 담는다.

## 1. 코드

```python
"""
33.6.3 셈밝힘 시험
============================

힘 북돋우는 배움 알고리즘을 견주는 셈밝힘 시험.
"""

import numpy as np
from typing import Tuple, List, Dict
from scipy import stats

# ========================================================================
# 메인
# ========================================================================


def welch_ttest(scores_a: np.ndarray, scores_b: np.ndarray) -> Dict:
    """두 알고리즘을 견주는 웰치 t 시험."""
    t_stat, p_value = stats.ttest_ind(scores_a, scores_b, equal_var=False)
    return {'t_statistic': t_stat, 'p_value': p_value,
            'significant_005': p_value < 0.05, 'significant_001': p_value < 0.01}


def mann_whitney_u(scores_a: np.ndarray, scores_b: np.ndarray) -> Dict:
    """값에 기대지 않는 만-휘트니 U 시험."""
    u_stat, p_value = stats.mannwhitneyu(scores_a, scores_b, alternative='two-sided')
    return {'u_statistic': u_stat, 'p_value': p_value,
            'significant_005': p_value < 0.05}


def bootstrap_ci(scores: np.ndarray, n_bootstrap: int = 10000,
                 ci: float = 0.95, stat_fn=np.mean) -> Tuple[float, float, float]:
    """셈밝힘 값의 부트스트랩 믿음 구간."""
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
    """짝지은 차이의 켜 나눈 부트스트랩 시험."""
    diffs = scores_a - scores_b
    mean_diff = diffs.mean()
    boot_diffs = np.array([
        np.random.choice(diffs, size=len(diffs), replace=True).mean()
        for _ in range(n_bootstrap)
    ])
    low = np.percentile(boot_diffs, 2.5)
    high = np.percentile(boot_diffs, 97.5)
    p_value = np.mean(boot_diffs < 0) if mean_diff > 0 else np.mean(boot_diffs > 0)
    p_value = 2 * min(p_value, 1 - p_value)  # 양쪽
    return {'mean_diff': mean_diff, 'ci_low': low, 'ci_high': high,
            'p_value': p_value, 'significant': low > 0 or high < 0}


def cohens_d(scores_a: np.ndarray, scores_b: np.ndarray) -> float:
    """코헨의 d 효과 크기."""
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
    """본페로니 여러 번 견주기 바로잡기."""
    n = len(p_values)
    return [min(1.0, p * n) for p in p_values]


def holm_bonferroni(p_values: List[float]) -> List[float]:
    """홀름-본페로니 걸음 내림 바로잡기."""
    n = len(p_values)
    sorted_idx = np.argsort(p_values)
    adjusted = np.zeros(n)
    for rank, idx in enumerate(sorted_idx):
        adjusted[idx] = min(1.0, p_values[idx] * (n - rank))
    # 한 방향임을 다짐
    for i in range(1, n):
        idx = sorted_idx[i]
        prev_idx = sorted_idx[i-1]
        adjusted[idx] = max(adjusted[idx], adjusted[prev_idx])
    return adjusted.tolist()


def comprehensive_comparison(name_a: str, scores_a: np.ndarray,
                              name_b: str, scores_b: np.ndarray) -> Dict:
    """알고리즘 둘 사이의 셈밝힘 시험을 모두 돌린다."""
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

    # 알고리즘 결과 흉내(씨앗 5개, 저마다 값 매김 마당 20번)
    algo_a = np.random.normal(180, 30, size=100)  # DQN
    algo_b = np.random.normal(200, 25, size=100)  # 겹 DQN
    algo_c = np.random.normal(185, 35, size=100)  # A와 비슷함

    # 짝별 견주기
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

    # 부트스트랩 믿음 구간
    print("\n--- Bootstrap Confidence Intervals ---")
    for name, scores in [("DQN", algo_a), ("DoubleDQN", algo_b)]:
        mean, low, high = bootstrap_ci(scores)
        print(f"  {name}: {mean:.1f} [{low:.1f}, {high:.1f}]")

    # 여러 번 견주기
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
```

**출력:**

```
============================================================
Statistical Testing Demo
============================================================

--- A vs B (clear difference) ---
  DQN: 176.9 ± 27.1
  DoubleDQN: 200.6 ± 23.7
  Welch's t-test: p=0.0000
  Mann-Whitney U: p=0.0000
  Bootstrap 95% CI: [-31.3, -16.3]
  Cohen's d: -0.929 (large)

--- A vs C (similar performance) ---
  DQN: 176.9 ± 27.1
  DQN-v2: 187.3 ± 37.8
  Welch's t-test: p=0.0274
  Cohen's d: -0.316 (small)

--- Bootstrap Confidence Intervals ---
  DQN: 176.9 [171.5, 182.0]
  DoubleDQN: 200.6 [195.9, 205.2]

--- Multiple Comparison Correction ---
  Raw p-values:  ['0.010', '0.030', '0.060', '0.150']
  Bonferroni:    ['0.040', '0.120', '0.240', '0.600']
  Holm-Bonf:     ['0.040', '0.090', '0.120', '0.150']

Statistical testing demo complete!
```

## 2. 논의

이 짜기는 통계 시험의 핵심 연산을 짜는 여러 연장 함수를 한가운데 둔다. 코드는 알고리즘 조각을 보여 주기와 따지기 논리에서 떼어 놓는 조각 짜기를 따른다.

보여 주기 함수는 핵심 움직임을 도드라지게 하는 만든 자료에서 이 조각들의 실제 쓰임을 보인다. 내놓기를 살펴보면 윗매개변수를 어떻게 고르고 문제를 어떻게 차리느냐에 따라 알고리즘의 성능이 어떻게 달라지는지 볼 수 있다.

실제 관점에서 이 짜기는 순수한 성능보다 또렷함을 앞세운다. 실제로 쓰는 얼개는 보통 묶음 셈, GPU 빠르게 하기, 더 정교한 윗매개변수 맞추기 같은 개선을 더한다. 그럼에도 여기 보인 핵심 알고리즘 생각은 큰 규모의 쓰임새로 곧바로 옮겨 간다.

## 연습문제

**연습문제 1.**
보여 주기 코드를 돌려 핵심 내놓기 잣대를 적어라. 윗매개변수 하나(배움 빠르기, 숨은 차원, 층 개수 같은 것)를 고치고 결과가 어떻게 바뀌는지 적어라.

??? success "연습문제 1 풀이"
    보여 주기를 돌린 뒤 나머지를 붙박아 두고 고른 윗매개변수를 차근히 바꾼다. 보기로 숨은 차원을 두 배로 하면 보통 나타냄 담이가 늘지만 셈 시간이 커진다. 배움 빠르기는 단조롭지 않은 영향을 준다. 너무 작으면 느리게 모이고 너무 크면 흔들린다. 고른 윗매개변수의 서로 다른 값을 적어도 셋 잡아 구체적인 수를 적어 두라.

---

**연습문제 2.**
이 짜기에서 핵심 얼개 고르기의 몫을 밝혀라. 왜 그 깨움 함수, 고르게 맞추기 셈속, 손실 함수를 쓰는가? 다른 것으로 바꾸면 어떻게 되는가?

??? success "연습문제 2 풀이"
    이 얼개 고르기는 힘 북돋우는 배움 따지기 방법에서 자리 잡은 가장 좋은 방식을 따른다. 보기로 ReLU 깨움은 비선형을 주면서 양의 들임에서 기울기가 사라지는 것을 피한다. 손실 함수는 일의 갈래에 맞게 고른다(가름에는 교차 엔트로피, 되돌이 맞춤에는 평균 제곱 어긋남). 다른 것으로 바꾸면(보기로 시그모이드 깨움, L1 손실) 가장 좋게 하기의 풍경이 바뀌어 성능이 나빠질 수 있지만 어떤 상황에서는 바꿈이 이로울 수도 있다.

---

**연습문제 3.**
더 어려운 상황을 다루도록 짜기를 넓혀라. 더 큰 자료 묶음, 다른 문제 변형, 특징 하나 더하기 가운데 하나이다. 고침을 적고 성능에 미친 영향을 따져라.

??? success "연습문제 3 풀이"
    자연스러운 넓힘 하나는 규칙 세우기(떨구기, 무게 줄임)나 더 정교한 얼개(층 더하기, 건너뛰는 이음)를 더하는 것이다. 고른 넓힘을 짜고 같은 자료로 익힌 뒤 앞뒤의 잣대를 견주어라. 이 넓힘은 본디 알고리즘과 고침의 이론 까닭을 모두 이해했음을 보여야 한다.

## 정리하며

**다룬 것** — 통계 시험

이 짜기는 통계 시험의 핵심 연산을 짜는 여러 연장 함수를 한가운데 둔다.

앞의 연습문제 3개로 스스로 따져 볼 수 있다.
