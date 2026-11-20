# 통계적 검정

강화학습 알고리즘을 견줄 때는 시드와 둘레에 따른 분산이 커서 참 성능 차이가 가려질 수 있으므로 엄밀한 통계적 검정이 꼭 필요하다. 이 구현은 웰치 t 검정, 만-휘트니 U 검정, 부트스트랩 신뢰 구간, 대응 부트스트랩 검정, 코헨의 d 효과 크기를 준다. 또한 알고리즘 여럿을 한꺼번에 견줄 때 거짓 발견율을 통제하는 다중 비교 보정(본페로니와 홀름-본페로니)도 담는다.

## 1. 코드

```python
"""
35.6.3 셈밝힘 시험
============================

힘 북돋우는 러닝 알고리즘을 견주는 셈밝힘 시험.
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
    """홀름-본페로니 스텝 내림 바로잡기."""
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

이 구현은 통계적 검정의 핵심 연산을 구현하는 여러 도구 함수를 중심에 둔다. 코드는 알고리즘 구성 요소를 데모·평가 논리와 분리하는 모듈식 설계를 따른다.

데모 함수는 핵심 동작을 잘 드러내는 합성 데이터에서 이 구성 요소들의 실제 쓰임을 보인다. 출력을 살펴보면 하이퍼파라미터 선택과 문제 설정에 따라 알고리즘의 성능이 어떻게 달라지는지 볼 수 있다.

실용적 관점에서 이 구현은 순수한 성능보다 명료함을 앞세운다. 실제로 쓰는 시스템은 보통 배치 계산, GPU 가속, 더 정교한 하이퍼파라미터 튜닝 같은 개선을 더한다. 그럼에도 여기 보인 핵심 알고리즘 아이디어는 대규모 응용으로 그대로 옮겨 간다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
데모 코드를 돌려 핵심 출력 지표를 기록하라. 하이퍼파라미터 하나(러닝레이트, 은닉 차원, 레이어 개수 같은 것)를 바꾸고 결과가 어떻게 바뀌는지 기록하라.

</div>

??? success "풀이"
    데모를 돌린 뒤 나머지를 고정하고 고른 하이퍼파라미터를 차례로 바꾼다. 예를 들어 은닉 차원을 두 배로 하면 보통 표현력이 늘지만 계산 시간이 커진다. 러닝레이트는 단조롭지 않은 영향을 준다. 너무 작으면 느리게 수렴하고 너무 크면 진동한다. 고른 하이퍼파라미터의 서로 다른 값을 적어도 셋 잡아 구체적인 수를 기록해 두라.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
이 구현에서 핵심 구조 선택의 역할을 설명하라. 왜 그 활성화 함수, 정규화 전략, 로스 함수를 쓰는가? 다른 것으로 바꾸면 어떻게 되는가?

</div>

??? success "풀이"
    이 구조 선택은 강화학습 평가 방법에서 자리 잡은 모범 사례를 따른다. 예를 들어 ReLU 활성화는 비선형성을 주면서 양의 입력에서 그래디언트가 사라지는 것을 피한다. 로스 함수는 과제 유형에 맞게 고른다(분류에는 교차 엔트로피, 회귀에는 평균 제곱 오차). 다른 것으로 바꾸면(예를 들어 시그모이드 활성화, L1 로스) 최적화 지형이 바뀌어 성능이 나빠질 수 있지만 어떤 상황에서는 바꾸는 것이 이로울 수도 있다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
더 어려운 상황을 다루도록 구현을 확장하라. 더 큰 데이터셋, 다른 문제 변형, 피처 추가 가운데 하나를 고른다. 수정 사항을 적고 성능에 미친 영향을 분석하라.

</div>

??? success "풀이"
    자연스러운 확장 하나는 정칙화(드롭아웃, 웨이트 감쇠)나 더 정교한 구조(레이어 추가, 스킵 연결)를 더하는 것이다. 고른 확장을 구현하고 같은 데이터로 학습한 뒤 전후의 지표를 비교하여라. 이 확장은 원래 알고리즘과 수정 사항의 이론적 근거를 모두 이해했음을 보여야 한다.

## 정리하며

**다룬 것** — 통계적 검정

이 구현은 통계적 검정의 핵심 연산을 구현하는 여러 도구 함수를 중심에 둔다.

앞의 연습문제 3개로 스스로 점검해 볼 수 있다.
