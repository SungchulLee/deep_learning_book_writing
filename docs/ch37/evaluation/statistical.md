# 통계적 검정

강화학습 알고리즘을 견줄 때는 시드와 환경에 따른 분산이 커서 참 성능 차이가 가려질 수 있으므로 엄밀한 통계적 검정이 꼭 필요하다. 이 구현은 웰치 t 검정, 만-휘트니 U 검정, 부트스트랩 신뢰 구간, 대응 부트스트랩 검정, 코헨의 d 효과 크기를 준다. 또한 알고리즘 여럿을 한꺼번에 견줄 때 **가족별 오류율**(FWER, 검정 여럿 가운데 하나라도 거짓 양성이 나올 확률)을 통제하는 다중 비교 보정(본페로니와 홀름-본페로니)도 담는다.

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

데모는 실제 학습 대신 정규분포에서 뽑은 합성 점수를 쓴다(DQN 은 평균 180·표준편차 30, Double DQN 은 200·25, 셋째 알고리즘은 185·35, 각 100개). 참값을 알고 있으니 검정이 무엇을 잡고 무엇을 놓치는지 확인할 수 있다.

DQN 과 Double DQN 은 웰치 t 검정과 만-휘트니 U 검정 모두 $p<0.0001$ 이고, 차이의 부트스트랩 95% 구간 $[-31.3, -16.3]$ 이 0 을 포함하지 않으며, 코헨의 d 가 $-0.929$ 로 큰 효과다. 네 도구가 같은 결론을 낸다.

흥미로운 것은 둘째 견줌이다. 코드는 이것을 「비슷한 성능」이라 이름 붙였지만 웰치 t 검정은 $p=0.0274$ 로 5% 수준에서 유의하다고 말한다. 참 평균이 180 과 185 로 실제로 다르고, 표본 100개면 이만한 차이도 잡히기 때문이다. 그런데 효과 크기는 $d=-0.316$ 으로 작다. **유의하다는 것과 차이가 크다는 것은 다른 말**이다 — 그래서 p-값과 효과 크기를 함께 적는다.

다중 비교 보정 부분은 원래 p-값 0.01, 0.03, 0.06, 0.15 를 보정한다. 본페로니는 모두에 4를 곱해 0.01 하나만 5% 수준에 남긴다. 홀름-본페로니는 작은 것부터 4, 3, 2, 1 을 곱하므로 0.090, 0.120, 0.150 처럼 본페로니보다 덜 보수적이다. 이 예에서는 결론이 같지만, 홀름 방법은 같은 FWER 을 지키면서 언제나 본페로니가 기각하는 가설을 모두 기각하고, 때로는 더 기각한다.

한 가지 주의할 점이 있다. `paired_bootstrap_test` 는 `scores_a - scores_b` 를 원소끼리 빼므로, 두 배열의 같은 자리가 같은 시드·같은 에피소드에서 나온 **짝**일 때만 뜻이 있다. 데모의 합성 점수는 서로 독립으로 뽑았으므로 짝이 없고, 이때의 부트스트랩 구간은 독립 표본의 평균 차를 대략 보여 줄 뿐이다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
데모의 표본 크기 `size=100` 을 10 과 1000 으로 바꾸어 A 와 C 의 견줌을 다시 돌려라. 웰치 t 검정의 p-값과 코헨의 d 는 각각 어떻게 달라지는가?

</div>

??? success "풀이"
    코헨의 d 는 평균 차를 합동 표준편차로 나눈 값이라 표본 크기와 상관없이 참값 근처(평균 차 5, 표준편차 30~35 이므로 약 $-0.15$)에 머물고, 표본이 작을수록 그 주위로 많이 흔들린다. 반면 t 통계량은 대략 $\sqrt{n}$ 에 비례해 커지므로 p-값은 표본이 커질수록 작아진다. 표본 10개면 거의 늘 유의하지 않고, 1000개면 거의 늘 유의하다. 같은 효과 크기가 표본 크기에 따라 「유의」와 「유의하지 않음」을 오간다는 것이 p-값만 보고하면 안 되는 까닭이다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
웰치 t 검정과 만-휘트니 U 검정은 각각 무엇을 가정하는가? 강화학습의 리턴 분포가 한쪽으로 치우치거나 봉우리가 둘인 경우(시드 몇 개만 학습에 실패한 경우) 어느 검정을 믿어야 하는가?

</div>

??? success "풀이"
    웰치 t 검정은 두 집단의 분산이 같다고 가정하지 않지만 표본 평균이 정규분포에 가깝다고 본다. 만-휘트니 U 검정은 분포의 형태를 가정하지 않는 **비모수 검정**으로, 값 자체 대신 순위를 견준다. 몇몇 시드가 학습에 실패해 리턴이 0 근처에 몰리면 평균과 분산이 그 시드들에 크게 끌려가 t 검정이 흔들리고, 표본이 작을 때는 정규 근사도 맞지 않는다. 이때는 순위 기반의 만-휘트니 검정이나 부트스트랩 구간이 더 믿을 만하다. 다만 만-휘트니 검정은 「한쪽 점수가 다른 쪽보다 크게 나올 확률이 1/2 인가」를 묻는 것이지 평균의 차를 묻는 것이 아니므로, 결과를 평균의 차로 읽으면 안 된다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
알고리즘 다섯 개를 서로 짝지어 모두 견주면 검정이 몇 번 필요한가? 각 검정을 5% 수준에서 하고 모든 귀무가설이 참이라면, 하나라도 거짓 양성이 나올 확률은 얼마쯤인가(검정이 독립이라고 보라)? 홀름-본페로니 보정으로 이것을 5% 이하로 묶는 과정을 `holm_bonferroni` 의 코드에 맞춰 설명하라.

</div>

??? success "풀이"
    짝은 $\binom{5}{2}=10$ 개다. 독립이면 하나라도 거짓 양성이 나올 확률은 $1-0.95^{10}\approx 0.40$ 으로, 열 번 견주면 열에 넷은 근거 없는 「차이」를 보고하게 된다. `holm_bonferroni` 는 p-값을 작은 순서로 정렬한 뒤 순위 $k$ (0부터)의 p-값에 $n-k$ 를 곱한다. 가장 작은 것에는 10, 다음에는 9, … 를 곱하는 셈이다. 그 뒤의 루프는 보정된 p-값이 순위를 따라 줄지 않도록 앞의 값과의 최댓값을 취한다. 보정된 p-값이 0.05 보다 작은 가설만 기각하면 가족별 오류율이 5% 이하로 묶인다.

## 정리하며

**다룬 것** — 통계적 검정

p-값은 차이가 우연인지를, 코헨의 d 는 차이가 얼마나 큰지를, 부트스트랩 구간은 차이가 어느 범위에 있는지를 말한다. 데모의 둘째 견줌처럼 셋이 서로 다른 인상을 줄 수 있으므로 함께 보고하고, 견줌이 여럿이면 다중 비교 보정을 건다.

앞의 연습문제 3개로 스스로 점검해 볼 수 있다.
