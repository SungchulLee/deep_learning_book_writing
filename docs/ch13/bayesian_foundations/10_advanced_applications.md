# 나아간 쓰임새

이 모듈은 베이즈 추론의 실전 쓰임새를 다루며 베이즈 A/B 시험에 초점을 맞춘다. 전통적인 A/B 시험은 p값과 붙박이 표본 크기에 기대지만, 베이즈 길은 어느 판이 더 나은지에 대한 확률을 곧바로 말해 주고, 일찍 멈추기를 받쳐 주며, 잇달아 시험해도 오류율을 부풀리지 않고 결과를 곧바로 알아볼 수 있게 한다.

## 코드

```python
"""
베이즈 추론 — 모듈 10: 나아간 쓰임새
수준: 나아간 단계
주제: A/B 시험, 베이즈 최적화, 바뀜점 찾기, 실전 쓰임새

이 모듈은 실제 상황에서 베이즈 추론이 쓰이는 실전 쓰임새를 다룬다.

지은이: 연세대학교 이성철 교수
전자우편: sungchulyonsei@gmail.com
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns

# ========================================================================
# 메인
# ========================================================================

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)

"""
베이즈 A/B 시험:

전통적인 A/B 시험은 p값과 붙박이 표본 크기를 쓴다.
베이즈 A/B 시험은 다음을 준다:
- 어느 판이 더 나은지에 대한 확률 진술
- 증거가 셀 때 일찍 멈출 수 있는 힘
- 결과를 곧바로 풀이하기
"""

def bayesian_ab_test(conversions_a, trials_a, conversions_b, trials_b):
    """
    전환율에 대한 베이즈 A/B 시험.
    """
    print("="*70)
    print("BAYESIAN A/B TESTING")
    print("="*70)
    
    print(f"\nVariant A: {conversions_a}/{trials_a} = {conversions_a/trials_a:.3f}")
    print(f"Variant B: {conversions_b}/{trials_b} = {conversions_b/trials_b:.3f}")
    
    # 베타 뒤확률(고른 앞확률)
    post_a = stats.beta(1 + conversions_a, 1 + trials_a - conversions_a)
    post_b = stats.beta(1 + conversions_b, 1 + trials_b - conversions_b)
    
    # P(B > A)을 셈하는 몬테카를로
    n_samples = 100000
    samples_a = post_a.rvs(n_samples)
    samples_b = post_b.rvs(n_samples)
    prob_b_better = np.mean(samples_b > samples_a)
    
    print(f"\nP(B > A) = {prob_b_better:.4f}")
    print(f"P(A > B) = {1-prob_b_better:.4f}")
    
    # 기대 들어올림
    lift = samples_b / samples_a - 1
    print(f"\nExpected lift (B vs A):")
    print(f"  Mean: {np.mean(lift)*100:.2f}%")
    print(f"  95% Credible Interval: [{np.percentile(lift, 2.5)*100:.2f}%, {np.percentile(lift, 97.5)*100:.2f}%]")
    
    # 시각화
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 그림 1: 뒤확률 분포
    p = np.linspace(0, 1, 1000)
    axes[0].plot(p, post_a.pdf(p), 'b-', linewidth=2, label=f'A ({conversions_a}/{trials_a})')
    axes[0].plot(p, post_b.pdf(p), 'r-', linewidth=2, label=f'B ({conversions_b}/{trials_b})')
    axes[0].axvline(post_a.mean(), color='blue', linestyle='--', alpha=0.7)
    axes[0].axvline(post_b.mean(), color='red', linestyle='--', alpha=0.7)
    axes[0].set_xlabel('Conversion Rate', fontsize=12)
    axes[0].set_ylabel('Density', fontsize=12)
    axes[0].set_title('Posterior Distributions', fontsize=13, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 그림 2: 들어올림 분포
    axes[1].hist(lift * 100, bins=50, alpha=0.7, color='green', edgecolor='black', density=True)
    axes[1].axvline(0, color='red', linestyle='--', linewidth=2, label='No difference')
    axes[1].axvline(np.mean(lift)*100, color='black', linestyle='-', linewidth=2, label=f'Mean lift={np.mean(lift)*100:.1f}%')
    axes[1].set_xlabel('Lift (% improvement)', fontsize=12)
    axes[1].set_ylabel('Density', fontsize=12)
    axes[1].set_title('Distribution of Lift (B vs A)', fontsize=13, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('bayesian_ab_test.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return prob_b_better

if __name__ == "__main__":
    print("\n" + "="*70)
    print("BAYESIAN INFERENCE - MODULE 10: ADVANCED APPLICATIONS")
    print("="*70)
    
    # 보기: A/B 시험
    prob_b_better = bayesian_ab_test(
        conversions_a=120, trials_a=1000,
        conversions_b=145, trials_b=1000
    )
    
    print("\n" + "="*70)
    print("MODULE 10 COMPLETE")
    print("="*70)
    print("\nKey takeaways:")
    print("1. Bayesian A/B testing provides probability statements")
    print("2. Can make early stopping decisions based on evidence")
    print("3. Direct interpretation: P(B better than A)")
    print("4. Naturally handles sequential testing")
    print("\nCongratulations! You've completed the Bayesian Inference curriculum.")
    print("="*70)```

## 논의

Bayesian A/B testing models conversion rates using Beta posteriors. Starting from a uniform prior $\text{Beta}(1,1)$, the posterior for variant $k$ after observing $s_k$ successes in $n_k$ trials is $\text{Beta}(1 + s_k, 1 + n_k - s_k)$. The probability that B is better than A is computed via Monte Carlo: draw samples from both posteriors and estimate $P(\theta_B > \theta_A)$.

A major advantage is computing the expected lift and its credible interval. The lift distribution $\theta_B / \theta_A - 1$ tells us not just whether B is better, but by how much. The 95% credible interval on lift provides a direct, interpretable range for practical significance, far more useful for decision-making than a binary significant/not-significant verdict.

베이즈 얼개는 원칙 있는 일찍 멈추기도 받쳐 준다. 자료를 여러 번 들여다본 것을 바로잡아야 하는 빈도주의 순차 시험과 달리, 베이즈 추론은 자료가 쌓이는 대로 믿음을 저절로 새로 고친다. '$P(B > A) > 0.95$이면 멈춘다' 같은 결정 규칙은 뒤확률을 곧바로 다스리므로 표본 크기를 크게 줄일 수 있다.

## 연습문제

**연습문제 1.**
쪽 A이 85/500, 쪽 B이 110/500의 전환을 낸 베이즈 A/B 시험을 돌려라. $P(B > A)$, 기대 들어올림, 그리고 들어올림의 95% 믿음 구간을 셈하여라.

??? success "연습문제 1 풀이"
    ```python
import numpy as np
from scipy import stats

np.random.seed(42)
samples_a = stats.beta(1+85, 1+415).rvs(100000)
samples_b = stats.beta(1+110, 1+390).rvs(100000)

prob_b_better = np.mean(samples_b > samples_a)
lift = samples_b / samples_a - 1

print(f'P(B > A) = {prob_b_better:.4f}')
print(f'Expected lift: {np.mean(lift)*100:.2f}%')
print(f'95% CI: [{np.percentile(lift,2.5)*100:.2f}%, {np.percentile(lift,97.5)*100:.2f}%]')
```


---

**연습문제 2.**
빈도주의 방법과 달리 베이즈 A/B 시험은 실험 도중 결과를 엿보아도 여러 번 견줌을 바로잡을 필요가 없는 까닭을 설명하여라.

??? success "연습문제 2 풀이"
    빈도주의 p값은 표본 크기를 붙박이로 놓고 귀무가설 아래에서 셈한다. 결과를 여러 번 엿보다가 뜻있어지면 멈추면 사실상 시험을 여러 번 돌리는 셈이라 거짓 양성률이 부풀어 오른다. 본페로니나 알파 나눠 쓰기 함수 같은 바로잡기가 필요하다.

Bayesian inference does not have this problem because the posterior probability $P(B > A | \text{data})$ is a valid probability statement at any point during the experiment. The posterior simply reflects the current state of evidence. If you observe more data, you update the posterior, but the probability statement remains coherent. There is no concept of 'multiple testing' because you are always working with the same model and the same posterior, just with more data incorporated.


---

**연습문제 3.**
베이즈 뒤확률로 살펴보기와 써먹기의 균형을 잡는 톰프슨 표집을 써서 여러 팔 슬롯머신을 구현하여라. 참된 전환율이 0.10, 0.12, 0.15인 판 셋을 다룬다.

??? success "연습문제 3 풀이"
    ```python
import numpy as np

np.random.seed(42)
true_rates = [0.10, 0.12, 0.15]
n_arms = 3
successes = np.ones(n_arms)
failures = np.ones(n_arms)
n_rounds = 5000

choices = []
for _ in range(n_rounds):
    samples = [np.random.beta(successes[k], failures[k]) for k in range(n_arms)]
    chosen = np.argmax(samples)
    choices.append(chosen)
    
    reward = np.random.binomial(1, true_rates[chosen])
    successes[chosen] += reward
    failures[chosen] += (1 - reward)

for k in range(n_arms):
    times_chosen = choices.count(k)
    print(f'Arm {k} (rate={true_rates[k]}): chosen {times_chosen} times '
          f'({times_chosen/n_rounds*100:.1f}%)')
print(f'Best arm chosen: {choices.count(2)/n_rounds*100:.1f}% of the time')
```
톰프슨 표집은 시간이 흐르면서 가장 좋은 팔에 저절로 몰리면서도 그것을 미덥게 알아낼 만큼의 살펴보기를 이어 간다.

