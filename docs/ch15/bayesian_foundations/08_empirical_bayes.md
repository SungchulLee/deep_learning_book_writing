# 경험 베이즈

경험 베이즈 방법은 웃매개변수를 자료 그 자체에서 어림하여, 온전한 베이즈와 빈도주의 사이의 실전적인 가운뎃길을 준다. 앞확률 매개변수를 미리 못 박는 대신, 경험 베이즈는 관측의 주변 분포로 가장 좋은 앞확률 설정을 정하며, 서로 이어진 매개변수를 한꺼번에 많이 어림할 때 특히 힘이 세다.

## 1. 코드

```python
"""
베이즈 추론 — 모듈 8: 경험 베이즈
수준: 나아간 단계
주제: 경험 베이즈 방법, 자료에서 웃매개변수 어림하기

경험 베이즈는 웃매개변수를 자료 그 자체에서 어림하여, 온전한 베이즈와
빈도주의 사이의 실전적인 가운뎃길을 준다.

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
경험 베이즈 방법론:

표준 베이즈: 앞확률 매개변수를 자료를 보기 전에 붙박아 둔다
경험 베이즈: 앞확률 매개변수를 자료에서 어림한다

절차:
1. 자료의 주변 분포에서 웃매개변수를 어림한다
2. 이 어림값을 낱낱의 추론을 위한 "앞확률"로 쓴다
3. 표준 베이즈 추론을 이어 간다

이름난 보기: 제임스-스타인 어림자
매개변수가 3개 이상이면 오그라들기 어림자가 MLE를 누름을 보인다
"""

def baseball_empirical_bayes_demo():
    """
    경험 베이즈의 고전적인 야구 타율 보기.
    """
    print("="*70)
    print("EMPIRICAL BAYES: Baseball Batting Averages")
    print("="*70)
    
    # 자료 흉내내기: 시즌 초 타율
    np.random.seed(42)
    n_players = 20
    true_abilities = np.random.beta(80, 220, n_players)  # 참된 타율
    at_bats = np.random.randint(20, 100, n_players)
    hits = np.array([np.random.binomial(ab, ta) for ab, ta in zip(at_bats, true_abilities)])
    
    observed_avg = hits / at_bats
    
    print(f"\nNumber of players: {n_players}")
    print(f"At-bats range: {at_bats.min()}-{at_bats.max()}")
    
    # 경험 베이즈: 관측된 타율에서 베타 앞확률 어림하기
    # 적률법
    mean_obs = np.mean(observed_avg)
    var_obs = np.var(observed_avg)
    
    # 베타 분포: 평균 = α/(α+β), 흩어짐 = αβ/[(α+β)²(α+β+1)]
    # α, β 풀기
    alpha_eb = mean_obs * (mean_obs * (1 - mean_obs) / var_obs - 1)
    beta_eb = (1 - mean_obs) * (mean_obs * (1 - mean_obs) / var_obs - 1)
    
    print(f"\nEmpirical Bayes prior: Beta({alpha_eb:.2f}, {beta_eb:.2f})")
    print(f"  Prior mean: {alpha_eb/(alpha_eb+beta_eb):.4f}")
    
    # 오그라들기 쓰기
    eb_estimates = (hits + alpha_eb) / (at_bats + alpha_eb + beta_eb)
    
    # MLE와 견주기
    mse_mle = np.mean((observed_avg - true_abilities)**2)
    mse_eb = np.mean((eb_estimates - true_abilities)**2)
    
    print(f"\nMean Squared Error:")
    print(f"  MLE:            {mse_mle:.6f}")
    print(f"  Empirical Bayes: {mse_eb:.6f}")
    print(f"  Improvement:     {(1 - mse_eb/mse_mle)*100:.1f}%")
    
    # 시각화
    plt.figure(figsize=(14, 6))
    
    plt.subplot(1, 2, 1)
    plt.scatter(observed_avg, true_abilities, s=at_bats, alpha=0.6, label='MLE')
    plt.scatter(eb_estimates, true_abilities, s=at_bats, alpha=0.6, label='Empirical Bayes')
    plt.plot([0, 0.5], [0, 0.5], 'k--', alpha=0.5)
    plt.xlabel('Estimate', fontsize=12)
    plt.ylabel('True Ability', fontsize=12)
    plt.title('Estimates vs True Ability', fontsize=13, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    for i in range(n_players):
        plt.plot([observed_avg[i], eb_estimates[i]], [i, i], 'r-', linewidth=1.5, alpha=0.7)
        plt.plot(observed_avg[i], i, 'bo', markersize=6)
        plt.plot(eb_estimates[i], i, 'ro', markersize=6)
    plt.axvline(mean_obs, color='green', linestyle=':', linewidth=2, label='Grand mean')
    plt.xlabel('Batting Average', fontsize=12)
    plt.ylabel('Player', fontsize=12)
    plt.title('Shrinkage Toward Prior Mean', fontsize=13, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig('empirical_bayes_baseball.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    print("\n" + "="*70)
    print("BAYESIAN INFERENCE - MODULE 8: EMPIRICAL BAYES")
    print("="*70)
    
    baseball_empirical_bayes_demo()
    
    print("\n" + "="*70)
    print("MODULE 8 COMPLETE")
    print("="*70)
    print("\nKey takeaways:")
    print("1. Empirical Bayes estimates prior from data")
    print("2. Provides automatic shrinkage without full Bayesian machinery")
    print("3. Often outperforms MLE, especially with many parameters")
    print("4. Related to James-Stein estimator")
    print("\nNext: Module 9 - Bayesian Linear Regression")
    print("="*70)
```

**출력:**

```

======================================================================
BAYESIAN INFERENCE - MODULE 8: EMPIRICAL BAYES
======================================================================
======================================================================
EMPIRICAL BAYES: Baseball Batting Averages
======================================================================

Number of players: 20
At-bats range: 20-98

Empirical Bayes prior: Beta(15.99, 43.52)
  Prior mean: 0.2687

Mean Squared Error:
  MLE:            0.002670
  Empirical Bayes: 0.000634
  Improvement:     76.2%

======================================================================
MODULE 8 COMPLETE
======================================================================

Key takeaways:
1. Empirical Bayes estimates prior from data
2. Provides automatic shrinkage without full Bayesian machinery
3. Often outperforms MLE, especially with many parameters
4. Related to James-Stein estimator

Next: Module 9 - Bayesian Linear Regression
======================================================================
```

## 2. 논의

경험 베이즈는 베이즈 통계와 빈도주의 통계 사이의 흥미로운 자리에 놓인다. 이 길은 주변 분포에 최대 가능도나 적률법 같은 빈도주의 방법을 써서 자료로부터 앞확률을 어림한 다음, 그 어림값을 베이즈 공식에 끼워 넣는다. 엄격한 베이즈 원칙을 어기기는 하지만 실전에서는 아주 좋은 결과를 준다.

야구 타율 보기는 경험 베이즈 오그라들기를 아름답게 보여 준다. 시즌 초 타율은 참된 능력에 대한 시끄러운 어림값이다. 관측된 타율 분포에 베타 앞확률을 맞추면 극단적인 관측을 모집단 평균 쪽으로 끌어당기는 어림값을 얻고, 시즌 끝 성적을 더 정확하게 미리 알 수 있다.

핵심 통찰은 매개변수를 셋 이상 한꺼번에 어림할 때 오그라들기 어림자가 최대 가능도 어림을 한결같이 앞선다는 것이다. 이것이 제임스-스타인 현상이다. 곧 차원이 셋 이상이면 MLE는 받아들일 수 없으며, 경험 베이즈는 그것을 누르는 오그라들기 어림자를 얻는 원칙 있는 길을 준다.

## 연습문제

**연습문제 1.**
정규-정규 모형에 대한 경험 베이즈 절차를 짜라. $\theta_i \sim N(\mu, \tau^2)$이고 관측값이 $y_i \sim N(\theta_i, 1)$인 10개가 주어졌을 때 주변 분포에서 $\mu$과 $\tau^2$을 어림하고 오그림 어림값을 셈하여라.

??? success "연습문제 1 풀이"
    ```python
import numpy as np

np.random.seed(42)
true_mu, true_tau = 3.0, 2.0
thetas = np.random.normal(true_mu, true_tau, 10)
y = np.array([np.random.normal(t, 1.0) for t in thetas])

# 적률법 어림값
mu_hat = np.mean(y)
tau2_hat = max(np.var(y) - 1.0, 0.01)

# 오그라들기
w = tau2_hat / (tau2_hat + 1.0)
eb_est = w * y + (1 - w) * mu_hat

print(f'MSE (MLE): {np.mean((y - thetas)**2):.4f}')
print(f'MSE (EB):  {np.mean((eb_est - thetas)**2):.4f}')
```


---

**연습문제 2.**
제임스-스타인 어림값이 차원 $p \geq 3$에서만 최대가능도 어림값을 앞서는 까닭을 밝혀라. 차원 1과 2에서는 어떻게 되는가?

??? success "연습문제 2 풀이"
    제임스-스타인 어림값은 최대가능도 어림값을 어떤 과녁 점 쪽으로 오그리며 $\hat{\theta}_{JS} = \bar{y} + (1 - (p-2)/\|y - \bar{y}\|^2)(y - \bar{y})$ 꼴이다. 여기서 $p$은 차원이다. 오그림 값 $(p-2)/\|y-\bar{y}\|^2$은 $p \geq 3$일 때만 양수이며, 그래서 이 결과에는 차원이 적어도 셋이어야 한다.

차원이 1과 2일 때 MLE는 받아들일 수 있다. 곧 평균 제곱 오차로 볼 때 어떤 어림자도 그것을 고르게 누를 수 없다는 뜻이다. 이것이 스타인의 역설이다. 낱낱의 어림값은 가장 좋지만, 매개변수를 셋 이상 함께 어림할 때는 공통 값 쪽으로 오그라뜨려 늘 더 잘할 수 있다.


---

**연습문제 3.**
야구 보기를 넓혀 경험 베이즈의 앞확률 갈래 셋을 견주어라. 베타, 로짓-정규, 그리고 비모수 알맹이 밀도 어림이다. 어느 것이 MSE을 가장 잘 줄이는가?

??? success "연습문제 3 풀이"
    ```python
import numpy as np
from scipy import stats
from scipy.optimize import minimize

np.random.seed(42)
n_players = 20
true_ab = np.random.beta(80, 220, n_players)
at_bats = np.random.randint(20, 100, n_players)
hits = np.array([np.random.binomial(ab, ta) for ab, ta in zip(at_bats, true_ab)])
obs_avg = hits / at_bats

# 1. 베타 앞확률(적률법)
m, v = np.mean(obs_avg), np.var(obs_avg)
a_b = m * (m*(1-m)/v - 1)
b_b = (1-m) * (m*(1-m)/v - 1)
eb_beta = (hits + a_b) / (at_bats + a_b + b_b)

# 2. 단순 오그라들기(로짓 눈금에서 정규)
logit_avg = np.log(obs_avg/(1-obs_avg+1e-10)+1e-10)
mu_l = np.mean(logit_avg)
sig_l = np.std(logit_avg)
w = sig_l**2 / (sig_l**2 + 1.0/at_bats)
eb_logit = 1/(1+np.exp(-(w*logit_avg + (1-w)*mu_l)))

# 3. 큰 평균 오그라들기
grand = np.mean(obs_avg)
eb_simple = 0.7*obs_avg + 0.3*grand

for name, est in [('Beta', eb_beta), ('Logit', eb_logit), ('Simple', eb_simple), ('Raw', obs_avg)]:
    print(f'{name:8s} MSE: {np.mean((est-true_ab)**2):.6f}')
```
자료가 저절로 베타-이항 모형을 따르므로 모수 가정이 실제와 잘 맞아, 타율에는 보통 베타 앞확률이 가장 좋다.

## 정리하며

**다룬 것** — 경험 베이즈

경험 베이즈는 베이즈 통계와 빈도주의 통계 사이의 흥미로운 자리에 놓인다.

앞의 연습문제 3개로 스스로 따져 볼 수 있다.
