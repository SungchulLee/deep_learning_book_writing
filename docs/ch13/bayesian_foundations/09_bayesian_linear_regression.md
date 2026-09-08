# 베이즈 선형 회귀

베이즈 선형 회귀는 모형 매개변수에 앞확률 분포를 얹고 온전한 뒤확률 분포를 셈하여 고전 선형 회귀를 넓힌다. 점 어림값을 내놓는 빈도주의 회귀와 달리 베이즈 길은 뒤확률로 불확실함을 정확히 재며, 예측 분포가 관측 잡음과 매개변수 불확실함을 모두 저절로 담는다.

## 1. 코드

```python
"""
베이즈 추론 — 모듈 9: 베이즈 선형 회귀
수준: 나아간 단계
주제: 베이즈 회귀, 예측 분포, 불확실함 재기

베이즈 선형 회귀는 매개변수와 예측에 걸친 온전한 뒤확률 분포를 주어
불확실함을 저절로 재어 준다.

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
베이즈 선형 회귀:

모형: y = Xβ + ε, 여기서 ε ~ N(0, σ²I)

앞확률: β ~ N(m₀, V₀)

뒤확률: β|y ~ N(mₙ, Vₙ)
여기서 각 기호는 다음과 같다.
  Vₙ = (V₀⁻¹ + (1/σ²)X'X)⁻¹
  mₙ = Vₙ(V₀⁻¹m₀ + (1/σ²)X'y)

예측 분포:
  y*|y ~ N(x*'mₙ, σ² + x*'Vₙx*)
  
덧붙은 x*'Vₙx* 항이 매개변수 불확실함을 담는다.
"""

def bayesian_linear_regression_demo():
    """
    불확실함 재기를 곁들인 베이즈 선형 회귀를 보인다.
    """
    print("="*70)
    print("BAYESIAN LINEAR REGRESSION")
    print("="*70)
    
    # 합성 데이터 생성
    np.random.seed(42)
    n = 30
    X = np.linspace(0, 10, n)
    true_intercept = 2.0
    true_slope = 1.5
    noise_std = 2.0
    
    y = true_intercept + true_slope * X + np.random.normal(0, noise_std, n)
    
    # 설계 행렬
    X_design = np.column_stack([np.ones(n), X])
    
    # 앞확률: 약하게만 알려 줌
    m0 = np.array([0, 0])
    V0 = np.eye(2) * 100
    
    # 뒤확률(잡음 흩어짐을 안다고 놓고)
    sigma_sq = noise_std ** 2
    V_inv = np.linalg.inv(V0) + (1/sigma_sq) * X_design.T @ X_design
    Vn = np.linalg.inv(V_inv)
    mn = Vn @ (np.linalg.inv(V0) @ m0 + (1/sigma_sq) * X_design.T @ y)
    
    print(f"\nTrue parameters: β₀={true_intercept}, β₁={true_slope}")
    print(f"Posterior mean: β₀={mn[0]:.3f}, β₁={mn[1]:.3f}")
    print(f"Posterior std:  β₀={np.sqrt(Vn[0,0]):.3f}, β₁={np.sqrt(Vn[1,1]):.3f}")
    
    # 예측
    X_test = np.linspace(-1, 11, 200)
    X_test_design = np.column_stack([np.ones(len(X_test)), X_test])
    
    # 평균 예측
    y_pred_mean = X_test_design @ mn
    
    # 예측 불확실함
    pred_var = sigma_sq + np.sum((X_test_design @ Vn) * X_test_design, axis=1)
    pred_std = np.sqrt(pred_var)
    
    # 시각화
    plt.figure(figsize=(14, 6))
    
    plt.subplot(1, 2, 1)
    plt.scatter(X, y, alpha=0.6, s=50, label='Data')
    plt.plot(X_test, y_pred_mean, 'r-', linewidth=2, label='Posterior mean')
    plt.fill_between(X_test, y_pred_mean - 2*pred_std, y_pred_mean + 2*pred_std,
                     alpha=0.3, color='red', label='95% Predictive interval')
    plt.xlabel('X', fontsize=12)
    plt.ylabel('y', fontsize=12)
    plt.title('Bayesian Linear Regression', fontsize=13, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 뒤확률에서 표집
    plt.subplot(1, 2, 2)
    plt.scatter(X, y, alpha=0.6, s=50)
    for _ in range(20):
        beta_sample = np.random.multivariate_normal(mn, Vn)
        y_sample = X_test_design @ beta_sample
        plt.plot(X_test, y_sample, 'r-', alpha=0.2, linewidth=1)
    plt.xlabel('X', fontsize=12)
    plt.ylabel('y', fontsize=12)
    plt.title('Posterior Samples of Regression Lines', fontsize=13, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('bayesian_linear_regression.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    print("\n" + "="*70)
    print("BAYESIAN INFERENCE - MODULE 9: BAYESIAN LINEAR REGRESSION")
    print("="*70)
    
    bayesian_linear_regression_demo()
    
    print("\n" + "="*70)
    print("MODULE 9 COMPLETE")
    print("="*70)
    print("\nKey takeaways:")
    print("1. Bayesian regression gives full posterior over parameters")
    print("2. Predictive distribution includes parameter uncertainty")
    print("3. Naturally regularized through prior")
    print("4. Uncertainty quantification is automatic")
    print("\nNext: Module 10 - Advanced Applications")
    print("="*70)```

## 2. 논의

베이즈 선형 회귀 모형은 $\varepsilon \sim N(0, \sigma^2 I)$인 $y = X\beta + \varepsilon$을 가정하고 정규 앞확률 $\beta \sim N(m_0, V_0)$을 둔다. 뒤확률은 켤레 정규다. 곧 $\beta | y \sim N(m_n, V_n)$이며 $V_n = (V_0^{-1} + \sigma^{-2} X^T X)^{-1}$이고 $m_n = V_n(V_0^{-1} m_0 + \sigma^{-2} X^T y)$이다.

고갱이 이점은 예측 분포다. 새 들임 $x_*$에 대해 예측은 $y_* | y \sim N(x_*^T m_n, \sigma^2 + x_*^T V_n x_*)$을 따른다. 덧붙은 마디 $x_*^T V_n x_*$이 매개변수 불확실성을 담아, 자료가 성긴 자리에서 예측 구간을 넓혀 불확실성 눈금을 저절로 맞춘다.

코드는 인공 자료를 만들고, 약하게 알려 주는 앞확률로 뒤확률을 해석적으로 셈하며, 뒤확률 평균 예측과 95% 예측 구간을 함께 그린다. $\beta$의 뒤확률 표본을 뽑으면 그럴듯한 회귀 직선의 무리가 보인다. 앞확률이 정칙화 노릇을 하며, $V_0 = \lambda^{-1} I$이면 베이즈 회귀가 능선 회귀와 이어진다.

## 연습문제

**연습문제 1.**
$y = \sin(x) + \varepsilon$에서 만든 자료에 대해 차수 3의 베이즈 다항 회귀를 짜라. 예측 평균과 95% 믿음 구간을 그려라.

??? success "연습문제 1 풀이"
    ```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
n = 40
X_raw = np.sort(np.random.uniform(0, 2*np.pi, n))
y = np.sin(X_raw) + np.random.normal(0, 0.3, n)
X_design = np.column_stack([X_raw**k for k in range(4)])

m0, V0 = np.zeros(4), np.eye(4) * 10
sigma_sq = 0.09
Vn = np.linalg.inv(np.linalg.inv(V0) + X_design.T @ X_design / sigma_sq)
mn = Vn @ (X_design.T @ y / sigma_sq)

X_test = np.linspace(0, 2*np.pi, 200)
X_td = np.column_stack([X_test**k for k in range(4)])
y_pred = X_td @ mn
pred_std = np.sqrt(sigma_sq + np.sum((X_td @ Vn) * X_td, axis=1))

plt.scatter(X_raw, y, alpha=0.6)
plt.plot(X_test, y_pred, 'r-', linewidth=2)
plt.fill_between(X_test, y_pred-2*pred_std, y_pred+2*pred_std, alpha=0.3, color='red')
plt.show()
```


---

**연습문제 2.**
앞확률이 $\beta \sim N(0, \lambda^{-1}I)$일 때 베이즈 뒤확률 평균이 능선 회귀 어림값과 같음을 보여라.

??? success "연습문제 2 풀이"
    뒤확률 평균은 $m_n = V_n(\sigma^{-2}X^Ty) = (\lambda I + \sigma^{-2}X^TX)^{-1}\sigma^{-2}X^Ty$이다. 분자와 분모에 $\sigma^2$을 곱하면 $m_n = (\lambda\sigma^2 I + X^TX)^{-1}X^Ty$이다. 이는 능선 회귀 풀이 $\hat{\beta}_{\text{ridge}} = (X^TX + \lambda\sigma^2 I)^{-1}X^Ty$과 꼭 같으므로 둘이 같음이 세워진다. $\square$


---

**연습문제 3.**
단순 선형 회귀에서 베이즈 예측 구간의 너비를 빈도주의 예측 구간과 견주어라. 자료점 30개를 만들고 시험 자리 100곳에서 둘 다 셈하여라.

??? success "연습문제 3 풀이"
    ```python
import numpy as np
from scipy import stats

np.random.seed(42)
n = 30
X = np.linspace(0, 10, n)
y = 2 + 1.5*X + np.random.normal(0, 2, n)
X_d = np.column_stack([np.ones(n), X])

# 베이즈
sig2 = 4.0
V0 = np.eye(2)*100
Vn = np.linalg.inv(np.linalg.inv(V0) + X_d.T@X_d/sig2)
mn = Vn @ (X_d.T@y/sig2)

X_test = np.linspace(-1, 11, 100)
X_td = np.column_stack([np.ones(100), X_test])
bayes_width = 2*1.96*np.sqrt(sig2 + np.sum((X_td@Vn)*X_td, axis=1))

# 빈도주의
beta_hat = np.linalg.lstsq(X_d, y, rcond=None)[0]
resid = y - X_d@beta_hat
s2 = np.sum(resid**2)/(n-2)
freq_width = 2*stats.t.ppf(0.975, n-2)*np.sqrt(s2*(1+np.sum((X_td@np.linalg.inv(X_d.T@X_d))*X_td, axis=1)))

print(f'Mean Bayesian width: {np.mean(bayes_width):.3f}')
print(f'Mean Frequentist width: {np.mean(freq_width):.3f}')
```
약하게만 알려 주는 앞확률을 쓰면 두 길은 아주 비슷한 구간을 내놓는다. 앞확률이 불확실함을 조금 더 보태므로 베이즈 구간이 살짝 더 넓다.

## 정리하며

**다룬 것** — 베이즈 선형 회귀

베이즈 선형 회귀 모형은 $\varepsilon \sim N(0, \sigma^2 I)$인 $y = X\beta + \varepsilon$을 가정하고 정규 앞확률 $\beta \sim N(m_0, V_0)$을 둔다.

앞의 연습문제 3개로 스스로 따져 볼 수 있다.
