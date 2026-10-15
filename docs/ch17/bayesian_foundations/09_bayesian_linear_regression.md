# 베이즈 선형 회귀

가우시안 잡음과 계수 사전분포를 가진 선형 회귀에서 사후평균·공분산을 계산하는 예제다. 잡음 분산을 알고 있다는 조건에서 켤레 갱신이 가능하다. 새 관측의 예측구간과 잠재 평균 함수의 신용구간을 구분한다.

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
    print("="*70)
```

**출력:**

```

======================================================================
BAYESIAN INFERENCE - MODULE 9: BAYESIAN LINEAR REGRESSION
======================================================================
======================================================================
BAYESIAN LINEAR REGRESSION
======================================================================

True parameters: β₀=2.0, β₁=1.5
Posterior mean: β₀=2.630, β₁=1.298
Posterior std:  β₀=0.711, β₁=0.122

======================================================================
MODULE 9 COMPLETE
======================================================================

Key takeaways:
1. Bayesian regression gives full posterior over parameters
2. Predictive distribution includes parameter uncertainty
3. Naturally regularized through prior
4. Uncertainty quantification is automatic

Next: Module 10 - Advanced Applications
======================================================================
```

## 2. 논의

모형은 $y=X\beta+\varepsilon$, $\varepsilon\sim N(0,\sigma^2I)$이며 $\beta\sim N(m_0,V_0)$다. σ²가 알려지고 V_0가 양의 정부호이면 다음 사후분포를 얻는다.

$$V_n=(V_0^{-1}+\sigma^{-2}X^\top X)^{-1},\qquad m_n=V_n(V_0^{-1}m_0+\sigma^{-2}X^\top y)$$

입력 x_*에서 잠재 평균의 분산은 x_*ᵀV_nx_*이고 새 관측 y_*의 예측 분산은 여기에 σ²를 더한다. 코드의 음영은 새 관측의 예측구간이다. ±2표준편차는 정규분포에서 약 95.45%이며 정확한 95% 분위수와 조금 다르다. 표시된 그림의 명칭과 이 근사를 구분한다.

사후분산은 선택한 선형 모형·사전·알려진 잡음 아래의 불확실성이다. 모형이 틀린 외삽 영역에서 실제 오차가 자동으로 보정된다는 뜻은 아니다. β 사후표본으로 그린 직선들은 잠재 평균 함수 표본이며 관측 잡음을 추가한 예측 표본과 다르다. 수치 구현은 역행렬을 직접 만들지만 큰 문제에서는 선형 시스템 풀이와 분해를 이용할 수 있다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff hard" title="어려움"></span>
아래 3차 다항 회귀의 음영이 잠재 평균의 신용구간인지 새 관측의 예측구간인지 판별하라.

</div>

??? success "연습문제 1 풀이"

    pred_std는 sigma_sq를 포함하므로 새 관측의 예측구간이다. 잠재 함수 구간에는 계수 불확실성 x_*ᵀV_nx_*만 사용한다. 다항식은 계수에 선형이지만 sin(x)를 정확히 표현하지는 않으므로 모형 오차도 남는다.

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

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff hard" title="어려움"></span>
계수 사전 N(0,λ⁻¹I)와 릿지 회귀의 계수 대응을 유도하라.

</div>

??? success "연습문제 2 풀이"

    $m_n=(X^\top X+\lambda\sigma^2I)^{-1}X^\top y$다. 잔차 제곱합에 대한 릿지 벌점 계수는 λσ²이며 사전 정밀도 λ 자체와 구분한다. 모든 계수에 같은 사전을 주면 절편도 벌점을 받는다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
아래 베이즈·빈도주의 구간 비교에서 잡음 분산 조건이 같은지 설명하라.

</div>

??? success "연습문제 3 풀이"

    베이즈는 참 잡음 분산 4를 알고 정규 분위수를 쓰고 빈도주의는 잔차로 분산을 추정해 t 분위수를 쓴다. 이 차이 때문에 어느 구간이 더 넓은지에 보편적 순서를 정할 수 없다. 같은 알려진 분산·평탄 사전 조건 또는 같은 미지 분산 조건을 맞춰 비교한다.

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


## 정리하며

알려진 잡음 분산의 정규 선형 모형에서는 계수 사후분포를 해석적으로 구한다. 잠재 함수와 새 관측의 불확실성을 구분하고, 비교 모형의 잡음·사전·절편 조건과 구간 수준을 맞춘다.
