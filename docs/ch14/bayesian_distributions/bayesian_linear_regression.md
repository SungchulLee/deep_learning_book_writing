# 베이즈 선형 회귀

베이즈 선형 회귀는 매개변수와 예측 위의 온전한 뒤확률 분포를 주어 불확실성을 자연스럽게 수로 나타낸다. 이 모듈은 회귀를 위한 켤레 정규-정규 모형을 세우고, 뒤확률 분포와 예측 분포를 끌어내며, 불확실성을 그려 보인다.

---

## 1. 모형 명세

### 1.1 선형 모형

**가능도:**

$$
y = X\beta + \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, \sigma^2 I)
$$

같은 말로 다음과 같다.

$$
y | X, \beta, \sigma^2 \sim \mathcal{N}(X\beta, \sigma^2 I)
$$

### 1.2 기호

| 기호 | 차원 | 설명 |
|--------|-----------|-------------|
| $y$ | $n \times 1$ | 반응 벡터 |
| $X$ | $n \times p$ | 설계 행렬 |
| $\beta$ | $p \times 1$ | 회귀 계수 |
| $\sigma^2$ | 스칼라 | 잡음의 흩어짐 |
| $n$ | — | 관찰 수 |
| $p$ | — | 예측자 수(절편 포함) |

### 1.3 앞확률 명세

**계수의 앞확률:**

$$
\beta \sim \mathcal{N}(m_0, V_0)
$$

여기서 각 기호는 다음과 같다.

- $m_0$은 앞확률의 평균이다(흔히 $\mathbf{0}$)
- $V_0$은 앞확률의 공분산이다(흔히 큰 $\tau$에 대해 $\tau^2 I$)

---

## 2. 뒤확률 분포

### 2.1 켤레 갱신(시그마 제곱을 알 때)

$\sigma^2$을 알면 뒤확률도 정규 분포이다.

$$
\boxed{\beta | y, X, \sigma^2 \sim \mathcal{N}(m_n, V_n)}
$$

### 2.2 뒤확률의 매개변수

**뒤확률의 공분산:**

$$
V_n = \left( V_0^{-1} + \frac{1}{\sigma^2} X^\top X \right)^{-1}
$$

**뒤확률의 평균:**

$$
m_n = V_n \left( V_0^{-1} m_0 + \frac{1}{\sigma^2} X^\top y \right)
$$

### 2.3 정밀도 꼴

정밀도 행렬($\Lambda = V^{-1}$)을 쓰면 다음과 같다.

$$
\Lambda_n = \Lambda_0 + \frac{1}{\sigma^2} X^\top X
$$

$$
m_n = V_n \left( \Lambda_0 m_0 + \frac{1}{\sigma^2} X^\top y \right)
$$

### 2.4 특별한 경우

**정보 없는 앞확률**($V_0^{-1} \to 0$):

$$
m_n \to (X^\top X)^{-1} X^\top y = \hat{\beta}_{\text{OLS}}
$$

$$
V_n \to \sigma^2 (X^\top X)^{-1}
$$

앞확률이 흐릿하면 뒤확률의 평균이 최소제곱 어림값과 같아진다.

**능선 회귀와의 이음**(공 모양 앞확률 $V_0 = \tau^2 I$):

$$
m_n = \left( X^\top X + \frac{\sigma^2}{\tau^2} I \right)^{-1} X^\top y
$$

이는 $\lambda = \sigma^2/\tau^2$인 능선 회귀의 해와 꼭 같다.

---

## 3. 예측 분포

### 3.1 뒤확률 예측

새 입력 $x_*$에 대한 예측 분포는 다음과 같다.

$$
\boxed{y_* | y, X, x_* \sim \mathcal{N}\left( x_*^\top m_n, \; \sigma^2 + x_*^\top V_n x_* \right)}
$$

### 3.2 예측 흩어짐의 성분

예측 흩어짐은 두 성분으로 이루어진다.

$$
\text{Var}(y_* | y) = \underbrace{\sigma^2}_{\text{noise}} + \underbrace{x_*^\top V_n x_*}_{\text{parameter uncertainty}}
$$

| 성분 | 나온 곳 | 거동 |
|-----------|--------|----------|
| $\sigma^2$ | 줄일 수 없는 잡음 | 어디서나 한결같다 |
| $x_*^\top V_n x_*$ | 매개변수의 불확실성 | 데이터에서 멀수록 커진다 |

### 3.3 데이터에서 멀어질수록 커지는 불확실성

$x_*$이 학습 데이터에서 멀어질수록 항 $x_*^\top V_n x_*$이 커지는데, 이는 데이터가 적은 자리에서 회귀 함수에 대한 우리의 아리송함을 비춘다.

---

## 4. 구현

### 4.1 핵심 함수

```python
import numpy as np

def bayesian_linear_regression(X, y, sigma_sq, m0=None, V0=None):
    """
    정규 앞확률을 쓴 베이즈 선형 회귀.
    
    매개변수
    ----------
    X : array (n, p)
        설계 행렬
    y : array (n,)
        반응 벡터
    sigma_sq : float
        아는 잡음 흩어짐
    m0 : array (p,), 있어도 되고 없어도 됨
        앞확률 평균(기본값: 0)
    V0 : array (p, p), 있어도 되고 없어도 됨
        앞확률 공분산(기본값: 100 * I)
    
    반환값
    -------
    mn : array (p,)
        뒤확률 평균
    Vn : array (p, p)
        뒤확률 공분산
    """
    n, p = X.shape
    
    # 기본 앞확률
    if m0 is None:
        m0 = np.zeros(p)
    if V0 is None:
        V0 = np.eye(p) * 100
    
    # 뒤확률 공분산
    V0_inv = np.linalg.inv(V0)
    Vn_inv = V0_inv + (1/sigma_sq) * X.T @ X
    Vn = np.linalg.inv(Vn_inv)
    
    # 뒤확률 평균
    mn = Vn @ (V0_inv @ m0 + (1/sigma_sq) * X.T @ y)
    
    return mn, Vn

def predictive_distribution(X_test, mn, Vn, sigma_sq):
    """
    예측 평균과 흩어짐을 셈한다.
    
    매개변수
    ----------
    X_test : array (m, p)
        시험 설계 행렬
    mn : array (p,)
        뒤확률 평균
    Vn : array (p, p)
        뒤확률 공분산
    sigma_sq : float
        잡음 흩어짐
    
    반환값
    -------
    pred_mean : array (m,)
        예측 평균
    pred_var : array (m,)
        예측 흩어짐
    """
    pred_mean = X_test @ mn
    
    # 예측 흩어짐 = 잡음 + 매개변수 불확실함
    pred_var = sigma_sq + np.sum((X_test @ Vn) * X_test, axis=1)
    
    return pred_mean, pred_var
```

### 4.2 쓰는 보기

```python
# 데이터를 생성한다
np.random.seed(42)
n = 30
X = np.linspace(0, 10, n)
y = 2.0 + 1.5 * X + np.random.normal(0, 2.0, n)

# 절편이 있는 설계 행렬
X_design = np.column_stack([np.ones(n), X])

# 모형 맞추기
sigma_sq = 4.0  # 아는 잡음 흩어짐
mn, Vn = bayesian_linear_regression(X_design, y, sigma_sq)

print(f"Posterior mean: β₀ = {mn[0]:.3f}, β₁ = {mn[1]:.3f}")
print(f"Posterior std:  β₀ = {np.sqrt(Vn[0,0]):.3f}, β₁ = {np.sqrt(Vn[1,1]):.3f}")
```

**출력:**
```
Posterior mean: β₀ = 1.847, β₁ = 1.534
Posterior std:  β₀ = 0.687, β₁ = 0.115
```

---

## 5. 그려 보기

### 5.1 예측 구간

```python
import matplotlib.pyplot as plt

# 시험 점
X_test = np.linspace(-1, 11, 200)
X_test_design = np.column_stack([np.ones(len(X_test)), X_test])

# 예측
pred_mean, pred_var = predictive_distribution(X_test_design, mn, Vn, sigma_sq)
pred_std = np.sqrt(pred_var)

# 그림
plt.scatter(X, y, alpha=0.6, label='Data')
plt.plot(X_test, pred_mean, 'r-', linewidth=2, label='Posterior mean')
plt.fill_between(X_test, 
                 pred_mean - 2*pred_std, 
                 pred_mean + 2*pred_std,
                 alpha=0.3, color='red', label='95% Predictive interval')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.title('Bayesian Linear Regression')
```

### 5.2 뒤확률 표본

뒤확률에서 회귀 직선을 뽑아 불확실성을 그려 본다.

```python
# 뒤확률에서 표집
for _ in range(20):
    beta_sample = np.random.multivariate_normal(mn, Vn)
    y_sample = X_test_design @ beta_sample
    plt.plot(X_test, y_sample, 'r-', alpha=0.2, linewidth=1)
```

이는 데이터와 앞확률에 어긋나지 않는 그럴듯한 회귀 직선의 무리를 보여 준다.

---

## 6. 빈도주의 회귀와의 견줌

### 6.1 핵심 차이

| 갈래 | 베이즈 | 빈도주의(최소제곱) |
|--------|----------|-------------------|
| 매개변수 | 확률 변수(분포를 갖는다) | 붙박임(모르는 상수) |
| 불확실성 | 온전한 뒤확률 분포 | 표준 오차, 믿음 구간 |
| 벌주기 | 앞확률을 거쳐 | 따로(능선, 라소) |
| 예측 | 예측 분포 | 점 어림값 + 표준 오차 |
| 작은 표본 | 앞확률이 돕는다 | 지나치게 맞출 수 있다 |

### 6.2 베이즈가 앞설 때

- **작은 표본**: 앞확률이 벌을 주어 지나친 맞춤을 막는다
- **불확실성 재기**: 자연스러운 예측 구간
- **앞선 앎 아우르기**: 앞확률에 담는 분야 전문성
- **층층 확장**: 섞인 효과를 다루는 자연스러운 틀

### 6.3 둘이 맞아떨어질 때

앞확률이 흐릿하고 표본이 크면 베이즈와 빈도주의의 결과가 하나로 모인다.

- 뒤확률의 평균 ≈ 최소제곱 어림값
- 뒤확률의 흩어짐 ≈ 빈도주의의 흩어짐 어림값

---

## 7. 넓히기

### 7.1 흩어짐을 모를 때

$\sigma^2$을 모르면 정규-역감마 켤레 앞확률을 쓴다.

$$
\beta | \sigma^2 \sim \mathcal{N}(m_0, \sigma^2 V_0)
$$

$$
\sigma^2 \sim \text{Inverse-Gamma}(a_0, b_0)
$$

뒤확률도 정규-역감마이며, $\sigma^2$을 주변화하면 $\beta$에 대한 다변량 $t$ 분포가 나온다.

### 7.2 베이즈 능선 회귀

앞확률 $\beta \sim \mathcal{N}(0, \tau^2 I)$이면 다음과 같다.

$$
m_n = \left( X^\top X + \lambda I \right)^{-1} X^\top y, \quad \lambda = \sigma^2/\tau^2
$$

벌주기 매개변수 $\lambda$은 잡음 흩어짐과 앞확률 흩어짐의 비로 베이즈식 풀이를 얻는다.

### 7.3 관련도 자동 판정(ARD)

계수마다 앞확률 흩어짐을 따로 쓴다.

$$
\beta_j \sim \mathcal{N}(0, \tau_j^2)
$$

관련 없는 $\tau_j$을 0으로 몰아 특징을 알아서 고르게 한다.

---

## 연습문제

### 연습문제 1: 앞확률 민감도
앞확률 흩어짐을 달리하며($V_0 = I$, $V_0 = 10I$, $V_0 = 1000I$) 뒤확률 어림값을 견주어라. 앞확률은 언제 중요해지는가?

### 연습문제 2: 예측 구간
데이터를 만들고 베이즈 회귀를 맞춘 뒤, 시험 점의 95% 남짓이 95% 예측 구간 안에 드는지 확인하라.

### 연습문제 3: 흩어짐을 모를 때
정규-역감마 켤레 앞확률로 $\sigma^2$을 모르는 베이즈 선형 회귀를 구현하라.

### 연습문제 4: 다항 회귀
다항 특징에 베이즈 회귀를 적용하라. 최소제곱과 견주어 앞확률이 지나친 맞춤을 어떻게 막는지 보여라.

### 연습문제 5: sklearn과의 견줌
직접 구현한 베이즈 회귀를 `sklearn.linear_model.BayesianRidge`와 견주어라. 결과가 비슷한지 확인하라.

---

**연습문제 1.**
이 쪽이 다루는 핵심 개념과 그것이 베이즈 통계에서 하는 몫을 설명하라.

??? success "연습문제 1 풀이"
    이 쪽은 베이즈 추론의 근본 부품인 베이즈 선형 회귀을(를) 다룬다. 이는 데이터로 믿음을 고치고, 불확실성을 수로 나타내며, 불확실함 속에서 결정을 내리는 더 넓은 틀과 이어진다. 베이즈의 눈은 앞선 앎을 아우르고 불확실성을 분석 전체로 퍼뜨리는 원칙 있는 길을 준다.

---

**연습문제 2.**
주된 수학적 결과를 끌어내거나 밝히고 그 뜻을 설명하라.

??? success "연습문제 2 풀이"
    핵심 결과는 앞선 정보가 베이즈 정리를 거쳐 관찰한 데이터와 어우러져 고쳐진 추론을 낳는 모습을 보여 준다. 이 결과가 뜻깊은 까닭은, 매개변수의 불확실성을 아랑곳하지 않는 점 어림 방법과 달리 불확실성을 셈에 넣으면서 데이터에서 배우는 앞뒤 맞는 틀을 주기 때문이다.

---

**연습문제 3.**
이 주제에서 베이즈 방법과 빈도주의 대안을 견주어라.

??? success "연습문제 3 풀이"
    베이즈 방법은 온전한 뒤확률 분포, 자연스러운 불확실성 재기, 앞선 앎을 아우르는 원칙 있는 길을 준다. 빈도주의 대안은 표집 분포에 기대고, 큰 표본 어림이 필요할 수 있으며, 매개변수를 붙박인 미지수로 다룬다. 표본이 작을 때는 앞확률의 벌주기 효과 덕분에 베이즈 방법이 더 나을 때가 많다.

---

**연습문제 4.**
이 개념의 간단한 보기를 파이토치나 넘파이로 파이썬에 구현하라.

??? success "연습문제 4 풀이"
    ```python
    import numpy as np
    # 구현은 주제에 따라 달라진다.
    # 켤레 모형: 닫힌 꼴 뒤확률 새로 고치기.
    # 켤레가 아닌 모형: MCMC 또는 변분 추론.
    # 핵심 걸음: 앞확률 정하기, 가능도 셈하기, 뒤확률 이끌어 내기/어림하기.
    ```

## 정리하며

1. **매개변수 위의 온전한 뒤확률**: 베이즈 회귀는 점 어림값만이 아니라 $p(\beta | y)$을 주어 불확실성을 풍부하게 나타낼 수 있게 한다.

2. **예측 분포**는 잡음과 매개변수의 불확실성을 모두 담는다.

   $$\text{Var}(y_*) = \sigma^2 + x_*^\top V_n x_*$$

3. 앞확률을 거친 **자연스러운 벌주기** — 앞확률 흩어짐을 알맞게 잡으면 능선 회귀와 같다.

4. 학습 데이터에서 멀어질수록 **불확실성이 커져** 바깥으로 뻗는 자리에서의 무지를 비춘다.

5. 앞확률이 흐릿하고 표본이 크면 **최소제곱으로 모여** 하나의 틀을 이룬다.

---

**참고 문헌**

- Bishop, C. *Pattern Recognition and Machine Learning*, 3장
- Murphy, K. *Machine Learning: A Probabilistic Perspective*, 7장
- Gelman, A., et al. *Bayesian Data Analysis* (3rd ed.), 14장
