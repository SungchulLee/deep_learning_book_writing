# 가우시안 과정

가우시안 과정(GP)은 유한한 입력 집합에서의 함수 값이 결합 정규분포를 따르도록 정의하는 함수 사전분포다. 커널로 입력 간 상관을 표현하며, 가우시안 관측 잡음과 고정 하이퍼파라미터의 회귀에서는 사후분포를 해석적으로 계산할 수 있다. 비가우시안 우도의 분류에서는 일반적으로 추가 근사가 필요하다.

## 1. 직관: 함수를 두고 따져 보기

함수의 평균과 입력 사이 공분산을 정하면 관측 전의 가능한 함수 분포를 정의할 수 있다. 관측 후에는 그 함수 값과 새 위치의 값 사이 관계를 이용해 조건부분포를 계산한다.

커널은 매끄러움·주기성·변화 척도 등의 가정을 담지만 실제 데이터에서 그 가정이 맞는지는 별도로 확인해야 한다. 사후분산은 선택한 모형 아래의 불확실성이며, 자료가 적거나 외삽한다는 이유만으로 실제 오차에 맞게 자동 보정되지는 않는다.

## 2. 정의

GP는 임의의 유한한 입력 집합에서 함수 값 벡터가 결합 가우시안 분포를 따르는 확률과정이다. 평균함수 m과 양의 준정부호 공분산 커널 k로 정의한다.

$$f\sim\mathcal{GP}(m,k),\qquad f(X)\sim N(m(X),K),\quad K_{ij}=k(x_i,x_j)$$

유효한 커널은 모든 유한 입력 집합에 대해 대칭 양의 준정부호 행렬을 만들어야 한다. 비모수라는 말이 커널·잡음 등에 하이퍼파라미터가 없다는 뜻은 아니다. 선형 커널의 GP처럼 유한 차원 웨이트 모형과 동등한 경우도 있다.

## 3. 커널 함수

커널의 형태와 하이퍼파라미터는 함수 값 사이의 상관을 정한다. 아래는 대표적인 공분산 함수다.

### 대표 커널

**제곱 지수(RBF):**

$$
k(\mathbf{x}, \mathbf{x}') = \sigma_f^2 \exp\left(-\frac{\|\mathbf{x} - \mathbf{x}'\|^2}{2\ell^2}\right)
$$

무한히 미분할 수 있는(아주 매끄러운) 함수를 낸다. **길이 척도** $\ell$이 상관이 미치는 거리를, **신호 분산** $\sigma_f^2$이 진폭을 다스린다.

**마테른 커널:**

$$
k_\nu(\mathbf{x}, \mathbf{x}') = \sigma_f^2 \frac{2^{1-\nu}}{\Gamma(\nu)} \left(\frac{\sqrt{2\nu}\,r}{\ell}\right)^\nu K_\nu\left(\frac{\sqrt{2\nu}\,r}{\ell}\right)
$$

여기서 $r = \|\mathbf{x} - \mathbf{x}'\|$이다. $\nu$으로 매끄러움을 다스리며 $\nu = 1/2$(오른슈타인-울렌벡), $\nu = 3/2$, $\nu = 5/2$이 흔한 선택이다. $\nu \to \infty$이면 RBF 커널이 된다.

**주기 커널:**

$$
k(\mathbf{x}, \mathbf{x}') = \sigma_f^2 \exp\left(-\frac{2\sin^2(\pi |\mathbf{x} - \mathbf{x}'|/p)}{\ell^2}\right)
$$

주기 커널 식은 1차원 입력에서 주기 p를 표현한다. 벡터 입력에는 어떤 거리나 축별 주기를 사용할지 별도로 정의한다.

### 커널 결합

커널을 섞어 더 풍부한 앞선 구조를 담을 수 있다.

| 연산 | 식 | 의미 |
|---|---|---|
| 합 | k₁+k₂ | 독립 GP 성분의 합의 공분산 |
| 곱 | k₁k₂ | 두 커널 특성을 결합한 유효 공분산 |
| 비음수 배율 | c k, c≥0 | 분산 크기 조절 |

커널 곱이 유효하더라도 독립 가우시안 함수 두 개를 점별로 곱한 과정 자체가 일반적으로 GP인 것은 아니다.

---

## 4. GP 회귀

### 모델

이하 수식은 평균함수 0, 독립 가우시안 관측 잡음을 가정한다. 평균이 0이 아니면 y에서 m(X)를 빼고 예측에 m(X_*)를 더한다.

$$
\mathbf{y} \sim \mathcal{N}\bigl(\mathbf{0}, \, \mathbf{K} + \sigma_n^2 \mathbf{I}\bigr)
$$

### 잠재 함수의 사후예측

시험 입력 $\mathbf{X}_*$에 대해 예측 분포는 가우시안 분포다.

$$
f_* \mid \mathbf{X}_*, \mathbf{X}, \mathbf{y} \sim \mathcal{N}(\bar{f}_*, \text{cov}(f_*))
$$

$$
\bar{f}_* = \mathbf{K}_*^\top (\mathbf{K} + \sigma_n^2 \mathbf{I})^{-1} \mathbf{y}
$$

$$
\text{cov}(f_*) = \mathbf{K}_{**} - \mathbf{K}_*^\top (\mathbf{K} + \sigma_n^2 \mathbf{I})^{-1} \mathbf{K}_*
$$

여기서 $\mathbf{K}_* = k(\mathbf{X}, \mathbf{X}_*)$이고 $\mathbf{K}_{**} = k(\mathbf{X}_*, \mathbf{X}_*)$이다.

### 주변 우도

하이퍼파라미터를 최적화할 때 쓰는 로그 주변 우도는 다음과 같다.

$$
\log p(\mathbf{y} \mid \mathbf{X}, \boldsymbol{\theta}) = -\frac{1}{2}\mathbf{y}^\top \mathbf{K}_y^{-1} \mathbf{y} - \frac{1}{2}\log|\mathbf{K}_y| - \frac{N}{2}\log(2\pi)
$$

K_y=K+σ_n²I다. 주변우도는 자료 적합도와 공분산 크기·구조를 함께 평가하지만 하이퍼파라미터의 유일한 최적점이나 올바른 모형을 보장하지 않는다. 이를 최대화해 고정한 하이퍼파라미터의 예측은 그 값의 사후 불확실성을 적분한 완전한 베이즈 예측과 다르다.

---

## 5. PyTorch 구현

아래 모델은 RBF 커널과 상수 잡음의 0평균 회귀만 구현한다. 양의 초기 길이·신호 분산·잡음 분산을 로그 좌표로 최적화한다. CPU에 하이퍼파라미터와 단위행렬을 만들므로 입력 장치·자료형을 맞춰야 한다. 모델을 nn.Module로 만들지 않아 일반적인 .to() 이동도 제공하지 않는다.

predict는 잠재 함수 f_*의 분산을 반환하며 새 관측 분산에는 noise_var를 더해야 한다. N=1에서는 alpha.squeeze()가 스칼라가 되어 행렬 연산의 차원이 맞지 않을 수 있다. 작은 잡음과 중복 입력의 조건수, Cholesky 실패와 분산의 수치 오차에 대한 처리가 별도로 필요하다.

```python
import torch

class GaussianProcessRegressor:
    """
    RBF 커널을 쓴 가우스 과정 회귀.
    
    웃파라메터는 주변 가능도를 최대로 만들어 맞춘다.
    """
    
    def __init__(self, length_scale: float = 1.0, signal_var: float = 1.0, 
                 noise_var: float = 0.1):
        self.log_length_scale = torch.tensor(
            [torch.log(torch.tensor(length_scale))], requires_grad=True)
        self.log_signal_var = torch.tensor(
            [torch.log(torch.tensor(signal_var))], requires_grad=True)
        self.log_noise_var = torch.tensor(
            [torch.log(torch.tensor(noise_var))], requires_grad=True)
    
    def rbf_kernel(self, X1, X2):
        """RBF 커널 행렬을 셈한다."""
        ell = torch.exp(self.log_length_scale)
        sf2 = torch.exp(self.log_signal_var)
        
        dist_sq = torch.cdist(X1 / ell, X2 / ell, p=2) ** 2
        return sf2 * torch.exp(-0.5 * dist_sq)
    
    def log_marginal_likelihood(self, X, y):
        """로그 주변 가능도를 셈한다."""
        sn2 = torch.exp(self.log_noise_var)
        K = self.rbf_kernel(X, X) + sn2 * torch.eye(len(X))
        
        L = torch.linalg.cholesky(K)
        alpha = torch.cholesky_solve(y.unsqueeze(1), L).squeeze()
        
        lml = -0.5 * y @ alpha - L.diagonal().log().sum() - 0.5 * len(y) * torch.log(
            torch.tensor(2 * torch.pi))
        return lml
    
    def fit(self, X, y, n_iter=100, lr=0.1):
        """주변 가능도로 웃파라메터를 맞춘다."""
        self.X_train = X
        self.y_train = y
        
        optimizer = torch.optim.Adam(
            [self.log_length_scale, self.log_signal_var, self.log_noise_var], lr=lr)
        
        for i in range(n_iter):
            optimizer.zero_grad()
            loss = -self.log_marginal_likelihood(X, y)
            loss.backward()
            optimizer.step()
        
        return self
    
    def predict(self, X_new):
        """뒤확률 예측 평균과 흩어짐."""
        with torch.no_grad():
            sn2 = torch.exp(self.log_noise_var)
            K = self.rbf_kernel(self.X_train, self.X_train) + sn2 * torch.eye(
                len(self.X_train))
            K_star = self.rbf_kernel(self.X_train, X_new)
            K_ss = self.rbf_kernel(X_new, X_new)
            
            L = torch.linalg.cholesky(K)
            alpha = torch.cholesky_solve(
                self.y_train.unsqueeze(1), L).squeeze()
            
            mean = K_star.T @ alpha
            v = torch.linalg.solve_triangular(L, K_star, upper=False)
            var = K_ss.diag() - (v ** 2).sum(dim=0)
            
        return mean, var
```

---

## 6. 계산에 대한 고려

밀집 GP 회귀의 N×N 공분산 분해는 일반적으로 O(N³) 계산과 O(N²) 저장을 사용한다. 학습 분해를 캐시한 뒤 새 점의 평균은 O(N), 분산은 O(N²)에 계산할 수 있지만 이 코드의 predict는 매 호출 학습 공분산을 다시 분해한다. 또한 M개 새 입력의 K_ss 전체를 만들므로 예측점 수에 대해 O(M²) 저장도 든다.

유도점·구조화 커널 등의 근사는 규모를 줄일 수 있지만 방법별 비용과 근사 가정이 다르다. 예를 들어 대표적인 유도점 방식은 유도점 수 I에 대해 O(NI²) 수준의 학습 계산을 사용한다. 이를 모든 희소 GP의 동일한 메모리·시간 보장으로 해석하지 않는다.

## 7. 계량 금융에서의 쓰임

함수 회귀·보간의 불확실성을 표현하는 모형으로 변동성 곡면이나 수익률 곡선 등에 적용해 볼 수 있다. 이때 가우시안 잡음·정상성·매끄러움 가정이 목표에 맞는지 확인한다. 일반 GP만으로 무차익 조건이나 꼬리 위험 한도를 보장하지는 않는다.

시계열은 시간 순서를 지켜 검증하고, 미래 정보를 사용한 커널·하이퍼파라미터 선택을 피한다. RBF 커널에서 학습 입력과 상관이 사라지면 잠재 예측은 지정한 사전 평균·분산으로 돌아가는 경향이 있다. 분산이 거리와 함께 무한히 증가한다는 뜻은 아니다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
GP의 정의와 해석적 사후추론이 가능한 조건을 설명하라.

</div>

??? success "연습문제 1 풀이"
    유한한 함수 값 벡터가 결합 가우시안인 확률과정이다. 고정 커널 하이퍼파라미터와 가우시안 관측 잡음의 회귀에서는 조건부 정규 공식으로 사후를 구한다. 로지스틱 분류 등 비가우시안 우도에서는 보통 근사 추론이 필요하다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
본문의 공분산 수식과 코드 반환값에 관측 잡음이 포함되는지 설명하라.

</div>

??? success "연습문제 2 풀이"
    사후의 f_* 공분산과 predict의 var는 잠재 함수에 대한 값이다. 독립 새 관측 y_*의 공분산에는 σ_n²I를 더한다. 학습 공분산에 잡음을 더했다는 사실과 새 예측 대상의 잡음 포함 여부는 다르다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
커널 하이퍼파라미터를 주변우도로 최적화하면 완전한 베이즈 예측인가?

</div>

??? success "연습문제 3 풀이"
    그 값에 조건부인 잠재 함수의 사후는 계산하지만 하이퍼파라미터의 불확실성은 적분하지 않는다. 데이터에서 추정한 값을 대입하는 경험적 베이즈 방식으로 볼 수 있다. 하이퍼사전과 추가 적분을 사용하는 설정과 구분한다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
아래 임포트만 있는 조각을 GP 구현으로 완성할 때 점검할 사항을 설명하라.

</div>

??? success "연습문제 4 풀이"
    유효한 커널과 양의 잡음을 정의하고 안정적인 공분산 분해로 조건부 평균·분산을 계산한다. 입력 모양·장치·자료형과 N=1을 점검하고, 잠재·관측 분산의 반환 규약을 명시해야 한다. import 한 줄은 구현이 아니다.

    ```python
    import numpy as np
    # 구현은 주제에 따라 달라진다.
    # 켤레 모형: 닫힌 꼴 뒤확률 새로 고치기.
    # 켤레가 아닌 모형: MCMC 또는 변분 추론.
    # 핵심 스텝: 앞확률 정하기, 가능도 셈하기, 뒤확률 이끌어 내기/어림하기.
    ```

## 정리하며

GP는 커널로 함수의 공분산을 정의한다. 가우시안 잡음 회귀의 조건부 계산과 하이퍼파라미터 추정·불확실성을 구분한다. 잠재 함수·새 관측 분산, 공분산 분해와 예측 캐시의 비용을 명시하고 모형 조건 아래의 구간을 실제 오차 보장으로 해석하지 않는다.

**참고 문헌**

- Rasmussen, C. E., & Williams, C. K. I. (2006). *Gaussian Processes for Machine Learning*. MIT Press.
- Murphy, K. P. (2012). *Machine Learning: A Probabilistic Perspective*. MIT Press. 15장.
- Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer. 7.4절.
