# 가우스 과정

**가우스 과정(GP)**은 유한한 매개변수 모임이 아니라 함수 위에 곧바로 앞확률을 정하는, 회귀와 분류를 위한 비모수 베이즈 방법이다. GP은 닫힌 꼴의 예측 분포로 정확한 뒤확률 추론을 주며, 불확실성을 자연스럽게 수로 나타내고 모형의 복잡도를 데이터에 맞춘다.

---

## 1. 직관: 함수를 두고 따져 보기

대부분의 기계 학습 방법은 데이터에서 매개변수를 어림한다. 신경망의 가중치나 선형 모형의 계수 같은 것이다. GP은 근본부터 다르게 다가간다. 데이터에 맞을 만한 함수의 **큰 틀의 성질**을 곧바로 따져 본다. "어떤 가중치가 가장 좋은가?"가 아니라 "우리가 본 것과 어긋나지 않는 함수는 어떤 것들인가?"라고 묻는다.

들쭉날쭉한 간격으로 관찰한 자산 수익률의 시계열을 생각해 보자. 데이터를 보기 전에도 우리에게는 앞선 믿음이 있다. 함수가 어느 정도 매끄러울 것이고, 어쩌면 되풀이될 것이며(철마다의 본새), 관찰이 적은 자리에서는 더 아리송하리라는 것이다. GP은 알맹이 함수를 고르는 것만으로 이 가정을 모두 곧바로 담아낸다.

!!! note "GP의 일 흐름"

    1. **앞확률 정하기**: 매끄러움, 되풀이됨, 크기에 대한 가정을 담는 알맹이를 고른다
    2. **데이터로 조건 짓기**: 관찰과 어긋나지 않는 함수 위의 뒤확률 분포를 셈한다
    3. **예측하기**: 뒤확률의 평균이 점 어림값을 주고 뒤확률의 흩어짐이 눈금 맞은 불확실성을 준다

GP의 핵심 성질은 관찰한 데이터 점에서 먼 자리일수록 **앎의 불확실성**(데이터가 적어서 생기는 아리송함)이 자연스럽게 커진다는 것이다. 무엇을 *모르는지* 아는 것이 정확히 맞히는 것만큼 중요한 계량 금융에서 특히 값지다.

---

## 2. 정의

가우스 과정은 확률 변수의 모임으로, 그 가운데 유한한 개수를 뽑으면 언제나 결합 가우스 분포를 따른다. GP은 **평균 함수** $m(\mathbf{x})$과 **공분산(알맹이) 함수** $k(\mathbf{x}, \mathbf{x}')$으로 온전히 정해진다.

$$
f \sim \mathcal{GP}\bigl(m(\mathbf{x}), \, k(\mathbf{x}, \mathbf{x}')\bigr)
$$

유한한 입력 모임 $\mathbf{X} = \{\mathbf{x}_1, \ldots, \mathbf{x}_N\}$에 대해 다음이 성립한다.

$$
\mathbf{f} = \begin{pmatrix} f(\mathbf{x}_1) \\ \vdots \\ f(\mathbf{x}_N) \end{pmatrix} \sim \mathcal{N}\bigl(\boldsymbol{\mu}, \mathbf{K}\bigr)
$$

여기서 $\mu_i = m(\mathbf{x}_i)$이고 $K_{ij} = k(\mathbf{x}_i, \mathbf{x}_j)$이다.

---

## 3. 알맹이 함수

알맹이는 함수에 대한 앞선 가정, 곧 매끄러움, 되풀이됨, 길이 눈금, 진폭을 담는다.

### 흔한 알맹이

**제곱 지수(RBF):**

$$
k(\mathbf{x}, \mathbf{x}') = \sigma_f^2 \exp\left(-\frac{\|\mathbf{x} - \mathbf{x}'\|^2}{2\ell^2}\right)
$$

무한히 미분할 수 있는(아주 매끄러운) 함수를 낸다. **길이 눈금** $\ell$이 상관이 미치는 거리를, **신호 흩어짐** $\sigma_f^2$이 진폭을 다스린다.

**마테른 알맹이:**

$$
k_\nu(\mathbf{x}, \mathbf{x}') = \sigma_f^2 \frac{2^{1-\nu}}{\Gamma(\nu)} \left(\frac{\sqrt{2\nu}\,r}{\ell}\right)^\nu K_\nu\left(\frac{\sqrt{2\nu}\,r}{\ell}\right)
$$

여기서 $r = \|\mathbf{x} - \mathbf{x}'\|$이다. $\nu$으로 매끄러움을 다스리며 $\nu = 1/2$(오른슈타인-울렌벡), $\nu = 3/2$, $\nu = 5/2$이 흔한 선택이다. $\nu \to \infty$이면 RBF 알맹이가 된다.

**되풀이 알맹이:**

$$
k(\mathbf{x}, \mathbf{x}') = \sigma_f^2 \exp\left(-\frac{2\sin^2(\pi |\mathbf{x} - \mathbf{x}'|/p)}{\ell^2}\right)
$$

되풀이 주기 $p$을 아는 함수에 쓴다.

### 알맹이 조합

알맹이를 섞어 더 풍부한 앞선 짜임을 담을 수 있다.

| 연산 | 식 | 풀이 |
|-----------|---------|---------------|
| 합 | $k_1 + k_2$ | 서로 독립인 본새의 겹침 |
| 곱 | $k_1 \cdot k_2$ | 본새끼리의 어울림 |
| 눈금 | $\sigma^2 k$ | 진폭 다스리기 |

---

## 4. GP 회귀

### 모델

잡음 모형 $y_i = f(\mathbf{x}_i) + \epsilon_i$, $\epsilon_i \sim \mathcal{N}(0, \sigma_n^2)$을 갖춘 학습 데이터 $\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^N$이 주어지면 다음과 같다.

$$
\mathbf{y} \sim \mathcal{N}\bigl(\mathbf{0}, \, \mathbf{K} + \sigma_n^2 \mathbf{I}\bigr)
$$

### 뒤확률 예측

시험 입력 $\mathbf{X}_*$에 대해 예측 분포는 가우스 분포이다.

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

### 주변 가능도

초매개변수를 최적화할 때 쓰는 로그 주변 가능도는 다음과 같다.

$$
\log p(\mathbf{y} \mid \mathbf{X}, \boldsymbol{\theta}) = -\frac{1}{2}\mathbf{y}^\top \mathbf{K}_y^{-1} \mathbf{y} - \frac{1}{2}\log|\mathbf{K}_y| - \frac{N}{2}\log(2\pi)
$$

여기서 $\mathbf{K}_y = \mathbf{K} + \sigma_n^2 \mathbf{I}$이다. 이는 데이터에 맞추는 것(첫 항)과 모형의 복잡도(둘째 항) 사이의 균형을 저절로 잡는다. 베이즈판 오컴의 면도날이다.

---

## 5. PyTorch 구현

```python
import torch

class GaussianProcessRegressor:
    """
    RBF 알맹이를 쓴 가우스 과정 회귀.
    
    웃매개변수는 주변 가능도를 최대로 만들어 맞춘다.
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
        """RBF 알맹이 행렬을 셈한다."""
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
        """주변 가능도로 웃매개변수를 맞춘다."""
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

| 갈래 | 보통의 GP | 어림 |
|--------|-------------|---------------|
| 학습 | $O(N^3)$ | 성긴 GP: $O(NM^2)$ |
| 예측 | 점마다 $O(N^2)$ | 점마다 $O(M^2)$ |
| 기억 | $O(N^2)$ | $O(NM)$ |
| 유도점 $M$ | — | $M \ll N$ |

큰 규모의 금융 데이터셋에서는 성긴 GP 어림(FITC, VFE)이나 짜임새 있는 알맹이 사이 메우기(SKI)가 필요하다.

---

## 7. 계량 금융에서의 쓰임

GP은 불확실성을 수로 나타내는 힘 덕분에 금융 모형에 특히 잘 맞는다.

- **변동성 면 모형 세우기**: 행사가와 만기의 함수로 내재 변동성을 불확실성 띠와 함께 맞추기
- **수익률 곡선 메우기**: 바깥으로 뻗을 때의 불확실성을 원칙 있게 다루는 매끄러운 사이 메우기
- **알파 신호 모형 세우기**: 포지션 크기를 정하는 데 쓸 앎의 불확실성을 갖춘 비선형 요인 모형
- **베이즈 최적화**: 거래 전략의 초매개변수 손질

---

## 연습문제

**연습문제 1.**
이 쪽이 다루는 핵심 개념과 그것이 베이즈 통계에서 하는 몫을 설명하라.

??? success "연습문제 1 풀이"
    이 쪽은 베이즈 추론의 근본 부품인 가우스 과정을(를) 다룬다. 이는 데이터로 믿음을 고치고, 불확실성을 수로 나타내며, 불확실함 속에서 결정을 내리는 더 넓은 틀과 이어진다. 베이즈의 눈은 앞선 앎을 아우르고 불확실성을 분석 전체로 퍼뜨리는 원칙 있는 길을 준다.

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

| 개념 | 핵심 |
|---------|-----------|
| **GP의 정의** | 평균과 알맹이로 정해지는 함수 위의 앞확률 |
| **알맹이 고르기** | 매끄러움, 되풀이됨을 비롯한 짜임의 가정을 담는다 |
| **뒤확률** | 닫힌 꼴의 정확한 가우스 예측 분포 |
| **주변 가능도** | 자연스러운 모형 고르기 잣대(베이즈판 오컴의 면도날) |
| **규모 확장성** | $O(N^3)$ 탓에 곧바로 쓰기 어렵고, 성긴 방법으로 더 큰 데이터셋까지 넓힌다 |

---

**참고 문헌**

- Rasmussen, C. E., & Williams, C. K. I. (2006). *Gaussian Processes for Machine Learning*. MIT Press.
- Murphy, K. P. (2012). *Machine Learning: A Probabilistic Perspective*. MIT Press. 15장.
- Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer. 6.4절.
