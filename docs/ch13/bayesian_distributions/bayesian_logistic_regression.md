# 베이즈 로지스틱 회귀
## 개요

베이즈 로지스틱 회귀는 보통의 로지스틱 회귀 모형에 가중치 벡터 위의 앞확률 분포를 얹어, 매개변수와 예측 위의 온전한 뒤확률 분포를 낸다. 베이즈 선형 회귀와 달리 뒤확률이 **켤레가 아니어서** 어림 추론 방법이 필요하며, 그래서 16장의 해석적 결과와 18~19장의 어림 추론 기법을 잇는 자연스러운 다리가 된다.

---

## 모형 명세

### 가능도

$y_i \in \{0, 1\}$인 관찰 $\{(\mathbf{x}_i, y_i)\}_{i=1}^N$의 이진 분류에서는 다음과 같다.

$$
p(y_i = 1 \mid \mathbf{x}_i, \mathbf{w}) = \sigma(\mathbf{w}^\top \mathbf{x}_i) = \frac{1}{1 + \exp(-\mathbf{w}^\top \mathbf{x}_i)}
$$

데이터 전체의 가능도는 다음과 같다.

$$
p(\mathbf{y} \mid \mathbf{X}, \mathbf{w}) = \prod_{i=1}^N \sigma(\mathbf{w}^\top \mathbf{x}_i)^{y_i} \left[1 - \sigma(\mathbf{w}^\top \mathbf{x}_i)\right]^{1-y_i}
$$

### 앞확률

가중치 위의 가우스 앞확률이 벌주기 노릇을 한다.

$$
p(\mathbf{w}) = \mathcal{N}(\mathbf{w} \mid \mathbf{0}, \alpha^{-1} \mathbf{I})
$$

여기서 $\alpha$은 앞확률의 정밀도(흩어짐의 역수)를 다스린다. 최대 뒤확률 어림을 할 때 이는 L2 벌주기에 해당한다.

### 뒤확률

뒤확률에는 닫힌 꼴의 해가 없다.

$$
p(\mathbf{w} \mid \mathbf{X}, \mathbf{y}) = \frac{p(\mathbf{y} \mid \mathbf{X}, \mathbf{w}) \, p(\mathbf{w})}{p(\mathbf{y} \mid \mathbf{X})} \propto p(\mathbf{y} \mid \mathbf{X}, \mathbf{w}) \, p(\mathbf{w})
$$

가우스 앞확률과 (시그모이드를 거친) 베르누이 가능도의 곱은 어떤 표준 분포족에도 들지 않는다.

---

## 라플라스 어림

**라플라스 어림**은 최빈값과 굽음을 맞추어 뒤확률에 가우스 분포를 맞춘다.

$$
p(\mathbf{w} \mid \mathcal{D}) \approx q(\mathbf{w}) = \mathcal{N}(\mathbf{w} \mid \mathbf{w}_{\text{MAP}}, \mathbf{H}^{-1})
$$

여기서 $\mathbf{w}_{\text{MAP}}$은 최대 뒤확률 어림값이고 $\mathbf{H}$은 $\mathbf{w}_{\text{MAP}}$에서 음의 로그 뒤확률의 헤세 행렬이다.

### 최대 뒤확률 어림값 찾기

로그 뒤확률은 다음과 같다.

$$
\log p(\mathbf{w} \mid \mathcal{D}) = \sum_{i=1}^N \left[ y_i \log \sigma_i + (1-y_i) \log(1-\sigma_i) \right] - \frac{\alpha}{2} \|\mathbf{w}\|^2 + \text{const}
$$

여기서 $\sigma_i = \sigma(\mathbf{w}^\top \mathbf{x}_i)$이다.

**기울기:**

$$
\nabla_{\mathbf{w}} \log p(\mathbf{w} \mid \mathcal{D}) = \mathbf{X}^\top (\mathbf{y} - \boldsymbol{\sigma}) - \alpha \mathbf{w}
$$

**헤세 행렬:**

$$
\mathbf{H} = -\nabla^2_{\mathbf{w}} \log p(\mathbf{w} \mid \mathcal{D}) = \mathbf{X}^\top \mathbf{S} \mathbf{X} + \alpha \mathbf{I}
$$

여기서 $\mathbf{S} = \text{diag}(\sigma_i(1-\sigma_i))$이다.

### 되풀이 재가중 최소제곱(IRLS)

최대 뒤확률 어림값은 뉴턴-랩슨 되풀이로 찾는다.

$$
\mathbf{w}^{(t+1)} = \mathbf{w}^{(t)} - \mathbf{H}^{-1} \nabla_{\mathbf{w}} \log p(\mathbf{w}^{(t)} \mid \mathcal{D})
$$

이는 가중 최소제곱 문제를 잇달아 푸는 것과 같다.

### 예측 분포

새 입력 $\mathbf{x}_*$에 대해 예측 분포는 어림한 뒤확률 위에서 적분한다.

$$
p(y_* = 1 \mid \mathbf{x}_*, \mathcal{D}) = \int \sigma(\mathbf{w}^\top \mathbf{x}_*) \, q(\mathbf{w}) \, d\mathbf{w}
$$

이 적분은 **프로빗 어림**으로 어림한다.

$$
p(y_* = 1 \mid \mathbf{x}_*, \mathcal{D}) \approx \sigma\left(\frac{\mu_a}{\sqrt{1 + \pi \sigma_a^2 / 8}}\right)
$$

여기서 $\mu_a = \mathbf{w}_{\text{MAP}}^\top \mathbf{x}_*$이고 $\sigma_a^2 = \mathbf{x}_*^\top \mathbf{H}^{-1} \mathbf{x}_*$이다.

---

## PyTorch 구현

```python
import torch
import torch.nn.functional as F

class BayesianLogisticRegression:
    """
    라플라스 어림을 쓴 베이즈 로지스틱 회귀.
    
    앞확률: w ~ N(0, alpha^{-1} I)
    가능도: y | x, w ~ 베르누이(sigmoid(w^T x))
    뒤확률 ≈ N(w_MAP, H^{-1}) — 라플라스 어림으로
    """
    
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.w_map = None
        self.H_inv = None
    
    def fit(self, X: torch.Tensor, y: torch.Tensor, 
            max_iter: int = 100, tol: float = 1e-6):
        """IRLS으로 MAP 어림값을 찾는다."""
        N, D = X.shape
        w = torch.zeros(D, dtype=torch.float64)
        
        for iteration in range(max_iter):
            # 순전파
            logits = X @ w
            sigma = torch.sigmoid(logits)
            
            # 로그 뒤확률의 기울기
            grad = X.T @ (y - sigma) - self.alpha * w
            
            # 음의 로그 뒤확률의 헤세 행렬
            S = sigma * (1 - sigma)
            H = X.T @ (S.unsqueeze(1) * X) + self.alpha * torch.eye(D, dtype=torch.float64)
            
            # 뉴턴 걸음
            delta = torch.linalg.solve(H, grad)
            w = w + delta
            
            if torch.norm(delta) < tol:
                break
        
        self.w_map = w
        self.H_inv = torch.linalg.inv(H)
        return self
    
    def predict_proba(self, X_new: torch.Tensor) -> torch.Tensor:
        """
        불확실함을 적분해 넣은 예측 확률.
        
        뒤확률에 걸쳐 적분하려고 프로빗 어림을 쓴다.
        """
        mu_a = X_new @ self.w_map
        sigma2_a = (X_new @ self.H_inv * X_new).sum(dim=1)
        
        # 프로빗 어림
        kappa = 1.0 / torch.sqrt(1.0 + torch.pi * sigma2_a / 8.0)
        return torch.sigmoid(kappa * mu_a)
    
    def predict_map(self, X_new: torch.Tensor) -> torch.Tensor:
        """MAP 어림값을 쓴 점 예측(불확실함 없음)."""
        return torch.sigmoid(X_new @ self.w_map)
```

---

## 견줌: 최대 뒤확률 예측과 온전한 베이즈 예측

| 갈래 | 최대 뒤확률 예측 | 베이즈 예측 |
|--------|---------------|---------------------|
| 식 | $\sigma(\mathbf{w}_{\text{MAP}}^\top \mathbf{x}_*)$ | $\int \sigma(\mathbf{w}^\top \mathbf{x}_*) q(\mathbf{w}) d\mathbf{w}$ |
| 불확실성 | 없음 | $\sigma_a^2$으로 나타낸 앎의 불확실성 |
| 데이터에서 멀 때 | 지나치게 자신함 | 알맞게 아리송함 |
| 판단 경계 | 날카로움 | 부드러움(넘어가는 자리가 넓다) |
| 눈금 맞음 | 나쁠 때가 많음 | 대체로 나음 |

베이즈 예측은 늘 최대 뒤확률 예측보다 **덜 자신하며**, 학습 데이터에서 먼 입력일수록 그렇다. 위험에 민감한 응용에서는 바람직한 성질이다.

---

## 다른 방법과의 이음

| 방법 | 장 | 관계 |
|--------|---------|-------------|
| 가우스 앞확률을 쓴 최대 뒤확률 | 이 장 | L2 벌주기 로지스틱 회귀와 같다 |
| 변분 추론 | 19장 | 라플라스의 대안이며 규모를 키우기 좋다 |
| MCMC 표집 | 18장 | 정확한 뒤확률이지만 더 값비싸다 |
| 베이즈 신경망 | 19장(BNN) | 비선형 모형으로의 일반화 |

---

## 참고 문헌

- Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer. 4.5절.
- Murphy, K. P. (2012). *Machine Learning: A Probabilistic Perspective*. MIT Press. 8장.
- MacKay, D. J. C. (1992). The evidence framework applied to classification networks. *Neural Computation*, 4(5), 720-736.

## 연습문제

**연습문제 1.**
이 쪽이 다루는 핵심 개념과 그것이 베이즈 통계에서 하는 몫을 설명하라.

??? success "연습문제 1 풀이"
    이 쪽은 베이즈 추론의 근본 부품인 베이즈 로지스틱 회귀을(를) 다룬다. 이는 데이터로 믿음을 고치고, 불확실성을 수로 나타내며, 불확실함 속에서 결정을 내리는 더 넓은 틀과 이어진다. 베이즈의 눈은 앞선 앎을 아우르고 불확실성을 분석 전체로 퍼뜨리는 원칙 있는 길을 준다.

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
