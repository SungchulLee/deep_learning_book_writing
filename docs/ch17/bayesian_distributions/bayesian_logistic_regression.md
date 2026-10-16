# 베이즈 로지스틱 회귀

베이즈 로지스틱 회귀는 이진 분류기의 웨이트에 사전분포를 두고 예측에서 그 불확실성을 반영한다. 정규 사전은 로지스틱 우도에 켤레가 아니므로 여기서는 MAP 주변의 라플라스 근사를 사용한다. 해석적인 정규 회귀와 근사 사후추론의 차이를 살펴본다.

## 1. 모델 명세

### 우도

$y_i \in \{0, 1\}$인 관찰 $\{(\mathbf{x}_i, y_i)\}_{i=1}^N$의 이진 분류에서는 다음과 같다.

$$
p(y_i = 1 \mid \mathbf{x}_i, \mathbf{w}) = \sigma(\mathbf{w}^\top \mathbf{x}_i) = \frac{1}{1 + \exp(-\mathbf{w}^\top \mathbf{x}_i)}
$$

데이터 전체의 우도는 다음과 같다.

$$
p(\mathbf{y} \mid \mathbf{X}, \mathbf{w}) = \prod_{i=1}^N \sigma(\mathbf{w}^\top \mathbf{x}_i)^{y_i} \left[1 - \sigma(\mathbf{w}^\top \mathbf{x}_i)\right]^{1-y_i}
$$

### 사전확률

웨이트 위의 가우스 사전확률이 정규화 노릇을 한다.

$$
p(\mathbf{w}) = \mathcal{N}(\mathbf{w} \mid \mathbf{0}, \alpha^{-1} \mathbf{I})
$$

여기서 $\alpha$은 사전확률의 정밀도(분산의 역수)를 다스린다. 최대사후확률 추정을 할 때 이는 L2 정규화에 해당한다.

### 사후확률

사후확률에는 닫힌 형태의 해가 없다.

$$
p(\mathbf{w} \mid \mathbf{X}, \mathbf{y}) = \frac{p(\mathbf{y} \mid \mathbf{X}, \mathbf{w}) \, p(\mathbf{w})}{p(\mathbf{y} \mid \mathbf{X})} \propto p(\mathbf{y} \mid \mathbf{X}, \mathbf{w}) \, p(\mathbf{w})
$$

정규 사전과 로지스틱 우도의 곱은 정규분포로 닫히지 않으며 정규화와 예측 적분도 일반적으로 해석적이지 않다. 분포의 비례식은 정확하지만 아래 가우시안 근사는 추가 근사다.

---

## 2. 라플라스 근사

**라플라스 근사**는 최빈값과 굽음을 맞추어 사후확률에 가우시안 분포를 맞춘다.

$$
p(\mathbf{w} \mid \mathcal{D}) \approx q(\mathbf{w}) = \mathcal{N}(\mathbf{w} \mid \mathbf{w}_{\text{MAP}}, \mathbf{H}^{-1})
$$

여기서 $\mathbf{w}_{\text{MAP}}$은 최대 사후확률 추정값이고 $\mathbf{H}$은 $\mathbf{w}_{\text{MAP}}$에서 음의 로그 사후확률의 헤시안 행렬이다.

### 최대 사후확률 추정값 찾기

로그 사후확률은 다음과 같다.

$$
\log p(\mathbf{w} \mid \mathcal{D}) = \sum_{i=1}^N \left[ y_i \log \sigma_i + (1-y_i) \log(1-\sigma_i) \right] - \frac{\alpha}{2} \|\mathbf{w}\|^2 + \text{const}
$$

여기서 $\sigma_i = \sigma(\mathbf{w}^\top \mathbf{x}_i)$이다.

**그래디언트:**

$$
\nabla_{\mathbf{w}} \log p(\mathbf{w} \mid \mathcal{D}) = \mathbf{X}^\top (\mathbf{y} - \boldsymbol{\sigma}) - \alpha \mathbf{w}
$$

**헤시안 행렬:**

$$
\mathbf{H} = -\nabla^2_{\mathbf{w}} \log p(\mathbf{w} \mid \mathcal{D}) = \mathbf{X}^\top \mathbf{S} \mathbf{X} + \alpha \mathbf{I}
$$

여기서 $\mathbf{S} = \text{diag}(\sigma_i(1-\sigma_i))$이다.

### 반복 재가중 최소제곱 IRLS

최대 사후확률 추정값은 뉴턴-랩슨 되풀이로 찾는다.

$$
\mathbf{w}^{(t+1)} = \mathbf{w}^{(t)} + \mathbf{H}^{-1} \nabla_{\mathbf{w}} \log p(\mathbf{w}^{(t)} \mid \mathcal{D})
$$

여기서 H는 음의 로그 사후밀도의 양의 헤시안으로 정의했으므로 로그 사후 경사에는 플러스 부호를 쓴다. 코드의 w+delta와 일치한다. α>0이면 모든 계수의 정규 벌점이 목적을 엄격히 볼록하게 만들지만 수치 안정성을 위해 스텝 감쇠와 수렴 확인이 필요할 수 있다.

### 예측 분포

새 입력 $\mathbf{x}_*$에 대해 예측 분포는 어림한 사후확률 위에서 적분한다.

$$
p(y_* = 1 \mid \mathbf{x}_*, \mathcal{D}) = \int \sigma(\mathbf{w}^\top \mathbf{x}_*) \, q(\mathbf{w}) \, d\mathbf{w}
$$

이 적분은 **프로빗 어림**으로 어림한다.

$$
p(y_* = 1 \mid \mathbf{x}_*, \mathcal{D}) \approx \sigma\left(\frac{\mu_a}{\sqrt{1 + \pi \sigma_a^2 / 8}}\right)
$$

여기서 $\mu_a = \mathbf{w}_{\text{MAP}}^\top \mathbf{x}_*$이고 $\sigma_a^2 = \mathbf{x}_*^\top \mathbf{H}^{-1} \mathbf{x}_*$이다.

---

## 3. PyTorch 구현

초기 w와 단위행렬을 CPU float64로 만들므로 입력 X·y도 호환되는 장치·자료형을 가져야 한다. α는 양수, max_iter는 양수여야 한다. 마지막 갱신 직전 계산한 H를 저장하므로 수렴하지 않았으면 최종 w에서의 헤시안과 다를 수 있다. 이 코드는 절편 열을 자동 추가하거나 최적화 성공을 별도로 보고하지 않는다.

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
            
            # 로그 뒤확률의 그래디언트
            grad = X.T @ (y - sigma) - self.alpha * w
            
            # 음의 로그 뒤확률의 헤세 행렬
            S = sigma * (1 - sigma)
            H = X.T @ (S.unsqueeze(1) * X) + self.alpha * torch.eye(D, dtype=torch.float64)
            
            # 뉴턴 스텝
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

## 4. 비교: 최대 사후확률 예측과 온전한 베이즈 예측

MAP 대입 예측은 σ(xᵀw_MAP)이고 라플라스 사후예측은 근사분포 q(w) 아래의 시그모이드 평균이다. 코드의 식은 이 평균에도 추가 근사를 적용한다. 정확한 사후분포 전체를 적분한 결과와 동일하지 않다.

이 근사식에서는 비음수 잠재 분산이 로짓 크기를 줄여 MAP보다 0.5에 가깝게 만든다. 그러나 같은 MAP를 기준으로 모든 베이즈 사후예측이 항상 덜 확신한다고 일반화할 수는 없다. 실제 확률 보정이 나아지는지도 독립적인 데이터에서 평가해야 한다.

## 5. 다른 방법과의 이음

- 가우시안 사전의 MAP는 적절한 스케일의 L2 정규화 로지스틱 회귀와 대응한다.
- 라플라스 근사는 최빈점과 국소 곡률로 가우시안을 구성한다.
- 변분 추론은 선택한 분포족에서 별도의 목적함수를 최적화한다.
- MCMC는 사후분포의 표본 근사를 제공하며 유한 표본·혼합 오차를 점검해야 한다.
- 베이즈 뉴럴 넷트웍은 비선형 함수의 웨이트 불확실성으로 이 관점을 확장한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
로지스틱 회귀에서 정규 사전의 역할과 비켤레성의 의미를 설명하라.

</div>

??? success "연습문제 1 풀이"
    정규 사전은 웨이트 크기에 이차 벌점을 주지만 사후분포를 정규로 만들지는 않는다. MAP는 최빈점 하나이고 라플라스는 그 주변을 정규로 근사한다. 사후분포 자체와 계산에 쓴 근사를 구분한다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
H를 음의 로그 사후 헤시안으로 정의하면 뉴턴 갱신의 부호는 무엇인가?

</div>

??? success "연습문제 2 풀이"
    로그 사후 경사 g를 사용할 때 w_new=w+H⁻¹g다. 음의 로그 사후 경사를 쓴다면 w_new=w−H⁻¹g_neg다. 두 경사와 헤시안의 부호 규약을 섞지 않아야 한다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
이 페이지의 예측은 두 단계의 근사를 어떻게 사용하는가?

</div>

??? success "연습문제 3 풀이"
    먼저 실제 사후분포를 MAP와 역헤시안의 정규분포로 근사한다. 그다음 정규 로짓에 대한 시그모이드 평균을 보정된 로짓 식으로 근사한다. MAP 대입, 라플라스 몬테카를로 예측과 이 추가 근사도 서로 다르다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
아래 임포트 조각을 완성하려면 어떤 조건을 확인해야 하는가?

</div>

??? success "연습문제 4 풀이"
    학습 데이터와 같은 피처·절편 구성을 사용하고 CPU float64 입력, 양의 α와 반복 수를 맞춘다. 최종 웨이트에서 헤시안을 다시 평가할지와 수렴 상태를 확인한다. 현재 import 한 줄은 학습·예측 구현이 아니다.

    ```python
    import numpy as np
    # 구현은 주제에 따라 달라진다.
    # 켤레 모형: 닫힌 꼴 뒤확률 새로 고치기.
    # 켤레가 아닌 모형: MCMC 또는 변분 추론.
    # 핵심 스텝: 앞확률 정하기, 가능도 셈하기, 뒤확률 이끌어 내기/어림하기.
    ```

## 정리하며

비켤레 로지스틱 사후분포를 라플라스로 근사하고 예측에서 웨이트 불확실성을 적분한다. 뉴턴 갱신의 부호와 입력 규약, 최종 헤시안과 수렴을 확인한다. 근사식의 확신 완화가 실제 확률 보정의 자동 보장은 아니다.

**참고 문헌**

- Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer. 4.5절.
- Murphy, K. P. (2012). *Machine Learning: A Probabilistic Perspective*. MIT Press. 8장.
- MacKay, D. J. C. (1992). The evidence framework applied to classification networks. *Neural Computation*, 4(5), 720-736.
