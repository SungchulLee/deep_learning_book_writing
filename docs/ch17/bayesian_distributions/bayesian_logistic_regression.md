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

<div class="storybox" markdown>

**그림으로 보는 AI 1.** 켤레가 없어서 1774년의 요령을 꺼내 쓴다

이 쪽의 머리말이 "정규 사전은 로지스틱 우도에 켤레가 아니므로"라고 담담히 적고 넘어가지만, 그 한 줄에서 이 장의 주된 장치가 멈춘다. [켤레 사전분포](../bayesian_foundations/conjugate_priors.md) 쪽의 표에 로지스틱이 없는 것은 빠뜨려서가 아니라 **없기 때문**이다. 더하고 세는 것으로 끝나던 갱신이 여기서는 통하지 않는다.

그래서 꺼내 쓰는 것이 **라플라스의 요령**이다. 피에르시몽 라플라스가 1774년에 쓴 방법으로, 적분값의 거의 전부가 피적분함수가 가장 큰 자리 **근처에서** 나온다는 관찰에 기댄다. 그 자리를 찾고 거기서의 굽음만 재어 가우시안으로 바꿔치기하는 것이다. 베이즈 정리의 일반적인 꼴을 적은 사람과 같은 사람이고 ([사전분포·우도·사후분포](../bayesian_foundations/prior_likelihood_posterior.md) 쪽의 상자를 보라), 그 사람이 남긴 다른 연장을 두 세기 뒤에 이렇게 쓰고 있다.

잘 듣는지 재어 보면 **자료에 따라 갈린다.**

![두 판으로 된 그림. 왼쪽은 잘 섞인 200점 자료의 계수 사후분포로, 정확한 사후분포 곡선과 라플라스 가우시안이 거의 포개져 있고 95퍼센트 구간이 각각 1.11에서 2.03, 1.07에서 1.98 로 거의 같으며 라플라스 구간이 실제로 담는 확률은 0.946 이다. 오른쪽은 완전히 분리되는 8점 자료로, 정확한 사후분포는 오른쪽으로 긴 꼬리를 끌며 0 근처에 거의 밀도가 없는데 라플라스 가우시안은 왼쪽으로 퍼져 0 을 넘어 음수까지 간다. 95퍼센트 구간이 정확한 쪽은 1.00에서 7.54, 라플라스는 마이너스 0.21에서 5.89 이고 라플라스 구간이 담는 확률은 0.897 이다.](figures/laplace_limits.svg)

**왼쪽은 흠잡을 데가 없다.** 자료가 200점이고 잘 섞여 있으면 사후분포가 거의 가우시안이라 두 곡선이 포개지고, 명목 95% 구간이 실제로 0.946 을 담는다. 이 쪽의 코드가 내놓는 수를 믿어도 되는 자리다.

**오른쪽이 문제다.** 여덟 점이 완전히 분리되면 — 작은 자료에서 흔한 일이다 — 사후분포가 오른쪽으로 길게 쏠린다. 우도만으로는 계수가 무한대로 가려 하고 사전분포가 겨우 붙들고 있는 상태다. 최빈값 하나와 굽음 하나로는 그 쏠림이 보이지 않으므로, 가우시안이 **왼쪽으로 새어 0 을 넘어간다.**

결과가 결론을 바꾼다. 라플라스의 95% 구간은 $[-0.21,\ 5.89]$ 라서 **"이 설명변수는 효과가 없을 수도, 반대 방향일 수도 있다"** 고 말한다. 정확한 구간은 $[1.00,\ 7.54]$ 로 **효과가 분명히 양수이고 1 보다 크다**고 말한다. 같은 자료에 같은 사전분포인데 한쪽은 부호를 확신하지 못하고 다른 쪽은 확신한다. 그 구간이 실제로 담는 사후확률도 0.897 이라 95% 라는 이름값을 못 한다.

**그러니 위 프로빗 어림의 수를 읽을 때 확인할 것은 자료의 분리 여부다.** 4절이 MAP 대입 예측과 라플라스 사후예측과 "정확한 사후분포 전체를 적분한 결과"를 셋으로 갈라 두는 까닭이 여기에 있다 — 셋이 가까운 자리와 갈라지는 자리가 따로 있고, 갈라지는 쪽이 하필 자료가 적을 때다.

_확인한 곳 — 라플라스의 방법은 1774년 글에서 처음 제시되었고, 적분의 주된 기여가 지수의 극값 근처에서 온다는 관찰에 기반한다. 베이즈 추론에서는 사후 정규화 상수를 어림하거나 사후분포를 최빈값 중심의 가우시안으로 바꾸는 데 쓴다. 그림의 수는 그림을 그리는 코드가 직접 셈하며, 정확한 사후분포는 촘촘한 격자에서 수치적분으로 구한다. 자료가 넉넉하면 라플라스가 잘 맞고 분리되면 구간이 0 을 넘는다는 것을 스스로 확인한 뒤에만 그린다._

</div>

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

??? success "풀이"
    정규 사전은 웨이트 크기에 이차 벌점을 주지만 사후분포를 정규로 만들지는 않는다. MAP는 최빈점 하나이고 라플라스는 그 주변을 정규로 근사한다. 사후분포 자체와 계산에 쓴 근사를 구분한다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
H를 음의 로그 사후 헤시안으로 정의하면 뉴턴 갱신의 부호는 무엇인가?

</div>

??? success "풀이"
    로그 사후 경사 g를 사용할 때 w_new=w+H⁻¹g다. 음의 로그 사후 경사를 쓴다면 w_new=w−H⁻¹g_neg다. 두 경사와 헤시안의 부호 규약을 섞지 않아야 한다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
이 페이지의 예측은 두 단계의 근사를 어떻게 사용하는가?

</div>

??? success "풀이"
    먼저 실제 사후분포를 MAP와 역헤시안의 정규분포로 근사한다. 그다음 정규 로짓에 대한 시그모이드 평균을 보정된 로짓 식으로 근사한다. MAP 대입, 라플라스 몬테카를로 예측과 이 추가 근사도 서로 다르다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
아래 임포트 조각을 완성하려면 어떤 조건을 확인해야 하는가?

</div>

??? success "풀이"
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
