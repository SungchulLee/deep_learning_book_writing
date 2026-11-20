# 섭동 유형

섭동을 어떻게 옭아매느냐가 적대자가 움직일 "예산"을 정한다. 노름이 다르면 알아챌 수 없음의 뜻도 달라지고 공격의 전략도 달라진다. 이 절은 여느 섭동 클래스와 그 형태의 피처, 그리고 제약된 최적화에 쓰는 투영 연산을 형식화한다.

---

## 1. L-무한 노름

$$
\|\boldsymbol{\delta}\|_\infty = \max_i |\delta_i| \leq \varepsilon
$$

**풀이:** 자리마다 많아야 $\varepsilon$만큼 바뀐다. 적대적 강건성 연구에서 가장 흔히 쓰는 제약이다.

**피처:**

- 피처마다의 가장 큰 바뀜을 제약한다
- 그림에 잘 맞는다. 픽셀마다의 섭동을 제한한다
- $\ell_\infty$ 공은 $\mathbb{R}^d$의 넘세모꼴 상자다
- 여느 기준: CIFAR-10에서 $\varepsilon = 8/255 \approx 0.031$, 이미지넷에서 $\varepsilon = 4/255$

**투영 연산:**

$$
\Pi_\varepsilon^{\infty}(\boldsymbol{\delta})_i = \text{clip}(\delta_i, -\varepsilon, \varepsilon)
$$

계산이 아주 쉽다. 자리마다 따로 잘라 내면 된다.

**금융에서의 풀이:** 금융 모델의 피처 공간 공격에서 $\ell_\infty$은 입력 피처 하나가 바뀔 수 있는 가장 큰 폭을 제한한다(어떤 금융 자도 $\varepsilon$을 넘게 바뀌지 않는다).

---

## 2. L2 노름

$$
\|\boldsymbol{\delta}\|_2 = \sqrt{\sum_i \delta_i^2} \leq \varepsilon
$$

**풀이:** 섭동의 온 유클리드 크기가 제한된다. 벡터의 온 길이만 작다면 자리 하나는 $\varepsilon$을 넘게 바뀔 수도 있다.

**피처:**

- 적은 차원에서 더 큰 바뀜을 받아 준다
- 현실 세계의 섭동과 이어지는 신호에 잘 맞는다
- $\ell_2$ 공은 $\mathbb{R}^d$의 넘세모꼴 공이다
- 여느 기준: CIFAR-10에서 $\varepsilon = 0.5$, 이미지넷에서 $\varepsilon = 3.0$

**투영 연산:**

$$
\Pi_\varepsilon^{2}(\boldsymbol{\delta}) = 
\begin{cases}
\boldsymbol{\delta} & \text{if } \|\boldsymbol{\delta}\|_2 \leq \varepsilon \\
\varepsilon \cdot \frac{\boldsymbol{\delta}}{\|\boldsymbol{\delta}\|_2} & \text{otherwise}
\end{cases}
$$

예산을 넘으면 섭동의 기준을 맞추어 $\varepsilon$ 공의 껍질에 놓는다.

---

## 3. L1 노름

$$
\|\boldsymbol{\delta}\|_1 = \sum_i |\delta_i| \leq \varepsilon
$$

**풀이:** 자리 모두에 걸친 온 바뀜의 크기가 제한된다.

**피처:**

- **성긴** 섭동을 이끈다(몇몇 자리만 크게 바뀐다)
- $\ell_1$ 공은 엇갈린 여러모꼴(마름모 형태)이다
- $\ell_\infty$이나 $\ell_2$보다 최적화 어렵다
- 볼록 완화로 $\ell_0$과 이어진다

**투영 연산(두치 등, 2008):**

$\ell_1$ 공으로 되비추려면 줄 세우기에 기댄 알고리즘이 있어야 한다.

```python
def project_l1(v: torch.Tensor, radius: float) -> torch.Tensor:
    """
    벡터 v를 주어진 반지름의 L1 공으로 되비춘다.
    두치 등(2008)의 알고리즘을 쓴다.
    """
    if torch.norm(v, p=1) <= radius:
        return v
    
    # 크기를 큰 것부터 줄 세운다
    u = torch.abs(v)
    sorted_u, _ = torch.sort(u, descending=True)
    
    # 쌓은 합으로 임계를 찾는다
    cumsum = torch.cumsum(sorted_u, dim=0)
    indices = torch.arange(1, len(u) + 1, device=v.device, dtype=v.dtype)
    rho = torch.where(
        sorted_u > (cumsum - radius) / indices,
        indices,
        torch.zeros_like(indices)
    ).max().long()
    
    theta = (cumsum[rho - 1] - radius) / rho
    
    # 부드러운 임계 자르기를 건다
    return torch.sign(v) * torch.clamp(torch.abs(v) - theta, min=0)
```

---

## 4. L0 "노름"

$$
\|\boldsymbol{\delta}\|_0 = |\{i : \delta_i \neq 0\}| \leq k
$$

**풀이:** 많아야 피처(픽셀) $k$개만 고칠 수 있으나, 그 피처는 얼마든지 바뀔 수 있다(흔히 옳은 입력 자리로는 잘라 낸다).

**피처:**

- 아주 성기다. 자리 $k$개만 흔든다
- 참 노름이 아니다(한결같음을 어긴다)
- 골라 샘플링이 얽혀 최적화 어렵다(대개 NP-어려움)
- 실제로는 흔히 $\ell_1$으로 완화한다
- 픽셀 레이어의 공격을 그린다(적대적 픽셀 몇 개 놓기 따위)

---

## 5. 형태로 비교

노름마다 낱 공의 형태가 $\mathbb{R}^d$에서 서로 다르다.

| 노름 | 낱 공의 형태 | 성김 | 최적화 | 꼭짓점 |
|------|-----------------|----------|--------------|----------|
| $\ell_\infty$ | 넘세모꼴 상자 | 없음 | 쉬움(낱낱이) | $2^d$ |
| $\ell_2$ | 넘세모꼴 공 | 없음 | 쉬움(그래디언트) | 이어짐 |
| $\ell_1$ | 엇갈린 여러모꼴 | 가운데 | 가운데 | $2d$ |
| $\ell_0$ | 띄엄한 모임 | 아주 성김 | NP-어려움 | 골라 샘플링 |

### 노름끼리의 사이

어떤 벡터 $\boldsymbol{\delta} \in \mathbb{R}^d$에서든

$$
\|\boldsymbol{\delta}\|_\infty \leq \|\boldsymbol{\delta}\|_2 \leq \|\boldsymbol{\delta}\|_1 \leq \sqrt{d} \|\boldsymbol{\delta}\|_2 \leq d \|\boldsymbol{\delta}\|_\infty
$$

이 부등식으로 노름 사이에서 섭동 예산을 옮길 수 있다. 다만 차원이 높으면 그 옮김이 헐거워진다.

---

## 6. PyTorch로 구현하기: 하나로 모은 투영

```python
import torch
import torch.nn as nn
from typing import Literal

class PerturbationProjector:
    """
    Lp 엡실론 공으로 하나로 모아 되비추기.
    
    옳은 자리로 잘라 내며 Linf, L2, L1 되비추기를 다룬다.
    
    Parameters
    ----------
    norm : str
        노름 클래스('linf', 'l2', 'l1')
    epsilon : float
        섭동 예산
    clip_min : float
        옳은 들임의 가장 작은 값
    clip_max : float
        옳은 들임의 가장 큰 값
    """
    
    def __init__(
        self,
        norm: Literal['linf', 'l2', 'l1'] = 'linf',
        epsilon: float = 8/255,
        clip_min: float = 0.0,
        clip_max: float = 1.0
    ):
        self.norm = norm
        self.epsilon = epsilon
        self.clip_min = clip_min
        self.clip_max = clip_max
    
    def project(
        self,
        delta: torch.Tensor,
        x: torch.Tensor
    ) -> torch.Tensor:
        """
        섭동을 엡실론 공으로 되비추고 옳은 자리로 잘라 낸다.
        
        Parameters
        ----------
        delta : torch.Tensor
            섭동 텐서, 형태 (N, ...)
        x : torch.Tensor
            본디 들임(옳은 자리로 잘라 내는 데 쓴다)
            
        Returns
        -------
        delta_proj : torch.Tensor
            되비춘 섭동
        """
        if self.norm == 'linf':
            delta = torch.clamp(delta, -self.epsilon, self.epsilon)
        
        elif self.norm == 'l2':
            batch_size = delta.shape[0]
            delta_flat = delta.view(batch_size, -1)
            norms = torch.norm(delta_flat, p=2, dim=1, keepdim=True)
            factor = torch.clamp(norms / self.epsilon, min=1.0)
            delta_flat = delta_flat / factor
            delta = delta_flat.view(delta.shape)
        
        elif self.norm == 'l1':
            batch_size = delta.shape[0]
            delta_flat = delta.view(batch_size, -1)
            for i in range(batch_size):
                if torch.norm(delta_flat[i], p=1) > self.epsilon:
                    delta_flat[i] = self._project_l1_single(
                        delta_flat[i], self.epsilon
                    )
            delta = delta_flat.view(delta.shape)
        
        # x + delta이 옳은 자리에 있도록 잘라 낸다
        delta = torch.clamp(x + delta, self.clip_min, self.clip_max) - x
        return delta
    
    @staticmethod
    def _project_l1_single(v: torch.Tensor, radius: float) -> torch.Tensor:
        """벡터 하나를 L1 공으로 되비춘다."""
        u = torch.abs(v)
        sorted_u, _ = torch.sort(u, descending=True)
        cumsum = torch.cumsum(sorted_u, dim=0)
        indices = torch.arange(1, len(u) + 1, device=v.device, dtype=v.dtype)
        rho = torch.where(
            sorted_u > (cumsum - radius) / indices,
            indices,
            torch.zeros_like(indices)
        ).max().long()
        theta = (cumsum[rho - 1] - radius) / rho
        return torch.sign(v) * torch.clamp(torch.abs(v) - theta, min=0)
    
    def random_init(self, shape: tuple, device: torch.device) -> torch.Tensor:
        """
        엡실론 공 안에서 첫자리를 아무렇게나 잡는다.
        
        Parameters
        ----------
        shape : tuple
            섭동 텐서의 형태
        device : torch.device
            텐서를 둘 장치
            
        Returns
        -------
        delta : torch.Tensor
            엡실론 공 안의 아무 섭동
        """
        if self.norm == 'linf':
            return torch.empty(shape, device=device).uniform_(
                -self.epsilon, self.epsilon
            )
        elif self.norm == 'l2':
            delta = torch.randn(shape, device=device)
            delta_flat = delta.view(shape[0], -1)
            norms = delta_flat.norm(p=2, dim=1, keepdim=True)
            delta_flat = delta_flat / norms  # 낱 공
            # 공 안에서 고르게 스케일을 맞춘다
            r = torch.rand(shape[0], 1, device=device) ** (1.0 / delta_flat.shape[1])
            delta_flat = delta_flat * r * self.epsilon
            return delta_flat.view(shape)
        elif self.norm == 'l1':
            # 어림: 고르게 뽑아 되비춘다
            delta = torch.empty(shape, device=device).uniform_(-1, 1)
            delta_flat = delta.view(shape[0], -1)
            for i in range(shape[0]):
                delta_flat[i] = self._project_l1_single(
                    delta_flat[i], self.epsilon
                )
            return delta_flat.view(shape)
```

---

## 7. 실제로 쓰는 섭동 예산

### 여느 기준

| 데이터 꾸러미 | $\ell_\infty$ | $\ell_2$ | 까닭 |
|---------|---------------|----------|---------------|
| MNIST | $0.3$ | $2.0$ | 숫자를 알아볼 수 있다 |
| CIFAR-10 | $8/255 \approx 0.031$ | $0.5$ | 사람이 바뀜을 알아채지 못한다 |
| 이미지넷 | $4/255 \approx 0.016$ | $3.0$ | 피처가 고와 예산이 더 빡빡하다 |

### 금융 응용

표로 된 금융 데이터에서는 섭동 예산이 그 분야에 맞아야 한다.

| 피처 클래스 | 내놓는 $\ell_\infty$ 예산 | 까닭 |
|-------------|-------------------------------|-----------|
| 기준 맞춘 값 | 0.01~0.05 | 값을 조금 흔들기 |
| 로그 돌아옴 | 0.005~0.02 | 그럴듯한 저자 잡음 |
| 두 값 표시 | 0(고정) | 클래스 값은 흔들 수 없다 |
| 이어지는 비교 | 0.01~0.1 | 피처에 매인다 |

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
선형 분류기 $f(x) = w^T x + b$에서 예측한 클래스를 바꾸는 데 드는 가장 작은 $\ell_\infty$ 섭동을 계산하여라. 이것이 뉴럴 네트워크의 강건성과 어떻게 이어지는지 설명하여라.

</div>

??? success "풀이"
    선형 분류기에서 $\ell_\infty$ 노름으로 잰 결정 경계까지의 거리는 $\frac{|w^T x + b|}{\|w\|_1}$이다. 가장 작은 섭동은 $\delta^* = \frac{|w^T x + b|}{\|w\|_1} \cdot \text{sign}(w)$이다. 뉴럴 네트워크에서는 국소 선형 근사 $f(x + \delta) \approx f(x) + \nabla_x f \cdot \delta$이 FGSM(그래디언트의 부호를 쓴다)이 왜 잘 통하는지를 설명해 준다. 차원이 높은 모델이 취약한 까닭은 $\|w\|_1$은 차원과 함께 커지는데 $|w^T x + b|$은 꼭 그렇지 않아 강건성의 여유가 줄어들기 때문이다. $\square$

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
이 절에서 다룬 공격이나 방어를 CIFAR-10의 ResNet-18 모델에 구현하여라. $\epsilon = 8/255$의 PGD-20 공격 아래에서 클린 정확도와 강건 정확도를 보고하여라.

</div>

??? success "풀이"
    일반 ResNet-18은 클린 정확도가 $\sim$93%이지만 PGD-20($\epsilon = 8/255$, 스텝 크기 $2/255$) 아래의 강건 정확도는 $\sim$0%이다. 이 절의 방법을 적용한 결과는 기법에 따라 다르다. 적대적 학습은 클린 정확도 $\sim$83%에 강건 정확도 $\sim$50%이고, 인증 가능한 방어는 더 낮지만 증명할 수 있는 바운드를 준다. 정확도와 강건성의 트레이드오프는 근본적인 것이라, 강건성을 높이면 클린 정확도가 흔히 5~15% 떨어진다. 무작위 시드 3개의 평균과 표준편차로 보고하여라. $\square$

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
섭동 공 안에서 클래스별 데이터의 서포트가 서로 겹친다고 볼 때, 모델이 용량을 키우지 않고서는 어떤 방어도 클린 데이터의 높은 정확도와 $\ell_\infty$ 섭동에 대한 높은 강건성을 함께 이룰 수 없음을 증명하여라.

</div>

??? success "풀이"
    두 클래스의 서포트가 거리 $\epsilon$ 안에서 겹치면(곧 $\|x_1 - x_2\|_\infty \leq 2\epsilon$인 $x_1 \in \text{클래스 1}, x_2 \in \text{클래스 2}$이 있으면), $x_1$과 $x_2$ 둘 다에서 강건한 분류기는 적어도 하나를 틀리게 가를 수밖에 없다(섭동 공이 겹치기 때문이다). 이것이 정확도와 강건성의 기초 트레이드오프다. 겹치는 서포트의 비중이 피할 수 없는 정확도 손실을 정한다. 자연 이미지 분포에서는 $\epsilon = 8/255$에서 겹침이 꽤 있어, 관찰되는 10~15%의 정확도 하락을 설명해 준다. 모델 용량을 키우면(더 넓은 네트워크) 복잡한 강건 결정 경계를 더 잘 그려 트레이드오프를 어느 정도 완화할 수 있다. $\square$

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
금융 머신러닝 시스템(사기 탐지나 거래 신호 생성 등)에서 적대적 강건성이 어떻게 드러나는지 논하여라. 위협 모델이 비전과 어떻게 다른가?

</div>

??? success "풀이"
    금융에서 적대자는 탐지 시스템에 맞추어 전략을 바꾸는 교묘한 집단(사기꾼, 시장 조작자)이다. 비전과 다른 핵심은 이렇다. (1) 섭동할 수 있는 피처가 경제적으로 그럴듯한 범위로 제약된다(사기꾼이 자기 거래 이력을 통째로 바꿀 수는 없다). (2) 공격이 순차적이고 적응적이다(적대자가 시스템의 응답을 보고 고친다). (3) 거짓 양성과 거짓 음성의 비용이 비대칭이다(정상 거래를 막는 것과 사기를 놓치는 것). (4) $\ell_p$ 노름은 의미가 없고 도메인에 맞는 섭동 모델이 있어야 한다. 방어는 적응형 적대자에게도 강건해야 하므로, 탐지 기준이 알려지면 회피할 수 있는 탐지 기반 방법은 쓸모가 크게 줄어든다. $\square$

## 정리하며

| 노름 | 제약 | 잘 맞는 자리 | 투영의 계산 비용 |
|------|-----------|----------|----------------------|
| $\ell_\infty$ | 자리마다의 가장 큼 | 그림 공격, 고른 바뀜 | $O(d)$ |
| $\ell_2$ | 유클리드 크기 | 현실 세계의 섭동 | $O(d)$ |
| $\ell_1$ | 온 바뀜의 크기 | 성긴 섭동 | $O(d \log d)$ |
| $\ell_0$ | 바뀐 자리의 수 | 픽셀 공격 | NP-어려움 |

노름은 위협 모델이 말하는 알아챌 수 없음과 맞아야 한다. 그림에서는 $\ell_\infty$과 $\ell_2$이 여느 길이다. 금융 피처에서는 분야에 맞는 제약이 어떤 노름 하나보다 앞서는 일이 잦다.

**살펴볼 거리**

1. Goodfellow, I., Shlens, J., & Szegedy, C. (2015). "Explaining and Harnessing Adversarial Examples." ICLR.
2. Carlini, N., & Wagner, D. (2017). "Towards Evaluating the Robustness of Neural Networks." IEEE S&P.
3. Duchi, J., et al. (2008). "Efficient Projections onto the L1-Ball for Learning in High Dimensions." ICML.
4. Croce, F., & Hein, M. (2021). "Mind the Box: L1-APGD for Sparse Adversarial Attacks on Image Classifiers." ICML.
