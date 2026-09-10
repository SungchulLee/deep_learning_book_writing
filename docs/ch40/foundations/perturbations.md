# 흔듦의 갈래

흔듦을 어떻게 옭아매느냐가 겨루는 이가 움직일 "예산"을 정한다. 노름이 다르면 알아챌 수 없음의 뜻도 달라지고 치기의 꾀도 달라진다. 이 마디는 여느 흔듦 갈래와 그 꼴의 결, 그리고 옭아맨 가장 좋게 하기에 쓰는 되비추는 셈을 꼴로 적는다.

---

## 1. L-무한 노름

$$
\|\boldsymbol{\delta}\|_\infty = \max_i |\delta_i| \leq \varepsilon
$$

**풀이:** 자리마다 많아야 $\varepsilon$만큼 바뀐다. 맞섬에 든든하기 연구에서 가장 흔히 쓰는 옭아맴이다.

**결:**

- 결마다의 가장 큰 바뀜을 옭아맨다
- 그림에 잘 맞는다. 낱그림점마다의 흔듦을 마디 짓는다
- $\ell_\infty$ 공은 $\mathbb{R}^d$의 넘세모꼴 상자다
- 여느 잣대: CIFAR-10에서 $\varepsilon = 8/255 \approx 0.031$, 이미지넷에서 $\varepsilon = 4/255$

**되비추는 셈:**

$$
\Pi_\varepsilon^{\infty}(\boldsymbol{\delta})_i = \text{clip}(\delta_i, -\varepsilon, \varepsilon)
$$

셈이 아주 쉽다. 자리마다 따로 잘라 내면 된다.

**금융에서의 풀이:** 금융 모형의 결 밭 치기에서 $\ell_\infty$은 들임 결 하나가 바뀔 수 있는 가장 큰 폭을 마디 짓는다(어떤 금융 자도 $\varepsilon$을 넘게 바뀌지 않는다).

---

## 2. L2 노름

$$
\|\boldsymbol{\delta}\|_2 = \sqrt{\sum_i \delta_i^2} \leq \varepsilon
$$

**풀이:** 흔듦의 온 유클리드 크기가 마디 지어진다. 벡터의 온 길이만 작다면 자리 하나는 $\varepsilon$을 넘게 바뀔 수도 있다.

**결:**

- 적은 차수에서 더 큰 바뀜을 받아 준다
- 참 세상의 흔듦과 이어지는 신호에 잘 맞는다
- $\ell_2$ 공은 $\mathbb{R}^d$의 넘세모꼴 공이다
- 여느 잣대: CIFAR-10에서 $\varepsilon = 0.5$, 이미지넷에서 $\varepsilon = 3.0$

**되비추는 셈:**

$$
\Pi_\varepsilon^{2}(\boldsymbol{\delta}) = 
\begin{cases}
\boldsymbol{\delta} & \text{if } \|\boldsymbol{\delta}\|_2 \leq \varepsilon \\
\varepsilon \cdot \frac{\boldsymbol{\delta}}{\|\boldsymbol{\delta}\|_2} & \text{otherwise}
\end{cases}
$$

예산을 넘으면 흔듦의 잣대를 맞추어 $\varepsilon$ 공의 껍질에 놓는다.

---

## 3. L1 노름

$$
\|\boldsymbol{\delta}\|_1 = \sum_i |\delta_i| \leq \varepsilon
$$

**풀이:** 자리 모두에 걸친 온 바뀜의 크기가 마디 지어진다.

**결:**

- **성긴** 흔듦을 이끈다(몇몇 자리만 크게 바뀐다)
- $\ell_1$ 공은 엇갈린 여러모꼴(마름모 꼴)이다
- $\ell_\infty$이나 $\ell_2$보다 가장 좋게 하기 어렵다
- 볼록 눅임으로 $\ell_0$과 이어진다

**되비추는 셈(두치 등, 2008):**

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
    
    # 쌓은 합으로 문턱을 찾는다
    cumsum = torch.cumsum(sorted_u, dim=0)
    indices = torch.arange(1, len(u) + 1, device=v.device, dtype=v.dtype)
    rho = torch.where(
        sorted_u > (cumsum - radius) / indices,
        indices,
        torch.zeros_like(indices)
    ).max().long()
    
    theta = (cumsum[rho - 1] - radius) / rho
    
    # 부드러운 문턱 자르기를 건다
    return torch.sign(v) * torch.clamp(torch.abs(v) - theta, min=0)
```

---

## 4. L0 "노름"

$$
\|\boldsymbol{\delta}\|_0 = |\{i : \delta_i \neq 0\}| \leq k
$$

**풀이:** 많아야 결(낱그림점) $k$개만 고칠 수 있으나, 그 결은 얼마든지 바뀔 수 있다(흔히 옳은 들임 자리로는 잘라 낸다).

**결:**

- 아주 성기다. 자리 $k$개만 흔든다
- 참 노름이 아니다(한결같음을 어긴다)
- 골라 뽑기가 얽혀 가장 좋게 하기 어렵다(대개 NP-어려움)
- 참으로는 흔히 $\ell_1$으로 눅인다
- 낱그림점 켜의 치기를 그린다(맞서는 낱그림점 몇 개 놓기 따위)

---

## 5. 꼴로 견주기

노름마다 낱 공의 꼴이 $\mathbb{R}^d$에서 서로 다르다.

| 노름 | 낱 공의 꼴 | 성김 | 가장 좋게 하기 | 꼭짓점 |
|------|-----------------|----------|--------------|----------|
| $\ell_\infty$ | 넘세모꼴 상자 | 없음 | 쉬움(낱낱이) | $2^d$ |
| $\ell_2$ | 넘세모꼴 공 | 없음 | 쉬움(기울기) | 이어짐 |
| $\ell_1$ | 엇갈린 여러모꼴 | 가운데 | 가운데 | $2d$ |
| $\ell_0$ | 띄엄한 모임 | 아주 성김 | NP-어려움 | 골라 뽑기 |

### 노름끼리의 사이

어떤 벡터 $\boldsymbol{\delta} \in \mathbb{R}^d$에서든

$$
\|\boldsymbol{\delta}\|_\infty \leq \|\boldsymbol{\delta}\|_2 \leq \|\boldsymbol{\delta}\|_1 \leq \sqrt{d} \|\boldsymbol{\delta}\|_2 \leq d \|\boldsymbol{\delta}\|_\infty
$$

이 부등식으로 노름 사이에서 흔듦 예산을 옮길 수 있다. 다만 차수가 높으면 그 옮김이 헐거워진다.

---

## 6. PyTorch로 짜기: 하나로 모은 되비추기

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
        노름 갈래('linf', 'l2', 'l1')
    epsilon : float
        흔듦 예산
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
        흔듦을 엡실론 공으로 되비추고 옳은 자리로 잘라 낸다.
        
        Parameters
        ----------
        delta : torch.Tensor
            흔듦 텐서, 꼴 (N, ...)
        x : torch.Tensor
            본디 들임(옳은 자리로 잘라 내는 데 쓴다)
            
        Returns
        -------
        delta_proj : torch.Tensor
            되비춘 흔듦
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
            흔듦 텐서의 꼴
        device : torch.device
            텐서를 둘 장치
            
        Returns
        -------
        delta : torch.Tensor
            엡실론 공 안의 아무 흔듦
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
            # 공 안에서 고르게 잣대를 잡는다
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

## 7. 참으로 쓰는 흔듦 예산

### 여느 잣대

| 자료 꾸러미 | $\ell_\infty$ | $\ell_2$ | 까닭 |
|---------|---------------|----------|---------------|
| MNIST | $0.3$ | $2.0$ | 숫자를 알아볼 수 있다 |
| CIFAR-10 | $8/255 \approx 0.031$ | $0.5$ | 사람이 바뀜을 알아채지 못한다 |
| 이미지넷 | $4/255 \approx 0.016$ | $3.0$ | 결이 고와 예산이 더 빡빡하다 |

### 금융에 쓰기

표로 된 금융 자료에서는 흔듦 예산이 그 밭에 맞아야 한다.

| 결 갈래 | 내놓는 $\ell_\infty$ 예산 | 까닭 |
|-------------|-------------------------------|-----------|
| 잣대 맞춘 값 | 0.01~0.05 | 값을 조금 흔들기 |
| 로그 돌아옴 | 0.005~0.02 | 그럴듯한 저자 잡음 |
| 두 값 표시 | 0(붙박이) | 갈래 값은 흔들 수 없다 |
| 이어지는 견줌 | 0.01~0.1 | 결에 매인다 |

---

## 연습문제

**연습문제 1.**
선형 가름개 $f(x) = w^T x + b$에서 미루어 본 갈래를 바꾸는 데 드는 가장 작은 $\ell_\infty$ 흔듦을 셈하여라. 이것이 신경 그물의 든든함과 어떻게 이어지는지 밝혀라.

??? success "연습문제 1 풀이"
    선형 가름개에서 $\ell_\infty$ 노름으로 잰 판단의 금까지의 거리는 $\frac{|w^T x + b|}{\|w\|_1}$이다. 가장 작은 흔듦은 $\delta^* = \frac{|w^T x + b|}{\|w\|_1} \cdot \text{sign}(w)$이다. 신경 그물에서는 그 자리의 선형 어림 $f(x + \delta) \approx f(x) + \nabla_x f \cdot \delta$이 FGSM(기울기의 부호를 쓴다)이 왜 잘 듣는지를 밝혀 준다. 차수가 높은 모형이 무른 까닭은 $\|w\|_1$은 차수와 함께 커지는데 $|w^T x + b|$은 꼭 그렇지 않아 든든함의 여유가 줄어들기 때문이다. $\square$

---

**연습문제 2.**
이 마디에서 다룬 치기나 막이를 CIFAR-10의 ResNet-18 모형에 짜 넣어라. $\epsilon = 8/255$의 PGD-20 치기 아래에서 맑은 맞음과 든든한 맞음을 알려라.

??? success "연습문제 2 풀이"
    여느 ResNet-18은 맑은 맞음이 $\sim$93%이지만 PGD-20($\epsilon = 8/255$, 걸음 크기 $2/255$) 아래의 든든한 맞음은 $\sim$0%이다. 이 마디의 방법을 걸면 결과는 재주에 따라 다르다. 맞서며 익히기는 맑은 맞음 $\sim$83%에 든든한 맞음 $\sim$50%이고, 밝혀 낸 막이는 더 낮지만 증명할 수 있는 테두리를 준다. 맞음과 든든함의 맞바꿈은 밑바탕부터 있는 것이라, 든든함을 높이면 맑은 맞음이 흔히 5~15% 든다. 아무렇게나 하는 씨앗 3개의 평균과 잣대 어긋남으로 알려라. $\square$

---

**연습문제 3.**
흔듦 공 안에서 갈래별 자료의 밑자리가 서로 겹친다고 볼 때, 모형이 담는 힘을 키우지 않고서는 어떤 막이도 맑은 자료의 높은 맞음과 $\ell_\infty$ 흔듦에 대한 높은 든든함을 함께 이룰 수 없음을 증명하여라.

??? success "연습문제 3 풀이"
    두 갈래의 밑자리가 거리 $\epsilon$ 안에서 겹치면(곧 $\|x_1 - x_2\|_\infty \leq 2\epsilon$인 $x_1 \in \text{갈래 1}, x_2 \in \text{갈래 2}$이 있으면), $x_1$과 $x_2$ 둘 다에서 든든한 가름개는 적어도 하나를 틀리게 가를 수밖에 없다(흔듦 공이 겹치기 때문이다). 이것이 맞음과 든든함의 밑바탕 맞바꿈이다. 겹치는 밑자리의 몫이 피할 수 없는 맞음 잃음을 정한다. 여느 그림 분포에서는 $\epsilon = 8/255$에서 겹침이 꽤 있어, 살펴본 10~15%의 맞음 떨어짐을 밝혀 준다. 모형이 담는 힘을 키우면(더 너른 그물) 얽힌 든든한 판단의 금을 더 잘 그려 맞바꿈을 얼마쯤 눅일 수 있다. $\square$

---

**연습문제 4.**
금융 기계 배움 얼개(속임수 알아내기나 거래 신호 만들기 따위)에서 맞섬의 든든함이 어떻게 드러나는지 다루어라. 으름 얼개가 보기 다룸과 어떻게 다른가?

??? success "연습문제 4 풀이"
    금융에서 겨루는 이는 알아내는 얼개에 맞추어 스스로 움직이는 꾀 많은 무리(속임수꾼, 저자 흔드는 이)다. 보기 다룸과 다른 고갱이는 이렇다. (1) 흔들 수 있는 밭이 돈으로 될 만한 것에 옭매인다(속임수꾼이 제 거래 자취를 통째로 바꿀 수는 없다). (2) 치기가 잇따르며 맞추어 간다(겨루는 이가 얼개의 되받음을 보고 손본다). (3) 헛 맞음과 놓침의 값이 서로 어긋난다(옳은 거래를 막는 것과 속임수를 놓치는 것). (4) $\ell_p$ 노름은 뜻이 없고 밭에 맞는 흔듦 모형이 있어야 한다. 막이는 맞추어 오는 겨루는 이에게도 든든해야 하므로, 알아내는 잣대가 알려지면 비껴갈 수 있는 알아내기 바탕의 길은 많이 걸러진다. $\square$

## 정리하며

| 노름 | 옭아맴 | 잘 맞는 자리 | 되비추기의 번거로움 |
|------|-----------|----------|----------------------|
| $\ell_\infty$ | 자리마다의 가장 큼 | 그림 치기, 고른 바뀜 | $O(d)$ |
| $\ell_2$ | 유클리드 크기 | 참 세상의 흔듦 | $O(d)$ |
| $\ell_1$ | 온 바뀜의 크기 | 성긴 흔듦 | $O(d \log d)$ |
| $\ell_0$ | 바뀐 자리의 수 | 낱그림점 치기 | NP-어려움 |

노름은 으름 얼개가 말하는 알아챌 수 없음과 맞아야 한다. 그림에서는 $\ell_\infty$과 $\ell_2$이 여느 길이다. 금융 결에서는 밭에 맞는 옭아맴이 어떤 노름 하나보다 앞서는 일이 잦다.

**살펴볼 거리**

1. Goodfellow, I., Shlens, J., & Szegedy, C. (2015). "Explaining and Harnessing Adversarial Examples." ICLR.
2. Carlini, N., & Wagner, D. (2017). "Towards Evaluating the Robustness of Neural Networks." IEEE S&P.
3. Duchi, J., et al. (2008). "Efficient Projections onto the L1-Ball for Learning in High Dimensions." ICML.
4. Croce, F., & Hein, M. (2021). "Mind the Box: L1-APGD for Sparse Adversarial Attacks on Image Classifiers." ICML.
