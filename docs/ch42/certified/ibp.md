# 구간 경계 전파(IBP)

**사이 테두리 퍼뜨리기(IBP)**(고왈 등, 2019)은 네트워크를 지나며 사이 테두리를 퍼뜨려 $\ell_\infty$ 강건성을 밝힌다. $\ell_2$ 강건성을 밝히는 아무렇게나 매끄럽게 하기와 달리, IBP은 예산 안의 어떤 $\ell_\infty$ 섭동도 추론을 바꿀 수 없음을 곧바로 밝힌다.

---

## 1. 수학 기초

### 핵심 깨침

입력 $\mathbf{x}$과 섭동 예산 $\varepsilon$이 있을 때 입력은 다음 사이에 놓인다.

$$
\mathbf{x} \in [\mathbf{x} - \varepsilon, \mathbf{x} + \varepsilon] = [\underline{\mathbf{x}}_0, \overline{\mathbf{x}}_0]
$$

IBP은 이 테두리를 네트워크의 켜마다 퍼뜨려 마지막 로짓의 테두리를 얻는다.

$$
[\underline{\mathbf{z}}_L, \overline{\mathbf{z}}_L] = \text{IBP}(f_\theta, [\underline{\mathbf{x}}_0, \overline{\mathbf{x}}_0])
$$

### 선형 켜를 지나 퍼뜨리기

입력 테두리가 $[\underline{\mathbf{x}}, \overline{\mathbf{x}}]$인 선형 켜 $\mathbf{z} = \mathbf{W}\mathbf{x} + \mathbf{b}$에서

$$
\begin{aligned}
\underline{z}_j &= \sum_i \min(W_{ji} \underline{x}_i, W_{ji} \overline{x}_i) + b_j \\
&= \sum_i [W_{ji}]_+ \underline{x}_i + [W_{ji}]_- \overline{x}_i + b_j
\end{aligned}
$$

$$
\overline{z}_j = \sum_i [W_{ji}]_+ \overline{x}_i + [W_{ji}]_- \underline{x}_i + b_j
$$

여기서 $[a]_+ = \max(a, 0)$이고 $[a]_- = \min(a, 0)$이다.

행렬 꼴로 적으면

$$
\begin{aligned}
\underline{\mathbf{z}} &= \mathbf{W}^+ \underline{\mathbf{x}} + \mathbf{W}^- \overline{\mathbf{x}} + \mathbf{b} \\
\overline{\mathbf{z}} &= \mathbf{W}^+ \overline{\mathbf{x}} + \mathbf{W}^- \underline{\mathbf{x}} + \mathbf{b}
\end{aligned}
$$

### ReLU을 지나 퍼뜨리기

테두리가 $[\underline{x}, \overline{x}]$인 ReLU 활성화 $z = \max(0, x)$에서

$$
\underline{z} = \max(0, \underline{x}), \quad \overline{z} = \max(0, \overline{x})
$$

### 밝히는 조건

참 클래스 로짓의 아래끝이 다른 클래스 모두의 위끝을 넘으면 그 추론은 든든하다고 밝혀진다.

$$
\underline{z}_y > \max_{k \neq y} \overline{z}_k \implies \text{certified robust}
$$

---

## 2. PyTorch로 짜기

```python
import torch
import torch.nn as nn
from typing import Tuple, Dict, Optional

class IBPBounds:
    """
    밝혀 낸 Linf 든든함을 위한 사이 테두리 퍼뜨리기.
    
    피드포워드 넷트웍을 지나며 사이 테두리를 퍼뜨려
    미루어 봄을 밝힌다.
    """
    
    @staticmethod
    def propagate_linear(
        layer: nn.Linear,
        lb: torch.Tensor,
        ub: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """선형 켜를 지나 테두리를 퍼뜨린다."""
        W = layer.weight
        b = layer.bias if layer.bias is not None else 0
        
        W_pos = torch.clamp(W, min=0)
        W_neg = torch.clamp(W, max=0)
        
        new_lb = lb @ W_pos.t() + ub @ W_neg.t() + b
        new_ub = ub @ W_pos.t() + lb @ W_neg.t() + b
        
        return new_lb, new_ub
    
    @staticmethod
    def propagate_relu(
        lb: torch.Tensor, ub: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """ReLU을 지나 테두리를 퍼뜨린다."""
        return torch.clamp(lb, min=0), torch.clamp(ub, min=0)
    
    @staticmethod
    def propagate_conv2d(
        layer: nn.Conv2d,
        lb: torch.Tensor,
        ub: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Conv2d을 지나 테두리를 퍼뜨린다."""
        W = layer.weight
        b = layer.bias
        
        W_pos = torch.clamp(W, min=0)
        W_neg = torch.clamp(W, max=0)
        
        new_lb = (nn.functional.conv2d(lb, W_pos, bias=b,
                    stride=layer.stride, padding=layer.padding) +
                  nn.functional.conv2d(ub, W_neg, bias=None,
                    stride=layer.stride, padding=layer.padding))
        new_ub = (nn.functional.conv2d(ub, W_pos, bias=b,
                    stride=layer.stride, padding=layer.padding) +
                  nn.functional.conv2d(lb, W_neg, bias=None,
                    stride=layer.stride, padding=layer.padding))
        
        return new_lb, new_ub

def certify_ibp(
    model: nn.Sequential,
    x: torch.Tensor,
    y: torch.Tensor,
    epsilon: float
) -> Dict[str, float]:
    """
    IBP으로 미루어 봄을 밝힌다.
    
    Parameters
    ----------
    model : nn.Sequential
        선형/Conv2d 켜와 ReLU 켜가 번갈아 오는 넷트웍
    x : torch.Tensor
        들임 배치
    y : torch.Tensor
        참 이름표
    epsilon : float
        Linf 흔듦 예산
    
    Returns
    -------
    results : certified_accuracy과 clean_accuracy을 담은 사전
    """
    ibp = IBPBounds()
    
    # 처음 테두리
    lb = torch.clamp(x - epsilon, 0, 1)
    ub = torch.clamp(x + epsilon, 0, 1)
    
    # 켜를 지나 퍼뜨린다
    for layer in model:
        if isinstance(layer, nn.Linear):
            lb, ub = ibp.propagate_linear(layer, lb, ub)
        elif isinstance(layer, nn.Conv2d):
            lb, ub = ibp.propagate_conv2d(layer, lb, ub)
        elif isinstance(layer, nn.ReLU):
            lb, ub = ibp.propagate_relu(lb, ub)
        elif isinstance(layer, nn.Flatten):
            lb = lb.flatten(1)
            ub = ub.flatten(1)
    
    # 밝혀졌는지 살핀다
    # 밝혀짐: 모든 k ≠ y에 대해 lower_bound[y] > upper_bound[k]
    true_lb = lb.gather(1, y.view(-1, 1))  # 참 클래스의 아래끝
    
    # 견주려고 참 클래스의 위끝을 -inf로 둔다
    ub_others = ub.clone()
    ub_others.scatter_(1, y.view(-1, 1), float('-inf'))
    max_other_ub = ub_others.max(dim=1)[0]
    
    certified = (true_lb.squeeze() > max_other_ub)
    
    # 맑은 맞음
    with torch.no_grad():
        clean_pred = model(x).argmax(dim=1)
        clean_correct = (clean_pred == y)
    
    return {
        'clean_accuracy': clean_correct.float().mean().item(),
        'certified_accuracy': (certified & clean_correct).float().mean().item(),
        'certification_rate': certified.float().mean().item()
    }
```

---

## 3. IBP으로 학습

인증 정확도를 올리려면 IBP에 기댄 로스로 모델을 익힐 수 있다.

$$
\mathcal{L}_{\text{IBP}} = \text{CE}(\underline{\mathbf{z}}_y - \max_{k \neq y} \overline{\mathbf{z}}_k, \mathbf{1})
$$

이는 네트워크가 참 클래스의 아래끝과 다른 클래스의 위끝 사이를 벌려 두도록 이끈다.

### 워밍업 구조

IBP 학습에는 조심스러운 구조가 있어야 한다. 여느 학습에서 비롯해 IBP 로스의 짐(그리고 $\varepsilon$)을 차츰 올려 학습이 흔들리지 않게 한다.

---

## 4. 한계

- **헐거운 테두리**: 네트워크가 깊을수록 IBP 테두리가 헐거워져 작은 네트워크에서만 밝힐 수 있다
- **작은 $\varepsilon$**: 섭동 예산이 작을 때 쓸 만하다. $\varepsilon$이 크면 테두리가 터진다
- **구조 제약**: 단순한 순방향 네트워크에 가장 잘 듣고 얽힌 구조에는 어렵다

---

## 5. 아무렇게나 매끄럽게 하기와 비교

| 피처 | IBP | 아무렇게나 매끄럽게 하기 |
|--------|-----|---------------------|
| 밝히는 노름 | $\ell_\infty$ | $\ell_2$ |
| 다짐의 클래스 | 붙박인 것 | 확률의 것 |
| 테두리의 촘촘함 | 깊은 네트워크에서 헐겁다 | 깊이와 상관없다 |
| 크게 늘리기 | 작은 네트워크 | 어떤 구조든 |
| 학습 값 | 가운데 | 낮음(잡음 불리기) |

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
선형 분류기 $f(x) = w^T x + b$에서 예측한 클래스를 바꾸는 데 드는 가장 작은 $\ell_\infty$ 섭동을 계산하여라. 이것이 뉴럴 넷트웍의 강건성과 어떻게 이어지는지 밝혀라.

</div>

??? success "풀이"
    선형 분류기에서 $\ell_\infty$ 노름으로 잰 결정 경계까지의 거리는 $\frac{|w^T x + b|}{\|w\|_1}$이다. 가장 작은 섭동은 $\delta^* = \frac{|w^T x + b|}{\|w\|_1} \cdot \text{sign}(w)$이다. 뉴럴 넷트웍에서는 그 자리의 선형 어림 $f(x + \delta) \approx f(x) + \nabla_x f \cdot \delta$이 FGSM(그래디언트의 부호를 쓴다)이 왜 잘 듣는지를 밝혀 준다. 차원이 높은 모델이 무른 까닭은 $\|w\|_1$은 차원과 함께 커지는데 $|w^T x + b|$은 꼭 그렇지 않아 강건성의 여유가 줄어들기 때문이다. $\square$

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
이 절에서 다룬 공격나 방어를 CIFAR-10의 ResNet-18 모델에 짜 넣어라. $\epsilon = 8/255$의 PGD-20 공격 아래에서 클린 정확도와 강건 정확도를 알려라.

</div>

??? success "풀이"
    여느 ResNet-18은 클린 정확도가 $\sim$93%이지만 PGD-20($\epsilon = 8/255$, 스텝 크기 $2/255$) 아래의 강건 정확도는 $\sim$0%이다. 이 절의 방법을 걸면 피처와는 재주에 따라 다르다. 적대적 학습은 클린 정확도 $\sim$83%에 강건 정확도 $\sim$50%이고, 인증 가능한 방어는 더 낮지만 증명할 수 있는 테두리를 준다. 정확도와 강건성의 트레이드오프는 기초부터 있는 것이라, 강건성을 높이면 클린 정확도가 흔히 5~15% 든다. 아무렇게나 하는 시드 3개의 평균과 표준편차로 알려라. $\square$

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
섭동 공 안에서 클래스별 데이터의 밑자리가 서로 겹친다고 볼 때, 모델이 담는 힘을 키우지 않고서는 어떤 방어도 맑은 데이터의 높은 정확도와 $\ell_\infty$ 섭동에 대한 높은 강건성을 함께 이룰 수 없음을 증명하여라.

</div>

??? success "풀이"
    두 클래스의 밑자리가 거리 $\epsilon$ 안에서 겹치면(곧 $\|x_1 - x_2\|_\infty \leq 2\epsilon$인 $x_1 \in \text{클래스 1}, x_2 \in \text{클래스 2}$이 있으면), $x_1$과 $x_2$ 둘 다에서 강건한 분류기는 적어도 하나를 틀리게 가를 수밖에 없다(섭동 공이 겹치기 때문이다). 이것이 정확도와 강건성의 기초 트레이드오프다. 겹치는 밑자리의 몫이 피할 수 없는 정확도 로스를 정한다. 여느 그림 분포에서는 $\epsilon = 8/255$에서 겹침이 꽤 있어, 살펴본 10~15%의 정확도 떨어짐을 밝혀 준다. 모델이 담는 힘을 키우면(더 너른 네트워크) 얽힌 강건한 결정 경계를 더 잘 그려 트레이드오프를 얼마쯤 눅일 수 있다. $\square$

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
금융 머신러닝 구조(속임수 알아내기나 거래 신호 만들기 따위)에서 맞섬의 강건성이 어떻게 드러나는지 다루어라. 위협 모델이 보기 다룸과 어떻게 다른가?

</div>

??? success "풀이"
    금융에서 겨루는 이는 알아내는 구조에 맞추어 스스로 움직이는 꾀 많은 무리(속임수꾼, 저자 흔드는 이)다. 보기 다룸과 다른 핵심은 이렇다. (1) 흔들 수 있는 밭이 돈으로 될 만한 것에 옭매인다(속임수꾼이 제 거래 자취를 통째로 바꿀 수는 없다). (2) 공격이 잇따르며 맞추어 간다(겨루는 이가 구조의 되받음을 보고 손본다). (3) 헛 정확도와 놓침의 값이 서로 어긋난다(옳은 거래를 막는 것과 속임수를 놓치는 것). (4) $\ell_p$ 노름은 뜻이 없고 밭에 맞는 섭동 모델이 있어야 한다. 방어는 맞추어 오는 겨루는 이에게도 든든해야 하므로, 알아내는 잣대가 알려지면 비껴갈 수 있는 알아내기 기초의 길은 많이 걸러진다. $\square$

## 정리하며

이 절에서는 수학 기초, PyTorch로 짜기, IBP으로 학습, 한계를 살펴보았다.

**살펴볼 거리**

1. Gowal, S., et al. (2019). "Scalable Verified Training for Provably Robust Image Classification." ICCV.
2. Mirman, M., Gehr, T., & Vechev, M. (2018). "Differentiable Abstract Interpretation for Provably Robust Neural Networks." ICML.
3. Wong, E., & Kolter, J. Z. (2018). "Provable Defenses Against Adversarial Examples via the Convex Outer Adversarial Polytope." ICML.
