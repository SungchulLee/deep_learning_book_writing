# 구간 경계 전파(IBP)

**구간 바운드 전파(Interval Bound Propagation, IBP)**(고왈 등, 2019)는 네트워크를 따라 구간 바운드를 전파해 $\ell_\infty$ 강건성을 인증한다. $\ell_2$ 강건성을 확률적으로 인증하는 무작위 평활화와 달리, IBP는 예산 안의 어떤 $\ell_\infty$ 섭동도 예측을 바꿀 수 없음을 결정론적으로 인증한다.

---

## 1. 수학 기초

### 핵심 통찰

입력 $\mathbf{x}$와 섭동 예산 $\varepsilon$이 있을 때 섭동된 입력은 다음 구간 안에 놓인다.

$$
\mathbf{x} \in [\mathbf{x} - \varepsilon, \mathbf{x} + \varepsilon] = [\underline{\mathbf{x}}_0, \overline{\mathbf{x}}_0]
$$

IBP는 이 바운드를 네트워크의 레이어마다 전파해 마지막 로짓의 바운드를 얻는다.

$$
[\underline{\mathbf{z}}_L, \overline{\mathbf{z}}_L] = \text{IBP}(f_\theta, [\underline{\mathbf{x}}_0, \overline{\mathbf{x}}_0])
$$

### 선형 레이어를 지나 전파

입력 바운드가 $[\underline{\mathbf{x}}, \overline{\mathbf{x}}]$인 선형 레이어 $\mathbf{z} = \mathbf{W}\mathbf{x} + \mathbf{b}$에서

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

행렬로 적으면

$$
\begin{aligned}
\underline{\mathbf{z}} &= \mathbf{W}^+ \underline{\mathbf{x}} + \mathbf{W}^- \overline{\mathbf{x}} + \mathbf{b} \\
\overline{\mathbf{z}} &= \mathbf{W}^+ \overline{\mathbf{x}} + \mathbf{W}^- \underline{\mathbf{x}} + \mathbf{b}
\end{aligned}
$$

### ReLU를 지나 전파

바운드가 $[\underline{x}, \overline{x}]$인 ReLU 활성화 $z = \max(0, x)$에서

$$
\underline{z} = \max(0, \underline{x}), \quad \overline{z} = \max(0, \overline{x})
$$

### 인증 조건

참 클래스 로짓의 아래끝이 다른 모든 클래스의 위끝보다 크면 그 예측은 강건하다고 인증된다.

$$
\underline{z}_y > \max_{k \neq y} \overline{z}_k \implies \text{certified robust}
$$

---

## 2. PyTorch로 구현하기

```python
import torch
import torch.nn as nn
from typing import Tuple, Dict, Optional

class IBPBounds:
    """
    밝혀 낸 Linf 강건성을 위한 사이 테두리 전파.
    
    피드포워드 네트워크를 지나며 사이 테두리를 퍼뜨려
    미루어 봄을 밝힌다.
    """
    
    @staticmethod
    def propagate_linear(
        layer: nn.Linear,
        lb: torch.Tensor,
        ub: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """선형 레이어를 지나 테두리를 퍼뜨린다."""
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
        선형/Conv2d 레이어와 ReLU 레이어가 번갈아 오는 네트워크
    x : torch.Tensor
        들임 배치
    y : torch.Tensor
        참 이름표
    epsilon : float
        Linf 섭동 예산
    
    Returns
    -------
    results : certified_accuracy과 clean_accuracy을 담은 사전
    """
    ibp = IBPBounds()
    
    # 처음 테두리
    lb = torch.clamp(x - epsilon, 0, 1)
    ub = torch.clamp(x + epsilon, 0, 1)
    
    # 레이어를 지나 퍼뜨린다
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
    
    # 맑은 정확도
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

## 3. IBP로 트레인

인증 정확도를 올리려면 IBP 기반 로스로 모델을 트레인할 수 있다.

$$
\mathcal{L}_{\text{IBP}} = \text{CE}(\underline{\mathbf{z}}_y - \max_{k \neq y} \overline{\mathbf{z}}_k, \mathbf{1})
$$

이는 네트워크가 참 클래스의 아래끝과 다른 클래스의 위끝 사이를 벌려 두도록 이끈다.

### 워밍업 스케줄

IBP 트레인에는 신중한 스케줄이 필요하다. 표준 트레인에서 출발해 IBP 로스의 가중치(그리고 $\varepsilon$)를 차츰 올려 트레인이 흔들리지 않게 한다.

---

## 4. 한계

- **헐거운 바운드**: 네트워크가 깊을수록 IBP 바운드가 헐거워져, 트레인 단계에서부터 IBP를 염두에 두지 않은 네트워크는 거의 인증하지 못한다
- **작은 $\varepsilon$**: 섭동 예산이 작을 때 쓸 만하다. $\varepsilon$이 크면 바운드가 지나치게 넓어진다
- **구조 제약**: 단순한 순방향 네트워크에 가장 잘 듣고 복잡한 구조에는 적용하기 어렵다

---

## 5. 무작위 평활화와 비교

| 항목 | IBP | 무작위 평활화 |
|--------|-----|---------------------|
| 인증하는 노름 | $\ell_\infty$ | $\ell_2$ |
| 보장의 종류 | 결정론적 | 확률적 |
| 바운드의 타이트함 | 깊은 네트워크에서 헐겁다 | 깊이와 상관없다 |
| 확장성 | 작은 네트워크 | 어떤 구조든 |
| 트레인 비용 | 중간 | 낮음(잡음 증강) |

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
    일반 ResNet-18은 클린 정확도가 $\sim$93%이지만 PGD-20($\epsilon = 8/255$, 스텝 크기 $2/255$) 아래의 강건 정확도는 $\sim$0%이다. 이 절의 방법을 적용한 결과는 기법에 따라 다르다. 적대적 트레인은 클린 정확도 $\sim$83%에 강건 정확도 $\sim$50%이고, 인증 가능한 방어는 더 낮지만 증명할 수 있는 바운드를 준다. 정확도와 강건성의 트레이드오프는 근본적인 것이라, 강건성을 높이면 클린 정확도가 흔히 5~15% 떨어진다. 무작위 시드 3개의 평균과 표준편차로 보고하여라. $\square$

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
섭동 공 안에서 클래스별 데이터의 서포트가 서로 겹친다고 볼 때, 모델이 용량을 키우지 않고서는 어떤 방어도 클린 데이터의 높은 정확도와 $\ell_\infty$ 섭동에 대한 높은 강건성을 함께 이룰 수 없음을 증명하여라.

</div>

??? success "풀이"
    두 클래스의 서포트가 거리 $\epsilon$ 안에서 겹치면(곧 $\|x_1 - x_2\|_\infty \leq 2\epsilon$인 $x_1 \in \text{클래스 1}, x_2 \in \text{클래스 2}$이 있으면), $x_1$과 $x_2$ 둘 다에서 강건한 분류기는 적어도 하나를 틀리게 가를 수밖에 없다(섭동 공이 겹치기 때문이다). 이것이 정확도와 강건성의 근본적인 트레이드오프다. 겹치는 서포트의 비중이 피할 수 없는 정확도 손실을 정한다. 자연 이미지 분포에서는 $\epsilon = 8/255$에서 겹침이 꽤 있어, 관찰되는 10~15%의 정확도 하락을 설명해 준다. 모델 용량을 키우면(더 넓은 네트워크) 복잡한 강건 결정 경계를 더 잘 그려 트레이드오프를 어느 정도 완화할 수 있다. $\square$

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
금융 머신러닝 시스템(사기 탐지나 거래 신호 생성 등)에서 적대적 강건성이 어떻게 드러나는지 논하여라. 위협 모델이 비전과 어떻게 다른가?

</div>

??? success "풀이"
    금융에서 적대자는 탐지 시스템에 맞추어 전략을 바꾸는 교묘한 집단(사기꾼, 시장 조작자)이다. 비전과 다른 핵심은 이렇다. (1) 섭동할 수 있는 피처가 경제적으로 그럴듯한 범위로 제약된다(사기꾼이 자기 거래 이력을 통째로 바꿀 수는 없다). (2) 공격이 순차적이고 적응적이다(적대자가 시스템의 응답을 보고 고친다). (3) 거짓 양성과 거짓 음성의 비용이 비대칭이다(정상 거래를 막는 것과 사기를 놓치는 것). (4) $\ell_p$ 노름은 의미가 없고 도메인에 맞는 섭동 모델이 있어야 한다. 방어는 적응형 적대자에게도 강건해야 하므로, 탐지 기준이 알려지면 회피할 수 있는 탐지 기반 방법은 쓸모가 크게 줄어든다. $\square$

## 정리하며

이 절에서는 IBP의 수학 기초와 PyTorch 구현, IBP로 트레인하는 법, 그리고 한계를 살펴보았다. IBP는 바운드가 헐거운 대신 계산이 순전파 두 번 값으로 싸서, 인증을 트레인 목표에 직접 넣을 수 있다.

**살펴볼 거리**

1. Gowal, S., et al. (2019). "Scalable Verified Training for Provably Robust Image Classification." ICCV.
2. Mirman, M., Gehr, T., & Vechev, M. (2018). "Differentiable Abstract Interpretation for Provably Robust Neural Networks." ICML.
3. Wong, E., & Kolter, J. Z. (2018). "Provable Defenses Against Adversarial Examples via the Convex Outer Adversarial Polytope." ICML.
