# 전이 기반 공격

**전이되는 공격**은 적대적 샘플의 기초 특징을 쓴다. 한 모델(**대신 쓰는 모델**)을 겨냥해 지은 섭동이 다른 모델(**타깃**)까지 속이는 일이 잦다는 것이다. 이로써 **물음이 없는** 블랙박스 공격이 이루어진다. 겨루는 이는 타깃 모델과 곧바로 마주할 일이 없다.

---

## 1. 전이이라는 일

### 적대적 샘플은 왜 전이되는이

모델을 넘나드는 전이는 몇 가지로 밝혀진다.

1. **나눠 가진 특징**: 비슷한 데이터로 학습된 모델은 비슷한 특징을 배우므로, 그 특징을 흔드는 적대적 섭동은 모델을 넘나들며 옮아간다
2. **비슷한 결정 경계**: 구조와 학습 절차가 비슷한 모델은 비슷한 결정 경계를 지닌다
3. **강건하지 않은 특징은 두루 있다**: 적대적 공격이 쓰는 강건하지 않은 특징은 모델 무리를 넘어 나눠 가진다

### 꼴로 적기

온전히 들여다볼 수 있는 대신 쓰는 모델 $f_s$과 볼 수 없는 타깃 모델 $f_t$이 있을 때

$$
\boldsymbol{\delta}^* = \text{WhiteBoxAttack}(f_s, \mathbf{x}, y) \implies f_t(\mathbf{x} + \boldsymbol{\delta}^*) \neq y \text{ (with non-trivial probability)}
$$

전이되는 비율은 대신 쓰는 모델과 타깃 모델의 사이에 따라 크게 달라진다.

---

## 2. 전이를 낫게 하기

### 밀어 나감 되돌이 FGSM(MI-FGSM)

동 등(2018)은 되돌이 공격에 **밀어 나감**을 더하면 전이가 크게 나아짐을 보였다. 여느 PGD은 대신 쓰는 모델의 손실 터에 지나치게 맞춰질 수 있는데, 밀어 나감이 공격의 방향을 든든하게 해 준다.

MI-FGSM의 고치는 규칙은

$$
\begin{aligned}
\mathbf{g}^{(t)} &= \mu \cdot \mathbf{g}^{(t-1)} + \frac{\nabla_\mathbf{x} \mathcal{L}(f_s(\mathbf{x}^{(t)}), y)}{\|\nabla_\mathbf{x} \mathcal{L}(f_s(\mathbf{x}^{(t)}), y)\|_1} \\
\mathbf{x}^{(t+1)} &= \Pi_\varepsilon\left(\mathbf{x}^{(t)} + \alpha \cdot \text{sign}(\mathbf{g}^{(t)})\right)
\end{aligned}
$$

여기서 $\mu$은 밀어 나감이 줄어드는 값이다(흔히 1.0).

### 앙상블 공격

대신 쓰는 모델을 앙상블로 치면 전이되는 비율이 나아진다.

$$
\mathcal{L}_{\text{ensemble}} = \sum_{k=1}^K w_k \cdot \mathcal{L}(f_k(\mathbf{x} + \boldsymbol{\delta}), y)
$$

여기서 $w_k$은 앙상블의 짐이다. 대신 쓰는 모델 여럿을 속이는 섭동은 본 적 없는 타깃에도 옮아갈 낌새가 크다.

### 입력 다양하게 하기(DI-FGSM)

셰 등(2019)은 공격을 만드는 동안 입력에 아무 바꿈을 걸자고 내놓았다.

$$
\nabla_\mathbf{x} \mathcal{L}(f_s(T(\mathbf{x}^{(t)})), y)
$$

여기서 $T$은 아무렇게나 크기를 바꾸고 덧대는 일을 한다. 이는 공격이 남다른 입력 특징에 지나치게 맞춰지는 것을 막는다.

### 옮겨도 그대로인 공격(TI-FGSM)

기울기를 알갱이와 엮으면 섭동이 옮겨도 그대로가 된다.

$$
\mathbf{g}^{(t)} = W * \nabla_\mathbf{x} \mathcal{L}(f_s(\mathbf{x}^{(t)}), y)
$$

여기서 $W$은 흔히 가우스 알갱이다.

---

## 3. PyTorch로 짜기

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict

class TransferAttack:
    """
    밀어 나감과 모둠을 쓰는 옮아감 바탕의 검은 상자 치기.
    
    대신 쓰는 모형을 겨냥해 맞서는 보기를 만들어
    모르는 과녁 모형으로 옮아가게 한다.
    
    Parameters
    ----------
    surrogate_models : list[nn.Module]
        대신 쓰는 모형 하나 이상
    epsilon : float
        흔듦 예산
    alpha : float
        걸음 크기
    num_iter : int
        치기 되돌이 횟수
    momentum : float
        밀어 나감이 줄어드는 값(0 = 밀어 나감 없음)
    input_diversity : bool
        들임 다양하게 하기를 쓸지
    """
    
    def __init__(
        self,
        surrogate_models: List[nn.Module],
        epsilon: float = 8/255,
        alpha: Optional[float] = None,
        num_iter: int = 20,
        momentum: float = 1.0,
        input_diversity: bool = True,
        device: Optional[torch.device] = None
    ):
        self.surrogates = surrogate_models
        self.epsilon = epsilon
        self.alpha = alpha if alpha else epsilon / num_iter * 2
        self.num_iter = num_iter
        self.momentum = momentum
        self.input_diversity = input_diversity
        self.device = device or next(surrogate_models[0].parameters()).device
        
        for model in self.surrogates:
            model.eval()
            model.to(self.device)
    
    def _input_diversity_transform(
        self, x: torch.Tensor, p: float = 0.5
    ) -> torch.Tensor:
        """들임을 다양하게 하려고 아무렇게나 크기를 바꾸고 덧댄다."""
        if not self.input_diversity or torch.rand(1).item() > p:
            return x
        
        img_size = x.shape[-1]
        rnd = torch.randint(img_size, img_size + 8, (1,)).item()
        
        x_resized = F.interpolate(
            x, size=(rnd, rnd), mode='bilinear', align_corners=False
        )
        
        pad_top = torch.randint(0, rnd - img_size + 1, (1,)).item()
        pad_bottom = rnd - img_size - pad_top
        pad_left = torch.randint(0, rnd - img_size + 1, (1,)).item()
        pad_right = rnd - img_size - pad_left
        
        x_padded = F.pad(x_resized, (pad_left, pad_right, pad_top, pad_bottom))
        x_padded = F.interpolate(
            x_padded, size=(img_size, img_size),
            mode='bilinear', align_corners=False
        )
        
        return x_padded
    
    def generate(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        targeted: bool = False,
        target_labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """옮아가는 맞서는 보기를 만든다."""
        x = x.to(self.device)
        y = y.to(self.device)
        
        x_adv = x.clone()
        g_momentum = torch.zeros_like(x)
        
        for t in range(self.num_iter):
            x_adv.requires_grad_(True)
            
            # 대신 쓰는 모형에 걸친 모둠 잃음
            total_loss = 0
            for model in self.surrogates:
                x_input = self._input_diversity_transform(x_adv)
                logits = model(x_input)
                
                if targeted:
                    loss = -F.cross_entropy(logits, target_labels)
                else:
                    loss = F.cross_entropy(logits, y)
                
                total_loss += loss / len(self.surrogates)
            
            # 기울기를 셈한다
            for model in self.surrogates:
                model.zero_grad()
            total_loss.backward()
            grad = x_adv.grad.data
            
            # 기울기의 잣대를 맞춘다(L1으로)
            grad_norm = grad / (grad.abs().mean(dim=[1, 2, 3], keepdim=True) + 1e-8)
            
            # 밀어 나감을 건다
            g_momentum = self.momentum * g_momentum + grad_norm
            
            # 고친다
            x_adv = x_adv.detach() + self.alpha * g_momentum.sign()
            
            # 되비춘다
            delta = torch.clamp(x_adv - x, -self.epsilon, self.epsilon)
            x_adv = torch.clamp(x + delta, 0, 1)
        
        return x_adv.detach()
```

---

## 4. 전이되는 비율 살피기

### 전이에 걸리는 것

| 걸리는 것 | 전이되는 비율에 미침 | 풀이 |
|--------|------------------------|-------------|
| 구조가 닮음 | 닮을수록 → 더 잘 전이 | 닮은 구조는 닮은 특징을 배운다 |
| 학습 데이터가 겹침 | 많이 겹칠수록 → 더 잘 전이 | 나눠 쓴 데이터가 나눠 가진 특징을 낳는다 |
| 밀어 나감 | 크게 나아짐 | 대신 쓰는 모델에 지나치게 맞춰지는 것을 비껴간다 |
| 앙상블 대신 모델 | 꽤 나아짐 | 모델을 넘나드는 특징이 더 두루 쓰인다 |
| 입력 다양하게 하기 | 웬만큼 나아짐 | 입력에 매인 지나친 맞춤을 줄인다 |
| 되돌이 | 보람이 줄어듦 | 되돌이가 많으면 대신 쓰는 모델에 지나치게 맞춰진다 |

### 흔한 전이되는 비율(CIFAR-10, epsilon = 8/255)

| 대신 모델 → 타깃 | FGSM | MI-FGSM-20 | 앙상블 MI-FGSM |
|---------------------|------|-----------|-------------------|
| ResNet-18 → VGG-16 | 약 30% | 약 50% | 약 65% |
| ResNet-18 → DenseNet | 약 35% | 약 55% | 약 70% |
| ResNet-18 → ResNet-50 | 약 45% | 약 65% | 약 75% |

---

## 5. 금융 응용

전이되는 공격은 다음과 같은 금융 자리에서 더욱 걸린다.

- 겨루는 이가 쓰인 모델의 갈래는 알아도(미쁨 점수에 기울기 북돋우기 따위) 정확한 매개변수는 모를 때
- 비슷한 데이터로 학습된 열린 모델을 대신 쓸 수 있을 때
- 규정이 모델 갈래를 못 박아 잿빛 상자의 앎이 생길 때

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
선형 분류기 $f(x) = w^T x + b$에서 예측한 갈래를 바꾸는 데 드는 가장 작은 $\ell_\infty$ 섭동을 계산하여라. 이것이 신경망의 강건성과 어떻게 이어지는지 밝혀라.

</div>

??? success "연습문제 1 풀이"
    선형 분류기에서 $\ell_\infty$ 노름으로 잰 결정 경계까지의 거리는 $\frac{|w^T x + b|}{\|w\|_1}$이다. 가장 작은 섭동은 $\delta^* = \frac{|w^T x + b|}{\|w\|_1} \cdot \text{sign}(w)$이다. 신경망에서는 그 자리의 선형 어림 $f(x + \delta) \approx f(x) + \nabla_x f \cdot \delta$이 FGSM(기울기의 부호를 쓴다)이 왜 잘 듣는지를 밝혀 준다. 차원이 높은 모델이 무른 까닭은 $\|w\|_1$은 차원과 함께 커지는데 $|w^T x + b|$은 꼭 그렇지 않아 강건성의 여유가 줄어들기 때문이다. $\square$

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
이 마디에서 다룬 공격나 방어를 CIFAR-10의 ResNet-18 모델에 짜 넣어라. $\epsilon = 8/255$의 PGD-20 공격 아래에서 클린 정확도와 강건 정확도를 알려라.

</div>

??? success "연습문제 2 풀이"
    여느 ResNet-18은 클린 정확도가 $\sim$93%이지만 PGD-20($\epsilon = 8/255$, 걸음 크기 $2/255$) 아래의 강건 정확도는 $\sim$0%이다. 이 마디의 방법을 걸면 특징과는 재주에 따라 다르다. 적대적 학습은 클린 정확도 $\sim$83%에 강건 정확도 $\sim$50%이고, 인증 가능한 방어는 더 낮지만 증명할 수 있는 테두리를 준다. 정확도와 강건성의 트레이드오프는 기초부터 있는 것이라, 강건성을 높이면 클린 정확도가 흔히 5~15% 든다. 아무렇게나 하는 시드 3개의 평균과 표준편차로 알려라. $\square$

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
섭동 공 안에서 갈래별 데이터의 밑자리가 서로 겹친다고 볼 때, 모델이 담는 힘을 키우지 않고서는 어떤 방어도 맑은 데이터의 높은 정확도와 $\ell_\infty$ 섭동에 대한 높은 강건성을 함께 이룰 수 없음을 증명하여라.

</div>

??? success "연습문제 3 풀이"
    두 갈래의 밑자리가 거리 $\epsilon$ 안에서 겹치면(곧 $\|x_1 - x_2\|_\infty \leq 2\epsilon$인 $x_1 \in \text{갈래 1}, x_2 \in \text{갈래 2}$이 있으면), $x_1$과 $x_2$ 둘 다에서 강건한 분류기는 적어도 하나를 틀리게 가를 수밖에 없다(섭동 공이 겹치기 때문이다). 이것이 정확도와 강건성의 기초 트레이드오프다. 겹치는 밑자리의 몫이 피할 수 없는 정확도 손실을 정한다. 여느 그림 분포에서는 $\epsilon = 8/255$에서 겹침이 꽤 있어, 살펴본 10~15%의 정확도 떨어짐을 밝혀 준다. 모델이 담는 힘을 키우면(더 너른 네트워크) 얽힌 강건한 결정 경계를 더 잘 그려 트레이드오프를 얼마쯤 눅일 수 있다. $\square$

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
금융 머신러닝 구조(속임수 알아내기나 거래 신호 만들기 따위)에서 맞섬의 강건성이 어떻게 드러나는지 다루어라. 위협 모델이 보기 다룸과 어떻게 다른가?

</div>

??? success "연습문제 4 풀이"
    금융에서 겨루는 이는 알아내는 구조에 맞추어 스스로 움직이는 꾀 많은 무리(속임수꾼, 저자 흔드는 이)다. 보기 다룸과 다른 핵심은 이렇다. (1) 흔들 수 있는 밭이 돈으로 될 만한 것에 옭매인다(속임수꾼이 제 거래 자취를 통째로 바꿀 수는 없다). (2) 공격이 잇따르며 맞추어 간다(겨루는 이가 구조의 되받음을 보고 손본다). (3) 헛 정확도와 놓침의 값이 서로 어긋난다(옳은 거래를 막는 것과 속임수를 놓치는 것). (4) $\ell_p$ 노름은 뜻이 없고 밭에 맞는 섭동 모델이 있어야 한다. 방어는 맞추어 오는 겨루는 이에게도 든든해야 하므로, 알아내는 잣대가 알려지면 비껴갈 수 있는 알아내기 기초의 길은 많이 걸러진다. $\square$

## 정리하며

이 마당은 전이이라는 일, 전이를 낫게 하기, PyTorch로 짜기, 전이되는 비율 살피기를 차례로 짚었다.

**살펴볼 거리**

1. Dong, Y., et al. (2018). "Boosting Adversarial Attacks with Momentum." CVPR.
2. Xie, C., et al. (2019). "Improving Transferability of Adversarial Examples with Input Diversity." CVPR.
3. Tramèr, F., et al. (2018). "Ensemble Adversarial Training: Attacks and Defenses." ICLR.
4. Papernot, N., McDaniel, P., & Goodfellow, I. (2016). "Transferability in Machine Learning: from Phenomena to Black-Box Attacks using Adversarial Samples." arXiv.
