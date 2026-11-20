# 전이 기반 공격

**전이 공격**은 적대적 샘플의 기본 성질 하나를 이용한다. 한 모델(**대리 모델**, surrogate)을 겨냥해 만든 섭동이 다른 모델(**타깃**)까지 속이는 일이 잦다는 것이다. 이로써 **질의가 필요 없는** 블랙박스 공격이 가능해진다. 적대자는 타깃 모델에 직접 접근할 필요가 없다.

---

## 1. 전이 현상

### 적대적 샘플은 왜 전이되는가

모델 사이의 전이는 몇 가지로 설명된다.

1. **공유된 피처**: 비슷한 데이터로 학습된 모델은 비슷한 피처를 배우므로, 그 피처를 흔드는 적대적 섭동은 다른 모델로도 옮아간다
2. **비슷한 결정 경계**: 구조와 학습 절차가 비슷한 모델은 비슷한 결정 경계를 갖는다
3. **강건하지 않은 피처의 보편성**: 적대적 공격이 이용하는 강건하지 않은 피처는 여러 모델 계열이 함께 갖고 있다

### 수식으로 적기

내부를 다 볼 수 있는 대리 모델 $f_s$와 볼 수 없는 타깃 모델 $f_t$가 있을 때

$$
\boldsymbol{\delta}^* = \text{WhiteBoxAttack}(f_s, \mathbf{x}, y) \implies f_t(\mathbf{x} + \boldsymbol{\delta}^*) \neq y \text{ (with non-trivial probability)}
$$

전이율은 대리 모델과 타깃 모델이 얼마나 닮았는지에 따라 크게 달라진다.

---

## 2. 전이성 높이기

### 모멘텀 반복 FGSM(MI-FGSM)

Dong 등(2018)은 반복 공격에 **모멘텀**을 더하면 전이가 크게 나아짐을 보였다. 일반 PGD는 대리 모델의 로스 지형에 과적합할 수 있는데, 모멘텀이 공격 방향을 안정시켜 준다.

MI-FGSM의 갱신 규칙은

$$
\begin{aligned}
\mathbf{g}^{(t)} &= \mu \cdot \mathbf{g}^{(t-1)} + \frac{\nabla_\mathbf{x} \mathcal{L}(f_s(\mathbf{x}^{(t)}), y)}{\|\nabla_\mathbf{x} \mathcal{L}(f_s(\mathbf{x}^{(t)}), y)\|_1} \\
\mathbf{x}^{(t+1)} &= \Pi_\varepsilon\left(\mathbf{x}^{(t)} + \alpha \cdot \text{sign}(\mathbf{g}^{(t)})\right)
\end{aligned}
$$

여기서 $\mu$는 모멘텀의 감쇠 계수다(흔히 1.0).

### 앙상블 공격

대리 모델 여럿의 앙상블을 공격하면 전이율이 높아진다.

$$
\mathcal{L}_{\text{ensemble}} = \sum_{k=1}^K w_k \cdot \mathcal{L}(f_k(\mathbf{x} + \boldsymbol{\delta}), y)
$$

여기서 $w_k$는 앙상블 가중치다. 대리 모델 여럿을 한꺼번에 속이는 섭동은 본 적 없는 타깃에도 옮아갈 확률이 크다.

### 입력 다양화(DI-FGSM)

Xie 등(2019)은 공격을 만드는 동안 입력에 무작위 변환을 걸자고 제안했다.

$$
\nabla_\mathbf{x} \mathcal{L}(f_s(T(\mathbf{x}^{(t)})), y)
$$

여기서 $T$는 무작위로 크기를 바꾸고 패딩을 넣는 변환이다. 이렇게 하면 공격이 특정한 입력 모양에 과적합하는 것을 막는다.

### 평행 이동 불변 공격(TI-FGSM)

그래디언트를 커널과 합성곱하면 섭동이 평행 이동에 덜 민감해진다.

$$
\mathbf{g}^{(t)} = W * \nabla_\mathbf{x} \mathcal{L}(f_s(\mathbf{x}^{(t)}), y)
$$

여기서 $W$는 흔히 가우스 커널이다.

---

## 3. PyTorch로 구현하기

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict

class TransferAttack:
    """
    밀어 나감과 모둠을 쓰는 옮아감 바탕의 검은 상자 치기.
    
    대신 쓰는 모형을 겨냥해 맞서는 보기를 만들어
    모르는 타깃 모형으로 옮아가게 한다.
    
    Parameters
    ----------
    surrogate_models : list[nn.Module]
        대신 쓰는 모형 하나 이상
    epsilon : float
        섭동 예산
    alpha : float
        스텝 크기
    num_iter : int
        치기 회귀 횟수
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
            
            # 대신 쓰는 모형에 걸친 모둠 로스
            total_loss = 0
            for model in self.surrogates:
                x_input = self._input_diversity_transform(x_adv)
                logits = model(x_input)
                
                if targeted:
                    loss = -F.cross_entropy(logits, target_labels)
                else:
                    loss = F.cross_entropy(logits, y)
                
                total_loss += loss / len(self.surrogates)
            
            # 그래디언트를 셈한다
            for model in self.surrogates:
                model.zero_grad()
            total_loss.backward()
            grad = x_adv.grad.data
            
            # 그래디언트의 기준을 맞춘다(L1으로)
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

## 4. 전이율 분석

### 전이에 영향을 주는 요인

| 요인 | 전이율에 미치는 영향 | 설명 |
|--------|------------------------|-------------|
| 구조가 닮음 | 닮을수록 → 더 잘 전이 | 닮은 구조는 닮은 피처를 배운다 |
| 학습 데이터가 겹침 | 많이 겹칠수록 → 더 잘 전이 | 공유한 데이터가 공유된 피처를 낳는다 |
| 모멘텀 | 크게 나아짐 | 대리 모델에 과적합하는 것을 피한다 |
| 대리 모델 앙상블 | 꽤 나아짐 | 여러 모델에 공통인 피처를 겨냥하게 된다 |
| 입력 다양화 | 어느 정도 나아짐 | 특정 입력에 매인 과적합을 줄인다 |
| 반복 횟수 | 수확 체감 | 반복이 많으면 대리 모델에 과적합한다 |

### 흔히 보고되는 전이율(CIFAR-10, epsilon = 8/255)

아래 값은 문헌에서 흔히 보고되는 크기이며 이 쪽에서 잰 값이 아니다.

| 대리 모델 → 타깃 | FGSM | MI-FGSM-20 | 앙상블 MI-FGSM |
|---------------------|------|-----------|-------------------|
| ResNet-18 → VGG-16 | 약 30% | 약 50% | 약 65% |
| ResNet-18 → DenseNet | 약 35% | 약 55% | 약 70% |
| ResNet-18 → ResNet-50 | 약 45% | 약 65% | 약 75% |

---

## 5. 금융 응용

전이 공격은 다음과 같은 금융 상황에서 특히 문제가 된다.

- 적대자가 쓰인 모델의 종류는 알아도(신용 점수에 그래디언트 부스팅을 쓴다는 것 따위) 정확한 파라메터는 모를 때
- 비슷한 데이터로 학습된 공개 모델을 대리 모델로 쓸 수 있을 때
- 규제가 쓸 수 있는 모델 종류를 정해 두어 공격자가 그레이박스 지식을 얻을 때

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
선형 분류기 $f(x) = w^T x + b$에서 예측한 클래스를 바꾸는 데 드는 가장 작은 $\ell_\infty$ 섭동을 계산하여라. 이것이 뉴럴 네트워크의 강건성과 어떻게 이어지는지 설명하여라.

</div>

??? success "풀이"
    선형 분류기에서 $\ell_\infty$ 노름으로 잰 결정 경계까지의 거리는 $\frac{|w^T x + b|}{\|w\|_1}$이다. 가장 작은 섭동은 $\delta^* = \frac{|w^T x + b|}{\|w\|_1} \cdot \text{sign}(w)$이다. 뉴럴 네트워크에서는 국소 선형 근사 $f(x + \delta) \approx f(x) + \nabla_x f \cdot \delta$가 FGSM(그래디언트의 부호를 쓴다)이 왜 잘 통하는지를 설명해 준다. 차원이 높은 모델이 취약한 까닭은 $\|w\|_1$은 차원과 함께 커지는데 $|w^T x + b|$은 꼭 그렇지 않아 강건성의 여유가 줄어들기 때문이다. $\square$

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
    두 클래스의 서포트가 거리 $\epsilon$ 안에서 겹치면(곧 $\|x_1 - x_2\|_\infty \leq 2\epsilon$인 $x_1 \in \text{클래스 1}, x_2 \in \text{클래스 2}$가 있으면), $x_1$과 $x_2$ 둘 다에서 강건한 분류기는 적어도 하나를 틀리게 가를 수밖에 없다(섭동 공이 겹치기 때문이다). 이것이 정확도와 강건성의 근본적인 트레이드오프다. 겹치는 서포트의 비중이 피할 수 없는 정확도 손실을 정한다. 자연 이미지 분포에서는 $\epsilon = 8/255$에서 겹침이 꽤 있어, 관찰되는 10~15%의 정확도 하락을 설명해 준다. 모델 용량을 키우면(더 넓은 네트워크) 복잡한 강건 결정 경계를 더 잘 그려 트레이드오프를 어느 정도 완화할 수 있다. $\square$

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
금융 머신러닝 시스템(사기 탐지나 거래 신호 생성 등)에서 적대적 강건성이 어떻게 드러나는지 논하여라. 위협 모델이 비전과 어떻게 다른가?

</div>

??? success "풀이"
    금융에서 적대자는 탐지 시스템에 맞추어 전략을 바꾸는 교묘한 집단(사기꾼, 시장 조작자)이다. 비전과 다른 핵심은 이렇다. (1) 섭동할 수 있는 피처가 경제적으로 그럴듯한 범위로 제약된다(사기꾼이 자기 거래 이력을 통째로 바꿀 수는 없다). (2) 공격이 순차적이고 적응적이다(적대자가 시스템의 응답을 보고 고친다). (3) 거짓 양성과 거짓 음성의 비용이 비대칭이다(정상 거래를 막는 것과 사기를 놓치는 것). (4) $\ell_p$ 노름은 의미가 없고 도메인에 맞는 섭동 모델이 있어야 한다. 방어는 적응형 적대자에게도 강건해야 하므로, 탐지 기준이 알려지면 회피할 수 있는 탐지 기반 방법은 쓸모가 크게 줄어든다. $\square$

## 정리하며

이 절에서는 전이 현상과 그 까닭, 전이성을 높이는 기법(MI-FGSM, 앙상블, DI-FGSM, TI-FGSM), PyTorch 구현, 전이율에 영향을 주는 요인을 살펴보았다.

**살펴볼 거리**

1. Dong, Y., et al. (2018). "Boosting Adversarial Attacks with Momentum." CVPR.
2. Xie, C., et al. (2019). "Improving Transferability of Adversarial Examples with Input Diversity." CVPR.
3. Tramèr, F., et al. (2018). "Ensemble Adversarial Training: Attacks and Defenses." ICLR.
4. Papernot, N., McDaniel, P., & Goodfellow, I. (2016). "Transferability in Machine Learning: from Phenomena to Black-Box Attacks using Adversarial Samples." arXiv.
