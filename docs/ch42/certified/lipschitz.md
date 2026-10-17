# 립시츠 제약 뉴럴 넷트웍

**립시츠로 제약된 네트워크**는 립시츠 값을 드러내 놓고 옭아매어 입력 섭동에 네트워크가 얼마나 예민한지를 다스려 강건성 인증을 이룬다. 네트워크의 립시츠 값이 $L$이면 출력의 바뀜은 입력 바뀜의 $L$배로 마디 지어지므로, 그 자체가 강건성의 밝힘이 된다.

---

## 1. 수학 기초

### 립시츠 이어짐

함수 $f: \mathbb{R}^d \to \mathbb{R}^m$이 노름 $\|\cdot\|$ 아래 다음을 채우면 **$L$-립시츠**라 한다.

$$
\|f(\mathbf{x}_1) - f(\mathbf{x}_2)\| \leq L \cdot \|\mathbf{x}_1 - \mathbf{x}_2\|
$$

밑밭의 모든 $\mathbf{x}_1, \mathbf{x}_2$에 대해서다.

### 강건성의 밝힘

틈이 $m(\mathbf{x}) = f(\mathbf{x})_y - \max_{k \neq y} f(\mathbf{x})_k$이고 립시츠 값이 $L$인 분류기에서 밝혀 낸 반지름은

$$
R(\mathbf{x}) = \frac{m(\mathbf{x})}{\sqrt{2} \cdot L}
$$

$\|\boldsymbol{\delta}\|_2 \leq R(\mathbf{x})$인 어떤 섭동도 추론을 바꾸지 않음이 다짐된다.

### 겹쳐 쌓은 립시츠 테두리

네트워크 $f = f_L \circ f_{L-1} \circ \cdots \circ f_1$에서 온 립시츠 값은 곱으로 마디 지어진다.

$$
L_f \leq \prod_{\ell=1}^L L_{f_\ell}
$$

짐 행렬이 $\mathbf{W}$인 선형 켜에서는 $L = \|\mathbf{W}\|_\sigma$(스펙트럼 노름)이고, ReLU에서는 $L = 1$이다.

---

## 2. 스펙트럼 잣대 맞추기

가장 흔한 길은 켜마다 스펙트럼 노름을 제약하는 것이다.

$$
\hat{\mathbf{W}} = \frac{\mathbf{W}}{\|\mathbf{W}\|_\sigma}
$$

이러면 선형 켜마다 립시츠 값이 많아야 1이 된다.

```python
import torch
import torch.nn as nn

class LipschitzLinear(nn.Module):
    """립시츠 테두리를 위해 스펙트럼 잣대를 맞춘 선형 켜."""
    
    def __init__(self, in_features, out_features, lip_const=1.0):
        super().__init__()
        self.linear = nn.utils.spectral_norm(
            nn.Linear(in_features, out_features)
        )
        self.lip_const = lip_const
    
    def forward(self, x):
        return self.lip_const * self.linear(x)

class LipschitzNetwork(nn.Module):
    """
    립시츠 값을 다스린 넷트웍.
    
    켜 모두에 스펙트럼 잣대 맞추기를 걸어
    온 립시츠 값을 마디 짓는다.
    """
    
    def __init__(self, input_dim, hidden_dims, num_classes):
        super().__init__()
        layers = []
        prev_dim = input_dim
        
        for h in hidden_dims:
            layers.append(LipschitzLinear(prev_dim, h))
            layers.append(nn.ReLU())  # 립시츠 값 = 1
            prev_dim = h
        
        layers.append(LipschitzLinear(prev_dim, num_classes))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x.flatten(1))
    
    def certify(self, x, y):
        """들임마다 밝혀 낸 반지름을 셈한다."""
        logits = self.forward(x)
        
        # 가름의 틈
        true_logit = logits.gather(1, y.view(-1, 1)).squeeze()
        mask = torch.ones_like(logits).scatter_(1, y.view(-1, 1), 0)
        max_other = (logits * mask - (1 - mask) * 1e9).max(dim=1)[0]
        margin = true_logit - max_other
        
        # 립시츠 값(스펙트럼 노름의 곱, 모두 ≤ 1)
        L = 1.0  # 스펙트럼 노름을 쓰면 켜마다 L ≤ 1
        
        # 밝혀 낸 L2 반지름
        radius = margin / (2**0.5 * L)
        radius = torch.clamp(radius, min=0)
        
        return radius
```

---

## 3. 한발 더 나간 길

### 곧게 어긋난 켜

곧게 어긋난 짐 행렬($\mathbf{W}^\top \mathbf{W} = \mathbf{I}$)을 쓰면 켜마다 꼭 $L = 1$이 되어 테두리가 헐거워지지 않는다.

- **케일리 바꿈**: 비스듬히 대칭인 $\mathbf{A}$으로 $\mathbf{W} = (\mathbf{I} - \mathbf{A})(\mathbf{I} + \mathbf{A})^{-1}$처럼 곧게 어긋난 행렬을 매긴다
- **하우스홀더 되비침**: 하우스홀더 되비침을 겹쳐 곧게 어긋난 행렬을 짓는다

### 무리 줄 세우기 활성화

**무리 줄 세우기** 활성화(아닐 등, 2019)은 소식을 더 많이 지키는 1-립시츠 ReLU 갈음이다.

$$
\text{GroupSort}_{k}(\mathbf{x}) = \text{sort groups of } k \text{ elements}
$$

$k=2$이면 짝을 줄 세우며, 이는 $(\min(x_1, x_2), \max(x_1, x_2))$과 같다.

---

## 4. 트레이드오프

| 피처 | 립시츠 네트워크 | IBP/CROWN | 아무렇게나 매끄럽게 하기 |
|--------|-------------------|-----------|---------------------|
| 밝힘의 클래스 | 붙박인 것 | 붙박인 것 | 확률의 것 |
| 노름 | $\ell_2$(제격) | $\ell_\infty$(제격) | $\ell_2$ |
| 구조 제약 | 계산 | 가운데 | 없음 |
| 밝혀 낸 반지름 | 작음~가운데 | 작음 | 큼 |
| 클린 정확도 | 마디 있음 | 가운데 | 더 높음 |

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
이 마디에서 다룬 공격나 방어를 CIFAR-10의 ResNet-18 모델에 짜 넣어라. $\epsilon = 8/255$의 PGD-20 공격 아래에서 클린 정확도와 강건 정확도를 알려라.

</div>

??? success "풀이"
    여느 ResNet-18은 클린 정확도가 $\sim$93%이지만 PGD-20($\epsilon = 8/255$, 스텝 크기 $2/255$) 아래의 강건 정확도는 $\sim$0%이다. 이 마디의 방법을 걸면 피처와는 재주에 따라 다르다. 적대적 학습은 클린 정확도 $\sim$83%에 강건 정확도 $\sim$50%이고, 인증 가능한 방어는 더 낮지만 증명할 수 있는 테두리를 준다. 정확도와 강건성의 트레이드오프는 기초부터 있는 것이라, 강건성을 높이면 클린 정확도가 흔히 5~15% 든다. 아무렇게나 하는 시드 3개의 평균과 표준편차로 알려라. $\square$

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

립시츠로 제약된 네트워크는 네트워크 구조 자체에서 깔끔한 강건성 밝힘을 준다. 이제까지 밝혀 낸 반지름은 아무렇게나 매끄럽게 하기보다 작지만, 붙박인 다짐과 말끔한 이론 틀 덕에 한창이고 앞날이 밝은 연구 클래스다.

**살펴볼 거리**

1. Anil, C., Lucas, J., & Grosse, R. (2019). "Sorting Out Lipschitz Function Approximation." ICML.
2. Li, Q., et al. (2019). "Preventing Gradient Attenuation in Lipschitz Constrained Convolutional Networks." NeurIPS.
3. Trockman, A., & Kolter, J. Z. (2021). "Orthogonalizing Convolutional Layers with the Cayley Transform." ICLR.
