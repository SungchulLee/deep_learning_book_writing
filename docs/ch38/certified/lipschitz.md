# 립시츠로 옭아맨 그물
## 들머리

**립시츠로 옭아맨 그물**은 립시츠 값을 드러내 놓고 옭아매어 들임 흔듦에 그물이 얼마나 예민한지를 다스려 밝혀 낸 든든함을 이룬다. 그물의 립시츠 값이 $L$이면 날임의 바뀜은 들임 바뀜의 $L$배로 마디 지어지므로, 그 자체가 든든함의 밝힘이 된다.

## 수학 밑바탕

### 립시츠 이어짐

함수 $f: \mathbb{R}^d \to \mathbb{R}^m$이 노름 $\|\cdot\|$ 아래 다음을 채우면 **$L$-립시츠**라 한다.

$$
\|f(\mathbf{x}_1) - f(\mathbf{x}_2)\| \leq L \cdot \|\mathbf{x}_1 - \mathbf{x}_2\|
$$

밑밭의 모든 $\mathbf{x}_1, \mathbf{x}_2$에 대해서다.

### 든든함의 밝힘

틈이 $m(\mathbf{x}) = f(\mathbf{x})_y - \max_{k \neq y} f(\mathbf{x})_k$이고 립시츠 값이 $L$인 가름개에서 밝혀 낸 반지름은

$$
R(\mathbf{x}) = \frac{m(\mathbf{x})}{\sqrt{2} \cdot L}
$$

$\|\boldsymbol{\delta}\|_2 \leq R(\mathbf{x})$인 어떤 흔듦도 미루어 봄을 바꾸지 않음이 다짐된다.

### 겹쳐 쌓은 립시츠 테두리

그물 $f = f_L \circ f_{L-1} \circ \cdots \circ f_1$에서 온 립시츠 값은 곱으로 마디 지어진다.

$$
L_f \leq \prod_{\ell=1}^L L_{f_\ell}
$$

짐 행렬이 $\mathbf{W}$인 선형 켜에서는 $L = \|\mathbf{W}\|_\sigma$(스펙트럼 노름)이고, ReLU에서는 $L = 1$이다.

## 스펙트럼 잣대 맞추기

가장 흔한 길은 켜마다 스펙트럼 노름을 옭아매는 것이다.

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
    립시츠 값을 다스린 그물.
    
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

## 한발 더 나간 길

### 곧게 어긋난 켜

곧게 어긋난 짐 행렬($\mathbf{W}^\top \mathbf{W} = \mathbf{I}$)을 쓰면 켜마다 꼭 $L = 1$이 되어 테두리가 헐거워지지 않는다.

- **케일리 바꿈**: 비스듬히 대칭인 $\mathbf{A}$으로 $\mathbf{W} = (\mathbf{I} - \mathbf{A})(\mathbf{I} + \mathbf{A})^{-1}$처럼 곧게 어긋난 행렬을 매긴다
- **하우스홀더 되비침**: 하우스홀더 되비침을 겹쳐 곧게 어긋난 행렬을 짓는다

### 무리 줄 세우기 살림

**무리 줄 세우기** 살림(아닐 등, 2019)은 소식을 더 많이 지키는 1-립시츠 ReLU 갈음이다.

$$
\text{GroupSort}_{k}(\mathbf{x}) = \text{sort groups of } k \text{ elements}
$$

$k=2$이면 짝을 줄 세우며, 이는 $(\min(x_1, x_2), \max(x_1, x_2))$과 같다.

## 맞바꿈

| 결 | 립시츠 그물 | IBP/CROWN | 아무렇게나 매끄럽게 하기 |
|--------|-------------------|-----------|---------------------|
| 밝힘의 갈래 | 붙박인 것 | 붙박인 것 | 낌새의 것 |
| 노름 | $\ell_2$(제격) | $\ell_\infty$(제격) | $\ell_2$ |
| 얼개 옭아맴 | 셈 | 가운데 | 없음 |
| 밝혀 낸 반지름 | 작음~가운데 | 작음 | 큼 |
| 맑은 맞음 | 마디 있음 | 가운데 | 더 높음 |

## 간추림

립시츠로 옭아맨 그물은 그물 얼개 자체에서 깔끔한 든든함 밝힘을 준다. 이제까지 밝혀 낸 반지름은 아무렇게나 매끄럽게 하기보다 작지만, 붙박인 다짐과 말끔한 이론 틀 덕에 한창이고 앞날이 밝은 연구 갈래다.

## 살펴볼 거리

1. Anil, C., Lucas, J., & Grosse, R. (2019). "Sorting Out Lipschitz Function Approximation." ICML.
2. Li, Q., et al. (2019). "Preventing Gradient Attenuation in Lipschitz Constrained Convolutional Networks." NeurIPS.
3. Trockman, A., & Kolter, J. Z. (2021). "Orthogonalizing Convolutional Layers with the Cayley Transform." ICLR.

## 익힘 문제

**익힘 1.**
선형 가름개 $f(x) = w^T x + b$에서 미루어 본 갈래를 바꾸는 데 드는 가장 작은 $\ell_\infty$ 흔듦을 셈하여라. 이것이 신경 그물의 든든함과 어떻게 이어지는지 밝혀라.

??? success "익힘 1 풀이"
    선형 가름개에서 $\ell_\infty$ 노름으로 잰 판단의 금까지의 거리는 $\frac{|w^T x + b|}{\|w\|_1}$이다. 가장 작은 흔듦은 $\delta^* = \frac{|w^T x + b|}{\|w\|_1} \cdot \text{sign}(w)$이다. 신경 그물에서는 그 자리의 선형 어림 $f(x + \delta) \approx f(x) + \nabla_x f \cdot \delta$이 FGSM(기울기의 부호를 쓴다)이 왜 잘 듣는지를 밝혀 준다. 차수가 높은 모형이 무른 까닭은 $\|w\|_1$은 차수와 함께 커지는데 $|w^T x + b|$은 꼭 그렇지 않아 든든함의 여유가 줄어들기 때문이다. $\square$

---

**익힘 2.**
이 마디에서 다룬 치기나 막이를 CIFAR-10의 ResNet-18 모형에 짜 넣어라. $\epsilon = 8/255$의 PGD-20 치기 아래에서 맑은 맞음과 든든한 맞음을 알려라.

??? success "익힘 2 풀이"
    여느 ResNet-18은 맑은 맞음이 $\sim$93%이지만 PGD-20($\epsilon = 8/255$, 걸음 크기 $2/255$) 아래의 든든한 맞음은 $\sim$0%이다. 이 마디의 방법을 걸면 결과는 재주에 따라 다르다. 맞서며 익히기는 맑은 맞음 $\sim$83%에 든든한 맞음 $\sim$50%이고, 밝혀 낸 막이는 더 낮지만 증명할 수 있는 테두리를 준다. 맞음과 든든함의 맞바꿈은 밑바탕부터 있는 것이라, 든든함을 높이면 맑은 맞음이 흔히 5~15% 든다. 아무렇게나 하는 씨앗 3개의 평균과 잣대 어긋남으로 알려라. $\square$

---

**익힘 3.**
흔듦 공 안에서 갈래별 자료의 밑자리가 서로 겹친다고 볼 때, 모형이 담는 힘을 키우지 않고서는 어떤 막이도 맑은 자료의 높은 맞음과 $\ell_\infty$ 흔듦에 대한 높은 든든함을 함께 이룰 수 없음을 증명하여라.

??? success "익힘 3 풀이"
    두 갈래의 밑자리가 거리 $\epsilon$ 안에서 겹치면(곧 $\|x_1 - x_2\|_\infty \leq 2\epsilon$인 $x_1 \in \text{갈래 1}, x_2 \in \text{갈래 2}$이 있으면), $x_1$과 $x_2$ 둘 다에서 든든한 가름개는 적어도 하나를 틀리게 가를 수밖에 없다(흔듦 공이 겹치기 때문이다). 이것이 맞음과 든든함의 밑바탕 맞바꿈이다. 겹치는 밑자리의 몫이 피할 수 없는 맞음 잃음을 정한다. 여느 그림 분포에서는 $\epsilon = 8/255$에서 겹침이 꽤 있어, 살펴본 10~15%의 맞음 떨어짐을 밝혀 준다. 모형이 담는 힘을 키우면(더 너른 그물) 얽힌 든든한 판단의 금을 더 잘 그려 맞바꿈을 얼마쯤 눅일 수 있다. $\square$

---

**익힘 4.**
금융 기계 배움 얼개(속임수 알아내기나 거래 신호 만들기 따위)에서 맞섬의 든든함이 어떻게 드러나는지 다루어라. 으름 얼개가 보기 다룸과 어떻게 다른가?

??? success "익힘 4 풀이"
    금융에서 겨루는 이는 알아내는 얼개에 맞추어 스스로 움직이는 꾀 많은 무리(속임수꾼, 저자 흔드는 이)다. 보기 다룸과 다른 고갱이는 이렇다. (1) 흔들 수 있는 밭이 돈으로 될 만한 것에 옭매인다(속임수꾼이 제 거래 자취를 통째로 바꿀 수는 없다). (2) 치기가 잇따르며 맞추어 간다(겨루는 이가 얼개의 되받음을 보고 손본다). (3) 헛 맞음과 놓침의 값이 서로 어긋난다(옳은 거래를 막는 것과 속임수를 놓치는 것). (4) $\ell_p$ 노름은 뜻이 없고 밭에 맞는 흔듦 모형이 있어야 한다. 막이는 맞추어 오는 겨루는 이에게도 든든해야 하므로, 알아내는 잣대가 알려지면 비껴갈 수 있는 알아내기 바탕의 길은 많이 걸러진다. $\square$
