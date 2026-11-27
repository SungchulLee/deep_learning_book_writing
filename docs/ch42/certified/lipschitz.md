# 립시츠 제약 뉴럴 네트워크

**립시츠로 제약된 네트워크**는 립시츠 상수를 명시적으로 제약해 입력 섭동에 대한 네트워크의 민감도를 통제함으로써 강건성 인증을 얻는다. 립시츠 상수는 입력이 바뀐 만큼에 비해 출력이 최대 몇 배까지 바뀔 수 있는지를 나타내는 값이다. 네트워크의 립시츠 상수가 $L$이면 출력의 변화는 입력 변화의 $L$배를 넘지 못하므로, 그 자체가 강건성 인증이 된다.

---

## 1. 수학 기초

### 립시츠 연속성

함수 $f: \mathbb{R}^d \to \mathbb{R}^m$이 노름 $\|\cdot\|$ 아래 다음을 만족하면 **$L$-립시츠**라 한다.

$$
\|f(\mathbf{x}_1) - f(\mathbf{x}_2)\| \leq L \cdot \|\mathbf{x}_1 - \mathbf{x}_2\|
$$

정의역의 모든 $\mathbf{x}_1, \mathbf{x}_2$에 대해서다.

### 강건성 인증

마진(정답 로짓과 나머지 가운데 가장 큰 로짓의 차)이 $m(\mathbf{x}) = f(\mathbf{x})_y - \max_{k \neq y} f(\mathbf{x})_k$이고 $\ell_2$ 립시츠 상수가 $L$인 분류기에서 인증 반지름은

$$
R(\mathbf{x}) = \frac{m(\mathbf{x})}{\sqrt{2} \cdot L}
$$

$\|\boldsymbol{\delta}\|_2 \leq R(\mathbf{x})$인 어떤 섭동도 예측을 바꾸지 않음이 보장된다. $\sqrt{2}$는 두 로짓의 차 $f_y - f_k$의 립시츠 상수가 많아야 $\sqrt{2}L$이라서 붙는다.

### 레이어를 쌓았을 때의 립시츠 바운드

네트워크 $f = f_L \circ f_{L-1} \circ \cdots \circ f_1$의 전체 립시츠 상수는 레이어별 립시츠 상수의 곱으로 위에서 바운드된다.

$$
L_f \leq \prod_{\ell=1}^L L_{f_\ell}
$$

가중치 행렬이 $\mathbf{W}$인 선형 레이어에서는 $L = \|\mathbf{W}\|_\sigma$(스펙트럼 노름, 곧 가장 큰 특잇값)이고, ReLU에서는 $L = 1$이다.

---

## 2. 스펙트럼 정규화

가장 흔한 방법은 레이어마다 스펙트럼 노름을 제약하는 것이다.

$$
\hat{\mathbf{W}} = \frac{\mathbf{W}}{\|\mathbf{W}\|_\sigma}
$$

이러면 선형 레이어마다 립시츠 상수가 1 이하가 된다. 다만 `spectral_norm`은 스펙트럼 노름을 거듭제곱법(power iteration)으로 근사하므로, 아래 코드처럼 $L = 1$로 두는 것은 근사가 충분히 수렴했다는 가정 위에 선다.

```python
import torch
import torch.nn as nn

class LipschitzLinear(nn.Module):
    """립시츠 테두리를 위해 스펙트럼 기준을 맞춘 선형 레이어."""
    
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
    립시츠 값을 다스린 네트워크.
    
    레이어 모두에 스펙트럼 기준 맞추기를 걸어
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
        L = 1.0  # 스펙트럼 노름을 쓰면 레이어마다 L ≤ 1
        
        # 밝혀 낸 L2 반지름
        radius = margin / (2**0.5 * L)
        radius = torch.clamp(radius, min=0)
        
        return radius
```

---

## 3. 심화 방법

### 직교 레이어

직교 가중치 행렬($\mathbf{W}^\top \mathbf{W} = \mathbf{I}$)을 쓰면 레이어마다 정확히 $L = 1$이 되고 그래디언트의 노름도 보존되어, 신호가 레이어를 지날수록 줄어드는 일이 없다.

- **케일리 변환**: 반대칭(skew-symmetric) 행렬 $\mathbf{A}$로 $\mathbf{W} = (\mathbf{I} - \mathbf{A})(\mathbf{I} + \mathbf{A})^{-1}$처럼 직교 행렬을 매개화한다
- **하우스홀더 반사**: 하우스홀더 반사를 여러 번 곱해 직교 행렬을 만든다

### GroupSort 엑티베이션

**GroupSort** 엑티베이션(아닐 등, 2019)는 ReLU보다 정보를 더 많이 보존하는 1-립시츠 엑티베이션으로, ReLU를 대신한다.

$$
\text{GroupSort}_{k}(\mathbf{x}) = \text{sort groups of } k \text{ elements}
$$

$k=2$이면 원소를 둘씩 짝지어 정렬하며, 이는 짝마다 $(\min(x_1, x_2), \max(x_1, x_2))$를 내놓는 것과 같다(MaxMin이라고도 한다).

---

## 4. 트레이드오프

| 항목 | 립시츠 네트워크 | IBP/CROWN | 무작위 평활화 |
|--------|-------------------|-----------|---------------------|
| 보장의 종류 | 결정론적 | 결정론적 | 확률적 |
| 노름 | $\ell_2$(자연스러움) | $\ell_\infty$(자연스러움) | $\ell_2$ |
| 구조 제약 | 강함 | 중간 | 없음 |
| 인증 반지름 | 작음~중간 | 작음 | 큼 |
| 클린 정확도 | 제한적 | 중간 | 더 높음 |

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

립시츠로 제약된 네트워크는 네트워크 구조 자체에서 깔끔한 강건성 인증을 얻는다. 지금까지는 인증 반지름이 무작위 평활화보다 작지만, 결정론적 보장과 간결한 이론 덕에 활발히 연구되는 분야다.

**살펴볼 거리**

1. Anil, C., Lucas, J., & Grosse, R. (2019). "Sorting Out Lipschitz Function Approximation." ICML.
2. Li, Q., et al. (2019). "Preventing Gradient Attenuation in Lipschitz Constrained Convolutional Networks." NeurIPS.
3. Trockman, A., & Kolter, J. Z. (2021). "Orthogonalizing Convolutional Layers with the Cayley Transform." ICLR.
