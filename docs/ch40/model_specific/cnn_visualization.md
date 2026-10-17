# CNN 시각화와 분해 방법

CNN에 맞춘 풀이 방법은 겹치는 뉴럴 넷트웍의 레이어 있고 자리가 있는 구조를 살려, 모델을 가리지 않는 길로는 얻을 수 없는 것을 들여다보게 한다. 이 마디는 **레이어마다의 쓸모 퍼뜨리기(LRP)**와 **DeepLIFT**을 다룬다. 둘 다 내놓기에서 네트워크를 거꾸로 거슬러 쓸모를 퍼뜨리는 쪼개기 방법이며, 그래디언트 방법에 없는 지켜짐 성능을 채운다.

---

## 1. 레이어마다의 쓸모 퍼뜨리기(LRP)

### 지켜짐 이치

LRP은 쓸모 점수를 레이어마다 거꾸로 퍼뜨려 뉴럴 넷트웍의 추론을 쪼개며, 레이어를 지나도 온 쓸모가 그대로 남는 지켜짐 성능을 채운다.

$$
\sum_j R_j^{(l)} = \sum_i R_i^{(l+1)} = \ldots = f(\mathbf{x})
$$

### 퍼뜨리는 규칙

**LRP-0(기본 규칙):**

$$
R_i^{(l)} = \sum_j \frac{a_i w_{ij}}{\sum_{i'} a_{i'} w_{i'j}} R_j^{(l+1)}
$$

**LRP-엡실론(든든하게 한 것):**

$$
R_i^{(l)} = \sum_j \frac{a_i w_{ij}}{\epsilon + \sum_{i'} a_{i'} w_{i'j}} R_j^{(l+1)}
$$

**LRP-감마(양수를 앞세운 것):**

$$
R_i^{(l)} = \sum_j \frac{a_i (w_{ij} + \gamma w_{ij}^+)}{\sum_{i'} a_{i'} (w_{i'j} + \gamma w_{i'j}^+)} R_j^{(l+1)}
$$

### 섞어 쓰는 꾀

깊이마다 다른 규칙을 쓰는 것이 잘 쓰는 길이다.

| 레이어 클래스 | 규칙 | 까닭 |
|------------|------|-----------|
| 입력 레이어 | LRP-zB(매인 것) | 입력 밭의 울타리를 지킨다 |
| 아래쪽 컨볼루션 레이어 | LRP-감마 ($\gamma=0.25$) | 양수 증거를 앞세운다 |
| 위쪽 레이어 | LRP-엡실론 ($\epsilon=0.01$) | 잡음을 누른다 |
| 빽빽한 레이어 | LRP-엡실론 | 쪼개기가 든든하다 |

### 구현

```python
import torch
import torch.nn as nn

class LRP:
    """켜마다의 쓸모 퍼뜨리기."""

    def __init__(self, model, epsilon=1e-6):
        self.model = model
        self.epsilon = epsilon
        self.activations = {}
        self._register_hooks()

    def _register_hooks(self):
        def get_activation(name):
            def hook(module, input, output):
                self.activations[name] = output.detach()
            return hook

        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                module.register_forward_hook(get_activation(name))

    def propagate_linear(self, layer, relevance, activation):
        """선형 켜를 지나는 LRP-엡실론."""
        W = layer.weight
        z = activation.unsqueeze(1) * W.unsqueeze(0)
        z_sum = z.sum(dim=2, keepdim=True) + self.epsilon
        s = relevance.unsqueeze(2) / z_sum
        c = z * s
        return c.sum(dim=1)

    def __call__(self, image_tensor, target_class, device):
        self.model.eval()
        image_tensor = image_tensor.to(device)
        output = self.model(image_tensor)

        relevance = torch.zeros_like(output)
        relevance[0, target_class] = output[0, target_class]

        return relevance
```

### Captum 쓰기

```python
from captum.attr import LRP as CaptumLRP

lrp = CaptumLRP(model)
attribution = lrp.attribute(input_tensor, target=target_class)
```

---

## 2. DeepLIFT

### 핵심 생각

DeepLIFT은 활성화를 비교 활성화와 비교하여 추론을 풀이하며, 잦아든 뉴런에서 그래디언트가 사라지는 문제를 다룬다.

$$
\Delta y = f(\mathbf{x}) - f(\mathbf{x}^0) = \sum_i C_{\Delta x_i \Delta y}
$$

### 잦아듦에서 나은 점

잦아든 자리($x = 10$, $x^0 = 0$)의 시그모이드 뉴런에서

- **그래디언트**: $\sigma'(10) \approx 0.000045$(거의 0)
- **DeepLIFT**: $(\sigma(10) - \sigma(0)) / (10 - 0) \approx 0.05$(뜻있다)

### 구현

```python
from captum.attr import DeepLift, DeepLiftShap

# 밑금 하나
deeplift = DeepLift(model)
attribution = deeplift.attribute(
    input_tensor, target=target_class,
    baselines=torch.zeros_like(input_tensor)
)

# 밑금 여럿 (DeepLIFT SHAP)
deeplift_shap = DeepLiftShap(model)
attribution = deeplift_shap.attribute(
    input_tensor, target=target_class,
    baselines=baseline_distribution
)
```

---

## 3. 피처 그림 그리기

기여도 분석을 넘어, CNN 그림 그리기에는 뉴런나 레이어마다 어떤 피처를 배웠는지 알아보는 재주도 든다.

### 활성화 최대화

정한 뉴런을 가장 크게 깨우는 입력을 찾는다.

$$
\mathbf{x}^* = \arg\max_{\mathbf{x}} a_k(\mathbf{x}) - \lambda \|\mathbf{x}\|^2
$$

### 거르개 그리기

```python
def visualize_filters(model, layer_name):
    """배운 겹치기 거르개를 그린다."""
    for name, module in model.named_modules():
        if name == layer_name and isinstance(module, nn.Conv2d):
            weights = module.weight.data.cpu()
            n_filters = min(weights.shape[0], 64)

            fig, axes = plt.subplots(8, 8, figsize=(12, 12))
            for i, ax in enumerate(axes.flat):
                if i < n_filters:
                    w = weights[i]
                    if w.shape[0] == 3:
                        w = (w - w.min()) / (w.max() - w.min())
                        ax.imshow(w.permute(1, 2, 0))
                    else:
                        ax.imshow(w[0], cmap='gray')
                ax.axis('off')
            return fig
```

---

## 4. 비교

| 방법 | 지켜짐 | 계산 | 잦아듦 다루기 | 이론 |
|--------|-------------|-------------|-------------------|--------|
| LRP | 예 | 가운데 | 규칙에 달렸다 | 테일러 쪼개기 |
| DeepLIFT | 예 | 빠름 | 예 | 비교와 맞대기 |
| Integrated Gradients | 예(온전함) | 느림 | 예 | 길 적분 |
| 맨 그래디언트 | 아니오 | 아주 빠름 | 아니오 | 그 자리 예민함 |

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
이 마디에서 밝힌 풀이 방법을, XOR 입력을 가르는 ReLU 활성화의 두 레이어 뉴럴 넷트웍에 걸어라. 입력 $x = [1, 1]$에 대한 풀이를 계산하여라.

</div>

??? success "풀이"
    웨이트가 $W_1, b_1, W_2, b_2$인 학습된 XOR 네트워크에서 내놓기는 $f(x) = W_2 \cdot \text{ReLU}(W_1 x + b_1) + b_2$이다. 풀이 방법은 입력 피처마다 몫을 내놓는다. $x = [1, 1]$(클래스 0)이면 두 피처 모두 음수 분류에 이바지한다. 몫 값은 방법마다 다르다. 그래디언트 기반 방법은 $\partial f / \partial x_i$을 계산하고, 흔들어 보는 방법은 피처를 가렸을 때 내놓기가 얼마나 바뀌는지 잰다. XOR 문제는 판단 금이 선형이 아니므로 선형 풀이 방법이 그르칠 수 있음을 보여 준다. $\square$

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff hard" title="어려움"></span>
이 마디의 풀이 방법이 온전함 공리를 채우는지, 곧 어떤 밑금 $x_0$에 대해 모든 피처 몫의 합이 $f(x) - f(x_0)$과 같은지 증명하거나 뒤집어라.

</div>

??? success "풀이"
    온전함 공리(섀플리 값 이론에서는 효율이라고도 한다)는 몫의 합이 입력에서의 모델 배포와 밑금에서의 내놓기의 차이와 같다는 것이다. 이 방법이 온전함을 채우는지는 그 세움새에 달렸다. 그래디언트 방법은 온전함을 채우지 못한다(그래디언트는 그 자리의 것이고 길을 따라 쌓은 것이 아니다). Integrated Gradients는 세움새 자체로 온전함을 채운다(길을 따라 미적분의 밑정리를 쓴다). SHAP 값은 섀플리 공리로 효율을 채운다. 온전함을 어기는 방법은 몫을 너무 많거나 적게 매길 수 있어, 온 몫을 온 세상 풀이로 믿기 어렵게 만든다. $\square$

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
이 방법이 내놓는 풀이가 얼마나 미더운지 따지는 시험을 꾸며라. 짚어 준 피처가 참으로 모델에 중요한지를 넣기와 빼기 곡선으로 재어라.

</div>

??? success "풀이"
    절차는 이렇다. (1) 시험 그림마다 피처 몫을 계산한다. (2) 빼기: 몫이 큰 차례로 피처를 하나씩 가리며 모델의 자신함이 떨어지는 모습을 적는다. 미더운 풀이면 자신함이 빠르게 떨어진다. (3) 넣기: 빈 밑금에서 시작해 몫이 큰 차례로 피처를 하나씩 드러내며 자신함이 오르는 모습을 적는다. 미더운 풀이면 자신함이 빠르게 오른다. (4) 두 곡선의 아래 넓이를 계산한다. (5) 아무렇게나 매긴 차례(밑금)와 다른 방법에 견준다. 미더운 방법이면 빼기 넓이가 작고 넣기 넓이가 커야 한다. 통계로 미더우려면 시험 표본 1000개 넘게 되풀이한다. $\square$

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
이 풀이 방법을 신용 부도를 예측하는 금융 모델에 어떻게 걸 수 있는지 다루어라. 풀이가 채워야 할 규정 요건은 무엇인가?

</div>

??? success "풀이"
    신용 모델에는 규정(ECOA, GDPR 22조)이 불리한 판단마다 그 사람에게 맞춘 풀이를 바란다. 방법은 다음을 내놓아야 한다. (1) 물리침에 가장 크게 이바지한 인자(불리한 처분 까닭). (2) 한결같은 풀이(비슷한 신청자는 비슷한 풀이를 받는다). (3) 손에 잡히는 풀이(신청자가 무엇을 바꾸어야 하는지 안다). 이 마디의 풀이 방법으로 피처의 중요함을 짚을 수 있으나, 든든함(입력이 조금 바뀌었다고 풀이가 확 달라지면 안 된다)과 옳음(중요한 피처를 없애면 추론이 바뀌어야 한다)을 따져 보아야 한다. 지켜야 할 성능은 대리 차별이 드러나지 않도록 조심히 다루어야 한다. $\square$

## 정리하며

CNN에 맞춘 쪼개기 방법인 LRP과 DeepLIFT은 지켜짐을 지키는 기여도 분석을 주어, 추론을 입력 피처에 꼭 맞게 나누어 담는다. 피처 그림 그리기 재주와 아우르면 CNN이 어떻게 판단하는지 두루 들여다볼 수 있다.

**살펴볼 거리**

1. Bach, S., et al. (2015). "On Pixel-wise Explanations for Non-Linear Classifier Decisions by Layer-wise Relevance Propagation." *PLoS ONE*.

2. Shrikumar, A., et al. (2017). "Learning Important Features Through Propagating Activation Differences." *ICML*.

3. Montavon, G., et al. (2019). "Layer-wise Relevance Propagation: An Overview." *Explainable AI*.

4. Olah, C., et al. (2017). "Feature Visualization." *Distill*.
