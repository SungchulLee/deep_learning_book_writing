# CNN 시각화와 분해 방법

CNN에 맞춘 설명 방법은 겹치는 뉴럴 네트워크의 레이어 있고 자리가 있는 구조를 살려, 모델에 무관한 길로는 얻을 수 없는 것을 들여다보게 한다. 이 절은 **레이어별 관련성 전파(LRP)**와 **DeepLIFT**을 다룬다. 둘 다 출력에서 네트워크를 거꾸로 거슬러 쓸모를 전파하는 쪼개기 방법이며, 그래디언트 방법에 없는 보존 성질을 채운다.

---

## 1. 레이어별 관련성 전파(LRP)

### 보존 원리

LRP은 관련성 점수를 레이어마다 거꾸로 전파해 뉴럴 네트워크의 추론을 쪼개며, 레이어를 지나도 온 쓸모가 그대로 남는 보존 성질을 채운다.

$$
\sum_j R_j^{(l)} = \sum_i R_i^{(l+1)} = \ldots = f(\mathbf{x})
$$

### 전파하는 규칙

**LRP-0(기본 규칙):**

$$
R_i^{(l)} = \sum_j \frac{a_i w_{ij}}{\sum_{i'} a_{i'} w_{i'j}} R_j^{(l+1)}
$$

**LRP-엡실론(강건하게 한 것):**

$$
R_i^{(l)} = \sum_j \frac{a_i w_{ij}}{\epsilon + \sum_{i'} a_{i'} w_{i'j}} R_j^{(l+1)}
$$

**LRP-감마(양수를 앞세운 것):**

$$
R_i^{(l)} = \sum_j \frac{a_i (w_{ij} + \gamma w_{ij}^+)}{\sum_{i'} a_{i'} (w_{i'j} + \gamma w_{i'j}^+)} R_j^{(l+1)}
$$

### 섞어 쓰는 전략

깊이마다 다른 규칙을 쓰는 것이 잘 쓰는 길이다.

| 레이어 클래스 | 규칙 | 까닭 |
|------------|------|-----------|
| 입력 레이어 | LRP-zB(매인 것) | 입력 분야의 울타리를 지킨다 |
| 아래쪽 컨볼루션 레이어 | LRP-감마 ($\gamma=0.25$) | 양수 증거를 앞세운다 |
| 위쪽 레이어 | LRP-엡실론 ($\epsilon=0.01$) | 잡음을 누른다 |
| 빽빽한 레이어 | LRP-엡실론 | 쪼개기가 강건하다 |

### 구현

```python
import torch
import torch.nn as nn

class LRP:
    """레이어마다의 쓸모 전파."""

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
        """선형 레이어를 지나는 LRP-엡실론."""
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
- **DeepLIFT**: $(\sigma(10) - \sigma(0)) / (10 - 0) \approx 0.05$(유의하다)

### 구현

```python
from captum.attr import DeepLift, DeepLiftShap

# 베이스라인 하나
deeplift = DeepLift(model)
attribution = deeplift.attribute(
    input_tensor, target=target_class,
    baselines=torch.zeros_like(input_tensor)
)

# 베이스라인 여럿 (DeepLIFT SHAP)
deeplift_shap = DeepLiftShap(model)
attribution = deeplift_shap.attribute(
    input_tensor, target=target_class,
    baselines=baseline_distribution
)
```

---

## 3. 피처 시각화

기여도 분석을 넘어, CNN 시각화에는 뉴런나 레이어마다 어떤 피처를 배웠는지 알아보는 기법도 든다.

### 활성화 최대화

정한 뉴런을 가장 크게 깨우는 입력을 찾는다.

$$
\mathbf{x}^* = \arg\max_{\mathbf{x}} a_k(\mathbf{x}) - \lambda \|\mathbf{x}\|^2
$$

### 필터 그리기

```python
def visualize_filters(model, layer_name):
    """배운 겹치기 필터를 그린다."""
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

| 방법 | 보존 성질 | 계산 | 잦아듦 다루기 | 이론 |
|--------|-------------|-------------|-------------------|--------|
| LRP | 예 | 가운데 | 규칙에 달렸다 | 테일러 쪼개기 |
| DeepLIFT | 예 | 빠름 | 예 | 비교와 맞대기 |
| Integrated Gradients | 예(완전성) | 느림 | 예 | 길 적분 |
| 바닐라 그래디언트 | 아니오 | 아주 빠름 | 아니오 | 그 자리 민감도 |

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
이 절에서 다룬 설명 방법을, XOR 입력을 분류하는 ReLU 활성화의 두 레이어 뉴럴 네트워크에 적용하라. 입력 $x = [1, 1]$에 대한 설명을 계산하여라.

</div>

??? success "풀이"
    웨이트가 $W_1, b_1, W_2, b_2$인 학습된 XOR 네트워크에서 출력은 $f(x) = W_2 \cdot \text{ReLU}(W_1 x + b_1) + b_2$이다. 설명 방법은 입력 피처마다 기여도를 내놓는다. $x = [1, 1]$(클래스 0)이면 두 피처 모두 음수 쪽 분류에 기여한다. 기여도 값은 방법마다 다르다. 그래디언트 기반 방법은 $\partial f / \partial x_i$을 계산하고, 섭동 기반 방법은 피처를 가렸을 때 출력이 얼마나 바뀌는지 잰다. XOR 문제는 결정 경계가 선형이 아니므로 선형 설명 방법이 틀릴 수 있음을 보여 준다. $\square$

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff hard" title="어려움"></span>
이 절의 설명 방법이 완전성 공리를 만족하는지, 곧 어떤 베이스라인 $x_0$에 대해 모든 피처 기여도의 합이 $f(x) - f(x_0)$과 같은지 증명하거나 반례를 들어라.

</div>

??? success "풀이"
    완전성 공리(섀플리 값 이론에서는 효율성이라고도 한다)는 기여도의 합이 입력에서의 모델 출력과 베이스라인에서의 출력의 차이와 같다는 것이다. 이 방법이 완전성을 만족하는지는 그 구성에 달렸다. 그래디언트 방법은 완전성을 만족하지 못한다(그래디언트는 한 점의 국소적인 양이고 경로를 따라 적분한 것이 아니다). Integrated Gradients는 구성 자체로 완전성을 만족한다(경로를 따라 미적분의 기본정리를 쓴다). SHAP 값은 섀플리 공리로 효율성을 만족한다. 완전성을 어기는 방법은 기여도를 너무 크거나 작게 매길 수 있어, 기여도 전체를 전역 설명으로 믿기 어렵게 만든다. $\square$

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
이 방법이 내놓는 설명이 얼마나 믿을 만한지 평가하는 실험을 설계하라. 방법이 중요하다고 짚은 피처가 실제로 모델에 중요한지를 삽입 곡선과 삭제 곡선으로 재어라.

</div>

??? success "풀이"
    절차는 이렇다. (1) 시험 이미지마다 피처 기여도를 계산한다. (2) 삭제: 기여도가 큰 차례로 피처를 하나씩 가리며 모델의 확신도가 떨어지는 모습을 기록한다. 믿을 만한 설명이면 확신도가 빠르게 떨어진다. (3) 삽입: 빈 베이스라인에서 시작해 기여도가 큰 차례로 피처를 하나씩 드러내며 확신도가 오르는 모습을 기록한다. 믿을 만한 설명이면 확신도가 빠르게 오른다. (4) 두 곡선 아래 넓이를 계산한다. (5) 무작위 순서(베이스라인)와 다른 방법에 견준다. 믿을 만한 방법이면 삭제 곡선의 넓이가 작고 삽입 곡선의 넓이가 커야 한다. 통계적으로 믿을 만하려면 시험 표본 1000개 넘게 되풀이한다. $\square$

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
이 설명 방법을 신용 부도를 예측하는 금융 모델에 어떻게 적용할 수 있는지 논하라. 설명이 채워야 할 규제 요건은 무엇인가?

</div>

??? success "풀이"
    신용 모델에는 규제(ECOA, GDPR 22조)가 불리한 결정마다 그 사람에게 맞춘 설명을 요구한다. 방법은 다음을 내놓아야 한다. (1) 거절에 가장 크게 기여한 요인(불리한 처분 사유). (2) 일관된 설명(비슷한 신청자는 비슷한 설명을 받는다). (3) 실행 가능한 설명(신청자가 무엇을 바꾸어야 하는지 안다). 이 절의 설명 방법으로 피처 중요도를 짚을 수 있으나, 강건성(입력이 조금 바뀌었다고 설명이 확 달라지면 안 된다)과 충실성(중요한 피처를 없애면 예측이 바뀌어야 한다)을 평가해야 한다. 보호 속성은 대리 차별이 드러나지 않도록 조심히 다루어야 한다. $\square$

## 정리하며

CNN에 맞춘 쪼개기 방법인 LRP과 DeepLIFT은 보존 성질을 만족하는 기여도 분석을 주어, 추론을 입력 피처에 꼭 맞게 나누어 담는다. 피처 시각화 기법과 아우르면 CNN이 어떻게 판단하는지 두루 들여다볼 수 있다.

**살펴볼 거리**

1. Bach, S., et al. (2015). "On Pixel-wise Explanations for Non-Linear Classifier Decisions by Layer-wise Relevance Propagation." *PLoS ONE*.

2. Shrikumar, A., et al. (2017). "Learning Important Features Through Propagating Activation Differences." *ICML*.

3. Montavon, G., et al. (2019). "Layer-wise Relevance Propagation: An Overview." *Explainable AI*.

4. Olah, C., et al. (2017). "Feature Visualization." *Distill*.
