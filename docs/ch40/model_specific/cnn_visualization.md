# CNN 시각화와 분해 방법

CNN에 맞춘 설명 방법은 컨볼루션 뉴럴 네트워크의 계층적이고 공간적인 구조를 살려, 모델 무관(model-agnostic) 방법으로는 얻을 수 없는 것을 들여다보게 한다. 이 절은 **레이어별 관련성 전파(LRP)**와 **DeepLIFT**를 다룬다. 둘 다 출력에서 네트워크를 거꾸로 거슬러 관련성을 전파하는 분해 방법이며, 그래디언트 방법에 없는 보존 성질을 만족한다.

---

## 1. 레이어별 관련성 전파(LRP)

### 보존 원리

LRP는 관련성 점수를 레이어마다 거꾸로 전파해 뉴럴 네트워크의 예측을 분해하며, 레이어를 지나도 관련성의 총합이 그대로 남는 보존 성질을 만족한다. 아래 등식이 정확히 성립하는 것은 편향이 없는 네트워크에 LRP-0을 쓸 때다. 편향이 있으면 그 몫의 관련성이 편향으로 새고, LRP-엡실론은 분모의 $\epsilon$만큼 관련성을 흡수하므로 보존이 근사로만 성립한다.

$$
\sum_j R_j^{(l)} = \sum_i R_i^{(l+1)} = \ldots = f(\mathbf{x})
$$

### 전파하는 규칙

**LRP-0(기본 규칙):**

$$
R_i^{(l)} = \sum_j \frac{a_i w_{ij}}{\sum_{i'} a_{i'} w_{i'j}} R_j^{(l+1)}
$$

**LRP-엡실론(분모를 안정화한 것):**

$$
R_i^{(l)} = \sum_j \frac{a_i w_{ij}}{\epsilon + \sum_{i'} a_{i'} w_{i'j}} R_j^{(l+1)}
$$

**LRP-감마(양의 기여를 강조한 것):**

$$
R_i^{(l)} = \sum_j \frac{a_i (w_{ij} + \gamma w_{ij}^+)}{\sum_{i'} a_{i'} (w_{i'j} + \gamma w_{i'j}^+)} R_j^{(l+1)}
$$

### 섞어 쓰는 전략

깊이마다 다른 규칙을 섞어 쓰는 것이 권장되는 방법이다.

| 레이어 종류 | 규칙 | 까닭 |
|------------|------|-----------|
| 입력 레이어 | LRP-zB(경계 제약) | 입력 값의 범위(경계)를 지킨다 |
| 하위 컨볼루션 레이어 | LRP-감마 ($\gamma=0.25$) | 양의 증거를 강조한다 |
| 상위 레이어 | LRP-엡실론 ($\epsilon=0.01$) | 잡음을 누른다 |
| 완전 연결 레이어 | LRP-엡실론 | 분해가 안정적이다 |

### 구현

```python
import torch
import torch.nn as nn

class LRP:
    """레이어별 관련성 전파(Layer-wise Relevance Propagation)."""

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

### 핵심 아이디어

DeepLIFT는 활성화를 참조(reference) 활성화와 비교하여 예측을 설명하며, 포화된 뉴런에서 그래디언트가 사라지는 문제를 다룬다.

$$
\Delta y = f(\mathbf{x}) - f(\mathbf{x}^0) = \sum_i C_{\Delta x_i \Delta y}
$$

### 포화 구간에서의 장점

포화된 지점($x = 10$, $x^0 = 0$)의 시그모이드 뉴런에서

- **그래디언트**: $\sigma'(10) \approx 0.000045$(거의 0)
- **DeepLIFT**: $(\sigma(10) - \sigma(0)) / (10 - 0) \approx 0.05$(0에서 뚜렷이 떨어져 있다)

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

기여도 분석을 넘어, CNN 시각화에는 뉴런이나 레이어마다 어떤 피처를 배웠는지 알아보는 기법도 있다.

### 활성화 최대화

특정 뉴런을 가장 크게 활성화하는 입력을 찾는다.

$$
\mathbf{x}^* = \arg\max_{\mathbf{x}} a_k(\mathbf{x}) - \lambda \|\mathbf{x}\|^2
$$

### 필터 시각화

```python
def visualize_filters(model, layer_name):
    """트레인된 컨볼루션 필터를 그린다."""
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

| 방법 | 보존 성질 | 계산 | 포화 처리 | 이론 |
|--------|-------------|-------------|-------------------|--------|
| LRP | 예 | 중간 | 규칙에 달렸다 | 테일러 분해 |
| DeepLIFT | 예 | 빠름 | 예 | 참조와의 차이 |
| Integrated Gradients | 예(완전성) | 느림 | 예 | 경로 적분 |
| 바닐라 그래디언트 | 아니오 | 아주 빠름 | 아니오 | 국소 민감도 |

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
편향이 없는 ReLU 두 레이어 네트워크 $f(x) = \text{ReLU}(x_1 - x_2) + \text{ReLU}(x_2 - x_1)$은 $\{0,1\}^2$에서 XOR를 정확히 계산한다. 입력 $x = [1, 0]$과 $x = [1, 1]$에서 LRP-0의 관련성과, 베이스라인 $x^0 = [0, 0]$인 DeepLIFT의 기여를 계산하여라.

</div>

??? success "풀이"
    $x = [1, 0]$이면 은닉 활성화가 $a = (1, 0)$이고 $f = 1$이다. 출력 레이어의 웨이트가 둘 다 1이므로 LRP-0은 $R_{a_1} = 1$, $R_{a_2} = 0$을 준다. 첫 은닉 뉴런으로 들어오는 기여는 $x_1 \cdot 1 = 1$, $x_2 \cdot (-1) = 0$이므로 $R_{x_1} = 1$, $R_{x_2} = 0$이고, 합 1이 $f(x)$와 같아 보존이 성립한다. DeepLIFT도 $\Delta a_1 = 1$이 전부 $\Delta x_1$에서 오므로 $C_{\Delta x_1 \Delta y} = 1$, $C_{\Delta x_2 \Delta y} = 0$으로 같은 답을 낸다. $x = [1, 1]$이면 두 은닉 뉴런이 모두 0이고 $f = 0 = f(x^0)$이므로, 두 방법 모두 모든 관련성과 기여를 0으로 준다. 실제로는 두 입력이 서로를 상쇄해 출력이 0이 된 것인데, 보존 성질 때문에 나눌 총량이 0이라 「둘이 함께 있어서 0이 되었다」는 사정이 설명에 드러나지 않는다. 피처마다 점수 하나를 주는 분해 방법이 상호작용 앞에서 보이는 한계다. $\square$

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff hard" title="어려움"></span>
이 절의 설명 방법이 완전성 공리를 만족하는지, 곧 어떤 베이스라인 $x_0$에 대해 모든 피처 기여도의 합이 $f(x) - f(x_0)$과 같은지 증명하거나 반례를 들어라.

</div>

??? success "풀이"
    완전성 공리(섀플리 값 이론에서는 효율성이라고도 한다)는 기여도의 합이 입력에서의 모델 출력과 베이스라인에서의 출력의 차이와 같다는 것이다. 이 방법이 완전성을 만족하는지는 그 구성에 달렸다. 그래디언트 방법은 완전성을 만족하지 못한다(그래디언트는 한 점의 국소적인 양이고 경로를 따라 적분한 것이 아니다). Integrated Gradients는 구성 자체로 완전성을 만족한다(경로를 따라 미적분의 기본정리를 쓴다). SHAP 값은 섀플리 공리로 효율성을 만족한다. 완전성을 어기는 방법은 기여도를 너무 크거나 작게 매길 수 있어, 기여도 전체를 전역 설명으로 믿기 어렵게 만든다. 이 쪽의 두 방법을 보면, DeepLIFT는 기여의 합이 $f(x) - f(x^0)$이 되도록 승수를 정의하므로 구성 자체로 완전성을 만족한다. LRP는 베이스라인이 없이 $f(x)$ 자체를 나누며, 편향이 없는 네트워크에 LRP-0을 쓸 때만 정확히 보존된다. 편향이 있거나 LRP-엡실론을 쓰면 합이 $f(x)$에서 조금 어긋나는 반례가 바로 나온다. $\square$

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
이 방법이 내놓는 설명이 얼마나 믿을 만한지 평가하는 실험을 설계하라. 방법이 중요하다고 짚은 피처가 실제로 모델에 중요한지를 삽입 곡선과 삭제 곡선으로 재어라.

</div>

??? success "풀이"
    절차는 이렇다. (1) 시험 이미지마다 피처 기여도를 계산한다. (2) 삭제: 기여도가 큰 차례로 피처를 하나씩 가리며 모델의 확신도가 떨어지는 모습을 기록한다. 믿을 만한 설명이면 확신도가 빠르게 떨어진다. (3) 삽입: 빈 베이스라인에서 시작해 기여도가 큰 차례로 피처를 하나씩 드러내며 확신도가 오르는 모습을 기록한다. 믿을 만한 설명이면 확신도가 빠르게 오른다. (4) 두 곡선 아래 넓이를 계산한다. (5) 무작위 순서(베이스라인)와 다른 방법과 비교한다. 믿을 만한 방법이면 삭제 곡선의 넓이가 작고 삽입 곡선의 넓이가 커야 한다. 통계적으로 믿을 만하려면 시험 표본 1000개 넘게 되풀이한다. $\square$

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
이 설명 방법을 신용 부도를 예측하는 금융 모델에 어떻게 적용할 수 있는지 논하라. 설명이 채워야 할 규제 요건은 무엇인가?

</div>

??? success "풀이"
    신용 모델에는 규제(ECOA, GDPR 22조)가 불리한 결정마다 그 사람에게 맞춘 설명을 요구한다. 방법은 다음을 내놓아야 한다. (1) 거절에 가장 크게 기여한 요인(불리한 처분 사유). (2) 일관된 설명(비슷한 신청자는 비슷한 설명을 받는다). (3) 실행 가능한 설명(신청자가 무엇을 바꾸어야 하는지 안다). 이 절의 설명 방법으로 피처 중요도를 짚을 수 있으나, 강건성(입력이 조금 바뀌었다고 설명이 확 달라지면 안 된다)과 충실성(중요한 피처를 없애면 예측이 바뀌어야 한다)을 평가해야 한다. 보호 속성은 대리 차별이 드러나지 않도록 조심히 다루어야 한다. $\square$

## 정리하며

CNN에 맞춘 분해 방법인 LRP와 DeepLIFT는 보존 성질을 만족하는 기여도 분석을 주어, 예측값을 입력 피처에 나누어 배분한다. 피처 시각화 기법과 함께 쓰면 CNN이 어떻게 판단하는지 폭넓게 들여다볼 수 있다.

**살펴볼 거리**

1. Bach, S., et al. (2015). "On Pixel-wise Explanations for Non-Linear Classifier Decisions by Layer-wise Relevance Propagation." *PLoS ONE*.

2. Shrikumar, A., et al. (2017). "Learning Important Features Through Propagating Activation Differences." *ICML*.

3. Montavon, G., et al. (2019). "Layer-wise Relevance Propagation: An Overview." *Explainable AI*.

4. Olah, C., et al. (2017). "Feature Visualization." *Distill*.
