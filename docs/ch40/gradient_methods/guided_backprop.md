# Guided Backpropagation

**Guided Backpropagation**은 그래디언트가 ReLU 엑티베이션 함수를 거꾸로 지나는 방식을 바꾸어 세밀하고 선명한 Saliency Map을 내는 시각화 기법이다. Saliency Map은 입력 픽셀마다 예측에 대한 중요도를 칠한 맵이다. 일반 역전파와 달리, Guided Backpropagation은 순전파에서 활성화되었던 뉴런을 지나는 양수 그래디언트만 전파한다.

슈프링엔베르크 외(2015)가 내놓은 이 방법은 예측에 관여한 세밀한 피처를 짚어 주는, 보기 좋고 선명한 맵을 낸다.

---

## 1. 수학 기초

### 일반 ReLU 역전파

ReLU 레이어를 지나는 일반 역전파에서는

**순전파:**

$$
y = \text{ReLU}(x) = \max(0, x)
$$

**일반 역전파:**

$$
\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \cdot \mathbf{1}[x > 0]
$$

### Guided Backpropagation의 수정

Guided Backpropagation은 조건을 하나 더 건다. 양수 그래디언트만 전파하는 것이다.

$$
\frac{\partial L}{\partial x}\bigg|_{\text{guided}} = \frac{\partial L}{\partial y} \cdot \mathbf{1}[x > 0] \cdot \mathbf{1}\left[\frac{\partial L}{\partial y} > 0\right]
$$

이는 다음 두 마스크를 합친 것이다.

1. **순전파 마스크**: 순전파에서 뉴런이 활성화된 자리만
2. **역전파 마스크**: 양수 그래디언트만

---

## 2. PyTorch 구현

### Guided Backpropagation을 적용한 사용자 정의 ReLU

```python
import torch
import torch.nn as nn
from torch.autograd import Function

class GuidedReLU(Function):
    """Guided Backpropagation을 적용한 ReLU."""

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        # 순전파 마스크: input > 0
        # 역전파 마스크: grad_output > 0
        return grad_output * (input > 0).float() * (grad_output > 0).float()

class GuidedBackpropagation:
    """Guided Backpropagation 구현."""

    def __init__(self, model):
        self.model = model
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        """ReLU 레이어마다 역전파 훅을 걸어 음수 그래디언트를 잘라 낸다."""
        def guided_relu_hook(module, grad_in, grad_out):
            return (torch.clamp(grad_in[0], min=0),)

        for module in self.model.modules():
            if isinstance(module, nn.ReLU):
                hook = module.register_backward_hook(guided_relu_hook)
                self.hooks.append(hook)

    def __call__(self, image_tensor, target_class, device):
        """Guided Backpropagation 샐리언시를 계산한다."""
        self.model.eval()
        image_tensor = image_tensor.to(device).requires_grad_(True)

        output = self.model(image_tensor)
        self.model.zero_grad()
        output[0, target_class].backward()

        saliency = image_tensor.grad.abs().max(dim=1)[0]
        return saliency

    def remove_hooks(self):
        """걸어 둔 훅을 제거한다."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
```

### 사용 예

```python
# 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = models.resnet50(pretrained=True).to(device).eval()

# Guided Backpropagation 객체를 만든다
guided_bp = GuidedBackpropagation(model)

# 이미지를 불러와 전처리한다
image_tensor = preprocess_image('dog.jpg').unsqueeze(0)

# 예측 클래스를 얻는다
with torch.no_grad():
    pred_class = model(image_tensor.to(device)).argmax(dim=1).item()

# Guided Backpropagation을 계산한다
saliency = guided_bp(image_tensor, pred_class, device)

# 훅을 제거한다
guided_bp.remove_hooks()
```

---

## 3. Guided Grad-CAM

Guided Backpropagation과 Grad-CAM을 결합하면 두 방법의 장점을 함께 얻는다.

- **Grad-CAM**: 해상도는 낮지만 클래스를 구별하는 위치 파악
- **Guided Backpropagation**: 해상도가 높고 세밀한 패턴

$$
\text{Guided Grad-CAM} = \text{Guided Backpropagation} \odot \text{Upsample}(\text{Grad-CAM})
$$

```python
def compute_guided_gradcam(
    model, target_layer, image_tensor, target_class, device
):
    """Guided Grad-CAM을 계산한다."""
    # Grad-CAM
    gradcam = GradCAM(model, target_layer)
    cam = gradcam(image_tensor, target_class, device)

    # 입력 크기로 업샘플링한다
    cam_upsampled = F.interpolate(
        cam.unsqueeze(0).unsqueeze(0),
        size=image_tensor.shape[2:],
        mode='bilinear'
    ).squeeze()

    # Guided Backpropagation
    guided_bp = GuidedBackpropagation(model)
    guided = guided_bp(image_tensor, target_class, device)
    guided_bp.remove_hooks()

    # 원소마다 곱한다
    guided_gradcam = guided.squeeze() * cam_upsampled

    return guided_gradcam, guided.squeeze(), cam
```

---

## 4. 다른 방법과 비교

| 방법 | 해상도 | 클래스 판별력 | 시각적 품질 |
|--------|------------|---------------------|----------------|
| 바닐라 그래디언트 | 높음 | 낮음 | 잡음 많음 |
| Grad-CAM | 낮음 | 높음 | 매끄럽지만 거침 |
| Guided Backpropagation | 높음 | 낮음 | 선명하고 세밀함 |
| Guided Grad-CAM | 높음 | 높음 | 전반적으로 가장 좋음 |

---

## 5. 한계

1. **혼자서는 클래스를 구별하지 못한다**: 클래스가 달라도 비슷한 맵이 나온다
2. **새니티 체크를 통과하지 못한다**: 참된 기여도 분석이 아니라 에지 검출기처럼 동작할 수 있다
3. **ReLU에 묶인다**: ReLU 엑티베이션에만 작동한다

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
이 절에서 다룬 설명 방법을, XOR 입력을 분류하는 ReLU 엑티베이션의 두 레이어 뉴럴 네트워크에 적용하라. 입력 $x = [1, 1]$에 대한 설명을 계산하여라.

</div>

??? success "풀이"
    웨이트가 $W_1, b_1, W_2, b_2$인 트레인된 XOR 네트워크에서 출력은 $f(x) = W_2 \cdot \text{ReLU}(W_1 x + b_1) + b_2$이다. 설명 방법은 입력 피처마다 기여도를 내놓는다. $x = [1, 1]$(클래스 0)이면 두 피처 모두 음수 쪽 분류에 기여한다. 기여도 값은 방법마다 다르다. 그래디언트 기반 방법은 $\partial f / \partial x_i$을 계산하고, 섭동 기반 방법은 피처를 가렸을 때 출력이 얼마나 바뀌는지 잰다. XOR 문제는 결정 경계가 선형이 아니므로 선형 설명 방법이 틀릴 수 있음을 보여 준다. $\square$

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff hard" title="어려움"></span>
이 절의 설명 방법이 완전성 공리를 만족하는지, 곧 어떤 베이스라인 $x_0$에 대해 모든 피처 기여도의 합이 $f(x) - f(x_0)$과 같은지 증명하거나 반례를 들어라.

</div>

??? success "풀이"
    완전성 공리(섀플리 값 이론에서는 효율성이라고도 한다)는 기여도의 합이 입력에서의 모델 출력과 베이스라인에서의 출력의 차이와 같다는 것이다. 이 방법이 완전성을 만족하는지는 그 구성에 달렸다. 그래디언트 방법은 완전성을 만족하지 못한다(그래디언트는 한 점의 국소적인 양이고 경로를 따라 적분한 것이 아니다). Integrated Gradients는 구성 자체로 완전성을 만족한다(경로를 따라 미적분의 기본정리를 쓴다). SHAP 값은 섀플리 공리로 효율성을 만족한다. Guided Backpropagation은 한 점에서 음수 그래디언트까지 잘라 낸 값이라 경로 적분도 아니므로 완전성을 만족하지 못한다. 완전성을 어기는 방법은 기여도를 너무 크거나 작게 매길 수 있어, 기여도 전체를 전역 설명으로 믿기 어렵게 만든다. $\square$

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

Guided Backpropagation은 역전파하는 동안 음수 그래디언트를 막아 선명하고 세밀한 맵을 낸다. Grad-CAM과 결합하면 Guided Grad-CAM이 되어 높은 해상도와 클래스 판별력을 함께 준다.

**핵심 식:**

$$
\frac{\partial L}{\partial x}\bigg|_{\text{guided}} = \frac{\partial L}{\partial y} \cdot \mathbf{1}[x > 0] \cdot \mathbf{1}\left[\frac{\partial L}{\partial y} > 0\right]
$$

**참고 문헌**

1. Springenberg, J.T., et al. (2015). *Striving for Simplicity: The All Convolutional Net*. ICLR Workshop.

2. Selvaraju, R.R., et al. (2017). *Grad-CAM: Visual Explanations from Deep Networks*. ICCV.
