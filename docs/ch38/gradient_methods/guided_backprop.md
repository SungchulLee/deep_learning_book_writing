# 이끈 되짚기

**이끈 되짚기**는 기울기가 ReLU 살림 함수를 거꾸로 지나는 길을 고쳐 결이 곱고 또렷한 두드러짐 그림을 내는 그림 그리기 재주다. 여느 되짚기와 달리, 이끈 되짚기는 앞으로 걸음에서 깨어 있던 신경 세포를 지나는 양수 기울기만 퍼뜨린다.

슈프링엔베르크 외(2015)가 들여온 이 방법은 미루어 봄에 걸린 잔 결을 짚어 주는, 보기 좋고 촘촘한 그림을 낸다.

---

## 1. 수학 밑바탕

### 여느 ReLU 되짚기 걸음

ReLU 켜를 지나는 여느 되짚기에서는

**앞으로 걸음:**

$$
y = \text{ReLU}(x) = \max(0, x)
$$

**여느 되짚기 걸음:**

$$
\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \cdot \mathbf{1}[x > 0]
$$

### 이끈 되짚기의 고침

이끈 되짚기는 조건을 하나 더 건다. 양수 기울기만 퍼뜨리는 것이다.

$$
\frac{\partial L}{\partial x}\bigg|_{\text{이끈}} = \frac{\partial L}{\partial y} \cdot \mathbf{1}[x > 0] \cdot \mathbf{1}\left[\frac{\partial L}{\partial y} > 0\right]
$$

이는 다음 둘을 아우른다.

1. **앞으로 가리개**: 신경 세포가 깨어 있던 자리만
2. **되짚기 가리개**: 양수 기울기만

---

## 2. PyTorch 짜보기

### 이끈 되짚기를 지닌 손수 만든 ReLU

```python
import torch
import torch.nn as nn
from torch.autograd import Function

class GuidedReLU(Function):
    """이끈 되짚기를 지닌 ReLU."""

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        # 앞으로 가리개: input > 0
        # 되짚기 가리개: grad_output > 0
        return grad_output * (input > 0).float() * (grad_output > 0).float()

class GuidedBackpropagation:
    """이끈 되짚기 짜보기."""

    def __init__(self, model):
        self.model = model
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        """ReLU 앞으로 걸음을 이끈 갈래로 갈음한다."""
        def guided_relu_hook(module, grad_in, grad_out):
            return (torch.clamp(grad_in[0], min=0),)

        for module in self.model.modules():
            if isinstance(module, nn.ReLU):
                hook = module.register_backward_hook(guided_relu_hook)
                self.hooks.append(hook)

    def __call__(self, image_tensor, target_class, device):
        """이끈 되짚기 두드러짐을 셈한다."""
        self.model.eval()
        image_tensor = image_tensor.to(device).requires_grad_(True)

        output = self.model(image_tensor)
        self.model.zero_grad()
        output[0, target_class].backward()

        saliency = image_tensor.grad.abs().max(dim=1)[0]
        return saliency

    def remove_hooks(self):
        """걸어 둔 갈고리를 치운다."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
```

### 쓰는 보기

```python
# 차림
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = models.resnet50(pretrained=True).to(device).eval()

# 이끈 되짚기 것을 만든다
guided_bp = GuidedBackpropagation(model)

# 그림을 부르고 미리 다듬는다
image_tensor = preprocess_image('dog.jpg').unsqueeze(0)

# 미루어 봄을 얻는다
with torch.no_grad():
    pred_class = model(image_tensor.to(device)).argmax(dim=1).item()

# 이끈 되짚기를 셈한다
saliency = guided_bp(image_tensor, pred_class, device)

# 치운다
guided_bp.remove_hooks()
```

---

## 3. 이끈 Grad-CAM

이끈 되짚기와 Grad-CAM을 아우르면 두 세계의 좋은 점을 함께 얻는다.

- **Grad-CAM**: 성기지만 갈래를 가려내는 자리 짚기
- **이끈 되짚기**: 결이 곱고 촘촘한 무늬

$$
\text{이끈 Grad-CAM} = \text{이끈 되짚기} \odot \text{키운}(\text{Grad-CAM})
$$

```python
def compute_guided_gradcam(
    model, target_layer, image_tensor, target_class, device
):
    """이끈 Grad-CAM을 셈한다."""
    # Grad-CAM
    gradcam = GradCAM(model, target_layer)
    cam = gradcam(image_tensor, target_class, device)

    # 들임 크기로 키운다
    cam_upsampled = F.interpolate(
        cam.unsqueeze(0).unsqueeze(0),
        size=image_tensor.shape[2:],
        mode='bilinear'
    ).squeeze()

    # 이끈 되짚기
    guided_bp = GuidedBackpropagation(model)
    guided = guided_bp(image_tensor, target_class, device)
    guided_bp.remove_hooks()

    # 원소마다 곱한다
    guided_gradcam = guided.squeeze() * cam_upsampled

    return guided_gradcam, guided.squeeze(), cam
```

---

## 4. 다른 방법과 견주기

| 방법 | 결 고움 | 갈래 가려냄 | 그림 됨됨이 |
|--------|------------|---------------------|----------------|
| 맨 기울기 | 높음 | 낮음 | 잡음 많음 |
| Grad-CAM | 낮음 | 높음 | 매끄럽되 성김 |
| 이끈 되짚기 | 높음 | 낮음 | 또렷하고 촘촘함 |
| 이끈 Grad-CAM | 높음 | 높음 | 두루 보아 가장 좋음 |

---

## 5. 한계

1. **혼자서는 갈래를 가려내지 못한다**: 갈래가 달라도 비슷한 결이 나온다
2. **제정신인지 살피기에 걸린다**: 참된 몫 매기기가 아니라 가장자리 찾개처럼 굴 수 있다
3. **ReLU에 매인다**: ReLU 살림에만 듣는다

---

## 연습문제

**연습문제 1.**
이 마디에서 밝힌 풀이 방법을, XOR 들임을 가르는 ReLU 살림의 두 켜 신경 그물에 걸어라. 들임 $x = [1, 1]$에 대한 풀이를 셈하여라.

??? success "연습문제 1 풀이"
    짐이 $W_1, b_1, W_2, b_2$인 익힌 XOR 그물에서 내놓기는 $f(x) = W_2 \cdot \text{ReLU}(W_1 x + b_1) + b_2$이다. 풀이 방법은 들임 결마다 몫을 내놓는다. $x = [1, 1]$(갈래 0)이면 두 결 모두 음수 가름에 이바지한다. 몫 값은 방법마다 다르다. 기울기 바탕 방법은 $\partial f / \partial x_i$을 셈하고, 흔들어 보는 방법은 결을 가렸을 때 내놓기가 얼마나 바뀌는지 잰다. XOR 문제는 판단 금이 선형이 아니므로 선형 풀이 방법이 그르칠 수 있음을 보여 준다. $\square$

---

**연습문제 2.**
이 마디의 풀이 방법이 온전함 공리를 채우는지, 곧 어떤 밑금 $x_0$에 대해 모든 결 몫의 합이 $f(x) - f(x_0)$과 같은지 증명하거나 뒤집어라.

??? success "연습문제 2 풀이"
    온전함 공리(섀플리 값 이론에서는 효율이라고도 한다)는 몫의 합이 들임에서의 모형 내놓기와 밑금에서의 내놓기의 차이와 같다는 것이다. 이 방법이 온전함을 채우는지는 그 세움새에 달렸다. 기울기 방법은 온전함을 채우지 못한다(기울기는 그 자리의 것이고 길을 따라 쌓은 것이 아니다). 쌓은 기울기는 세움새 자체로 온전함을 채운다(길을 따라 미적분의 밑정리를 쓴다). SHAP 값은 섀플리 공리로 효율을 채운다. 온전함을 어기는 방법은 몫을 너무 많거나 적게 매길 수 있어, 온 몫을 온 세상 풀이로 믿기 어렵게 만든다. $\square$

---

**연습문제 3.**
이 방법이 내놓는 풀이가 얼마나 미더운지 따지는 시험을 꾸며라. 짚어 준 결이 참으로 모형에 중요한지를 넣기와 빼기 곡선으로 재어라.

??? success "연습문제 3 풀이"
    절차는 이렇다. (1) 시험 그림마다 결 몫을 셈한다. (2) 빼기: 몫이 큰 차례로 결을 하나씩 가리며 모형의 자신함이 떨어지는 모습을 적는다. 미더운 풀이면 자신함이 빠르게 떨어진다. (3) 넣기: 빈 밑금에서 시작해 몫이 큰 차례로 결을 하나씩 드러내며 자신함이 오르는 모습을 적는다. 미더운 풀이면 자신함이 빠르게 오른다. (4) 두 곡선의 아래 넓이를 셈한다. (5) 아무렇게나 매긴 차례(밑금)와 다른 방법에 견준다. 미더운 방법이면 빼기 넓이가 작고 넣기 넓이가 커야 한다. 통계로 미더우려면 시험 표본 1000개 넘게 되풀이한다. $\square$

---

**연습문제 4.**
이 풀이 방법을 신용 부도를 미루어 보는 금융 모형에 어떻게 걸 수 있는지 다루어라. 풀이가 채워야 할 규정 요건은 무엇인가?

??? success "연습문제 4 풀이"
    신용 모형에는 규정(ECOA, GDPR 22조)이 불리한 판단마다 그 사람에게 맞춘 풀이를 바란다. 방법은 다음을 내놓아야 한다. (1) 물리침에 가장 크게 이바지한 인자(불리한 처분 까닭). (2) 한결같은 풀이(비슷한 신청자는 비슷한 풀이를 받는다). (3) 손에 잡히는 풀이(신청자가 무엇을 바꾸어야 하는지 안다). 이 마디의 풀이 방법으로 결의 중요함을 짚을 수 있으나, 든든함(들임이 조금 바뀌었다고 풀이가 확 달라지면 안 된다)과 옳음(중요한 결을 없애면 미루어 봄이 바뀌어야 한다)을 따져 보아야 한다. 지켜야 할 됨됨이는 대리 차별이 드러나지 않도록 조심히 다루어야 한다. $\square$

## 정리하며

이끈 되짚기는 되짚는 동안 음수 기울기를 가려 또렷하고 촘촘한 그림을 낸다. Grad-CAM과 아우르면 이끈 Grad-CAM이 되어 결 고움과 갈래 가려냄을 함께 준다.

**고갱이 식:**

$$
\frac{\partial L}{\partial x}\bigg|_{\text{이끈}} = \frac{\partial L}{\partial y} \cdot \mathbf{1}[x > 0] \cdot \mathbf{1}\left[\frac{\partial L}{\partial y} > 0\right]
$$

**살펴볼 거리**

1. Springenberg, J.T., et al. (2015). *Striving for Simplicity: The All Convolutional Net*. ICLR Workshop.

2. Selvaraju, R.R., et al. (2017). *Grad-CAM: Visual Explanations from Deep Networks*. ICCV.
