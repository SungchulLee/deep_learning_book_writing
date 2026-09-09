# Grad-CAM++: 나아진 눈 풀이

**Grad-CAM++**은 Grad-CAM을 낫게 한 것으로, 특히 같은 갈래의 물체가 그림에 여럿 있을 때 자리를 더 잘 짚는다. Grad-CAM이 기울기를 온 세상에 걸쳐 고르게 모아 중요함 짐을 셈하는 데 견주어, Grad-CAM++은 갈래 점수에 양수로 크게 이바지하는 그림점에 더 큰 중요함을 주는 **짐 실은 아우름**을 쓴다.

차토파디야이 외(2018)가 들여온 Grad-CAM++은 물체가 여럿이거나 물체가 그림의 작은 자리만 차지할 때 처음 Grad-CAM이 지닌 고갱이 걸림돌을 다룬다.

---

## 1. 왜 있어야 하는가

### Grad-CAM의 한계

Grad-CAM은 온 세상 고르게 모으기로 중요함 짐을 셈한다.

$$
\alpha_k^c = \frac{1}{Z} \sum_i \sum_j \frac{\partial y^c}{\partial A_{ij}^k}
$$

이 길에는 걸림돌이 여럿 있다.

1. **물체가 여럿일 때**: 같은 갈래의 물체가 여럿 있으면 고르게 하면서 물체마다의 신호가 묽어진다
2. **덜 덮음**: 걸린 자리를 다 덮지 못하고 가장 눈에 띄는 물체만 짚을 수 있다
3. **똑같이 다룸**: 참으로 얼마나 이바지하든 모든 자리를 똑같이 헤아린다
4. **작은 물체**: 작은 자리를 차지하는 물체는 짐이 모자라게 매겨질 수 있다

### Grad-CAM++이 푸는 길

Grad-CAM++은 한결같이 고르게 하는 대신 **그림점마다의 짐**을 써서 이를 다룬다.

$$
w_k^c = \sum_i \sum_j \alpha_{ij}^{kc} \cdot \text{ReLU}\left(\frac{\partial y^c}{\partial A_{ij}^k}\right)
$$

여기서 $\alpha_{ij}^{kc}$은 자리마다의 **견준 중요함**을 담아내는 그림점마다의 짐이다.

---

## 2. 수학 밑바탕

### 이차 기울기로 이끌어 내기

Grad-CAM++은 이차와 삼차 편미분으로 그림점마다의 짐을 이끌어 낸다. 갈래 점수를 결 그림의 짐 실은 합으로 보는 데서 시작한다.

$$
y^c = \sum_k w_k^c \sum_i \sum_j A_{ij}^k
$$

여기서 $w_k^c$은 갈래 $c$에 대한 결 그림 $k$의 중요함이다.

잇달아 편미분하면

$$
\frac{\partial y^c}{\partial A_{ij}^k} = w_k^c
$$

$$
\frac{\partial^2 y^c}{\partial (A_{ij}^k)^2} = \frac{\partial w_k^c}{\partial A_{ij}^k}
$$

### 그림점마다의 짐 셈하기

그림점마다의 짐은 이렇게 셈한다.

$$
\alpha_{ij}^{kc} = \frac{\frac{\partial^2 y^c}{(\partial A_{ij}^k)^2}}{2 \cdot \frac{\partial^2 y^c}{(\partial A_{ij}^k)^2} + \sum_{a,b} A_{ab}^k \cdot \frac{\partial^3 y^c}{(\partial A_{ij}^k)^3}}
$$

### 참으로 쓰는 단순한 셈

삼차 미분을 또렷이 셈하는 것은 비싸므로, 기울기의 거듭제곱을 바탕으로 단순하게 줄인 꼴을 쓴다.

$$
\alpha_{ij}^{kc} = \frac{(g_{ij}^{kc})^2}{2(g_{ij}^{kc})^2 + \sum_{a,b} A_{ab}^k \cdot (g_{ij}^{kc})^3 + \epsilon}
$$

여기서

- $g_{ij}^{kc} = \frac{\partial y^c}{\partial A_{ij}^k}$은 일차 기울기
- $(g_{ij}^{kc})^2$은 원소마다의 제곱(이차 미분을 어림함)
- $(g_{ij}^{kc})^3$은 원소마다의 세제곱(삼차 미분을 어림함)
- $\epsilon$은 셈이 흔들리지 않게 하는 작은 붙박이 수

### 마지막 열 그림 세움새

Grad-CAM++ 열 그림은 이렇다.

$$
L^c_{\text{Grad-CAM++}} = \text{ReLU}\left(\sum_k w_k^c A^k\right)
$$

여기서 통로 짐이 그림점마다의 중요함을 담는다.

$$
w_k^c = \sum_i \sum_j \alpha_{ij}^{kc} \cdot \text{ReLU}\left(\frac{\partial y^c}{\partial A_{ij}^k}\right)
$$

**고갱이 깨침**: 짐을 매기기 앞서 기울기에 ReLU을 걸므로, Grad-CAM++은 갈래 점수에 **양수로 이바지하는** 그림점에만 눈길을 둔다.

---

## 3. 알고리즘

```
알고리즘: Grad-CAM++
들임: 그림 I, CNN 모형 f, 겨눈 갈래 c, 겨눈 켜 l
내놓기: 열 그림 L_GradCAM++

1. 앞으로 걸음: 켜 l의 살아남 A^k을 셈한다
2. 앞으로 걸음: 갈래 점수 y^c(또는 소프트맥스 S^c)을 셈한다
3. 되짚기 걸음: 기울기 g = ∂y^c/∂A^k을 셈한다
4. 기울기의 거듭제곱을 셈한다:
   g² = g ⊙ g (원소마다의 제곱)
   g³ = g ⊙ g ⊙ g (원소마다의 세제곱)
5. 결 그림 k마다:
   a. 자리 합을 셈한다: sum_A = Σ_{a,b} A^k_{ab}
   b. 그림점마다의 짐을 셈한다:
      α_{ij}^{kc} = g²_{ij} / (2·g²_{ij} + sum_A · g³_{ij} + ε)
   c. 통로 짐을 셈한다:
      w_k^c = Σ_{i,j} α_{ij}^{kc} · ReLU(g_{ij}^{kc})
6. 열 그림을 셈한다: L = ReLU(Σ_k w_k^c · A^k)
7. 고르게 하고 들임 결 고움으로 키운다
8. L을 내놓는다
```

---

## 4. PyTorch 짜보기

### 온전한 GradCAMPlusPlus 클래스

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class GradCAMPlusPlus:
    """
    CNN 그림 그리기를 낫게 하는 Grad-CAM++ 짜보기.

    Grad-CAM보다 자리를 잘 짚는데 특히 다음에서 그렇다:
    - 같은 갈래의 물체가 여럿일 때
    - 물체가 작을 때
    - 얼마쯤 가려졌을 때

    살펴볼 거리: Chattopadhyay et al., "Grad-CAM++: Improved Visual Explanations
    for Deep Convolutional Networks" (WACV 2018)
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        """
        Args:
            model: PyTorch CNN 모형
            target_layer: 그릴 겨눈 겹치는 켜
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # 갈고리를 건다
        self._register_hooks()

    def _register_hooks(self):
        """겨눈 켜에 앞으로 걸음과 되짚기 갈고리를 건다."""

        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def __call__(
        self, 
        input_tensor: torch.Tensor, 
        target_class: int = None,
        device: torch.device = None
    ) -> torch.Tensor:
        """
        Grad-CAM++ 열 그림을 만든다.

        Args:
            input_tensor: 들임 그림 텐서 (1, C, H, W)
            target_class: 겨눈 갈래 번호. None이면 미루어 본 갈래를 쓴다.
            device: 셈하는 장치

        Returns:
            [0, 1]으로 고르게 한 열 그림 텐서 [H, W]
        """
        if device is None:
            device = next(self.model.parameters()).device

        self.model.eval()
        input_tensor = input_tensor.to(device)

        # 앞으로 걸음
        output = self.model(input_tensor)

        # 겨눈 갈래를 정한다
        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # 기울기가 더 잘 굴러가도록 소프트맥스 점수를 쓴다
        score = F.softmax(output, dim=1)[0, target_class]

        # 되짚기 걸음
        self.model.zero_grad()
        score.backward(retain_graph=True)

        # 살아남과 기울기를 집는다
        A = self.activations  # [1, K, H', W']
        g = self.gradients    # [1, K, H', W']

        # 기울기의 거듭제곱을 셈한다
        g_squared = g.pow(2)  # 이차 어림
        g_cubed = g.pow(3)    # 삼차 어림

        # (A * g³)을 자리 축으로 더한다
        # 곧: Σ_{a,b} A^k_{ab} · (g^k_{ab})³
        sum_Ag3 = (A * g_cubed).sum(dim=(2, 3), keepdim=True)  # [1, K, 1, 1]

        # 그림점마다의 알파 짐을 셈한다
        # α_{ij}^{kc} = g² / (2·g² + Σ(A·g³) + ε)
        denominator = 2 * g_squared + sum_Ag3 + 1e-8

        # 셈이 흔들리지 않게 다룬다
        denominator = torch.where(
            denominator != 0,
            denominator,
            torch.ones_like(denominator)
        )

        alpha = g_squared / denominator  # [1, K, H', W']

        # 통로 짐을 셈한다: w_k = Σ_{i,j} α_{ij} · ReLU(g_{ij})
        positive_gradients = F.relu(g)
        weights = (alpha * positive_gradients).sum(dim=(2, 3), keepdim=True)  # [1, K, 1, 1]

        # 살아남 그림을 짐 실어 아우른다
        heatmap = (weights * A).sum(dim=1, keepdim=True)  # [1, 1, H', W']

        # ReLU을 건다(양수 이바지에만 눈길)
        heatmap = F.relu(heatmap)

        # [0, 1]으로 고르게 한다
        heatmap = heatmap - heatmap.min()
        heatmap = heatmap / (heatmap.max() + 1e-8)

        # 들임 결 고움으로 키운다
        heatmap = F.interpolate(
            heatmap,
            size=input_tensor.shape[2:],
            mode='bilinear',
            align_corners=False
        )

        return heatmap.squeeze()  # [H, W]

    def generate_cam(
        self, 
        input_tensor: torch.Tensor, 
        target_class: int = None
    ) -> np.ndarray:
        """
        Grad-CAM++ 열 그림을 numpy 배열로 만든다.

        Args:
            input_tensor: 들임 그림 텐서 (1, C, H, W)
            target_class: 겨눈 갈래 번호

        Returns:
            numpy 배열 열 그림 [H', W'] (결 그림 결 고움)
        """
        device = next(self.model.parameters()).device
        heatmap = self(input_tensor.to(device), target_class, device)
        return heatmap.cpu().numpy()

    def generate_visualization(
        self,
        input_tensor: torch.Tensor,
        original_image: np.ndarray = None,
        target_class: int = None,
        alpha: float = 0.4
    ) -> np.ndarray:
        """
        열 그림을 겹쳐 그린 그림을 만든다.

        Args:
            input_tensor: 들임 그림 텐서
            original_image: [0, 255] 너비의 numpy 배열 (H, W, 3) 본디 그림
            target_class: 겨눈 갈래 번호
            alpha: 겹침의 비침 정도

        Returns:
            [0, 255] 너비의 numpy 배열 (H, W, 3) 그림
        """
        import cv2

        cam = self.generate_cam(input_tensor, target_class)

        if original_image is not None:
            h, w = original_image.shape[:2]
        else:
            h, w = input_tensor.shape[2:]

        cam_resized = cv2.resize(cam, (w, h))
        heatmap = cv2.applyColorMap(
            np.uint8(255 * cam_resized), 
            cv2.COLORMAP_JET
        )
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        if original_image is not None:
            if original_image.max() <= 1.0:
                original_image = (original_image * 255).astype(np.uint8)
            visualization = cv2.addWeighted(
                original_image, 1 - alpha, heatmap, alpha, 0
            )
        else:
            visualization = heatmap

        return visualization
```

---

## 5. 견주기: Grad-CAM 대 Grad-CAM++

### 이론에서의 차이

| 결 | Grad-CAM | Grad-CAM++ |
|--------|----------|------------|
| 짐 셈하기 | 온 세상 고르게 모으기 | 그림점마다 짐 실은 합 |
| 기울기 쓰임 | 일차만 | 일차, 이차, 삼차 |
| 물체가 여럿일 때 | 똑같이 이바지(묽어짐) | 센 살아남에 더 큰 짐 |
| 작은 물체 | 놓치거나 짐이 모자람 | 자리를 더 잘 짚음 |
| 셈 값 | 낮음 | 높음(기울기 거듭제곱) |
| 셈의 흔들림 | 더 든든함 | 조심히 다루어야 함 |

### 눈으로 견주기

```python
import torch
import torchvision.models as models
import matplotlib.pyplot as plt
import numpy as np

def compare_gradcam_methods(model, input_tensor, target_class=None):
    """
    Grad-CAM과 Grad-CAM++ 그림을 나란히 견준다.
    """
    target_layer = model.layer4[-1]

    # Grad-CAM
    gradcam = GradCAM(model, target_layer)
    cam_gc = gradcam.generate_cam(input_tensor, target_class)

    # Grad-CAM++
    gradcam_pp = GradCAMPlusPlus(model, target_layer)
    cam_gcpp = gradcam_pp.generate_cam(input_tensor, target_class)

    # 견줌을 그린다
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 보여 주려고 들임을 되돌려 고른다
    img = input_tensor[0].permute(1, 2, 0).cpu().numpy()
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = img * std + mean
    img = np.clip(img, 0, 1)

    axes[0].imshow(img)
    axes[0].set_title('본디 그림', fontsize=12)
    axes[0].axis('off')

    axes[1].imshow(cam_gc, cmap='jet')
    axes[1].set_title('Grad-CAM', fontsize=12)
    axes[1].axis('off')

    axes[2].imshow(cam_gcpp, cmap='jet')
    axes[2].set_title('Grad-CAM++', fontsize=12)
    axes[2].axis('off')

    plt.tight_layout()
    return fig

# 쓰임
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
model.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# 제 그림 텐서를 부른다
input_tensor = torch.randn(1, 3, 224, 224).to(device)
fig = compare_gradcam_methods(model, input_tensor, target_class=281)  # 얼룩 고양이
```

---

## 6. 물체가 여럿일 때의 자리 짚기

### Grad-CAM++이 나은 까닭

한 그림에 같은 갈래의 물체가 여럿 있을 때

1. **Grad-CAM**: 온 세상 고르게 모으기가 모든 자리를 똑같이 다루므로, 가장 눈에 띄는 물체만 짚거나 모든 물체에 걸쳐 고르게 되어 신호가 묽어지기 쉽다

2. **Grad-CAM++**: 그림점마다의 짐 $\alpha_{ij}^{kc}$이 양수 기울기가 센 그림점에 더 큰 중요함을 주므로 물체 하나하나를 더 잘 붙든다

### 보이기

```python
def multi_instance_demonstration(model, image_tensor, target_class):
    """
    물체가 여럿인 그림에서 Grad-CAM++이 나음을 보인다.
    """
    target_layer = model.layer4[-1]

    # Grad-CAM
    gc = GradCAM(model, target_layer)
    cam_gc = gc.generate_cam(image_tensor, target_class)

    # Grad-CAM++
    gcpp = GradCAMPlusPlus(model, target_layer)
    cam_gcpp = gcpp.generate_cam(image_tensor, target_class)

    # 문턱을 달리하며 덮는 만큼을 잰다
    thresholds = [0.3, 0.5, 0.7]

    print("덮는 만큼 살피기(그림에서 짚어 준 몫):")
    print("-" * 50)
    for thresh in thresholds:
        coverage_gc = (cam_gc > thresh).sum() / cam_gc.size
        coverage_gcpp = (cam_gcpp > thresh).sum() / cam_gcpp.size
        print(f"문턱 {thresh}: Grad-CAM={coverage_gc:.2%}, Grad-CAM++={coverage_gcpp:.2%}")

    # 물체가 여럿이면 Grad-CAM++이 흔히 더 잘 덮는다
    return cam_gc, cam_gcpp
```

---

## 7. 수로 따지기

### 짚기 놀이 자

짚기 놀이는 가장 큰 살아남이 참값 자리 안에 드는지를 따진다.

```python
def pointing_game(cam: np.ndarray, bbox: tuple) -> int:
    """
    가장 큰 살아남이 테두리 상자 안에 드는지 따진다.

    Args:
        cam: [0, 1]으로 고르게 한 갈래 살아남 그림 (H, W)
        bbox: 테두리 상자 [x_min, y_min, x_max, y_max]

    Returns:
        가장 큰 점이 상자 안이면 1, 아니면 0
    """
    max_idx = np.unravel_index(cam.argmax(), cam.shape)
    y, x = max_idx

    x_min, y_min, x_max, y_max = bbox
    if x_min <= x <= x_max and y_min <= y <= y_max:
        return 1
    return 0

def pointing_game_evaluation(model, dataloader, method='gradcam++'):
    """
    자료 전체에 걸쳐 짚기 놀이 맞음률을 따진다.

    Args:
        model: CNN 모형
        dataloader: 그림과 테두리 상자를 지닌 DataLoader
        method: 'gradcam'이나 'gradcam++'

    Returns:
        짚기 놀이 맞음률
    """
    target_layer = model.layer4[-1]

    if method == 'gradcam++':
        cam_method = GradCAMPlusPlus(model, target_layer)
    else:
        cam_method = GradCAM(model, target_layer)

    hits = 0
    total = 0

    for images, labels, bboxes in dataloader:
        for img, label, bbox in zip(images, labels, bboxes):
            cam = cam_method.generate_cam(img.unsqueeze(0), label.item())
            hits += pointing_game(cam, bbox)
            total += 1

    accuracy = hits / total
    return accuracy
```

### 고른 떨어짐 / 오름 자

이 자들은 풀이가 얼마나 미더운지를 잰다.

```python
def average_drop_increase(
    model: nn.Module,
    input_tensor: torch.Tensor,
    cam: np.ndarray,
    target_class: int,
    device: torch.device
) -> tuple:
    """
    고른 떨어짐과 고른 오름 자를 셈한다.

    고른 떨어짐: 짚어 주지 않은 자리를 가릴 때 자신함이
                 얼마나 떨어지는가?
    고른 오름: 짚어 준 자리만 남길 때 자신함이 오르는 일이
               얼마나 잦은가?

    Args:
        model: 가름개 모형
        input_tensor: 본디 들임
        cam: 갈래 살아남 그림
        target_class: 겨눈 갈래 번호
        device: 셈하는 장치

    Returns:
        (고른 떨어짐, 고른 오름)
    """
    import cv2

    model.eval()
    input_tensor = input_tensor.to(device)

    # 본디 미루어 봄
    with torch.no_grad():
        orig_output = model(input_tensor)
        orig_conf = F.softmax(orig_output, dim=1)[0, target_class].item()

    # 가린 들임을 만든다(짚어 준 자리만 남긴다)
    h, w = input_tensor.shape[2:]
    cam_resized = cv2.resize(cam, (w, h))
    mask = torch.tensor(cam_resized, device=device).unsqueeze(0).unsqueeze(0)
    masked_input = input_tensor * mask

    # 가린 미루어 봄
    with torch.no_grad():
        masked_output = model(masked_input)
        masked_conf = F.softmax(masked_output, dim=1)[0, target_class].item()

    # 자를 셈한다
    # 고른 떨어짐: (본디 - 가림) / 본디, [0, 1]으로 자른다
    drop = max(0, orig_conf - masked_conf) / (orig_conf + 1e-8)

    # 고른 오름: masked_conf > orig_conf이면 1, 아니면 0
    increase = 1 if masked_conf > orig_conf else 0

    return drop, increase

def evaluate_faithfulness(model, dataloader, cam_method, device):
    """
    자료 전체에 걸쳐 미더움 자를 따진다.
    """
    total_drop = 0
    total_increase = 0
    n_samples = 0

    for images, labels in dataloader:
        for img, label in zip(images, labels):
            img = img.unsqueeze(0).to(device)
            cam = cam_method.generate_cam(img, label.item())

            drop, increase = average_drop_increase(
                model, img, cam, label.item(), device
            )

            total_drop += drop
            total_increase += increase
            n_samples += 1

    avg_drop = total_drop / n_samples
    avg_increase = total_increase / n_samples

    print(f"고른 떨어짐: {avg_drop:.2%} (작을수록 좋다)")
    print(f"고른 오름: {avg_increase:.2%} (작을수록 좋다)")

    return avg_drop, avg_increase
```

---

## 8. 어느 것을 언제 쓸까

### Grad-CAM을 쓸 때:
- 그림마다 물체가 하나인 것이 흔할 때
- 셈이 잘 들어야 할 때
- 빠르게 벌레잡고 둘러볼 때
- 밑금으로 견주어야 할 때

### Grad-CAM++을 쓸 때:
- 같은 갈래의 물체가 그림에 **여럿** 있을 때
- **작은 물체**의 자리를 짚어야 할 때
- 걸린 자리를 **더 잘 덮어야** 할 때
- 촘촘히 살피려고 더 또렷해야 할 때
- 학술로 따질 때(짚기 놀이 등)

### 판단 표

| 자리 | 권하는 방법 | 까닭 |
|----------|-------------------|--------|
| 물체 하나가 판칠 때 | Grad-CAM | 넉넉하고 빠르다 |
| 같은 갈래 물체가 여럿 | Grad-CAM++ | 더 잘 덮는다 |
| 작은 물체 | Grad-CAM++ | 자리를 더 잘 짚는다 |
| 제때 도는 쓰임 | Grad-CAM | 늦음이 적다 |
| 연구/논문 | Grad-CAM++ | 가장 앞선 것 |
| 빠른 벌레잡기 | Grad-CAM | 되풀이가 빠르다 |

---

## 9. 한계

### 1. 셈 값이 더 크다

기울기의 거듭제곱(제곱, 세제곱)을 셈하느라 Grad-CAM보다 짐이 붙는다.

- Grad-CAM보다 ~1.5~2곱절 느리다
- 기울기 거듭제곱을 담느라 기억을 더 쓴다

### 2. 셈이 흔들릴 수 있다

$\alpha_{ij}^{kc}$을 셈할 때의 나눗셈이 흔들릴 수 있다.

- 아랫자리가 작으면 셈에 탈이 난다
- 엡실론을 조심히 다루어야 한다
- 기울기가 작은 자리에서 자국이 생길 수 있다

### 3. 얻는 것이 줄어든다

다음에서는 Grad-CAM보다 나아지는 만큼이 적다.

- 물체가 하나인 그림
- 물체가 화면을 가득 채운 그림
- 여러 갈래가 잘 갈라져 있는 자리

### 4. Grad-CAM과 같은 밑바탕 한계

- 여전히 겨눈 켜의 결 고움에 매인다(성긴 자리 짚기)
- 그림점 낱으로 또렷할 수 없다
- 겨눈 흔듦에 약하다

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

Grad-CAM++은 높은 차수의 기울기 소식에서 이끌어 낸 **그림점마다의 중요함 짐**을 써서 Grad-CAM을 낫게 하며, 물체가 여럿이거나 작을 때 자리를 더 잘 짚는다.

### 고갱이 식

**그림점마다의 짐:**

$$
\alpha_{ij}^{kc} = \frac{(g_{ij}^{kc})^2}{2(g_{ij}^{kc})^2 + \sum_{a,b} A_{ab}^k \cdot (g_{ij}^{kc})^3 + \epsilon}
$$

**통로 짐:**

$$
w_k^c = \sum_{i,j} \alpha_{ij}^{kc} \cdot \text{ReLU}(g_{ij}^{kc})
$$

**마지막 열 그림:**

$$
L^c_{\text{Grad-CAM++}} = \text{ReLU}\left(\sum_k w_k^c A^k\right)
$$

### Grad-CAM보다 나아진 것

| 결 | 나아진 만큼 |
|--------|-------------|
| 물체가 여럿일 때의 자리 짚기 | 크게 낫다 |
| 작은 물체 찾기 | 더 잘 덮는다 |
| 이론 밑바탕 | 이차/삼차 기울기 |
| 짚기 놀이 맞음률 | 점수가 높다 |

### 잘 쓰는 길

1. Grad-CAM이 시원찮은 **물체 여럿 자리에 쓴다**
2. 알맞은 엡실론 값으로 **셈이 흔들리지 않게 다룬다**
3. 제 자리에서 참으로 나아지는지 **Grad-CAM과 견준다**
4. 짚기 놀이와 미더움 자로 **수로 따진다**
5. 제때 도는 쓰임에서는 **셈 값의 맞바꿈을 헤아린다**

**살펴볼 거리**

1. Chattopadhyay, A., Sarkar, A., Howlader, P., & Balasubramanian, V. N. (2018). "Grad-CAM++: Improved Visual Explanations for Deep Convolutional Networks." *WACV 2018*.

2. Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra, D. (2017). "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization." *ICCV 2017*.

3. Zhang, J., Bargal, S. A., Lin, Z., Brandt, J., Shen, X., & Sclaroff, S. (2018). "Top-Down Neural Attention by Excitation Backprop." *International Journal of Computer Vision*.
