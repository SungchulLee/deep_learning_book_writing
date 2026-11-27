# 03: SmoothGrad

이 쪽은 Saliency Map의 잡음을 줄이는 SmoothGrad를 직접 구현한다. 잡음 섞은 입력 여러 개에서 구한 그래디언트를 평균 내면, 입력이 조금만 바뀌어도 크게 흔들리는 그래디언트 성분은 상쇄되고 일관된 신호만 남는다. 이론은 [SmoothGrad](smoothgrad.md) 쪽에서 다룬다.

## 1. 코드

```python
"""
03: SmoothGrad - 샐리언시 맵의 잡음 줄이기
=================================================

난이도: 낮음

설명:
SmoothGrad는 입력에 잡음을 섞은 것들에서 계산한 그래디언트를 평균해
샐리언시 맵의 눈에 띄는 잡음을 줄인다. 잡음을 더해 평균하면
날카롭고 흔들리는 그래디언트가 서로 상쇄되고 의미 있는 신호만 남는다.

수학적 배경:
    SmoothGrad(x) = (1/n) Σᵢ₌₁ⁿ |∂y_c/∂(x + N(0, σ²))|

여기서:
- n: 잡음 섞은 표본의 수
- N(0, σ²): 표준편차가 σ인 가우스 잡음
- 평균이 그래디언트 잡음을 완화한다

주요 파라메터:
- n (num_samples): 표본이 많을수록 매끄럽지만 느리다(흔히 20~50)
- σ (noise_level): 잡음의 크기를 조절한다(흔히 입력 범위의 0.1~0.2)

지은이: 교육용 자료
"""

import torch
import torch.nn as nn
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
# 무작위로 뽑는 값이 아래에 나온다. 시드를 고정해야 이 쪽에 실린
# 수가 다시 나온다 — 고정하지 않으면 돌릴 때마다 다른 수가 찍힌다
torch.manual_seed(0)
np.random.seed(0)

# ========================================================================
# 도우미 함수
# ========================================================================

# ImageNet으로 트레인한 모델이 쓰는 값. 전처리할 때와 되돌릴 때 같은 값을 써야 한다.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_device():
    """쓸 수 있는 가장 빠른 장치를 고른다."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_pretrained_model(model_name="resnet50", device=None, eval_mode=True):
    """torchvision에서 사전 트레인된 모델을 불러온다.

    다운로드가 안 되면(네트워크가 없을 때) 트레인하지 않은 모델로 대체한다. 그래디언트
    방법을 보이는 데에는 가중치가 실제로 트레인된 것이 아니어도 맵이 나온다.
    """
    from torchvision import models

    device = device or get_device()
    try:
        model = getattr(models, model_name)(weights="DEFAULT")
    except Exception:
        print(f"  사전 트레인 가중치를 받지 못했다 — {model_name}을 무작위 초기화 상태로 쓴다.")
        model = getattr(models, model_name)(weights=None)

    model = model.to(device)
    if eval_mode:
        model.eval()
    return model


def preprocess_image(image, size=224, device=None, requires_grad=False):
    """이미지를 모델이 받는 (1, 3, size, size) 텐서로 전처리한다.

    인수:
        image: 파일 경로이거나 PIL 이미지
        size: 자를 크기
        device: 텐서를 올릴 장치
        requires_grad: True면 입력에 대한 그래디언트를 받을 수 있게 한다

    파일이 없으면 회색 이미지로 대체한다. 예제가 이미지 하나 때문에
    통째로 멈추지 않게 하기 위해서다.
    """
    from torchvision import transforms

    device = device or get_device()

    if isinstance(image, (str, os.PathLike)):
        try:
            image = Image.open(image).convert("RGB")
        except FileNotFoundError:
            print(f"  '{image}' 파일을 찾지 못했다 — 단색 회색 이미지로 대신한다.")
            image = Image.new("RGB", (size, size), (128, 128, 128))
    elif not isinstance(image, Image.Image):
        raise TypeError("image는 파일 경로이거나 PIL 이미지여야 한다")

    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    tensor = transform(image).unsqueeze(0).to(device)
    return tensor.requires_grad_(requires_grad)


def denormalize(image_tensor):
    """정규화한 텐서를 [0,1]로 되돌려 시각화할 수 있게 만든다."""
    t = image_tensor.detach().cpu()
    if t.dim() == 4:
        t = t[0]
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (t * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()


def _as_map(saliency):
    """샐리언시를 (H, W) 넘파이 배열로 맞추고 [0,1]로 정규화한다."""
    s = saliency.detach().cpu() if torch.is_tensor(saliency) else torch.as_tensor(saliency)
    while s.dim() > 2:
        s = s.squeeze(0) if s.shape[0] == 1 else s.max(0).values
    s = s.numpy().astype(np.float32)
    lo, hi = s.min(), s.max()
    return (s - lo) / (hi - lo) if hi > lo else np.zeros_like(s)


def visualize_multiple_saliencies(image_tensor, saliencies, colormap="jet",
                                  alpha=0.4, save_path=None):
    """여러 방법의 샐리언시를 한 줄에 늘어놓아 비교한다.

    인수:
        saliencies: {이름: 샐리언시} 딕셔너리. 이름은 표준 출력으로 알린다
    """
    img = denormalize(image_tensor)
    names = list(saliencies)
    fig, axes = plt.subplots(1, len(names) + 1, figsize=(4 * (len(names) + 1), 4))
    axes[0].imshow(img); axes[0].set_title("Input"); axes[0].axis("off")

    for ax, name in zip(axes[1:], names):
        sal = _as_map(saliencies[name])
        if sal.shape != img.shape[:2]:
            sal = np.array(Image.fromarray((sal * 255).astype(np.uint8))
                           .resize((img.shape[1], img.shape[0]))) / 255.0
        ax.imshow(img)
        ax.imshow(sal, cmap=colormap, alpha=alpha)
        ax.axis("off")
    print("  [그림] " + " | ".join(n.replace("\n", " ") for n in names))
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def create_output_dir(path="outputs"):
    """출력 이미지를 저장할 디렉터리를 만들고 그 경로를 돌려준다."""
    os.makedirs(path, exist_ok=True)
    return path


# ========================================================================
# 메인
# ========================================================================

def compute_smoothgrad(
    model: nn.Module,
    image_tensor: torch.Tensor,
    target_class: int,
    device: torch.device,
    num_samples: int = 25,
    noise_level: float = 0.15
) -> torch.Tensor:
    """
    SmoothGrad 샐리언시 맵을 계산한다.

    알고리즘:
    1. i = 1에서 n까지:
       a. 입력에 가우스 잡음을 더한다: x_noisy = x + N(0, σ²)
       b. 그래디언트를 계산한다: gᵢ = |∂y_c/∂x_noisy|
    2. 그래디언트를 평균한다: SmoothGrad = (1/n) Σᵢ gᵢ

    왜 효과가 있는가:
    - 날카롭고 흔들리는 그래디언트는 잡음에 따라 무작위로 달라진다
    - 의미 있는 그래디언트는 일관되게 남는다
    - 평균이 잡음을 상쇄하고 신호를 남긴다
    - 머신러닝의 앙상블 방법과 결이 같다

    Args:
        model: 사전 트레인된 모델
        image_tensor: 입력 이미지 [1, 3, H, W]
        target_class: 대상 클래스 번호
        device: 계산 장치
        num_samples: 잡음 섞은 표본의 수
        noise_level: 가우스 잡음의 표준편차(입력 범위에 대한 비율)

    Returns:
        torch.Tensor: 매끄럽게 한 샐리언시 맵 [1, H, W]
    """
    model.eval()
    image_tensor = image_tensor.to(device)

    # 그래디언트를 누적할 곳
    accumulated_gradients = torch.zeros_like(image_tensor)

    # 잡음의 표준편차를 계산한다
    # 흔한 입력 범위(정규화한 뒤)에 맞추어 잡는다
    noise_std = noise_level

    for i in range(num_samples):
        # 잡음 섞은 입력을 만든다
        noise = torch.randn_like(image_tensor) * noise_std
        noisy_image = image_tensor + noise
        noisy_image.requires_grad = True

        # 순전파 스텝
        output = model(noisy_image)
        target_score = output[0, target_class]

        # 역전파 스텝
        model.zero_grad()
        target_score.backward()

        # 그래디언트를 누적한다
        accumulated_gradients += noisy_image.grad

    # 그래디언트를 평균한다
    mean_gradients = accumulated_gradients / num_samples

    # 절댓값을 취하고 채널 방향으로 모은다
    abs_gradients = torch.abs(mean_gradients)
    saliency = torch.max(abs_gradients, dim=1)[0]

    return saliency


def example_1_smoothgrad_vs_vanilla():
    """SmoothGrad와 바닐라 그래디언트를 비교한다."""
    print("\n" + "="*60)
    print("예제 1: SmoothGrad 대 바닐라 그래디언트")
    print("="*60)

    device = get_device()
    create_output_dir('outputs')
    model = load_pretrained_model('resnet50', device)

    # 잡음 많은 테스트 이미지를 만든다
    img_array = np.random.randint(50, 200, (224, 224, 3), dtype=np.uint8)
    test_image = Image.fromarray(img_array)
    image_tensor = preprocess_image(test_image, requires_grad=True)

    with torch.no_grad():
        output = model(image_tensor.to(device))
        target_class = output.argmax(dim=1).item()

    print(f"\n대상 클래스: {target_class}")
    print("바닐라 그래디언트를 계산한다...")

    # 바닐라 그래디언트
    image_vanilla = preprocess_image(test_image, requires_grad=True)
    output = model(image_vanilla.to(device))
    output[0, target_class].backward()
    vanilla = torch.max(torch.abs(image_vanilla.grad), dim=1)[0]

    print("SmoothGrad를 계산한다(더 오래 걸린다)...")
    # SmoothGrad
    image_smooth = preprocess_image(test_image, requires_grad=False)
    smoothgrad = compute_smoothgrad(
        model, image_smooth, target_class, device,
        num_samples=25, noise_level=0.15
    )

    # 시각화한다
    visualize_multiple_saliencies(
        image_tensor,
        {
            '바닐라 그래디언트\n(잡음 많음)': vanilla,
            'SmoothGrad\n(깨끗함)': smoothgrad
        },
        save_path='outputs/03_smoothgrad_comparison.png'
    )

    print("\n✓ SmoothGrad가 더 깨끗한 맵을 낸다!")


def example_2_parameter_sensitivity():
    """num_samples와 noise_level이 미치는 영향을 살핀다."""
    print("\n" + "="*60)
    print("예제 2: 파라메터 민감도")
    print("="*60)

    device = get_device()
    model = load_pretrained_model('resnet50', device)

    test_image = Image.new('RGB', (224, 224), color=(120, 150, 180))
    image_tensor = preprocess_image(test_image, requires_grad=False)

    with torch.no_grad():
        output = model(image_tensor.to(device))
        target_class = output.argmax(dim=1).item()

    # 파라메터를 달리해 시험한다
    configs = [
        (10, 0.10, "n=10, σ=0.10"),
        (25, 0.15, "n=25, σ=0.15"),
        (50, 0.20, "n=50, σ=0.20"),
    ]

    saliencies = {}
    for num_samples, noise_level, label in configs:
        print(f"{label} 계산하는 중...")
        sal = compute_smoothgrad(
            model, image_tensor, target_class, device,
            num_samples, noise_level
        )
        saliencies[label] = sal

    visualize_multiple_saliencies(
        image_tensor, saliencies,
        save_path='outputs/03_parameter_comparison.png'
    )

    print("\n눈여겨볼 것:")
    print("- 표본이 많을수록 매끄럽지만 느리다")
    print("- 잡음이 클수록 더 매끄러워진다")
    print("- 세밀한 디테일과 깨끗함 사이의 트레이드오프")
    print("\n✓ 흔한 설정: n=25, σ=0.15")


def main():
    print("\n" + "="*70)
    print(" "*20 + "SMOOTHGRAD 튜토리얼")
    print("="*70)

    try:
        example_1_smoothgrad_vs_vanilla()
        example_2_parameter_sensitivity()

        print("\n" + "="*70)
        print("핵심:")
        print("1. SmoothGrad는 평균해서 눈에 띄는 잡음을 줄인다")
        print("2. 잡음을 더했는데 되레 잡음이 준다!")
        print("3. 파라메터: n=20~50, σ=0.10~0.20")
        print("4. 계산 비용이 크다(순전파/역전파를 n번 한다)")
        print("\n다음: 꾸러미 04 - Integrated Gradients")
        print("="*70)
    except Exception as e:
        print(f"오류: {e}")

if __name__ == "__main__":
    main()
```

**출력:**

```

======================================================================
                    SMOOTHGRAD 튜토리얼
======================================================================

============================================================
예제 1: SmoothGrad 대 바닐라 그래디언트
============================================================

대상 클래스: 21
바닐라 그래디언트를 계산한다...
SmoothGrad를 계산한다(더 오래 걸린다)...
  [그림] 바닐라 그래디언트 (잡음 많음) | SmoothGrad (깨끗함)

✓ SmoothGrad가 더 깨끗한 맵을 낸다!

============================================================
예제 2: 파라메터 민감도
============================================================
n=10, σ=0.10 계산하는 중...
n=25, σ=0.15 계산하는 중...
n=50, σ=0.20 계산하는 중...
  [그림] n=10, σ=0.10 | n=25, σ=0.15 | n=50, σ=0.20

눈여겨볼 것:
- 표본이 많을수록 매끄럽지만 느리다
- 잡음이 클수록 더 매끄러워진다
- 세밀한 디테일과 깨끗함 사이의 트레이드오프

✓ 흔한 설정: n=25, σ=0.15

======================================================================
핵심:
1. SmoothGrad는 평균해서 눈에 띄는 잡음을 줄인다
2. 잡음을 더했는데 되레 잡음이 준다!
3. 파라메터: n=20~50, σ=0.10~0.20
4. 계산 비용이 크다(순전파/역전파를 n번 한다)

다음: 꾸러미 04 - Integrated Gradients
======================================================================
```

## 2. 논의

`compute_smoothgrad`는 입력에 표준편차 `noise_level`인 가우스 잡음을 더한 표본 `num_samples`개에서 그래디언트를 구해 평균 낸 뒤, 절댓값을 잡고 채널 방향 최댓값으로 집계한다. 이론 쪽 구현과 달리 잡음의 표준편차를 입력 값 범위에 비례시키지 않고 `noise_level`을 그대로 쓴다. ImageNet 방식으로 정규화한 입력은 대략 −2.1~2.6 범위에 있으므로, 0.15는 그 범위의 3% 남짓이다. 또 독스트링의 식은 절댓값을 먼저 잡고 평균 내는 형태로 적혀 있지만, 코드는 평균을 먼저 내고 절댓값을 잡는다. 두 순서는 결과가 다르다.

위 출력에서 보듯 프로그램은 두 예제를 끝까지 돈다. 다만 출력에 찍히는 것은 진행 문구와 미리 적어 둔 결론뿐이고, 비교 그림은 `outputs/` 에 파일로 저장될 뿐 이 쪽에 실리지 않는다. 입력도 합성 이미지라 「SmoothGrad가 더 깨끗한 맵을 낸다」는 마지막 줄은 측정 결과가 아니라 고정 문구다. 깨끗함을 확인하려면 저장된 그림을 직접 보거나 두 맵의 잡음을 수로 재어야 한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
코드를 읽고 핵심 설계 선택을 짚어라. 구현에서 고른 것 셋을 들고, 저마다 왜 그래디언트 기반 설명에 알맞은지 설명하라.

</div>

??? success "풀이"
    설계 선택은 구현마다 다르나 흔히 이런 것이 있다. (1) 엑티베이션 함수 선택 -- ReLU 계열은 그래디언트가 소실되지 않아 트레인이 빠르다. (2) 정규화 전략 -- 배치 정규화가 내부 공변량 이동을 줄여 트레인을 안정시킨다. (3) 잔차 연결 -- 있으면 스킵 경로를 주어 깊은 네트워크에서 그래디언트가 흐르게 한다. 각 선택은 표현력, 계산 비용, 트레인 안정성 사이의 트레이드오프를 드러낸다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
입력의 형태와 데이터 타입이 기대한 대로인지 확인하는 입력 검증을 주요 함수나 클래스에 더하여라. 올바르지 않은 입력에는 알아보기 쉬운 오류 메시지를 띄워라.

</div>

??? success "풀이"
    `forward` 메서드(또는 해당 함수)의 앞부분에 `assert x.dim() == expected_dims, f'Expected {expected_dims}D input, got {x.dim()}D'`이나 `assert x.dtype == torch.float32, f'Expected float32, got {x.dtype}'` 같은 검사를 더한다. 형태를 검사하려면 중요한 차원을 본다. `B, C, H, W = x.shape; assert C == self.expected_channels`. 알아보기 쉬운 오류 메시지는 디버깅을 크게 앞당기고 코드를 재사용하기 안전하게 한다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
이 구현이 실패할 만한 유형 둘을 들고, 저마다 어떻게 탐지하고 고칠지 설명하라.

</div>

??? success "풀이"
    흔한 실패 유형은 이렇다. (1) **그래디언트가 사라지거나 터짐** -- 그래디언트 크기를 모니터링해 탐지한다(`torch.nn.utils.clip_grad_norm_`이나 레이어마다 `param.grad.norm()` 기록). 그래디언트 클리핑, 더 나은 초기화(Xavier/Kaiming), 구조 수정(잔차 연결, 정규화 레이어)으로 고친다. (2) **과적합** -- 트레인 로스는 줄어드는데 검증 로스가 오르면 탐지한다. 정칙화(드롭아웃, 웨이트 감쇠, 데이터 증강)나 모델 축소로 고친다. 트레인 지표와 검증 지표를 늘 함께 모니터링해 이를 일찍 잡아야 한다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
03: SmoothGrad 구현을 검사하는 포괄적인 테스트 함수를 작성하라. 빈 입력, 원소 하나짜리 입력, 아주 큰 입력, 그리고 극단값(0, 아주 큰 수)이 든 입력 같은 경계 사례를 테스트하여라.

</div>

??? success "풀이"
    경계 조건을 두루 다루는 테스트 함수를 작성한다.
    ```python
    def test_03: smoothgrad():
        model = 03: SmoothGrad(...)
        # 일반 입력
        assert model(normal_input).shape == expected_shape
        # 원소 하나짜리 배치
        assert model(single_input).shape == (1, ...)
        # 큰 값(오버플로를 확인한다)
        out = model(torch.ones(...) * 1000)
        assert torch.isfinite(out).all()
        # 그래디언트 흐름
        out = model(normal_input)
        out.sum().backward()
        for p in model.parameters():
            assert p.grad is not None
    ```
    구조가 엔드투엔드 트레인을 지원하는지 알려면 그래디언트 흐름을 테스트하는 것이 특히 중요하다.

## 정리하며

**다룬 것** — 03: SmoothGrad

이 쪽은 SmoothGrad를 PyTorch로 직접 구현했다. 실린 출력은 끝까지 돈 기록이지만 그림은 파일로만 저장되고, 결론 문구는 측정값이 아니다.

앞의 연습문제 4개로 스스로 점검해 볼 수 있다.
