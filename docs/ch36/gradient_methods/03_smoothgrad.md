# 03: SmoothGrad

03: SmoothGrad - 두드러짐 그림의 잡음 줄이기. 밝힘:

신경 그물이 무엇을 배우는지 아는 일은 믿음을 쌓고 모형의 벌레를 잡는 데 종요롭다. 이 꾸러미는 모형이 들임을 어떻게 다루고 어떻게 판단하는지 드러내는 기울기 바탕 풀이 재주를 보이며, 그물의 움직임을 눈으로 보고 수로 재게 해 준다.

## 코드

```python
"""
03: SmoothGrad - 두드러짐 그림의 잡음 줄이기
=================================================

어려움: 낮음

밝힘:
SmoothGrad은 들임에 잡음을 섞은 것들에서 셈한 기울기를 고르게 해
두드러짐 그림의 눈에 띄는 잡음을 줄인다. 잡음을 더해 고르게 하면
날카롭고 흔들리는 기울기가 서로 지워지고 뜻있는 신호만 남는다.

수학 밑바탕:
    SmoothGrad(x) = (1/n) Σᵢ₌₁ⁿ |∂y_c/∂(x + N(0, σ²))|

여기서:
- n: 잡음 섞은 표본의 수
- N(0, σ²): 잣대 벗어남이 σ인 가우스 잡음
- 고르게 하기가 기울기 잡음을 눅인다

으뜸 매개변수:
- n (num_samples): 표본이 많을수록 매끄럽지만 느리다(흔히 20~50)
- σ (noise_level): 잡음의 크기를 다스린다(흔히 들임 자의 0.1~0.2)

지은이: 가르치기 몫
"""

import torch
import torch.nn as nn
import numpy as np
from utils import *
from PIL import Image

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
    SmoothGrad 두드러짐 그림을 셈한다.

    알고리즘:
    1. i = 1에서 n까지:
       a. 들임에 가우스 잡음을 더한다: x_noisy = x + N(0, σ²)
       b. 기울기를 셈한다: gᵢ = |∂y_c/∂x_noisy|
    2. 기울기를 고르게 한다: SmoothGrad = (1/n) Σᵢ gᵢ

    왜 듣는가:
    - 날카롭고 흔들리는 기울기는 잡음에 따라 아무렇게나 달라진다
    - 뜻있는 기울기는 한결같이 남는다
    - 고르게 하기가 잡음을 지우고 신호를 남긴다
    - 기계 배움의 모둠 방법과 결이 같다

    Args:
        model: 미리 익힌 모형
        image_tensor: 들임 그림 [1, 3, H, W]
        target_class: 겨눈 갈래 번호
        device: 셈하는 장치
        num_samples: 잡음 섞은 표본의 수
        noise_level: 가우스 잡음의 잣대 벗어남(들임 자에 대한 몫)

    Returns:
        torch.Tensor: 매끄럽게 한 두드러짐 그림 [1, H, W]
    """
    model.eval()
    image_tensor = image_tensor.to(device)

    # 기울기를 쌓아 둘 곳
    accumulated_gradients = torch.zeros_like(image_tensor)

    # 잡음의 잣대 벗어남을 셈한다
    # 흔한 들임 너비(고르게 한 뒤)에 맞추어 잡는다
    noise_std = noise_level

    for i in range(num_samples):
        # 잡음 섞은 들임을 만든다
        noise = torch.randn_like(image_tensor) * noise_std
        noisy_image = image_tensor + noise
        noisy_image.requires_grad = True

        # 앞으로 걸음
        output = model(noisy_image)
        target_score = output[0, target_class]

        # 되짚기 걸음
        model.zero_grad()
        target_score.backward()

        # 기울기를 쌓는다
        accumulated_gradients += noisy_image.grad

    # 기울기를 고르게 한다
    mean_gradients = accumulated_gradients / num_samples

    # 절댓값을 잡고 통로를 가로질러 모은다
    abs_gradients = torch.abs(mean_gradients)
    saliency = torch.max(abs_gradients, dim=1)[0]

    return saliency


def example_1_smoothgrad_vs_vanilla():
    """SmoothGrad과 맨 기울기를 견준다."""
    print("\n" + "="*60)
    print("보기 1: SmoothGrad 대 맨 기울기")
    print("="*60)

    device = get_device()
    create_output_dir('outputs')
    model = load_pretrained_model('resnet50', device)

    # 잡음 많은 시험 그림을 만든다
    img_array = np.random.randint(50, 200, (224, 224, 3), dtype=np.uint8)
    test_image = Image.fromarray(img_array)
    image_tensor = preprocess_image(test_image, requires_grad=True)

    with torch.no_grad():
        output = model(image_tensor.to(device))
        target_class = output.argmax(dim=1).item()

    print(f"\n겨눈 갈래: {target_class}")
    print("맨 기울기를 셈한다...")

    # 맨 기울기
    image_vanilla = preprocess_image(test_image, requires_grad=True)
    output = model(image_vanilla.to(device))
    output[0, target_class].backward()
    vanilla = torch.max(torch.abs(image_vanilla.grad), dim=1)[0]

    print("SmoothGrad을 셈한다(더 오래 걸린다)...")
    # SmoothGrad
    image_smooth = preprocess_image(test_image, requires_grad=False)
    smoothgrad = compute_smoothgrad(
        model, image_smooth, target_class, device,
        num_samples=25, noise_level=0.15
    )

    # 그림으로 보인다
    visualize_multiple_saliencies(
        image_tensor,
        {
            '맨 기울기\n(잡음 많음)': vanilla,
            'SmoothGrad\n(깨끗함)': smoothgrad
        },
        save_path='outputs/03_smoothgrad_comparison.png'
    )

    print("\n✓ SmoothGrad이 더 깨끗한 그림을 낸다!")


def example_2_parameter_sensitivity():
    """num_samples과 noise_level이 미치는 힘을 살핀다."""
    print("\n" + "="*60)
    print("보기 2: 매개변수에 대한 예민함")
    print("="*60)

    device = get_device()
    model = load_pretrained_model('resnet50', device)

    test_image = Image.new('RGB', (224, 224), color=(120, 150, 180))
    image_tensor = preprocess_image(test_image, requires_grad=False)

    with torch.no_grad():
        output = model(image_tensor.to(device))
        target_class = output.argmax(dim=1).item()

    # 매개변수를 달리해 시험한다
    configs = [
        (10, 0.10, "n=10, σ=0.10"),
        (25, 0.15, "n=25, σ=0.15"),
        (50, 0.20, "n=50, σ=0.20"),
    ]

    saliencies = {}
    for num_samples, noise_level, label in configs:
        print(f"{label} 셈하는 중...")
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
    print("- 잔 무늬와 깨끗함 사이의 맞바꿈")
    print("\n✓ 흔한 차림: n=25, σ=0.15")


def main():
    print("\n" + "="*70)
    print(" "*20 + "SMOOTHGRAD 익히기")
    print("="*70)

    try:
        example_1_smoothgrad_vs_vanilla()
        example_2_parameter_sensitivity()

        print("\n" + "="*70)
        print("고갱이:")
        print("1. SmoothGrad은 고르게 해서 눈에 띄는 잡음을 줄인다")
        print("2. 잡음을 더했는데 되레 잡음이 준다!")
        print("3. 매개변수: n=20~50, σ=0.10~0.20")
        print("4. 셈이 비싸다(앞으로/되짚기를 n번 한다)")
        print("\n다음: 꾸러미 04 - 쌓은 기울기")
        print("="*70)
    except Exception as e:
        print(f"어긋남: {e}")

if __name__ == "__main__":
    main()```

## 논의

이 짜보기는 깔끔하고 읽기 쉬운 PyTorch 코드로 기울기 바탕 풀이의 고갱이가 되는 생각을 보여 준다. 조각으로 나눈 얼개 덕에 부분마다 따로 살피고 다른 일이나 자료에 맞추어 고치기 쉽다.

여기서 보인 결은 더 까다로운 자리로도 자연스레 넓혀진다. 하이퍼파라미터, 얼개의 갈래, 여러 자료를 바꿔 가며 해 보면 이해가 깊어지고 모형 풀이하기에 대한 감이 몸에 붙는다.

## 익힘 문제

**익힘 1.**
코드를 읽고 고갱이가 되는 설계 판단을 짚어라. 짜기에서 고른 것 셋을 들고, 저마다 왜 기울기 바탕 풀이에 알맞은지 밝혀라.

??? success "익힘 1 풀이"
    설계 판단은 짜보기마다 다르나 흔히 이런 것이 있다. (1) 살림 함수 고르기 -- ReLU 갈래는 기울기가 잦아들지 않아 익히기가 빠르다. (2) 고르게 하는 꾀 -- 묶음 고르게 하기가 안쪽 함께 바뀌는 옮겨감을 줄여 익힘을 든든하게 한다. (3) 나머지 이음 -- 있으면 건너뛰는 길을 주어 깊은 그물에서 기울기가 흐르게 한다. 고른 것마다 나타내는 힘, 셈 값, 익힘의 든든함 사이의 맞바꿈을 드러낸다.

---

**익힘 2.**
들임의 꼴과 자료 갈래가 바라는 대로인지 살피는 들임 살피기를 으뜸 함수나 클래스에 더하여라. 올바르지 않은 들임에는 알아듣기 쉬운 어긋남 알림을 띄워라.

??? success "익힘 2 풀이"
    `forward` 방법(또는 알맞은 함수)의 첫머리에 `assert x.dim() == expected_dims, f'Expected {expected_dims}D input, got {x.dim()}D'`이나 `assert x.dtype == torch.float32, f'Expected float32, got {x.dtype}'` 같은 살핌을 더한다. 꼴을 살피려면 종요로운 차원을 본다. `B, C, H, W = x.shape; assert C == self.expected_channels`. 알아듣기 쉬운 어긋남 알림은 벌레잡기를 크게 앞당기고 코드를 되쓰기 든든하게 한다.

---

**익힘 3.**
이 짜보기가 무너질 만한 결 둘을 밝히고, 저마다 어떻게 짚어내고 고칠지 밝혀라.

??? success "익힘 3 풀이"
    흔히 무너지는 결은 이렇다. (1) **기울기가 사라지거나 터짐** -- 기울기 크기를 지켜보아 짚어낸다(`torch.nn.utils.clip_grad_norm_`이나 켜마다 `param.grad.norm()` 적기). 기울기 자르기, 더 나은 첫값 잡기(Xavier/Kaiming), 얼개 고치기(나머지 이음, 고르게 하기)로 고친다. (2) **지나치게 맞추기** -- 익힘 잃음은 줄어드는데 살핌 잃음이 오르면 짚어낸다. 정칙화(드롭아웃, 짐 줄이기, 자료 늘리기)나 모형 크기 줄이기로 고친다. 익힘과 살핌 자를 늘 함께 지켜보아 이를 일찍 잡아야 한다.

---

**익힘 4.**
03: SmoothGrad 짜보기를 살피는 두루 갖춘 시험 함수를 써라. 빈 들임, 원소 하나짜리 들임, 아주 큰 들임, 그리고 끝자락 값(0, 아주 큰 수)이 든 들임 같은 가장자리 자리를 시험하여라.

??? success "익힘 4 풀이"
    금 언저리 조건을 두루 건드리는 시험 함수를 짓는다.
    ```python
    def test_03: smoothgrad():
        model = 03: SmoothGrad(...)
        # 여느 들임
        assert model(normal_input).shape == expected_shape
        # 원소 하나짜리 묶음
        assert model(single_input).shape == (1, ...)
        # 큰 값(넘침을 살핀다)
        out = model(torch.ones(...) * 1000)
        assert torch.isfinite(out).all()
        # 기울기 흐름
        out = model(normal_input)
        out.sum().backward()
        for p in model.parameters():
            assert p.grad is not None
    ```
    얼개가 끝에서 끝까지 익히기를 받치는지 알려면 기울기 흐름을 시험하는 것이 특히 중요하다.
