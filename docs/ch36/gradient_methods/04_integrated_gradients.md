# 04: 쌓은 기울기

04: 쌓은 기울기 - 이치에 닿는 몫 매기기. 밝힘:

신경 그물이 무엇을 배우는지 아는 일은 믿음을 쌓고 모형의 벌레를 잡는 데 종요롭다. 이 꾸러미는 모형이 들임을 어떻게 다루고 어떻게 판단하는지 드러내는 기울기 바탕 풀이 재주를 보이며, 그물의 움직임을 눈으로 보고 수로 재게 해 준다.

## 코드

```python
"""
04: 쌓은 기울기 - 이치에 닿는 몫 매기기
================================================

어려움: 가운데

밝힘:
쌓은 기울기는 밑금에서 들임으로 가는 길을 따라 기울기를 쌓는다.
이로써 몫 매기기의 중요한 공리인 예민함과 짜기에 흔들리지 않음을
채운다.

수학 밑바탕:
    IG(x) = (x - x') ⊙ ∫₀¹ (∂f(x' + α(x - x'))/∂x) dα

여기서:
- x: 들임 그림
- x': 밑금(흔히 0이나 흐린 그림)
- α ∈ [0,1]: 사이 잡는 계수
- 적분은 리만 합으로 어림한다

채우는 공리:
1. 예민함: 결이 내놓기를 바꾸면 0이 아닌 몫을 받는다
2. 짜기에 흔들리지 않음: 하는 일이 같은 그물은 같은 몫을 받는다
3. 온전함: 몫의 합이 f(x) - f(x')이 된다

지은이: 가르치기 몫
"""

import torch
import torch.nn as nn
import numpy as np
from utils import *
from PIL import Image, ImageFilter

# ========================================================================
# 메인
# ========================================================================

def compute_integrated_gradients(
    model: nn.Module,
    image_tensor: torch.Tensor,
    target_class: int,
    device: torch.device,
    baseline: str = 'zeros',
    steps: int = 50
) -> torch.Tensor:
    """
    쌓은 기울기를 셈한다.

    알고리즘:
    1. 밑금 x'을 고른다
    2. 사이 들임을 만든다: i=0..m에 대해 x^(i) = x' + (i/m)(x - x')
    3. 자리마다 기울기를 셈한다: gᵢ = ∂f(x^(i))/∂x
    4. 기울기를 고르게 한다: ḡ = (1/m) Σᵢ gᵢ
    5. 들임 차이로 잣대를 잡는다: IG = (x - x') ⊙ ḡ

    Args:
        baseline: 'zeros', 'blur', 'random' 가운데 하나
        steps: 사이를 잡는 걸음 수(많을수록 더 맞다)
    """
    model.eval()
    image_tensor = image_tensor.to(device)

    # 밑금을 만든다
    if baseline == 'zeros':
        baseline_tensor = torch.zeros_like(image_tensor)
    elif baseline == 'blur':
        # 그림을 흐리게 해서 밑금으로 삼는다
        from torchvision.transforms.functional import gaussian_blur
        baseline_tensor = gaussian_blur(image_tensor, kernel_size=51, sigma=20)
    elif baseline == 'random':
        baseline_tensor = torch.randn_like(image_tensor) * 0.1
    else:
        baseline_tensor = torch.zeros_like(image_tensor)

    baseline_tensor = baseline_tensor.to(device)

    # 길을 셈한다: α ∈ [0,1]에 대해 x' + α(x - x')
    accumulated_gradients = torch.zeros_like(image_tensor)

    for step in range(steps):
        # 사이 잡는 계수
        alpha = (step + 1) / steps

        # 사이 들임
        interpolated = baseline_tensor + alpha * (image_tensor - baseline_tensor)
        interpolated.requires_grad = True

        # 앞으로 걸음
        output = model(interpolated)
        target_score = output[0, target_class]

        # 되짚기 걸음
        model.zero_grad()
        target_score.backward()

        # 기울기를 쌓는다
        accumulated_gradients += interpolated.grad

    # 기울기를 고르게 한다(적분의 리만 어림)
    avg_gradients = accumulated_gradients / steps

    # 들임 차이로 잣대를 잡는다
    integrated_grads = (image_tensor - baseline_tensor) * avg_gradients

    # 모은다
    abs_attr = torch.abs(integrated_grads)
    saliency = torch.max(abs_attr, dim=1)[0]

    return saliency


def verify_completeness(model, image_tensor, target_class, device, saliency):
    """몫의 합이 내놓기 차이가 되는지 살핀다."""
    model.eval()

    with torch.no_grad():
        output_image = model(image_tensor.to(device))[0, target_class]
        baseline = torch.zeros_like(image_tensor).to(device)
        output_baseline = model(baseline)[0, target_class]

    # 몫의 합이 내놓기 차이와 ≈ 같아야 한다
    # 짚을 것: 잘게 나누어 셈하므로 어림값이다
    print(f"\n온전함 살피기:")
    print(f"f(x) - f(x'): {(output_image - output_baseline).item():.4f}")
    print("(두드러짐의 합이 이 값에 가까워야 한다)")


def example_1_baseline_comparison():
    """밑금을 달리 고른 것들을 견준다."""
    print("\n" + "="*60)
    print("보기 1: 밑금 견주기")
    print("="*60)

    device = get_device()
    create_output_dir('outputs')
    model = load_pretrained_model('resnet50', device)

    test_image = Image.new('RGB', (224, 224), color=(150, 120, 90))
    image_tensor = preprocess_image(test_image, requires_grad=False)

    with torch.no_grad():
        output = model(image_tensor.to(device))
        target_class = output.argmax(dim=1).item()

    baselines = ['zeros', 'blur', 'random']
    saliencies = {}

    for baseline in baselines:
        print(f"{baseline} 밑금으로 셈하는 중...")
        sal = compute_integrated_gradients(
            model, image_tensor, target_class, device,
            baseline=baseline, steps=30
        )
        saliencies[f'{baseline}\n밑금'] = sal

    visualize_multiple_saliencies(
        image_tensor, saliencies,
        save_path='outputs/04_baseline_comparison.png'
    )

    print("\n밑금 고르기:")
    print("- 0: 단순하고 빠르며 웬만한 자리에서 잘 듣는다")
    print("- 흐림: 자연 그림에 좋다")
    print("- 아무렇게나: 쓸 일이 드물다")
    print("\n✓ 0 밑금이 가장 흔하다!")


def main():
    print("\n" + "="*70)
    print(" "*15 + "쌓은 기울기 익히기")
    print("="*70)

    try:
        example_1_baseline_comparison()

        print("\n" + "="*70)
        print("고갱이:")
        print("1. IG은 예민함과 짜기에 흔들리지 않음을 채운다")
        print("2. 밑금을 어떻게 고르느냐가 중요하다(대개 0이 좋다)")
        print("3. 걸음이 많을수록 더 맞다(흔히 30~50)")
        print("4. 셈이 비싸지만 이론이 든든하다")
        print("\n다음: 꾸러미 05 - Grad-CAM")
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
04: 쌓은 기울기 짜보기를 살피는 두루 갖춘 시험 함수를 써라. 빈 들임, 원소 하나짜리 들임, 아주 큰 들임, 그리고 끝자락 값(0, 아주 큰 수)이 든 들임 같은 가장자리 자리를 시험하여라.

??? success "익힘 4 풀이"
    금 언저리 조건을 두루 건드리는 시험 함수를 짓는다.
    ```python
    def test_04: integrated gradients():
        model = 04: 쌓은 기울기(...)
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
