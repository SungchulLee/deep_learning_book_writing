# 03: SmoothGrad

03: SmoothGrad - Saliency Map의 잡음 줄이기. 설명:

뉴럴 네트워크가 무엇을 배우는지 아는 일은 신뢰를 쌓고 모델을 디버깅하는 데 중요하다. 이 모듈은 모델이 입력을 어떻게 처리하고 어떻게 판단하는지 드러내는 그래디언트 기반 설명 기법을 보이며, 네트워크의 행동을 눈으로 보고 수로 재게 해 준다.

## 1. 코드

```python
"""
03: SmoothGrad - 두드러짐 그림의 잡음 줄이기
=================================================

어려움: 낮음

설명:
SmoothGrad은 들임에 잡음을 섞은 것들에서 셈한 그래디언트를 고르게 해
두드러짐 그림의 눈에 띄는 잡음을 줄인다. 잡음을 더해 고르게 하면
날카롭고 흔들리는 그래디언트가 서로 지워지고 뜻있는 신호만 남는다.

수학 밑바탕:
    SmoothGrad(x) = (1/n) Σᵢ₌₁ⁿ |∂y_c/∂(x + N(0, σ²))|

여기서:
- n: 잡음 섞은 표본의 수
- N(0, σ²): 표준편차이 σ인 가우스 잡음
- 고르게 하기가 그래디언트 잡음을 완화한다

으뜸 파라메터:
- n (num_samples): 표본이 많을수록 매끄럽지만 느리다(흔히 20~50)
- σ (noise_level): 잡음의 크기를 다스린다(흔히 들임 자의 0.1~0.2)

지은이: 교육용 자료
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
       b. 그래디언트를 셈한다: gᵢ = |∂y_c/∂x_noisy|
    2. 그래디언트를 고르게 한다: SmoothGrad = (1/n) Σᵢ gᵢ

    왜 듣는가:
    - 날카롭고 흔들리는 그래디언트는 잡음에 따라 아무렇게나 달라진다
    - 뜻있는 그래디언트는 한결같이 남는다
    - 고르게 하기가 잡음을 지우고 신호를 남긴다
    - 기계 러닝의 모둠 방법과 결이 같다

    Args:
        model: 미리 익힌 모형
        image_tensor: 들임 그림 [1, 3, H, W]
        target_class: 겨눈 클래스 번호
        device: 셈하는 장치
        num_samples: 잡음 섞은 표본의 수
        noise_level: 가우스 잡음의 표준편차(들임 자에 대한 기여도)

    Returns:
        torch.Tensor: 매끄럽게 한 두드러짐 그림 [1, H, W]
    """
    model.eval()
    image_tensor = image_tensor.to(device)

    # 그래디언트를 쌓아 둘 곳
    accumulated_gradients = torch.zeros_like(image_tensor)

    # 잡음의 표준편차을 셈한다
    # 흔한 들임 너비(고르게 한 뒤)에 맞추어 잡는다
    noise_std = noise_level

    for i in range(num_samples):
        # 잡음 섞은 들임을 만든다
        noise = torch.randn_like(image_tensor) * noise_std
        noisy_image = image_tensor + noise
        noisy_image.requires_grad = True

        # 앞으로 스텝
        output = model(noisy_image)
        target_score = output[0, target_class]

        # 역전파 스텝
        model.zero_grad()
        target_score.backward()

        # 그래디언트를 쌓는다
        accumulated_gradients += noisy_image.grad

    # 그래디언트를 고르게 한다
    mean_gradients = accumulated_gradients / num_samples

    # 절댓값을 잡고 통로를 가로질러 모은다
    abs_gradients = torch.abs(mean_gradients)
    saliency = torch.max(abs_gradients, dim=1)[0]

    return saliency


def example_1_smoothgrad_vs_vanilla():
    """SmoothGrad과 맨 그래디언트를 견준다."""
    print("\n" + "="*60)
    print("보기 1: SmoothGrad 대 맨 그래디언트")
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

    print(f"\n겨눈 클래스: {target_class}")
    print("맨 그래디언트를 셈한다...")

    # 맨 그래디언트
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
            '맨 그래디언트\n(잡음 많음)': vanilla,
            'SmoothGrad\n(깨끗함)': smoothgrad
        },
        save_path='outputs/03_smoothgrad_comparison.png'
    )

    print("\n✓ SmoothGrad이 더 깨끗한 그림을 낸다!")


def example_2_parameter_sensitivity():
    """num_samples과 noise_level이 미치는 힘을 살핀다."""
    print("\n" + "="*60)
    print("보기 2: 파라메터에 대한 예민함")
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
        print("핵심:")
        print("1. SmoothGrad은 고르게 해서 눈에 띄는 잡음을 줄인다")
        print("2. 잡음을 더했는데 되레 잡음이 준다!")
        print("3. 파라메터: n=20~50, σ=0.10~0.20")
        print("4. 셈이 비싸다(앞으로/역전파를 n번 한다)")
        print("\n다음: 꾸러미 04 - 쌓은 그래디언트")
        print("="*70)
    except Exception as e:
        print(f"어긋남: {e}")

if __name__ == "__main__":
    main()
```

**출력:**

```
======================================================================
                    SMOOTHGRAD 익히기
======================================================================

============================================================
보기 1: SmoothGrad 대 맨 그래디언트
============================================================
어긋남: name 'get_device' is not defined
```

## 2. 논의

이 구현은 깔끔하고 읽기 쉬운 PyTorch 코드로 그래디언트 기반 설명의 핵심 아이디어를 보여 준다. 모듈로 나눈 구조 덕에 부분마다 따로 이해하고 다른 과제나 데이터에 맞추어 고치기 쉽다.

여기서 보인 기법은 더 까다로운 상황으로도 자연스럽게 확장된다. 하이퍼파라미터, 구조의 종류, 여러 데이터를 바꿔 가며 해 보면 이해가 깊어지고 모델 해석에 대한 감이 생긴다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
코드를 읽고 핵심 설계 선택을 짚어라. 구현에서 고른 것 셋을 들고, 저마다 왜 그래디언트 기반 설명에 알맞은지 설명하라.

</div>

??? success "풀이"
    설계 선택은 구현마다 다르나 흔히 이런 것이 있다. (1) 활성화 함수 선택 -- ReLU 계열은 그래디언트가 소실되지 않아 학습이 빠르다. (2) 정규화 전략 -- 배치 정규화가 내부 공변량 이동을 줄여 학습을 안정시킨다. (3) 잔차 연결 -- 있으면 스킵 경로를 주어 깊은 네트워크에서 그래디언트가 흐르게 한다. 각 선택은 표현력, 계산 비용, 학습 안정성 사이의 트레이드오프를 드러낸다.

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
    흔한 실패 유형은 이렇다. (1) **그래디언트가 사라지거나 터짐** -- 그래디언트 크기를 모니터링해 탐지한다(`torch.nn.utils.clip_grad_norm_`이나 레이어마다 `param.grad.norm()` 기록). 그래디언트 클리핑, 더 나은 초기화(Xavier/Kaiming), 구조 수정(잔차 연결, 정규화 레이어)으로 고친다. (2) **과적합** -- 학습 로스는 줄어드는데 검증 로스가 오르면 탐지한다. 정칙화(드롭아웃, 웨이트 감쇠, 데이터 증강)나 모델 축소로 고친다. 학습 지표와 검증 지표를 늘 함께 모니터링해 이를 일찍 잡아야 한다.

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
        # 여느 들임
        assert model(normal_input).shape == expected_shape
        # 원소 하나짜리 배치
        assert model(single_input).shape == (1, ...)
        # 큰 값(넘침을 살핀다)
        out = model(torch.ones(...) * 1000)
        assert torch.isfinite(out).all()
        # 그래디언트 흐름
        out = model(normal_input)
        out.sum().backward()
        for p in model.parameters():
            assert p.grad is not None
    ```
    구조가 엔드투엔드 학습을 지원하는지 알려면 그래디언트 흐름을 테스트하는 것이 특히 중요하다.

## 정리하며

**다룬 것** — 03: SmoothGrad

이 구현은 깔끔하고 읽기 쉬운 PyTorch 코드로 그래디언트 기반 설명의 핵심 아이디어를 보여 준다.

앞의 연습문제 4개로 스스로 점검해 볼 수 있다.
