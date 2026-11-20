# 04: Integrated Gradients

Integrated Gradients(IG)를 구현해 베이스라인 선택에 따라 기여도 맵이 어떻게 달라지는지 견주는 쪽이다. IG는 베이스라인에서 입력으로 가는 직선 경로를 따라 그래디언트를 적분해 기여도를 매기는, 공리에 근거한 방법이다.

뉴럴 네트워크가 무엇을 배우는지 아는 일은 신뢰를 쌓고 모델을 디버깅하는 데 중요하다. 이 모듈은 모델이 입력을 어떻게 처리하고 어떻게 판단하는지 드러내는 그래디언트 기반 설명 기법을 보이며, 네트워크의 행동을 눈으로 보고 수로 재게 해 준다.

## 1. 코드

```python
"""
04: 쌓은 그래디언트 - 이치에 닿는 기여도 매기기
================================================

어려움: 가운데

설명:
쌓은 그래디언트는 베이스라인에서 들임으로 가는 길을 따라 그래디언트를 쌓는다.
이로써 기여도 매기기의 중요한 공리인 예민함과 짜기에 흔들리지 않음을
채운다.

수학 밑바탕:
    IG(x) = (x - x') ⊙ ∫₀¹ (∂f(x' + α(x - x'))/∂x) dα

여기서:
- x: 들임 그림
- x': 베이스라인(흔히 0이나 흐린 그림)
- α ∈ [0,1]: 사이 잡는 계수
- 적분은 리만 합으로 어림한다

채우는 공리:
1. 예민함: 결이 출력을 바꾸면 0이 아닌 기여도를 받는다
2. 짜기에 흔들리지 않음: 하는 일이 같은 네트워크는 같은 기여도를 받는다
3. 완전성: 기여도의 합이 f(x) - f(x')이 된다

지은이: 교육용 자료
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
    쌓은 그래디언트를 셈한다.

    알고리즘:
    1. 베이스라인 x'을 고른다
    2. 사이 들임을 만든다: i=0..m에 대해 x^(i) = x' + (i/m)(x - x')
    3. 자리마다 그래디언트를 셈한다: gᵢ = ∂f(x^(i))/∂x
    4. 그래디언트를 고르게 한다: ḡ = (1/m) Σᵢ gᵢ
    5. 들임 차이로 스케일을 맞춘다: IG = (x - x') ⊙ ḡ

    Args:
        baseline: 'zeros', 'blur', 'random' 가운데 하나
        steps: 사이를 잡는 스텝 수(많을수록 더 맞다)
    """
    model.eval()
    image_tensor = image_tensor.to(device)

    # 베이스라인을 만든다
    if baseline == 'zeros':
        baseline_tensor = torch.zeros_like(image_tensor)
    elif baseline == 'blur':
        # 그림을 흐리게 해서 베이스라인으로 삼는다
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

        # 앞으로 스텝
        output = model(interpolated)
        target_score = output[0, target_class]

        # 역전파 스텝
        model.zero_grad()
        target_score.backward()

        # 그래디언트를 쌓는다
        accumulated_gradients += interpolated.grad

    # 그래디언트를 고르게 한다(적분의 리만 어림)
    avg_gradients = accumulated_gradients / steps

    # 들임 차이로 스케일을 맞춘다
    integrated_grads = (image_tensor - baseline_tensor) * avg_gradients

    # 모은다
    abs_attr = torch.abs(integrated_grads)
    saliency = torch.max(abs_attr, dim=1)[0]

    return saliency


def verify_completeness(model, image_tensor, target_class, device, saliency):
    """기여도의 합이 출력 차이가 되는지 살핀다."""
    model.eval()

    with torch.no_grad():
        output_image = model(image_tensor.to(device))[0, target_class]
        baseline = torch.zeros_like(image_tensor).to(device)
        output_baseline = model(baseline)[0, target_class]

    # 기여도의 합이 출력 차이와 ≈ 같아야 한다
    # 짚을 것: 잘게 나누어 셈하므로 어림값이다
    print(f"\n완전성 살피기:")
    print(f"f(x) - f(x'): {(output_image - output_baseline).item():.4f}")
    print("(두드러짐의 합이 이 값에 가까워야 한다)")


def example_1_baseline_comparison():
    """베이스라인을 달리 고른 것들을 견준다."""
    print("\n" + "="*60)
    print("보기 1: 베이스라인 견주기")
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
        print(f"{baseline} 베이스라인으로 셈하는 중...")
        sal = compute_integrated_gradients(
            model, image_tensor, target_class, device,
            baseline=baseline, steps=30
        )
        saliencies[f'{baseline}\n베이스라인'] = sal

    visualize_multiple_saliencies(
        image_tensor, saliencies,
        save_path='outputs/04_baseline_comparison.png'
    )

    print("\n베이스라인 고르기:")
    print("- 0: 단순하고 빠르며 웬만한 자리에서 잘 듣는다")
    print("- 흐림: 자연 그림에 좋다")
    print("- 아무렇게나: 쓸 일이 드물다")
    print("\n✓ 0 베이스라인이 가장 흔하다!")


def main():
    print("\n" + "="*70)
    print(" "*15 + "쌓은 그래디언트 익히기")
    print("="*70)

    try:
        example_1_baseline_comparison()

        print("\n" + "="*70)
        print("핵심:")
        print("1. IG는 예민함과 짜기에 흔들리지 않음을 채운다")
        print("2. 베이스라인을 어떻게 고르느냐가 중요하다(대개 0이 좋다)")
        print("3. 스텝이 많을수록 더 맞다(흔히 30~50)")
        print("4. 셈이 비싸지만 이론이 강건하다")
        print("\n다음: 꾸러미 05 - Grad-CAM")
        print("="*70)
    except Exception as e:
        print(f"어긋남: {e}")

if __name__ == "__main__":
    main()
```

**출력:**

```
======================================================================
               쌓은 그래디언트 익히기
======================================================================

============================================================
보기 1: 베이스라인 견주기
============================================================
어긋남: name 'get_device' is not defined
```

## 2. 논의

위 출력에서 보듯 이 프로그램은 첫 보기에서 멈춘다. `get_device`·`load_pretrained_model` 같은 도우미를 `from utils import *` 로 가져오는데, 그 `utils` 모듈이 없어 `NameError` 가 났다. 그러니 베이스라인별 기여도 맵은 이 쪽에 실리지 않았고, 핵심 계산은 `compute_integrated_gradients` 의 코드로 읽어야 한다.

이 함수는 그래디언트 기반 설명의 핵심 아이디어를 짧은 PyTorch 코드로 보여 준다. 베이스라인 생성, 경로 위 그래디언트 누적, 입력 차이로 스케일 맞추기가 차례로 나뉘어 있어 부분마다 따로 이해하고 다른 과제나 데이터에 맞추어 고치기 쉽다.

여기서 보인 기법은 더 까다로운 상황으로도 자연스럽게 확장된다. 하이퍼파라미터, 구조의 종류, 여러 데이터를 바꿔 가며 해 보면 이해가 깊어지고 모델 해석에 대한 감이 생긴다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
코드를 읽고 핵심 설계 선택을 짚어라. 구현에서 고른 것 셋을 들고, 저마다 왜 그래디언트 기반 설명에 알맞은지 설명하라.

</div>

??? success "풀이"
    `compute_integrated_gradients` 에서 셋을 들 수 있다. (1) **베이스라인 선택**(`'zeros'`·`'blur'`·`'random'`) -- IG의 기여도는 「베이스라인에 견주어」 매기는 값이므로 무엇을 정보 없는 입력으로 볼지가 결과를 정한다. (2) **오른쪽 끝점 리만 합**(`alpha = (step + 1) / steps`) -- 적분을 $m$개 점의 평균으로 근사하며, 스텝이 많을수록 완전성 오차가 줄어든다. (3) **채널 축 절댓값 최대**(`torch.max(abs_attr, dim=1)`) -- RGB 세 채널의 기여도를 한 장의 맵으로 모아 보기 쉽게 하되, 부호 정보는 버린다. 각 선택은 정확도, 계산 비용, 해석 용이성 사이의 트레이드오프를 드러낸다.

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
    흔한 실패 유형은 이렇다. (1) **스텝이 모자라 완전성이 깨짐** -- 기여도의 합을 $f(x) - f(x')$ 과 견주어 탐지한다(`verify_completeness` 가 뒤엣것을 찍는다. 다만 이 함수는 절댓값을 취하기 전의 기여도를 받아야 견줄 수 있다). 상대 오차가 5%를 넘으면 스텝을 늘린다. (2) **베이스라인 편향** -- 0 베이스라인에서는 $x_i - x'_i = 0$ 인 검은 픽셀의 기여도가 언제나 0이 되어, 어두운 물체가 중요해도 맵에 나타나지 않는다. 베이스라인 여럿의 맵을 견주어 탐지하고, 흐린 이미지나 여러 베이스라인의 평균(Expected Gradients)으로 고친다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
04: Integrated Gradients 구현을 검사하는 포괄적인 테스트 함수를 작성하라. 빈 입력, 원소 하나짜리 입력, 아주 큰 입력, 그리고 극단값(0, 아주 큰 수)이 든 입력 같은 경계 사례를 테스트하여라.

</div>

??? success "풀이"
    경계 조건을 두루 다루는 테스트 함수를 작성한다.
    ```python
    def test_04: integrated gradients():
        model = 04: 쌓은 그래디언트(...)
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

**다룬 것** — 04: Integrated Gradients

이 구현은 깔끔하고 읽기 쉬운 PyTorch 코드로 그래디언트 기반 설명의 핵심 아이디어를 보여 준다.

앞의 연습문제 4개로 스스로 점검해 볼 수 있다.
