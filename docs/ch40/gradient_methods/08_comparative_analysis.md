# 비교 분석

앞의 쪽들에서 다룬 Saliency Map 방법 일곱 가지(바닐라 그래디언트, 그래디언트 × 입력, SmoothGrad, Integrated Gradients, Grad-CAM, Guided Backpropagation, Guided Grad-CAM)를 속도와 맵의 품질로 비교하려는 쪽이다.

뉴럴 네트워크가 무엇을 배우는지 아는 일은 신뢰를 쌓고 모델을 디버깅하는 데 중요하다. 이 모듈은 모델이 입력을 어떻게 처리하고 어떻게 판단하는지 드러내는 그래디언트 기반 설명 기법을 보이며, 네트워크의 행동을 눈으로 보고 수로 재게 해 준다.

## 1. 코드

```python
"""
08: 샐리언시 방법 비교 분석
==========================================

난이도: 높음

설명:
지금까지 배운 샐리언시 방법을 두루 비교한다.
계산 비용, 장단점, 쓰임새를 살핀다.

비교하는 방법:
1. 바닐라 그래디언트
2. 그래디언트 × 입력
3. SmoothGrad
4. Integrated Gradients
5. Grad-CAM
6. Guided Backpropagation
7. Guided Grad-CAM

지은이: 교육용 자료
"""

import torch
import time
import os
from PIL import Image

# ========================================================================
# 도우미 함수
# ========================================================================

# ImageNet으로 학습한 모델이 쓰는 값. 전처리할 때와 되돌릴 때 같은 값을 써야 한다.
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
    """torchvision에서 사전 학습된 모델을 불러온다.

    다운로드가 안 되면(네트워크가 없을 때) 학습하지 않은 모델로 대체한다. 그래디언트
    방법을 보이는 데에는 가중치가 실제로 학습된 것이 아니어도 맵이 나온다.
    """
    from torchvision import models

    device = device or get_device()
    try:
        model = getattr(models, model_name)(weights="DEFAULT")
    except Exception:
        print(f"  사전 학습 가중치를 받지 못했다 — {model_name}을 무작위 초기화 상태로 쓴다.")
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


def create_output_dir(path="outputs"):
    """출력 이미지를 저장할 디렉터리를 만들고 그 경로를 돌려준다."""
    os.makedirs(path, exist_ok=True)
    return path


# ========================================================================
# 메인
# ========================================================================

def benchmark_methods(model, image_tensor, target_class, device):
    """방법마다 속도와 품질을 잰다."""

    results = {}

    print("\n" + "="*60)
    print("샐리언시 방법 벤치마크")
    print("="*60)

    # 1. 바닐라 그래디언트
    print("\n[1/5] 바닐라 그래디언트...")
    start = time.time()
    img = preprocess_image(Image.new('RGB', (224, 224)), requires_grad=True)
    output = model(img.to(device))
    output[0, target_class].backward()
    vanilla = torch.max(torch.abs(img.grad), dim=1)[0]
    results['바닐라 그래디언트'] = {
        'time': time.time() - start,
        'complexity': 'O(순전파 1번 + 역전파 1번)',
        'quality': '잡음 많음',
        'resolution': '픽셀 단위'
    }

    # 다른 방법도 이와 같이...

    # 비교 표를 찍는다
    print("\n" + "="*60)
    print("방법 비교")
    print("="*60)
    print(f"{'방법':<25} {'시간(초)':<12} {'품질':<15} {'해상도'}")
    print("-"*60)
    for method, props in results.items():
        print(f"{method:<25} {props['time']:<12.3f} {props['quality']:<15} {props['resolution']}")

    return results


def example_1_all_methods_comparison():
    """모든 방법을 나란히 비교한다."""
    print("\n" + "="*60)
    print("예제 1: 모든 방법 비교")
    print("="*60)

    device = get_device()
    create_output_dir('outputs')
    model = load_pretrained_model('resnet50', device)

    test_image = Image.new('RGB', (224, 224), color=(120, 150, 180))

    print("\n샐리언시 방법 7가지를 비교한다...")
    print("\n방법 선택 가이드:")
    print("-" * 60)
    print("빠른 디버깅 → 바닐라 그래디언트")
    print("더 나은 기여도 계산 → 그래디언트 × 입력")
    print("깨끗한 맵 → SmoothGrad")
    print("이론 보장 → Integrated Gradients")
    print("거친 위치 파악 → Grad-CAM")
    print("세밀한 디테일 → Guided Backpropagation")
    print("전반적으로 가장 좋음 → Guided Grad-CAM")
    print("-" * 60)

    print("\n✓ 방법마다 알맞은 쓰임새가 따로 있다!")


def main():
    print("\n" + "="*70)
    print(" "*15 + "비교 분석 튜토리얼")
    print("="*70)

    try:
        example_1_all_methods_comparison()

        print("\n" + "="*70)
        print("요약 표:")
        print("-" * 70)
        print("방법                    | 속도   | 품질       | 쓰임새")
        print("-" * 70)
        print("바닐라 그래디언트       | ⚡⚡⚡ | ⭐         | 빠른 디버깅")
        print("그래디언트 × 입력       | ⚡⚡⚡ | ⭐⭐       | 더 나은 기여도 계산")
        print("SmoothGrad              | ⚡     | ⭐⭐⭐     | 깨끗한 맵")
        print("Integrated Gradients    | ⚡     | ⭐⭐⭐⭐   | 이론적 근거")
        print("Grad-CAM                | ⚡⚡   | ⭐⭐⭐     | 위치 파악")
        print("Guided Backpropagation  | ⚡⚡   | ⭐⭐⭐     | 세밀한 디테일")
        print("Guided Grad-CAM         | ⚡⚡   | ⭐⭐⭐⭐⭐ | 전반적으로 가장 좋음")
        print("="*70)
    except Exception as e:
        print(f"오류: {e}")

if __name__ == "__main__":
    main()
```

**출력:**

```

======================================================================
               비교 분석 튜토리얼
======================================================================

============================================================
예제 1: 모든 방법 비교
============================================================

샐리언시 방법 7가지를 비교한다...

방법 선택 가이드:
------------------------------------------------------------
빠른 디버깅 → 바닐라 그래디언트
더 나은 기여도 계산 → 그래디언트 × 입력
깨끗한 맵 → SmoothGrad
이론 보장 → Integrated Gradients
거친 위치 파악 → Grad-CAM
세밀한 디테일 → Guided Backpropagation
전반적으로 가장 좋음 → Guided Grad-CAM
------------------------------------------------------------

✓ 방법마다 알맞은 쓰임새가 따로 있다!

======================================================================
요약 표:
----------------------------------------------------------------------
방법                    | 속도   | 품질       | 쓰임새
----------------------------------------------------------------------
바닐라 그래디언트       | ⚡⚡⚡ | ⭐         | 빠른 디버깅
그래디언트 × 입력       | ⚡⚡⚡ | ⭐⭐       | 더 나은 기여도 계산
SmoothGrad              | ⚡     | ⭐⭐⭐     | 깨끗한 맵
Integrated Gradients    | ⚡     | ⭐⭐⭐⭐   | 이론적 근거
Grad-CAM                | ⚡⚡   | ⭐⭐⭐     | 위치 파악
Guided Backpropagation  | ⚡⚡   | ⭐⭐⭐     | 세밀한 디테일
Guided Grad-CAM         | ⚡⚡   | ⭐⭐⭐⭐⭐ | 전반적으로 가장 좋음
======================================================================
```

## 2. 논의

위 출력에서 보듯 프로그램은 끝까지 돌지만, 찍히는 것은 방법 선택 가이드와 요약 표뿐이다. 표의 속도·품질 별점은 이 코드가 잰 값이 아니라 고정 문구이므로, 방법 사이의 실제 속도나 품질 차이로 인용하지 않는다.

코드가 돌았더라도 비교는 이루어지지 않는다. `benchmark_methods` 는 바닐라 그래디언트 하나만 재고 나머지는 `# 다른 방법도 이와 같이...` 로 비워 두었으며, 이 함수를 부르는 곳도 없다. `example_1_all_methods_comparison` 과 `main` 은 방법 선택 가이드와 별점 표를 찍을 뿐인데, 그 별점은 잰 값이 아니라 적어 넣은 평가다. 실제로 비교하려면 같은 이미지와 같은 타깃 클래스로 일곱 방법을 모두 계산하고, 시간과 함께 삭제·삽입 곡선 같은 충실성 지표를 재야 한다.

여기서 보인 기법은 더 까다로운 상황으로도 자연스럽게 확장된다. 하이퍼파라미터, 구조의 종류, 여러 데이터를 바꿔 가며 해 보면 이해가 깊어지고 모델 해석에 대한 감이 생긴다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
코드를 읽고 핵심 설계 선택을 짚어라. 구현에서 고른 것 셋을 들고, 저마다 왜 그래디언트 기반 설명에 알맞은지 설명하라.

</div>

??? success "풀이"
    `benchmark_methods` 에서 셋을 들 수 있다. (1) **같은 모델·같은 타깃 클래스로 비교하기** -- 방법마다 설명하는 대상이 같아야 맵의 차이가 방법의 차이가 된다. (2) **벽시계 시간 재기**(`time.time()`) -- 실제로 얼마나 기다려야 하는지 바로 보여 준다. 다만 GPU에서는 연산이 비동기로 돌아 `torch.cuda.synchronize()` 없이 재면 시간이 짧게 나오고, 첫 호출은 워밍업 비용까지 들어간다. (3) **계산 비용을 순전파·역전파 횟수로 적기**(`'complexity'`) -- 기계와 무관한 척도라 다른 환경에서도 비교할 수 있다. 바닐라 그래디언트는 한 번씩, SmoothGrad와 Integrated Gradients는 표본 수나 스텝 수만큼 든다.

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
    흔한 실패 유형은 이렇다. (1) **시간을 잘못 잼** -- GPU 연산은 비동기라 `time.time()` 만으로는 커널이 끝나기 전에 시계를 멈출 수 있고, 처음 부른 방법은 워밍업 비용까지 떠안는다. 같은 방법을 여러 번 재어 값이 크게 흔들리면 의심한다. 재기 전후에 `torch.cuda.synchronize()` 를 넣고, 한 번 미리 돌린 뒤 여러 번 잰 중앙값을 쓴다. (2) **앞 방법이 뒤 방법을 오염시킴** -- Guided Backpropagation이 건 ReLU 훅을 지우지 않거나 같은 입력 텐서의 `.grad` 를 비우지 않으면 뒤에 계산하는 방법의 그래디언트가 바뀐다. 방법을 혼자 돌린 결과와 다른 방법 뒤에 돌린 결과를 비교해 탐지하고, 방법마다 입력 텐서를 새로 만들고 훅을 바로 지워 고친다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
이 쪽의 비교 코드를 검사하는 포괄적인 테스트 함수를 작성하라. 빈 입력, 원소 하나짜리 입력, 아주 큰 입력, 그리고 극단값(0, 아주 큰 수)이 든 입력 같은 경계 사례를 테스트하여라.

</div>

??? success "풀이"
    경계 조건을 두루 다루는 테스트 함수를 작성한다.
    ```python
    def test_comparative analysis():
        model = 비교 분석(...)
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
    구조가 엔드투엔드 학습을 지원하는지 알려면 그래디언트 흐름을 테스트하는 것이 특히 중요하다.

## 정리하며

**다룬 것** — 비교 분석

Saliency Map 방법 일곱 가지를 비교하는 틀을 보았다. 다만 이 코드는 바닐라 그래디언트 하나만 재며, 별점 표는 잰 값이 아니다.

앞의 연습문제 4개로 스스로 점검해 볼 수 있다.
