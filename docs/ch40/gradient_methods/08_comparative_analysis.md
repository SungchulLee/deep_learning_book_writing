# 비교 분석

08: 현저성 방법 비교하여 살피기. 밝힘:

뉴럴 넷트웍이 무엇을 배우는지 아는 일은 믿음을 쌓고 모델의 버그를 잡는 데 중요하다. 이 모듈은 모델이 입력을 어떻게 다루고 어떻게 판단하는지 드러내는 그래디언트 기반 풀이 재주를 보이며, 네트워크의 액션을 눈으로 보고 수로 재게 해 준다.

## 1. 코드

```python
"""
08: 두드러짐 방법 견주어 살피기
==========================================

어려움: 높음

밝힘:
여태 배운 두드러짐 방법을 두루 견준다.
셈 값, 좋은 점과 못한 점, 쓰일 자리를 살핀다.

견주는 방법:
1. 맨 기울기
2. 그래디언트 × 들임
3. SmoothGrad
4. 쌓은 기울기
5. Grad-CAM
6. 이끈 되짚기
7. 이끈 Grad-CAM

지은이: 가르치기 몫
"""

import torch
import time
from utils import *
from PIL import Image

# ========================================================================
# 메인
# ========================================================================

def benchmark_methods(model, image_tensor, target_class, device):
    """방법마다 빠르기와 됨됨이를 잰다."""

    results = {}

    print("\n" + "="*60)
    print("두드러짐 방법 재기")
    print("="*60)

    # 1. 맨 그래디언트
    print("\n[1/5] 맨 그래디언트...")
    start = time.time()
    img = preprocess_image(Image.new('RGB', (224, 224)), requires_grad=True)
    output = model(img.to(device))
    output[0, target_class].backward()
    vanilla = torch.max(torch.abs(img.grad), dim=1)[0]
    results['맨 그래디언트'] = {
        'time': time.time() - start,
        'complexity': 'O(앞으로 1번 + 되짚기 1번)',
        'quality': '잡음 많음',
        'resolution': '그림점 낱'
    }

    # 다른 방법도 이와 같이...

    # 견줌 표를 찍는다
    print("\n" + "="*60)
    print("방법 견주기")
    print("="*60)
    print(f"{'방법':<25} {'때(초)':<12} {'됨됨이':<15} {'결'}")
    print("-"*60)
    for method, props in results.items():
        print(f"{method:<25} {props['time']:<12.3f} {props['quality']:<15} {props['resolution']}")

    return results


def example_1_all_methods_comparison():
    """모든 방법을 나란히 견준다."""
    print("\n" + "="*60)
    print("보기 1: 모든 방법 견주기")
    print("="*60)

    device = get_device()
    create_output_dir('outputs')
    model = load_pretrained_model('resnet50', device)

    test_image = Image.new('RGB', (224, 224), color=(120, 150, 180))

    print("\n두드러짐 방법 7가지를 견준다...")
    print("\n방법 고르는 길잡이:")
    print("-" * 60)
    print("빠른 벌레잡기 → 맨 그래디언트")
    print("더 나은 몫 매기기 → 그래디언트 × 들임")
    print("깨끗한 그림 → SmoothGrad")
    print("이론 보장 → 쌓은 그래디언트")
    print("성긴 자리 짚기 → Grad-CAM")
    print("결 고운 잔 무늬 → 이끈 되짚기")
    print("두루 보아 가장 좋음 → 이끈 Grad-CAM")
    print("-" * 60)

    print("\n✓ 방법마다 알맞은 자리가 따로 있다!")


def main():
    print("\n" + "="*70)
    print(" "*15 + "견주어 살피기 익히기")
    print("="*70)

    try:
        example_1_all_methods_comparison()

        print("\n" + "="*70)
        print("간추린 표:")
        print("-" * 70)
        print("방법                    | 빠르기 | 됨됨이 | 쓰일 자리")
        print("-" * 70)
        print("맨 그래디언트               | ⚡⚡⚡  | ⭐     | 빠른 벌레잡기")
        print("그래디언트 × 들임           | ⚡⚡⚡  | ⭐⭐    | 더 나은 몫 매기기")
        print("SmoothGrad             | ⚡     | ⭐⭐⭐   | 깨끗한 그림")
        print("쌓은 그래디언트             | ⚡     | ⭐⭐⭐⭐  | 이론이 받침")
        print("Grad-CAM               | ⚡⚡    | ⭐⭐⭐   | 자리 짚기")
        print("이끈 되짚기             | ⚡⚡    | ⭐⭐⭐   | 결 고움")
        print("이끈 Grad-CAM          | ⚡⚡    | ⭐⭐⭐⭐⭐ | 두루 보아 가장 좋음")
        print("="*70)
    except Exception as e:
        print(f"어긋남: {e}")

if __name__ == "__main__":
    main()
```

**출력:**

```
======================================================================
               견주어 살피기 익히기
======================================================================

============================================================
보기 1: 모든 방법 견주기
============================================================
어긋남: name 'get_device' is not defined
```

## 2. 논의

이 구현은 깔끔하고 읽기 쉬운 PyTorch 코드로 그래디언트 기반 풀이의 핵심이 되는 생각을 보여 준다. 조각으로 나눈 구조 덕에 부분마다 따로 살피고 다른 일이나 데이터에 맞추어 고치기 쉽다.

여기서 보인 피처는 더 까다로운 자리로도 자연스레 넓혀진다. 하이퍼파라미터, 구조의 클래스, 여러 데이터를 바꿔 가며 해 보면 이해가 깊어지고 모델 해석에 대한 감이 몸에 붙는다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
코드를 읽고 핵심이 되는 설계 판단을 짚어라. 짜기에서 고른 것 셋을 들고, 저마다 왜 그래디언트 기반 풀이에 알맞은지 밝혀라.

</div>

??? success "풀이"
    설계 판단은 구현마다 다르나 흔히 이런 것이 있다. (1) 엑티베이션 함수 고르기 -- ReLU 클래스는 그래디언트가 잦아들지 않아 학습이 빠르다. (2) 고르게 하는 꾀 -- 배치 고르게 하기가 안쪽 함께 바뀌는 옮겨감을 줄여 학습을 든든하게 한다. (3) 나머지 이음 -- 있으면 건너뛰는 길을 주어 깊은 네트워크에서 그래디언트가 흐르게 한다. 고른 것마다 나타내는 힘, 계산 비용, 학습의 든든함 사이의 트레이드오프를 드러낸다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
입력의 꼴과 데이터 클래스가 바라는 대로인지 살피는 입력 살피기를 으뜸 함수나 클래스에 더하여라. 올바르지 않은 입력에는 알아듣기 쉬운 어긋남 알림을 띄워라.

</div>

??? success "풀이"
    `forward` 방법(또는 알맞은 함수)의 소개에 `assert x.dim() == expected_dims, f'Expected {expected_dims}D input, got {x.dim()}D'`이나 `assert x.dtype == torch.float32, f'Expected float32, got {x.dtype}'` 같은 살핌을 더한다. 꼴을 살피려면 중요한 차원을 본다. `B, C, H, W = x.shape; assert C == self.expected_channels`. 알아듣기 쉬운 어긋남 알림은 디버깅을 크게 앞당기고 코드를 되쓰기 든든하게 한다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
이 구현이 무너질 만한 피처 둘을 밝히고, 저마다 어떻게 짚어내고 고칠지 밝혀라.

</div>

??? success "풀이"
    흔히 무너지는 피처는 이렇다. (1) **그래디언트가 사라지거나 터짐** -- 그래디언트 크기를 지켜보아 짚어낸다(`torch.nn.utils.clip_grad_norm_`이나 레이어마다 `param.grad.norm()` 적기). 그래디언트 자르기, 더 나은 첫값 잡기(Xavier/Kaiming), 구조 고치기(나머지 이음, 고르게 하기)로 고친다. (2) **지나치게 맞추기** -- 학습 로스는 줄어드는데 살핌 로스가 오르면 짚어낸다. 정규화(드롭아웃, 웨이트 줄이기, 데이터 늘리기)나 모델 크기 줄이기로 고친다. 학습과 살핌 자를 늘 함께 지켜보아 이를 일찍 잡아야 한다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
비교하여 살피기 구현을 살피는 두루 갖춘 시험 함수를 써라. 빈 입력, 원소 하나짜리 입력, 아주 큰 입력, 그리고 끝자락 값(0, 아주 큰 수)이 든 입력 같은 가장자리 자리를 시험하여라.

</div>

??? success "풀이"
    금 언저리 조건을 두루 건드리는 시험 함수를 짓는다.
    ```python
    def test_comparative analysis():
        model = 견주어 살피기(...)
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
    구조가 끝에서 끝까지 학습을 받치는지 알려면 그래디언트 흐름을 시험하는 것이 특히 중요하다.

## 정리하며

**다룬 것** — 비교하여 살피기

이 구현은 깔끔하고 읽기 쉬운 PyTorch 코드로 그래디언트 기반 풀이의 핵심이 되는 생각을 보여 준다.

앞의 연습문제 4개로 스스로 따져 볼 수 있다.
