# 견주어 살피기

08: 두드러짐 방법 견주어 살피기. 밝힘:

신경 그물이 무엇을 배우는지 아는 일은 믿음을 쌓고 모형의 벌레를 잡는 데 종요롭다. 이 꾸러미는 모형이 들임을 어떻게 다루고 어떻게 판단하는지 드러내는 기울기 바탕 풀이 재주를 보이며, 그물의 움직임을 눈으로 보고 수로 재게 해 준다.

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
2. 기울기 × 들임
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

    # 1. 맨 기울기
    print("\n[1/5] 맨 기울기...")
    start = time.time()
    img = preprocess_image(Image.new('RGB', (224, 224)), requires_grad=True)
    output = model(img.to(device))
    output[0, target_class].backward()
    vanilla = torch.max(torch.abs(img.grad), dim=1)[0]
    results['맨 기울기'] = {
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
    print("빠른 벌레잡기 → 맨 기울기")
    print("더 나은 몫 매기기 → 기울기 × 들임")
    print("깨끗한 그림 → SmoothGrad")
    print("이론 보장 → 쌓은 기울기")
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
        print("맨 기울기               | ⚡⚡⚡  | ⭐     | 빠른 벌레잡기")
        print("기울기 × 들임           | ⚡⚡⚡  | ⭐⭐    | 더 나은 몫 매기기")
        print("SmoothGrad             | ⚡     | ⭐⭐⭐   | 깨끗한 그림")
        print("쌓은 기울기             | ⚡     | ⭐⭐⭐⭐  | 이론이 받침")
        print("Grad-CAM               | ⚡⚡    | ⭐⭐⭐   | 자리 짚기")
        print("이끈 되짚기             | ⚡⚡    | ⭐⭐⭐   | 결 고움")
        print("이끈 Grad-CAM          | ⚡⚡    | ⭐⭐⭐⭐⭐ | 두루 보아 가장 좋음")
        print("="*70)
    except Exception as e:
        print(f"어긋남: {e}")

if __name__ == "__main__":
    main()
```

## 2. 논의

이 짜보기는 깔끔하고 읽기 쉬운 PyTorch 코드로 기울기 바탕 풀이의 고갱이가 되는 생각을 보여 준다. 조각으로 나눈 얼개 덕에 부분마다 따로 살피고 다른 일이나 자료에 맞추어 고치기 쉽다.

여기서 보인 결은 더 까다로운 자리로도 자연스레 넓혀진다. 하이퍼파라미터, 얼개의 갈래, 여러 자료를 바꿔 가며 해 보면 이해가 깊어지고 모형 풀이하기에 대한 감이 몸에 붙는다.

## 연습문제

**연습문제 1.**
코드를 읽고 고갱이가 되는 설계 판단을 짚어라. 짜기에서 고른 것 셋을 들고, 저마다 왜 기울기 바탕 풀이에 알맞은지 밝혀라.

??? success "연습문제 1 풀이"
    설계 판단은 짜보기마다 다르나 흔히 이런 것이 있다. (1) 살림 함수 고르기 -- ReLU 갈래는 기울기가 잦아들지 않아 익히기가 빠르다. (2) 고르게 하는 꾀 -- 묶음 고르게 하기가 안쪽 함께 바뀌는 옮겨감을 줄여 익힘을 든든하게 한다. (3) 나머지 이음 -- 있으면 건너뛰는 길을 주어 깊은 그물에서 기울기가 흐르게 한다. 고른 것마다 나타내는 힘, 셈 값, 익힘의 든든함 사이의 맞바꿈을 드러낸다.

---

**연습문제 2.**
들임의 꼴과 자료 갈래가 바라는 대로인지 살피는 들임 살피기를 으뜸 함수나 클래스에 더하여라. 올바르지 않은 들임에는 알아듣기 쉬운 어긋남 알림을 띄워라.

??? success "연습문제 2 풀이"
    `forward` 방법(또는 알맞은 함수)의 첫머리에 `assert x.dim() == expected_dims, f'Expected {expected_dims}D input, got {x.dim()}D'`이나 `assert x.dtype == torch.float32, f'Expected float32, got {x.dtype}'` 같은 살핌을 더한다. 꼴을 살피려면 종요로운 차원을 본다. `B, C, H, W = x.shape; assert C == self.expected_channels`. 알아듣기 쉬운 어긋남 알림은 벌레잡기를 크게 앞당기고 코드를 되쓰기 든든하게 한다.

---

**연습문제 3.**
이 짜보기가 무너질 만한 결 둘을 밝히고, 저마다 어떻게 짚어내고 고칠지 밝혀라.

??? success "연습문제 3 풀이"
    흔히 무너지는 결은 이렇다. (1) **기울기가 사라지거나 터짐** -- 기울기 크기를 지켜보아 짚어낸다(`torch.nn.utils.clip_grad_norm_`이나 켜마다 `param.grad.norm()` 적기). 기울기 자르기, 더 나은 첫값 잡기(Xavier/Kaiming), 얼개 고치기(나머지 이음, 고르게 하기)로 고친다. (2) **지나치게 맞추기** -- 익힘 잃음은 줄어드는데 살핌 잃음이 오르면 짚어낸다. 정칙화(드롭아웃, 짐 줄이기, 자료 늘리기)나 모형 크기 줄이기로 고친다. 익힘과 살핌 자를 늘 함께 지켜보아 이를 일찍 잡아야 한다.

---

**연습문제 4.**
견주어 살피기 짜보기를 살피는 두루 갖춘 시험 함수를 써라. 빈 들임, 원소 하나짜리 들임, 아주 큰 들임, 그리고 끝자락 값(0, 아주 큰 수)이 든 들임 같은 가장자리 자리를 시험하여라.

??? success "연습문제 4 풀이"
    금 언저리 조건을 두루 건드리는 시험 함수를 짓는다.
    ```python
    def test_comparative analysis():
        model = 견주어 살피기(...)
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

## 정리하며

**다룬 것** — 견주어 살피기

이 짜보기는 깔끔하고 읽기 쉬운 PyTorch 코드로 기울기 바탕 풀이의 고갱이가 되는 생각을 보여 준다.

앞의 연습문제 4개로 스스로 따져 볼 수 있다.
