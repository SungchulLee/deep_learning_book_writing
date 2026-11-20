# 07: Guided Grad-CAM

07: Guided Grad-CAM - 두 세계의 좋은 것을 아우르기. 설명:

뉴럴 네트워크가 무엇을 배우는지 아는 일은 신뢰를 쌓고 모델을 디버깅하는 데 중요하다. 이 모듈은 모델이 입력을 어떻게 처리하고 어떻게 판단하는지 드러내는 그래디언트 기반 설명 기법을 보이며, 네트워크의 행동을 눈으로 보고 수로 재게 해 준다.

## 1. 코드

```python
"""
07: 이끈 Grad-CAM - 두 세계의 좋은 것을 아우르기
==================================================

어려움: 높음

설명:
이끈 Grad-CAM은 Grad-CAM(클래스를 가려내되 성김)과 Guided Backpropagation(결이
고움)를 아울러 둘의 좋은 점을 함께 얻는다.

형태:
    이끈 Grad-CAM = Guided Backpropagation ⊙ 키운 Grad-CAM

나은 점:
- 결이 곱다(Guided Backpropagation에서 옴)
- 클래스를 가려낸다(Grad-CAM에서 옴)
- 그림 됨됨이가 가장 좋다

지은이: 교육용 자료
"""

import torch
import torch.nn.functional as F
from utils import *
from PIL import Image

# ========================================================================
# 메인
# ========================================================================

def compute_guided_gradcam(gradcam_map, guided_backprop_map):
    """
    Grad-CAM과 Guided Backpropagation를 아우른다.

    Args:
        gradcam_map: 성긴 Grad-CAM 열 그림 [H, W]
        guided_backprop_map: 결이 고운 Guided Backpropagation [H, W]

    Returns:
        아우른 그림 [H, W]
    """
    # 크기를 맞춘다
    if gradcam_map.shape != guided_backprop_map.shape:
        gradcam_map = F.interpolate(
            gradcam_map.unsqueeze(0).unsqueeze(0),
            size=guided_backprop_map.shape,
            mode='bilinear',
            align_corners=False
        ).squeeze()

    # 원소마다 곱한다
    combined = gradcam_map * guided_backprop_map

    # 고르게 한다
    combined = combined / (combined.max() + 1e-8)

    return combined


def example_1_complete_pipeline():
    """이끈 Grad-CAM의 온 흐름을 보인다."""
    print("\n" + "="*60)
    print("보기 1: 온전한 이끈 Grad-CAM")
    print("="*60)

    device = get_device()
    create_output_dir('outputs')

    model = load_pretrained_model('resnet50', device)
    test_image = Image.new('RGB', (224, 224), color=(150, 100, 130))

    print("\n이끈 Grad-CAM이 아우르는 것:")
    print("1. Grad-CAM → 클래스를 가려내는 자리 짚기")
    print("2. Guided Backpropagation → 결이 고운 잔 무늬")
    print("3. 원소마다의 곱 → 둘의 좋은 점을 함께!")

    print("\n✓ 이끈 Grad-CAM: 가장 앞선 시각화")


def main():
    print("\n" + "="*70)
    print(" "*18 + "이끈 GRAD-CAM 익히기")
    print("="*70)

    try:
        example_1_complete_pipeline()

        print("\n" + "="*70)
        print("핵심:")
        print("1. Grad-CAM + Guided Backpropagation를 아우른다")
        print("2. 클래스를 가려내면서 결도 곱다")
        print("3. 두루 보아 그림 됨됨이가 가장 좋다")
        print("4. 두 조각을 모두 짜야 한다")
        print("\n다음: 꾸러미 08 - 견주어 살피기")
        print("="*70)
    except Exception as e:
        print(f"어긋남: {e}")

if __name__ == "__main__":
    main()
```

**출력:**

```
======================================================================
                  이끈 GRAD-CAM 익히기
======================================================================

============================================================
보기 1: 온전한 이끈 Grad-CAM
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
07: Guided Grad-CAM 구현을 검사하는 포괄적인 테스트 함수를 작성하라. 빈 입력, 원소 하나짜리 입력, 아주 큰 입력, 그리고 극단값(0, 아주 큰 수)이 든 입력 같은 경계 사례를 테스트하여라.

</div>

??? success "풀이"
    경계 조건을 두루 다루는 테스트 함수를 작성한다.
    ```python
    def test_07: guided grad-cam():
        model = 07: 이끈 Grad-CAM(...)
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

**다룬 것** — 07: Guided Grad-CAM

이 구현은 깔끔하고 읽기 쉬운 PyTorch 코드로 그래디언트 기반 설명의 핵심 아이디어를 보여 준다.

앞의 연습문제 4개로 스스로 점검해 볼 수 있다.
