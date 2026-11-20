# 07: Guided Grad-CAM

Guided Grad-CAM으로 두 방법의 장점을 결합하는 쪽이다. 클래스를 구별하지만 해상도가 낮은 Grad-CAM 맵을 입력 크기로 키운 뒤, 해상도는 높지만 클래스를 잘 구별하지 못하는 Guided Backpropagation 맵과 원소별로 곱한다.

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

위 출력에서 보듯 이 프로그램은 첫 보기에서 멈춘다. `get_device` 같은 도우미를 `from utils import *` 로 가져오는데, 그 `utils` 모듈이 없어 `NameError` 가 났다. 게다가 `example_1_complete_pipeline` 은 모델과 이미지를 준비한 뒤 설명 문장만 찍을 뿐, `compute_guided_gradcam` 을 부르지 않는다. 실제 계산은 이 함수 하나에 들어 있다.

`compute_guided_gradcam` 은 Guided Grad-CAM의 정의를 그대로 옮긴다. 두 맵의 크기가 다르면 Grad-CAM 맵을 쌍선형 보간으로 키우고, 원소별로 곱한 뒤 최댓값으로 나누어 [0, 1] 범위로 맞춘다. Grad-CAM 맵과 Guided Backpropagation 맵을 만드는 일은 앞의 두 쪽(Grad-CAM, Guided Backpropagation)의 코드에 맡긴다.

여기서 보인 기법은 더 까다로운 상황으로도 자연스럽게 확장된다. 하이퍼파라미터, 구조의 종류, 여러 데이터를 바꿔 가며 해 보면 이해가 깊어지고 모델 해석에 대한 감이 생긴다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
코드를 읽고 핵심 설계 선택을 짚어라. 구현에서 고른 것 셋을 들고, 저마다 왜 그래디언트 기반 설명에 알맞은지 설명하라.

</div>

??? success "풀이"
    `compute_guided_gradcam` 에서 셋을 들 수 있다. (1) **Grad-CAM 맵을 키워 맞추기** -- 7×7 같은 거친 맵을 쌍선형 보간으로 입력 크기까지 키워야 픽셀 단위 맵과 곱할 수 있다. (2) **원소별 곱** -- 곱셈은 두 맵이 모두 큰 곳만 남기는 「그리고」 연산처럼 작동한다. Grad-CAM이 클래스와 무관하다고 본 영역의 세밀한 무늬는 지워지고, 그 클래스의 영역 안의 무늬만 남는다. (3) **최댓값으로 나누기** -- 결과를 [0, 1]로 맞추어 표시하기 쉽게 한다. 다만 곱하기 전에 두 맵의 척도를 맞추지 않으면 한쪽이 결과를 지배할 수 있다.

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
    흔한 실패 유형은 이렇다. (1) **두 맵의 클래스가 어긋남** -- Grad-CAM과 Guided Backpropagation을 서로 다른 타깃 클래스로 계산하면 곱이 무의미해진다. 두 호출에 같은 `target_class` 를 넘겼는지 확인해 탐지하고, 한 함수 안에서 클래스를 한 번 정해 둘 다에 넘겨 고친다. (2) **Guided Backpropagation 훅이 Grad-CAM을 오염시킴** -- 같은 모델에 ReLU 역전파 훅을 건 채로 Grad-CAM을 계산하면 Grad-CAM의 그래디언트도 수정된 규칙을 따른다. Grad-CAM 맵이 훅 유무에 따라 달라지는지 견주어 탐지하고, Grad-CAM을 먼저 계산하거나 Guided Backpropagation 계산이 끝나면 훅을 바로 지워 고친다.

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
