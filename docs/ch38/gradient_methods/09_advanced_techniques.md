# 앞선 재주

09: 앞선 두드러짐 재주. 밝힘:

신경 그물이 무엇을 배우는지 아는 일은 믿음을 쌓고 모형의 벌레를 잡는 데 종요롭다. 이 꾸러미는 모형이 들임을 어떻게 다루고 어떻게 판단하는지 드러내는 기울기 바탕 풀이 재주를 보이며, 그물의 움직임을 눈으로 보고 수로 재게 해 준다.

## 1. 코드

```python
"""
09: 앞선 두드러짐 재주
==============================

어려움: 높음

밝힘:
가장 앞선 두드러짐 방법 들머리:
- 켜마다의 쓸모 퍼뜨리기(LRP)
- 눈길 굴리기(변환기에 씀)
- DeepLIFT
- 깊은 배움을 위한 SHAP

이 방법들은 기울기 바탕 길의 한계를 다룬다.

지은이: 가르치기 몫
"""

import torch
from utils import *

# ========================================================================
# 메인
# ========================================================================

def overview_advanced_methods():
    """앞선 재주 살펴보기."""

    print("\n" + "="*70)
    print(" "*15 + "앞선 두드러짐 재주")
    print("="*70)

    print("\n1. 켜마다의 쓸모 퍼뜨리기(LRP)")
    print("-" * 70)
    print("생각: 쓸모를 거꾸로 퍼뜨려 미루어 봄을 쪼갠다")
    print("꼴: R_i = Σ_j (z_ij / Σ_k z_kj) R_j")
    print("나은 점: 지켜짐 됨됨이를 채운다")
    print("쓰일 자리: 쓸모를 정확히 쪼개야 할 때")

    print("\n2. 눈길 굴리기(변환기)")
    print("-" * 70)
    print("생각: 켜마다의 눈길 그림을 한데 모은다")
    print("꼴: Att = Π_l Att^(l), Att^(l)은 켜마다의 눈길")
    print("나은 점: 변환기가 어디를 보는지 그린다")
    print("쓰일 자리: 눈 변환기, BERT 등")

    print("\n3. DEEPLIFT")
    print("-" * 70)
    print("생각: 살아남을 견줌 살아남과 견준다")
    print("꼴: 밑금과의 차이로 몫을 매긴다")
    print("나은 점: 기울기가 잦아든 자리를 더 잘 다룬다")
    print("쓰일 자리: 기울기가 사라지거나 터질 때")

    print("\n4. SHAP(더해지는 섀플리 풀이)")
    print("-" * 70)
    print("생각: 놀이 이론으로 몫을 매긴다")
    print("꼴: 함께 하는 놀이 이론의 섀플리 값")
    print("나은 점: 이론으로 가장 좋고 고르게 몫을 매긴다")
    print("쓰일 자리: 고름을 증명할 수 있는 풀이가 있어야 할 때")

    print("\n5. DECONVNET")
    print("-" * 70)
    print("생각: 이끈 되짚기와 비슷하되 ReLU 다루기가 다르다")
    print("쓰일 자리: 이끈 되짚기를 갈음할 때")

    print("\n" + "="*70)
    print("짜기에 도움 되는 곳:")
    print("-" * 70)
    print("• Captum (PyTorch): https://captum.ai/")
    print("• SHAP: https://github.com/slundberg/shap")
    print("• LRP Toolbox: https://github.com/sebastian-lapuschkin/lrp_toolbox")
    print("• 변환기 풀이하기: 눈길 굴리기 논문")
    print("="*70)


def example_1_when_to_use_what():
    """알맞은 방법을 고르는 길잡이."""

    print("\n" + "="*60)
    print("판단 나무: 어느 방법을 쓸까?")
    print("="*60)

    print("\n물음 1: 무엇을 하려는가?")
    print("  A) 빠른 벌레잡기 → 맨 기울기")
    print("  B) 논문에 실을 그림 → 이끈 Grad-CAM")
    print("  C) 이론 보장 → 쌓은 기울기나 SHAP")
    print("  D) 변환기 알아보기 → 눈길 굴리기")

    print("\n물음 2: 모형 갈래는?")
    print("  A) CNN → Grad-CAM, 이끈 Grad-CAM")
    print("  B) 변환기 → 눈길 굴리기")
    print("  C) 아무거나 → 쌓은 기울기, SHAP")

    print("\n물음 3: 무엇에 매였는가?")
    print("  A) 빠르기 → 맨 기울기")
    print("  B) 됨됨이 → SmoothGrad, 쌓은 기울기")
    print("  C) 결 고움 → 이끈 방법들")

    print("\n물음 4: 어디에 쓰는가?")
    print("  A) 학술 논문 → 쌓은 기울기(인용이 많다)")
    print("  B) 보임/발표 → 이끈 Grad-CAM(눈에 잘 든다)")
    print("  C) 서비스에 올림 → Grad-CAM(빠르다)")
    print("  D) 벌레잡기 → 맨 기울기(빨리 되풀이한다)")


def main():
    print("\n" + "="*70)
    print(" "*15 + "앞선 재주 살펴보기")
    print("="*70)

    try:
        overview_advanced_methods()
        example_1_when_to_use_what()

        print("\n" + "="*70)
        print("잘하셨습니다!")
        print("="*70)
        print("\n두드러짐 그림 익히기 묶음을 다 마쳤습니다!")
        print("\n배운 것:")
        print("✓ 기울기 바탕 방법(맨 기울기, 기울기×들임)")
        print("✓ 잡음 줄이기(SmoothGrad)")
        print("✓ 길 따라 쌓기(쌓은 기울기)")
        print("✓ 자리 짚기(Grad-CAM)")
        print("✓ 결 고움(이끈 되짚기)")
        print("✓ 아우른 방법(이끈 Grad-CAM)")
        print("✓ 방법 고르기와 견주기")
        print("✓ 앞선 재주 살펴보기")

        print("\n다음 걸음:")
        print("1. 제 모형에 이 방법들을 걸어 본다")
        print("2. Captum 곳집에서 더 많은 재주를 살핀다")
        print("3. 처음 논문을 읽어 더 깊이 안다")
        print("4. 여러 얼개로 해 본다")

        print("\n즐겁게 풀이하시길! 🎉")
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
눈길 짐 뒤(값과 곱하기 앞)에 드롭아웃 켜를 더하여라. 익히는 동안 드롭아웃 비율을 0.1으로 잡아라. 눈길 드롭아웃이 정칙화에 왜 도움이 되는지 밝혀라.

??? success "연습문제 2 풀이"
    `__init__`에 `self.attn_dropout = nn.Dropout(0.1)`을 더하고 소프트맥스 뒤에 건다. `attn_weights = self.attn_dropout(F.softmax(scores, dim=-1))`. 눈길 드롭아웃은 익히는 동안 눈길 짐 몇몇을 아무렇게나 0으로 만들어, 모형이 특정 낱말끼리의 얽힘에 지나치게 기대는 것을 막는다. 그래서 모형이 눈길을 더 고루 나누고 더 든든한 나타냄을 배우게 되는데, 여느 드롭아웃이 신경 세포끼리 함께 굳는 것을 막는 것과 같은 결이다.

---

**연습문제 3.**
제 눈길의 셈 복잡도를 열 길이 $n$과 모형 차원 $d$의 함수로 밝혀라. 이것이 왜 긴 열에 Longformer이나 Linformer 같은 얼개를 부르는가?

??? success "연습문제 3 풀이"
    여느 제 눈길은 $n \times n$ 눈길 행렬을 셈하므로 때가 $O(n^2 d)$이고 눈길 짐에 드는 기억이 $O(n^2)$이다. 열이 길면(보기로 $n = 4096$) 감당할 수 없다. Longformer는 그 자리 미끄럼 창 눈길($O(n \cdot w \cdot d)$, $w$은 창 크기)과 고른 낱말에 대한 성긴 온 세상 눈길을 아울러 쓴다. Linformer는 열쇠와 값을 낮은 차원 $k \ll n$으로 쏘아 복잡도를 $O(n \cdot k \cdot d)$으로 줄인다. 둘 다 나타내는 힘을 얼마쯤 내주고 긴 들임에서의 쓸모를 얻는다.

---
**연습문제 4.**
앞선 재주 짜보기를 살피는 두루 갖춘 시험 함수를 써라. 빈 들임, 원소 하나짜리 들임, 아주 큰 들임, 그리고 끝자락 값(0, 아주 큰 수)이 든 들임 같은 가장자리 자리를 시험하여라.

??? success "연습문제 4 풀이"
    금 언저리 조건을 두루 건드리는 시험 함수를 짓는다.
    ```python
    def test_advanced techniques():
        model = 앞선 재주(...)
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

**다룬 것** — 앞선 재주

이 짜보기는 깔끔하고 읽기 쉬운 PyTorch 코드로 기울기 바탕 풀이의 고갱이가 되는 생각을 보여 준다.

앞의 연습문제 4개로 스스로 따져 볼 수 있다.
