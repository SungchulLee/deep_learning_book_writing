# 고급 기법

그래디언트 기반 방법 너머의 설명 기법 다섯 — LRP(Layer-wise Relevance Propagation), Attention Rollout, DeepLIFT, SHAP, DeconvNet — 을 짧게 훑고, 상황에 맞는 방법을 고르는 길잡이를 찍는 쪽이다. 계산은 하지 않고 설명 문장만 출력한다.

뉴럴 네트워크가 무엇을 배우는지 아는 일은 신뢰를 쌓고 모델을 디버깅하는 데 중요하다. 이 모듈은 모델이 입력을 어떻게 처리하고 어떻게 판단하는지 드러내는 그래디언트 기반 설명 기법을 보이며, 네트워크의 행동을 눈으로 보고 수로 재게 해 준다.

## 1. 코드

```python
"""
09: 고급 샐리언시 기법
==============================

난이도: 높음

설명:
가장 발전된 샐리언시 방법 소개:
- Layer-wise Relevance Propagation(LRP)
- Attention Rollout(트랜스포머에 씀)
- DeepLIFT
- 딥러닝을 위한 SHAP

이 방법들은 그래디언트 기반 방법의 한계를 다룬다.

지은이: 교육용 자료
"""

import torch

# ========================================================================
# 메인
# ========================================================================

def overview_advanced_methods():
    """고급 기법 개관."""

    print("\n" + "="*70)
    print(" "*15 + "고급 샐리언시 기법")
    print("="*70)

    print("\n1. Layer-wise Relevance Propagation(LRP)")
    print("-" * 70)
    print("아이디어: 관련도를 거꾸로 전파해 예측을 분해한다")
    print("식: R_i = Σ_j (z_ij / Σ_k z_kj) R_j")
    print("장점: 보존 성질을 만족한다")
    print("쓰임: 관련도를 정확히 분해해야 할 때")

    print("\n2. Attention Rollout(트랜스포머)")
    print("-" * 70)
    print("아이디어: 레이어마다의 어텐션 맵을 한데 결합한다")
    print("식: Att = Π_l Att^(l), Att^(l)은 레이어마다의 어텐션")
    print("장점: 트랜스포머가 어디를 보는지 시각화한다")
    print("쓰임: 비전 트랜스포머, BERT 등")

    print("\n3. DEEPLIFT")
    print("-" * 70)
    print("아이디어: 활성값을 기준 활성값과 비교한다")
    print("방식: 베이스라인과의 차이로 기여도를 매긴다")
    print("장점: 그래디언트가 포화된 영역을 더 잘 다룬다")
    print("쓰임: 그래디언트가 소실되거나 폭발할 때")

    print("\n4. SHAP(SHapley Additive exPlanations)")
    print("-" * 70)
    print("아이디어: 게임 이론으로 기여도를 매긴다")
    print("방식: 협력 게임 이론의 섀플리 값")
    print("장점: 이론적으로 최적이고 공정하게 기여도를 매긴다")
    print("쓰임: 공정성이 증명된 설명이 필요할 때")

    print("\n5. DECONVNET")
    print("-" * 70)
    print("아이디어: Guided Backpropagation과 비슷하되 ReLU 처리가 다르다")
    print("쓰임: Guided Backpropagation을 대신할 때")

    print("\n" + "="*70)
    print("구현에 도움 되는 자료:")
    print("-" * 70)
    print("• Captum (PyTorch): https://captum.ai/")
    print("• SHAP: https://github.com/slundberg/shap")
    print("• LRP Toolbox: https://github.com/sebastian-lapuschkin/lrp_toolbox")
    print("• 트랜스포머 해석: Attention Rollout 논문")
    print("="*70)


def example_1_when_to_use_what():
    """알맞은 방법을 고르는 가이드."""

    print("\n" + "="*60)
    print("의사결정 트리: 어느 방법을 쓸까?")
    print("="*60)

    print("\n질문 1: 무엇을 하려는가?")
    print("  A) 빠른 디버깅 → 바닐라 그래디언트")
    print("  B) 논문에 실을 그림 → Guided Grad-CAM")
    print("  C) 이론적 보장 → Integrated Gradients나 SHAP")
    print("  D) 트랜스포머 해석 → Attention Rollout")

    print("\n질문 2: 모델 종류는?")
    print("  A) CNN → Grad-CAM, Guided Grad-CAM")
    print("  B) 트랜스포머 → Attention Rollout")
    print("  C) 무엇이든 → Integrated Gradients, SHAP")

    print("\n질문 3: 제약은 무엇인가?")
    print("  A) 속도 → 바닐라 그래디언트")
    print("  B) 품질 → SmoothGrad, Integrated Gradients")
    print("  C) 세밀한 디테일 → Guided 방법들")

    print("\n질문 4: 어디에 쓰는가?")
    print("  A) 학술 논문 → Integrated Gradients(인용이 많다)")
    print("  B) 데모/발표 → Guided Grad-CAM(눈에 잘 띈다)")
    print("  C) 서비스 배포 → Grad-CAM(빠르다)")
    print("  D) 디버깅 → 바닐라 그래디언트(빨리 반복한다)")


def main():
    print("\n" + "="*70)
    print(" "*15 + "고급 기법 개관")
    print("="*70)

    try:
        overview_advanced_methods()
        example_1_when_to_use_what()

        print("\n" + "="*70)
        print("잘하셨습니다!")
        print("="*70)
        print("\nSaliency Map 튜토리얼 시리즈를 모두 마쳤습니다!")
        print("\n배운 것:")
        print("✓ 그래디언트 기반 방법(바닐라 그래디언트, 그래디언트×입력)")
        print("✓ 잡음 줄이기(SmoothGrad)")
        print("✓ 경로 적분(Integrated Gradients)")
        print("✓ 위치 파악(Grad-CAM)")
        print("✓ 세밀한 디테일(Guided Backpropagation)")
        print("✓ 결합 방법(Guided Grad-CAM)")
        print("✓ 방법 선택과 비교")
        print("✓ 고급 기법 개관")

        print("\n다음 스텝:")
        print("1. 자신의 모델에 이 방법들을 적용해 본다")
        print("2. Captum 라이브러리에서 더 많은 기법을 살핀다")
        print("3. 원 논문을 읽어 더 깊이 이해한다")
        print("4. 여러 구조로 실험해 본다")

        print("\n즐겁게 해석하시길! 🎉")
        print("="*70)

    except Exception as e:
        print(f"오류: {e}")

if __name__ == "__main__":
    main()
```

??? note "전체 출력 (101줄)"

    ```

    ======================================================================
                   고급 기법 개관
    ======================================================================

    ======================================================================
                   고급 샐리언시 기법
    ======================================================================

    1. Layer-wise Relevance Propagation(LRP)
    ----------------------------------------------------------------------
    아이디어: 관련도를 거꾸로 전파해 예측을 분해한다
    식: R_i = Σ_j (z_ij / Σ_k z_kj) R_j
    장점: 보존 성질을 만족한다
    쓰임: 관련도를 정확히 분해해야 할 때

    2. Attention Rollout(트랜스포머)
    ----------------------------------------------------------------------
    아이디어: 레이어마다의 어텐션 맵을 한데 결합한다
    식: Att = Π_l Att^(l), Att^(l)은 레이어마다의 어텐션
    장점: 트랜스포머가 어디를 보는지 시각화한다
    쓰임: 비전 트랜스포머, BERT 등

    3. DEEPLIFT
    ----------------------------------------------------------------------
    아이디어: 활성값을 기준 활성값과 비교한다
    방식: 베이스라인과의 차이로 기여도를 매긴다
    장점: 그래디언트가 포화된 영역을 더 잘 다룬다
    쓰임: 그래디언트가 소실되거나 폭발할 때

    4. SHAP(SHapley Additive exPlanations)
    ----------------------------------------------------------------------
    아이디어: 게임 이론으로 기여도를 매긴다
    방식: 협력 게임 이론의 섀플리 값
    장점: 이론적으로 최적이고 공정하게 기여도를 매긴다
    쓰임: 공정성이 증명된 설명이 필요할 때

    5. DECONVNET
    ----------------------------------------------------------------------
    아이디어: Guided Backpropagation과 비슷하되 ReLU 처리가 다르다
    쓰임: Guided Backpropagation을 대신할 때

    ======================================================================
    구현에 도움 되는 자료:
    ----------------------------------------------------------------------
    • Captum (PyTorch): https://captum.ai/
    • SHAP: https://github.com/slundberg/shap
    • LRP Toolbox: https://github.com/sebastian-lapuschkin/lrp_toolbox
    • 트랜스포머 해석: Attention Rollout 논문
    ======================================================================

    ============================================================
    의사결정 트리: 어느 방법을 쓸까?
    ============================================================

    질문 1: 무엇을 하려는가?
      A) 빠른 디버깅 → 바닐라 그래디언트
      B) 논문에 실을 그림 → Guided Grad-CAM
      C) 이론적 보장 → Integrated Gradients나 SHAP
      D) 트랜스포머 해석 → Attention Rollout

    질문 2: 모델 종류는?
      A) CNN → Grad-CAM, Guided Grad-CAM
      B) 트랜스포머 → Attention Rollout
      C) 무엇이든 → Integrated Gradients, SHAP

    질문 3: 제약은 무엇인가?
      A) 속도 → 바닐라 그래디언트
      B) 품질 → SmoothGrad, Integrated Gradients
      C) 세밀한 디테일 → Guided 방법들

    질문 4: 어디에 쓰는가?
      A) 학술 논문 → Integrated Gradients(인용이 많다)
      B) 데모/발표 → Guided Grad-CAM(눈에 잘 띈다)
      C) 서비스 배포 → Grad-CAM(빠르다)
      D) 디버깅 → 바닐라 그래디언트(빨리 반복한다)

    ======================================================================
    잘하셨습니다!
    ======================================================================

    Saliency Map 튜토리얼 시리즈를 모두 마쳤습니다!

    배운 것:
    ✓ 그래디언트 기반 방법(바닐라 그래디언트, 그래디언트×입력)
    ✓ 잡음 줄이기(SmoothGrad)
    ✓ 경로 적분(Integrated Gradients)
    ✓ 위치 파악(Grad-CAM)
    ✓ 세밀한 디테일(Guided Backpropagation)
    ✓ 결합 방법(Guided Grad-CAM)
    ✓ 방법 선택과 비교
    ✓ 고급 기법 개관

    다음 스텝:
    1. 자신의 모델에 이 방법들을 적용해 본다
    2. Captum 라이브러리에서 더 많은 기법을 살핀다
    3. 원 논문을 읽어 더 깊이 이해한다
    4. 여러 구조로 실험해 본다

    즐겁게 해석하시길! 🎉
    ======================================================================
    ```
## 2. 논의

이 프로그램은 모델을 부르지 않고 문장만 찍으므로 끝까지 돈다. 출력에 적힌 식 두 개는 읽을 때 짚어 둘 것이 있다. LRP의 `R_i = Σ_j (z_ij / Σ_k z_kj) R_j` 는 가장 단순한 LRP-0 규칙으로, 위 레이어의 관련도 $R_j$ 를 아래 뉴런의 기여 $z_{ij}$ 비율대로 나누어 주므로 레이어마다 관련도의 합이 보존된다. Attention Rollout의 `Att = Π_l Att^(l)` 은 레이어마다의 어텐션 행렬을 곱한 것인데, 원래 방법은 잔차 연결을 반영하려고 각 행렬에 단위행렬을 더해 다시 정규화한 뒤 곱한다. 출력의 식은 그 단계를 생략했다.

여기서 보인 기법은 더 까다로운 상황으로도 자연스럽게 확장된다. 하이퍼파라미터, 구조의 종류, 여러 데이터를 바꿔 가며 해 보면 이해가 깊어지고 모델 해석에 대한 감이 생긴다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
이 쪽이 소개하는 방법 가운데 셋을 골라, 저마다 그래디언트 기반 설명의 어떤 한계를 다루는지 설명하라.

</div>

??? success "풀이"
    (1) **LRP** -- 그래디언트는 한 점의 국소 민감도라 기여의 합이 예측값과 이어지지 않는다. LRP는 출력의 관련도를 레이어마다 아래로 나누어 주면서 합을 보존하므로, 입력 피처의 관련도를 모두 더하면 출력이 된다. (2) **DeepLIFT** -- 포화 영역에서는 그래디언트가 0이어도 입력이 출력을 바꾼다. DeepLIFT는 기울기 대신 베이스라인과의 차이를 전파하므로, 기울기가 0인 곳에서도 기여가 사라지지 않는다. (3) **Attention Rollout** -- 트랜스포머에서는 한 레이어의 어텐션만 보면 앞 레이어에서 이미 섞인 정보를 놓친다. 레이어마다의 어텐션 행렬(잔차 연결을 반영해 단위행렬을 더한 것)을 곱해 입력 토큰까지 거슬러 올라간 흐름을 본다. 다만 셋 모두 대가가 있다. LRP와 DeepLIFT는 레이어마다 전파 규칙을 정해야 하고 구현 불변성을 지키지 못하며, Attention Rollout은 값 벡터의 크기와 MLP 레이어를 무시한다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
어텐션 웨이트 뒤(값과 곱하기 전)에 드롭아웃 레이어를 더하여라. 학습하는 동안 드롭아웃 비율을 0.1로 잡아라. 어텐션 드롭아웃이 정칙화에 왜 도움이 되는지 설명하라.

</div>

??? success "풀이"
    `__init__`에 `self.attn_dropout = nn.Dropout(0.1)`을 더하고 소프트맥스 뒤에 적용한다. `attn_weights = self.attn_dropout(F.softmax(scores, dim=-1))`. 어텐션 드롭아웃은 학습하는 동안 어텐션 웨이트 일부를 무작위로 0으로 만들어, 모델이 특정 토큰 사이의 상관에 지나치게 기대는 것을 막는다. 그래서 모델이 어텐션을 더 고르게 나누고 더 강건한 표현을 배우게 되는데, 보통의 드롭아웃이 뉴런끼리의 공적응(co-adaptation)을 막는 것과 같은 원리다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
셀프 어텐션의 계산 복잡도를 시퀀스 길이 $n$과 모델 차원 $d$의 함수로 밝혀라. 이것이 왜 긴 시퀀스에 Longformer나 Linformer 같은 구조를 필요로 하는가?

</div>

??? success "풀이"
    보통의 셀프 어텐션은 $n \times n$ 어텐션 행렬을 계산하므로 시간이 $O(n^2 d)$이고 어텐션 웨이트에 드는 메모리가 $O(n^2)$이다. 시퀀스가 길면(예를 들어 $n = 4096$) 감당할 수 없다. Longformer는 국소 슬라이딩 윈도 어텐션($O(n \cdot w \cdot d)$, $w$는 윈도 크기)과 고른 토큰에 대한 희소 전역 어텐션을 함께 쓴다. Linformer는 키와 값을 낮은 차원 $k \ll n$으로 사영해 복잡도를 $O(n \cdot k \cdot d)$로 줄인다. 둘 다 표현력을 얼마쯤 내주고 긴 입력에서의 효율을 얻는다.

---
<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
이 쪽이 소개하는 고급 기법의 구현을 검사하는 포괄적인 테스트 함수를 작성하라. 빈 입력, 원소 하나짜리 입력, 아주 큰 입력, 그리고 극단값(0, 아주 큰 수)이 든 입력 같은 경계 사례를 테스트하여라.

</div>

??? success "풀이"
    경계 조건을 두루 다루는 테스트 함수를 작성한다.
    ```python
    def test_advanced techniques():
        model = 앞선 기법(...)
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

**다룬 것** — 고급 기법

LRP, Attention Rollout, DeepLIFT, SHAP, DeconvNet이 그래디언트 기반 방법의 어떤 한계를 다루는지 훑고, 목적·모델·제약·쓰임에 따라 방법을 고르는 길잡이를 보았다.

앞의 연습문제 4개로 스스로 점검해 볼 수 있다.
