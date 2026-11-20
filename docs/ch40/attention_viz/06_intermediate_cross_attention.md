# 중간레이어 크로스 어텐션

가운데 레이어: Seq2Seq 모델의 크로스 어텐션 그리기. 이 모듈은 인코더-디코더 구조의 크로스 어텐션을 그리는 데 어텐션을 둔다,

뉴럴 네트워크가 무엇을 배우는지 아는 일은 신뢰를 쌓고 모델을 디버깅하는 데 중요하다. 이 모듈은 모델이 입력을 어떻게 처리하고 어떻게 판단하는지 드러내는 어텐션 시각화 기법을 보이며, 네트워크의 행동을 눈으로 보고 수로 재게 해 준다.

## 1. 코드

```python
"""
가운데 레이어: Seq2Seq 모형의 엇갈린 어텐션 그리기

이 꾸러미는 인코더-푸는 개 구조의 엇갈린 어텐션을 그리는 데 어텐션을 두며,
기계 옮김, 간추리기, seq2seq 일감에 특히 쓸모 있다.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional, Tuple

# ========================================================================
# 메인
# ========================================================================

class CrossAttentionVisualizer:
    """인코더-푸는 개의 엇갈린 어텐션을 그리는 개."""

    def __init__(self, figsize=(12, 8)):
        self.figsize = figsize

    def plot_cross_attention(self, 
                            cross_attention: torch.Tensor,
                            source_tokens: List[str],
                            target_tokens: List[str],
                            title: str = "엇갈린 어텐션",
                            save_path: Optional[str] = None):
        """
        보내는 열과 받는 열 사이의 엇갈린 어텐션을 그린다.

        파라메터:
        ----------
        cross_attention : torch.Tensor
            엇갈린 어텐션 짐, 형태: (받는 열 길이, 보내는 열 길이)
        source_tokens : list
            보내는 열의 단어
        target_tokens : list
            받는 열의 단어
        """
        if isinstance(cross_attention, torch.Tensor):
            cross_attention = cross_attention.cpu().numpy()

        fig, ax = plt.subplots(figsize=self.figsize)

        sns.heatmap(
            cross_attention,
            xticklabels=source_tokens,
            yticklabels=target_tokens,
            cmap='YlOrRd',
            square=False,
            cbar_kws={'label': '어텐션 짐'},
            ax=ax,
            vmin=0,
            vmax=1
        )

        ax.set_xlabel('보내는 쪽 (인코더)', fontsize=12, fontweight='bold')
        ax.set_ylabel('받는 쪽 (푸는 개)', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.show()

    def plot_alignment_matrix(self,
                             cross_attention: torch.Tensor,
                             source_tokens: List[str],
                             target_tokens: List[str]):
        """
        기계 옮김에서 흔히 쓰는 맞춤 행렬 그림을 만든다.

        보내는 쪽의 어느 단어가 받는 쪽의 어느 단어와 맞물리는지 보인다.
        """
        if isinstance(cross_attention, torch.Tensor):
            cross_attention = cross_attention.cpu().numpy()

        # 받는 단어마다 어텐션이 가장 큰 곳을 찾는다
        max_alignments = np.argmax(cross_attention, axis=1)

        print("\n단어 맞춤:")
        print("-" * 50)
        for target_idx, source_idx in enumerate(max_alignments):
            target_word = target_tokens[target_idx]
            source_word = source_tokens[source_idx]
            attention_weight = cross_attention[target_idx, source_idx]
            print(f"{target_word:15s} <- {source_word:15s} (짐: {attention_weight:.3f})")

        # 그림으로 보인다
        self.plot_cross_attention(cross_attention, source_tokens, target_tokens,
                                 "단어 맞춤 행렬")

def example_translation_attention():
    """보기: 기계 옮김의 엇갈린 어텐션."""
    print("=" * 70)
    print("엇갈린 어텐션 그리기 보기")
    print("=" * 70)

    # 영어에서 프랑스말로 옮기는 보기
    source = ["I", "love", "machine", "learning"]
    target = ["J'", "adore", "l'", "apprentissage", "automatique"]

    # 지어낸 엇갈린 어텐션을 만든다
    # 받는 열 길이 x 보내는 열 길이
    cross_attn = torch.zeros(len(target), len(source))

    # 그럴듯한 맞춤을 흉내낸다
    cross_attn[0, 0] = 0.8  # J' <- I
    cross_attn[1, 1] = 0.7  # adore <- love
    cross_attn[2, 2] = 0.3  # l' <- machine (관사)
    cross_attn[3, 2] = 0.6  # apprentissage <- machine
    cross_attn[3, 3] = 0.3  # apprentissage <- learning
    cross_attn[4, 3] = 0.7  # automatique <- learning

    # 바탕 어텐션을 조금 더한다
    cross_attn += torch.rand(len(target), len(source)) * 0.05

    # 고르게 한다
    cross_attn = cross_attn / cross_attn.sum(dim=1, keepdim=True)

    # 그림으로 보인다
    viz = CrossAttentionVisualizer()
    viz.plot_alignment_matrix(cross_attn, source, target)

if __name__ == "__main__":
    torch.manual_seed(42)
    example_translation_attention()

    print("\n핵심 깨침:")
    print("  - 엇갈린 어텐션은 보내는 쪽과 받는 쪽의 얽힘을 보인다")
    print("  - 옮김과 지어내기를 알아보는 데 쓸모 있다")
    print("  - 단어가 맞물리는 결을 드러낸다")
```

**출력:**

```
======================================================================
엇갈린 어텐션 그리기 보기
======================================================================

단어 맞춤:
--------------------------------------------------
J'              <- I               (짐: 0.882)
adore           <- love            (짐: 0.910)
l'              <- machine         (짐: 0.806)
apprentissage   <- machine         (짐: 0.618)
automatique     <- learning        (짐: 0.894)

핵심 깨침:
  - 엇갈린 어텐션은 보내는 쪽과 받는 쪽의 얽힘을 보인다
  - 옮김과 지어내기를 알아보는 데 쓸모 있다
  - 단어가 맞물리는 결을 드러낸다
```

## 2. 논의

시각화는 모델의 행동을 이해하고 학습의 문제를 탐지하는 데 큰 역할을 한다. 시각화 코드는 학습된 표현, 수렴하는 패턴, 평가 지표를 들여다보게 해서 추상적인 계산을 눈에 보이게 한다.

여기서 보인 기법은 더 까다로운 상황으로도 자연스럽게 확장된다. 하이퍼파라미터, 구조의 종류, 여러 데이터를 바꿔 가며 해 보면 이해가 깊어지고 모델 해석에 대한 감이 생긴다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
코드를 읽고 핵심 설계 선택을 짚어라. 구현에서 고른 것 셋을 들고, 저마다 왜 어텐션 시각화에 알맞은지 설명하라.

</div>

??? success "풀이"
    설계 선택은 구현마다 다르나 흔히 이런 것이 있다. (1) 활성화 함수 선택 -- ReLU 계열은 그래디언트가 소실되지 않아 학습이 빠르다. (2) 정규화 전략 -- 배치 정규화가 내부 공변량 이동을 줄여 학습을 안정시킨다. (3) 잔차 연결 -- 있으면 스킵 경로를 주어 깊은 네트워크에서 그래디언트가 흐르게 한다. 각 선택은 표현력, 계산 비용, 학습 안정성 사이의 트레이드오프를 드러낸다.

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
    보통의 셀프 어텐션은 $n \times n$ 어텐션 행렬을 계산하므로 시간이 $O(n^2 d)$이고 어텐션 웨이트에 드는 메모리가 $O(n^2)$이다. 시퀀스가 길면(예를 들어 $n = 4096$) 감당할 수 없다. Longformer는 국소 슬라이딩 윈도 어텐션($O(n \cdot w \cdot d)$, $w$은 윈도 크기)과 고른 토큰에 대한 성긴 전역 어텐션을 함께 쓴다. Linformer는 키와 값을 낮은 차원 $k \ll n$으로 사영해 복잡도를 $O(n \cdot k \cdot d)$으로 줄인다. 둘 다 표현력을 얼마쯤 내주고 긴 입력에서의 효율을 얻는다.

---
<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
가운데 레이어 구현을 검사하는 포괄적인 테스트 함수를 작성하라. 빈 입력, 원소 하나짜리 입력, 아주 큰 입력, 그리고 극단값(0, 아주 큰 수)이 든 입력 같은 경계 사례를 테스트하여라.

</div>

??? success "풀이"
    경계 조건을 두루 다루는 테스트 함수를 작성한다.
    ```python
    def test_crossattentionvisualizer():
        model = 가운데 레이어(...)
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

**다룬 것** — 가운데 레이어

시각화는 모델의 행동을 이해하고 학습의 문제를 탐지하는 데 큰 역할을 한다.

핵심 클래스는 `CrossAttentionVisualizer`이며 앞의 연습문제 4개로 스스로 따져 볼 수 있다.
