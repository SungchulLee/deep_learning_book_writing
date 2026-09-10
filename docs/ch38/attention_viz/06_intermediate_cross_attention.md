# 가운데 켜

가운데 켜: Seq2Seq 모형의 엇갈린 눈길 그리기. 이 꾸러미는 부호기-푸는 개 얼개의 엇갈린 눈길을 그리는 데 눈길을 둔다,

신경 그물이 무엇을 배우는지 아는 일은 믿음을 쌓고 모형의 벌레를 잡는 데 종요롭다. 이 꾸러미는 모형이 들임을 어떻게 다루고 어떻게 판단하는지 드러내는 눈길 그림 그리기 재주를 보이며, 그물의 움직임을 눈으로 보고 수로 재게 해 준다.

## 1. 코드

```python
"""
가운데 켜: Seq2Seq 모형의 엇갈린 눈길 그리기

이 꾸러미는 부호기-푸는 개 얼개의 엇갈린 눈길을 그리는 데 눈길을 두며,
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
    """부호기-푸는 개의 엇갈린 눈길을 그리는 개."""

    def __init__(self, figsize=(12, 8)):
        self.figsize = figsize

    def plot_cross_attention(self, 
                            cross_attention: torch.Tensor,
                            source_tokens: List[str],
                            target_tokens: List[str],
                            title: str = "엇갈린 눈길",
                            save_path: Optional[str] = None):
        """
        보내는 열과 받는 열 사이의 엇갈린 눈길을 그린다.

        매개변수:
        ----------
        cross_attention : torch.Tensor
            엇갈린 눈길 짐, 꼴: (받는 열 길이, 보내는 열 길이)
        source_tokens : list
            보내는 열의 낱말
        target_tokens : list
            받는 열의 낱말
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
            cbar_kws={'label': '눈길 짐'},
            ax=ax,
            vmin=0,
            vmax=1
        )

        ax.set_xlabel('보내는 쪽 (부호기)', fontsize=12, fontweight='bold')
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

        보내는 쪽의 어느 낱말이 받는 쪽의 어느 낱말과 맞물리는지 보인다.
        """
        if isinstance(cross_attention, torch.Tensor):
            cross_attention = cross_attention.cpu().numpy()

        # 받는 낱말마다 눈길이 가장 큰 곳을 찾는다
        max_alignments = np.argmax(cross_attention, axis=1)

        print("\n낱말 맞춤:")
        print("-" * 50)
        for target_idx, source_idx in enumerate(max_alignments):
            target_word = target_tokens[target_idx]
            source_word = source_tokens[source_idx]
            attention_weight = cross_attention[target_idx, source_idx]
            print(f"{target_word:15s} <- {source_word:15s} (짐: {attention_weight:.3f})")

        # 그림으로 보인다
        self.plot_cross_attention(cross_attention, source_tokens, target_tokens,
                                 "낱말 맞춤 행렬")

def example_translation_attention():
    """보기: 기계 옮김의 엇갈린 눈길."""
    print("=" * 70)
    print("엇갈린 눈길 그리기 보기")
    print("=" * 70)

    # 영어에서 프랑스말로 옮기는 보기
    source = ["I", "love", "machine", "learning"]
    target = ["J'", "adore", "l'", "apprentissage", "automatique"]

    # 지어낸 엇갈린 눈길을 만든다
    # 받는 열 길이 x 보내는 열 길이
    cross_attn = torch.zeros(len(target), len(source))

    # 그럴듯한 맞춤을 흉내낸다
    cross_attn[0, 0] = 0.8  # J' <- I
    cross_attn[1, 1] = 0.7  # adore <- love
    cross_attn[2, 2] = 0.3  # l' <- machine (관사)
    cross_attn[3, 2] = 0.6  # apprentissage <- machine
    cross_attn[3, 3] = 0.3  # apprentissage <- learning
    cross_attn[4, 3] = 0.7  # automatique <- learning

    # 바탕 눈길을 조금 더한다
    cross_attn += torch.rand(len(target), len(source)) * 0.05

    # 고르게 한다
    cross_attn = cross_attn / cross_attn.sum(dim=1, keepdim=True)

    # 그림으로 보인다
    viz = CrossAttentionVisualizer()
    viz.plot_alignment_matrix(cross_attn, source, target)

if __name__ == "__main__":
    torch.manual_seed(42)
    example_translation_attention()

    print("\n고갱이 깨침:")
    print("  - 엇갈린 눈길은 보내는 쪽과 받는 쪽의 얽힘을 보인다")
    print("  - 옮김과 지어내기를 알아보는 데 쓸모 있다")
    print("  - 낱말이 맞물리는 결을 드러낸다")
```

**출력:**

```
======================================================================
엇갈린 눈길 그리기 보기
======================================================================

낱말 맞춤:
--------------------------------------------------
J'              <- I               (짐: 0.882)
adore           <- love            (짐: 0.910)
l'              <- machine         (짐: 0.806)
apprentissage   <- machine         (짐: 0.618)
automatique     <- learning        (짐: 0.894)

고갱이 깨침:
  - 엇갈린 눈길은 보내는 쪽과 받는 쪽의 얽힘을 보인다
  - 옮김과 지어내기를 알아보는 데 쓸모 있다
  - 낱말이 맞물리는 결을 드러낸다
```

## 2. 논의

그림으로 보이기는 모형의 움직임을 알고 익힘의 탈을 짚어내는 데 큰 몫을 한다. 그리는 코드는 배운 나타냄, 모여 가는 결, 따짐 자를 들여다보게 해서 손에 잡히지 않던 셈을 눈에 보이게 한다.

여기서 보인 결은 더 까다로운 자리로도 자연스레 넓혀진다. 하이퍼파라미터, 얼개의 갈래, 여러 자료를 바꿔 가며 해 보면 이해가 깊어지고 모형 풀이하기에 대한 감이 몸에 붙는다.

## 연습문제

**연습문제 1.**
코드를 읽고 고갱이가 되는 설계 판단을 짚어라. 짜기에서 고른 것 셋을 들고, 저마다 왜 눈길 그림 그리기에 알맞은지 밝혀라.

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
가운데 켜 짜보기를 살피는 두루 갖춘 시험 함수를 써라. 빈 들임, 원소 하나짜리 들임, 아주 큰 들임, 그리고 끝자락 값(0, 아주 큰 수)이 든 들임 같은 가장자리 자리를 시험하여라.

??? success "연습문제 4 풀이"
    금 언저리 조건을 두루 건드리는 시험 함수를 짓는다.
    ```python
    def test_crossattentionvisualizer():
        model = 가운데 켜(...)
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

**다룬 것** — 가운데 켜

그림으로 보이기는 모형의 움직임을 알고 익힘의 탈을 짚어내는 데 큰 몫을 한다.

고갱이 갈래는 `CrossAttentionVisualizer`이며 앞의 연습문제 4개로 스스로 따져 볼 수 있다.
