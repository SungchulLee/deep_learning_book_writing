# 앞선 켜

앞선 켜: 눈길 흐름 살피기. 눈길 짐과 기울기 소식을 아울러 알아본다

신경 그물이 무엇을 배우는지 아는 일은 믿음을 쌓고 모형의 벌레를 잡는 데 종요롭다. 이 꾸러미는 모형이 들임을 어떻게 다루고 어떻게 판단하는지 드러내는 눈길 그림 그리기 재주를 보이며, 그물의 움직임을 눈으로 보고 수로 재게 해 준다.

## 코드

```python
"""
앞선 켜: 눈길 흐름 살피기

눈길 짐과 기울기 소식을 아울러, 어느 눈길 이음이 미루어 봄에
가장 중요한지 알아본다.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional, Tuple

# ========================================================================
# 메인
# ========================================================================

class AttentionFlowAnalyzer:
    """
    기울기로 눈길 흐름을 살피는 개.

    눈길 짐만으로는 어느 이음이 중요한지 알 수 없다.
    눈길과 기울기를 아우르면 종요로운 길을 짚어낼 수 있다.

    꼴:
    -------
    흐름 = 눈길 × |기울기|

    여기서 기울기는 내놓기에 대한 것이다.
    """

    def __init__(self):
        pass

    def compute_attention_flow(self,
                              attention: torch.Tensor,
                              gradients: torch.Tensor) -> torch.Tensor:
        """
        눈길과 기울기를 아울러 눈길 흐름을 셈한다.

        매개변수:
        ----------
        attention : torch.Tensor
            눈길 짐, 꼴: (열 길이, 열 길이)
        gradients : torch.Tensor
            눈길에 대한 내놓기의 기울기, 꼴이 같다

        Returns:
        -------
        torch.Tensor
            눈길 흐름 행렬
        """
        # 기울기의 절댓값을 잡는다(크기만 본다)
        grad_magnitude = torch.abs(gradients)

        # 눈길에 기울기 크기를 곱한다
        flow = attention * grad_magnitude

        # 고르게 한다
        flow = flow / (flow.sum(dim=1, keepdim=True) + 1e-10)

        return flow

    def visualize_flow(self,
                      attention: torch.Tensor,
                      flow: torch.Tensor,
                      tokens: List[str],
                      save_path: Optional[str] = None):
        """
        눈길 짐과 눈길 흐름을 견준다.

        어느 눈길 이음이 참으로 미루어 봄에 걸리는지 보인다.
        """
        if isinstance(attention, torch.Tensor):
            attention = attention.cpu().numpy()
        if isinstance(flow, torch.Tensor):
            flow = flow.cpu().numpy()

        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        # 눈길을 그린다
        sns.heatmap(
            attention,
            xticklabels=tokens,
            yticklabels=tokens,
            cmap='viridis',
            square=True,
            cbar_kws={'label': '짐'},
            ax=axes[0],
            vmin=0,
            vmax=1
        )
        axes[0].set_title('눈길 짐', fontsize=13, fontweight='bold')
        axes[0].set_xlabel('열쇠 낱말')
        axes[0].set_ylabel('물음 낱말')

        # 흐름을 그린다
        sns.heatmap(
            flow,
            xticklabels=tokens,
            yticklabels=tokens,
            cmap='YlOrRd',
            square=True,
            cbar_kws={'label': '흐름'},
            ax=axes[1],
            vmin=0,
            vmax=flow.max()
        )
        axes[1].set_title('눈길 흐름 (기울기를 곁들임)', fontsize=13, fontweight='bold')
        axes[1].set_xlabel('열쇠 낱말')
        axes[1].set_ylabel('물음 낱말')

        for ax in axes:
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

        plt.suptitle('눈길 대 흐름: 어느 이음이 중요한가?',
                    fontsize=15, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')

        plt.show()

    def identify_critical_connections(self,
                                     flow: torch.Tensor,
                                     tokens: List[str],
                                     top_k: int = 5) -> List[Tuple]:
        """
        흐름을 바탕으로 가장 종요로운 눈길 이음을 짚어낸다.

        Returns:
        -------
        튜플의 목록
            (물음 낱말, 열쇠 낱말, 흐름 값)
        """
        if isinstance(flow, torch.Tensor):
            flow = flow.cpu().numpy()

        # 앞선 k개의 이음을 집는다
        flat_indices = np.argsort(flow.flatten())[-top_k:][::-1]

        critical = []
        for idx in flat_indices:
            i = idx // flow.shape[1]
            j = idx % flow.shape[1]
            critical.append((tokens[i], tokens[j], flow[i, j]))

        return critical

def example_attention_flow():
    """보기: 눈길 흐름 셈하기."""
    print("=" * 70)
    print("눈길 흐름 살피기")
    print("=" * 70)

    # 보기를 만든다
    tokens = ["The", "cat", "sat", "on", "the", "mat"]
    seq_len = len(tokens)

    # 지어낸 눈길
    attention = torch.softmax(torch.randn(seq_len, seq_len), dim=1)

    # 지어낸 기울기(중요함을 흉내낸다)
    # 몇몇 이음에 큰 기울기를 준다
    gradients = torch.rand(seq_len, seq_len) * 0.1
    gradients[1, 0] = 2.0  # "cat" <- "The"이 중요하다
    gradients[2, 1] = 1.5  # "sat" <- "cat"이 중요하다
    gradients[5, 3] = 1.8  # "mat" <- "on"이 중요하다

    # 흐름을 셈한다
    analyzer = AttentionFlowAnalyzer()
    flow = analyzer.compute_attention_flow(attention, gradients)

    # 그림으로 보인다
    analyzer.visualize_flow(attention, flow, tokens)

    # 종요로운 이음을 찾는다
    print("\n앞선 5개의 종요로운 눈길 이음:")
    print("-" * 50)
    critical = analyzer.identify_critical_connections(flow, tokens, top_k=5)
    for query, key, flow_val in critical:
        print(f"  {query:10s} <- {key:10s} : {flow_val:.4f}")

if __name__ == "__main__":
    torch.manual_seed(42)
    example_attention_flow()

    print("\n고갱이 깨침:")
    print("  - 눈길 짐은 모든 이음을 보인다")
    print("  - 기울기는 어느 이음이 내놓기를 흔드는지 보인다")
    print("  - 흐름은 둘을 아울러 참된 중요함을 준다")
    print("  - 몫 매기기와 풀이하기에 종요롭다")```

## 논의

그림으로 보이기는 모형의 움직임을 알고 익힘의 탈을 짚어내는 데 큰 몫을 한다. 그리는 코드는 배운 나타냄, 모여 가는 결, 따짐 자를 들여다보게 해서 손에 잡히지 않던 셈을 눈에 보이게 한다.

여기서 보인 결은 더 까다로운 자리로도 자연스레 넓혀진다. 하이퍼파라미터, 얼개의 갈래, 여러 자료를 바꿔 가며 해 보면 이해가 깊어지고 모형 풀이하기에 대한 감이 몸에 붙는다.

## 익힘 문제

**익힘 1.**
코드를 읽고 고갱이가 되는 설계 판단을 짚어라. 짜기에서 고른 것 셋을 들고, 저마다 왜 눈길 그림 그리기에 알맞은지 밝혀라.

??? success "익힘 1 풀이"
    설계 판단은 짜보기마다 다르나 흔히 이런 것이 있다. (1) 살림 함수 고르기 -- ReLU 갈래는 기울기가 잦아들지 않아 익히기가 빠르다. (2) 고르게 하는 꾀 -- 묶음 고르게 하기가 안쪽 함께 바뀌는 옮겨감을 줄여 익힘을 든든하게 한다. (3) 나머지 이음 -- 있으면 건너뛰는 길을 주어 깊은 그물에서 기울기가 흐르게 한다. 고른 것마다 나타내는 힘, 셈 값, 익힘의 든든함 사이의 맞바꿈을 드러낸다.

---

**익힘 2.**
눈길 짐 뒤(값과 곱하기 앞)에 드롭아웃 켜를 더하여라. 익히는 동안 드롭아웃 비율을 0.1으로 잡아라. 눈길 드롭아웃이 정칙화에 왜 도움이 되는지 밝혀라.

??? success "익힘 2 풀이"
    `__init__`에 `self.attn_dropout = nn.Dropout(0.1)`을 더하고 소프트맥스 뒤에 건다. `attn_weights = self.attn_dropout(F.softmax(scores, dim=-1))`. 눈길 드롭아웃은 익히는 동안 눈길 짐 몇몇을 아무렇게나 0으로 만들어, 모형이 특정 낱말끼리의 얽힘에 지나치게 기대는 것을 막는다. 그래서 모형이 눈길을 더 고루 나누고 더 든든한 나타냄을 배우게 되는데, 여느 드롭아웃이 신경 세포끼리 함께 굳는 것을 막는 것과 같은 결이다.

---

**익힘 3.**
제 눈길의 셈 복잡도를 열 길이 $n$과 모형 차원 $d$의 함수로 밝혀라. 이것이 왜 긴 열에 Longformer이나 Linformer 같은 얼개를 부르는가?

??? success "익힘 3 풀이"
    여느 제 눈길은 $n \times n$ 눈길 행렬을 셈하므로 때가 $O(n^2 d)$이고 눈길 짐에 드는 기억이 $O(n^2)$이다. 열이 길면(보기로 $n = 4096$) 감당할 수 없다. Longformer은 그 자리 미끄럼 창 눈길($O(n \cdot w \cdot d)$, $w$은 창 크기)과 고른 낱말에 대한 성긴 온 세상 눈길을 아울러 쓴다. Linformer은 열쇠와 값을 낮은 차원 $k \ll n$으로 쏘아 복잡도를 $O(n \cdot k \cdot d)$으로 줄인다. 둘 다 나타내는 힘을 얼마쯤 내주고 긴 들임에서의 쓸모를 얻는다.

---
**익힘 4.**
앞선 켜 짜보기를 살피는 두루 갖춘 시험 함수를 써라. 빈 들임, 원소 하나짜리 들임, 아주 큰 들임, 그리고 끝자락 값(0, 아주 큰 수)이 든 들임 같은 가장자리 자리를 시험하여라.

??? success "익힘 4 풀이"
    금 언저리 조건을 두루 건드리는 시험 함수를 짓는다.
    ```python
    def test_attentionflowanalyzer():
        model = 앞선 켜(...)
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
