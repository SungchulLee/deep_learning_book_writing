# 어텐션 흐름 분석 심화

어텐션 웨이트에 그 웨이트에 대한 그래디언트의 크기를 곱해, 어느 어텐션 연결이 예측에 중요한지 가려 보는 심화 예제다. 어텐션 웨이트는 모델이 어디를 보는지만 알려 주고, 그래디언트는 그 연결이 바뀌면 출력이 얼마나 바뀌는지를 알려 준다.

뉴럴 네트워크가 무엇을 배우는지 아는 일은 신뢰를 쌓고 모델을 디버깅하는 데 중요하다. 이 모듈은 모델이 입력을 어떻게 처리하고 어떻게 판단하는지 드러내는 어텐션 시각화 기법을 보이며, 네트워크의 행동을 눈으로 보고 수로 재게 해 준다.

## 1. 코드

```python
"""
심화: 어텐션 흐름 분석

어텐션 웨이트와 그래디언트 정보를 결합해, 어느 어텐션 연결이 예측에
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
    그래디언트로 어텐션 흐름을 분석하는 클래스.

    어텐션 웨이트만으로는 어느 연결이 중요한지 알 수 없다.
    어텐션과 그래디언트를 결합하면 중요한 경로를 찾아낼 수 있다.

    식:
    -------
    흐름 = 어텐션 × |그래디언트|

    여기서 그래디언트는 출력에 대한 것이다.
    """

    def __init__(self):
        pass

    def compute_attention_flow(self,
                              attention: torch.Tensor,
                              gradients: torch.Tensor) -> torch.Tensor:
        """
        어텐션과 그래디언트를 결합해 어텐션 흐름을 계산한다.

        파라메터:
        ----------
        attention : torch.Tensor
            어텐션 웨이트, 형태: (시퀀스 길이, 시퀀스 길이)
        gradients : torch.Tensor
            어텐션에 대한 출력의 그래디언트, 형태가 같다

        Returns:
        -------
        torch.Tensor
            어텐션 흐름 행렬
        """
        # 그래디언트의 절댓값을 취한다(크기만 본다)
        grad_magnitude = torch.abs(gradients)

        # 어텐션에 그래디언트 크기를 곱한다
        flow = attention * grad_magnitude

        # 행마다 합이 1이 되게 정규화한다
        flow = flow / (flow.sum(dim=1, keepdim=True) + 1e-10)

        return flow

    def visualize_flow(self,
                      attention: torch.Tensor,
                      flow: torch.Tensor,
                      tokens: List[str],
                      save_path: Optional[str] = None):
        """
        어텐션 웨이트와 어텐션 흐름을 비교한다.

        어느 어텐션 연결이 실제로 예측에 관여하는지 보인다.
        """
        if isinstance(attention, torch.Tensor):
            attention = attention.cpu().numpy()
        if isinstance(flow, torch.Tensor):
            flow = flow.cpu().numpy()

        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        # 어텐션을 그린다
        sns.heatmap(
            attention,
            xticklabels=tokens,
            yticklabels=tokens,
            cmap='viridis',
            square=True,
            cbar_kws={'label': '웨이트'},
            ax=axes[0],
            vmin=0,
            vmax=1
        )
        axes[0].set_title('어텐션 웨이트', fontsize=13, fontweight='bold')
        axes[0].set_xlabel('키 토큰')
        axes[0].set_ylabel('쿼리 토큰')

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
        axes[1].set_title('어텐션 흐름 (그래디언트 반영)', fontsize=13, fontweight='bold')
        axes[1].set_xlabel('키 토큰')
        axes[1].set_ylabel('쿼리 토큰')

        for ax in axes:
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

        plt.suptitle('어텐션 대 흐름: 어느 연결이 중요한가?',
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
        흐름을 바탕으로 가장 중요한 어텐션 연결을 찾아낸다.

        Returns:
        -------
        튜플의 목록
            (쿼리 토큰, 키 토큰, 흐름 값)
        """
        if isinstance(flow, torch.Tensor):
            flow = flow.cpu().numpy()

        # 상위 k개의 연결을 고른다
        flat_indices = np.argsort(flow.flatten())[-top_k:][::-1]

        critical = []
        for idx in flat_indices:
            i = idx // flow.shape[1]
            j = idx % flow.shape[1]
            critical.append((tokens[i], tokens[j], flow[i, j]))

        return critical

def example_attention_flow():
    """예제: 어텐션 흐름 계산."""
    print("=" * 70)
    print("어텐션 흐름 분석")
    print("=" * 70)

    # 예제를 만든다
    tokens = ["The", "cat", "sat", "on", "the", "mat"]
    seq_len = len(tokens)

    # 합성한 어텐션
    attention = torch.softmax(torch.randn(seq_len, seq_len), dim=1)

    # 합성한 그래디언트(중요도를 흉내 낸다)
    # 몇몇 연결에 큰 그래디언트를 준다
    gradients = torch.rand(seq_len, seq_len) * 0.1
    gradients[1, 0] = 2.0  # "cat" <- "The"가 중요하다
    gradients[2, 1] = 1.5  # "sat" <- "cat"이 중요하다
    gradients[5, 3] = 1.8  # "mat" <- "on"이 중요하다

    # 흐름을 계산한다
    analyzer = AttentionFlowAnalyzer()
    flow = analyzer.compute_attention_flow(attention, gradients)

    # 시각화한다
    analyzer.visualize_flow(attention, flow, tokens)

    # 중요한 연결을 찾는다
    print("\n상위 5개의 중요한 어텐션 연결:")
    print("-" * 50)
    critical = analyzer.identify_critical_connections(flow, tokens, top_k=5)
    for query, key, flow_val in critical:
        print(f"  {query:10s} <- {key:10s} : {flow_val:.4f}")

if __name__ == "__main__":
    torch.manual_seed(42)
    example_attention_flow()

    print("\n핵심 정리:")
    print("  - 어텐션 웨이트는 모든 연결을 보인다")
    print("  - 그래디언트는 어느 연결이 출력을 바꾸는지 보인다")
    print("  - 흐름은 둘을 결합해 실제 중요도를 준다")
    print("  - 기여도 분석과 해석에 중요하다")
```

**출력:**

```
======================================================================
어텐션 흐름 분석
======================================================================

상위 5개의 중요한 어텐션 연결:
--------------------------------------------------
  mat        <- on         : 0.9463
  cat        <- The        : 0.9217
  sat        <- cat        : 0.7819
  the        <- sat        : 0.5280
  The        <- cat        : 0.5073

핵심 정리:
  - 어텐션 웨이트는 모든 연결을 보인다
  - 그래디언트는 어느 연결이 출력을 바꾸는지 보인다
  - 흐름은 둘을 결합해 실제 중요도를 준다
  - 기여도 분석과 해석에 중요하다
```

## 2. 논의

먼저 이름부터 짚어 둔다. 이 쪽의 「흐름」은 어텐션 × |그래디언트| 를 행마다 합이 1이 되게 나눈 것으로, 아브나르와 주이데마(2020)의 Attention Flow — 레이어를 그래프로 보고 최대 흐름을 푸는 방법 — 와는 다르다. 어텐션과 그래디언트도 모델에서 나온 것이 아니라 지어낸 것이다. 그래디언트는 거의 모든 칸이 0.1 아래의 잡음이고, 세 칸(`cat <- The`, `sat <- cat`, `mat <- on`)에만 1.5~2.0을 심었다.

출력의 위 세 줄은 심어 둔 그 세 칸이다. 그러니 이 결과는 방법이 중요한 연결을 찾아냈다는 증거가 아니라, 심은 것이 그대로 나온다는 확인이다. 넷째와 다섯째 줄(`the <- sat` 0.5280, `The <- cat` 0.5073)이 더 볼 만하다. 이 두 행에는 큰 그래디언트를 심지 않았는데도 값이 0.5를 넘는다. 행 정규화 때문에 잡음뿐인 행에서도 가장 큰 칸이 크게 부풀기 때문이다. 행마다 따로 정규화한 값을 모아 상위 $k$개를 고르면 행 사이의 비교가 무너진다는 뜻이다. 출력 끝의 「핵심 정리」은 이 실행이 보여 준 것이 아니라 미리 적어 둔 문장이다.

여기서 보인 기법은 더 까다로운 상황으로도 자연스럽게 확장된다. 하이퍼파라미터, 구조의 종류, 여러 데이터를 바꿔 가며 해 보면 이해가 깊어지고 모델 해석에 대한 감이 생긴다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
코드를 읽고 핵심 설계 선택을 짚어라. 구현에서 고른 것 셋을 들고, 저마다 왜 어텐션 시각화에 알맞은지 설명하라.

</div>

??? success "풀이"
    `AttentionFlowAnalyzer` 에서 셋을 들 수 있다. (1) **어텐션과 그래디언트 크기의 곱** -- 어텐션이 커도 출력이 그 연결에 둔감하면 값이 작아지고, 그 반대도 마찬가지라 두 조건을 함께 만족하는 연결만 남는다. (2) **그래디언트의 절댓값**(`torch.abs`) -- 출력을 올리는 연결과 내리는 연결을 똑같이 중요하다고 본다. 대신 방향 정보는 버린다. (3) **행마다 합이 1이 되게 정규화** -- 쿼리마다 그 쿼리의 흐름을 분포로 읽게 해 준다. 다만 위 논의에서 보았듯 잡음뿐인 행도 가장 큰 칸이 크게 부풀므로, 행끼리 비교해 상위 $k$개를 고를 때는 정규화 전의 값을 써야 한다.

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
    보통의 셀프 어텐션은 $n \times n$ 어텐션 행렬을 계산하므로 시간이 $O(n^2 d)$이고 어텐션 웨이트에 드는 메모리가 $O(n^2)$이다. 시퀀스가 길면(예를 들어 $n = 4096$) 감당할 수 없다. Longformer는 국소 슬라이딩 윈도 어텐션($O(n \cdot w \cdot d)$, $w$는 윈도 크기)과 고른 토큰에 대한 희소한 전역 어텐션을 함께 쓴다. Linformer는 키와 값을 낮은 차원 $k \ll n$으로 사영해 복잡도를 $O(n \cdot k \cdot d)$로 줄인다. 둘 다 표현력을 얼마쯤 내주고 긴 입력에서의 효율을 얻는다.

---
<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
`AttentionFlowAnalyzer` 구현을 검사하는 포괄적인 테스트 함수를 작성하라. 빈 입력, 원소 하나짜리 입력, 아주 큰 입력, 그리고 극단값(0, 아주 큰 수)이 든 입력 같은 경계 사례를 테스트하여라.

</div>

??? success "풀이"
    경계 조건을 두루 다루는 테스트 함수를 작성한다.
    ```python
    def test_attentionflowanalyzer():
        model = AttentionFlowAnalyzer(...)
        # 보통 입력
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

**다룬 것** — 어텐션 흐름 분석 심화

어텐션 웨이트에 그래디언트 크기를 곱해 중요한 연결을 가리는 법을 보았다. 행 정규화가 잡음뿐인 행까지 부풀린다는 점도 출력에서 확인했다.

핵심 클래스는 `AttentionFlowAnalyzer`이며 앞의 연습문제 4개로 스스로 따져 볼 수 있다.
