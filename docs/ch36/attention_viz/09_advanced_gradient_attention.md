# 앞선 켜

앞선 켜: 기울기 바탕 눈길 몫 매기기. 쌓은 기울기를 비롯한 기울기 바탕 방법을 짜 넣는다

신경 그물이 무엇을 배우는지 아는 일은 믿음을 쌓고 모형의 벌레를 잡는 데 종요롭다. 이 꾸러미는 모형이 들임을 어떻게 다루고 어떻게 판단하는지 드러내는 눈길 그림 그리기 재주를 보이며, 그물의 움직임을 눈으로 보고 수로 재게 해 준다.

## 1. 코드

```python
"""
앞선 켜: 기울기 바탕 눈길 몫 매기기

눈길 몫 매기기와 모형 풀이하기를 위해 쌓은 기울기를 비롯한
기울기 바탕 방법을 짜 넣는다.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Callable

# ========================================================================
# 메인
# ========================================================================

class GradientAttentionAnalyzer:
    """
    기울기 바탕 눈길 몫 매기기 방법.

    이 방법들은 기울기를 써서 어느 눈길 짐이 모형의 미루어 봄에
    가장 크게 이바지하는지 가려낸다.
    """

    def __init__(self, n_steps: int = 50):
        """
        매개변수:
        ----------
        n_steps : int
            쌓은 기울기의 사이 잡는 걸음 수
        """
        self.n_steps = n_steps

    def integrated_gradients_attention(self,
                                      model: Callable,
                                      input_ids: torch.Tensor,
                                      baseline_ids: Optional[torch.Tensor] = None,
                                      target_idx: Optional[int] = None) -> torch.Tensor:
        """
        눈길에 대한 쌓은 기울기를 셈한다.

        쌓은 기울기:
        --------------------
        IG(x) = (x - x') × ∫[α=0에서 1] ∂F(x' + α(x - x'))/∂x dα

        여기서:
        - x은 들임
        - x'은 밑금
        - α은 사이 잡는 계수
        - 적분은 리만 합으로 어림한다

        매개변수:
        ----------
        model : callable
            눈길 짐을 내놓는 모형
        input_ids : torch.Tensor
            들임 낱말 번호
        baseline_ids : torch.Tensor, 없어도 됨
            밑금 들임(맡긴 값: 0)
        target_idx : int, 없어도 됨
            겨눈 내놓기 자리

        Returns:
        -------
        torch.Tensor
            눈길에 대한 몫 점수
        """
        if baseline_ids is None:
            baseline_ids = torch.zeros_like(input_ids)

        # 사이 들임을 만든다
        alphas = torch.linspace(0, 1, self.n_steps)

        attributions = []

        for alpha in alphas:
            # 사이를 잡는다
            interpolated = baseline_ids + alpha * (input_ids - baseline_ids)
            interpolated = interpolated.long()

            # 기울기를 켠다
            interpolated.requires_grad = True

            # 앞으로 걸음
            outputs = model(interpolated, output_attentions=True)

            # 눈길을 집고 기울기를 셈한다
            attention = outputs.attentions[-1][0, 0]  # 마지막 켜, 첫 머리

            if target_idx is not None:
                # 정한 자리에 대한 기울기
                target_attn = attention[target_idx].sum()
            else:
                # 모든 자리에 대한 기울기
                target_attn = attention.sum()

            # 되짚기
            if interpolated.grad is not None:
                interpolated.grad.zero_()

            target_attn.backward()

            # 기울기를 갈무리한다
            if interpolated.grad is not None:
                attributions.append(interpolated.grad.detach())

        # 리만 합으로 적분을 어림한다
        attributions = torch.stack(attributions)
        integrated_grads = attributions.mean(dim=0)

        # (x - x')을 곱한다
        final_attribution = (input_ids - baseline_ids).float() * integrated_grads

        return final_attribution

    def gradient_x_input(self,
                        model: Callable,
                        input_ids: torch.Tensor,
                        target_idx: Optional[int] = None) -> torch.Tensor:
        """
        단순한 기울기 × 들임 몫 매기기.

        몫 = ∂내놓기/∂들임 × 들임

        쌓은 기울기를 갈음하는 더 단순한 길이다.
        """
        input_ids.requires_grad = True

        # 앞으로
        outputs = model(input_ids, output_attentions=True)
        attention = outputs.attentions[-1][0, 0]

        if target_idx is not None:
            target = attention[target_idx].sum()
        else:
            target = attention.sum()

        # 되짚기
        if input_ids.grad is not None:
            input_ids.grad.zero_()

        target.backward()

        # 몫
        attribution = input_ids.grad * input_ids.float()

        return attribution.detach()

    def visualize_attribution(self,
                             attribution: torch.Tensor,
                             tokens: List[str],
                             title: str = "눈길 몫 매기기",
                             save_path: Optional[str] = None):
        """몫 점수를 그린다."""
        if isinstance(attribution, torch.Tensor):
            attribution = attribution.cpu().numpy()

        if attribution.ndim > 1:
            attribution = attribution.squeeze()

        fig, ax = plt.subplots(figsize=(12, 5))

        positions = np.arange(len(tokens))
        colors = ['red' if a < 0 else 'green' for a in attribution]

        bars = ax.bar(positions, np.abs(attribution), color=colors, alpha=0.7, edgecolor='black')

        ax.set_xlabel('낱말', fontsize=12, fontweight='bold')
        ax.set_ylabel('몫 점수(크기)', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

        ax.set_xticks(positions)
        ax.set_xticklabels(tokens, rotation=45, ha='right')

        # 값 이름표를 붙인다
        for pos, val in zip(positions, attribution):
            ax.text(pos, abs(val) + max(abs(attribution)) * 0.02, 
                   f'{val:.3f}', ha='center', va='bottom', fontsize=9)

        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(0, color='black', linewidth=0.5)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')

        plt.show()

def example_gradient_attribution():
    """보기: 기울기 바탕 몫 매기기(단순하게 줄임)."""
    print("=" * 70)
    print("기울기 바탕 눈길 몫 매기기")
    print("=" * 70)

    print("\n짚을 것: 이 보기는 지어낸 자료를 쓴다.")
    print("참 모형에는 실제 변환기 모형과 함께 쓰라.")

    tokens = ["The", "cat", "chased", "the", "mouse"]
    seq_len = len(tokens)

    # 몫 점수를 흉내낸다
    attribution = torch.tensor([0.2, 0.8, 0.6, 0.1, 0.9])

    analyzer = GradientAttentionAnalyzer()
    analyzer.visualize_attribution(
        attribution,
        tokens,
        title="낱말 몫 점수(지어낸 것)"
    )

    print("\n읽는 법:")
    print("  - 점수가 클수록 미루어 봄에 더 중요하다")
    print("  - 'cat'과 'mouse'의 몫이 크다")
    print("  - 관사('the')의 몫은 작다")

if __name__ == "__main__":
    example_gradient_attribution()

    print("\n고갱이 생각:")
    print("  - 쌓은 기울기: 든든한 몫 매기기 방법")
    print("  - 기울기 × 들임: 더 단순한 갈음")
    print("  - 어느 낱말이 미루어 봄을 이끄는지 드러낸다")
    print("  - 모형 풀이하기에 꼭 있어야 한다")
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
앞선 켜 짜보기를 살피는 두루 갖춘 시험 함수를 써라. 빈 들임, 원소 하나짜리 들임, 아주 큰 들임, 그리고 끝자락 값(0, 아주 큰 수)이 든 들임 같은 가장자리 자리를 시험하여라.

??? success "연습문제 4 풀이"
    금 언저리 조건을 두루 건드리는 시험 함수를 짓는다.
    ```python
    def test_gradientattentionanalyzer():
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

## 정리하며

**다룬 것** — 앞선 켜

그림으로 보이기는 모형의 움직임을 알고 익힘의 탈을 짚어내는 데 큰 몫을 한다.

고갱이 갈래는 `GradientAttentionAnalyzer`이며 앞의 연습문제 4개로 스스로 따져 볼 수 있다.
