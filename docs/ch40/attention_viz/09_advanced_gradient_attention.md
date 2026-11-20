# 그래디언트 기반 어텐션 심화

트랜스포머의 어텐션에 Integrated Gradients와 그래디언트 × 입력 같은 그래디언트 기반 기여도 분석을 걸어 보려는 심화 예제다. 기여도 분석은 예측에 입력의 각 부분이 얼마나 기여했는지를 수로 매기는 일이다.

뉴럴 네트워크가 무엇을 배우는지 아는 일은 신뢰를 쌓고 모델을 디버깅하는 데 중요하다. 이 모듈은 모델이 입력을 어떻게 처리하고 어떻게 판단하는지 드러내는 어텐션 시각화 기법을 보이며, 네트워크의 행동을 눈으로 보고 수로 재게 해 준다.

## 1. 코드

```python
"""
앞선 레이어: 그래디언트 바탕 어텐션 기여도 매기기

어텐션 기여도 매기기와 모형 풀이하기를 위해 쌓은 그래디언트를 비롯한
그래디언트 바탕 방법을 짜 넣는다.
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
    그래디언트 바탕 어텐션 기여도 매기기 방법.

    이 방법들은 그래디언트를 써서 어느 어텐션 짐이 모형의 미루어 봄에
    가장 크게 기여하는지 가려낸다.
    """

    def __init__(self, n_steps: int = 50):
        """
        파라메터:
        ----------
        n_steps : int
            쌓은 그래디언트의 사이 잡는 스텝 수
        """
        self.n_steps = n_steps

    def integrated_gradients_attention(self,
                                      model: Callable,
                                      input_ids: torch.Tensor,
                                      baseline_ids: Optional[torch.Tensor] = None,
                                      target_idx: Optional[int] = None) -> torch.Tensor:
        """
        어텐션에 대한 쌓은 그래디언트를 셈한다.

        쌓은 그래디언트:
        --------------------
        IG(x) = (x - x') × ∫[α=0에서 1] ∂F(x' + α(x - x'))/∂x dα

        여기서:
        - x은 들임
        - x'은 베이스라인
        - α은 사이 잡는 계수
        - 적분은 리만 합으로 어림한다

        파라메터:
        ----------
        model : callable
            어텐션 짐을 내놓는 모형
        input_ids : torch.Tensor
            들임 단어 번호
        baseline_ids : torch.Tensor, 없어도 됨
            베이스라인 들임(맡긴 값: 0)
        target_idx : int, 없어도 됨
            겨눈 출력 자리

        Returns:
        -------
        torch.Tensor
            어텐션에 대한 기여도 점수
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

            # 그래디언트를 켠다
            interpolated.requires_grad = True

            # 앞으로 스텝
            outputs = model(interpolated, output_attentions=True)

            # 어텐션을 집고 그래디언트를 셈한다
            attention = outputs.attentions[-1][0, 0]  # 마지막 레이어, 첫 머리

            if target_idx is not None:
                # 정한 자리에 대한 그래디언트
                target_attn = attention[target_idx].sum()
            else:
                # 모든 자리에 대한 그래디언트
                target_attn = attention.sum()

            # 역전파
            if interpolated.grad is not None:
                interpolated.grad.zero_()

            target_attn.backward()

            # 그래디언트를 갈무리한다
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
        단순한 그래디언트 × 들임 기여도 매기기.

        기여도 = ∂출력/∂들임 × 들임

        쌓은 그래디언트를 갈음하는 더 단순한 길이다.
        """
        input_ids.requires_grad = True

        # 앞으로
        outputs = model(input_ids, output_attentions=True)
        attention = outputs.attentions[-1][0, 0]

        if target_idx is not None:
            target = attention[target_idx].sum()
        else:
            target = attention.sum()

        # 역전파
        if input_ids.grad is not None:
            input_ids.grad.zero_()

        target.backward()

        # 기여도
        attribution = input_ids.grad * input_ids.float()

        return attribution.detach()

    def visualize_attribution(self,
                             attribution: torch.Tensor,
                             tokens: List[str],
                             title: str = "어텐션 기여도 매기기",
                             save_path: Optional[str] = None):
        """기여도 점수를 그린다."""
        if isinstance(attribution, torch.Tensor):
            attribution = attribution.cpu().numpy()

        if attribution.ndim > 1:
            attribution = attribution.squeeze()

        fig, ax = plt.subplots(figsize=(12, 5))

        positions = np.arange(len(tokens))
        colors = ['red' if a < 0 else 'green' for a in attribution]

        bars = ax.bar(positions, np.abs(attribution), color=colors, alpha=0.7, edgecolor='black')

        ax.set_xlabel('단어', fontsize=12, fontweight='bold')
        ax.set_ylabel('기여도 점수(크기)', fontsize=12, fontweight='bold')
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
    """보기: 그래디언트 바탕 기여도 매기기(단순하게 줄임)."""
    print("=" * 70)
    print("그래디언트 바탕 어텐션 기여도 매기기")
    print("=" * 70)

    print("\n짚을 것: 이 보기는 지어낸 자료를 쓴다.")
    print("참 모형에는 실제 트랜스포머 모형과 함께 쓰라.")

    tokens = ["The", "cat", "chased", "the", "mouse"]
    seq_len = len(tokens)

    # 기여도 점수를 흉내낸다
    attribution = torch.tensor([0.2, 0.8, 0.6, 0.1, 0.9])

    analyzer = GradientAttentionAnalyzer()
    analyzer.visualize_attribution(
        attribution,
        tokens,
        title="단어 기여도 점수(지어낸 것)"
    )

    print("\n읽는 법:")
    print("  - 점수가 클수록 미루어 봄에 더 중요하다")
    print("  - 'cat'과 'mouse'의 기여도가 크다")
    print("  - 관사('the')의 기여도는 작다")

if __name__ == "__main__":
    example_gradient_attribution()

    print("\n핵심 생각:")
    print("  - 쌓은 그래디언트: 강건한 기여도 매기기 방법")
    print("  - 그래디언트 × 들임: 더 단순한 갈음")
    print("  - 어느 단어가 미루어 봄을 이끄는지 드러낸다")
    print("  - 모형 풀이하기에 꼭 있어야 한다")
```

**출력:**

```
======================================================================
그래디언트 바탕 어텐션 기여도 매기기
======================================================================

짚을 것: 이 보기는 지어낸 자료를 쓴다.
참 모형에는 실제 트랜스포머 모형과 함께 쓰라.

읽는 법:
  - 점수가 클수록 미루어 봄에 더 중요하다
  - 'cat'과 'mouse'의 기여도가 크다
  - 관사('the')의 기여도는 작다

핵심 생각:
  - 쌓은 그래디언트: 강건한 기여도 매기기 방법
  - 그래디언트 × 들임: 더 단순한 갈음
  - 어느 단어가 미루어 봄을 이끄는지 드러낸다
  - 모형 풀이하기에 꼭 있어야 한다
```

## 2. 논의

실행되는 것은 `example_gradient_attribution` 하나뿐이고, 이 함수는 손으로 적은 점수 `[0.2, 0.8, 0.6, 0.1, 0.9]` 를 막대그래프로 그릴 뿐이다. 출력의 「읽는 법」도 그 지어낸 점수를 풀어 쓴 것이다. 기여도를 실제로 계산하는 두 메서드는 한 번도 불리지 않는데, 불렀다면 두 가지 까닭으로 제대로 돌지 않는다.

첫째, 토큰 번호에 그래디언트를 걸 수 없다. `integrated_gradients_attention` 은 정수인 토큰 번호를 보간한 뒤 `.long()` 으로 다시 정수로 바꾸고 `requires_grad = True` 를 건다. PyTorch는 부동소수점 텐서에만 그래디언트를 허용하므로 여기서 오류가 난다. 설령 걸리더라도 번호 3과 7 사이의 「4.6번 토큰」은 뜻이 없다. 텍스트에 IG를 쓸 때는 임베딩 벡터를 베이스라인 임베딩에서 입력 임베딩까지 보간한다. `gradient_x_input` 도 같은 문제를 안고 있다.

둘째, 미분하는 대상이 상수다. 어텐션 웨이트는 소프트맥스를 거쳐 행마다 합이 1이므로 `attention[target_idx].sum()` 은 늘 1이고 `attention.sum()` 은 늘 시퀀스 길이다. 상수의 그래디언트는 0이니 기여도도 모두 0이 된다. 어텐션이 무엇에 기여하는지 보려면 어텐션이 아니라 모델의 출력(예를 들어 타깃 클래스의 로짓)을 미분해야 한다.

여기서 보인 기법은 더 까다로운 상황으로도 자연스럽게 확장된다. 하이퍼파라미터, 구조의 종류, 여러 데이터를 바꿔 가며 해 보면 이해가 깊어지고 모델 해석에 대한 감이 생긴다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
코드를 읽고 핵심 설계 선택을 짚어라. 구현에서 고른 것 셋을 들고, 저마다 왜 어텐션 시각화에 알맞은지 설명하라.

</div>

??? success "풀이"
    `GradientAttentionAnalyzer` 에서 셋을 들 수 있다. (1) **IG와 그래디언트 × 입력을 함께 두기** -- 그래디언트 × 입력은 순전파·역전파 한 번이면 되고, IG는 $m$번이 들지만 포화 영역에서도 기여를 놓치지 않는다. 빠른 탐색과 엄밀한 분석을 골라 쓰게 한다. (2) **리만 합으로 적분 근사**(`attributions.mean(dim=0)` 뒤 `(x - x')` 곱하기) -- IG의 정의를 그대로 옮긴 것이라 스텝 수만 늘리면 근사가 정확해진다. (3) **부호를 색으로, 크기를 막대 높이로**(`visualize_attribution`) -- 예측을 올린 토큰과 내린 토큰을 한 그림에서 가른다. 다만 위 논의에서 보았듯 이 구현은 토큰 번호를 보간하고 상수(어텐션의 합)를 미분하므로, 설계는 맞아도 그대로는 쓸 수 없다. 임베딩을 보간하고 모델 출력을 미분하도록 고쳐야 한다.

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
    보통의 셀프 어텐션은 $n \times n$ 어텐션 행렬을 계산하므로 시간이 $O(n^2 d)$이고 어텐션 웨이트에 드는 메모리가 $O(n^2)$이다. 시퀀스가 길면(예를 들어 $n = 4096$) 감당할 수 없다. Longformer는 국소 슬라이딩 윈도 어텐션($O(n \cdot w \cdot d)$, $w$는 윈도 크기)과 고른 토큰에 대한 성긴 전역 어텐션을 함께 쓴다. Linformer는 키와 값을 낮은 차원 $k \ll n$으로 사영해 복잡도를 $O(n \cdot k \cdot d)$로 줄인다. 둘 다 표현력을 얼마쯤 내주고 긴 입력에서의 효율을 얻는다.

---
<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
`GradientAttentionAnalyzer` 구현을 검사하는 포괄적인 테스트 함수를 작성하라. 빈 입력, 원소 하나짜리 입력, 아주 큰 입력, 그리고 극단값(0, 아주 큰 수)이 든 입력 같은 경계 사례를 테스트하여라.

</div>

??? success "풀이"
    경계 조건을 두루 다루는 테스트 함수를 작성한다.
    ```python
    def test_gradientattentionanalyzer():
        model = 앞선 레이어(...)
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

**다룬 것** — 그래디언트 기반 어텐션 심화

트랜스포머에 IG와 그래디언트 × 입력을 거는 틀을 보았다. 토큰 번호가 아니라 임베딩을 보간하고, 어텐션의 합이 아니라 모델 출력을 미분해야 한다는 점이 핵심이다.

핵심 클래스는 `GradientAttentionAnalyzer`이며 앞의 연습문제 4개로 스스로 따져 볼 수 있다.
