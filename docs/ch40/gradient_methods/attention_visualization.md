# 어텐션 시각화

트랜스포머 모델을 위한 어텐션 시각화. 트랜스포머 구조의 셀프 어텐션 피처를 그린다.

뉴럴 네트워크가 무엇을 배우는지 아는 일은 신뢰를 쌓고 모델을 디버깅하는 데 중요하다. 이 모듈은 모델이 입력을 어떻게 처리하고 어떻게 판단하는지 드러내는 그래디언트 기반 설명 기법을 보이며, 네트워크의 행동을 눈으로 보고 수로 재게 해 준다.

## 1. 코드

```python
"""
트랜스포머 모형을 위한 어텐션 시각화
트랜스포머 구조의 제 어텐션 결을 그린다
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List, Tuple, Dict
import warnings

# ========================================================================
# 메인
# ========================================================================


class AttentionVisualizer:
    """
    트랜스포머 모형의 어텐션 짐을 그린다.
    """

    def __init__(self, model: nn.Module):
        """
        어텐션 그리개의 첫자리를 잡는다.

        Args:
            model: 트랜스포머 모형(보기로 BERT, GPT, ViT)
        """
        self.model = model
        self.attention_maps = {}
        self.hooks = []

    def register_hooks(self, layer_names: Optional[List[str]] = None):
        """
        어텐션 짐을 붙들려고 앞으로 스텝 갈고리를 건다.

        Args:
            layer_names: 갈고리를 걸 레이어 이름 목록. None이면 모든 어텐션 레이어에 건다.
        """
        def hook_fn(name):
            def hook(module, input, output):
                # 어텐션 짐을 갈무리한다
                # 출력 형태는 모형 구조마다 다르다
                if isinstance(output, tuple) and len(output) > 1:
                    # 흔히 (출력, 어텐션 짐)
                    self.attention_maps[name] = output[1].detach()
                else:
                    self.attention_maps[name] = output.detach()
            return hook

        # 어텐션 꾸러미를 찾아 갈고리를 건다
        for name, module in self.model.named_modules():
            # 흔한 어텐션 꾸러미 이름
            if any(x in name.lower() for x in ['attention', 'attn', 'self_attn']):
                if layer_names is None or name in layer_names:
                    handle = module.register_forward_hook(hook_fn(name))
                    self.hooks.append(handle)

    def remove_hooks(self):
        """걸어 둔 갈고리를 모두 치운다."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def get_attention_maps(self) -> Dict[str, torch.Tensor]:
        """
        붙든 어텐션 그림을 내놓는다.

        Returns:
            레이어 이름을 어텐션 텐서에 이어 주는 사전
        """
        return self.attention_maps

    def visualize_attention_head(self, attention_weights: torch.Tensor,
                                tokens: Optional[List[str]] = None,
                                head_idx: int = 0,
                                figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
        """
        정한 머리의 어텐션 짐을 그린다.

        Args:
            attention_weights: (배치, 머리, 열 길이, 열 길이) 형태 어텐션 텐서
            tokens: 이름표로 쓸 단어 목록
            head_idx: 그릴 어텐션 머리의 번호
            figsize: 그림 크기

        Returns:
            Matplotlib 그림
        """
        # 머리 하나를 뽑는다
        if attention_weights.dim() == 4:
            attn = attention_weights[0, head_idx].cpu().numpy()
        elif attention_weights.dim() == 3:
            attn = attention_weights[head_idx].cpu().numpy()
        else:
            attn = attention_weights.cpu().numpy()

        # 그림을 만든다
        fig, ax = plt.subplots(figsize=figsize)

        # 열 그림을 그린다
        sns.heatmap(attn, annot=False, cmap='viridis', square=True,
                   cbar_kws={'label': '어텐션 짐'}, ax=ax)

        # 단어가 있으면 이름표를 붙인다
        if tokens:
            ax.set_xticklabels(tokens, rotation=90)
            ax.set_yticklabels(tokens, rotation=0)

        ax.set_xlabel('열쇠 자리')
        ax.set_ylabel('질문 자리')
        ax.set_title(f'어텐션 머리 {head_idx}')

        plt.tight_layout()
        return fig

    def visualize_all_heads(self, attention_weights: torch.Tensor,
                           tokens: Optional[List[str]] = None,
                           max_heads: int = 8,
                           figsize: Tuple[int, int] = (16, 12)) -> plt.Figure:
        """
        어텐션 머리 여럿을 격자로 그린다.

        Args:
            attention_weights: (배치, 머리, 열 길이, 열 길이) 형태 어텐션 텐서
            tokens: 단어 목록
            max_heads: 보일 머리의 가장 큰 수
            figsize: 그림 크기

        Returns:
            Matplotlib 그림
        """
        if attention_weights.dim() == 4:
            attn = attention_weights[0].cpu().numpy()
        else:
            attn = attention_weights.cpu().numpy()

        num_heads = min(attn.shape[0], max_heads)
        ncols = 4
        nrows = (num_heads + ncols - 1) // ncols

        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        axes = axes.flatten() if num_heads > 1 else [axes]

        for i in range(num_heads):
            ax = axes[i]
            sns.heatmap(attn[i], annot=False, cmap='viridis',
                       square=True, cbar=False, ax=ax)
            ax.set_title(f'머리 {i}')

            if tokens and len(tokens) <= 20:
                ax.set_xticklabels(tokens, rotation=90, fontsize=8)
                ax.set_yticklabels(tokens, rotation=0, fontsize=8)
            else:
                ax.set_xticks([])
                ax.set_yticks([])

        # 쓰지 않은 칸을 감춘다
        for i in range(num_heads, len(axes)):
            axes[i].axis('off')

        plt.suptitle('여러 머리 어텐션 결', fontsize=16)
        plt.tight_layout()
        return fig

    def visualize_layer_attention(self, layer_name: str,
                                  tokens: Optional[List[str]] = None,
                                  average_heads: bool = True,
                                  figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
        """
        정한 레이어의 어텐션을 그린다.

        Args:
            layer_name: 레이어 이름
            tokens: 단어 목록
            average_heads: 머리를 가로질러 고르게 할지
            figsize: 그림 크기

        Returns:
            Matplotlib 그림
        """
        if layer_name not in self.attention_maps:
            raise ValueError(f"어텐션 그림에 레이어 {layer_name}이 없다")

        attn = self.attention_maps[layer_name]

        if average_heads and attn.dim() >= 3:
            # 머리를 가로질러 고르게 한다
            if attn.dim() == 4:
                attn = attn[0].mean(dim=0).cpu().numpy()
            else:
                attn = attn.mean(dim=0).cpu().numpy()
        else:
            return self.visualize_all_heads(attn, tokens, figsize=figsize)

        # 그림을 만든다
        fig, ax = plt.subplots(figsize=figsize)

        sns.heatmap(attn, annot=False, cmap='viridis', square=True,
                   cbar_kws={'label': '고른 어텐션'}, ax=ax)

        if tokens:
            ax.set_xticklabels(tokens, rotation=90)
            ax.set_yticklabels(tokens, rotation=0)

        ax.set_xlabel('열쇠 자리')
        ax.set_ylabel('질문 자리')
        ax.set_title(f'레이어: {layer_name} (고른 어텐션)')

        plt.tight_layout()
        return fig

    def plot_attention_flow(self, attention_weights: torch.Tensor,
                           tokens: List[str],
                           query_idx: int,
                           figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
        """
        정한 질문 단어에서 뻗는 어텐션 흐름을 그린다.

        Args:
            attention_weights: 어텐션 텐서
            tokens: 단어 목록
            query_idx: 질문 단어의 번호
            figsize: 그림 크기

        Returns:
            Matplotlib 그림
        """
        if attention_weights.dim() == 4:
            # 배치와 머리를 가로질러 고르게 한다
            attn = attention_weights[0].mean(dim=0)[query_idx].cpu().numpy()
        elif attention_weights.dim() == 3:
            attn = attention_weights.mean(dim=0)[query_idx].cpu().numpy()
        else:
            attn = attention_weights[query_idx].cpu().numpy()

        fig, ax = plt.subplots(figsize=figsize)

        positions = np.arange(len(tokens))
        ax.bar(positions, attn, color='steelblue', alpha=0.7)
        ax.set_xticks(positions)
        ax.set_xticklabels(tokens, rotation=45, ha='right')
        ax.set_ylabel('어텐션 짐')
        ax.set_title(f'"{tokens[query_idx]}"에서 다른 단어로 가는 어텐션')
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        return fig


class BERTAttentionVisualizer(AttentionVisualizer):
    """
    BERT 결 모형에 맞춘 그리개.
    """

    def extract_attention_from_output(self, outputs):
        """
        BERT 모형의 출력에서 어텐션 짐을 뽑아낸다.

        Args:
            outputs: BERT(transformers 곳집) 모형의 출력

        Returns:
            레이어마다 하나씩인 어텐션 텐서 목록
        """
        if hasattr(outputs, 'attentions') and outputs.attentions is not None:
            return outputs.attentions
        else:
            warnings.warn("모형 출력에 어텐션 짐이 없다. "
                        "모형을 output_attentions=True로 부르라")
            return []


def visualize_token_attention(attention_weights: torch.Tensor,
                              tokens: List[str],
                              layer_idx: int = -1,
                              save_path: Optional[str] = None) -> plt.Figure:
    """
    단어에 대한 어텐션을 빠르게 그리는 도움 함수.

    Args:
        attention_weights: 어텐션 텐서 (레이어, 배치, 머리, 열, 열)
        tokens: 단어 목록
        layer_idx: 그릴 레이어 번호(-1이면 마지막 레이어)
        save_path: 그림을 갈무리할 길

    Returns:
        Matplotlib 그림
    """
    # 레이어를 고른다
    if isinstance(attention_weights, (list, tuple)):
        attn = attention_weights[layer_idx]
    else:
        attn = attention_weights

    # 배치와 머리를 가로질러 고르게 한다
    if attn.dim() == 4:
        attn = attn[0].mean(dim=0)
    elif attn.dim() == 3:
        attn = attn.mean(dim=0)

    attn = attn.cpu().numpy()

    # 그림을 만든다
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(attn, xticklabels=tokens, yticklabels=tokens,
               cmap='viridis', square=True,
               cbar_kws={'label': '어텐션 짐'})

    ax.set_xlabel('열쇠 자리')
    ax.set_ylabel('질문 자리')
    ax.set_title(f'어텐션 결 (레이어 {layer_idx})')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


if __name__ == "__main__":
    pass
```

## 2. 논의

시각화는 모델의 행동을 이해하고 학습의 문제를 탐지하는 데 큰 역할을 한다. 시각화 코드는 학습된 표현, 수렴하는 패턴, 평가 지표를 들여다보게 해서 추상적인 계산을 눈에 보이게 한다.

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
어텐션 시각화 구현을 검사하는 포괄적인 테스트 함수를 작성하라. 빈 입력, 원소 하나짜리 입력, 아주 큰 입력, 그리고 극단값(0, 아주 큰 수)이 든 입력 같은 경계 사례를 테스트하여라.

</div>

??? success "풀이"
    경계 조건을 두루 다루는 테스트 함수를 작성한다.
    ```python
    def test_attentionvisualizer():
        model = 어텐션 시각화(...)
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

**다룬 것** — 어텐션 시각화

시각화는 모델의 행동을 이해하고 학습의 문제를 탐지하는 데 큰 역할을 한다.

핵심 클래스는 `AttentionVisualizer`, `BERTAttentionVisualizer`이며 앞의 연습문제 4개로 스스로 따져 볼 수 있다.
