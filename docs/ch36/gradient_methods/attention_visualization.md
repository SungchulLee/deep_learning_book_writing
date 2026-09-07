# 눈길 그림 그리기

변환기 모형을 위한 눈길 그림 그리기. 변환기 얼개의 제 눈길 결을 그린다.

신경 그물이 무엇을 배우는지 아는 일은 믿음을 쌓고 모형의 벌레를 잡는 데 종요롭다. 이 꾸러미는 모형이 들임을 어떻게 다루고 어떻게 판단하는지 드러내는 기울기 바탕 풀이 재주를 보이며, 그물의 움직임을 눈으로 보고 수로 재게 해 준다.

## 코드

```python
"""
변환기 모형을 위한 눈길 그림 그리기
변환기 얼개의 제 눈길 결을 그린다
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
    변환기 모형의 눈길 짐을 그린다.
    """

    def __init__(self, model: nn.Module):
        """
        눈길 그리개의 첫자리를 잡는다.

        Args:
            model: 변환기 모형(보기로 BERT, GPT, ViT)
        """
        self.model = model
        self.attention_maps = {}
        self.hooks = []

    def register_hooks(self, layer_names: Optional[List[str]] = None):
        """
        눈길 짐을 붙들려고 앞으로 걸음 갈고리를 건다.

        Args:
            layer_names: 갈고리를 걸 켜 이름 목록. None이면 모든 눈길 켜에 건다.
        """
        def hook_fn(name):
            def hook(module, input, output):
                # 눈길 짐을 갈무리한다
                # 내놓기 꼴은 모형 얼개마다 다르다
                if isinstance(output, tuple) and len(output) > 1:
                    # 흔히 (내놓기, 눈길 짐)
                    self.attention_maps[name] = output[1].detach()
                else:
                    self.attention_maps[name] = output.detach()
            return hook

        # 눈길 꾸러미를 찾아 갈고리를 건다
        for name, module in self.model.named_modules():
            # 흔한 눈길 꾸러미 이름
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
        붙든 눈길 그림을 내놓는다.

        Returns:
            켜 이름을 눈길 텐서에 이어 주는 사전
        """
        return self.attention_maps

    def visualize_attention_head(self, attention_weights: torch.Tensor,
                                tokens: Optional[List[str]] = None,
                                head_idx: int = 0,
                                figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
        """
        정한 머리의 눈길 짐을 그린다.

        Args:
            attention_weights: (묶음, 머리, 열 길이, 열 길이) 꼴 눈길 텐서
            tokens: 이름표로 쓸 낱말 목록
            head_idx: 그릴 눈길 머리의 번호
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
                   cbar_kws={'label': '눈길 짐'}, ax=ax)

        # 낱말이 있으면 이름표를 붙인다
        if tokens:
            ax.set_xticklabels(tokens, rotation=90)
            ax.set_yticklabels(tokens, rotation=0)

        ax.set_xlabel('열쇠 자리')
        ax.set_ylabel('물음 자리')
        ax.set_title(f'눈길 머리 {head_idx}')

        plt.tight_layout()
        return fig

    def visualize_all_heads(self, attention_weights: torch.Tensor,
                           tokens: Optional[List[str]] = None,
                           max_heads: int = 8,
                           figsize: Tuple[int, int] = (16, 12)) -> plt.Figure:
        """
        눈길 머리 여럿을 격자로 그린다.

        Args:
            attention_weights: (묶음, 머리, 열 길이, 열 길이) 꼴 눈길 텐서
            tokens: 낱말 목록
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

        plt.suptitle('여러 머리 눈길 결', fontsize=16)
        plt.tight_layout()
        return fig

    def visualize_layer_attention(self, layer_name: str,
                                  tokens: Optional[List[str]] = None,
                                  average_heads: bool = True,
                                  figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
        """
        정한 켜의 눈길을 그린다.

        Args:
            layer_name: 켜 이름
            tokens: 낱말 목록
            average_heads: 머리를 가로질러 고르게 할지
            figsize: 그림 크기

        Returns:
            Matplotlib 그림
        """
        if layer_name not in self.attention_maps:
            raise ValueError(f"눈길 그림에 켜 {layer_name}이 없다")

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
                   cbar_kws={'label': '고른 눈길'}, ax=ax)

        if tokens:
            ax.set_xticklabels(tokens, rotation=90)
            ax.set_yticklabels(tokens, rotation=0)

        ax.set_xlabel('열쇠 자리')
        ax.set_ylabel('물음 자리')
        ax.set_title(f'켜: {layer_name} (고른 눈길)')

        plt.tight_layout()
        return fig

    def plot_attention_flow(self, attention_weights: torch.Tensor,
                           tokens: List[str],
                           query_idx: int,
                           figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
        """
        정한 물음 낱말에서 뻗는 눈길 흐름을 그린다.

        Args:
            attention_weights: 눈길 텐서
            tokens: 낱말 목록
            query_idx: 물음 낱말의 번호
            figsize: 그림 크기

        Returns:
            Matplotlib 그림
        """
        if attention_weights.dim() == 4:
            # 묶음과 머리를 가로질러 고르게 한다
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
        ax.set_ylabel('눈길 짐')
        ax.set_title(f'"{tokens[query_idx]}"에서 다른 낱말로 가는 눈길')
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        return fig


class BERTAttentionVisualizer(AttentionVisualizer):
    """
    BERT 결 모형에 맞춘 그리개.
    """

    def extract_attention_from_output(self, outputs):
        """
        BERT 모형의 내놓기에서 눈길 짐을 뽑아낸다.

        Args:
            outputs: BERT(transformers 곳집) 모형의 내놓기

        Returns:
            켜마다 하나씩인 눈길 텐서 목록
        """
        if hasattr(outputs, 'attentions') and outputs.attentions is not None:
            return outputs.attentions
        else:
            warnings.warn("모형 내놓기에 눈길 짐이 없다. "
                        "모형을 output_attentions=True로 부르라")
            return []


def visualize_token_attention(attention_weights: torch.Tensor,
                              tokens: List[str],
                              layer_idx: int = -1,
                              save_path: Optional[str] = None) -> plt.Figure:
    """
    낱말에 대한 눈길을 빠르게 그리는 도움 함수.

    Args:
        attention_weights: 눈길 텐서 (켜, 묶음, 머리, 열, 열)
        tokens: 낱말 목록
        layer_idx: 그릴 켜 번호(-1이면 마지막 켜)
        save_path: 그림을 갈무리할 길

    Returns:
        Matplotlib 그림
    """
    # 켜를 고른다
    if isinstance(attention_weights, (list, tuple)):
        attn = attention_weights[layer_idx]
    else:
        attn = attention_weights

    # 묶음과 머리를 가로질러 고르게 한다
    if attn.dim() == 4:
        attn = attn[0].mean(dim=0)
    elif attn.dim() == 3:
        attn = attn.mean(dim=0)

    attn = attn.cpu().numpy()

    # 그림을 만든다
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(attn, xticklabels=tokens, yticklabels=tokens,
               cmap='viridis', square=True,
               cbar_kws={'label': '눈길 짐'})

    ax.set_xlabel('열쇠 자리')
    ax.set_ylabel('물음 자리')
    ax.set_title(f'눈길 결 (켜 {layer_idx})')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


if __name__ == "__main__":
    pass```

## 논의

그림으로 보이기는 모형의 움직임을 알고 익힘의 탈을 짚어내는 데 큰 몫을 한다. 그리는 코드는 배운 나타냄, 모여 가는 결, 따짐 자를 들여다보게 해서 손에 잡히지 않던 셈을 눈에 보이게 한다.

여기서 보인 결은 더 까다로운 자리로도 자연스레 넓혀진다. 하이퍼파라미터, 얼개의 갈래, 여러 자료를 바꿔 가며 해 보면 이해가 깊어지고 모형 풀이하기에 대한 감이 몸에 붙는다.

## 익힘 문제

**익힘 1.**
코드를 읽고 고갱이가 되는 설계 판단을 짚어라. 짜기에서 고른 것 셋을 들고, 저마다 왜 기울기 바탕 풀이에 알맞은지 밝혀라.

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
    여느 제 눈길은 $n \times n$ 눈길 행렬을 셈하므로 때가 $O(n^2 d)$이고 눈길 짐에 드는 기억이 $O(n^2)$이다. 열이 길면(보기로 $n = 4096$) 감당할 수 없다. Longformer는 그 자리 미끄럼 창 눈길($O(n \cdot w \cdot d)$, $w$은 창 크기)과 고른 낱말에 대한 성긴 온 세상 눈길을 아울러 쓴다. Linformer는 열쇠와 값을 낮은 차원 $k \ll n$으로 쏘아 복잡도를 $O(n \cdot k \cdot d)$으로 줄인다. 둘 다 나타내는 힘을 얼마쯤 내주고 긴 들임에서의 쓸모를 얻는다.

---
**익힘 4.**
눈길 그림 그리기 짜보기를 살피는 두루 갖춘 시험 함수를 써라. 빈 들임, 원소 하나짜리 들임, 아주 큰 들임, 그리고 끝자락 값(0, 아주 큰 수)이 든 들임 같은 가장자리 자리를 시험하여라.

??? success "익힘 4 풀이"
    금 언저리 조건을 두루 건드리는 시험 함수를 짓는다.
    ```python
    def test_attentionvisualizer():
        model = 눈길 그림 그리기(...)
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
