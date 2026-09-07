# 쓰는 보기

쓰는 보기: 변환기의 눈길 그림 그리기. 이 글은 변환기 모형의 눈길 결을 그리는 법을 보인다.

신경 그물이 무엇을 배우는지 아는 일은 믿음을 쌓고 모형의 벌레를 잡는 데 종요롭다. 이 꾸러미는 모형이 들임을 어떻게 다루고 어떻게 판단하는지 드러내는 기울기 바탕 풀이 재주를 보이며, 그물의 움직임을 눈으로 보고 수로 재게 해 준다.

## 코드

```python
"""
쓰는 보기: 변환기의 눈길 그림 그리기

이 글은 변환기 모형의 눈길 결을 그리는 법을 보인다.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from attention_visualization import (

# ========================================================================
# 메인
# ========================================================================
    AttentionVisualizer, 
    BERTAttentionVisualizer,
    visualize_token_attention
)

# 짚을 것: 이 보기들은 transformers 곳집이 있어야 한다
# 이렇게 깐다: pip install transformers


def example_simple_attention():
    """
    보기 1: 지어낸 눈길 결 그리기
    """
    print("=" * 60)
    print("보기 1: 단순한 눈길 그림 그리기")
    print("=" * 60)

    # 지어낸 눈길 짐을 만든다
    # 꼴: (묶음=1, 머리=8, 열 길이=10, 열 길이=10)
    seq_len = 10
    num_heads = 8

    attention = torch.zeros(1, num_heads, seq_len, seq_len)

    # 머리마다 다른 결을 만든다
    for h in range(num_heads):
        if h == 0:  # 대각선 결(제 눈길)
            attention[0, h] = torch.eye(seq_len)
        elif h == 1:  # 앞 낱말을 본다
            attention[0, h] = torch.diag(torch.ones(seq_len - 1), -1)
        elif h == 2:  # 첫 낱말을 본다
            attention[0, h, :, 0] = 1.0
        else:  # 아무렇게나 만든 결
            attention[0, h] = torch.softmax(torch.randn(seq_len, seq_len), dim=-1)

    # 낱말을 만든다
    tokens = [f"Token_{i}" for i in range(seq_len)]

    # 그림으로 보인다
    print("그림을 만드는 중...")
    fig = visualize_token_attention(attention, tokens, layer_idx=0)
    plt.savefig('attention_simple.png', dpi=150, bbox_inches='tight')
    print("'attention_simple.png'에 갈무리했다")
    plt.close()

    # 모든 머리를 그린다
    from attention_visualization import AttentionVisualizer
    visualizer = AttentionVisualizer(None)  # 이 보기에는 모형이 없어도 된다
    fig = visualizer.visualize_all_heads(attention, tokens, max_heads=8)
    plt.savefig('attention_all_heads.png', dpi=150, bbox_inches='tight')
    print("'attention_all_heads.png'에 갈무리했다")
    plt.close()


def example_bert_attention():
    """
    보기 2: BERT 눈길 그리기

    있어야 할 것: pip install transformers
    """
    print("\n" + "=" * 60)
    print("보기 2: BERT 눈길 그림 그리기")
    print("=" * 60)

    try:
        from transformers import BertTokenizer, BertModel
    except ImportError:
        print("transformers 곳집이 없다. 이렇게 깐다: pip install transformers")
        return

    # 모형과 낱말 쪼개개를 부른다
    print("BERT 모형을 부르는 중...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased', output_attentions=True)
    model.eval()

    # 들임 글
    text = "The quick brown fox jumps over the lazy dog."
    print(f"들임 글: {text}")

    # 낱말로 쪼갠다
    inputs = tokenizer(text, return_tensors='pt')
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    print(f"낱말: {tokens}")

    # 눈길을 곁들인 모형 내놓기를 얻는다
    with torch.no_grad():
        outputs = model(**inputs)

    # 눈길 짐을 뽑아낸다
    # outputs.attentions은 켜 수만큼의 텐서 튜플이다
    # 텐서마다 꼴이 (묶음, 머리, 열 길이, 열 길이)다
    attentions = outputs.attentions

    print(f"\n켜의 수: {len(attentions)}")
    print(f"켜마다의 눈길 꼴: {attentions[0].shape}")

    # 마지막 켜를 그린다
    print("\n마지막 켜의 눈길을 그리는 중...")
    fig = visualize_token_attention(attentions, tokens, layer_idx=-1)
    plt.savefig('bert_attention_last_layer.png', dpi=150, bbox_inches='tight')
    print("'bert_attention_last_layer.png'에 갈무리했다")
    plt.close()

    # 마지막 켜의 모든 머리를 그린다
    visualizer = BERTAttentionVisualizer(model)
    fig = visualizer.visualize_all_heads(attentions[-1], tokens, max_heads=12)
    plt.savefig('bert_attention_all_heads.png', dpi=150, bbox_inches='tight')
    print("'bert_attention_all_heads.png'에 갈무리했다")
    plt.close()

    # 정한 낱말에서 뻗는 눈길 흐름을 그린다
    query_token = "fox"
    query_idx = tokens.index(query_token)
    fig = visualizer.plot_attention_flow(attentions[-1], tokens, query_idx)
    plt.savefig('bert_attention_flow.png', dpi=150, bbox_inches='tight')
    print("'bert_attention_flow.png'에 갈무리했다")
    plt.close()


def example_gpt2_attention():
    """
    보기 3: GPT-2 눈길 그리기

    있어야 할 것: pip install transformers
    """
    print("\n" + "=" * 60)
    print("보기 3: GPT-2 눈길 그림 그리기")
    print("=" * 60)

    try:
        from transformers import GPT2Tokenizer, GPT2LMHeadModel
    except ImportError:
        print("transformers 곳집이 없다. 이렇게 깐다: pip install transformers")
        return

    # 모형과 낱말 쪼개개를 부른다
    print("GPT-2 모형을 부르는 중...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2', output_attentions=True)
    model.eval()

    # 들임 글
    text = "Artificial intelligence is transforming"
    print(f"들임 글: {text}")

    # 낱말로 쪼갠다
    inputs = tokenizer(text, return_tensors='pt')
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    print(f"낱말: {tokens}")

    # 눈길을 곁들인 내놓기를 얻는다
    with torch.no_grad():
        outputs = model(**inputs)

    attentions = outputs.attentions

    print(f"\n켜의 수: {len(attentions)}")
    print(f"켜마다의 눈길 꼴: {attentions[0].shape}")

    # 인과 눈길 결을 그린다(GPT-2은 인과 가리개를 쓴다)
    print("\n인과 눈길 결을 그리는 중...")
    fig = visualize_token_attention(attentions, tokens, layer_idx=-1)
    plt.savefig('gpt2_attention_causal.png', dpi=150, bbox_inches='tight')
    print("'gpt2_attention_causal.png'에 갈무리했다")
    plt.close()

    # 여러 켜를 그린다
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    layer_indices = [0, 2, 4, 6, 8, 11]  # 뽑아 본 켜

    for idx, layer_idx in enumerate(layer_indices):
        attn = attentions[layer_idx][0].mean(dim=0).cpu().numpy()

        im = axes[idx].imshow(attn, cmap='viridis', aspect='auto')
        axes[idx].set_title(f'켜 {layer_idx}')
        axes[idx].set_xlabel('열쇠 자리')
        axes[idx].set_ylabel('물음 자리')

        if len(tokens) <= 10:
            axes[idx].set_xticks(range(len(tokens)))
            axes[idx].set_yticks(range(len(tokens)))
            axes[idx].set_xticklabels(tokens, rotation=90, fontsize=8)
            axes[idx].set_yticklabels(tokens, fontsize=8)

    plt.tight_layout()
    plt.savefig('gpt2_attention_layers.png', dpi=150, bbox_inches='tight')
    print("'gpt2_attention_layers.png'에 갈무리했다")
    plt.close()


def example_attention_patterns():
    """
    보기 4: 여러 눈길 결 살피기
    """
    print("\n" + "=" * 60)
    print("보기 4: 흔한 눈길 결")
    print("=" * 60)

    seq_len = 12
    tokens = [f"T{i}" for i in range(seq_len)]

    # 여러 눈길 결을 만든다
    patterns = {
        '그 자리': torch.zeros(seq_len, seq_len),
        '온 세상': torch.zeros(seq_len, seq_len),
        '인과': torch.zeros(seq_len, seq_len),
        '띄엄띄엄': torch.zeros(seq_len, seq_len),
    }

    # 그 자리 눈길(창 크기 3)
    for i in range(seq_len):
        start = max(0, i - 1)
        end = min(seq_len, i + 2)
        patterns['그 자리'][i, start:end] = 1.0
    patterns['그 자리'] = torch.softmax(patterns['그 자리'], dim=-1)

    # 온 세상 눈길(모두를 본다)
    patterns['온 세상'] = torch.ones(seq_len, seq_len) / seq_len

    # 인과 눈길(지난 것만 본다)
    for i in range(seq_len):
        patterns['인과'][i, :i+1] = 1.0
    patterns['인과'] = torch.softmax(patterns['인과'], dim=-1)

    # 띄엄띄엄 눈길(일정한 사이의 자리를 본다)
    for i in range(seq_len):
        for j in range(0, seq_len, 3):
            patterns['띄엄띄엄'][i, j] = 1.0
    patterns['띄엄띄엄'] = torch.softmax(patterns['띄엄띄엄'], dim=-1)

    # 모든 결을 그린다
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.flatten()

    for idx, (name, pattern) in enumerate(patterns.items()):
        im = axes[idx].imshow(pattern.numpy(), cmap='viridis', aspect='auto')
        axes[idx].set_title(f'{name} 눈길 결')
        axes[idx].set_xlabel('열쇠 자리')
        axes[idx].set_ylabel('물음 자리')
        plt.colorbar(im, ax=axes[idx])

    plt.tight_layout()
    plt.savefig('attention_patterns.png', dpi=150, bbox_inches='tight')
    print("'attention_patterns.png'에 갈무리했다")
    plt.close()


def example_attention_statistics():
    """
    보기 5: 눈길 셈속 셈하기
    """
    print("\n" + "=" * 60)
    print("보기 5: 눈길 셈속")
    print("=" * 60)

    # 지어낸 여러 켜 눈길을 만든다
    num_layers = 12
    num_heads = 8
    seq_len = 16

    attentions = []
    for _ in range(num_layers):
        attn = torch.softmax(torch.randn(1, num_heads, seq_len, seq_len), dim=-1)
        attentions.append(attn)

    # 셈속을 셈한다
    print("\n눈길 셈속을 셈하는 중...")

    # 엔트로피(눈길이 얼마나 모여 있는가?)
    entropies = []
    for layer_attn in attentions:
        # 묶음과 머리를 가로질러 고르게 한다
        attn = layer_attn[0].mean(dim=0)
        # 물음 자리마다 엔트로피를 셈한다
        entropy = -(attn * torch.log(attn + 1e-10)).sum(dim=-1).mean()
        entropies.append(entropy.item())

    # 눈길 거리(눈길이 얼마나 멀리 뻗는가?)
    distances = []
    for layer_attn in attentions:
        attn = layer_attn[0].mean(dim=0)
        positions = torch.arange(seq_len).float()
        avg_dist = 0
        for i in range(seq_len):
            weighted_pos = (attn[i] * positions).sum()
            avg_dist += abs(weighted_pos - i)
        distances.append(avg_dist.item() / seq_len)

    # 셈속을 그린다
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(range(num_layers), entropies, marker='o')
    axes[0].set_xlabel('켜')
    axes[0].set_ylabel('고른 엔트로피')
    axes[0].set_title('켜마다의 눈길 엔트로피')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(range(num_layers), distances, marker='o', color='coral')
    axes[1].set_xlabel('켜')
    axes[1].set_ylabel('고른 거리')
    axes[1].set_title('켜마다의 눈길 거리')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('attention_statistics.png', dpi=150, bbox_inches='tight')
    print("'attention_statistics.png'에 갈무리했다")
    plt.close()

    print(f"\n켜마다의 셈속:")
    for i in range(num_layers):
        print(f"  켜 {i}: 엔트로피={entropies[i]:.3f}, 거리={distances[i]:.3f}")


if __name__ == "__main__":
    print("눈길 그림 그리기 보기\n")

    # 보기를 돌린다
    example_simple_attention()
    example_attention_patterns()
    example_attention_statistics()

    # 이것들은 transformers 곳집이 있어야 한다
    try:
        example_bert_attention()
        example_gpt2_attention()
    except Exception as e:
        print(f"\n변환기 보기를 건너뛴다: {e}")
        print("transformers을 이렇게 깐다: pip install transformers")

    print("\n" + "=" * 60)
    print("보기를 모두 마쳤다!")
    print("=" * 60)```

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
    여느 제 눈길은 $n \times n$ 눈길 행렬을 셈하므로 때가 $O(n^2 d)$이고 눈길 짐에 드는 기억이 $O(n^2)$이다. 열이 길면(보기로 $n = 4096$) 감당할 수 없다. Longformer은 그 자리 미끄럼 창 눈길($O(n \cdot w \cdot d)$, $w$은 창 크기)과 고른 낱말에 대한 성긴 온 세상 눈길을 아울러 쓴다. Linformer은 열쇠와 값을 낮은 차원 $k \ll n$으로 쏘아 복잡도를 $O(n \cdot k \cdot d)$으로 줄인다. 둘 다 나타내는 힘을 얼마쯤 내주고 긴 들임에서의 쓸모를 얻는다.

---
**익힘 4.**
쓰는 보기 짜보기를 살피는 두루 갖춘 시험 함수를 써라. 빈 들임, 원소 하나짜리 들임, 아주 큰 들임, 그리고 끝자락 값(0, 아주 큰 수)이 든 들임 같은 가장자리 자리를 시험하여라.

??? success "익힘 4 풀이"
    금 언저리 조건을 두루 건드리는 시험 함수를 짓는다.
    ```python
    def test_example usage():
        model = 쓰는 보기(...)
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
