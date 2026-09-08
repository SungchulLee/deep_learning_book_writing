# 눈길 결 살피기

눈길 짐을 그저 그리는 데서 나아가, **눈길 결이 머리와 켜와 갈래를 가로질러 어떻게 짜이는지** 아는 일이 변환기 모형을 풀이하는 데 꼭 있어야 한다. 이 마디는 여러 머리 살피기, 켜를 따라가는 흐름, 엇갈린 눈길 풀이하기, 그림 그리는 연장, 그리고 눈길과 몫 매기기의 종요로운 차이를 한데 모은다.

---

## 1. 여러 머리 눈길 살피기

### 수학 뒷그림

여러 머리 눈길은 $H$개의 눈길 분포를 나란히 셈한다.

$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_H) W^O
$$

여기서 머리마다 이렇게 셈한다.

$$
\text{head}_h = \text{Attention}(Q W_h^Q, K W_h^K, V W_h^V)
$$

머리마다 $d_k = d_{\text{model}} / H$ 차원의 밑밭에서 돌며, 서로 다른 말결이나 얼개 결에 맡은 몫이 갈릴 수 있다.

### 머리가 맡는 결

연구로 되풀이해 나타나는 맡음새가 여럿 드러났다.

| 결 | 밝힘 | 흔한 켜 |
|---------|-------------|---------------|
| **자리** | 이웃한 낱말을 본다 | 앞선 켜 |
| **월 얼개** | 매인 얽힘을 따라간다 | 가운데 켜 |
| **가름표** | 가르는 표([SEP], 문장 부호)를 본다 | 여러 켜 |
| **드문 낱말** | 잦지 않은 낱말에 눈길을 둔다 | 가운데 켜 |
| **뜻** | 뜻이 이어진 개념을 본다 | 뒤쪽 켜 |
| **넓게/고루** | 눈길을 고루 흩는다 | 여러 켜 |

### 여러 머리 그리기

```python
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_all_heads(
    attention_weights: torch.Tensor,
    tokens: list,
    layer: int = 0,
    figsize: tuple = (20, 16)
) -> plt.Figure:
    """
    한 켜의 모든 머리의 눈길 결을 그린다.

    Args:
        attention_weights: [묶음, 머리 수, 열 길이, 열 길이]
        tokens: 낱말 목록
        layer: 그릴 켜 번호
        figsize: 그림 크기

    Returns:
        Matplotlib 그림
    """
    attn = attention_weights[0].detach().cpu().numpy()
    num_heads = attn.shape[0]

    ncols = 4
    nrows = (num_heads + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten()

    for head_idx in range(num_heads):
        ax = axes[head_idx]
        sns.heatmap(
            attn[head_idx],
            xticklabels=tokens,
            yticklabels=tokens,
            ax=ax,
            cmap='Blues',
            vmin=0, vmax=1,
            cbar=False
        )
        ax.set_title(f'머리 {head_idx}', fontsize=10)
        ax.tick_params(labelsize=7)

    for idx in range(num_heads, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(f'켜 {layer} - 모든 눈길 머리', fontsize=14)
    plt.tight_layout()
    return fig
```

### 머리가 서로 얼마나 다른지 살피기

머리마다 얼마나 다르게 보는지 재면 군더더기와 맡음새를 가늠할 수 있다.

```python
def compute_head_diversity(
    attention_weights: torch.Tensor
) -> dict:
    """
    눈길 머리 사이의 다름을 살핀다.

    Args:
        attention_weights: [묶음, 머리 수, 열 길이, 열 길이]

    Returns:
        다름 자를 담은 사전
    """
    attn = attention_weights[0].detach().cpu()
    num_heads = attn.shape[0]
    seq_len = attn.shape[1]

    # 머리마다의 엔트로피
    entropies = []
    for h in range(num_heads):
        head_attn = attn[h]
        eps = 1e-10
        entropy = -(head_attn * torch.log(head_attn + eps)).sum(dim=-1).mean()
        entropies.append(entropy.item())

    # 머리끼리의 코사인 닮음
    flat_heads = attn.reshape(num_heads, -1)
    flat_normed = flat_heads / flat_heads.norm(dim=1, keepdim=True)
    similarity_matrix = (flat_normed @ flat_normed.T).numpy()

    # 대각선 밖의 고른 닮음
    mask = ~np.eye(num_heads, dtype=bool)
    avg_similarity = similarity_matrix[mask].mean()

    return {
        'entropies': entropies,
        'similarity_matrix': similarity_matrix,
        'avg_pairwise_similarity': avg_similarity,
        'diversity_score': 1.0 - avg_similarity
    }
```

### 머리의 중요함과 쳐내기

머리마다 이바지가 같지 않다. 머리를 하나씩 없애 보고 그 미침을 재어 중요함을 어림할 수 있다.

```python
def compute_head_importance(
    model: nn.Module,
    data_loader,
    num_layers: int,
    num_heads: int,
    device: torch.device
) -> np.ndarray:
    """
    기울기 바탕 점수로 머리의 중요함을 셈한다.

    미셸 외(2019)의 방법을 쓴다:
    중요함 = E[|grad(L) * 눈길|]

    Returns:
        importance_matrix: [켜 수, 머리 수]
    """
    importance = torch.zeros(num_layers, num_heads).to(device)

    model.eval()
    for batch in data_loader:
        inputs = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(inputs, labels=labels, output_attentions=True)
        loss = outputs.loss
        attentions = outputs.attentions

        for layer_idx, attn in enumerate(attentions):
            attn.retain_grad()

        loss.backward()

        for layer_idx, attn in enumerate(attentions):
            if attn.grad is not None:
                head_importance = (attn * attn.grad).abs().sum(dim=(0, 2, 3))
                importance[layer_idx] += head_importance

    return importance.cpu().numpy()
```

---

## 2. 켜를 따라가는 눈길 결

### 이론 뒷그림

소식이 변환기의 켜를 지나면서 눈길 결이 짜임새 있게 바뀐다.

- **앞선 켜**(1~3): 그 자리의 월 얼개 얽힘을 담는다. 이웃 낱말 눈길, 자리 결
- **가운데 켜**(4~8): 월 얼개를 담는다. 매인 얽힘, 같은 것 가리키기
- **뒤쪽 켜**(9~12): 추린 뜻 결을 쌓는다. 주제 눈길, 멀리 미치는 매임

이 흐름은 CNN에서 보이는 켜 있는 결 배움과 나란하다.

### 켜 견주기 짜보기

```python
def compare_layers(
    attention_weights_by_layer: list,
    tokens: list,
    layers_to_compare: list = None
) -> plt.Figure:
    """
    변환기 켜를 가로질러 눈길 결을 견준다.

    Args:
        attention_weights_by_layer: 켜마다의 [묶음, 머리, 열, 열] 목록
        tokens: 낱말 목록
        layers_to_compare: 보일 켜
    """
    if layers_to_compare is None:
        n_layers = len(attention_weights_by_layer)
        layers_to_compare = [0, n_layers // 3, 2 * n_layers // 3, n_layers - 1]

    fig, axes = plt.subplots(1, len(layers_to_compare), figsize=(5 * len(layers_to_compare), 5))

    for idx, layer in enumerate(layers_to_compare):
        attn = attention_weights_by_layer[layer][0].mean(dim=0).detach().cpu().numpy()

        sns.heatmap(
            attn,
            xticklabels=tokens,
            yticklabels=tokens,
            ax=axes[idx],
            cmap='Blues',
            vmin=0,
            cbar=idx == len(layers_to_compare) - 1
        )
        axes[idx].set_title(f'켜 {layer}')

    plt.tight_layout()
    return fig

def compute_locality_score(attention_weights: torch.Tensor, window: int = 3) -> float:
    """
    눈길이 그 자리에 머무는지 멀리 미치는지 잰다.

    그 자리 셈 = 대각선에서 ±창 안에 든 눈길 무게의 몫.
    """
    attn = attention_weights[0].mean(dim=0).detach().cpu().numpy()
    seq_len = attn.shape[0]

    local_mass = 0.0
    total_mass = 0.0

    for i in range(seq_len):
        for j in range(seq_len):
            total_mass += attn[i, j]
            if abs(i - j) <= window:
                local_mass += attn[i, j]

    return local_mass / total_mass if total_mass > 0 else 0.0
```

### 고갱이 열매

여러 변환기 얼개에 걸친 연구에서 한결같은 결이 드러난다.

1. **깊어질수록 그 자리에 덜 머문다**: 앞선 켜는 그 자리를 보고(창 3 안에 ~80%), 뒤쪽 켜는 멀리 본다(~40%)
2. **머리를 쳐내도 견디는 만큼이 켜마다 다르다**: 뒤쪽 켜가 머리를 없애도 더 든든하다
3. **나머지 이음이 중요하다**: 소식이 건너뛰는 이음으로 눈길을 지나치므로, 눈길만으로는 온 소식 흐름을 담을 수 없다

---

## 3. 엇갈린 눈길 풀이하기

엇갈린 눈길은 서로 다른 두 열을 이어 부호기와 푸는 개 사이에 소식이 흐르게 한다. 옮김, 간추리기, 물음 답하기 얼개를 풀이하려면 엇갈린 눈길을 알아야 한다.

### 수학 밑바탕

부호기의 나타냄 $K^e, V^e$과 푸는 개의 물음 $Q^d$이 주어지면

$$
\text{CrossAttn}(Q^d, K^e, V^e) = \text{softmax}\left(\frac{Q^d (K^e)^\top}{\sqrt{d_k}}\right) V^e
$$

눈길 짐이 맞춤 행렬 $A \in \mathbb{R}^{T_d \times T_e}$을 이루며, $A_{ij}$은 푸는 개의 자리 $i$이 부호기의 자리 $j$을 얼마나 보는지 알린다.

### 짜보기

```python
def visualize_cross_attention(
    cross_attention: torch.Tensor,
    source_tokens: list,
    target_tokens: list,
    head: int = None
) -> plt.Figure:
    """
    부호기-푸는 개의 엇갈린 눈길을 그린다.

    Args:
        cross_attention: [묶음, 머리, 받는 열 길이, 보내는 열 길이]
        source_tokens: 부호기 들임 낱말
        target_tokens: 푸는 개 내놓기 낱말
        head: 정한 머리(None이면 모든 머리를 고르게 한다)
    """
    if head is not None:
        attn = cross_attention[0, head].detach().cpu().numpy()
        title = f'엇갈린 눈길 (머리 {head})'
    else:
        attn = cross_attention[0].mean(dim=0).detach().cpu().numpy()
        title = '엇갈린 눈길 (고르게 함)'

    fig, ax = plt.subplots(figsize=(10, 8))

    sns.heatmap(
        attn,
        xticklabels=source_tokens,
        yticklabels=target_tokens,
        ax=ax,
        cmap='Blues',
        vmin=0, vmax=1
    )
    ax.set_xlabel('보내는 쪽 (부호기)')
    ax.set_ylabel('받는 쪽 (푸는 개)')
    ax.set_title(title)

    plt.tight_layout()
    return fig

def analyze_cross_attention_alignment(
    cross_attention: torch.Tensor,
    source_tokens: list,
    target_tokens: list
) -> dict:
    """
    엇갈린 눈길에서 맞춤의 됨됨이를 살핀다.

    맞춤이 얼마나 또렷하고 얼마나 덮는지를 자로 내놓는다.
    """
    attn = cross_attention[0].mean(dim=0).detach().cpu()

    # 맞춤 엔트로피(작을수록 맞춤이 또렷하다)
    eps = 1e-10
    entropy = -(attn * torch.log(attn + eps)).sum(dim=-1).mean().item()

    # 덮음: 문턱을 넘게 눈길을 받은 보내는 쪽 낱말의 몫
    max_attn_per_source = attn.max(dim=0)[0]
    coverage = (max_attn_per_source > 0.1).float().mean().item()

    # 한 방향으로 감: 맞춤이 얼마나 한 방향인가
    argmax_positions = attn.argmax(dim=-1).float()
    diffs = argmax_positions[1:] - argmax_positions[:-1]
    monotonicity = (diffs >= 0).float().mean().item()

    return {
        'alignment_entropy': entropy,
        'source_coverage': coverage,
        'monotonicity': monotonicity
    }
```

---

## 4. 그림 그리는 연장: BertViz과 손수 만든 것

### BertViz 살펴보기

BertViz(빅, 2019)은 그리는 결 셋을 준다.

| 결 | 보이는 것 | 쓰일 자리 |
|------|-------|----------|
| **눈길 머리 봄** | 머리 하나의 눈길 결 | 정한 머리의 움직임 살피기 |
| **모형 봄** | 모든 켜의 모든 머리 | 눈길 퍼짐을 두루 보기 |
| **신경 세포 봄** | 물음-열쇠 쪼갬 | 무엇이 눈길을 이끄는지 알기 |

### BertViz 쓰기

```python
from bertviz import head_view, model_view
from transformers import AutoTokenizer, AutoModel

def interactive_attention_visualization(
    model_name: str,
    text: str,
    text_pair: str = None
):
    """
    주고받는 BertViz 그림을 만든다.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, output_attentions=True)

    inputs = tokenizer(
        text, text_pair,
        return_tensors='pt',
        padding=True,
        truncation=True
    )

    with torch.no_grad():
        outputs = model(**inputs)

    attention = outputs.attentions
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

    # 머리 봄 - 머리 하나를 촘촘히 살핀다
    head_view(attention, tokens)

    # 모형 봄 - 모든 머리, 모든 켜
    model_view(attention, tokens)
```

### 손수 그림 짓기

```python
def create_attention_heatmap_grid(
    attention_by_layer: list,
    tokens: list,
    num_layers: int = 12,
    num_heads: int = 12,
    figsize: tuple = (24, 20)
) -> plt.Figure:
    """
    켜 × 머리의 눈길 격자를 두루 만든다.
    """
    fig, axes = plt.subplots(num_layers, num_heads, figsize=figsize)

    for layer_idx in range(num_layers):
        for head_idx in range(num_heads):
            attn = attention_by_layer[layer_idx][0, head_idx].detach().cpu().numpy()
            ax = axes[layer_idx, head_idx]
            ax.imshow(attn, cmap='Blues', vmin=0, vmax=1, aspect='auto')
            ax.set_xticks([])
            ax.set_yticks([])

            if layer_idx == 0:
                ax.set_title(f'H{head_idx}', fontsize=8)
            if head_idx == 0:
                ax.set_ylabel(f'L{layer_idx}', fontsize=8)

    fig.suptitle('눈길 결: 켜(가로줄) × 머리(세로줄)', fontsize=14)
    plt.tight_layout()
    return fig

def create_attention_flow_sankey(
    attention_weights: torch.Tensor,
    tokens: list,
    source_idx: int,
    top_k: int = 5
):
    """
    정한 낱말에서 눈길이 어디로 흐르는지 보이는
    생키 결의 흐름 그림을 만든다.
    """
    attn = attention_weights[0].detach().cpu().numpy()
    num_heads = attn.shape[0]

    flows = []
    for head_idx in range(num_heads):
        head_attn = attn[head_idx, source_idx]
        top_targets = np.argsort(head_attn)[::-1][:top_k]

        for target_idx in top_targets:
            if head_attn[target_idx] > 0.05:
                flows.append({
                    'head': head_idx,
                    'source': tokens[source_idx],
                    'target': tokens[target_idx],
                    'weight': head_attn[target_idx]
                })

    return flows
```

---

## 5. 눈길 대 몫 매기기: 종요로운 가름

### 고갱이 문제

모형 풀이하기에서 종요로운 가름은 **눈길 짐**과 **몫 점수**의 차이다.

- **눈길**은 모형이 어디를 "보는지" 보인다. 곧 들임 낱말에 걸친 짐의 퍼짐이다
- **몫 매기기**는 무엇이 참으로 내놓기를 흔드는지, 곧 들임마다의 인과 이바지를 드러낸다

이 둘은 **같은 것이 아니다**. 제인과 월리스(2019)는 눈길 짐이 기울기 바탕 몫과 얽히지 않을 때가 많으며, 눈길 분포를 달리해도 똑같은 미루어 봄이 나올 수 있음을 밝혔다.

### 눈길 ≠ 풀이인 까닭

눈길을 풀이 방법으로 삼기 어렵게 하는 몫이 여럿 있다.

**값 바꿈**: 눈길 짐은 어느 값 벡터를 아우를지 정하지만, 값 벡터 자체도 바뀐다. 어떤 낱말에 눈길이 크다고 그 낱말의 소식이 내놓기를 판치는 것은 아니다.

$$
\text{output}_i = \sum_j \alpha_{ij} V_j W^V
$$

이바지는 $\alpha_{ij}$과 $V_j$의 속살에 함께 달렸다.

**나머지 이음**: 소식이 건너뛰는 이음으로 눈길을 지나친다. 내놓기 나타냄에는 눈길이 모은 소식과 본디 들임이 함께 들어 있으므로 눈길 짐만으로는 그림이 온전하지 않다.

$$
\mathbf{h}_i^{(l+1)} = \mathbf{h}_i^{(l)} + \text{Attn}(\mathbf{h}^{(l)})
$$

**여러 켜 겹침**: 여러 켜 변환기에서는 한 낱말의 소식이 여러 눈길 길을 거쳐 내놓기에 이를 수 있다. 켜 하나의 눈길로는 이런 에두른 미침을 담을 수 없다.

### 눈길과 몫 매기기 견주기

```python
def compare_attention_and_attribution(
    model,
    input_ids: torch.Tensor,
    target_class: int,
    layer: int = -1,
    device: torch.device = None
) -> dict:
    """
    눈길 짐과 기울기 바탕 몫을 견준다.
    """
    model.eval()
    input_ids = input_ids.to(device)

    # 눈길 짐을 얻는다
    with torch.no_grad():
        outputs = model(input_ids, output_attentions=True)
        attention = outputs.attentions[layer][0].mean(dim=0)

    # 기울기 바탕 몫을 얻는다
    embeddings = model.get_input_embeddings()(input_ids)
    embeddings.requires_grad_(True)

    outputs = model(inputs_embeds=embeddings)
    logits = outputs.logits
    logits[0, target_class].backward()

    gradient_attr = embeddings.grad.abs().sum(dim=-1)[0]

    # [CLS] 낱말에서 뻗는 눈길(흔히 고른다)
    attn_scores = attention[0].detach().cpu().numpy()
    grad_scores = gradient_attr.detach().cpu().numpy()

    # 고르게 한다
    attn_scores = attn_scores / attn_scores.sum()
    grad_scores = grad_scores / grad_scores.sum()

    # 얽힘
    from scipy.stats import spearmanr
    correlation, p_value = spearmanr(attn_scores, grad_scores)

    return {
        'attention_scores': attn_scores,
        'gradient_scores': grad_scores,
        'spearman_correlation': correlation,
        'p_value': p_value
    }
```

### 어느 것을 언제 쓸까

| 방법 | 알맞은 자리 | 한계 |
|--------|----------|------------|
| **눈길 짐** | 모형의 움직임 둘러보기, 얼개 알기, 가설 세우기 | 미더운 몫이 아니고 값 속살에 흔들린다 |
| **눈길 굴리기** | 켜를 가로지르는 소식 흐름 좇기 | 눈길 ≈ 소식 흐름이라고 여긴다 |
| **눈길 흐름** | 여러 켜에 걸쳐 더 미더운 몫 매기기 | 셈이 비싸다 |
| **기울기 몫 매기기** | 미더운 중요함 재기 | 눈길만의 깨침을 놓친다 |
| **아울러 쓰기** | 가장 두루 알기 | 풀이하기가 더 까다롭다 |

---

## 6. 온전한 보기

```python
def comprehensive_attention_analysis(
    model,
    tokenizer,
    text: str,
    device: torch.device
) -> dict:
    """
    여러 재주를 아우른 두루 갖춘 눈길 살피기.
    """
    # 낱말로 쪼갠다
    inputs = tokenizer(text, return_tensors='pt', padding=True)
    input_ids = inputs['input_ids'].to(device)
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

    # 눈길을 곁들인 앞으로 걸음
    model.eval()
    with torch.no_grad():
        outputs = model(**{k: v.to(device) for k, v in inputs.items()},
                       output_attentions=True)

    attentions = outputs.attentions

    results = {
        'tokens': tokens,
        'num_layers': len(attentions),
        'num_heads': attentions[0].shape[1],
        'analyses': {}
    }

    # 켜마다 살핀다
    for layer_idx, attn in enumerate(attentions):
        diversity = compute_head_diversity(attn)
        locality = compute_locality_score(attn)

        results['analyses'][f'layer_{layer_idx}'] = {
            'diversity_score': diversity['diversity_score'],
            'head_entropies': diversity['entropies'],
            'locality_score': locality
        }

    # 켜를 가로지르는 그 자리 머묾의 흐름
    localities = [
        results['analyses'][f'layer_{i}']['locality_score']
        for i in range(len(attentions))
    ]
    results['locality_progression'] = localities

    return results
```

---

## 연습문제

**연습문제 1.**
이 마디에서 밝힌 풀이 방법을, XOR 들임을 가르는 ReLU 살림의 두 켜 신경 그물에 걸어라. 들임 $x = [1, 1]$에 대한 풀이를 셈하여라.

??? success "연습문제 1 풀이"
    짐이 $W_1, b_1, W_2, b_2$인 익힌 XOR 그물에서 내놓기는 $f(x) = W_2 \cdot \text{ReLU}(W_1 x + b_1) + b_2$이다. 풀이 방법은 들임 결마다 몫을 내놓는다. $x = [1, 1]$(갈래 0)이면 두 결 모두 음수 가름에 이바지한다. 몫 값은 방법마다 다르다. 기울기 바탕 방법은 $\partial f / \partial x_i$을 셈하고, 흔들어 보는 방법은 결을 가렸을 때 내놓기가 얼마나 바뀌는지 잰다. XOR 문제는 판단 금이 선형이 아니므로 선형 풀이 방법이 그르칠 수 있음을 보여 준다. $\square$

---

**연습문제 2.**
이 마디의 풀이 방법이 온전함 공리를 채우는지, 곧 어떤 밑금 $x_0$에 대해 모든 결 몫의 합이 $f(x) - f(x_0)$과 같은지 증명하거나 뒤집어라.

??? success "연습문제 2 풀이"
    온전함 공리(섀플리 값 이론에서는 효율이라고도 한다)는 몫의 합이 들임에서의 모형 내놓기와 밑금에서의 내놓기의 차이와 같다는 것이다. 이 방법이 온전함을 채우는지는 그 세움새에 달렸다. 기울기 방법은 온전함을 채우지 못한다(기울기는 그 자리의 것이고 길을 따라 쌓은 것이 아니다). 쌓은 기울기는 세움새 자체로 온전함을 채운다(길을 따라 미적분의 밑정리를 쓴다). SHAP 값은 섀플리 공리로 효율을 채운다. 온전함을 어기는 방법은 몫을 너무 많거나 적게 매길 수 있어, 온 몫을 온 세상 풀이로 믿기 어렵게 만든다. $\square$

---

**연습문제 3.**
이 방법이 내놓는 풀이가 얼마나 미더운지 따지는 시험을 꾸며라. 짚어 준 결이 참으로 모형에 중요한지를 넣기와 빼기 곡선으로 재어라.

??? success "연습문제 3 풀이"
    절차는 이렇다. (1) 시험 그림마다 결 몫을 셈한다. (2) 빼기: 몫이 큰 차례로 결을 하나씩 가리며 모형의 자신함이 떨어지는 모습을 적는다. 미더운 풀이면 자신함이 빠르게 떨어진다. (3) 넣기: 빈 밑금에서 시작해 몫이 큰 차례로 결을 하나씩 드러내며 자신함이 오르는 모습을 적는다. 미더운 풀이면 자신함이 빠르게 오른다. (4) 두 곡선의 아래 넓이를 셈한다. (5) 아무렇게나 매긴 차례(밑금)와 다른 방법에 견준다. 미더운 방법이면 빼기 넓이가 작고 넣기 넓이가 커야 한다. 통계로 미더우려면 시험 표본 1000개 넘게 되풀이한다. $\square$

---

**연습문제 4.**
이 풀이 방법을 신용 부도를 미루어 보는 금융 모형에 어떻게 걸 수 있는지 다루어라. 풀이가 채워야 할 규정 요건은 무엇인가?

??? success "연습문제 4 풀이"
    신용 모형에는 규정(ECOA, GDPR 22조)이 불리한 판단마다 그 사람에게 맞춘 풀이를 바란다. 방법은 다음을 내놓아야 한다. (1) 물리침에 가장 크게 이바지한 인자(불리한 처분 까닭). (2) 한결같은 풀이(비슷한 신청자는 비슷한 풀이를 받는다). (3) 손에 잡히는 풀이(신청자가 무엇을 바꾸어야 하는지 안다). 이 마디의 풀이 방법으로 결의 중요함을 짚을 수 있으나, 든든함(들임이 조금 바뀌었다고 풀이가 확 달라지면 안 된다)과 옳음(중요한 결을 없애면 미루어 봄이 바뀌어야 한다)을 따져 보아야 한다. 지켜야 할 됨됨이는 대리 차별이 드러나지 않도록 조심히 다루어야 한다. $\square$

## 정리하며

눈길 결 살피기는 변환기의 움직임을 넉넉히 들여다보게 해 주지만 조심히 읽어야 한다.

1. **여러 머리의 다름**이 맡음새를 드러낸다. 너무 비슷한 머리는 쳐낼 수 있다
2. **켜를 따라가는 흐름**이 그 자리에서 온 세상으로 가는 켜 있는 결 쌓기를 보인다
3. **엇갈린 눈길의 맞춤**은 부호기-푸는 개 모형에서 알려 주는 바가 많다
4. **눈길 ≠ 몫 매기기**: 눈길로 얻은 깨침은 늘 기울기 바탕 방법으로 따져 보라
5. BertViz 같은 **주고받는 연장**은 둘러보기에 값지지만 수로 재는 자를 곁들여야 한다

**살펴볼 거리**

1. Vig, J. (2019). "A Multiscale Visualization of Attention in the Transformer Model." *ACL Demo*.

2. Clark, K., et al. (2019). "What Does BERT Look At? An Analysis of BERT's Attention." *BlackboxNLP*.

3. Jain, S., & Wallace, B. C. (2019). "Attention is not Explanation." *NAACL*.

4. Wiegreffe, S., & Pinter, Y. (2019). "Attention is not not Explanation." *EMNLP*.

5. Michel, P., Levy, O., & Neubig, G. (2019). "Are Sixteen Heads Really Better than One?" *NeurIPS*.

6. Voita, E., et al. (2019). "Analyzing Multi-Head Self-Attention: Specialized Heads Do the Heavy Lifting." *ACL*.

7. Abnar, S., & Zuidema, W. (2020). "Quantifying Attention Flow in Transformers." *ACL*.
