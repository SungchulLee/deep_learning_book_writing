# 어텐션 패턴 분석

어텐션 웨이트를 그저 그리는 데서 나아가, **어텐션 패턴이 헤드와 레이어에 따라 어떻게 달라지는지** 아는 일이 트랜스포머 모델을 해석하는 데 필요하다. 이 절은 멀티헤드 분석, 레이어에 따른 변화, 크로스 어텐션 해석, 시각화 도구, 그리고 어텐션과 기여도 분석의 중요한 차이를 한데 모은다.

---

## 1. 멀티헤드 어텐션 살피기

### 수학 배경

멀티헤드 어텐션은 $H$개의 어텐션 분포를 나란히 계산한다.

$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_H) W^O
$$

여기서 헤드마다 이렇게 계산한다.

$$
\text{head}_h = \text{Attention}(Q W_h^Q, K W_h^K, V W_h^V)
$$

헤드마다 $d_k = d_{\text{model}} / H$ 차원의 부분공간에서 돌므로, 헤드마다 서로 다른 언어적·구조적 패턴을 맡을 수 있다.

### 헤드가 맡는 패턴

연구에서 되풀이해 나타나는 역할이 여럿 드러났다.

| 패턴 | 설명 | 흔한 레이어 |
|---------|-------------|---------------|
| **위치** | 이웃한 단어를 본다 | 앞쪽 레이어 |
| **구문** | 의존 관계를 따라간다 | 가운데 레이어 |
| **구분자** | 구분 토큰([SEP], 문장 부호)을 본다 | 여러 레이어 |
| **드문 단어** | 드물게 나오는 단어에 어텐션을 둔다 | 가운데 레이어 |
| **의미** | 의미가 이어진 개념을 본다 | 뒤쪽 레이어 |
| **넓게 퍼짐** | 어텐션을 고르게 퍼뜨린다 | 여러 레이어 |

### 여러 헤드 시각화

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
    한 레이어의 모든 헤드의 어텐션 패턴을 그린다.

    Args:
        attention_weights: [배치, 헤드 수, 시퀀스 길이, 시퀀스 길이]
        tokens: 토큰 목록
        layer: 그릴 레이어 번호
        figsize: figure 크기

    Returns:
        Matplotlib figure
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
        ax.set_title(f'헤드 {head_idx}', fontsize=10)
        ax.tick_params(labelsize=7)

    for idx in range(num_heads, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(f'레이어 {layer} - 모든 어텐션 헤드', fontsize=14)
    plt.tight_layout()
    return fig
```

### 헤드가 서로 얼마나 다른지 살피기

헤드마다 얼마나 다르게 보는지 재면 헤드 사이의 중복과 역할을 가늠할 수 있다.

```python
def compute_head_diversity(
    attention_weights: torch.Tensor
) -> dict:
    """
    어텐션 헤드 사이의 다양성을 분석한다.

    Args:
        attention_weights: [배치, 헤드 수, 시퀀스 길이, 시퀀스 길이]

    Returns:
        다양성 지표를 담은 딕셔너리
    """
    attn = attention_weights[0].detach().cpu()
    num_heads = attn.shape[0]
    seq_len = attn.shape[1]

    # 헤드마다의 엔트로피
    entropies = []
    for h in range(num_heads):
        head_attn = attn[h]
        eps = 1e-10
        entropy = -(head_attn * torch.log(head_attn + eps)).sum(dim=-1).mean()
        entropies.append(entropy.item())

    # 헤드끼리의 코사인 유사도
    flat_heads = attn.reshape(num_heads, -1)
    flat_normed = flat_heads / flat_heads.norm(dim=1, keepdim=True)
    similarity_matrix = (flat_normed @ flat_normed.T).numpy()

    # 대각선 밖의 평균 유사도
    mask = ~np.eye(num_heads, dtype=bool)
    avg_similarity = similarity_matrix[mask].mean()

    return {
        'entropies': entropies,
        'similarity_matrix': similarity_matrix,
        'avg_pairwise_similarity': avg_similarity,
        'diversity_score': 1.0 - avg_similarity
    }
```

### 헤드 중요도와 가지치기

헤드마다 기여가 같지 않다. 헤드를 하나 없앴을 때 로스가 얼마나 바뀌는지로 중요도를 정의하고, 아래 코드처럼 그 변화를 그래디언트로 근사해 추정할 수 있다(미셸 외, 2019).

```python
def compute_head_importance(
    model: nn.Module,
    data_loader,
    num_layers: int,
    num_heads: int,
    device: torch.device
) -> np.ndarray:
    """
    그래디언트 기반 점수로 헤드의 중요도를 계산한다.

    미셸 외(2019)의 방법을 쓴다:
    중요도 = E[|grad(L) * 어텐션|]

    Returns:
        importance_matrix: [레이어 수, 헤드 수]
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

## 2. 레이어에 따른 어텐션 패턴

### 이론 배경

정보가 트랜스포머의 레이어를 지나면서 어텐션 패턴이 체계적으로 바뀐다.

- **앞쪽 레이어**(1~3): 국소적인 구문 관계를 담는다. 이웃 단어 어텐션, 위치 패턴
- **가운데 레이어**(4~8): 구문 구조를 담는다. 의존 관계, 상호참조
- **뒤쪽 레이어**(9~12): 추상적인 의미 패턴을 쌓는다. 과제에 특화된 어텐션, 장거리 의존

이 흐름은 CNN에서 보이는 계층적 피처 학습과 나란하다.

### 레이어 비교 구현

```python
def compare_layers(
    attention_weights_by_layer: list,
    tokens: list,
    layers_to_compare: list = None
) -> plt.Figure:
    """
    트랜스포머 레이어에 걸쳐 어텐션 패턴을 비교한다.

    Args:
        attention_weights_by_layer: 레이어마다의 [배치, 헤드, 시퀀스, 시퀀스] 목록
        tokens: 토큰 목록
        layers_to_compare: 보일 레이어
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
        axes[idx].set_title(f'레이어 {layer}')

    plt.tight_layout()
    return fig

def compute_locality_score(attention_weights: torch.Tensor, window: int = 3) -> float:
    """
    어텐션이 국소적인지 멀리 미치는지 잰다.

    국소성 점수 = 대각선에서 ±윈도 안에 든 어텐션 웨이트의 비율.
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

### 핵심 결과

여러 트랜스포머 구조에 걸친 연구에서 한결같은 패턴이 드러난다.

1. **깊어질수록 국소성이 줄어든다**: 앞쪽 레이어는 가까운 토큰을 보고(창 3 안에 ~80%), 뒤쪽 레이어는 멀리 본다(~40%)
2. **헤드를 가지치기해도 견디는 정도가 레이어마다 다르다**: 뒤쪽 레이어가 헤드를 없애도 더 강건하다
3. **잔차 연결이 중요하다**: 정보가 스킵 연결로 어텐션을 지나치므로, 어텐션만으로는 온 정보 흐름을 담을 수 없다

---

## 3. 크로스 어텐션 해석하기

크로스 어텐션은 서로 다른 두 시퀀스를 이어 인코더와 디코더 사이에 정보가 흐르게 한다. 번역, 요약, 질의응답 모델을 해석하려면 크로스 어텐션을 알아야 한다.

### 수학 기초

인코더의 표현 $K^e, V^e$와 디코더의 쿼리 $Q^d$가 주어지면

$$
\text{CrossAttn}(Q^d, K^e, V^e) = \text{softmax}\left(\frac{Q^d (K^e)^\top}{\sqrt{d_k}}\right) V^e
$$

어텐션 웨이트가 정렬(alignment) 행렬 $A \in \mathbb{R}^{T_d \times T_e}$를 이루며, $A_{ij}$는 디코더의 위치 $i$가 인코더의 위치 $j$를 얼마나 보는지 나타낸다.

### 구현

```python
def visualize_cross_attention(
    cross_attention: torch.Tensor,
    source_tokens: list,
    target_tokens: list,
    head: int = None
) -> plt.Figure:
    """
    인코더-디코더의 크로스 어텐션을 그린다.

    Args:
        cross_attention: [배치, 헤드, 타깃 길이, 소스 길이]
        source_tokens: 인코더 입력 토큰
        target_tokens: 디코더 출력 토큰
        head: 지정한 헤드(None이면 모든 헤드를 평균한다)
    """
    if head is not None:
        attn = cross_attention[0, head].detach().cpu().numpy()
        title = f'크로스 어텐션 (헤드 {head})'
    else:
        attn = cross_attention[0].mean(dim=0).detach().cpu().numpy()
        title = '크로스 어텐션 (헤드 평균)'

    fig, ax = plt.subplots(figsize=(10, 8))

    sns.heatmap(
        attn,
        xticklabels=source_tokens,
        yticklabels=target_tokens,
        ax=ax,
        cmap='Blues',
        vmin=0, vmax=1
    )
    ax.set_xlabel('소스 (인코더)')
    ax.set_ylabel('타깃 (디코더)')
    ax.set_title(title)

    plt.tight_layout()
    return fig

def analyze_cross_attention_alignment(
    cross_attention: torch.Tensor,
    source_tokens: list,
    target_tokens: list
) -> dict:
    """
    크로스 어텐션에서 정렬의 품질을 분석한다.

    정렬이 얼마나 뚜렷하고 소스를 얼마나 덮는지 지표로 내놓는다.
    """
    attn = cross_attention[0].mean(dim=0).detach().cpu()

    # 정렬 엔트로피(작을수록 정렬이 뚜렷하다)
    eps = 1e-10
    entropy = -(attn * torch.log(attn + eps)).sum(dim=-1).mean().item()

    # 커버리지: 임계값을 넘는 어텐션을 받은 소스 토큰의 비율
    max_attn_per_source = attn.max(dim=0)[0]
    coverage = (max_attn_per_source > 0.1).float().mean().item()

    # 단조성: 정렬이 얼마나 한 방향으로 나아가는가
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

## 4. 시각화 도구: BertViz와 직접 만든 것

### BertViz 살펴보기

BertViz(비그, 2019)는 시각화 보기 세 가지를 준다.

| 보기 | 보이는 것 | 쓰일 자리 |
|------|-------|----------|
| **어텐션 헤드 뷰** | 헤드 하나의 어텐션 패턴 | 특정 헤드의 동작 살피기 |
| **모델 뷰** | 모든 레이어의 모든 헤드 | 어텐션 분포를 두루 보기 |
| **뉴런 뷰** | 쿼리-키 분해 | 무엇이 어텐션을 이끄는지 알기 |

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
    인터랙티브 BertViz 시각화를 만든다.
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

    # 헤드 뷰 - 헤드 하나를 자세히 본다
    head_view(attention, tokens)

    # 모델 뷰 - 모든 헤드, 모든 레이어
    model_view(attention, tokens)
```

### 직접 시각화 만들기

```python
def create_attention_heatmap_grid(
    attention_by_layer: list,
    tokens: list,
    num_layers: int = 12,
    num_heads: int = 12,
    figsize: tuple = (24, 20)
) -> plt.Figure:
    """
    레이어 × 헤드의 전체 어텐션 격자를 만든다.
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

    fig.suptitle('어텐션 패턴: 레이어(행) × 헤드(열)', fontsize=14)
    plt.tight_layout()
    return fig

def create_attention_flow_sankey(
    attention_weights: torch.Tensor,
    tokens: list,
    source_idx: int,
    top_k: int = 5
):
    """
    지정한 토큰에서 어텐션이 어디로 흐르는지 보이는
    생키 다이어그램용 흐름 데이터를 만든다.
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

## 5. 어텐션 대 기여도 분석: 중요한 구분

### 핵심 문제

모델 해석에서 중요한 구분은 **어텐션 웨이트**와 **기여도 점수**의 차이다.

- **어텐션**은 모델이 어디를 "보는지" 보인다. 곧 입력 토큰에 걸친 웨이트의 분포다
- **기여도 분석**은 무엇이 실제로 출력을 좌우하는지, 곧 입력마다의 인과적 기여를 드러낸다

이 둘은 **같은 것이 아니다**. 제인과 월리스(2019)는 어텐션 웨이트가 그래디언트 기반 기여도와 상관이 낮을 때가 많으며, 어텐션 분포를 달리해도 똑같은 예측이 나올 수 있음을 밝혔다.

### 어텐션 ≠ 설명인 까닭

어텐션을 설명 방법으로 삼기 어렵게 하는 요인이 여럿 있다.

**값 벡터의 크기**: 어텐션 웨이트는 값 벡터를 섞는 비율을 정할 뿐이고, 값 벡터 자체의 크기와 방향은 토큰마다 다르다. 어떤 단어에 어텐션이 크다고 그 단어의 정보가 출력을 지배하는 것은 아니다.

$$
\text{output}_i = \sum_j \alpha_{ij} V_j W^V
$$

기여는 $\alpha_{ij}$와 $V_j$의 내용에 함께 달렸다.

**잔차 연결**: 정보가 스킵 연결로 어텐션을 지나친다. 출력 표현에는 어텐션이 모은 정보와 원래 입력이 함께 들어 있으므로 어텐션 웨이트만으로는 전체 그림을 볼 수 없다.

$$
\mathbf{h}_i^{(l+1)} = \mathbf{h}_i^{(l)} + \text{Attn}(\mathbf{h}^{(l)})
$$

**레이어 여럿의 누적**: 여러 레이어로 된 트랜스포머에서는 한 단어의 정보가 여러 어텐션 경로를 거쳐 출력에 이를 수 있다. 레이어 하나의 어텐션으로는 이런 간접 영향을 담을 수 없다.

### 어텐션과 기여도 분석 비교

```python
def compare_attention_and_attribution(
    model,
    input_ids: torch.Tensor,
    target_class: int,
    layer: int = -1,
    device: torch.device = None
) -> dict:
    """
    어텐션 웨이트와 그래디언트 기반 기여도를 비교한다.
    """
    model.eval()
    input_ids = input_ids.to(device)

    # 어텐션 웨이트를 얻는다
    with torch.no_grad():
        outputs = model(input_ids, output_attentions=True)
        attention = outputs.attentions[layer][0].mean(dim=0)

    # 그래디언트 기반 기여도를 얻는다
    embeddings = model.get_input_embeddings()(input_ids)
    embeddings.requires_grad_(True)

    outputs = model(inputs_embeds=embeddings)
    logits = outputs.logits
    logits[0, target_class].backward()

    gradient_attr = embeddings.grad.abs().sum(dim=-1)[0]

    # [CLS] 토큰에서 나가는 어텐션(흔히 이것을 쓴다)
    attn_scores = attention[0].detach().cpu().numpy()
    grad_scores = gradient_attr.detach().cpu().numpy()

    # 합이 1이 되게 정규화한다
    attn_scores = attn_scores / attn_scores.sum()
    grad_scores = grad_scores / grad_scores.sum()

    # 상관
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
| **어텐션 웨이트** | 모델의 행동 둘러보기, 구조 알기, 가설 세우기 | 믿을 만한 기여도가 아니고 값 벡터에 따라 달라진다 |
| **Attention Rollout** | 레이어를 가로지르는 정보 흐름 좇기 | 어텐션 ≈ 정보 흐름이라고 가정한다 |
| **Attention Flow** | 여러 레이어에 걸쳐 더 믿을 만한 기여도 분석 | 계산 비용이 크다 |
| **그래디언트 기여도 분석** | 믿을 만한 중요도 재기 | 어텐션만의 통찰을 놓친다 |
| **함께 쓰기** | 가장 폭넓은 이해 | 해석하기가 더 까다롭다 |

---

## 6. 종합 예제

```python
def comprehensive_attention_analysis(
    model,
    tokenizer,
    text: str,
    device: torch.device
) -> dict:
    """
    여러 기법을 결합한 종합 어텐션 분석.
    """
    # 토큰화한다
    inputs = tokenizer(text, return_tensors='pt', padding=True)
    input_ids = inputs['input_ids'].to(device)
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

    # 어텐션을 함께 출력하는 순전파
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

    # 레이어마다 분석한다
    for layer_idx, attn in enumerate(attentions):
        diversity = compute_head_diversity(attn)
        locality = compute_locality_score(attn)

        results['analyses'][f'layer_{layer_idx}'] = {
            'diversity_score': diversity['diversity_score'],
            'head_entropies': diversity['entropies'],
            'locality_score': locality
        }

    # 레이어에 따른 국소성의 변화
    localities = [
        results['analyses'][f'layer_{i}']['locality_score']
        for i in range(len(attentions))
    ]
    results['locality_progression'] = localities

    return results
```

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
이 절에서 다룬 설명 방법을, XOR 입력을 분류하는 ReLU 활성화의 두 레이어 뉴럴 네트워크에 적용하라. 입력 $x = [1, 1]$에 대한 설명을 계산하여라.

</div>

??? success "풀이"
    웨이트가 $W_1, b_1, W_2, b_2$인 학습된 XOR 네트워크에서 출력은 $f(x) = W_2 \cdot \text{ReLU}(W_1 x + b_1) + b_2$이다. 설명 방법은 입력 피처마다 기여도를 내놓는다. $x = [1, 1]$(클래스 0)이면 두 피처 모두 음수 쪽 분류에 기여한다. 기여도 값은 방법마다 다르다. 그래디언트 기반 방법은 $\partial f / \partial x_i$을 계산하고, 섭동 기반 방법은 피처를 가렸을 때 출력이 얼마나 바뀌는지 잰다. XOR 문제는 결정 경계가 선형이 아니므로 선형 설명 방법이 틀릴 수 있음을 보여 준다. $\square$

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff hard" title="어려움"></span>
이 절의 설명 방법이 완전성 공리를 만족하는지, 곧 어떤 베이스라인 $x_0$에 대해 모든 피처 기여도의 합이 $f(x) - f(x_0)$과 같은지 증명하거나 반례를 들어라.

</div>

??? success "풀이"
    완전성 공리(섀플리 값 이론에서는 효율성이라고도 한다)는 기여도의 합이 입력에서의 모델 출력과 베이스라인에서의 출력의 차이와 같다는 것이다. 이 방법이 완전성을 만족하는지는 그 구성에 달렸다. 그래디언트 방법은 완전성을 만족하지 못한다(그래디언트는 한 점의 국소적인 양이고 경로를 따라 적분한 것이 아니다). Integrated Gradients는 구성 자체로 완전성을 만족한다(경로를 따라 미적분의 기본정리를 쓴다). SHAP 값은 섀플리 공리로 효율성을 만족한다. 어텐션 웨이트는 값 벡터를 섞는 비율이라 행마다 합이 1일 뿐 출력 차이와 이어지지 않으므로 완전성을 만족하지 못한다. 완전성을 어기는 방법은 기여도를 너무 크거나 작게 매길 수 있어, 기여도 전체를 전역 설명으로 믿기 어렵게 만든다. $\square$

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
이 방법이 내놓는 설명이 얼마나 믿을 만한지 평가하는 실험을 설계하라. 방법이 중요하다고 짚은 피처가 실제로 모델에 중요한지를 삽입 곡선과 삭제 곡선으로 재어라.

</div>

??? success "풀이"
    절차는 이렇다. (1) 시험 이미지마다 피처 기여도를 계산한다. (2) 삭제: 기여도가 큰 차례로 피처를 하나씩 가리며 모델의 확신도가 떨어지는 모습을 기록한다. 믿을 만한 설명이면 확신도가 빠르게 떨어진다. (3) 삽입: 빈 베이스라인에서 시작해 기여도가 큰 차례로 피처를 하나씩 드러내며 확신도가 오르는 모습을 기록한다. 믿을 만한 설명이면 확신도가 빠르게 오른다. (4) 두 곡선 아래 넓이를 계산한다. (5) 무작위 순서(베이스라인)와 다른 방법과 비교한다. 믿을 만한 방법이면 삭제 곡선의 넓이가 작고 삽입 곡선의 넓이가 커야 한다. 통계적으로 믿을 만하려면 시험 표본 1000개 넘게 되풀이한다. $\square$

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
이 설명 방법을 신용 부도를 예측하는 금융 모델에 어떻게 적용할 수 있는지 논하라. 설명이 채워야 할 규제 요건은 무엇인가?

</div>

??? success "풀이"
    신용 모델에는 규제(ECOA, GDPR 22조)가 불리한 결정마다 그 사람에게 맞춘 설명을 요구한다. 방법은 다음을 내놓아야 한다. (1) 거절에 가장 크게 기여한 요인(불리한 처분 사유). (2) 일관된 설명(비슷한 신청자는 비슷한 설명을 받는다). (3) 실행 가능한 설명(신청자가 무엇을 바꾸어야 하는지 안다). 이 절의 설명 방법으로 피처 중요도를 짚을 수 있으나, 강건성(입력이 조금 바뀌었다고 설명이 확 달라지면 안 된다)과 충실성(중요한 피처를 없애면 예측이 바뀌어야 한다)을 평가해야 한다. 보호 속성은 대리 차별이 드러나지 않도록 조심히 다루어야 한다. $\square$

## 정리하며

어텐션 패턴 분석은 트랜스포머의 동작을 폭넓게 들여다보게 해 주지만 조심히 읽어야 한다.

1. **헤드 사이의 다양성**이 역할을 드러낸다. 너무 비슷한 헤드는 가지치기할 수 있다
2. **레이어에 따른 변화**가 국소에서 전역으로 가는 계층적 피처 학습을 보인다
3. **크로스 어텐션의 맞춤**은 인코더-디코더 모델에서 알려 주는 바가 많다
4. **어텐션 ≠ 기여도 분석**: 어텐션으로 얻은 통찰은 늘 그래디언트 기반 방법으로 따져 보라
5. BertViz 같은 **인터랙티브 도구**는 탐색에 값지지만 정량 지표를 곁들여야 한다

**참고 문헌**

1. Vig, J. (2019). "A Multiscale Visualization of Attention in the Transformer Model." *ACL Demo*.

2. Clark, K., et al. (2019). "What Does BERT Look At? An Analysis of BERT's Attention." *BlackboxNLP*.

3. Jain, S., & Wallace, B. C. (2019). "Attention is not Explanation." *NAACL*.

4. Wiegreffe, S., & Pinter, Y. (2019). "Attention is not not Explanation." *EMNLP*.

5. Michel, P., Levy, O., & Neubig, G. (2019). "Are Sixteen Heads Really Better than One?" *NeurIPS*.

6. Voita, E., et al. (2019). "Analyzing Multi-Head Self-Attention: Specialized Heads Do the Heavy Lifting." *ACL*.

7. Abnar, S., & Zuidema, W. (2020). "Quantifying Attention Flow in Transformers." *ACL*.
