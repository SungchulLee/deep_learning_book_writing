# Attention Fundamentals

## Introduction

Attention mechanisms represent one of the most significant breakthroughs in deep learning, fundamentally changing how neural networks process sequential and structured data. Originally introduced to address limitations in sequence-to-sequence models, attention has become the cornerstone of modern architectures including Transformers, which power state-of-the-art models in natural language processing, computer vision, and beyond.

The core intuition behind attention is elegantly simple: when processing information, not all inputs are equally relevant to the current task. Rather than treating all information uniformly, attention mechanisms allow models to dynamically focus on the most pertinent parts of the input, much like how humans selectively attend to specific aspects of their environment.

## The Core Intuition: Information Retrieval

Attention can be understood through a **social media search analogy**:

| Social Media | Attention | Purpose |
|--------------|-----------|---------|
| Search query | Query ($\mathbf{Q}$) | Express information need |
| Hashtags | Key ($\mathbf{K}$) | Enable discovery |
| Post content | Value ($\mathbf{V}$) | Provide actual information |

When you search social media for "sunset photography tips":
1. **Query**: Your search intent—what you're looking for
2. **Keys**: Hashtags on posts (#photography, #sunset, #tips) optimized for findability
3. **Values**: The actual post content you came for

A post about sunset photography might have hashtag #lighting, but its actual content discusses color temperature, exposure settings, atmospheric conditions, and timing. **The hashtag gets you there; the content is what you came for.**

## Historical Context

### The Sequence-to-Sequence Problem

Before attention, sequence-to-sequence (seq2seq) models relied on encoder-decoder architectures where the encoder compressed an entire input sequence into a fixed-length context vector. The decoder then generated the output sequence conditioned solely on this vector.

$$
\mathbf{c} = f_{\text{encoder}}(\mathbf{x}_1, \mathbf{x}_2, \ldots, \mathbf{x}_T)
$$

This approach suffered from the **information bottleneck problem**: compressing arbitrarily long sequences into fixed-size representations inevitably loses information. Performance degraded significantly on longer sequences, as the context vector simply could not capture all necessary details.

### The Attention Solution

Bahdanau et al. (2014) introduced attention to address this limitation. Instead of relying on a single context vector, the decoder could now "attend" to different parts of the encoder output at each decoding step:

$$
\mathbf{c}_t = \sum_{i=1}^{T} \alpha_{ti} \mathbf{h}_i
$$

where $\alpha_{ti}$ represents the attention weight indicating how much the decoder should focus on encoder hidden state $\mathbf{h}_i$ when generating output at time $t$.

## Query-Key-Value Formulation

Modern attention mechanisms operate on three fundamental components with distinct, optimized roles:

Given an input sequence $\mathbf{X} \in \mathbb{R}^{n \times d}$, attention computes three projections:

$$\mathbf{Q} = \mathbf{X}\mathbf{W}_Q \quad \text{(Queries: "What am I looking for?")}$$

$$\mathbf{K} = \mathbf{X}\mathbf{W}_K \quad \text{(Keys: "What do I offer?")}$$

$$\mathbf{V} = \mathbf{X}\mathbf{W}_V \quad \text{(Values: "What will I contribute?")}$$

where $\mathbf{W}_Q, \mathbf{W}_K \in \mathbb{R}^{d \times d_k}$ and $\mathbf{W}_V \in \mathbb{R}^{d \times d_v}$.

### Why Three Separate Projections?

The separation serves distinct purposes that enable richer representations:

**Keys are optimized for findability**—concise, categorical, searchable descriptors that identify what information a token offers. Like hashtags, they're designed to match well with queries.

**Values are optimized for information transfer**—rich semantic content that gets transferred once a match is established. They carry the actual payload of information.

**Queries encode the search intent**—what the current position needs to know. Different from keys, queries represent the "demand side" of the information retrieval.

This formulation provides a general framework:

$$
\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{Aggregate}(\text{Compatibility}(\mathbf{Q}, \mathbf{K}), \mathbf{V})
$$

### The Learned Aspect

The network learns all three projections simultaneously:

- $\mathbf{W}_Q$: How to formulate effective searches
- $\mathbf{W}_K$: How to tag content for discoverability  
- $\mathbf{W}_V$: How to package content for useful transmission

Training optimizes all three—learning not just what to search for, but how tokens should advertise themselves and what information they should deliver.

## The Attention Mechanism

### Complete Formulation

The complete attention computation:

$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}\right)\mathbf{V}$$

### Step-by-Step Breakdown

**Step 1: Compute Relevance Scores**
$$\mathbf{S} = \mathbf{Q}\mathbf{K}^T \in \mathbb{R}^{n \times n}$$

Entry $S_{ij}$ measures how much position $i$ should attend to position $j$—comparing the query at position $i$ against the key at position $j$.

**Step 2: Scale by $\sqrt{d_k}$**
$$\mathbf{S}_{\text{scaled}} = \frac{\mathbf{S}}{\sqrt{d_k}}$$

This prevents dot products from growing too large with dimension, which would push softmax into saturation (explored in detail in the scaling section).

**Step 3: Apply Softmax**
$$\mathbf{A} = \text{softmax}(\mathbf{S}_{\text{scaled}})$$

Row $i$ becomes a probability distribution over all positions, indicating where position $i$ should gather information.

**Step 4: Weighted Sum of Values**
$$\mathbf{Z} = \mathbf{A}\mathbf{V}$$

Each output position receives a weighted combination of value vectors:
$$\mathbf{z}_i = \sum_{j=1}^{n} A_{ij} \mathbf{v}_j$$

### The Attention Pipeline

```
Query ──┐
        ├──► Score Function ──► Softmax ──► Attention Weights
Keys ───┘                                          │
                                                   ▼
Values ─────────────────────────────────► Weighted Sum ──► Output
```

## A Concrete Example: Pronoun Resolution

Consider: "The cat sat on the mat because it was soft."

When processing "it":

| Component | Role | Example |
|-----------|------|---------|
| **Query** | Search intent | "I'm searching for #noun #antecedent #nearby" |
| **Keys** | Token descriptors | "cat": #noun #animate #subject; "mat": #noun #inanimate #object |
| **Attention weights** | Relevance scores | Higher weight on "mat" (because "soft" implies inanimate) |
| **Value** | Retrieved content | Semantic content of "mat" (thing-you-sit-on, surface, location) |

The key (hashtag) identifies "mat" as a potential antecedent. The value provides the semantic information needed to resolve the pronoun.

## Score Functions

Different score functions have been proposed, each with distinct properties:

**Dot-Product (Multiplicative) Attention**:
$$
\text{score}(\mathbf{q}, \mathbf{k}) = \mathbf{q}^T \mathbf{k}
$$

Computationally efficient but requires $d_q = d_k$.

**Additive (Bahdanau) Attention**:
$$
\text{score}(\mathbf{q}, \mathbf{k}) = \mathbf{v}^T \tanh(\mathbf{W}_q \mathbf{q} + \mathbf{W}_k \mathbf{k})
$$

More flexible but computationally expensive. Introduces learnable parameters $\mathbf{W}_q$, $\mathbf{W}_k$, and $\mathbf{v}$.

**General Attention**:
$$
\text{score}(\mathbf{q}, \mathbf{k}) = \mathbf{q}^T \mathbf{W} \mathbf{k}
$$

Introduces a learnable weight matrix to mediate the interaction.

**Scaled Dot-Product Attention**:
$$
\text{score}(\mathbf{q}, \mathbf{k}) = \frac{\mathbf{q}^T \mathbf{k}}{\sqrt{d_k}}
$$

The standard in Transformer architectures, combining efficiency with numerical stability.

## Attention as Soft Dictionary Lookup

Attention can be viewed as a **differentiable dictionary**:

| Hard Dictionary | Soft Attention |
|-----------------|----------------|
| Exact key match | Similarity-weighted matching |
| Returns one value | Returns weighted combination |
| Discrete lookup | Continuous, differentiable |
| `dict[query] → value` | `query → weighted_sum(values, similarity(query, keys))` |

**Hard selection**: $\mathbf{y} = \mathbf{v}_{i^*}$ where $i^* = \arg\max_i \text{score}(\mathbf{q}, \mathbf{k}_i)$

**Soft selection**: $\mathbf{y} = \sum_i \alpha_i \mathbf{v}_i$ where $\alpha_i = \text{softmax}(\text{scores})_i$

This soft matching enables gradient flow and end-to-end learning—a crucial property that makes attention trainable.

## PyTorch Implementation

### Basic Attention Module

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class BasicAttention(nn.Module):
    """
    Basic Attention Mechanism (Additive/Bahdanau Attention)
    
    Computes attention weights using a learned alignment model.
    Score(query, key) = v^T * tanh(W_q * query + W_k * key)
    
    This formulation allows queries and keys to have different dimensions
    and learns a non-linear alignment function.
    """
    
    def __init__(self, query_dim: int, key_dim: int, hidden_dim: int):
        """
        Args:
            query_dim: Dimension of query vectors
            key_dim: Dimension of key vectors
            hidden_dim: Hidden dimension for alignment model
        """
        super().__init__()
        self.query_projection = nn.Linear(query_dim, hidden_dim)
        self.key_projection = nn.Linear(key_dim, hidden_dim)
        self.score_projection = nn.Linear(hidden_dim, 1)
        
    def forward(
        self, 
        query: torch.Tensor, 
        keys: torch.Tensor, 
        values: torch.Tensor, 
        mask: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: (batch_size, query_dim) - Single query vector per batch
            keys: (batch_size, seq_len, key_dim) - Key vectors to attend over
            values: (batch_size, seq_len, value_dim) - Value vectors
            mask: (batch_size, seq_len) - Optional mask (1=attend, 0=ignore)
            
        Returns:
            context: (batch_size, value_dim) - Weighted sum of values
            attention_weights: (batch_size, seq_len) - Attention distribution
        """
        batch_size, seq_len, _ = keys.shape
        
        # Project query: (batch_size, hidden_dim) -> (batch_size, 1, hidden_dim)
        query_proj = self.query_projection(query).unsqueeze(1)
        
        # Project keys: (batch_size, seq_len, hidden_dim)
        keys_proj = self.key_projection(keys)
        
        # Compute alignment scores via additive attention
        # Broadcasting: (batch_size, 1, hidden_dim) + (batch_size, seq_len, hidden_dim)
        alignment = torch.tanh(query_proj + keys_proj)
        
        # Score projection: (batch_size, seq_len, 1) -> (batch_size, seq_len)
        scores = self.score_projection(alignment).squeeze(-1)
        
        # Apply mask if provided (set masked positions to -inf)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Normalize to get attention weights
        attention_weights = F.softmax(scores, dim=-1)
        
        # Compute context as weighted sum of values
        # (batch_size, 1, seq_len) @ (batch_size, seq_len, value_dim)
        context = torch.bmm(attention_weights.unsqueeze(1), values).squeeze(1)
        
        return context, attention_weights
```

### Demonstration

```python
def demonstrate_basic_attention():
    """Demonstrate the basic attention mechanism."""
    
    # Configuration
    batch_size = 2
    seq_len = 5
    query_dim = 8
    key_dim = 8
    value_dim = 16
    hidden_dim = 32
    
    # Create sample data
    query = torch.randn(batch_size, query_dim)
    keys = torch.randn(batch_size, seq_len, key_dim)
    values = torch.randn(batch_size, seq_len, value_dim)
    
    # Create attention module
    attention = BasicAttention(query_dim, key_dim, hidden_dim)
    
    # Compute attention
    context, weights = attention(query, keys, values)
    
    print("Basic Attention Demonstration")
    print("-" * 40)
    print(f"Query shape:      {query.shape}")
    print(f"Keys shape:       {keys.shape}")
    print(f"Values shape:     {values.shape}")
    print(f"Context shape:    {context.shape}")
    print(f"Weights shape:    {weights.shape}")
    print(f"\nAttention weights (sample 0): {weights[0].detach().numpy()}")
    print(f"Weights sum:      {weights[0].sum().item():.6f}")


if __name__ == "__main__":
    demonstrate_basic_attention()
```

**Output:**
```
Basic Attention Demonstration
----------------------------------------
Query shape:      torch.Size([2, 8])
Keys shape:       torch.Size([2, 5, 8])
Values shape:     torch.Size([2, 5, 16])
Context shape:    torch.Size([2, 16])
Weights shape:    torch.Size([2, 5])

Attention weights (sample 0): [0.187 0.203 0.196 0.211 0.203]
Weights sum:      1.000000
```

## Attention Variants by Information Flow

| Variant | Q Source | K, V Source | Use Case |
|---------|----------|-------------|----------|
| **Self-Attention** | Same sequence | Same sequence | Internal context modeling |
| **Cross-Attention** | One sequence | Different sequence | Encoder-decoder bridging |
| **Masked Attention** | Same sequence | Past positions only | Autoregressive generation |

### Self-Attention

In self-attention, queries, keys, and values all derive from the same input sequence:

$$
\mathbf{Q} = \mathbf{X}\mathbf{W}^Q, \quad \mathbf{K} = \mathbf{X}\mathbf{W}^K, \quad \mathbf{V} = \mathbf{X}\mathbf{W}^V
$$

**Use cases**: Transformer encoders (BERT), understanding relationships within a single sequence.

### Cross-Attention

In cross-attention, queries come from one sequence while keys and values come from another:

$$
\mathbf{Q} = \mathbf{X}_{\text{decoder}}\mathbf{W}^Q, \quad \mathbf{K} = \mathbf{X}_{\text{encoder}}\mathbf{W}^K, \quad \mathbf{V} = \mathbf{X}_{\text{encoder}}\mathbf{W}^V
$$

**Use cases**: Machine translation (decoder attending to encoder), image captioning, multimodal fusion.

### Causal (Masked) Attention

Causal attention restricts each position to only attend to previous positions (including itself):

$$
\alpha_{ij} = 0 \quad \text{for all } j > i
$$

**Use cases**: Autoregressive generation (GPT), language modeling.

## Properties of Attention

### Permutation Equivariance

If input positions are permuted, attention outputs are permuted identically:

$$\text{Attention}(\mathbf{P}\mathbf{X}) = \mathbf{P} \cdot \text{Attention}(\mathbf{X})$$

Attention treats positions symmetrically—positional information must be added separately.

### No Built-in Position Awareness

Attention scores depend only on content, not position. The same word at different positions produces identical queries and keys (before positional encoding). **Position must be explicitly encoded.**

### Dynamic Weighting

Unlike fixed weights in fully connected layers, attention weights are computed dynamically based on the input:

1. **Input-dependent processing**: The same network processes different inputs differently
2. **Variable-length handling**: Naturally handles sequences of different lengths
3. **Interpretability**: Attention weights reveal which inputs influence outputs

### Computational Complexity

For a sequence of length $n$ with dimension $d$:

| Operation | Time Complexity | Memory |
|-----------|-----------------|--------|
| Score computation | $O(n^2 d)$ | $O(n^2)$ |
| Softmax | $O(n^2)$ | $O(n^2)$ |
| Value aggregation | $O(n^2 d)$ | $O(nd)$ |
| **Total** | $O(n^2 d)$ | $O(n^2 + nd)$ |

The quadratic complexity in sequence length is a key limitation, motivating efficient attention variants like sparse attention, linear attention, and FlashAttention.

## Masking in Attention

### Padding Masks

When batching sequences of different lengths, shorter sequences are padded. Padding masks prevent attention to padding tokens:

```python
def create_padding_mask(seq: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
    """
    Create mask to ignore padding tokens.
    
    Args:
        seq: (batch_size, seq_len) - Token indices
        pad_idx: Index used for padding
        
    Returns:
        mask: (batch_size, 1, 1, seq_len) - Broadcastable mask
    """
    return (seq != pad_idx).unsqueeze(1).unsqueeze(2)
```

### Causal Masks

For autoregressive models, causal masks prevent attending to future positions:

```python
def create_causal_mask(size: int) -> torch.Tensor:
    """
    Create lower-triangular mask for causal attention.
    
    Args:
        size: Sequence length
        
    Returns:
        mask: (1, 1, size, size) - Broadcastable causal mask
    """
    mask = torch.tril(torch.ones(size, size))
    return mask.unsqueeze(0).unsqueeze(0)
```

### Combined Masks

In practice, padding and causal masks are often combined:

```python
def combine_masks(
    padding_mask: torch.Tensor, 
    causal_mask: torch.Tensor
) -> torch.Tensor:
    """
    Combine padding and causal masks.
    """
    return padding_mask & causal_mask.bool()
```

## Applications

### Natural Language Processing

Attention mechanisms power modern NLP:
- **Machine Translation**: Aligning source and target language representations
- **Text Summarization**: Identifying salient content for compression
- **Question Answering**: Matching questions with relevant context passages
- **Sentiment Analysis**: Focusing on sentiment-bearing words

### Computer Vision

Visual attention has transformed image understanding:
- **Image Classification**: Focusing on discriminative regions (Vision Transformer)
- **Object Detection**: Attending to object locations
- **Image Captioning**: Grounding generated words in image regions
- **Visual Question Answering**: Relating questions to relevant image areas

### Quantitative Finance

Attention mechanisms find natural applications in finance:
- **Time Series Forecasting**: Identifying relevant historical patterns
- **Portfolio Optimization**: Weighting asset contributions to portfolio risk
- **Event Detection**: Attending to anomalous market conditions
- **Document Analysis**: Extracting relevant information from financial reports

## Summary

Attention provides a mechanism for:

1. **Dynamic routing**: Each position decides where to gather information
2. **Content-based addressing**: Relevance determined by semantic similarity
3. **Parallel computation**: All positions computed simultaneously
4. **Differentiable lookup**: Enables end-to-end learning

The Query-Key-Value formulation separates the concerns of searching (Q-K matching) from information transfer (V retrieval), giving the model flexibility to learn optimal representations for each role.

**Key takeaways:**

| Concept | Description |
|---------|-------------|
| **Core idea** | Dynamically compute weighted combinations of values based on query-key compatibility |
| **Components** | Queries (what to look for), Keys (what's available), Values (actual content) |
| **Benefits** | Input-dependent processing, variable-length handling, interpretability |
| **Variants** | Self-attention, cross-attention, causal attention for different use cases |
| **Challenge** | Quadratic complexity limits applicability to very long sequences |

The following sections explore specific attention variants in detail, starting with the scaled dot-product attention that forms the foundation of Transformer architectures.

## References

1. Bahdanau, D., Cho, K., & Bengio, Y. (2014). Neural Machine Translation by Jointly Learning to Align and Translate. *arXiv preprint arXiv:1409.0473*.

2. Luong, M.-T., Pham, H., & Manning, C. D. (2015). Effective Approaches to Attention-based Neural Machine Translation. *EMNLP*.

3. Vaswani, A., et al. (2017). Attention Is All You Need. *NeurIPS*.

4. Lin, Z., et al. (2017). A Structured Self-attentive Sentence Embedding. *ICLR*.
