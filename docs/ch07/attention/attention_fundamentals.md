# Attention Fundamentals

Attention mechanisms allow neural networks to dynamically focus on relevant parts of their input, rather than treating all information uniformly. Originally introduced to address limitations in sequence-to-sequence models, attention has become the foundation of modern architectures including Transformers.

## Core Intuition: Attention as Soft Dictionary Lookup

One of the most illuminating ways to understand attention is through the lens of **soft dictionary lookup**—a differentiable generalization of key-value retrieval. Traditional dictionaries perform exact matching: given a query, they return the value associated with the matching key. Attention extends this concept by performing **soft, weighted retrieval** across all entries based on similarity.

### Hard vs. Soft Retrieval

A standard dictionary maps keys to values with exact matching:

```python
dictionary = {
    "cat": [0.2, 0.8, 0.1],   # key -> value
    "dog": [0.3, 0.7, 0.2],
    "mat": [0.1, 0.1, 0.9],
}

result = dictionary["cat"]  # Returns [0.2, 0.8, 0.1]
```

Attention generalizes this to continuous, differentiable retrieval:

```python
def soft_lookup(query, keys, values):
    similarities = [dot(query, k) for k in keys]
    weights = softmax(similarities)
    result = sum(w * v for w, v in zip(weights, values))
    return result
```

| Aspect | Hard Dictionary | Soft Attention |
|--------|-----------------|----------------|
| Matching | Exact key match | Similarity-weighted |
| Selection | Binary (0 or 1) | Continuous $[0, 1]$ |
| Output | Single value | Weighted combination |
| Differentiable | No | Yes |
| Learnable | No | Yes (Q, K, V projections) |

This simple shift—from hard to soft matching—unlocks the power of learnable, differentiable memory access. Given a query, attention computes similarity scores against all keys, normalises them via softmax, and returns a weighted sum of corresponding values.

### The Social Media Analogy

| Social media | Attention | Purpose |
|--------------|-----------|---------|
| Search query | Query ($\mathbf{Q}$) | Express information need |
| Hashtags | Key ($\mathbf{K}$) | Enable discovery |
| Post content | Value ($\mathbf{V}$) | Provide actual information |

When searching for "sunset photography tips", hashtags (#photography, #sunset) get you to relevant posts, but the post content is what you came for. **Keys are optimised for findability; values carry the payload.**

### Why the Soft Dictionary View Matters

**Separation of concerns.** In a dictionary, the key used for lookup differs from the value returned (`{"isbn-123": "The Great Gatsby"}`). Similarly, keys are optimized for **findability** while values are optimized for **information content**, allowing the model to learn different representations for "how to be found" versus "what information to provide."

**Content-addressable memory.** Traditional computer memory uses position-based addressing (`memory[address]`). Attention enables content-based addressing (`memory[content_similar_to_query]`), retrieving based on meaning rather than arbitrary positions.

**Graceful degradation.** Hard lookup fails catastrophically with slight variations—if the key isn't exactly right, you get nothing. Soft lookup degrades gracefully: similar queries retrieve similar weighted combinations, and small perturbations lead to small changes in output.

**Compositional retrieval.** A query between two keys retrieves a blend of both values, allowing the model to synthesize new responses from stored primitives.

### Temperature: Controlling Soft vs. Hard

The softmax temperature controls how "hard" the lookup becomes:

$$\alpha_i = \frac{\exp(s_i / T)}{\sum_j \exp(s_j / T)}$$

| Temperature | Behavior | Analogy |
|-------------|----------|---------|
| $T \to 0$ | Hard attention (argmax) | Exact dictionary lookup |
| $T = 1$ | Standard softmax | Soft mixture |
| $T \to \infty$ | Uniform weights | Return average of all values |

The scaling factor $\sqrt{d_k}$ in Transformer attention acts as a temperature that adapts to dimensionality.

## Historical Motivation: The Seq2Seq Bottleneck

### The Problem

Before attention, encoder-decoder models compressed the entire input sequence into a single fixed-length context vector:

$$\mathbf{c} = f_{\text{encoder}}(x_1, x_2, \ldots, x_T)$$

This creates an **information bottleneck**: compressing $O(Td)$ bits into $O(d)$ bits guarantees information loss for long sequences. Performance degraded significantly as sequence length increased.

### The Attention Solution

Bahdanau et al. (2014) proposed allowing the decoder to "attend" to different encoder positions at each step:

$$\mathbf{c}_t = \sum_{i=1}^{T} \alpha_{ti} \mathbf{h}_i$$

where $\alpha_{ti}$ indicates how much the decoder should focus on encoder state $\mathbf{h}_i$ when generating output at time $t$. The attention weights satisfy $\sum_i \alpha_{ti} = 1$ and $\alpha_{ti} \geq 0$.

This maintains $O(Td)$ accessible information throughout decoding, circumventing the bottleneck.

### Alignment Interpretation

For translation, attention weights represent **soft alignment** between source and target positions:

```
Source:  The  cat  sat  on  the  mat
Weights: 0.1  0.7  0.1  0.0  0.0  0.1  → "chat" (French for cat)
         0.0  0.1  0.0  0.0  0.1  0.8  → "tapis" (French for mat)
```

The model learns alignment structure implicitly, without explicit supervision.

## The Query-Key-Value Formulation

Modern attention operates on three projections from input $\mathbf{X} \in \mathbb{R}^{n \times d}$:

$$\mathbf{Q} = \mathbf{X}\mathbf{W}_Q, \quad \mathbf{K} = \mathbf{X}\mathbf{W}_K, \quad \mathbf{V} = \mathbf{X}\mathbf{W}_V$$

The separation serves distinct purposes:

- **Queries** encode search intent—what the current position needs
- **Keys** are optimised for matching—concise descriptors for findability
- **Values** carry content—rich information for transfer once a match is found

This gives the model flexibility to learn different representations for searching versus information delivery—mirroring the structure of traditional databases where query language (SQL) specifies what to find, indexes enable efficient lookup, and stored records contain the actual data.

### Scaled Dot-Product Attention

The standard formulation (Vaswani et al., 2017):

$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}}\right)\mathbf{V}$$

**Step by step:**

1. **Compute scores**: $\mathbf{S} = \mathbf{Q}\mathbf{K}^\top$ — entry $S_{ij}$ measures compatibility between position $i$'s query and position $j$'s key
2. **Scale**: $\mathbf{S} / \sqrt{d_k}$ — prevents dot products from growing with dimension, avoiding softmax saturation
3. **Normalise**: $\mathbf{A} = \text{softmax}(\mathbf{S})$ — row $i$ becomes a distribution over positions
4. **Aggregate**: $\mathbf{Z} = \mathbf{A}\mathbf{V}$ — each output is a weighted sum of value vectors

## Score Functions

Different score functions trade off expressiveness and efficiency:

**Dot-Product** (requires $d_q = d_k$):
$$\text{score}(\mathbf{q}, \mathbf{k}) = \mathbf{q}^\top \mathbf{k}$$

**Scaled Dot-Product** (standard in Transformers):
$$\text{score}(\mathbf{q}, \mathbf{k}) = \frac{\mathbf{q}^\top \mathbf{k}}{\sqrt{d_k}}$$

**General** (learnable, allows dimension mismatch):
$$\text{score}(\mathbf{q}, \mathbf{k}) = \mathbf{q}^\top \mathbf{W} \mathbf{k}$$

**Additive / Bahdanau** (more expressive, slower):
$$\text{score}(\mathbf{q}, \mathbf{k}) = \mathbf{v}^\top \tanh(\mathbf{W}_q \mathbf{q} + \mathbf{W}_k \mathbf{k})$$

**Cosine Similarity** (scale-invariant):
$$\text{score}(\mathbf{q}, \mathbf{k}) = \frac{\mathbf{q}^\top \mathbf{k}}{\|\mathbf{q}\| \|\mathbf{k}\|}$$

| Score function | Parameters | Complexity | Use case |
|----------------|------------|------------|----------|
| Dot-product | 0 | $O(d)$ | Same-dimension Q, K |
| Scaled dot-product | 0 | $O(d)$ | Transformers (stable gradients) |
| General | $d_q \times d_k$ | $O(d_q d_k)$ | Different dimensions |
| Additive | $d_a(d_q + d_k) + d_a$ | $O(d_a)$ | RNN seq2seq |
| Cosine | 0 | $O(d)$ | Bounded similarity |

### Why Scaling Matters

For random unit vectors in $d$ dimensions, dot products have variance $\approx d$. Without scaling, large $d$ pushes softmax into saturation where gradients vanish. The $\sqrt{d_k}$ factor normalises variance to 1, maintaining healthy gradient flow. See [Scaled Dot-Product](scaled_dot_product.md) for a detailed analysis.

## Attention Variants

| Variant | Q source | K, V source | Use case |
|---------|----------|-------------|----------|
| **Self-attention** | Same sequence | Same sequence | Internal context (BERT encoder) |
| **Cross-attention** | Decoder | Encoder | Encoder-decoder bridging (translation) |
| **Causal attention** | Same sequence | Past positions only | Autoregressive generation (GPT) |

### Self-Attention

All projections derive from the same input:

$$\mathbf{Q} = \mathbf{X}\mathbf{W}^Q, \quad \mathbf{K} = \mathbf{X}\mathbf{W}^K, \quad \mathbf{V} = \mathbf{X}\mathbf{W}^V$$

Each position can attend to all positions, capturing global dependencies. See [Self-Attention](self_attention.md) for full treatment.

### Cross-Attention

Queries come from one sequence (decoder), keys and values from another (encoder):

$$\mathbf{Q} = \mathbf{X}_{\text{dec}}\mathbf{W}^Q, \quad \mathbf{K} = \mathbf{X}_{\text{enc}}\mathbf{W}^K, \quad \mathbf{V} = \mathbf{X}_{\text{enc}}\mathbf{W}^V$$

This bridges encoder and decoder in seq2seq models. Cross-attention patterns are discussed in [Attention Patterns](attention_patterns.md).

### Causal (Masked) Attention

Position $i$ can only attend to positions $j \leq i$:

$$\alpha_{ij} = 0 \quad \text{for } j > i$$

Essential for autoregressive generation where future tokens are unavailable.

## Masking

### Padding Mask

Prevents attention to padding tokens when batching variable-length sequences:

```python
def padding_mask(seq, pad_idx=0):
    """(batch, seq_len) -> (batch, 1, 1, seq_len)"""
    return (seq != pad_idx).unsqueeze(1).unsqueeze(2)
```

### Causal Mask

Lower-triangular mask for autoregressive models:

```python
def causal_mask(size):
    """(1, 1, size, size)"""
    return torch.tril(torch.ones(size, size)).unsqueeze(0).unsqueeze(0)
```

Masks are applied by setting scores to $-\infty$ before softmax.

## PyTorch Implementation

### Scaled Dot-Product Attention

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Args:
        Q: (batch, ..., seq_q, d_k)
        K: (batch, ..., seq_k, d_k)
        V: (batch, ..., seq_k, d_v)
        mask: broadcastable to (batch, ..., seq_q, seq_k)

    Returns:
        output: (batch, ..., seq_q, d_v)
        weights: (batch, ..., seq_q, seq_k)
    """
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    weights = F.softmax(scores, dim=-1)
    output = torch.matmul(weights, V)
    return output, weights
```

### Attention Layer with Projections

```python
class Attention(nn.Module):
    def __init__(self, d_model, d_k=None, d_v=None):
        super().__init__()
        d_k = d_k or d_model
        d_v = d_v or d_model

        self.W_q = nn.Linear(d_model, d_k)
        self.W_k = nn.Linear(d_model, d_k)
        self.W_v = nn.Linear(d_model, d_v)
        self.d_k = d_k

    def forward(self, query, key, value, mask=None):
        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)
        return scaled_dot_product_attention(Q, K, V, mask)
```

### Soft Dictionary Module

```python
class SoftDictionary(nn.Module):
    """
    Attention as a Soft Dictionary Lookup.
    
    Emphasizes the dictionary interpretation:
    - Keys index the dictionary
    - Values are the stored content
    - Queries retrieve relevant content via soft matching
    """
    
    def __init__(self, key_dim: int, value_dim: int, num_entries: int = None):
        super().__init__()
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.scale = key_dim ** -0.5
        
        if num_entries is not None:
            self.keys = nn.Parameter(torch.randn(num_entries, key_dim))
            self.values = nn.Parameter(torch.randn(num_entries, value_dim))
        else:
            self.keys = None
            self.values = None
            
    def forward(self, query, keys=None, values=None, temperature=1.0):
        """
        Args:
            query: (batch, query_dim) or (batch, num_queries, query_dim)
            keys: (batch, num_entries, key_dim) or None to use learned keys
            values: (batch, num_entries, value_dim) or None to use learned values
            temperature: Controls sharpness (lower = harder selection)
        """
        if keys is None:
            keys = self.keys.unsqueeze(0).expand(query.size(0), -1, -1)
        if values is None:
            values = self.values.unsqueeze(0).expand(query.size(0), -1, -1)
            
        if query.dim() == 2:
            query = query.unsqueeze(1)
            
        scores = torch.matmul(query, keys.transpose(-2, -1)) * self.scale
        scores = scores / temperature
        weights = F.softmax(scores, dim=-1)
        retrieved = torch.matmul(weights, values)
        
        return retrieved.squeeze(1), weights.squeeze(1)
```

## Properties of Attention

### Permutation Equivariance

If inputs are permuted, outputs are permuted identically:

$$\text{Attention}(\mathbf{P}\mathbf{X}) = \mathbf{P} \cdot \text{Attention}(\mathbf{X})$$

Attention treats positions symmetrically—positional information must be added explicitly.

### Computational Complexity

For sequence length $n$ and dimension $d$:

| Operation | Time | Space |
|-----------|------|-------|
| Score computation | $O(n^2 d)$ | $O(n^2)$ |
| Softmax + aggregation | $O(n^2)$ | $O(n^2)$ |
| **Total** | $O(n^2 d)$ | $O(n^2)$ |

The quadratic complexity limits applicability to very long sequences, motivating efficient variants (sparse attention, linear attention, FlashAttention).

## Connections to Other Concepts

### Memory Networks

Memory Networks (Weston et al., 2015) explicitly formalize the attention-as-memory view:

$$\mathbf{o} = \sum_i p_i \mathbf{c}_i$$

where $p_i = \text{softmax}(\mathbf{q}^T \mathbf{m}_i)$ are addressing weights, $\mathbf{m}_i$ are memory keys, and $\mathbf{c}_i$ are memory values. Transformers generalize Memory Networks by making the memory the sequence itself—this is self-attention.

### Hopfield Networks and Associative Memory

Modern Hopfield networks (Ramsauer et al., 2020) reveal that attention is a continuous relaxation of associative memory:

$$\text{new state} = \text{softmax}(\beta \cdot \text{state} \cdot \text{patterns}^T) \cdot \text{patterns}$$

This is exactly the attention formula! The connection shows that attention implements **pattern completion**—given a partial pattern (query), retrieve stored patterns (values). The exponential storage capacity of modern Hopfield networks explains why Transformers can handle long contexts.

### Retrieval-Augmented Generation (RAG)

RAG systems use attention-like retrieval at scale:

| RAG Component | Attention Analogy |
|---------------|-------------------|
| Query encoder | Query projection $\mathbf{W}_Q$ |
| Document index | Keys |
| Document content | Values |
| Retrieval | Attention computation |
| Reader | Downstream processing |

RAG can be viewed as attention over an external knowledge base, extending the model's "memory" beyond its parameters.

## From Seq2Seq Attention to Transformers

The evolution from RNN attention to Transformers revealed key insights:

1. **Attention does the heavy lifting**: In seq2seq with attention, most information flows through attention, not recurrence
2. **Self-attention**: If attention connects encoder-decoder, why not positions within the same sequence?
3. **Parallelisation**: Removing recurrence enables massive parallelisation
4. **Positional encoding**: Without recurrence, position must be encoded explicitly

| Year | Development | Impact |
|------|-------------|--------|
| 2014 | Seq2seq with LSTM | Encoder-decoder paradigm |
| 2015 | Bahdanau attention | Solved bottleneck problem |
| 2015 | Luong attention | Efficient alternatives |
| 2017 | Transformer | Self-attention replaces recurrence |

## Summary

| Concept | Description |
|---------|-------------|
| **Core idea** | Differentiable soft dictionary: weighted combination of values based on query-key compatibility |
| **Q-K-V roles** | Q: search intent; K: findability; V: content |
| **Score functions** | Dot-product (fast), additive (expressive), scaled (stable) |
| **Variants** | Self (within sequence), cross (between sequences), causal (autoregressive) |
| **Complexity** | $O(n^2 d)$ time and $O(n^2)$ space |
| **Temperature** | Controls soft-to-hard spectrum; $\sqrt{d_k}$ scaling adapts to dimensionality |

## References

1. Bahdanau, D., Cho, K., & Bengio, Y. (2014). "Neural Machine Translation by Jointly Learning to Align and Translate." *arXiv:1409.0473*.
2. Luong, M.-T., Pham, H., & Manning, C. D. (2015). "Effective Approaches to Attention-based Neural Machine Translation." *EMNLP*.
3. Vaswani, A., et al. (2017). "Attention Is All You Need." *NeurIPS*.
4. Weston, J., Chopra, S., & Bordes, A. (2015). "Memory Networks." *ICLR*.
5. Sukhbaatar, S., et al. (2015). "End-To-End Memory Networks." *NeurIPS*.
6. Ramsauer, H., et al. (2020). "Hopfield Networks is All You Need." *ICLR*.
7. Graves, A., Wayne, G., & Danihelka, I. (2014). "Neural Turing Machines." *arXiv:1410.5401*.
