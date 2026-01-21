# Self-Attention

## Definition

Self-attention is attention where queries, keys, and values all come from the **same sequence**:

$$\mathbf{Q} = \mathbf{X}\mathbf{W}_Q, \quad \mathbf{K} = \mathbf{X}\mathbf{W}_K, \quad \mathbf{V} = \mathbf{X}\mathbf{W}_V$$

Each position attends to all positions (including itself) within the same sequence.

## Contrast with Cross-Attention

| Aspect | Self-Attention | Cross-Attention |
|--------|----------------|-----------------|
| Q source | Same sequence | Decoder sequence |
| K, V source | Same sequence | Encoder sequence |
| Purpose | Internal context | External reference |
| Typical use | Within encoder/decoder | Decoder-to-encoder |

## How Self-Attention Works

Given a sequence $\mathbf{X} = (\mathbf{x}_1, \ldots, \mathbf{x}_n) \in \mathbb{R}^{n \times d}$:

### Step 1: Project to Q, K, V

$$\mathbf{Q} = \mathbf{X}\mathbf{W}_Q \in \mathbb{R}^{n \times d_k}$$
$$\mathbf{K} = \mathbf{X}\mathbf{W}_K \in \mathbb{R}^{n \times d_k}$$
$$\mathbf{V} = \mathbf{X}\mathbf{W}_V \in \mathbb{R}^{n \times d_v}$$

Each token generates:
- A **query**: "What information am I looking for?"
- A **key**: "What information do I offer?"
- A **value**: "What will I contribute if selected?"

### Step 2: Compute Attention

$$\text{SelfAttention}(\mathbf{X}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}\right)\mathbf{V}$$

The attention matrix $\mathbf{A} \in \mathbb{R}^{n \times n}$ is **square**—every position attends to every position.

## Self-Attention Enables Global Context

### The Key Advantage

Consider: "The cat sat on the mat because it was soft."

Traditional approaches (RNNs) process sequentially:
- "it" at position 8 must wait for information from "mat" at position 6 to propagate through positions 7

Self-attention processes in parallel:
- Position 8 ("it") directly attends to position 6 ("mat")
- Path length: 1 (constant, regardless of distance)

### Path Length Comparison

| Architecture | Path Length (positions $i$ to $j$) |
|--------------|-----------------------------------|
| RNN | $O(|i - j|)$ |
| CNN | $O(\log_{k}|i - j|)$ with kernel size $k$ |
| Self-Attention | $O(1)$ |

Short paths enable better gradient flow for learning long-range dependencies.

## Visualizing Self-Attention

For the sentence "The cat sat":

```
         Keys
         The  cat  sat
        ┌────┬────┬────┐
    The │ .6 │ .2 │ .2 │  Query "The" attends mostly to itself
        ├────┼────┼────┤
Q   cat │ .1 │ .7 │ .2 │  Query "cat" attends mostly to itself
        ├────┼────┼────┤
    sat │ .2 │ .5 │ .3 │  Query "sat" attends to "cat" (subject)
        └────┴────┴────┘
```

Each row is a probability distribution (sums to 1).

## Bidirectional vs Causal Self-Attention

### Bidirectional (Encoder-style)

Every position sees every other position:

$$\mathbf{A} = \begin{pmatrix} a_{11} & a_{12} & a_{13} \\ a_{21} & a_{22} & a_{23} \\ a_{31} & a_{32} & a_{33} \end{pmatrix}$$

**Use case**: Understanding tasks (BERT) where full context is available.

### Causal/Masked (Decoder-style)

Position $i$ can only attend to positions $1, \ldots, i$:

$$\mathbf{A} = \begin{pmatrix} a_{11} & 0 & 0 \\ a_{21} & a_{22} & 0 \\ a_{31} & a_{32} & a_{33} \end{pmatrix}$$

**Use case**: Generation tasks (GPT) where we predict the next token.

See [Masked Self-Attention](../transformers_nlp/masked_attention.md) for details.

## Properties of Self-Attention

### Permutation Equivariance

If we permute the input positions, the output is permuted identically:

$$\text{SelfAttn}(\mathbf{P}\mathbf{X}) = \mathbf{P} \cdot \text{SelfAttn}(\mathbf{X})$$

where $\mathbf{P}$ is a permutation matrix.

**Implication**: Self-attention treats positions symmetrically. Positional information must be added explicitly via positional encodings.

### Computational Complexity

- **Time**: $O(n^2 d)$ — quadratic in sequence length
- **Memory**: $O(n^2)$ — storing the attention matrix

This is the primary bottleneck for long sequences.

### No Inductive Bias for Order

Unlike RNNs (sequential) or CNNs (local), self-attention has no built-in notion of position or locality. This is both a strength (flexibility) and a weakness (requires positional encoding).

## What Self-Attention Learns

Research has shown that different layers and heads learn different patterns:

### Early Layers

- Local patterns (attending to adjacent tokens)
- Positional patterns (fixed offsets)
- Syntactic basics (punctuation, function words)

### Middle Layers

- Syntactic relationships (subject-verb, modifier-head)
- Coreference (pronoun-antecedent)
- Named entity patterns

### Later Layers

- Task-specific patterns
- Semantic relationships
- Long-range dependencies

## Self-Attention vs Fully Connected

Self-attention might seem similar to a fully connected layer, but there are key differences:

| Aspect | Fully Connected | Self-Attention |
|--------|-----------------|----------------|
| Weights | Fixed per position | Dynamic, content-based |
| Sharing | None (different weights per position) | Same Q, K, V matrices |
| Flexibility | Rigid | Adapts to input |

Self-attention computes **dynamic** weights based on content, while fully connected layers have **static** weights.

## PyTorch Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SelfAttention(nn.Module):
    def __init__(self, d_model, d_k=None, d_v=None):
        super().__init__()
        self.d_k = d_k or d_model
        self.d_v = d_v or d_model
        
        self.W_q = nn.Linear(d_model, self.d_k)
        self.W_k = nn.Linear(d_model, self.d_k)
        self.W_v = nn.Linear(d_model, self.d_v)
    
    def forward(self, X, mask=None):
        """
        Args:
            X: Input sequence (batch, seq_len, d_model)
            mask: Optional attention mask
        
        Returns:
            output: Self-attended output (batch, seq_len, d_v)
            attn_weights: Attention weights (batch, seq_len, seq_len)
        """
        # Project to Q, K, V
        Q = self.W_q(X)  # (batch, seq_len, d_k)
        K = self.W_k(X)  # (batch, seq_len, d_k)
        V = self.W_v(X)  # (batch, seq_len, d_v)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, V)
        
        return output, attn_weights

# Example usage
d_model = 512
seq_len = 10
batch_size = 2

self_attn = SelfAttention(d_model)
X = torch.randn(batch_size, seq_len, d_model)

output, weights = self_attn(X)
print(f"Input shape: {X.shape}")           # (2, 10, 512)
print(f"Output shape: {output.shape}")      # (2, 10, 512)
print(f"Attention shape: {weights.shape}")  # (2, 10, 10)
```

## Self-Attention in Context

### In Encoder (BERT-style)

- Bidirectional: all positions see all positions
- Learns contextual representations for understanding

### In Decoder (GPT-style)

- Causal: each position only sees previous positions
- Maintains autoregressive property for generation

### Combined (Encoder-Decoder)

- Encoder: bidirectional self-attention
- Decoder: causal self-attention + cross-attention to encoder

## Summary

Self-attention is characterized by:

| Property | Description |
|----------|-------------|
| **Source** | Q, K, V from same sequence |
| **Scope** | Every position attends to every position |
| **Path length** | $O(1)$ between any two positions |
| **Complexity** | $O(n^2 d)$ time, $O(n^2)$ space |
| **Inductive bias** | None (requires positional encoding) |

Self-attention is the mechanism that gives Transformers their ability to model arbitrary dependencies within a sequence, forming the foundation for both understanding (BERT) and generation (GPT) architectures.

## References

- Vaswani et al., "Attention Is All You Need" (2017)
- Devlin et al., "BERT" (2019)
