# Attention Fundamentals

## Overview

Attention is the core mechanism that enables Transformers to dynamically focus on relevant parts of the input when computing representations. Unlike fixed convolutions or recurrence, attention allows each position to selectively gather information from any other position based on learned relevance.

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

## Query-Key-Value Formulation

Given an input sequence $\mathbf{X} \in \mathbb{R}^{n \times d}$, attention computes three projections:

$$\mathbf{Q} = \mathbf{X}\mathbf{W}_Q \quad \text{(Queries: "What am I looking for?")}$$

$$\mathbf{K} = \mathbf{X}\mathbf{W}_K \quad \text{(Keys: "What do I offer?")}$$

$$\mathbf{V} = \mathbf{X}\mathbf{W}_V \quad \text{(Values: "What will I contribute?")}$$

where $\mathbf{W}_Q, \mathbf{W}_K \in \mathbb{R}^{d \times d_k}$ and $\mathbf{W}_V \in \mathbb{R}^{d \times d_v}$.

### Why Three Separate Projections?

The separation serves distinct purposes:

**Keys are optimized for findability**—concise, categorical, searchable descriptors that identify what information a token offers.

**Values are optimized for information transfer**—rich semantic content that gets transferred once a match is established.

A post about sunset photography might have hashtag #lighting, but its actual content discusses color temperature, exposure settings, atmospheric conditions, and timing. The hashtag gets you there; the content is what you came for.

## The Attention Mechanism

The complete attention computation:

$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}\right)\mathbf{V}$$

Breaking this down:

### Step 1: Compute Relevance Scores

$$\mathbf{S} = \mathbf{Q}\mathbf{K}^T \in \mathbb{R}^{n \times n}$$

Entry $S_{ij}$ measures how much position $i$ should attend to position $j$—comparing the query at position $i$ against the key at position $j$.

### Step 2: Scale by $\sqrt{d_k}$

$$\mathbf{S}_{\text{scaled}} = \frac{\mathbf{S}}{\sqrt{d_k}}$$

This prevents dot products from growing too large with dimension, which would push softmax into saturation.

### Step 3: Apply Softmax

$$\mathbf{A} = \text{softmax}(\mathbf{S}_{\text{scaled}})$$

Row $i$ becomes a probability distribution over all positions, indicating where position $i$ should gather information.

### Step 4: Weighted Sum of Values

$$\mathbf{Z} = \mathbf{A}\mathbf{V}$$

Each output position receives a weighted combination of value vectors:

$$\mathbf{z}_i = \sum_{j=1}^{n} A_{ij} \mathbf{v}_j$$

## A Concrete Example

Consider: "The cat sat on the mat because it was soft."

When processing "it":

| Component | Role |
|-----------|------|
| **Query** | "I'm searching for #noun #antecedent #nearby" |
| **Keys** (from other tokens) | "cat": #noun #animate #subject; "mat": #noun #inanimate #object |
| **Attention weights** | Higher weight on "mat" (because "soft" implies inanimate) |
| **Value** | Semantic content of "mat" (thing-you-sit-on, surface, location) |

The key (hashtag) identifies "mat" as a potential antecedent. The value provides the semantic information needed to resolve the pronoun.

## The Learned Aspect

The network learns:

- $\mathbf{W}_Q$: How to formulate effective searches
- $\mathbf{W}_K$: How to tag content for discoverability
- $\mathbf{W}_V$: How to package content for useful transmission

Training optimizes all three simultaneously—learning not just what to search for, but how tokens should advertise themselves and what information they should deliver.

## Attention as Soft Dictionary Lookup

Attention can be viewed as a **differentiable dictionary**:

| Hard Dictionary | Soft Attention |
|-----------------|----------------|
| Exact key match | Similarity-weighted matching |
| Returns one value | Returns weighted combination |
| Discrete lookup | Continuous, differentiable |

Traditional dictionary: `dict[query] → value` (exact match)

Attention: `query → weighted_sum(values, similarity(query, keys))`

This soft matching enables gradient flow and end-to-end learning.

## Properties of Attention

### Permutation Equivariance

If input positions are permuted, attention outputs are permuted identically:

$$\text{Attention}(\mathbf{P}\mathbf{X}) = \mathbf{P} \cdot \text{Attention}(\mathbf{X})$$

Attention treats positions symmetrically—positional information must be added separately.

### Computational Complexity

- **Time**: $O(n^2 d)$ for sequence length $n$ and dimension $d$
- **Memory**: $O(n^2)$ for the attention matrix

The quadratic complexity in sequence length is the primary bottleneck for long sequences.

### No Built-in Position Awareness

Attention scores depend only on content, not position. The same word at different positions produces identical queries and keys (before positional encoding). Position must be explicitly encoded.

## Attention Variants

| Variant | Modification | Use Case |
|---------|--------------|----------|
| Self-Attention | Q, K, V from same sequence | Internal context modeling |
| Cross-Attention | Q from one sequence, K/V from another | Encoder-decoder bridging |
| Masked Attention | Prevent attending to certain positions | Autoregressive generation |
| Sparse Attention | Attend to subset of positions | Long sequences |

## Summary

Attention provides a mechanism for:

1. **Dynamic routing**: Each position decides where to gather information
2. **Content-based addressing**: Relevance determined by semantic similarity
3. **Parallel computation**: All positions computed simultaneously
4. **Differentiable lookup**: Enables end-to-end learning

The Query-Key-Value formulation separates the concerns of searching (Q-K matching) from information transfer (V retrieval), giving the model flexibility to learn optimal representations for each role.

## References

- Vaswani et al., "Attention Is All You Need" (2017)
- Bahdanau et al., "Neural Machine Translation by Jointly Learning to Align and Translate" (2015)
