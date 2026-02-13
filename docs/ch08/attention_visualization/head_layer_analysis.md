# Head and Layer Analysis

## Overview

Different attention heads and layers learn distinct patterns. Research shows heads specialize into positional (adjacent tokens, early layers), syntactic (dependency structure, middle layers), semantic (related concepts, later layers), and separator patterns ([CLS]/[SEP] attention).

## Attention Entropy

The entropy of attention measures focus: $H(A_h) = -\sum_j \alpha_{hj} \log \alpha_{hj}$. Low entropy indicates focused attention on few tokens; high entropy indicates diffuse, broad attention.

```python
def attention_entropy(attn_weights):
    attn = attn_weights.clamp(min=1e-12)
    return -(attn * attn.log()).sum(dim=-1)
```

## Head Importance

Estimate via gradient sensitivity: $I_h = \mathbb{E}_x |\frac{\partial \mathcal{L}}{\partial A_h} \odot A_h|$. Michel et al. (2019) showed many heads can be pruned post-training with minimal performance loss, and some can even be pruned during training.

## Probing Across Layers

Different linguistic properties emerge at different depths: layers 1-3 capture surface features (word identity, position), layers 4-8 capture syntactic features (POS tags, dependency arcs), and layers 9-12 capture semantic features (NER, coreference, sentiment). This hierarchy holds across BERT, GPT, and other transformer architectures.
