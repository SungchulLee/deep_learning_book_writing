# Multimodal Fusion

## Learning Objectives

By the end of this section, you will be able to:

- Distinguish early, late, and mid-level fusion strategies
- Implement cross-attention for vision-language fusion
- Understand the trade-offs between dual-encoder and fusion-based architectures

## Fusion Strategies

### Early Fusion

Concatenate raw or lightly processed features from different modalities, then process jointly:

$$\mathbf{z} = f([\mathbf{v}; \mathbf{t}])$$

**Advantage**: Maximum interaction between modalities
**Disadvantage**: Computationally expensive, requires paired data

### Late Fusion

Process each modality independently, combine only at the decision level:

$$\mathbf{z} = g(\mathbf{v}) + h(\mathbf{t})$$

**Advantage**: Efficient, modalities can be pre-computed independently
**Disadvantage**: Limited cross-modal interaction

### Mid-Level Fusion (Cross-Attention)

Use attention mechanisms to selectively combine modality-specific features:

```python
import torch
import torch.nn as nn

class CrossAttentionFusion(nn.Module):
    """Cross-attention for vision-language fusion."""
    def __init__(self, d_model=768, nhead=12):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )
    
    def forward(self, query, context):
        """
        Args:
            query: (B, N_q, D) features to be enriched
            context: (B, N_c, D) features to attend to
        """
        attended, _ = self.cross_attn(query, context, context)
        query = self.norm(query + attended)
        query = self.norm(query + self.ffn(query))
        return query
```

## Architecture Comparison

| Architecture | Fusion Type | Retrieval Speed | Understanding Depth |
|-------------|------------|----------------|-------------------|
| CLIP/ALIGN | Late (dual-encoder) | Very fast | Limited |
| BLIP (ITM) | Mid (cross-attention) | Moderate | Deep |
| Flamingo | Early (interleaved) | Slow | Deepest |

## Summary

The choice of fusion strategy depends on the application: dual-encoders for retrieval (pre-compute embeddings), cross-attention for understanding tasks (VQA, grounding), and early fusion for maximum reasoning capability.

## References

1. Li, J., et al. (2022). BLIP: Bootstrapping Language-Image Pre-training. ICML.
2. Alayrac, J.-B., et al. (2022). Flamingo: a Visual Language Model for Few-Shot Learning. NeurIPS.
