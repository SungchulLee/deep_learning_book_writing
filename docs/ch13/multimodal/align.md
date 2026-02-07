# ALIGN: A Large-scale ImaGe and Noisy-text Embedding

## Learning Objectives

By the end of this section, you will be able to:

- Understand ALIGN's approach to scaling vision-language pre-training with noisy data
- Compare ALIGN's training strategy with CLIP's curated data approach
- Explain the dual-encoder architecture for efficient cross-modal retrieval

## Overview

ALIGN (Jia et al., 2021) scales vision-language pre-training to 1.8 billion noisy image-text pairs harvested from the web, demonstrating that data scale can compensate for noise in supervision.

## Key Differences from CLIP

| Aspect | CLIP | ALIGN |
|--------|------|-------|
| Training data | 400M curated pairs | 1.8B noisy pairs |
| Data quality | Carefully filtered | Raw alt-text, minimal filtering |
| Image encoder | ViT or ResNet | EfficientNet |
| Text encoder | Transformer | BERT |
| Key insight | Curation matters | Scale compensates for noise |

## Architecture

ALIGN uses the same dual-encoder contrastive learning framework as CLIP:

$$\mathcal{L} = -\frac{1}{N}\sum_{i=1}^{N}\left[\log\frac{\exp(\text{sim}(v_i, t_i)/\tau)}{\sum_j \exp(\text{sim}(v_i, t_j)/\tau)} + \log\frac{\exp(\text{sim}(t_i, v_i)/\tau)}{\sum_j \exp(\text{sim}(t_i, v_j)/\tau)}\right]$$

The primary contribution is demonstrating that with sufficient scale (1.8B pairs), the dual-encoder approach achieves strong zero-shot transfer even with noisy training data.

## Summary

ALIGN established that vision-language pre-training can scale effectively with web-harvested data, providing an alternative to CLIP's carefully curated approach. Both methods achieve strong zero-shot transfer, suggesting that the contrastive learning framework is robust to data quality at sufficient scale.

## References

1. Jia, C., et al. (2021). Scaling Up Visual and Vision-Language Representation Learning With Noisy Text Supervision. ICML.
