# CLIP Score

The **CLIP Score** measures alignment between generated images and their text conditioning. It is the primary metric for evaluating text-to-image models.

## Definition

Given a generated image $x$ and its text prompt $c$:

$$\text{CLIPScore}(x, c) = \max\bigl(\cos(E_I(x),\, E_T(c)),\, 0\bigr) \times 100$$

where $E_I$ and $E_T$ are CLIP's image and text encoders, and $\cos(\cdot, \cdot)$ is cosine similarity.

## Interpretation

Higher CLIP Score indicates better text-image alignment. The metric ranges from 0 to ~35 in practice, with scores above 30 indicating strong alignment.

| Quality level | CLIP Score range |
|--------------|-----------------|
| Poor alignment | < 20 |
| Moderate | 20–25 |
| Good | 25–30 |
| Excellent | > 30 |

## Variants

**Reference-free CLIP Score.** Compares generated image to text prompt only. Used when no reference image exists.

**Reference-based CLIPScore.** $\text{RefCLIPScore} = \text{harmonic\_mean}(\text{CLIPScore}(x, c),\, \cos(E_I(x), E_I(x_{\text{ref}})))$. Incorporates similarity to a reference image.

## Trade-off with FID

Guidance scale creates a tension between FID and CLIP Score:

| Guidance scale $w$ | FID | CLIP Score |
|-------------------|-----|------------|
| 1 (no guidance) | Best | Worst |
| 3–5 | Good | Good |
| 7.5 (typical) | Moderate | Very good |
| 15+ | Degraded | Highest |

The optimal guidance scale balances sample quality (FID) against text faithfulness (CLIP Score). A Pareto frontier analysis helps select the best operating point.

## PyTorch Computation

```python
import torch
import torch.nn.functional as F


def clip_score(
    image_features: torch.Tensor,
    text_features: torch.Tensor,
) -> torch.Tensor:
    """Compute CLIP Score between image and text features.

    Args:
        image_features: [N, D] L2-normalised CLIP image embeddings.
        text_features: [N, D] L2-normalised CLIP text embeddings.

    Returns:
        [N] CLIP scores (0-100 scale).
    """
    similarity = F.cosine_similarity(image_features, text_features)
    return torch.clamp(similarity, min=0) * 100
```

## Limitations

CLIP Score inherits CLIP's biases (trained on internet image-text pairs), may not capture fine-grained spatial relationships, and can be saturated by generic visual features. Human evaluation remains the gold standard for subjective quality assessment.

## References

1. Hessel, J., et al. (2021). "CLIPScore: A Reference-free Evaluation Metric for Image Captioning." *EMNLP*.
2. Radford, A., et al. (2021). "Learning Transferable Visual Models From Natural Language Supervision." *ICML*.
