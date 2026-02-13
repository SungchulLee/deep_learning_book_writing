# Self-Supervised Learning Overview

Self-supervised learning (SSL) learns meaningful representations from unlabeled data by creating supervision signals directly from the input data structure, without requiring human annotations.

## Motivation

The key bottleneck in deep learning is labeled data. SSL addresses this by exploiting the structure inherent in data:

| Paradigm | Labels needed | Data efficiency | Representation quality |
|----------|--------------|-----------------|----------------------|
| Supervised | $n$ per class | Low | High (on that task) |
| Self-supervised | 0 | High | General-purpose |
| Semi-supervised | Few | Moderate | Moderate |

## The SSL Framework

Given unlabeled data $\mathcal{D} = \{x_1, ..., x_n\}$, SSL learns an encoder $f_\theta$ by solving a pretext task:

$$\mathcal{L}_{\text{SSL}} = \mathbb{E}_{x \sim \mathcal{D}} \left[ \ell(f_\theta(T(x)), y(x)) \right]$$

where $T(x)$ transforms the input and $y(x)$ is a pseudo-label derived from $x$ itself.

## Evolution of SSL Methods

```
Classical Pretext Tasks (2015-2018)
  └─ Rotation, Jigsaw, Colorization, Inpainting
      │
Contrastive Learning (2018-2021)
  └─ SimCLR, MoCo, BYOL, SimSiam, Barlow Twins
      │
Masked Modeling (2021-present)
  └─ MAE, BEiT, Data2Vec
      │
Self-Distillation (2021-present)
  └─ DINO, DINOv2
```

## Taxonomy

| Category | Key idea | Examples |
|----------|----------|---------|
| Pretext tasks | Predict data transformations | Rotation, jigsaw, colorization |
| Contrastive | Pull positives close, push negatives apart | SimCLR, MoCo |
| Non-contrastive | Match representations without negatives | BYOL, SimSiam, Barlow Twins |
| Masked modeling | Reconstruct masked inputs | MAE, BEiT |
| Self-distillation | Student-teacher without labels | DINO, DINOv2 |

## Evaluation Paradigm

SSL representations are evaluated by how well they transfer:

1. **Linear probing**: Freeze encoder, train linear classifier
2. **k-NN evaluation**: Nearest-neighbour classification in feature space
3. **Fine-tuning**: End-to-end fine-tuning on downstream task
4. **Few-shot**: Transfer with very few labeled examples

The [evaluation section](../self_supervised_evaluation/evaluation.md) covers these protocols in detail.

## References

1. Liu, X., et al. (2021). "Self-Supervised Learning: Generative or Contrastive." *IEEE TKDE*.
2. Balestriero, R., et al. (2023). "A Cookbook of Self-Supervised Learning." arXiv.
