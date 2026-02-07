# Inception Score (IS) for Diffusion Models

The Inception Score evaluates generated image quality and diversity using a pre-trained Inception-v3 classifier. For the full mathematical derivation, implementation, and limitations analysis, see [IS in §24.6](../../ch24/gan_evaluation/inception_score.md). This page covers diffusion-specific usage.

## Definition Recap

$$
\text{IS} = \exp\!\left(\mathbb{E}_{x \sim p_g}\!\left[D_{\text{KL}}\bigl(p(y|x) \,\|\, p(y)\bigr)\right]\right)
$$

Higher IS indicates both confident class predictions (quality) and diverse class coverage (diversity). Scores range from 1 (worst) to theoretically 1000 (ImageNet classes).

## Role in Diffusion Model Evaluation

IS is a **secondary metric** for diffusion models. FID is preferred as the primary benchmark because it directly compares against real data. IS is useful for:

- **Quick sanity checks** during training (cheaper than FID)
- **Complementary signal** when FID alone is ambiguous
- **Historical comparison** with earlier GAN results

### Typical Diffusion Model IS Values

| Model | IS (CIFAR-10) ↑ |
|-------|-----------------|
| Real data | 11.24 |
| DDPM | 9.46 |
| ADM | ~10.9 |
| BigGAN (GAN baseline) | 14.73 |

!!! note "IS Can Exceed Real Data"
    GANs sometimes achieve IS above real data because mode collapse concentrates predictions on fewer, more confident classes. This is why IS alone can be misleading — always pair with FID and [Precision/Recall](../../ch24/gan_evaluation/precision_recall.md).

## Guidance Scale Effect on IS

Like FID, IS is affected by classifier-free guidance:

| Guidance scale $w$ | IS | FID |
|-------------------|-----|-----|
| 1.0 | Lower | Higher |
| 3.0–5.0 | Good | **Best** |
| 10+ | **Highest** | Degraded |

IS monotonically increases with guidance scale because stronger guidance produces more class-confident images. However, this comes at the cost of diversity, which IS partially misses. The FID-optimal guidance scale is typically lower than the IS-optimal one.

## Limitations for Diffusion Models

IS has the same fundamental limitations as for GANs (see [§24.6](../../ch24/gan_evaluation/inception_score.md#limitations-and-pitfalls)), with additional diffusion-specific caveats:

- **Text-conditioned models**: IS only measures ImageNet class diversity, not text–image alignment. Use [CLIP Score](clip_score.md) for text-to-image evaluation.
- **High-resolution generation**: IS was designed for ImageNet-scale images; it may not capture quality differences at 512×512+ resolutions.
- **Unconditional vs conditional**: IS is more meaningful for class-conditional generation than for unconditional or text-conditional models.

## Recommended Evaluation Protocol

For diffusion models, report IS as a supplement to FID:

```
Evaluation Results:
  FID-50K:          3.17  (primary metric)
  IS (50K, 10 splits): 9.46 ± 0.11  (secondary metric)
  CLIP Score:       28.5  (text-to-image only)
```

See the comprehensive IS treatment in [§24.6](../../ch24/gan_evaluation/inception_score.md) for implementation code, information-theoretic interpretation, and best practices.

## References

1. Salimans, T., et al. (2016). "Improved Techniques for Training GANs." *NeurIPS*.
2. Ho, J., Jain, A., & Abbeel, P. (2020). "Denoising Diffusion Probabilistic Models." *NeurIPS*.
3. Dhariwal, P., & Nichol, A. (2021). "Diffusion Models Beat GANs on Image Synthesis." *NeurIPS*.
