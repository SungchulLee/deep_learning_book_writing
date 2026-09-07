# Inception Score (IS) for Diffusion Models
The Inception Score evaluates generated image quality and diversity using a pre-trained Inception-v3 classifier. For the full mathematical derivation, implementation, and limitations analysis, see [IS in §25.6](../../ch25/gan_evaluation/inception_score.md). This page covers diffusion-specific usage.

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
    GANs sometimes achieve IS above real data because mode collapse concentrates predictions on fewer, more confident classes. This is why IS alone can be misleading — always pair with FID and [Precision/Recall](../../ch25/gan_evaluation/precision_recall.md).

## Guidance Scale Effect on IS

Like FID, IS is affected by classifier-free guidance:

| Guidance scale $w$ | IS | FID |
|-------------------|-----|-----|
| 1.0 | Lower | Higher |
| 3.0–5.0 | Good | **Best** |
| 10+ | **Highest** | Degraded |

IS monotonically increases with guidance scale because stronger guidance produces more class-confident images. However, this comes at the cost of diversity, which IS partially misses. The FID-optimal guidance scale is typically lower than the IS-optimal one.

## Limitations for Diffusion Models

IS has the same fundamental limitations as for GANs (see [§25.6](../../ch25/gan_evaluation/inception_score.md#limitations-and-pitfalls)), with additional diffusion-specific caveats:

- **Text-conditioned models**: IS only measures ImageNet class diversity, not text–image alignment. Use CLIP Score for text-to-image evaluation.
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

See the comprehensive IS treatment in [§25.6](../../ch25/gan_evaluation/inception_score.md) for implementation code, information-theoretic interpretation, and best practices.

## References

1. Salimans, T., et al. (2016). "Improved Techniques for Training GANs." *NeurIPS*.
2. Ho, J., Jain, A., & Abbeel, P. (2020). "Denoising Diffusion Probabilistic Models." *NeurIPS*.
3. Dhariwal, P., & Nichol, A. (2021). "Diffusion Models Beat GANs on Image Synthesis." *NeurIPS*.

## 익힘 문제

**익힘 1.**
Explain why log-likelihood is a useful metric for evaluating diffusion models. What are its limitations?

??? success "익힘 1 풀이"
    Log-likelihood measures how well the model assigns probability to held-out test data: $\mathcal{L} = \frac{1}{N}\sum_i \log p_\theta(x_i)$. It is useful because: (1) it is a proper scoring rule (maximized by the true distribution), (2) it provides a single number for model comparison, (3) it penalizes both poor sample quality and mode collapse. **Limitations**: (1) models with high likelihood can produce poor samples (e.g., mixtures that assign mass to unrealistic regions), (2) exact computation is often intractable for diffusion models (requires the ELBO or expensive ODE-based evaluation), (3) it does not directly measure perceptual quality.

---

**익힘 2.**
Compare FID, Inception Score, and log-likelihood as evaluation metrics for generative models.

??? success "익힘 2 풀이"
    | Metric | Measures | Requires Real Data | Detects Mode Collapse | Perceptual Quality |
    |--------|---------|-------------------|----------------------|-------------------|
    | **FID** | Distributional similarity | Yes | Yes | Good |
    | **IS** | Quality + diversity | No | Partially | Moderate |
    | **Log-likelihood** | Density accuracy | Yes (test set) | Yes | Weak |

    FID is the most widely used because it correlates well with human judgment and captures both quality and diversity. IS only evaluates generated samples. Log-likelihood is theoretically principled but can disagree with perceptual quality. Best practice: report all three.

---

**익힘 3.**
What is the bits-per-dimension (BPD) metric? How is it computed for diffusion models?

??? success "익힘 3 풀이"
    BPD normalizes the negative log-likelihood by the data dimensionality and converts to bits: $\text{BPD} = -\frac{\log_2 p(x)}{d}$ where $d$ is the number of dimensions (e.g., $3 \times 32 \times 32 = 3072$ for CIFAR-10). For diffusion models, the log-likelihood is bounded by the ELBO: $\log p(x) \geq \text{ELBO} = -\sum_t L_t$ where $L_t$ are the KL divergence terms. Exact computation uses the probability flow ODE and the instantaneous change of variables formula. Lower BPD indicates a better model. State-of-the-art diffusion models achieve $\sim$2.5 BPD on CIFAR-10.

---

**익힘 4.**
Why can a generative model with excellent FID still fail in production applications? What additional evaluations are needed?

??? success "익힘 4 풀이"
    FID measures average distributional quality but misses: (1) **Tail behavior**: rare but important failure modes (artifacts, offensive content) are averaged out, (2) **Conditional fidelity**: FID is typically computed unconditionally; class-conditional or text-conditional FID may differ, (3) **Memorization**: a model that memorizes training data achieves low FID but is useless for generation, (4) **Diversity within conditions**: FID may be low even if the model generates the same image for similar prompts. Additional evaluations: precision/recall curves, per-class FID, memorization detection (nearest-neighbor distance to training set), human evaluation for quality and diversity, and application-specific metrics (e.g., text-image alignment for text-to-image models).
