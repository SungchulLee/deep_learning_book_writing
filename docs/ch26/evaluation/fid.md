# Fréchet Inception Distance (FID) for Diffusion Models

FID is the standard metric for evaluating diffusion model sample quality. For the full mathematical derivation, implementation, and best practices, see [FID in §24.6](../../ch25/gan_evaluation/fid.md). This page focuses on diffusion-specific considerations.

## Definition Recap

$$
\text{FID} = \|\mu_r - \mu_g\|^2 + \text{tr}\!\left(\Sigma_r + \Sigma_g - 2(\Sigma_r \Sigma_g)^{1/2}\right)
$$

Lower FID indicates generated samples are closer to the real data distribution in Inception-v3 feature space.

## Diffusion-Specific Considerations

### Sampling Steps and FID

Diffusion models face a unique quality–speed tradeoff controlled by the number of denoising steps:

| Sampler | Steps | Typical CIFAR-10 FID |
|---------|-------|---------------------|
| DDPM | 1000 | ~3.17 |
| DDIM | 50 | ~4.67 |
| DDIM | 10 | ~13.36 |
| DPM-Solver++ | 20 | ~2.80 |

More steps generally improve FID but increase inference cost linearly. Modern solvers (DPM-Solver, DEIS) achieve strong FID with far fewer steps than the original DDPM sampler.

### Guidance Scale and FID

Classifier-free guidance trades diversity for fidelity, creating a characteristic FID curve:

| Guidance scale $w$ | FID | Precision | Recall |
|-------------------|-----|-----------|--------|
| 1.0 (no guidance) | Higher | Lower | Higher |
| 2.0–4.0 | **Optimal** | Good | Good |
| 7.5 (common default) | Moderate | High | Lower |
| 15+ | Degraded | Highest | Low |

The optimal guidance scale minimizes FID and represents the best balance between sample quality and diversity. Beyond this point, FID degrades because recall (diversity) drops faster than precision improves.

### Noise Schedule Impact

The noise schedule $\beta_t$ affects the learned score function and therefore sample quality:

- **Linear schedule**: Standard choice, FID benchmarks typically use this
- **Cosine schedule**: Often yields better FID on smaller images (Nichol & Dhariwal, 2021)
- **Learned schedules**: Can further reduce FID through end-to-end optimization

### FID-50K Convention

Standard practice for diffusion model evaluation:

1. Generate **50,000** samples using the full sampling pipeline
2. Compute Inception-v3 pool3 features (2048-d) for both real and generated sets
3. Use **consistent preprocessing**: bilinear resize to 299×299, ImageNet normalization
4. Report FID with the exact sampler configuration (steps, guidance scale, noise schedule)

!!! warning "Preprocessing Matters"
    Small differences in resize interpolation or normalization can shift FID by several points. Always use the same preprocessing pipeline as the reference you compare against. The `torch-fidelity` and `clean-fid` libraries help ensure consistency.

## State-of-the-Art Benchmarks

| Model | CIFAR-10 FID ↓ | ImageNet 256×256 FID ↓ |
|-------|----------------|------------------------|
| DDPM (Ho et al., 2020) | 3.17 | — |
| ADM (Dhariwal & Nichol, 2021) | — | 10.94 |
| ADM + classifier guidance | — | 4.59 |
| ADM + classifier-free guidance | — | 3.94 |
| LDM / Stable Diffusion | — | ~3.60 |
| DiT-XL/2 (Peebles & Xie, 2023) | — | 2.27 |
| Consistency Models (Song et al., 2023) | 2.93 | 3.55 |

## When FID Falls Short for Diffusion

FID captures overall distributional similarity but may not reflect:

- **Text-image alignment** in conditional generation → use [CLIP Score](clip_score.md) instead
- **Fine-grained perceptual quality** → complement with [human evaluation](human_evaluation.md)
- **Likelihood fit** → use [BPD/NLL](likelihood.md) for models with tractable ELBO

A complete diffusion model evaluation should report FID alongside complementary metrics. See the comprehensive FID treatment in [§24.6](../../ch25/gan_evaluation/fid.md) for implementation details, sample size analysis, and bootstrap confidence intervals.

## References

1. Ho, J., Jain, A., & Abbeel, P. (2020). "Denoising Diffusion Probabilistic Models." *NeurIPS*.
2. Dhariwal, P., & Nichol, A. (2021). "Diffusion Models Beat GANs on Image Synthesis." *NeurIPS*.
3. Peebles, W., & Xie, S. (2023). "Scalable Diffusion Models with Transformers." *ICCV*.
4. Parmar, G., et al. (2022). "On Aliased Resizing and Surprising Subtleties in GAN Evaluation." *CVPR*.
