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

!!! note "인셉션 점수는 참 자료를 넘어설 수 있다"
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
로그 가능도가 퍼짐 모형을 따지는 데 쓸모 있는 자인 까닭을 밝혀라. 그 한계는 무엇인가?

??? success "익힘 1 풀이"
    로그 가능도는 남겨 둔 시험 자료에 모형이 확률을 얼마나 잘 매기는지 잰다. $\mathcal{L} = \frac{1}{N}\sum_i \log p_\theta(x_i)$이다. 쓸모 있는 까닭은 이렇다. (1) 올바른 점수 규칙이다(참 분포에서 가장 커진다). (2) 모형을 견줄 수 있는 수 하나를 준다. (3) 표본이 나쁜 것과 최빈값 무너짐을 모두 벌한다. **한계**는 이렇다. (1) 가능도가 높아도 표본이 나쁠 수 있다(참되지 않은 자리에 무게를 두는 섞음 따위). (2) 퍼짐 모형에서는 딱 맞게 셈하기가 흔히 어렵다(ELBO이나 값비싼 상미분 방정식 따짐이 든다). (3) 느낌의 좋음을 곧바로 재지는 않는다.

---

**익힘 2.**
만들개 모형을 따지는 자로서 FID, 인셉션 점수, 로그 가능도를 견주어라.

??? success "익힘 2 풀이"
    | 자 | 재는 것 | 참 자료가 드는가 | 최빈값 무너짐을 알아내는가 | 느낌의 좋음 |
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
