# Quick Reference: Generative Model Evaluation Metrics

## Metric Cheat Sheet

### Likelihood-Based Metrics

| Metric | Formula | Range | Better | Use Case |
|--------|---------|-------|--------|----------|
| **NLL** | `-E[log p(x)]` | [0, ∞) | Lower | Models with tractable likelihood |
| **BPD** | `NLL / (dims × log(2))` | [0, ∞) | Lower | Comparing across dimensions |
| **Perplexity** | `exp(NLL per token)` | [1, ∞) | Lower | Language models |

### Sample-Based Metrics

| Metric | Formula | Range | Better | Use Case |
|--------|---------|-------|--------|----------|
| **FID** | `‖μ_r - μ_g‖² + Tr(Σ_r + Σ_g - 2√(Σ_rΣ_g))` | [0, ∞) | Lower | Image generation quality |
| **IS** | `exp(E[KL(p(y\|x) \|\| p(y))])` | [1, ∞) | Higher | Image quality + diversity |
| **KID** | `MMD²(F_r, F_g)` | [0, ∞) | Lower | Unbiased alternative to FID |
| **Precision** | Coverage of real manifold | [0, 1] | Higher | Sample quality |
| **Recall** | Coverage of generated manifold | [0, 1] | Higher | Sample diversity |

### Reconstruction Metrics

| Metric | Formula | Range | Better | Use Case |
|--------|---------|-------|--------|----------|
| **MSE** | `E[(x - x̂)²]` | [0, ∞) | Lower | Pixel-wise similarity |
| **PSNR** | `10 log₁₀(MAX²/MSE)` | [0, ∞) | Higher | Signal quality (dB) |
| **SSIM** | Structural similarity | [0, 1] | Higher | Perceptual similarity |
| **LPIPS** | Learned perceptual distance | [0, ∞) | Lower | Deep perceptual similarity |

## When to Use Each Metric

### For VAEs
- ✅ ELBO / NLL (training objective)
- ✅ BPD (normalized comparison)
- ✅ FID (sample quality)
- ✅ Reconstruction MSE/PSNR
- ❌ IS (not designed for VAEs)

### For GANs
- ✅ FID (primary metric)
- ✅ IS (quality + diversity)
- ✅ Precision/Recall (mode coverage)
- ✅ PPL (latent space smoothness)
- ❌ Likelihood (GANs don't have explicit likelihood)

### For Diffusion Models
- ✅ FID (sample quality)
- ✅ IS (if ImageNet-like)
- ✅ NFE (sampling efficiency)
- ✅ BPD (if likelihood available)
- ✅ LPIPS (perceptual quality)

### For Autoregressive Models
- ✅ NLL / Perplexity (primary metrics)
- ✅ BPD (cross-domain comparison)
- ✅ BLEU/ROUGE (if text)
- ✅ FID (if images)

### For Normalizing Flows
- ✅ NLL (exact likelihood)
- ✅ BPD (normalized)
- ✅ FID (sample quality)
- ✅ Reverse NLL (invertibility check)

## Typical Benchmark Values

### FID Scores (ImageNet 256×256)
| Model | FID ↓ | Year |
|-------|-------|------|
| Real Data | 0.0 | - |
| StyleGAN2 | 2.8 | 2020 |
| BigGAN | 6.9 | 2019 |
| DDPM | 3.2 | 2020 |
| Baseline GAN | 50-100 | - |
| Random Images | 300+ | - |

### Inception Score (ImageNet)
| Model | IS ↑ | Year |
|-------|------|------|
| Real ImageNet | 11.2 | - |
| BigGAN-deep | 171.4 | 2019 |
| StyleGAN2 | ~9.0 | 2020 |
| DCGAN | ~6.5 | 2016 |
| Random Images | 1.0 | - |

### Perplexity (Language Models)
| Model | Perplexity ↓ | Dataset |
|-------|-------------|---------|
| GPT-3 | 20.5 | WikiText-103 |
| GPT-2 | 35.8 | WikiText-103 |
| LSTM Baseline | 60-100 | WikiText-103 |
| Random | vocab_size | Any |

## Quick Decision Tree

```
Need to evaluate generative model?
│
├─ Has tractable likelihood?
│  ├─ YES → Use NLL/BPD as primary
│  │        + FID for sample quality
│  │
│  └─ NO → Use FID as primary
│           + IS for diversity
│           + Precision/Recall
│
├─ What type of data?
│  ├─ Images → FID, IS, LPIPS
│  ├─ Text → Perplexity, BLEU, BERTScore
│  ├─ Audio → Fréchet Audio Distance
│  └─ Other → Domain-specific metrics
│
└─ What aspect to measure?
   ├─ Quality → FID, Precision, LPIPS
   ├─ Diversity → IS, Recall, KID
   ├─ Coverage → Recall, Mode Count
   └─ Likelihood → NLL, BPD, Perplexity
```

## Command Line Examples

### Using torch-fidelity
```bash
# Install
pip install torch-fidelity

# Compute FID
fidelity --gpu 0 --fid --input1 real_images/ --input2 generated_images/

# Compute IS
fidelity --gpu 0 --isc --input1 generated_images/

# Compute both
fidelity --gpu 0 --fid --isc --input1 real_images/ --input2 generated_images/
```

### Using pytorch-fid
```bash
# Install
pip install pytorch-fid

# Compute FID
python -m pytorch_fid real_images/ generated_images/ --device cuda:0
```

### Using clean-fid
```bash
# Install
pip install clean-fid

# Compute FID with consistent preprocessing
python -m cleanfid fid real_images/ generated_images/
```

## Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| FID very high (>100) | Check sample diversity, preprocessing, training |
| IS low (<2) | Check image quality, ensure sharp predictions |
| Metrics disagree | Use multiple metrics, include visual inspection |
| Unstable metrics | Increase sample size (≥10K for FID) |
| OOM errors | Process in batches, reduce batch size |

## Metric Limitations

| Metric | Main Limitation |
|--------|----------------|
| **FID** | Assumes Gaussian features, biased by feature extractor |
| **IS** | Ignores within-class diversity, ImageNet-specific |
| **NLL** | High likelihood ≠ good samples |
| **MSE** | Poor correlation with perceptual quality |
| **Precision** | Doesn't measure quality within manifold |
| **Recall** | Can be high even with mode collapse |

## Best Practices Checklist

- [ ] Use ≥10,000 samples for FID
- [ ] Compute metrics with multiple random seeds
- [ ] Report confidence intervals (mean ± std)
- [ ] Use consistent preprocessing
- [ ] Combine multiple complementary metrics
- [ ] Include visual inspection
- [ ] Compare to established baselines
- [ ] Document exact evaluation setup
- [ ] Use standard implementations when possible
- [ ] Report negative results honestly

## Python Quick Start

```python
# Basic evaluation setup
from torch_fidelity import calculate_metrics

metrics = calculate_metrics(
    input1='path/to/real',
    input2='path/to/generated',
    cuda=True,
    isc=True,  # Inception Score
    fid=True,  # Fréchet Inception Distance
    kid=True,  # Kernel Inception Distance
    verbose=True
)

print(f"FID: {metrics['frechet_inception_distance']:.2f}")
print(f"IS: {metrics['inception_score_mean']:.2f}")
print(f"KID: {metrics['kernel_inception_distance_mean']:.4f}")
```

---

**Remember:** No single metric is perfect! Always use multiple metrics and visual inspection.
