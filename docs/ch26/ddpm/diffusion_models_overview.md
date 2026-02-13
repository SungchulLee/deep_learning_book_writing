# 🎨 Diffusion Models Collection

A comprehensive educational collection of diffusion model implementations in PyTorch, from basic concepts to advanced techniques like DDIM, Latent Diffusion, and Classifier-Free Guidance.

## 📚 What's Inside

This collection provides **5 complete implementations** plus **utility scripts** to help you understand and experiment with diffusion models:

### Core Implementations

| File | Description | Complexity | Key Features |
|------|-------------|------------|--------------|
| `01_ddpm_toy.py` | Toy DDPM on MNIST | ⭐ Beginner | Simple U-Net, minimal code (~200 lines) |
| `02_ddpm.py` | Full DDPM Implementation | ⭐⭐ Intermediate | Proper U-Net with attention, cosine schedule |
| `03_conditional_ddpm.py` | Class-Conditional DDPM | ⭐⭐⭐ Advanced | Label conditioning + Classifier-Free Guidance |
| `04_ddim_fast_sampling.py` | DDIM Fast Sampling | ⭐⭐⭐ Advanced | 20x faster sampling with DDIM |
| `05_latent_diffusion.py` | Latent Diffusion Model | ⭐⭐⭐⭐ Expert | VAE + diffusion in latent space (Stable Diffusion concept) |

### Utilities

| File | Purpose |
|------|---------|
| `utils/visualize.py` | Forward/reverse process visualization, noise schedule comparison |
| `utils/interpolation.py` | Latent space interpolation, smooth transitions |

## 🚀 Quick Start

### Installation

```bash
# Clone or download this repository
# Then install dependencies:
pip install -r requirements.txt
```

### Run Your First Model

```bash
# Start with the toy example (fastest, ~5 minutes on CPU)
python 01_ddpm_toy.py

# Or try the full DDPM (better quality, needs GPU)
python 02_ddpm.py

# For conditional generation (generate specific digits/objects)
python 03_conditional_ddpm.py

# For fast sampling with DDIM
python 04_ddim_fast_sampling.py

# For latent diffusion (Stable Diffusion concept)
python 05_latent_diffusion.py
```

### Visualization

```bash
# See how noise is added to images
python utils/visualize.py

# After training, visualize interpolations
python -c "from utils.interpolation import *; # see file for examples"
```

## 📖 Learning Path

### 1️⃣ Start Here: Toy DDPM (`01_ddpm_toy.py`)
- **What you'll learn**: Core diffusion concepts
- **Time**: 5-10 minutes training
- **Output**: Generated MNIST digits
- **Key concepts**: 
  - Forward diffusion (adding noise)
  - Reverse diffusion (denoising)
  - U-Net noise prediction

### 2️⃣ Full Implementation: DDPM (`02_ddpm.py`)
- **What you'll learn**: Production-quality implementation
- **Time**: 30-60 minutes training
- **Output**: High-quality CIFAR-10/MNIST samples
- **Key concepts**:
  - Proper U-Net architecture with residual blocks
  - Self-attention mechanisms
  - Cosine beta schedule
  - Sinusoidal time embeddings

### 3️⃣ Conditional Generation (`03_conditional_ddpm.py`)
- **What you'll learn**: Controlled generation
- **Time**: 30-60 minutes training
- **Output**: Class-specific samples (e.g., "generate a cat")
- **Key concepts**:
  - Label conditioning
  - Classifier-Free Guidance (CFG)
  - Guidance scale tuning

### 4️⃣ Fast Sampling: DDIM (`04_ddim_fast_sampling.py`)
- **What you'll learn**: Efficient sampling
- **Time**: Same training, 20x faster sampling!
- **Output**: Same quality in 50 steps vs 1000
- **Key concepts**:
  - Deterministic sampling
  - Timestep subsampling
  - Eta parameter (stochasticity control)

### 5️⃣ Advanced: Latent Diffusion (`05_latent_diffusion.py`)
- **What you'll learn**: Foundation of Stable Diffusion
- **Time**: 1-2 hours total training
- **Output**: Generated samples via compressed latents
- **Key concepts**:
  - VAE for compression
  - Diffusion in latent space
  - Memory efficiency
  - Why Stable Diffusion is "stable"

## 🔬 Key Concepts Explained

### What is Diffusion?

Diffusion models learn to generate data by **reversing a gradual noising process**:

```
Forward Process (Training):
Clean Image → [Add noise] → Slightly Noisy → [Add more noise] → Very Noisy → Pure Noise

Reverse Process (Sampling):
Pure Noise → [Denoise] → Slightly Less Noisy → [Denoise] → Clean Image
```

### How Training Works

1. **Take a clean image** from your dataset
2. **Pick a random timestep** (e.g., t=500 out of 1000)
3. **Add the corresponding amount of noise** to the image
4. **Train a neural network** to predict the noise
5. **Repeat** millions of times

### How Sampling Works

1. **Start with pure random noise**
2. **Use the trained network** to predict the noise
3. **Remove a bit of the predicted noise**
4. **Repeat** for all timesteps (1000 → 0)
5. **Result**: A clean generated image!

## 🎯 Advanced Techniques

### Classifier-Free Guidance (CFG)

Makes conditional generation stronger:
```python
# Without CFG: Just condition on label
noise_pred = model(x, t, class_label)

# With CFG: Mix conditional and unconditional
noise_uncond = model(x, t, null_label)  
noise_cond = model(x, t, class_label)
noise_pred = noise_uncond + scale * (noise_cond - noise_uncond)  # scale > 1
```

**Effect**: Higher guidance scale = more accurate to the condition but less diverse

### DDIM Sampling

Instead of denoising every timestep (1000 steps), skip most of them:
```python
# DDPM: Use all 1000 steps
for t in [999, 998, 997, ..., 2, 1, 0]:
    x = denoise(x, t)

# DDIM: Use only 50 steps
for t in [999, 979, 959, ..., 39, 19, 0]:  # every 20th step
    x = denoise(x, t)  # special formula
```

**Result**: 20x faster sampling with similar quality!

### Latent Diffusion

Run diffusion on compressed representations:
```python
# Pixel Diffusion (e.g., DALL-E 2):
image (3×512×512) → add noise → denoise → image
# Memory: Very high!

# Latent Diffusion (e.g., Stable Diffusion):
image (3×512×512) → VAE encode → latent (4×64×64) → add noise → denoise → VAE decode → image
# Memory: 64x less!
```

**Result**: Much faster training and sampling, enables high-resolution generation

## 🛠️ Customization Guide

### Change Dataset

Edit the `DATASET` variable in any file:
```python
DATASET = "MNIST"     # 28x28 grayscale digits
DATASET = "CIFAR10"   # 32x32 color images
```

### Adjust Model Size

```python
BASE_CH = 32   # Smaller, faster (for testing)
BASE_CH = 64   # Default
BASE_CH = 128  # Larger, better quality (slower)
```

### Training Duration

```python
EPOCHS = 3     # Quick test
EPOCHS = 10    # Decent results
EPOCHS = 50    # High quality (needs time!)
```

### Sampling Speed

```python
# DDPM: Full quality
samples = ddpm.sample(n=16)

# DDIM: Faster
samples = ddim.ddim_sample(n=16, ddim_steps=50)  # 20x faster
```

## 📊 Performance Comparison

| Method | Training Time | Sampling Time | Quality | Memory |
|--------|--------------|---------------|---------|--------|
| DDPM | 1x | 1x (slow) | ⭐⭐⭐⭐ | High |
| DDIM | 1x | 0.05x (20x faster!) | ⭐⭐⭐⭐ | High |
| Latent Diffusion | 1.5x | 0.05x | ⭐⭐⭐⭐⭐ | Low |
| Conditional + CFG | 1x | 2x (two forward passes) | ⭐⭐⭐⭐⭐ | High |

## 🎓 Research Papers

This collection implements ideas from:

1. **DDPM**: [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) (Ho et al., 2020)
2. **Improved DDPM**: [Improved Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2102.09672) (Nichol & Dhariwal, 2021)
3. **DDIM**: [Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502) (Song et al., 2020)
4. **Classifier-Free Guidance**: [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598) (Ho & Salimans, 2022)
5. **Latent Diffusion**: [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) (Rombach et al., 2022) - *Stable Diffusion*

## 🏗️ Architecture Overview

### Simple U-Net (Toy)
```
Input Image → Conv → Conv → Conv → Output Noise Prediction
   (28×28)     (64)   (64)   (28)        (28×28)
```

### Full U-Net (DDPM)
```
Input (32×32×3)
    ↓
[Encoder with ResBlocks + Attention]
    ↓ (downsample)
    ↓ (16×16)
    ↓ (downsample)
    ↓ (8×8)
[Middle: ResBlocks + Attention]
    ↑ (upsample)
    ↑ (16×16)
    ↑ (upsample)
[Decoder with ResBlocks + Attention]
    ↓
Output (32×32×3)
```

### Latent Diffusion
```
Image → [VAE Encoder] → Latent → [U-Net Diffusion] → Denoised Latent → [VAE Decoder] → Image
(512²)                  (64²)                          (64²)                             (512²)
```

## 💡 Tips & Tricks

### Training
- **Start small**: Try toy implementation first
- **Use GPU**: Diffusion models are slow on CPU
- **Monitor loss**: Should decrease steadily
- **Save checkpoints**: Training takes time!
- **Increase epochs**: More training = better quality

### Sampling
- **Use DDIM**: Much faster than DDPM
- **Adjust eta**: 0=deterministic, 1=stochastic
- **Try different seeds**: Each gives different results
- **Batch sampling**: Generate multiple at once

### Quality
- **Increase model size**: More channels = better quality
- **Use attention**: Helps with fine details
- **Cosine schedule**: Better than linear
- **Longer training**: Most important factor!
- **Use CFG**: For conditional models, guidance_scale=3-7

## 🐛 Troubleshooting

### Problem: NaN Loss
- **Solution**: Reduce learning rate or use gradient clipping
- Check: `torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)`

### Problem: Blurry Samples
- **Solution**: Train longer, increase model size, or use attention
- Try: More epochs (50+) with larger BASE_CH (128+)

### Problem: Slow Training
- **Solution**: Reduce batch size, use smaller model, or use latent diffusion
- Quick test: `BASE_CH=32, BATCH_SIZE=32`

### Problem: Out of Memory
- **Solution**: Reduce batch size or model size
- Try: `BATCH_SIZE=32, BASE_CH=32`

### Problem: Generated samples all look the same
- **Solution**: Train longer or reduce guidance scale (if using CFG)

## 🌟 Next Steps

After mastering these implementations, explore:

1. **Text Conditioning**: Add CLIP or T5 for text-to-image
2. **Super-Resolution**: Cascade models for high-res images
3. **Inpainting**: Condition on masked images
4. **ControlNet**: Add spatial conditioning (edges, depth)
5. **Stable Diffusion**: Full implementation with text encoder
6. **Video Diffusion**: Extend to temporal dimension

## 📁 Project Structure

```
diffusion_models_collection/
├── 01_ddpm_toy.py              # Beginner-friendly toy implementation
├── 02_ddpm.py                  # Full DDPM with proper U-Net
├── 03_conditional_ddpm.py      # Class-conditional with CFG
├── 04_ddim_fast_sampling.py    # DDIM for fast sampling
├── 05_latent_diffusion.py      # Latent space diffusion (Stable Diffusion concept)
├── utils/
│   ├── visualize.py            # Visualization tools
│   └── interpolation.py        # Latent space exploration
├── requirements.txt            # Python dependencies
└── README.md                   # This file!
```

## 📝 License

This is an educational collection. Use it to learn, experiment, and build upon!

For production use, please cite the original papers.

## 🙏 Acknowledgments

This collection is inspired by and builds upon:
- The original DDPM paper by Ho et al.
- Hugging Face's diffusers library
- Phil Wang's implementations
- Various educational resources in the community

## 📧 Questions?

These implementations are designed to be educational and easy to understand. Each file is heavily commented - read the code!

If something is unclear:
1. Read the comments in the code
2. Check the paper references
3. Experiment with the parameters
4. Try the visualization tools

Happy diffusing! 🎨✨
