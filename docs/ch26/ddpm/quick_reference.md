# 📊 Quick Reference: Diffusion Models Comparison

## At-a-Glance Comparison

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        DIFFUSION MODELS QUICK REFERENCE                      │
└─────────────────────────────────────────────────────────────────────────────┘

File: 01_ddpm_toy.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Technique:    Toy DDPM (Educational)
Complexity:   ⭐ Beginner
Training:     5-10 minutes
Sampling:     Slow (1000 steps)
Quality:      Basic
Best For:     Learning the basics
Key Feature:  Simple, easy to understand
Code Size:    ~200 lines
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

File: 02_ddpm.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Technique:    Full DDPM (Ho et al., 2020)
Complexity:   ⭐⭐ Intermediate
Training:     30-60 minutes
Sampling:     Slow (1000 steps)
Quality:      Good
Best For:     Understanding proper architecture
Key Feature:  Production-quality U-Net, cosine schedule
Code Size:    ~400 lines
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

File: 03_conditional_ddpm.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Technique:    Conditional DDPM + CFG
Complexity:   ⭐⭐⭐ Advanced
Training:     30-60 minutes
Sampling:     Medium (1000 steps, 2 forward passes)
Quality:      Excellent (with CFG)
Best For:     Controlled generation (specific classes)
Key Feature:  Classifier-Free Guidance
Code Size:    ~430 lines
Notes:        Can specify what to generate!
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

File: 04_ddim_fast_sampling.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Technique:    DDIM (Song et al., 2020)
Complexity:   ⭐⭐⭐ Advanced
Training:     30-60 minutes (same as DDPM)
Sampling:     Fast (50 steps) ⚡
Quality:      Excellent
Best For:     Fast generation, deterministic samples
Key Feature:  20x faster sampling!
Code Size:    ~500 lines
Notes:        Same training, much faster inference
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

File: 05_latent_diffusion.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Technique:    Latent Diffusion (Rombach et al., 2022)
Complexity:   ⭐⭐⭐⭐ Expert
Training:     1-2 hours (2 stages: VAE + Diffusion)
Sampling:     Fast (operates on compressed latents)
Quality:      Outstanding
Best For:     High-res generation, memory efficiency
Key Feature:  Core concept of Stable Diffusion
Code Size:    ~550 lines
Notes:        64x less memory than pixel diffusion!
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

## 🎯 Which One Should You Use?

### Just Learning? → `01_ddpm_toy.py`
- Simplest implementation
- Run it first to understand basics
- Takes only 5-10 minutes

### Want Quality? → `02_ddpm.py`
- Proper architecture
- Good results
- Foundation for everything else

### Need Control? → `03_conditional_ddpm.py`
- Generate specific classes
- Classifier-Free Guidance
- Better quality with CFG

### Need Speed? → `04_ddim_fast_sampling.py`
- 20x faster than DDPM
- Same quality
- Deterministic if you want

### Want Efficiency? → `05_latent_diffusion.py`
- Much less memory
- Scales to high resolution
- Stable Diffusion concept

## 📈 Speed Comparison (Sampling)

```
DDPM:             ████████████████████  1000 steps  (baseline)
DDIM:             █                      50 steps   (20x faster!)
Latent Diffusion: ██                    100 steps   (10x faster + efficient)
```

## 💾 Memory Comparison

```
DDPM (pixel):        ████████████████████  100%  (3×32×32 = 3072 values)
Latent Diffusion:    ███                    16%  (4×8×8 = 256 values)
                                                   ↑ 64x compression!
```

## 🎨 Quality vs Speed Tradeoff

```
Quality
  ↑
  │  ⑤ Latent
  │  ④ DDIM
  │  ③ Conditional
  │  ② Full DDPM
  │  ① Toy DDPM
  └──────────────────→ Speed
```

## 🔑 Key Techniques Used

| Technique | Files Using It | Purpose |
|-----------|----------------|---------|
| Noise Prediction | All | Core idea of diffusion |
| U-Net Architecture | 2,3,4,5 | Denoise at different scales |
| Residual Blocks | 2,3,4,5 | Better gradient flow |
| Self-Attention | 2,3,4,5 | Capture long-range dependencies |
| Time Embeddings | 2,3,4,5 | Tell model which timestep |
| Cosine Schedule | 2,3,4,5 | Better noise distribution |
| Label Conditioning | 3 | Control generation |
| CFG | 3 | Strengthen conditioning |
| Timestep Skipping | 4 | Fast sampling |
| VAE Compression | 5 | Memory efficiency |

## 💡 When to Combine Techniques

### Best Overall Setup
```python
# Combine the best of everything:
- Latent Diffusion (memory efficient)
+ DDIM Sampling (fast)
+ Conditional (controllable)
+ CFG (high quality)
= Stable Diffusion! 🎯
```

### For Learning
```python
01_ddpm_toy.py  → Understand basics
↓
02_ddpm.py      → Proper architecture
↓
03_conditional  → Add control
↓
04_ddim        → Make it fast
↓
05_latent      → Make it efficient
```

### For Production
```python
Start with: 05_latent_diffusion.py
Add: DDIM sampling (50 steps)
Add: Conditional generation
Add: CFG (scale=7.5)
Result: Production-ready system
```

## 🚀 Performance Tips

### For Faster Training
- Use GPU
- Increase batch size
- Use latent diffusion
- Reduce model size (BASE_CH=32)

### For Better Quality
- Train longer (50+ epochs)
- Increase model size (BASE_CH=128)
- Use attention at multiple scales
- Use cosine schedule
- Try CFG with conditional model

### For Faster Sampling
- Use DDIM (50 steps)
- Use latent diffusion
- Reduce number of attention layers
- Use smaller model

### For Less Memory
- Reduce batch size
- Use latent diffusion
- Use gradient checkpointing
- Reduce model size

## 🎯 Cheat Sheet

```bash
# Quickest start (5 min)
python 01_ddpm_toy.py

# Best quality (1 hour)
python 02_ddpm.py  # then increase EPOCHS=50

# Controlled generation
python 03_conditional_ddpm.py  # edit cfg_scale

# Fastest sampling
python 04_ddim_fast_sampling.py  # change ddim_steps

# Most efficient
python 05_latent_diffusion.py

# Compare methods
python examples/compare_methods.py
```

## 📚 Quick Command Reference

```python
# Change dataset
DATASET = "MNIST"      # or "CIFAR10"

# Adjust speed/quality
EPOCHS = 10            # More = better quality
BASE_CH = 64           # Bigger = better but slower
BATCH_SIZE = 128       # Bigger = faster training

# Sampling options
T = 1000               # Total training steps
DDIM_STEPS = 50        # Sampling steps (DDIM)
CFG_SCALE = 3.0        # Guidance strength
```

---

**Keep this reference handy while exploring! 🚀**
