# Training Stable Diffusion

This document explains how Stable Diffusion is trained, covering the two-stage process and compute requirements.

## Overview

In **classic Stable Diffusion**, training is basically two-stage (plus a pre-trained text encoder):

1. **Train a VAE** (encoder + decoder) on images
2. **Freeze the VAE**, then train the diffusion U-Net on the VAE's latent space, conditioned on text

## Stage 1: Train the VAE (Image ↔ Latent)

### Objective

Train a **Variational Autoencoder** on images:
- **Encoder**: image → latent tensor $z$
- **Decoder**: latent $z$ → reconstructed image

### Loss Function

$$\mathcal{L}_{\text{VAE}} = \mathcal{L}_{\text{recon}} + \beta \cdot D_{\text{KL}}(q(z|x) \| p(z))$$

In practice, Stable Diffusion uses:
- **Reconstruction loss**: L1 + perceptual (LPIPS)
- **KL regularization**: Very small weight ($\beta \approx 10^{-6}$)
- **Adversarial loss**: PatchGAN discriminator for sharpness

### Result

A VAE that:
- **Compresses** 512×512×3 images to ~64×64×4 latents (8× spatial compression)
- **Preserves** visual information for high-quality reconstruction
- Has a **regularized latent space** suitable for diffusion

### Training Details

| Aspect | Typical Value |
|--------|---------------|
| Dataset | Large-scale images (LAION, etc.) |
| Resolution | 256×256 or 512×512 |
| Batch size | 64-256 |
| Training steps | ~500K-1M |
| Optimizer | Adam, lr ~1e-4 |

## Stage 2: Train Diffusion in Latent Space

### Setup

- **VAE is frozen** (no more updates)
- **Text encoder** (CLIP) is typically frozen
- **Only U-Net is trained**

### Training Loop

For each training image:

1. **Encode**: image → latent $z_0$ with VAE encoder
2. **Add noise**: Choose timestep $t$, compute $z_t = \sqrt{\bar{\alpha}_t} z_0 + \sqrt{1-\bar{\alpha}_t} \epsilon$
3. **Condition**: Get text embedding from CLIP
4. **Predict**: U-Net predicts noise $\epsilon_\theta(z_t, t, \text{text})$
5. **Loss**: $\|\epsilon - \epsilon_\theta\|^2$

### What's Trainable?

| Component | Status |
|-----------|--------|
| VAE encoder | ❄️ Frozen |
| VAE decoder | ❄️ Frozen |
| CLIP text encoder | ❄️ Frozen (usually) |
| U-Net | 🔥 Trained |

### Training Details

| Aspect | SD v1.4/1.5 | SD v2.0 | SDXL |
|--------|-------------|---------|------|
| Base resolution | 512×512 | 768×768 | 1024×1024 |
| U-Net params | ~860M | ~865M | ~2.6B |
| Training data | LAION-2B | LAION-5B | Internal + LAION |
| Training compute | ~150K A100 hours | ~200K A100 hours | ~500K+ A100 hours |
| Batch size | 2048 | 2048 | Large |

## Pseudocode

```python
# Stage 1: VAE Training (done separately)
vae = VAE()
for images in dataset:
    z = vae.encode(images)
    x_recon = vae.decode(z)
    loss = reconstruction_loss(images, x_recon) + kl_loss(z) + adversarial_loss(x_recon)
    loss.backward()
    optimizer.step()

# Save VAE
torch.save(vae.state_dict(), "vae.pt")

# Stage 2: Latent Diffusion Training
vae = VAE().load("vae.pt").eval()  # Frozen
text_encoder = CLIP().eval()        # Frozen
unet = UNet()                       # Trainable

for images, captions in dataset:
    with torch.no_grad():
        z_0 = vae.encode(images)
        text_emb = text_encoder(captions)
    
    # Diffusion forward
    t = torch.randint(0, T, (batch_size,))
    noise = torch.randn_like(z_0)
    z_t = sqrt_alpha_bar[t] * z_0 + sqrt_one_minus_alpha_bar[t] * noise
    
    # Predict and compute loss
    noise_pred = unet(z_t, t, text_emb)
    loss = F.mse_loss(noise_pred, noise)
    
    loss.backward()
    optimizer.step()
```

## Inference Pipeline

At inference time:

1. **Text encoding**: caption → CLIP → text embedding
2. **Initialize**: Sample $z_T \sim \mathcal{N}(0, I)$ in latent space
3. **Denoise**: Run reverse diffusion with U-Net for $T$ steps
4. **Decode**: Final $z_0$ → VAE decoder → image

```python
@torch.no_grad()
def generate(prompt, num_steps=50):
    # Text encoding
    text_emb = text_encoder(prompt)
    
    # Start from noise
    z = torch.randn(1, 4, 64, 64)
    
    # Reverse diffusion
    for t in reversed(range(num_steps)):
        noise_pred = unet(z, t, text_emb)
        z = ddim_step(z, noise_pred, t)  # or DDPM step
    
    # Decode
    image = vae.decode(z)
    return image
```

## Why Two-Stage?

### Advantages

1. **Efficiency**: Diffusion in 64×64×4 vs 512×512×3 is ~48× fewer pixels
2. **Modularity**: Can improve VAE and U-Net independently
3. **Reusability**: Same VAE works for different diffusion models
4. **Proven**: Works well empirically

### Limitations

1. **Information bottleneck**: VAE compression can lose details
2. **Suboptimal**: End-to-end training might find better solutions
3. **Fixed latent space**: Can't adapt latent to diffusion needs

## Recent Developments

### End-to-End Training

Recent research explores joint VAE + diffusion training:
- **REPA-E**: Unlocks VAE for end-to-end tuning
- **Various papers**: Show benefits of unfreezing VAE during fine-tuning

### Fine-tuning

For fine-tuning existing models:
- **LoRA**: Low-rank adaptation of U-Net
- **DreamBooth**: Subject-specific fine-tuning
- **Textual Inversion**: Learn new text embeddings

## Compute Requirements

### Training from Scratch

| Model | GPU Hours | Cost (estimate) |
|-------|-----------|-----------------|
| SD v1.4 | ~150K A100 | ~$300K-500K |
| SD v2.0 | ~200K A100 | ~$400K-600K |
| SDXL | ~500K+ A100 | ~$1M+ |

### Fine-tuning

| Method | GPU Hours | Cost |
|--------|-----------|------|
| LoRA | 1-4 A100 | ~$5-20 |
| DreamBooth | 4-8 A100 | ~$20-50 |
| Full fine-tune | 100+ A100 | ~$500+ |

## Summary

Stable Diffusion training follows a two-stage approach:

1. **First**: Train VAE to compress images to a latent space
2. **Then**: Freeze VAE, train diffusion U-Net on latents with text conditioning

This design enables efficient training and high-quality generation while keeping compute requirements manageable (relatively speaking for large-scale models).

## Navigation

- **Previous**: [Stable Diffusion](stable_diffusion.md)
- **Next**: [VAE Component](vae_component.md)
