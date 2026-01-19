# Stable Diffusion Architecture

**Stable Diffusion** is a latent diffusion model that performs the diffusion process in a compressed latent space rather than pixel space, enabling efficient high-resolution image generation.

## Overview

Stable Diffusion consists of three main components:

1. **Variational Autoencoder (VAE)**: Compresses images to/from latent space
2. **U-Net Denoiser**: Performs diffusion in latent space
3. **Text Encoder (CLIP)**: Encodes text prompts for conditioning

```
Text Prompt → CLIP → Text Embeddings ─┐
                                      │
                                      ▼
Noise ──→ U-Net (denoise) ──→ Clean Latent ──→ VAE Decoder ──→ Image
              ▲
              │
        Timestep Embedding
```

## Why Latent Space?

### The Dimensionality Problem

A 512×512 RGB image has 786,432 dimensions. Diffusion in this space is:
- **Computationally expensive**: U-Net processes huge tensors
- **Memory intensive**: Batch sizes limited
- **Slow**: Many operations scale with resolution

### Compression via VAE

The VAE compresses by a factor of 8:
- **Input image**: 512×512×3 = 786,432 dimensions
- **Latent**: 64×64×4 = 16,384 dimensions
- **Compression ratio**: ~48×

This makes diffusion tractable while preserving perceptual quality.

## The VAE Component

### Architecture

```
Encoder: Image (512×512×3) → Latent (64×64×4)
Decoder: Latent (64×64×4) → Image (512×512×3)
```

### Training

The VAE is trained separately with:
1. **Reconstruction loss**: $\|x - \text{Dec}(\text{Enc}(x))\|$
2. **KL regularization**: $D_\text{KL}(\mathcal{N}(\mu, \sigma) \| \mathcal{N}(0, I))$
3. **Perceptual loss**: VGG feature matching
4. **Adversarial loss**: Discriminator for sharpness

### Key Properties

- **Fixed after training**: VAE is not updated during diffusion training
- **Perceptually lossless**: Reconstructions are visually nearly identical
- **Smooth latent space**: Good for interpolation and manipulation

## The U-Net Denoiser

### Inputs

The U-Net receives:
1. **Noisy latent** $z_t$: Shape (batch, 4, 64, 64)
2. **Timestep** $t$: Encoded via sinusoidal embedding
3. **Conditioning** $c$: Text embeddings from CLIP

### Architecture Overview

```
z_t (4×64×64)
    │
    ▼
┌─────────────────────────────────────────────┐
│              Downsampling Path              │
│  Conv → ResBlock → Attention → Downsample   │
│        (×3 at resolutions 64→32→16→8)       │
└─────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────┐
│              Middle Block                   │
│     ResBlock → Attention → ResBlock         │
└─────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────┐
│              Upsampling Path                │
│  Upsample → ResBlock → Attention → Conv     │
│   + Skip connections from encoder           │
└─────────────────────────────────────────────┘
    │
    ▼
ε_θ (4×64×64)
```

### Key Components

#### Timestep Embedding

$$
\text{emb}(t) = \text{MLP}(\text{SinusoidalPosEmb}(t))
$$

Added to ResBlock hidden states via:
$$
h = h + \text{Linear}(\text{emb}(t))
$$

#### Cross-Attention for Text Conditioning

Text embeddings $c$ are injected via cross-attention:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V
$$

where:
- $Q = W_Q \cdot h$ (queries from image features)
- $K = W_K \cdot c$ (keys from text)
- $V = W_V \cdot c$ (values from text)

#### Self-Attention

At low resolutions (8×8, 16×16), self-attention captures global context:
- Queries, keys, values all from image features
- Computationally feasible at small spatial sizes

## The Text Encoder (CLIP)

### Architecture

Stable Diffusion v1 uses CLIP ViT-L/14:
- **Input**: Text string (max 77 tokens)
- **Output**: Embeddings (77, 768)

### Tokenization

```python
from transformers import CLIPTokenizer

tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
tokens = tokenizer(
    "a cat sitting on a couch",
    padding="max_length",
    max_length=77,
    return_tensors="pt"
)
```

### Encoding

```python
from transformers import CLIPTextModel

text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")
embeddings = text_encoder(tokens.input_ids)[0]  # (batch, 77, 768)
```

## Training Pipeline

### Data Preparation

1. Load image $x$
2. Encode to latent: $z_0 = \text{Enc}(x)$
3. Sample timestep: $t \sim \text{Uniform}(1, T)$
4. Add noise: $z_t = \sqrt{\bar{\alpha}_t} z_0 + \sqrt{1-\bar{\alpha}_t} \epsilon$

### Loss Function

Standard DDPM loss in latent space:

$$
\mathcal{L} = \mathbb{E}_{z_0, t, \epsilon, c}\left[\|\epsilon - \epsilon_\theta(z_t, t, c)\|^2\right]
$$

### Training Code Sketch

```python
def training_step(vae, unet, text_encoder, images, captions):
    # Encode images to latent
    with torch.no_grad():
        latents = vae.encode(images).latent_dist.sample()
        latents = latents * 0.18215  # Scaling factor
    
    # Encode text
    with torch.no_grad():
        text_embeddings = text_encoder(captions)[0]
    
    # Sample noise and timesteps
    noise = torch.randn_like(latents)
    timesteps = torch.randint(0, T, (batch_size,))
    
    # Add noise
    noisy_latents = scheduler.add_noise(latents, noise, timesteps)
    
    # Predict noise
    noise_pred = unet(noisy_latents, timesteps, text_embeddings)
    
    # MSE loss
    loss = F.mse_loss(noise_pred, noise)
    
    return loss
```

## Inference Pipeline

### Text-to-Image Generation

```python
@torch.no_grad()
def generate(prompt, num_steps=50, guidance_scale=7.5):
    # 1. Encode text
    text_emb = text_encoder(tokenizer(prompt))
    uncond_emb = text_encoder(tokenizer(""))  # Null condition
    
    # 2. Initialize latent noise
    latents = torch.randn((1, 4, 64, 64))
    
    # 3. Denoise with classifier-free guidance
    for t in scheduler.timesteps:
        # Expand latents for CFG
        latent_input = torch.cat([latents, latents])
        text_input = torch.cat([uncond_emb, text_emb])
        
        # Predict noise
        noise_pred = unet(latent_input, t, text_input)
        noise_uncond, noise_cond = noise_pred.chunk(2)
        
        # Apply guidance
        noise_guided = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
        
        # DDIM/PNDM step
        latents = scheduler.step(noise_guided, t, latents)
    
    # 4. Decode to image
    latents = latents / 0.18215
    image = vae.decode(latents)
    
    return image
```

## Stable Diffusion Versions

| Version | VAE | Text Encoder | U-Net | Resolution |
|---------|-----|--------------|-------|------------|
| SD 1.4 | KL-8 | CLIP ViT-L/14 | 860M params | 512×512 |
| SD 1.5 | KL-8 | CLIP ViT-L/14 | 860M params | 512×512 |
| SD 2.0 | KL-8 | OpenCLIP ViT-H/14 | 865M params | 512/768 |
| SD 2.1 | KL-8 | OpenCLIP ViT-H/14 | 865M params | 512/768 |
| SDXL | KL-8 | CLIP + OpenCLIP | 2.6B params | 1024×1024 |

## Key Hyperparameters

| Parameter | Typical Value | Notes |
|-----------|---------------|-------|
| Latent channels | 4 | VAE bottleneck |
| Latent spatial | 64×64 | For 512×512 images |
| Guidance scale | 7.5 | CFG strength |
| Inference steps | 20-50 | Quality vs speed |
| VAE scale | 0.18215 | Normalization factor |

## Conditioning Mechanisms

### Text Conditioning

Primary method via cross-attention with CLIP embeddings.

### Image Conditioning (ControlNet)

Add spatial control via:
- Edge maps (Canny)
- Depth maps
- Pose skeletons
- Segmentation maps

### Inpainting

Concatenate mask and masked image to latent input:
- U-Net input: 4 + 4 + 1 = 9 channels
- Mask channel indicates regions to regenerate

## Advantages of Latent Diffusion

1. **Efficiency**: 48× fewer dimensions than pixel space
2. **Quality**: Perceptual compression preserves visual quality
3. **Flexibility**: Same framework for various tasks
4. **Scalability**: Enables high-resolution generation

## Limitations

1. **Two-stage artifacts**: VAE can introduce subtle blurring
2. **Fine details**: Small text/faces can be challenging
3. **Memory**: Still requires significant VRAM
4. **Training data**: Quality depends heavily on dataset

## Summary

Stable Diffusion combines a VAE for compression, a U-Net for denoising, and CLIP for text understanding. By operating in latent space, it achieves efficient high-resolution generation while maintaining quality. The architecture has become the foundation for many text-to-image applications and extensions.
