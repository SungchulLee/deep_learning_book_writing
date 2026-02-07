# VQ-VAE: Vector Quantized Variational Autoencoder

Discrete latent representations through vector quantization.

---

## Learning Objectives

By the end of this section, you will be able to:

- Explain the motivation for discrete latent spaces
- Understand vector quantization and codebook learning
- Implement VQ-VAE from scratch
- Apply VQ-VAE for high-fidelity image generation

---

## Motivation: Why Discrete Latents?

### Limitations of Continuous Latents

Standard VAEs use continuous Gaussian latent spaces:
- Reconstruction often blurry due to averaging
- KL divergence can lead to posterior collapse
- Difficult to use with autoregressive priors

### Benefits of Discrete Latents

| Advantage | Explanation |
|-----------|-------------|
| **No posterior collapse** | No KL term to minimize |
| **Sharp reconstructions** | Discrete codes avoid averaging |
| **Composable** | Can use powerful autoregressive priors |
| **Efficient** | Compact codebook representation |

---

## VQ-VAE Architecture

### Overview

```
Input x → Encoder → z_e → Quantization → z_q → Decoder → x̂
                           ↓
                     Codebook e
                   (K embedding vectors)
```

### Components

1. **Encoder:** Maps input to continuous latent $z_e$
2. **Codebook:** $K$ learnable embedding vectors $e_1, ..., e_K$
3. **Quantization:** Replace $z_e$ with nearest codebook entry
4. **Decoder:** Reconstruct from quantized code $z_q$

---

## Vector Quantization

### The Quantization Operation

For encoder output $z_e$, find nearest codebook vector:

$$z_q = e_k \quad \text{where} \quad k = \arg\min_j \|z_e - e_j\|^2$$

### Handling Non-Differentiability

The $\arg\min$ operation has zero gradient everywhere. Solution: **straight-through estimator**.

**Forward pass:** Use quantized $z_q$
**Backward pass:** Copy gradients from $z_q$ to $z_e$

```python
# Straight-through estimator
z_q = z_e + (z_q - z_e).detach()  # Forward: z_q, Backward: gradient flows to z_e
```

---

## VQ-VAE Loss Function

### Three Components

$$\mathcal{L} = \underbrace{\|x - \hat{x}\|^2}_{\text{Reconstruction}} + \underbrace{\|sg[z_e] - e\|^2}_{\text{Codebook Loss}} + \underbrace{\beta \|z_e - sg[e]\|^2}_{\text{Commitment Loss}}$$

where $sg[\cdot]$ is the stop-gradient operator.

### Loss Breakdown

| Term | Formula | Purpose |
|------|---------|---------|
| **Reconstruction** | $\|x - \hat{x}\|^2$ | Match input |
| **Codebook** | $\|sg[z_e] - e\|^2$ | Move codebook toward encoder outputs |
| **Commitment** | $\beta\|z_e - sg[e]\|^2$ | Keep encoder outputs near codebook |

---

## Implementation

### Vector Quantizer Module

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class VectorQuantizer(nn.Module):
    """
    Vector Quantization layer for VQ-VAE.
    
    Args:
        num_embeddings: Size of codebook (K)
        embedding_dim: Dimension of each embedding vector
        commitment_cost: Weight for commitment loss (β)
    """
    
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super().__init__()
        
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        
        # Codebook: K embedding vectors of dimension D
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)
    
    def forward(self, z_e):
        """
        Quantize encoder output.
        
        Args:
            z_e: Encoder output [B, D, H, W] or [B, D]
        
        Returns:
            z_q: Quantized output
            loss: VQ loss (codebook + commitment)
            encoding_indices: Selected codebook indices
        """
        # Flatten spatial dimensions if present
        if z_e.dim() == 4:
            B, D, H, W = z_e.shape
            z_e_flat = z_e.permute(0, 2, 3, 1).contiguous()  # [B, H, W, D]
            z_e_flat = z_e_flat.view(-1, D)  # [B*H*W, D]
        else:
            z_e_flat = z_e
            B, D = z_e.shape
            H, W = 1, 1
        
        # Compute distances to all codebook vectors
        # ||z_e - e||^2 = ||z_e||^2 + ||e||^2 - 2*z_e·e
        d = (z_e_flat.pow(2).sum(dim=1, keepdim=True) 
             + self.embedding.weight.pow(2).sum(dim=1)
             - 2 * torch.matmul(z_e_flat, self.embedding.weight.t()))
        
        # Find nearest codebook entry
        encoding_indices = torch.argmin(d, dim=1)
        
        # Quantize
        z_q_flat = self.embedding(encoding_indices)
        
        # Reshape to original spatial dimensions
        if z_e.dim() == 4:
            z_q = z_q_flat.view(B, H, W, D).permute(0, 3, 1, 2).contiguous()
            z_e_for_loss = z_e.permute(0, 2, 3, 1).contiguous().view(-1, D)
        else:
            z_q = z_q_flat
            z_e_for_loss = z_e_flat
        
        # Compute losses
        codebook_loss = F.mse_loss(z_q_flat.detach(), z_e_for_loss)
        commitment_loss = F.mse_loss(z_q_flat, z_e_for_loss.detach())
        
        vq_loss = codebook_loss + self.commitment_cost * commitment_loss
        
        # Straight-through estimator
        z_q = z_e + (z_q - z_e).detach()
        
        return z_q, vq_loss, encoding_indices.view(B, H, W) if z_e.dim() == 4 else encoding_indices
```

### VQ-VAE Model

```python
class VQVAE(nn.Module):
    """
    Vector Quantized VAE for images.
    
    Args:
        in_channels: Input image channels
        hidden_dim: Hidden dimension
        num_embeddings: Codebook size
        embedding_dim: Dimension of codebook vectors
        commitment_cost: β for commitment loss
    """
    
    def __init__(self, in_channels=1, hidden_dim=128, 
                 num_embeddings=512, embedding_dim=64, commitment_cost=0.25):
        super().__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, embedding_dim, 3, padding=1),
        )
        
        # Vector Quantizer
        self.vq = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(embedding_dim, hidden_dim, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_dim, hidden_dim, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_dim, in_channels, 4, stride=2, padding=1),
            nn.Sigmoid()
        )
    
    def encode(self, x):
        """Encode to continuous latent."""
        return self.encoder(x)
    
    def quantize(self, z_e):
        """Quantize continuous latent."""
        return self.vq(z_e)
    
    def decode(self, z_q):
        """Decode from quantized latent."""
        return self.decoder(z_q)
    
    def forward(self, x):
        """
        Full forward pass.
        
        Returns:
            recon: Reconstructed image
            vq_loss: Vector quantization loss
            indices: Codebook indices used
        """
        z_e = self.encode(x)
        z_q, vq_loss, indices = self.quantize(z_e)
        recon = self.decode(z_q)
        return recon, vq_loss, indices
```

---

## Training VQ-VAE

### Loss Function

```python
def vqvae_loss(recon_x, x, vq_loss):
    """
    VQ-VAE total loss.
    
    Args:
        recon_x: Reconstructed image
        x: Original image
        vq_loss: VQ loss from quantizer
    
    Returns:
        total_loss, recon_loss, vq_loss
    """
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    total_loss = recon_loss + vq_loss
    return total_loss, recon_loss, vq_loss
```

### Training Loop

```python
def train_vqvae(model, dataloader, optimizer, device, epochs=20):
    """Train VQ-VAE."""
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        total_recon = 0
        total_vq = 0
        
        for batch_idx, (data, _) in enumerate(dataloader):
            data = data.to(device)
            
            optimizer.zero_grad()
            
            recon, vq_loss, _ = model(data)
            loss, recon_loss, _ = vqvae_loss(recon, data, vq_loss)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_vq += vq_loss.item()
        
        n = len(dataloader.dataset)
        print(f"Epoch {epoch+1}: Loss={total_loss/n:.4f}, "
              f"Recon={total_recon/n:.4f}, VQ={total_vq/n:.4f}")
```

---

## Exponential Moving Average (EMA) Updates

### Alternative to Codebook Loss

Instead of gradient-based codebook updates, use EMA:

```python
class VectorQuantizerEMA(nn.Module):
    """VQ with EMA codebook updates (more stable)."""
    
    def __init__(self, num_embeddings, embedding_dim, 
                 commitment_cost=0.25, decay=0.99, epsilon=1e-5):
        super().__init__()
        
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon
        
        # Codebook
        self.register_buffer('embedding', torch.randn(num_embeddings, embedding_dim))
        self.register_buffer('cluster_size', torch.zeros(num_embeddings))
        self.register_buffer('ema_w', torch.randn(num_embeddings, embedding_dim))
    
    def forward(self, z_e):
        # ... (same distance computation and quantization)
        
        if self.training:
            # EMA update
            encodings_one_hot = F.one_hot(encoding_indices, self.num_embeddings).float()
            
            self.cluster_size.data.mul_(self.decay).add_(
                encodings_one_hot.sum(0), alpha=1 - self.decay)
            
            dw = torch.matmul(encodings_one_hot.t(), z_e_flat)
            self.ema_w.data.mul_(self.decay).add_(dw, alpha=1 - self.decay)
            
            n = self.cluster_size.sum()
            cluster_size = (self.cluster_size + self.epsilon) / (n + self.num_embeddings * self.epsilon) * n
            
            self.embedding.data.copy_(self.ema_w / cluster_size.unsqueeze(1))
        
        # Only commitment loss (no codebook loss with EMA)
        commitment_loss = self.commitment_cost * F.mse_loss(z_q_flat, z_e_for_loss.detach())
        
        return z_q, commitment_loss, encoding_indices
```

---

## VQ-VAE-2

### Hierarchical Quantization

VQ-VAE-2 uses multiple levels of quantization:

```
Input → Encoder_bottom → VQ_bottom → Decoder_bottom
                ↓
        Encoder_top → VQ_top → Decoder_top
```

**Benefits:**
- Bottom level: Local details (textures)
- Top level: Global structure (shape, layout)
- Combined: High-fidelity reconstruction

---

## Comparison with Standard VAE

| Aspect | VAE | VQ-VAE |
|--------|-----|--------|
| **Latent space** | Continuous Gaussian | Discrete codebook |
| **Regularization** | KL divergence | Commitment loss |
| **Reconstruction** | Often blurry | Sharp |
| **Prior** | Fixed $\mathcal{N}(0,I)$ | Learnable (autoregressive) |
| **Generation** | Sample $z \sim p(z)$ | Sample codes autoregressively |

---

## Applications

### Image Generation Pipeline

1. **Train VQ-VAE:** Learn codebook and encoder/decoder
2. **Train PixelCNN prior:** Learn $p(z)$ over discrete codes
3. **Generate:** Sample codes from prior, decode to images

### Other Applications

- **Audio synthesis:** WaveNet with VQ-VAE
- **Video prediction:** Temporal VQ-VAE
- **Image compression:** Efficient discrete codes

---

## Summary

| Concept | Key Point |
|---------|-----------|
| **Vector Quantization** | Map continuous to discrete via nearest neighbor |
| **Codebook** | $K$ learnable embedding vectors |
| **Straight-through** | Copy gradients past quantization |
| **Loss** | Reconstruction + codebook + commitment |
| **Advantage** | Sharp reconstructions, no posterior collapse |

---

## Exercises

### Exercise 1: Codebook Size

Train VQ-VAE with $K \in \{64, 256, 512, 1024\}$. Compare:
- Reconstruction quality
- Codebook utilization
- Training stability

### Exercise 2: Codebook Analysis

Visualize the learned codebook:
- Decode each codebook vector individually
- Plot codebook utilization histogram
- Identify "dead" codes (never used)

### Exercise 3: Prior Learning

After training VQ-VAE, train a PixelCNN to model $p(z)$ over the discrete codes. Generate new images by sampling.

---

## What's Next

The next section covers [Conditional VAE (CVAE)](cvae.md) for controlled generation by conditioning on labels.
