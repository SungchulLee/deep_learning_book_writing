# StyleGAN2

StyleGAN2 (Karras et al., 2020) addressed artifacts in StyleGAN and further improved image quality through architectural refinements.

## Addressing StyleGAN Artifacts

### Problem: Blob Artifacts

StyleGAN produced characteristic "blob" artifacts due to AdaIN normalizing feature statistics.

### Solution: Weight Demodulation

Replace AdaIN with modulation/demodulation in convolution:

```python
class ModulatedConv2d(nn.Module):
    """Modulated convolution (StyleGAN2)."""
    
    def __init__(self, in_ch, out_ch, kernel_size, w_dim, demodulate=True):
        super().__init__()
        self.out_ch = out_ch
        self.kernel_size = kernel_size
        self.demodulate = demodulate
        
        self.weight = nn.Parameter(
            torch.randn(out_ch, in_ch, kernel_size, kernel_size)
        )
        self.modulation = EqualizedLinear(w_dim, in_ch)
    
    def forward(self, x, w):
        batch_size = x.size(0)
        
        # Modulation
        style = self.modulation(w)  # [B, in_ch]
        style = style.view(batch_size, 1, -1, 1, 1)  # [B, 1, in_ch, 1, 1]
        
        # Scale weights by style
        weight = self.weight.unsqueeze(0) * style  # [B, out, in, k, k]
        
        # Demodulation
        if self.demodulate:
            demod = (weight.pow(2).sum([2, 3, 4]) + 1e-8).rsqrt()
            weight = weight * demod.view(batch_size, self.out_ch, 1, 1, 1)
        
        # Reshape for grouped convolution
        weight = weight.view(batch_size * self.out_ch, -1, 
                            self.kernel_size, self.kernel_size)
        x = x.view(1, batch_size * x.size(1), x.size(2), x.size(3))
        
        # Convolution
        out = F.conv2d(x, weight, groups=batch_size, padding=self.kernel_size//2)
        return out.view(batch_size, self.out_ch, out.size(2), out.size(3))
```

## Key Improvements

### 1. Redesigned Normalization

| StyleGAN | StyleGAN2 |
|----------|-----------|
| AdaIN (normalize → scale + shift) | Modulate → Conv → Demodulate |
| Feature statistics normalized | Weight statistics normalized |

### 2. Path Length Regularization

Encourage smooth latent space:

```python
def path_length_regularization(generator, z, mean_path_length):
    """Encourage consistent gradient magnitude in w space."""
    w = generator.mapping(z)
    fake = generator.synthesis(w)
    
    # Gradient of output w.r.t. w
    noise = torch.randn_like(fake) / math.sqrt(fake.numel() / fake.size(0))
    grad = torch.autograd.grad(
        outputs=(fake * noise).sum(), inputs=w, create_graph=True
    )[0]
    
    path_length = grad.pow(2).sum(dim=1).mean().sqrt()
    path_penalty = (path_length - mean_path_length).pow(2)
    
    return path_penalty, path_length.detach()
```

### 3. No Progressive Growing

StyleGAN2 trains at full resolution from the start using:
- Skip connections
- Residual architecture
- Proper initialization

### 4. Lazy Regularization

Apply regularization every N batches:

```python
def train_step(G, D, real, z, step, reg_interval=16):
    # Normal GAN loss every step
    fake = G(z)
    d_loss = discriminator_loss(D, real, fake)
    g_loss = generator_loss(D, fake)
    
    # Regularization every reg_interval steps
    if step % reg_interval == 0:
        d_loss += r1_regularization(D, real) * reg_interval
        g_loss += path_length_regularization(G, z) * reg_interval
    
    return g_loss, d_loss
```

## Architecture Comparison

| Component | StyleGAN | StyleGAN2 |
|-----------|----------|-----------|
| Normalization | AdaIN | Weight demod |
| Training | Progressive | Full resolution |
| Artifacts | Blob artifacts | Minimal |
| Regularization | None | Path length + R1 |

## Results

- FID of 2.84 on FFHQ (vs 4.40 for StyleGAN)
- Eliminated blob artifacts
- Improved latent space smoothness
- Better for image editing/projection

## Summary

StyleGAN2's weight demodulation and path length regularization produce cleaner images with better latent space properties, enabling superior image manipulation and projection capabilities.
