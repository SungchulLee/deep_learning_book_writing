# CycleGAN

CycleGAN (Zhu et al., 2017) enables unpaired image-to-image translation using cycle consistency loss.

## Problem: Unpaired Translation

Unlike Pix2Pix (paired data), CycleGAN works with unpaired datasets:
- Horses ↔ Zebras
- Summer ↔ Winter
- Photos ↔ Monet paintings

## Architecture

### Two Generators, Two Discriminators

```
Domain X (e.g., horses)     Domain Y (e.g., zebras)
    x ----G_XY----> ŷ
    x <---G_YX---- ŷ (cycle)
    
    y ----G_YX----> x̂
    y <---G_XY---- x̂ (cycle)
```

### Cycle Consistency Loss

Key insight: Translation should be reversible.

$$\mathcal{L}_{cyc}(G_{XY}, G_{YX}) = \mathbb{E}_x[\|G_{YX}(G_{XY}(x)) - x\|_1] + \mathbb{E}_y[\|G_{XY}(G_{YX}(y)) - y\|_1]$$

```python
def cycle_consistency_loss(G_XY, G_YX, real_X, real_Y):
    # X -> Y -> X
    fake_Y = G_XY(real_X)
    cycled_X = G_YX(fake_Y)
    loss_cycle_X = F.l1_loss(cycled_X, real_X)
    
    # Y -> X -> Y
    fake_X = G_YX(real_Y)
    cycled_Y = G_XY(fake_X)
    loss_cycle_Y = F.l1_loss(cycled_Y, real_Y)
    
    return loss_cycle_X + loss_cycle_Y
```

### Identity Loss (Optional)

Preserve color when already in target domain:

```python
def identity_loss(G_XY, G_YX, real_X, real_Y):
    # G_XY should be identity on Y
    same_Y = G_XY(real_Y)
    loss_identity_Y = F.l1_loss(same_Y, real_Y)
    
    # G_YX should be identity on X
    same_X = G_YX(real_X)
    loss_identity_X = F.l1_loss(same_X, real_X)
    
    return loss_identity_X + loss_identity_Y
```

## Full Objective

$$\mathcal{L} = \mathcal{L}_{GAN}(G_{XY}, D_Y) + \mathcal{L}_{GAN}(G_{YX}, D_X) + \lambda_{cyc}\mathcal{L}_{cyc} + \lambda_{id}\mathcal{L}_{id}$$

```python
def cyclegan_loss(G_XY, G_YX, D_X, D_Y, real_X, real_Y, lambda_cyc=10, lambda_id=5):
    # GAN losses
    fake_Y = G_XY(real_X)
    loss_GAN_XY = mse_loss(D_Y(fake_Y), ones)
    
    fake_X = G_YX(real_Y)
    loss_GAN_YX = mse_loss(D_X(fake_X), ones)
    
    # Cycle loss
    loss_cycle = cycle_consistency_loss(G_XY, G_YX, real_X, real_Y)
    
    # Identity loss
    loss_identity = identity_loss(G_XY, G_YX, real_X, real_Y)
    
    # Total generator loss
    G_loss = loss_GAN_XY + loss_GAN_YX + lambda_cyc * loss_cycle + lambda_id * loss_identity
    
    return G_loss
```

## Generator Architecture

ResNet-based generator (9 residual blocks for 256×256):

```python
class ResNetGenerator(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, ngf=64, n_blocks=9):
        super().__init__()
        
        # Initial convolution
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_ch, ngf, 7),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(True)
        ]
        
        # Downsampling
        for i in range(2):
            mult = 2 ** i
            model += [
                nn.Conv2d(ngf*mult, ngf*mult*2, 3, 2, 1),
                nn.InstanceNorm2d(ngf*mult*2),
                nn.ReLU(True)
            ]
        
        # Residual blocks
        mult = 2 ** 2
        for i in range(n_blocks):
            model += [ResidualBlock(ngf * mult)]
        
        # Upsampling
        for i in range(2):
            mult = 2 ** (2 - i)
            model += [
                nn.ConvTranspose2d(ngf*mult, ngf*mult//2, 3, 2, 1, 1),
                nn.InstanceNorm2d(ngf*mult//2),
                nn.ReLU(True)
            ]
        
        # Output
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, out_ch, 7),
            nn.Tanh()
        ]
        
        self.model = nn.Sequential(*model)
```

## Training Tips

1. **Use LSGAN loss** (MSE instead of BCE)
2. **Image buffer** to stabilize D training
3. **Instance normalization** (not batch)
4. **Learning rate decay** after 100 epochs

## Summary

| Component | Value |
|-----------|-------|
| Generators | 2 (G_XY, G_YX) |
| Discriminators | 2 (D_X, D_Y) |
| λ_cycle | 10 |
| λ_identity | 5 |
| Normalization | Instance Norm |
