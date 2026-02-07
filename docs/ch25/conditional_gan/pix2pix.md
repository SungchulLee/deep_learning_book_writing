# Pix2Pix

Pix2Pix (Isola et al., 2017) is a conditional GAN for paired image-to-image translation, transforming images from one domain to another.

## Problem Formulation

Given paired images $(x, y)$ where $x$ is input and $y$ is target, learn mapping $G: x \to y$.

**Examples:**
- Edges → Photo
- Semantic labels → Street scene
- Black & white → Color
- Day → Night

## Architecture

### Generator: U-Net

Encoder-decoder with skip connections:

```python
class UNetGenerator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, features=64):
        super().__init__()
        
        # Encoder
        self.enc1 = self.encoder_block(in_channels, features, bn=False)
        self.enc2 = self.encoder_block(features, features*2)
        self.enc3 = self.encoder_block(features*2, features*4)
        self.enc4 = self.encoder_block(features*4, features*8)
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features*8, features*8, 4, 2, 1),
            nn.ReLU()
        )
        
        # Decoder with skip connections
        self.dec1 = self.decoder_block(features*8, features*8, dropout=True)
        self.dec2 = self.decoder_block(features*16, features*4)  # *16 due to skip
        self.dec3 = self.decoder_block(features*8, features*2)
        self.dec4 = self.decoder_block(features*4, features)
        
        self.final = nn.Sequential(
            nn.ConvTranspose2d(features*2, out_channels, 4, 2, 1),
            nn.Tanh()
        )
    
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        
        # Bottleneck
        b = self.bottleneck(e4)
        
        # Decoder with skip connections
        d1 = self.dec1(b)
        d1 = torch.cat([d1, e4], dim=1)
        d2 = self.dec2(d1)
        d2 = torch.cat([d2, e3], dim=1)
        d3 = self.dec3(d2)
        d3 = torch.cat([d3, e2], dim=1)
        d4 = self.dec4(d3)
        d4 = torch.cat([d4, e1], dim=1)
        
        return self.final(d4)
```

### Discriminator: PatchGAN

Classify N×N patches instead of whole image:

```python
class PatchDiscriminator(nn.Module):
    """70x70 PatchGAN discriminator."""
    
    def __init__(self, in_channels=6):  # input + target concatenated
        super().__init__()
        
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(256, 512, 4, 1, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(512, 1, 4, 1, 1)  # Output: (N, 1, H', W')
        )
    
    def forward(self, x, y):
        return self.model(torch.cat([x, y], dim=1))
```

## Loss Function

$$\mathcal{L} = \mathcal{L}_{cGAN}(G, D) + \lambda \mathcal{L}_{L1}(G)$$

```python
def pix2pix_loss(G, D, real_x, real_y, lambda_l1=100):
    # GAN loss
    fake_y = G(real_x)
    
    # D loss
    real_pred = D(real_x, real_y)
    fake_pred = D(real_x, fake_y.detach())
    d_loss = (bce(real_pred, ones) + bce(fake_pred, zeros)) / 2
    
    # G loss = GAN + L1
    fake_pred = D(real_x, fake_y)
    g_gan = bce(fake_pred, ones)
    g_l1 = F.l1_loss(fake_y, real_y)
    g_loss = g_gan + lambda_l1 * g_l1
    
    return g_loss, d_loss
```

## Why L1 Loss?

- Encourages low-frequency correctness
- GAN handles high-frequency detail
- Together: sharp AND accurate

## Summary

| Component | Choice |
|-----------|--------|
| Generator | U-Net |
| Discriminator | PatchGAN (70×70) |
| Loss | cGAN + λ×L1 |
| λ | 100 |
