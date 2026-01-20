# StyleGAN

StyleGAN (Karras et al., 2019) introduced style-based generation, enabling unprecedented control over generated image attributes at different scales.

## Key Innovation: Style-Based Architecture

### Mapping Network

Transform latent z to intermediate w space:

$$w = f(z), \quad f: \mathcal{Z} \rightarrow \mathcal{W}$$

```python
class MappingNetwork(nn.Module):
    """8-layer MLP mapping z → w."""
    
    def __init__(self, latent_dim=512, w_dim=512, n_layers=8):
        super().__init__()
        layers = []
        for i in range(n_layers):
            layers.extend([
                EqualizedLinear(latent_dim if i == 0 else w_dim, w_dim),
                nn.LeakyReLU(0.2)
            ])
        self.mapping = nn.Sequential(*layers)
    
    def forward(self, z):
        return self.mapping(z)
```

### Adaptive Instance Normalization (AdaIN)

Inject style at each layer:

$$\text{AdaIN}(x, y) = y_s \cdot \frac{x - \mu(x)}{\sigma(x)} + y_b$$

```python
class AdaIN(nn.Module):
    """Adaptive Instance Normalization."""
    
    def __init__(self, channels, w_dim):
        super().__init__()
        self.norm = nn.InstanceNorm2d(channels)
        self.style = EqualizedLinear(w_dim, channels * 2)
    
    def forward(self, x, w):
        style = self.style(w)
        gamma, beta = style.chunk(2, dim=1)
        gamma = gamma.view(-1, gamma.size(1), 1, 1)
        beta = beta.view(-1, beta.size(1), 1, 1)
        return gamma * self.norm(x) + beta
```

### Noise Injection

Add stochastic variation at each layer:

```python
class NoiseInjection(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1, channels, 1, 1))
    
    def forward(self, x):
        noise = torch.randn(x.size(0), 1, x.size(2), x.size(3), device=x.device)
        return x + self.weight * noise
```

## Complete Generator

```python
class StyleGANGenerator(nn.Module):
    """StyleGAN Generator."""
    
    def __init__(self, latent_dim=512, w_dim=512):
        super().__init__()
        
        self.mapping = MappingNetwork(latent_dim, w_dim)
        self.const = nn.Parameter(torch.randn(1, 512, 4, 4))
        
        # Synthesis blocks (4x4 → 1024x1024)
        self.blocks = nn.ModuleList([
            SynthesisBlock(512, 512, w_dim),  # 4→8
            SynthesisBlock(512, 512, w_dim),  # 8→16
            SynthesisBlock(512, 512, w_dim),  # 16→32
            SynthesisBlock(512, 256, w_dim),  # 32→64
            SynthesisBlock(256, 128, w_dim),  # 64→128
            SynthesisBlock(128, 64, w_dim),   # 128→256
            SynthesisBlock(64, 32, w_dim),    # 256→512
            SynthesisBlock(32, 16, w_dim),    # 512→1024
        ])
        
    def forward(self, z):
        w = self.mapping(z)
        x = self.const.expand(z.size(0), -1, -1, -1)
        for block in self.blocks:
            x = block(x, w)
        return x
```

## Style Mixing

Use different w vectors at different layers:

```python
def style_mixing(self, z1, z2, crossover_layer):
    """Mix styles from two latent vectors."""
    w1 = self.mapping(z1)
    w2 = self.mapping(z2)
    
    x = self.const.expand(z1.size(0), -1, -1, -1)
    for i, block in enumerate(self.blocks):
        w = w1 if i < crossover_layer else w2
        x = block(x, w)
    return x
```

## Hierarchical Style Control

| Layer Level | Controls |
|-------------|----------|
| Coarse (4×4 – 8×8) | Pose, face shape, eyeglasses |
| Medium (16×16 – 32×32) | Facial features, hairstyle |
| Fine (64×64 – 1024×1024) | Color scheme, microstructure |

## Innovations Summary

| Component | Purpose |
|-----------|---------|
| Mapping network | Disentangled w space |
| AdaIN | Style injection |
| Noise injection | Stochastic detail |
| Constant input | No direct z dependency |
| Style mixing | Regularization + control |
