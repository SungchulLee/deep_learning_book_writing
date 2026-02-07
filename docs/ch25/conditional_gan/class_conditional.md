# Conditional GAN (cGAN)

Conditional GANs extend the GAN framework to generate data conditioned on additional information like class labels, text, or other attributes.

## Mathematical Formulation

### Objective

$$\min_G \max_D V(D, G) = \mathbb{E}_{x,y}[\log D(x|y)] + \mathbb{E}_{z,y}[\log(1 - D(G(z|y)|y))]$$

where $y$ is the conditioning information.

### Key Difference from Standard GAN

| Standard GAN | Conditional GAN |
|--------------|-----------------|
| $G(z) \to x$ | $G(z, y) \to x$ |
| $D(x) \to [0,1]$ | $D(x, y) \to [0,1]$ |

## Implementation

### Conditioning Methods

#### 1. Concatenation (Simple)

```python
class ConditionalGenerator(nn.Module):
    def __init__(self, latent_dim, n_classes, img_shape):
        super().__init__()
        self.label_embed = nn.Embedding(n_classes, n_classes)
        
        self.model = nn.Sequential(
            nn.Linear(latent_dim + n_classes, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, np.prod(img_shape)),
            nn.Tanh()
        )
    
    def forward(self, z, labels):
        label_embedding = self.label_embed(labels)
        gen_input = torch.cat([z, label_embedding], dim=1)
        return self.model(gen_input).view(-1, *img_shape)
```

#### 2. Conditional Batch Normalization

```python
class ConditionalBatchNorm2d(nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.bn = nn.BatchNorm2d(num_features, affine=False)
        self.gamma = nn.Embedding(num_classes, num_features)
        self.beta = nn.Embedding(num_classes, num_features)
        
    def forward(self, x, y):
        out = self.bn(x)
        gamma = self.gamma(y).view(-1, out.size(1), 1, 1)
        beta = self.beta(y).view(-1, out.size(1), 1, 1)
        return gamma * out + beta
```

#### 3. Projection Discriminator

```python
class ProjectionDiscriminator(nn.Module):
    """Discriminator with projection-based conditioning."""
    
    def __init__(self, img_channels, n_classes, feature_maps=64):
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(img_channels, feature_maps, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(feature_maps, feature_maps * 2, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool2d(1)
        )
        
        self.fc = nn.Linear(feature_maps * 2, 1)
        self.embed = nn.Embedding(n_classes, feature_maps * 2)
    
    def forward(self, x, y):
        h = self.features(x).view(x.size(0), -1)
        out = self.fc(h)
        
        # Projection: add inner product with class embedding
        embed = self.embed(y)
        out += (h * embed).sum(dim=1, keepdim=True)
        
        return torch.sigmoid(out)
```

## Training

```python
def train_cgan_step(G, D, real_imgs, labels, z, criterion, g_opt, d_opt):
    batch_size = real_imgs.size(0)
    
    # Train Discriminator
    d_opt.zero_grad()
    
    real_validity = D(real_imgs, labels)
    d_real_loss = criterion(real_validity, torch.ones_like(real_validity))
    
    fake_imgs = G(z, labels)
    fake_validity = D(fake_imgs.detach(), labels)
    d_fake_loss = criterion(fake_validity, torch.zeros_like(fake_validity))
    
    d_loss = (d_real_loss + d_fake_loss) / 2
    d_loss.backward()
    d_opt.step()
    
    # Train Generator
    g_opt.zero_grad()
    
    fake_imgs = G(z, labels)
    validity = D(fake_imgs, labels)
    g_loss = criterion(validity, torch.ones_like(validity))
    
    g_loss.backward()
    g_opt.step()
    
    return g_loss.item(), d_loss.item()
```

## Applications

| Application | Conditioning |
|-------------|-------------|
| Class-conditional generation | Class labels |
| Text-to-image | Text embeddings |
| Image-to-image | Input image |
| Attribute manipulation | Binary attributes |

## Summary

cGAN enables controlled generation by conditioning both G and D on auxiliary information, forming the basis for many practical applications.
