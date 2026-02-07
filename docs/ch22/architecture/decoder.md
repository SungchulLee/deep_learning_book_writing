# Decoder Network

The generative model: mapping latent codes to data distributions.

---

## Learning Objectives

By the end of this section, you will be able to:

- Design decoder architectures for different data types
- Choose appropriate output distributions (Bernoulli, Gaussian, Laplace)
- Implement both MLP and convolutional decoders
- Understand the relationship between decoder choice and reconstruction loss

---

## The Decoder's Role

The decoder defines the generative model $p_\theta(x|z)$: given a latent code $z$, it specifies a distribution over data $x$. During training, the decoder reconstructs inputs from latent samples. During generation, it maps samples from the prior $z \sim p(z)$ to new data points.

The decoder architecture mirrors the encoder, transforming low-dimensional latent codes back to data space. For images, this means upsampling from compact representations to full resolution.

---

## Output Distributions

### Bernoulli Decoder

For binary or $[0, 1]$-bounded data:

$$p_\theta(x_i|z) = \text{Bernoulli}(\hat{x}_i) = \hat{x}_i^{x_i}(1 - \hat{x}_i)^{1-x_i}$$

where $\hat{x}_i = \sigma(\text{decoder}(z)_i)$ uses a sigmoid activation. The reconstruction loss is Binary Cross-Entropy (BCE).

### Gaussian Decoder

For continuous, real-valued data:

$$p_\theta(x|z) = \mathcal{N}(x; \mu_\theta(z), \sigma^2 I)$$

With fixed variance, the reconstruction loss reduces to Mean Squared Error (MSE). With learned variance, the decoder outputs both mean and log-variance.

### Choosing the Right Decoder

| Data Type | Decoder | Final Activation | Loss |
|-----------|---------|-----------------|------|
| Binary images | Bernoulli | Sigmoid | BCE |
| [0,1] normalized images | Bernoulli | Sigmoid | BCE |
| Continuous, unbounded | Gaussian (fixed var) | None | MSE |
| Continuous, heteroscedastic | Gaussian (learned var) | None (mean), None (logvar) | NLL |
| Robust to outliers | Laplace | None | L1 |

---

## MLP Decoder

```python
import torch
import torch.nn as nn

class MLPDecoder(nn.Module):
    """
    Fully connected decoder for vector outputs.
    """
    
    def __init__(self, latent_dim=32, hidden_dim=256, output_dim=784,
                 output_distribution='bernoulli'):
        super().__init__()
        
        self.output_distribution = output_distribution
        
        self.layers = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        if output_distribution == 'bernoulli':
            self.output_head = nn.Sequential(
                nn.Linear(hidden_dim, output_dim),
                nn.Sigmoid()
            )
        elif output_distribution == 'gaussian':
            self.fc_mu = nn.Linear(hidden_dim, output_dim)
            self.fc_logvar = nn.Linear(hidden_dim, output_dim)
        else:
            self.output_head = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, z):
        h = self.layers(z)
        
        if self.output_distribution == 'gaussian':
            return self.fc_mu(h), self.fc_logvar(h)
        return self.output_head(h)
```

---

## Convolutional Decoder

### Transposed Convolution Approach

```python
class ConvDecoder(nn.Module):
    """
    Convolutional decoder using transposed convolutions for upsampling.
    """
    
    def __init__(self, latent_dim=32, hidden_channels=32, out_channels=1):
        super().__init__()
        
        self.fc = nn.Linear(latent_dim, hidden_channels * 4 * 4 * 4)
        self.hidden_channels = hidden_channels * 4
        
        self.conv_layers = nn.Sequential(
            # 4x4 → 7x7
            nn.ConvTranspose2d(hidden_channels * 4, hidden_channels * 2, 
                              3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(hidden_channels * 2),
            
            # 7x7 → 14x14
            nn.ConvTranspose2d(hidden_channels * 2, hidden_channels, 
                              4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(hidden_channels),
            
            # 14x14 → 28x28
            nn.ConvTranspose2d(hidden_channels, out_channels, 
                              4, stride=2, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, z):
        h = self.fc(z)
        h = h.view(-1, self.hidden_channels, 4, 4)
        return self.conv_layers(h)
```

### Checkerboard Artifacts

Transposed convolutions can produce checkerboard artifacts when stride doesn't evenly divide the kernel size. An alternative approach uses nearest-neighbor upsampling followed by standard convolution:

```python
class UpsampleConvDecoder(nn.Module):
    """Decoder using upsample + conv to avoid checkerboard artifacts."""
    
    def __init__(self, latent_dim=32, hidden_channels=64, out_channels=1):
        super().__init__()
        
        self.fc = nn.Linear(latent_dim, hidden_channels * 7 * 7)
        
        self.layers = nn.Sequential(
            # 7x7 → 14x14
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(hidden_channels, hidden_channels // 2, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(hidden_channels // 2),
            
            # 14x14 → 28x28
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(hidden_channels // 2, out_channels, 3, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, z):
        h = self.fc(z).view(-1, 64, 7, 7)
        return self.layers(h)
```

---

## Learned Variance Decoder

### Motivation

A fixed-variance Gaussian decoder assumes uniform noise across all pixels and all latent codes. A **learned variance** decoder can express spatially varying uncertainty:

$$p_\theta(x|z) = \mathcal{N}(x; \mu_\theta(z), \text{diag}(\sigma^2_\theta(z)))$$

### Implementation

```python
class GaussianDecoder(nn.Module):
    """Decoder that outputs both mean and per-pixel variance."""
    
    def __init__(self, latent_dim=32, hidden_dim=256, output_dim=784):
        super().__init__()
        
        self.shared = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.fc_mu = nn.Linear(hidden_dim, output_dim)
        self.fc_logvar = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, z):
        h = self.shared(z)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


def gaussian_nll(x, mu, logvar):
    """
    Negative log-likelihood for Gaussian with learned variance.
    
    -log p(x|z) = 0.5 * (log(2π) + logvar + (x - mu)² / exp(logvar))
    """
    return 0.5 * (logvar + (x - mu).pow(2) / logvar.exp()).sum()
```

### Benefits

Learned variance provides adaptive uncertainty (different regions can have different confidence levels), better calibration (the model knows where it is uncertain), and improved generation (captures heteroscedastic structure in data).

---

## Decoder Capacity and Blurriness

### Why VAE Samples Are Blurry

VAE reconstructions tend to be blurry compared to GAN outputs. This arises from the combination of Gaussian decoder assumptions (averaging over possible outputs), regularization from the KL term (limiting information in $z$), and the fact that the decoder must produce outputs that are likely for all $z$ in the high-probability region of $q_\phi(z|x)$.

### Mitigations

Strategies to improve sharpness include using more expressive decoders (deeper networks, skip connections), learned variance (concentrates on plausible outputs), adversarial training (VAE-GAN hybrid), and discrete latent spaces (VQ-VAE avoids the averaging problem).

---

## Summary

| Component | Description | Key Consideration |
|-----------|-------------|-------------------|
| **Bernoulli decoder** | Sigmoid output, BCE loss | Natural for binary/[0,1] data |
| **Gaussian decoder** | Linear output, MSE loss | Natural for continuous data |
| **Learned variance** | Two-head output | Better calibration |
| **Conv decoder** | Upsample to image resolution | Avoid checkerboard artifacts |

---

## Exercises

### Exercise 1: Decoder Comparison

Train VAEs with Bernoulli vs Gaussian decoders on MNIST. Compare reconstruction quality (visual and quantitative).

### Exercise 2: Learned Variance

Implement a Gaussian decoder with learned per-pixel variance. Visualize the learned variance map — which regions have high uncertainty?

### Exercise 3: Depth Ablation

Compare decoders with 2, 4, and 6 hidden layers. Does deeper decoder improve sample quality?

---

## What's Next

The next section discusses [Prior Selection](prior.md), examining alternatives to the standard Gaussian prior.
