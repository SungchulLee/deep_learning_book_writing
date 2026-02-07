# Adversarial Training

Generative Adversarial Networks (GANs) represent a paradigm shift in generative modeling. Unlike likelihood-based approaches (VAEs, Flows, Diffusion models), GANs learn through **adversarial competition** between two neural networks.

## The Core Intuition

Consider an analogy from art forgery:

**The Forger (Generator)**: Learns to create fake paintings that look authentic
**The Expert (Discriminator)**: Learns to distinguish genuine art from forgeries

As the expert improves at detecting fakes, the forger must create more convincing forgeries. This adversarial dynamic drives both parties to continuously improve. Eventually, the forger produces paintings indistinguishable from authentic masterpieces.

This competitive process, when translated to neural networks, enables generators to produce remarkably realistic data without ever explicitly computing probability densities.

## Historical Context

Ian Goodfellow introduced GANs in 2014, describing them as a breakthrough that enabled neural networks to learn generative models through a game-theoretic framework. The original paper demonstrated that adversarial training could produce sharp, realistic images—a significant improvement over the blurry outputs typical of variational autoencoders at the time.

## The Adversarial Framework

GANs consist of two neural networks trained simultaneously:

### Generator $G$

The generator transforms random noise into synthetic data:

$$G: \mathcal{Z} \rightarrow \mathcal{X}$$

where:

- $\mathcal{Z}$ is the latent space (typically standard Gaussian $\mathcal{N}(0, I)$)
- $\mathcal{X}$ is the data space (e.g., images)
- $z \sim p_z(z)$ is the input noise vector
- $G(z)$ is the generated sample

### Discriminator $D$

The discriminator classifies inputs as real or generated:

$$D: \mathcal{X} \rightarrow [0, 1]$$

where:

- $D(x)$ outputs the probability that $x$ is real
- $D(x) \approx 1$ indicates "definitely real"
- $D(x) \approx 0$ indicates "definitely fake"

## The Two-Player Game

GAN training is formulated as a **minimax game** between G and D:

$$\min_G \max_D V(D, G) = \mathbb{E}_{x \sim p_{\text{data}}(x)}[\log D(x)] + \mathbb{E}_{z \sim p_z(z)}[\log(1 - D(G(z)))]$$

This objective has two components:

1. **$\mathbb{E}_{x \sim p_{\text{data}}}[\log D(x)]$**: Expected log-probability that D correctly classifies real data
2. **$\mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))]$**: Expected log-probability that D correctly classifies fake data

### Discriminator's Goal

The discriminator maximizes $V(D, G)$, which means:

- Maximize $D(x)$ for real samples (push toward 1)
- Minimize $D(G(z))$ for fake samples (push toward 0)

This is equivalent to **binary cross-entropy classification**.

### Generator's Goal

The generator minimizes $V(D, G)$, which means:

- Maximize $D(G(z))$ (fool the discriminator)
- Generate samples that D classifies as real

## Why Adversarial Training Works

The adversarial framework provides several advantages:

### Implicit Distribution Matching

Unlike VAEs that require explicit likelihood computation, GANs implicitly match distributions through the discriminator's feedback. The generator never needs to compute $p_G(x)$ directly—it only needs to produce samples that fool D.

### Sharp Sample Generation

The discriminator provides detailed feedback about sample quality. Unlike reconstruction losses that average over all pixels, the discriminator focuses on features that distinguish real from fake, enabling sharper outputs.

### Flexible Architecture

Both G and D can be any differentiable function (typically neural networks). This flexibility allows GANs to leverage powerful architectures like CNNs for image generation.

## The Learning Process

### Phase 1: Early Training

Initially, the generator produces random noise-like outputs:

- D easily distinguishes real from fake: $D(x) \approx 1$, $D(G(z)) \approx 0$
- G receives strong gradient signal to improve
- Both networks learn rapidly

### Phase 2: Competitive Improvement

As training progresses:

- G produces increasingly realistic samples
- D must become more sophisticated to detect fakes
- The "arms race" drives quality improvements

### Phase 3: Equilibrium (Ideal)

At convergence:

- G produces samples indistinguishable from real data
- D outputs 0.5 for all inputs (cannot distinguish)
- Neither network can improve unilaterally

## Basic Training Algorithm

```python
import torch
import torch.nn as nn
import torch.optim as optim

def train_gan(generator, discriminator, dataloader, latent_dim, 
              n_epochs, device):
    """
    Basic GAN training loop.
    
    Args:
        generator: Generator network G(z) -> x
        discriminator: Discriminator network D(x) -> [0, 1]
        dataloader: DataLoader for real data
        latent_dim: Dimension of latent vector z
        n_epochs: Number of training epochs
        device: Training device (cpu/cuda)
    """
    # Loss and optimizers
    criterion = nn.BCELoss()
    g_optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    
    for epoch in range(n_epochs):
        for real_data, _ in dataloader:
            batch_size = real_data.size(0)
            real_data = real_data.to(device)
            
            # Labels
            real_labels = torch.ones(batch_size, 1, device=device)
            fake_labels = torch.zeros(batch_size, 1, device=device)
            
            # ==================
            # Train Discriminator
            # ==================
            d_optimizer.zero_grad()
            
            # Loss on real data
            d_real = discriminator(real_data)
            d_loss_real = criterion(d_real, real_labels)
            
            # Loss on fake data
            z = torch.randn(batch_size, latent_dim, device=device)
            fake_data = generator(z)
            d_fake = discriminator(fake_data.detach())  # Detach to avoid G gradients
            d_loss_fake = criterion(d_fake, fake_labels)
            
            # Total discriminator loss
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            d_optimizer.step()
            
            # ===============
            # Train Generator
            # ===============
            g_optimizer.zero_grad()
            
            # Generate new fake samples
            z = torch.randn(batch_size, latent_dim, device=device)
            fake_data = generator(z)
            d_fake = discriminator(fake_data)
            
            # G wants D to think fakes are real
            g_loss = criterion(d_fake, real_labels)
            g_loss.backward()
            g_optimizer.step()
```

## Key Training Principles

### 1. Alternating Optimization

Train D and G in alternation:
- Update D with fixed G
- Update G with fixed D
- Repeat

### 2. Balanced Training

Neither network should dominate:
- If D is too strong, G gets vanishing gradients
- If G is too strong, D provides poor feedback

### 3. Detachment for Discriminator Training

When training D on fake samples, **detach** the generator output to prevent gradients from flowing back to G:

```python
d_fake = discriminator(fake_data.detach())
```

### 4. Fresh Samples for Generator Training

Generate new fake samples when training G (don't reuse samples from D training):

```python
# For G training
z = torch.randn(batch_size, latent_dim)  # New noise
fake_data = generator(z)  # Fresh samples
```

## Comparison with Other Generative Models

| Aspect | GANs | VAEs | Flows | Diffusion |
|--------|------|------|-------|-----------|
| **Training** | Adversarial game | ELBO maximization | MLE | Denoising |
| **Likelihood** | Implicit | Lower bound | Exact | Lower bound |
| **Sample Quality** | Sharp | Blurry | Good | Excellent |
| **Training Stability** | Challenging | Stable | Stable | Stable |
| **Sampling Speed** | Fast | Fast | Fast | Slow |

## Advantages of Adversarial Training

1. **No explicit density**: Avoids intractable partition functions
2. **Flexible architectures**: Any differentiable network works
3. **Sharp samples**: Discriminator focuses on perceptually important features
4. **Fast inference**: Single forward pass through generator

## Challenges

1. **Training instability**: Careful hyperparameter tuning required
2. **Mode collapse**: Generator may ignore parts of the data distribution
3. **No likelihood**: Cannot evaluate $p_G(x)$ directly
4. **Evaluation difficulty**: Quality metrics are imperfect

## Summary

Adversarial training is a powerful paradigm for generative modeling:

- Two networks compete in a minimax game
- The generator learns to produce realistic samples
- The discriminator provides learning signal through classification
- Competition drives quality improvements
- Results in high-quality samples without explicit density modeling

This framework has spawned numerous variants and applications, from image synthesis to drug discovery, making it one of the most influential ideas in modern deep learning.
