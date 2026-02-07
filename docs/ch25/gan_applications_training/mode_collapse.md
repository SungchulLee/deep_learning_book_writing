# Mode Collapse

Mode collapse is the most common and challenging problem in GAN training, where the generator produces limited variety of outputs.

## What is Mode Collapse?

The generator learns to produce only a few "safe" outputs that fool the discriminator, ignoring large portions of the data distribution.

### Types

| Type | Description |
|------|-------------|
| **Complete** | G produces nearly identical outputs |
| **Partial** | G covers some modes, misses others |
| **Oscillating** | G cycles between modes |

## Why It Happens

### Generator's Incentive Problem

$$\mathcal{L}_G = -\mathbb{E}_z[\log D(G(z))]$$

If G finds one output that fools D, there's no gradient pushing it to explore other modes.

### Training Dynamics

```
1. G finds output x* that fools D
2. D learns to reject x*
3. G finds new x** that fools D
4. D learns to reject x**
5. G returns to x* (D has "forgotten")
6. Cycle continues...
```

## Detection

### Visual Inspection

```python
def check_mode_collapse(G, latent_dim, n_samples=100):
    """Generate samples and check diversity."""
    z = torch.randn(n_samples, latent_dim)
    samples = G(z)
    
    # Compute pairwise distances
    flat = samples.view(n_samples, -1)
    distances = torch.cdist(flat, flat)
    
    # Check for near-duplicates
    min_dist = distances[~torch.eye(n_samples, dtype=bool)].min()
    mean_dist = distances[~torch.eye(n_samples, dtype=bool)].mean()
    
    if min_dist < 0.01 or mean_dist < 0.1:
        print("WARNING: Possible mode collapse detected")
```

### Monitoring D(fake) Variance

Low variance in D outputs on fake data suggests collapse:

```python
def monitor_collapse(d_fake_history, window=100):
    recent = d_fake_history[-window:]
    variance = np.var(recent)
    if variance < 0.001:
        print("Warning: Low variance in D(fake) - possible collapse")
```

## Solutions

### 1. Minibatch Discrimination

Let D see batch statistics to detect collapse:

```python
class MinibatchDiscrimination(nn.Module):
    def __init__(self, in_features, out_features, kernel_dims):
        super().__init__()
        self.T = nn.Parameter(torch.randn(in_features, out_features, kernel_dims))
    
    def forward(self, x):
        M = torch.mm(x, self.T.view(x.size(1), -1))
        M = M.view(-1, self.T.size(1), self.T.size(2))
        
        # Compute L1 distance between samples
        diffs = M.unsqueeze(0) - M.unsqueeze(1)
        abs_diffs = diffs.abs().sum(2)
        minibatch_features = torch.exp(-abs_diffs).sum(0)
        
        return torch.cat([x, minibatch_features], 1)
```

### 2. Feature Matching

Train G to match feature statistics, not fool D:

```python
def feature_matching_loss(D, real_data, fake_data):
    real_features = D.get_features(real_data)
    fake_features = D.get_features(fake_data)
    return F.mse_loss(fake_features.mean(0), real_features.mean(0))
```

### 3. Unrolled GAN

Update G considering future D updates:

```python
def unrolled_generator_loss(G, D, z, k_steps=5):
    """Compute G loss with unrolled D updates."""
    D_copy = copy.deepcopy(D)
    
    for _ in range(k_steps):
        # Simulate D update
        fake = G(z)
        d_loss = discriminator_loss(D_copy, real, fake)
        d_loss.backward()
        optimizer_step(D_copy)
    
    # G loss using unrolled D
    fake = G(z)
    return generator_loss(D_copy, fake)
```

### 4. Wasserstein Loss

Continuous gradients help avoid collapse:

```python
# Use WGAN-GP instead of standard GAN
loss_fn = WassersteinGANLoss(lambda_gp=10)
```

### 5. Multiple Generators

Train several generators to cover different modes:

```python
class MultipleGenerators(nn.Module):
    def __init__(self, n_generators, latent_dim):
        super().__init__()
        self.generators = nn.ModuleList([
            Generator(latent_dim) for _ in range(n_generators)
        ])
    
    def forward(self, z, generator_idx=None):
        if generator_idx is None:
            generator_idx = torch.randint(len(self.generators), (1,)).item()
        return self.generators[generator_idx](z)
```

## Summary

| Solution | Mechanism |
|----------|-----------|
| Minibatch discrimination | D sees batch diversity |
| Feature matching | Match statistics, not fool D |
| Unrolled GAN | G considers future D |
| WGAN | Continuous gradients |
| Multiple generators | Explicit mode coverage |
