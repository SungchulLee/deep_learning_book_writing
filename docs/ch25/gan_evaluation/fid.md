# Fréchet Inception Distance (FID)

## Overview

The Fréchet Inception Distance (FID) is the most widely adopted metric for evaluating generative models, particularly for image synthesis. Introduced by Heusel et al. (2017), FID measures the distance between the distribution of generated images and real images in the feature space of a pre-trained Inception network.

!!! info "Learning Objectives"
    By the end of this section, you will be able to:
    
    - Derive and understand the mathematical foundation of FID
    - Implement FID computation from scratch with proper numerical handling
    - Understand why FID uses Inception features instead of pixel space
    - Recognize FID's advantages over IS and its remaining limitations
    - Apply FID correctly in research and production settings

## Mathematical Foundation

### The Fréchet Distance

The Fréchet distance (also known as Wasserstein-2 distance for Gaussians) measures the distance between two probability distributions. For multivariate Gaussian distributions, it has a closed-form solution.

Given two Gaussian distributions:

- Real data features: $\mathcal{N}(\mu_r, \Sigma_r)$
- Generated data features: $\mathcal{N}(\mu_g, \Sigma_g)$

The Fréchet distance is:

$$
\text{FID} = \|\mu_r - \mu_g\|_2^2 + \text{Tr}\left(\Sigma_r + \Sigma_g - 2(\Sigma_r \Sigma_g)^{1/2}\right)
$$

### Component Breakdown

**Mean Difference Term: $\|\mu_r - \mu_g\|_2^2$**

This measures how different the "average" real and generated images are in feature space. A large mean difference indicates systematic bias in generation.

**Covariance Terms:**

- $\text{Tr}(\Sigma_r)$: Total variance in real data features
- $\text{Tr}(\Sigma_g)$: Total variance in generated data features  
- $-2\text{Tr}((\Sigma_r \Sigma_g)^{1/2})$: Covariance overlap penalty

The covariance terms capture whether the generator produces the same range and correlation structure as real data.

### Why Assume Gaussian?

The Gaussian assumption for feature distributions is justified because:

1. **Central Limit Theorem**: Deep network activations tend toward Gaussianity due to averaging effects
2. **Computational Tractability**: Closed-form solution exists
3. **Empirical Validation**: Works well in practice for image features
4. **Mathematical Foundation**: Optimal transport distance for Gaussians

### Connection to Optimal Transport

FID is the squared 2-Wasserstein distance for Gaussian distributions:

$$
W_2^2(\mathcal{N}(\mu_1, \Sigma_1), \mathcal{N}(\mu_2, \Sigma_2)) = \text{FID}
$$

This means FID measures the minimum "cost" to transform one distribution into another, where cost is squared Euclidean distance.

## Mathematical Derivation

### Starting from Wasserstein Distance

The 2-Wasserstein distance between distributions $P$ and $Q$ is:

$$
W_2(P, Q) = \left(\inf_{\gamma \in \Gamma(P,Q)} \mathbb{E}_{(x,y)\sim\gamma}[\|x - y\|_2^2]\right)^{1/2}
$$

where $\Gamma(P,Q)$ is the set of all joint distributions with marginals $P$ and $Q$.

### Closed Form for Gaussians

For Gaussians, the optimal transport plan is known, giving:

$$
W_2^2(\mathcal{N}(\mu_1, \Sigma_1), \mathcal{N}(\mu_2, \Sigma_2)) = \|\mu_1 - \mu_2\|_2^2 + \text{Bures}(\Sigma_1, \Sigma_2)
$$

where the Bures metric for positive definite matrices is:

$$
\text{Bures}(\Sigma_1, \Sigma_2) = \text{Tr}(\Sigma_1) + \text{Tr}(\Sigma_2) - 2\text{Tr}\left((\Sigma_1^{1/2}\Sigma_2\Sigma_1^{1/2})^{1/2}\right)
$$

### Simplification

Using the cyclic property of trace and positive definiteness:

$$
\text{Tr}\left((\Sigma_1^{1/2}\Sigma_2\Sigma_1^{1/2})^{1/2}\right) = \text{Tr}\left((\Sigma_1\Sigma_2)^{1/2}\right)
$$

This gives us the standard FID formula.

## Why Inception Features?

### The Problem with Pixel Space

Comparing images directly in pixel space is problematic:

1. **Perceptually Irrelevant**: Small pixel shifts cause large distances but are visually imperceptible
2. **Scale Sensitivity**: Pixel distances depend heavily on resolution
3. **No Semantic Understanding**: A cat and dog may have similar pixel statistics

### Inception Network as Feature Extractor

InceptionV3 (trained on ImageNet) provides:

| Feature | Benefit |
|---------|---------|
| Semantic understanding | Captures object identity and scene content |
| Perceptual alignment | Features correlate with human perception |
| Hierarchical representation | Captures both low-level and high-level features |
| Standardization | Same network enables fair comparisons |

### Pool3 Layer (2048-dimensional)

FID uses the output of the global average pooling layer before the final classification:

```
InceptionV3 Architecture:
┌─────────────┐
│   Input     │ 299×299×3
├─────────────┤
│   Conv/Pool │ Multiple layers
├─────────────┤
│  Mixed_7c   │ 8×8×2048
├─────────────┤
│  AvgPool    │ 2048 ← FID uses this!
├─────────────┤
│    FC       │ 1000 (classes)
└─────────────┘
```

## PyTorch Implementation

### Complete FID Calculator

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import linalg
from typing import Tuple, Optional, Union
from torchvision.models import inception_v3, Inception_V3_Weights


class FIDCalculator:
    """
    Comprehensive Fréchet Inception Distance calculator.
    
    FID measures the distance between real and generated image distributions
    in the feature space of a pre-trained Inception network.
    
    Lower FID indicates better quality and more similar distributions.
    
    Attributes:
        device: Computation device
        inception: Pre-trained InceptionV3 model
        feature_dim: Dimension of extracted features (2048)
    """
    
    def __init__(self, 
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize FID calculator.
        
        Args:
            device: Computation device ('cuda' or 'cpu')
        """
        self.device = device
        self.inception = None
        self.feature_dim = 2048
        
    def _load_inception(self):
        """
        Load and modify InceptionV3 for feature extraction.
        
        We extract features from the pool3 layer (2048-dimensional)
        which captures high-level semantic information.
        """
        # Load pre-trained InceptionV3
        self.inception = inception_v3(
            weights=Inception_V3_Weights.IMAGENET1K_V1,
            transform_input=False
        )
        
        # Remove the final classification layer
        # We want features from the pooling layer
        self.inception.fc = nn.Identity()
        
        # Set to evaluation mode
        self.inception.eval()
        self.inception.to(self.device)
        
        # Disable gradients for efficiency
        for param in self.inception.parameters():
            param.requires_grad = False
    
    def _preprocess(self, images: torch.Tensor) -> torch.Tensor:
        """
        Preprocess images for InceptionV3.
        
        Requirements:
        - Size: 299×299
        - Range: Normalized with ImageNet statistics
        - Channels: 3 (RGB)
        
        Args:
            images: Input images [B, C, H, W] in range [0, 1]
            
        Returns:
            Preprocessed images
        """
        # Resize if needed
        if images.shape[2] != 299 or images.shape[3] != 299:
            images = F.interpolate(
                images,
                size=(299, 299),
                mode='bilinear',
                align_corners=False
            )
        
        # Handle grayscale
        if images.shape[1] == 1:
            images = images.repeat(1, 3, 1, 1)
        
        # Normalize with ImageNet statistics
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        
        mean = mean.to(images.device)
        std = std.to(images.device)
        
        return (images - mean) / std
    
    def extract_features(self,
                        images: torch.Tensor,
                        batch_size: int = 64) -> np.ndarray:
        """
        Extract InceptionV3 features from images.
        
        Args:
            images: Images [N, C, H, W] in range [0, 1]
            batch_size: Batch size for processing
            
        Returns:
            Features [N, 2048]
        """
        if self.inception is None:
            self._load_inception()
        
        all_features = []
        n_images = len(images)
        
        with torch.no_grad():
            for i in range(0, n_images, batch_size):
                batch = images[i:i+batch_size].to(self.device)
                batch = self._preprocess(batch)
                
                # Extract features
                features = self.inception(batch)
                all_features.append(features.cpu().numpy())
        
        return np.concatenate(all_features, axis=0)
    
    @staticmethod
    def compute_statistics(features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute mean and covariance of features.
        
        Args:
            features: Feature vectors [N, D]
            
        Returns:
            Tuple of (mean [D], covariance [D, D])
            
        Mathematical Notes:
            μ = (1/N) Σ x_i
            Σ = (1/(N-1)) Σ (x_i - μ)(x_i - μ)ᵀ
        """
        mu = np.mean(features, axis=0)
        sigma = np.cov(features, rowvar=False)
        
        return mu, sigma
    
    @staticmethod
    def calculate_frechet_distance(mu1: np.ndarray,
                                   sigma1: np.ndarray,
                                   mu2: np.ndarray,
                                   sigma2: np.ndarray,
                                   eps: float = 1e-6) -> float:
        """
        Calculate Fréchet distance between two Gaussians.
        
        FID = ||μ₁ - μ₂||² + Tr(Σ₁ + Σ₂ - 2(Σ₁Σ₂)^{1/2})
        
        Args:
            mu1: Mean of first distribution [D]
            sigma1: Covariance of first distribution [D, D]
            mu2: Mean of second distribution [D]
            sigma2: Covariance of second distribution [D, D]
            eps: Small constant for numerical stability
            
        Returns:
            FID value (scalar, lower is better)
        """
        # Ensure numpy arrays
        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)
        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)
        
        assert mu1.shape == mu2.shape, f"Mean shapes differ: {mu1.shape} vs {mu2.shape}"
        assert sigma1.shape == sigma2.shape, f"Cov shapes differ: {sigma1.shape} vs {sigma2.shape}"
        
        # 1. Mean difference term: ||μ₁ - μ₂||²
        diff = mu1 - mu2
        mean_term = np.dot(diff, diff)
        
        # 2. Matrix square root: (Σ₁Σ₂)^{1/2}
        # This is the computationally expensive step
        
        # Product of covariances
        product = sigma1 @ sigma2
        
        # Matrix square root using scipy
        covmean, _ = linalg.sqrtm(product, disp=False)
        
        # Handle numerical issues
        if not np.isfinite(covmean).all():
            print(f"Warning: Non-finite values in matrix sqrt. Adding {eps} to diagonal.")
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset) @ (sigma2 + offset))
        
        # Handle imaginary components (numerical artifact)
        if np.iscomplexobj(covmean):
            if np.allclose(covmean.imag, 0, atol=1e-3):
                covmean = covmean.real
            else:
                raise ValueError(f"Significant imaginary component: {np.max(np.abs(covmean.imag))}")
        
        # 3. Trace terms
        trace_term = np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean)
        
        # 4. Final FID
        fid = mean_term + trace_term
        
        return float(fid)
    
    def calculate_fid(self,
                     real_images: torch.Tensor,
                     generated_images: torch.Tensor,
                     batch_size: int = 64) -> float:
        """
        Calculate FID between real and generated images.
        
        Complete pipeline:
        1. Extract Inception features from real images
        2. Extract Inception features from generated images
        3. Compute statistics (μ, Σ) for both
        4. Calculate Fréchet distance
        
        Args:
            real_images: Real images [N_r, C, H, W]
            generated_images: Generated images [N_g, C, H, W]
            batch_size: Batch size for feature extraction
            
        Returns:
            FID score (lower is better)
        """
        print(f"Extracting features from {len(real_images)} real images...")
        real_features = self.extract_features(real_images, batch_size)
        
        print(f"Extracting features from {len(generated_images)} generated images...")
        gen_features = self.extract_features(generated_images, batch_size)
        
        print("Computing statistics...")
        mu_real, sigma_real = self.compute_statistics(real_features)
        mu_gen, sigma_gen = self.compute_statistics(gen_features)
        
        print("Calculating Fréchet distance...")
        fid = self.calculate_frechet_distance(mu_real, sigma_real, mu_gen, sigma_gen)
        
        print(f"FID = {fid:.4f}")
        return fid
    
    def calculate_fid_from_statistics(self,
                                      mu_real: np.ndarray,
                                      sigma_real: np.ndarray,
                                      generated_images: torch.Tensor,
                                      batch_size: int = 64) -> float:
        """
        Calculate FID using pre-computed real data statistics.
        
        This is more efficient when comparing multiple generators
        against the same real dataset.
        
        Args:
            mu_real: Pre-computed mean of real features [D]
            sigma_real: Pre-computed covariance of real features [D, D]
            generated_images: Generated images [N, C, H, W]
            batch_size: Batch size for feature extraction
            
        Returns:
            FID score
        """
        gen_features = self.extract_features(generated_images, batch_size)
        mu_gen, sigma_gen = self.compute_statistics(gen_features)
        
        return self.calculate_frechet_distance(mu_real, sigma_real, mu_gen, sigma_gen)


def save_reference_statistics(real_images: torch.Tensor,
                             save_path: str,
                             batch_size: int = 64):
    """
    Pre-compute and save statistics for a reference dataset.
    
    This enables efficient FID computation during training
    without re-computing real data statistics each time.
    
    Args:
        real_images: Real images [N, C, H, W]
        save_path: Path to save statistics (.npz file)
        batch_size: Batch size for feature extraction
    """
    calculator = FIDCalculator()
    features = calculator.extract_features(real_images, batch_size)
    mu, sigma = FIDCalculator.compute_statistics(features)
    
    np.savez(save_path, mu=mu, sigma=sigma)
    print(f"Saved statistics to {save_path}")
    print(f"  Shape: μ={mu.shape}, Σ={sigma.shape}")


def load_reference_statistics(load_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load pre-computed statistics.
    
    Args:
        load_path: Path to .npz file
        
    Returns:
        Tuple of (mean, covariance)
    """
    data = np.load(load_path)
    return data['mu'], data['sigma']
```

### Demonstration with Synthetic Data

```python
def demonstrate_fid_computation():
    """
    Demonstrate FID computation with controlled scenarios.
    """
    print("=" * 70)
    print("Fréchet Inception Distance Demonstration")
    print("=" * 70)
    
    # Use synthetic features for demonstration
    # (In practice, these would come from Inception)
    n_samples = 5000
    feature_dim = 2048
    
    # Scenario 1: Identical distributions
    print("\n📊 Scenario 1: Identical Distributions")
    print("-" * 50)
    
    real_features = np.random.randn(n_samples, feature_dim)
    gen_features = np.random.randn(n_samples, feature_dim)
    
    mu_r, sigma_r = FIDCalculator.compute_statistics(real_features)
    mu_g, sigma_g = FIDCalculator.compute_statistics(gen_features)
    
    fid = FIDCalculator.calculate_frechet_distance(mu_r, sigma_r, mu_g, sigma_g)
    print(f"FID: {fid:.4f}")
    print("Note: Small non-zero FID due to finite sample estimation")
    
    # Scenario 2: Shifted mean
    print("\n📊 Scenario 2: Shifted Mean")
    print("-" * 50)
    
    gen_features_shifted = np.random.randn(n_samples, feature_dim) + 0.5
    
    mu_g2, sigma_g2 = FIDCalculator.compute_statistics(gen_features_shifted)
    fid_shifted = FIDCalculator.calculate_frechet_distance(mu_r, sigma_r, mu_g2, sigma_g2)
    
    print(f"FID: {fid_shifted:.4f}")
    print("Note: Mean shift increases FID significantly")
    
    # Scenario 3: Reduced variance (mode collapse indicator)
    print("\n📊 Scenario 3: Reduced Variance (Mode Collapse)")
    print("-" * 50)
    
    gen_features_collapsed = np.random.randn(n_samples, feature_dim) * 0.5
    
    mu_g3, sigma_g3 = FIDCalculator.compute_statistics(gen_features_collapsed)
    fid_collapsed = FIDCalculator.calculate_frechet_distance(mu_r, sigma_r, mu_g3, sigma_g3)
    
    print(f"FID: {fid_collapsed:.4f}")
    print("Note: Reduced variance indicates mode collapse")
    
    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"{'Scenario':<30} {'FID':>10}")
    print("-" * 40)
    print(f"{'Identical (baseline)':<30} {fid:>10.2f}")
    print(f"{'Shifted mean':<30} {fid_shifted:>10.2f}")
    print(f"{'Mode collapse':<30} {fid_collapsed:>10.2f}")
    print("\nLower FID = Better (more similar to real distribution)")


demonstrate_fid_computation()
```

## Interpreting FID Values

### Typical Ranges for Natural Images

| FID Value | Quality Level | Interpretation |
|-----------|---------------|----------------|
| < 5 | Excellent | Near-indistinguishable from real |
| 5 - 20 | Very Good | High-quality generation |
| 20 - 50 | Good | Minor artifacts or mode dropping |
| 50 - 100 | Moderate | Noticeable quality issues |
| > 100 | Poor | Significant distribution mismatch |

### Dataset-Specific Benchmarks

Different datasets have different FID ranges:

| Dataset | State-of-the-Art FID |
|---------|---------------------|
| CIFAR-10 | ~2 |
| CelebA-HQ 256 | ~5 |
| FFHQ 256 | ~3 |
| ImageNet 256 | ~2-5 |
| LSUN Bedroom | ~2-5 |

### Factors Affecting FID

1. **Sample Size**: More samples → more stable FID
2. **Image Resolution**: Different resolutions may need different preprocessing
3. **Color Space**: RGB vs grayscale affects features
4. **Truncation**: GAN truncation trades diversity for quality

## Sample Size Analysis

### Minimum Requirements

FID requires sufficient samples for reliable covariance estimation:

```python
def analyze_fid_sample_size():
    """
    Analyze how sample size affects FID stability.
    """
    feature_dim = 2048
    true_mu = np.zeros(feature_dim)
    true_sigma = np.eye(feature_dim)
    
    sample_sizes = [100, 500, 1000, 2048, 5000, 10000, 50000]
    n_trials = 10
    
    results = []
    
    for n in sample_sizes:
        fids = []
        for _ in range(n_trials):
            # Sample from same distribution
            features1 = np.random.randn(n, feature_dim)
            features2 = np.random.randn(n, feature_dim)
            
            mu1, sigma1 = FIDCalculator.compute_statistics(features1)
            mu2, sigma2 = FIDCalculator.compute_statistics(features2)
            
            fid = FIDCalculator.calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
            fids.append(fid)
        
        results.append({
            'n': n,
            'mean_fid': np.mean(fids),
            'std_fid': np.std(fids)
        })
        print(f"N={n:>6}: FID = {np.mean(fids):.2f} ± {np.std(fids):.2f}")
    
    return results
```

**Recommendations:**

- **Minimum**: 2,048 samples (= feature dimension)
- **Recommended**: 10,000+ samples
- **Best**: 50,000+ samples for stable comparisons

### Bootstrap Confidence Intervals

```python
def bootstrap_fid(real_features: np.ndarray,
                  gen_features: np.ndarray,
                  n_bootstrap: int = 1000,
                  sample_size: Optional[int] = None) -> Tuple[float, float, float]:
    """
    Compute FID with bootstrap confidence intervals.
    
    Args:
        real_features: Real data features [N, D]
        gen_features: Generated data features [N, D]
        n_bootstrap: Number of bootstrap samples
        sample_size: Size of each bootstrap sample (default: min(N_real, N_gen))
        
    Returns:
        Tuple of (FID, lower_ci, upper_ci) for 95% CI
    """
    n_real = len(real_features)
    n_gen = len(gen_features)
    
    if sample_size is None:
        sample_size = min(n_real, n_gen)
    
    bootstrap_fids = []
    
    for _ in range(n_bootstrap):
        # Bootstrap sample
        idx_real = np.random.choice(n_real, sample_size, replace=True)
        idx_gen = np.random.choice(n_gen, sample_size, replace=True)
        
        real_sample = real_features[idx_real]
        gen_sample = gen_features[idx_gen]
        
        mu_r, sigma_r = FIDCalculator.compute_statistics(real_sample)
        mu_g, sigma_g = FIDCalculator.compute_statistics(gen_sample)
        
        fid = FIDCalculator.calculate_frechet_distance(mu_r, sigma_r, mu_g, sigma_g)
        bootstrap_fids.append(fid)
    
    # Compute confidence interval
    fid_mean = np.mean(bootstrap_fids)
    lower = np.percentile(bootstrap_fids, 2.5)
    upper = np.percentile(bootstrap_fids, 97.5)
    
    return fid_mean, lower, upper
```

## FID vs IS: Comparative Analysis

| Aspect | FID | IS |
|--------|-----|-----|
| **Distribution comparison** | Compares two distributions | Only evaluates generated |
| **Mode collapse detection** | Sensitive (via covariance) | Less sensitive |
| **Sample size requirement** | Higher (10K+) | Lower (5K+) |
| **Reference dataset needed** | Yes | No |
| **Computation** | More expensive (covariance) | Faster |
| **Theoretical foundation** | Optimal transport | Information theory |

### When to Use Which

- **FID**: Primary metric for image generation quality
- **IS**: Quick sanity check, useful when no reference dataset available
- **Both**: Report both for comprehensive evaluation

## Limitations and Pitfalls

### 1. Gaussian Assumption

FID assumes features follow Gaussian distributions. This may fail for:

- Highly multi-modal feature spaces
- Small sample sizes
- Out-of-domain images

### 2. Inception Bias

FID is tied to InceptionV3's learned representations:

```python
def demonstrate_inception_bias():
    """
    Show that FID depends on feature extractor choice.
    """
    # FID with InceptionV3 vs VGG vs CLIP would give different values
    # The "correct" FID depends on what semantic similarity means
    print("Different feature extractors give different FIDs:")
    print("- InceptionV3: Standard choice, trained on ImageNet")
    print("- CLIP: Better for text-to-image evaluation")
    print("- SwAV: Self-supervised features, less class-biased")
```

### 3. Preprocessing Sensitivity

Inconsistent preprocessing leads to incorrect FID:

```python
# BAD: Inconsistent preprocessing
real_images = resize(real_images, 299)  # Bilinear
gen_images = resize(gen_images, 299)    # Different method!

# GOOD: Consistent preprocessing
def consistent_preprocess(images):
    return F.interpolate(images, size=(299, 299), 
                        mode='bilinear', align_corners=False)
```

### 4. FID Cannot Detect Everything

FID may miss:

- Subtle artifacts (blur, noise patterns)
- Memorization (copying training data)
- Perceptual issues that don't affect statistics

## Best Practices

### 1. Use Established Libraries

```python
# Recommended: torch-fidelity
from torch_fidelity import calculate_metrics

metrics = calculate_metrics(
    input1='path/to/real',
    input2='path/to/generated',
    cuda=True,
    fid=True,
    verbose=True
)
print(f"FID: {metrics['frechet_inception_distance']}")
```

### 2. Pre-compute Reference Statistics

```python
# Save real data statistics once
save_reference_statistics(real_images, 'cifar10_stats.npz')

# Reuse during training
mu_real, sigma_real = load_reference_statistics('cifar10_stats.npz')

for epoch in range(epochs):
    # Generate samples
    fake_images = generator.sample(10000)
    
    # Compute FID efficiently
    fid = calculator.calculate_fid_from_statistics(
        mu_real, sigma_real, fake_images
    )
    print(f"Epoch {epoch}: FID = {fid:.2f}")
```

### 3. Report with Context

```python
def report_fid(fid: float, n_real: int, n_gen: int):
    """Properly report FID with context."""
    print(f"FID: {fid:.2f}")
    print(f"  Real samples: {n_real:,}")
    print(f"  Generated samples: {n_gen:,}")
    print(f"  Feature extractor: InceptionV3 (ImageNet)")
    print(f"  Preprocessing: 299×299, bilinear, ImageNet normalization")
```

## Summary

!!! success "Key Takeaways"
    
    1. **FID Formula**: $\text{FID} = \|\mu_r - \mu_g\|^2 + \text{Tr}(\Sigma_r + \Sigma_g - 2(\Sigma_r\Sigma_g)^{1/2})$
    
    2. **Interpretation**: Lower FID = more similar to real data distribution
    
    3. **Typical values**: Excellent (<5), Good (5-20), Moderate (20-50), Poor (>100)
    
    4. **Sample requirements**: Minimum 2048, recommended 10,000+
    
    5. **Best practices**: Use established libraries, pre-compute statistics, ensure consistent preprocessing

## References

1. Heusel, M., et al. (2017). "GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium." *NeurIPS*.

2. Parmar, G., et al. (2021). "On Aliased Resizing and Surprising Subtleties in GAN Evaluation." *CVPR*.

3. Bińkowski, M., et al. (2018). "Demystifying MMD GANs." *ICLR*.

4. Chong, M. J., & Forsyth, D. (2020). "Effectively Unbiased FID and Inception Score and Where to Find Them." *CVPR*.
