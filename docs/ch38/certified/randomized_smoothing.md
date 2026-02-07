# Certified Robustness

## Introduction

**Certified robustness** provides mathematical guarantees that a classifier's prediction will not change under any perturbation within a specified radius. Unlike empirical defenses (which are tested against specific attacks), certified defenses offer **provable guarantees** that hold against all possible attacks.

## Empirical vs Certified Robustness

| Aspect | Empirical | Certified |
|--------|-----------|-----------|
| **Guarantee** | None (tested against known attacks) | Mathematical proof |
| **Security** | May be broken by new attacks | Provably secure within radius |
| **Accuracy** | Higher | Lower |
| **Certification** | N/A | Computed per-example |

## Randomized Smoothing

### Core Idea

**Randomized smoothing** (Cohen et al., 2019) transforms any classifier into a certifiably robust one by averaging predictions over Gaussian noise.

Given a base classifier $f: \mathbb{R}^d \to \mathcal{Y}$, construct the **smoothed classifier**:

$$
g(\mathbf{x}) = \arg\max_c \mathbb{P}_{\boldsymbol{\epsilon} \sim \mathcal{N}(0, \sigma^2 I)}[f(\mathbf{x} + \boldsymbol{\epsilon}) = c]
$$

**Intuition:** Add Gaussian noise to input, take majority vote over noisy predictions.

### Certification Theorem

**Theorem (Cohen et al., 2019):** If the smoothed classifier $g$ predicts class $c_A$ at input $\mathbf{x}$ with probability $p_A$, and the runner-up class $c_B$ has probability $p_B$, then $g(\mathbf{x}) = c_A$ is certifiably robust within $\ell_2$ radius:

$$
R = \frac{\sigma}{2}\left(\Phi^{-1}(p_A) - \Phi^{-1}(p_B)\right)
$$

where $\Phi^{-1}$ is the inverse CDF of the standard normal distribution.

### Intuition

- $p_A$ is the probability that class $c_A$ wins the majority vote
- If $p_A \gg p_B$, the certified radius $R$ is large (high confidence)
- If $p_A \approx p_B$, the certified radius $R$ is small (low confidence)
- $\sigma$ scales the radius: larger noise = larger certified region

### Certification Algorithm

**Two-stage process:**

1. **Selection:** Find the predicted class
   - Sample $n_0$ noisy predictions
   - Take majority vote to determine $\hat{c}$

2. **Certification:** Estimate radius
   - Sample $n$ more noisy predictions
   - Compute confidence intervals for $p_A$
   - Calculate certified radius $R$

### Monte Carlo Estimation

We estimate probabilities via sampling:

$$
\hat{p}_A = \frac{1}{N} \sum_{i=1}^N \mathbf{1}[f(\mathbf{x} + \boldsymbol{\epsilon}_i) = c_A], \quad \boldsymbol{\epsilon}_i \sim \mathcal{N}(0, \sigma^2 I)
$$

Using **Clopper-Pearson confidence intervals**, we obtain:

$$
\mathbb{P}(p_A \geq \underline{p}_A) \geq 1 - \alpha
$$

This gives certified radius with probability $\geq 1 - \alpha$.

## PyTorch Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.stats import norm, binom
from typing import Optional, Dict, Tuple
from tqdm import tqdm
import math

class RandomizedSmoothing:
    """
    Certified Robustness via Randomized Smoothing.
    
    Provides provable L2 robustness guarantees by smoothing
    predictions with Gaussian noise.
    
    Parameters
    ----------
    base_classifier : nn.Module
        Base classifier to smooth
    sigma : float
        Standard deviation of Gaussian noise
        Larger σ: larger certified radius, lower accuracy
    """
    
    def __init__(
        self,
        base_classifier: nn.Module,
        sigma: float = 0.25,
        device: Optional[torch.device] = None
    ):
        self.base_classifier = base_classifier
        self.sigma = sigma
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.base_classifier.eval()
        self.base_classifier.to(self.device)
        self.num_classes = None
    
    def _sample_predictions(
        self,
        x: torch.Tensor,
        num_samples: int,
        batch_size: int = 1000
    ) -> torch.Tensor:
        """
        Sample predictions under Gaussian noise.
        
        Parameters
        ----------
        x : torch.Tensor
            Single input image, shape (C, H, W)
        num_samples : int
            Number of noisy samples
        batch_size : int
            Batch size for processing
            
        Returns
        -------
        counts : torch.Tensor
            Prediction counts for each class
        """
        with torch.no_grad():
            # Determine number of classes
            if self.num_classes is None:
                test_out = self.base_classifier(x.unsqueeze(0).to(self.device))
                self.num_classes = test_out.shape[1]
            
            counts = torch.zeros(self.num_classes, device=self.device)
            
            num_batches = math.ceil(num_samples / batch_size)
            remaining = num_samples
            
            for _ in range(num_batches):
                current_batch = min(batch_size, remaining)
                remaining -= current_batch
                
                # Repeat input
                batch = x.unsqueeze(0).repeat(current_batch, 1, 1, 1).to(self.device)
                
                # Add Gaussian noise
                noise = torch.randn_like(batch) * self.sigma
                noisy_batch = batch + noise
                
                # Get predictions
                logits = self.base_classifier(noisy_batch)
                predictions = logits.argmax(dim=1)
                
                # Count predictions
                for c in range(self.num_classes):
                    counts[c] += (predictions == c).sum()
            
            return counts
    
    def _lower_confidence_bound(
        self,
        count: int,
        n: int,
        alpha: float
    ) -> float:
        """
        Compute lower confidence bound for binomial proportion.
        Uses Clopper-Pearson (exact) method.
        """
        if count == 0:
            return 0.0
        return binom.ppf(alpha / 2, n, count / n) / n
    
    def _compute_radius(self, p_A: float, p_B: float) -> float:
        """
        Compute certified radius from probabilities.
        
        R = σ/2 * (Φ^{-1}(p_A) - Φ^{-1}(p_B))
        """
        if p_A <= 0.5:
            return 0.0
        
        # Clamp to avoid infinity
        p_A = min(p_A, 0.999999)
        p_B = max(p_B, 0.000001)
        
        radius = (self.sigma / 2) * (norm.ppf(p_A) - norm.ppf(p_B))
        return max(0.0, radius)
    
    def certify(
        self,
        x: torch.Tensor,
        n0: int = 100,
        n: int = 10000,
        alpha: float = 0.001,
        batch_size: int = 1000
    ) -> Tuple[int, float]:
        """
        Certify a single input.
        
        Parameters
        ----------
        x : torch.Tensor
            Input image, shape (C, H, W)
        n0 : int
            Samples for selection phase
        n : int
            Samples for certification phase
        alpha : float
            Confidence level (default: 99.9%)
        batch_size : int
            Batch size for Monte Carlo
            
        Returns
        -------
        predicted_class : int
            Predicted class (-1 if abstain)
        certified_radius : float
            Certified L2 radius (0 if abstain)
        """
        x = x.to(self.device)
        
        # Stage 1: Selection
        counts_selection = self._sample_predictions(x, n0, batch_size)
        top_class = counts_selection.argmax().item()
        
        # Stage 2: Certification
        counts_cert = self._sample_predictions(x, n, batch_size)
        
        # Count for top class
        count_top = counts_cert[top_class].item()
        
        # Lower confidence bound for p_A
        p_A_lower = self._lower_confidence_bound(int(count_top), n, alpha)
        
        # If p_A_lower <= 0.5, we cannot certify
        if p_A_lower <= 0.5:
            return -1, 0.0  # Abstain
        
        # Compute certified radius
        # For binary case, p_B_upper = 1 - p_A_lower
        # For multiclass, use more conservative bound
        radius = self.sigma * norm.ppf(p_A_lower)
        
        return top_class, radius
    
    def certify_batch(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        n: int = 10000,
        alpha: float = 0.001,
        batch_size: int = 1000,
        radii_to_check: list = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
    ) -> Dict[str, float]:
        """
        Certify a batch of images.
        
        Returns
        -------
        results : dict
            - clean_accuracy: Fraction correctly predicted
            - certified_accuracy_r=X: Fraction certified at radius X
            - avg_certified_radius: Average certified radius
        """
        num_images = len(images)
        predictions = []
        radii = []
        
        for i in tqdm(range(num_images), desc="Certifying"):
            pred, radius = self.certify(images[i], n=n, alpha=alpha, batch_size=batch_size)
            predictions.append(pred)
            radii.append(radius)
        
        predictions = torch.tensor(predictions, device=labels.device)
        radii = torch.tensor(radii)
        
        # Metrics
        correct = (predictions == labels)
        abstain = (predictions == -1)
        
        results = {
            'clean_accuracy': correct.float().mean().item(),
            'abstain_rate': abstain.float().mean().item(),
            'avg_certified_radius': radii[correct & ~abstain].mean().item() if (correct & ~abstain).any() else 0.0
        }
        
        # Certified accuracy at different radii
        for r in radii_to_check:
            certified = correct & (radii >= r)
            results[f'certified_accuracy_r={r}'] = certified.float().mean().item()
        
        return results
    
    def predict(
        self,
        x: torch.Tensor,
        n: int = 1000,
        batch_size: int = 500
    ) -> int:
        """Predict class (without certification)."""
        counts = self._sample_predictions(x, n, batch_size)
        return counts.argmax().item()


class SmoothClassifier(nn.Module):
    """
    Wrapper that makes a classifier smooth for training/inference.
    """
    
    def __init__(self, base_classifier: nn.Module, sigma: float, num_samples: int = 1):
        super().__init__()
        self.base_classifier = base_classifier
        self.sigma = sigma
        self.num_samples = num_samples
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with noise augmentation.
        
        During training: single noise sample (for efficiency)
        During inference: average over multiple samples
        """
        if self.training:
            # Single noise sample during training
            noise = torch.randn_like(x) * self.sigma
            return self.base_classifier(x + noise)
        else:
            # Average over multiple samples during inference
            batch_size = x.shape[0]
            outputs = []
            
            for _ in range(self.num_samples):
                noise = torch.randn_like(x) * self.sigma
                outputs.append(self.base_classifier(x + noise))
            
            return torch.stack(outputs).mean(dim=0)
```

### Usage Example

```python
import torchvision
import torchvision.transforms as transforms

# Load model
base_model = torchvision.models.resnet18(num_classes=10)
base_model.load_state_dict(torch.load('cifar10_resnet18.pth'))

# Create smoothed classifier
smoother = RandomizedSmoothing(base_model, sigma=0.25)

# Load test data
transform = transforms.ToTensor()
testset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transform)
test_loader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False)

# Get a batch
images, labels = next(iter(test_loader))

# Certify
results = smoother.certify_batch(
    images[:50], labels[:50],  # Certify 50 examples (slow)
    n=10000,                    # 10k samples per example
    alpha=0.001                 # 99.9% confidence
)

print("Certification Results:")
print(f"  Clean Accuracy: {results['clean_accuracy']:.2%}")
print(f"  Abstain Rate: {results['abstain_rate']:.2%}")
print(f"  Avg Certified Radius: {results['avg_certified_radius']:.4f}")
print(f"  Certified @ r=0.25: {results['certified_accuracy_r=0.25']:.2%}")
print(f"  Certified @ r=0.50: {results['certified_accuracy_r=0.5']:.2%}")
print(f"  Certified @ r=1.00: {results['certified_accuracy_r=1.0']:.2%}")
```

## Training for Certified Robustness

### Gaussian Data Augmentation

The simplest approach: train with Gaussian noise augmentation.

```python
def train_with_noise(model, train_loader, sigma, epochs):
    """Train with Gaussian noise augmentation."""
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    
    for epoch in range(epochs):
        for x, y in train_loader:
            # Add Gaussian noise
            noise = torch.randn_like(x) * sigma
            x_noisy = x + noise
            
            optimizer.zero_grad()
            loss = F.cross_entropy(model(x_noisy), y)
            loss.backward()
            optimizer.step()
```

### Consistency Regularization

Encourage consistent predictions across noise samples:

$$
\mathcal{L} = \mathcal{L}_{\text{CE}} + \lambda \cdot \text{KL}(f(\mathbf{x} + \boldsymbol{\epsilon}_1) \| f(\mathbf{x} + \boldsymbol{\epsilon}_2))
$$

## Key Parameters and Trade-offs

### Noise Level $\sigma$

| $\sigma$ | Certified Radius | Clean Accuracy |
|----------|------------------|----------------|
| 0.12 | Small (~0.25) | High (~85%) |
| 0.25 | Medium (~0.5) | Medium (~75%) |
| 0.50 | Large (~1.0) | Low (~60%) |
| 1.00 | Very large (~2.0) | Very low (~40%) |

**Trade-off:** Larger $\sigma$ = larger certified region, but lower accuracy.

### Sampling Parameters

| Parameter | Value | Effect |
|-----------|-------|--------|
| $n_0$ (selection) | 100 | Higher = more reliable selection |
| $n$ (certification) | 10,000+ | Higher = tighter confidence |
| $\alpha$ (confidence) | 0.001 | Lower = more conservative |

### Computational Cost

Certification is **expensive**:
- $n = 10,000$ samples per input
- Each sample requires full forward pass
- Certifying 10,000 test examples: 100M forward passes

## Comparison: Empirical vs Certified

**CIFAR-10, $\varepsilon = 0.5$ (L2):**

| Method | Clean Acc | Robust Acc | Certified? |
|--------|-----------|------------|------------|
| Standard | 95% | 0% | No |
| PGD-AT | 85% | ~50% | No |
| Randomized Smoothing | 75% | ~60% | **Yes** |

Certified accuracy may exceed empirical robust accuracy because:
- Empirical attacks may not find optimal adversarial examples
- Certification provides guaranteed lower bound

## Limitations

1. **L2 norm only**: Randomized smoothing certifies L2 perturbations, not L∞
2. **Accuracy drop**: Significant clean accuracy reduction
3. **Computational cost**: Slow certification
4. **Limited scalability**: Challenging for large models/datasets

## Advanced Topics

### Certified Robustness for L∞

Approaches like **Interval Bound Propagation (IBP)** provide L∞ certification:

$$
[\underline{z}, \overline{z}] = \text{IBP}(f, [\mathbf{x} - \varepsilon, \mathbf{x} + \varepsilon])
$$

If $\underline{z}_y > \max_{i \neq y} \overline{z}_i$, the prediction is certified.

### Tighter Certificates

- **SmoothAdv**: Adversarial training + smoothing
- **MACER**: Maximize certified radius during training
- **Denoised smoothing**: Train denoiser to improve base accuracy

## Summary

| Concept | Key Point |
|---------|-----------|
| **Randomized smoothing** | Average over Gaussian noise |
| **Certified radius** | $R = \frac{\sigma}{2}(\Phi^{-1}(p_A) - \Phi^{-1}(p_B))$ |
| **Trade-off** | Larger $\sigma$ = larger R, lower accuracy |
| **Guarantee** | Provable for all perturbations $\|\boldsymbol{\delta}\|_2 \leq R$ |

Certified robustness provides the strongest theoretical guarantees, at the cost of accuracy and computational overhead.

## References

1. Cohen, J., Rosenfeld, E., & Kolter, Z. (2019). "Certified Adversarial Robustness via Randomized Smoothing." ICML.
2. Salman, H., et al. (2019). "Provably Robust Deep Learning via Adversarially Trained Smoothed Classifiers." NeurIPS.
3. Zhai, R., et al. (2020). "MACER: Attack-Free and Scalable Robust Training via Maximizing Certified Radius." ICLR.
