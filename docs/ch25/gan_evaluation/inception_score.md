# Inception Score (IS)

## Overview

The Inception Score (IS) is one of the most widely used metrics for evaluating generative models, particularly Generative Adversarial Networks (GANs). Introduced by Salimans et al. (2016), IS provides a single scalar value that captures both the quality and diversity of generated images.

!!! info "Learning Objectives"
    By the end of this section, you will be able to:
    
    - Understand the mathematical foundation of Inception Score
    - Implement IS computation from scratch in PyTorch
    - Interpret IS values correctly and understand their limitations
    - Apply IS in practical evaluation workflows

## Mathematical Foundation

### Core Formula

The Inception Score is defined as:

$$
\text{IS} = \exp\left(\mathbb{E}_{x \sim p_g}\left[D_{KL}(p(y|x) \| p(y))\right]\right)
$$

where:

- $x$ is a generated image sampled from the generator distribution $p_g$
- $p(y|x)$ is the conditional class distribution given image $x$ (from InceptionV3)
- $p(y) = \mathbb{E}_{x}[p(y|x)]$ is the marginal class distribution
- $D_{KL}$ is the Kullback-Leibler divergence

### Intuition Behind the Components

**Conditional Distribution $p(y|x)$:**

This represents how confident the Inception classifier is about the image's class. A sharp, peaked distribution indicates the classifier is confident—suggesting the image contains a clear, recognizable object.

$$
p(y|x) = \text{softmax}(f_{\text{Inception}}(x))
$$

where $f_{\text{Inception}}(x)$ returns logits for 1000 ImageNet classes.

**Marginal Distribution $p(y)$:**

This is the average class distribution across all generated images:

$$
p(y) = \frac{1}{N}\sum_{i=1}^{N} p(y|x_i)
$$

A uniform marginal distribution indicates the generator produces diverse images covering many classes.

**KL Divergence:**

The KL divergence measures how much the conditional distribution differs from the marginal:

$$
D_{KL}(p(y|x) \| p(y)) = \sum_{c=1}^{C} p(y=c|x) \log\frac{p(y=c|x)}{p(y=c)}
$$

### What IS Actually Measures

| Component | High Value Indicates | Low Value Indicates |
|-----------|---------------------|---------------------|
| $p(y\|x)$ entropy | Uncertain predictions | Confident predictions (quality) |
| $p(y)$ entropy | Diverse classes (diversity) | Mode collapse |
| KL divergence | Both quality AND diversity | Poor quality OR low diversity |

The IS captures both aspects simultaneously:

- **Quality**: Each image should produce a confident classification (low entropy in $p(y|x)$)
- **Diversity**: Generated images should cover many classes (high entropy in $p(y)$)

## Mathematical Derivation

### Expanding the KL Divergence

Starting from the definition:

$$
\begin{aligned}
D_{KL}(p(y|x) \| p(y)) &= \sum_{y} p(y|x) \log\frac{p(y|x)}{p(y)} \\
&= \sum_{y} p(y|x) \log p(y|x) - \sum_{y} p(y|x) \log p(y) \\
&= -H(y|x) + H_{\text{cross}}(p(y|x), p(y))
\end{aligned}
$$

where $H(y|x)$ is the conditional entropy.

### Expected Value

Taking the expectation over generated samples:

$$
\mathbb{E}_x[D_{KL}(p(y|x) \| p(y))] = -\mathbb{E}_x[H(y|x)] + H(y)
$$

The first term represents **average conditional entropy** (lower is better for quality), and the second term is the **marginal entropy** (higher is better for diversity).

### Final Score

$$
\text{IS} = \exp\left(H(y) - \mathbb{E}_x[H(y|x)]\right)
$$

This can be interpreted as the **effective number of classes** the generator can produce with confident predictions.

## PyTorch Implementation

### Complete Implementation from Scratch

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
from scipy import stats


class InceptionScoreCalculator:
    """
    Comprehensive Inception Score calculator with detailed documentation.
    
    The Inception Score measures both quality and diversity of generated images
    by analyzing the class predictions of a pre-trained Inception network.
    """
    
    def __init__(self, 
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize the IS calculator.
        
        Args:
            device: Computation device ('cuda' or 'cpu')
        """
        self.device = device
        self.inception_model = None
        
    def _load_inception(self):
        """Load InceptionV3 model pre-trained on ImageNet."""
        from torchvision.models import inception_v3, Inception_V3_Weights
        
        # Load pre-trained InceptionV3
        self.inception_model = inception_v3(
            weights=Inception_V3_Weights.IMAGENET1K_V1,
            transform_input=False  # We'll handle preprocessing ourselves
        )
        self.inception_model.eval()
        self.inception_model.to(self.device)
        
        # Disable auxiliary outputs
        self.inception_model.aux_logits = False
        
    def _preprocess_images(self, images: torch.Tensor) -> torch.Tensor:
        """
        Preprocess images for InceptionV3.
        
        InceptionV3 expects:
        - Images of size 299×299
        - Normalized with ImageNet mean and std
        
        Args:
            images: Input images [B, C, H, W] in range [0, 1]
            
        Returns:
            Preprocessed images ready for Inception
        """
        # Resize to 299×299 if needed
        if images.shape[2] != 299 or images.shape[3] != 299:
            images = F.interpolate(
                images, 
                size=(299, 299), 
                mode='bilinear', 
                align_corners=False
            )
        
        # Convert grayscale to RGB if needed
        if images.shape[1] == 1:
            images = images.repeat(1, 3, 1, 1)
        
        # Normalize with ImageNet statistics
        # Note: Inception expects [-1, 1] range internally
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(images.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(images.device)
        
        images = (images - mean) / std
        
        return images
    
    def get_predictions(self, 
                       images: torch.Tensor, 
                       batch_size: int = 32) -> np.ndarray:
        """
        Get Inception predictions for a batch of images.
        
        Args:
            images: Generated images [N, C, H, W] in range [0, 1]
            batch_size: Batch size for processing
            
        Returns:
            Softmax probabilities [N, 1000]
        """
        if self.inception_model is None:
            self._load_inception()
            
        all_probs = []
        n_images = len(images)
        
        with torch.no_grad():
            for i in range(0, n_images, batch_size):
                batch = images[i:i+batch_size].to(self.device)
                batch = self._preprocess_images(batch)
                
                # Forward pass through Inception
                logits = self.inception_model(batch)
                
                # Apply softmax to get probabilities
                probs = F.softmax(logits, dim=1)
                all_probs.append(probs.cpu().numpy())
        
        return np.concatenate(all_probs, axis=0)
    
    def calculate_inception_score(self,
                                  images: torch.Tensor,
                                  splits: int = 10,
                                  batch_size: int = 32) -> Tuple[float, float]:
        """
        Calculate Inception Score with confidence intervals.
        
        Algorithm:
        1. Get p(y|x) from Inception for each image
        2. Split data into `splits` groups for computing variance
        3. For each split:
           a. Compute marginal p(y) = mean(p(y|x))
           b. Compute KL(p(y|x) || p(y)) for each sample
           c. Average KL and exponentiate
        4. Return mean and std across splits
        
        Args:
            images: Generated images [N, C, H, W] in range [0, 1]
            splits: Number of splits for computing std
            batch_size: Batch size for Inception inference
            
        Returns:
            Tuple of (IS mean, IS std)
        """
        # Get predictions
        probs = self.get_predictions(images, batch_size)
        
        # Calculate IS with splits
        scores = []
        n = len(probs)
        split_size = n // splits
        
        for k in range(splits):
            # Get split
            start = k * split_size
            end = start + split_size if k < splits - 1 else n
            part = probs[start:end]
            
            # Compute marginal: p(y) = (1/N) Σ p(y|x_i)
            p_y = np.mean(part, axis=0, keepdims=True)
            
            # Compute KL divergence for each sample
            # KL(p(y|x) || p(y)) = Σ p(y|x) * log(p(y|x) / p(y))
            eps = 1e-16
            part = np.clip(part, eps, 1.0)
            p_y = np.clip(p_y, eps, 1.0)
            
            # Log ratio
            log_ratio = np.log(part) - np.log(p_y)
            
            # KL divergence per sample
            kl_per_sample = np.sum(part * log_ratio, axis=1)
            
            # Average KL and exponentiate
            mean_kl = np.mean(kl_per_sample)
            is_score = np.exp(mean_kl)
            
            scores.append(is_score)
        
        return float(np.mean(scores)), float(np.std(scores))


def compute_inception_score_step_by_step(probs: np.ndarray) -> dict:
    """
    Compute IS with detailed intermediate results for educational purposes.
    
    This function breaks down the IS computation into interpretable steps,
    making it easier to understand what each component measures.
    
    Args:
        probs: Class probabilities [N, C] from Inception
        
    Returns:
        Dictionary containing intermediate values and final IS
    """
    eps = 1e-16
    probs = np.clip(probs, eps, 1.0)
    
    # Step 1: Compute marginal distribution p(y)
    # This represents the overall class distribution across all samples
    p_y = np.mean(probs, axis=0)
    
    # Step 2: Compute entropy of marginal H(y)
    # Higher entropy means more diverse samples (covering more classes)
    h_marginal = -np.sum(p_y * np.log(p_y))
    
    # Step 3: Compute conditional entropy H(y|x) for each sample
    # Lower conditional entropy means more confident predictions (higher quality)
    h_conditional_per_sample = -np.sum(probs * np.log(probs), axis=1)
    h_conditional = np.mean(h_conditional_per_sample)
    
    # Step 4: Compute KL divergence
    # KL(p(y|x) || p(y)) = H(y) - H(y|x) in expectation
    # But we compute it directly for accuracy
    kl_per_sample = np.sum(probs * (np.log(probs) - np.log(p_y)), axis=1)
    mean_kl = np.mean(kl_per_sample)
    
    # Step 5: Final IS = exp(mean_kl)
    inception_score = np.exp(mean_kl)
    
    # Additional insights
    effective_classes = np.exp(h_marginal)  # Effective number of classes used
    avg_confidence = np.exp(-h_conditional)  # Average prediction confidence
    
    return {
        'inception_score': inception_score,
        'mean_kl_divergence': mean_kl,
        'marginal_entropy': h_marginal,
        'conditional_entropy': h_conditional,
        'effective_classes': effective_classes,
        'avg_confidence': avg_confidence,
        'marginal_distribution': p_y
    }
```

### Practical Usage Example

```python
import torch
import matplotlib.pyplot as plt


def demonstrate_inception_score():
    """
    Demonstrate IS computation with different quality scenarios.
    """
    n_samples = 1000
    n_classes = 10  # Simplified for demonstration
    
    print("=" * 70)
    print("Inception Score Demonstration")
    print("=" * 70)
    
    # Scenario 1: High quality + High diversity (Ideal)
    print("\n📊 Scenario 1: High Quality + High Diversity")
    print("-" * 50)
    
    probs_ideal = np.zeros((n_samples, n_classes))
    for i in range(n_samples):
        class_idx = i % n_classes  # Uniform coverage
        probs_ideal[i, class_idx] = 0.9
        probs_ideal[i, :] += 0.01  # Small uniform noise
    probs_ideal = probs_ideal / probs_ideal.sum(axis=1, keepdims=True)
    
    results_ideal = compute_inception_score_step_by_step(probs_ideal)
    print(f"  IS: {results_ideal['inception_score']:.2f}")
    print(f"  Effective classes: {results_ideal['effective_classes']:.2f}")
    print(f"  Average confidence: {results_ideal['avg_confidence']:.4f}")
    
    # Scenario 2: Low quality (uncertain predictions)
    print("\n📊 Scenario 2: Low Quality (Uncertain Predictions)")
    print("-" * 50)
    
    probs_uncertain = np.ones((n_samples, n_classes)) / n_classes
    
    results_uncertain = compute_inception_score_step_by_step(probs_uncertain)
    print(f"  IS: {results_uncertain['inception_score']:.2f}")
    print(f"  Effective classes: {results_uncertain['effective_classes']:.2f}")
    print(f"  Average confidence: {results_uncertain['avg_confidence']:.4f}")
    print("  Note: Minimum IS = 1.0 when all predictions are uniform")
    
    # Scenario 3: Mode collapse (only one class)
    print("\n📊 Scenario 3: Mode Collapse (Single Class)")
    print("-" * 50)
    
    probs_collapse = np.zeros((n_samples, n_classes))
    probs_collapse[:, 0] = 0.95
    probs_collapse[:, 1:] = 0.05 / (n_classes - 1)
    
    results_collapse = compute_inception_score_step_by_step(probs_collapse)
    print(f"  IS: {results_collapse['inception_score']:.2f}")
    print(f"  Effective classes: {results_collapse['effective_classes']:.2f}")
    print(f"  Note: Confident but not diverse!")
    
    return {
        'ideal': results_ideal,
        'uncertain': results_uncertain,
        'collapse': results_collapse
    }


# Run demonstration
results = demonstrate_inception_score()
```

## Interpreting IS Values

### Typical Ranges

| IS Value | Quality Level | Interpretation |
|----------|---------------|----------------|
| < 2.0 | Very Poor | Images unrecognizable or highly uncertain predictions |
| 2.0 - 5.0 | Poor to Moderate | Some structure but limited quality or diversity |
| 5.0 - 8.0 | Good | Clear images with reasonable diversity |
| > 8.0 | Excellent | High-quality, diverse image generation |
| ~11.2 | Real ImageNet | Benchmark from real ImageNet images |

### Theoretical Bounds

**Minimum IS = 1.0**: Achieved when $p(y|x) = p(y)$ for all $x$ (uniform predictions).

**Maximum IS**: Theoretically bounded by the number of classes (1000 for ImageNet), achieved when each image is perfectly classified into a unique class.

## Limitations and Pitfalls

### 1. Cannot Detect Memorization

IS cannot distinguish between a model that generates novel images and one that simply memorizes training data:

```python
def demonstrate_memorization_blindness():
    """
    Shows that IS cannot detect if a model memorizes training data.
    """
    # A model that generates the same 10 images perfectly
    # will still achieve high IS if those images are classified confidently
    n_unique = 10
    n_total = 1000
    
    probs_memorized = np.zeros((n_total, 10))
    for i in range(n_total):
        class_idx = i % n_unique  # Only 10 unique "images"
        probs_memorized[i, class_idx] = 0.95
        probs_memorized[i, :] += 0.005
    
    probs_memorized = probs_memorized / probs_memorized.sum(axis=1, keepdims=True)
    results = compute_inception_score_step_by_step(probs_memorized)
    
    print(f"IS with memorization: {results['inception_score']:.2f}")
    print("This is HIGH despite only 10 unique images!")
```

### 2. Ignores Within-Class Diversity

IS only measures class-level diversity, not visual diversity within classes:

- 1000 identical cat images → High IS (confident "cat" classification)
- But zero visual diversity!

### 3. Dataset Dependency

IS is only meaningful for ImageNet-like natural images. It may fail for:

- Medical images
- Satellite imagery
- Abstract art
- Domain-specific images

### 4. Can Be Gamed

Adversarial strategies can artificially inflate IS:

```python
def demonstrate_gaming_is():
    """
    Shows how IS can be 'gamed' with adversarial strategies.
    """
    # Strategy: Generate exactly one image per class
    n_classes = 1000
    probs_gamed = np.eye(n_classes)  # Perfect classification for each class
    
    results = compute_inception_score_step_by_step(probs_gamed)
    print(f"Gamed IS: {results['inception_score']:.2f}")
    print("Maximum possible IS with only 1000 unique images!")
```

## Best Practices

### 1. Sample Size

```python
def analyze_sample_size_effect(generator, sample_sizes=[100, 500, 1000, 5000, 10000]):
    """
    Analyze how sample size affects IS stability.
    """
    calculator = InceptionScoreCalculator()
    
    results = []
    for n in sample_sizes:
        images = generator.generate(n)
        is_mean, is_std = calculator.calculate_inception_score(images)
        results.append({
            'n_samples': n,
            'is_mean': is_mean,
            'is_std': is_std,
            'relative_std': is_std / is_mean
        })
    
    return results
```

**Recommendations:**

- Minimum: 5,000 samples
- Recommended: 10,000+ samples
- Always report confidence intervals

### 2. Splits for Variance Estimation

```python
# Standard practice: 10 splits
is_mean, is_std = calculator.calculate_inception_score(images, splits=10)

# Report as: IS = mean ± std
print(f"IS = {is_mean:.2f} ± {is_std:.2f}")
```

### 3. Combine with Other Metrics

IS should never be used alone. Always combine with:

- **FID**: Detects mode collapse better
- **Precision/Recall**: Measures quality vs coverage tradeoff
- **Visual inspection**: Human judgment remains essential

## Connection to Information Theory

The IS has a beautiful information-theoretic interpretation:

$$
\text{IS} = \exp\left(I(X; Y)\right)
$$

where $I(X; Y)$ is the mutual information between generated images $X$ and their predicted classes $Y$.

**Mutual information decomposes as:**

$$
I(X; Y) = H(Y) - H(Y|X)
$$

- **$H(Y)$**: Entropy of class predictions (diversity)
- **$H(Y|X)$**: Average uncertainty in predictions (quality)

Higher mutual information means:
- The generated images carry more information about class labels
- Both quality and diversity contribute positively

## Summary

!!! success "Key Takeaways"
    
    1. **IS Formula**: $\text{IS} = \exp(\mathbb{E}[D_{KL}(p(y|x) \| p(y))])$
    
    2. **Measures Both**: Quality (confident predictions) and diversity (class coverage)
    
    3. **Range**: 1.0 (minimum) to ~1000 (theoretical max), real ImageNet ≈ 11.2
    
    4. **Limitations**: Cannot detect memorization, ignores within-class diversity, ImageNet-specific
    
    5. **Best Practice**: Use 10,000+ samples, 10 splits, combine with FID and visual inspection

## References

1. Salimans, T., et al. (2016). "Improved Techniques for Training GANs." *NeurIPS*.

2. Barratt, S., & Sharma, R. (2018). "A Note on the Inception Score." *ICML Workshop*.

3. Borji, A. (2019). "Pros and Cons of GAN Evaluation Measures." *Computer Vision and Image Understanding*.
