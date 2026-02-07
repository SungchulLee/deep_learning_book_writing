# Precision and Recall for Generative Models

## Overview

Precision and Recall metrics, adapted from classification to generative modeling, provide a nuanced understanding of generation quality that FID alone cannot capture. These metrics separately measure **fidelity** (are generated samples realistic?) and **diversity** (does the model cover all modes of the data distribution?).

!!! info "Learning Objectives"
    By the end of this section, you will be able to:
    
    - Understand why precision/recall is needed beyond FID
    - Implement k-NN based precision and recall computation
    - Interpret the precision-recall tradeoff in generative models
    - Apply improved precision and recall metrics (IPR)
    - Diagnose mode collapse and quality issues using these metrics

## Motivation: Why Beyond FID?

### The Limitations of Single-Number Metrics

FID conflates two distinct failure modes:

| Failure Mode | Description | FID Response |
|--------------|-------------|--------------|
| **Low Fidelity** | Generated samples look fake | FID increases |
| **Low Diversity** | Model only generates a subset of modes | FID increases |

A single FID value cannot distinguish between these fundamentally different problems:

```
Model A: FID = 50 (high-quality but mode collapsed)
Model B: FID = 50 (diverse but low-quality)

Same FID, completely different issues!
```

### Precision and Recall Decomposition

**Precision (Fidelity)**: What fraction of generated samples are realistic?

$$
\text{Precision} = \frac{\text{Generated samples that look real}}{\text{All generated samples}}
$$

**Recall (Coverage)**: What fraction of real data modes does the model cover?

$$
\text{Recall} = \frac{\text{Real data covered by generator}}{\text{All real data}}
$$

## Mathematical Foundation

### Manifold-Based Interpretation

Real data and generated data lie on manifolds in feature space:

- **Real manifold** $\mathcal{M}_r$: The "true" data distribution support
- **Generated manifold** $\mathcal{M}_g$: The generator's output distribution support

**Precision** measures: $P(\mathcal{M}_g \cap \mathcal{M}_r | \mathcal{M}_g)$

**Recall** measures: $P(\mathcal{M}_g \cap \mathcal{M}_r | \mathcal{M}_r)$

### K-Nearest Neighbor Approach

Since we cannot access the true manifolds, we approximate them using k-NN:

**Key Idea**: A point $x$ belongs to a manifold if its k-th nearest neighbor in that manifold is "close enough."

Given:
- Real features: $X_r = \{x_r^1, ..., x_r^{N_r}\}$
- Generated features: $X_g = \{x_g^1, ..., x_g^{N_g}\}$

For each point, compute:

$$
d_k(x, X) = \|x - \text{NN}_k(x, X)\|_2
$$

where $\text{NN}_k(x, X)$ is the k-th nearest neighbor of $x$ in set $X$.

### Precision Definition

A generated sample $x_g$ is considered "real-like" if it falls within the k-NN ball of some real sample:

$$
\text{Precision} = \frac{1}{N_g} \sum_{i=1}^{N_g} \mathbb{1}\left[d_k(x_g^i, X_r) \leq d_k(\text{NN}_k(x_g^i, X_r), X_r)\right]
$$

**Simplified**: A fake sample is realistic if it's close to real samples.

### Recall Definition

Similarly, recall measures how many real samples have a generated sample nearby:

$$
\text{Recall} = \frac{1}{N_r} \sum_{i=1}^{N_r} \mathbb{1}\left[d_k(x_r^i, X_g) \leq d_k(\text{NN}_k(x_r^i, X_g), X_g)\right]
$$

**Simplified**: Recall measures if real data modes are covered by generated samples.

## PyTorch Implementation

### Basic Implementation

```python
import torch
import numpy as np
from typing import Tuple, Optional
from scipy.spatial.distance import cdist


class PrecisionRecallCalculator:
    """
    Precision and Recall metrics for generative models.
    
    Based on Sajjadi et al. (2018) and Kynkäänniemi et al. (2019).
    
    Precision measures fidelity: Are generated samples realistic?
    Recall measures diversity: Does the model cover all modes?
    """
    
    def __init__(self,
                 k: int = 3,
                 row_batch_size: int = 10000,
                 col_batch_size: int = 10000):
        """
        Initialize calculator.
        
        Args:
            k: Number of neighbors for k-NN
            row_batch_size: Batch size for distance computation (rows)
            col_batch_size: Batch size for distance computation (cols)
        """
        self.k = k
        self.row_batch_size = row_batch_size
        self.col_batch_size = col_batch_size
    
    def _batch_pairwise_distances(self,
                                  X: np.ndarray,
                                  Y: np.ndarray) -> np.ndarray:
        """
        Compute pairwise Euclidean distances in batches.
        
        Args:
            X: First set of points [N, D]
            Y: Second set of points [M, D]
            
        Returns:
            Distance matrix [N, M]
        """
        n = len(X)
        m = len(Y)
        distances = np.zeros((n, m), dtype=np.float32)
        
        for i in range(0, n, self.row_batch_size):
            end_i = min(i + self.row_batch_size, n)
            
            for j in range(0, m, self.col_batch_size):
                end_j = min(j + self.col_batch_size, m)
                
                distances[i:end_i, j:end_j] = cdist(
                    X[i:end_i], Y[j:end_j], metric='euclidean'
                )
        
        return distances
    
    def _compute_knn_distances(self,
                              X: np.ndarray,
                              Y: np.ndarray) -> np.ndarray:
        """
        Compute k-th nearest neighbor distances from X to Y.
        
        Args:
            X: Query points [N, D]
            Y: Reference points [M, D]
            
        Returns:
            k-th NN distances for each point in X [N]
        """
        distances = self._batch_pairwise_distances(X, Y)
        
        # Sort and get k-th smallest distance
        # Note: k=0 would be the point itself if X==Y
        kth_distances = np.partition(distances, self.k, axis=1)[:, self.k]
        
        return kth_distances
    
    def compute_precision_recall(self,
                                real_features: np.ndarray,
                                generated_features: np.ndarray) -> Tuple[float, float]:
        """
        Compute Precision and Recall.
        
        Algorithm:
        1. Compute k-NN balls for real data
        2. Precision: fraction of generated falling in real k-NN balls
        3. Compute k-NN balls for generated data
        4. Recall: fraction of real falling in generated k-NN balls
        
        Args:
            real_features: Features from real images [N_r, D]
            generated_features: Features from generated images [N_g, D]
            
        Returns:
            Tuple of (precision, recall)
        """
        print(f"Computing precision/recall with k={self.k}")
        print(f"  Real samples: {len(real_features)}")
        print(f"  Generated samples: {len(generated_features)}")
        
        # Compute manifold radii for real data
        # (distance to k-th nearest neighbor within real data)
        real_nn_distances = self._compute_knn_distances(real_features, real_features)
        
        # Compute distances from generated to real
        gen_to_real_distances = self._compute_knn_distances(generated_features, real_features)
        
        # Precision: generated samples within real manifold
        # For each generated sample, check if its nearest real neighbor
        # is within the k-NN ball of that real sample
        distances_gen_to_real = self._batch_pairwise_distances(
            generated_features, real_features
        )
        nearest_real_idx = np.argmin(distances_gen_to_real, axis=1)
        nearest_real_dist = distances_gen_to_real[np.arange(len(generated_features)), nearest_real_idx]
        
        # Check if within manifold
        precision_mask = nearest_real_dist <= real_nn_distances[nearest_real_idx]
        precision = np.mean(precision_mask)
        
        # Compute manifold radii for generated data
        gen_nn_distances = self._compute_knn_distances(generated_features, generated_features)
        
        # Recall: real samples within generated manifold
        distances_real_to_gen = self._batch_pairwise_distances(
            real_features, generated_features
        )
        nearest_gen_idx = np.argmin(distances_real_to_gen, axis=1)
        nearest_gen_dist = distances_real_to_gen[np.arange(len(real_features)), nearest_gen_idx]
        
        # Check if within manifold
        recall_mask = nearest_gen_dist <= gen_nn_distances[nearest_gen_idx]
        recall = np.mean(recall_mask)
        
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        
        return float(precision), float(recall)


class ImprovedPrecisionRecall:
    """
    Improved Precision and Recall (IPR) from Kynkäänniemi et al. (2019).
    
    Key improvements over original:
    1. Uses hypersphere-based manifold estimation
    2. More robust to outliers
    3. Better handling of sparse regions
    """
    
    def __init__(self,
                 k: int = 3,
                 row_batch_size: int = 10000,
                 col_batch_size: int = 10000):
        """
        Initialize IPR calculator.
        
        Args:
            k: Number of neighbors for manifold estimation
            row_batch_size: Batch size for distance computation
            col_batch_size: Batch size for distance computation
        """
        self.k = k
        self.row_batch_size = row_batch_size
        self.col_batch_size = col_batch_size
    
    def _batch_pairwise_distances(self,
                                  X: np.ndarray,
                                  Y: np.ndarray) -> np.ndarray:
        """Compute pairwise distances in batches."""
        n = len(X)
        m = len(Y)
        distances = np.zeros((n, m), dtype=np.float32)
        
        for i in range(0, n, self.row_batch_size):
            end_i = min(i + self.row_batch_size, n)
            
            for j in range(0, m, self.col_batch_size):
                end_j = min(j + self.col_batch_size, m)
                
                distances[i:end_i, j:end_j] = cdist(
                    X[i:end_i], Y[j:end_j], metric='euclidean'
                )
        
        return distances
    
    def _compute_manifold(self, features: np.ndarray) -> np.ndarray:
        """
        Compute manifold radii using k-NN.
        
        For each point, the manifold radius is the distance to its k-th
        nearest neighbor (excluding itself).
        
        Args:
            features: Feature vectors [N, D]
            
        Returns:
            Manifold radii [N]
        """
        distances = self._batch_pairwise_distances(features, features)
        
        # Set diagonal to infinity (exclude self)
        np.fill_diagonal(distances, np.inf)
        
        # Get k-th smallest distance for each point
        radii = np.partition(distances, self.k - 1, axis=1)[:, self.k - 1]
        
        return radii
    
    def compute_improved_precision_recall(self,
                                         real_features: np.ndarray,
                                         generated_features: np.ndarray) -> Tuple[float, float]:
        """
        Compute Improved Precision and Recall.
        
        Improved P&R uses hypersphere-based manifold estimation:
        - Real manifold: union of hyperspheres around real samples
        - Gen manifold: union of hyperspheres around generated samples
        
        Args:
            real_features: Real image features [N_r, D]
            generated_features: Generated image features [N_g, D]
            
        Returns:
            Tuple of (precision, recall)
        """
        print(f"Computing Improved Precision/Recall (k={self.k})")
        
        # Compute manifold radii
        real_radii = self._compute_manifold(real_features)
        gen_radii = self._compute_manifold(generated_features)
        
        # Compute distances between real and generated
        dist_gen_to_real = self._batch_pairwise_distances(generated_features, real_features)
        dist_real_to_gen = self._batch_pairwise_distances(real_features, generated_features)
        
        # Precision: for each generated sample, check if any real sample
        # has it within its manifold (i.e., distance <= real radius)
        # Shape: [N_g, N_r] - real_radii [N_r]
        in_real_manifold = dist_gen_to_real <= real_radii[np.newaxis, :]
        precision = np.mean(np.any(in_real_manifold, axis=1))
        
        # Recall: for each real sample, check if any generated sample
        # has it within its manifold
        in_gen_manifold = dist_real_to_gen <= gen_radii[np.newaxis, :]
        recall = np.mean(np.any(in_gen_manifold, axis=1))
        
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        
        return float(precision), float(recall)


def compute_density_coverage(real_features: np.ndarray,
                            generated_features: np.ndarray,
                            k: int = 5) -> Tuple[float, float]:
    """
    Compute Density and Coverage metrics (Naeem et al., 2020).
    
    These are alternative formulations of precision/recall:
    - Density: Average number of real neighbors for generated samples
    - Coverage: Fraction of real samples with at least one generated neighbor
    
    Args:
        real_features: Real features [N_r, D]
        generated_features: Generated features [N_g, D]
        k: Number of neighbors
        
    Returns:
        Tuple of (density, coverage)
    """
    from scipy.spatial.distance import cdist
    
    # Compute k-NN radii for real data
    real_to_real = cdist(real_features, real_features, 'euclidean')
    np.fill_diagonal(real_to_real, np.inf)
    real_radii = np.partition(real_to_real, k-1, axis=1)[:, k-1]
    
    # Distance from generated to real
    gen_to_real = cdist(generated_features, real_features, 'euclidean')
    
    # Density: average number of real samples within each generated sample's
    # neighborhood (defined by real manifold)
    in_ball = gen_to_real <= real_radii[np.newaxis, :]
    density = np.mean(np.sum(in_ball, axis=1)) / k
    
    # Coverage: fraction of real samples with at least one generated neighbor
    # within their manifold
    real_to_gen = cdist(real_features, generated_features, 'euclidean')
    covered = np.any(real_to_gen <= real_radii[:, np.newaxis], axis=1)
    coverage = np.mean(covered)
    
    return float(density), float(coverage)
```

### Demonstration

```python
def demonstrate_precision_recall():
    """
    Demonstrate precision/recall with controlled scenarios.
    """
    print("=" * 70)
    print("Precision and Recall Demonstration")
    print("=" * 70)
    
    np.random.seed(42)
    n_samples = 5000
    feature_dim = 128
    
    # Real data: mixture of 4 Gaussian clusters
    cluster_centers = np.array([
        [-2, -2],
        [-2, 2],
        [2, -2],
        [2, 2]
    ])
    
    real_features = []
    for center in cluster_centers:
        cluster = np.random.randn(n_samples // 4, 2) * 0.5 + center
        real_features.append(cluster)
    real_features = np.concatenate(real_features, axis=0)
    
    # Pad to higher dimension (simulating Inception features)
    real_features = np.concatenate([
        real_features,
        np.zeros((len(real_features), feature_dim - 2))
    ], axis=1)
    
    calculator = ImprovedPrecisionRecall(k=3)
    
    # Scenario 1: Ideal generation (covers all modes with quality)
    print("\n📊 Scenario 1: Ideal Generation")
    print("-" * 50)
    
    gen_ideal = []
    for center in cluster_centers:
        cluster = np.random.randn(n_samples // 4, 2) * 0.5 + center
        gen_ideal.append(cluster)
    gen_ideal = np.concatenate(gen_ideal, axis=0)
    gen_ideal = np.concatenate([
        gen_ideal,
        np.zeros((len(gen_ideal), feature_dim - 2))
    ], axis=1)
    
    p1, r1 = calculator.compute_improved_precision_recall(real_features, gen_ideal)
    
    # Scenario 2: Mode collapse (high precision, low recall)
    print("\n📊 Scenario 2: Mode Collapse (only 1 cluster)")
    print("-" * 50)
    
    gen_collapse = np.random.randn(n_samples, 2) * 0.5 + cluster_centers[0]
    gen_collapse = np.concatenate([
        gen_collapse,
        np.zeros((len(gen_collapse), feature_dim - 2))
    ], axis=1)
    
    p2, r2 = calculator.compute_improved_precision_recall(real_features, gen_collapse)
    print("Note: High precision (realistic), low recall (missing modes)")
    
    # Scenario 3: Low quality (low precision, high recall)
    print("\n📊 Scenario 3: Low Quality (noisy but diverse)")
    print("-" * 50)
    
    gen_noisy = []
    for center in cluster_centers:
        cluster = np.random.randn(n_samples // 4, 2) * 2.0 + center  # High noise
        gen_noisy.append(cluster)
    gen_noisy = np.concatenate(gen_noisy, axis=0)
    gen_noisy = np.concatenate([
        gen_noisy,
        np.zeros((len(gen_noisy), feature_dim - 2))
    ], axis=1)
    
    p3, r3 = calculator.compute_improved_precision_recall(real_features, gen_noisy)
    print("Note: Low precision (unrealistic), high recall (covers modes)")
    
    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"{'Scenario':<25} {'Precision':>12} {'Recall':>12}")
    print("-" * 50)
    print(f"{'Ideal':<25} {p1:>12.4f} {r1:>12.4f}")
    print(f"{'Mode Collapse':<25} {p2:>12.4f} {r2:>12.4f}")
    print(f"{'Low Quality':<25} {p3:>12.4f} {r3:>12.4f}")
    
    return {
        'ideal': (p1, r1),
        'collapse': (p2, r2),
        'noisy': (p3, r3)
    }


demonstrate_precision_recall()
```

## Interpreting Precision and Recall

### The Precision-Recall Tradeoff

Generative models often exhibit a tradeoff:

```
Precision ↑  ←→  Recall ↓  (Conservative generation)
Precision ↓  ←→  Recall ↑  (Diverse but noisy)
```

### Diagnostic Matrix

| Precision | Recall | Diagnosis |
|-----------|--------|-----------|
| High | High | **Ideal**: Realistic and diverse |
| High | Low | **Mode collapse**: Good quality but missing modes |
| Low | High | **Low fidelity**: Covers data but poor quality |
| Low | Low | **Failure**: Neither realistic nor diverse |

### Visualizing the Tradeoff

```python
import matplotlib.pyplot as plt


def plot_precision_recall_tradeoff(models_results: dict):
    """
    Plot precision-recall for multiple models.
    
    Args:
        models_results: Dict mapping model name to (precision, recall)
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    for name, (precision, recall) in models_results.items():
        ax.scatter(recall, precision, s=100, label=name)
        ax.annotate(name, (recall + 0.02, precision + 0.02))
    
    # Reference lines
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(0.5, color='gray', linestyle='--', alpha=0.5)
    
    # Quadrant labels
    ax.text(0.25, 0.75, 'Mode Collapse\n(good quality, low diversity)', 
            ha='center', va='center', fontsize=10, alpha=0.7)
    ax.text(0.75, 0.75, 'Ideal\n(good quality, high diversity)', 
            ha='center', va='center', fontsize=10, alpha=0.7)
    ax.text(0.25, 0.25, 'Failure\n(poor quality, low diversity)', 
            ha='center', va='center', fontsize=10, alpha=0.7)
    ax.text(0.75, 0.25, 'Low Fidelity\n(poor quality, high diversity)', 
            ha='center', va='center', fontsize=10, alpha=0.7)
    
    ax.set_xlabel('Recall (Coverage)', fontsize=12)
    ax.set_ylabel('Precision (Fidelity)', fontsize=12)
    ax.set_title('Precision-Recall Tradeoff', fontsize=14)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig
```

## Relationship to FID

### How P&R Complements FID

```python
def compare_fid_and_pr():
    """
    Show how P&R provides insights FID cannot.
    """
    # Two models with same FID but different P&R profiles
    
    # Model A: Mode collapse (high P, low R)
    # Model B: Low quality (low P, high R)
    
    print("Two models with similar FID:")
    print("-" * 40)
    print("Model A (Mode Collapse):")
    print("  FID = 25, Precision = 0.95, Recall = 0.30")
    print("\nModel B (Low Quality):")
    print("  FID = 25, Precision = 0.30, Recall = 0.95")
    print("\n→ FID alone cannot distinguish these cases!")
    print("→ P&R reveals the specific failure mode.")
```

### Using Both Metrics

**Recommended evaluation approach:**

1. **FID**: Overall quality score for quick comparison
2. **Precision**: Detect quality degradation
3. **Recall**: Detect mode collapse

```python
def comprehensive_evaluation(real_features, gen_features):
    """
    Comprehensive evaluation using multiple metrics.
    """
    from fid_calculator import FIDCalculator
    
    # FID
    fid_calc = FIDCalculator()
    mu_r, sigma_r = fid_calc.compute_statistics(real_features)
    mu_g, sigma_g = fid_calc.compute_statistics(gen_features)
    fid = fid_calc.calculate_frechet_distance(mu_r, sigma_r, mu_g, sigma_g)
    
    # Precision and Recall
    pr_calc = ImprovedPrecisionRecall(k=3)
    precision, recall = pr_calc.compute_improved_precision_recall(
        real_features, gen_features
    )
    
    # Report
    print("Comprehensive Evaluation:")
    print(f"  FID: {fid:.2f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    
    # Diagnosis
    if fid < 20 and precision > 0.7 and recall > 0.7:
        print("  Diagnosis: Excellent generation")
    elif precision > 0.7 and recall < 0.5:
        print("  Diagnosis: Mode collapse detected")
    elif precision < 0.5 and recall > 0.7:
        print("  Diagnosis: Low quality but good coverage")
    else:
        print("  Diagnosis: Needs improvement")
    
    return fid, precision, recall
```

## Advanced: F1 Score and F-beta

### Combining Precision and Recall

When a single number is needed, use F-scores:

$$
F_\beta = (1 + \beta^2) \cdot \frac{\text{Precision} \cdot \text{Recall}}{\beta^2 \cdot \text{Precision} + \text{Recall}}
$$

**Common choices:**

- $F_1$: Equal weight to precision and recall
- $F_{0.5}$: Emphasize precision (quality-focused)
- $F_2$: Emphasize recall (coverage-focused)

```python
def compute_f_score(precision: float, recall: float, beta: float = 1.0) -> float:
    """
    Compute F-beta score.
    
    Args:
        precision: Precision value
        recall: Recall value
        beta: Weight parameter (beta > 1 favors recall)
        
    Returns:
        F-beta score
    """
    if precision + recall == 0:
        return 0.0
    
    beta_sq = beta ** 2
    f_score = (1 + beta_sq) * precision * recall / (beta_sq * precision + recall)
    
    return f_score


# Example usage
precision, recall = 0.8, 0.6

f1 = compute_f_score(precision, recall, beta=1.0)
f05 = compute_f_score(precision, recall, beta=0.5)
f2 = compute_f_score(precision, recall, beta=2.0)

print(f"F1 (balanced): {f1:.4f}")
print(f"F0.5 (precision-focused): {f05:.4f}")
print(f"F2 (recall-focused): {f2:.4f}")
```

## Best Practices

### 1. Choose k Carefully

```python
def analyze_k_sensitivity(real_features, gen_features, k_values=[1, 3, 5, 10, 20]):
    """
    Analyze sensitivity to k parameter.
    """
    results = []
    
    for k in k_values:
        calc = ImprovedPrecisionRecall(k=k)
        p, r = calc.compute_improved_precision_recall(real_features, gen_features)
        results.append({'k': k, 'precision': p, 'recall': r})
        print(f"k={k:2d}: P={p:.4f}, R={r:.4f}")
    
    return results
```

**Recommendations:**

- Default: k=3 (robust choice)
- Sparse data: k=1 (but noisy)
- Dense data: k=5-10 (smoother)

### 2. Sample Size Considerations

- Minimum: 5,000 samples for stable estimates
- Recommended: 10,000+ samples
- Report results with confidence intervals

### 3. Feature Space Choice

```python
# Standard: InceptionV3 (same as FID)
from torchvision.models import inception_v3

# Alternative for specific domains:
# - CLIP features for text-to-image
# - Domain-specific networks for medical/satellite images
```

## Summary

!!! success "Key Takeaways"
    
    1. **Precision measures fidelity**: Are generated samples realistic?
    
    2. **Recall measures diversity**: Does the model cover all data modes?
    
    3. **Key diagnostics**:
       - High P, Low R → Mode collapse
       - Low P, High R → Low quality
       - Both high → Excellent generation
    
    4. **Use alongside FID**: P&R provides insights FID cannot
    
    5. **Recommended settings**: k=3, 10,000+ samples, InceptionV3 features

## References

1. Sajjadi, M.S.M., et al. (2018). "Assessing Generative Models via Precision and Recall." *NeurIPS*.

2. Kynkäänniemi, T., et al. (2019). "Improved Precision and Recall Metric for Assessing Generative Models." *NeurIPS*.

3. Naeem, M.F., et al. (2020). "Reliable Fidelity and Diversity Metrics for Generative Models." *ICML*.

4. Simon, L., et al. (2019). "Revisiting Precision and Recall Definition for Generative Model Evaluation." *ICML*.
