# Interpolation and Latent Arithmetic

Exploring the structure of learned latent spaces through vector arithmetic, smooth interpolation, and semantic direction discovery.

---

## Overview

**What you'll learn:**

- Latent space vector arithmetic: analogies and transformations
- Smooth interpolation between data points via latent codes
- Discovering semantic directions in latent space
- Spherical vs linear interpolation (slerp vs lerp)
- Evaluating interpolation quality and smoothness

---

## Mathematical Foundation

### Latent Space Arithmetic

If the latent space is well-structured, meaningful arithmetic operations are possible:

**Vector arithmetic (analogies):**

$$z_{\text{result}} = z_A + (z_B - z_C)$$

For example, in MNIST: $z_{\text{digit 0}} + (z_{\text{digit 8}} - z_{\text{digit 1}})$ might produce something resembling a digit that has the "roundness" of 8 applied to the structure of 0.

**Interpolation:**

$$z_t = (1-t) \cdot z_1 + t \cdot z_2, \quad t \in [0, 1]$$

Decoding $z_t$ for different values of $t$ should produce smooth transitions between the reconstructions of $z_1$ and $z_2$.

**Class averaging:**

$$\bar{z}_c = \frac{1}{n_c} \sum_{i: y_i = c} z_i$$

The average latent vector for a class captures the "prototype" representation.

---

## Part 1: Latent Space Vector Arithmetic

```python
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def collect_class_latents(model, test_loader, device, num_per_class=10):
    """
    Collect latent representations organized by class.
    Returns dict: {class_label: mean_latent_vector}
    """
    model.eval()
    
    digit_examples = {i: [] for i in range(10)}
    
    with torch.no_grad():
        for images, labels in test_loader:
            images_flat = images.view(images.size(0), -1).to(device)
            z = model.encode(images_flat)
            
            for i in range(10):
                mask = labels == i
                if mask.sum() > 0 and len(digit_examples[i]) < num_per_class:
                    digit_examples[i].extend(
                        z[mask][:num_per_class - len(digit_examples[i])].cpu()
                    )
            
            if all(len(v) >= num_per_class for v in digit_examples.values()):
                break
    
    # Compute mean latent vector per class
    class_latents = {}
    for i in range(10):
        stacked = torch.stack(digit_examples[i])
        class_latents[i] = stacked.mean(dim=0, keepdim=True).to(device)
    
    return class_latents


def demonstrate_latent_arithmetic(model, test_loader, device):
    """
    Demonstrate arithmetic operations in latent space.
    
    Example: z_0 + (z_8 - z_1) transforms digit 0 by adding
    the difference between 8 and 1 — potentially adding
    "roundness" features.
    """
    class_latents = collect_class_latents(model, test_loader, device)
    
    # Define arithmetic operations
    operations = [
        ("z_0 + (z_8 - z_1)", 0, 8, 1),
        ("z_1 + (z_7 - z_4)", 1, 7, 4),
        ("z_3 + (z_9 - z_7)", 3, 9, 7),
        ("z_5 + (z_0 - z_6)", 5, 0, 6),
    ]
    
    fig, axes = plt.subplots(len(operations), 5, figsize=(12, 3 * len(operations)))
    
    for row, (name, base, add, sub) in enumerate(operations):
        z_base = class_latents[base]
        z_add = class_latents[add]
        z_sub = class_latents[sub]
        
        z_result = z_base + (z_add - z_sub)
        
        with torch.no_grad():
            img_base = model.decode(z_base).cpu().numpy().reshape(28, 28)
            img_add = model.decode(z_add).cpu().numpy().reshape(28, 28)
            img_sub = model.decode(z_sub).cpu().numpy().reshape(28, 28)
            img_result = model.decode(z_result).cpu().numpy().reshape(28, 28)
        
        axes[row, 0].imshow(img_base, cmap='gray')
        axes[row, 0].set_title(f'z_{base}')
        
        axes[row, 1].text(0.5, 0.5, '+', fontsize=24, ha='center', va='center',
                         transform=axes[row, 1].transAxes)
        axes[row, 1].set_title(f'(z_{add} - z_{sub})')
        
        axes[row, 2].imshow(img_add, cmap='gray')
        axes[row, 2].set_title(f'z_{add}')
        
        axes[row, 3].imshow(img_sub, cmap='gray')
        axes[row, 3].set_title(f'z_{sub}')
        
        axes[row, 4].imshow(img_result, cmap='gray')
        axes[row, 4].set_title('Result')
        
        for ax in axes[row]:
            ax.axis('off')
    
    plt.suptitle('Latent Space Vector Arithmetic')
    plt.tight_layout()
    plt.savefig('latent_arithmetic.png', dpi=150)
    plt.show()
```

---

## Part 2: Smooth Interpolation

### Linear Interpolation (Lerp)

```python
def interpolate_linear(model, z_start, z_end, num_steps=10):
    """
    Linear interpolation between two latent points.
    
    z_t = (1-t) * z_start + t * z_end
    """
    model.eval()
    
    interpolations = []
    
    with torch.no_grad():
        for t in np.linspace(0, 1, num_steps):
            z_interp = (1 - t) * z_start + t * z_end
            img = model.decode(z_interp).cpu().numpy().reshape(28, 28)
            interpolations.append(img)
    
    return interpolations


def interpolate_spherical(model, z_start, z_end, num_steps=10):
    """
    Spherical linear interpolation (slerp) between two latent points.
    
    Slerp follows the great circle on the hypersphere, which can
    produce smoother transitions than linear interpolation when
    latent codes have similar norms but different directions.
    
    slerp(z₁, z₂, t) = sin((1-t)θ)/sin(θ) * z₁ + sin(tθ)/sin(θ) * z₂
    where θ = arccos(z₁·z₂ / (||z₁|| ||z₂||))
    """
    model.eval()
    
    # Normalize
    z1 = z_start.squeeze()
    z2 = z_end.squeeze()
    
    z1_norm = z1 / (torch.norm(z1) + 1e-8)
    z2_norm = z2 / (torch.norm(z2) + 1e-8)
    
    # Angle between vectors
    cos_theta = torch.clamp(torch.dot(z1_norm, z2_norm), -1, 1)
    theta = torch.acos(cos_theta)
    
    interpolations = []
    
    with torch.no_grad():
        for t in np.linspace(0, 1, num_steps):
            if theta.item() < 1e-6:
                # Vectors are nearly identical, use linear
                z_interp = (1 - t) * z_start + t * z_end
            else:
                # Slerp
                w1 = torch.sin((1 - t) * theta) / torch.sin(theta)
                w2 = torch.sin(t * theta) / torch.sin(theta)
                z_interp = (w1 * z_start + w2 * z_end)
            
            img = model.decode(z_interp).cpu().numpy().reshape(28, 28)
            interpolations.append(img)
    
    return interpolations


def visualize_interpolation(model, test_loader, device, 
                            digit_pairs=[(0, 8), (1, 7), (3, 5)]):
    """
    Visualize interpolation between digit pairs using
    both linear and spherical interpolation.
    """
    class_latents = collect_class_latents(model, test_loader, device)
    
    num_steps = 10
    num_pairs = len(digit_pairs)
    
    fig, axes = plt.subplots(num_pairs * 2, num_steps, 
                             figsize=(num_steps * 1.5, num_pairs * 3))
    
    for pair_idx, (d1, d2) in enumerate(digit_pairs):
        z1 = class_latents[d1]
        z2 = class_latents[d2]
        
        # Linear interpolation
        linear_imgs = interpolate_linear(model, z1, z2, num_steps)
        for i, img in enumerate(linear_imgs):
            row = pair_idx * 2
            axes[row, i].imshow(img, cmap='gray')
            axes[row, i].axis('off')
        axes[row, 0].set_ylabel(f'{d1}→{d2}\nLinear', fontsize=8)
        
        # Spherical interpolation
        slerp_imgs = interpolate_spherical(model, z1, z2, num_steps)
        for i, img in enumerate(slerp_imgs):
            row = pair_idx * 2 + 1
            axes[row, i].imshow(img, cmap='gray')
            axes[row, i].axis('off')
        axes[row, 0].set_ylabel(f'{d1}→{d2}\nSlerp', fontsize=8)
    
    plt.suptitle('Linear vs Spherical Interpolation')
    plt.tight_layout()
    plt.savefig('interpolation_comparison.png', dpi=150)
    plt.show()
```

---

## Part 3: Semantic Direction Discovery

```python
def discover_semantic_directions(model, test_loader, device):
    """
    Discover meaningful directions in latent space by computing
    difference vectors between class centroids.
    
    Each direction d_ij = mean(z_i) - mean(z_j) captures the
    transformation from class j to class i.
    """
    class_latents = collect_class_latents(model, test_loader, device, 
                                          num_per_class=50)
    
    # Compute all pairwise directions
    directions = {}
    for i in range(10):
        for j in range(10):
            if i != j:
                directions[(i, j)] = class_latents[i] - class_latents[j]
    
    return directions


def apply_semantic_direction(model, z_source, direction, 
                              strengths=[-2, -1, 0, 1, 2]):
    """
    Apply a semantic direction to a source latent code
    at various strengths.
    """
    model.eval()
    
    images = []
    with torch.no_grad():
        for alpha in strengths:
            z_modified = z_source + alpha * direction
            img = model.decode(z_modified).cpu().numpy().reshape(28, 28)
            images.append(img)
    
    return images


def visualize_semantic_directions(model, test_loader, device):
    """
    Visualize discovered semantic directions applied to different digits.
    """
    class_latents = collect_class_latents(model, test_loader, device, 
                                          num_per_class=50)
    directions = discover_semantic_directions(model, test_loader, device)
    
    # Select interesting directions
    selected = [(8, 1), (0, 6), (9, 7)]  # (target, source) pairs
    source_digits = [3, 5, 2]  # Digits to apply direction to
    
    strengths = np.linspace(-1.5, 1.5, 7)
    
    fig, axes = plt.subplots(len(selected) * len(source_digits), 
                             len(strengths),
                             figsize=(len(strengths) * 1.5, 
                                      len(selected) * len(source_digits) * 1.5))
    
    row = 0
    for (tgt, src) in selected:
        direction = directions[(tgt, src)]
        
        for digit in source_digits:
            z_base = class_latents[digit]
            
            with torch.no_grad():
                for col, alpha in enumerate(strengths):
                    z_mod = z_base + alpha * direction
                    img = model.decode(z_mod).cpu().numpy().reshape(28, 28)
                    axes[row, col].imshow(img, cmap='gray')
                    axes[row, col].axis('off')
            
            axes[row, 0].set_ylabel(f'd_{digit}\n({tgt}-{src})', fontsize=7)
            row += 1
    
    plt.suptitle('Semantic Directions Applied to Different Digits')
    plt.tight_layout()
    plt.savefig('semantic_directions.png', dpi=150)
    plt.show()
```

---

## Part 4: Interpolation Quality Metrics

```python
def measure_interpolation_smoothness(model, z_start, z_end, 
                                      num_steps=50):
    """
    Measure smoothness of interpolation by computing the variance
    of step-to-step reconstruction changes.
    
    Smooth interpolation: low variance (uniform step sizes in pixel space).
    Non-smooth: high variance (sudden jumps).
    """
    model.eval()
    
    reconstructions = []
    with torch.no_grad():
        for t in np.linspace(0, 1, num_steps):
            z_interp = (1 - t) * z_start + t * z_end
            recon = model.decode(z_interp).cpu().numpy().flatten()
            reconstructions.append(recon)
    
    reconstructions = np.array(reconstructions)
    
    # Step-to-step L2 distances
    step_distances = np.array([
        np.linalg.norm(reconstructions[i+1] - reconstructions[i])
        for i in range(len(reconstructions) - 1)
    ])
    
    smoothness = {
        'mean_step_distance': np.mean(step_distances),
        'std_step_distance': np.std(step_distances),
        'max_step_distance': np.max(step_distances),
        'coefficient_of_variation': np.std(step_distances) / (np.mean(step_distances) + 1e-8)
    }
    
    return smoothness


def compare_interpolation_methods(model, test_loader, device):
    """
    Compare linear and spherical interpolation quality
    across multiple digit pairs.
    """
    class_latents = collect_class_latents(model, test_loader, device)
    
    pairs = [(0, 1), (2, 7), (4, 9), (3, 8), (5, 6)]
    
    print("Interpolation Smoothness Comparison")
    print("-" * 60)
    print(f"{'Pair':<10} {'Linear CV':<15} {'Slerp CV':<15} {'Better'}")
    print("-" * 60)
    
    for d1, d2 in pairs:
        z1 = class_latents[d1]
        z2 = class_latents[d2]
        
        linear_smooth = measure_interpolation_smoothness(model, z1, z2)
        # For slerp, measure in the same way
        slerp_smooth = measure_interpolation_smoothness(model, z1, z2)
        
        better = "Linear" if linear_smooth['coefficient_of_variation'] < \
                 slerp_smooth['coefficient_of_variation'] else "Slerp"
        
        print(f"{d1}→{d2:<8} "
              f"{linear_smooth['coefficient_of_variation']:<15.4f} "
              f"{slerp_smooth['coefficient_of_variation']:<15.4f} "
              f"{better}")
```

---

## Quantitative Finance Application

Latent interpolation and arithmetic have direct applications in quantitative finance:

- **Scenario analysis:** Interpolate between market states (e.g., 2008 crisis latent code and 2017 low-vol code) to generate continuous stress scenarios that smoothly transition between regimes
- **Portfolio transformation:** Use vector arithmetic to answer "what if my portfolio had the sector exposure of portfolio B instead of portfolio A?" by computing $z_{\text{my portfolio}} + (z_B - z_A)$ in latent space
- **Synthetic data generation:** Generate realistic synthetic market scenarios by interpolating between historical encoded states, useful for backtesting strategies on scenarios that never occurred but are plausible
- **Semantic factors:** Discover interpretable directions in latent space that correspond to economic factors (e.g., the "risk-on/risk-off" direction, the "rates sensitivity" direction)

---

## Exercises

### Exercise 1: Arithmetic Quality
Perform vector arithmetic across all digit pairs and evaluate whether the results are recognizable. Which operations work best?

### Exercise 2: Linear vs Spherical
Compare linear and spherical interpolation for multiple digit pairs. Compute smoothness metrics. Under what conditions does slerp outperform lerp?

### Exercise 3: Semantic Direction Catalog
Compute all 90 pairwise directions between digit classes. Apply each direction to a fixed reference digit. Which directions produce the most interpretable transformations?

### Exercise 4: Interpolation in Different Architectures
Compare interpolation quality across vanilla, denoising, and sparse autoencoders. Which architecture produces the smoothest interpolations?

---

## Summary

| Operation | Formula | Use Case |
|-----------|---------|----------|
| **Vector arithmetic** | $z_A + (z_B - z_C)$ | Analogies, style transfer |
| **Linear interpolation** | $(1-t) z_1 + t z_2$ | Simple smooth transitions |
| **Spherical interpolation** | Slerp$(z_1, z_2, t)$ | Better for normalized latents |
| **Semantic directions** | $\bar{z}_i - \bar{z}_j$ | Controllable transformations |

**Key Insight:** The quality of latent interpolation and arithmetic is a direct measure of how well-structured the latent space is. Smooth, semantically meaningful transitions indicate that the autoencoder has learned a latent geometry that captures the underlying data manifold, rather than just memorizing training examples.
