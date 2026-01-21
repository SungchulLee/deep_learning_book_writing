# Why Scale by $\sqrt{d_k}$?

## The Problem

Without scaling, attention scores are computed as:

$$S_{ij} = \mathbf{q}_i^T \mathbf{k}_j$$

As the dimension $d_k$ increases, these dot products **grow in magnitude**, causing the softmax to saturate and gradients to vanish.

## Mathematical Analysis

### Variance of Dot Products

Assume query and key components are independent with:
- $\mathbb{E}[q_\ell] = \mathbb{E}[k_\ell] = 0$
- $\text{Var}(q_\ell) = \text{Var}(k_\ell) = 1$

The dot product is:

$$\mathbf{q}^T \mathbf{k} = \sum_{\ell=1}^{d_k} q_\ell k_\ell$$

Each term $q_\ell k_\ell$ has:
- $\mathbb{E}[q_\ell k_\ell] = \mathbb{E}[q_\ell]\mathbb{E}[k_\ell] = 0$ (by independence)
- $\text{Var}(q_\ell k_\ell) = \mathbb{E}[q_\ell^2]\mathbb{E}[k_\ell^2] = 1 \cdot 1 = 1$

For the sum:

$$\text{Var}(\mathbf{q}^T \mathbf{k}) = \sum_{\ell=1}^{d_k} \text{Var}(q_\ell k_\ell) = d_k$$

**Key insight**: The variance grows linearly with dimension:

$$\mathbf{q}^T \mathbf{k} \sim \mathcal{N}(0, d_k)$$

### The Scaling Fix

Dividing by $\sqrt{d_k}$:

$$\text{Var}\left(\frac{\mathbf{q}^T \mathbf{k}}{\sqrt{d_k}}\right) = \frac{\text{Var}(\mathbf{q}^T \mathbf{k})}{d_k} = \frac{d_k}{d_k} = 1$$

The scaled scores have **unit variance regardless of dimension**.

## Softmax Saturation

### Why Large Inputs Are Problematic

Consider softmax with input $\mathbf{s} = (s_1, s_2, \ldots, s_m)$:

$$\text{softmax}(\mathbf{s})_i = \frac{\exp(s_i)}{\sum_j \exp(s_j)}$$

When inputs have large variance (say, standard deviation $\sigma$):

**If $s_1 \gg s_2, \ldots, s_m$:**

$$\text{softmax}(\mathbf{s}) \approx (1, 0, 0, \ldots, 0)$$

**If scores are comparable but large:**

Small differences get exponentially amplified, leading to near-one-hot outputs.

### Numerical Example

Let $d_k = 512$. Unscaled dot products have standard deviation $\sqrt{512} \approx 22.6$.

Consider attention scores: $\mathbf{s} = (20, 22, 18, 21)$

**Unscaled softmax:**

$$\text{softmax}(20, 22, 18, 21) \approx (0.018, 0.731, 0.002, 0.249)$$

**Scaled (dividing by $\sqrt{512} \approx 22.6$):**

$$\mathbf{s}_{\text{scaled}} = (0.88, 0.97, 0.80, 0.93)$$

$$\text{softmax}(0.88, 0.97, 0.80, 0.93) \approx (0.227, 0.249, 0.210, 0.314)$$

The scaled version has a much smoother distribution, allowing meaningful gradients for all positions.

## Gradient Analysis

### Softmax Gradient

The Jacobian of softmax is:

$$\frac{\partial \text{softmax}(\mathbf{s})_i}{\partial s_j} = \text{softmax}(\mathbf{s})_i \left(\delta_{ij} - \text{softmax}(\mathbf{s})_j\right)$$

### When Softmax Saturates

If $\text{softmax}(\mathbf{s}) \approx (1, 0, \ldots, 0)$:

$$\frac{\partial \text{softmax}(\mathbf{s})_1}{\partial s_j} \approx 1 \cdot (1 - 1) = 0 \quad \text{for } j = 1$$

$$\frac{\partial \text{softmax}(\mathbf{s})_1}{\partial s_j} \approx 1 \cdot (0 - 0) = 0 \quad \text{for } j \neq 1$$

**All gradients vanish!** The model cannot learn to adjust attention weights.

### With Proper Scaling

When softmax outputs are moderate (say, around $1/m$):

$$\frac{\partial \text{softmax}(\mathbf{s})_i}{\partial s_j} \approx \frac{1}{m}\left(\delta_{ij} - \frac{1}{m}\right)$$

Gradients are small but non-zero, enabling learning.

## Visualization

```
Without Scaling (d_k = 512)
Attention Scores:  [-15, 8, 22, -3]     (high variance)
After Softmax:     [0.00, 0.00, 1.00, 0.00]  (saturated)
Gradients:         [~0, ~0, ~0, ~0]     (vanished)

With Scaling (÷ √512 ≈ 22.6)
Attention Scores:  [-0.66, 0.35, 0.97, -0.13]  (unit variance)
After Softmax:     [0.12, 0.32, 0.47, 0.20]    (smooth)
Gradients:         [meaningful values]          (learning happens)
```

## Why $\sqrt{d_k}$ Specifically?

The choice of $\sqrt{d_k}$ achieves **unit variance** for the scores:

| Scaling Factor | Resulting Variance | Effect |
|----------------|-------------------|--------|
| None | $d_k$ | Softmax saturates |
| $d_k$ | $1/d_k$ | Scores too small, near-uniform attention |
| $\sqrt{d_k}$ | $1$ | Just right |

Alternatives:
- Dividing by $d_k$: Scores have variance $1/d_k$, too small
- Dividing by $\sqrt{d_k/2}$: Variance is 2, still manageable but not standard

The $\sqrt{d_k}$ choice is elegant because it normalizes to unit variance assuming standard initialization.

## Connection to Temperature

The scaling can be viewed as setting softmax temperature:

$$\text{softmax}\left(\frac{\mathbf{s}}{\sqrt{d_k}}\right) = \text{softmax}\left(\frac{\mathbf{s}}{T}\right) \text{ where } T = \sqrt{d_k}$$

| Temperature | Behavior |
|-------------|----------|
| $T \to 0$ | Hard attention (argmax) |
| $T = 1$ | Standard softmax |
| $T = \sqrt{d_k}$ | Scaled attention |
| $T \to \infty$ | Uniform attention |

Scaling provides a temperature that adapts to the dimensionality.

## PyTorch Implementation

```python
import torch
import math

def scaled_dot_product_attention(Q, K, V):
    d_k = Q.size(-1)
    
    # Without scaling (problematic for large d_k)
    # scores = torch.matmul(Q, K.transpose(-2, -1))
    
    # With scaling (stable)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    
    attn_weights = torch.softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, V)
    
    return output, attn_weights

# Demonstration
d_k = 512
Q = torch.randn(1, 10, d_k)  # 10 queries
K = torch.randn(1, 10, d_k)  # 10 keys
V = torch.randn(1, 10, 64)   # 10 values

# Check variance of dot products
raw_scores = torch.matmul(Q, K.transpose(-2, -1))
scaled_scores = raw_scores / math.sqrt(d_k)

print(f"Raw scores variance: {raw_scores.var().item():.2f}")      # ≈ 512
print(f"Scaled scores variance: {scaled_scores.var().item():.2f}")  # ≈ 1
```

## Empirical Validation

The original Transformer paper notes:

> "We suspect that for large values of $d_k$, the dot products grow large in magnitude, pushing the softmax function into regions where it has extremely small gradients. To counteract this effect, we scale the dot products by $1/\sqrt{d_k}$."

Ablation studies confirm that without scaling:
- Training becomes unstable for large $d_k$
- Attention patterns collapse to near-uniform or one-hot
- Model performance degrades significantly

## Summary

| Aspect | Without Scaling | With $\sqrt{d_k}$ Scaling |
|--------|-----------------|--------------------------|
| Score variance | $d_k$ (grows with dimension) | $1$ (stable) |
| Softmax behavior | Saturates | Smooth gradients |
| Attention distribution | Near one-hot | Distributed |
| Learning | Vanishing gradients | Stable training |

The $\sqrt{d_k}$ scaling is a simple but crucial detail that enables Transformers to work across different dimensions and depths. It ensures attention scores remain in the regime where softmax produces meaningful gradients.

## References

- Vaswani et al., "Attention Is All You Need" (2017)
