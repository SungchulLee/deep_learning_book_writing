# Jacobian of Softmax

## Learning Objectives

By the end of this section, you will be able to:

- Derive the Jacobian matrix of the softmax function from first principles
- Understand the structure and properties of the softmax Jacobian
- Implement Jacobian computation in PyTorch
- Apply the Jacobian in backpropagation analysis
- Connect the Jacobian to gradient flow in neural networks

---

## Prerequisites: Calculus Review

### Partial Derivatives and the Jacobian

For a vector-valued function $\mathbf{f}: \mathbb{R}^n \to \mathbb{R}^m$, the **Jacobian matrix** $\mathbf{J} \in \mathbb{R}^{m \times n}$ contains all partial derivatives:

$$\mathbf{J}_{ij} = \frac{\partial f_i}{\partial x_j}$$

The Jacobian represents the best linear approximation to $\mathbf{f}$ near a point:

$$\mathbf{f}(\mathbf{x} + \Delta\mathbf{x}) \approx \mathbf{f}(\mathbf{x}) + \mathbf{J} \cdot \Delta\mathbf{x}$$

### The Chain Rule for Vector Functions

For composed functions $\mathbf{h} = \mathbf{g} \circ \mathbf{f}$, the chain rule states:

$$\mathbf{J}_{\mathbf{h}} = \mathbf{J}_{\mathbf{g}} \cdot \mathbf{J}_{\mathbf{f}}$$

This is the foundation of backpropagation.

---

## Softmax Function Recap

The softmax function $\boldsymbol{\sigma}: \mathbb{R}^K \to \mathbb{R}^K$ is defined as:

$$\sigma_i(\mathbf{z}) = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}} = \frac{e^{z_i}}{S}$$

where we define $S = \sum_{j=1}^{K} e^{z_j}$ for convenience.

**Key property:** The output depends on ALL inputs (not just $z_i$), so all partial derivatives are non-zero.

---

## Deriving the Jacobian

### Case 1: Diagonal Elements ($i = j$)

We need $\frac{\partial \sigma_i}{\partial z_i}$ when differentiating with respect to the same index.

Using the quotient rule for $\sigma_i = \frac{e^{z_i}}{S}$:

$$\frac{\partial \sigma_i}{\partial z_i} = \frac{\frac{\partial}{\partial z_i}(e^{z_i}) \cdot S - e^{z_i} \cdot \frac{\partial S}{\partial z_i}}{S^2}$$

Computing the derivatives:
- $\frac{\partial}{\partial z_i}(e^{z_i}) = e^{z_i}$
- $\frac{\partial S}{\partial z_i} = \frac{\partial}{\partial z_i}\sum_j e^{z_j} = e^{z_i}$

Substituting:

$$\frac{\partial \sigma_i}{\partial z_i} = \frac{e^{z_i} \cdot S - e^{z_i} \cdot e^{z_i}}{S^2} = \frac{e^{z_i}}{S} - \frac{e^{2z_i}}{S^2}$$

$$= \frac{e^{z_i}}{S}\left(1 - \frac{e^{z_i}}{S}\right) = \sigma_i(1 - \sigma_i)$$

$$\boxed{\frac{\partial \sigma_i}{\partial z_i} = \sigma_i(1 - \sigma_i)}$$

### Case 2: Off-Diagonal Elements ($i \neq j$)

We need $\frac{\partial \sigma_i}{\partial z_j}$ when differentiating with respect to a different index.

Again using the quotient rule for $\sigma_i = \frac{e^{z_i}}{S}$:

$$\frac{\partial \sigma_i}{\partial z_j} = \frac{\frac{\partial}{\partial z_j}(e^{z_i}) \cdot S - e^{z_i} \cdot \frac{\partial S}{\partial z_j}}{S^2}$$

Computing the derivatives:
- $\frac{\partial}{\partial z_j}(e^{z_i}) = 0$ (since $i \neq j$)
- $\frac{\partial S}{\partial z_j} = e^{z_j}$

Substituting:

$$\frac{\partial \sigma_i}{\partial z_j} = \frac{0 \cdot S - e^{z_i} \cdot e^{z_j}}{S^2} = -\frac{e^{z_i}}{S} \cdot \frac{e^{z_j}}{S}$$

$$\boxed{\frac{\partial \sigma_i}{\partial z_j} = -\sigma_i \sigma_j \quad (i \neq j)}$$

---

## The Complete Jacobian Matrix

### Unified Formula

Combining both cases using the Kronecker delta $\delta_{ij}$ (1 if $i=j$, 0 otherwise):

$$\frac{\partial \sigma_i}{\partial z_j} = \sigma_i(\delta_{ij} - \sigma_j)$$

### Matrix Form

The Jacobian matrix $\mathbf{J}_\sigma \in \mathbb{R}^{K \times K}$ can be written as:

$$\mathbf{J}_\sigma = \text{diag}(\boldsymbol{\sigma}) - \boldsymbol{\sigma} \boldsymbol{\sigma}^T$$

where:
- $\text{diag}(\boldsymbol{\sigma})$ is a diagonal matrix with $\sigma_i$ on the diagonal
- $\boldsymbol{\sigma} \boldsymbol{\sigma}^T$ is the outer product of $\boldsymbol{\sigma}$ with itself

### Explicit Form (K=3 Example)

For 3 classes with $\boldsymbol{\sigma} = (\sigma_1, \sigma_2, \sigma_3)^T$:

$$\mathbf{J}_\sigma = \begin{pmatrix}
\sigma_1(1-\sigma_1) & -\sigma_1\sigma_2 & -\sigma_1\sigma_3 \\
-\sigma_2\sigma_1 & \sigma_2(1-\sigma_2) & -\sigma_2\sigma_3 \\
-\sigma_3\sigma_1 & -\sigma_3\sigma_2 & \sigma_3(1-\sigma_3)
\end{pmatrix}$$

---

## Properties of the Softmax Jacobian

### Property 1: Symmetry

$$\mathbf{J}_\sigma = \mathbf{J}_\sigma^T$$

The Jacobian is symmetric because $\frac{\partial \sigma_i}{\partial z_j} = -\sigma_i\sigma_j = -\sigma_j\sigma_i = \frac{\partial \sigma_j}{\partial z_i}$.

### Property 2: Row/Column Sums Equal Zero

$$\sum_{j=1}^K \frac{\partial \sigma_i}{\partial z_j} = \sigma_i - \sigma_i \sum_{j=1}^K \sigma_j = \sigma_i - \sigma_i \cdot 1 = 0$$

This reflects the constraint that $\sum_i \sigma_i = 1$ is invariant—increasing one probability must decrease others.

### Property 3: Positive Semi-Definiteness

$\mathbf{J}_\sigma$ is positive semi-definite (all eigenvalues $\geq 0$).

**Proof sketch:** For any $\mathbf{v}$:
$$\mathbf{v}^T \mathbf{J}_\sigma \mathbf{v} = \sum_i \sigma_i v_i^2 - \left(\sum_i \sigma_i v_i\right)^2 \geq 0$$
by the Cauchy-Schwarz inequality.

### Property 4: Rank Deficiency

$\text{rank}(\mathbf{J}_\sigma) = K - 1$

Since the rows (and columns) sum to zero, $\mathbf{1}$ is in the null space: $\mathbf{J}_\sigma \mathbf{1} = \mathbf{0}$.

### Property 5: Eigenvalue Structure

- One eigenvalue is 0 (corresponding to eigenvector $\mathbf{1}$)
- All other eigenvalues are positive
- The maximum eigenvalue is bounded by $\frac{1}{4}$ (achieved when $\sigma_i = 0.5$)

---

## PyTorch Implementation

### Computing the Jacobian

```python
import torch
import torch.nn.functional as F

def softmax_jacobian(z: torch.Tensor) -> torch.Tensor:
    """
    Compute the Jacobian matrix of softmax.
    
    Args:
        z: Logits tensor of shape (K,) for a single sample
    
    Returns:
        Jacobian matrix of shape (K, K)
    
    Formula: J[i,j] = σ_i(δ_ij - σ_j) = diag(σ) - σσᵀ
    """
    sigma = F.softmax(z, dim=0)
    K = len(sigma)
    
    # Method 1: Using the formula directly
    diag_sigma = torch.diag(sigma)
    outer_product = sigma.unsqueeze(1) @ sigma.unsqueeze(0)  # σσᵀ
    jacobian = diag_sigma - outer_product
    
    return jacobian

def softmax_jacobian_batched(z: torch.Tensor) -> torch.Tensor:
    """
    Compute softmax Jacobian for a batch.
    
    Args:
        z: Logits tensor of shape (batch_size, K)
    
    Returns:
        Jacobian tensor of shape (batch_size, K, K)
    """
    sigma = F.softmax(z, dim=1)  # (batch, K)
    batch_size, K = sigma.shape
    
    # Create diagonal matrices: (batch, K, K)
    diag_sigma = torch.diag_embed(sigma)
    
    # Outer product: σσᵀ for each sample
    outer_product = sigma.unsqueeze(2) @ sigma.unsqueeze(1)  # (batch, K, 1) @ (batch, 1, K)
    
    return diag_sigma - outer_product

# Example usage
z = torch.tensor([2.0, 1.0, 0.5])
J = softmax_jacobian(z)
sigma = F.softmax(z, dim=0)

print("Logits z:", z.numpy())
print("Softmax σ:", sigma.numpy().round(4))
print("\nJacobian matrix:")
print(J.numpy().round(4))
```

### Verifying Properties

```python
def verify_jacobian_properties(z: torch.Tensor):
    """Verify theoretical properties of the softmax Jacobian."""
    J = softmax_jacobian(z)
    sigma = F.softmax(z, dim=0)
    K = len(z)
    
    print("=" * 60)
    print("SOFTMAX JACOBIAN PROPERTY VERIFICATION")
    print("=" * 60)
    
    # Property 1: Symmetry
    is_symmetric = torch.allclose(J, J.T, atol=1e-6)
    print(f"\n1. Symmetry: J = Jᵀ? {is_symmetric}")
    
    # Property 2: Row/column sums = 0
    row_sums = J.sum(dim=1)
    col_sums = J.sum(dim=0)
    print(f"2. Row sums: {row_sums.numpy().round(6)}")
    print(f"   Column sums: {col_sums.numpy().round(6)}")
    
    # Property 3: Positive semi-definiteness
    eigenvalues = torch.linalg.eigvalsh(J)
    print(f"3. Eigenvalues: {eigenvalues.numpy().round(6)}")
    print(f"   All ≥ 0? {(eigenvalues >= -1e-6).all().item()}")
    
    # Property 4: Rank deficiency
    rank = torch.linalg.matrix_rank(J).item()
    print(f"4. Rank: {rank} (expected: {K-1})")
    
    # Property 5: Null space
    null_vec = torch.ones(K)
    J_times_ones = J @ null_vec
    print(f"5. J @ 1 = {J_times_ones.numpy().round(6)} (should be 0)")
    
    # Additional: diagonal elements = σᵢ(1-σᵢ)
    expected_diag = sigma * (1 - sigma)
    print(f"\n6. Diagonal verification:")
    print(f"   Actual diagonal: {torch.diag(J).numpy().round(6)}")
    print(f"   σᵢ(1-σᵢ):        {expected_diag.numpy().round(6)}")

# Run verification
z = torch.tensor([2.0, 1.0, 0.5, -0.5])
verify_jacobian_properties(z)
```

### Using PyTorch Autograd for Verification

```python
def jacobian_via_autograd(z: torch.Tensor) -> torch.Tensor:
    """
    Compute Jacobian using PyTorch's automatic differentiation.
    
    This serves as a verification of our analytical formula.
    """
    z = z.clone().requires_grad_(True)
    
    # Compute Jacobian using torch.autograd.functional.jacobian
    def softmax_fn(z):
        return F.softmax(z, dim=0)
    
    J = torch.autograd.functional.jacobian(softmax_fn, z)
    return J

# Compare analytical vs autograd
z = torch.tensor([2.0, 1.0, 0.5])
J_analytical = softmax_jacobian(z)
J_autograd = jacobian_via_autograd(z)

print("Analytical Jacobian:")
print(J_analytical.numpy().round(6))
print("\nAutograd Jacobian:")
print(J_autograd.numpy().round(6))
print(f"\nMatrices match: {torch.allclose(J_analytical, J_autograd, atol=1e-5)}")
```

---

## Application: Vector-Jacobian Products

### Backpropagation Context

In backpropagation, we don't compute the full Jacobian. Instead, we compute the **vector-Jacobian product (VJP)**:

$$\frac{\partial \mathcal{L}}{\partial \mathbf{z}} = \mathbf{J}_\sigma^T \frac{\partial \mathcal{L}}{\partial \boldsymbol{\sigma}} = \mathbf{J}_\sigma \frac{\partial \mathcal{L}}{\partial \boldsymbol{\sigma}}$$

(using symmetry of the softmax Jacobian)

### Efficient VJP Computation

```python
def softmax_vjp(grad_output: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """
    Compute vector-Jacobian product for softmax.
    
    Args:
        grad_output: Gradient from subsequent layer ∂L/∂σ, shape (K,)
        sigma: Softmax output σ, shape (K,)
    
    Returns:
        Gradient w.r.t. logits ∂L/∂z, shape (K,)
    
    Derivation:
        ∂L/∂zⱼ = Σᵢ (∂L/∂σᵢ)(∂σᵢ/∂zⱼ)
               = Σᵢ gᵢ σᵢ(δᵢⱼ - σⱼ)
               = σⱼgⱼ - σⱼΣᵢ gᵢσᵢ
               = σⱼ(gⱼ - Σᵢ gᵢσᵢ)
               = σ ⊙ (g - <g, σ>)
    """
    # <g, σ> = Σᵢ gᵢσᵢ (weighted sum of gradients)
    dot_product = (grad_output * sigma).sum()
    
    # σ ⊙ (g - <g, σ>)
    grad_input = sigma * (grad_output - dot_product)
    
    return grad_input

# Example
z = torch.tensor([2.0, 1.0, 0.5], requires_grad=True)
sigma = F.softmax(z, dim=0)

# Simulate gradient from loss
grad_output = torch.tensor([0.1, -0.3, 0.2])  # ∂L/∂σ

# Our efficient VJP
grad_z_efficient = softmax_vjp(grad_output, sigma.detach())

# Verify with full Jacobian multiplication
J = softmax_jacobian(z.detach())
grad_z_full = J @ grad_output

print("Efficient VJP:", grad_z_efficient.numpy().round(6))
print("Full Jacobian:", grad_z_full.numpy().round(6))
print(f"Match: {torch.allclose(grad_z_efficient, grad_z_full, atol=1e-6)}")
```

---

## Visualization: Jacobian Structure

```python
import matplotlib.pyplot as plt
import numpy as np

def visualize_jacobian(z: torch.Tensor):
    """Visualize the structure of the softmax Jacobian."""
    J = softmax_jacobian(z).numpy()
    sigma = F.softmax(z, dim=0).numpy()
    K = len(z)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Plot 1: Jacobian matrix as heatmap
    im1 = axes[0].imshow(J, cmap='RdBu', vmin=-0.3, vmax=0.3)
    axes[0].set_title('Jacobian Matrix')
    axes[0].set_xlabel('Input logit index j')
    axes[0].set_ylabel('Output prob index i')
    plt.colorbar(im1, ax=axes[0])
    
    # Plot 2: Diagonal elements (variance-like)
    diagonal = np.diag(J)
    axes[1].bar(range(K), diagonal, color='steelblue')
    axes[1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    axes[1].set_title('Diagonal: σᵢ(1-σᵢ)')
    axes[1].set_xlabel('Index i')
    axes[1].set_ylabel('Value')
    
    # Plot 3: Softmax probabilities
    axes[2].bar(range(K), sigma, color='coral')
    axes[2].set_title('Softmax Probabilities')
    axes[2].set_xlabel('Class')
    axes[2].set_ylabel('Probability')
    
    plt.tight_layout()
    return fig

# Create visualization
z = torch.tensor([2.0, 1.0, 0.5, -0.5, 0.0])
# fig = visualize_jacobian(z)
# plt.show()
```

---

## Connection to Gradient of Cross-Entropy Loss

### The Key Result

When softmax is followed by cross-entropy loss with true class $c$:

$$\mathcal{L} = -\log \sigma_c$$

The gradient with respect to logits becomes remarkably simple:

$$\frac{\partial \mathcal{L}}{\partial z_j} = \sigma_j - \mathbb{1}[j = c] = \sigma_j - y_j$$

where $y_j$ is the one-hot encoding of the true class.

### Derivation

Using the chain rule:

$$\frac{\partial \mathcal{L}}{\partial z_j} = \sum_i \frac{\partial \mathcal{L}}{\partial \sigma_i} \frac{\partial \sigma_i}{\partial z_j}$$

For cross-entropy: $\frac{\partial \mathcal{L}}{\partial \sigma_i} = -\frac{y_i}{\sigma_i}$ (only non-zero for $i = c$)

So:

$$\frac{\partial \mathcal{L}}{\partial z_j} = -\frac{1}{\sigma_c} \cdot \frac{\partial \sigma_c}{\partial z_j}$$

For $j = c$ (diagonal):
$$\frac{\partial \mathcal{L}}{\partial z_c} = -\frac{1}{\sigma_c} \cdot \sigma_c(1 - \sigma_c) = -(1 - \sigma_c) = \sigma_c - 1$$

For $j \neq c$ (off-diagonal):
$$\frac{\partial \mathcal{L}}{\partial z_j} = -\frac{1}{\sigma_c} \cdot (-\sigma_c\sigma_j) = \sigma_j$$

Unified: $\frac{\partial \mathcal{L}}{\partial z_j} = \sigma_j - y_j$ ✓

---

## Summary

### The Softmax Jacobian

$$\boxed{\mathbf{J}_\sigma = \text{diag}(\boldsymbol{\sigma}) - \boldsymbol{\sigma}\boldsymbol{\sigma}^T}$$

$$\boxed{\frac{\partial \sigma_i}{\partial z_j} = \sigma_i(\delta_{ij} - \sigma_j) = \begin{cases}
\sigma_i(1 - \sigma_i) & \text{if } i = j \\
-\sigma_i\sigma_j & \text{if } i \neq j
\end{cases}}$$

### Key Properties

| Property | Statement |
|----------|-----------|
| Symmetry | $\mathbf{J} = \mathbf{J}^T$ |
| Sum constraint | Row and column sums = 0 |
| Positive semi-definite | All eigenvalues $\geq 0$ |
| Rank | $K - 1$ (null space = $\mathbf{1}$) |

### Efficient VJP

$$\frac{\partial \mathcal{L}}{\partial \mathbf{z}} = \boldsymbol{\sigma} \odot \left(\frac{\partial \mathcal{L}}{\partial \boldsymbol{\sigma}} - \left\langle \frac{\partial \mathcal{L}}{\partial \boldsymbol{\sigma}}, \boldsymbol{\sigma} \right\rangle \mathbf{1}\right)$$

---

## Next Steps

With the Jacobian understood, you can now fully appreciate:

1. **Gradient Derivation** — Complete gradient computation for softmax regression
2. **PyTorch Implementation** — Understanding what happens inside `backward()`
3. **Advanced Topics** — Second-order methods using the Hessian

---

## References

1. Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*, Chapter 6.2
2. Petersen, K. B., & Pedersen, M. S. (2012). *The Matrix Cookbook*, Section 8.3
3. Baydin, A. G., et al. (2017). *Automatic Differentiation in Machine Learning: A Survey*
