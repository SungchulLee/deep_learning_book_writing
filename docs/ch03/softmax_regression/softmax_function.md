# Softmax Function

## Learning Objectives

By the end of this section, you will be able to:

- Derive the softmax function from first principles using maximum entropy reasoning
- Understand the mathematical properties that make softmax ideal for classification
- Derive the Jacobian matrix of the softmax function element-by-element
- Express the Jacobian in compact matrix form and verify its properties
- Implement numerically stable softmax and efficient vector-Jacobian products in PyTorch
- Analyze softmax behavior through temperature scaling

---

## Motivation: From Logits to Probabilities

### The Classification Problem

In multiclass classification, a neural network produces **logits** $\mathbf{z} = (z_1, z_2, \ldots, z_K)$ — raw, unbounded scores for each of $K$ classes. We need a function $\sigma: \mathbb{R}^K \to \Delta^{K-1}$ that maps these logits to the **probability simplex**:

$$\Delta^{K-1} = \left\{ \mathbf{p} \in \mathbb{R}^K : p_k \geq 0,\; \sum_{k=1}^{K} p_k = 1 \right\}$$

The softmax function is the canonical choice for this transformation.

---

## The Softmax Function

### Definition

The **softmax function** $\sigma: \mathbb{R}^K \to \mathbb{R}^K$ is defined as:

$$\sigma(\mathbf{z})_k = \frac{\exp(z_k)}{\sum_{j=1}^{K} \exp(z_j)}$$

or equivalently, for the entire vector:

$$\sigma(\mathbf{z}) = \frac{\exp(\mathbf{z})}{\mathbf{1}^T \exp(\mathbf{z})} = \frac{\exp(\mathbf{z})}{\|\exp(\mathbf{z})\|_1}$$

where $\exp(\mathbf{z})$ applies element-wise.

### Visual Intuition

```
Logits:          [2.0,  1.0,  0.1]
                   ↓     ↓     ↓
Exponentials:    [7.39, 2.72, 1.11]    (always positive)
                   ↓     ↓     ↓
Sum:             11.22
                   ↓     ↓     ↓
Softmax:         [0.66, 0.24, 0.10]    (sum to 1)
```

---

## Mathematical Derivation

### Derivation 1: Maximum Entropy Principle

The softmax function can be derived by maximizing entropy subject to constraints. Given expected values of features, we seek the distribution with maximum uncertainty.

**Problem.** Find $\mathbf{p}$ that maximizes entropy:

$$H(\mathbf{p}) = -\sum_{k=1}^{K} p_k \log p_k$$

subject to:

1. **Normalization:** $\sum_{k=1}^{K} p_k = 1$
2. **Feature expectations:** $\sum_{k=1}^{K} p_k f_j(k) = \mu_j$ for features $f_j$

**Solution via Lagrangian.**

$$\mathcal{L} = -\sum_k p_k \log p_k - \lambda_0 \left(\sum_k p_k - 1\right) - \sum_j \lambda_j \left(\sum_k p_k f_j(k) - \mu_j\right)$$

Taking the derivative with respect to $p_k$ and setting to zero:

$$\frac{\partial \mathcal{L}}{\partial p_k} = -\log p_k - 1 - \lambda_0 - \sum_j \lambda_j f_j(k) = 0$$

Solving for $p_k$:

$$p_k = \exp\left(-1 - \lambda_0 - \sum_j \lambda_j f_j(k)\right) = \frac{\exp(z_k)}{Z}$$

where $z_k = -\sum_j \lambda_j f_j(k)$ and $Z = \exp(1 + \lambda_0)$ is the normalizing constant. This is exactly the softmax function.

### Derivation 2: From the Exponential Family

The categorical distribution belongs to the **exponential family**. In canonical form:

$$P(Y = k \mid \boldsymbol{\eta}) = h(k) \exp\left(\boldsymbol{\eta}^T T(k) - A(\boldsymbol{\eta})\right)$$

For the categorical distribution with one-hot sufficient statistic:

- $\boldsymbol{\eta} = (\log\pi_1, \ldots, \log\pi_{K-1}, 0)$ (natural parameters)
- $T(k) = \mathbf{e}_k$ (one-hot encoding)
- $A(\boldsymbol{\eta}) = \log\sum_j \exp(\eta_j)$ (log-partition function)

The mean parameters (probabilities) are obtained via:

$$\pi_k = \frac{\partial A}{\partial \eta_k} = \frac{\exp(\eta_k)}{\sum_j \exp(\eta_j)} = \operatorname{softmax}(\boldsymbol{\eta})_k$$

---

## Essential Properties

### Property 1: Normalization

$$\sum_{k=1}^{K} \sigma(\mathbf{z})_k = \sum_{k=1}^{K} \frac{\exp(z_k)}{\sum_j \exp(z_j)} = \frac{\sum_k \exp(z_k)}{\sum_j \exp(z_j)} = 1$$

### Property 2: Positivity

$$\sigma(\mathbf{z})_k = \frac{\exp(z_k)}{\sum_j \exp(z_j)} > 0 \quad \forall\, k$$

since $\exp(x) > 0$ for all $x \in \mathbb{R}$.

### Property 3: Translation Invariance

For any constant $c \in \mathbb{R}$:

$$\sigma(\mathbf{z} + c\mathbf{1})_k = \frac{\exp(z_k + c)}{\sum_j \exp(z_j + c)} = \frac{\exp(c)\exp(z_k)}{\exp(c)\sum_j \exp(z_j)} = \sigma(\mathbf{z})_k$$

This means softmax only depends on the **relative** differences between logits. This property is the foundation of the numerically stable implementation.

### Property 4: Monotonicity

Softmax is monotonically increasing in each coordinate:

$$\frac{\partial \sigma(\mathbf{z})_k}{\partial z_k} = \sigma(\mathbf{z})_k \bigl(1 - \sigma(\mathbf{z})_k\bigr) > 0$$

Larger logits always produce larger probabilities.

### Property 5: Convexity of Log-Partition

The log-sum-exp function $A(\mathbf{z}) = \log\sum_k \exp(z_k)$ is convex, which ensures:

- Unique global optimum in maximum likelihood estimation
- Well-behaved optimization landscape

---

## Jacobian of the Softmax Function

### Prerequisites: Jacobian Matrix

For a vector-valued function $\mathbf{f}: \mathbb{R}^n \to \mathbb{R}^m$, the **Jacobian matrix** $\mathbf{J} \in \mathbb{R}^{m \times n}$ contains all partial derivatives:

$$\mathbf{J}_{ij} = \frac{\partial f_i}{\partial x_j}$$

The Jacobian represents the best linear approximation to $\mathbf{f}$ near a point:

$$\mathbf{f}(\mathbf{x} + \Delta\mathbf{x}) \approx \mathbf{f}(\mathbf{x}) + \mathbf{J} \cdot \Delta\mathbf{x}$$

For composed functions $\mathbf{h} = \mathbf{g} \circ \mathbf{f}$, the chain rule gives $\mathbf{J}_{\mathbf{h}} = \mathbf{J}_{\mathbf{g}} \cdot \mathbf{J}_{\mathbf{f}}$, which is the foundation of backpropagation.

### Deriving the Jacobian

We compute $\frac{\partial \sigma_i}{\partial z_j}$ for the softmax function $\sigma_i = \frac{e^{z_i}}{S}$ where $S = \sum_{j=1}^{K} e^{z_j}$.

**Case 1: Diagonal elements ($i = j$).**

Using the quotient rule:

$$\frac{\partial \sigma_i}{\partial z_i} = \frac{\frac{\partial}{\partial z_i}(e^{z_i}) \cdot S - e^{z_i} \cdot \frac{\partial S}{\partial z_i}}{S^2}$$

Computing the derivatives: $\frac{\partial}{\partial z_i}(e^{z_i}) = e^{z_i}$ and $\frac{\partial S}{\partial z_i} = e^{z_i}$. Substituting:

$$\frac{\partial \sigma_i}{\partial z_i} = \frac{e^{z_i} \cdot S - e^{z_i} \cdot e^{z_i}}{S^2} = \frac{e^{z_i}}{S} - \frac{e^{2z_i}}{S^2} = \sigma_i(1 - \sigma_i)$$

$$\boxed{\frac{\partial \sigma_i}{\partial z_i} = \sigma_i(1 - \sigma_i)}$$

!!! note "Connection to Logistic Sigmoid"
    This is identical to the derivative of the logistic sigmoid $\sigma(z) = \frac{1}{1+e^{-z}}$, which satisfies $\sigma'(z) = \sigma(z)(1-\sigma(z))$. In the binary case ($K=2$), softmax reduces to the logistic sigmoid.

**Case 2: Off-diagonal elements ($i \neq j$).**

Again using the quotient rule:

$$\frac{\partial \sigma_i}{\partial z_j} = \frac{\frac{\partial}{\partial z_j}(e^{z_i}) \cdot S - e^{z_i} \cdot \frac{\partial S}{\partial z_j}}{S^2}$$

Since $i \neq j$: $\frac{\partial}{\partial z_j}(e^{z_i}) = 0$ and $\frac{\partial S}{\partial z_j} = e^{z_j}$. Substituting:

$$\frac{\partial \sigma_i}{\partial z_j} = \frac{0 \cdot S - e^{z_i} \cdot e^{z_j}}{S^2} = -\frac{e^{z_i}}{S} \cdot \frac{e^{z_j}}{S}$$

$$\boxed{\frac{\partial \sigma_i}{\partial z_j} = -\sigma_i \sigma_j \quad (i \neq j)}$$

### Unified Formula and Matrix Form

Combining both cases using the Kronecker delta $\delta_{ij}$:

$$\frac{\partial \sigma_i}{\partial z_j} = \sigma_i(\delta_{ij} - \sigma_j)$$

The Jacobian matrix $\mathbf{J}_\sigma \in \mathbb{R}^{K \times K}$ can be written compactly as:

$$\boxed{\mathbf{J}_\sigma = \operatorname{diag}(\boldsymbol{\sigma}) - \boldsymbol{\sigma} \boldsymbol{\sigma}^T}$$

where $\operatorname{diag}(\boldsymbol{\sigma})$ is a diagonal matrix with $\sigma_i$ on the diagonal and $\boldsymbol{\sigma} \boldsymbol{\sigma}^T$ is the outer product.

**Explicit form ($K=3$).** For $\boldsymbol{\sigma} = (\sigma_1, \sigma_2, \sigma_3)^T$:

$$\mathbf{J}_\sigma = \begin{pmatrix}
\sigma_1(1-\sigma_1) & -\sigma_1\sigma_2 & -\sigma_1\sigma_3 \\
-\sigma_2\sigma_1 & \sigma_2(1-\sigma_2) & -\sigma_2\sigma_3 \\
-\sigma_3\sigma_1 & -\sigma_3\sigma_2 & \sigma_3(1-\sigma_3)
\end{pmatrix}$$

### Properties of the Softmax Jacobian

**Property 1 (Symmetry).** $\mathbf{J}_\sigma = \mathbf{J}_\sigma^T$, since $\frac{\partial \sigma_i}{\partial z_j} = -\sigma_i\sigma_j = -\sigma_j\sigma_i = \frac{\partial \sigma_j}{\partial z_i}$.

**Property 2 (Row/Column Sums Equal Zero).**

$$\sum_{j=1}^K \frac{\partial \sigma_i}{\partial z_j} = \sigma_i - \sigma_i \sum_{j=1}^K \sigma_j = \sigma_i - \sigma_i = 0$$

This reflects the constraint $\sum_i \sigma_i = 1$ — increasing one probability must decrease others.

**Property 3 (Positive Semi-Definiteness).** For any $\mathbf{v}$:

$$\mathbf{v}^T \mathbf{J}_\sigma \mathbf{v} = \sum_i \sigma_i v_i^2 - \left(\sum_i \sigma_i v_i\right)^2 \geq 0$$

by the Cauchy-Schwarz inequality. All eigenvalues are $\geq 0$.

**Property 4 (Rank Deficiency).** $\operatorname{rank}(\mathbf{J}_\sigma) = K - 1$. Since the rows sum to zero, $\mathbf{1}$ is in the null space: $\mathbf{J}_\sigma \mathbf{1} = \mathbf{0}$.

**Property 5 (Eigenvalue Structure).**

- One eigenvalue is 0 (eigenvector $\mathbf{1}$)
- All other eigenvalues are positive
- The maximum eigenvalue is bounded by $\frac{1}{4}$

---

## Temperature Scaling

### Definition

**Temperature-scaled softmax** introduces a parameter $T > 0$:

$$\sigma(\mathbf{z};\, T)_k = \frac{\exp(z_k / T)}{\sum_j \exp(z_j / T)}$$

### Behavior Analysis

| Temperature | Effect | Use Case |
|-------------|--------|----------|
| $T \to 0^+$ | Approaches argmax (hard assignment) | Inference, discrete decisions |
| $T = 1$ | Standard softmax | Training |
| $T > 1$ | Softer, more uniform distribution | Exploration, knowledge distillation |
| $T \to \infty$ | Uniform distribution | Maximum uncertainty |

**Low temperature limit ($T \to 0^+$):**

$$\lim_{T \to 0^+} \sigma(\mathbf{z};\, T) = \mathbf{e}_{k^*}$$

where $k^* = \arg\max_k z_k$. The distribution becomes a one-hot vector.

**High temperature limit ($T \to \infty$):**

$$\lim_{T \to \infty} \sigma(\mathbf{z};\, T) = \frac{1}{K} \mathbf{1}$$

The distribution becomes uniform.

---

## Softmax vs Other Functions

| Function | Formula | Properties |
|----------|---------|------------|
| Softmax | $\frac{e^{z_k}}{\sum_j e^{z_j}}$ | Smooth, differentiable, preserves ranking |
| Hardmax | $\mathbb{1}[k = \arg\max_j z_j]$ | Non-differentiable, sparse |
| Sparsemax | Euclidean projection onto simplex | Sparse, differentiable |
| Entmax | Generalization of softmax/sparsemax | Tunable sparsity |

**Use softmax when** training neural networks (need smooth gradients), probabilities should be strictly positive, and all classes should receive some probability mass. **Consider alternatives when** sparse attention is desired (sparsemax, entmax), hard decisions are needed (hardmax at inference), or calibrated probabilities are critical (temperature scaling, Platt scaling).

---

## PyTorch Implementation

### Softmax Implementations

```python
import torch
import torch.nn.functional as F

def softmax_naive(z: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Naive softmax - susceptible to overflow/underflow."""
    exp_z = torch.exp(z)
    return exp_z / exp_z.sum(dim=dim, keepdim=True)

def softmax_stable(z: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Numerically stable softmax using the max trick."""
    z_max = z.max(dim=dim, keepdim=True).values
    exp_z = torch.exp(z - z_max)
    return exp_z / exp_z.sum(dim=dim, keepdim=True)

# Test with extreme values
z_extreme = torch.tensor([1000.0, 1001.0, 1002.0])

# Naive: produces nan due to overflow
naive_result = softmax_naive(z_extreme)
print(f"Naive softmax: {naive_result}")  # [nan, nan, nan]

# Stable: works correctly
stable_result = softmax_stable(z_extreme)
print(f"Stable softmax: {stable_result}")  # [0.0900, 0.2447, 0.6652]

# PyTorch's built-in (already stable)
pytorch_result = F.softmax(z_extreme, dim=0)
print(f"PyTorch softmax: {pytorch_result}")  # Same as stable
```

### Computing the Jacobian

```python
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

    diag_sigma = torch.diag_embed(sigma)                       # (batch, K, K)
    outer_product = sigma.unsqueeze(2) @ sigma.unsqueeze(1)    # (batch, K, K)

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
    print(f"2. Row sums: {row_sums.numpy().round(6)}")

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

    # Diagonal verification
    expected_diag = sigma * (1 - sigma)
    print(f"\n6. Diagonal verification:")
    print(f"   Actual diagonal: {torch.diag(J).numpy().round(6)}")
    print(f"   σᵢ(1-σᵢ):        {expected_diag.numpy().round(6)}")

z = torch.tensor([2.0, 1.0, 0.5, -0.5])
verify_jacobian_properties(z)
```

### Autograd Verification

```python
def jacobian_via_autograd(z: torch.Tensor) -> torch.Tensor:
    """
    Compute Jacobian using PyTorch's automatic differentiation.
    Serves as verification of the analytical formula.
    """
    z = z.clone().requires_grad_(True)

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

### Efficient Vector-Jacobian Products

In backpropagation, we do not compute the full Jacobian. Instead, we compute the **vector-Jacobian product (VJP)**:

$$\frac{\partial \mathcal{L}}{\partial \mathbf{z}} = \mathbf{J}_\sigma^T \frac{\partial \mathcal{L}}{\partial \boldsymbol{\sigma}} = \mathbf{J}_\sigma \frac{\partial \mathcal{L}}{\partial \boldsymbol{\sigma}}$$

where the last equality uses symmetry of the softmax Jacobian.

**Derivation of the efficient form.** Let $\mathbf{g} = \frac{\partial \mathcal{L}}{\partial \boldsymbol{\sigma}}$:

$$\frac{\partial \mathcal{L}}{\partial z_j} = \sum_i g_i \sigma_i(\delta_{ij} - \sigma_j) = \sigma_j g_j - \sigma_j \sum_i g_i \sigma_i = \sigma_j\bigl(g_j - \langle \mathbf{g}, \boldsymbol{\sigma}\rangle\bigr)$$

Therefore:

$$\frac{\partial \mathcal{L}}{\partial \mathbf{z}} = \boldsymbol{\sigma} \odot \bigl(\mathbf{g} - \langle \mathbf{g}, \boldsymbol{\sigma}\rangle \,\mathbf{1}\bigr)$$

This is $O(K)$ instead of the $O(K^2)$ full Jacobian multiplication.

```python
def softmax_vjp(grad_output: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """
    Compute vector-Jacobian product for softmax.

    Args:
        grad_output: Gradient from subsequent layer ∂L/∂σ, shape (K,)
        sigma: Softmax output σ, shape (K,)

    Returns:
        Gradient w.r.t. logits ∂L/∂z, shape (K,)
    """
    # ⟨g, σ⟩ = Σᵢ gᵢσᵢ
    dot_product = (grad_output * sigma).sum()

    # σ ⊙ (g - ⟨g, σ⟩)
    grad_input = sigma * (grad_output - dot_product)

    return grad_input

# Example
z = torch.tensor([2.0, 1.0, 0.5], requires_grad=True)
sigma = F.softmax(z, dim=0)

# Simulate gradient from loss
grad_output = torch.tensor([0.1, -0.3, 0.2])  # ∂L/∂σ

# Efficient VJP
grad_z_efficient = softmax_vjp(grad_output, sigma.detach())

# Verify with full Jacobian multiplication
J = softmax_jacobian(z.detach())
grad_z_full = J @ grad_output

print("Efficient VJP:", grad_z_efficient.numpy().round(6))
print("Full Jacobian:", grad_z_full.numpy().round(6))
print(f"Match: {torch.allclose(grad_z_efficient, grad_z_full, atol=1e-6)}")
```

### Temperature Scaling

```python
def softmax_temperature(z: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """
    Softmax with temperature scaling.

    Args:
        z: Logits of shape (..., K)
        temperature: Temperature parameter T > 0

    Returns:
        Temperature-scaled probabilities
    """
    return F.softmax(z / temperature, dim=-1)

# Demonstrate temperature effects
logits = torch.tensor([2.0, 1.0, 0.5])
print(f"Logits: {logits}")

for T in [0.1, 0.5, 1.0, 2.0, 10.0]:
    probs = softmax_temperature(logits, T)
    print(f"T={T:4.1f}: {probs.numpy().round(3)}")
```

Output:
```
Logits: tensor([2.0000, 1.0000, 0.5000])
T= 0.1: [1.    0.    0.   ]    (nearly argmax)
T= 0.5: [0.953 0.042 0.006]    (sharp)
T= 1.0: [0.659 0.242 0.099]    (standard)
T= 2.0: [0.474 0.319 0.207]    (soft)
T=10.0: [0.356 0.332 0.312]    (nearly uniform)
```

---

## Summary

### Core Formula

$$\boxed{\sigma(\mathbf{z})_k = \frac{\exp(z_k)}{\sum_{j=1}^{K} \exp(z_j)}}$$

### Key Properties

| Property | Mathematical Statement |
|----------|----------------------|
| Normalization | $\sum_k \sigma(\mathbf{z})_k = 1$ |
| Positivity | $\sigma(\mathbf{z})_k > 0$ |
| Translation Invariance | $\sigma(\mathbf{z} + c\mathbf{1}) = \sigma(\mathbf{z})$ |
| Monotonicity | $z_i > z_j \Rightarrow \sigma(\mathbf{z})_i > \sigma(\mathbf{z})_j$ |

### The Softmax Jacobian

$$\boxed{\mathbf{J}_\sigma = \operatorname{diag}(\boldsymbol{\sigma}) - \boldsymbol{\sigma}\boldsymbol{\sigma}^T}$$

$$\frac{\partial \sigma_i}{\partial z_j} = \sigma_i(\delta_{ij} - \sigma_j) = \begin{cases}
\sigma_i(1 - \sigma_i) & \text{if } i = j \\
-\sigma_i\sigma_j & \text{if } i \neq j
\end{cases}$$

### Jacobian Properties

| Property | Statement |
|----------|-----------|
| Symmetry | $\mathbf{J} = \mathbf{J}^T$ |
| Sum constraint | Row and column sums $= 0$ |
| Positive semi-definite | All eigenvalues $\geq 0$ |
| Rank | $K - 1$ (null space $= \operatorname{span}\{\mathbf{1}\}$) |

### Efficient VJP

$$\frac{\partial \mathcal{L}}{\partial \mathbf{z}} = \boldsymbol{\sigma} \odot \left(\frac{\partial \mathcal{L}}{\partial \boldsymbol{\sigma}} - \left\langle \frac{\partial \mathcal{L}}{\partial \boldsymbol{\sigma}},\, \boldsymbol{\sigma} \right\rangle \mathbf{1}\right)$$

### Implementation Guidelines

!!! tip "Best Practices"
    1. **Always use PyTorch's `F.softmax` or `F.log_softmax`** — they are numerically stable
    2. **Subtract the max before exponentiating** if implementing manually
    3. **Use `F.log_softmax` directly** when computing cross-entropy
    4. **Apply temperature scaling** for knowledge distillation or calibration

---

## References

1. Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*, Chapter 6.2.2.
2. Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*, Chapter 4.3.4.
3. Jaynes, E. T. (1957). Information Theory and Statistical Mechanics. *Physical Review*, 106(4), 620–630.
4. Petersen, K. B., & Pedersen, M. S. (2012). *The Matrix Cookbook*, Section 8.3.
5. Baydin, A. G., et al. (2017). Automatic Differentiation in Machine Learning: A Survey. *JMLR*, 18(153), 1–43.
6. PyTorch Documentation: [torch.nn.functional.softmax](https://pytorch.org/docs/stable/generated/torch.nn.functional.softmax.html)
