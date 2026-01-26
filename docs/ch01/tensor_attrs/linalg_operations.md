# Linear Algebra Operations

Linear algebra operations form the computational backbone of deep learning. PyTorch provides comprehensive support for matrix operations, decompositions, and solving linear systems through both the core API and the `torch.linalg` module.

## Matrix Multiplication

### Element-wise vs Matrix Multiplication

The distinction between `*` and `@` is fundamental—one of the most common sources of bugs:

```python
import torch

A = torch.tensor([[1., 2.], [3., 4.]])
B = torch.tensor([[5., 6.], [7., 8.]])

# Element-wise multiplication (Hadamard product): same position multiplied
element_wise = A * B
# tensor([[ 5., 12.],
#         [21., 32.]])

# Matrix multiplication: row-column dot products
matrix_mult = A @ B
# tensor([[19., 22.],
#         [43., 50.]])
```

### The `@` Operator and `matmul`

The `@` operator is the recommended way to perform matrix multiplication:

```python
# Matrix-matrix multiplication
A = torch.randn(3, 4)
B = torch.randn(4, 5)
C = A @ B  # Shape: (3, 5)

# Equivalent forms:
C = torch.matmul(A, B)
C = torch.mm(A, B)  # Only for 2D matrices
C = A.mm(B)
```

### Matrix-Vector Multiplication

```python
A = torch.randn(3, 4)  # Matrix
x = torch.randn(4)     # Vector

y = A @ x  # Shape: (3,)
# Also available:
y = torch.mv(A, x)
```

### Batch Matrix Multiplication

```python
# Explicit batch multiplication (same batch size)
batch_A = torch.randn(32, 3, 4)
batch_B = torch.randn(32, 4, 5)
batch_C = torch.bmm(batch_A, batch_B)  # Shape: (32, 3, 5)

# With broadcasting via matmul/@
A = torch.randn(3, 4)
batch_B = torch.randn(10, 4, 5)
result = torch.matmul(A, batch_B)  # A broadcasts: (10, 3, 5)

# Complex broadcasting
A = torch.randn(2, 1, 3, 4)
B = torch.randn(1, 5, 4, 2)
C = A @ B  # Shape: (2, 5, 3, 2)
```

**Key distinction**: `bmm` requires exact batch dimensions; `matmul`/`@` broadcasts batch dimensions.

## Vector Operations

### Dot Product

```python
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])

# Dot product (1D tensors only)
dot = torch.dot(a, b)  # tensor(32.) = 1*4 + 2*5 + 3*6

# Inner product notation (equivalent)
inner = a @ b  # tensor(32.)

# Batched dot products
batch_a = torch.randn(32, 128)
batch_b = torch.randn(32, 128)
batch_dot = (batch_a * batch_b).sum(dim=1)  # Shape: (32,)
```

### Outer Product

```python
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0])

outer = torch.outer(a, b)
# tensor([[ 4.,  5.],
#         [ 8., 10.],
#         [12., 15.]])

# Via broadcasting (equivalent)
outer_broadcast = a.unsqueeze(1) * b.unsqueeze(0)
```

### Cross Product

For 3D vectors only:

```python
a = torch.tensor([1., 0., 0.])
b = torch.tensor([0., 1., 0.])

cross = torch.cross(a, b)  # tensor([0., 0., 1.])
```

## Einstein Summation (`einsum`)

`einsum` provides a powerful, flexible notation for complex tensor operations:

### Basic Operations

```python
# Matrix multiplication: 'ik,kj->ij'
A = torch.randn(3, 4)
B = torch.randn(4, 5)
C = torch.einsum('ik,kj->ij', A, B)  # Shape: (3, 5)

# Batch matrix multiplication: 'bik,bkj->bij'
batch_A = torch.randn(10, 3, 4)
batch_B = torch.randn(10, 4, 5)
C = torch.einsum('bik,bkj->bij', batch_A, batch_B)  # Shape: (10, 3, 5)
```

### Common einsum Patterns

```python
A = torch.randn(5, 5)
a = torch.randn(3)
b = torch.randn(4)

# Transpose: 'ij->ji'
At = torch.einsum('ij->ji', A)  # Shape: (5, 5)

# Trace (sum of diagonal): 'ii->'
trace = torch.einsum('ii->', A)  # Scalar

# Batch trace: 'bii->b'
batch_A = torch.randn(10, 5, 5)
traces = torch.einsum('bii->b', batch_A)  # Shape: (10,)

# Diagonal elements: 'ii->i'
diag = torch.einsum('ii->i', A)  # Shape: (5,)

# Outer product: 'i,j->ij'
outer = torch.einsum('i,j->ij', a, b)  # Shape: (3, 4)

# Dot product: 'i,i->'
dot = torch.einsum('i,i->', a, a)  # Scalar

# Attention scores: 'bhqd,bhkd->bhqk'
Q = torch.randn(8, 4, 10, 64)  # (batch, heads, queries, dim)
K = torch.randn(8, 4, 20, 64)  # (batch, heads, keys, dim)
scores = torch.einsum('bhqd,bhkd->bhqk', Q, K)  # Shape: (8, 4, 10, 20)
```

### When to Use einsum

| Use einsum | Use built-in functions |
|------------|------------------------|
| Complex contractions | Simple matmul |
| Multiple simultaneous operations | Single operation |
| Custom dimension handling | Standard patterns |
| Clarity in complex operations | Performance-critical code |

## Matrix Properties

### Transpose

```python
A = torch.randn(3, 4)

# 2D transpose
A_T = A.T  # Shape: (4, 3)

# Multi-dimensional transpose
B = torch.randn(2, 3, 4)
B_transposed = B.transpose(1, 2)  # Swap dims 1 and 2: (2, 4, 3)
```

### Trace

Sum of diagonal elements:

```python
A = torch.tensor([[1., 2., 3.],
                  [4., 5., 6.],
                  [7., 8., 9.]])

trace = torch.trace(A)  # tensor(15.) = 1 + 5 + 9

# Equivalent:
trace = torch.diag(A).sum()
```

### Determinant

```python
A = torch.tensor([[1.0, 2.0],
                  [3.0, 4.0]])

det = torch.linalg.det(A)  # tensor(-2.) = 1*4 - 2*3
```

### Matrix Rank

```python
A = torch.tensor([[1.0, 2.0, 3.0],
                  [4.0, 5.0, 6.0],
                  [7.0, 8.0, 9.0]])

rank = torch.linalg.matrix_rank(A)  # tensor(2) - rows are linearly dependent
```

### Condition Number

Measures numerical stability—large values indicate ill-conditioned matrices:

```python
A = torch.randn(5, 5)
cond = torch.linalg.cond(A)
```

## Matrix Inverse and Pseudo-Inverse

### Inverse

```python
A = torch.tensor([[1.0, 2.0],
                  [3.0, 4.0]])

A_inv = torch.linalg.inv(A)

# Verify: A @ A^(-1) ≈ I
identity = A @ A_inv
print(torch.allclose(identity, torch.eye(2)))  # True
```

### Pseudo-Inverse (Moore-Penrose)

For non-square or singular matrices:

```python
A = torch.randn(3, 5)

A_pinv = torch.linalg.pinv(A)
# Original: (3, 5), Pseudo-inverse: (5, 3)

# For overdetermined systems: A_pinv = (A^T A)^(-1) A^T
# For underdetermined systems: A_pinv = A^T (A A^T)^(-1)
```

### Matrix Power

```python
A = torch.tensor([[1., 2.], [3., 4.]])

A_squared = torch.linalg.matrix_power(A, 2)
print(torch.allclose(A_squared, A @ A))  # True
```

## Matrix Decompositions

### Eigenvalue Decomposition

For a square matrix $A$, find eigenvalues $\lambda$ and eigenvectors $v$ such that $Av = \lambda v$:

```python
A = torch.tensor([[4.0, -2.0],
                  [1.0, 1.0]])

# General case (may return complex values)
eigenvalues, eigenvectors = torch.linalg.eig(A)

# For symmetric/Hermitian matrices (real eigenvalues, sorted ascending)
A_sym = torch.tensor([[4.0, 1.0],
                      [1.0, 3.0]])
eigenvalues_real, eigenvectors_real = torch.linalg.eigh(A_sym)
```

### Singular Value Decomposition (SVD)

For any matrix $A = U \Sigma V^H$:

```python
A = torch.randn(4, 3)

# Full SVD
U, S, Vh = torch.linalg.svd(A)
# U: (4, 4), S: (3,), Vh: (3, 3)

# Reduced SVD (more efficient, recommended)
U, S, Vh = torch.linalg.svd(A, full_matrices=False)
# U: (4, 3), S: (3,), Vh: (3, 3)

# Verify reconstruction
A_reconstructed = U @ torch.diag(S) @ Vh
print(torch.allclose(A, A_reconstructed, atol=1e-5))  # True
```

### QR Decomposition

For a matrix $A = QR$ where $Q$ is orthogonal and $R$ is upper triangular:

```python
A = torch.randn(5, 3)

Q, R = torch.linalg.qr(A)
# Q: (5, 3), R: (3, 3)

# Q is orthonormal: Q^T @ Q = I
print(torch.allclose(Q.T @ Q, torch.eye(3), atol=1e-5))  # True

# Verify reconstruction
print(torch.allclose(Q @ R, A, atol=1e-5))  # True
```

### Cholesky Decomposition

For a positive-definite matrix $A = LL^T$:

```python
# Create positive definite matrix
A = torch.randn(3, 3)
A = A @ A.T + torch.eye(3)  # Ensure positive definite

L = torch.linalg.cholesky(A)  # Lower triangular

# Verify: L @ L^T = A
print(torch.allclose(L @ L.T, A))  # True
```

### LU Decomposition

```python
A = torch.randn(4, 4)

P, L, U = torch.linalg.lu(A)
# P is permutation, L is lower triangular, U is upper triangular
# P @ A = L @ U
```

## Solving Linear Systems

### Basic Linear System: Ax = b

```python
A = torch.tensor([[3.0, 1.0],
                  [1.0, 2.0]])
b = torch.tensor([9.0, 8.0])

x = torch.linalg.solve(A, b)

# Verify: Ax ≈ b
print(torch.allclose(A @ x, b))  # True
```

### Batch Solving

```python
batch_A = torch.randn(10, 3, 3)
batch_b = torch.randn(10, 3)

# Make A invertible
batch_A = batch_A + 5 * torch.eye(3)

batch_x = torch.linalg.solve(batch_A, batch_b)  # Shape: (10, 3)
```

### Least Squares

For overdetermined systems (more equations than unknowns), minimize $\|Ax - b\|$:

```python
A = torch.randn(5, 3)  # 5 equations, 3 unknowns
b = torch.randn(5)

solution = torch.linalg.lstsq(A, b)
x = solution.solution
residuals = solution.residuals
rank = solution.rank
```

## Norms

### Vector Norms

```python
v = torch.tensor([3.0, 4.0])

# L2 norm (Euclidean)
l2 = torch.linalg.norm(v)  # tensor(5.) = sqrt(3² + 4²)

# L1 norm (Manhattan)
l1 = torch.linalg.norm(v, ord=1)  # tensor(7.) = |3| + |4|

# L-infinity norm (max absolute value)
linf = torch.linalg.norm(v, ord=float('inf'))  # tensor(4.)
```

### Matrix Norms

```python
A = torch.randn(3, 4)

# Frobenius norm (default for matrices)
fro = torch.linalg.norm(A)

# Nuclear norm (sum of singular values)
nuclear = torch.linalg.norm(A, ord='nuc')

# Spectral norm (largest singular value)
spectral = torch.linalg.norm(A, ord=2)
```

## Practical Applications

### Scaled Dot-Product Attention

```python
def scaled_dot_product_attention(Q, K, V):
    """
    Args:
        Q: (batch, heads, seq_q, d_k)
        K: (batch, heads, seq_k, d_k)
        V: (batch, heads, seq_k, d_v)
    Returns:
        output: (batch, heads, seq_q, d_v)
    """
    d_k = Q.size(-1)
    
    # Attention scores: Q @ K^T
    scores = Q @ K.transpose(-2, -1)  # (batch, heads, seq_q, seq_k)
    scores = scores / (d_k ** 0.5)   # Scale
    
    attn_weights = torch.softmax(scores, dim=-1)
    output = attn_weights @ V
    
    return output

# Usage
batch, heads, seq, d_k, d_v = 8, 4, 10, 64, 64
Q = torch.randn(batch, heads, seq, d_k)
K = torch.randn(batch, heads, seq, d_k)
V = torch.randn(batch, heads, seq, d_v)

output = scaled_dot_product_attention(Q, K, V)
# Shape: (8, 4, 10, 64)
```

### Batch Linear Transformation

```python
def batch_linear(x, W, b=None):
    """
    Args:
        x: (batch, in_features)
        W: (out_features, in_features)
        b: (out_features,) optional
    Returns:
        (batch, out_features)
    """
    output = x @ W.T
    if b is not None:
        output = output + b
    return output

x = torch.randn(32, 512)
W = torch.randn(256, 512)
b = torch.randn(256)

y = batch_linear(x, W, b)  # Shape: (32, 256)
```

### Covariance Matrix

```python
def covariance(X):
    """
    Compute covariance matrix.
    Args:
        X: (n_samples, n_features)
    Returns:
        (n_features, n_features)
    """
    n = X.size(0)
    X_centered = X - X.mean(dim=0, keepdim=True)
    return (X_centered.T @ X_centered) / (n - 1)

X = torch.randn(100, 5)
cov = covariance(X)  # Shape: (5, 5)
```

### Principal Component Analysis (PCA)

```python
def pca(X, n_components):
    """
    Args:
        X: (n_samples, n_features)
        n_components: number of components to keep
    Returns:
        X_projected: (n_samples, n_components)
        components: (n_components, n_features)
        explained_variance_ratio: (n_components,)
    """
    # Center data
    X_centered = X - X.mean(dim=0)
    
    # SVD
    U, S, Vh = torch.linalg.svd(X_centered, full_matrices=False)
    
    # Keep top components
    components = Vh[:n_components]
    X_projected = X_centered @ components.T
    
    # Explained variance
    explained_variance = (S ** 2) / (S ** 2).sum()
    
    return X_projected, components, explained_variance[:n_components]

X = torch.randn(1000, 50)
X_pca, components, var_ratio = pca(X, n_components=10)
print(f"Reduced: {X.shape[1]} → {X_pca.shape[1]} dimensions")
print(f"Explained variance: {var_ratio.sum():.2%}")
```

### Gram Matrix (Style Transfer)

```python
def gram_matrix(features):
    """
    Compute Gram matrix for style transfer.
    Args:
        features: (batch, channels, height, width)
    Returns:
        (batch, channels, channels)
    """
    b, c, h, w = features.shape
    features = features.view(b, c, h * w)  # (b, c, n)
    gram = features @ features.transpose(-2, -1)  # (b, c, c)
    return gram / (c * h * w)

features = torch.randn(8, 256, 14, 14)
gram = gram_matrix(features)  # Shape: (8, 256, 256)
```

### Orthogonalization with QR

```python
# Gram-Schmidt orthogonalization via QR
vectors = torch.randn(5, 3)  # 3 vectors in 5D

Q, R = torch.linalg.qr(vectors.T)
orthogonal_vectors = Q.T

# Verify orthogonality
dot_products = orthogonal_vectors @ orthogonal_vectors.T
print(torch.allclose(dot_products, torch.eye(3), atol=1e-5))  # True
```

### Ridge Regression

```python
# Ridge regression: minimize ||Ax - b||² + λ||x||²
# Closed-form solution: x = (A^T A + λI)^(-1) A^T b

X = torch.randn(100, 10)  # Features
y = torch.randn(100)      # Targets
lambda_reg = 0.1          # Regularization strength

A = X.T @ X + lambda_reg * torch.eye(10)
b = X.T @ y
weights = torch.linalg.solve(A, b)
```

## Performance Tips

1. **Prefer `@` over function calls** for readability:
   ```python
   # Preferred
   C = A @ B
   # Also fine
   C = torch.matmul(A, B)
   ```

2. **Batch operations** are much faster than loops:
   ```python
   # Fast: single batched operation
   batch_result = batch_A @ batch_B
   
   # Slow: loop
   results = [A @ B for A, B in zip(batch_A, batch_B)]
   ```

3. **Choose the right function**:
   - `mm`: 2D only, minimal overhead
   - `bmm`: 3D batched, no broadcasting
   - `matmul`/`@`: flexible with broadcasting

4. **Consider numerical stability** for decompositions and inverses

5. **Profile** when choosing between einsum and explicit operations

## Quick Reference

| Operation | Function | Notes |
|-----------|----------|-------|
| Matrix multiply | `@`, `matmul`, `mm` | `@` broadcasts |
| Batch matmul | `bmm`, `@` | `bmm` no broadcast |
| Vector dot | `dot` | 1D only |
| Outer product | `outer` | |
| Cross product | `cross` | 3D only |
| einsum | `einsum` | Flexible notation |
| Inverse | `linalg.inv` | |
| Pseudo-inverse | `linalg.pinv` | Non-square matrices |
| Determinant | `linalg.det` | |
| Solve Ax=b | `linalg.solve` | |
| Least squares | `linalg.lstsq` | Overdetermined systems |
| Matrix power | `linalg.matrix_power` | |
| Trace | `trace` | |
| Rank | `linalg.matrix_rank` | |
| Condition number | `linalg.cond` | Numerical stability |
| Eigenvalues | `linalg.eig`, `linalg.eigh` | `eigh` for symmetric |
| SVD | `linalg.svd` | |
| QR | `linalg.qr` | |
| Cholesky | `linalg.cholesky` | Positive definite |
| LU | `linalg.lu` | |
| Vector norms | `linalg.norm` | L1, L2, L∞ |
| Matrix norms | `linalg.norm` | Frobenius, nuclear, spectral |

## Key Takeaways

1. **`@` is the go-to** for matrix multiplication—readable and broadcasts automatically
2. **`matmul` broadcasts**, `mm`/`bmm` don't—choose based on your needs
3. **`einsum` is powerful** for complex operations but profile for performance
4. **Use the `linalg` module** for decompositions and solving systems
5. **Batch operations** are essential—avoid Python loops over matrices
6. **Check numerical stability** via condition numbers before inverse/solve
7. **Understand broadcasting rules** in matmul to avoid dimension errors
8. **SVD is versatile**—use it for PCA, low-rank approximations, and pseudo-inverses

## See Also

- [Broadcasting Rules](broadcasting_rules.md) - Implicit shape matching
- [Shape and Dimensions](shape_dimensions.md) - Understanding tensor shapes
- [Memory Layout and Strides](../tensors/memory_layout_strides.md) - Contiguity for operations
