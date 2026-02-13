# Shape Manipulation

## Concatenation and Stacking

### `cat` — Concatenate along existing dimension

```python
import torch

a = torch.randn(2, 3)
b = torch.randn(4, 3)

# Concatenate along dim 0 (stack vertically)
c = torch.cat([a, b], dim=0)    # Shape: (6, 3)

# Concatenate along dim 1 (stack horizontally)
d = torch.randn(2, 5)
e = torch.cat([a, d], dim=1)    # Shape: (2, 8)
```

### `stack` — Create new dimension

```python
a = torch.randn(3, 4)
b = torch.randn(3, 4)
c = torch.randn(3, 4)

# Stack creates a new dimension
s = torch.stack([a, b, c], dim=0)  # Shape: (3, 3, 4)
s = torch.stack([a, b, c], dim=1)  # Shape: (3, 3, 4)
```

### Splitting

```python
x = torch.randn(10, 4)

# Split into chunks of size 3 (last chunk may be smaller)
chunks = torch.split(x, 3, dim=0)  # List of tensors: shapes (3,4), (3,4), (3,4), (1,4)

# Split into exactly n pieces
pieces = torch.chunk(x, 3, dim=0)  # 3 pieces along dim 0
```

## Linear Algebra Operations

Linear algebra forms the computational backbone of deep learning and quantitative finance. PyTorch provides comprehensive support through both the core API and `torch.linalg`.

### Matrix Multiplication

The distinction between `*` (element-wise) and `@` (matrix) multiplication is fundamental:

```python
A = torch.tensor([[1., 2.], [3., 4.]])
B = torch.tensor([[5., 6.], [7., 8.]])

# Element-wise (Hadamard product)
element_wise = A * B      # tensor([[ 5., 12.], [21., 32.]])

# Matrix multiplication
matrix_mult = A @ B       # tensor([[19., 22.], [43., 50.]])
```

The `@` operator is the recommended way to perform matrix multiplication:

```python
A = torch.randn(3, 4)
B = torch.randn(4, 5)
C = A @ B                         # Shape: (3, 5)

# Equivalent forms
C = torch.matmul(A, B)
C = torch.mm(A, B)               # 2D only, minimal overhead
```

### Matrix-Vector and Batch Multiplication

```python
# Matrix-vector
A = torch.randn(3, 4)
x = torch.randn(4)
y = A @ x                        # Shape: (3,)

# Batch matrix multiplication
batch_A = torch.randn(32, 3, 4)
batch_B = torch.randn(32, 4, 5)
batch_C = torch.bmm(batch_A, batch_B)  # Shape: (32, 3, 5)

# With broadcasting via @
A = torch.randn(3, 4)
batch_B = torch.randn(10, 4, 5)
result = torch.matmul(A, batch_B)      # A broadcasts: (10, 3, 5)
```

**Key distinction**: `bmm` requires exact batch dimensions; `matmul`/`@` broadcasts batch dimensions.

### Vector Operations

```python
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])

# Dot product (1D only)
dot = torch.dot(a, b)            # tensor(32.)

# Outer product
outer = torch.outer(a, b)        # Shape: (3, 3)

# Cross product (3D only)
cross = torch.cross(a, b)
```

### Einstein Summation

`einsum` provides flexible notation for complex tensor contractions:

```python
# Matrix multiplication
A = torch.randn(3, 4)
B = torch.randn(4, 5)
C = torch.einsum('ik,kj->ij', A, B)

# Batch matrix multiplication
batch_A = torch.randn(10, 3, 4)
batch_B = torch.randn(10, 4, 5)
C = torch.einsum('bik,bkj->bij', batch_A, batch_B)

# Trace
A = torch.randn(5, 5)
trace = torch.einsum('ii->', A)

# Attention scores
Q = torch.randn(8, 4, 10, 64)  # (batch, heads, queries, dim)
K = torch.randn(8, 4, 20, 64)
scores = torch.einsum('bhqd,bhkd->bhqk', Q, K)
```

### Matrix Properties

```python
A = torch.tensor([[1., 2., 3.],
                  [4., 5., 6.],
                  [7., 8., 9.]])

# Transpose
print(A.T)                         # Or A.t() for 2D

# Trace
print(torch.trace(A))             # tensor(15.) = 1 + 5 + 9

# Determinant
det = torch.linalg.det(A[:2, :2])

# Rank
rank = torch.linalg.matrix_rank(A)  # tensor(2) — linearly dependent rows
```

### Matrix Inverse and Solving Linear Systems

```python
A = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

# Inverse
A_inv = torch.linalg.inv(A)
print(torch.allclose(A @ A_inv, torch.eye(2)))  # True

# Solve Ax = b
b = torch.tensor([9.0, 8.0])
x = torch.linalg.solve(A, b)
print(torch.allclose(A @ x, b))  # True

# Pseudo-inverse (for non-square matrices)
A_rect = torch.randn(3, 5)
A_pinv = torch.linalg.pinv(A_rect)

# Least squares (overdetermined systems)
A = torch.randn(5, 3)
b = torch.randn(5)
solution = torch.linalg.lstsq(A, b)
```

### Matrix Decompositions

```python
# Eigenvalue decomposition (symmetric)
A_sym = torch.tensor([[4.0, 1.0], [1.0, 3.0]])
eigenvalues, eigenvectors = torch.linalg.eigh(A_sym)

# SVD: A = U Σ V^H
A = torch.randn(4, 3)
U, S, Vh = torch.linalg.svd(A, full_matrices=False)
A_reconstructed = U @ torch.diag(S) @ Vh

# QR decomposition
Q, R = torch.linalg.qr(A)
print(torch.allclose(Q.T @ Q, torch.eye(3), atol=1e-5))  # True

# Cholesky (positive definite)
M = torch.randn(3, 3)
M = M @ M.T + torch.eye(3)  # Ensure positive definite
L = torch.linalg.cholesky(M)
print(torch.allclose(L @ L.T, M))  # True
```

### Practical Applications

```python
# PCA via SVD
def pca(X, n_components):
    X_centered = X - X.mean(dim=0)
    U, S, Vh = torch.linalg.svd(X_centered, full_matrices=False)
    components = Vh[:n_components]
    X_projected = X_centered @ components.T
    explained_var = (S ** 2) / (S ** 2).sum()
    return X_projected, components, explained_var[:n_components]

# Ridge regression: x = (X^T X + λI)^{-1} X^T y
def ridge_regression(X, y, lambda_reg=0.1):
    A = X.T @ X + lambda_reg * torch.eye(X.size(1))
    b = X.T @ y
    return torch.linalg.solve(A, b)

# Scaled dot-product attention
def scaled_dot_product_attention(Q, K, V):
    d_k = Q.size(-1)
    scores = Q @ K.transpose(-2, -1) / (d_k ** 0.5)
    attn_weights = torch.softmax(scores, dim=-1)
    return attn_weights @ V
```

## Quick Reference

| Operation | Function | Notes |
|-----------|----------|-------|
| Concatenate | `torch.cat` | Along existing dim |
| Stack | `torch.stack` | Creates new dim |
| Split | `torch.split` | By chunk size |
| Chunk | `torch.chunk` | By number of pieces |
| Matrix multiply | `@`, `matmul`, `mm` | `@` broadcasts |
| Batch matmul | `bmm`, `@` | `bmm` no broadcast |
| Dot product | `torch.dot` | 1D only |
| Outer product | `torch.outer` | — |
| einsum | `torch.einsum` | Flexible contractions |
| Inverse | `linalg.inv` | — |
| Solve Ax=b | `linalg.solve` | — |
| Least squares | `linalg.lstsq` | Overdetermined |
| SVD | `linalg.svd` | — |
| Eigenvalues | `linalg.eigh` | Symmetric |
| QR | `linalg.qr` | — |
| Cholesky | `linalg.cholesky` | Positive definite |
