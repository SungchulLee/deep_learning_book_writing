#!/usr/bin/env python3
"""
Matrix operations and linear algebra.

Covers:
- Matrix multiplication: matmul (@), mm, bmm
- Element-wise multiplication: mul (*)
- Vector operations: dot, cross, outer
- einsum for complex operations
- Batch matrix operations
- Linear algebra: inv, det, solve, eig
- Matrix decompositions: SVD, QR, Cholesky
"""

import torch

def header(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

def main():
    torch.manual_seed(42)

    # -------------------------------------------------------------------------
    header("Element-wise multiplication: * or mul()")
    a = torch.tensor([[1., 2.], [3., 4.]])
    b = torch.tensor([[5., 6.], [7., 8.]])
    
    element_wise = a * b  # Element-wise (Hadamard product)
    print("a:\n", a)
    print("b:\n", b)
    print("a * b (element-wise):\n", element_wise)

    # -------------------------------------------------------------------------
    header("Matrix multiplication: @ or matmul()")
    a = torch.tensor([[1., 2.], [3., 4.]])  # (2, 2)
    b = torch.tensor([[5., 6.], [7., 8.]])  # (2, 2)
    
    mat_mul = a @ b  # Matrix multiplication
    print("a @ b (matrix multiplication):\n", mat_mul)
    
    # Also works with matmul()
    mat_mul2 = torch.matmul(a, b)
    print("torch.matmul(a, b):\n", mat_mul2)
    print("Equal:", torch.allclose(mat_mul, mat_mul2))

    # -------------------------------------------------------------------------
    header("Matrix-vector multiplication")
    A = torch.randn(3, 4)  # Matrix
    x = torch.randn(4)     # Vector
    
    result = A @ x  # Matrix-vector product
    print("A.shape:", A.shape)
    print("x.shape:", x.shape)
    print("(A @ x).shape:", result.shape)  # (3,)
    
    # Also works with mv()
    result2 = torch.mv(A, x)
    print("torch.mv(A, x).shape:", result2.shape)

    # -------------------------------------------------------------------------
    header("Vector dot product")
    a = torch.tensor([1., 2., 3.])
    b = torch.tensor([4., 5., 6.])
    
    dot_product = torch.dot(a, b)
    print("a:", a)
    print("b:", b)
    print("dot(a, b):", dot_product.item())
    print("Manual: 1*4 + 2*5 + 3*6 =", 1*4 + 2*5 + 3*6)

    # -------------------------------------------------------------------------
    header("Matrix multiplication: mm() for 2D only")
    a = torch.randn(3, 4)
    b = torch.randn(4, 5)
    
    # mm() requires exactly 2D tensors
    result = torch.mm(a, b)
    print("mm(a, b).shape:", result.shape)  # (3, 5)
    
    # matmul() is more flexible
    result2 = torch.matmul(a, b)
    print("matmul(a, b).shape:", result2.shape)

    # -------------------------------------------------------------------------
    header("Batch matrix multiplication: bmm()")
    # Batch of matrix multiplications
    batch_a = torch.randn(10, 3, 4)  # 10 matrices of (3, 4)
    batch_b = torch.randn(10, 4, 5)  # 10 matrices of (4, 5)
    
    result = torch.bmm(batch_a, batch_b)
    print("bmm(batch_a, batch_b).shape:", result.shape)  # (10, 3, 5)
    
    # matmul() also handles this with broadcasting
    result2 = batch_a @ batch_b
    print("batch_a @ batch_b shape:", result2.shape)

    # -------------------------------------------------------------------------
    header("Broadcasting with matmul")
    # matmul broadcasts batch dimensions
    a = torch.randn(5, 3, 4)    # 5 batches
    b = torch.randn(4, 2)       # Single matrix
    
    # b broadcasts across batch dimension
    result = a @ b
    print("a.shape:", a.shape)
    print("b.shape:", b.shape)
    print("(a @ b).shape:", result.shape)  # (5, 3, 2)
    
    # More complex broadcasting
    a = torch.randn(2, 1, 3, 4)
    b = torch.randn(1, 5, 4, 2)
    result = a @ b
    print("\nComplex broadcasting:")
    print("a.shape:", a.shape)
    print("b.shape:", b.shape)
    print("(a @ b).shape:", result.shape)  # (2, 5, 3, 2)

    # -------------------------------------------------------------------------
    header("Outer product")
    a = torch.tensor([1., 2., 3.])
    b = torch.tensor([4., 5., 6., 7.])
    
    outer = torch.outer(a, b)
    print("a:", a)
    print("b:", b)
    print("outer(a, b):\n", outer)
    print("Shape:", outer.shape)  # (3, 4)

    # -------------------------------------------------------------------------
    header("Cross product (3D vectors only)")
    a = torch.tensor([1., 0., 0.])
    b = torch.tensor([0., 1., 0.])
    
    cross = torch.cross(a, b)
    print("a:", a)
    print("b:", b)
    print("cross(a, b):", cross)  # Should be [0, 0, 1]

    # -------------------------------------------------------------------------
    header("einsum: Einstein summation notation")
    # Matrix multiplication
    a = torch.randn(3, 4)
    b = torch.randn(4, 5)
    
    # 'ik,kj->ij' means sum over k
    result = torch.einsum('ik,kj->ij', a, b)
    print("einsum matrix multiply:")
    print("  'ik,kj->ij'")
    print("  a.shape:", a.shape)
    print("  b.shape:", b.shape)
    print("  result.shape:", result.shape)  # (3, 5)
    
    # Batch matrix multiply
    a = torch.randn(10, 3, 4)
    b = torch.randn(10, 4, 5)
    result = torch.einsum('bik,bkj->bij', a, b)
    print("\nBatch matmul: 'bik,bkj->bij'")
    print("  result.shape:", result.shape)  # (10, 3, 5)
    
    # Transpose
    a = torch.randn(3, 4)
    result = torch.einsum('ij->ji', a)
    print("\nTranspose: 'ij->ji'")
    print("  a.shape:", a.shape)
    print("  result.shape:", result.shape)  # (4, 3)
    
    # Trace (diagonal sum)
    a = torch.randn(5, 5)
    trace = torch.einsum('ii->', a)
    print("\nTrace: 'ii->'")
    print("  result:", trace.item())
    
    # Batch trace
    a = torch.randn(10, 5, 5)
    traces = torch.einsum('bii->b', a)
    print("\nBatch trace: 'bii->b'")
    print("  result.shape:", traces.shape)  # (10,)

    # -------------------------------------------------------------------------
    header("Matrix power")
    a = torch.tensor([[1., 2.], [3., 4.]])
    
    # Square the matrix
    a_squared = torch.linalg.matrix_power(a, 2)
    print("a:\n", a)
    print("a²:\n", a_squared)
    print("Check: a @ a:\n", a @ a)

    # -------------------------------------------------------------------------
    header("Matrix inverse")
    a = torch.tensor([[1., 2.], [3., 4.]])
    
    # Compute inverse
    a_inv = torch.linalg.inv(a)
    print("a:\n", a)
    print("inv(a):\n", a_inv)
    
    # Verify: A @ A^(-1) = I
    identity = a @ a_inv
    print("a @ inv(a) (should be I):\n", identity)

    # -------------------------------------------------------------------------
    header("Determinant")
    a = torch.tensor([[1., 2.], [3., 4.]])
    det = torch.linalg.det(a)
    print("a:\n", a)
    print("det(a):", det.item())
    print("Manual: 1*4 - 2*3 =", 1*4 - 2*3)

    # -------------------------------------------------------------------------
    header("Solving linear systems: Ax = b")
    A = torch.tensor([[3., 1.], [1., 2.]], dtype=torch.float32)
    b = torch.tensor([[9.], [8.]], dtype=torch.float32)
    
    # Solve for x
    x = torch.linalg.solve(A, b)
    print("A:\n", A)
    print("b:\n", b)
    print("Solution x:\n", x)
    
    # Verify: A @ x = b
    result = A @ x
    print("Verify A @ x:\n", result)
    print("Close to b:", torch.allclose(result, b))

    # -------------------------------------------------------------------------
    header("Eigenvalues and eigenvectors")
    A = torch.tensor([[4., -2.], [1., 1.]], dtype=torch.float32)
    
    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = torch.linalg.eig(A)
    print("A:\n", A)
    print("Eigenvalues:", eigenvalues)
    print("Eigenvectors:\n", eigenvectors)

    # -------------------------------------------------------------------------
    header("Singular Value Decomposition (SVD)")
    A = torch.randn(5, 3)
    
    # A = U @ S @ V^T
    U, S, Vh = torch.linalg.svd(A, full_matrices=False)
    print("A.shape:", A.shape)
    print("U.shape:", U.shape)  # (5, 3)
    print("S.shape:", S.shape)  # (3,)
    print("Vh.shape:", Vh.shape)  # (3, 3)
    
    # Reconstruct
    S_mat = torch.diag(S)
    A_reconstructed = U @ S_mat @ Vh
    print("Reconstruction error:", (A - A_reconstructed).abs().max().item())

    # -------------------------------------------------------------------------
    header("QR decomposition")
    A = torch.randn(5, 3)
    
    # A = Q @ R
    Q, R = torch.linalg.qr(A)
    print("A.shape:", A.shape)
    print("Q.shape:", Q.shape)  # (5, 3)
    print("R.shape:", R.shape)  # (3, 3)
    
    # Q is orthogonal: Q^T @ Q = I
    orthogonal_check = Q.t() @ Q
    print("Q^T @ Q (should be I):\n", orthogonal_check)

    # -------------------------------------------------------------------------
    header("Cholesky decomposition")
    # For positive definite matrices: A = L @ L^T
    A = torch.tensor([[4., 2.], [2., 3.]], dtype=torch.float32)
    
    L = torch.linalg.cholesky(A)
    print("A:\n", A)
    print("L (lower triangular):\n", L)
    
    # Reconstruct
    A_reconstructed = L @ L.t()
    print("L @ L^T:\n", A_reconstructed)

    # -------------------------------------------------------------------------
    header("Matrix rank and condition number")
    A = torch.randn(5, 3)
    
    rank = torch.linalg.matrix_rank(A)
    print("A.shape:", A.shape)
    print("Rank:", rank.item())
    
    # Condition number (ratio of largest to smallest singular value)
    cond = torch.linalg.cond(A)
    print("Condition number:", cond.item())

    # -------------------------------------------------------------------------
    header("Trace (sum of diagonal elements)")
    A = torch.randn(4, 4)
    
    trace = torch.trace(A)
    print("Trace:", trace.item())
    
    # Also sum of diagonal
    diag = torch.diag(A)
    trace2 = diag.sum()
    print("Sum of diagonal:", trace2.item())
    print("Equal:", torch.allclose(trace, trace2))

    # -------------------------------------------------------------------------
    header("Practical: attention mechanism")
    # Simplified attention: Q @ K^T @ V
    Q = torch.randn(8, 10, 64)  # (batch, seq, dim)
    K = torch.randn(8, 10, 64)
    V = torch.randn(8, 10, 64)
    
    # Attention scores: Q @ K^T
    scores = Q @ K.transpose(-2, -1)  # (8, 10, 10)
    print("Attention scores shape:", scores.shape)
    
    # Apply softmax and multiply with values
    import torch.nn.functional as F
    attn_weights = F.softmax(scores, dim=-1)
    output = attn_weights @ V  # (8, 10, 64)
    print("Attention output shape:", output.shape)

    # -------------------------------------------------------------------------
    header("Practical: batch linear transformation")
    # Transform batch of vectors with same matrix
    x = torch.randn(100, 512)  # Batch of 100 vectors
    W = torch.randn(512, 256)  # Weight matrix
    b = torch.randn(256)       # Bias
    
    # Linear transformation
    output = x @ W + b
    print("Input shape:", x.shape)
    print("Weight shape:", W.shape)
    print("Output shape:", output.shape)  # (100, 256)

    # -------------------------------------------------------------------------
    header("Quick reference: matrix operations")
    print("\nMultiplication:")
    print("  * or mul()     - Element-wise multiplication")
    print("  @ or matmul()  - Matrix multiplication (flexible)")
    print("  mm()           - Matrix multiply (2D only)")
    print("  mv()           - Matrix-vector multiply")
    print("  bmm()          - Batch matrix multiply")
    
    print("\nVector operations:")
    print("  dot()          - Dot product")
    print("  cross()        - Cross product (3D)")
    print("  outer()        - Outer product")
    
    print("\nAdvanced:")
    print("  einsum()       - Einstein summation")
    
    print("\nLinear algebra:")
    print("  linalg.inv()   - Matrix inverse")
    print("  linalg.det()   - Determinant")
    print("  linalg.solve() - Solve Ax=b")
    print("  linalg.eig()   - Eigenvalues/vectors")
    print("  trace()        - Sum of diagonal")
    
    print("\nDecompositions:")
    print("  linalg.svd()   - Singular value decomposition")
    print("  linalg.qr()    - QR decomposition")
    print("  linalg.cholesky() - Cholesky decomposition")
    
    print("\nTips:")
    print("  - Use @ for clean matrix multiplication")
    print("  - matmul broadcasts, mm/bmm don't")
    print("  - einsum is powerful but can be slower")
    print("  - For large matrices, check numerical stability")

if __name__ == "__main__":
    main()
