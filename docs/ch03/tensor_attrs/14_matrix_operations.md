# 행렬 연산

이 스크립트는 텐서의 행렬 연산을 보여준다. 이 개념들을 이해하는 것은 효과적인 PyTorch 프로그래밍과 딥러닝 모델 개발에 필수적이다.

## 1. 코드

```python
#!/usr/bin/env python3
"""
행렬 셈과 선형대수.

Covers:
- 행렬 곱: matmul (@), mm, bmm
- 원소별 곱: mul (*)
- 벡터 셈: dot, cross, outer
- 복잡한 셈을 위한 einsum
- 배치 행렬 셈
- 선형대수: inv, det, solve, eig
- 행렬 쪼개기: SVD, QR, 촐레스키
"""

import torch

# ========================================================================
# 메인
# ========================================================================

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
    
    # matmul()로도 된다
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
    
    # mv()로도 된다
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
    
    # mm()은 정확히 2차원 텐서를 요구한다
    result = torch.mm(a, b)
    print("mm(a, b).shape:", result.shape)  # (3, 5)
    
    # matmul()이 더 유연하다
    result2 = torch.matmul(a, b)
    print("matmul(a, b).shape:", result2.shape)

    # -------------------------------------------------------------------------
    header("Batch matrix multiplication: bmm()")
    # 행렬 곱의 배치
    batch_a = torch.randn(10, 3, 4)  # 10 matrices of (3, 4)
    batch_b = torch.randn(10, 4, 5)  # 10 matrices of (4, 5)
    
    result = torch.bmm(batch_a, batch_b)
    print("bmm(batch_a, batch_b).shape:", result.shape)  # (10, 3, 5)
    
    # matmul()도 브로드캐스팅으로 이를 처리한다
    result2 = batch_a @ batch_b
    print("batch_a @ batch_b shape:", result2.shape)

    # -------------------------------------------------------------------------
    header("Broadcasting with matmul")
    # matmul은 배치 차원을 브로드캐스팅한다
    a = torch.randn(5, 3, 4)    # 5 batches
    b = torch.randn(4, 2)       # Single matrix
    
    # b가 배치 차원에 걸쳐 브로드캐스팅된다
    result = a @ b
    print("a.shape:", a.shape)
    print("b.shape:", b.shape)
    print("(a @ b).shape:", result.shape)  # (5, 3, 2)
    
    # 더 복잡한 브로드캐스팅
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
    # 행렬 곱
    a = torch.randn(3, 4)
    b = torch.randn(4, 5)
    
    # 'ik,kj->ij'는 k에 대해 더한다는 뜻이다
    result = torch.einsum('ik,kj->ij', a, b)
    print("einsum matrix multiply:")
    print("  'ik,kj->ij'")
    print("  a.shape:", a.shape)
    print("  b.shape:", b.shape)
    print("  result.shape:", result.shape)  # (3, 5)
    
    # 배치 행렬 곱
    a = torch.randn(10, 3, 4)
    b = torch.randn(10, 4, 5)
    result = torch.einsum('bik,bkj->bij', a, b)
    print("\nBatch matmul: 'bik,bkj->bij'")
    print("  result.shape:", result.shape)  # (10, 3, 5)
    
    # 전치
    a = torch.randn(3, 4)
    result = torch.einsum('ij->ji', a)
    print("\nTranspose: 'ij->ji'")
    print("  a.shape:", a.shape)
    print("  result.shape:", result.shape)  # (4, 3)
    
    # 대각합
    a = torch.randn(5, 5)
    trace = torch.einsum('ii->', a)
    print("\nTrace: 'ii->'")
    print("  result:", trace.item())
    
    # 배치 대각합
    a = torch.randn(10, 5, 5)
    traces = torch.einsum('bii->b', a)
    print("\nBatch trace: 'bii->b'")
    print("  result.shape:", traces.shape)  # (10,)

    # -------------------------------------------------------------------------
    header("Matrix power")
    a = torch.tensor([[1., 2.], [3., 4.]])
    
    # 행렬 제곱
    a_squared = torch.linalg.matrix_power(a, 2)
    print("a:\n", a)
    print("a²:\n", a_squared)
    print("Check: a @ a:\n", a @ a)

    # -------------------------------------------------------------------------
    header("Matrix inverse")
    a = torch.tensor([[1., 2.], [3., 4.]])
    
    # 역행렬 계산
    a_inv = torch.linalg.inv(a)
    print("a:\n", a)
    print("inv(a):\n", a_inv)
    
    # 확인: A @ A^(-1) = I
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
    
    # x에 대해 풀기
    x = torch.linalg.solve(A, b)
    print("A:\n", A)
    print("b:\n", b)
    print("Solution x:\n", x)
    
    # 확인: A @ x = b
    result = A @ x
    print("Verify A @ x:\n", result)
    print("Close to b:", torch.allclose(result, b))

    # -------------------------------------------------------------------------
    header("Eigenvalues and eigenvectors")
    A = torch.tensor([[4., -2.], [1., 1.]], dtype=torch.float32)
    
    # 고윳값과 고유벡터 계산
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
    
    # 복원
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
    
    # Q는 직교행렬이다: Q^T @ Q = I
    orthogonal_check = Q.t() @ Q
    print("Q^T @ Q (should be I):\n", orthogonal_check)

    # -------------------------------------------------------------------------
    header("Cholesky decomposition")
    # 양의 정부호 행렬에 대해: A = L @ L^T
    A = torch.tensor([[4., 2.], [2., 3.]], dtype=torch.float32)
    
    L = torch.linalg.cholesky(A)
    print("A:\n", A)
    print("L (lower triangular):\n", L)
    
    # 복원
    A_reconstructed = L @ L.t()
    print("L @ L^T:\n", A_reconstructed)

    # -------------------------------------------------------------------------
    header("Matrix rank and condition number")
    A = torch.randn(5, 3)
    
    rank = torch.linalg.matrix_rank(A)
    print("A.shape:", A.shape)
    print("Rank:", rank.item())
    
    # 조건수(가장 큰 특이값과 가장 작은 특이값의 비)
    cond = torch.linalg.cond(A)
    print("Condition number:", cond.item())

    # -------------------------------------------------------------------------
    header("Trace (sum of diagonal elements)")
    A = torch.randn(4, 4)
    
    trace = torch.trace(A)
    print("Trace:", trace.item())
    
    # 대각 성분의 합이기도 하다
    diag = torch.diag(A)
    trace2 = diag.sum()
    print("Sum of diagonal:", trace2.item())
    print("Equal:", torch.allclose(trace, trace2))

    # -------------------------------------------------------------------------
    header("Practical: attention mechanism")
    # 단순화한 어텐션: Q @ K^T @ V
    Q = torch.randn(8, 10, 64)  # (batch, seq, dim)
    K = torch.randn(8, 10, 64)
    V = torch.randn(8, 10, 64)
    
    # 어텐션 점수: Q @ K^T
    scores = Q @ K.transpose(-2, -1)  # (8, 10, 10)
    print("Attention scores shape:", scores.shape)
    
    # 소프트맥스를 적용하고 값과 곱한다
    import torch.nn.functional as F
    attn_weights = F.softmax(scores, dim=-1)
    output = attn_weights @ V  # (8, 10, 64)
    print("Attention output shape:", output.shape)

    # -------------------------------------------------------------------------
    header("Practical: batch linear transformation")
    # 같은 행렬로 벡터 배치를 변환
    x = torch.randn(100, 512)  # Batch of 100 vectors
    W = torch.randn(512, 256)  # Weight matrix
    b = torch.randn(256)       # Bias
    
    # 선형 변환
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
```

## 2. 논의

브로드캐스팅은 작은 텐서를 가상으로 확장하여 모양이 다른 텐서 사이의 원소별 연산을 가능하게 한다. PyTorch는 차원을 오른쪽부터 맞추며, 각 차원 쌍이 서로 같거나, 둘 중 하나가 1이거나, 아예 없을 것을 요구한다. 이로써 데이터를 명시적으로 복제하지 않아도 되어 메모리 효율이 좋고 빠르다.

행렬 연산은 신경망 계산의 핵심을 이룬다. `@` 연산자와 `torch.matmul()`은 배치 차원에 대한 자동 브로드캐스팅과 함께 행렬 곱을 처리하고, `torch.mm()`이나 `torch.bmm()` 같은 특화된 함수는 명료함과 성능을 위해 텐서의 계수를 특정 값으로 강제한다.

텐서 생성 함수는 데이터를 초기화하는 유연한 방법을 제공한다. `torch.zeros`, `torch.randn`, `torch.arange` 같은 팩토리 함수는 `dtype`, `device`, `requires_grad` 매개변수를 받으므로 불필요한 복사 없이 목표 장치에 곧바로 할당할 수 있다.

## 연습문제

**연습문제 1.**
모양이 $(3, 1, 5)$와 $(4, 5)$인 텐서가 주어졌을 때 덧셈 후의 출력 모양을 구하고 PyTorch로 확인하라.

??? success "연습문제 1 풀이"
    ```python
    import torch
    a = torch.randn(3, 1, 5)
    b = torch.randn(4, 5)
    c = a + b
    print(c.shape)  # torch.Size([3, 4, 5])
    # b는 암묵적으로 (1, 4, 5)가 되고, 그다음 둘 다 (3, 4, 5)로 브로드캐스팅된다.
    ```

---


**연습문제 2.**
모양이 $(3, 4)$와 $(5,)$인 텐서의 덧셈이 실패하는 이유를 설명하라. 이 불일치를 고치는 서로 다른 두 가지 방법을 제안하라.

??? success "연습문제 2 풀이"
    마지막 차원이 4와 5인데 서로 같지도 않고 1도 아니다. 해결 1: 벡터를 열에 맞도록 바꾼다. 예를 들어 $(4,)$ 벡터를 쓴다. 해결 2: 벡터를 $(5, 1)$로 바꾸고 행렬을 $(5, 4)$로 바꾸어 올바른 축을 따라 브로드캐스팅이 되게 한다.

---


**연습문제 3.**
`torch.outer`를 쓰지 않고 브로드캐스팅으로 두 벡터 $a = [1, 2, 3]$과 $b = [4, 5]$의 외적을 계산하라.

??? success "연습문제 3 풀이"
    ```python
    a = torch.tensor([1., 2., 3.]).unsqueeze(1)  # (3, 1)
    b = torch.tensor([4., 5.]).unsqueeze(0)       # (1, 2)
    outer = a * b  # (3, 2)
    print(outer)
    # tensor([[ 4.,  5.],
    #         [ 8., 10.],
    #         [12., 15.]])
    ```

## 정리하며

**다룬 것** — 행렬 연산

브로드캐스팅은 작은 텐서를 가상으로 확장하여 모양이 다른 텐서 사이의 원소별 연산을 가능하게 한다.

앞의 연습문제 3개로 직접 확인할 수 있다.
