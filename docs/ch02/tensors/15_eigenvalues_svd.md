# 고윳값과 SVD - 고급 행렬 분해

이 스크립트는 고윳값과 SVD 같은 고급 행렬 분해을 보여준다. 이 개념들을 이해하는 것은 효과적인 PyTorch 프로그래밍과 딥러닝 모델 개발에 필수적이다.

## 코드

```python
"""익힘 15: 고윳값과 특잇값 쪼개기 - 앞선 행렬 쪼개기"""
import torch

# ========================================================================
# 메인
# ========================================================================

def header(title): print(f"\n{'='*70}\n{title}\n{'='*70}")

def main():
    header("1. Eigenvalues and Eigenvectors")
    A = torch.tensor([[2.0, 1.0], [1.0, 2.0]])
    print(f"A =\n{A}")
    eigenvalues, eigenvectors = torch.linalg.eig(A)
    print(f"Eigenvalues: {eigenvalues}")
    print(f"Eigenvectors:\n{eigenvectors}")
    v1 = eigenvectors[:, 0].real
    lambda1 = eigenvalues[0].real
    print(f"\nVerification: A @ v1 ≈ λ1 * v1")
    print(f"A @ v1 = {A @ v1}")
    print(f"λ1 * v1 = {lambda1 * v1}")
    
    header("2. Singular Value Decomposition (SVD)")
    M = torch.randn(4, 3)
    print(f"M shape: {M.shape}")
    U, S, Vh = torch.linalg.svd(M, full_matrices=False)
    print(f"U shape: {U.shape}")  # (4, 3)
    print(f"S shape: {S.shape}")  # (3,) - singular values
    print(f"Vh shape: {Vh.shape}")  # (3, 3)
    print(f"\nSingular values: {S}")
    M_reconstructed = U @ torch.diag(S) @ Vh
    print(f"Reconstruction error: {torch.norm(M - M_reconstructed):.2e}")
    
    header("3. Matrix Rank")
    M = torch.tensor([[1.0, 2.0, 3.0], 
                      [4.0, 5.0, 6.0], 
                      [7.0, 8.0, 9.0]])
    print(f"M =\n{M}")
    rank = torch.linalg.matrix_rank(M)
    print(f"Rank: {rank}")  # This matrix is rank-deficient
    M_full = torch.randn(3, 3)
    print(f"\nRandom matrix rank: {torch.linalg.matrix_rank(M_full)}")
    
    header("4. QR Decomposition")
    A = torch.randn(5, 3)
    Q, R = torch.linalg.qr(A)
    print(f"A shape: {A.shape}")
    print(f"Q shape: {Q.shape}")  # (5, 3) - orthonormal columns
    print(f"R shape: {R.shape}")  # (3, 3) - upper triangular
    print(f"Q is orthonormal: {torch.allclose(Q.T @ Q, torch.eye(3))}")
    print(f"Reconstruction: {torch.allclose(Q @ R, A)}")
    
    header("5. Cholesky Decomposition")
    A = torch.tensor([[4.0, 2.0], [2.0, 3.0]])  # Positive definite
    print(f"A (positive definite) =\n{A}")
    L = torch.linalg.cholesky(A)
    print(f"L (lower triangular) =\n{L}")
    print(f"L @ L.T =\n{L @ L.T}")  # Should equal A
    
    header("6. Practical: PCA with SVD")
    data = torch.randn(100, 10)  # 100 samples, 10 features
    print(f"Data shape: {data.shape}")
    data_centered = data - data.mean(dim=0)
    U, S, Vh = torch.linalg.svd(data_centered, full_matrices=False)
    n_components = 3
    print(f"Top {n_components} principal components:")
    print(f"Explained variance: {S[:n_components]**2 / (S**2).sum()}")
    data_reduced = data_centered @ Vh.T[:, :n_components]
    print(f"Reduced data shape: {data_reduced.shape}")

if __name__ == "__main__":
    main()```

## 논의

행렬 연산은 신경망 계산의 핵심을 이룬다. `@` 연산자와 `torch.matmul()`은 배치 차원에 대한 자동 브로드캐스팅과 함께 행렬 곱을 처리하고, `torch.mm()`이나 `torch.bmm()` 같은 특화된 함수는 명료함과 성능을 위해 텐서의 계수를 특정 값으로 강제한다.

텐서 생성 함수는 데이터를 초기화하는 유연한 방법을 제공한다. `torch.zeros`, `torch.randn`, `torch.arange` 같은 팩토리 함수는 `dtype`, `device`, `requires_grad` 매개변수를 받으므로 불필요한 복사 없이 목표 장치에 곧바로 할당할 수 있다.

`sum()`, `mean()`, `min()`, `max()` 같은 축약 연산은 텐서의 하나 이상의 차원을 접는다. `dim` 매개변수가 어느 차원을 축약할지 지정하며, `keepdim=True`는 축약된 차원을 크기 1로 남겨 둔다. 이는 이후 연산에서 올바른 브로드캐스팅을 위해 필수적이다.

## 연습문제

**연습문제 1.**
$(2, 3)$ 행렬과 $(3, 4)$ 행렬의 곱을 `@`, `torch.mm()`, `torch.matmul()` 세 가지 방법으로 계산하라. 결과가 동일함을 확인하라.

??? success "연습문제 1 풀이"
    ```python
    A = torch.randn(2, 3)
    B = torch.randn(3, 4)
    r1 = A @ B
    r2 = torch.mm(A, B)
    r3 = torch.matmul(A, B)
    print(torch.allclose(r1, r2) and torch.allclose(r2, r3))  # True
    ```

---


**연습문제 2.**
원소별 곱(`*`)과 행렬 곱(`@`)의 차이를 설명하라. 같은 입력에 대해 서로 다른 결과를 내는 예를 들라.

??? success "연습문제 2 풀이"
    원소별 곱은 대응하는 원소끼리 곱한다. 같은 모양의 행렬 $A, B$에 대해 $(A * B)_{ij} = A_{ij} B_{ij}$이다. 행렬 곱은 안쪽 차원에 대해 더한다. $(A @ B)_{ij} = \sum_k A_{ik} B_{kj}$이다. $A = B = [[1,2],[3,4]]$일 때 원소별 곱은 $[[1,4],[9,16]]$을, 행렬 곱은 $[[7,10],[15,22]]$을 준다.

---


**연습문제 3.**
`torch.einsum`을 사용하여 모양 $(10, 4, 4)$인 텐서의 배치 대각합을 계산하고 모양 $(10,)$인 텐서를 반환하라.

??? success "연습문제 3 풀이"
    ```python
    A = torch.randn(10, 4, 4)
    traces = torch.einsum('bii->b', A)
    print(traces.shape)  # torch.Size([10])
    # 직접 계산한 값과 대조 확인:
    manual = torch.stack([torch.trace(A[i]) for i in range(10)])
    print(torch.allclose(traces, manual))  # True
    ```
