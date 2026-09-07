# 행렬 연산 - 기계학습에 필요한 선형대수

이 스크립트는 행렬 연산, 즉 기계학습에 필요한 선형대수을 보여준다. 이 개념들을 이해하는 것은 효과적인 PyTorch 프로그래밍과 딥러닝 모델 개발에 필수적이다.

## 코드

```python
"""튜토리얼 14: 행렬 셈 - 기계 학습에 꼭 필요한 선형대수"""
import torch

# ========================================================================
# 메인
# ========================================================================

def header(title): print(f"\n{'='*70}\n{title}\n{'='*70}")

def main():
    header("1. Matrix Multiplication")
    A = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
    B = torch.tensor([[5, 6], [7, 8]], dtype=torch.float32)
    print(f"A =\n{A}\nB =\n{B}\n")
    C = A @ B  # Matrix multiplication
    print(f"A @ B =\n{C}")
    print(f"torch.mm(A, B) =\n{torch.mm(A, B)}")
    print(f"torch.matmul(A, B) =\n{torch.matmul(A, B)}")
    A_batch = torch.randn(10, 3, 4)  # Batch of matrices
    B_batch = torch.randn(10, 4, 5)
    C_batch = A_batch @ B_batch  # Batch matrix multiplication
    print(f"\nBatch: {A_batch.shape} @ {B_batch.shape} = {C_batch.shape}")
    
    header("2. Dot Product")
    v1 = torch.tensor([1, 2, 3], dtype=torch.float32)
    v2 = torch.tensor([4, 5, 6], dtype=torch.float32)
    dot = torch.dot(v1, v2)
    print(f"v1 = {v1}\nv2 = {v2}")
    print(f"dot(v1, v2) = {dot}")  # 1*4 + 2*5 + 3*6 = 32
    dot_manual = (v1 * v2).sum()
    print(f"Manual: (v1 * v2).sum() = {dot_manual}")
    
    header("3. Matrix-Vector Multiplication")
    M = torch.randn(3, 4)
    v = torch.randn(4)
    result = M @ v  # or torch.mv(M, v)
    print(f"M: {M.shape}, v: {v.shape}")
    print(f"M @ v: {result.shape}")  # (3,)
    result_mv = torch.mv(M, v)
    print(f"torch.mv(M, v): {result_mv.shape}")
    
    header("4. Outer Product")
    v1 = torch.tensor([1, 2, 3])
    v2 = torch.tensor([4, 5])
    outer = torch.outer(v1, v2)
    print(f"v1 = {v1}, v2 = {v2}")
    print(f"outer(v1, v2) =\n{outer}")
    print(f"Shape: {outer.shape}")  # (3, 2)
    
    header("5. Matrix Transpose")
    M = torch.arange(6).reshape(2, 3)
    print(f"M =\n{M}")
    print(f"M.T =\n{M.T}")
    print(f"M.transpose(0, 1) =\n{M.transpose(0, 1)}")
    
    header("6. Matrix Inverse")
    A = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    print(f"A =\n{A}")
    A_inv = torch.inverse(A)
    print(f"A^(-1) =\n{A_inv}")
    identity = A @ A_inv
    print(f"A @ A^(-1) ≈ I:\n{identity}")
    
    header("7. Determinant and Trace")
    A = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    det = torch.det(A)
    trace = torch.trace(A)
    print(f"A =\n{A}")
    print(f"det(A) = {det}")
    print(f"trace(A) = {trace}")  # Sum of diagonal
    
    header("8. Matrix Norms")
    A = torch.randn(3, 4)
    print(f"A shape: {A.shape}")
    fro_norm = torch.norm(A, p='fro')  # Frobenius norm
    print(f"Frobenius norm: {fro_norm:.4f}")
    nuc_norm = torch.norm(A, p='nuc')  # Nuclear norm
    print(f"Nuclear norm: {nuc_norm:.4f}")
    
    header("9. Solving Linear Systems")
    A = torch.tensor([[3.0, 1.0], [1.0, 2.0]])
    b = torch.tensor([9.0, 8.0])
    print(f"Solve Ax = b")
    print(f"A =\n{A}\nb = {b}")
    x = torch.linalg.solve(A, b)
    print(f"x = {x}")
    print(f"Verification: A @ x = {A @ x}")

if __name__ == "__main__":
    main()```

## 논의

행렬 연산은 신경망 계산의 핵심을 이룬다. `@` 연산자와 `torch.matmul()`은 배치 차원에 대한 자동 브로드캐스팅과 함께 행렬 곱을 처리하고, `torch.mm()`이나 `torch.bmm()` 같은 특화된 함수는 명료함과 성능을 위해 텐서의 계수를 특정 값으로 강제한다.

재구성 연산은 데이터를 반드시 복사하지 않으면서 텐서의 논리적 배치를 바꾼다. `view()` 메서드는 연속된 메모리를 요구하며 항상 뷰를 반환하고, `reshape()`는 (필요하면 복사하여) 어떤 텐서에서도 동작한다. `transpose()`나 `permute()` 같은 연산은 데이터 배치가 아니라 스트라이드를 바꾸므로 결과가 연속적이지 않을 수 있다.

텐서 생성 함수는 데이터를 초기화하는 유연한 방법을 제공한다. `torch.zeros`, `torch.randn`, `torch.arange` 같은 팩토리 함수는 `dtype`, `device`, `requires_grad` 매개변수를 받으므로 불필요한 복사 없이 목표 장치에 곧바로 할당할 수 있다.

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
