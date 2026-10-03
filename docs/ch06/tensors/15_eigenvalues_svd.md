# 고윳값과 SVD - 고급 행렬 분해

행렬을 쪼개는 함수들은 수학책의 정의와 그대로 맞아떨어지지 않는다. **대칭 행렬에 `eig`를 써도 복소수가 나오고**, 고윳값과 특잇값은 **정렬 차례가 서로 반대**이며, 고유벡터의 부호는 아예 정해져 있지 않다. 계수(rank)도 재는 값이라 허용오차에 달려 있다. 이 쪽은 그 어긋남을 본다.

## 1. 코드

```python
"""튜토리얼 15: 고윳값과 특잇값 쪼개기 - 앞선 행렬 쪼개기"""
import torch
# 무작위로 뽑는 값이 아래에 나온다. 씨앗을 고정해야 이 쪽에 실린
# 수가 다시 나온다 — 고정하지 않으면 돌릴 때마다 다른 수가 찍힌다
torch.manual_seed(0)

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
    main()
```

**출력:**

```

======================================================================
1. Eigenvalues and Eigenvectors
======================================================================
A =
tensor([[2., 1.],
        [1., 2.]])
Eigenvalues: tensor([3.+0.j, 1.+0.j])
Eigenvectors:
tensor([[ 0.7071+0.j, -0.7071+0.j],
        [ 0.7071+0.j,  0.7071+0.j]])

Verification: A @ v1 ≈ λ1 * v1
A @ v1 = tensor([2.1213, 2.1213])
λ1 * v1 = tensor([2.1213, 2.1213])

======================================================================
2. Singular Value Decomposition (SVD)
======================================================================
M shape: torch.Size([4, 3])
U shape: torch.Size([4, 3])
S shape: torch.Size([3])
Vh shape: torch.Size([3, 3])

Singular values: tensor([3.2245, 1.4537, 0.2943])
Reconstruction error: 8.39e-07

======================================================================
3. Matrix Rank
======================================================================
M =
tensor([[1., 2., 3.],
        [4., 5., 6.],
        [7., 8., 9.]])
Rank: 2

Random matrix rank: 3

======================================================================
4. QR Decomposition
======================================================================
A shape: torch.Size([5, 3])
Q shape: torch.Size([5, 3])
R shape: torch.Size([3, 3])
Q is orthonormal: False
Reconstruction: True

======================================================================
5. Cholesky Decomposition
======================================================================
A (positive definite) =
tensor([[4., 2.],
        [2., 3.]])
L (lower triangular) =
tensor([[2.0000, 0.0000],
        [1.0000, 1.4142]])
L @ L.T =
tensor([[4., 2.],
        [2., 3.]])

======================================================================
6. Practical: PCA with SVD
======================================================================
Data shape: torch.Size([100, 10])
Top 3 principal components:
Explained variance: tensor([0.1721, 0.1411, 0.1305])
Reduced data shape: torch.Size([100, 3])
```

## 2. 논의

**`eig`는 복소수를 준다. 대칭 행렬이어도 그렇다.**

```python
A = torch.tensor([[2., 1.], [1., 2.]])      # 대칭
torch.linalg.eig(A)[0]      # tensor([3.+0.j, 1.+0.j])   dtype=complex64
torch.linalg.eigh(A)[0]     # tensor([1., 3.])           dtype=float32
```

까닭은 `eig`가 **아무 정사각 행렬**을 받기 때문이다. 실수 행렬이라도 고윳값이 복소수일 수 있으므로 — 회전 행렬이 그 보기다 —

```python
B = torch.tensor([[0., -1.], [1., 0.]])     # 90도 회전
torch.linalg.eig(B)[0]      # tensor([0.+1.j, 0.-1.j])
```

`eig`는 자료형을 미리 복소수로 정해 둔다. 그래서 대칭 행렬을 넣어도 `+0.j`가 붙어 나오고, 그 값을 그대로 실수 셈에 넘기면 자료형이 어긋난다.

**대칭이면 `eigh`를 쓴다.** 대칭(에르미트) 행렬은 고윳값이 반드시 실수임이 보장되므로 `eigh`는 실수를 돌려준다. 게다가 더 빠르고 더 안정적이다. 공분산 행렬, 그람 행렬, 헤세 행렬은 모두 대칭이므로 거의 늘 `eigh` 쪽이다.

!!! warning "고윳값은 오름차례, 특잇값은 내림차례다"
    두 함수의 정렬 규약이 **반대**다.

    ```python
    C = torch.tensor([[1., 0.], [0., 5.]])
    torch.linalg.eigh(C)[0]         # tensor([1., 5.])   ← 작은 것부터
    torch.linalg.svdvals(C)         # tensor([5., 1.])   ← 큰 것부터
    ```

    그래서 "가장 큰 것을 고른다"는 코드가 한쪽에서는 `[-1]`, 다른 쪽에서는 `[0]`이 된다. PCA에서 주성분을 고를 때 이 자리를 잘못 짚으면 **가장 덜 중요한 축**을 가장 중요한 축으로 쓰게 되는데, 모양은 맞으므로 오류가 나지 않는다.

    `eig`(대칭이 아닌 쪽)는 **아예 정렬하지 않는다.** 순서를 쓰려면 직접 정렬해야 한다.

**고유벡터와 특이벡터의 부호는 정해져 있지 않다.** $v$가 고유벡터면 $-v$도 같은 고유벡터이므로, 어느 쪽을 돌려줄지는 알고리즘과 판본에 달렸다.

```python
torch.linalg.eigh(A)[1][:, 0]     # tensor([-0.7071,  0.7071])
```

이 값이 `[0.7071, -0.7071]`로 나오는 기계도 맞다. 그러므로 **고유벡터를 눈으로 견주어 같은지 판단하지 않는다.** 부호에 매이지 않는 값 — 고윳값, 투영한 결과의 크기, 복원 오차 — 으로 확인한다. PCA의 주성분도 부호가 뒤집혀 그려질 수 있는데, 틀린 것이 아니다.

**계수(rank)는 세는 값이 아니라 재는 값이다.** 0인 특잇값이 몇 개인지를 세야 하는데, 실수 셈에서 정확히 0이 되는 일은 드물다.

```python
D = torch.tensor([[1., 2.], [2., 4.0000001]])
torch.linalg.svdvals(D)           # tensor([5.0000e+00, 4.8449e-09])
torch.linalg.matrix_rank(D)       # 1
```

수학적으로 $D$는 두 행이 정확히 비례하지 않으므로 계수가 2다. 그런데 둘째 특잇값이 $4.8 \times 10^{-9}$이라 허용오차 아래로 떨어져 **1**이 나온다. 어느 쪽이 맞는가 하면 — 목적에 달렸다. 역행렬을 구할 생각이라면 1이 더 쓸모 있는 답이다. 그만큼 가까우면 사실상 특이행렬이기 때문이다. 허용오차는 `matrix_rank(D, tol=...)`로 직접 줄 수 있다.

**`cholesky`는 양정부호를 요구하고, 아니면 오류를 낸다.**

```python
torch.linalg.cholesky(torch.tensor([[1., 2.], [2., 1.]]))
# _LinAlgError: ... the input is not positive-definite
```

대칭인 것만으로는 모자라다. 공분산 행렬을 다루다가 수치 오차로 고윳값이 아주 작은 음수가 되면 이 오류를 만나므로, 대각선에 작은 수를 더해(`+ 1e-6 * I`) 밀어 주는 것이 흔한 처방이다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
대칭 행렬 `A = torch.tensor([[2., 1.], [1., 2.]])`에 `torch.linalg.eig`와 `torch.linalg.eigh`를 각각 써서 고윳값과 그 `dtype`을 찍어라. 대칭 행렬인데도 `eig`가 복소수를 주는 까닭을 설명하라.

</div>

??? success "연습문제 1 풀이"
    ```python
    import torch

    A = torch.tensor([[2., 1.], [1., 2.]])
    w_eig = torch.linalg.eig(A)[0]
    w_eigh = torch.linalg.eigh(A)[0]
    print("eig :", w_eig, w_eig.dtype)
    print("eigh:", w_eigh, w_eigh.dtype)
    ```

    ```
    eig : tensor([3.+0.j, 1.+0.j]) torch.complex64
    eigh: tensor([1., 3.]) torch.float32
    ```

    `eig`는 **아무 정사각 행렬**을 받는 함수다. 실수 행렬이라도 고윳값이 복소수일 수
    있으므로 자료형을 미리 복소수로 정해 둔다. 그래서 값이 실수로 떨어지는 대칭 행렬을
    넣어도 `+0.j`가 붙는다.

    ```python
    B = torch.tensor([[0., -1.], [1., 0.]])     # 90도 회전
    print("회전 행렬:", torch.linalg.eig(B)[0])
    ```

    ```
    회전 행렬: tensor([0.+1.j, 0.-1.j])
    ```

    이것이 복소수가 필요한 까닭이다. 회전은 늘어나는 방향이 없으니 실수 고윳값이 없다.

    대칭(에르미트) 행렬은 고윳값이 반드시 실수임이 보장되므로 `eigh`가 실수를 돌려준다.
    공분산·그람·헤세 행렬은 모두 대칭이라 거의 늘 `eigh` 쪽이고, 더 빠르고 안정적이다.

---


<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
`C = torch.tensor([[1., 0.], [0., 5.]])`에 대해 `torch.linalg.eigh(C)[0]`과 `torch.linalg.svdvals(C)`를 찍어 정렬 차례를 비교하라. "가장 큰 것을 고르는" 코드를 두 경우에 각각 어떻게 적어야 하는지 쓰고, 이것을 잘못 짚으면 PCA에서 무슨 일이 생기는지 설명하라.

</div>

??? success "연습문제 2 풀이"
    ```python
    import torch

    C = torch.tensor([[1., 0.], [0., 5.]])
    ev = torch.linalg.eigh(C)[0]
    sv = torch.linalg.svdvals(C)
    print("eigh    :", ev)
    print("svdvals :", sv)
    print("가장 큰 고윳값 :", ev[-1].item())
    print("가장 큰 특잇값 :", sv[0].item())
    ```

    ```
    eigh    : tensor([1., 5.])
    svdvals : tensor([5., 1.])
    가장 큰 고윳값 : 5.0
    가장 큰 특잇값 : 5.0
    ```

    규약이 반대다. `eigh`는 **오름차례**라 가장 큰 것이 `[-1]`이고, `svdvals`는
    **내림차례**라 가장 큰 것이 `[0]`이다.

    PCA에서 이 자리를 잘못 짚으면 **가장 덜 중요한 축을 주성분으로 쓴다.** 분산이 가장
    작은 방향을 골라 놓고 가장 큰 방향이라 믿는 셈이다. 모양은 맞으므로 오류가 나지
    않고, 그린 그림이 이상해 보이는 것 말고는 알려 주는 것이 없다.

    덧붙여 `eig`(대칭이 아닌 쪽)는 **아예 정렬하지 않는다.** 순서가 필요하면 직접
    정렬해야 하며, 정렬되어 있다고 가정하면 안 된다.

---


<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
`D = torch.tensor([[1., 2.], [2., 4.0000001]])`의 특잇값과 `torch.linalg.matrix_rank(D)`를 찍어라. 수학적으로 $D$의 계수가 얼마인지 따진 뒤, PyTorch의 답과 어긋나는 까닭을 밝혀라. **어느 쪽이 맞는 답인지** 논하라.

</div>

??? success "연습문제 3 풀이"
    ```python
    import torch

    D = torch.tensor([[1., 2.], [2., 4.0000001]])
    print("특잇값:", torch.linalg.svdvals(D))
    print("rank  :", torch.linalg.matrix_rank(D).item())
    print("행렬식:", torch.linalg.det(D).item())
    ```

    ```
    특잇값: tensor([5.0000e+00, 4.8449e-09])
    rank  : 1
    행렬식: -0.0
    ```

    **수학적으로는 계수가 2다.** 두 행 $[1, 2]$와 $[2, 4.0000001]$은 정확히 비례하지
    않는다. 비례하려면 둘째 행이 $[2, 4]$여야 한다.

    **PyTorch는 1이라고 한다.** 둘째 특잇값이 $4.8 \times 10^{-9}$인데, 이것이 허용오차
    아래로 떨어지므로 0으로 센다. 계수는 "0인 특잇값이 몇 개인가"를 세는 값인데, 실수
    셈에서 정확히 0이 되는 일은 드물기 때문에 **어디까지를 0으로 볼지** 정해야 한다.
    행렬식이 `-0.0`으로 찍히는 것도 같은 사정이다 — `float32`가 그 차이를 담지 못한다.

    **어느 쪽이 맞는가는 무엇을 하려는지에 달렸다.** 역행렬을 구하거나 연립방정식을
    풀 생각이라면 **1이 더 쓸모 있는 답**이다. 특잇값의 비가 $10^9$이면 사실상 특이행렬이고,
    억지로 풀면 답이 입력의 미세한 오차에 휘둘린다. 반대로 기호 셈을 하고 있다면 2가
    맞다.

    허용오차를 내가 정할 수도 있다.

    ```python
    print("tol=1e-12:", torch.linalg.matrix_rank(D, tol=1e-12).item())
    ```

    ```
    tol=1e-12: 2
    ```

    같은 행렬이 허용오차에 따라 계수가 1도 되고 2도 된다. 그러므로 계수는 행렬의
    성질이라기보다 **행렬과 허용오차가 함께 정하는 값**이다.

    덧붙여 `cholesky`도 같은 자리에서 걸린다. 양정부호를 요구하는데, 수치 오차로
    고윳값이 아주 작은 음수가 되면 거절한다.

    ```python
    try:
        torch.linalg.cholesky(torch.tensor([[1., 2.], [2., 1.]]))
    except Exception as e:
        print(type(e).__name__, "-", str(e)[:58])
    ```

    ```
    _LinAlgError - linalg.cholesky: The factorization could not be
    ```

    공분산 행렬을 다룰 때 흔한 처방은 대각선을 조금 밀어 주는 것이다 — `S + 1e-6 * I`.

## 정리하며

행렬을 쪼개는 함수들은 수학책의 정의와 네 군데서 어긋난다.

- **`eig`는 복소수를 준다.** 아무 정사각 행렬을 받으므로 자료형을 미리 복소수로 정해 두기 때문이다. 대칭이면 `eigh`를 써서 실수를 받는다 — 더 빠르고 안정적이다.
- **정렬 규약이 반대다.** `eigh`는 오름차례, `svdvals`는 내림차례다. `eig`는 정렬하지 않는다. "가장 큰 것"이 한쪽에서는 `[-1]`, 다른 쪽에서는 `[0]`이다.
- **고유벡터의 부호는 정해지지 않는다.** $v$와 $-v$가 같은 고유벡터이므로, 눈으로 견주어 같은지 판단하지 않고 부호에 매이지 않는 값으로 확인한다.
- **계수는 재는 값이다.** 허용오차 아래의 특잇값을 0으로 세므로, 같은 행렬이 허용오차에 따라 계수가 달라진다. 행렬의 성질이 아니라 행렬과 허용오차가 함께 정하는 값이다.

`cholesky`가 거절하는 것도 같은 자리다. 수치 오차로 고윳값이 조금 음수가 되면 양정부호가 아니라고 보므로, 대각선을 조금 밀어 준다.
