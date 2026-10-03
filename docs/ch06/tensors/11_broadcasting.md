# PyTorch의 브로드캐스팅

모양이 다른 텐서끼리 셈할 수 있게 해 주는 규칙이 펴 맞추기(broadcasting)다. 편한 만큼 위험하기도 하다. 모양이 **어긋나면 오류가 나서 알게 되지만, 어긋나지 않으면서 뜻과 다른 모양이 되면** 아무 말 없이 지나간다. 이 쪽은 규칙과 그 함정을 함께 본다.

## 1. 코드

```python
"""
튜토리얼 11: PyTorch의 펴 맞추기
=====================================

펴 맞추기 덕에 PyTorch는 모양이 다른 텐서끼리도 데이터를 베끼지 않고
절로 맞는 꼴로 늘려 셈할 수 있다.

핵심 개념:
- 펴 맞추기 규칙
- 흔한 펴 맞추기 무늬
- 펴 맞추기가 안 되는 때
- 기억 자리 아끼기
- 신경망에서의 펴 맞추기
"""

import torch
# 무작위로 뽑는 값이 아래에 나온다. 씨앗을 고정해야 이 쪽에 실린
# 수가 다시 나온다 — 고정하지 않으면 돌릴 때마다 다른 수가 찍힌다
torch.manual_seed(0)

# ========================================================================
# 메인
# ========================================================================


def header(title: str):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def main():
    # -------------------------------------------------------------------------
    # 1. 브로드캐스팅이란?
    # -------------------------------------------------------------------------
    header("1. What is Broadcasting?")
    
    print("""
    펴 맞추기는 원소별 셈을 할 때 모양이 다른 텐서도 다룰 수 있게 하는
    힘센 구조다.
    
    텐서를 직접 같은 크기로 늘리는 대신 PyTorch가 절로
    해 준다(데이터를 참으로 베끼지 않고 겉으로만).
    """)
    
    # 간단한 예: 텐서에 스칼라 더하기
    vec = torch.tensor([1, 2, 3, 4])
    scalar = 10
    
    print(f"vec = {vec}")
    print(f"scalar = {scalar}")
    print(f"vec + scalar = {vec + scalar}")  # Scalar broadcasts to [10, 10, 10, 10]
    
    # -------------------------------------------------------------------------
    # 2. 브로드캐스팅 규칙
    # -------------------------------------------------------------------------
    header("2. Broadcasting Rules")
    
    print("""
    다음이면 두 텐서를 "펴 맞출 수 있다".
    
    1. 텐서마다 차원이 적어도 하나 있거나, 한쪽이 홑값이다
    2. 마지막 차원부터 거꾸로 훑을 때
       - 차원이 같거나,
       - 한쪽 차원이 1이거나,
       - 한쪽 차원이 없다
    
    Examples:
    """)
    
    # 규칙 예시
    examples = [
        ("(3, 1) with (1, 4)", (3, 1), (1, 4), "(3, 4)"),
        ("(3, 4) with (3, 1)", (3, 4), (3, 1), "(3, 4)"),
        ("(1, 4, 5) with (3, 1, 5)", (1, 4, 5), (3, 1, 5), "(3, 4, 5)"),
        ("(3,) with (4, 3)", (3,), (4, 3), "(4, 3)"),
    ]
    
    for desc, shape1, shape2, result in examples:
        t1 = torch.randn(shape1)
        t2 = torch.randn(shape2)
        result_tensor = t1 + t2
        print(f"  {desc}: → {result_tensor.shape}")
    
    # -------------------------------------------------------------------------
    # 3. 기본 브로드캐스팅 패턴
    # -------------------------------------------------------------------------
    header("3. Basic Broadcasting Patterns")
    
    # 패턴 1: 스칼라와 임의의 텐서
    matrix = torch.arange(6).reshape(2, 3)
    scalar_val = 100
    print("Pattern 1: Scalar with matrix")
    print(f"Matrix:\n{matrix}")
    print(f"Matrix + 100:\n{matrix + scalar_val}\n")
    
    # 패턴 2: 벡터와 행렬(마지막 차원이 같다)
    vec = torch.tensor([10, 20, 30])
    print("Pattern 2: Row vector with matrix")
    print(f"vec = {vec}")
    result = matrix + vec  # vec broadcasts to (1, 3) then to (2, 3)
    print(f"matrix + vec:\n{result}\n")
    
    # 패턴 3: 열 벡터와 행렬
    col_vec = torch.tensor([[10], [20]])  # Shape (2, 1)
    print("Pattern 3: Column vector with matrix")
    print(f"col_vec:\n{col_vec}")
    result = matrix + col_vec  # col_vec broadcasts to (2, 3)
    print(f"matrix + col_vec:\n{result}\n")
    
    # -------------------------------------------------------------------------
    # 4. 브로드캐스팅 시각화
    # -------------------------------------------------------------------------
    header("4. Visualizing Broadcasting")
    
    # 시각화를 위한 간단한 텐서 만들기
    A = torch.arange(3).reshape(3, 1)  # Column vector
    B = torch.arange(4).reshape(1, 4)  # Row vector
    
    print("A (3x1):\n", A)
    print("B (1x4):\n", B)
    
    # 브로드캐스팅이 3x4 결과를 만든다
    C = A + B
    print(f"\nA + B (broadcasts to 3x4):\n{C}")
    print("""
    어떻게 도는가:
    - A broadcasts: [[0],     →  [[0, 0, 0, 0],
                     [1],          [1, 1, 1, 1],
                     [2]]          [2, 2, 2, 2]]
    
    - B broadcasts: [[0, 1, 2, 3]] → [[0, 1, 2, 3],
                                      [0, 1, 2, 3],
                                      [0, 1, 2, 3]]
    """)
    
    # -------------------------------------------------------------------------
    # 5. 기계학습에서의 흔한 용례
    # -------------------------------------------------------------------------
    header("5. Common Use Cases in ML")
    
    # 용례 1: 층 출력에 편향 더하기
    print("Use case 1: Adding bias")
    batch_size, features = 32, 10
    layer_output = torch.randn(batch_size, features)
    bias = torch.randn(features)  # Shape (10,)
    
    output_with_bias = layer_output + bias  # bias broadcasts to (32, 10)
    print(f"Layer output: {layer_output.shape}")
    print(f"Bias: {bias.shape}")
    print(f"Result: {output_with_bias.shape}\n")
    
    # 용례 2: 데이터 정규화
    print("Use case 2: Data normalization")
    data = torch.randn(100, 5)  # 100 samples, 5 features
    mean = data.mean(dim=0, keepdim=True)  # Shape (1, 5)
    std = data.std(dim=0, keepdim=True)    # Shape (1, 5)
    
    normalized = (data - mean) / std  # Broadcasting!
    print(f"Data: {data.shape}")
    print(f"Mean: {mean.shape}")
    print(f"Normalized: {normalized.shape}\n")
    
    # 용례 3: 쌍별 거리
    print("Use case 3: Pairwise distances")
    points_a = torch.randn(5, 2)  # 5 points in 2D
    points_b = torch.randn(3, 2)  # 3 points in 2D
    
    # 모든 쌍별 거리 계산
    # 브로드캐스팅을 위한 재구성: (5, 1, 2) - (1, 3, 2) = (5, 3, 2)
    diff = points_a.unsqueeze(1) - points_b.unsqueeze(0)
    distances = torch.sqrt((diff ** 2).sum(dim=2))
    print(f"Points A: {points_a.shape}")
    print(f"Points B: {points_b.shape}")
    print(f"Pairwise distances: {distances.shape}")  # (5, 3)
    
    # -------------------------------------------------------------------------
    # 6. 브로드캐스팅이 실패할 때
    # -------------------------------------------------------------------------
    header("6. When Broadcasting Fails")
    
    print("Broadcasting fails when shapes are incompatible:\n")
    
    # 예제 1: 호환되지 않는 모양
    try:
        t1 = torch.randn(3, 4)
        t2 = torch.randn(2, 3)
        result = t1 + t2  # Will fail!
    except RuntimeError as e:
        print(f"Error with (3,4) + (2,3): Shapes incompatible")
        print(f"  Reason: Neither dimension matches (3≠2, 4≠3)\n")
    
    # 예제 2: 고치는 방법
    print("Fix: Reshape or unsqueeze to make compatible")
    t1 = torch.randn(3, 4)
    t2 = torch.randn(3, 1)  # Now compatible!
    result = t1 + t2
    print(f"(3,4) + (3,1) = {result.shape} ✓\n")
    
    # -------------------------------------------------------------------------
    # 7. keepdim 매개변수 - 차원 보존하기
    # -------------------------------------------------------------------------
    header("7. keepdim=True for Broadcasting")
    
    data = torch.arange(12).reshape(3, 4).float()
    print(f"Data:\n{data}\n")
    
    # keepdim 없이 - 차원이 제거된다
    row_sum = data.sum(dim=1)
    print(f"sum(dim=1): {row_sum}")
    print(f"Shape: {row_sum.shape}")  # (3,) - lost dimension!
    
    # keepdim 사용 - 차원이 크기 1로 남는다
    row_sum_keep = data.sum(dim=1, keepdim=True)
    print(f"\nsum(dim=1, keepdim=True):\n{row_sum_keep}")
    print(f"Shape: {row_sum_keep.shape}")  # (3, 1) - can broadcast!
    
    # 이제 브로드캐스팅이 가능하다
    normalized_rows = data / row_sum_keep  # Works!
    print(f"\nData / row_sum:\n{normalized_rows}")
    
    # -------------------------------------------------------------------------
    # 8. 메모리 효율
    # -------------------------------------------------------------------------
    header("8. Memory Efficiency")
    
    print("""
    핵심 통찰: 펴 맞추기는 데이터를 참으로 베끼지 않는다!
    
    (1, 100) 텐서를 (1000, 100)으로 펴 맞출 때 PyTorch는 베낌 1000개를
    만들지 않는다. 대신 슬기로운 자리 잡기로 같은 데이터를 겉으로만
    되쓴다.
    
    그래서 펴 맞추기는 빠르면서도 기억 자리를 아낀다.
    """)
    
    # 시연
    small = torch.tensor([[1, 2, 3]])  # (1, 3)
    large = torch.randn(1000, 3)
    
    result = small + large  # No actual copying happens!
    
    print(f"Small tensor: {small.shape}")
    print(f"Large tensor: {large.shape}")
    print(f"Result: {result.shape}")
    print(f"Memory multiplier: {result.numel() / small.numel()}x")
    print("But only the small tensor is stored once!")
    
    # -------------------------------------------------------------------------
    # 9. expand()를 이용한 명시적 브로드캐스팅
    # -------------------------------------------------------------------------
    header("9. Explicit Broadcasting with expand()")
    
    # expand()는 데이터를 복사하지 않고 명시적으로 브로드캐스팅한다
    vec = torch.tensor([1, 2, 3])
    print(f"Original vector: {vec}, shape: {vec.shape}")
    
    # (4, 3)으로 확장
    expanded = vec.expand(4, 3)
    print(f"\nExpanded: {expanded.shape}")
    print(f"Values:\n{expanded}")
    
    # 메모리 확인: expand는 복사하지 않는다!
    print(f"\nSame storage? {vec.untyped_storage().data_ptr() == expanded.untyped_storage().data_ptr()}")
    
    # 주의: 확장된 텐서를 수정하면 예상치 못한 결과가 생길 수 있다!
    # 확장된 텐서를 제자리에서 수정하지 말 것

    # broadcast_to() — expand와 같은 일을 하지만 **읽는 사람에게 뜻이 분명하다.**
    # expand는 "길이 1인 차원을 늘린다"는 제 규칙을 알아야 읽히고,
    # broadcast_to는 "이 모양으로 펴 맞춘다"고 그대로 적는다.
    bt = vec.broadcast_to(4, 3)
    print(f"\nbroadcast_to(4, 3): {bt.shape}")
    print(f"expand와 같은가: {torch.equal(bt, expanded)}")
    print(f"역시 복사 안 함: {vec.untyped_storage().data_ptr() == bt.untyped_storage().data_ptr()}")

    # 둘은 하는 일이 같다. 앞쪽에 차원을 새로 붙이는 것도, -1로 "그대로 두기"를
    # 적는 것도 양쪽 다 된다. 고르는 기준은 읽는 맛뿐이다.
    print(f"\nexpand(2, 4, 3):       {vec.expand(2, 4, 3).shape}")
    print(f"broadcast_to(2, 4, 3): {vec.broadcast_to(2, 4, 3).shape}")
    print(f"둘이 같은가: {torch.equal(vec.expand(2, 4, 3), vec.broadcast_to(2, 4, 3))}")
    # broadcast_to는 NumPy의 np.broadcast_to와 이름이 같아서, 두 쪽을 함께
    # 읽는 코드에서는 이쪽이 눈에 덜 걸린다.
    
    # -------------------------------------------------------------------------
    # 10. 고급: einsum을 이용한 브로드캐스팅
    # -------------------------------------------------------------------------
    header("10. Advanced: einsum")
    
    print("""
    einsum(아인슈타인 합)은 펴 맞추기를 드러내 놓고 다스리며 텐서 셈을
    짧게 적는 길을 준다.
    """)
    
    # einsum을 이용한 행렬-벡터 곱
    matrix = torch.randn(3, 4)
    vec = torch.randn(4)
    
    # 전통적인 방식: matrix @ vec
    result_traditional = matrix @ vec
    
    # einsum으로: 'ij,j->i'는 (i,j) * (j) -> (i)를 뜻한다
    result_einsum = torch.einsum('ij,j->i', matrix, vec)
    
    print(f"Matrix: {matrix.shape}")
    print(f"Vector: {vec.shape}")
    print(f"Result: {result_einsum.shape}")
    print(f"Match? {torch.allclose(result_traditional, result_einsum)}")
    
    # -------------------------------------------------------------------------
    # 연습 문제
    # -------------------------------------------------------------------------
    header("Practice Exercises")
    
    print("""
    다음 학습을 해 보아라.
    
    1. (5,) 텐서를 (3, 5) 행렬에 더하여라
    2. (4, 1) 칸 벡터와 (1, 3) 줄 벡터를 곱하여라
    3. (10, 20) 행렬에서 칸 평균을 빼어 정규화하어라
    4. 1차원의 점 5개로 (5, 5) 거리 행렬을 만들어라
    5. (2, 3, 1) 텐서와 (1, 1, 4) 텐서를 펴 맞추어라
    """)
    
    # 해답
    t1 = torch.randn(5)
    t2 = torch.randn(3, 5)
    ex1 = t1 + t2
    print(f"\n1. {t1.shape} + {t2.shape} = {ex1.shape}")
    
    col = torch.randn(4, 1)
    row = torch.randn(1, 3)
    ex2 = col * row
    print(f"2. {col.shape} * {row.shape} = {ex2.shape}")
    
    data = torch.randn(10, 20)
    col_mean = data.mean(dim=0, keepdim=True)
    ex3 = data - col_mean
    print(f"3. Normalized: {ex3.shape}")
    
    points = torch.randn(5)
    distances = (points.unsqueeze(0) - points.unsqueeze(1)).abs()
    print(f"4. Distance matrix: {distances.shape}")
    
    t_a = torch.randn(2, 3, 1)
    t_b = torch.randn(1, 1, 4)
    ex5 = t_a + t_b
    print(f"5. {t_a.shape} + {t_b.shape} = {ex5.shape}")


if __name__ == "__main__":
    main()
```

??? note "전체 출력 (207줄)"

    ```

    ======================================================================
    1. What is Broadcasting?
    ======================================================================

        펴 맞추기는 원소별 셈을 할 때 모양이 다른 텐서도 다룰 수 있게 하는
        힘센 구조다.
        
        텐서를 직접 같은 크기로 늘리는 대신 PyTorch가 절로
        해 준다(데이터를 참으로 베끼지 않고 겉으로만).
        
    vec = tensor([1, 2, 3, 4])
    scalar = 10
    vec + scalar = tensor([11, 12, 13, 14])

    ======================================================================
    2. Broadcasting Rules
    ======================================================================

        다음이면 두 텐서를 "펴 맞출 수 있다".
        
        1. 텐서마다 차원이 적어도 하나 있거나, 한쪽이 홑값이다
        2. 마지막 차원부터 거꾸로 훑을 때
           - 차원이 같거나,
           - 한쪽 차원이 1이거나,
           - 한쪽 차원이 없다
        
        Examples:
        
      (3, 1) with (1, 4): → torch.Size([3, 4])
      (3, 4) with (3, 1): → torch.Size([3, 4])
      (1, 4, 5) with (3, 1, 5): → torch.Size([3, 4, 5])
      (3,) with (4, 3): → torch.Size([4, 3])

    ======================================================================
    3. Basic Broadcasting Patterns
    ======================================================================
    Pattern 1: Scalar with matrix
    Matrix:
    tensor([[0, 1, 2],
            [3, 4, 5]])
    Matrix + 100:
    tensor([[100, 101, 102],
            [103, 104, 105]])

    Pattern 2: Row vector with matrix
    vec = tensor([10, 20, 30])
    matrix + vec:
    tensor([[10, 21, 32],
            [13, 24, 35]])

    Pattern 3: Column vector with matrix
    col_vec:
    tensor([[10],
            [20]])
    matrix + col_vec:
    tensor([[10, 11, 12],
            [23, 24, 25]])


    ======================================================================
    4. Visualizing Broadcasting
    ======================================================================
    A (3x1):
     tensor([[0],
            [1],
            [2]])
    B (1x4):
     tensor([[0, 1, 2, 3]])

    A + B (broadcasts to 3x4):
    tensor([[0, 1, 2, 3],
            [1, 2, 3, 4],
            [2, 3, 4, 5]])

        어떻게 도는가:
        - A broadcasts: [[0],     →  [[0, 0, 0, 0],
                         [1],          [1, 1, 1, 1],
                         [2]]          [2, 2, 2, 2]]
        
        - B broadcasts: [[0, 1, 2, 3]] → [[0, 1, 2, 3],
                                          [0, 1, 2, 3],
                                          [0, 1, 2, 3]]
        

    ======================================================================
    5. Common Use Cases in ML
    ======================================================================
    Use case 1: Adding bias
    Layer output: torch.Size([32, 10])
    Bias: torch.Size([10])
    Result: torch.Size([32, 10])

    Use case 2: Data normalization
    Data: torch.Size([100, 5])
    Mean: torch.Size([1, 5])
    Normalized: torch.Size([100, 5])

    Use case 3: Pairwise distances
    Points A: torch.Size([5, 2])
    Points B: torch.Size([3, 2])
    Pairwise distances: torch.Size([5, 3])

    ======================================================================
    6. When Broadcasting Fails
    ======================================================================
    Broadcasting fails when shapes are incompatible:

    Error with (3,4) + (2,3): Shapes incompatible
      Reason: Neither dimension matches (3≠2, 4≠3)

    Fix: Reshape or unsqueeze to make compatible
    (3,4) + (3,1) = torch.Size([3, 4]) ✓


    ======================================================================
    7. keepdim=True for Broadcasting
    ======================================================================
    Data:
    tensor([[ 0.,  1.,  2.,  3.],
            [ 4.,  5.,  6.,  7.],
            [ 8.,  9., 10., 11.]])

    sum(dim=1): tensor([ 6., 22., 38.])
    Shape: torch.Size([3])

    sum(dim=1, keepdim=True):
    tensor([[ 6.],
            [22.],
            [38.]])
    Shape: torch.Size([3, 1])

    Data / row_sum:
    tensor([[0.0000, 0.1667, 0.3333, 0.5000],
            [0.1818, 0.2273, 0.2727, 0.3182],
            [0.2105, 0.2368, 0.2632, 0.2895]])

    ======================================================================
    8. Memory Efficiency
    ======================================================================

        핵심 통찰: 펴 맞추기는 데이터를 참으로 베끼지 않는다!
        
        (1, 100) 텐서를 (1000, 100)으로 펴 맞출 때 PyTorch는 베낌 1000개를
        만들지 않는다. 대신 슬기로운 자리 잡기로 같은 데이터를 겉으로만
        되쓴다.
        
        그래서 펴 맞추기는 빠르면서도 기억 자리를 아낀다.
        
    Small tensor: torch.Size([1, 3])
    Large tensor: torch.Size([1000, 3])
    Result: torch.Size([1000, 3])
    Memory multiplier: 1000.0x
    But only the small tensor is stored once!

    ======================================================================
    9. Explicit Broadcasting with expand()
    ======================================================================
    Original vector: tensor([1, 2, 3]), shape: torch.Size([3])

    Expanded: torch.Size([4, 3])
    Values:
    tensor([[1, 2, 3],
            [1, 2, 3],
            [1, 2, 3],
            [1, 2, 3]])

    Same storage? True

    broadcast_to(4, 3): torch.Size([4, 3])
    expand와 같은가: True
    역시 복사 안 함: True

    expand(2, 4, 3):       torch.Size([2, 4, 3])
    broadcast_to(2, 4, 3): torch.Size([2, 4, 3])
    둘이 같은가: True

    ======================================================================
    10. Advanced: einsum
    ======================================================================

        einsum(아인슈타인 합)은 펴 맞추기를 드러내 놓고 다스리며 텐서 셈을
        짧게 적는 길을 준다.
        
    Matrix: torch.Size([3, 4])
    Vector: torch.Size([4])
    Result: torch.Size([3])
    Match? True

    ======================================================================
    Practice Exercises
    ======================================================================

        다음 학습을 해 보아라.
        
        1. (5,) 텐서를 (3, 5) 행렬에 더하여라
        2. (4, 1) 칸 벡터와 (1, 3) 줄 벡터를 곱하여라
        3. (10, 20) 행렬에서 칸 평균을 빼어 정규화하어라
        4. 1차원의 점 5개로 (5, 5) 거리 행렬을 만들어라
        5. (2, 3, 1) 텐서와 (1, 1, 4) 텐서를 펴 맞추어라
        

    1. torch.Size([5]) + torch.Size([3, 5]) = torch.Size([3, 5])
    2. torch.Size([4, 1]) * torch.Size([1, 3]) = torch.Size([4, 3])
    3. Normalized: torch.Size([10, 20])
    4. Distance matrix: torch.Size([5, 5])
    5. torch.Size([2, 3, 1]) + torch.Size([1, 1, 4]) = torch.Size([2, 3, 4])
    ```
## 2. 논의

**규칙은 셋이다.** 두 모양을 **오른쪽에 맞추어** 세우고, 각 자리를 따로 본다.

1. 길이가 같으면 그대로 둔다.
2. 한쪽이 **1**이면 그쪽을 늘린다.
3. 한쪽에 자리가 **없으면** 1로 여기고 늘린다.

어느 자리에서도 이 셋에 걸리지 않으면 — 곧 둘 다 1이 아니고 서로 다르면 — 오류다.

```text
      (3, 1, 5)          (3, 4)
      (   4, 5)   →        (5,)   →  오류
    ───────────        ────────
      (3, 4, 5)          4 ≠ 5, 둘 다 1이 아니다
```

늘리는 일은 **베끼는 일이 아니다.** 스트라이드를 0으로 두어 같은 값을 여러 번 읽게 할 뿐이므로 메모리가 늘지 않는다. `(32, 1)`을 `(32, 1000)`으로 펴 맞추어도 숫자는 32개뿐이다.

!!! warning "오류가 나지 않는 쪽이 더 위험하다"
    모양이 어긋나면 오류가 나므로 금방 안다. 문제는 규칙에 **맞으면서** 뜻과 다른 모양이 나오는 경우다.

    **`(N, 1)`과 `(N,)`.** 가장 흔한 자리다. 모델의 출력이 `(32, 1)`이고 정답이 `(32,)`이면, 둘을 빼면 `(32,)`가 아니라 `(32, 32)`가 된다.

    ```python
    pred = torch.randn(32, 1); tgt = torch.randn(32)
    (pred - tgt).shape        # torch.Size([32, 32])  ← 32개가 1024개가 되었다
    ```

    오른쪽에 맞추면 `(32, 1)`과 `(1, 32)`가 되어 두 자리 모두 늘어나기 때문이다. 손실 함수도 막아 주지 않는다 — 경고만 하고 **다른 값을 돌려준다.**

    ```python
    F.mse_loss(pred, tgt)                 # 2.1565  ← 경고만 난다
    F.mse_loss(pred, tgt.unsqueeze(1))    # 2.3581  ← 이것이 맞다
    ```

    학습이 돌아가고 손실도 줄어드는데 재는 것이 엉뚱해진다. `squeeze(-1)`이나 `unsqueeze(1)`로 **모양을 맞춰서** 넘긴다.

    **축약 뒤의 `keepdim`.** 평균을 빼서 중심을 맞추는 흔한 셈에서도 같은 일이 생긴다.

    ```python
    x - x.mean(1)                  # 뜻과 다르다
    x - x.mean(1, keepdim=True)    # 이것이 맞다
    ```

    `x.mean(1)`은 그 차원을 **없애므로** `(4, 5)`가 `(5,)`가 아니라 `(4,)`가 된다. 오른쪽에 맞추면 `(4,)`가 마지막 자리에 붙어 열 방향으로 늘어난다.

    고약한 것은 **정사각 텐서에서는 오류가 나지 않는다**는 점이다. `(5, 5)`에서는 길이가 맞아떨어져 조용히 다른 값을 낸다.

    ```python
    x = torch.arange(25.).reshape(5, 5)
    (x - x.mean(1))[0]                 # [-2., -6., -10., -14., -18.]
    (x - x.mean(1, keepdim=True))[0]   # [-2., -1.,   0.,   1.,   2.]
    ```

    작은 정사각 예제로 시험하면 지나가고, 직사각 자료를 만나면 오류가 난다. 축약한 뒤 그 결과를 원래 텐서와 셈할 것이라면 **늘 `keepdim=True`**를 쓴다.

**지키는 습관 하나.** 펴 맞추기에 기대는 줄에서는 모양을 눈으로 확인하거나, 오른쪽에 맞춘 뒤 어떻게 늘어나는지 소리내어 읽어 본다. 모양을 적어 두는 것도 값싼 보험이다.

```python
assert pred.shape == tgt.shape, (pred.shape, tgt.shape)
```

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
`(3, 1, 5)`와 `(4, 5)`를 더하면 모양이 무엇이 되는지 **오른쪽에 맞추어** 손으로 셈한 뒤 확인하라. 이어서 `torch.outer`를 쓰지 않고 펴 맞추기만으로 $a = [1, 2, 3]$과 $b = [4, 5]$의 외적을 만들어라.

</div>

??? success "연습문제 1 풀이"
    오른쪽에 맞추어 세운다.

    ```text
      (3, 1, 5)
      (   4, 5)
    ───────────
      (3, 4, 5)
    ```

    마지막 자리는 5와 5로 같고, 가운데는 1이 4로 늘어나고, 맨 앞은 자리가 없으니 1로 여겨 3이 된다.

    ```python
    import torch

    print((torch.randn(3, 1, 5) + torch.randn(4, 5)).shape)

    a = torch.tensor([1., 2., 3.])
    b = torch.tensor([4., 5.])
    outer = a.unsqueeze(1) * b.unsqueeze(0)      # (3,1) * (1,2) -> (3,2)
    print(outer)
    print("torch.outer와 같은가:", torch.allclose(outer, torch.outer(a, b)))
    ```

    ```
    torch.Size([3, 4, 5])
    tensor([[ 4.,  5.],
            [ 8., 10.],
            [12., 15.]])
    torch.outer와 같은가: True
    ```

    외적이 펴 맞추기 한 번인 까닭은, `(3, 1)`과 `(1, 2)`가 두 자리 모두 늘어나
    `(3, 2)`가 되면서 모든 짝이 한 번씩 만나기 때문이다.

---


<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
`torch.randn(3, 4) + torch.randn(5)`는 오류가 난다. 오류 메시지를 적고 왜 규칙에 걸리는지 설명하라. 그리고 **뜻이 서로 다른** 두 가지 고침을 제시하라.

</div>

??? success "연습문제 2 풀이"
    ```python
    import torch

    try:
        torch.randn(3, 4) + torch.randn(5)
    except RuntimeError as e:
        print(str(e)[:78])
    ```

    ```
    The size of tensor a (4) must match the size of tensor b (5) at non-singleton
    ```

    오른쪽에 맞추면 마지막 자리가 4와 5다. 같지도 않고 어느 쪽도 1이 아니므로 세 규칙에
    모두 걸리지 않는다.

    고침은 **내가 무엇을 하려 했는지**에 달려 있다. 모양만 맞추는 문제가 아니다.

    ```python
    # (가) 길이 5가 맞다면 — 행 넷에 길이 5인 벡터를 더하려던 것
    x = torch.randn(3, 5)
    print((x + torch.randn(5)).shape)

    # (나) 길이 3이 맞다면 — 행마다 다른 수 하나를 더하려던 것
    y = torch.randn(3, 4)
    print((y + torch.randn(3).unsqueeze(1)).shape)
    ```

    ```
    torch.Size([3, 5])
    torch.Size([3, 4])
    ```

    (가)는 벡터를 **행마다** 더하고 (나)는 **열마다** 더한다. 둘은 전혀 다른 셈이다.
    그래서 "모양을 맞추면 된다"고 아무 쪽이나 고르면 오류는 사라지지만 셈이 틀린다.

---


<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
아래 둘은 오류가 나지 않는다. 그런데 둘 다 뜻과 다르다.

```python
# (가) 모델 출력과 정답
pred = torch.randn(32, 1); tgt = torch.randn(32)
diff = pred - tgt

# (나) 행마다 평균을 뺀다
x = torch.arange(25.).reshape(5, 5)
centered = x - x.mean(1)
```

(가)에서 `diff`의 모양을 확인하고 `F.mse_loss(pred, tgt)`가 올바른 값과 얼마나 다른지 재어라. (나)에서는 `x`를 `(4, 5)`로 바꾸면 무슨 일이 생기는지 확인하고, **정사각일 때 오류가 나지 않는 것이 왜 더 나쁜지** 설명하라.

</div>

??? success "연습문제 3 풀이"
    ```python
    import torch
    import torch.nn.functional as F

    torch.manual_seed(0)
    pred = torch.randn(32, 1)
    tgt = torch.randn(32)
    print("(pred - tgt).shape :", (pred - tgt).shape)
    print("mse_loss(pred, tgt)             :", round(F.mse_loss(pred, tgt).item(), 4))
    print("mse_loss(pred, tgt.unsqueeze(1)):",
          round(F.mse_loss(pred, tgt.unsqueeze(1)).item(), 4))
    ```

    ```
    (pred - tgt).shape : torch.Size([32, 32])
    mse_loss(pred, tgt)             : 2.1565
    mse_loss(pred, tgt.unsqueeze(1)): 2.3581
    ```

    `(32, 1)`과 `(32,)`를 오른쪽에 맞추면 `(32, 1)`과 `(1, 32)`가 되어 **두 자리 모두**
    늘어난다. 그래서 32개를 견주려던 것이 1024개의 짝이 되고, 손실도 2.1565와 2.3581로
    다르다. 손실 함수는 경고만 하고 값을 돌려준다 — 학습은 돌아가고 손실도 줄지만 재는
    것이 엉뚱하다.

    ```python
    for n in (5, 4):
        x = torch.arange(n * 5.).reshape(n, 5)
        try:
            print(f"({n},5):", (x - x.mean(1)).shape)
        except RuntimeError as e:
            print(f"({n},5): RuntimeError -", str(e)[:52])
    ```

    ```
    (5,5): torch.Size([5, 5])
    (4,5): RuntimeError - The size of tensor a (5) must match the size of
    ```

    `x.mean(1)`은 1번 차원을 **없애므로** `(5, 5)`에서 `(5,)`가 나온다. 오른쪽에 맞추면
    그 `(5,)`가 **마지막** 자리에 붙어 열 방향으로 늘어난다. 행마다 빼려 했는데 열마다
    빼는 셈이 된다.

    ```python
    x = torch.arange(25.).reshape(5, 5)
    print("wrong:", (x - x.mean(1))[0].tolist())
    print("right:", (x - x.mean(1, keepdim=True))[0].tolist())
    ```

    ```
    wrong: [-2.0, -6.0, -10.0, -14.0, -18.0]
    right: [-2.0, -1.0, 0.0, 1.0, 2.0]
    ```

    **정사각일 때 오류가 나지 않는 것이 더 나쁜 까닭**은 시험이 통과하기 때문이다.
    `(5, 5)`로 적은 작은 예제에서는 길이가 맞아떨어져 조용히 다른 값을 내고, 그래서
    맞다고 믿고 넘어간다. 직사각 자료를 만나는 순간 오류가 나는데, 그때는 이 줄이
    이미 통과한 줄로 남아 있어 의심하지 않는다.

    축약한 결과를 원래 텐서와 셈할 것이라면 **늘 `keepdim=True`**를 쓴다.

## 정리하며

펴 맞추기는 두 모양을 **오른쪽에 맞추어** 자리마다 따로 본다. 같으면 두고, 한쪽이 1이면 늘리고, 자리가 없으면 1로 여긴다. 어느 자리에서도 이에 걸리지 않으면 오류다.

늘리는 일은 **베끼는 일이 아니다.** 스트라이드를 0으로 두어 같은 값을 여러 번 읽으므로 메모리가 늘지 않는다.

**조심할 것은 오류가 나는 쪽이 아니라 나지 않는 쪽이다.** 모양이 어긋나면 그 자리에서 알게 되지만, 규칙에 맞으면서 뜻과 다른 모양이 되면 아무 말이 없다.

- **`(N, 1)`과 `(N,)`** — 빼면 `(N,)`이 아니라 `(N, N)`이 된다. 손실 함수도 경고만 하고 다른 값을 돌려주므로, 학습이 돌아가면서 재는 것이 엉뚱해진다.
- **축약 뒤의 `keepdim`** — `x.mean(1)`은 차원을 없애므로 다시 붙을 때 자리가 밀린다. **정사각 텐서에서는 오류가 나지 않아** 작은 예제를 통과하고 실제 자료에서 드러난다.

그러므로 펴 맞추기에 기대는 줄에서는 모양을 확인한다. `assert pred.shape == tgt.shape`는 값싼 보험이다.
