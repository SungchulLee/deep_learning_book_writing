# PyTorch의 브로드캐스팅

이 스크립트는 PyTorch의 브로드캐스팅을 보여준다. 이 개념들을 이해하는 것은 효과적인 PyTorch 프로그래밍과 딥러닝 모델 개발에 필수적이다.

## 코드

```python
"""
익힘 11: PyTorch의 펴 맞추기
=====================================

펴 맞추기 덕에 PyTorch은 꼴이 다른 텐서끼리도 자료를 베끼지 않고
절로 맞는 꼴로 늘려 셈할 수 있다.

고갱이 개념:
- 펴 맞추기 규칙
- 흔한 펴 맞추기 무늬
- 펴 맞추기가 안 되는 때
- 기억 자리 아끼기
- 신경망에서의 펴 맞추기
"""

import torch

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
    펴 맞추기는 원소별 셈을 할 때 꼴이 다른 텐서도 다룰 수 있게 하는
    힘센 얼개다.
    
    텐서를 손수 같은 크기로 늘리는 대신 PyTorch이 절로
    해 준다(자료를 참으로 베끼지 않고 겉으로만).
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
    고갱이 눈썰미: 펴 맞추기는 자료를 참으로 베끼지 않는다!
    
    (1, 100) 텐서를 (1000, 100)으로 펴 맞출 때 PyTorch은 베낌 1000개를
    만들지 않는다. 대신 슬기로운 자리 잡기로 같은 자료를 겉으로만
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
    print(f"\nSame storage? {vec.storage().data_ptr() == expanded.storage().data_ptr()}")
    
    # 주의: 확장된 텐서를 수정하면 예상치 못한 결과가 생길 수 있다!
    # 확장된 텐서를 제자리에서 수정하지 말 것
    
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
    다음 익힘을 해 보아라.
    
    1. (5,) 텐서를 (3, 5) 행렬에 더하여라
    2. Multiply a (4, 1) column vector with a (1, 3) row vector
    3. Normalize a (10, 20) matrix by subtracting column means
    4. Create a (5, 5) distance matrix from 5 points in 1D
    5. Broadcast a (2, 3, 1) tensor with a (1, 1, 4) tensor
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
    main()```

## 논의

브로드캐스팅은 작은 텐서를 가상으로 확장하여 모양이 다른 텐서 사이의 원소별 연산을 가능하게 한다. PyTorch는 차원을 오른쪽부터 맞추며, 각 차원 쌍이 서로 같거나, 둘 중 하나가 1이거나, 아예 없을 것을 요구한다. 이로써 데이터를 명시적으로 복제하지 않아도 되어 메모리 효율이 좋고 빠르다.

행렬 연산은 신경망 계산의 핵심을 이룬다. `@` 연산자와 `torch.matmul()`은 배치 차원에 대한 자동 브로드캐스팅과 함께 행렬 곱을 처리하고, `torch.mm()`이나 `torch.bmm()` 같은 특화된 함수는 명료함과 성능을 위해 텐서의 계수를 특정 값으로 강제한다.

재구성 연산은 데이터를 반드시 복사하지 않으면서 텐서의 논리적 배치를 바꾼다. `view()` 메서드는 연속된 메모리를 요구하며 항상 뷰를 반환하고, `reshape()`는 (필요하면 복사하여) 어떤 텐서에서도 동작한다. `transpose()`나 `permute()` 같은 연산은 데이터 배치가 아니라 스트라이드를 바꾸므로 결과가 연속적이지 않을 수 있다.

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
