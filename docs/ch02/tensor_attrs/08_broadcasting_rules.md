# 브로드캐스팅 규칙

이 스크립트는 브로드캐스팅 규칙을 보여준다. 이 개념들을 이해하는 것은 효과적인 PyTorch 프로그래밍과 딥러닝 모델 개발에 필수적이다.

## 코드

```python
#!/usr/bin/env python3
"""
펴 맞추기 규칙: 원소별 셈이 되도록 텐서를 맞춘다.

Covers:
- 펴 맞추기 규칙 셋
- 흔한 펴 맞추기 무늬
- 차원 맞추기(뒤쪽 차원부터)
- 크기 1인 차원 늘리기
- 모양을 곁들인 눈으로 보는 보기
- 흔한 함정과 탈 잡기
"""

import torch

# ========================================================================
# 메인
# ========================================================================

def header(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

def show_broadcast(a, b, op_name="+"):
    """펴 맞추기를 그림으로 보이는 도우미."""
    result = a + b
    print(f"  {a.shape} {op_name} {b.shape} → {result.shape}")
    return result

def main():
    torch.manual_seed(0)

    # -------------------------------------------------------------------------
    header("Broadcasting Rule 1: Align from trailing (rightmost) dimensions")
    a = torch.randn(3, 4, 5)  # Shape: (3, 4, 5)
    b = torch.randn(5)        # Shape: (5,)
    # 정렬:
    #   a: (3, 4, 5)
    #   b:    (  5)  ← 암묵적으로 (1, 1, 5)
    result = show_broadcast(a, b)

    # -------------------------------------------------------------------------
    header("Broadcasting Rule 2: Prepend 1s to shorter tensor")
    a = torch.randn(3, 4, 5)  # Shape: (3, 4, 5)
    b = torch.randn(4, 5)     # Shape: (4, 5)
    # 앞에 1을 붙인 뒤:
    #   a: (3, 4, 5)
    #   b: (1, 4, 5)  ← 앞에 1이 붙는다
    result = show_broadcast(a, b)

    # -------------------------------------------------------------------------
    header("Broadcasting Rule 3: Size-1 dimensions expand to match")
    a = torch.randn(3, 1, 5)  # Shape: (3, 1, 5)
    b = torch.randn(1, 4, 5)  # Shape: (1, 4, 5)
    # 브로드캐스팅:
    #   a: (3, 1, 5) → (3, 4, 5)
    #   b: (1, 4, 5) → (3, 4, 5)
    result = show_broadcast(a, b)
    print(f"  Both broadcast to {result.shape}")

    # -------------------------------------------------------------------------
    header("Scalar broadcasting (0-D tensor)")
    a = torch.randn(3, 4)
    scalar = torch.tensor(5.0)  # Shape: ()
    # 스칼라는 어떤 모양으로도 브로드캐스팅된다
    result = show_broadcast(a, scalar, "*")

    # -------------------------------------------------------------------------
    header("Common pattern: (batch, features) + (features,)")
    batch_data = torch.randn(32, 128)  # 32 samples, 128 features
    bias = torch.randn(128)             # Per-feature bias
    # 브로드캐스팅: (32, 128) + (128,) → (32, 128)
    result = show_broadcast(batch_data, bias)
    print("  Common in neural networks: adding bias to batched data")

    # -------------------------------------------------------------------------
    header("Common pattern: (batch, channels, H, W) + (channels, 1, 1)")
    images = torch.randn(8, 3, 64, 64)  # Batch of RGB images
    channel_scale = torch.randn(3, 1, 1)  # Per-channel scaling
    # 브로드캐스팅: (8, 3, 64, 64) + (3, 1, 1) → (8, 3, 64, 64)
    result = show_broadcast(images, channel_scale, "*")
    print("  Common in CNNs: per-channel operations")

    # -------------------------------------------------------------------------
    header("Matrix-vector broadcasting")
    matrix = torch.randn(5, 4)  # Shape: (5, 4)
    vector = torch.randn(4)     # Shape: (4,)
    # 행 방향 연산: (5, 4) + (4,) → (5, 4)
    row_result = matrix + vector
    print(f"  Row-wise: {matrix.shape} + {vector.shape} → {row_result.shape}")
    
    # 열 방향으로 하려면 벡터를 재구성해야 한다
    col_vector = torch.randn(5, 1)  # Shape: (5, 1)
    col_result = matrix + col_vector
    print(f"  Col-wise: {matrix.shape} + {col_vector.shape} → {col_result.shape}")

    # -------------------------------------------------------------------------
    header("Outer product via broadcasting")
    a = torch.tensor([1., 2., 3.]).view(-1, 1)  # (3, 1)
    b = torch.tensor([10., 20., 30., 40.]).view(1, -1)  # (1, 4)
    # 브로드캐스팅: (3, 1) * (1, 4) → (3, 4)
    outer = a * b
    print(f"  {a.shape} * {b.shape} → {outer.shape}")
    print("  Outer product:\n", outer)

    # -------------------------------------------------------------------------
    header("Broadcasting incompatibility (will fail)")
    a = torch.randn(3, 4)
    b = torch.randn(5)
    try:
        result = a + b  # Incompatible: (3, 4) vs (5,)
    except RuntimeError as e:
        print(f"  ERROR (expected): {str(e)[:60]}...")
        print(f"  (3, 4) vs (5,) → dimension 1: 4 ≠ 5")

    # -------------------------------------------------------------------------
    header("Broadcasting with unsqueeze to add size-1 dimensions")
    a = torch.randn(3, 4)     # (3, 4)
    b = torch.randn(4)        # (4,)
    
    # 1번 차원이 아니라 0번 차원을 따라 브로드캐스팅하고 싶다
    b_col = b.unsqueeze(0)    # (1, 4)
    result = a + b_col
    print(f"  {a.shape} + {b_col.shape} → {result.shape}")
    
    # 또는 열로 브로드캐스팅
    b_col2 = b.unsqueeze(1)   # (4, 1)
    try:
        result = a + b_col2   # (3, 4) vs (4, 1) → incompatible!
    except RuntimeError as e:
        print(f"  ERROR: (3, 4) + (4, 1) incompatible")
    
    # 열 방향 브로드캐스팅을 위해 제대로 재구성해야 한다
    c = torch.randn(3)
    c_col = c.unsqueeze(1)    # (3, 1)
    result = a + c_col        # (3, 4) + (3, 1) → (3, 4)
    print(f"  {a.shape} + {c_col.shape} → {result.shape} ✓")

    # -------------------------------------------------------------------------
    header("Explicit broadcast_to for clarity")
    a = torch.tensor([1., 2., 3.])  # (3,)
    target_shape = (4, 3)
    
    # 자동 브로드캐스팅
    b = torch.zeros(4, 3)
    result = b + a  # (4, 3) + (3,) → (4, 3)
    print(f"  Automatic: {b.shape} + {a.shape} → {result.shape}")
    
    # 명시적 broadcast_to(뷰를 반환한다)
    a_broadcast = a.broadcast_to(target_shape)
    print(f"  Explicit: broadcast_to({target_shape}) → {a_broadcast.shape}")
    print(f"  Is view (shares storage): {id(a.storage()) == id(a_broadcast.storage())}")

    # -------------------------------------------------------------------------
    header("Common bug: unintended broadcasting with missing dimensions")
    # 이미지 배치가 있고 이미지별 평균을 구하고 싶다고 하자
    images = torch.randn(10, 3, 32, 32)  # 10 RGB images, 32x32
    
    # 잘못된 방법: 공간 차원에 대해 평균을 내면서 keepdim을 잊었다
    wrong_mean = images.mean(dim=(2, 3))  # Shape: (10, 3)
    print(f"  Wrong mean shape: {wrong_mean.shape}")
    try:
        normalized = images - wrong_mean  # (10,3,32,32) - (10,3) → broadcasts incorrectly!
        print(f"  Result shape: {normalized.shape}")
        print("  ⚠️  This broadcasts but NOT as intended!")
    except:
        pass
    
    # 올바른 방법: keepdim=True를 쓴다
    correct_mean = images.mean(dim=(2, 3), keepdim=True)  # Shape: (10, 3, 1, 1)
    print(f"  Correct mean shape: {correct_mean.shape}")
    normalized = images - correct_mean  # (10,3,32,32) - (10,3,1,1) ✓
    print(f"  Correct result shape: {normalized.shape}")

    # -------------------------------------------------------------------------
    header("Broadcasting with torch.where (conditional selection)")
    a = torch.randn(3, 4)
    b = torch.randn(4)        # Broadcasts to (3, 4)
    condition = a > 0
    
    result = torch.where(condition, a, b)  # Select a or b based on condition
    print(f"  where: {a.shape}, {b.shape} → {result.shape}")

    # -------------------------------------------------------------------------
    header("Memory efficiency: broadcasting creates views, not copies")
    small = torch.tensor([1., 2., 3.])
    large_shape = (1000, 3)
    
    # 브로드캐스팅은 데이터를 복사하지 않는다
    broadcasted = small.expand(large_shape)  # Returns a view
    print(f"  Broadcasted shape: {broadcasted.shape}")
    print(f"  Shares storage: {id(small.storage()) == id(broadcasted.storage())}")
    print(f"  Storage size: {small.storage().size()} elements (only original data)")

    # -------------------------------------------------------------------------
    header("Quick reference: broadcasting shape compatibility")
    print("  Examples of compatible shapes:")
    examples = [
        ((5, 1, 7), (3, 7), (5, 3, 7)),
        ((3, 1), (1, 4), (3, 4)),
        ((8, 1, 6, 1), (7, 1, 5), (8, 7, 6, 5)),
        ((5,), (3, 5), (3, 5)),
        ((), (3, 4), (3, 4)),  # Scalar
    ]
    for shape_a, shape_b, result_shape in examples:
        print(f"    {str(shape_a):20s} + {str(shape_b):20s} → {result_shape}")
    
    print("\n  Examples of INCOMPATIBLE shapes:")
    incompatible = [
        ((3, 4), (5,), "dims don't match: 4 ≠ 5"),
        ((2, 3), (3, 2), "both dims non-1: 2≠3 and 3≠2"),
    ]
    for shape_a, shape_b, reason in incompatible:
        print(f"    {str(shape_a):20s} + {str(shape_b):20s} ✗ ({reason})")

if __name__ == "__main__":
    main()```

## 논의

브로드캐스팅은 작은 텐서를 가상으로 확장하여 모양이 다른 텐서 사이의 원소별 연산을 가능하게 한다. PyTorch는 차원을 오른쪽부터 맞추며, 각 차원 쌍이 서로 같거나, 둘 중 하나가 1이거나, 아예 없을 것을 요구한다. 이로써 데이터를 명시적으로 복제하지 않아도 되어 메모리 효율이 좋고 빠르다.

재구성 연산은 데이터를 반드시 복사하지 않으면서 텐서의 논리적 배치를 바꾼다. `view()` 메서드는 연속된 메모리를 요구하며 항상 뷰를 반환하고, `reshape()`는 (필요하면 복사하여) 어떤 텐서에서도 동작한다. `transpose()`나 `permute()` 같은 연산은 데이터 배치가 아니라 스트라이드를 바꾸므로 결과가 연속적이지 않을 수 있다.

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
