# 축약 연산

이 스크립트는 텐서의 축약 연산을 보여준다. 이 개념들을 이해하는 것은 효과적인 PyTorch 프로그래밍과 딥러닝 모델 개발에 필수적이다.

## 코드

```python
"""튜토리얼 12: 줄이는 셈 - 텐서 데이터 모으기(sum, mean, min, max 따위)"""
import torch

# ========================================================================
# 메인
# ========================================================================

def header(title): print(f"\n{'='*70}\n{title}\n{'='*70}")

def main():
    header("1. Basic Reductions - Sum, Mean, Product")
    x = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)
    print(f"Tensor:\n{x}\n")
    print(f"sum(): {x.sum()}")  # All elements
    print(f"mean(): {x.mean()}")
    print(f"prod(): {x.prod()}")  # Product of all
    print(f"sum(dim=0): {x.sum(dim=0)}")  # Column sums
    print(f"sum(dim=1): {x.sum(dim=1)}")  # Row sums
    print(f"sum(dim=1, keepdim=True):\n{x.sum(dim=1, keepdim=True)}")
    
    header("2. Min and Max")
    print(f"min(): {x.min()}")
    print(f"max(): {x.max()}")
    min_val, min_idx = x.min(dim=1)  # Returns values and indices
    print(f"min(dim=1): values={min_val}, indices={min_idx}")
    print(f"argmin(): {x.argmin()}")  # Flattened index
    print(f"argmax(dim=0): {x.argmax(dim=0)}")
    
    header("3. Statistical Operations")
    data = torch.randn(100, 10)
    print(f"Data shape: {data.shape}")
    print(f"std(): {data.std():.4f}")  # Standard deviation
    print(f"var(): {data.var():.4f}")  # Variance
    print(f"median(): {data.median():.4f}")
    print(f"std(dim=0) shape: {data.std(dim=0).shape}")
    
    header("4. Logical Reductions")
    bool_tensor = torch.tensor([[True, False, True], [False, False, True]])
    print(f"Bool tensor:\n{bool_tensor}\n")
    print(f"all(): {bool_tensor.all()}")  # Are all True?
    print(f"any(): {bool_tensor.any()}")  # Is any True?
    print(f"all(dim=1): {bool_tensor.all(dim=1)}")
    print(f"any(dim=0): {bool_tensor.any(dim=0)}")
    
    header("5. Counting and Finding")
    x = torch.tensor([1, 2, 3, 2, 1, 2, 3])
    print(f"Tensor: {x}")
    print(f"unique(): {torch.unique(x)}")
    print(f"Number of elements: {x.numel()}")
    counts = torch.bincount(x)
    print(f"bincount(): {counts}")  # Count occurrences of each value
    
    header("6. Norm Operations")
    vec = torch.tensor([3.0, 4.0])
    print(f"Vector: {vec}")
    print(f"L2 norm (Euclidean): {torch.norm(vec, p=2)}")
    print(f"L1 norm (Manhattan): {torch.norm(vec, p=1)}")
    mat = torch.randn(3, 4)
    print(f"\nMatrix norm: {torch.norm(mat)}")
    print(f"Frobenius norm: {torch.norm(mat, p='fro')}")
    
    header("7. Cumulative Operations")
    x = torch.tensor([1, 2, 3, 4, 5])
    print(f"Tensor: {x}")
    print(f"cumsum(): {torch.cumsum(x, dim=0)}")  # Cumulative sum
    print(f"cumprod(): {torch.cumprod(x, dim=0)}")  # Cumulative product
    mat = torch.arange(12).reshape(3, 4)
    print(f"\nMatrix:\n{mat}")
    print(f"cumsum(dim=1):\n{torch.cumsum(mat, dim=1)}")

if __name__ == "__main__":
    main()```

## 논의

재구성 연산은 데이터를 반드시 복사하지 않으면서 텐서의 논리적 배치를 바꾼다. `view()` 메서드는 연속된 메모리를 요구하며 항상 뷰를 반환하고, `reshape()`는 (필요하면 복사하여) 어떤 텐서에서도 동작한다. `transpose()`나 `permute()` 같은 연산은 데이터 배치가 아니라 스트라이드를 바꾸므로 결과가 연속적이지 않을 수 있다.

텐서 생성 함수는 데이터를 초기화하는 유연한 방법을 제공한다. `torch.zeros`, `torch.randn`, `torch.arange` 같은 팩토리 함수는 `dtype`, `device`, `requires_grad` 매개변수를 받으므로 불필요한 복사 없이 목표 장치에 곧바로 할당할 수 있다.

`sum()`, `mean()`, `min()`, `max()` 같은 축약 연산은 텐서의 하나 이상의 차원을 접는다. `dim` 매개변수가 어느 차원을 축약할지 지정하며, `keepdim=True`는 축약된 차원을 크기 1로 남겨 둔다. 이는 이후 연산에서 올바른 브로드캐스팅을 위해 필수적이다.

## 연습문제

**연습문제 1.**
모양이 $(24,)$인 텐서를 만들어 $(2, 3, 4)$로 재구성하라. 그런 다음 차원을 $(4, 2, 3)$으로 바꾸고 전체 원소 개수가 변하지 않았음을 확인하라.

??? success "연습문제 1 풀이"
    ```python
    x = torch.arange(24)
    x_3d = x.reshape(2, 3, 4)
    x_perm = x_3d.permute(2, 0, 1)
    assert x_perm.shape == (4, 2, 3)
    assert x_perm.numel() == 24
    ```

---


**연습문제 2.**
`transpose()` 후에 `view()`는 실패하지만 `reshape()`는 성공하는 이유를 설명하라. `contiguous()`는 무엇을 하는가?

??? success "연습문제 2 풀이"
    `transpose()`는 스트라이드는 다르지만 바탕 저장소는 같은 뷰를 반환한다. 데이터가 더 이상 행 우선 순서가 아니므로 (연속된 메모리를 요구하는) `view()`가 실패한다. `reshape()`는 비연속성을 감지하여 복사본을 만든다. `.contiguous()`를 호출하면 행 우선 배치를 가진 새 텐서가 명시적으로 만들어지며, 그 뒤에는 `view()`가 동작한다.

---


**연습문제 3.**
NCHW 형식의 모양 $(32, 3, 224, 224)$인 이미지 배치가 주어졌을 때 `permute()`로 NHWC 형식으로 바꾸고, 공간 차원을 펼쳐 모양 $(32, 3, 50176)$을 얻어라.

??? success "연습문제 3 풀이"
    ```python
    images = torch.randn(32, 3, 224, 224)
    nhwc = images.permute(0, 2, 3, 1)  # (32, 224, 224, 3)
    # (32, 3, 50176)을 얻으려면 원본에서 H와 W를 펼친다:
    flat = images.flatten(start_dim=2)  # (32, 3, 50176)
    print(flat.shape)
    ```
