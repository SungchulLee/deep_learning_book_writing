# 이어 붙이기와 쌓기 - 텐서 결합

이 스크립트는 이어 붙이기와 쌓기로 텐서를 결합하는 방법을 보여준다. 이 개념들을 이해하는 것은 효과적인 PyTorch 프로그래밍과 딥러닝 모델 개발에 필수적이다.

## 코드

```python
"""학습 16: 이어 붙이기와 쌓기 - 텐서 아우르기"""
import torch

# ========================================================================
# 메인
# ========================================================================

def header(title): print(f"\n{'='*70}\n{title}\n{'='*70}")

def main():
    header("1. Concatenation - torch.cat()")
    a = torch.tensor([[1, 2], [3, 4]])
    b = torch.tensor([[5, 6], [7, 8]])
    print(f"a =\n{a}\nb =\n{b}\n")
    cat_dim0 = torch.cat([a, b], dim=0)  # Vertical stacking
    print(f"cat(dim=0) - Stack vertically:\n{cat_dim0}")
    cat_dim1 = torch.cat([a, b], dim=1)  # Horizontal stacking
    print(f"\ncat(dim=1) - Stack horizontally:\n{cat_dim1}")
    c = torch.tensor([[9, 10], [11, 12]])
    cat_multi = torch.cat([a, b, c], dim=0)
    print(f"\nCat multiple tensors:\n{cat_multi}")
    
    header("2. Stacking - torch.stack()")
    x = torch.tensor([1, 2, 3])
    y = torch.tensor([4, 5, 6])
    print(f"x = {x}\ny = {y}\n")
    stack_dim0 = torch.stack([x, y], dim=0)
    print(f"stack(dim=0):\n{stack_dim0}")
    print(f"Shape: {stack_dim0.shape}")  # (2, 3)
    stack_dim1 = torch.stack([x, y], dim=1)
    print(f"\nstack(dim=1):\n{stack_dim1}")
    print(f"Shape: {stack_dim1.shape}")  # (3, 2)
    
    header("3. cat() vs stack()")
    a = torch.randn(2, 3)
    b = torch.randn(2, 3)
    print(f"a shape: {a.shape}, b shape: {b.shape}")
    cat_result = torch.cat([a, b], dim=0)
    stack_result = torch.stack([a, b], dim=0)
    print(f"cat(dim=0) shape: {cat_result.shape}")  # (4, 3)
    print(f"stack(dim=0) shape: {stack_result.shape}")  # (2, 2, 3)
    print("\nKey difference:")
    print("- cat(): Concatenates along existing dimension")
    print("- stack(): Creates new dimension for stacking")
    
    header("4. Splitting - torch.split()")
    tensor = torch.arange(10)
    print(f"Tensor: {tensor}")
    splits = torch.split(tensor, 3)  # Split into chunks of size 3
    print(f"split(3): {splits}")
    splits_sizes = torch.split(tensor, [3, 3, 4])  # Custom sizes
    print(f"split([3,3,4]): {splits_sizes}")
    
    header("5. Chunking - torch.chunk()")
    tensor = torch.arange(12).reshape(3, 4)
    print(f"Tensor:\n{tensor}")
    chunks = torch.chunk(tensor, 2, dim=0)  # Split into 2 chunks along dim 0
    print(f"chunk(2, dim=0):")
    for i, chunk in enumerate(chunks):
        print(f"  Chunk {i}:\n{chunk}")
    
    header("6. Unbinding - torch.unbind()")
    stacked = torch.arange(12).reshape(3, 4)
    print(f"Stacked:\n{stacked}")
    unbound = torch.unbind(stacked, dim=0)  # Unpack along dimension
    print(f"unbind(dim=0): {len(unbound)} tensors")
    for i, t in enumerate(unbound):
        print(f"  Tensor {i}: {t}")
    
    header("7. Practical: Building Batch")
    sample1 = torch.randn(3, 32, 32)  # Image 1
    sample2 = torch.randn(3, 32, 32)  # Image 2
    sample3 = torch.randn(3, 32, 32)  # Image 3
    batch = torch.stack([sample1, sample2, sample3], dim=0)
    print(f"Batch shape: {batch.shape}")  # (3, 3, 32, 32)
    print("Format: (batch_size, channels, height, width)")
    
    header("8. Practical: Feature Concatenation")
    features_a = torch.randn(10, 64)  # 10 samples, 64 features
    features_b = torch.randn(10, 32)  # 10 samples, 32 features
    combined = torch.cat([features_a, features_b], dim=1)
    print(f"features_a: {features_a.shape}")
    print(f"features_b: {features_b.shape}")
    print(f"combined: {combined.shape}")  # (10, 96)

if __name__ == "__main__":
    main()```

## 논의

재구성 연산은 데이터를 반드시 복사하지 않으면서 텐서의 논리적 배치를 바꾼다. `view()` 메서드는 연속된 메모리를 요구하며 항상 뷰를 반환하고, `reshape()`는 (필요하면 복사하여) 어떤 텐서에서도 동작한다. `transpose()`나 `permute()` 같은 연산은 데이터 배치가 아니라 스트라이드를 바꾸므로 결과가 연속적이지 않을 수 있다.

텐서 생성 함수는 데이터를 초기화하는 유연한 방법을 제공한다. `torch.zeros`, `torch.randn`, `torch.arange` 같은 팩토리 함수는 `dtype`, `device`, `requires_grad` 매개변수를 받으므로 불필요한 복사 없이 목표 장치에 곧바로 할당할 수 있다.

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
