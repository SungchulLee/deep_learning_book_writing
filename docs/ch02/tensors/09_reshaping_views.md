# 재구성과 뷰

이 스크립트는 텐서의 재구성과 뷰을 보여준다. 이 개념들을 이해하는 것은 효과적인 PyTorch 프로그래밍과 딥러닝 모델 개발에 필수적이다.

## 코드

```python
"""
Tutorial 09: Reshaping and Views
=================================

Learn how to change tensor dimensions without copying data (when possible).
Understanding views vs copies is crucial for memory efficiency.

Key Concepts:
- reshape() vs view() vs contiguous()
- Adding/removing dimensions (unsqueeze/squeeze)
- Permuting dimensions (transpose/permute)
- Flattening tensors
- Memory layout and contiguity
"""

import torch

# ========================================================================
# 메인
# ========================================================================


def header(title: str):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def print_tensor_info(tensor, name="Tensor"):
    """Helper to display tensor properties."""
    print(f"{name}:")
    print(f"  Value: {tensor}")
    print(f"  Shape: {tensor.shape}")
    print(f"  Stride: {tensor.stride()}")
    print(f"  Contiguous: {tensor.is_contiguous()}")
    print()


def main():
    # -------------------------------------------------------------------------
    # 1. 기본 재구성 - reshape()와 view()
    # -------------------------------------------------------------------------
    header("1. Basic Reshaping - reshape() vs view()")
    
    # 1차원 텐서 만들기
    vec = torch.arange(12)
    print(f"Original 1D tensor: {vec}")
    print(f"Shape: {vec.shape}")  # torch.Size([12])
    
    # reshape() - 항상 동작하는 안전한 방법
    mat_reshape = vec.reshape(3, 4)
    print(f"\nReshape to (3, 4):\n{mat_reshape}")
    
    # view() - 더 빠르지만 연속된 메모리를 요구한다
    mat_view = vec.view(3, 4)
    print(f"\nView as (3, 4):\n{mat_view}")
    
    # 모양은 다르지만 전체 원소 수는 같다
    cube_reshape = vec.reshape(2, 2, 3)
    print(f"\nReshape to (2, 2, 3):\n{cube_reshape}")
    
    # 핵심 차이: view()는 비연속 텐서에서 실패한다
    # reshape()는 항상 동작한다(필요하면 복사한다)
    
    # -------------------------------------------------------------------------
    # 2. -1을 이용한 자동 크기 추론
    # -------------------------------------------------------------------------
    header("2. Automatic Size Inference with -1")
    
    # 한 차원에 -1을 쓴다 - PyTorch가 자동으로 추론한다
    vec_24 = torch.arange(24)
    
    # 행의 개수는 PyTorch가 계산하게 한다
    auto_rows = vec_24.reshape(-1, 4)  # -1 means "figure it out" → 6 rows
    print(f"reshape(-1, 4):\n{auto_rows}")
    print(f"Shape: {auto_rows.shape}")  # torch.Size([6, 4])
    
    # 열의 개수는 PyTorch가 계산하게 한다
    auto_cols = vec_24.reshape(3, -1)  # → 8 columns
    print(f"\nreshape(3, -1):\n{auto_cols}")
    print(f"Shape: {auto_cols.shape}")  # torch.Size([3, 8])
    
    # 재구성 한 번에 -1은 한 번만 쓸 수 있다
    # auto_both = vec_24.reshape(-1, -1)  # ❌ 오류: 한 차원만 추론할 수 있다
    
    # -------------------------------------------------------------------------
    # 3. Flatten - 1차원으로 바꾸기
    # -------------------------------------------------------------------------
    header("3. Flatten - Convert to 1D")
    
    mat_3d = torch.arange(24).reshape(2, 3, 4)
    print(f"3D tensor shape: {mat_3d.shape}")
    
    # flatten() - 지정한 차원을 펼친다
    flat_all = mat_3d.flatten()  # Flatten all dimensions
    print(f"flatten(): {flat_all}")
    print(f"Shape: {flat_all.shape}")  # torch.Size([24])
    
    # 특정 차원 펼치기
    flat_partial = mat_3d.flatten(start_dim=1)  # Keep dim 0, flatten rest
    print(f"\nflatten(start_dim=1) shape: {flat_partial.shape}")  # torch.Size([2, 12])
    print(f"Values:\n{flat_partial}")
    
    # 대안: -1로 재구성하기
    flat_reshape = mat_3d.reshape(-1)
    print(f"\nreshape(-1): {flat_reshape}")
    
    # 연속이면 view(-1)도 가능
    flat_view = mat_3d.view(-1)
    print(f"view(-1): {flat_view}")
    
    # -------------------------------------------------------------------------
    # 4. 차원 추가 - unsqueeze()
    # -------------------------------------------------------------------------
    header("4. Adding Dimensions - unsqueeze()")
    
    vec = torch.tensor([1, 2, 3, 4, 5])
    print(f"Original vector: {vec}")
    print(f"Shape: {vec.shape}")  # torch.Size([5])
    
    # 0 위치에 차원 추가(행 벡터/행렬이 된다)
    vec_row = vec.unsqueeze(0)
    print(f"\nunsqueeze(0) - Row vector:\n{vec_row}")
    print(f"Shape: {vec_row.shape}")  # torch.Size([1, 5])
    
    # 1 위치에 차원 추가(열 벡터/행렬이 된다)
    vec_col = vec.unsqueeze(1)
    print(f"\nunsqueeze(1) - Column vector:\n{vec_col}")
    print(f"Shape: {vec_col.shape}")  # torch.Size([5, 1])
    
    # -1 위치(끝)에 차원 추가
    vec_end = vec.unsqueeze(-1)
    print(f"\nunsqueeze(-1):\n{vec_end}")
    print(f"Shape: {vec_end.shape}")  # torch.Size([5, 1])
    
    # unsqueeze 여러 번
    vec_3d = vec.unsqueeze(0).unsqueeze(2)  # Shape: [1, 5, 1]
    print(f"\nDouble unsqueeze shape: {vec_3d.shape}")
    
    # 대안: None으로 인덱싱하기
    vec_row_alt = vec[None, :]  # Equivalent to unsqueeze(0)
    vec_col_alt = vec[:, None]  # Equivalent to unsqueeze(1)
    print(f"vec[None, :] shape: {vec_row_alt.shape}")  # torch.Size([1, 5])
    print(f"vec[:, None] shape: {vec_col_alt.shape}")  # torch.Size([5, 1])
    
    # -------------------------------------------------------------------------
    # 5. 차원 제거 - squeeze()
    # -------------------------------------------------------------------------
    header("5. Removing Dimensions - squeeze()")
    
    # 크기 1인 차원을 가진 텐서 만들기
    tensor_with_ones = torch.randn(1, 5, 1, 3, 1)
    print(f"Original shape: {tensor_with_ones.shape}")  # torch.Size([1, 5, 1, 3, 1])
    
    # squeeze() - 크기 1인 모든 차원을 제거한다
    squeezed_all = tensor_with_ones.squeeze()
    print(f"squeeze() shape: {squeezed_all.shape}")  # torch.Size([5, 3])
    
    # squeeze(dim) - 특정 차원을 제거한다(크기가 1일 때만)
    squeezed_dim0 = tensor_with_ones.squeeze(0)  # Remove first dim
    print(f"squeeze(0) shape: {squeezed_dim0.shape}")  # torch.Size([5, 1, 3, 1])
    
    squeezed_dim2 = tensor_with_ones.squeeze(2)  # Remove third dim
    print(f"squeeze(2) shape: {squeezed_dim2.shape}")  # torch.Size([1, 5, 3, 1])
    
    # 크기가 1이 아닌 차원을 squeeze하려 하면 아무 일도 일어나지 않는다
    squeezed_dim1 = tensor_with_ones.squeeze(1)  # Dim 1 is size 5
    print(f"squeeze(1) shape: {squeezed_dim1.shape}")  # torch.Size([1, 5, 1, 3, 1]) - unchanged
    
    # -------------------------------------------------------------------------
    # 6. 전치 - 차원 맞바꾸기
    # -------------------------------------------------------------------------
    header("6. Transpose - Swap Dimensions")
    
    mat = torch.arange(12).reshape(3, 4)
    print(f"Original matrix (3x4):\n{mat}")
    
    # transpose() - 두 차원을 맞바꾼다
    mat_T = mat.transpose(0, 1)  # Swap dimensions 0 and 1
    print(f"\ntranspose(0, 1) - Now (4x3):\n{mat_T}")
    
    # .T 속성 - 2차원 전치의 축약형
    mat_T_short = mat.T
    print(f"\nmat.T (same as transpose):\n{mat_T_short}")
    
    # 더 높은 차원에는 transpose나 permute를 쓴다
    tensor_3d = torch.arange(24).reshape(2, 3, 4)
    print(f"\n3D tensor shape: {tensor_3d.shape}")  # torch.Size([2, 3, 4])
    
    transposed_3d = tensor_3d.transpose(0, 2)  # Swap dims 0 and 2
    print(f"transpose(0, 2) shape: {transposed_3d.shape}")  # torch.Size([4, 3, 2])
    
    # -------------------------------------------------------------------------
    # 7. Permute - 여러 차원 재배열하기
    # -------------------------------------------------------------------------
    header("7. Permute - Rearrange Multiple Dimensions")
    
    # permute() - 모든 차원의 새 순서를 지정한다
    tensor_4d = torch.randn(2, 3, 4, 5)
    print(f"Original shape: {tensor_4d.shape}")  # torch.Size([2, 3, 4, 5])
    
    # (5, 3, 2, 4)로 재배열 - 차원: [3, 1, 0, 2]
    permuted = tensor_4d.permute(3, 1, 0, 2)
    print(f"permute(3, 1, 0, 2) shape: {permuted.shape}")  # torch.Size([5, 3, 2, 4])
    
    # 흔한 용례: NCHW에서 NHWC로 바꾸기(배치, 채널, 높이, 너비 → 배치, 높이, 너비, 채널)
    image_batch = torch.randn(32, 3, 224, 224)  # 32 images, 3 channels, 224x224
    print(f"\nImage batch (NCHW): {image_batch.shape}")
    
    image_batch_hwc = image_batch.permute(0, 2, 3, 1)  # Keep batch, move channels to end
    print(f"Image batch (NHWC): {image_batch_hwc.shape}")  # torch.Size([32, 224, 224, 3])
    
    # -------------------------------------------------------------------------
    # 8. 연속성 - 메모리 배치가 중요하다
    # -------------------------------------------------------------------------
    header("8. Contiguity - Memory Layout Matters")
    
    # 연속 텐서는 원소가 메모리에 순회 순서와 같은 순서로 놓여 있다
    vec_c = torch.arange(6)
    mat_c = vec_c.reshape(2, 3)
    print(f"Original (contiguous): {mat_c.is_contiguous()}")
    print_tensor_info(mat_c, "Contiguous matrix")
    
    # 전치는 비연속 뷰를 만든다
    mat_T = mat_c.T
    print(f"After transpose (non-contiguous): {mat_T.is_contiguous()}")
    print_tensor_info(mat_T, "Transposed matrix")
    
    # view()는 연속된 메모리를 요구한다
    try:
        # mat_T가 연속이 아니므로 이것은 실패한다
        mat_T.view(-1)
    except RuntimeError as e:
        print(f"Error with view() on non-contiguous: {e}\n")
    
    # 해결 1: contiguous()로 연속 복사본을 만든다
    mat_T_cont = mat_T.contiguous()
    print(f"After contiguous(): {mat_T_cont.is_contiguous()}")
    flat_T = mat_T_cont.view(-1)  # Now works!
    print(f"Flattened transposed matrix: {flat_T}")
    
    # 해결 2: 대신 reshape()를 쓴다(비연속을 자동으로 처리한다)
    flat_T_reshape = mat_T.reshape(-1)  # Works without contiguous()
    print(f"Using reshape() instead: {flat_T_reshape}")
    
    # 성능 참고: contiguous()는 복사본을 만들므로 시간과 메모리가 든다
    print(f"\nShared storage before contiguous? {mat_T.storage().data_ptr() == mat_c.storage().data_ptr()}")  # True
    print(f"Shared storage after contiguous? {mat_T_cont.storage().data_ptr() == mat_c.storage().data_ptr()}")  # False
    
    # -------------------------------------------------------------------------
    # 9. 흔한 재구성 패턴
    # -------------------------------------------------------------------------
    header("9. Common Reshaping Patterns")
    
    # 패턴 1: 벡터 배치를 행렬로
    batch_size, feature_dim = 64, 128
    batch_vectors = torch.randn(batch_size, feature_dim)
    print(f"Batch of vectors: {batch_vectors.shape}")  # torch.Size([64, 128])
    
    # 패턴 2: 완전연결 층을 위해 이미지 펼치기
    # 이미지: (배치, 채널, 높이, 너비)
    images = torch.randn(32, 3, 28, 28)
    flat_images = images.reshape(32, -1)  # (32, 3*28*28) = (32, 2352)
    print(f"Flattened images: {flat_images.shape}")
    
    # 패턴 3: 합성곱을 위한 재구성
    # 완전연결 출력 → 합성곱 입력
    fc_output = torch.randn(16, 512)  # 16 samples, 512 features
    conv_input = fc_output.reshape(16, 512, 1, 1)  # Add spatial dimensions
    print(f"Conv input shape: {conv_input.shape}")
    
    # 패턴 4: 텐서를 그룹으로 나누기
    big_tensor = torch.arange(60)
    groups = big_tensor.reshape(3, 20)  # 3 groups of 20 elements
    print(f"Grouped tensor shape: {groups.shape}")
    print(f"Groups:\n{groups}")
    
    # 패턴 5: 배치 차원 추가
    single_image = torch.randn(3, 224, 224)
    batch_of_one = single_image.unsqueeze(0)  # Add batch dim
    print(f"Single image: {single_image.shape}")
    print(f"As batch: {batch_of_one.shape}")
    
    # -------------------------------------------------------------------------
    # 10. 모범 사례
    # -------------------------------------------------------------------------
    header("10. Best Practices and Tips")
    
    print("""
    Key Takeaways:
    
    1. **reshape() vs view()**
       - Use reshape(): Safer, works always (copies if needed)
       - Use view(): Faster IF you know the tensor is contiguous
    
    2. **Contiguity**
       - Operations like transpose() create non-contiguous views
       - Call contiguous() before view() if unsure
       - reshape() handles this automatically
    
    3. **Memory Efficiency**
       - Reshaping operations are usually FREE (create views)
       - contiguous() creates a COPY (takes time and memory)
       - Only call contiguous() when necessary
    
    4. **Dimension Management**
       - unsqueeze() adds dimensions (useful for broadcasting)
       - squeeze() removes size-1 dimensions
       - Use -1 for automatic size inference in reshape()
    
    5. **Common Pitfalls**
       - Always check tensor shapes after reshaping
       - Be aware that views share memory with original
       - Remember: total elements must match in reshape
    """)
    
    # -------------------------------------------------------------------------
    # 연습 문제
    # -------------------------------------------------------------------------
    header("Practice Exercises")
    
    print("""
    Try these exercises:
    
    1. Create a tensor of shape (4, 5) and reshape it to (2, 2, 5)
    2. Take a (3, 224, 224) image and add a batch dimension at the start
    3. Flatten a (10, 5, 4) tensor into a (10, 20) tensor
    4. Convert a column vector (10, 1) to a row vector (1, 10)
    5. Create a (2, 3, 4, 5) tensor and rearrange to (5, 2, 4, 3)
    
    Solutions:
    """)
    
    # 해결 1
    t1 = torch.randn(4, 5)
    t1_reshaped = t1.reshape(2, 2, 5)
    print(f"1. Shape: {t1.shape} → {t1_reshaped.shape}")
    
    # 해결 2
    img = torch.randn(3, 224, 224)
    img_batch = img.unsqueeze(0)
    print(f"2. Shape: {img.shape} → {img_batch.shape}")
    
    # 해결 3
    t3 = torch.randn(10, 5, 4)
    t3_flat = t3.reshape(10, -1)
    print(f"3. Shape: {t3.shape} → {t3_flat.shape}")
    
    # 해결 4
    col = torch.randn(10, 1)
    row = col.reshape(1, 10)  # or col.T or col.squeeze().unsqueeze(0)
    print(f"4. Shape: {col.shape} → {row.shape}")
    
    # 해결 5
    t5 = torch.randn(2, 3, 4, 5)
    t5_perm = t5.permute(3, 0, 2, 1)
    print(f"5. Shape: {t5.shape} → {t5_perm.shape}")


if __name__ == "__main__":
    main()```

## 논의

브로드캐스팅은 작은 텐서를 가상으로 확장하여 모양이 다른 텐서 사이의 원소별 연산을 가능하게 한다. PyTorch는 차원을 오른쪽부터 맞추며, 각 차원 쌍이 서로 같거나, 둘 중 하나가 1이거나, 아예 없을 것을 요구한다. 이로써 데이터를 명시적으로 복제하지 않아도 되어 메모리 효율이 좋고 빠르다.

재구성 연산은 데이터를 반드시 복사하지 않으면서 텐서의 논리적 배치를 바꾼다. `view()` 메서드는 연속된 메모리를 요구하며 항상 뷰를 반환하고, `reshape()`는 (필요하면 복사하여) 어떤 텐서에서도 동작한다. `transpose()`나 `permute()` 같은 연산은 데이터 배치가 아니라 스트라이드를 바꾸므로 결과가 연속적이지 않을 수 있다.

PyTorch는 (저장소를 공유하는 뷰를 반환하는) 기본 슬라이싱과 (복사본을 반환하는) 불리언 마스크나 정수 배열을 이용한 고급 인덱싱을 모두 지원한다. 이 구분을 이해하는 것은 메모리 효율을 위해서도, 인덱싱한 결과를 수정할 때 의도치 않은 부작용을 피하기 위해서도 중요하다.

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
