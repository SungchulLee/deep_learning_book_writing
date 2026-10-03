# 재구성과 뷰

모양을 바꾸는 일은 대개 숫자를 옮기지 않는다. 저장소는 그대로 두고 "몇 칸씩 건너뛰며 읽을까"만 고치면 되기 때문이다. 그런데 그렇게 할 수 **없는** 경우가 있고, 그때 `view`와 `reshape`가 서로 다르게 행동한다. 이 쪽은 그 갈림길을 본다.

## 1. 코드

```python
"""
튜토리얼 09: 꼴 바꾸기와 보기
=================================

될 수 있으면 데이터를 베끼지 않고 텐서의 차원을 바꾸는 법을 배운다.
보기와 베낌의 다름을 아는 것이 기억 자리를 아끼는 데 꼭 필요하다.

핵심 개념:
- reshape(), view(), contiguous() 견주기
- 차원 더하기와 없애기(unsqueeze/squeeze)
- 차원 자리바꿈(transpose/permute)
- 텐서 펼치기
- 기억 자리 구조와 이어짐
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


def print_tensor_info(tensor, name="Tensor"):
    """텐서의 속성을 보여 주는 도우미."""
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

    # movedim() - 옮길 차원 하나만 말한다. 나머지는 순서를 지킨 채 밀려난다.
    # permute는 **모든** 차원의 새 순서를 적어야 하므로, 차원이 넷만 되어도
    # 건드리지 않을 셋까지 써야 한다. 바로 위의 NCHW→NHWC가 그 예다.
    moved = torch.movedim(image_batch, 1, 3)   # 채널(1번)을 끝으로 옮긴다
    print(f"\nmovedim(1, 3): {moved.shape}")   # permute(0, 2, 3, 1)과 같다
    print(f"permute와 같은가: {torch.equal(moved, image_batch_hwc)}")
    # 여러 개를 한꺼번에 옮길 수도 있다
    moved2 = torch.movedim(tensor_4d, [0, 1], [2, 3])
    print(f"movedim([0,1], [2,3]): {moved2.shape}")

    # swapdims() - transpose의 다른 이름이다. 뜻이 더 분명해서 읽기에 낫다.
    swapped = torch.swapdims(tensor_4d, 0, 2)
    print(f"\nswapdims(0, 2): {swapped.shape}")
    print(f"transpose와 같은가: {torch.equal(swapped, tensor_4d.transpose(0, 2))}")

    # .t() - 2차원 전용 줄임말이다. 계수가 3 이상이면 오류를 낸다.
    print(f"\nmat.t() shape: {mat.t().shape}")
    try:
        tensor_4d.t()
    except RuntimeError as e:
        print("4차원에 .t():", str(e)[:60])
    
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
    핵심 학습:
    
    1. **reshape()과 view() 견주기**
       - reshape()을 써라: 더 안전하고 늘 된다(필요하면 베낀다)
       - view()를 써라: 텐서가 이어져 있음을 안다면 더 빠르다
    
    2. **Contiguity**
       - transpose() 같은 셈은 이어지지 않은 보기를 만든다
       - 확실하지 않으면 view() 앞에 contiguous()을 불러라
       - reshape()은 이를 절로 다룬다
    
    3. **기억 자리 아끼기**
       - 꼴 바꾸기는 대개 공짜다(보기를 만든다)
       - contiguous()은 베낌을 만든다(때와 기억 자리가 든다)
       - 꼭 필요할 때만 contiguous()을 불러라
    
    4. **차원 다루기**
       - unsqueeze()은 차원을 더한다(펴 맞추기에 쓸모 있다)
       - squeeze()은 크기 1인 차원을 없앤다
       - reshape()에서 크기를 절로 미루게 하려면 -1을 써라
    
    5. **흔한 함정**
       - 모양을 바꾼 뒤에는 늘 텐서 모양을 살펴라
       - 보기는 본디 텐서와 기억 자리를 나눠 쓴다는 것을 잊지 마라
       - 기억하라: 모양을 바꿀 때 온 원소 수가 맞아야 한다
    """)
    
    # -------------------------------------------------------------------------
    # 연습 문제
    # -------------------------------------------------------------------------
    header("Practice Exercises")
    
    print("""
    다음 학습을 해 보아라.
    
    1. 모양이 (4, 5)인 텐서를 만들고 (2, 2, 5)으로 바꾸어라
    2. (3, 224, 224) 그림에 앞쪽으로 배치 차원을 더하여라
    3. (10, 5, 4) 텐서를 (10, 20) 텐서로 펼쳐라
    4. 칸 벡터 (10, 1)을 줄 벡터 (1, 10)으로 바꾸어라
    5. (2, 3, 4, 5) 텐서를 만들어 (5, 2, 4, 3)으로 자리를 바꾸어라
    
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
    main()
```

??? note "전체 출력 (237줄)"

    ```

    ======================================================================
    1. Basic Reshaping - reshape() vs view()
    ======================================================================
    Original 1D tensor: tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11])
    Shape: torch.Size([12])

    Reshape to (3, 4):
    tensor([[ 0,  1,  2,  3],
            [ 4,  5,  6,  7],
            [ 8,  9, 10, 11]])

    View as (3, 4):
    tensor([[ 0,  1,  2,  3],
            [ 4,  5,  6,  7],
            [ 8,  9, 10, 11]])

    Reshape to (2, 2, 3):
    tensor([[[ 0,  1,  2],
             [ 3,  4,  5]],

            [[ 6,  7,  8],
             [ 9, 10, 11]]])

    ======================================================================
    2. Automatic Size Inference with -1
    ======================================================================
    reshape(-1, 4):
    tensor([[ 0,  1,  2,  3],
            [ 4,  5,  6,  7],
            [ 8,  9, 10, 11],
            [12, 13, 14, 15],
            [16, 17, 18, 19],
            [20, 21, 22, 23]])
    Shape: torch.Size([6, 4])

    reshape(3, -1):
    tensor([[ 0,  1,  2,  3,  4,  5,  6,  7],
            [ 8,  9, 10, 11, 12, 13, 14, 15],
            [16, 17, 18, 19, 20, 21, 22, 23]])
    Shape: torch.Size([3, 8])

    ======================================================================
    3. Flatten - Convert to 1D
    ======================================================================
    3D tensor shape: torch.Size([2, 3, 4])
    flatten(): tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,
            18, 19, 20, 21, 22, 23])
    Shape: torch.Size([24])

    flatten(start_dim=1) shape: torch.Size([2, 12])
    Values:
    tensor([[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11],
            [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]])

    reshape(-1): tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,
            18, 19, 20, 21, 22, 23])
    view(-1): tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,
            18, 19, 20, 21, 22, 23])

    ======================================================================
    4. Adding Dimensions - unsqueeze()
    ======================================================================
    Original vector: tensor([1, 2, 3, 4, 5])
    Shape: torch.Size([5])

    unsqueeze(0) - Row vector:
    tensor([[1, 2, 3, 4, 5]])
    Shape: torch.Size([1, 5])

    unsqueeze(1) - Column vector:
    tensor([[1],
            [2],
            [3],
            [4],
            [5]])
    Shape: torch.Size([5, 1])

    unsqueeze(-1):
    tensor([[1],
            [2],
            [3],
            [4],
            [5]])
    Shape: torch.Size([5, 1])

    Double unsqueeze shape: torch.Size([1, 5, 1])
    vec[None, :] shape: torch.Size([1, 5])
    vec[:, None] shape: torch.Size([5, 1])

    ======================================================================
    5. Removing Dimensions - squeeze()
    ======================================================================
    Original shape: torch.Size([1, 5, 1, 3, 1])
    squeeze() shape: torch.Size([5, 3])
    squeeze(0) shape: torch.Size([5, 1, 3, 1])
    squeeze(2) shape: torch.Size([1, 5, 3, 1])
    squeeze(1) shape: torch.Size([1, 5, 1, 3, 1])

    ======================================================================
    6. Transpose - Swap Dimensions
    ======================================================================
    Original matrix (3x4):
    tensor([[ 0,  1,  2,  3],
            [ 4,  5,  6,  7],
            [ 8,  9, 10, 11]])

    transpose(0, 1) - Now (4x3):
    tensor([[ 0,  4,  8],
            [ 1,  5,  9],
            [ 2,  6, 10],
            [ 3,  7, 11]])

    mat.T (same as transpose):
    tensor([[ 0,  4,  8],
            [ 1,  5,  9],
            [ 2,  6, 10],
            [ 3,  7, 11]])

    3D tensor shape: torch.Size([2, 3, 4])
    transpose(0, 2) shape: torch.Size([4, 3, 2])

    ======================================================================
    7. Permute - Rearrange Multiple Dimensions
    ======================================================================
    Original shape: torch.Size([2, 3, 4, 5])
    permute(3, 1, 0, 2) shape: torch.Size([5, 3, 2, 4])

    Image batch (NCHW): torch.Size([32, 3, 224, 224])
    Image batch (NHWC): torch.Size([32, 224, 224, 3])

    movedim(1, 3): torch.Size([32, 224, 224, 3])
    permute와 같은가: True
    movedim([0,1], [2,3]): torch.Size([4, 5, 2, 3])

    swapdims(0, 2): torch.Size([4, 3, 2, 5])
    transpose와 같은가: True

    mat.t() shape: torch.Size([4, 3])
    4차원에 .t(): t() expects a tensor with <= 2 dimensions, but self is 4D

    ======================================================================
    8. Contiguity - Memory Layout Matters
    ======================================================================
    Original (contiguous): True
    Contiguous matrix:
      Value: tensor([[0, 1, 2],
            [3, 4, 5]])
      Shape: torch.Size([2, 3])
      Stride: (3, 1)
      Contiguous: True

    After transpose (non-contiguous): False
    Transposed matrix:
      Value: tensor([[0, 3],
            [1, 4],
            [2, 5]])
      Shape: torch.Size([3, 2])
      Stride: (1, 3)
      Contiguous: False

    Error with view() on non-contiguous: view size is not compatible with input tensor's size and stride (at least one dimension spans across two contiguous subspaces). Use .reshape(...) instead.

    After contiguous(): True
    Flattened transposed matrix: tensor([0, 3, 1, 4, 2, 5])
    Using reshape() instead: tensor([0, 3, 1, 4, 2, 5])

    Shared storage before contiguous? True
    Shared storage after contiguous? False

    ======================================================================
    9. Common Reshaping Patterns
    ======================================================================
    Batch of vectors: torch.Size([64, 128])
    Flattened images: torch.Size([32, 2352])
    Conv input shape: torch.Size([16, 512, 1, 1])
    Grouped tensor shape: torch.Size([3, 20])
    Groups:
    tensor([[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,
             18, 19],
            [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37,
             38, 39],
            [40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
             58, 59]])
    Single image: torch.Size([3, 224, 224])
    As batch: torch.Size([1, 3, 224, 224])

    ======================================================================
    10. Best Practices and Tips
    ======================================================================

        핵심 학습:
        
        1. **reshape()과 view() 견주기**
           - reshape()을 써라: 더 안전하고 늘 된다(필요하면 베낀다)
           - view()를 써라: 텐서가 이어져 있음을 안다면 더 빠르다
        
        2. **Contiguity**
           - transpose() 같은 셈은 이어지지 않은 보기를 만든다
           - 확실하지 않으면 view() 앞에 contiguous()을 불러라
           - reshape()은 이를 절로 다룬다
        
        3. **기억 자리 아끼기**
           - 꼴 바꾸기는 대개 공짜다(보기를 만든다)
           - contiguous()은 베낌을 만든다(때와 기억 자리가 든다)
           - 꼭 필요할 때만 contiguous()을 불러라
        
        4. **차원 다루기**
           - unsqueeze()은 차원을 더한다(펴 맞추기에 쓸모 있다)
           - squeeze()은 크기 1인 차원을 없앤다
           - reshape()에서 크기를 절로 미루게 하려면 -1을 써라
        
        5. **흔한 함정**
           - 모양을 바꾼 뒤에는 늘 텐서 모양을 살펴라
           - 보기는 본디 텐서와 기억 자리를 나눠 쓴다는 것을 잊지 마라
           - 기억하라: 모양을 바꿀 때 온 원소 수가 맞아야 한다
        

    ======================================================================
    Practice Exercises
    ======================================================================

        다음 학습을 해 보아라.
        
        1. 모양이 (4, 5)인 텐서를 만들고 (2, 2, 5)으로 바꾸어라
        2. (3, 224, 224) 그림에 앞쪽으로 배치 차원을 더하여라
        3. (10, 5, 4) 텐서를 (10, 20) 텐서로 펼쳐라
        4. 칸 벡터 (10, 1)을 줄 벡터 (1, 10)으로 바꾸어라
        5. (2, 3, 4, 5) 텐서를 만들어 (5, 2, 4, 3)으로 자리를 바꾸어라
        
        Solutions:
        
    1. Shape: torch.Size([4, 5]) → torch.Size([2, 2, 5])
    2. Shape: torch.Size([3, 224, 224]) → torch.Size([1, 3, 224, 224])
    3. Shape: torch.Size([10, 5, 4]) → torch.Size([10, 20])
    4. Shape: torch.Size([10, 1]) → torch.Size([1, 10])
    5. Shape: torch.Size([2, 3, 4, 5]) → torch.Size([5, 2, 4, 3])
    ```
## 2. 논의

**모양을 바꾸는 일은 스트라이드를 고치는 일이다.** 저장소에 숫자가 한 줄로 놓여 있고, 모양과 스트라이드가 "그것을 어떻게 읽을지"를 말해 준다. 그래서 `(3, 4)`를 `(4, 3)`으로 바꾸는 데는 숫자를 옮길 필요가 없다 — 읽는 규칙만 바꾸면 된다.

할 수 없는 때가 있다. `t()`나 `permute()`를 거친 텐서는 스트라이드가 메모리 순서와 어긋나 있어서(**비연속**), 원하는 모양을 어떤 스트라이드로도 나타낼 수 없는 일이 생긴다. 여기서 둘이 갈린다.

| | 약속하는 것 | 할 수 없으면 |
|---|---|---|
| `view(…)` | **저장소를 함께 쓴다** | **거절한다** — `RuntimeError` |
| `reshape(…)` | **결과를 준다** | 조용히 베껴서 준다 |

**`view`가 거절하는 것은 모자람이 아니라 알려 주는 일이다.** "네가 생각한 대로 메모리가 놓여 있지 않다"는 뜻이기 때문이다. `reshape`는 같은 자리에서 베껴서 넘어가므로, 뷰를 받았다고 믿고 제자리 연산을 하면 원본이 바뀌지 않는다. 그래서 **공유가 중요하면 `view`를, 결과만 필요하면 `reshape`를** 쓴다.

`contiguous()`는 비연속 텐서를 메모리 순서에 맞게 다시 깔아 준다. 이미 연속이면 아무 일도 하지 않고 그대로 돌려주므로, `t.contiguous().view(…)`는 "필요할 때만 베끼고 나머지는 공유"라는 뜻이 된다.

**`-1`은 한 자리에만 쓴다.** 나머지 길이로부터 나눗셈 한 번으로 정해지기 때문이다. 두 자리에 쓰면 `only one dimension can be inferred`다.

!!! warning "인수 없는 `squeeze()`는 배치 차원도 지운다"
    `squeeze()`는 크기가 1인 차원을 **모두** 지운다. 그래서 배치 크기가 1일 때 배치 차원까지 함께 사라진다.

    ```python
    torch.randn(4, 3, 1, 1).squeeze().shape   # (4, 3) — 배치가 남는다
    torch.randn(1, 3, 1, 1).squeeze().shape   # (3,)   — 배치가 사라졌다
    ```

    배치 4로 짠 코드가 돌다가 마지막 배치 하나에서 모양이 어긋나는 흔한 버그가 이것이다. 지울 차원을 **밝혀서** 적으면 생기지 않는다.

    ```python
    x.squeeze(-1).squeeze(-1)   # 마지막 둘만 지운다
    x.flatten(1)                # 0번은 그대로, 뒤를 펼친다
    ```

    `squeeze(dim)`에 크기가 1이 아닌 차원을 주면 오류가 아니라 **아무 일도 하지 않는다.** 모양이 그대로이므로 지워졌다고 믿고 지나가기 쉽다.

**차원을 옮기는 함수가 넷이다.** 하는 일이 겹치지만 적는 품이 다르다.

| | 적는 법 | 쓰기 좋은 때 |
|---|---|---|
| `t()` | 2차원 전용 | 행렬 하나를 뒤집을 때. 계수가 3 이상이면 오류다 |
| `transpose(i, j)` / `swapdims(i, j)` | 둘을 맞바꾼다 | 두 축만 바꿀 때. `swapdims`가 이름이 더 분명하다 |
| `permute(…)` | **모든** 차원의 새 순서 | 전체를 재배열할 때 |
| `movedim(src, dst)` | 옮길 것만 말한다 | 한둘만 옮길 때 — 나머지는 순서를 지킨 채 밀려난다 |

NCHW를 NHWC로 바꾸는 일이 둘의 차이를 잘 보여 준다. `permute(0, 2, 3, 1)`은 건드리지 않을 차원까지 넷을 다 적어야 하는데, `movedim(1, 3)`은 "채널을 끝으로"라고만 적는다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
`a = torch.arange(12.).reshape(3, 4)`와 그 전치 `t = a.t()`에 대해, `a.reshape(4, 3)`과 `t.reshape(12)`가 각각 `a`와 저장소를 함께 쓰는지 확인하라. 같은 `reshape`인데 답이 다른 까닭을 밝혀라.

</div>

??? success "연습문제 1 풀이"
    ```python
    import torch

    def shares(x, y):
        return x.untyped_storage().data_ptr() == y.untyped_storage().data_ptr()

    a = torch.arange(12.).reshape(3, 4)
    t = a.t()
    print("t가 이어져 있는가:", t.is_contiguous())
    print("a.reshape(4, 3)  :", shares(a, a.reshape(4, 3)))
    print("t.reshape(12)    :", shares(a, t.reshape(12)))
    ```

    ```
    t가 이어져 있는가: False
    a.reshape(4, 3)  : True
    t.reshape(12)    : False
    ```

    `a`는 연속이므로 `(4, 3)`으로 읽는 스트라이드가 존재한다. 숫자를 옮길 필요 없이
    읽는 규칙만 바꾸면 되므로 **공유한다.**

    `t`는 전치라 스트라이드가 메모리 순서와 어긋나 있다. 그 상태로 `(12,)`로 한 줄로
    읽을 방법이 없으므로 `reshape`는 **조용히 베낀다.**

    곧 `reshape`가 무엇을 돌려주는지는 받는 텐서의 메모리 상태에 달려 있다. 같은
    함수 호출이 어떤 때는 뷰, 어떤 때는 사본이다.

---


<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
위의 `t = a.t()`에 `t.view(12)`를 해 보라. 오류 메시지를 적고, `reshape`는 되는데 `view`는 안 되는 까닭을 설명하라. `view`의 이 거절이 왜 **쓸모 있는 성질**인지 밝히고, `contiguous()`로 넘기는 법을 보여라.

</div>

??? success "연습문제 2 풀이"
    ```python
    import torch

    a = torch.arange(12.).reshape(3, 4)
    t = a.t()
    try:
        t.view(12)
    except RuntimeError as e:
        print(str(e)[:84])

    print("contiguous 뒤:", t.contiguous().view(12).shape)
    ```

    ```
    view size is not compatible with input tensor's size and stride (at least one dime
    contiguous 뒤: torch.Size([12])
    ```

    `view`는 **저장소를 함께 쓴다**는 것을 약속한다. `t`를 한 줄로 읽는 스트라이드가
    없으므로 약속을 지킬 수 없고, 그래서 거절한다. `reshape`는 결과를 주겠다고만
    약속하므로 베껴서 지킨다.

    거절이 쓸모 있는 까닭은 **잘못된 믿음을 그 자리에서 깨 주기** 때문이다. 뷰를
    받았다고 믿고 제자리 연산을 하면 원본이 바뀌어야 하는데, `reshape`가 몰래 베낀
    경우에는 바뀌지 않는다. 그 어긋남은 한참 뒤에 드러난다. `view`를 쓰면 그 자리에서
    멈춘다.

    `contiguous()`는 메모리를 다시 깔아 준다. 이미 연속이면 그대로 돌려주므로,
    `t.contiguous().view(…)`는 "필요할 때만 베낀다"는 뜻이 된다.

---


<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
합성곱의 출력처럼 모양이 `(N, 3, 1, 1)`인 텐서를 `(N, 3)`으로 줄이려 한다. `x.squeeze()`로 적고 `N = 4`와 `N = 1`에서 각각 모양을 확인하라. 왜 이것이 **배치 크기에 따라 달라지는 버그**가 되는지 설명하고, 안전하게 적는 두 가지 방법을 보여라.

</div>

??? success "연습문제 3 풀이"
    ```python
    import torch

    for n in (4, 1):
        x = torch.randn(n, 3, 1, 1)
        print(f"N={n}: squeeze() -> {tuple(x.squeeze().shape)}")
    ```

    ```
    N=4: squeeze() -> (4, 3)
    N=1: squeeze() -> (3,)
    ```

    인수 없는 `squeeze()`는 크기가 1인 차원을 **모두** 지운다. `N = 4`에서는 지울 것이
    뒤의 두 개뿐이라 `(4, 3)`이 나와 뜻한 대로다. 그런데 `N = 1`에서는 배치 차원도
    크기가 1이므로 **함께 사라져** `(3,)`이 된다.

    이것이 고약한 까닭은 **거의 언제나 잘 돌아가기** 때문이다. 배치 크기가 32면 멀쩡히
    돌다가, 자료 수가 배치로 나누어떨어지지 않아 마지막 배치가 하나만 남는 순간
    모양이 어긋난다. 학습은 몇 시간 뒤 마지막 배치에서 깨지고, 원인은 이 한 줄에 있다.

    안전하게 적는 길은 **지울 차원을 밝히는 것**이다.

    ```python
    for n in (4, 1):
        x = torch.randn(n, 3, 1, 1)
        print(f"N={n}: squeeze(-1).squeeze(-1) -> {tuple(x.squeeze(-1).squeeze(-1).shape)}"
              f"   flatten(1) -> {tuple(x.flatten(1).shape)}")
    ```

    ```
    N=4: squeeze(-1).squeeze(-1) -> (4, 3)   flatten(1) -> (4, 3)
    N=1: squeeze(-1).squeeze(-1) -> (1, 3)   flatten(1) -> (1, 3)
    ```

    둘 다 배치 차원을 건드리지 않는다. `flatten(1)`은 "0번은 그대로 두고 뒤를 전부
    펼친다"는 뜻이라 뒤의 차원이 `1`이 아니어도 통하므로 더 넓게 쓰인다.

    덧붙여, `squeeze(dim)`에 크기가 1이 **아닌** 차원을 주면 오류가 아니라 아무 일도
    하지 않는다. 모양이 그대로이므로 지워진 줄 알고 넘어가기 쉽다.

## 정리하며

모양을 바꾸는 일은 숫자를 옮기는 일이 아니라 **스트라이드를 고치는 일**이다. 그렇게 할 수 없을 때 무엇을 하느냐가 함수마다 다르다.

- **`view`는 공유를 약속하고, 못 지키면 거절한다.** 그 거절이 "메모리가 네 생각과 다르게 놓여 있다"고 알려 준다.
- **`reshape`는 결과를 약속하고, 못 지키면 베낀다.** 그래서 뷰를 받았는지 사본을 받았는지 호출만 보고는 알 수 없다. 받는 텐서가 연속인지에 달려 있다.
- `contiguous()`는 필요할 때만 다시 깔아 준다. 이미 연속이면 그대로 돌려준다.

공유가 중요하면 `view`, 결과만 필요하면 `reshape`를 쓴다.

차원을 옮기는 함수는 적는 품으로 고른다. `t()`는 2차원 전용, `transpose`/`swapdims`는 둘을 맞바꾸고, `permute`는 전부 적고, `movedim`은 옮길 것만 적는다.

마지막으로 **인수 없는 `squeeze()`를 쓰지 않는다.** 크기가 1인 차원을 모두 지우므로 배치가 1인 날 배치 차원까지 사라진다. 거의 언제나 잘 돌아가다가 마지막 배치에서 깨지는 버그가 되므로, `squeeze(-1)`이나 `flatten(1)`처럼 **어디를 지울지 밝혀서** 적는다.
