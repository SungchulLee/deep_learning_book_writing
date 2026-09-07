# 텐서의 인덱싱과 슬라이싱

이 스크립트는 텐서의 인덱싱과 슬라이싱을 보여준다. 이 개념들을 이해하는 것은 효과적인 PyTorch 프로그래밍과 딥러닝 모델 개발에 필수적이다.

## 코드

```python
"""
학습 08: 텐서 자리 잡기와 자르기
==========================================

어떤 원소, 줄, 칸, 아래 텐서에 닿고 그것을 고치는 법을 배운다.
데이터를 다루고 신경망을 셈하는 데 꼭 필요하다.

고갱이 개념:
- 기본 자리 잡기(원소 하나)
- 자르기(원소의 범위)
- 앞선 자리 잡기(참거짓 가림, 멋진 자리 잡기)
- 여러 차원 자리 잡기
- 자리 잡기로 제자리에서 고치기
"""

import torch

# ========================================================================
# 메인
# ========================================================================


def print_section(title: str):
    """마디 머리글을 찍는 도우미."""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def main():
    # -------------------------------------------------------------------------
    # 준비: 시연용 예시 텐서 만들기
    # -------------------------------------------------------------------------
    print_section("Setup: Sample Tensors")
    
    # 1차원 텐서
    vec = torch.tensor([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    print("1D tensor (vec):", vec)
    
    # 2차원 텐서(3x4 행렬)
    mat = torch.arange(1, 13).reshape(3, 4)
    print("2D tensor (mat):\n", mat)
    
    # 3차원 텐서(2x3x4 - 모양 3x4인 행렬 2개로 생각하면 된다)
    tensor_3d = torch.arange(1, 25).reshape(2, 3, 4)
    print("3D tensor:\n", tensor_3d)
    
    # -------------------------------------------------------------------------
    # 1. 기본 인덱싱 - 단일 원소
    # -------------------------------------------------------------------------
    print_section("1. Basic Indexing - Single Elements")
    
    # 1차원 인덱싱(파이썬 방식, 0부터 시작)
    elem = vec[3]  # Fourth element
    print(f"vec[3] = {elem}")  # 40
    
    # 음수 인덱싱(끝에서부터)
    last = vec[-1]  # Last element
    second_last = vec[-2]  # Second to last
    print(f"vec[-1] = {last}, vec[-2] = {second_last}")  # 100, 90
    
    # 2차원 인덱싱 - [행, 열]
    elem_2d = mat[1, 2]  # Row 1, Column 2
    print(f"mat[1, 2] = {elem_2d}")  # 7
    
    # 3차원 인덱싱 - [깊이, 행, 열]
    elem_3d = tensor_3d[0, 1, 2]  # First matrix, row 1, column 2
    print(f"tensor_3d[0, 1, 2] = {elem_3d}")  # 7
    
    # 중요: 원소 하나를 인덱싱하면 0차원 텐서(스칼라)가 반환된다
    print(f"Type: {type(elem)}, Shape: {elem.shape}")  # Shape is torch.Size([])
    
    # 파이썬 스칼라를 얻으려면 .item()을 쓴다
    python_int = elem.item()
    print(f"Python int: {python_int}, Type: {type(python_int)}")
    
    # -------------------------------------------------------------------------
    # 2. 슬라이싱 - 부분 텐서 뽑아내기
    # -------------------------------------------------------------------------
    print_section("2. Slicing - Extracting Sub-tensors")
    
    # 문법: tensor[start:end:step]
    # - start: 포함(기본값 0)
    # - end: 제외(기본 길이)
    # - step: 보폭(기본값 1)
    
    # 기본 슬라이싱
    sub_vec = vec[2:5]  # Elements at indices 2, 3, 4
    print(f"vec[2:5] = {sub_vec}")  # tensor([30, 40, 50])
    
    # start 생략(0부터 시작한다)
    start_slice = vec[:4]  # First 4 elements
    print(f"vec[:4] = {start_slice}")  # tensor([10, 20, 30, 40])
    
    # end 생략(끝까지 간다)
    end_slice = vec[6:]  # From index 6 to end
    print(f"vec[6:] = {end_slice}")  # tensor([70, 80, 90, 100])
    
    # 보폭 사용(원소 건너뛰기)
    every_other = vec[::2]  # Every 2nd element
    print(f"vec[::2] = {every_other}")  # tensor([10, 30, 50, 70, 90])
    
    # 텐서 뒤집기
    reversed_vec = vec[::-1]
    print(f"vec[::-1] = {reversed_vec}")  # tensor([100, 90, 80, ..., 10])
    
    # -------------------------------------------------------------------------
    # 3. 다차원 슬라이싱
    # -------------------------------------------------------------------------
    print_section("3. Multi-dimensional Slicing")
    
    print("Original matrix (mat):\n", mat)
    # tensor([[ 1,  2,  3,  4],
    #         [ 5,  6,  7,  8],
    #         [ 9, 10, 11, 12]])
    
    # 행 전체 선택(1번 행)
    row_1 = mat[1, :]  # or simply mat[1]
    print(f"Row 1 (mat[1, :]): {row_1}")  # tensor([5, 6, 7, 8])
    
    # 열 전체 선택(2번 열)
    col_2 = mat[:, 2]
    print(f"Column 2 (mat[:, 2]): {col_2}")  # tensor([ 3,  7, 11])
    
    # 부분 행렬 선택(0-1행, 1-2열)
    sub_mat = mat[0:2, 1:3]
    print(f"Sub-matrix (mat[0:2, 1:3]):\n{sub_mat}")
    # tensor([[2, 3],
    #         [6, 7]])
    
    # 보폭을 두고 선택
    every_other_row = mat[::2, :]  # Rows 0, 2
    print(f"Every other row:\n{every_other_row}")
    
    # -------------------------------------------------------------------------
    # 4. 생략 부호(...) - 빠진 차원 채우기
    # -------------------------------------------------------------------------
    print_section("4. Ellipsis (...) - Shorthand for ':' across dimensions")
    
    # 생략 부호는 명시적으로 지정하지 않은 모든 차원을 나타낸다
    # 고차원 텐서에 유용하다
    
    # 3차원 텐서: 나머지 전체에 대해 마지막 차원의 첫 원소를 선택한다
    result = tensor_3d[..., 0]  # Equivalent to tensor_3d[:, :, 0]
    print(f"tensor_3d[..., 0] shape: {result.shape}")  # torch.Size([2, 3])
    print(f"tensor_3d[..., 0]:\n{result}")
    
    # 가운데 "행렬" 선택(depth=1)
    middle = tensor_3d[1, ...]  # Equivalent to tensor_3d[1, :, :]
    print(f"tensor_3d[1, ...] shape: {middle.shape}")  # torch.Size([3, 4])
    
    # -------------------------------------------------------------------------
    # 5. 불리언 인덱싱(마스킹)
    # -------------------------------------------------------------------------
    print_section("5. Boolean Indexing (Masking)")
    
    # 불리언 마스크 만들기
    mask = vec > 50  # Elements greater than 50
    print(f"Mask (vec > 50): {mask}")
    # tensor([False, False, False, False, False,  True,  True,  True,  True,  True])
    
    # 마스크로 걸러내기
    filtered = vec[mask]
    print(f"vec[mask] (elements > 50): {filtered}")  # tensor([ 60,  70,  80,  90, 100])
    
    # &(AND)와 |(OR)로 여러 조건 결합
    # 참고: 'and'/'or'가 아니라 &와 |를 쓴다(전자는 원소별로 동작하지 않는다)
    mask_complex = (vec > 30) & (vec < 80)
    print(f"vec[(vec > 30) & (vec < 80)]: {vec[mask_complex]}")  # tensor([40, 50, 60, 70])
    
    # 2차원 텐서에 대한 불리언 인덱싱
    mask_2d = mat > 6
    print(f"Elements > 6 in mat: {mat[mask_2d]}")  # Returns 1D tensor of matching elements
    
    # -------------------------------------------------------------------------
    # 6. 고급 인덱싱 - 인덱스 텐서
    # -------------------------------------------------------------------------
    print_section("6. Advanced Indexing - Index Tensors")
    
    # 인덱스 텐서로 원소를 선택한다
    indices = torch.tensor([0, 2, 4])
    selected = vec[indices]
    print(f"vec[[0, 2, 4]]: {selected}")  # tensor([10, 30, 50])
    
    # 2차원 텐서에 대한 팬시 인덱싱
    row_indices = torch.tensor([0, 1, 2])
    col_indices = torch.tensor([1, 2, 3])
    # mat[0,1], mat[1,2], mat[2,3] 선택
    diagonal_like = mat[row_indices, col_indices]
    print(f"mat[row_indices, col_indices]: {diagonal_like}")  # tensor([ 2,  7, 12])
    
    # -------------------------------------------------------------------------
    # 7. 인덱싱을 통한 제자리 수정
    # -------------------------------------------------------------------------
    print_section("7. In-place Modification via Indexing")
    
    # 수정할 복사본 만들기
    vec_copy = vec.clone()
    print(f"Original: {vec_copy}")
    
    # 원소 하나 수정
    vec_copy[3] = 999
    print(f"After vec_copy[3] = 999: {vec_copy}")
    
    # 슬라이스 수정
    vec_copy[5:8] = 0
    print(f"After vec_copy[5:8] = 0: {vec_copy}")
    
    # 불리언 마스크로 수정
    vec_copy[vec_copy < 40] = -1
    print(f"After setting elements < 40 to -1: {vec_copy}")
    
    # 2차원 수정
    mat_copy = mat.clone()
    mat_copy[0, :] = 0  # Set first row to zeros
    mat_copy[:, -1] = 99  # Set last column to 99
    print(f"Modified matrix:\n{mat_copy}")
    
    # -------------------------------------------------------------------------
    # 8. 뷰와 복사본 - 중요한 메모리 고려사항
    # -------------------------------------------------------------------------
    print_section("8. View vs Copy - Memory Behavior")
    
    # 슬라이싱은 뷰(VIEW)를 만든다(원본과 메모리를 공유한다)
    original = torch.tensor([1, 2, 3, 4, 5])
    view = original[1:4]
    
    print(f"Original: {original}")
    print(f"View: {view}")
    
    # 뷰를 수정하면 원본이 바뀐다!
    view[0] = 999
    print(f"After view[0] = 999:")
    print(f"Original: {original}")  # Changed!
    print(f"View: {view}")
    
    # 이를 피하려면 .clone()을 쓴다
    original2 = torch.tensor([1, 2, 3, 4, 5])
    true_copy = original2[1:4].clone()
    true_copy[0] = 999
    print(f"\nWith .clone():")
    print(f"Original: {original2}")  # Unchanged
    print(f"Copy: {true_copy}")
    
    # 텐서들이 저장소를 공유하는지 확인
    print(f"\nShared storage? {original.data_ptr() == view.data_ptr()}")  # False (different data pointer due to offset)
    print(f"Same underlying storage? {original.storage().data_ptr() == view.storage().data_ptr()}")  # True!
    
    # -------------------------------------------------------------------------
    # 9. 흔한 패턴과 용례
    # -------------------------------------------------------------------------
    print_section("9. Common Patterns and Use Cases")
    
    # 패턴 1: 행렬의 대각 성분 얻기
    diag = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    diagonal = torch.diagonal(diag)
    print(f"Diagonal: {diagonal}")  # tensor([1, 5, 9])
    
    # 패턴 2: 특정 행 선택
    data = torch.randn(100, 10)  # 100 samples, 10 features
    batch_indices = torch.tensor([0, 5, 10, 15])
    batch = data[batch_indices]
    print(f"Selected batch shape: {batch.shape}")  # torch.Size([4, 10])
    
    # 패턴 3: 원소 제거(나머지를 선택하는 방식으로)
    vec_to_filter = torch.tensor([1, 2, 3, 4, 5, 6])
    keep_mask = torch.tensor([True, False, True, True, False, True])
    filtered_result = vec_to_filter[keep_mask]
    print(f"After filtering: {filtered_result}")  # tensor([1, 3, 4, 6])
    
    # 패턴 4: 조건에 따라 값 바꾸기
    data_with_outliers = torch.tensor([1.0, 2.0, 100.0, 3.0, -50.0, 4.0])
    data_clipped = data_with_outliers.clone()
    data_clipped[data_clipped > 10] = 10.0
    data_clipped[data_clipped < 0] = 0.0
    print(f"Clipped data: {data_clipped}")  # tensor([1., 2., 10., 3., 0., 4.])
    
    # -------------------------------------------------------------------------
    # 연습 문제
    # -------------------------------------------------------------------------
    print_section("Practice Exercises")
    
    print("""
    이해했는지 다음 학습으로 따져 보아라.
    
    1. 5x5 행렬을 만들고 네 모서리(원소 4개: [0,0], [0,4], [4,0], [4,4])를 뽑아라
    2. 원소 20개짜리 1차원 텐서에서 번호 1부터 세 칸마다 하나씩 골라라
    3. 4x6 행렬을 만들고 둘째 줄과 셋째 칸의 원소를 모두 0으로 두어라
    4. 참거짓 자리 잡기로 텐서에서 5와 15 사이의 원소를 모두 찾아라
    5. 3x3 행렬을 만들고 자리 잡기로 첫 줄과 마지막 줄을 맞바꾸어라
    
    풀이는 아래에 있다...
    """)
    
    # 해결 1
    mat_5x5 = torch.arange(25).reshape(5, 5)
    corners_indices = torch.tensor([[0, 0], [0, 4], [4, 0], [4, 4]])
    corners = mat_5x5[corners_indices[:, 0], corners_indices[:, 1]]
    print(f"Exercise 1 - Corners: {corners}")
    
    # 해결 2
    vec_20 = torch.arange(20)
    every_third = vec_20[1::3]
    print(f"Exercise 2 - Every 3rd from index 1: {every_third}")
    
    # 해결 3
    mat_4x6 = torch.ones(4, 6)
    mat_4x6[1, :] = 0  # 2nd row
    mat_4x6[:, 2] = 0  # 3rd column
    print(f"Exercise 3 - Modified matrix:\n{mat_4x6}")
    
    # 해결 4
    test_vec = torch.arange(20)
    between = test_vec[(test_vec >= 5) & (test_vec <= 15)]
    print(f"Exercise 4 - Elements between 5 and 15: {between}")
    
    # 해결 5
    mat_3x3 = torch.arange(9).reshape(3, 3)
    mat_3x3[[0, 2]] = mat_3x3[[2, 0]]  # Swap rows 0 and 2
    print(f"Exercise 5 - After swapping rows:\n{mat_3x3}")


if __name__ == "__main__":
    main()```

## 논의

재구성 연산은 데이터를 반드시 복사하지 않으면서 텐서의 논리적 배치를 바꾼다. `view()` 메서드는 연속된 메모리를 요구하며 항상 뷰를 반환하고, `reshape()`는 (필요하면 복사하여) 어떤 텐서에서도 동작한다. `transpose()`나 `permute()` 같은 연산은 데이터 배치가 아니라 스트라이드를 바꾸므로 결과가 연속적이지 않을 수 있다.

PyTorch는 (저장소를 공유하는 뷰를 반환하는) 기본 슬라이싱과 (복사본을 반환하는) 불리언 마스크나 정수 배열을 이용한 고급 인덱싱을 모두 지원한다. 이 구분을 이해하는 것은 메모리 효율을 위해서도, 인덱싱한 결과를 수정할 때 의도치 않은 부작용을 피하기 위해서도 중요하다.

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
