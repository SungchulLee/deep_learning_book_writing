# 텐서의 인덱싱과 슬라이싱

색인하는 길이 두 갈래다. `a[1:3]`처럼 **자리를 범위로 말하는 것**과 `a[a>0]`처럼 **어느 원소인지 골라 말하는 것**이다. 둘은 겉모양이 비슷한데 하나는 뷰를 주고 하나는 베낀 것을 준다. 게다가 등호의 왼쪽에 놓이면 규칙이 또 달라진다.

## 1. 코드

```python
"""
튜토리얼 08: 텐서 자리 잡기와 자르기
==========================================

어떤 원소, 줄, 칸, 아래 텐서에 닿고 그것을 고치는 법을 배운다.
데이터를 다루고 신경망을 셈하는 데 꼭 필요하다.

핵심 개념:
- 기본 자리 잡기(원소 하나)
- 자르기(원소의 범위)
- 앞선 자리 잡기(참거짓 가림, 멋진 자리 잡기)
- 여러 차원 자리 잡기
- 자리 잡기로 제자리에서 고치기
"""

import torch
# 무작위로 뽑는 값이 아래에 나온다. 씨앗을 고정해야 이 쪽에 실린
# 수가 다시 나온다 — 고정하지 않으면 돌릴 때마다 다른 수가 찍힌다
torch.manual_seed(0)

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
    
    # 텐서 뒤집기.
    # 넘파이와 달리 PyTorch는 음수 보폭 자르기(vec[::-1])를 받지 않는다.
    # "step must be greater than zero" 오류가 나므로 torch.flip을 쓴다
    reversed_vec = torch.flip(vec, dims=[0])
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
    main()
```

??? note "전체 출력 (138줄)"

    ```

    ======================================================================
    Setup: Sample Tensors
    ======================================================================
    1D tensor (vec): tensor([ 10,  20,  30,  40,  50,  60,  70,  80,  90, 100])
    2D tensor (mat):
     tensor([[ 1,  2,  3,  4],
            [ 5,  6,  7,  8],
            [ 9, 10, 11, 12]])
    3D tensor:
     tensor([[[ 1,  2,  3,  4],
             [ 5,  6,  7,  8],
             [ 9, 10, 11, 12]],

            [[13, 14, 15, 16],
             [17, 18, 19, 20],
             [21, 22, 23, 24]]])

    ======================================================================
    1. Basic Indexing - Single Elements
    ======================================================================
    vec[3] = 40
    vec[-1] = 100, vec[-2] = 90
    mat[1, 2] = 7
    tensor_3d[0, 1, 2] = 7
    Type: <class 'torch.Tensor'>, Shape: torch.Size([])
    Python int: 40, Type: <class 'int'>

    ======================================================================
    2. Slicing - Extracting Sub-tensors
    ======================================================================
    vec[2:5] = tensor([30, 40, 50])
    vec[:4] = tensor([10, 20, 30, 40])
    vec[6:] = tensor([ 70,  80,  90, 100])
    vec[::2] = tensor([10, 30, 50, 70, 90])
    vec[::-1] = tensor([100,  90,  80,  70,  60,  50,  40,  30,  20,  10])

    ======================================================================
    3. Multi-dimensional Slicing
    ======================================================================
    Original matrix (mat):
     tensor([[ 1,  2,  3,  4],
            [ 5,  6,  7,  8],
            [ 9, 10, 11, 12]])
    Row 1 (mat[1, :]): tensor([5, 6, 7, 8])
    Column 2 (mat[:, 2]): tensor([ 3,  7, 11])
    Sub-matrix (mat[0:2, 1:3]):
    tensor([[2, 3],
            [6, 7]])
    Every other row:
    tensor([[ 1,  2,  3,  4],
            [ 9, 10, 11, 12]])

    ======================================================================
    4. Ellipsis (...) - Shorthand for ':' across dimensions
    ======================================================================
    tensor_3d[..., 0] shape: torch.Size([2, 3])
    tensor_3d[..., 0]:
    tensor([[ 1,  5,  9],
            [13, 17, 21]])
    tensor_3d[1, ...] shape: torch.Size([3, 4])

    ======================================================================
    5. Boolean Indexing (Masking)
    ======================================================================
    Mask (vec > 50): tensor([False, False, False, False, False,  True,  True,  True,  True,  True])
    vec[mask] (elements > 50): tensor([ 60,  70,  80,  90, 100])
    vec[(vec > 30) & (vec < 80)]: tensor([40, 50, 60, 70])
    Elements > 6 in mat: tensor([ 7,  8,  9, 10, 11, 12])

    ======================================================================
    6. Advanced Indexing - Index Tensors
    ======================================================================
    vec[[0, 2, 4]]: tensor([10, 30, 50])
    mat[row_indices, col_indices]: tensor([ 2,  7, 12])

    ======================================================================
    7. In-place Modification via Indexing
    ======================================================================
    Original: tensor([ 10,  20,  30,  40,  50,  60,  70,  80,  90, 100])
    After vec_copy[3] = 999: tensor([ 10,  20,  30, 999,  50,  60,  70,  80,  90, 100])
    After vec_copy[5:8] = 0: tensor([ 10,  20,  30, 999,  50,   0,   0,   0,  90, 100])
    After setting elements < 40 to -1: tensor([ -1,  -1,  -1, 999,  50,  -1,  -1,  -1,  90, 100])
    Modified matrix:
    tensor([[ 0,  0,  0, 99],
            [ 5,  6,  7, 99],
            [ 9, 10, 11, 99]])

    ======================================================================
    8. View vs Copy - Memory Behavior
    ======================================================================
    Original: tensor([1, 2, 3, 4, 5])
    View: tensor([2, 3, 4])
    After view[0] = 999:
    Original: tensor([  1, 999,   3,   4,   5])
    View: tensor([999,   3,   4])

    With .clone():
    Original: tensor([1, 2, 3, 4, 5])
    Copy: tensor([999,   3,   4])

    Shared storage? False
    Same underlying storage? True

    ======================================================================
    9. Common Patterns and Use Cases
    ======================================================================
    Diagonal: tensor([1, 5, 9])
    Selected batch shape: torch.Size([4, 10])
    After filtering: tensor([1, 3, 4, 6])
    Clipped data: tensor([ 1.,  2., 10.,  3.,  0.,  4.])

    ======================================================================
    Practice Exercises
    ======================================================================

        이해했는지 다음 학습으로 따져 보아라.
        
        1. 5x5 행렬을 만들고 네 모서리(원소 4개: [0,0], [0,4], [4,0], [4,4])를 뽑아라
        2. 원소 20개짜리 1차원 텐서에서 번호 1부터 세 칸마다 하나씩 골라라
        3. 4x6 행렬을 만들고 둘째 줄과 셋째 칸의 원소를 모두 0으로 두어라
        4. 참거짓 자리 잡기로 텐서에서 5와 15 사이의 원소를 모두 찾아라
        5. 3x3 행렬을 만들고 자리 잡기로 첫 줄과 마지막 줄을 맞바꾸어라
        
        풀이는 아래에 있다...
        
    Exercise 1 - Corners: tensor([ 0,  4, 20, 24])
    Exercise 2 - Every 3rd from index 1: tensor([ 1,  4,  7, 10, 13, 16, 19])
    Exercise 3 - Modified matrix:
    tensor([[1., 1., 0., 1., 1., 1.],
            [0., 0., 0., 0., 0., 0.],
            [1., 1., 0., 1., 1., 1.],
            [1., 1., 0., 1., 1., 1.]])
    Exercise 4 - Elements between 5 and 15: tensor([ 5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15])
    Exercise 5 - After swapping rows:
    tensor([[6, 7, 8],
            [3, 4, 5],
            [0, 1, 2]])
    ```


## 2. 논의

**기본 슬라이싱은 뷰를 주고, 골라 말하는 색인은 베낀 것을 준다.**

| 적는 법 | 저장소를 | 갈래 |
|---|---|---|
| `a[1:3]`, `a[::2]`, `a[...]`, `a[0]` | 함께 쓴다 | 기본 슬라이싱 |
| `a[a > 2]` | 베낀다 | 불리언 마스크 |
| `a[torch.tensor([0, 2])]` | 베낀다 | 색인 텐서 |

가르는 기준은 **규칙적인가**다. 범위와 걸음으로 적히는 것은 스트라이드만 고쳐서 같은 저장소를 다르게 보면 되므로 베낄 일이 없다. 그런데 마스크가 고르는 자리는 띄엄띄엄할 수 있어서 어떤 스트라이드로도 나타낼 수 없다. 그래서 모아서 새 저장소에 담는다.

**그런데 등호의 왼쪽에서는 규칙이 뒤집힌다.** 마스크로 **읽으면** 베낀 것이 나오지만, 마스크로 **쓰면** 원본이 바뀐다.

```python
sel = a[a > 2]; sel[0] = -1      # a는 그대로다 — sel은 베낀 것이다
a[a > 2] = -1                    # a가 바뀐다
```

같은 `a[a > 2]`인데 결과가 반대다. 파이썬이 두 경우를 다른 함수로 넘기기 때문이다. 오른쪽에 있으면 `__getitem__`이 불려 **새 텐서를 만들어 돌려주고**, 왼쪽에 있으면 `__setitem__`이 불려 **원본에 직접 써 넣는다.** 중간에 텐서를 만들 일이 없으므로 베낄 일도 없다.

여기서 조용히 틀리는 자리가 나온다.

```python
a[a > 2][0] = -1      # 아무 일도 일어나지 않는다. 오류도 나지 않는다
```

`a[a > 2]`가 먼저 베낀 것을 만들고, `[0] = -1`은 **그 베낀 것**에 써 넣는다. 베낀 것은 아무도 붙들고 있지 않으므로 그대로 버려진다. `a`는 그대로이고 오류도 나지 않는다.

기본 슬라이싱에서는 같은 사슬이 **먹힌다.** `a[0]`이 뷰이므로 `a[0][1] = 99`는 원본을 고친다. 그래서 "사슬로 적으면 되는가"가 어느 색인을 썼는지에 달려 있다 — **한 줄에 두 번 색인하지 않는 것**이 안전한 규칙이다.

```python
a[0, 1] = 99          # 한 번에 적는다. 늘 먹힌다
```

**생략 부호와 음수 색인.** `a[..., 0]`은 "앞의 차원은 모두 그대로, 마지막 차원의 0번"이라는 뜻이다. 계수가 몇이든 같은 뜻으로 읽히므로 `a[:, :, 0]`처럼 차원 수에 매여 적지 않아도 된다. 음수는 뒤에서부터 센다 — `a[-1]`이 마지막 원소다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
`a = torch.arange(6.)`에 대해 아래 넷이 `a`와 저장소를 함께 쓰는지 각각 판정하라. 가르는 기준을 한 문장으로 적어라.

```python
a[1:3]      a[::2]      a[a > 2]      a[torch.tensor([0, 2])]
```

</div>

??? success "연습문제 1 풀이"
    ```python
    import torch

    def shares(x, y):
        return x.untyped_storage().data_ptr() == y.untyped_storage().data_ptr()

    a = torch.arange(6.)
    print("a[1:3]           :", shares(a, a[1:3]))
    print("a[::2]           :", shares(a, a[::2]))
    print("a[a > 2]         :", shares(a, a[a > 2]))
    print("a[tensor([0, 2])]:", shares(a, a[torch.tensor([0, 2])]))
    ```

    ```
    a[1:3]           : True
    a[::2]           : True
    a[a > 2]         : False
    a[tensor([0, 2])]: False
    ```

    기준은 **고른 자리가 규칙적인가**다. 범위와 걸음으로 적히면 스트라이드만 고쳐
    같은 저장소를 다르게 보면 되므로 베낄 일이 없다. `a[::2]`도 걸음이 2인 규칙이라
    뷰다. 반면 마스크나 색인 텐서가 고르는 자리는 띄엄띄엄할 수 있어 어떤 스트라이드로도
    나타낼 수 없으므로, 모아서 새 저장소에 담는다.

---


<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
아래 둘을 모두 돌려 보라. 같은 `a[a > 2]`를 쓰는데 `a`가 바뀌는 쪽과 바뀌지 않는 쪽이 갈린다. 왜 그런지 설명하라.

```python
# (가)
a = torch.arange(6.); sel = a[a > 2]; sel[0] = -1

# (나)
b = torch.arange(6.); b[b > 2] = -1
```

</div>

??? success "연습문제 2 풀이"
    ```python
    import torch

    a = torch.arange(6.)
    sel = a[a > 2]
    sel[0] = -1
    print("(가) a:", a)

    b = torch.arange(6.)
    b[b > 2] = -1
    print("(나) b:", b)
    ```

    ```
    (가) a: tensor([0., 1., 2., 3., 4., 5.])
    (나) b: tensor([ 0.,  1.,  2., -1., -1., -1.])
    ```

    `a[a > 2]`가 **등호의 어느 쪽에 있는가**가 다르다.

    (가)에서는 오른쪽에 있으므로 `__getitem__`이 불린다. 이것은 고른 원소를 모아
    **새 텐서**를 만들어 돌려준다. `sel`은 베낀 것이고, 거기에 쓰는 일은 `a`와 상관이 없다.

    (나)에서는 왼쪽에 있으므로 `__setitem__`이 불린다. 이것은 텐서를 만들지 않고
    `b`의 저장소에서 마스크가 참인 자리를 찾아 **직접 써 넣는다.** 중간에 베낄 것이 없다.

    곧 "마스크 색인은 베낀다"는 말은 **읽을 때만** 맞다.

---


<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
아래 두 줄 가운데 하나는 먹히고 하나는 **아무 일도 하지 않으면서 오류도 내지 않는다.**

```python
a = torch.arange(6.).reshape(2, 3);  a[0][1] = 99.
b = torch.arange(6.);                b[b > 2][0] = -1.
```

어느 쪽이 먹히는지 확인하고, 둘이 갈리는 까닭을 밝혀라. 그리고 이런 실수를 아예 만들지 않는 적는 법을 제시하라.

</div>

??? success "연습문제 3 풀이"
    ```python
    import torch

    a = torch.arange(6.).reshape(2, 3)
    a[0][1] = 99.
    print("a[0][1] = 99  ->", a[0].tolist())

    b = torch.arange(6.)
    b[b > 2][0] = -1.
    print("b[b>2][0] = -1 ->", b.tolist())
    ```

    ```
    a[0][1] = 99  -> [0.0, 99.0, 2.0]
    b[b>2][0] = -1 -> [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    ```

    두 줄 모두 **색인을 두 번** 한다. 그래서 앞의 색인이 무엇을 돌려주는지가 결과를
    정한다.

    - `a[0]`은 기본 슬라이싱이라 **뷰**다. 뒤의 `[1] = 99`는 그 뷰를 통해 원본 저장소에
      써 넣으므로 `a`가 바뀐다.
    - `b[b > 2]`는 마스크 색인이라 **베낀 것**이다. 뒤의 `[0] = -1`은 그 베낀 것에 써
      넣는다. 그런데 그 베낀 것을 아무도 붙들고 있지 않으므로 다음 순간 버려진다.
      `b`는 그대로이고, 버려졌다는 말은 아무도 해 주지 않는다.

    `__setitem__`이 불리는 것은 **마지막** 색인뿐이고, 그 앞의 색인은 모두
    `__getitem__`이다. 그러므로 앞에서 한 번이라도 베끼면 쓰기가 허공에 떨어진다.

    아예 만들지 않는 방법은 **한 줄에 색인을 한 번만 하는 것**이다.

    ```python
    a = torch.arange(6.).reshape(2, 3)
    a[0, 1] = 99.          # 쉼표로 한 번에 — 늘 먹힌다

    b = torch.arange(6.)
    b[b > 2] = -1.         # 마스크로 한 번에 — 먹힌다
    ```

    대괄호를 두 번 여는 자리가 보이면 앞의 것이 뷰인지 베낀 것인지 따져야 한다는
    신호다. 쉼표로 합칠 수 있으면 합친다.

## 정리하며

색인은 두 갈래이고, 읽을 때와 쓸 때가 또 다르다.

- **읽을 때** — 범위와 걸음으로 적는 기본 슬라이싱(`a[1:3]`, `a[::2]`, `a[...]`, `a[0]`)은 **뷰**다. 마스크나 색인 텐서로 골라 말하면 **베낀 것**이다. 고른 자리가 규칙적이면 스트라이드로 나타낼 수 있고, 띄엄띄엄하면 모아 담아야 하기 때문이다.
- **쓸 때** — 마스크든 색인 텐서든 **원본에 직접 써 넣는다.** `__setitem__`은 중간 텐서를 만들지 않는다.

그래서 "마스크 색인은 베낀다"는 말은 반만 맞다. 등호의 오른쪽에서만 맞다.

여기서 나오는 함정이 **한 줄에 두 번 색인하는 것**이다. `a[b > 2][0] = -1`은 베낀 것에 써 넣고 그것을 버리므로, 오류 없이 아무 일도 하지 않는다. 마지막 색인만 쓰기가 되고 앞의 것은 모두 읽기이기 때문이다.

규칙 하나로 줄이면 — **쉼표로 합칠 수 있으면 합친다.** `a[0, 1] = 99`는 뷰인지 베낀 것인지 따질 필요가 없다.
