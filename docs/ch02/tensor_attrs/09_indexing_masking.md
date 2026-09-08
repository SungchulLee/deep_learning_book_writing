# 인덱싱과 마스킹

이 스크립트는 텐서의 인덱싱과 마스킹을 보여준다. 이 개념들을 이해하는 것은 효과적인 PyTorch 프로그래밍과 딥러닝 모델 개발에 필수적이다.

## 1. 코드

```python
#!/usr/bin/env python3
"""
앞선 자리 잡기, 자르기, 가리기 셈.

Covers:
- 기본 자르기와 앞선 자리 잡기 견주기
- 참거짓 가리기(조건에 따라 고르기)
- 정수 배열로 자리 잡기(멋진 자리 잡기)
- 조건 셈을 위한 torch.where
- masked_fill, masked_select, masked_scatter
- 너그러운 자리 잡기를 위한 줄임표(...)
- 보기와 베낌의 다름
"""

import torch

# ========================================================================
# 메인
# ========================================================================

def header(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

def main():
    torch.manual_seed(0)

    # -------------------------------------------------------------------------
    header("Basic slicing (creates views)")
    a = torch.arange(20).reshape(4, 5)
    print("a:\n", a)
    
    view = a[1:3, 2:4]  # Slice rows 1-2, cols 2-3
    print("view = a[1:3, 2:4]:\n", view)
    print("Is view (shares storage):", id(a.storage()) == id(view.storage()))
    
    # 뷰를 수정하면 원본이 바뀐다
    view[0, 0] = 999
    print("After view[0,0]=999, a[1,2]:", a[1, 2].item())

    # -------------------------------------------------------------------------
    header("Boolean masking (fancy indexing - creates copies)")
    a = torch.tensor([1, -2, 3, -4, 5, -6])
    print("a:", a)
    
    # 불리언 마스크 만들기
    mask = a > 0
    print("mask (a > 0):", mask)
    
    # 마스크로 인덱싱(1차원 텐서를 반환하며 뷰가 아니라 복사본이다)
    positive = a[mask]
    print("a[mask] (positives):", positive)
    
    # 수정해도 원본에 영향이 없다(복사본이기 때문이다)
    positive[0] = 100
    print("After positive[0]=100, a:", a)

    # -------------------------------------------------------------------------
    header("Boolean mask assignment (in-place modification)")
    a = torch.tensor([1, -2, 3, -4, 5, -6])
    print("Before: a:", a)
    
    a[a < 0] = 0  # Set all negative values to 0
    print("After a[a < 0] = 0:", a)
    
    # &(and)와 |(or)로 여러 조건 결합
    b = torch.randn(10)
    print("\nb:", b)
    b[(b > -0.5) & (b < 0.5)] = 0  # Set small values to 0
    print("After clipping to zero:", b)

    # -------------------------------------------------------------------------
    header("torch.where for conditional selection")
    a = torch.randn(5)
    b = torch.randn(5)
    print("a:", a)
    print("b:", b)
    
    # a>0인 곳은 a에서, 그 밖에는 b에서 선택
    result = torch.where(a > 0, a, b)
    print("where(a > 0, a, b):", result)
    
    # 스칼라를 쓸 수도 있다
    clamped = torch.where(a > 0, a, torch.tensor(0.0))
    print("Clamp negatives to 0:", clamped)

    # -------------------------------------------------------------------------
    header("Integer array indexing (fancy indexing)")
    a = torch.arange(12).reshape(3, 4)
    print("a:\n", a)
    
    # 정수 텐서로 인덱싱
    row_idx = torch.tensor([0, 2, 1])
    col_idx = torch.tensor([1, 3, 2])
    
    # 이것은 a[0,1], a[2,3], a[1,2]를 선택한다
    selected = a[row_idx, col_idx]
    print("a[row_idx, col_idx]:", selected)
    
    # 특정 원소를 모을 때 쓸 수 있다
    rows = torch.tensor([0, 0, 1, 1, 2, 2])
    cols = torch.tensor([0, 1, 2, 3, 0, 1])
    gathered = a[rows, cols]
    print("Gathered elements:", gathered)

    # -------------------------------------------------------------------------
    header("Advanced indexing with broadcasting")
    a = torch.arange(12).reshape(3, 4)
    print("a:\n", a)
    
    # 행 전체 선택
    row_indices = torch.tensor([0, 2])
    selected_rows = a[row_indices]  # Shape: (2, 4)
    print("a[row_indices]:\n", selected_rows)
    
    # 열 전체 선택(첫 차원에 :가 필요하다)
    col_indices = torch.tensor([1, 3])
    selected_cols = a[:, col_indices]  # Shape: (3, 2)
    print("a[:, col_indices]:\n", selected_cols)
    
    # 브로드캐스팅을 이용한 2차원 인덱싱
    row_idx = torch.tensor([[0], [1], [2]])  # (3, 1)
    col_idx = torch.tensor([[0, 2]])          # (1, 2)
    # 브로드캐스팅되어 (0,0), (0,2), (1,0), (1,2), (2,0), (2,2)를 선택한다
    subgrid = a[row_idx, col_idx]  # Shape: (3, 2)
    print("Broadcasted 2D indexing:\n", subgrid)

    # -------------------------------------------------------------------------
    header("masked_fill: fill values based on mask")
    a = torch.randn(3, 4)
    print("a:\n", a)
    
    mask = a > 0
    filled = a.masked_fill(mask, value=-999)
    print("masked_fill(a > 0, -999):\n", filled)
    
    # 제자리 버전
    a.masked_fill_(a.abs() < 0.5, value=0)
    print("After masking small values in-place:\n", a)

    # -------------------------------------------------------------------------
    header("masked_select: extract values matching mask")
    a = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    print("a:\n", a)
    
    mask = a > 4
    print("mask (a > 4):\n", mask)
    
    selected = a.masked_select(mask)
    print("masked_select:", selected)

    # -------------------------------------------------------------------------
    header("masked_scatter: scatter values based on mask")
    a = torch.zeros(3, 4)
    mask = torch.tensor([[True, False, True, False],
                         [False, True, False, True],
                         [True, True, False, False]])
    source = torch.arange(1, 7)  # Values to scatter
    
    result = a.masked_scatter(mask, source)
    print("mask:\n", mask)
    print("source:", source)
    print("masked_scatter result:\n", result)

    # -------------------------------------------------------------------------
    header("Ellipsis (...) for flexible multi-dimensional indexing")
    a = torch.randn(2, 3, 4, 5, 6)
    print("a.shape:", a.shape)
    
    # ...은 필요한 만큼 :로 확장된다
    slice1 = a[..., 0]        # Same as a[:, :, :, :, 0]
    print("a[..., 0].shape:", slice1.shape)
    
    slice2 = a[0, ..., 2]     # Same as a[0, :, :, :, 2]
    print("a[0, ..., 2].shape:", slice2.shape)
    
    slice3 = a[..., 1:3, :]   # Same as a[:, :, :, 1:3, :]
    print("a[..., 1:3, :].shape:", slice3.shape)

    # -------------------------------------------------------------------------
    header("None/newaxis for adding dimensions")
    a = torch.randn(3, 4)
    print("a.shape:", a.shape)
    
    expanded1 = a[None, :, :]     # Same as a.unsqueeze(0)
    print("a[None, :, :].shape:", expanded1.shape)
    
    expanded2 = a[:, None, :]     # Add dimension in middle
    print("a[:, None, :].shape:", expanded2.shape)
    
    expanded3 = a[:, :, None]     # Same as a.unsqueeze(2)
    print("a[:, :, None].shape:", expanded3.shape)

    # -------------------------------------------------------------------------
    header("Combining slicing and masking")
    a = torch.randn(4, 5)
    print("a:\n", a)
    
    # 먼저 슬라이싱하고 그다음 마스킹
    submatrix = a[1:3, :]      # Rows 1-2, all columns
    positive_in_sub = submatrix[submatrix > 0]
    print("Positive values in rows 1-2:", positive_in_sub)
    
    # 특정 행 마스킹
    row_mask = torch.tensor([True, False, True, False])
    masked_rows = a[row_mask]
    print("Masked rows (0 and 2):\n", masked_rows)

    # -------------------------------------------------------------------------
    header("nonzero and where for finding indices")
    a = torch.tensor([[0, 1, 0], [2, 0, 3], [0, 4, 0]])
    print("a:\n", a)
    
    # 0이 아닌 원소의 인덱스 얻기
    indices = a.nonzero()
    print("nonzero() indices:\n", indices)
    
    # 대안: torch.where는 인덱스 튜플을 반환한다
    row_idx, col_idx = torch.where(a > 0)
    print("where(a > 0) row indices:", row_idx)
    print("where(a > 0) col indices:", col_idx)
    
    # 인덱스로 원소에 접근
    values = a[row_idx, col_idx]
    print("Values at those indices:", values)

    # -------------------------------------------------------------------------
    header("View vs copy: when does indexing copy?")
    a = torch.arange(12).reshape(3, 4)
    
    # 기본 슬라이싱 → 뷰
    view = a[1:3, 2:4]
    print("Basic slice is view:", id(a.storage()) == id(view.storage()))
    
    # 불리언 인덱싱 → 복사본
    mask = a > 5
    copy1 = a[mask]
    print("Boolean indexing is copy:", id(a.storage()) != id(copy1.storage()))
    
    # 정수 배열 인덱싱 → 복사본
    indices = torch.tensor([0, 2])
    copy2 = a[indices]
    print("Integer array indexing is copy:", id(a.storage()) != id(copy2.storage()))
    
    # 보폭 슬라이싱 → 뷰
    view2 = a[::2, ::2]
    print("Step slicing is view:", id(a.storage()) == id(view2.storage()))

    # -------------------------------------------------------------------------
    header("Practical example: attention masking")
    # 어텐션 점수 흉내 내기(batch=2, heads=1, seq_len=4)
    scores = torch.randn(2, 1, 4, 4)
    print("Attention scores shape:", scores.shape)
    
    # 인과 마스크 만들기(하삼각)
    mask = torch.triu(torch.ones(4, 4), diagonal=1).bool()
    print("Causal mask:\n", mask)
    
    # 마스크 적용(미래 위치를 -inf로 설정)
    scores.masked_fill_(mask, float('-inf'))
    print("Masked scores (first batch):\n", scores[0, 0])

    # -------------------------------------------------------------------------
    header("Practical example: data filtering")
    # 특징과 레이블을 가진 데이터셋
    data = torch.randn(100, 5)      # 100 samples, 5 features
    labels = torch.randint(0, 3, (100,))  # 3 classes
    
    # 클래스 1의 표본만 선택
    class_1_mask = labels == 1
    class_1_data = data[class_1_mask]
    print(f"Total samples: {len(data)}, Class 1 samples: {len(class_1_data)}")
    
    # 첫 번째 특징이 0보다 큰 표본 선택
    feature_mask = data[:, 0] > 0
    filtered_data = data[feature_mask]
    print(f"Samples with feature[0] > 0: {len(filtered_data)}")

if __name__ == "__main__":
    main()```

## 2. 논의

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

## 정리하며

**다룬 것** — 인덱싱과 마스킹

브로드캐스팅은 작은 텐서를 가상으로 확장하여 모양이 다른 텐서 사이의 원소별 연산을 가능하게 한다.

앞의 연습문제 3개로 직접 확인할 수 있다.
