# 비교, 범위 제한, 정렬

이 스크립트는 텐서의 비교, 범위 제한, 정렬을 보여준다. 이 개념들을 이해하는 것은 효과적인 PyTorch 프로그래밍과 딥러닝 모델 개발에 필수적이다.

## 1. 코드

```python
#!/usr/bin/env python3
"""
견주기, 조건 셈, 줄 세우기, 고르기.

Covers:
- 견주기 셈: ==, !=, <, <=, >, >=
- 논리 셈: &, |, ~
- 조건에 따라 고르는 torch.where
- 제한하기: clamp, clip
- 줄 세우기: sort, argsort
- 위 k개 고르기: topk, kthvalue
- 원소 찾기: eq, ne, lt, le, gt, ge
- isnan, isinf, isfinite 살피기
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
    torch.manual_seed(42)

    # -------------------------------------------------------------------------
    header("Basic comparison operators")
    a = torch.tensor([1, 2, 3, 4, 5])
    b = torch.tensor([1, 1, 3, 5, 5])
    
    print("a:", a)
    print("b:", b)
    print("a == b:", a == b)
    print("a != b:", a != b)
    print("a < b:", a < b)
    print("a <= b:", a <= b)
    print("a > b:", a > b)
    print("a >= b:", a >= b)

    # -------------------------------------------------------------------------
    header("Comparison with scalars")
    x = torch.randn(5)
    print("x:", x)
    print("x > 0:", x > 0)
    print("x <= 0.5:", x <= 0.5)
    print("x == 0:", x == 0)

    # -------------------------------------------------------------------------
    header("Element-wise comparison functions")
    a = torch.tensor([1., 2., 3.])
    b = torch.tensor([2., 2., 2.])
    
    print("torch.eq(a, b):", torch.eq(a, b))  # Equal
    print("torch.ne(a, b):", torch.ne(a, b))  # Not equal
    print("torch.lt(a, b):", torch.lt(a, b))  # Less than
    print("torch.le(a, b):", torch.le(a, b))  # Less or equal
    print("torch.gt(a, b):", torch.gt(a, b))  # Greater than
    print("torch.ge(a, b):", torch.ge(a, b))  # Greater or equal

    # -------------------------------------------------------------------------
    header("Logical operators: &, |, ~ (and, or, not)")
    a = torch.tensor([True, True, False, False])
    b = torch.tensor([True, False, True, False])
    
    print("a:", a)
    print("b:", b)
    print("a & b (and):", a & b)
    print("a | b (or):", a | b)
    print("~a (not):", ~a)
    
    # 조건 결합
    x = torch.randn(10)
    print("\nx:", x)
    in_range = (x > -0.5) & (x < 0.5)
    print("In range [-0.5, 0.5]:", in_range)
    print("Values:", x[in_range])

    # -------------------------------------------------------------------------
    header("torch.where: conditional selection")
    condition = torch.tensor([True, False, True, False])
    a = torch.tensor([1, 2, 3, 4])
    b = torch.tensor([10, 20, 30, 40])
    
    # 조건이 참인 곳은 a에서, 그 밖에는 b에서 선택
    result = torch.where(condition, a, b)
    print("condition:", condition)
    print("a:", a)
    print("b:", b)
    print("where(condition, a, b):", result)
    
    # 브로드캐스팅 사용
    x = torch.randn(5)
    result = torch.where(x > 0, x, torch.tensor(0.))
    print("\nx:", x)
    print("ReLU (x if x>0 else 0):", result)

    # -------------------------------------------------------------------------
    header("clamp: limit values to range")
    x = torch.tensor([-2., -1., 0., 1., 2., 3., 4.])
    print("x:", x)
    
    # [0, 2]로 범위 제한
    clamped = torch.clamp(x, min=0, max=2)
    print("clamp(0, 2):", clamped)
    
    # 최솟값만
    clamped_min = torch.clamp(x, min=0)
    print("clamp(min=0):", clamped_min)
    
    # 최댓값만
    clamped_max = torch.clamp(x, max=2)
    print("clamp(max=2):", clamped_max)
    
    # 제자리 버전
    x_copy = x.clone()
    x_copy.clamp_(0, 2)
    print("After clamp_(0, 2):", x_copy)

    # -------------------------------------------------------------------------
    header("clip: alias for clamp")
    x = torch.randn(5)
    clipped = torch.clip(x, -1, 1)
    clamped = torch.clamp(x, -1, 1)
    print("clip and clamp are identical:", torch.allclose(clipped, clamped))

    # -------------------------------------------------------------------------
    header("sort: sort values along dimension")
    x = torch.tensor([3, 1, 4, 1, 5, 9, 2, 6, 5])
    print("x:", x)
    
    # 오름차순 정렬(기본값)
    sorted_vals, sorted_indices = torch.sort(x)
    print("Sorted values:", sorted_vals)
    print("Sorted indices:", sorted_indices)
    
    # 내림차순 정렬
    sorted_desc, indices_desc = torch.sort(x, descending=True)
    print("Sorted descending:", sorted_desc)
    
    # 2차원 정렬
    mat = torch.randint(0, 10, (3, 4))
    print("\nMatrix:\n", mat)
    sorted_rows, _ = torch.sort(mat, dim=1)
    print("Sorted rows:\n", sorted_rows)
    sorted_cols, _ = torch.sort(mat, dim=0)
    print("Sorted columns:\n", sorted_cols)

    # -------------------------------------------------------------------------
    header("argsort: indices that would sort the tensor")
    x = torch.tensor([3, 1, 4, 1, 5, 9, 2, 6, 5])
    print("x:", x)
    
    indices = torch.argsort(x)
    print("argsort:", indices)
    
    # 인덱스로 정렬
    sorted_x = x[indices]
    print("x[argsort]:", sorted_x)
    
    # 내림차순
    indices_desc = torch.argsort(x, descending=True)
    print("argsort(descending):", indices_desc)

    # -------------------------------------------------------------------------
    header("topk: k largest (or smallest) elements")
    x = torch.tensor([3, 1, 4, 1, 5, 9, 2, 6, 5])
    print("x:", x)
    
    # 상위 3개 값
    top_vals, top_indices = torch.topk(x, k=3)
    print("Top 3 values:", top_vals)
    print("Top 3 indices:", top_indices)
    
    # 하위 3개(가장 작은 값들)
    bottom_vals, bottom_indices = torch.topk(x, k=3, largest=False)
    print("Bottom 3 values:", bottom_vals)
    print("Bottom 3 indices:", bottom_indices)
    
    # 2차원 topk
    mat = torch.randint(0, 10, (3, 5))
    print("\nMatrix:\n", mat)
    top_vals, top_indices = torch.topk(mat, k=2, dim=1)
    print("Top 2 per row:\n", top_vals)

    # -------------------------------------------------------------------------
    header("kthvalue: k-th smallest element")
    x = torch.tensor([3, 1, 4, 1, 5, 9, 2, 6, 5])
    print("x:", x)
    
    # 다섯 번째로 작은 값(원소가 9개일 때의 중앙값)
    kth_val, kth_idx = torch.kthvalue(x, k=5)
    print("5th smallest value:", kth_val.item())
    print("5th smallest index:", kth_idx.item())

    # -------------------------------------------------------------------------
    header("Checking for special values: nan, inf")
    x = torch.tensor([1., float('nan'), 3., float('inf'), -float('inf')])
    print("x:", x)
    
    print("isnan:", torch.isnan(x))
    print("isinf:", torch.isinf(x))
    print("isfinite:", torch.isfinite(x))
    print("isposinf:", torch.isposinf(x))
    print("isneginf:", torch.isneginf(x))
    
    # 특수한 값 세기
    print("Number of NaNs:", torch.isnan(x).sum().item())
    print("Number of infs:", torch.isinf(x).sum().item())

    # -------------------------------------------------------------------------
    header("allclose and isclose: approximate equality")
    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.tensor([1.0001, 2.0001, 3.0001])
    
    print("a:", a)
    print("b:", b)
    print("a == b:", (a == b).all().item())
    print("allclose (default tol):", torch.allclose(a, b))
    print("allclose (strict tol):", torch.allclose(a, b, atol=1e-5, rtol=1e-5))
    
    # 원소별 확인
    close = torch.isclose(a, b)
    print("isclose:", close)

    # -------------------------------------------------------------------------
    header("equal: exact equality of all elements")
    a = torch.tensor([1, 2, 3])
    b = torch.tensor([1, 2, 3])
    c = torch.tensor([1, 2, 4])
    
    print("equal(a, b):", torch.equal(a, b))
    print("equal(a, c):", torch.equal(a, c))

    # -------------------------------------------------------------------------
    header("maximum and minimum: element-wise max/min of two tensors")
    a = torch.tensor([1, 5, 3, 7])
    b = torch.tensor([4, 2, 6, 1])
    
    print("a:", a)
    print("b:", b)
    print("maximum(a, b):", torch.maximum(a, b))
    print("minimum(a, b):", torch.minimum(a, b))

    # -------------------------------------------------------------------------
    header("Practical: filtering with conditions")
    # 온도 데이터
    temps = torch.tensor([15.5, 18.2, 22.1, 19.8, 25.3, 28.7, 23.4])
    print("Temperatures:", temps)
    
    # 적당한 범위 찾기
    comfortable = (temps >= 18) & (temps <= 25)
    print("Comfortable days:", comfortable)
    print("Comfortable temps:", temps[comfortable])
    
    # 개수
    print("Number of comfortable days:", comfortable.sum().item())

    # -------------------------------------------------------------------------
    header("Practical: outlier detection")
    data = torch.randn(100)
    
    # 표준편차 2배를 넘는 값을 이상치로 정의한다
    mean = data.mean()
    std = data.std()
    
    outliers = (data < mean - 2*std) | (data > mean + 2*std)
    print("Total points:", len(data))
    print("Outliers:", outliers.sum().item())
    print("Outlier percentage:", f"{100*outliers.float().mean():.1f}%")

    # -------------------------------------------------------------------------
    header("Practical: top-k accuracy")
    # 모델 예측과 목표값 흉내 내기
    logits = torch.randn(10, 5)  # 10 samples, 5 classes
    targets = torch.randint(0, 5, (10,))
    
    print("Logits shape:", logits.shape)
    print("Targets:", targets)
    
    # 상위 1개 정확도
    pred_top1 = logits.argmax(dim=1)
    top1_acc = (pred_top1 == targets).float().mean()
    print("Top-1 accuracy:", f"{100*top1_acc:.1f}%")
    
    # 상위 3개 정확도
    _, pred_top3 = logits.topk(k=3, dim=1)
    top3_acc = (pred_top3 == targets.unsqueeze(1)).any(dim=1).float().mean()
    print("Top-3 accuracy:", f"{100*top3_acc:.1f}%")

    # -------------------------------------------------------------------------
    header("Practical: thresholding")
    # 이미지 형태의 데이터
    image = torch.rand(5, 5)
    print("Image:\n", image)
    
    # 0.5를 기준으로 이진화
    binary = (image > 0.5).float()
    print("Binary (threshold=0.5):\n", binary)
    
    # 부드러운 임계 처리(clamp)
    threshold = 0.3
    soft = torch.where(image > threshold, 
                       image - threshold, 
                       torch.zeros_like(image))
    print("Soft threshold (0.3):\n", soft)

    # -------------------------------------------------------------------------
    header("Practical: NaN handling")
    data = torch.tensor([1., 2., float('nan'), 4., float('nan'), 6.])
    print("Data with NaNs:", data)
    
    # NaN을 유효한 값들의 평균으로 대체
    valid_mask = ~torch.isnan(data)
    mean_val = data[valid_mask].mean()
    cleaned = torch.where(torch.isnan(data), mean_val, data)
    print("After replacing NaNs with mean:", cleaned)
    
    # 또는 NaN을 제거한다
    clean_data = data[~torch.isnan(data)]
    print("After removing NaNs:", clean_data)

    # -------------------------------------------------------------------------
    header("Practical: ranking")
    scores = torch.tensor([85, 92, 78, 95, 88, 92, 70])
    print("Scores:", scores)
    
    # 순위 얻기(1부터 시작. 점수가 높을수록 순위 번호가 작다)
    sorted_scores, indices = torch.sort(scores, descending=True)
    ranks = torch.zeros_like(scores)
    ranks[indices] = torch.arange(1, len(scores) + 1)
    print("Ranks:", ranks)
    print("(1 = highest, 7 = lowest)")

    # -------------------------------------------------------------------------
    header("Quick reference: comparisons and selection")
    print("\nComparison operators:")
    print("  ==, !=, <, <=, >, >=  - Element-wise comparison")
    print("  torch.eq, ne, lt, le, gt, ge - Functional forms")
    
    print("\nLogical operators:")
    print("  &, |, ~               - Logical and, or, not")
    print("  torch.logical_and/or/not/xor - Functional forms")
    
    print("\nConditional selection:")
    print("  torch.where(cond, a, b) - Select from a or b based on condition")
    
    print("\nClamping:")
    print("  torch.clamp(x, min, max) - Limit values to range")
    print("  torch.clip(x, min, max)  - Alias for clamp")
    
    print("\nSorting:")
    print("  torch.sort(x)         - Sort values and return indices")
    print("  torch.argsort(x)      - Indices that would sort")
    
    print("\nSelection:")
    print("  torch.topk(x, k)      - Top k values/indices")
    print("  torch.kthvalue(x, k)  - k-th smallest value")
    
    print("\nSpecial value checks:")
    print("  torch.isnan(x)        - Check for NaN")
    print("  torch.isinf(x)        - Check for infinity")
    print("  torch.isfinite(x)     - Check for finite values")
    
    print("\nEquality checks:")
    print("  torch.equal(a, b)     - Exact equality")
    print("  torch.allclose(a, b)  - Approximate equality")
    print("  torch.isclose(a, b)   - Element-wise approximate equality")
    
    print("\nElement-wise max/min:")
    print("  torch.maximum(a, b)   - Element-wise maximum")
    print("  torch.minimum(a, b)   - Element-wise minimum")

if __name__ == "__main__":
    main()```

## 2. 논의

브로드캐스팅은 작은 텐서를 가상으로 확장하여 모양이 다른 텐서 사이의 원소별 연산을 가능하게 한다. PyTorch는 차원을 오른쪽부터 맞추며, 각 차원 쌍이 서로 같거나, 둘 중 하나가 1이거나, 아예 없을 것을 요구한다. 이로써 데이터를 명시적으로 복제하지 않아도 되어 메모리 효율이 좋고 빠르다.

텐서 생성 함수는 데이터를 초기화하는 유연한 방법을 제공한다. `torch.zeros`, `torch.randn`, `torch.arange` 같은 팩토리 함수는 `dtype`, `device`, `requires_grad` 매개변수를 받으므로 불필요한 복사 없이 목표 장치에 곧바로 할당할 수 있다.

`sum()`, `mean()`, `min()`, `max()` 같은 축약 연산은 텐서의 하나 이상의 차원을 접는다. `dim` 매개변수가 어느 차원을 축약할지 지정하며, `keepdim=True`는 축약된 차원을 크기 1로 남겨 둔다. 이는 이후 연산에서 올바른 브로드캐스팅을 위해 필수적이다.

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

**다룬 것** — 비교, 범위 제한, 정렬

브로드캐스팅은 작은 텐서를 가상으로 확장하여 모양이 다른 텐서 사이의 원소별 연산을 가능하게 한다.

앞의 연습문제 3개로 직접 확인할 수 있다.
