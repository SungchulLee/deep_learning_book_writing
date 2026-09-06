# 축약과 통계

이 스크립트는 텐서의 축약 연산과 통계을 보여준다. 이 개념들을 이해하는 것은 효과적인 PyTorch 프로그래밍과 딥러닝 모델 개발에 필수적이다.

## 코드

```python
#!/usr/bin/env python3
"""
Reduction operations and statistical functions.

Covers:
- Basic reductions: sum, prod, mean, std, var
- Min/max operations: min, max, argmin, argmax, aminmax
- Dimension-wise reductions with dim parameter
- keepdim for preserving dimensions
- Quantiles and percentiles
- Norms: norm, dist
- Logical reductions: all, any
- Counting operations: numel, count_nonzero
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
    header("Basic reductions: sum, mean, prod")
    x = torch.randn(3, 4)
    print("x:\n", x)
    
    print("sum:", x.sum().item())
    print("mean:", x.mean().item())
    print("prod (product):", x.prod().item())

    # -------------------------------------------------------------------------
    header("Dimension-wise reductions")
    x = torch.tensor([[1., 2., 3.],
                      [4., 5., 6.],
                      [7., 8., 9.]])
    print("x:\n", x)
    
    # 0번 차원을 따라 합(행을 접는다)
    sum_dim0 = x.sum(dim=0)
    print("sum(dim=0):", sum_dim0)  # [12, 15, 18]
    print("  Result shape:", sum_dim0.shape)  # (3,)
    
    # 1번 차원을 따라 합(열을 접는다)
    sum_dim1 = x.sum(dim=1)
    print("sum(dim=1):", sum_dim1)  # [6, 15, 24]
    print("  Result shape:", sum_dim1.shape)  # (3,)

    # -------------------------------------------------------------------------
    header("keepdim: preserve reduced dimensions as size 1")
    x = torch.randn(3, 4, 5)
    print("x.shape:", x.shape)
    
    # keepdim 없이(기본값)
    mean_no_keep = x.mean(dim=1)
    print("mean(dim=1).shape:", mean_no_keep.shape)  # (3, 5)
    
    # keepdim 사용
    mean_keep = x.mean(dim=1, keepdim=True)
    print("mean(dim=1, keepdim=True).shape:", mean_keep.shape)  # (3, 1, 5)
    
    # 브로드캐스팅에 유용하다
    normalized = x - mean_keep  # Broadcasts correctly
    print("Normalized shape:", normalized.shape)  # (3, 4, 5)

    # -------------------------------------------------------------------------
    header("Multiple dimension reductions")
    x = torch.randn(2, 3, 4, 5)
    print("x.shape:", x.shape)
    
    # 여러 차원에 걸쳐 축약
    mean_multi = x.mean(dim=(1, 3))
    print("mean(dim=(1,3)).shape:", mean_multi.shape)  # (2, 4)
    
    mean_multi_keep = x.mean(dim=(1, 3), keepdim=True)
    print("mean(dim=(1,3), keepdim=True).shape:", mean_multi_keep.shape)  # (2, 1, 4, 1)

    # -------------------------------------------------------------------------
    header("Standard deviation and variance")
    x = torch.randn(100)
    
    print("std:", x.std().item())
    print("var:", x.var().item())
    print("Relation: var = std²:", x.var().item(), "≈", (x.std() ** 2).item())
    
    # 불편 추정량과 편향 추정량
    print("\nBiased (default, Bessel correction):")
    print("  std(correction=1):", x.std(correction=1).item())
    print("Unbiased:")
    print("  std(correction=0):", x.std(correction=0).item())

    # -------------------------------------------------------------------------
    header("Min and max operations")
    x = torch.randn(3, 4)
    print("x:\n", x)
    
    print("min:", x.min().item())
    print("max:", x.max().item())
    
    # 차원별 min/max는 값과 인덱스를 함께 반환한다
    min_vals, min_indices = x.min(dim=1)
    print("min(dim=1) values:", min_vals)
    print("min(dim=1) indices:", min_indices)
    
    max_vals, max_indices = x.max(dim=1)
    print("max(dim=1) values:", max_vals)
    print("max(dim=1) indices:", max_indices)

    # -------------------------------------------------------------------------
    header("argmin and argmax: just the indices")
    x = torch.randn(3, 4)
    print("x:\n", x)
    
    # 펼친 뒤 전역 argmin/argmax 찾기
    print("argmin (global):", x.argmin().item())
    print("argmax (global):", x.argmax().item())
    
    # 차원별
    print("argmin(dim=1):", x.argmin(dim=1))
    print("argmax(dim=1):", x.argmax(dim=1))

    # -------------------------------------------------------------------------
    header("aminmax: min and max together")
    x = torch.randn(3, 4)
    min_val, max_val = x.aminmax()
    print("aminmax:", min_val.item(), max_val.item())
    
    # 차원별
    min_vals, max_vals = x.aminmax(dim=1)
    print("aminmax(dim=1):")
    print("  mins:", min_vals)
    print("  maxs:", max_vals)

    # -------------------------------------------------------------------------
    header("Quantiles and percentiles")
    x = torch.randn(1000)
    
    # 중앙값(50번째 백분위수)
    median = x.median()
    print("median:", median.item())
    
    # 특정 분위수
    q25 = x.quantile(0.25)
    q75 = x.quantile(0.75)
    print("25th percentile:", q25.item())
    print("75th percentile:", q75.item())
    print("IQR:", (q75 - q25).item())
    
    # 여러 분위수를 한 번에
    quantiles = x.quantile(torch.tensor([0.1, 0.5, 0.9]))
    print("10th, 50th, 90th percentiles:", quantiles)

    # -------------------------------------------------------------------------
    header("Norms and distances")
    x = torch.tensor([3., 4.])
    
    # L2 노름(유클리드)
    l2 = x.norm(p=2)
    print("L2 norm:", l2.item(), "(expect 5.0)")
    
    # L1 노름(맨해튼)
    l1 = x.norm(p=1)
    print("L1 norm:", l1.item())
    
    # 무한대 노름(절댓값의 최댓값)
    linf = x.norm(p=float('inf'))
    print("L∞ norm:", linf.item())
    
    # 행렬 노름
    A = torch.randn(3, 4)
    frobenius = A.norm(p='fro')
    print("Frobenius norm:", frobenius.item())
    
    # 두 텐서 사이의 거리
    y = torch.tensor([6., 8.])
    dist = torch.dist(x, y, p=2)  # L2 distance
    print("Distance between vectors:", dist.item())

    # -------------------------------------------------------------------------
    header("Logical reductions: all, any")
    x = torch.tensor([[True, True, True],
                      [True, False, True],
                      [False, False, False]])
    print("x:\n", x)
    
    print("all() (all True):", x.all().item())
    print("any() (any True):", x.any().item())
    
    # 차원별
    print("all(dim=1):", x.all(dim=1))
    print("any(dim=1):", x.any(dim=1))
    
    # 실용적 용도: 조건 확인
    values = torch.randn(5)
    print("\nvalues:", values)
    all_positive = (values > 0).all()
    any_positive = (values > 0).any()
    print("All positive:", all_positive.item())
    print("Any positive:", any_positive.item())

    # -------------------------------------------------------------------------
    header("Counting operations")
    x = torch.tensor([[1, 0, 3], [0, 0, 6], [7, 8, 0]])
    print("x:\n", x)
    
    # 전체 원소 개수
    print("numel (total elements):", x.numel())
    
    # 0이 아닌 원소 세기
    print("count_nonzero:", torch.count_nonzero(x).item())
    
    # 차원별로 세기
    print("count_nonzero(dim=0):", torch.count_nonzero(x, dim=0))
    print("count_nonzero(dim=1):", torch.count_nonzero(x, dim=1))

    # -------------------------------------------------------------------------
    header("Cumulative operations")
    x = torch.tensor([1., 2., 3., 4., 5.])
    print("x:", x)
    
    # 누적 합
    cumsum = x.cumsum(dim=0)
    print("cumsum:", cumsum)  # [1, 3, 6, 10, 15]
    
    # 누적 곱
    cumprod = x.cumprod(dim=0)
    print("cumprod:", cumprod)  # [1, 2, 6, 24, 120]
    
    # 2차원 예제
    mat = torch.tensor([[1., 2., 3.],
                        [4., 5., 6.]])
    print("\nmat:\n", mat)
    print("cumsum(dim=0):\n", mat.cumsum(dim=0))
    print("cumsum(dim=1):\n", mat.cumsum(dim=1))

    # -------------------------------------------------------------------------
    header("Mode and unique values")
    x = torch.tensor([3, 1, 4, 1, 5, 9, 2, 6, 5])
    print("x:", x)
    
    # 가장 흔한 값(최빈값)
    mode_val, mode_idx = x.mode()
    print("mode value:", mode_val.item())
    print("mode index:", mode_idx.item())
    
    # 서로 다른 값들
    unique = torch.unique(x)
    print("unique values:", unique)
    
    # 서로 다른 값과 그 개수
    unique_vals, counts = torch.unique(x, return_counts=True)
    print("unique with counts:")
    for val, count in zip(unique_vals, counts):
        print(f"  {val.item()}: {count.item()} times")

    # -------------------------------------------------------------------------
    header("Batch statistics example: normalizing batches")
    # 이미지 배치: (배치, 채널, 높이, 너비)
    batch = torch.randn(4, 3, 32, 32)
    print("Batch shape:", batch.shape)
    
    # 채널별 평균과 표준편차 계산(배치와 공간 차원에 걸쳐)
    # 축약할 차원: 0(배치), 2(높이), 3(너비)
    mean = batch.mean(dim=(0, 2, 3), keepdim=True)
    std = batch.std(dim=(0, 2, 3), keepdim=True)
    print("Mean shape:", mean.shape)  # (1, 3, 1, 1)
    print("Std shape:", std.shape)    # (1, 3, 1, 1)
    
    # 정규화
    normalized = (batch - mean) / (std + 1e-5)
    print("Normalized shape:", normalized.shape)
    print("Normalized mean ≈ 0:", normalized.mean(dim=(0, 2, 3)))
    print("Normalized std ≈ 1:", normalized.std(dim=(0, 2, 3)))

    # -------------------------------------------------------------------------
    header("Statistical summary function")
    def summarize(tensor, name="tensor"):
        """Print statistical summary of a tensor."""
        print(f"\n{name}:")
        print(f"  Shape: {tensor.shape}")
        print(f"  dtype: {tensor.dtype}")
        print(f"  Min: {tensor.min().item():.4f}")
        print(f"  Max: {tensor.max().item():.4f}")
        print(f"  Mean: {tensor.mean().item():.4f}")
        print(f"  Std: {tensor.std().item():.4f}")
        print(f"  Median: {tensor.median().item():.4f}")
    
    x = torch.randn(100, 50)
    summarize(x, "Random matrix")

    # -------------------------------------------------------------------------
    header("Quick reference: reduction operations")
    print("\nBasic statistics:")
    print("  .sum()     - Sum of all elements")
    print("  .mean()    - Average of all elements")
    print("  .std()     - Standard deviation")
    print("  .var()     - Variance")
    print("  .prod()    - Product of all elements")
    
    print("\nExtrema:")
    print("  .min()     - Minimum value (+ index if dim specified)")
    print("  .max()     - Maximum value (+ index if dim specified)")
    print("  .argmin()  - Index of minimum")
    print("  .argmax()  - Index of maximum")
    print("  .aminmax() - Min and max together")
    
    print("\nDistribution:")
    print("  .median()  - Median value")
    print("  .quantile(q) - q-th quantile")
    print("  .mode()    - Most frequent value")
    
    print("\nNorms:")
    print("  .norm(p)   - p-norm (p=1, 2, inf, 'fro')")
    print("  torch.dist(a, b, p) - Distance between tensors")
    
    print("\nLogical:")
    print("  .all()     - True if all elements True")
    print("  .any()     - True if any element True")
    print("  torch.count_nonzero() - Count non-zero elements")
    
    print("\nCumulative:")
    print("  .cumsum(dim) - Cumulative sum")
    print("  .cumprod(dim) - Cumulative product")
    
    print("\nNote: Most operations support dim and keepdim parameters")

if __name__ == "__main__":
    main()```

## 논의

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
