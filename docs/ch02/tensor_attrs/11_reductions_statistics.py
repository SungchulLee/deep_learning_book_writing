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
    
    # Sum along dimension 0 (collapse rows)
    sum_dim0 = x.sum(dim=0)
    print("sum(dim=0):", sum_dim0)  # [12, 15, 18]
    print("  Result shape:", sum_dim0.shape)  # (3,)
    
    # Sum along dimension 1 (collapse columns)
    sum_dim1 = x.sum(dim=1)
    print("sum(dim=1):", sum_dim1)  # [6, 15, 24]
    print("  Result shape:", sum_dim1.shape)  # (3,)

    # -------------------------------------------------------------------------
    header("keepdim: preserve reduced dimensions as size 1")
    x = torch.randn(3, 4, 5)
    print("x.shape:", x.shape)
    
    # Without keepdim (default)
    mean_no_keep = x.mean(dim=1)
    print("mean(dim=1).shape:", mean_no_keep.shape)  # (3, 5)
    
    # With keepdim
    mean_keep = x.mean(dim=1, keepdim=True)
    print("mean(dim=1, keepdim=True).shape:", mean_keep.shape)  # (3, 1, 5)
    
    # Useful for broadcasting
    normalized = x - mean_keep  # Broadcasts correctly
    print("Normalized shape:", normalized.shape)  # (3, 4, 5)

    # -------------------------------------------------------------------------
    header("Multiple dimension reductions")
    x = torch.randn(2, 3, 4, 5)
    print("x.shape:", x.shape)
    
    # Reduce over multiple dimensions
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
    
    # Unbiased vs biased estimator
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
    
    # Dimension-wise min/max returns values AND indices
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
    
    # Flatten and find global argmin/argmax
    print("argmin (global):", x.argmin().item())
    print("argmax (global):", x.argmax().item())
    
    # Dimension-wise
    print("argmin(dim=1):", x.argmin(dim=1))
    print("argmax(dim=1):", x.argmax(dim=1))

    # -------------------------------------------------------------------------
    header("aminmax: min and max together")
    x = torch.randn(3, 4)
    min_val, max_val = x.aminmax()
    print("aminmax:", min_val.item(), max_val.item())
    
    # Dimension-wise
    min_vals, max_vals = x.aminmax(dim=1)
    print("aminmax(dim=1):")
    print("  mins:", min_vals)
    print("  maxs:", max_vals)

    # -------------------------------------------------------------------------
    header("Quantiles and percentiles")
    x = torch.randn(1000)
    
    # Median (50th percentile)
    median = x.median()
    print("median:", median.item())
    
    # Specific quantiles
    q25 = x.quantile(0.25)
    q75 = x.quantile(0.75)
    print("25th percentile:", q25.item())
    print("75th percentile:", q75.item())
    print("IQR:", (q75 - q25).item())
    
    # Multiple quantiles at once
    quantiles = x.quantile(torch.tensor([0.1, 0.5, 0.9]))
    print("10th, 50th, 90th percentiles:", quantiles)

    # -------------------------------------------------------------------------
    header("Norms and distances")
    x = torch.tensor([3., 4.])
    
    # L2 norm (Euclidean)
    l2 = x.norm(p=2)
    print("L2 norm:", l2.item(), "(expect 5.0)")
    
    # L1 norm (Manhattan)
    l1 = x.norm(p=1)
    print("L1 norm:", l1.item())
    
    # Infinity norm (max absolute value)
    linf = x.norm(p=float('inf'))
    print("L∞ norm:", linf.item())
    
    # Matrix norms
    A = torch.randn(3, 4)
    frobenius = A.norm(p='fro')
    print("Frobenius norm:", frobenius.item())
    
    # Distance between two tensors
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
    
    # Dimension-wise
    print("all(dim=1):", x.all(dim=1))
    print("any(dim=1):", x.any(dim=1))
    
    # Practical use: checking conditions
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
    
    # Total number of elements
    print("numel (total elements):", x.numel())
    
    # Count non-zero elements
    print("count_nonzero:", torch.count_nonzero(x).item())
    
    # Count per dimension
    print("count_nonzero(dim=0):", torch.count_nonzero(x, dim=0))
    print("count_nonzero(dim=1):", torch.count_nonzero(x, dim=1))

    # -------------------------------------------------------------------------
    header("Cumulative operations")
    x = torch.tensor([1., 2., 3., 4., 5.])
    print("x:", x)
    
    # Cumulative sum
    cumsum = x.cumsum(dim=0)
    print("cumsum:", cumsum)  # [1, 3, 6, 10, 15]
    
    # Cumulative product
    cumprod = x.cumprod(dim=0)
    print("cumprod:", cumprod)  # [1, 2, 6, 24, 120]
    
    # 2D example
    mat = torch.tensor([[1., 2., 3.],
                        [4., 5., 6.]])
    print("\nmat:\n", mat)
    print("cumsum(dim=0):\n", mat.cumsum(dim=0))
    print("cumsum(dim=1):\n", mat.cumsum(dim=1))

    # -------------------------------------------------------------------------
    header("Mode and unique values")
    x = torch.tensor([3, 1, 4, 1, 5, 9, 2, 6, 5])
    print("x:", x)
    
    # Most common value (mode)
    mode_val, mode_idx = x.mode()
    print("mode value:", mode_val.item())
    print("mode index:", mode_idx.item())
    
    # Unique values
    unique = torch.unique(x)
    print("unique values:", unique)
    
    # Unique with counts
    unique_vals, counts = torch.unique(x, return_counts=True)
    print("unique with counts:")
    for val, count in zip(unique_vals, counts):
        print(f"  {val.item()}: {count.item()} times")

    # -------------------------------------------------------------------------
    header("Batch statistics example: normalizing batches")
    # Batch of images: (batch, channels, height, width)
    batch = torch.randn(4, 3, 32, 32)
    print("Batch shape:", batch.shape)
    
    # Compute mean and std per channel (across batch and spatial dims)
    # Dims to reduce: 0 (batch), 2 (height), 3 (width)
    mean = batch.mean(dim=(0, 2, 3), keepdim=True)
    std = batch.std(dim=(0, 2, 3), keepdim=True)
    print("Mean shape:", mean.shape)  # (1, 3, 1, 1)
    print("Std shape:", std.shape)    # (1, 3, 1, 1)
    
    # Normalize
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
    main()
