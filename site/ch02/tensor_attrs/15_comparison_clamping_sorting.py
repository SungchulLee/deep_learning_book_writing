#!/usr/bin/env python3
"""
Comparison, conditional operations, sorting, and selection.

Covers:
- Comparison operators: ==, !=, <, <=, >, >=
- Logical operators: &, |, ~
- torch.where for conditional selection
- Clamping: clamp, clip
- Sorting: sort, argsort
- Top-k selection: topk, kthvalue
- Finding elements: eq, ne, lt, le, gt, ge
- isnan, isinf, isfinite checks
"""

import torch

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
    
    # Combined conditions
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
    
    # Select from a where condition is True, else from b
    result = torch.where(condition, a, b)
    print("condition:", condition)
    print("a:", a)
    print("b:", b)
    print("where(condition, a, b):", result)
    
    # With broadcasting
    x = torch.randn(5)
    result = torch.where(x > 0, x, torch.tensor(0.))
    print("\nx:", x)
    print("ReLU (x if x>0 else 0):", result)

    # -------------------------------------------------------------------------
    header("clamp: limit values to range")
    x = torch.tensor([-2., -1., 0., 1., 2., 3., 4.])
    print("x:", x)
    
    # Clamp to [0, 2]
    clamped = torch.clamp(x, min=0, max=2)
    print("clamp(0, 2):", clamped)
    
    # Only min
    clamped_min = torch.clamp(x, min=0)
    print("clamp(min=0):", clamped_min)
    
    # Only max
    clamped_max = torch.clamp(x, max=2)
    print("clamp(max=2):", clamped_max)
    
    # In-place version
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
    
    # Sort ascending (default)
    sorted_vals, sorted_indices = torch.sort(x)
    print("Sorted values:", sorted_vals)
    print("Sorted indices:", sorted_indices)
    
    # Sort descending
    sorted_desc, indices_desc = torch.sort(x, descending=True)
    print("Sorted descending:", sorted_desc)
    
    # 2D sorting
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
    
    # Use indices to sort
    sorted_x = x[indices]
    print("x[argsort]:", sorted_x)
    
    # Descending
    indices_desc = torch.argsort(x, descending=True)
    print("argsort(descending):", indices_desc)

    # -------------------------------------------------------------------------
    header("topk: k largest (or smallest) elements")
    x = torch.tensor([3, 1, 4, 1, 5, 9, 2, 6, 5])
    print("x:", x)
    
    # Top 3 values
    top_vals, top_indices = torch.topk(x, k=3)
    print("Top 3 values:", top_vals)
    print("Top 3 indices:", top_indices)
    
    # Bottom 3 (smallest)
    bottom_vals, bottom_indices = torch.topk(x, k=3, largest=False)
    print("Bottom 3 values:", bottom_vals)
    print("Bottom 3 indices:", bottom_indices)
    
    # 2D topk
    mat = torch.randint(0, 10, (3, 5))
    print("\nMatrix:\n", mat)
    top_vals, top_indices = torch.topk(mat, k=2, dim=1)
    print("Top 2 per row:\n", top_vals)

    # -------------------------------------------------------------------------
    header("kthvalue: k-th smallest element")
    x = torch.tensor([3, 1, 4, 1, 5, 9, 2, 6, 5])
    print("x:", x)
    
    # 5th smallest (median for 9 elements)
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
    
    # Count special values
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
    
    # Element-wise check
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
    # Temperature data
    temps = torch.tensor([15.5, 18.2, 22.1, 19.8, 25.3, 28.7, 23.4])
    print("Temperatures:", temps)
    
    # Find comfortable range
    comfortable = (temps >= 18) & (temps <= 25)
    print("Comfortable days:", comfortable)
    print("Comfortable temps:", temps[comfortable])
    
    # Count
    print("Number of comfortable days:", comfortable.sum().item())

    # -------------------------------------------------------------------------
    header("Practical: outlier detection")
    data = torch.randn(100)
    
    # Define outliers as values beyond 2 std devs
    mean = data.mean()
    std = data.std()
    
    outliers = (data < mean - 2*std) | (data > mean + 2*std)
    print("Total points:", len(data))
    print("Outliers:", outliers.sum().item())
    print("Outlier percentage:", f"{100*outliers.float().mean():.1f}%")

    # -------------------------------------------------------------------------
    header("Practical: top-k accuracy")
    # Simulated model predictions and targets
    logits = torch.randn(10, 5)  # 10 samples, 5 classes
    targets = torch.randint(0, 5, (10,))
    
    print("Logits shape:", logits.shape)
    print("Targets:", targets)
    
    # Top-1 accuracy
    pred_top1 = logits.argmax(dim=1)
    top1_acc = (pred_top1 == targets).float().mean()
    print("Top-1 accuracy:", f"{100*top1_acc:.1f}%")
    
    # Top-3 accuracy
    _, pred_top3 = logits.topk(k=3, dim=1)
    top3_acc = (pred_top3 == targets.unsqueeze(1)).any(dim=1).float().mean()
    print("Top-3 accuracy:", f"{100*top3_acc:.1f}%")

    # -------------------------------------------------------------------------
    header("Practical: thresholding")
    # Image-like data
    image = torch.rand(5, 5)
    print("Image:\n", image)
    
    # Binary threshold at 0.5
    binary = (image > 0.5).float()
    print("Binary (threshold=0.5):\n", binary)
    
    # Soft threshold (clamp)
    threshold = 0.3
    soft = torch.where(image > threshold, 
                       image - threshold, 
                       torch.zeros_like(image))
    print("Soft threshold (0.3):\n", soft)

    # -------------------------------------------------------------------------
    header("Practical: NaN handling")
    data = torch.tensor([1., 2., float('nan'), 4., float('nan'), 6.])
    print("Data with NaNs:", data)
    
    # Replace NaNs with mean of valid values
    valid_mask = ~torch.isnan(data)
    mean_val = data[valid_mask].mean()
    cleaned = torch.where(torch.isnan(data), mean_val, data)
    print("After replacing NaNs with mean:", cleaned)
    
    # Or remove NaNs
    clean_data = data[~torch.isnan(data)]
    print("After removing NaNs:", clean_data)

    # -------------------------------------------------------------------------
    header("Practical: ranking")
    scores = torch.tensor([85, 92, 78, 95, 88, 92, 70])
    print("Scores:", scores)
    
    # Get ranks (1-indexed, higher score = lower rank number)
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
    main()
