#!/usr/bin/env python3
"""
Advanced indexing, slicing, and masking operations.

Covers:
- Basic slicing vs advanced indexing
- Boolean masking (conditional selection)
- Integer array indexing (fancy indexing)
- torch.where for conditional operations
- masked_fill, masked_select, masked_scatter
- Ellipsis (...) for flexible indexing
- Difference between views and copies
"""

import torch

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
    
    # Modifying view affects original
    view[0, 0] = 999
    print("After view[0,0]=999, a[1,2]:", a[1, 2].item())

    # -------------------------------------------------------------------------
    header("Boolean masking (fancy indexing - creates copies)")
    a = torch.tensor([1, -2, 3, -4, 5, -6])
    print("a:", a)
    
    # Create boolean mask
    mask = a > 0
    print("mask (a > 0):", mask)
    
    # Index with mask (returns 1D tensor, COPY not view)
    positive = a[mask]
    print("a[mask] (positives):", positive)
    
    # Modifying doesn't affect original (it's a copy)
    positive[0] = 100
    print("After positive[0]=100, a:", a)

    # -------------------------------------------------------------------------
    header("Boolean mask assignment (in-place modification)")
    a = torch.tensor([1, -2, 3, -4, 5, -6])
    print("Before: a:", a)
    
    a[a < 0] = 0  # Set all negative values to 0
    print("After a[a < 0] = 0:", a)
    
    # Multiple conditions with & (and) and | (or)
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
    
    # Select from a where a>0, otherwise from b
    result = torch.where(a > 0, a, b)
    print("where(a > 0, a, b):", result)
    
    # Can also use scalars
    clamped = torch.where(a > 0, a, torch.tensor(0.0))
    print("Clamp negatives to 0:", clamped)

    # -------------------------------------------------------------------------
    header("Integer array indexing (fancy indexing)")
    a = torch.arange(12).reshape(3, 4)
    print("a:\n", a)
    
    # Index with integer tensors
    row_idx = torch.tensor([0, 2, 1])
    col_idx = torch.tensor([1, 3, 2])
    
    # This selects a[0,1], a[2,3], a[1,2]
    selected = a[row_idx, col_idx]
    print("a[row_idx, col_idx]:", selected)
    
    # Can use for gathering specific elements
    rows = torch.tensor([0, 0, 1, 1, 2, 2])
    cols = torch.tensor([0, 1, 2, 3, 0, 1])
    gathered = a[rows, cols]
    print("Gathered elements:", gathered)

    # -------------------------------------------------------------------------
    header("Advanced indexing with broadcasting")
    a = torch.arange(12).reshape(3, 4)
    print("a:\n", a)
    
    # Select entire rows
    row_indices = torch.tensor([0, 2])
    selected_rows = a[row_indices]  # Shape: (2, 4)
    print("a[row_indices]:\n", selected_rows)
    
    # Select entire columns (need : for first dimension)
    col_indices = torch.tensor([1, 3])
    selected_cols = a[:, col_indices]  # Shape: (3, 2)
    print("a[:, col_indices]:\n", selected_cols)
    
    # 2D indexing with broadcasting
    row_idx = torch.tensor([[0], [1], [2]])  # (3, 1)
    col_idx = torch.tensor([[0, 2]])          # (1, 2)
    # Broadcasts to select (0,0), (0,2), (1,0), (1,2), (2,0), (2,2)
    subgrid = a[row_idx, col_idx]  # Shape: (3, 2)
    print("Broadcasted 2D indexing:\n", subgrid)

    # -------------------------------------------------------------------------
    header("masked_fill: fill values based on mask")
    a = torch.randn(3, 4)
    print("a:\n", a)
    
    mask = a > 0
    filled = a.masked_fill(mask, value=-999)
    print("masked_fill(a > 0, -999):\n", filled)
    
    # In-place version
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
    
    # ... expands to : as many times as needed
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
    
    # First slice, then mask
    submatrix = a[1:3, :]      # Rows 1-2, all columns
    positive_in_sub = submatrix[submatrix > 0]
    print("Positive values in rows 1-2:", positive_in_sub)
    
    # Mask specific rows
    row_mask = torch.tensor([True, False, True, False])
    masked_rows = a[row_mask]
    print("Masked rows (0 and 2):\n", masked_rows)

    # -------------------------------------------------------------------------
    header("nonzero and where for finding indices")
    a = torch.tensor([[0, 1, 0], [2, 0, 3], [0, 4, 0]])
    print("a:\n", a)
    
    # Get indices of non-zero elements
    indices = a.nonzero()
    print("nonzero() indices:\n", indices)
    
    # Alternative: torch.where returns tuple of indices
    row_idx, col_idx = torch.where(a > 0)
    print("where(a > 0) row indices:", row_idx)
    print("where(a > 0) col indices:", col_idx)
    
    # Use indices to access elements
    values = a[row_idx, col_idx]
    print("Values at those indices:", values)

    # -------------------------------------------------------------------------
    header("View vs copy: when does indexing copy?")
    a = torch.arange(12).reshape(3, 4)
    
    # Basic slicing → VIEW
    view = a[1:3, 2:4]
    print("Basic slice is view:", id(a.storage()) == id(view.storage()))
    
    # Boolean indexing → COPY
    mask = a > 5
    copy1 = a[mask]
    print("Boolean indexing is copy:", id(a.storage()) != id(copy1.storage()))
    
    # Integer array indexing → COPY
    indices = torch.tensor([0, 2])
    copy2 = a[indices]
    print("Integer array indexing is copy:", id(a.storage()) != id(copy2.storage()))
    
    # Step slicing → VIEW
    view2 = a[::2, ::2]
    print("Step slicing is view:", id(a.storage()) == id(view2.storage()))

    # -------------------------------------------------------------------------
    header("Practical example: attention masking")
    # Simulate attention scores (batch=2, heads=1, seq_len=4)
    scores = torch.randn(2, 1, 4, 4)
    print("Attention scores shape:", scores.shape)
    
    # Create causal mask (lower triangular)
    mask = torch.triu(torch.ones(4, 4), diagonal=1).bool()
    print("Causal mask:\n", mask)
    
    # Apply mask (set future positions to -inf)
    scores.masked_fill_(mask, float('-inf'))
    print("Masked scores (first batch):\n", scores[0, 0])

    # -------------------------------------------------------------------------
    header("Practical example: data filtering")
    # Dataset with features and labels
    data = torch.randn(100, 5)      # 100 samples, 5 features
    labels = torch.randint(0, 3, (100,))  # 3 classes
    
    # Select only samples from class 1
    class_1_mask = labels == 1
    class_1_data = data[class_1_mask]
    print(f"Total samples: {len(data)}, Class 1 samples: {len(class_1_data)}")
    
    # Select samples where first feature > 0
    feature_mask = data[:, 0] > 0
    filtered_data = data[feature_mask]
    print(f"Samples with feature[0] > 0: {len(filtered_data)}")

if __name__ == "__main__":
    main()
