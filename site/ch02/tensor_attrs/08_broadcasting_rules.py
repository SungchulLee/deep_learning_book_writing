#!/usr/bin/env python3
"""
Broadcasting rules: making tensors compatible for element-wise operations.

Covers:
- The three broadcasting rules
- Common broadcasting patterns
- Dimension alignment (trailing dimensions)
- Size-1 dimension expansion
- Visual examples with shapes
- Common pitfalls and debugging
"""

import torch

def header(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

def show_broadcast(a, b, op_name="+"):
    """Helper to visualize broadcasting."""
    result = a + b
    print(f"  {a.shape} {op_name} {b.shape} → {result.shape}")
    return result

def main():
    torch.manual_seed(0)

    # -------------------------------------------------------------------------
    header("Broadcasting Rule 1: Align from trailing (rightmost) dimensions")
    a = torch.randn(3, 4, 5)  # Shape: (3, 4, 5)
    b = torch.randn(5)        # Shape: (5,)
    # Alignment:
    #   a: (3, 4, 5)
    #   b:    (  5)  ← implicitly (1, 1, 5)
    result = show_broadcast(a, b)

    # -------------------------------------------------------------------------
    header("Broadcasting Rule 2: Prepend 1s to shorter tensor")
    a = torch.randn(3, 4, 5)  # Shape: (3, 4, 5)
    b = torch.randn(4, 5)     # Shape: (4, 5)
    # After prepending 1s:
    #   a: (3, 4, 5)
    #   b: (1, 4, 5)  ← prepended with 1
    result = show_broadcast(a, b)

    # -------------------------------------------------------------------------
    header("Broadcasting Rule 3: Size-1 dimensions expand to match")
    a = torch.randn(3, 1, 5)  # Shape: (3, 1, 5)
    b = torch.randn(1, 4, 5)  # Shape: (1, 4, 5)
    # Broadcasting:
    #   a: (3, 1, 5) → (3, 4, 5)
    #   b: (1, 4, 5) → (3, 4, 5)
    result = show_broadcast(a, b)
    print(f"  Both broadcast to {result.shape}")

    # -------------------------------------------------------------------------
    header("Scalar broadcasting (0-D tensor)")
    a = torch.randn(3, 4)
    scalar = torch.tensor(5.0)  # Shape: ()
    # Scalar broadcasts to any shape
    result = show_broadcast(a, scalar, "*")

    # -------------------------------------------------------------------------
    header("Common pattern: (batch, features) + (features,)")
    batch_data = torch.randn(32, 128)  # 32 samples, 128 features
    bias = torch.randn(128)             # Per-feature bias
    # Broadcasting: (32, 128) + (128,) → (32, 128)
    result = show_broadcast(batch_data, bias)
    print("  Common in neural networks: adding bias to batched data")

    # -------------------------------------------------------------------------
    header("Common pattern: (batch, channels, H, W) + (channels, 1, 1)")
    images = torch.randn(8, 3, 64, 64)  # Batch of RGB images
    channel_scale = torch.randn(3, 1, 1)  # Per-channel scaling
    # Broadcasting: (8, 3, 64, 64) + (3, 1, 1) → (8, 3, 64, 64)
    result = show_broadcast(images, channel_scale, "*")
    print("  Common in CNNs: per-channel operations")

    # -------------------------------------------------------------------------
    header("Matrix-vector broadcasting")
    matrix = torch.randn(5, 4)  # Shape: (5, 4)
    vector = torch.randn(4)     # Shape: (4,)
    # Row-wise operation: (5, 4) + (4,) → (5, 4)
    row_result = matrix + vector
    print(f"  Row-wise: {matrix.shape} + {vector.shape} → {row_result.shape}")
    
    # For column-wise, need to reshape vector
    col_vector = torch.randn(5, 1)  # Shape: (5, 1)
    col_result = matrix + col_vector
    print(f"  Col-wise: {matrix.shape} + {col_vector.shape} → {col_result.shape}")

    # -------------------------------------------------------------------------
    header("Outer product via broadcasting")
    a = torch.tensor([1., 2., 3.]).view(-1, 1)  # (3, 1)
    b = torch.tensor([10., 20., 30., 40.]).view(1, -1)  # (1, 4)
    # Broadcasting: (3, 1) * (1, 4) → (3, 4)
    outer = a * b
    print(f"  {a.shape} * {b.shape} → {outer.shape}")
    print("  Outer product:\n", outer)

    # -------------------------------------------------------------------------
    header("Broadcasting incompatibility (will fail)")
    a = torch.randn(3, 4)
    b = torch.randn(5)
    try:
        result = a + b  # Incompatible: (3, 4) vs (5,)
    except RuntimeError as e:
        print(f"  ERROR (expected): {str(e)[:60]}...")
        print(f"  (3, 4) vs (5,) → dimension 1: 4 ≠ 5")

    # -------------------------------------------------------------------------
    header("Broadcasting with unsqueeze to add size-1 dimensions")
    a = torch.randn(3, 4)     # (3, 4)
    b = torch.randn(4)        # (4,)
    
    # Want to broadcast along dim 0 instead of dim 1
    b_col = b.unsqueeze(0)    # (1, 4)
    result = a + b_col
    print(f"  {a.shape} + {b_col.shape} → {result.shape}")
    
    # Or broadcast as column
    b_col2 = b.unsqueeze(1)   # (4, 1)
    try:
        result = a + b_col2   # (3, 4) vs (4, 1) → incompatible!
    except RuntimeError as e:
        print(f"  ERROR: (3, 4) + (4, 1) incompatible")
    
    # Need to reshape properly for column broadcast
    c = torch.randn(3)
    c_col = c.unsqueeze(1)    # (3, 1)
    result = a + c_col        # (3, 4) + (3, 1) → (3, 4)
    print(f"  {a.shape} + {c_col.shape} → {result.shape} ✓")

    # -------------------------------------------------------------------------
    header("Explicit broadcast_to for clarity")
    a = torch.tensor([1., 2., 3.])  # (3,)
    target_shape = (4, 3)
    
    # Automatic broadcasting
    b = torch.zeros(4, 3)
    result = b + a  # (4, 3) + (3,) → (4, 3)
    print(f"  Automatic: {b.shape} + {a.shape} → {result.shape}")
    
    # Explicit broadcast_to (returns a view)
    a_broadcast = a.broadcast_to(target_shape)
    print(f"  Explicit: broadcast_to({target_shape}) → {a_broadcast.shape}")
    print(f"  Is view (shares storage): {id(a.storage()) == id(a_broadcast.storage())}")

    # -------------------------------------------------------------------------
    header("Common bug: unintended broadcasting with missing dimensions")
    # Suppose we have batch of images and want per-image mean
    images = torch.randn(10, 3, 32, 32)  # 10 RGB images, 32x32
    
    # WRONG: compute mean over spatial dims but forget to keepdim
    wrong_mean = images.mean(dim=(2, 3))  # Shape: (10, 3)
    print(f"  Wrong mean shape: {wrong_mean.shape}")
    try:
        normalized = images - wrong_mean  # (10,3,32,32) - (10,3) → broadcasts incorrectly!
        print(f"  Result shape: {normalized.shape}")
        print("  ⚠️  This broadcasts but NOT as intended!")
    except:
        pass
    
    # CORRECT: use keepdim=True
    correct_mean = images.mean(dim=(2, 3), keepdim=True)  # Shape: (10, 3, 1, 1)
    print(f"  Correct mean shape: {correct_mean.shape}")
    normalized = images - correct_mean  # (10,3,32,32) - (10,3,1,1) ✓
    print(f"  Correct result shape: {normalized.shape}")

    # -------------------------------------------------------------------------
    header("Broadcasting with torch.where (conditional selection)")
    a = torch.randn(3, 4)
    b = torch.randn(4)        # Broadcasts to (3, 4)
    condition = a > 0
    
    result = torch.where(condition, a, b)  # Select a or b based on condition
    print(f"  where: {a.shape}, {b.shape} → {result.shape}")

    # -------------------------------------------------------------------------
    header("Memory efficiency: broadcasting creates views, not copies")
    small = torch.tensor([1., 2., 3.])
    large_shape = (1000, 3)
    
    # Broadcasting doesn't copy data
    broadcasted = small.expand(large_shape)  # Returns a view
    print(f"  Broadcasted shape: {broadcasted.shape}")
    print(f"  Shares storage: {id(small.storage()) == id(broadcasted.storage())}")
    print(f"  Storage size: {small.storage().size()} elements (only original data)")

    # -------------------------------------------------------------------------
    header("Quick reference: broadcasting shape compatibility")
    print("  Examples of compatible shapes:")
    examples = [
        ((5, 1, 7), (3, 7), (5, 3, 7)),
        ((3, 1), (1, 4), (3, 4)),
        ((8, 1, 6, 1), (7, 1, 5), (8, 7, 6, 5)),
        ((5,), (3, 5), (3, 5)),
        ((), (3, 4), (3, 4)),  # Scalar
    ]
    for shape_a, shape_b, result_shape in examples:
        print(f"    {str(shape_a):20s} + {str(shape_b):20s} → {result_shape}")
    
    print("\n  Examples of INCOMPATIBLE shapes:")
    incompatible = [
        ((3, 4), (5,), "dims don't match: 4 ≠ 5"),
        ((2, 3), (3, 2), "both dims non-1: 2≠3 and 3≠2"),
    ]
    for shape_a, shape_b, reason in incompatible:
        print(f"    {str(shape_a):20s} + {str(shape_b):20s} ✗ ({reason})")

if __name__ == "__main__":
    main()
