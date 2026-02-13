"""
Tutorial 11: Broadcasting in PyTorch
=====================================

Broadcasting allows PyTorch to perform operations on tensors of different shapes
by automatically expanding them to compatible shapes without copying data.

Key Concepts:
- Broadcasting rules
- Common broadcasting patterns
- When broadcasting fails
- Memory efficiency
- Broadcasting in neural networks
"""

import torch


def header(title: str):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def main():
    # -------------------------------------------------------------------------
    # 1. What is Broadcasting?
    # -------------------------------------------------------------------------
    header("1. What is Broadcasting?")
    
    print("""
    Broadcasting is a powerful mechanism that allows PyTorch to work with
    tensors of different shapes when performing element-wise operations.
    
    Instead of manually expanding tensors to the same size, PyTorch does it
    automatically (virtually, without actually copying data).
    """)
    
    # Simple example: Adding scalar to tensor
    vec = torch.tensor([1, 2, 3, 4])
    scalar = 10
    
    print(f"vec = {vec}")
    print(f"scalar = {scalar}")
    print(f"vec + scalar = {vec + scalar}")  # Scalar broadcasts to [10, 10, 10, 10]
    
    # -------------------------------------------------------------------------
    # 2. Broadcasting Rules
    # -------------------------------------------------------------------------
    header("2. Broadcasting Rules")
    
    print("""
    Two tensors are "broadcastable" if:
    
    1. Each tensor has at least one dimension, OR one tensor is a scalar
    2. When iterating from the LAST dimension backwards:
       - Dimensions are equal, OR
       - One dimension is 1, OR  
       - One dimension doesn't exist
    
    Examples:
    """)
    
    # Rule examples
    examples = [
        ("(3, 1) with (1, 4)", (3, 1), (1, 4), "(3, 4)"),
        ("(3, 4) with (3, 1)", (3, 4), (3, 1), "(3, 4)"),
        ("(1, 4, 5) with (3, 1, 5)", (1, 4, 5), (3, 1, 5), "(3, 4, 5)"),
        ("(3,) with (4, 3)", (3,), (4, 3), "(4, 3)"),
    ]
    
    for desc, shape1, shape2, result in examples:
        t1 = torch.randn(shape1)
        t2 = torch.randn(shape2)
        result_tensor = t1 + t2
        print(f"  {desc}: → {result_tensor.shape}")
    
    # -------------------------------------------------------------------------
    # 3. Basic Broadcasting Patterns
    # -------------------------------------------------------------------------
    header("3. Basic Broadcasting Patterns")
    
    # Pattern 1: Scalar with any tensor
    matrix = torch.arange(6).reshape(2, 3)
    scalar_val = 100
    print("Pattern 1: Scalar with matrix")
    print(f"Matrix:\n{matrix}")
    print(f"Matrix + 100:\n{matrix + scalar_val}\n")
    
    # Pattern 2: Vector with matrix (same last dimension)
    vec = torch.tensor([10, 20, 30])
    print("Pattern 2: Row vector with matrix")
    print(f"vec = {vec}")
    result = matrix + vec  # vec broadcasts to (1, 3) then to (2, 3)
    print(f"matrix + vec:\n{result}\n")
    
    # Pattern 3: Column vector with matrix
    col_vec = torch.tensor([[10], [20]])  # Shape (2, 1)
    print("Pattern 3: Column vector with matrix")
    print(f"col_vec:\n{col_vec}")
    result = matrix + col_vec  # col_vec broadcasts to (2, 3)
    print(f"matrix + col_vec:\n{result}\n")
    
    # -------------------------------------------------------------------------
    # 4. Visualizing Broadcasting
    # -------------------------------------------------------------------------
    header("4. Visualizing Broadcasting")
    
    # Create simple tensors for visualization
    A = torch.arange(3).reshape(3, 1)  # Column vector
    B = torch.arange(4).reshape(1, 4)  # Row vector
    
    print("A (3x1):\n", A)
    print("B (1x4):\n", B)
    
    # Broadcasting creates a 3x4 result
    C = A + B
    print(f"\nA + B (broadcasts to 3x4):\n{C}")
    print("""
    How it works:
    - A broadcasts: [[0],     →  [[0, 0, 0, 0],
                     [1],          [1, 1, 1, 1],
                     [2]]          [2, 2, 2, 2]]
    
    - B broadcasts: [[0, 1, 2, 3]] → [[0, 1, 2, 3],
                                      [0, 1, 2, 3],
                                      [0, 1, 2, 3]]
    """)
    
    # -------------------------------------------------------------------------
    # 5. Common Use Cases in Machine Learning
    # -------------------------------------------------------------------------
    header("5. Common Use Cases in ML")
    
    # Use case 1: Adding bias to layer output
    print("Use case 1: Adding bias")
    batch_size, features = 32, 10
    layer_output = torch.randn(batch_size, features)
    bias = torch.randn(features)  # Shape (10,)
    
    output_with_bias = layer_output + bias  # bias broadcasts to (32, 10)
    print(f"Layer output: {layer_output.shape}")
    print(f"Bias: {bias.shape}")
    print(f"Result: {output_with_bias.shape}\n")
    
    # Use case 2: Normalizing data
    print("Use case 2: Data normalization")
    data = torch.randn(100, 5)  # 100 samples, 5 features
    mean = data.mean(dim=0, keepdim=True)  # Shape (1, 5)
    std = data.std(dim=0, keepdim=True)    # Shape (1, 5)
    
    normalized = (data - mean) / std  # Broadcasting!
    print(f"Data: {data.shape}")
    print(f"Mean: {mean.shape}")
    print(f"Normalized: {normalized.shape}\n")
    
    # Use case 3: Pairwise distances
    print("Use case 3: Pairwise distances")
    points_a = torch.randn(5, 2)  # 5 points in 2D
    points_b = torch.randn(3, 2)  # 3 points in 2D
    
    # Compute all pairwise distances
    # Reshape for broadcasting: (5, 1, 2) - (1, 3, 2) = (5, 3, 2)
    diff = points_a.unsqueeze(1) - points_b.unsqueeze(0)
    distances = torch.sqrt((diff ** 2).sum(dim=2))
    print(f"Points A: {points_a.shape}")
    print(f"Points B: {points_b.shape}")
    print(f"Pairwise distances: {distances.shape}")  # (5, 3)
    
    # -------------------------------------------------------------------------
    # 6. When Broadcasting Fails
    # -------------------------------------------------------------------------
    header("6. When Broadcasting Fails")
    
    print("Broadcasting fails when shapes are incompatible:\n")
    
    # Example 1: Incompatible shapes
    try:
        t1 = torch.randn(3, 4)
        t2 = torch.randn(2, 3)
        result = t1 + t2  # Will fail!
    except RuntimeError as e:
        print(f"Error with (3,4) + (2,3): Shapes incompatible")
        print(f"  Reason: Neither dimension matches (3≠2, 4≠3)\n")
    
    # Example 2: How to fix it
    print("Fix: Reshape or unsqueeze to make compatible")
    t1 = torch.randn(3, 4)
    t2 = torch.randn(3, 1)  # Now compatible!
    result = t1 + t2
    print(f"(3,4) + (3,1) = {result.shape} ✓\n")
    
    # -------------------------------------------------------------------------
    # 7. Keepdim Parameter - Preserving Dimensions
    # -------------------------------------------------------------------------
    header("7. keepdim=True for Broadcasting")
    
    data = torch.arange(12).reshape(3, 4).float()
    print(f"Data:\n{data}\n")
    
    # Without keepdim - dimension is removed
    row_sum = data.sum(dim=1)
    print(f"sum(dim=1): {row_sum}")
    print(f"Shape: {row_sum.shape}")  # (3,) - lost dimension!
    
    # With keepdim - dimension is kept as size 1
    row_sum_keep = data.sum(dim=1, keepdim=True)
    print(f"\nsum(dim=1, keepdim=True):\n{row_sum_keep}")
    print(f"Shape: {row_sum_keep.shape}")  # (3, 1) - can broadcast!
    
    # Now we can broadcast
    normalized_rows = data / row_sum_keep  # Works!
    print(f"\nData / row_sum:\n{normalized_rows}")
    
    # -------------------------------------------------------------------------
    # 8. Memory Efficiency
    # -------------------------------------------------------------------------
    header("8. Memory Efficiency")
    
    print("""
    Key insight: Broadcasting doesn't actually copy data!
    
    When you broadcast a (1, 100) tensor to (1000, 100), PyTorch doesn't
    create 1000 copies. Instead, it uses clever indexing to reuse the same
    data virtually.
    
    This makes broadcasting both FAST and MEMORY-EFFICIENT.
    """)
    
    # Demonstration
    small = torch.tensor([[1, 2, 3]])  # (1, 3)
    large = torch.randn(1000, 3)
    
    result = small + large  # No actual copying happens!
    
    print(f"Small tensor: {small.shape}")
    print(f"Large tensor: {large.shape}")
    print(f"Result: {result.shape}")
    print(f"Memory multiplier: {result.numel() / small.numel()}x")
    print("But only the small tensor is stored once!")
    
    # -------------------------------------------------------------------------
    # 9. Explicit Broadcasting with expand()
    # -------------------------------------------------------------------------
    header("9. Explicit Broadcasting with expand()")
    
    # expand() explicitly broadcasts without copying data
    vec = torch.tensor([1, 2, 3])
    print(f"Original vector: {vec}, shape: {vec.shape}")
    
    # Expand to (4, 3)
    expanded = vec.expand(4, 3)
    print(f"\nExpanded: {expanded.shape}")
    print(f"Values:\n{expanded}")
    
    # Check memory: expand doesn't copy!
    print(f"\nSame storage? {vec.storage().data_ptr() == expanded.storage().data_ptr()}")
    
    # Warning: Modifying expanded tensors can have unexpected effects!
    # Don't modify expanded tensors in-place
    
    # -------------------------------------------------------------------------
    # 10. Advanced: Broadcasting with einsum
    # -------------------------------------------------------------------------
    header("10. Advanced: einsum")
    
    print("""
    einsum (Einstein summation) provides a concise way to express tensor
    operations with explicit control over broadcasting.
    """)
    
    # Matrix-vector multiplication with einsum
    matrix = torch.randn(3, 4)
    vec = torch.randn(4)
    
    # Traditional: matrix @ vec
    result_traditional = matrix @ vec
    
    # With einsum: 'ij,j->i' means (i,j) * (j) -> (i)
    result_einsum = torch.einsum('ij,j->i', matrix, vec)
    
    print(f"Matrix: {matrix.shape}")
    print(f"Vector: {vec.shape}")
    print(f"Result: {result_einsum.shape}")
    print(f"Match? {torch.allclose(result_traditional, result_einsum)}")
    
    # -------------------------------------------------------------------------
    # Practice Exercises
    # -------------------------------------------------------------------------
    header("Practice Exercises")
    
    print("""
    Try these exercises:
    
    1. Add a (5,) tensor to a (3, 5) matrix
    2. Multiply a (4, 1) column vector with a (1, 3) row vector
    3. Normalize a (10, 20) matrix by subtracting column means
    4. Create a (5, 5) distance matrix from 5 points in 1D
    5. Broadcast a (2, 3, 1) tensor with a (1, 1, 4) tensor
    """)
    
    # Solutions
    t1 = torch.randn(5)
    t2 = torch.randn(3, 5)
    ex1 = t1 + t2
    print(f"\n1. {t1.shape} + {t2.shape} = {ex1.shape}")
    
    col = torch.randn(4, 1)
    row = torch.randn(1, 3)
    ex2 = col * row
    print(f"2. {col.shape} * {row.shape} = {ex2.shape}")
    
    data = torch.randn(10, 20)
    col_mean = data.mean(dim=0, keepdim=True)
    ex3 = data - col_mean
    print(f"3. Normalized: {ex3.shape}")
    
    points = torch.randn(5)
    distances = (points.unsqueeze(0) - points.unsqueeze(1)).abs()
    print(f"4. Distance matrix: {distances.shape}")
    
    t_a = torch.randn(2, 3, 1)
    t_b = torch.randn(1, 1, 4)
    ex5 = t_a + t_b
    print(f"5. {t_a.shape} + {t_b.shape} = {ex5.shape}")


if __name__ == "__main__":
    main()
