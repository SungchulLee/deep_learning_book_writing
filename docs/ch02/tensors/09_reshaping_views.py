"""
Tutorial 09: Reshaping and Views
=================================

Learn how to change tensor dimensions without copying data (when possible).
Understanding views vs copies is crucial for memory efficiency.

Key Concepts:
- reshape() vs view() vs contiguous()
- Adding/removing dimensions (unsqueeze/squeeze)
- Permuting dimensions (transpose/permute)
- Flattening tensors
- Memory layout and contiguity
"""

import torch


def header(title: str):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def print_tensor_info(tensor, name="Tensor"):
    """Helper to display tensor properties."""
    print(f"{name}:")
    print(f"  Value: {tensor}")
    print(f"  Shape: {tensor.shape}")
    print(f"  Stride: {tensor.stride()}")
    print(f"  Contiguous: {tensor.is_contiguous()}")
    print()


def main():
    # -------------------------------------------------------------------------
    # 1. Basic Reshaping - reshape() and view()
    # -------------------------------------------------------------------------
    header("1. Basic Reshaping - reshape() vs view()")
    
    # Create a 1D tensor
    vec = torch.arange(12)
    print(f"Original 1D tensor: {vec}")
    print(f"Shape: {vec.shape}")  # torch.Size([12])
    
    # reshape() - Safe method that works always
    mat_reshape = vec.reshape(3, 4)
    print(f"\nReshape to (3, 4):\n{mat_reshape}")
    
    # view() - Faster but requires contiguous memory
    mat_view = vec.view(3, 4)
    print(f"\nView as (3, 4):\n{mat_view}")
    
    # Different shapes, same total elements
    cube_reshape = vec.reshape(2, 2, 3)
    print(f"\nReshape to (2, 2, 3):\n{cube_reshape}")
    
    # Key difference: view() fails on non-contiguous tensors
    # reshape() works always (copies if needed)
    
    # -------------------------------------------------------------------------
    # 2. Automatic Size Inference with -1
    # -------------------------------------------------------------------------
    header("2. Automatic Size Inference with -1")
    
    # Use -1 for one dimension - PyTorch infers it automatically
    vec_24 = torch.arange(24)
    
    # Let PyTorch compute the number of rows
    auto_rows = vec_24.reshape(-1, 4)  # -1 means "figure it out" → 6 rows
    print(f"reshape(-1, 4):\n{auto_rows}")
    print(f"Shape: {auto_rows.shape}")  # torch.Size([6, 4])
    
    # Let PyTorch compute the number of columns
    auto_cols = vec_24.reshape(3, -1)  # → 8 columns
    print(f"\nreshape(3, -1):\n{auto_cols}")
    print(f"Shape: {auto_cols.shape}")  # torch.Size([3, 8])
    
    # Can only use -1 once per reshape
    # auto_both = vec_24.reshape(-1, -1)  # ❌ Error: only one dimension can be inferred
    
    # -------------------------------------------------------------------------
    # 3. Flatten - Convert to 1D
    # -------------------------------------------------------------------------
    header("3. Flatten - Convert to 1D")
    
    mat_3d = torch.arange(24).reshape(2, 3, 4)
    print(f"3D tensor shape: {mat_3d.shape}")
    
    # flatten() - Flattens specified dimensions
    flat_all = mat_3d.flatten()  # Flatten all dimensions
    print(f"flatten(): {flat_all}")
    print(f"Shape: {flat_all.shape}")  # torch.Size([24])
    
    # Flatten specific dimensions
    flat_partial = mat_3d.flatten(start_dim=1)  # Keep dim 0, flatten rest
    print(f"\nflatten(start_dim=1) shape: {flat_partial.shape}")  # torch.Size([2, 12])
    print(f"Values:\n{flat_partial}")
    
    # Alternative: reshape to -1
    flat_reshape = mat_3d.reshape(-1)
    print(f"\nreshape(-1): {flat_reshape}")
    
    # Or view(-1) if contiguous
    flat_view = mat_3d.view(-1)
    print(f"view(-1): {flat_view}")
    
    # -------------------------------------------------------------------------
    # 4. Adding Dimensions - unsqueeze()
    # -------------------------------------------------------------------------
    header("4. Adding Dimensions - unsqueeze()")
    
    vec = torch.tensor([1, 2, 3, 4, 5])
    print(f"Original vector: {vec}")
    print(f"Shape: {vec.shape}")  # torch.Size([5])
    
    # Add dimension at position 0 (makes it a row vector/matrix)
    vec_row = vec.unsqueeze(0)
    print(f"\nunsqueeze(0) - Row vector:\n{vec_row}")
    print(f"Shape: {vec_row.shape}")  # torch.Size([1, 5])
    
    # Add dimension at position 1 (makes it a column vector/matrix)
    vec_col = vec.unsqueeze(1)
    print(f"\nunsqueeze(1) - Column vector:\n{vec_col}")
    print(f"Shape: {vec_col.shape}")  # torch.Size([5, 1])
    
    # Add dimension at position -1 (end)
    vec_end = vec.unsqueeze(-1)
    print(f"\nunsqueeze(-1):\n{vec_end}")
    print(f"Shape: {vec_end.shape}")  # torch.Size([5, 1])
    
    # Multiple unsqueezes
    vec_3d = vec.unsqueeze(0).unsqueeze(2)  # Shape: [1, 5, 1]
    print(f"\nDouble unsqueeze shape: {vec_3d.shape}")
    
    # Alternative: indexing with None
    vec_row_alt = vec[None, :]  # Equivalent to unsqueeze(0)
    vec_col_alt = vec[:, None]  # Equivalent to unsqueeze(1)
    print(f"vec[None, :] shape: {vec_row_alt.shape}")  # torch.Size([1, 5])
    print(f"vec[:, None] shape: {vec_col_alt.shape}")  # torch.Size([5, 1])
    
    # -------------------------------------------------------------------------
    # 5. Removing Dimensions - squeeze()
    # -------------------------------------------------------------------------
    header("5. Removing Dimensions - squeeze()")
    
    # Create tensor with singleton dimensions
    tensor_with_ones = torch.randn(1, 5, 1, 3, 1)
    print(f"Original shape: {tensor_with_ones.shape}")  # torch.Size([1, 5, 1, 3, 1])
    
    # squeeze() - Remove ALL dimensions of size 1
    squeezed_all = tensor_with_ones.squeeze()
    print(f"squeeze() shape: {squeezed_all.shape}")  # torch.Size([5, 3])
    
    # squeeze(dim) - Remove specific dimension (only if size 1)
    squeezed_dim0 = tensor_with_ones.squeeze(0)  # Remove first dim
    print(f"squeeze(0) shape: {squeezed_dim0.shape}")  # torch.Size([5, 1, 3, 1])
    
    squeezed_dim2 = tensor_with_ones.squeeze(2)  # Remove third dim
    print(f"squeeze(2) shape: {squeezed_dim2.shape}")  # torch.Size([1, 5, 3, 1])
    
    # Trying to squeeze a dimension that's not size 1 does nothing
    squeezed_dim1 = tensor_with_ones.squeeze(1)  # Dim 1 is size 5
    print(f"squeeze(1) shape: {squeezed_dim1.shape}")  # torch.Size([1, 5, 1, 3, 1]) - unchanged
    
    # -------------------------------------------------------------------------
    # 6. Transpose - Swap Dimensions
    # -------------------------------------------------------------------------
    header("6. Transpose - Swap Dimensions")
    
    mat = torch.arange(12).reshape(3, 4)
    print(f"Original matrix (3x4):\n{mat}")
    
    # transpose() - Swap two dimensions
    mat_T = mat.transpose(0, 1)  # Swap dimensions 0 and 1
    print(f"\ntranspose(0, 1) - Now (4x3):\n{mat_T}")
    
    # .T property - Shorthand for 2D transpose
    mat_T_short = mat.T
    print(f"\nmat.T (same as transpose):\n{mat_T_short}")
    
    # For higher dimensions, use transpose or permute
    tensor_3d = torch.arange(24).reshape(2, 3, 4)
    print(f"\n3D tensor shape: {tensor_3d.shape}")  # torch.Size([2, 3, 4])
    
    transposed_3d = tensor_3d.transpose(0, 2)  # Swap dims 0 and 2
    print(f"transpose(0, 2) shape: {transposed_3d.shape}")  # torch.Size([4, 3, 2])
    
    # -------------------------------------------------------------------------
    # 7. Permute - Rearrange Multiple Dimensions
    # -------------------------------------------------------------------------
    header("7. Permute - Rearrange Multiple Dimensions")
    
    # permute() - Specify new order of ALL dimensions
    tensor_4d = torch.randn(2, 3, 4, 5)
    print(f"Original shape: {tensor_4d.shape}")  # torch.Size([2, 3, 4, 5])
    
    # Rearrange to (5, 3, 2, 4) - dims: [3, 1, 0, 2]
    permuted = tensor_4d.permute(3, 1, 0, 2)
    print(f"permute(3, 1, 0, 2) shape: {permuted.shape}")  # torch.Size([5, 3, 2, 4])
    
    # Common use case: Change from NCHW to NHWC (batch, channel, height, width → batch, height, width, channel)
    image_batch = torch.randn(32, 3, 224, 224)  # 32 images, 3 channels, 224x224
    print(f"\nImage batch (NCHW): {image_batch.shape}")
    
    image_batch_hwc = image_batch.permute(0, 2, 3, 1)  # Keep batch, move channels to end
    print(f"Image batch (NHWC): {image_batch_hwc.shape}")  # torch.Size([32, 224, 224, 3])
    
    # -------------------------------------------------------------------------
    # 8. Contiguity - Memory Layout Matters
    # -------------------------------------------------------------------------
    header("8. Contiguity - Memory Layout Matters")
    
    # Contiguous tensors have elements in memory in the same order as iteration
    vec_c = torch.arange(6)
    mat_c = vec_c.reshape(2, 3)
    print(f"Original (contiguous): {mat_c.is_contiguous()}")
    print_tensor_info(mat_c, "Contiguous matrix")
    
    # Transpose creates a non-contiguous view
    mat_T = mat_c.T
    print(f"After transpose (non-contiguous): {mat_T.is_contiguous()}")
    print_tensor_info(mat_T, "Transposed matrix")
    
    # view() requires contiguous memory
    try:
        # This will FAIL because mat_T is not contiguous
        mat_T.view(-1)
    except RuntimeError as e:
        print(f"Error with view() on non-contiguous: {e}\n")
    
    # Solution 1: Use contiguous() to create a contiguous copy
    mat_T_cont = mat_T.contiguous()
    print(f"After contiguous(): {mat_T_cont.is_contiguous()}")
    flat_T = mat_T_cont.view(-1)  # Now works!
    print(f"Flattened transposed matrix: {flat_T}")
    
    # Solution 2: Use reshape() instead (handles non-contiguous automatically)
    flat_T_reshape = mat_T.reshape(-1)  # Works without contiguous()
    print(f"Using reshape() instead: {flat_T_reshape}")
    
    # Performance note: contiguous() creates a copy, which takes time and memory
    print(f"\nShared storage before contiguous? {mat_T.storage().data_ptr() == mat_c.storage().data_ptr()}")  # True
    print(f"Shared storage after contiguous? {mat_T_cont.storage().data_ptr() == mat_c.storage().data_ptr()}")  # False
    
    # -------------------------------------------------------------------------
    # 9. Common Reshaping Patterns
    # -------------------------------------------------------------------------
    header("9. Common Reshaping Patterns")
    
    # Pattern 1: Batch of vectors to matrix
    batch_size, feature_dim = 64, 128
    batch_vectors = torch.randn(batch_size, feature_dim)
    print(f"Batch of vectors: {batch_vectors.shape}")  # torch.Size([64, 128])
    
    # Pattern 2: Flatten images for fully connected layer
    # Images: (batch, channels, height, width)
    images = torch.randn(32, 3, 28, 28)
    flat_images = images.reshape(32, -1)  # (32, 3*28*28) = (32, 2352)
    print(f"Flattened images: {flat_images.shape}")
    
    # Pattern 3: Reshape for convolution
    # Fully connected output → convolutional input
    fc_output = torch.randn(16, 512)  # 16 samples, 512 features
    conv_input = fc_output.reshape(16, 512, 1, 1)  # Add spatial dimensions
    print(f"Conv input shape: {conv_input.shape}")
    
    # Pattern 4: Split tensor into groups
    big_tensor = torch.arange(60)
    groups = big_tensor.reshape(3, 20)  # 3 groups of 20 elements
    print(f"Grouped tensor shape: {groups.shape}")
    print(f"Groups:\n{groups}")
    
    # Pattern 5: Add batch dimension
    single_image = torch.randn(3, 224, 224)
    batch_of_one = single_image.unsqueeze(0)  # Add batch dim
    print(f"Single image: {single_image.shape}")
    print(f"As batch: {batch_of_one.shape}")
    
    # -------------------------------------------------------------------------
    # 10. Best Practices
    # -------------------------------------------------------------------------
    header("10. Best Practices and Tips")
    
    print("""
    Key Takeaways:
    
    1. **reshape() vs view()**
       - Use reshape(): Safer, works always (copies if needed)
       - Use view(): Faster IF you know the tensor is contiguous
    
    2. **Contiguity**
       - Operations like transpose() create non-contiguous views
       - Call contiguous() before view() if unsure
       - reshape() handles this automatically
    
    3. **Memory Efficiency**
       - Reshaping operations are usually FREE (create views)
       - contiguous() creates a COPY (takes time and memory)
       - Only call contiguous() when necessary
    
    4. **Dimension Management**
       - unsqueeze() adds dimensions (useful for broadcasting)
       - squeeze() removes size-1 dimensions
       - Use -1 for automatic size inference in reshape()
    
    5. **Common Pitfalls**
       - Always check tensor shapes after reshaping
       - Be aware that views share memory with original
       - Remember: total elements must match in reshape
    """)
    
    # -------------------------------------------------------------------------
    # Practice Exercises
    # -------------------------------------------------------------------------
    header("Practice Exercises")
    
    print("""
    Try these exercises:
    
    1. Create a tensor of shape (4, 5) and reshape it to (2, 2, 5)
    2. Take a (3, 224, 224) image and add a batch dimension at the start
    3. Flatten a (10, 5, 4) tensor into a (10, 20) tensor
    4. Convert a column vector (10, 1) to a row vector (1, 10)
    5. Create a (2, 3, 4, 5) tensor and rearrange to (5, 2, 4, 3)
    
    Solutions:
    """)
    
    # Solution 1
    t1 = torch.randn(4, 5)
    t1_reshaped = t1.reshape(2, 2, 5)
    print(f"1. Shape: {t1.shape} → {t1_reshaped.shape}")
    
    # Solution 2
    img = torch.randn(3, 224, 224)
    img_batch = img.unsqueeze(0)
    print(f"2. Shape: {img.shape} → {img_batch.shape}")
    
    # Solution 3
    t3 = torch.randn(10, 5, 4)
    t3_flat = t3.reshape(10, -1)
    print(f"3. Shape: {t3.shape} → {t3_flat.shape}")
    
    # Solution 4
    col = torch.randn(10, 1)
    row = col.reshape(1, 10)  # or col.T or col.squeeze().unsqueeze(0)
    print(f"4. Shape: {col.shape} → {row.shape}")
    
    # Solution 5
    t5 = torch.randn(2, 3, 4, 5)
    t5_perm = t5.permute(3, 0, 2, 1)
    print(f"5. Shape: {t5.shape} → {t5_perm.shape}")


if __name__ == "__main__":
    main()
