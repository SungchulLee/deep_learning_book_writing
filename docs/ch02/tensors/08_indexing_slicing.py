"""
Tutorial 08: Indexing and Slicing Tensors
==========================================

Learn how to access and modify specific elements, rows, columns, or sub-tensors.
This is crucial for data manipulation and neural network operations.

Key Concepts:
- Basic indexing (single elements)
- Slicing (ranges of elements)
- Advanced indexing (boolean masks, fancy indexing)
- Multi-dimensional indexing
- In-place modifications via indexing
"""

import torch


def print_section(title: str):
    """Helper to print section headers."""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def main():
    # -------------------------------------------------------------------------
    # Setup: Create sample tensors for demonstrations
    # -------------------------------------------------------------------------
    print_section("Setup: Sample Tensors")
    
    # 1D tensor
    vec = torch.tensor([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    print("1D tensor (vec):", vec)
    
    # 2D tensor (3x4 matrix)
    mat = torch.arange(1, 13).reshape(3, 4)
    print("2D tensor (mat):\n", mat)
    
    # 3D tensor (2x3x4 - think of it as 2 matrices of shape 3x4)
    tensor_3d = torch.arange(1, 25).reshape(2, 3, 4)
    print("3D tensor:\n", tensor_3d)
    
    # -------------------------------------------------------------------------
    # 1. Basic Indexing - Single Elements
    # -------------------------------------------------------------------------
    print_section("1. Basic Indexing - Single Elements")
    
    # 1D indexing (Python-style, 0-indexed)
    elem = vec[3]  # Fourth element
    print(f"vec[3] = {elem}")  # 40
    
    # Negative indexing (from the end)
    last = vec[-1]  # Last element
    second_last = vec[-2]  # Second to last
    print(f"vec[-1] = {last}, vec[-2] = {second_last}")  # 100, 90
    
    # 2D indexing - [row, column]
    elem_2d = mat[1, 2]  # Row 1, Column 2
    print(f"mat[1, 2] = {elem_2d}")  # 7
    
    # 3D indexing - [depth, row, column]
    elem_3d = tensor_3d[0, 1, 2]  # First matrix, row 1, column 2
    print(f"tensor_3d[0, 1, 2] = {elem_3d}")  # 7
    
    # Important: Single element indexing returns a 0-D tensor (scalar)
    print(f"Type: {type(elem)}, Shape: {elem.shape}")  # Shape is torch.Size([])
    
    # To get Python scalar, use .item()
    python_int = elem.item()
    print(f"Python int: {python_int}, Type: {type(python_int)}")
    
    # -------------------------------------------------------------------------
    # 2. Slicing - Extracting Sub-tensors
    # -------------------------------------------------------------------------
    print_section("2. Slicing - Extracting Sub-tensors")
    
    # Syntax: tensor[start:end:step]
    # - start: inclusive (default 0)
    # - end: exclusive (default length)
    # - step: stride (default 1)
    
    # Basic slicing
    sub_vec = vec[2:5]  # Elements at indices 2, 3, 4
    print(f"vec[2:5] = {sub_vec}")  # tensor([30, 40, 50])
    
    # Omit start (begins from 0)
    start_slice = vec[:4]  # First 4 elements
    print(f"vec[:4] = {start_slice}")  # tensor([10, 20, 30, 40])
    
    # Omit end (goes to the end)
    end_slice = vec[6:]  # From index 6 to end
    print(f"vec[6:] = {end_slice}")  # tensor([70, 80, 90, 100])
    
    # Use step (skip elements)
    every_other = vec[::2]  # Every 2nd element
    print(f"vec[::2] = {every_other}")  # tensor([10, 30, 50, 70, 90])
    
    # Reverse a tensor
    reversed_vec = vec[::-1]
    print(f"vec[::-1] = {reversed_vec}")  # tensor([100, 90, 80, ..., 10])
    
    # -------------------------------------------------------------------------
    # 3. Multi-dimensional Slicing
    # -------------------------------------------------------------------------
    print_section("3. Multi-dimensional Slicing")
    
    print("Original matrix (mat):\n", mat)
    # tensor([[ 1,  2,  3,  4],
    #         [ 5,  6,  7,  8],
    #         [ 9, 10, 11, 12]])
    
    # Select entire row (row 1)
    row_1 = mat[1, :]  # or simply mat[1]
    print(f"Row 1 (mat[1, :]): {row_1}")  # tensor([5, 6, 7, 8])
    
    # Select entire column (column 2)
    col_2 = mat[:, 2]
    print(f"Column 2 (mat[:, 2]): {col_2}")  # tensor([ 3,  7, 11])
    
    # Select sub-matrix (rows 0-1, columns 1-2)
    sub_mat = mat[0:2, 1:3]
    print(f"Sub-matrix (mat[0:2, 1:3]):\n{sub_mat}")
    # tensor([[2, 3],
    #         [6, 7]])
    
    # Select with steps
    every_other_row = mat[::2, :]  # Rows 0, 2
    print(f"Every other row:\n{every_other_row}")
    
    # -------------------------------------------------------------------------
    # 4. Ellipsis (...) - Filling in missing dimensions
    # -------------------------------------------------------------------------
    print_section("4. Ellipsis (...) - Shorthand for ':' across dimensions")
    
    # Ellipsis represents all dimensions not explicitly specified
    # Useful for high-dimensional tensors
    
    # For 3D tensor: select first element of last dimension across all others
    result = tensor_3d[..., 0]  # Equivalent to tensor_3d[:, :, 0]
    print(f"tensor_3d[..., 0] shape: {result.shape}")  # torch.Size([2, 3])
    print(f"tensor_3d[..., 0]:\n{result}")
    
    # Select middle "matrix" (depth=1)
    middle = tensor_3d[1, ...]  # Equivalent to tensor_3d[1, :, :]
    print(f"tensor_3d[1, ...] shape: {middle.shape}")  # torch.Size([3, 4])
    
    # -------------------------------------------------------------------------
    # 5. Boolean Indexing (Masking)
    # -------------------------------------------------------------------------
    print_section("5. Boolean Indexing (Masking)")
    
    # Create a boolean mask
    mask = vec > 50  # Elements greater than 50
    print(f"Mask (vec > 50): {mask}")
    # tensor([False, False, False, False, False,  True,  True,  True,  True,  True])
    
    # Use mask to filter
    filtered = vec[mask]
    print(f"vec[mask] (elements > 50): {filtered}")  # tensor([ 60,  70,  80,  90, 100])
    
    # Multiple conditions with & (AND) and | (OR)
    # Note: Use & and |, not 'and'/'or' (those don't work element-wise)
    mask_complex = (vec > 30) & (vec < 80)
    print(f"vec[(vec > 30) & (vec < 80)]: {vec[mask_complex]}")  # tensor([40, 50, 60, 70])
    
    # Boolean indexing on 2D tensors
    mask_2d = mat > 6
    print(f"Elements > 6 in mat: {mat[mask_2d]}")  # Returns 1D tensor of matching elements
    
    # -------------------------------------------------------------------------
    # 6. Advanced Indexing - Index Tensors
    # -------------------------------------------------------------------------
    print_section("6. Advanced Indexing - Index Tensors")
    
    # Use a tensor of indices to select elements
    indices = torch.tensor([0, 2, 4])
    selected = vec[indices]
    print(f"vec[[0, 2, 4]]: {selected}")  # tensor([10, 30, 50])
    
    # Fancy indexing with 2D tensors
    row_indices = torch.tensor([0, 1, 2])
    col_indices = torch.tensor([1, 2, 3])
    # Select mat[0,1], mat[1,2], mat[2,3]
    diagonal_like = mat[row_indices, col_indices]
    print(f"mat[row_indices, col_indices]: {diagonal_like}")  # tensor([ 2,  7, 12])
    
    # -------------------------------------------------------------------------
    # 7. In-place Modification via Indexing
    # -------------------------------------------------------------------------
    print_section("7. In-place Modification via Indexing")
    
    # Create a copy to modify
    vec_copy = vec.clone()
    print(f"Original: {vec_copy}")
    
    # Modify single element
    vec_copy[3] = 999
    print(f"After vec_copy[3] = 999: {vec_copy}")
    
    # Modify slice
    vec_copy[5:8] = 0
    print(f"After vec_copy[5:8] = 0: {vec_copy}")
    
    # Modify via boolean mask
    vec_copy[vec_copy < 40] = -1
    print(f"After setting elements < 40 to -1: {vec_copy}")
    
    # 2D modification
    mat_copy = mat.clone()
    mat_copy[0, :] = 0  # Set first row to zeros
    mat_copy[:, -1] = 99  # Set last column to 99
    print(f"Modified matrix:\n{mat_copy}")
    
    # -------------------------------------------------------------------------
    # 8. View vs Copy - Important Memory Consideration
    # -------------------------------------------------------------------------
    print_section("8. View vs Copy - Memory Behavior")
    
    # Slicing creates a VIEW (shares memory with original)
    original = torch.tensor([1, 2, 3, 4, 5])
    view = original[1:4]
    
    print(f"Original: {original}")
    print(f"View: {view}")
    
    # Modifying the view affects the original!
    view[0] = 999
    print(f"After view[0] = 999:")
    print(f"Original: {original}")  # Changed!
    print(f"View: {view}")
    
    # To avoid this, use .clone()
    original2 = torch.tensor([1, 2, 3, 4, 5])
    true_copy = original2[1:4].clone()
    true_copy[0] = 999
    print(f"\nWith .clone():")
    print(f"Original: {original2}")  # Unchanged
    print(f"Copy: {true_copy}")
    
    # Check if tensors share storage
    print(f"\nShared storage? {original.data_ptr() == view.data_ptr()}")  # False (different data pointer due to offset)
    print(f"Same underlying storage? {original.storage().data_ptr() == view.storage().data_ptr()}")  # True!
    
    # -------------------------------------------------------------------------
    # 9. Common Patterns and Use Cases
    # -------------------------------------------------------------------------
    print_section("9. Common Patterns and Use Cases")
    
    # Pattern 1: Get diagonal of a matrix
    diag = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    diagonal = torch.diagonal(diag)
    print(f"Diagonal: {diagonal}")  # tensor([1, 5, 9])
    
    # Pattern 2: Select specific rows
    data = torch.randn(100, 10)  # 100 samples, 10 features
    batch_indices = torch.tensor([0, 5, 10, 15])
    batch = data[batch_indices]
    print(f"Selected batch shape: {batch.shape}")  # torch.Size([4, 10])
    
    # Pattern 3: Remove elements (by selecting others)
    vec_to_filter = torch.tensor([1, 2, 3, 4, 5, 6])
    keep_mask = torch.tensor([True, False, True, True, False, True])
    filtered_result = vec_to_filter[keep_mask]
    print(f"After filtering: {filtered_result}")  # tensor([1, 3, 4, 6])
    
    # Pattern 4: Conditionally replace values
    data_with_outliers = torch.tensor([1.0, 2.0, 100.0, 3.0, -50.0, 4.0])
    data_clipped = data_with_outliers.clone()
    data_clipped[data_clipped > 10] = 10.0
    data_clipped[data_clipped < 0] = 0.0
    print(f"Clipped data: {data_clipped}")  # tensor([1., 2., 10., 3., 0., 4.])
    
    # -------------------------------------------------------------------------
    # Practice Exercises
    # -------------------------------------------------------------------------
    print_section("Practice Exercises")
    
    print("""
    Try these exercises to test your understanding:
    
    1. Create a 5x5 matrix and extract its corners (4 elements: [0,0], [0,4], [4,0], [4,4])
    2. Given a 1D tensor of 20 elements, select every 3rd element starting from index 1
    3. Create a 4x6 matrix and set all elements in the 2nd row and 3rd column to zero
    4. Use boolean indexing to find all elements in a tensor that are between 5 and 15
    5. Create a 3x3 matrix and swap its first and last rows using indexing
    
    Solutions below...
    """)
    
    # Solution 1
    mat_5x5 = torch.arange(25).reshape(5, 5)
    corners_indices = torch.tensor([[0, 0], [0, 4], [4, 0], [4, 4]])
    corners = mat_5x5[corners_indices[:, 0], corners_indices[:, 1]]
    print(f"Exercise 1 - Corners: {corners}")
    
    # Solution 2
    vec_20 = torch.arange(20)
    every_third = vec_20[1::3]
    print(f"Exercise 2 - Every 3rd from index 1: {every_third}")
    
    # Solution 3
    mat_4x6 = torch.ones(4, 6)
    mat_4x6[1, :] = 0  # 2nd row
    mat_4x6[:, 2] = 0  # 3rd column
    print(f"Exercise 3 - Modified matrix:\n{mat_4x6}")
    
    # Solution 4
    test_vec = torch.arange(20)
    between = test_vec[(test_vec >= 5) & (test_vec <= 15)]
    print(f"Exercise 4 - Elements between 5 and 15: {between}")
    
    # Solution 5
    mat_3x3 = torch.arange(9).reshape(3, 3)
    mat_3x3[[0, 2]] = mat_3x3[[2, 0]]  # Swap rows 0 and 2
    print(f"Exercise 5 - After swapping rows:\n{mat_3x3}")


if __name__ == "__main__":
    main()
