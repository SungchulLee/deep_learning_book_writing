"""Tutorial 13: Logical Operations - Boolean operations and masking"""
import torch

def header(title): print(f"\n{'='*70}\n{title}\n{'='*70}")

def main():
    header("1. Comparison Operations")
    a = torch.tensor([1, 2, 3, 4, 5])
    b = torch.tensor([5, 4, 3, 2, 1])
    print(f"a = {a}\nb = {b}\n")
    print(f"a > b: {a > b}")
    print(f"a == b: {a == b}")
    print(f"torch.eq(a, b): {torch.eq(a, b)}")
    print(f"torch.gt(a, b): {torch.gt(a, b)}")
    
    header("2. Logical Operations - AND, OR, NOT")
    x = torch.tensor([True, True, False, False])
    y = torch.tensor([True, False, True, False])
    print(f"x = {x}\ny = {y}\n")
    print(f"x & y (AND): {x & y}")
    print(f"x | y (OR): {x | y}")
    print(f"~x (NOT): {~x}")
    print(f"x ^ y (XOR): {x ^ y}")
    print(f"torch.logical_and(x, y): {torch.logical_and(x, y)}")
    
    header("3. Boolean Masking")
    data = torch.tensor([10, 20, 5, 30, 15])
    print(f"Data: {data}")
    mask = data > 15
    print(f"Mask (data > 15): {mask}")
    filtered = data[mask]
    print(f"Filtered data: {filtered}")
    complex_mask = (data > 10) & (data < 25)
    print(f"Complex mask: {complex_mask}")
    print(f"Filtered: {data[complex_mask]}")
    
    header("4. Conditional Selection")
    x = torch.tensor([-2, -1, 0, 1, 2])
    print(f"x = {x}")
    result = torch.where(x > 0, x, torch.zeros_like(x))  # ReLU
    print(f"ReLU (where x>0, x, 0): {result}")
    a = torch.tensor([1, 2, 3])
    b = torch.tensor([10, 20, 30])
    condition = torch.tensor([True, False, True])
    selected = torch.where(condition, a, b)
    print(f"\nSelect from a or b: {selected}")
    
    header("5. Element-wise Comparison")
    x = torch.tensor([[1, 2], [3, 4]])
    y = torch.tensor([[2, 2], [2, 4]])
    print(f"x:\n{x}\ny:\n{y}\n")
    print(f"torch.eq(x, y):\n{torch.eq(x, y)}")
    print(f"torch.allclose(x, y): {torch.allclose(x.float(), y.float())}")
    z = torch.tensor([[1.0001, 2.0], [3.0, 4.0]])
    w = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    print(f"\nClose values? {torch.allclose(z, w, atol=1e-3)}")
    
    header("6. Practical Example: Data Cleaning")
    data = torch.tensor([1.0, 2.0, float('nan'), 4.0, float('inf')])
    print(f"Raw data: {data}")
    is_finite = torch.isfinite(data)
    print(f"is_finite: {is_finite}")
    clean_data = data[is_finite]
    print(f"Clean data: {clean_data}")
    data_with_outliers = torch.tensor([1, 2, 100, 3, 4, 200])
    mask = (data_with_outliers > 0) & (data_with_outliers < 50)
    cleaned = data_with_outliers[mask]
    print(f"\nOutlier removal: {cleaned}")

if __name__ == "__main__":
    main()
