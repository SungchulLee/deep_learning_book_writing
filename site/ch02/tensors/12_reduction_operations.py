"""Tutorial 12: Reduction Operations - Aggregating tensor data (sum, mean, min, max, etc.)"""
import torch

def header(title): print(f"\n{'='*70}\n{title}\n{'='*70}")

def main():
    header("1. Basic Reductions - Sum, Mean, Product")
    x = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)
    print(f"Tensor:\n{x}\n")
    print(f"sum(): {x.sum()}")  # All elements
    print(f"mean(): {x.mean()}")
    print(f"prod(): {x.prod()}")  # Product of all
    print(f"sum(dim=0): {x.sum(dim=0)}")  # Column sums
    print(f"sum(dim=1): {x.sum(dim=1)}")  # Row sums
    print(f"sum(dim=1, keepdim=True):\n{x.sum(dim=1, keepdim=True)}")
    
    header("2. Min and Max")
    print(f"min(): {x.min()}")
    print(f"max(): {x.max()}")
    min_val, min_idx = x.min(dim=1)  # Returns values and indices
    print(f"min(dim=1): values={min_val}, indices={min_idx}")
    print(f"argmin(): {x.argmin()}")  # Flattened index
    print(f"argmax(dim=0): {x.argmax(dim=0)}")
    
    header("3. Statistical Operations")
    data = torch.randn(100, 10)
    print(f"Data shape: {data.shape}")
    print(f"std(): {data.std():.4f}")  # Standard deviation
    print(f"var(): {data.var():.4f}")  # Variance
    print(f"median(): {data.median():.4f}")
    print(f"std(dim=0) shape: {data.std(dim=0).shape}")
    
    header("4. Logical Reductions")
    bool_tensor = torch.tensor([[True, False, True], [False, False, True]])
    print(f"Bool tensor:\n{bool_tensor}\n")
    print(f"all(): {bool_tensor.all()}")  # Are all True?
    print(f"any(): {bool_tensor.any()}")  # Is any True?
    print(f"all(dim=1): {bool_tensor.all(dim=1)}")
    print(f"any(dim=0): {bool_tensor.any(dim=0)}")
    
    header("5. Counting and Finding")
    x = torch.tensor([1, 2, 3, 2, 1, 2, 3])
    print(f"Tensor: {x}")
    print(f"unique(): {torch.unique(x)}")
    print(f"Number of elements: {x.numel()}")
    counts = torch.bincount(x)
    print(f"bincount(): {counts}")  # Count occurrences of each value
    
    header("6. Norm Operations")
    vec = torch.tensor([3.0, 4.0])
    print(f"Vector: {vec}")
    print(f"L2 norm (Euclidean): {torch.norm(vec, p=2)}")
    print(f"L1 norm (Manhattan): {torch.norm(vec, p=1)}")
    mat = torch.randn(3, 4)
    print(f"\nMatrix norm: {torch.norm(mat)}")
    print(f"Frobenius norm: {torch.norm(mat, p='fro')}")
    
    header("7. Cumulative Operations")
    x = torch.tensor([1, 2, 3, 4, 5])
    print(f"Tensor: {x}")
    print(f"cumsum(): {torch.cumsum(x, dim=0)}")  # Cumulative sum
    print(f"cumprod(): {torch.cumprod(x, dim=0)}")  # Cumulative product
    mat = torch.arange(12).reshape(3, 4)
    print(f"\nMatrix:\n{mat}")
    print(f"cumsum(dim=1):\n{torch.cumsum(mat, dim=1)}")

if __name__ == "__main__":
    main()
