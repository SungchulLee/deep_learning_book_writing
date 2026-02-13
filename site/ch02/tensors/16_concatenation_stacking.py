"""Tutorial 16: Concatenation and Stacking - Combining tensors"""
import torch

def header(title): print(f"\n{'='*70}\n{title}\n{'='*70}")

def main():
    header("1. Concatenation - torch.cat()")
    a = torch.tensor([[1, 2], [3, 4]])
    b = torch.tensor([[5, 6], [7, 8]])
    print(f"a =\n{a}\nb =\n{b}\n")
    cat_dim0 = torch.cat([a, b], dim=0)  # Vertical stacking
    print(f"cat(dim=0) - Stack vertically:\n{cat_dim0}")
    cat_dim1 = torch.cat([a, b], dim=1)  # Horizontal stacking
    print(f"\ncat(dim=1) - Stack horizontally:\n{cat_dim1}")
    c = torch.tensor([[9, 10], [11, 12]])
    cat_multi = torch.cat([a, b, c], dim=0)
    print(f"\nCat multiple tensors:\n{cat_multi}")
    
    header("2. Stacking - torch.stack()")
    x = torch.tensor([1, 2, 3])
    y = torch.tensor([4, 5, 6])
    print(f"x = {x}\ny = {y}\n")
    stack_dim0 = torch.stack([x, y], dim=0)
    print(f"stack(dim=0):\n{stack_dim0}")
    print(f"Shape: {stack_dim0.shape}")  # (2, 3)
    stack_dim1 = torch.stack([x, y], dim=1)
    print(f"\nstack(dim=1):\n{stack_dim1}")
    print(f"Shape: {stack_dim1.shape}")  # (3, 2)
    
    header("3. cat() vs stack()")
    a = torch.randn(2, 3)
    b = torch.randn(2, 3)
    print(f"a shape: {a.shape}, b shape: {b.shape}")
    cat_result = torch.cat([a, b], dim=0)
    stack_result = torch.stack([a, b], dim=0)
    print(f"cat(dim=0) shape: {cat_result.shape}")  # (4, 3)
    print(f"stack(dim=0) shape: {stack_result.shape}")  # (2, 2, 3)
    print("\nKey difference:")
    print("- cat(): Concatenates along existing dimension")
    print("- stack(): Creates new dimension for stacking")
    
    header("4. Splitting - torch.split()")
    tensor = torch.arange(10)
    print(f"Tensor: {tensor}")
    splits = torch.split(tensor, 3)  # Split into chunks of size 3
    print(f"split(3): {splits}")
    splits_sizes = torch.split(tensor, [3, 3, 4])  # Custom sizes
    print(f"split([3,3,4]): {splits_sizes}")
    
    header("5. Chunking - torch.chunk()")
    tensor = torch.arange(12).reshape(3, 4)
    print(f"Tensor:\n{tensor}")
    chunks = torch.chunk(tensor, 2, dim=0)  # Split into 2 chunks along dim 0
    print(f"chunk(2, dim=0):")
    for i, chunk in enumerate(chunks):
        print(f"  Chunk {i}:\n{chunk}")
    
    header("6. Unbinding - torch.unbind()")
    stacked = torch.arange(12).reshape(3, 4)
    print(f"Stacked:\n{stacked}")
    unbound = torch.unbind(stacked, dim=0)  # Unpack along dimension
    print(f"unbind(dim=0): {len(unbound)} tensors")
    for i, t in enumerate(unbound):
        print(f"  Tensor {i}: {t}")
    
    header("7. Practical: Building Batch")
    sample1 = torch.randn(3, 32, 32)  # Image 1
    sample2 = torch.randn(3, 32, 32)  # Image 2
    sample3 = torch.randn(3, 32, 32)  # Image 3
    batch = torch.stack([sample1, sample2, sample3], dim=0)
    print(f"Batch shape: {batch.shape}")  # (3, 3, 32, 32)
    print("Format: (batch_size, channels, height, width)")
    
    header("8. Practical: Feature Concatenation")
    features_a = torch.randn(10, 64)  # 10 samples, 64 features
    features_b = torch.randn(10, 32)  # 10 samples, 32 features
    combined = torch.cat([features_a, features_b], dim=1)
    print(f"features_a: {features_a.shape}")
    print(f"features_b: {features_b.shape}")
    print(f"combined: {combined.shape}")  # (10, 96)

if __name__ == "__main__":
    main()
