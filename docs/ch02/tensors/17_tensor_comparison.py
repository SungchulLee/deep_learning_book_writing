"""Tutorial 17: Tensor Comparison - Element-wise and tensor comparisons"""
import torch

def header(title): print(f"\n{'='*70}\n{title}\n{'='*70}")

def main():
    header("1. Element-wise Comparison")
    a = torch.tensor([1, 2, 3, 4, 5])
    b = torch.tensor([5, 4, 3, 2, 1])
    print(f"a = {a}\nb = {b}\n")
    print(f"a > b: {a > b}")
    print(f"a >= b: {a >= b}")
    print(f"a < b: {a < b}")
    print(f"a == b: {a == b}")
    print(f"a != b: {a != b}")
    
    header("2. Tensor Equality - torch.equal()")
    x = torch.tensor([1, 2, 3])
    y = torch.tensor([1, 2, 3])
    z = torch.tensor([1, 2, 4])
    print(f"x = {x}\ny = {y}\nz = {z}\n")
    print(f"torch.equal(x, y): {torch.equal(x, y)}")  # True
    print(f"torch.equal(x, z): {torch.equal(x, z)}")  # False
    print("\nNote: equal() requires EXACT match")
    
    header("3. Approximate Equality - torch.allclose()")
    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.tensor([1.0001, 2.0001, 3.0001])
    print(f"a = {a}\nb = {b}\n")
    print(f"equal(): {torch.equal(a, b)}")  # False
    print(f"allclose() default: {torch.allclose(a, b)}")  # True
    print(f"allclose(atol=1e-5): {torch.allclose(a, b, atol=1e-5)}")  # False
    print(f"allclose(atol=1e-3): {torch.allclose(a, b, atol=1e-3)}")  # True
    
    header("4. Finding Matches - torch.eq()")
    a = torch.tensor([[1, 2, 3], [4, 5, 6]])
    b = torch.tensor([[1, 0, 3], [4, 0, 6]])
    print(f"a =\n{a}\nb =\n{b}\n")
    matches = torch.eq(a, b)
    print(f"Element-wise equality:\n{matches}")
    num_matches = matches.sum().item()
    print(f"Number of matching elements: {num_matches}")
    
    header("5. Top-k and Sorting")
    scores = torch.tensor([3.2, 1.5, 4.7, 2.1, 5.3])
    print(f"Scores: {scores}")
    top_k_values, top_k_indices = torch.topk(scores, k=3)
    print(f"Top 3 values: {top_k_values}")
    print(f"Top 3 indices: {top_k_indices}")
    sorted_values, sorted_indices = torch.sort(scores, descending=True)
    print(f"\nSorted (descending): {sorted_values}")
    print(f"Sorted indices: {sorted_indices}")
    
    header("6. Element-wise Max/Min")
    a = torch.tensor([1, 5, 3])
    b = torch.tensor([2, 4, 6])
    print(f"a = {a}\nb = {b}\n")
    max_elem = torch.max(a, b)
    min_elem = torch.min(a, b)
    print(f"Element-wise max: {max_elem}")
    print(f"Element-wise min: {min_elem}")
    
    header("7. Practical: Finding Best Predictions")
    logits = torch.randn(5, 10)  # 5 samples, 10 classes
    print(f"Logits shape: {logits.shape}")
    predictions = torch.argmax(logits, dim=1)
    print(f"Predicted classes: {predictions}")
    max_scores, _ = torch.max(logits, dim=1)
    print(f"Max scores: {max_scores}")

if __name__ == "__main__":
    main()
