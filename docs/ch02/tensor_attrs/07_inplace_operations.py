#!/usr/bin/env python3
"""
In-place operations: efficiency and gotchas.

Covers:
- Naming convention: operations ending with underscore (_)
- Performance benefits and memory sharing implications
- Autograd restrictions with in-place ops
- Common in-place operations: add_, mul_, clamp_, etc.
- When to use and when to avoid in-place ops
"""

import torch
import torch.nn as nn

def header(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

def main():
    torch.manual_seed(0)

    # -------------------------------------------------------------------------
    header("Basic in-place operations")
    x = torch.tensor([1., 2., 3., 4., 5.])
    print("Original x:", x)
    
    x.add_(10)  # x = x + 10, in-place
    print("After x.add_(10):", x)
    
    x.mul_(2)   # x = x * 2, in-place
    print("After x.mul_(2):", x)
    
    x.clamp_(0, 30)  # Clamp values in-place
    print("After x.clamp_(0, 30):", x)

    # -------------------------------------------------------------------------
    header("Memory sharing with in-place ops")
    a = torch.randn(3, 4)
    b = a  # b is just another reference to the same tensor
    a_id = id(a)
    
    a.add_(1)  # Modifies the underlying data
    print("a and b share memory:", id(a) == id(b) == a_id)
    print("b also changed:", b[0, :3])

    # -------------------------------------------------------------------------
    header("Out-of-place vs in-place comparison")
    import time
    
    # Large tensor for timing
    big = torch.randn(1000, 1000)
    
    # Out-of-place (creates new tensor)
    start = time.time()
    for _ in range(100):
        result = big + 1.0
    out_of_place_time = time.time() - start
    
    # In-place (modifies existing tensor)
    start = time.time()
    for _ in range(100):
        big.add_(1.0)
    in_place_time = time.time() - start
    
    print(f"Out-of-place: {out_of_place_time:.4f}s")
    print(f"In-place:     {in_place_time:.4f}s")
    print(f"Speedup:      {out_of_place_time/in_place_time:.2f}x")

    # -------------------------------------------------------------------------
    header("Autograd restriction: in-place on leaf tensors with requires_grad")
    leaf = torch.tensor([1., 2., 3.], requires_grad=True)
    
    try:
        leaf.add_(1)  # This will fail!
    except RuntimeError as e:
        print("ERROR (expected):", str(e)[:80] + "...")
    
    # Solution 1: Use out-of-place operation
    leaf2 = torch.tensor([1., 2., 3.], requires_grad=True)
    result = leaf2 + 1  # This works fine
    print("Out-of-place works:", result)
    
    # Solution 2: Use torch.no_grad() for parameter updates
    leaf3 = torch.tensor([1., 2., 3.], requires_grad=True)
    with torch.no_grad():
        leaf3.add_(1)  # OK inside no_grad
    print("In-place with no_grad:", leaf3)

    # -------------------------------------------------------------------------
    header("In-place ops on non-leaf tensors (intermediate results)")
    x = torch.tensor([1., 2., 3.], requires_grad=True)
    y = x * 2  # Non-leaf tensor
    
    try:
        y.add_(1)  # Also problematic for autograd
    except RuntimeError as e:
        print("ERROR (expected):", str(e)[:80] + "...")
    print("In-place on non-leaf can break autograd graph")

    # -------------------------------------------------------------------------
    header("Common in-place operations showcase")
    t = torch.randn(5)
    print("Original t:", t)
    
    t_copy = t.clone()
    t_copy.abs_()
    print("abs_():", t_copy)
    
    t_copy = t.clone()
    t_copy.neg_()
    print("neg_():", t_copy)
    
    t_copy = t.clone()
    t_copy.sqrt_().abs_()  # Chain in-place ops
    print("sqrt_().abs_():", t_copy)
    
    t_copy = t.clone()
    t_copy.pow_(2)
    print("pow_(2):", t_copy)
    
    t_copy = torch.randn(5)
    t_copy.uniform_(-1, 1)  # Fill with uniform random
    print("uniform_(-1, 1):", t_copy)
    
    t_copy = torch.zeros(5)
    t_copy.normal_(mean=0, std=1)  # Fill with normal random
    print("normal_(0, 1):", t_copy)

    # -------------------------------------------------------------------------
    header("fill_ and zero_ operations")
    m = torch.randn(3, 3)
    print("Before fill_:\n", m)
    
    m.fill_(7.0)
    print("After fill_(7.0):\n", m)
    
    m.zero_()
    print("After zero_():\n", m)

    # -------------------------------------------------------------------------
    header("Indexed in-place assignment")
    arr = torch.zeros(5)
    arr[1:4] = torch.tensor([10., 20., 30.])
    print("After indexed assignment:", arr)
    
    arr[arr > 15] = -1  # Boolean masking with assignment
    print("After boolean mask assignment:", arr)

    # -------------------------------------------------------------------------
    header("copy_ for in-place copying")
    src = torch.randn(3, 3)
    dst = torch.zeros(3, 3)
    print("dst before copy_:\n", dst)
    
    dst.copy_(src)  # Copy src data into dst
    print("dst after copy_(src):\n", dst)

    # -------------------------------------------------------------------------
    header("Best practices: when to use in-place")
    print("✓ Use in-place ops for:")
    print("  - Parameter updates inside torch.no_grad()")
    print("  - Memory-critical situations")
    print("  - Explicit tensor initialization (fill_, zero_, normal_)")
    print("  - When you KNOW autograd won't be needed")
    print("\n✗ Avoid in-place ops for:")
    print("  - Leaf tensors with requires_grad=True")
    print("  - Intermediate computation results in autograd")
    print("  - When code clarity is more important than speed")
    print("  - When tensors might be aliased unexpectedly")

if __name__ == "__main__":
    main()
