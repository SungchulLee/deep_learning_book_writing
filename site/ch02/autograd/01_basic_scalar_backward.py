#!/usr/bin/env python3
"""
==============================================================================
BEGINNER TUTORIAL 01: Basic Scalar Backward Pass
==============================================================================

LEARNING OBJECTIVES:
-------------------
1. Understand what a "leaf" tensor is in PyTorch's autograd system
2. Learn how to enable gradient tracking with requires_grad=True
3. See how to compute gradients using .backward() on a scalar loss
4. Understand the difference between leaf and non-leaf tensors
5. Learn about .retain_grad() for non-leaf tensors

KEY CONCEPTS:
------------
- Leaf Tensor: A tensor created directly by the user (not from operations)
- Computational Graph: PyTorch builds this automatically to track operations
- Gradient Accumulation: Gradients add up unless explicitly zeroed
- Vector-Jacobian Product (VJP): How PyTorch computes gradients efficiently

==============================================================================
"""

import torch


def main():
    """
    Demonstrates basic gradient computation for a scalar loss function.
    
    We'll compute: loss = sum(x^2)
    Expected gradient: d(loss)/dx = 2*x
    """
    
    # Set random seed for reproducibility
    torch.manual_seed(0)
    
    print("="*70)
    print("PART 1: Creating a Leaf Tensor with Gradient Tracking")
    print("="*70)
    
    # Create a leaf tensor with gradient tracking enabled
    # A "leaf" tensor is one created by the user, not by an operation
    x = torch.randn(3, requires_grad=True)  # Shape: (3,)
    
    # Alternative: You can create from a list
    # x = torch.tensor([1., 2., 3.], requires_grad=True)
    
    print(f"x: {x}")
    print(f"x.shape: {x.shape}")
    print(f"x.requires_grad: {x.requires_grad}")
    print(f"x.is_leaf: {x.is_leaf}")  # True because we created it directly
    print(f"x.grad_fn: {x.grad_fn}")  # None for leaf tensors
    print(f"x.grad (before backward): {x.grad}")  # None initially
    print()
    
    print("="*70)
    print("PART 2: Forward Pass - Building the Computational Graph")
    print("="*70)
    
    # Forward pass: compute loss = sum(x^2)
    # PyTorch automatically builds a computational graph:
    # x → x**2 → sum → loss
    loss = (x ** 2).sum()  # Shape: scalar ()
    
    print(f"loss: {loss}")
    print(f"loss.shape: {loss.shape}")
    print(f"loss.is_leaf: {loss.is_leaf}")  # False - it's computed from x
    print(f"loss.grad_fn: {loss.grad_fn}")  # Shows the operation that created it
    print()
    
    print("="*70)
    print("PART 3: Backward Pass - Computing Gradients")
    print("="*70)
    print("Calling loss.backward()...")
    print("This computes: d(loss)/dx = d(sum(x^2))/dx = 2*x")
    print()
    
    # Backward pass: compute gradients
    # For scalar tensors, .backward() is equivalent to .backward(torch.tensor(1.0))
    loss.backward()
    
    print(f"x.grad (after backward): {x.grad}")
    print(f"Expected gradient (2*x): {2*x.detach()}")
    print(f"Match? {torch.allclose(x.grad, 2*x.detach())}")
    print()
    
    print("="*70)
    print("PART 4: Understanding Leaf vs Non-Leaf Tensors")
    print("="*70)
    
    # Reset for demonstration
    x2 = torch.tensor([1., 2., 3.], requires_grad=True)
    y = 2 * x2  # y is non-leaf (created by operation)
    z = (y ** 2).sum()  # z is also non-leaf
    
    print(f"x2 (leaf): is_leaf={x2.is_leaf}, grad_fn={x2.grad_fn}")
    print(f"y (non-leaf): is_leaf={y.is_leaf}, grad_fn={y.grad_fn}")
    print(f"z (non-leaf): is_leaf={z.is_leaf}, grad_fn={z.grad_fn}")
    print()
    
    # By default, gradients are only kept for leaf tensors
    z.backward()
    print(f"x2.grad: {x2.grad}  ← Stored (leaf tensor)")
    print(f"y.grad: {y.grad}  ← Not stored (non-leaf tensor)")
    print()
    
    print("="*70)
    print("PART 5: Retaining Gradients for Non-Leaf Tensors")
    print("="*70)
    
    # If you need gradients for non-leaf tensors, use .retain_grad()
    x3 = torch.tensor([1., 2., 3.], requires_grad=True)
    y3 = 2 * x3
    y3.retain_grad()  # ← Tell PyTorch to keep y3's gradient
    z3 = (y3 ** 2).sum()
    
    z3.backward()
    
    print("With .retain_grad() called on y3:")
    print(f"x3.grad: {x3.grad}")
    print(f"y3.grad: {y3.grad}  ← Now stored!")
    print(f"Expected y3.grad (d(z3)/d(y3) = 2*y3): {2*y3.detach()}")
    print()
    
    print("="*70)
    print("UNDERSTANDING: What is grad_output / upstream gradient?")
    print("="*70)
    print("""
    PyTorch uses the CHAIN RULE via Vector-Jacobian Products (VJP):
    
    If y = f(x) and loss = g(y), then:
        d(loss)/dx = (d(loss)/dy) * (dy/dx)
                   = upstream_grad * local_jacobian
    
    For scalar loss.backward():
    - The implicit upstream gradient is 1.0
    - Each operation computes: incoming_grad * local_jacobian
    - This propagates backwards through the computational graph
    
    For non-scalar outputs, you must provide the upstream gradient explicitly!
    (We'll see this in advanced tutorials)
    """)
    
    print("="*70)
    print("KEY TAKEAWAYS")
    print("="*70)
    print("""
    ✓ Leaf tensors: Created by user, not by operations
    ✓ requires_grad=True: Enables gradient tracking
    ✓ .backward(): Computes gradients for scalar losses
    ✓ .grad: Stores computed gradients (only for leaves by default)
    ✓ .retain_grad(): Use this to keep gradients for non-leaf tensors
    ✓ .grad_fn: Shows which operation created a tensor (None for leaves)
    """)


if __name__ == "__main__":
    main()
