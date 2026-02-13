#!/usr/bin/env python3
"""
==============================================================================
BEGINNER TUTORIAL 02: Gradient Accumulation
==============================================================================

LEARNING OBJECTIVES:
-------------------
1. Understand that gradients ACCUMULATE by default in PyTorch
2. Learn why you need to zero gradients between training steps
3. See practical examples of gradient accumulation behavior
4. Understand when gradient accumulation is useful vs problematic

KEY CONCEPTS:
------------
- Gradient Accumulation: Gradients add up with each .backward() call
- Zero Gradients: Must explicitly clear with .zero_() or .grad = None
- Use Cases: Large batch simulation, multiple loss terms

COMMON PITFALL:
--------------
Forgetting to zero gradients leads to incorrect updates!

==============================================================================
"""

import torch


def main():
    """
    Demonstrates how gradients accumulate across multiple backward passes
    and why zeroing gradients is essential for correct training.
    """
    
    torch.manual_seed(0)
    
    print("="*70)
    print("PART 1: Default Behavior - Gradients Accumulate!")
    print("="*70)
    
    # Create a simple parameter
    x = torch.tensor([2.0], requires_grad=True)
    
    print(f"Initial x: {x}")
    print(f"Initial x.grad: {x.grad}")
    print()
    
    # First backward pass
    print("First backward pass: loss1 = x^2")
    loss1 = x ** 2  # loss1 = 4.0, d(loss1)/dx = 2x = 4.0
    loss1.backward()
    print(f"After 1st backward: x.grad = {x.grad}")
    print(f"  (Expected: 2*x = 2*2.0 = 4.0)")
    print()
    
    # Second backward pass WITHOUT zeroing gradients
    print("Second backward pass: loss2 = 3*x")
    loss2 = 3 * x  # d(loss2)/dx = 3.0
    loss2.backward()
    print(f"After 2nd backward: x.grad = {x.grad}")
    print(f"  (NOT what we want! 4.0 + 3.0 = 7.0)")
    print(f"  (Gradients accumulated: 1st gradient + 2nd gradient)")
    print()
    
    print("="*70)
    print("PART 2: Correct Behavior - Zero Gradients Between Steps")
    print("="*70)
    
    # Reset
    x2 = torch.tensor([2.0], requires_grad=True)
    
    # First step
    loss1 = x2 ** 2
    loss1.backward()
    print(f"After 1st backward: x2.grad = {x2.grad}")
    
    # Zero the gradient before next step
    x2.grad.zero_()
    print(f"After zeroing: x2.grad = {x2.grad}")
    
    # Second step
    loss2 = 3 * x2
    loss2.backward()
    print(f"After 2nd backward: x2.grad = {x2.grad}")
    print(f"  (Correct! Only the gradient from loss2 = 3.0)")
    print()
    
    print("="*70)
    print("PART 3: Two Ways to Zero Gradients")
    print("="*70)
    
    x3 = torch.tensor([5.0], requires_grad=True)
    loss = (x3 ** 2).sum()
    loss.backward()
    print(f"Before zeroing: x3.grad = {x3.grad}")
    
    print("\nMethod 1: .zero_() - Sets values to 0")
    x3.grad.zero_()
    print(f"After .zero_(): x3.grad = {x3.grad}")
    print(f"  Type: {type(x3.grad)}")
    
    # Accumulate again
    loss.backward()
    print(f"After another backward: x3.grad = {x3.grad}")
    
    print("\nMethod 2: Set to None - Frees memory")
    x3.grad = None
    print(f"After setting to None: x3.grad = {x3.grad}")
    
    # Next backward will create fresh gradient tensor
    loss.backward()
    print(f"After backward: x3.grad = {x3.grad}")
    print()
    
    print("="*70)
    print("PART 4: Intentional Gradient Accumulation (Valid Use Case)")
    print("="*70)
    print("Use case: Simulating large batch when GPU memory is limited")
    print()
    
    torch.manual_seed(42)
    w = torch.randn(1, requires_grad=True)
    lr = 0.01
    
    # Simulate processing a large batch in smaller chunks
    micro_batches = 4
    print(f"Simulating batch_size={micro_batches * 10} by accumulating {micro_batches} micro-batches")
    print()
    
    # Zero gradient at start
    if w.grad is not None:
        w.grad.zero_()
    
    # Accumulate gradients from multiple micro-batches
    for i in range(micro_batches):
        # Each micro-batch computes its loss
        x_batch = torch.randn(10, 1)
        y_batch = 2 * x_batch + 1 + 0.1 * torch.randn(10, 1)
        
        pred = x_batch * w
        loss = ((pred - y_batch) ** 2).mean()
        
        # Backward WITHOUT zeroing - gradients accumulate
        loss.backward()
        
        print(f"  Micro-batch {i+1}: loss={loss.item():.4f}, w.grad={w.grad.item():.4f}")
    
    # Average the accumulated gradient
    w.grad /= micro_batches
    print(f"\nAfter averaging: w.grad = {w.grad.item():.4f}")
    
    # Now update parameters
    with torch.no_grad():
        w -= lr * w.grad
    
    print(f"Updated w: {w.item():.4f}")
    print()
    
    print("="*70)
    print("PART 5: Multiple Loss Terms")
    print("="*70)
    
    x4 = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    
    # Multiple loss terms that we want to combine
    loss_mse = ((x4 - 2) ** 2).mean()  # MSE loss
    loss_l1 = x4.abs().mean()           # L1 regularization
    
    # Method 1: Combine before backward (preferred)
    total_loss = loss_mse + 0.1 * loss_l1
    total_loss.backward()
    print(f"Method 1 - Combined loss backward:")
    print(f"  x4.grad = {x4.grad}")
    
    # Method 2: Accumulate from separate backwards (valid but less common)
    x5 = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    loss_mse = ((x5 - 2) ** 2).mean()
    loss_l1 = x5.abs().mean()
    
    # Zero first
    if x5.grad is not None:
        x5.grad.zero_()
    
    # Backward on first term
    loss_mse.backward(retain_graph=True)  # Need retain_graph for second backward
    grad_after_mse = x5.grad.clone()
    
    # Backward on second term (accumulates)
    scaled_l1 = 0.1 * loss_l1
    scaled_l1.backward()
    
    print(f"\nMethod 2 - Separate backwards with accumulation:")
    print(f"  After MSE: {grad_after_mse}")
    print(f"  After both: {x5.grad}")
    print(f"  Match Method 1? {torch.allclose(x4.grad, x5.grad)}")
    print()
    
    print("="*70)
    print("KEY TAKEAWAYS")
    print("="*70)
    print("""
    ✓ Gradients ACCUMULATE by default - this is a feature, not a bug!
    ✓ Always zero gradients between training steps: .zero_() or = None
    ✓ .zero_() keeps the tensor, = None frees memory (slightly faster)
    ✓ Intentional accumulation is useful for:
        • Simulating large batches with limited memory
        • Combining multiple loss terms
        • Gradient accumulation in distributed training
    ✓ NEVER forget to zero gradients in your training loop!
    """)


if __name__ == "__main__":
    main()
