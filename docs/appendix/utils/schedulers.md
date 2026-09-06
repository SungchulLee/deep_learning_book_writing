# Learning Rate Schedulers

Includes:

This implementation provides a concise, educational reference for Learning Rate Schedulers. The code focuses on the core architecture and forward pass, making it straightforward to study the key design patterns and adapt them for experimentation.

## Code

```python
#!/usr/bin/env python3
"""
Learning Rate Schedulers - Common patterns
Includes:
  - Warmup + cosine decay
  - Step decay (PyTorch built-in note)
  - ReduceLROnPlateau (PyTorch built-in note)

File: appendix/utils/schedulers.py
Note: Educational scheduler that returns lr multiplier given step.
"""

import math

# ========================================================================
# Main
# ========================================================================


def warmup_cosine_lr(step, warmup_steps, total_steps, base_lr):
    """
    Warmup + cosine decay schedule.

    - Linearly increase lr from 0 -> base_lr during warmup
    - Then cosine decay from base_lr -> 0

    Returns:
      lr at current step
    """
    if step < warmup_steps:
        return base_lr * (step / max(1, warmup_steps))

    # Cosine decay phase
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


# Note:
# For production, you typically use PyTorch schedulers:
#   torch.optim.lr_scheduler.StepLR
#   torch.optim.lr_scheduler.CosineAnnealingLR
#   torch.optim.lr_scheduler.ReduceLROnPlateau


if __name__ == "__main__":
    pass```

## Discussion

This implementation demonstrates key concepts in utility module using clean, readable PyTorch code. The modular structure makes it easy to study individual components and adapt them for different tasks or datasets.

As a reference implementation, this code prioritizes clarity over optimization. Production systems would typically add mixed-precision training, distributed data parallelism, and more sophisticated data augmentation. Nevertheless, the core architectural ideas shown here remain the same regardless of scale.

## Exercises

**Exercise 1.**
Read through the code and identify the key design decisions. List three specific implementation choices and explain why each is appropriate for utility module.

??? success "Solution to Exercise 1"
    Design decisions vary by implementation but commonly include: (1) choice of activation functions -- ReLU variants provide non-saturating gradients for faster training; (2) normalization strategy -- batch normalization stabilizes training by reducing internal covariate shift; (3) residual connections -- when present, they enable gradient flow in deep networks by providing skip paths. Each choice reflects a trade-off between expressiveness, computational cost, and training stability.

---

**Exercise 2.**
Add input validation to the main function or class to check that inputs have the expected shape and dtype. Raise informative error messages for invalid inputs.

??? success "Solution to Exercise 2"
    At the start of the `forward` method (or relevant function), add checks like: `assert x.dim() == expected_dims, f'Expected {expected_dims}D input, got {x.dim()}D'` and `assert x.dtype == torch.float32, f'Expected float32, got {x.dtype}'`. For shape validation, check critical dimensions: `B, C, H, W = x.shape; assert C == self.expected_channels`. Informative error messages significantly speed up debugging and make the code more robust for reuse.

---

**Exercise 3.**
Describe two potential failure modes of this implementation and explain how you would diagnose and fix each one.

??? success "Solution to Exercise 3"
    Common failure modes include: (1) **Vanishing/exploding gradients** -- diagnosed by monitoring gradient norms (`torch.nn.utils.clip_grad_norm_` or logging `param.grad.norm()` per layer). Fix with gradient clipping, better initialization (Xavier/Kaiming), or architectural changes (residual connections, normalization). (2) **Overfitting** -- diagnosed when training loss decreases but validation loss increases. Fix with regularization (dropout, weight decay, data augmentation) or reducing model capacity. Always monitor both training and validation metrics to catch these issues early.

---

**Exercise 4.**
Write a comprehensive test function that validates the Learning Rate Schedulers implementation. Test edge cases including empty inputs, single-element inputs, very large inputs, and inputs with extreme values (zeros, very large numbers).

??? success "Solution to Exercise 4"
    Create a test function that exercises boundary conditions:
    ```python
    def test_learning rate schedulers():
        model = Learning Rate Schedulers(...)
        # Normal input
        assert model(normal_input).shape == expected_shape
        # Single element batch
        assert model(single_input).shape == (1, ...)
        # Large values (check for overflow)
        out = model(torch.ones(...) * 1000)
        assert torch.isfinite(out).all()
        # Gradient flow
        out = model(normal_input)
        out.sum().backward()
        for p in model.parameters():
            assert p.grad is not None
    ```
    Testing gradient flow is especially important to ensure the architecture supports end-to-end training.
