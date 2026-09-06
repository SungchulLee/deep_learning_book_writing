# Data Augmentation

Data Augmentation - Common vision augmentations (tensor-based) Includes:

This implementation provides a concise, educational reference for Data Augmentation. The code focuses on the core architecture and forward pass, making it straightforward to study the key design patterns and adapt them for experimentation.

## Code

```python
#!/usr/bin/env python3
"""
Data Augmentation - Common vision augmentations (tensor-based)
Includes:
  - Random horizontal flip
  - Random crop (naive)
  - Color jitter (simple brightness/contrast)
  - Mixup (classification)

File: appendix/utils/augmentation.py
Note: Educational implementations (not as feature-complete as torchvision.transforms).
"""

import torch

# ========================================================================
# Main
# ========================================================================


def random_horizontal_flip(x, p=0.5):
    """
    Randomly flip images horizontally.

    x: (B, C, H, W)
    """
    if torch.rand(1).item() < p:
        return torch.flip(x, dims=[3])  # flip width dimension
    return x


def random_crop(x, crop_h, crop_w):
    """
    Naive random crop.

    x: (B, C, H, W)
    """
    B, C, H, W = x.shape
    if crop_h > H or crop_w > W:
        raise ValueError("Crop size must be <= image size")

    top = torch.randint(0, H - crop_h + 1, (1,)).item()
    left = torch.randint(0, W - crop_w + 1, (1,)).item()
    return x[:, :, top:top + crop_h, left:left + crop_w]


def mixup(x, y, alpha=0.2):
    """
    Mixup augmentation for classification.

    x: (B, C, H, W)
    y: (B,) class indices OR (B, num_classes) one-hot

    Returns:
      x_mix, y_a, y_b, lam
    """
    if alpha <= 0:
        return x, y, y, 1.0

    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    B = x.size(0)
    perm = torch.randperm(B)

    x_mix = lam * x + (1 - lam) * x[perm]
    y_a = y
    y_b = y[perm]
    return x_mix, y_a, y_b, lam


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
Write a comprehensive test function that validates the Data Augmentation implementation. Test edge cases including empty inputs, single-element inputs, very large inputs, and inputs with extreme values (zeros, very large numbers).

??? success "Solution to Exercise 4"
    Create a test function that exercises boundary conditions:
    ```python
    def test_data augmentation():
        model = Data Augmentation(...)
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
