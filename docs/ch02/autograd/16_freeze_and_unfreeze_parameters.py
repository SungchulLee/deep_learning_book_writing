# freeze_unfreeze_demo.py
import torch
import torch.nn as nn
import torch.nn.functional as F

def freeze_layer0(model: nn.Sequential):
    """
    Freeze the first Linear layer (index 0) inside a Sequential.
    We match param names that start with '0.' → '0.weight', '0.bias'.
    """
    for name, p in model.named_parameters():
        if name.startswith("0."):
            p.requires_grad_(False)

def unfreeze_all(model: nn.Module):
    """Make every parameter trainable again."""
    for p in model.parameters():
        p.requires_grad_(True)

def grad_status(model: nn.Module, title="Grad status"):
    print(title)
    for name, p in model.named_parameters():
        print(f"{name:20s} requires_grad={p.requires_grad}  grad_is_None={p.grad is None}")
    print()

def main():
    # Tiny model: Linear → ReLU → Linear
    model = nn.Sequential(
        nn.Linear(5, 3),  # index 0 (we'll freeze this one)
        nn.ReLU(),        # index 1
        nn.Linear(3, 1)   # index 2
    )

    grad_status(model, title="Before freezing layer 0")

    # Freeze first Linear (index 0)
    freeze_layer0(model)
    grad_status(model, title="After freezing layer 0")

    # Forward + backward with dummy data
    x = torch.randn(4, 5)  # data tensors default to requires_grad=False
    y = torch.randn(4, 1)
    out = model(x)
    loss = F.mse_loss(out, y)
    loss.backward()

    # After backward: frozen params keep grad=None; others get gradients
    grad_status(model, title="After backward")

    # Unfreeze everything again (optional)
    unfreeze_all(model)
    grad_status(model, title="After unfreeze")

if __name__ == "__main__":
    main()