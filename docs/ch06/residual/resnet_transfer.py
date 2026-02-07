"""
ResNet — Feature Extraction and Fine-Tuning
============================================

Demonstrates how to load a pre-trained ResNet-18, inspect its layers,
freeze all weights, and then unfreeze (replace) the final fully-connected
layer for transfer learning on a new task.

Source: PyTorch ResNet tutorial / Transfer Learning with ResNet

Workflow
--------
1.  Load ``resnet18`` with ImageNet-pretrained weights.
2.  Run a forward pass, backward pass, and one optimiser step on random data
    to verify the model works end-to-end.
3.  Inspect the layer hierarchy (``layer4``, ``conv2``, ``fc``).
4.  **Freeze** every parameter in the network (``requires_grad = False``).
5.  **Replace** the final ``fc`` layer with a new ``nn.Linear(512, 10)`` so
    that only the new head is trainable — the standard *feature-extraction*
    transfer-learning pattern.

Run
---
    python resnet_transfer.py
"""

# =============================================================================
# Imports
# =============================================================================
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


# =============================================================================
# 1. Basic forward / backward with pre-trained ResNet
# =============================================================================
def demo_forward_backward():
    """Load ResNet-18, run one forward + backward + optimiser step."""
    print("=" * 60)
    print("1. Forward / backward with pre-trained ResNet-18")
    print("=" * 60)

    data = torch.rand(1, 3, 64, 64)
    labels = torch.rand(1, 1000)

    model = resnet18(weights=ResNet18_Weights.DEFAULT)

    # Forward
    prediction = model(data)
    print(f"  prediction.shape = {prediction.shape}")
    print(f"  softmax sum      = {nn.Softmax(dim=1)(prediction).sum().item():.4f}")
    print(f"  prediction[0, 0] = {prediction[0, 0].item():.6f}")

    # Backward
    loss = (prediction - labels).sum()
    loss.backward()

    # One SGD step
    optim = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
    optim.step()

    # Forward again to see the change
    prediction = model(data)
    print(f"  prediction[0, 0] after step = {prediction[0, 0].item():.6f}")
    print()


# =============================================================================
# 2. Inspecting ResNet layers
# =============================================================================
def demo_inspect_layers():
    """Print the structure of selected ResNet sub-modules."""
    print("=" * 60)
    print("2. Inspecting ResNet-18 layers")
    print("=" * 60)

    model = resnet18(weights=ResNet18_Weights.DEFAULT)

    print(f"  model.layer4         = {model.layer4}")
    print()
    print(f"  model.layer4[0]      = {model.layer4[0]}")
    print()
    print(f"  model.layer4[0].conv2 = {model.layer4[0].conv2}")
    print()
    print(f"  model.fc             = {model.fc}")
    print()

    print("  fc parameters:")
    for param in model.fc.parameters():
        print(f"    shape={str(param.shape):>20s}  requires_grad={param.requires_grad}")
    print()


# =============================================================================
# 3. Freeze all weights
# =============================================================================
def demo_freeze_all():
    """Set ``requires_grad = False`` for every parameter in the network."""
    print("=" * 60)
    print("3. Freeze all weights")
    print("=" * 60)

    model = resnet18(weights=ResNet18_Weights.DEFAULT)

    for param in model.parameters():
        param.requires_grad = False

    print("  fc parameters after freezing:")
    for param in model.fc.parameters():
        print(f"    shape={str(param.shape):>20s}  requires_grad={param.requires_grad}")
    print()


# =============================================================================
# 4. Replace fc and unfreeze only the new head
# =============================================================================
def demo_replace_fc():
    """Freeze backbone, replace ``fc`` with a fresh linear layer → only
    the new head is trainable (feature-extraction transfer learning)."""
    print("=" * 60)
    print("4. Replace fc → only new head is trainable")
    print("=" * 60)

    model = resnet18(weights=ResNet18_Weights.DEFAULT)

    # Freeze everything
    for param in model.parameters():
        param.requires_grad = False

    print("  fc BEFORE replacement:")
    for param in model.fc.parameters():
        print(f"    shape={str(param.shape):>20s}  requires_grad={param.requires_grad}")

    # Replace the last layer (new parameters default to requires_grad=True)
    num_classes = 10
    model.fc = nn.Linear(512, num_classes)

    print(f"\n  fc AFTER replacement (num_classes={num_classes}):")
    for param in model.fc.parameters():
        print(f"    shape={str(param.shape):>20s}  requires_grad={param.requires_grad}")

    # Quick sanity check: count trainable vs frozen parameters
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    print(f"\n  Total params:     {total:>10,}")
    print(f"  Trainable params: {trainable:>10,}")
    print(f"  Frozen params:    {frozen:>10,}")

    # Forward pass with new head
    data = torch.rand(4, 3, 64, 64)
    output = model(data)
    print(f"\n  output.shape = {output.shape}  (batch=4, classes={num_classes})")
    print()


# =============================================================================
# 5. Main
# =============================================================================
def main():
    demo_forward_backward()
    demo_inspect_layers()
    demo_freeze_all()
    demo_replace_fc()


if __name__ == "__main__":
    main()
