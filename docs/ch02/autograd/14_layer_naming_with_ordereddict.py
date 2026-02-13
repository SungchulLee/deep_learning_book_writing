from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F

def freeze_encoder(model: nn.Sequential):
    """
    Freeze the 'encoder' Linear layer (named explicitly in Sequential).
    Matches 'encoder.weight' / 'encoder.bias'.
    """
    for name, p in model.named_parameters():
        if name.startswith("encoder."):
            p.requires_grad_(False)

def unfreeze_all(model: nn.Module):
    for p in model.parameters():
        p.requires_grad_(True)

def grad_status(model: nn.Module, title="Grad status"):
    print(title)
    for name, p in model.named_parameters():
        print(f"{name:20s} requires_grad={p.requires_grad}  grad_is_None={p.grad is None}")
    print()

def main():
    # Use OrderedDict to name layers
    model = nn.Sequential(OrderedDict([
        ("encoder", nn.Linear(5, 3)),
        ("act",     nn.ReLU()),
        ("head",    nn.Linear(3, 1)),
    ]))
    # Error without OrderedDict:
    # nn.Sequential doesn’t accept raw (name, module) tuples as positional args. 
    # It accepts either:
    #   a list/iterable of modules only, or
    #   a single OrderedDict[str, nn.Module] mapping names → modules.
    # model = nn.Sequential(
    #     ("encoder", nn.Linear(5, 3)),
    #     ("act",     nn.ReLU()),
    #     ("head",    nn.Linear(3, 1)),
    # )

    grad_status(model, title="Before freezing 'encoder'")

    freeze_encoder(model)
    grad_status(model, title="After freezing 'encoder'")

    x = torch.randn(4, 5)
    y = torch.randn(4, 1)
    out = model(x)
    loss = F.mse_loss(out, y)
    loss.backward()

    grad_status(model, title="After backward")

    unfreeze_all(model)
    grad_status(model, title="After unfreeze")

if __name__ == "__main__":
    main()