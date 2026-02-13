import torch.nn as nn

def main():
    # Tiny model: Linear → ReLU → Linear
    model = nn.Sequential(
        nn.Linear(5, 3),  # index 0
        nn.ReLU(),        # index 1
        nn.Linear(3, 1)   # index 2
    )
    # PyTorch nn.Linear(in_features, out_features) stores:
    #   • weight: (out_features, in_features)
    #   • bias  : (out_features,)
    # Forward computes: output = input @ weight.T + bias

    # 0) If it's a Sequential, index-based listing is the quickest
    if isinstance(model, nn.Sequential):
        print("== Sequential index listing ==")
        for i, layer in enumerate(model):
            print(f"[{i}] {layer}")
        print()

    # 1) Top-level children (works for any nn.Module)
    print("== Top-level children ==")
    for name, layer in model.named_children():
        print(f"{name}: {layer}")
    print()

    # 2) Full module tree (includes nested submodules; name '' is the root)
    print("== Full module tree ==")
    for name, layer in model.named_modules():
        print(f"{name or '<root>'}: {layer}")
    print()

    # 3) Targeted info: Linear weight/bias shapes
    print("== Linear shapes ==")
    for name, layer in model.named_modules():
        if isinstance(layer, nn.Linear):
            W = tuple(layer.weight.shape)  # (out_features, in_features)
            b = tuple(layer.bias.shape) if layer.bias is not None else None
            print(f"{name}: Linear weight {W}, bias {b}")
    print()

if __name__ == "__main__":
    main()