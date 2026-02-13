import torch

def main():

    x = torch.randn(3, requires_grad=True)
    y = x.sin().sum()  # scalar loss built from differentiable ops
    y.backward()
    # By default, PyTorch tracks gradients for tensors with requires_grad=True
    print("Grad tracked by default → x.grad:", x.grad)

    # In some contexts we want to STOP gradient tracking with torch.no_grad():
    #   - During inference/evaluation: saves memory & computation since no grads needed
    #   - During parameter updates: e.g. w -= lr * w.grad should not be tracked
    #   - For any ops you explicitly do NOT want contributing to future gradients
    print("Typical places to STOP tracking:")
    print("- Inference/evaluation forward pass")
    print("- Optimizer-like parameter updates")
    print("- Any operation that should NOT influence future gradients")

if __name__ == "__main__":
    main()