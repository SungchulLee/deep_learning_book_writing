import torch

def main():

    torch.manual_seed(0)

    # Forward: x ∈ R^3 → y ∈ R^3 (vector output, not scalar).
    x = torch.randn(3, requires_grad=True)
    y = torch.sin(x)  # elementwise sine, shape [3]
    print("x:", x)
    print("y = sin(x):", y, "| shape:", tuple(y.shape))

    # Autograd requires a scalar loss to call .backward() without arguments.
    # If y is non-scalar, PyTorch needs an explicit "vector" v to form v^T J.
    try:
        y.backward()
    except RuntimeError as e:
        print("As expected, calling y.backward() fails for non-scalar y:")
        print("  ", e)

    # Supply v with same shape as y. Autograd computes:
    #   x.grad = J^T v, where J = ∂y/∂x.
    # This is the vector-Jacobian product (VJP).
    v = torch.tensor([0.1, 1.0, 0.01], dtype=torch.float32)
    y.backward(v)
    print("Chosen v:", v)
    print("x.grad (v^T * J):", x.grad)

if __name__ == "__main__":
    main()