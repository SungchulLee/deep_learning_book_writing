import torch

def main():

    torch.manual_seed(3)

    x = torch.randn(4, requires_grad=True)
    y = torch.tanh(x)  # elementwise nonlinearity, output is vector (non-scalar)
    print("x:", x)
    print("y = tanh(x):", y)

    # (a) Wrong v shape → error.
    # For non-scalar outputs, backward requires an explicit gradient argument v
    # (a "vector-Jacobian product").
    # Mathematically: sum_i v_i * ∂y_i/∂x_j
    # So the length of v must equal the length of y.
    try:
        v_wrong = torch.tensor([1.0, 2.0])  # wrong length → mismatch with y of length 4
        y.backward(v_wrong)
    except RuntimeError as e:
        print("Shape mismatch as expected:", e)

    # (b) Correct v shape.
    # Here we choose v as a uniform weighting over outputs (just for demo).
    v = torch.ones_like(y) / y.numel()
    # Backward computes the vector-Jacobian product v^T * J
    y.backward(v)
    print("After first backward, x.grad:", x.grad)

    # (c) Gradients accumulate unless cleared.
    # Another backward on a different output will add to the existing .grad values.
    y2 = x ** 2
    y2.backward(v)  # second backward accumulates into x.grad
    print("After second backward (accumulated), x.grad:", x.grad)

    # To start fresh, clear grads (zero_() or set to None).
    x.grad.zero_()
    print("After zero_(), x.grad:", x.grad)

if __name__ == "__main__":
    main()
