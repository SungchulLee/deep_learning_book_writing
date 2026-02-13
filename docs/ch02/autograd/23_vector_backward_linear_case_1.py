import torch

def main():

    torch.manual_seed(1)

    # Linear map setup:
    # A has shape (2×3), x has shape (3,), so y = A @ x has shape (2,).
    A = torch.tensor([[2.0, 0.0, -1.0],
                      [0.5, 3.0,  1.0]], dtype=torch.float32)  # (2×3)
    x = torch.tensor([1.0, -2.0, 0.5], requires_grad=True)     # (3,), leaf tensor
    y = A @ x                                                  # (2,)
    print("A:\n", A)
    print("x:", x)
    print("y = A @ x:", y)

    # Choose upstream gradient v with the same shape as y (2,).
    # This corresponds to selecting weights for the vector-Jacobian product v^T J.
    v = torch.tensor([3.0, -1.0], dtype=torch.float32)

    # Backward pass:
    # For linear map y = A x, the Jacobian J = A.
    # Autograd computes: x.grad = J^T v = A^T v.
    y.backward(v)
    print("v:", v)
    print("x.grad (autograd):", x.grad)

    # Closed form check: multiply A^T by v directly.
    with torch.no_grad():
        expected = A.t() @ v
    print("A^T v (expected):  ", expected)

if __name__ == "__main__":
    main()