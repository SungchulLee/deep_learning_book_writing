import torch

def main():

    torch.manual_seed(1)

    # Setup: Linear map A: R^m → R^n
    # A has shape (2,3), x has shape (3,1), so y = A @ x has shape (2,1).
    A = torch.tensor([[2.0, 0.0, -1.0],
                      [0.5, 3.0,  1.0]], dtype=torch.float32)        # (2×3)
    x = torch.tensor([[1.0], [-2.0], [0.5]], requires_grad=True)     # (3×1), leaf param
    y = A @ x                                                        # (2×1)
    print("A:\n", A)
    print("x:", x)
    print("y = A @ x:", y)

    # Choose a vector v ∈ R^2 with the same shape as y (2×1).
    # This plays the role of the upstream gradient in the vector-Jacobian product.
    v = torch.tensor([[3.0], [-1.0]], dtype=torch.float32)

    # Backward pass:
    # Autograd computes: x.grad = J^T v, where J = ∂y/∂x = A (constant here).
    # So mathematically, x.grad = A^T v.
    y.backward(v)
    print("v:", v)
    print("x.grad (autograd):", x.grad)

    # Closed-form manual computation for verification: A^T v
    with torch.no_grad():
        expected = A.t() @ v
    print("A^T v (expected):  ", expected)

if __name__ == "__main__":
    main()
