import torch

def main():
    torch.manual_seed(4)

    B, m, n = 2, 3, 2
    A = torch.randn(n, m)  # fixed weight matrix (not trainable here)

    # ⚠️ Treating the inputs as trainable parameters
    x_b = torch.randn(B, m, requires_grad=True)

    # Forward: batch_output = x_b @ A^T
    batch_output = x_b @ A.t()

    # Upstream gradient (same shape as batch_output)
    v_b = torch.tensor([[1.0, 0.0], [0.5, -2.0]], dtype=torch.float32)
    batch_output.backward(v_b)

    # Expected: grad wrt x_b = v_b @ A
    expected = v_b @ A
    print("=== Trainable Inputs Demo ===")
    print("A:\n", A)
    print("x_b.grad (autograd):\n", x_b.grad)
    print("Expected v_b @ A:\n", expected)
    print()

if __name__ == "__main__":
    main()