import torch

def main():

    torch.manual_seed(2)

    # Define a small nonlinear mapping f: R^2 -> R^2
    def f(x):
        # x: shape [2]
        # Returns a 2D vector output depending on both components of x
        return torch.stack([
            torch.sin(x[0]) + x[1] ** 2,   # first component
            torch.exp(x[0]) - torch.cos(x[1])  # second component
        ])

    x = torch.tensor([0.3, -0.7], requires_grad=True)
    y = f(x)  # output is a vector (shape [2])
    print("x:", x)
    print("y = f(x):", y)

    # To get a gradient for vector outputs, we must supply a "v" vector.
    # This computes the vector-Jacobian product: v^T J(x).
    v = torch.tensor([0.2, -1.5], dtype=torch.float32)
    y.backward(v)  # backward on vector → computes v^T J wrt x
    print("v:", v)
    print("x.grad (v^T J):", x.grad)

    # For verification: explicitly compute the full Jacobian J(x).
    # Then compare autograd's v^T J result with manual v @ J.
    # Note: Jacobian construction is costly for larger problems.
    try:
        from torch.autograd.functional import jacobian

        def f_for_jac(x_vec):
            return f(x_vec)  # required to wrap for jacobian API

        J = jacobian(f_for_jac, x.detach().requires_grad_(True))  # shape (2,2)
        with torch.no_grad():
            vT_J = v @ J  # shape (2,) after multiplication
        print("Full Jacobian J:\n", J)
        print("v^T J via full J:", vT_J)
        print("Difference |x.grad - (v^T J)|:", (x.grad - vT_J).abs())
    except Exception as e:
        print("Could not compute full Jacobian (version/platform issue):", e)

if __name__ == "__main__":
    main()