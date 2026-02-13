import torch

def main():

    torch.manual_seed(0)

    # Tiny dataset: linear regression toy example
    # Input-output pairs: x ∈ {-1, 0, 1}, target y ∈ {1, 0, -1}
    x = torch.tensor([[-1.0], [0.0], [1.0]])  # shape (3,1)
    y = torch.tensor([[1.0], [0.0], [-1.0]])  # shape (3,1)

    # Trainable parameters (leaf tensors).
    # Leaf tensors are those created by the user with requires_grad=True.
    # Autograd computes gradients for these directly.
    w = torch.randn(1, requires_grad=True)  # dtype is torch.float32 by default
    b = torch.zeros(1, requires_grad=True)  # dtype is torch.float32 by default

    lr = 0.1
    for step in range(3):
        # ---------------- Forward pass ----------------
        y_hat = x * w + b                     # linear model, shape (3,1)
        loss = torch.mean((y_hat - y) ** 2)   # mean squared error (scalar)
        # Note:
        #   - loss is a 0-dim tensor (scalar tensor).
        #   - loss.grad_fn = <MeanBackward0>, because it was produced
        #     by a mean operation in the computation graph.
        #   - grad_fn points to the function that knows how to compute
        #     the gradient of this tensor w.r.t. its parents.
        #   - Leaf tensors (like w, b) have grad_fn=None.

        # ---------------- Backward pass ----------------
        # Gradients accumulate by default, so clear old grads first.
        # Use .zero_() (in-place) to reset gradients to None, not 0. 
        # Resetting gradients to None, not 0 is the default behavior of zero_().
        # PyTorch does not have .zero() (only .zero_()).
        if w.grad is not None: w.grad.zero_()
        if b.grad is not None: b.grad.zero_()
        print(f"{w.grad = }")

        # Computes d(loss)/dw and d(loss)/db by traversing the computation graph
        # from loss.backward() → grad_fn → parent tensors → chain rule.
        loss.backward()

        # ---------------- Parameter update ----------------
        # IMPORTANT: wrap in torch.no_grad() so autograd does not
        # record these in-place updates in the computation graph.
        with torch.no_grad():
            w -= lr * w.grad
            b -= lr * b.grad

        # ---------------- Logging ----------------
        # Why .item()? Because:
        #   - loss, w, b are tensors (scalars with shape []).
        #   - Printing a tensor directly shows "tensor(..., grad_fn=...)".
        #   - .item() extracts the raw Python float, making formatting like
        #     :.6f or :.4f work without error.
        print(f"step {step}: loss={loss.item():.6f} | w={w.item():.4f} | b={b.item():.4f}")

    # Final learned parameters
    print("Final params:", {"w": w.item(), "b": b.item()})
    # - About loss.item():
    #   * loss.item() **returns a Python float on the CPU**. If `loss` lives on GPU/MPS,
    #     calling .item() triggers a **device→host copy and a synchronization barrier**
    #     (the host waits until all preceding GPU ops that produce `loss` complete).
    #   * This stalls the GPU pipeline, so in tight training loops it’s better to **avoid
    #     calling .item() every step**. Prefer:
    #       - Log less frequently (e.g., every k steps).
    #       - Accumulate a tensor value and call `.detach().cpu()` outside the hot path.
    #       - For averages: keep a running sum as a CPU float **after** occasional syncs.
    #   * Use .item() when you truly need a Python float (printing, early stopping checks),
    #     but be mindful of the sync point it creates.

if __name__ == "__main__":
    main()