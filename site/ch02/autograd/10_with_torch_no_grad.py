import torch

def main():

    x = torch.randn(3, requires_grad=True)

    # Normal tracking:
    # ----------------
    # Since x has requires_grad=True, any operation involving x will also
    # produce a tensor that requires_grad=True. Autograd records a grad_fn
    # for non-leaf results (like y below) so it can compute gradients later.
    y = (x ** 2).sum()
    print("y.requires_grad (expect True):", y.requires_grad)

    # Scoped no-grad context:
    # -----------------------
    # with torch.no_grad(): temporarily DISABLES autograd recording.
    # - Any new tensors created inside this block default to requires_grad=False.
    # - Even if they are built from tensors that normally require gradients.
    # - This is why optimizers wrap parameter updates in no_grad: so that
    #   w -= lr * w.grad does not become part of the computation graph.
    #
    # NOTE:
    # - This differs from x.grad.zero_():
    #     * x.grad.zero_() only clears the gradient buffer (safe, not tracked).
    #     * Updates to x itself (e.g. x -= lr * x.grad) MUST be inside no_grad()
    #       or autograd will think the update op is part of the graph, causing
    #       memory leaks and breaking leaf tensor rules.
    with torch.no_grad():
        y_ng = (x ** 2).sum()
        print("Inside no_grad → y_ng.requires_grad (expect False):", y_ng.requires_grad)

    # Exiting the no_grad() context:
    # ------------------------------
    # After the block, autograd tracking is automatically restored.
    # Now new ops on x again require gradients, and grad_fn is attached.
    z = (x + 1).sum()
    print("After block → z.requires_grad (expect True):", z.requires_grad)

if __name__ == "__main__":
    main()
