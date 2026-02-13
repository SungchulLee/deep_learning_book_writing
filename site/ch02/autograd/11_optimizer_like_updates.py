import torch

def main():

    w = torch.randn(3, requires_grad=True)
    lr = 0.1
    for step in range(3):
        # ---------------- Forward pass ----------------
        # Simple quadratic loss = sum of squares.
        # Since w.requires_grad=True, this loss tensor also requires grad
        # and has a grad_fn (<SumBackward0>) so autograd knows how to
        # compute ∂loss/∂w when we call backward().
        loss = (w ** 2).sum()

        # ---------------- Gradient clearing ----------------
        # By default, PyTorch ACCUMULATES gradients into .grad buffers.
        # So we must clear them explicitly before calling backward() again.
        # w.grad.zero_() is SAFE:
        #   - .grad is just a buffer tensor, not part of the computation graph.
        #   - In-place ops on .grad do not confuse autograd.
        #   - No need for torch.no_grad() here.
        if w.grad is not None:
            w.grad.zero_()

        # ---------------- Backward pass ----------------
        # Computes d(loss)/dw and stores result in w.grad.
        # Internally this is done by traversing the graph via loss.grad_fn.
        loss.backward()

        # ---------------- Parameter update ----------------
        # ❌ WRONG: w -= lr * w.grad   (outside no_grad)
        #   - Would be recorded as part of the graph (grad_fn=<SubBackward0>).
        #   - w would stop being a "leaf" tensor and .grad would no longer update correctly.
        #   - Graph would grow every step → memory leaks.
        #
        # ✅ RIGHT: wrap in torch.no_grad()
        #   - Temporarily disables gradient tracking.
        #   - In-place update happens, but is excluded from the graph.
        #   - w remains a leaf tensor with requires_grad=True.
        with torch.no_grad():
            w -= lr * w.grad

        # Use .item() for scalar loss so formatting works (:.6f).
        print(f"step {step} | loss={loss.item():.6f} | w={w}")

    # ---------------- Post-training notes ----------------
    # - w.requires_grad is STILL True: w is still trainable.
    # - The updates were skipped by autograd because of no_grad().
    # - This is exactly how torch.optim.SGD / Adam implement .step().
    print("Final w.requires_grad (still True, updates not tracked):", w.requires_grad)

if __name__ == "__main__":
    main()