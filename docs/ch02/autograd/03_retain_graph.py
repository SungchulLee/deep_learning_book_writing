import torch

def main():

    torch.manual_seed(0)

    x = torch.randn(3, requires_grad=True)
    print("x:", x)

    # Build one computation graph once (loss = sum of squares).
    loss = (x ** 2).sum()

    # First backward: computes grads and fills x.grad.
    # retain_graph=True tells PyTorch NOT to free the graph after use.
    # This allows another backward on the SAME graph.
    loss.backward(retain_graph=True)
    print("After 1st backward, x.grad:", x.grad)

    # Second backward on the SAME graph (no new forward here).
    # Because we retained the graph above, this call is valid.
    # Here retain_graph=False (default), so after this second backward
    # the graph WILL be freed and cannot be used again.
    # Gradients accumulate into x.grad by default.
    loss.backward()
    print("After 2nd backward (accumulated), x.grad:", x.grad)

    # Clear grads so we can see a clean slate before the next computation.
    # IMPORTANT:
    #   - PyTorch tensors implement .zero_(), which works IN-PLACE.
    #   - There is NO .zero() method on tensors; calling it would raise AttributeError.
    #   - The trailing underscore convention (_): means the operation mutates the tensor.
    #   - This is the standard, idiomatic way to reset gradients.
    x.grad.zero_()
    print("After zero_, x.grad:", x.grad)

    # Fresh forward pass: builds a brand new computation graph.
    # Backward now computes grads again, starting from zero.
    loss = (x ** 2).sum()
    loss.backward()
    print("After 3rd backward (from zero), x.grad:", x.grad)

    # Note:
    #   - If you want to reset ALL model parameters' gradients in one call,
    #     use optimizer.zero_grad().
    #   - optimizer.zero_grad(set_to_none=True) sets grads to None instead of 0.
    #     This can save memory and avoid some subtle accumulation issues.
    #   - But for manual per-tensor clearing, always use .zero_().

if __name__ == "__main__":
    main()