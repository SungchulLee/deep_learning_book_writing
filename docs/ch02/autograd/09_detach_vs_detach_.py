import torch

def main():
    # -----------------------------------------------------------------------------
    # Autograd & detaching — quick rules of thumb
    # -----------------------------------------------------------------------------
    # • Leaf tensor: created by the user (not the result of an autograd op).
    #   - has grad_fn=None
    #   - can toggle requires_grad with .requires_grad_(True/False)
    #
    # • Non-leaf tensor: result of ops on tensors that require grad.
    #   - has a grad_fn like <AddBackward0>, <SinBackward>, etc.
    #   - CANNOT toggle requires_grad_ (will error) but CAN be detached.
    #
    # • Detach variants:
    #   - t.detach()  → returns a NEW tensor that shares storage but is not tracked
    #                   (requires_grad=False, grad_fn=None).
    #   - t.detach_() → IN-PLACE: mutates t itself to stop tracking
    #                   (requires_grad=False, grad_fn=None). Allowed on non-leaf too.
    #
    # • To “start fresh” tracking from a given value:
    #       t = t.detach().requires_grad_(True)   # new leaf from here on
    # -----------------------------------------------------------------------------

    a = torch.linspace(-1, 1, steps=5, requires_grad=True)
    # a is a LEAF tensor because it was created directly by the user
    # with requires_grad=True. Leaf tensors always have grad_fn=None,
    # even though autograd will compute their gradients during backward.
    print("a.requires_grad:", a.requires_grad, "| a.grad_fn:", a.grad_fn)  # grad_fn=None

    b = a * a + 1.0
    # b is computed from a, so b.requires_grad=True.
    # BUT b is NON-leaf:
    #   - b.grad_fn is NOT the function that computed b's forward value.
    #   - Instead, it’s the backward function object (e.g. <AddBackward0>)
    #     that knows how to compute ∂b/∂a when backpropagating.
    print("b.requires_grad:", b.requires_grad, "| b.grad_fn:", b.grad_fn)

    # detach(): creates a NEW tensor that shares storage with b
    # but is not connected to the computation graph:
    #   - requires_grad=False
    #   - grad_fn=None
    # Use this when you want a read-only view wrt autograd without mutating b.
    b_det = b.detach()
    print("\n--- Detach example ---")
    print("b_det.requires_grad (expect False):", b_det.requires_grad)
    print("b_det.grad_fn (expect None):", b_det.grad_fn)
    print("b_det is b? (expect False):", b_det is b)  # new object

    # detach_(): IN-PLACE version of detach()
    # This mutates the tensor itself so that autograd no longer tracks it.
    # After this, the tensor has requires_grad=False and grad_fn=None.
    # Safe & valid on NON-leaf tensors too.
    b2 = (a + 3).sin()
    print("\n--- In-place detach_ example ---")
    print("Before detach_(): b2.requires_grad:", b2.requires_grad, "| b2.grad_fn:", b2.grad_fn)
    b2.detach_()  # in-place: removes gradient tracking (now behaves like a leaf going forward)
    print("After  detach_(): b2.requires_grad:", b2.requires_grad, "| b2.grad_fn:", b2.grad_fn)

    # If you later want to re-enable tracking from this value, re-leaf it:
    # b2 = b2.detach().requires_grad_(True)

if __name__ == "__main__":
    main()