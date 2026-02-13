import torch


def header(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def ptr(t: torch.Tensor) -> int:
    """Return the base storage data pointer (decimal).

    • Points to the beginning of the underlying storage buffer (not the logical
      first element if the tensor is a view with offset/strides).
    • If two tensors share the *same* storage, their data_ptr() values will be
      equal (even if their shapes/strides differ).
    """
    return t.storage().data_ptr()


def main():
    torch.manual_seed(0)

    # ----------------------------------------------------------------------------
    # 0) Setup a base tensor
    # ----------------------------------------------------------------------------
    # requires_grad_(True) flips the flag in-place so autograd tracks ops on `base`.
    base = torch.arange(1, 7, dtype=torch.float32).reshape(2, 3).requires_grad_(True)
    header("Base tensor")
    print("base:\n", base)
    print("base.requires_grad:", base.requires_grad)
    print("ptr(base):", ptr(base))

    # ----------------------------------------------------------------------------
    # 1) Plain Python assignment: NO COPY (just another reference)
    # ----------------------------------------------------------------------------
    header("1) Plain assignment: alias reference (NO COPY)")
    # `alias` and `base` are the exact same Python object → same storage, same grad flag.
    alias = base
    print("alias is base?       ", alias is base)     # True (same object identity)
    print("ptr(alias) == ptr(base)?", ptr(alias) == ptr(base))

    # Any in-place change through one name is visible via the other (same object).
    base.add_(100)
    print("\nAfter base.add_(100):")
    print("base:\n", base)
    print("alias (same object):\n", alias)

    # Revert for next demos (in-place subtraction).
    base.sub_(100)

    # ----------------------------------------------------------------------------
    # 2) Views (share storage): slicing / view / reshape
    # ----------------------------------------------------------------------------
    header("2) Views that SHARE storage (slicing / view / reshape)")
    # Many shape/stride transformations return *views* that alias the same storage.
    view_slice = base[:, 1:]          # slice → shares when possible
    view_view  = base.view(2, 3)      # view with same shape → shares
    view_resh  = base.reshape(2, 3)   # may return a view; may allocate if needed

    print("ptr(view_slice):", ptr(view_slice))
    print("ptr(view_view) :", ptr(view_view))
    print("ptr(view_resh) :", ptr(view_resh))
    print("All share storage with base? ->",
          ptr(view_slice) == ptr(base) and ptr(view_view) == ptr(base))

    # In-place change via a view updates the base (shared storage).
    view_slice.mul_(10)
    print("\nAfter view_slice.mul_(10):")
    print("base:\n", base)
    print("view_slice:\n", view_slice)

    # Revert for next parts.
    view_slice.div_(10)

    # ----------------------------------------------------------------------------
    # 3) clone(): deep copy of the underlying data (NO storage sharing)
    # ----------------------------------------------------------------------------
    header("3) .clone(): DEEP COPY (no storage sharing)")
    # clone() creates a new tensor with its own storage; autograd graph is preserved.
    c = base.clone()
    print("ptr(clone):", ptr(c), "  ptr(base):", ptr(base))
    print("Shares storage? ->", ptr(c) == ptr(base))

    # In-place change on base does NOT affect the clone (independent buffers).
    base.add_(1000)
    print("\nAfter base.add_(1000):")
    print("base:\n", base)
    print("clone (unchanged):\n", c)

    # Revert
    base.sub_(1000)

    # ----------------------------------------------------------------------------
    # 4) detach(): shares storage but breaks grad tracking
    # ----------------------------------------------------------------------------
    header("4) .detach(): shares storage, stops grad")
    # detach() returns a tensor that aliases the same storage but with requires_grad=False
    # and no grad_fn relationship to the original graph.
    d = base.detach()
    print("d.requires_grad:", d.requires_grad)
    print("ptr(detach) == ptr(base)?", ptr(d) == ptr(base))

    # In-place change on base is visible in d (shared storage).
    base.add_(5)
    print("\nAfter base.add_(5):")
    print("base:\n", base)
    print("detach (reflects change):\n", d)

    # Revert
    base.sub_(5)

    # ----------------------------------------------------------------------------
    # 5) detach().clone(): breaks grad + NO storage sharing
    # ----------------------------------------------------------------------------
    header("5) .detach().clone(): no grad + deep copy")
    # Pattern for “safe snapshot” not connected to autograd and independent memory.
    dc = base.detach().clone()
    print("dc.requires_grad:", dc.requires_grad)
    print("ptr(detach().clone) == ptr(base)?", ptr(dc) == ptr(base))

    # In-place change on base does NOT affect dc (independent).
    base.mul_(2)
    print("\nAfter base.mul_(2):")
    print("base:\n", base)
    print("detach().clone (unchanged):\n", dc)

    # Revert (divide by 2)
    base.div_(2)

    # ----------------------------------------------------------------------------
    # 6) Autograd note: clone keeps gradient flow; detach does not
    # ----------------------------------------------------------------------------
    header("6) Autograd note: clone vs detach")
    x = torch.ones(3, requires_grad=True)

    # clone(): preserves the computation graph; gradients can flow back to `x`.
    y_clone = x.clone() * 3.0  # grad_fn=MulBackward; clone keeps graph connectivity

    # detach(): severs the graph; subsequent ops won’t contribute to x.grad.
    y_detach = x.detach() * 3.0  # computed from a leaf with requires_grad=False

    y_clone.sum().backward()   # d/dx of (sum(3*x)) = 3
    print("x.grad from clone-path:", x.grad)  # tensor([3., 3., 3.])

    x.grad.zero_()
    try:
        y_detach.sum().backward()
    except RuntimeError as e:
        # Backward through a graph that doesn’t include x → no grad for x.
        print("backward on detach path raised:", e)

    # ----------------------------------------------------------------------------
    # 7) In-place ops and shared storages: be careful
    # ----------------------------------------------------------------------------
    header("7) In-place ops can silently affect ALL tensors sharing the storage")
    # Any in-place operation on a tensor affects every view/alias that shares storage.
    a = torch.tensor([1., 2., 3.], requires_grad=True)
    v = a[1:]      # view: shares storage (elements a[1], a[2])
    c = a.clone()  # independent copy

    print("Before in-place on view:")
    print("a:", a, " ptr:", ptr(a))
    print("v:", v, " ptr:", ptr(v))
    print("c:", c, " ptr:", ptr(c))

    v.add_(100)  # in-place on the view → updates shared positions in `a`
    print("\nAfter v.add_(100):")
    print("a (affected):", a)  # a[1], a[2] changed
    print("v (view):    ", v)
    print("c (clone):   ", c)  # unchanged (separate storage)

    # ----------------------------------------------------------------------------
    # 8) Quick summary
    # ----------------------------------------------------------------------------
    header("8) Summary")
    print(
        "• alias = base           : NO COPY, same Python object & storage\n"
        "• view/slice/reshape     : SHARE storage (when possible)\n"
        "• clone()                : COPY, independent storage; keeps autograd link\n"
        "• detach()               : SHARE storage; breaks autograd link\n"
        "• detach().clone()       : COPY + no grad (safe snapshot)\n"
        "• In-place ops affect ALL tensors sharing the storage; use with care.\n"
    )


if __name__ == "__main__":
    main()