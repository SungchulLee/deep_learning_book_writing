import torch

def print_info(t):
    """Pretty printer for quick inspection.

    Shows:
      - `t` itself  : value display depends on dtype/precision
      - `t.shape`   : `torch.Size([])` means a scalar (rank 0)
      - `t.dtype`   : inferred from the Python value unless specified
      - `t.requires_grad` : whether autograd will track ops on `t`
    """
    print(f"{t = }", f"{t.shape = }", f"{t.dtype = }", f"{t.requires_grad = }",
          sep="\n", end="\n\n")

def main():
    # --------------------------------------------
    # 1) Wrap a Python int directly → scalar tensor
    # --------------------------------------------
    scalar_val = 42
    t1 = torch.tensor(scalar_val)  # dtype inferred → torch.int64
    print_info(t1)
    # Result: tensor(42), shape=[], dtype=int64 → a true scalar (rank 0).
    # Integer tensors cannot require gradients (autograd works on float/complex).

    # --------------------------------------------
    # 2) Same, but force a dtype (float32 here)
    # --------------------------------------------
    t2 = torch.tensor(scalar_val, dtype=torch.float32)
    print_info(t2)
    # Still a scalar, now float32. This enables autograd *if* you set requires_grad=True.
    # e.g., torch.tensor(scalar_val, dtype=torch.float32, requires_grad=True) would
    # create a **leaf** scalar that participates in gradient computation.

    # --------------------------------------------
    # 3) Put the scalar in a list → NOT a scalar anymore
    # --------------------------------------------
    t3 = torch.tensor([scalar_val])
    print_info(t3)
    # Shape is [1]. This is rank-1 (length-1 vector), not rank-0.

    # --------------------------------------------
    # 4) torch.scalar_tensor: convenient alias for scalar input
    # --------------------------------------------
    t4 = torch.scalar_tensor(scalar_val)
    print_info(t4)
    # Equivalent to torch.tensor(scalar_val) for scalar inputs (dtype inferred).

    # --------------------------------------------
    # 5) From a Python float → dtype defaults to float32
    # --------------------------------------------
    float_val = 3.14
    t5 = torch.tensor(float_val)  # default float dtype is float32
    print_info(t5)

    # --------------------------------------------
    # 6) Convert a 1-element tensor to a Python scalar and back
    # --------------------------------------------
    vec = torch.tensor([10])
    scalar_extracted = vec.item()   # works only when numel()==1
    t6 = torch.tensor(scalar_extracted)  # back to a scalar tensor
    print_info(t6)
    # ❓ Does `item()` work for rank-0/1/2...?
    # • **Yes if and only if the tensor has exactly one element**:
    #     OK: shape [], [1], [1,1], ... (numel()==1)
    #     ERROR: shape [2], [1,2], ... (numel()>1)

    # Mini demo: item() success and failure
    ok1 = torch.tensor(7)          # shape []
    ok2 = torch.tensor([[7]])      # shape [1,1]
    bad = torch.tensor([1, 2])     # shape [2]
    _ = ok1.item()                 # OK
    _ = ok2.item()                 # OK (still one element)
    try:
        _ = bad.item()             # ValueError: only one element tensors can be converted
    except ValueError as e:
        print("item() on multi-element tensor →", e, "\n")

    # --------------------------------------------
    # 7) Make a scalar via torch.full with empty shape `()`
    # --------------------------------------------
    t7 = torch.full((), 7.7)  # empty shape → rank-0 scalar
    print_info(t7)

    # --------------------------------------------
    # 8) Enable autograd explicitly (for float/complex tensors)
    # --------------------------------------------
    t8 = torch.tensor(5.0, requires_grad=True)  # leaf scalar with grad tracking
    print_info(t8)
    # This scalar now participates in autograd. Note: requires_grad=True is valid only
    # for floating/complex dtypes (not integers).

    # Mini demo: backprop through a float scalar
    y = 0.5 * (t8 ** 2)  # y = 1/2 x^2
    y.backward()         # dy/dx = x
    print("t8:", t8.item(), "requires_grad:", t8.requires_grad)
    print("t8.grad (expected 5.0):", t8.grad.item(), "\n")

if __name__ == "__main__":
    main()