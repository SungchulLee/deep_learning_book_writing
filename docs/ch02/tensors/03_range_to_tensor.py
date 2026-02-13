import torch

def print_info(t):
    # Helper to inspect a tensor quickly.
    # - t: value preview (PyTorch pretty-prints based on dtype)
    # - t.shape: [N] for 1-D, [R,C] for 2-D, [] for scalar (0-D)
    # - t.dtype: inferred unless specified (ints→int64, floats→float32 by default)
    # - requires_grad: autograd flag (False unless you set True on float/complex tensors)
    print(f"{t = }", f"{t.shape = }", f"{t.dtype = }", f"{t.requires_grad = }", sep="\n", end="\n\n")

def main():
    # --------------------------------------------
    # 1) Python range  →  Tensor (COPY)
    # --------------------------------------------
    # torch.tensor(range(...)) copies elements from the iterable.
    # Dtype is inferred; all ints → torch.int64 by default.
    r1 = range(5)   # [0, 1, 2, 3, 4]
    t1 = torch.tensor(r1)
    print_info(t1)
    # Expect: tensor([0, 1, 2, 3, 4])   torch.Size([5])   torch.int64

    # --------------------------------------------
    # 2) range with (start, stop, step)
    # --------------------------------------------
    # Python’s range is half-open: includes start, excludes stop.
    r2 = range(2, 10, 2)  # [2, 4, 6, 8]
    t2 = torch.tensor(r2)
    print_info(t2)
    # Expect: tensor([2, 4, 6, 8])   torch.Size([4])   torch.int64

    # --------------------------------------------
    # 3) torch.arange  — preferred over wrapping range
    # --------------------------------------------
    # Creates the tensor directly (can set dtype/device/requires_grad).
    # Like Python range: half-open [start, stop).
    t3 = torch.arange(0, 10, 2)
    print_info(t3)
    # Expect: tensor([0, 2, 4, 6, 8])   torch.Size([5])   torch.int64

    # --------------------------------------------
    # 4) torch.arange with float step
    # --------------------------------------------
    # Floating steps may accumulate rounding error; result is still half-open.
    # Default float dtype is float32 unless set via dtype=...
    t4 = torch.arange(0.0, 1.0, 0.2)
    print_info(t4)
    # Example output: tensor([0.0000, 0.2000, 0.4000, 0.6000, 0.8000])

    # --------------------------------------------
    # 5) torch.linspace — includes BOTH endpoints
    # --------------------------------------------
    # Returns `steps` evenly spaced points from start to end (inclusive).
    # This differs from arange’s half-open behavior.
    t5 = torch.linspace(0, 1, steps=5)
    print_info(t5)
    # Expect: tensor([0.0000, 0.2500, 0.5000, 0.7500, 1.0000])

    # --------------------------------------------
    # 6) torch.range — deprecated (use arange instead)
    # --------------------------------------------
    # Historically existed, but deprecated due to ambiguity; prefer torch.arange.
    # Here we show the *replacement* call to avoid deprecation warnings.
    t6 = torch.arange(1, 5)  # [1, 2, 3, 4]
    print_info(t6)

    # ---------------------- Extras (FYI) ----------------------
    # • You can control dtype/device directly with torch.arange/linspace:
    #     torch.arange(0, 10, 2, dtype=torch.float32, device='cpu', requires_grad=False)
    # • For evenly spaced *counts* (include both ends), prefer linspace.
    #   For step-based sequences (exclude stop), prefer arange.
    # • For floating steps where you care about the last value landing exactly
    #   on the endpoint, linspace is usually the safer choice.

if __name__ == "__main__":
    main()