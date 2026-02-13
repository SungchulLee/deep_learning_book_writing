import torch
import pandas as pd
import numpy as np

def print_info(t):
    # Quick inspector for a tensor:
    # - t: prints values with dtype-aware formatting
    # - t.shape: rank/size; [] means scalar, [N] 1-D, [R,C] 2-D, etc.
    # - t.dtype: inferred or forced; NumPy float64 → torch.float64 by default
    # - requires_grad: autograd flag (False unless you set True on float/complex)
    print(f"{t = }", f"{t.shape = }", f"{t.dtype = }", f"{t.requires_grad = }", sep="\n", end="\n\n")

def main():
    # --------------------------------------------
    # 1) Pandas Series[int] → Tensor  (**COPY**)
    # --------------------------------------------
    # s.values / s.to_numpy(...) → NumPy array; torch.tensor(...) then **COPIES**.
    s1 = pd.Series([1, 2, 3, 4, 5])
    t1 = torch.tensor(s1.values)   # COPY (independent storage)
    print_info(t1)
    # Expect: tensor([1, 2, 3, 4, 5])   dtype=torch.int64

    # --------------------------------------------
    # 2) Pandas Series[float] → Tensor  (**COPY**)
    # --------------------------------------------
    # Pandas/NumPy default float is float64 → torch.float64 unless overridden.
    s2 = pd.Series([0.1, 0.2, 0.3])
    t2 = torch.tensor(s2.values)   # COPY (dtype follows NumPy, likely float64)
    print_info(t2)

    # --------------------------------------------
    # 3) Explicit dtype conversion  (**COPY**)
    # --------------------------------------------
    # Best practice when you care about precision/perf: be explicit.
    t3 = torch.tensor(s2.values, dtype=torch.float32)  # COPY (float32)
    print_info(t3)

    # --------------------------------------------
    # 4) Boolean Series → torch.bool Tensor  (**COPY**)
    # --------------------------------------------
    s4 = pd.Series([True, False, True])
    t4 = torch.tensor(s4.values)   # COPY
    print_info(t4)
    # Expect: tensor([ True, False,  True])   dtype=torch.bool

    # --------------------------------------------
    # 5) Shared memory via torch.from_numpy  (**SHARE**)
    # --------------------------------------------
    # torch.from_numpy(ndarray) **SHARES** storage with the ndarray (no copy).
    # Mutating either reflects in the other (requirements: numeric, writable, supported layout).
    arr = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    s5 = pd.Series(arr)                    # wraps the SAME ndarray (no copy)
    t5 = torch.from_numpy(s5.values)       # SHARE (no copy)
    print_info(t5)

    arr[0] = 99.0   # mutate underlying NumPy array
    print("   NumPy arr after:", arr)
    print("   Tensor after    :", t5)  # reflects change (shared memory)

    # TIP:
    # - Need independence?  t5_ind = torch.from_numpy(s5.values).clone()  # COPY after share
    # - If Series isn't writable/contiguous, .from_numpy may error → use s.to_numpy(..., copy=True).

    # --------------------------------------------
    # 6) Non-numeric Series (object dtype) → ERROR
    # --------------------------------------------
    try:
        s6 = pd.Series(["a", "b", "c"])
        torch.tensor(s6.values)  # object dtype → ValueError / TypeError
    except Exception as e:
        print("Non-numeric Series error:", e)

    # ---------------------- Notes (COPY / SHARE / TRY-TO-SHARE) ----------------------
    # • Pandas → NumPy:
    #     s.to_numpy(dtype=..., copy=False)     # may SHARE with backing data (no copy) or make a view
    #     s.values                               # same idea as to_numpy(); use to_numpy for explicit control
    #
    # • NumPy → Torch:
    #     torch.tensor(ndarray)        → **COPY** (always new, independent storage)
    #     torch.from_numpy(ndarray)    → **SHARE** (no copy; changes reflect both ways)
    #     torch.as_tensor(ndarray)     → **TRY TO SHARE** (shares if compatible: numeric, writable,
    #                                        supported strides; otherwise falls back to COPY)
    #
    # • With Python lists/tuples (not NumPy):
    #     torch.tensor(list_like)      → **COPY**
    #     torch.as_tensor(list_like)   → **COPY** (nothing to share)
    #
    # • Autograd:
    #     Newly created tensors have requires_grad=False.
    #     Set requires_grad=True on floating/complex tensors if you’ll do backprop.
    #
    # • Device/dtype:
    #     Prefer explicit dtype (e.g., float32 for training). Move device as needed:
    #         t = torch.from_numpy(arr).to("cuda")   # share on CPU first, then COPY to GPU
    #         t = torch.tensor(df.to_numpy(np.float32), device="cuda")  # direct **COPY** to GPU

if __name__ == "__main__":
    main()
