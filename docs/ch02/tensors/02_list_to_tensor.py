import torch

def print_info(t):
    # Helper to inspect common attributes of a tensor.
    # - t            : prints values (PyTorch pretty-prints by dtype)
    # - t.shape      : shows rank/lengths; [] means a 0-D scalar, [N] 1-D, [R,C] 2-D, etc.
    # - t.dtype      : inferred unless explicitly provided (ints→int64, floats→float32 by default)
    # - requires_grad: autograd tracking flag (False by default; only meaningful for float/complex)
    print(f"{t = }", f"{t.shape = }", f"{t.dtype = }", f"{t.requires_grad = }", sep="\n", end="\n\n")

def main():
    # --------------------------------------------
    # 1) 1D Python list  →  1D Tensor (COPY)
    # --------------------------------------------
    # torch.tensor(...) **copies** data from Python sequences into a brand-new tensor.
    # The resulting dtype is inferred: a list of floats → float32 by default.
    list1 = [1.0, 2.0, 3.0]
    t1 = torch.tensor(list1)   # copy data from list → independent storage
    print_info(t1)
    # Expect: tensor([1., 2., 3.])   torch.Size([3])   torch.float32

    # --------------------------------------------
    # 2) Nested (rectangular) list  →  multi-dimensional Tensor
    # --------------------------------------------
    # All inner lists must have the **same length** (rectangular). Otherwise, it’s ragged.
    # Integer values infer dtype=int64 by default.
    list2 = [[1, 2, 3], [4, 5, 6]]
    t2 = torch.tensor(list2)
    print_info(t2)
    # Expect: tensor([[1, 2, 3],
    #                 [4, 5, 6]])   torch.Size([2, 3])   torch.int64

    # --------------------------------------------
    # 3) Explicit dtype specification
    # --------------------------------------------
    # You can override the inference. Here we force float64 (double precision).
    t3 = torch.tensor(list1, dtype=torch.float64)
    print_info(t3)

    # --------------------------------------------
    # 4) Mixed types in a list  →  dtype promotion
    # --------------------------------------------
    # PyTorch promotes to a common dtype that can represent all values.
    # int + float → float (default float32 unless you force otherwise).
    list4 = [1, 2.5, 3]  # int + float
    t4 = torch.tensor(list4)  # auto-promotes to float
    print_info(t4)
    # Expect dtype: torch.float32

    # --------------------------------------------
    # 5) Empty list  →  empty 1D tensor (length 0)
    # --------------------------------------------
    # Default floating dtype (float32). Shape is [0].
    empty_list = []
    t5 = torch.tensor(empty_list)
    print_info(t5)
    # Expect: tensor([])   torch.Size([0])   torch.float32

    # --------------------------------------------
    # 6) Boolean list  →  torch.bool tensor
    # --------------------------------------------
    # Useful for masks and indexing.
    bool_list = [True, False, True]
    t6 = torch.tensor(bool_list)
    print_info(t6)
    # Expect: tensor([ True, False,  True])   dtype=torch.bool

    # --------------------------------------------
    # 7) Ragged (non-rectangular) nested lists raise an error
    # --------------------------------------------
    # Inner lists have different lengths → PyTorch cannot form a proper tensor shape.
    try:
        ragged = [[1, 2], [3, 4, 5]]
        torch.tensor(ragged)  # inconsistent inner lengths → ValueError
    except Exception as e:
        print("Ragged list error:", e)

    # ---------------------- Extras (FYI) ----------------------
    # • COPY vs SHARE with NumPy:
    #     - torch.tensor(np_array)      → **ALWAYS copies** (new, independent storage).
    #     - torch.as_tensor(np_array)   → **tries to avoid copy** (often shares like from_numpy
    #                                    if dtype/stride/writable allow; otherwise copies).
    #     - torch.from_numpy(np_array)  → **ALWAYS shares** memory (no copy; mutations reflect both ways).
    # • With **Python lists/tuples** (not NumPy):
    #     - torch.tensor(list_like)     → copies (as used above).
    #     - torch.as_tensor(list_like)  → still copies (there’s nothing to share).
    # • requires_grad defaults to False for all creations above. Set requires_grad=True on floating
    #   tensors if you want autograd to track operations for backprop.
    # • For device placement, pass device=... (e.g., device='cuda') when creating the tensor.

if __name__ == "__main__":
    main()