#!/usr/bin/env python3
"""
Tensor memory layout & view/copy behavior.

Covers:
- transpose()/permute() views, stride changes, and .contiguous()
- view() vs reshape() vs clone()
- expand() vs repeat()
- Simple gather()/scatter_() mini-demo (indexing-based movement)
"""

import torch

def header(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

def main():
    torch.manual_seed(0)

    # -------------------------------------------------------------------------
    header("transpose()/permute() return VIEWS (stride changes)")
    t = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    T = t.transpose(0, 1)     # (4, 3), view
    P = t.permute(1, 0)       # (4, 3), view
    print("t:\n", t)
    print("T = t.transpose(0,1):\n", T, "| is_contiguous:", T.is_contiguous(), "| stride:", T.stride())
    print("P = t.permute(1,0):\n", P, "| is_contiguous:", P.is_contiguous(), "| stride:", P.stride())

    # -------------------------------------------------------------------------
    header("Make contiguous copy: .contiguous()")
    Tc = T.contiguous()
    print("Tc.is_contiguous:", Tc.is_contiguous(), "| Tc.stride:", Tc.stride())

    # -------------------------------------------------------------------------
    header("view() vs reshape() vs clone()")
    v = t.view(12)            # view (when possible) → shares storage
    r = t.reshape(6, 2)       # may return view or copy
    c = t.clone()             # always copy
    print("view shares storage:", id(t.storage()) == id(v.storage()))
    print("clone shares storage:", id(t.storage()) == id(c.storage()))
    print("Before in-place, v[:5]:", v[:5])
    t[0, 0] = -999
    print("After  in-place, v[:5]:", v[:5])
    print("Clone first row (unchanged):", c[0])

    # -------------------------------------------------------------------------
    header("expand() vs repeat()")
    b = torch.tensor([1., 2., 3.])   # (3,)
    print("b.expand(2,3):\n", b.expand(2, 3))  # view (no alloc): stride tricks
    print("b.repeat(2,1):\n", b.repeat(2, 1))  # real data replication

    # -------------------------------------------------------------------------
    header("gather() / scatter_() mini-demo")
    A = torch.arange(1, 13).reshape(3, 4)  # [[1..4],[5..8],[9..12]]
    idx = torch.tensor([[0, 2], [1, 3], [0, 0]])  # per-row indices
    picked = A.gather(dim=1, index=idx)
    tgt = torch.zeros_like(A)
    tgt.scatter_(dim=1, index=idx, src=torch.tensor([[9, 9], [8, 8], [7, 7]]))
    print("A:\n", A)
    print("gather(dim=1, idx):\n", picked)
    print("scatter_ into zeros:\n", tgt)

    header("Done")

if __name__ == "__main__":
    main()