#!/usr/bin/env python3
"""
Data-facing utilities on Tensors.

Covers:
- item(), tolist()
- detach() and detach().clone()
- .cpu().numpy() (and why detach/cpu matters)
- .data caveat (bypasses autograd)
- Basic conversion gotchas with CUDA
"""

import torch

def header(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

def main():
    torch.manual_seed(0)

    # -------------------------------------------------------------------------
    header("Scalars and Python numbers: .item()")
    s = torch.tensor(3.14159)  # 0-dim tensor
    print("s:", s, "| s.item():", s.item())

    # -------------------------------------------------------------------------
    header("Lists: .tolist()")
    v = torch.tensor([[1., 2.], [3., 4.]])
    print("v:\n", v, "\nv.tolist():", v.tolist())

    # -------------------------------------------------------------------------
    header("detach() vs detach().clone() (storage alias vs snapshot)")
    a = torch.randn(4, requires_grad=True)
    b = a.detach()             # shares storage, no grad
    c = a.detach().clone()     # copy to new storage, no grad
    a.add_(10)
    print("a:", a)
    print("b (shares storage, updated with a):", b)
    print("c (snapshot, unchanged):", c)

    # -------------------------------------------------------------------------
    header("NumPy interop: .cpu().numpy()")
    x = torch.randn(5, requires_grad=True)
    x_np = x.detach().cpu().numpy()       # detach if requires_grad=True
    print("x_np (shape, dtype):", x_np.shape, x_np.dtype)

    # GPU example (guarded)
    if torch.cuda.is_available():
        g = torch.randn(3, device="cuda")
        try:
            _ = g.numpy()
        except Exception as e:
            print("CUDA tensor .numpy() fails (expected):", e)
        print("g.cpu().numpy() works shape:", g.cpu().numpy().shape)
    else:
        print("CUDA not available; skipping GPU numpy demo.")

    # -------------------------------------------------------------------------
    header(".data caveat: bypasses autograd (use sparingly)")
    q = torch.tensor([1., 2., 3.], requires_grad=True)
    print("Before .data in-place, q:", q)
    q.data.mul_(1000.)  # no autograd tracking
    print("After  .data in-place,  q:", q)

if __name__ == "__main__":
    main()