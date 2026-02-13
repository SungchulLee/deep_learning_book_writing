#!/usr/bin/env python3
"""
Autograd-focused Tensor attributes and behaviors.

Covers:
- requires_grad / is_leaf / grad_fn
- .backward() on scalar vs non-scalar (VJP with gradient arg)
- Gradient accumulation & clearing (p.grad.zero_ vs opt.zero_grad)
- torch.no_grad() for safe parameter updates
- retain_graph=True for multiple backward passes on same graph
"""

import torch
import torch.nn as nn

def header(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

def main():
    torch.manual_seed(0)

    # -------------------------------------------------------------------------
    header("Leaf / non-leaf, requires_grad, grad_fn, is_leaf")
    w = torch.randn(3, requires_grad=True)   # LEAF tensor
    y = (w * 2.0).sum()                      # non-leaf scalar with grad_fn
    print("w:", w)
    print("w.requires_grad:", w.requires_grad, "| w.grad_fn:", w.grad_fn, "| w.is_leaf:", w.is_leaf)
    print("y.requires_grad:", y.requires_grad, "| y.grad_fn:", y.grad_fn, "| y.is_leaf:", y.is_leaf)

    # -------------------------------------------------------------------------
    header("Backward on scalar output → fills w.grad (accumulates)")
    print("Before backward, w.grad:", w.grad)
    y.backward()
    print("After  1st backward, w.grad:", w.grad)
    w.grad.zero_()
    ((w ** 2).sum()).backward()
    print("After  2nd backward, w.grad (fresh):", w.grad)

    # -------------------------------------------------------------------------
    header("Non-scalar output requires gradient arg (VJP)")
    x = torch.randn(4, 3, requires_grad=True)
    A = torch.randn(2, 3)         # fixed linear map (no grads)
    out = x @ A.t()                # shape (4, 2) → non-scalar
    v = torch.tensor([[1., 0.], [0.5, -1.], [0., 0.], [2., 3.]])  # same shape as out
    x.grad = None
    out.backward(v)                # computes VJP wrt x
    print("x.grad shape (expect (4,3)):", x.grad.shape)

    # -------------------------------------------------------------------------
    header("Optimizer-style clearing: zero_grad(set_to_none=True)")
    model = nn.Linear(3, 1, bias=True)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    xb = torch.randn(5, 3)
    yb = torch.randn(5, 1)
    opt.zero_grad(set_to_none=True)   # sets .grad to None (not zeros)
    nn.functional.mse_loss(model(xb), yb).backward()
    # .grad now allocated; another .backward() would ACCUMULATE unless cleared again.
    print("Param grad is None? ->", [p.grad is None for p in model.parameters()])
    opt.zero_grad(set_to_none=True)
    print("After zero_grad(set_to_none=True):", [p.grad is None for p in model.parameters()])

    # -------------------------------------------------------------------------
    header("torch.no_grad() for parameter updates (avoid graph pollution)")
    p = torch.randn(3, requires_grad=True)
    loss = (p ** 2).sum()
    loss.backward()
    with torch.no_grad():          # update excluded from graph
        p -= 0.1 * p.grad
    print("p.requires_grad stays True:", p.requires_grad)

    # -------------------------------------------------------------------------
    header("retain_graph=True for repeated backward on the SAME graph")
    a = torch.tensor([1., 2., 3.], requires_grad=True)
    z = (a ** 2).sum()             # one graph
    z.backward(retain_graph=True)  # keep graph alive
    z.backward()                   # reuse retained graph, now it will be freed
    print("a.grad (accumulated from two backward passes):", a.grad)

if __name__ == "__main__":
    main()