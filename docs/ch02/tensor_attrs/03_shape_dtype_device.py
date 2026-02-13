#!/usr/bin/env python3
"""
Shape / dtype / device / layout attributes and helpers.

Covers:
- shape / size() / ndim
- dtype / device / requires_grad
- layout
- is_cuda / is_contiguous() / stride() / storage_offset()
- transpose helpers: T / mT / H
"""

import torch

def header(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

def main():
    torch.manual_seed(0)

    # -------------------------------------------------------------------------
    header("Base tensor setup")
    x = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    print("x:\n", x)

    # -------------------------------------------------------------------------
    header("Shape / ndim")
    print("x.shape     :", x.shape)
    print("x.size()    :", x.size())
    print("x.ndim      :", x.ndim)

    # -------------------------------------------------------------------------
    header("dtype / device / requires_grad / layout")
    print("x.dtype       :", x.dtype)
    print("x.device      :", x.device)
    print("x.requires_grad:", x.requires_grad)
    print("x.layout      :", x.layout)

    # -------------------------------------------------------------------------
    header("Memory layout attributes")
    print("x.is_cuda       :", x.is_cuda)
    print("x.is_contiguous :", x.is_contiguous())
    print("x.stride()      :", x.stride())
    print("x.storage_offset:", x.storage_offset())

    # -------------------------------------------------------------------------
    header("Transpose helpers (T / mT / H)")
    print("x.T  (simple transpose):\n", x.T)
    print("x.mT (matrix transpose):\n", x.mT)
    print("x.H  (Hermitian transpose):\n", x.H)

    # -------------------------------------------------------------------------
    header("Autograd quick peek")
    y = x.clone().requires_grad_(True)
    z = (y * y).sum()
    print("y.is_leaf :", y.is_leaf)  # True
    print("z.grad_fn :", z.grad_fn)  # e.g., <SumBackward0>
    z.backward()
    print("y.grad:\n", y.grad)

if __name__ == "__main__":
    main()