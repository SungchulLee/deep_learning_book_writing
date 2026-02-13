#!/usr/bin/env python3
"""
PyTorch nn.Linear weight convention & equivalences.

Key points:
- nn.Linear(in_features, out_features) stores:
    weight.shape == (out_features, in_features)
    bias.shape   == (out_features,)  (if bias=True)
- Forward pass equals:
    out = F.linear(x, weight, bias) == x @ weight.T + bias
  where x.shape == (batch, in_features), out.shape == (batch, out_features)

We verify:
- Shapes and equality with/without bias
- That weight.T is a view (no copy), and .contiguous() if needed
- Autograd gradients shapes
- "Row = one output neuron" intuition by slicing weight rows
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

def header(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

def main():
    torch.manual_seed(0)

    B, in_features, out_features = 4, 5, 3
    x = torch.randn(B, in_features)

    # -------------------------------------------------------------------------
    header("nn.Linear weight/bias shapes and equivalence to x @ W.T + b")
    lin = nn.Linear(in_features, out_features, bias=True)

    W = lin.weight  # (out_features, in_features)
    b = lin.bias    # (out_features,)
    print("W.shape:", W.shape, "| b.shape:", b.shape)

    # F.linear vs matmul equivalence
    out_F = F.linear(x, W, b)        # preferred functional form
    out_mm = x @ W.t() + b           # explicit matmul
    print("Allclose(F.linear, x @ W.T + b):", torch.allclose(out_F, out_mm, atol=1e-6))

    # -------------------------------------------------------------------------
    header("Without bias: same equivalence")
    lin_nobias = nn.Linear(in_features, out_features, bias=False)
    W2 = lin_nobias.weight
    out_F2 = F.linear(x, W2, None)
    out_mm2 = x @ W2.t()
    print("W2.shape:", W2.shape, "| bias=None")
    print("Allclose(no-bias):", torch.allclose(out_F2, out_mm2, atol=1e-6))

    # -------------------------------------------------------------------------
    header("Transpose view & contiguity")
    WT = W.t()  # view with different strides (no data copy)
    print("W.is_contiguous:", W.is_contiguous(), "| W.t().is_contiguous:", WT.is_contiguous())
    WTc = WT.contiguous()
    print("After .contiguous(), WTc.is_contiguous:", WTc.is_contiguous())

    # -------------------------------------------------------------------------
    header("Autograd shapes: grads wrt W and b")
    # Simple scalar loss so we can backprop
    x_req = x.clone().requires_grad_(True)  # data as leaf (normally requires_grad=False)
    out = F.linear(x_req, W, b)
    loss = out.sum()
    # Clear grads on params
    lin.zero_grad(set_to_none=True)
    loss.backward()
    print("W.grad.shape:", W.grad.shape, "| b.grad.shape:", b.grad.shape)
    print("x_req.grad.shape (grad wrt inputs):", x_req.grad.shape)

    # -------------------------------------------------------------------------
    header("Row = one output neuron (intuition)")
    # Each output unit k uses weight row W[k] (incoming weights) and bias b[k]
    # For a single example x[i], output[k] ≈ x[i] @ W[k].T + b[k]
    i, k = 0, 1
    manual_k = x[i] @ W[k].t() + b[k]
    print(f"x[{i}].shape:", x[i].shape, "| W[{k}].shape:", W[k].shape)
    print(f"lin(x)[{i},{k}] =", lin(x)[i, k].item(), "| manual =", manual_k.item())

    # -------------------------------------------------------------------------
    header("Batch/out dims check with random shapes")
    for (Bb, inf, outf) in [(2, 4, 6), (7, 3, 1)]:
        xx = torch.randn(Bb, inf)
        ll = nn.Linear(inf, outf, bias=True)
        out1 = ll(xx)
        out2 = xx @ ll.weight.t() + ll.bias
        print(f"(B={Bb}, in={inf}, out={outf}) -> out.shape:", out1.shape,
              "| equal:", torch.allclose(out1, out2, atol=1e-6))

    # -------------------------------------------------------------------------
    header("Sanity: training step matches both forms")
    # One SGD step comparing F.linear vs x @ W.T + b
    model = nn.Linear(in_features, out_features, bias=True)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    xbatch = torch.randn(B, in_features)
    ytarget = torch.randn(B, out_features)

    # Forward both ways (they're the same)
    outA = F.linear(xbatch, model.weight, model.bias)
    outB = xbatch @ model.weight.t() + model.bias
    print("Forward equal (pre-step):", torch.allclose(outA, outB, atol=1e-6))

    lossA = F.mse_loss(outA, ytarget, reduction="mean")
    opt.zero_grad(set_to_none=True)
    lossA.backward()
    opt.step()

    outA2 = F.linear(xbatch, model.weight, model.bias)
    outB2 = xbatch @ model.weight.t() + model.bias
    print("Forward equal (post-step):", torch.allclose(outA2, outB2, atol=1e-6))

    # -------------------------------------------------------------------------
    header("Done")

if __name__ == "__main__":
    main()