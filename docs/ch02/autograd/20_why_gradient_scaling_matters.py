import torch
import torch.nn as nn

def main():

    torch.manual_seed(3)

    modelA = nn.Linear(3, 1, bias=False)
    modelB = nn.Linear(3, 1, bias=False)

    # Copy identical weights so both models start from the same point
    with torch.no_grad():
        modelB.weight.copy_(modelA.weight)

    X = torch.randn(4, 3)
    y = torch.randn(4, 1)

    # -- Baseline: full-batch mean loss → reference gradient
    # Compute grad for loss = MSE(X→y, reduction="mean") over the ENTIRE batch.
    # We'll compare microbatch accumulation against this "ground truth."
    pred_full = modelA(X)
    loss_full = nn.functional.mse_loss(pred_full, y, reduction="mean")
    loss_full.backward()

    # About param.grad:
    # - modelA.weight.grad is a *gradient buffer tensor* allocated by autograd.
    # - It is NOT part of the forward graph (requires_grad=False, grad_fn=None).
    # - Therefore, .detach() is NOT necessary to avoid tracking — it's already not tracked.
    # - However, cloning is useful to take a stable snapshot before we mutate/clear grads later.
    #   (e.g., opt.zero_grad(set_to_none=True) would set grad to None; further backward() overwrites it.)
    #
    # Choices:
    #   grad_full = modelA.weight.grad.clone()                # sufficient (no tracking anyway)
    #   grad_full = modelA.weight.grad.detach().clone()       # also fine; extra .detach() is redundant (defensive coding)
    #   grad_full = modelA.weight.grad.detach()               # shares storage; risky if grad changes later
    grad_full = modelA.weight.grad.detach().clone()

    # -- Microbatch accumulation for modelB
    modelB.zero_grad(set_to_none=True)
    mb1 = (X[:2], y[:2])
    mb2 = (X[2:], y[2:])

    # (i) WRONG: backprop the **mean** loss for each microbatch and just accumulate.
    # Intuition: each microbatch's mean already divides by its own size.
    # Accumulating two such grads ≈ doubles the full-batch mean grad (for equal halves).
    # Hence the result is too large (~2× here).
    pred1 = modelB(mb1[0])
    loss1 = nn.functional.mse_loss(pred1, mb1[1], reduction="mean")
    loss1.backward()  # grads accumulate → already too big
    pred2 = modelB(mb2[0])
    loss2 = nn.functional.mse_loss(pred2, mb2[1], reduction="mean")
    loss2.backward()  # accumulates again → now ≈ 2× full-batch mean grad
    wrong_grad = modelB.weight.grad.detach().clone()  # snapshot for fair comparison

    # (ii) RIGHT: scale each microbatch mean-loss by 1 / (#microbatches).
    # With K microbatches, backprop (loss_mean / K) per microbatch.
    # The accumulated gradient then matches the gradient of the full-batch mean loss.
    modelB.zero_grad(set_to_none=True)
    pred1 = modelB(mb1[0])
    (nn.functional.mse_loss(pred1, mb1[1], reduction="mean") / 2).backward()
    pred2 = modelB(mb2[0])
    (nn.functional.mse_loss(pred2, mb2[1], reduction="mean") / 2).backward()
    right_grad = modelB.weight.grad.detach().clone()  # safe snapshot again

    print("Full-batch grad:\n", grad_full)
    print("Accumulated grad (WRONG, unscaled):\n", wrong_grad)
    print("Accumulated grad (RIGHT, scaled):\n", right_grad)
    print("max |full - right|:", (grad_full - right_grad).abs().max().item())

if __name__ == "__main__":
    main()