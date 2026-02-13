import torch

def main():
    torch.manual_seed(1)

    w = torch.randn(3, requires_grad=True)
    opt = torch.optim.SGD([w], lr=0.1)

    # (a) Default behavior: optimizer.zero_grad() now sets gradients to None
    #     (this changed in recent PyTorch versions; older versions used zeros).
    loss = (w ** 2).sum()
    loss.backward()
    print("1) .grad after backward:", w.grad)
    opt.zero_grad()  # default set_to_none=True → clears grads by setting them to None
    print("   after opt.zero_grad():", w.grad)

    # (b) Explicitly zero-fill: set_to_none=False → replaces gradients with a zero tensor
    #     This guarantees .grad is a tensor of zeros (legacy behavior).
    loss = (w ** 2).sum()
    loss.backward()
    print("2) .grad after backward:", w.grad)
    opt.zero_grad(set_to_none=False)
    print("   after opt.zero_grad(set_to_none=False):", w.grad)  # zero tensor

    # (c) Manual reset: explicitly assign w.grad=None (same as set_to_none=True for that tensor)
    loss = (w ** 2).sum()
    loss.backward()
    print("3) .grad after backward:", w.grad)
    w.grad = None
    print("   after w.grad=None:", w.grad)

if __name__ == "__main__":
    main()