import torch

def main():

    # (a) Mixing tracked & untracked tensors:
    # Rule: if ANY input has requires_grad=True, the result will also track gradients.
    a = torch.randn(3, requires_grad=True)
    b = torch.randn(3, requires_grad=False)
    c = a + b
    print("a.requires_grad:", a.requires_grad, "| b.requires_grad:", b.requires_grad)
    print("c = a + b → c.requires_grad (expect True):", c.requires_grad)

    # (b) Inference-time context:
    # Use torch.no_grad() to temporarily disable gradient tracking.
    # This saves memory and compute, since no backward graph is constructed.
    m = torch.nn.Linear(4, 2)
    x = torch.randn(8, 4, requires_grad=True)
    with torch.no_grad():
        y = m(x)
    print("Inference y.requires_grad (expect False):", y.requires_grad)

if __name__ == "__main__":
    main()