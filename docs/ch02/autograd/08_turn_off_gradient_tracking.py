import torch

def main():

    # By default, tensors are created with requires_grad=False
    # (because most tensors are just data, not parameters).
    # You must explicitly set requires_grad=True for parameters you want to optimize.
    w = torch.randn(4, requires_grad=True)  # Equivalent: w = torch.randn(4).requires_grad_()
    print("Before: w.requires_grad =", w.requires_grad)  # True

    # --- requires_grad_ vs torch.no_grad (discussion) -----------------
    # • w.requires_grad_(False)
    #     - Flips ONLY the metadata flag, IN-PLACE. It does NOT change tensor values.
    #     - This action is not tracked by autograd, so wrapping it in torch.no_grad()
    #       is unnecessary and adds no benefit.
    #
    # • with torch.no_grad():
    #     - Use this when you perform VALUE-CHANGING ops you do NOT want recorded
    #       in the autograd graph (e.g., w.add_(...), w.copy_(...), manual updates).
    #     - Example:
    #         with torch.no_grad():
    #             w.add_(update)   # mutate value without tracking
    # ---------------------------------------------------------------------------
    print("Disabling gradient tracking for w ...")
    w.requires_grad_(False)  # metadata toggle; no value change; no need for torch.no_grad()
    print("After:  w.requires_grad =", w.requires_grad)  # False

    # Rule of autograd:
    #   If ALL inputs to an op have requires_grad=False,
    #   the result also has requires_grad=False (no autograd history).
    loss = (w ** 2).sum()
    print("loss.requires_grad (expect False):", loss.requires_grad)

    try:
        loss.backward()
    except Exception as e:
        print(e)

if __name__ == "__main__":
    main()