import torch
import torch.nn as nn

def main():

    torch.manual_seed(5)

    model = nn.Linear(4, 1)
    opt = torch.optim.SGD(model.parameters(), lr=0.05)

    # Fake dataset with N samples (features=4, target=1)
    N = 12
    X = torch.randn(N, 4)
    y = torch.randn(N, 1)

    batch_size = 3
    accumulation_steps = 2  # effective batch size = batch_size * accumulation_steps = 6

    for epoch in range(2):
        opt.zero_grad(set_to_none=True)
        running_loss = 0.0

        for step in range(0, N, batch_size):
            xb = X[step : step + batch_size]
            yb = y[step : step + batch_size]

            # Forward pass
            pred = model(xb)
            loss = nn.functional.mse_loss(pred, yb, reduction="mean")

            # Scale loss so accumulated grads match full-batch mean loss
            (loss / accumulation_steps).backward()
            running_loss += loss.item()

            # Only update weights after "accumulation_steps" microbatches
            if ((step // batch_size) + 1) % accumulation_steps == 0:
                opt.step()                     # apply one update
                opt.zero_grad(set_to_none=True)  # clear grads for next accumulation

        # Report average loss (per microbatch) for monitoring
        print(f"Epoch {epoch}: avg loss per microbatch = {running_loss / (N / batch_size):.6f}")

    # Print final learned parameters (detach() so they are plain tensors)
    print("Final model weights:\n", {n: p.detach() for n, p in model.named_parameters()})

if __name__ == "__main__":
    main()