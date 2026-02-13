import torch
import torch.nn as nn

def main():
    
    torch.manual_seed(2)

    # Tiny linear model: y = Wx + b
    model = nn.Linear(5, 1, bias=True)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)

    # Goal: simulate a larger "effective batch size" using smaller mini-batches
    # Example: we want effective_batch_size = 8, but can only fit batch_size = 2
    batch_size = 2
    accumulation_steps = 4  # 2 × 4 = 8 effective batch size
    N = batch_size * accumulation_steps

    # Fake dataset of size N
    X = torch.randn(N, 5)
    y = torch.randn(N, 1)

    # ------------------------------------------------------------
    # Gradient clearing
    # ------------------------------------------------------------
    opt.zero_grad()

    # ------------------------------------------------------------
    # Accumulation loop
    # ------------------------------------------------------------
    for step in range(accumulation_steps):
        # Slice out each mini-batch
        xb = X[step * batch_size : (step + 1) * batch_size]
        yb = y[step * batch_size : (step + 1) * batch_size]

        # Forward + compute loss on the mini-batch
        pred = model(xb)
        loss = nn.functional.mse_loss(pred, yb, reduction="mean")

        # Scale loss before backward:
        #   - Without scaling: each .backward() accumulates full gradients,
        #     so after 4 mini-batches, grads are 4× too large.
        #   - With scaling: accumulated gradients ≈ one big batch of size 8.
        (loss / accumulation_steps).backward()

        # Important: do NOT call opt.step() yet — wait until all mini-batches processed

    # After accumulation, apply a single optimizer update
    opt.step()
    opt.zero_grad()
    print("Finished one optimizer step using accumulation with proper scaling.")

if __name__ == "__main__":
    main()