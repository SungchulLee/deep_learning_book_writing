#!/usr/bin/env python3
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

def main():
    torch.manual_seed(0)

    # ------------------------------------------------------------
    # 1) Synthetic data: y = 1 + 2x + ε,  ε ~ N(0, 0.2^2)
    # ------------------------------------------------------------
    n = 100
    x = torch.randn(n, 1)                    # x ~ N(0,1)
    noise = 0.2 * torch.randn(n, 1)
    y = 1.0 + 2.0 * x + noise                # target

    # ------------------------------------------------------------
    # 2) Model: nn.Linear(1,1)
    # ------------------------------------------------------------
    model = nn.Linear(1, 1)                  # y_hat = w*x + b
    criterion = nn.MSELoss()                 # mean squared error
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    # ------------------------------------------------------------
    # 3) Training loop
    # ------------------------------------------------------------
    steps = 200
    losses = []
    alphas, betas = [], []   # store bias (α) and weight (β)

    for step in range(steps):
        # -------- Forward --------
        y_hat = model(x)
        loss = criterion(y_hat, y)

        # -------- Backward --------
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Log values
        losses.append(loss.item())
        betas.append(model.weight.item())
        alphas.append(model.bias.item())

        if step % 20 == 0 or step == steps - 1:
            print(f"step {step:3d}: loss={loss.item():.6f} | β={model.weight.item():.4f} | α={model.bias.item():.4f}")

    print("Final params:", {"beta (slope)": model.weight.item(), "alpha (intercept)": model.bias.item()})

    # ------------------------------------------------------------
    # 4) Visualization
    # ------------------------------------------------------------
    x_np = x.detach().cpu().numpy().reshape(-1)
    y_np = y.detach().cpu().numpy().reshape(-1)

    # For a clean fitted line
    sort_idx = x_np.argsort()
    x_sorted = x_np[sort_idx]
    yhat_sorted = (model.weight.item() * x_sorted + model.bias.item())

    fig, (ax0, ax1, ax2, ax3) = plt.subplots(1, 4, figsize=(18, 4))

    # (a) data + fit
    ax0.scatter(x_np, y_np, alpha=0.5, label="data")
    ax0.plot(x_sorted, yhat_sorted, lw=3, label="fitted line")
    ax0.set_title("Linear Fit on Synthetic Data")
    ax0.set_xlabel("x")
    ax0.set_ylabel("y")
    ax0.legend()

    # (b) loss curve
    ax1.plot(range(steps), losses, lw=2)
    ax1.set_title("Training Loss (MSE) per Step")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss")

    # (c) α (bias) trajectories
    #ax2.plot(range(steps), betas, label="β (slope)", lw=2)
    ax2.plot(range(steps), alphas, label="α (intercept)", lw=2)
    #ax2.axhline(2.0, color="k", ls="--", lw=1, alpha=0.7, label="true β=2")
    ax2.axhline(1.0, color="gray", ls="--", lw=1, alpha=0.7, label="true α=1")
    ax2.set_title("Parameter Alpha Convergence")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("alpha")
    ax2.legend()

    # (d) β (weight) trajectories
    ax3.plot(range(steps), betas, label="β (slope)", lw=2)
    #ax3.plot(range(steps), alphas, label="α (intercept)", lw=2)
    ax3.axhline(2.0, color="k", ls="--", lw=1, alpha=0.7, label="true β=2")
    #ax3.axhline(1.0, color="gray", ls="--", lw=1, alpha=0.7, label="true α=1")
    ax3.set_title("Parameter Beta Convergence")
    ax3.set_xlabel("epoch")
    ax3.set_ylabel("beta")
    ax3.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()