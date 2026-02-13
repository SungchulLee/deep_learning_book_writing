"""
Linear Regression with nn.Linear — Dataset from Tensor
========================================================

Idiomatic PyTorch linear regression using nn.Linear, F.mse_loss,
optim.SGD, DataLoader, device handling, and state_dict save/load.

This is the canonical pattern that generalises to all later chapters.

Demonstrates:
- nn.Module subclass
- TensorDataset / DataLoader pipeline
- Device-agnostic training (CPU / GPU)
- optimizer.zero_grad() → loss.backward() → optimizer.step()
- model.eval() + torch.no_grad() for inference
- torch.save / torch.load for model persistence

Author: Deep Learning Foundations Curriculum
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

# ============================================================================
# Configuration
# ============================================================================

parser = argparse.ArgumentParser(description="nn.Linear linear regression")
parser.add_argument("--n-samples", type=int, default=500)
parser.add_argument("--n-features", type=int, default=3)
parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--save-path", type=str, default="linear_model.pt")
ARGS = parser.parse_args()

torch.manual_seed(ARGS.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ============================================================================
# Data
# ============================================================================

n, p = ARGS.n_samples, ARGS.n_features
X = torch.randn(n, p)
w_true = torch.randn(p)
b_true = 3.0
y = X @ w_true + b_true + 0.3 * torch.randn(n)

n_train = int(0.8 * n)
X_train = X[:n_train].to(device)
X_test = X[n_train:].to(device)
y_train = y[:n_train].to(device)
y_test = y[n_train:].to(device)

train_ds = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_ds, batch_size=ARGS.batch_size, shuffle=True)

# ============================================================================
# Model
# ============================================================================


class LinearRegression(nn.Module):
    def __init__(self, in_features: int, out_features: int = 1):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(-1)


model = LinearRegression(in_features=p).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=ARGS.lr)

print(model)
print(f"Parameters: {sum(p.numel() for p in model.parameters())}")

# ============================================================================
# Training
# ============================================================================

history = []
for epoch in range(ARGS.epochs):
    model.train()
    epoch_loss = 0.0
    for X_batch, y_batch in train_loader:
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * len(y_batch)

    avg_loss = epoch_loss / len(train_ds)
    history.append(avg_loss)

    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1:3d}  MSE = {avg_loss:.6f}")

# ============================================================================
# Evaluation
# ============================================================================


@torch.no_grad()
def evaluate(model, X, y):
    model.eval()
    y_pred = model(X)
    mse = F.mse_loss(y_pred, y).item()
    ss_res = ((y - y_pred) ** 2).sum().item()
    ss_tot = ((y - y.mean()) ** 2).sum().item()
    r2 = 1.0 - ss_res / ss_tot
    return {"mse": mse, "r2": r2}


train_metrics = evaluate(model, X_train, y_train)
test_metrics = evaluate(model, X_test, y_test)
print(f"\nTrain MSE: {train_metrics['mse']:.6f}  R²: {train_metrics['r2']:.4f}")
print(f"Test  MSE: {test_metrics['mse']:.6f}  R²: {test_metrics['r2']:.4f}")

w_learned = model.linear.weight.detach().cpu().squeeze().numpy()
b_learned = model.linear.bias.detach().cpu().item()
print(f"\nLearned w: {w_learned}")
print(f"True    w: {w_true.numpy()}")
print(f"Learned b: {b_learned:.4f}  (true: {b_true})")

# ============================================================================
# Save / Load
# ============================================================================

torch.save(model.state_dict(), ARGS.save_path)
print(f"\nModel saved to {ARGS.save_path}")

model2 = LinearRegression(in_features=p).to(device)
model2.load_state_dict(torch.load(ARGS.save_path, map_location=device))
model2.eval()

test_metrics2 = evaluate(model2, X_test, y_test)
assert abs(test_metrics["mse"] - test_metrics2["mse"]) < 1e-6
print("Model loaded and verified — predictions match.")

# ============================================================================
# Convergence Plot
# ============================================================================

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(history, lw=1.5)
ax.set_xlabel("Epoch")
ax.set_ylabel("MSE")
ax.set_title("nn.Linear — Training Loss")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("dataset_from_tensor.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: dataset_from_tensor.png")
