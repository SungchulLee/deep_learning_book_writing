"""Wavenet gated."""
# ---
# title: "WaveNet: Gated Residual Blocks"
# description: "Full WaveNet architecture with gated activations and skip connections"
# ---
#
# This script implements the full WaveNet architecture in PyTorch, including:
#   1. Gated Activation Unit  (tanh ⊙ sigmoid)
#   2. Residual blocks with dilated causal convolutions
#   3. Skip connections aggregated across all layers
#
# The gated activation is the key difference from vanilla dilated convolutions:
#
#       z = tanh(W_f * x) ⊙ σ(W_g * x)
#
# where W_f and W_g are separate filter and gate convolution weights.
#
# Adapted from: O'Reilly Hands-On ML, Chapter 15 (TF → PyTorch)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# ========================================================================
# Main
# ========================================================================


# ─── Causal Conv1d ─────────────────────────────────────────────────────────
class CausalConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=2, dilation=1):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation)

    def forward(self, x):
        return self.conv(F.pad(x, (self.pad, 0)))


# ─── Gated Activation Unit ────────────────────────────────────────────────
class GatedActivationUnit(nn.Module):
    """Splits the channel dimension in half and applies:
       output = tanh(first_half) * sigmoid(second_half)
    """

    def forward(self, x):
        # x: (batch, 2*n_filters, seq_len)
        n_filters = x.size(1) // 2
        return torch.tanh(x[:, :n_filters]) * torch.sigmoid(x[:, n_filters:])


# ─── WaveNet Residual Block ───────────────────────────────────────────────
class WaveNetResidualBlock(nn.Module):
    """One residual block:
       dilated_conv → gated_activation → 1x1 conv → add residual
       The 1x1 output is also returned as a skip connection.
    """

    def __init__(self, n_filters, kernel_size=2, dilation=1):
        super().__init__()
        self.dilated_conv = CausalConv1d(n_filters, 2 * n_filters, kernel_size, dilation)
        self.gate = GatedActivationUnit()
        self.residual_conv = nn.Conv1d(n_filters, n_filters, kernel_size=1)
        self.skip_conv = nn.Conv1d(n_filters, n_filters, kernel_size=1)

    def forward(self, x):
        z = self.dilated_conv(x)
        z = self.gate(z)
        skip = self.skip_conv(z)
        residual = self.residual_conv(z) + x
        return residual, skip


# ─── Full WaveNet ──────────────────────────────────────────────────────────
class WaveNet(nn.Module):
    """Complete WaveNet with multiple blocks of dilated causal convolutions,
    gated activations, residual + skip connections.

    Args:
        in_channels:        Number of input features (1 for univariate)
        n_filters:          Number of hidden channels per layer
        n_layers_per_block: Dilation rates go 1, 2, …, 2^(n-1) within a block
        n_blocks:           Number of dilation-rate cycles
        n_outputs:          Number of output channels (e.g. 256 for μ-law)
    """

    def __init__(self, in_channels=1, n_filters=32, n_layers_per_block=3,
                 n_blocks=1, n_outputs=10, kernel_size=2):
        super().__init__()
        self.input_conv = CausalConv1d(in_channels, n_filters, kernel_size)

        self.residual_blocks = nn.ModuleList()
        for _ in range(n_blocks):
            for i in range(n_layers_per_block):
                dilation = 2 ** i
                self.residual_blocks.append(
                    WaveNetResidualBlock(n_filters, kernel_size, dilation)
                )

        self.post_conv1 = nn.Conv1d(n_filters, n_filters, kernel_size=1)
        self.post_conv2 = nn.Conv1d(n_filters, n_outputs, kernel_size=1)

    def forward(self, x):
        z = self.input_conv(x)
        skips = []
        for block in self.residual_blocks:
            z, skip = block(z)
            skips.append(skip)
        # Sum all skip connections
        s = torch.stack(skips, dim=0).sum(dim=0)
        s = F.relu(s)
        s = F.relu(self.post_conv1(s))
        return self.post_conv2(s)

    def receptive_field(self):
        """Calculate the theoretical receptive field size."""
        rf = 1
        for block in self.residual_blocks:
            dilation = block.dilated_conv.conv.dilation[0]
            kernel = block.dilated_conv.conv.kernel_size[0]
            rf += (kernel - 1) * dilation
        return rf


# ─── Data ──────────────────────────────────────────────────────────────────
def generate_time_series(batch_size, n_steps):
    freq1, freq2 = np.random.uniform(0.1, 0.5, (2, batch_size, 1))
    off1, off2 = np.random.uniform(0, 2 * np.pi, (2, batch_size, 1))
    time = np.linspace(0, 1, n_steps + 10).reshape(1, -1)
    series = 0.5 * np.sin((time - off1) * (n_steps * freq1))
    series += 0.2 * np.sin((time - off2) * (n_steps * freq2))
    series += 0.1 * (np.random.randn(batch_size, n_steps + 10) - 0.5)
    return series.astype(np.float32)


np.random.seed(42)
torch.manual_seed(42)

n_steps = 50
series = generate_time_series(10000, n_steps)
X_train = torch.tensor(series[:7000, :n_steps]).unsqueeze(1)
Y_train = torch.tensor(series[:7000, 1:n_steps + 1]).unsqueeze(1)
X_valid = torch.tensor(series[7000:9000, :n_steps]).unsqueeze(1)
Y_valid = torch.tensor(series[7000:9000, 1:n_steps + 1]).unsqueeze(1)

# ─── Model ─────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = WaveNet(
    in_channels=1,
    n_filters=32,         # 128 in the original paper
    n_layers_per_block=3, # 10 in the original paper
    n_blocks=1,           # 3 in the original paper
    n_outputs=1,          # regression (1 output channel)
).to(device)

print(f"WaveNet parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Receptive field:    {model.receptive_field()} time steps")
print(f"Device:             {device}\n")

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

# ─── Training ──────────────────────────────────────────────────────────────
epochs = 25
batch_size = 128
train_losses, val_losses = [], []

for epoch in range(1, epochs + 1):
    model.train()
    perm = torch.randperm(X_train.size(0))
    epoch_loss, n = 0.0, 0
    for i in range(0, len(perm), batch_size):
        idx = perm[i : i + batch_size]
        xb = X_train[idx].to(device)
        yb = Y_train[idx].to(device)
        pred = model(xb)
        loss = loss_fn(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        n += 1
    train_losses.append(epoch_loss / n)

    model.eval()
    with torch.no_grad():
        vp = model(X_valid.to(device))
        vl = loss_fn(vp, Y_valid.to(device)).item()
    val_losses.append(vl)

    if epoch % 5 == 0 or epoch == 1:
        print(f"Epoch {epoch:3d}  train={train_losses[-1]:.5f}  val={val_losses[-1]:.5f}")

# ─── Visualization ─────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Loss curves
ax1.plot(train_losses, label="train")
ax1.plot(val_losses, label="valid")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("MSE")
ax1.set_title("WaveNet Training Curve")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Sample prediction
model.eval()
with torch.no_grad():
    sample_x = X_valid[:3].to(device)
    sample_pred = model(sample_x).cpu().squeeze(1).numpy()
    sample_gt = Y_valid[:3].squeeze(1).numpy()

for i in range(3):
    ax2.plot(sample_gt[i], alpha=0.4, label=f"true {i}" if i == 0 else None, color="C0")
    ax2.plot(sample_pred[i], alpha=0.7, label=f"pred {i}" if i == 0 else None, color="C1", linestyle="--")
ax2.set_title("WaveNet Predictions (3 samples)")
ax2.set_xlabel("Time step")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("wavenet_gated_results.png", dpi=150)
plt.show()
print("Done.")


if __name__ == "__main__":
    pass