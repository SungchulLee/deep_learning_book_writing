#!/usr/bin/env python3
# ==========================================================
# ae_principal_manifold_2d_to_1d.py
# ==========================================================
# Learn a smooth 1-D manifold in 2-D with an autoencoder.
# Visualizes:
#   • original noisy S-curve points
#   • learned manifold (decoder over a latent sweep)
#   • projections decoder(encoder(x))
#   • tiny segments from x to its projection
#
# Run: python ae_principal_manifold_2d_to_1d.py

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

# -----------------------------
# 1) Make a 2D S-curve dataset
# -----------------------------
def make_s_curve(n=400, noise=0.07, seed=0):
    rng = np.random.default_rng(seed)
    # latent parameter t in [-2.5, 2.5]
    t = rng.uniform(-2.5, 2.5, size=n)
    x = t
    y = np.sin(1.5 * t)
    X = np.stack([x, y], axis=1)
    X += rng.normal(scale=noise, size=X.shape)
    return X.astype(np.float32)

X = make_s_curve(n=600, noise=0.06, seed=42)   # (n, 2)
X_mean = X.mean(axis=0, keepdims=True)
X_std  = X.std(axis=0, keepdims=True)
Xn = (X - X_mean) / X_std                       # normalize for easier training

device = "cpu"
Xt = torch.from_numpy(Xn).to(device)

# -----------------------------
# 2) Define a tiny AE: 2 -> 1 -> 2
# -----------------------------
class AE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(2, 32),
            nn.Tanh(),
            nn.Linear(32, 16),
            nn.Tanh(),
            nn.Linear(16, 1),        # bottleneck (the 1-D manifold coordinate)
        )
        self.decoder = nn.Sequential(
            nn.Linear(1, 16),
            nn.Tanh(),
            nn.Linear(16, 32),
            nn.Tanh(),
            nn.Linear(32, 2),
        )

    def forward(self, x):
        z = self.encoder(x)
        xhat = self.decoder(z)
        return xhat, z

ae = AE().to(device)

# -----------------------------
# 3) Train with MSE (recon loss)
# -----------------------------
opt = optim.Adam(ae.parameters(), lr=5e-3)
loss_fn = nn.MSELoss()

ae.train()
epochs = 60000
batch = 128
perm = torch.randperm(len(Xt))
for ep in range(1, epochs + 1):
    # simple minibatch loop
    total = 0.0
    for i in range(0, len(Xt), batch):
        idx = perm[i:i+batch]
        xb = Xt[idx]
        opt.zero_grad()
        xhat, _ = ae(xb)
        loss = loss_fn(xhat, xb)
        loss.backward()
        opt.step()
        total += loss.item() * len(idx)
    if ep % 100 == 0:
        print(f"epoch {ep:4d} | recon MSE ~ {total/len(Xt):.5f}")

ae.eval()
with torch.no_grad():
    Xproj_n, Z = ae(Xt)             # projections in normalized space + 1D coords
    Xproj = Xproj_n.cpu().numpy() * X_std + X_mean  # back to original scale
    z = Z.cpu().numpy().ravel()

# -----------------------------
# 4) Trace the learned manifold
# -----------------------------
# Sweep the latent in the range observed from data and decode
with torch.no_grad():
    z_min, z_max = np.percentile(z, 1), np.percentile(z, 99)
    zs = torch.linspace(z_min, z_max, 400, device=device).unsqueeze(1)
    curve_n = ae.decoder(zs).cpu().numpy()
    curve = curve_n * X_std + X_mean

# Also compute projections for the data
Xproj = Xproj.astype(np.float32)

# -----------------------------
# 5) Plot
# -----------------------------
fig, ax = plt.subplots(figsize=(8, 6))

# original points
ax.scatter(X[:, 0], X[:, 1], s=18, alpha=0.55, label="Original points")

# reconstructed projections
ax.scatter(Xproj[:, 0], Xproj[:, 1], s=12, marker="x", alpha=0.9,
           label="Projection via AE (decode(encode(x)))")

# tiny segments from x to its projection (subsample for clarity)
n = len(X)
step = max(1, n // 60)
for i in range(0, n, step):
    ax.plot([X[i, 0], Xproj[i, 0]], [X[i, 1], Xproj[i, 1]],
            linewidth=0.8, alpha=0.5)

# the learned 1-D manifold
ax.plot(curve[:, 0], curve[:, 1], linewidth=2.2, label="Learned 1-D manifold")

ax.set_title("Autoencoder ‘Principal Manifold’: 2D → 1D → 2D")
ax.set_xlabel("x₁")
ax.set_ylabel("x₂")
ax.axis("equal")
ax.grid(True, linestyle="--", alpha=0.3)
ax.legend(loc="best")
plt.tight_layout()
plt.show()

