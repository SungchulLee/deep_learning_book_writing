"""
================================================================================
make_figures.py - 5장의 그림을 만든다
================================================================================

만드는 그림:
    pca_reconstructions.svg   MNIST를 주성분 2, 16, 32, 64개로 줄였다 되살린 모습.
                              2개로는 읽을 수 없고 64개로는 거의 원본이다

그림 규칙(CLAUDE.md):
    - SVG로 저장하고 svg.fonttype='path'로 글자를 외곽선으로 만든다
    - transparent=True
    - 그림 안의 글자는 모두 ASCII (기본 글꼴에 한글이 없다)

실행:
    python make_figures.py
    DATA_ROOT=~/data python make_figures.py        # 내려받아 둔 자료를 쓸 때
================================================================================
"""

import os

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

matplotlib.rcParams["svg.fonttype"] = "path"

DATA = os.environ.get("DATA_ROOT", "./data")
MEAN, STD = 0.1307, 0.3081


def load_flat():
    """정규화한 MNIST 학습·시험 자료를 (N, 784)로 돌려준다."""
    tf = transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize((MEAN,), (STD,))])
    tr = torchvision.datasets.MNIST(DATA, train=True, download=True, transform=tf)
    te = torchvision.datasets.MNIST(DATA, train=False, download=True, transform=tf)

    def cat(ds):
        xs, ys = [], []
        for x, y in torch.utils.data.DataLoader(ds, batch_size=2000, shuffle=False):
            xs.append(x.flatten(1)); ys.append(y)
        return torch.cat(xs), torch.cat(ys)

    return cat(tr), cat(te)


def pca_basis(X):
    """공분산의 고유분해. torch.linalg.eigh는 이 크기에서 불안정한 빌드가 있어
    numpy의 LAPACK 경로를 float64로 쓴다."""
    mu = X.mean(0, keepdim=True)
    Xc = X - mu
    cov = (Xc.T @ Xc) / (Xc.shape[0] - 1)
    ev, evec = np.linalg.eigh(cov.double().numpy())
    evals = torch.from_numpy(np.ascontiguousarray(ev[::-1])).float()
    evecs = torch.from_numpy(np.ascontiguousarray(evec[:, ::-1])).float()
    return mu, evecs, evals


def show(ax, flat):
    """정규화를 되돌려 28x28로 그린다."""
    img = (flat.reshape(28, 28) * STD + MEAN).clamp(0, 1)
    ax.imshow(img, cmap="gray", vmin=0, vmax=1)
    ax.axis("off")


# === 그림 1: 주성분 개수에 따른 복원 ========================================
def fig_pca_reconstructions():
    (Xtr, _), (Xte, yte) = load_flat()
    mu, evecs, evals = pca_basis(Xtr)

    # 숫자 0~9를 한 장씩 고른다
    idx = [int((yte == d).nonzero()[0]) for d in range(10)]
    ks = [2, 16, 32, 64]

    rows = [("original", Xte[idx])]
    for k in ks:
        V = evecs[:, :k]
        rec = ((Xte[idx] - mu) @ V) @ V.T + mu
        share = 100 * (evals[:k].sum() / evals.sum()).item()
        rows.append((f"PCA-{k}  ({share:.0f}% var)", rec))

    fig, axes = plt.subplots(len(rows), 10, figsize=(11, 5.9))
    for r, (label, imgs) in enumerate(rows):
        for c in range(10):
            show(axes[r, c], imgs[c])
        axes[r, 0].text(-0.18, 0.5, label, transform=axes[r, 0].transAxes,
                        ha="right", va="center", fontsize=9)

    fig.subplots_adjust(left=0.16, right=0.995, top=0.995, bottom=0.005,
                        wspace=0.06, hspace=0.10)
    fig.savefig("pca_reconstructions.svg", transparent=True)
    plt.close(fig)
    print("wrote pca_reconstructions.svg")


# === 학습해 둔 자기 부호기를 읽어 오기 위한 구조 정의 =======================
# mnist_ladder.py가 저장한 state_dict와 짝이 맞아야 한다
import torch.nn as nn

MODELS = os.environ.get("MODEL_ROOT", ".")


class LinearAE(nn.Module):
    def __init__(self, k):
        super().__init__(); self.enc = nn.Linear(784, k); self.dec = nn.Linear(k, 784)
    def forward(self, x): return self.dec(self.enc(x))


class MLPAE(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, k))
        self.dec = nn.Sequential(nn.Linear(k, 256), nn.ReLU(), nn.Linear(256, 784))
    def forward(self, x): return self.dec(self.enc(x))


class ConvAE(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(32 * 7 * 7, k))
        self.dec = nn.Sequential(
            nn.Linear(k, 32 * 7 * 7), nn.ReLU(), nn.Unflatten(1, (32, 7, 7)),
            nn.ConvTranspose2d(32, 16, 2, stride=2), nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 2, stride=2))
    def forward(self, x): return self.dec(self.enc(x))


def load(cls, k, tag):
    m = cls(k)
    m.load_state_dict(torch.load(f"{MODELS}/mnist_{tag}.pt",
                                 map_location="cpu", weights_only=True))
    return m.eval()


# === 그림 2: 부호 64에서 네 걸음의 복원을 나란히 ============================
@torch.no_grad()
def fig_recon_ladder():
    (Xtr, _), (Xte, yte) = load_flat()
    mu, evecs, _ = pca_basis(Xtr)
    idx = [int((yte == d).nonzero()[0]) for d in range(10)]
    x_flat, x_img = Xte[idx], Xte[idx].reshape(-1, 1, 28, 28)

    V = evecs[:, :64]
    rows = [("original", x_flat),
            ("PCA-64", ((x_flat - mu) @ V) @ V.T + mu),
            ("AE_Linear-64", load(LinearAE, 64, "ae_linear64")(x_flat)),
            ("AE_MLP-64", load(MLPAE, 64, "ae_mlp64")(x_flat)),
            ("AE_CNN-64", load(ConvAE, 64, "ae_conv64")(x_img).flatten(1))]

    fig, axes = plt.subplots(len(rows), 10, figsize=(11, 5.9))
    for r, (label, imgs) in enumerate(rows):
        for c in range(10):
            show(axes[r, c], imgs[c])
        axes[r, 0].text(-0.18, 0.5, label, transform=axes[r, 0].transAxes,
                        ha="right", va="center", fontsize=9)
    fig.subplots_adjust(left=0.17, right=0.995, top=0.995, bottom=0.005,
                        wspace=0.06, hspace=0.10)
    fig.savefig("recon_ladder.svg", transparent=True)
    plt.close(fig)
    print("wrote recon_ladder.svg")


# === 그림 3: 2차원 잠재 공간 — 선형과 비선형 ================================
@torch.no_grad()
def fig_latent_2d():
    """왼쪽은 흩어진 점, 오른쪽은 그 공간을 격자로 훑어 복호한 것.

    자기 부호기의 잠재 공간에는 **빈 곳**이 있다는 점이 요점이다. 빈 곳을
    복호하면 숫자가 아닌 것이 나온다. VAE가 고치려는 것이 바로 이것이다.
    """
    (Xtr, _), (Xte, yte) = load_flat()
    mu, evecs, _ = pca_basis(Xtr)

    pca_z = (Xte - mu) @ evecs[:, :2]
    ae = load(MLPAE, 2, "ae_mlp2")
    ae_z = ae.enc(Xte)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4))
    for ax, z, title in [(axes[0], pca_z, "PCA-2 latent"),
                         (axes[1], ae_z, "AE_MLP-2 latent")]:
        # rasterized=True — 점 1만 개를 벡터로 두면 SVG가 3MB를 넘는다.
        # 흩어진 점은 래스터로 묻어도 읽는 데 지장이 없다
        sc = ax.scatter(z[:, 0], z[:, 1], c=yte, cmap="tab10", s=1.2,
                        alpha=0.55, linewidths=0, rasterized=True)
        ax.set_title(title, fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
    cb = fig.colorbar(sc, ax=axes[1], fraction=0.046, ticks=range(10))
    cb.ax.tick_params(labelsize=8)

    # AE 잠재 공간을 격자로 훑어 복호한다
    lo = ae_z.quantile(0.02, dim=0); hi = ae_z.quantile(0.98, dim=0)
    n = 16
    gx = torch.linspace(lo[0], hi[0], n); gy = torch.linspace(lo[1], hi[1], n)
    grid = torch.stack([torch.stack([x, y]) for y in gy.flip(0) for x in gx])
    dec = ae.dec(grid).reshape(n, n, 28, 28)
    canvas = torch.cat([torch.cat([dec[i, j] for j in range(n)], dim=1)
                        for i in range(n)], dim=0)
    axes[2].imshow((canvas * STD + MEAN).clamp(0, 1), cmap="gray", vmin=0, vmax=1)
    axes[2].set_title("AE_MLP-2 decoded over a grid", fontsize=11)
    axes[2].axis("off")

    fig.subplots_adjust(left=0.01, right=0.99, top=0.93, bottom=0.02, wspace=0.12)
    fig.savefig("latent_2d.svg", transparent=True)
    plt.close(fig)
    print("wrote latent_2d.svg")


if __name__ == "__main__":
    fig_pca_reconstructions()
    fig_recon_ladder()
    fig_latent_2d()
