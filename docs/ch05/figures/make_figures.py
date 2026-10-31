"""
================================================================================
make_figures.py - 5장의 그림을 만든다
================================================================================

만드는 그림:
    pca_reconstructions.svg   MNIST를 주성분 2, 16, 32, 64개로 줄였다 되살린 모습.
                              2개로는 읽을 수 없고 64개로는 거의 원본이다
    ae_linear_descent.svg     선형 AE가 100 에포크에 걸쳐 PCA의 바닥까지 내려가는
                              곡선과, 그 끝에서 나온 복원. 이 그림만 모델을 직접
                              학습하므로 혼자 5분쯤 걸린다

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


# === 학습해 둔 오토인코더를 읽어 오기 위한 구조 정의 =======================
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

    오토인코더의 잠재 공간에는 **빈 곳**이 있다는 점이 요점이다. 빈 곳을
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


class VAEModel(nn.Module):
    def __init__(self, k, n_cond=0):
        super().__init__()
        self.k = k
        self.enc = nn.Sequential(nn.Linear(784 + n_cond, 256), nn.ReLU())
        self.mu = nn.Linear(256, k); self.logvar = nn.Linear(256, k)
        self.dec = nn.Sequential(nn.Linear(k + n_cond, 256), nn.ReLU(), nn.Linear(256, 784))


# === 그림 4: 2차원에서는 오토인코더도 멀쩡하다 =============================
@torch.no_grad()
def fig_ae_vs_vae_2d():
    """AE-2와 VAE-2의 잠재 공간을 나란히 둔다.

    교과서 그림은 "AE에는 빈 곳이 있고 VAE가 메운다"고 말하지만, 2차원에서
    실제로 재어 보면 AE도 고르게 뽑힌다(고름 0.977 대 0.952). 이 그림은 그
    사실을 보이기 위한 것이지 VAE의 우월함을 보이기 위한 것이 아니다.
    """
    (Xtr, _), (Xte, yte) = load_flat()
    ae = load(MLPAE, 2, "ae_mlp2")
    vae = VAEModel(2)
    vae.load_state_dict(torch.load(f"{MODELS}/mnist_vae2_beta1.0.pt",
                                   map_location="cpu", weights_only=True))
    vae.eval()

    ae_z = ae.enc(Xte)
    vae_z = vae.mu(vae.enc(Xte))

    fig, axes = plt.subplots(2, 2, figsize=(9.5, 9.0))
    for col, (z, dec, title) in enumerate([
            (ae_z, ae.dec, "AE_MLP-2"),
            (vae_z, vae.dec, "VAE-2  (beta=1)")]):
        ax = axes[0, col]
        ax.scatter(z[:, 0], z[:, 1], c=yte, cmap="tab10", s=1.2,
                   alpha=0.55, linewidths=0, rasterized=True)
        ax.set_title(f"{title} latent", fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])

        lo, hi = z.quantile(0.02, dim=0), z.quantile(0.98, dim=0)
        n = 16
        gx = torch.linspace(lo[0], hi[0], n); gy = torch.linspace(lo[1], hi[1], n)
        grid = torch.stack([torch.stack([x, y]) for y in gy.flip(0) for x in gx])
        d = dec(grid).reshape(n, n, 28, 28)
        canvas = torch.cat([torch.cat([d[i, j] for j in range(n)], 1) for i in range(n)], 0)
        axes[1, col].imshow((canvas * STD + MEAN).clamp(0, 1), cmap="gray", vmin=0, vmax=1)
        axes[1, col].set_title(f"{title} decoded over a grid", fontsize=11)
        axes[1, col].axis("off")

    fig.subplots_adjust(left=0.01, right=0.99, top=0.96, bottom=0.01,
                        wspace=0.06, hspace=0.10)
    fig.savefig("ae_vs_vae_2d.svg", transparent=True)
    plt.close(fig)
    print("wrote ae_vs_vae_2d.svg")


# === 그림 5: 부호 64에서 무너지는 것 — 클래스 몫 ============================
def fig_class_share():
    """심판이 표본 5,000장을 어느 클래스로 읽었는지의 몫.

    고르면 0.1씩이다. 오토인코더는 한 클래스가 66%를 가져간다.
    2차원 그림으로는 보이지 않는 실패이며, 그래서 수로 재야 한다.
    """
    import json
    with open(f"{MODELS}/mnist_class_share.json") as f:
        share = json.load(f)

    fig, ax = plt.subplots(figsize=(9.5, 3.6))
    w = 0.38
    xs = np.arange(10)
    ax.bar(xs - w/2, share["ae_mlp64"], w, label="AE_MLP-64  (balance 0.54)")
    ax.bar(xs + w/2, share["vae64"], w, label="VAE-64  (balance 0.88)")
    ax.axhline(0.1, color="black", lw=1, ls="--", label="uniform (0.1)")
    ax.set_xticks(xs); ax.set_xlabel("digit the judge reads", fontsize=10)
    ax.set_ylabel("share of 5,000 samples", fontsize=10)
    ax.legend(fontsize=9, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.97, bottom=0.16)
    fig.savefig("class_share.svg", transparent=True)
    plt.close(fig)
    print("wrote class_share.svg")


# === 그림 6: 경사 하강법이 PCA의 바닥까지 내려간다 ==========================
# 다른 그림과 달리 학습해 둔 .pt를 읽지 않고 여기서 직접 학습한다. 에포크마다의
# 시험 MSE는 저장된 웨이트에서 되살릴 수 없기 때문이다.
SEED, BATCH, LR, AE_EPOCHS = 42, 100, 1e-3, 100


def train_linear_ae(k, Xtr, Xte, device):
    """선형 AE를 학습하면서 에포크마다 시험 MSE를 잰다.

    씨앗은 초기화 **앞**에 둔다. 뒤에 두면 nn.Linear의 초깃값이 씨앗 밖에
    남아 곡선이 돌릴 때마다 달라진다.
    """
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(SEED)
    model = LinearAE(k).to(device)
    opt = optim.Adam(model.parameters(), lr=LR)
    g = torch.Generator().manual_seed(SEED)
    loader = DataLoader(TensorDataset(Xtr), batch_size=BATCH, shuffle=True, generator=g)

    @torch.no_grad()
    def test_mse():
        model.eval()
        tot = 0.0
        for i in range(0, len(Xte), 1000):
            xb = Xte[i:i + 1000].to(device)
            tot += ((model(xb) - xb) ** 2).mean().item() * len(xb)
        return tot / len(Xte)

    curve = [test_mse()]                      # 에포크 0 = 아직 한 걸음도 딛기 전
    for _ in range(AE_EPOCHS):
        model.train()
        for (xb,) in loader:
            xb = xb.to(device)
            opt.zero_grad()
            F.mse_loss(model(xb), xb).backward()
            opt.step()
        curve.append(test_mse())
    return model.eval().cpu(), curve


def fig_ae_linear_descent():
    """위는 에포크마다의 시험 MSE, 아래는 100 에포크 끝에서 나온 복원.

    위 칸의 요점은 **점선에 가 닿는다**는 것이다. 점선은 PCA가 eigh 한 번으로
    셈한 값이고, 실선은 경사 하강법이 그 값을 더듬어 찾아간 자취다.

    아래 칸의 요점은 **PCA 줄과 AE 줄이 구별되지 않는다**는 것이다. 부호 2에서
    두 줄이 같은 숫자를 같은 모양으로 망가뜨리는 것까지 같다.
    """
    (Xtr, _), (Xte, yte) = load_flat()
    mu, evecs, _ = pca_basis(Xtr)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    trained, floors = {}, {}
    for k in (64, 2):
        V = evecs[:, :k]
        floors[k] = (((((Xte - mu) @ V) @ V.T + mu) - Xte) ** 2).mean().item()
        trained[k] = train_linear_ae(k, Xtr, Xte, device)
        print(f"  k={k:2d}  epoch 100 {trained[k][1][-1]:.5f}   "
              f"PCA floor {floors[k]:.5f}")

    fig = plt.figure(figsize=(11.0, 8.4))
    gs_top = fig.add_gridspec(1, 2, left=0.075, right=0.985, top=0.955, bottom=0.650,
                              wspace=0.20)
    gs_bot = fig.add_gridspec(5, 10, left=0.170, right=0.985, top=0.575, bottom=0.015,
                              wspace=0.06, hspace=0.10)

    # --- 위: 내려가는 곡선과 PCA의 바닥 -----------------------------------
    # y축은 바닥에 바싹 붙여 자른다. 0부터 그리면 둘 다 1 에포크 만에 납작해져
    # "그 뒤로도 떨고 있다"가 보이지 않는다. 에포크 0(학습 전)은 눈금 밖이다.
    panels = [(64, "tab:blue", (0.0940, 0.1270)),
              (2, "tab:orange", (0.5840, 0.7050))]
    for col, (k, color, ylim) in enumerate(panels):
        ax = fig.add_subplot(gs_top[col])
        curve = trained[k][1]
        ax.plot(range(1, len(curve)), curve[1:], color=color, lw=1.7,
                label=f"AE_Linear-{k}, gradient descent")
        ax.axhline(floors[k], color="black", lw=1.1, ls="--",
                   label=f"PCA-{k} floor (eigh) = {floors[k]:.5f}")
        ax.set_xlim(0, AE_EPOCHS); ax.set_ylim(*ylim)
        ax.set_xlabel("epoch", fontsize=10)
        ax.set_ylabel("test reconstruction MSE", fontsize=10)
        ax.legend(fontsize=9, frameon=False, loc="upper right")
        ax.spines[["top", "right"]].set_visible(False)

    # --- 아래: 그 끝에서 나온 복원 ----------------------------------------
    idx = [int((yte == d).nonzero()[0]) for d in range(10)]
    x_flat = Xte[idx]
    V64, V2 = evecs[:, :64], evecs[:, :2]
    with torch.no_grad():
        rows = [("input", x_flat),
                ("PCA-64", ((x_flat - mu) @ V64) @ V64.T + mu),
                ("AE_Linear-64", trained[64][0](x_flat)),
                ("PCA-2", ((x_flat - mu) @ V2) @ V2.T + mu),
                ("AE_Linear-2", trained[2][0](x_flat))]
    for r, (label, imgs) in enumerate(rows):
        for c in range(10):
            a = fig.add_subplot(gs_bot[r, c])
            show(a, imgs[c])
            if c == 0:
                a.text(-0.18, 0.5, label, transform=a.transAxes,
                       ha="right", va="center", fontsize=9)

    fig.savefig("ae_linear_descent.svg", transparent=True)
    plt.close(fig)
    print("wrote ae_linear_descent.svg")


# === 그림 7: 부호 하나를 열 가지 라벨에 걸어 본다 ===========================
class JudgeCNN(nn.Module):
    """3장에서 학습해 둔 심판. 만든 그림이 어느 숫자로 읽히는지 묻는다."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2); self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(64 * 7 * 7, 128); self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        return self.fc2(self.dropout(torch.relu(self.fc1(x.flatten(1)))))


def show_boxed(ax, flat, boxed):
    """show()와 같되, 테두리를 두를 수 있다."""
    img = (flat.reshape(28, 28) * STD + MEAN).clamp(0, 1)
    ax.imshow(img, cmap="gray", vmin=0, vmax=1)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(bool(boxed))
        if boxed:
            sp.set_color("#e8384f"); sp.set_linewidth(2.0)


@torch.no_grad()
def fig_cvae_label_sweep(n_code=8):
    """줄마다 부호 하나를 쥐고, 열 가지 라벨을 차례로 걸어 복호한다.

    cVAE-2는 한 줄이 한 가지 필체로 열 가지 숫자를 써 낸다 — 라벨이 정체를
    맡고 부호는 모양만 맡았다는 뜻이다. cVAE-64는 줄마다 **좋아하는 숫자**가
    있어 라벨을 걸어도 그리로 돌아간다.

    부호는 03_cvae.md 4절이 잰 2,000개 가운데 앞에서부터 n_code개다. 고르지
    않았다는 뜻이며, 같은 씨앗(11)을 쓰므로 그 절의 수와 같은 표본이다.
    """
    judge = JudgeCNN()
    judge.load_state_dict(torch.load(f"{MODELS}/mnist_judge.pt",
                                     map_location="cpu", weights_only=True))
    judge.eval()

    fig = plt.figure(figsize=(13.6, 6.1))
    for col, k in enumerate((2, 64)):
        m = VAEModel(k, 10)
        m.load_state_dict(torch.load(f"{MODELS}/mnist_cvae{k}.pt",
                                     map_location="cpu", weights_only=True))
        m.eval()

        g = torch.Generator().manual_seed(11)          # 4절과 같은 씨앗
        z = torch.randn(2000, k, generator=g)[:n_code]

        left = 0.045 + col * 0.495
        gs = fig.add_gridspec(n_code, 10, left=left, right=left + 0.445,
                              top=0.885, bottom=0.075, wspace=0.06, hspace=0.06)
        for r in range(n_code):
            for lab in range(10):
                c = torch.zeros(1, 10); c[0, lab] = 1.0
                x = m.dec(torch.cat([z[r:r + 1], c], 1))
                read = int(judge(x.reshape(1, 1, 28, 28)).argmax(1))
                ax = fig.add_subplot(gs[r, lab])
                show_boxed(ax, x[0], read != lab)
                if r == 0:
                    ax.set_title(str(lab), fontsize=9, pad=4)
        fig.text(left + 0.2225, 0.945, f"cVAE-{k}", ha="center", fontsize=12)
    fig.text(0.5, 0.028, "each row is one code z, swept across all ten requested labels"
             "   (red box = the judge read something else)",
             ha="center", fontsize=9)
    fig.savefig("cvae_label_sweep.svg", transparent=True)
    plt.close(fig)
    print("wrote cvae_label_sweep.svg")


# === 그림 8: 두 자가 네 스텝을 어떻게 줄 세우는가 ===========================
def fig_two_rulers():
    """5.1 개요의 두 표를 하나의 평면에 놓는다.

    그 절의 요점은 "재는 자가 둘이다"이다. 가로가 복원 MSE, 세로가 숫자 정체이고
    둘은 같은 순서를 주지 않는다. 잠재 차원 64에서는 네 점이 세로로 거의 붙은 채
    가로로만 벌어지고(MSE는 네 배 차이, 정체는 0.7%포인트), 2에서는 두 축이
    함께 움직인다.

    수는 각 쪽에서 재어 개요 표에 실린 값 그대로다. 학습이 필요 없다.
    """
    # (name, MSE, identity, label dx, dy, ha) — PCA와 AE_Linear는 거의 같은 자리에
    # 떨어진다. 그것이 이 절의 주장이므로 겹침을 숨기지 않고 라벨만 위아래로 가른다.
    rows64 = [("PCA",       0.0953, 97.81, -9, -13, "right"),
              ("AE_Linear", 0.0958, 97.83,  9,   5, "left"),
              ("AE_MLP",    0.0546, 98.51,  9,  -4, "left"),
              ("AE_CNN",    0.0243, 98.52,  9,  -4, "left")]
    rows2 = [("PCA",       0.5865, 35.66, -9, -13, "right"),
             ("AE_Linear", 0.5867, 35.54,  9,   4, "left"),
             ("AE_MLP",    0.4223, 70.13,  9,  -4, "left"),
             ("AE_CNN",    0.4742, 61.78,  9,  -4, "left")]
    JUDGE = 98.83                      # 원본을 심판에 넣은 값

    colors = {"PCA": "#777777", "AE_Linear": "tab:blue",
              "AE_MLP": "tab:orange", "AE_CNN": "tab:green"}
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.6))
    for ax, rows, k, ylim in ((axes[0], rows64, 64, (97.3, 99.15)),
                              (axes[1], rows2, 2, (28, 78))):
        for name, mse, acc, dx, dy, ha in rows:
            ax.scatter(mse, acc, s=72, color=colors[name], zorder=3)
            ax.annotate(name, (mse, acc), textcoords="offset points",
                        xytext=(dx, dy), fontsize=9, color=colors[name], ha=ha)
        if k == 64:                     # 천장은 64 쪽에서만 눈금 안에 들어온다
            ax.axhline(JUDGE, color="black", lw=1, ls="--")
            ax.text(0.99, JUDGE + 0.03, "original 98.83%", transform=
                    ax.get_yaxis_transform(), fontsize=8, ha="right", va="bottom")
        ax.set_xlabel("reconstruction MSE   (left is better)", fontsize=10)
        ax.set_ylabel("read as the same digit (%)", fontsize=10)
        ax.set_title(f"latent dimension {k}", fontsize=11)
        ax.set_ylim(*ylim)
        ax.margins(x=0.30)
        ax.spines[["top", "right"]].set_visible(False)
    fig.subplots_adjust(left=0.065, right=0.985, top=0.91, bottom=0.135, wspace=0.24)
    fig.savefig("two_rulers.svg", transparent=True)
    plt.close(fig)
    print("wrote two_rulers.svg")


# === 그림 9: 앞으로 가는 흐름과 그 일정 =====================================
def fig_forward_diffusion(T=1000):
    """5.4절의 앞 흐름을 그림으로 보인다. 학습한 모델이 필요 없다.

    x_t = sqrt(abar_t) x_0 + sqrt(1 - abar_t) eps 이므로 어느 t든 한 번에 간다.
    아래 칸이 요점이다 — 일정이 선형이라고 해서 보이는 것이 고르게 바뀌지는
    않는다. t=400에서 신호 계수가 이미 0.44이고 t=800에서는 0.039이라, 1,000
    스텝 가운데 뒤쪽 수백 스텝은 이미 잡음인 것을 더 흔들 뿐이다.
    """
    (_, _), (Xte, yte) = load_flat()
    betas = torch.linspace(1e-4, 0.02, T)          # 5.4절 코드와 같은 일정
    abar = torch.cumprod(1.0 - betas, 0)
    ts = [0, 50, 100, 200, 300, 400, 600, 800, T - 1]
    digits = [7, 3]                                 # 획이 다른 둘

    g = torch.Generator().manual_seed(0)
    fig = plt.figure(figsize=(11.4, 5.4))
    gs_img = fig.add_gridspec(len(digits), len(ts), left=0.055, right=0.985,
                              top=0.955, bottom=0.50, wspace=0.07, hspace=0.07)
    for r, d in enumerate(digits):
        x0 = Xte[int((yte == d).nonzero()[0])]
        eps = torch.randn(784, generator=g)         # 한 줄에 같은 잡음을 쓴다
        for c, tt in enumerate(ts):
            xt = abar[tt].sqrt() * x0 + (1 - abar[tt]).sqrt() * eps
            ax = fig.add_subplot(gs_img[r, c])
            show(ax, xt)
            if r == 0:
                ax.set_title(f"t={tt}", fontsize=9, pad=4)

    ax = fig.add_subplot(fig.add_gridspec(1, 1, left=0.055, right=0.985,
                                          top=0.42, bottom=0.10)[0])
    tt = torch.arange(T)
    ax.plot(tt, abar.sqrt(), color="tab:blue", lw=1.8,
            label=r"signal  $\sqrt{\bar\alpha_t}$")
    ax.plot(tt, (1 - abar).sqrt(), color="tab:red", lw=1.8,
            label=r"noise  $\sqrt{1-\bar\alpha_t}$")
    for x in ts:
        ax.axvline(x, color="0.8", lw=0.8, zorder=0)
    ax.set_xlim(0, T - 1); ax.set_ylim(0, 1.05)
    ax.set_xlabel("t", fontsize=10)
    ax.set_ylabel("mixing coefficient", fontsize=10)
    ax.legend(fontsize=9, frameon=False, loc="center right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig("forward_diffusion.svg", transparent=True)
    plt.close(fig)
    print("wrote forward_diffusion.svg")


# === 그림 10: KL 항이 무엇을 사고 무엇을 파는가 =============================
def fig_kl_tradeoff():
    """5.2 개요의 '미리 보는 결과' 표를 손잡이 하나의 자취로 보인다.

    가로가 복원 MSE, 세로가 표본 클래스 균형도다. beta를 키우면 오른쪽 위로
    움직인다 — 되살리기를 팔아 커버리지를 산다. 확신도는 거의 그대로여서
    점 옆에 숫자로만 적는다. 학습이 필요 없다. 표에 실린 값 그대로다.
    """
    # (label, 복원 MSE, 클래스 균형도, 확신도)
    pts = [("AE  (no KL)",       0.0546, 0.541, 0.777),
           (r"VAE  $\beta$=1",   0.0750, 0.877, 0.769),
           (r"$\beta$-VAE  $\beta$=4", 0.1351, 0.950, 0.794)]

    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    xs = [q[1] for q in pts]; ys = [q[2] for q in pts]
    ax.plot(xs, ys, color="0.75", lw=1.2, zorder=1)          # 손잡이의 자취
    for (lab, x, y, conf), col in zip(pts, ("#777777", "tab:blue", "tab:orange")):
        ax.scatter(x, y, s=95, color=col, zorder=3)
        ax.annotate(f"{lab}\nconfidence {conf:.3f}", (x, y),
                    textcoords="offset points", xytext=(11, -16),
                    fontsize=9, color=col)
    # 자취 자체가 방향을 보이므로 화살표는 끝에 하나만 둔다
    ax.annotate("", xy=(0.1351, 0.950), xytext=(0.0750, 0.877),
                arrowprops=dict(arrowstyle="-|>", color="0.6", lw=1.2))
    ax.text(0.0545, 0.72, "larger  " + r"$\beta$", fontsize=9.5, color="0.4",
            rotation=62, rotation_mode="anchor")
    ax.set_xlabel("reconstruction MSE   (left is better)", fontsize=10)
    ax.set_ylabel("sample class balance   (up is better)", fontsize=10)
    ax.set_xlim(0.03, 0.175); ax.set_ylim(0.47, 1.02)
    ax.spines[["top", "right"]].set_visible(False)
    fig.subplots_adjust(left=0.11, right=0.985, top=0.96, bottom=0.135)
    fig.savefig("kl_tradeoff.svg", transparent=True)
    plt.close(fig)
    print("wrote kl_tradeoff.svg")


# === 그림 11: 품질은 순전파로 산다 ==========================================
def fig_quality_vs_cost():
    """5장 개요의 마지막 표를 값(순전파)에 대해 그린다.

    가로는 표본 한 장에 드는 순전파 횟수이고 로그 눈금이다. 네 모델이 1에
    모여 있고 DDPM만 1,000에 있다. DDIM이 10에서 그 품질을 거의 되찾는 것이
    이 장의 마지막 이야기다. 오른쪽 칸의 점선은 진짜 MNIST의 1.00이다.

    표에 실린 값 그대로이며 학습이 필요 없다.
    """
    # (label, 순전파, 확신도, 클래스 안 다양함, dy_conf, dy_div)
    # 마지막 두 수는 왼쪽·오른쪽 칸의 라벨 세로 어긋냄이다. 두 칸에서 위아래
    # 순서가 뒤집히는 짝(AE/VAE)이 있어 칸마다 따로 준다.
    rows = [("AE",      1, 0.737, 0.640, -15,   7),
            ("VAE",     1, 0.758, 0.628,   7, -16),
            ("GAN",     1, 0.908, 0.953, -15, -16),
            ("DCGAN",   1, 0.909, 1.085,   7,   7),
            ("DDPM", 1000, 0.940, 1.047,   7,   7),
            ("DDIM-10", 10, 0.904, 0.979,   7, -16)]
    col = {"AE": "#777777", "VAE": "tab:blue", "GAN": "tab:green",
           "DCGAN": "tab:olive", "DDPM": "tab:red", "DDIM-10": "tab:orange"}

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.4))
    for ax, j, ylab, ylim in ((axes[0], 2, "judge confidence", (0.70, 0.98)),
                              (axes[1], 3, "within-class variety", (0.55, 1.16))):
        for r in rows:
            name, n, conf, div = r[0], r[1], r[2], r[3]
            y = r[j]
            dy = r[4] if j == 2 else r[5]
            ax.scatter(n, y, s=85, color=col[name], zorder=3)
            ax.annotate(name, (n, y), textcoords="offset points",
                        xytext=(9, dy), fontsize=9, color=col[name])
        if j == 3:
            ax.axhline(1.00, color="black", lw=1, ls="--")
            ax.text(1.05, 1.005, "real MNIST 1.00", fontsize=8, va="bottom")
        ax.set_xscale("log")
        ax.set_xlim(0.6, 3000)
        ax.set_ylim(*ylim)
        ax.set_xlabel("forward passes per sample   (log)", fontsize=10)
        ax.set_ylabel(ylab, fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
    fig.subplots_adjust(left=0.07, right=0.985, top=0.96, bottom=0.145, wspace=0.22)
    fig.savefig("quality_vs_cost.svg", transparent=True)
    plt.close(fig)
    print("wrote quality_vs_cost.svg")


# === 그림 12: GAN과 DCGAN이 만든 그림 ======================================
def _grid(ax, samples, n=8):
    """[-1, 1] 범위의 (N,1,28,28)을 n x n 한 장으로 이어 붙여 그린다."""
    im = samples[:n * n, 0]
    canvas = torch.cat([torch.cat([im[r * n + c] for c in range(n)], 1)
                        for r in range(n)], 0)
    ax.imshow((canvas + 1) / 2, cmap="gray", vmin=0, vmax=1)
    ax.axis("off")


def fig_gan_samples():
    """5.3절이 만든 표본. 학습해 둔 생성기에서 뽑아 저장한 것을 읽어 쓴다."""
    s = torch.load(f"{MODELS}/mnist_gan_samples.pt", map_location="cpu",
                   weights_only=True)
    fig, ax = plt.subplots(figsize=(5.6, 5.6))
    _grid(ax, s)
    fig.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005)
    fig.savefig("gan_samples.svg", transparent=True)
    plt.close(fig)
    print("wrote gan_samples.svg")


def fig_gan_vs_dcgan():
    """두 구조가 만든 표본을 나란히 둔다.

    두 표본은 같은 씨앗(7)의 같은 z 64개를 넣어 얻은 것이다. 다만 **칸마다
    짝이 맞지는 않는다** — 따로 학습한 두 생성기는 같은 z를 서로 다른 숫자로
    보내므로, 견줄 것은 칸 대 칸이 아니라 획의 결이다.
    """
    g = torch.load(f"{MODELS}/mnist_gan_samples.pt", map_location="cpu",
                   weights_only=True)
    d = torch.load(f"{MODELS}/mnist_dcgan_samples.pt", map_location="cpu",
                   weights_only=True)
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.75))
    for ax, s, title in ((axes[0], g, "GAN  (MLP generator)"),
                         (axes[1], d, "DCGAN  (conv generator)")):
        _grid(ax, s)
        ax.set_title(title, fontsize=11, pad=6)
    fig.subplots_adjust(left=0.005, right=0.995, top=0.945, bottom=0.005,
                        wspace=0.04)
    fig.savefig("gan_vs_dcgan.svg", transparent=True)
    plt.close(fig)
    print("wrote gan_vs_dcgan.svg")


# === 그림 13: 스텝을 줄여도 같은 그림이 나오는가 ============================
def fig_ddim_steps():
    """같은 z에서 스텝 수만 바꾸어 뽑은 것을 줄마다 늘어놓는다.

    sample_ddim이 안에서 씨앗을 고정하므로 네 줄은 **같은 z**에서 출발한다.
    그래서 여기서는 칸끼리 짝이 맞고, 달라진 것은 스텝 수뿐이다. 100에서
    20까지는 같은 숫자가 거의 그대로 나오고 10에서야 획이 뭉개지기 시작한다.
    """
    d = torch.load(f"{MODELS}/mnist_ddim_steps.pt", map_location="cpu",
                   weights_only=True)
    steps = [100, 50, 20, 10]
    conf = {100: 0.924, 50: 0.923, 20: 0.916, 10: 0.904}   # 쪽의 출력 블록에서
    n = d[100].shape[0]

    fig, axes = plt.subplots(len(steps), n, figsize=(10.4, 5.5))
    for r, s in enumerate(steps):
        im = d[s]
        for c in range(n):
            ax = axes[r, c]
            ax.imshow((im[c, 0] + 1) / 2, cmap="gray", vmin=0, vmax=1)
            ax.axis("off")
        axes[r, 0].text(-0.14, 0.5, f"{s} steps\nconf {conf[s]:.3f}",
                        transform=axes[r, 0].transAxes, ha="right",
                        va="center", fontsize=9)
    fig.subplots_adjust(left=0.135, right=0.995, top=0.99, bottom=0.01,
                        wspace=0.06, hspace=0.08)
    fig.savefig("ddim_steps.svg", transparent=True)
    plt.close(fig)
    print("wrote ddim_steps.svg")


# === 그림 14: 자를 바꾸면 줄이 바뀐다 =======================================
def fig_rulers_disagree():
    """5장 개요의 표를 자마다 등수로 바꾸어 세 줄로 세운다.

    같은 여섯 모델, 같은 한 벌의 표본인데 자를 바꾸면 줄이 바뀐다. 선이
    엇갈리는 자리가 요점이다. 수는 모두 개요 표의 값이고 등수만 여기서
    셈한다 — 다양함은 1.00이 기준이므로 |1 - x|로 줄을 세운다.
    """
    m = {"AE": (0.737, 0.580, 0.640), "VAE": (0.758, 0.887, 0.628),
         "GAN": (0.908, 0.945, 0.953), "DCGAN": (0.909, 0.970, 1.085),
         "DDPM": (0.940, 0.986, 1.047), "DDIM-10": (0.904, 0.971, 0.979)}
    rulers = [("judge\nconfidence", lambda v: -v[0]),
              ("class\nbalance", lambda v: -v[1]),
              ("within-class variety\nscored as |1 - x|", lambda v: abs(v[2] - 1))]
    order = [sorted(m, key=lambda n, k=k: k(m[n])) for _, k in rulers]
    rank = {k: [o.index(k) + 1 for o in order] for k in m}
    assert rank["DDIM-10"] == [4, 2, 1] and rank["DCGAN"] == [2, 3, 4]
    assert rank["AE"][1:] == [6, 5] and rank["VAE"][1:] == [5, 6]

    col = {"AE": "#999999", "VAE": "tab:blue", "GAN": "tab:green",
           "DCGAN": "tab:olive", "DDPM": "tab:red", "DDIM-10": "tab:orange"}
    xs = [0, 1, 2]
    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    for name, rk in rank.items():
        ax.plot(xs, rk, "-o", color=col[name], lw=2.0, ms=7, zorder=3)
        ax.text(-0.08, rk[0], name, ha="right", va="center",
                fontsize=9.5, color=col[name])
        ax.text(2.08, rk[2], name, ha="left", va="center",
                fontsize=9.5, color=col[name])
    for x in xs:
        ax.axvline(x, color="0.88", lw=1.0, zorder=0)
    ax.set_xticks(xs)
    ax.set_xticklabels([r for r, _ in rulers], fontsize=9.5)
    ax.set_yticks(range(1, 7))
    ax.set_ylabel("rank", fontsize=10)
    ax.set_ylim(6.6, 0.4)                               # 1등이 위
    ax.set_xlim(-0.95, 2.95)
    ax.tick_params(axis="x", length=0)
    ax.spines[["top", "right", "bottom"]].set_visible(False)
    fig.subplots_adjust(left=0.16, right=0.86, top=0.97, bottom=0.14)
    fig.savefig("rulers_disagree.svg", transparent=True)
    plt.close(fig)
    print("wrote rulers_disagree.svg")


# === 그림 15: 같은 스텝2, 다른 결과 =========================================
def fig_step2_two_ladders():
    """두 사다리의 스텝2가 같은 수법인데 결과가 갈린다.

    둘 다 "닫힌 형태 해 -> 경사 하강법"이고 둘 다 같은 자(심판이 읽은 %)로
    쟀다. 분류 쪽은 10.32%포인트를 벌고 표현 쪽은 0.02%포인트에 그친다.
    표현 쪽의 0은 실험이 모자라서가 아니라 1988년에 미리 알려진 값이다.

    수는 4.1절과 이 절의 표에 실린 것이다.
    """
    bars = [("classification ladder\n(templates -> linear, trained)", 10.32, "tab:green"),
            ("representation ladder\n(PCA -> AE_Linear)", 0.02, "#777777")]
    fig, ax = plt.subplots(figsize=(7.8, 4.0))
    for i, (lab, v, c) in enumerate(bars):
        ax.barh(i, v, height=0.46, color=c, alpha=0.85, zorder=3)
        ax.text(v + 0.22, i, f"+{v:.2f} points", va="center", fontsize=10,
                color=c, fontweight="bold")
    ax.axvline(0, color="0.5", lw=1.0)
    # 아래 막대에는 이론이 미리 적어 둔 값이 있다
    # 아래(표현 사다리) 막대를 가리켜야 한다 - invert_yaxis 뒤 index 1 이 아래다
    ax.annotate("theory, 1988: this step\ncannot gain anything",
                xy=(0.10, 1.20), xytext=(3.0, 1.50), fontsize=9, color="#777777",
                arrowprops=dict(arrowstyle="->", color="#999999", lw=1.1))
    ax.set_yticks(range(len(bars)))
    ax.set_yticklabels([b[0] for b in bars], fontsize=9.5)
    ax.set_ylim(-0.6, 1.75)
    ax.set_xlim(-0.4, 14.2)
    ax.set_xlabel("what step 2 gained, in points of judge-read accuracy", fontsize=10)
    ax.invert_yaxis()
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.subplots_adjust(left=0.30, right=0.985, top=0.95, bottom=0.17)
    fig.savefig("step2_two_ladders.svg", transparent=True)
    plt.close(fig)
    print("wrote step2_two_ladders.svg")


# === 그림 16: 피어슨이 1901년에 던진 물음 ===================================
def fig_pearson_two_lines():
    """같은 점에 세 직선. 둘은 "무엇이 정답인가"에 따라 갈리고 하나는 갈리지 않는다.

    점은 씨앗 0으로 이 코드가 만든 것이고 기울기 셋도 여기서 잰다. 피어슨
    논문에 실린 수가 아니다 — 보이려는 것은 수가 아니라 **갈린다는 사실**이다.
    """
    rng = np.random.default_rng(0)
    n = 120
    x = rng.normal(0, 1, n)
    y = 0.8 * x + rng.normal(0, 0.6, n)
    mx, my = x.mean(), y.mean()

    b_yx = np.polyfit(x, y, 1)[0]                       # y 를 정답으로 두고
    b_xy = 1.0 / np.polyfit(y, x, 1)[0]                 # x 를 정답으로 두고
    Xc = np.c_[x - mx, y - my]
    Vt = np.linalg.svd(Xc, full_matrices=False)[2]
    b_pca = Vt[0, 1] / Vt[0, 0]
    # 축을 바꿔 다시 재면 같은 직선으로 돌아온다
    Vs = np.linalg.svd(np.c_[y - my, x - mx], full_matrices=False)[2]
    assert abs(1.0 / (Vs[0, 1] / Vs[0, 0]) - b_pca) < 1e-9

    fig, ax = plt.subplots(figsize=(7.4, 5.4))
    ax.scatter(x, y, s=13, color="0.55", alpha=0.65, linewidths=0, zorder=2)
    xs = np.linspace(x.min() - 0.3, x.max() + 0.3, 2)
    for b, c, lab in ((b_yx, "tab:blue", f"treat y as the answer   slope {b_yx:.2f}"),
                      (b_xy, "tab:green", f"treat x as the answer   slope {b_xy:.2f}"),
                      (b_pca, "#c2410c", f"ask neither (PCA)        slope {b_pca:.2f}")):
        ax.plot(xs, my + b * (xs - mx), color=c, lw=2.1, label=lab, zorder=3)
    ax.text(0.02, 0.97,
            "swap x and y, redo the orange line:\nit comes back the same line",
            transform=ax.transAxes, va="top", fontsize=9, color="#c2410c")
    ax.legend(fontsize=9, frameon=False, loc="lower right")
    ax.set_xlabel("x", fontsize=10); ax.set_ylabel("y", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    fig.subplots_adjust(left=0.10, right=0.98, top=0.97, bottom=0.11)
    fig.savefig("pearson_two_lines.svg", transparent=True)
    plt.close(fig)
    print("wrote pearson_two_lines.svg")


def fig_diffusion_attention(T=1000):
    """2015년 방식과 2020년 방식이 1,000스텝에 품을 어떻게 나누어 주는가.

    왼쪽 칸의 날짜는 각 논문이 arXiv 에 처음 올라온 달이고, 오른쪽 칸의
    곡선은 **이 쪽이 쓰는 일정**(`betas = linspace(1e-4, 0.02, 1000)`)으로
    여기서 셈한 것이다. 어느 논문에 실린 수도 아니다.

    무게는 Ho 2020 식 (12) 가 L_simple 에서 떼어 낸 계수
    beta_t^2 / (2 sigma_t^2 alpha_t (1 - abar_t)) 에 sigma_t^2 = beta_t 를
    넣은 것이다. 둘을 나란히 보려고 각각 합이 100%가 되게 고쳐 적었다 —
    L_simple 쪽의 상수는 그 자체로는 뜻이 없기 때문이다.
    """
    betas = np.linspace(1e-4, 0.02, T)
    alphas = 1.0 - betas
    abar = np.cumprod(alphas)
    lam = betas / (2 * alphas * (1 - abar))
    old = lam / lam.sum() * 100                      # 2015: 무게가 붙은 하한
    new = np.full(T, 100.0 / T)                      # 2020: 모든 스텝이 같다
    t = np.arange(1, T + 1)
    head = old[:100].sum()                           # 앞 100스텝이 가져가는 몫
    assert abs(old.sum() - 100) < 1e-6 and abs(new.sum() - 100) < 1e-6

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.6, 4.9),
                                   gridspec_kw={"width_ratios": [1.05, 1]})

    # --- 왼쪽: 이 장의 여섯 모델이 세상에 나온 차례 -------------------------
    rows = [("VAE", 2013.96, "tab:blue"), ("GAN", 2014.46, "tab:red"),
            ("diffusion\n(the idea)", 2015.21, "#c2410c"),
            ("DCGAN", 2015.88, "tab:red"),
            ("DDPM", 2020.46, "#c2410c"), ("DDIM", 2020.79, "#c2410c")]
    for i, (name, yr, c) in enumerate(rows):
        up = i % 2 == 0
        axL.plot([yr, yr], [0, 0.42 if up else -0.42], color=c, lw=1.0, alpha=0.5)
        axL.plot([yr], [0], "o", color=c, ms=8, zorder=3)
        axL.text(yr, 0.52 if up else -0.52, name, ha="center",
                 va="bottom" if up else "top", fontsize=9.5, color=c)
    axL.annotate("", xy=(2020.40, -0.95), xytext=(2015.27, -0.95),
                 arrowprops=dict(arrowstyle="<->", color="0.4", lw=1.2))
    axL.text(2017.8, -1.08, "five years, and almost nobody used it",
             ha="center", va="top", fontsize=9.5, color="0.35")
    axL.axhline(0, color="0.3", lw=1.1)
    axL.set_xlim(2013.4, 2021.3); axL.set_ylim(-1.75, 1.3)
    axL.set_yticks([]); axL.set_xticks(range(2014, 2022, 2))
    axL.set_xlabel("year the paper was first posted", fontsize=10)
    axL.set_title("the idea did not arrive last", fontsize=11)
    axL.spines[["top", "right", "left"]].set_visible(False)

    # --- 오른쪽: 품을 어떻게 나누는가 ---------------------------------------
    axR.plot(t, old, color="#c2410c", lw=2.0, label="the 2015 recipe")
    axR.plot(t, new, color="tab:green", lw=2.0, label="the 2020 recipe (this page)")
    axR.set_yscale("log")
    axR.fill_between(t[:100], 1e-4, old[:100], color="#c2410c", alpha=0.12)
    axR.annotate(f"the first 100 steps take {head:.0f}%\nof all the practice",
                 xy=(28, old[27]), xytext=(150, 1.9),
                 arrowprops=dict(arrowstyle="->", color="#c2410c", lw=1.1),
                 fontsize=9.5, color="#c2410c")
    axR.text(640, 0.145, "the 2020 recipe gives\nthose same 100 steps 10%",
             fontsize=9.5, color="tab:green", ha="center")
    axR.set_xlabel("step  (1 = picture almost clean,  1000 = pure noise)", fontsize=10)
    axR.set_ylabel("share of all the practice,\nper step  (%)", fontsize=10)
    axR.set_ylim(3e-3, 9)
    axR.legend(fontsize=9.5, frameon=False, loc="lower center")
    axR.set_title("what 2020 changed: where the practice goes", fontsize=11)
    axR.spines[["top", "right"]].set_visible(False)

    fig.text(0.5, 0.012, "left: posting dates.  right: computed here from this "
             "page's own schedule - not a number from either paper.",
             ha="center", fontsize=8.5, color="0.45")
    fig.subplots_adjust(left=0.055, right=0.985, top=0.92, bottom=0.17, wspace=0.26)
    fig.savefig("diffusion_attention.svg", transparent=True)
    plt.close(fig)
    print(f"wrote diffusion_attention.svg  (first 100 steps: {head:.1f}% vs 10%)")


def fig_gan_saturation():
    """유명한 식과 실제로 돌아가는 식이 생성기에 주는 힌트의 세기.

    왼쪽은 꼴이다. 판별기가 가짜에 매기는 점수 D 에 대해 기울기 크기가
    버려진 식은 D, 쓰이는 식은 1 - D 라 둘의 비가 정확히 D / (1 - D) 다.
    D = 0.5 에서만 둘이 같다.

    오른쪽 막대는 **이 쪽의 로스 표**에서 되짚은 것이다. 이 쪽 코드가
    `bce(D(fake), ones)` 를 쓰므로 L_G = -log D 이고, 에포크 평균을
    되짚으면 D 의 기하평균이 나온다. 논문에 실린 수가 아니다.
    """
    d = np.linspace(1e-3, 0.999, 600)
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.2, 4.6),
                                   gridspec_kw={"width_ratios": [1.15, 1]})

    # --- 왼쪽: 힌트의 세기 --------------------------------------------------
    axL.plot(d, d, color="#b91c1c", lw=2.1,
             label="the famous equation (abandoned in sec. 3)")
    axL.plot(d, 1 - d, color="tab:green", lw=2.1,
             label="the one this page runs")
    axL.axvline(0.5, color="0.6", lw=1.0, ls=":")
    axL.plot([0.5], [0.5], "o", color="0.35", ms=7, zorder=4)
    axL.annotate("they are equally strong\nonly at the balance point",
                 xy=(0.5, 0.5), xytext=(0.60, 0.80), fontsize=9.5, color="0.35",
                 arrowprops=dict(arrowstyle="->", color="0.5", lw=1.0))
    axL.annotate("judge is sure it is fake:\nthe forger gets almost no hint",
                 xy=(0.05, 0.05), xytext=(0.53, 0.26), fontsize=9.5,
                 color="#b91c1c", ha="center",
                 arrowprops=dict(arrowstyle="->", color="#b91c1c", lw=1.1))
    axL.set_xlabel("D - the score the judge gives a fake picture", fontsize=10)
    axL.set_ylabel("how strong a hint the forger gets", fontsize=10)
    axL.set_xlim(0, 1); axL.set_ylim(0, 1.05)
    axL.legend(fontsize=9.3, frameon=False, loc="upper center")
    axL.set_title("the shape", fontsize=11)
    axL.spines[["top", "right"]].set_visible(False)

    # --- 오른쪽: 이 쪽의 네 에포크 -------------------------------------------
    eps = [(1, 1.687), (10, 2.607), (20, 1.360), (30, 1.310)]
    labs = [f"epoch {e}" for e, _ in eps]
    vals = [(1 - np.exp(-lg)) / np.exp(-lg) for _, lg in eps]
    cols = ["0.65", "#b91c1c", "0.65", "0.65"]
    bars = axR.bar(labs, vals, color=cols, width=0.62)
    for b, v in zip(bars, vals):
        axR.text(b.get_x() + b.get_width() / 2, v + 0.35, f"{v:.1f}x",
                 ha="center", fontsize=10)
    axR.axhline(1.0, color="0.4", lw=1.1, ls="--")
    axR.text(3.45, 1.45, "1x = no difference", ha="right", fontsize=9,
             color="0.4")
    axR.text(1, 16.6, "the epoch this page marks\nas the judge pulling ahead",
             ha="center", va="top", fontsize=9.3, color="#b91c1c")
    axR.set_ylim(0, 17)
    axR.set_ylabel("how much fainter the hint\nwould have been", fontsize=10)
    axR.set_title("this page's own run", fontsize=11)
    axR.spines[["top", "right"]].set_visible(False)

    fig.text(0.5, 0.015, "right panel: read back out of this page's L_G column "
             "- not a number from the paper.",
             ha="center", fontsize=8.5, color="0.45")
    fig.subplots_adjust(left=0.065, right=0.985, top=0.92, bottom=0.175,
                        wspace=0.27)
    fig.savefig("gan_saturation.svg", transparent=True)
    plt.close(fig)
    print(f"wrote gan_saturation.svg  (epoch 10: {vals[1]:.1f}x fainter)")


def fig_dcgan_bundle():
    """GAN 에서 DCGAN 으로 넘어갈 때 실제로 몇 가지가 바뀌었는가.

    왼쪽 목록과 오른쪽 막대는 모두 **이 책 두 쪽의 코드에서 읽은 것**이다.
    파라미터 수는 거기 적힌 모듈을 그대로 세운 뒤 센 것이고, 어느 논문의
    수도 아니다.
    """
    import torch.nn as nn
    Z = 64
    g_mlp = nn.Sequential(nn.Linear(Z, 256), nn.LeakyReLU(0.2),
                          nn.Linear(256, 512), nn.LeakyReLU(0.2),
                          nn.Linear(512, 784), nn.Tanh())
    d_mlp = nn.Sequential(nn.Flatten(), nn.Linear(784, 512), nn.LeakyReLU(0.2),
                          nn.Linear(512, 256), nn.LeakyReLU(0.2),
                          nn.Linear(256, 1))
    g_cnv = nn.ModuleList([
        nn.Sequential(nn.Linear(Z, 128 * 7 * 7), nn.BatchNorm1d(128 * 7 * 7),
                      nn.ReLU()),
        nn.Sequential(nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64),
                      nn.ReLU(), nn.ConvTranspose2d(64, 1, 4, 2, 1), nn.Tanh())])
    d_cnv = nn.Sequential(nn.Conv2d(1, 64, 4, 2, 1), nn.LeakyReLU(0.2),
                          nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128),
                          nn.LeakyReLU(0.2), nn.Flatten(),
                          nn.Linear(128 * 7 * 7, 1))
    cnt = lambda m: sum(p.numel() for p in m.parameters())
    gm, gc, dm, dc = cnt(g_mlp), cnt(g_cnv), cnt(d_mlp), cnt(d_cnv)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.8, 4.5),
                                   gridspec_kw={"width_ratios": [1.45, 1]})

    # --- 왼쪽: 무엇이 그대로이고 무엇이 바뀌었나 -----------------------------
    same = ["seed 42", "batch 100", "30 epochs", "z size 64",
            "Adam, lr 2e-4", "betas (0.5, 0.999)", "the same loss",
            "the same judge"]
    diff = ["fully connected  ->  convolution",
            "no batchnorm  ->  batchnorm, both sides",
            "generator's LeakyReLU  ->  ReLU",
            f"discriminator {dm:,}  ->  {dc:,} weights"]
    axL.text(0.02, 0.95, "held fixed", fontsize=11, color="0.35",
             transform=axL.transAxes, weight="bold")
    for i, s in enumerate(same):
        axL.text(0.035, 0.86 - i * 0.072, f"-  {s}", fontsize=9.6, color="0.45",
                 transform=axL.transAxes)
    axL.text(0.46, 0.95, "changed - the page names only the first",
             fontsize=11, color="#b91c1c", transform=axL.transAxes,
             weight="bold")
    for i, s in enumerate(diff):
        axL.text(0.475, 0.86 - i * 0.105, f"{i+1}.  {s}", fontsize=9.6,
                 color="#b91c1c", transform=axL.transAxes)
    axL.axvline(0.44, color="0.8", lw=1.0)
    axL.set_xticks([]); axL.set_yticks([])
    axL.spines[:].set_visible(False)
    axL.set_title("one arrow in the table, four changes in the code",
                  fontsize=11)

    # --- 오른쪽: 파라미터 수 -------------------------------------------------
    x = np.arange(2)
    axR.bar(x - 0.19, [gm, dm], 0.37, label="MLP  (prev. page)", color="0.62")
    axR.bar(x + 0.19, [gc, dc], 0.37, label="conv  (this page)", color="tab:blue")
    for xi, (a, b) in zip(x, [(gm, gc), (dm, dc)]):
        axR.text(xi - 0.19, a + 14000, f"{a/1000:.0f}k", ha="center", fontsize=9.3)
        axR.text(xi + 0.19, b + 14000, f"{b/1000:.0f}k", ha="center", fontsize=9.3)
    axR.text(0, 625000, "same size (ratio 1.00)", ha="center", fontsize=9.6,
             color="0.3")
    axR.annotate(f"a quarter of it\n(ratio {dc/dm:.2f})", xy=(1.19, 160000),
                 xytext=(1.19, 320000), ha="center", fontsize=9.6,
                 color="#b91c1c",
                 arrowprops=dict(arrowstyle="->", color="#b91c1c", lw=1.1))
    axR.set_xticks(x)
    axR.set_xticklabels(["the one that draws", "the one that judges"],
                        fontsize=10)
    axR.set_ylabel("number of weights", fontsize=10)
    axR.set_ylim(0, 790000)
    axR.legend(fontsize=9.3, frameon=False, loc="upper right")
    axR.set_title("and one of them quietly shrank", fontsize=11)
    axR.spines[["top", "right"]].set_visible(False)

    fig.text(0.5, 0.015, "every number here was read off the code on these two "
             "pages - none of it comes from the DCGAN paper.",
             ha="center", fontsize=8.5, color="0.45")
    fig.subplots_adjust(left=0.025, right=0.985, top=0.91, bottom=0.12,
                        wspace=0.18)
    fig.savefig("dcgan_bundle.svg", transparent=True)
    plt.close(fig)
    print(f"wrote dcgan_bundle.svg  (G {gm:,}->{gc:,}, D {dm:,}->{dc:,})")


def fig_ddim_invertible(T=1000):
    """DDIM 을 거꾸로 돌려 z 를 되찾을 때, 스텝을 줄이면 얼마나 어긋나는가.

    왼쪽은 **장난감**이다. 자료를 1차원 두 봉우리 섞음으로 두면 참
    잡음예측기를 식으로 적을 수 있어 학습한 모델 없이 DDIM 의 두 식을
    그대로 돌릴 수 있다. MNIST 에서 잰 수가 아니다.

    되짚기가 근사인 까닭: 뽑는 걸음은 eps(x_t, t) 로 t -> t' 로 가는데,
    거꾸로 가려면 그 eps 가 있어야 하지만 손에 있는 것은 x_t' 뿐이라
    eps(x_t', t') 로 바꿔치기한다. 스텝이 많을수록 그 틈이 작다.

    오른쪽은 이 쪽 표의 시간 열을 그대로 옮긴 것이다.
    """
    betas = np.linspace(1e-4, 0.02, T)
    abar = np.cumprod(1.0 - betas)
    MU, SD, W = np.array([-2.0, 2.0]), np.array([0.5, 0.5]), np.array([0.5, 0.5])

    def eps_true(x, t):
        a = abar[t]
        var = a * SD ** 2 + (1 - a)
        d = x[:, None] - (np.sqrt(a) * MU)[None, :]
        lp = -0.5 * d ** 2 / var - 0.5 * np.log(var) + np.log(W)
        w = np.exp(lp - lp.max(1, keepdims=True)); w /= w.sum(1, keepdims=True)
        return -np.sqrt(1 - a) * (w * (-d / var)).sum(1)

    def hop(x, t, tn):
        e = eps_true(x, t)
        x0 = (x - np.sqrt(1 - abar[t]) * e) / np.sqrt(abar[t])
        an = abar[tn] if tn >= 0 else 1.0
        return np.sqrt(an) * x0 + np.sqrt(1 - an) * e

    def roundtrip(n):
        ts = np.linspace(T - 1, 0, n + 1).round().astype(int)
        z = np.random.default_rng(0).normal(size=4000)
        x = z.copy()
        for i in range(n):
            x = hop(x, ts[i], ts[i + 1])
        for i in range(n, 0, -1):
            x = hop(x, ts[i], ts[i - 1])
        return np.sqrt(np.mean((x - z) ** 2)) / z.std() * 100

    steps = [10, 20, 50, 100, 200, 500, 1000]
    err = [roundtrip(n) for n in steps]
    page = {1000: 2495, 100: 290, 50: 158, 20: 69, 10: 26}

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.4, 4.7))

    axL.plot(steps, err, "o-", color="#c2410c", lw=2.0, ms=6)
    for n in (10, 100, 1000):
        e = err[steps.index(n)]
        axL.annotate(f"{e:.1f}%", xy=(n, e), xytext=(16, 7),
                     textcoords="offset points", ha="left", fontsize=10,
                     color="#c2410c")
    axL.axhspan(10, 60, color="#c2410c", alpha=0.07)
    axL.text(620, 22, "a z this wrong is\nno use as a code", fontsize=9.5,
             color="#c2410c", ha="right")
    axL.set_xscale("log"); axL.set_yscale("log")
    axL.set_xlabel("number of steps", fontsize=10)
    axL.set_ylabel("how far off the recovered z is  (%)", fontsize=10)
    axL.set_title("toy run - can you get the z back?", fontsize=11)
    axL.grid(alpha=0.25, which="both", lw=0.5)
    axL.spines[["top", "right"]].set_visible(False)

    ks = sorted(page)
    axR.plot(ks, [page[k] for k in ks], "s-", color="tab:blue", lw=2.0, ms=6)
    for k in ks:
        axR.annotate(f"{page[k]}s", xy=(k, page[k]), xytext=(0, 11),
                     textcoords="offset points", ha="center", fontsize=9.5,
                     color="tab:blue")
    axR.set_xscale("log"); axR.set_yscale("log")
    axR.set_xlim(7.5, 1700); axR.set_ylim(18, 4600)
    axR.set_xlabel("number of steps", fontsize=10)
    axR.set_ylabel("seconds to draw 5,000 pictures", fontsize=10)
    axR.set_title("this page's table - what the steps cost", fontsize=11)
    axR.grid(alpha=0.25, which="both", lw=0.5)
    axR.spines[["top", "right"]].set_visible(False)

    fig.text(0.5, 0.015, "the same dial moves both: the cheap end is also the "
             "end where the code cannot be recovered.",
             ha="center", fontsize=9, color="0.4")
    fig.subplots_adjust(left=0.07, right=0.985, top=0.91, bottom=0.17,
                        wspace=0.24)
    fig.savefig("ddim_invertible.svg", transparent=True)
    plt.close(fig)
    print(f"wrote ddim_invertible.svg  (10 steps {err[0]:.1f}%, "
          f"1000 steps {err[-1]:.1f}%)")


def fig_cvae_condition(N=12000, K=8):
    """조건이 답을 얼마나 좁히는가 — 사분면을 주는 것과 라벨을 주는 것.

    CVAE 를 내놓은 글이 MNIST 에서 쓴 조건은 라벨이 아니라 **그림의 사분면
    한둘셋**이었다. 그 조건은 답을 하나로 좁히지 못한다. 여기서는 모델을
    쓰지 않고 **진짜 MNIST 만으로** 그것을 센다 — 주어진 자리가 가장 비슷한
    진짜 숫자 K 개를 찾아 그 라벨이 몇 가지인지 보는 것이다.

    왼쪽 띠의 질의 하나는 효과가 잘 보이게 고른 것이고, 오른쪽 막대가
    고르지 않고 센 값이다.
    """
    tf = transforms.ToTensor()
    ds = torchvision.datasets.MNIST(DATA, train=True, download=True, transform=tf)
    X = torch.stack([ds[i][0][0] for i in range(N)])
    y = torch.tensor([ds[i][1] for i in range(N)])

    def feats(nq):
        if nq == 1:
            return X[:, :14, :14].reshape(N, -1)
        if nq == 2:
            return X[:, :14, :].reshape(N, -1)
        f = X.clone(); f[:, 14:, 14:] = 0
        return f.reshape(N, -1)

    g = torch.Generator().manual_seed(0)
    qs = torch.randperm(N, generator=g)[:800]
    spans = []
    for nq in (1, 2, 3):
        F = feats(nq); c = []
        for qi in qs:
            d = ((F - F[qi]) ** 2).sum(1); d[qi] = 1e9
            c.append(len(set(y[d.argsort()[:K]].tolist())))
        spans.append(float(torch.tensor(c, dtype=torch.float).mean()))

    # 효과가 잘 보이는 질의 하나를 고른다 (왼쪽 띠 전용)
    F1 = feats(1)
    ink = F1.sum(1)
    floor = float(ink.median())          # 거의 빈 사분면은 아무것과나 맞는다
    best, bq = -1, None
    for qi in qs[:300]:
        if ink[qi] < floor:
            continue
        d = ((F1 - F1[qi]) ** 2).sum(1); d[qi] = 1e9
        nb = d.argsort()[:6]
        k = len(set(y[nb].tolist()))
        if k > best:
            best, bq, bnb = k, int(qi), nb

    fig = plt.figure(figsize=(12.6, 4.4))
    gs = fig.add_gridspec(2, 8, height_ratios=[1, 1.05],
                          left=0.105, right=0.985, top=0.84, bottom=0.14,
                          wspace=0.35, hspace=0.45)

    # --- 왼쪽 위: 주어진 자리 + 그와 맞는 진짜 숫자들 -----------------------
    axq = fig.add_subplot(gs[0, 0])
    shown = X[bq].clone(); shown[14:, :] = 0.12; shown[:14, 14:] = 0.12
    axq.imshow(shown, cmap="gray_r"); axq.set_xticks([]); axq.set_yticks([])
    axq.set_title("what you are given", fontsize=9)
    for s in axq.spines.values():
        s.set_color("#c2410c"); s.set_linewidth(1.6)
    for i, j in enumerate(bnb):
        ax = fig.add_subplot(gs[0, i + 1])
        ax.imshow(X[j], cmap="gray_r"); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"a {y[j].item()}", fontsize=9)
        ax.spines[:].set_visible(False)
    fig.text(0.105, 0.90, "one quadrant, and the real digits that match it - "
             f"{best} different answers",
             fontsize=10.5, color="#c2410c")

    # --- 아래: 조건이 답을 얼마나 좁히는가 ----------------------------------
    axb = fig.add_subplot(gs[1, :])
    labs = ["1 quadrant", "2 quadrants", "3 quadrants", "the label\n(this page)"]
    vals = spans + [1.0]
    cols = ["#c2410c"] * 3 + ["tab:blue"]
    bars = axb.barh(labs[::-1], vals[::-1], color=cols[::-1], height=0.6)
    for b, v in zip(bars, vals[::-1]):
        axb.text(v + 0.03, b.get_y() + b.get_height() / 2, f"{v:.2f}",
                 va="center", fontsize=10)
    axb.axvline(1.0, color="0.55", lw=1.0, ls="--")
    axb.set_xlim(0, 2.75)
    axb.set_xlabel(f"how many different digits the {K} closest real matches "
                   "still span", fontsize=10)
    axb.text(2.70, 1.5, "the paper's three settings", ha="right", fontsize=9.5,
             color="#c2410c")
    axb.spines[["top", "right"]].set_visible(False)
    axb.tick_params(labelsize=9.5)

    fig.text(0.5, 0.012, "counted on real MNIST only - no model was run. "
             "the top strip is one query chosen to show the effect.",
             ha="center", fontsize=8.5, color="0.45")
    fig.savefig("cvae_condition.svg", transparent=True)
    plt.close(fig)
    print(f"wrote cvae_condition.svg  (spans {spans[0]:.2f}/{spans[1]:.2f}/"
          f"{spans[2]:.2f})")


def fig_variety_ruler():
    """클래스 안 다양함이라는 자가 무엇을 다양함으로 세는가.

    진짜 MNIST 를 **그대로 두고 자리만 옮긴다.** 새로 만들어진 다양함은
    하나도 없는데 자는 올라간다. 한 칸만 옮겨도 이 장의 가장 높은 값인
    DCGAN 의 1.085 를 넘는다.

    모델은 쓰지 않는다. 모든 수는 여기서 진짜 MNIST 로 센 것이다.
    """
    tf = transforms.ToTensor()
    te = torchvision.datasets.MNIST(DATA, train=False, download=True, transform=tf)
    from torch.utils.data import DataLoader
    X = torch.cat([x for x, _ in DataLoader(te, batch_size=2000)])
    y = torch.cat([l for _, l in DataLoader(te, batch_size=2000)])

    def ruler(Z):
        return sum(Z[y == c].var(0, unbiased=False).mean().item()
                   for c in range(10)) / 10

    base = ruler(X)

    def jitter(m, seed=0):
        if m == 0:
            return X.clone()
        g = torch.Generator().manual_seed(seed)
        dx = torch.randint(-m, m + 1, (len(X),), generator=g)
        dy = torch.randint(-m, m + 1, (len(X),), generator=g)
        Z = X.clone()
        for i in range(len(X)):
            Z[i, 0] = torch.roll(X[i, 0], (int(dy[i]), int(dx[i])), (0, 1))
        return Z

    shifts = [0, 1, 2, 3, 4]
    zs = {m: jitter(m) for m in shifts}
    vals = [ruler(zs[m]) / base for m in shifts]

    fig = plt.figure(figsize=(12.6, 4.8))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.12, 1], height_ratios=[1, 1],
                          left=0.065, right=0.985, top=0.88, bottom=0.13,
                          wspace=0.2, hspace=0.42)
    axc = fig.add_subplot(gs[:, 0])

    axc.plot(shifts, vals, "o-", color="#c2410c", lw=2.1, ms=7)
    for m, v in zip(shifts, vals):
        axc.annotate(f"{v:.3f}", xy=(m, v), xytext=(0, 10),
                     textcoords="offset points", ha="center", fontsize=9.5,
                     color="#c2410c")
    for v, n, c in ((0.953, "GAN", "0.5"), (1.047, "DDPM", "0.5"),
                    (1.085, "DCGAN - the chapter's best", "tab:blue")):
        axc.plot([-0.25, 4.15], [v, v], color=c, lw=1.0, ls="--")
        axc.text(4.28, v, n, va="center", fontsize=9, color=c)
    axc.set_xticks(shifts)
    axc.set_xlabel("every real digit moved by up to this many pixels",
                   fontsize=10)
    axc.set_ylabel("what the variety ruler reports", fontsize=10)
    axc.set_xlim(-0.25, 5.6); axc.set_ylim(0.9, 1.55)
    axc.set_title("nothing new was drawn - the digits were only nudged",
                  fontsize=11)
    axc.spines[["top", "right"]].set_visible(False)

    # --- 오른쪽: 눈으로 보는 같은 일 ----------------------------------------
    pick = [3, 2, 1, 18, 4, 8, 11, 0]
    for row, (m, lab) in enumerate([(0, f"untouched - ruler {vals[0]:.3f}"),
                                    (1, f"moved by 1 pixel - ruler {vals[1]:.3f}")]):
        sub = gs[row, 1].subgridspec(1, len(pick), wspace=0.08)
        for k, idx in enumerate(pick):
            ax = fig.add_subplot(sub[0, k])
            ax.imshow(zs[m][idx, 0], cmap="gray_r", vmin=0, vmax=1)
            ax.set_xticks([]); ax.set_yticks([]); ax.spines[:].set_visible(False)
        fig.text(0.60, 0.845 - row * 0.425, lab, fontsize=9.6,
                 color="0.3" if m == 0 else "#c2410c")

    fig.text(0.5, 0.015, "every picture on the right is a real MNIST digit. "
             "counted here on the 10,000 test digits - no model was run.",
             ha="center", fontsize=8.5, color="0.45")
    fig.savefig("variety_ruler.svg", transparent=True)
    plt.close(fig)
    print(f"wrote variety_ruler.svg  (1 pixel -> {vals[1]:.3f} vs DCGAN 1.085)")


def fig_where_nonlinearity():
    """굽은 곳이 인코더냐 디코더냐에 따라 복원오차가 어디로 가는가.

    네 수는 03_ae_mlp.md 의 규약(씨앗 42, Adam 1e-3, 배치 100, 100 에포크,
    폭 256, 잠재 64)을 그대로 써서 네 벌을 따로 학습해 잰 것이다. 「둘 다
    굽음」이 그 쪽의 0.05462 를 되살리므로(0.05448, MPS 흔들림 안) 나머지
    세 칸도 같은 자로 잰 값이다.

    왜 이렇게 되는지는 그림이 아니라 셈으로 안다. 디코더가 선형이면 복원은
    k 차원 아핀 부분공간 안에만 놓이므로, 인코더를 아무리 굽혀도 그 부분
    공간 중 최선인 PCA 바닥 아래로 내려갈 수 없다.
    """
    pca = 0.09530
    cells = {("linear", "linear"): 0.09639, ("curved", "linear"): 0.09662,
             ("linear", "curved"): 0.05709, ("curved", "curved"): 0.05448}

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.4, 4.6),
                                   gridspec_kw={"width_ratios": [1, 1.1]})

    # --- 왼쪽: 2x2 판 -------------------------------------------------------
    axL.set_xlim(0, 2); axL.set_ylim(0, 2)
    for (enc, dec), v in cells.items():
        i = 0 if enc == "linear" else 1
        j = 0 if dec == "linear" else 1
        hit = v < 0.07
        axL.add_patch(plt.Rectangle((i, j), 1, 1, facecolor=
                                    "#dcfce7" if hit else "#fee2e2",
                                    edgecolor="white", lw=3))
        axL.text(i + 0.5, j + 0.62, f"{v:.5f}", ha="center", fontsize=14,
                 color="#166534" if hit else "#991b1b")
        axL.text(i + 0.5, j + 0.34,
                 "beats PCA" if hit else "stuck at PCA", ha="center",
                 fontsize=9.5, color="#166534" if hit else "#991b1b")
    axL.set_xticks([0.5, 1.5]); axL.set_xticklabels(["straight", "curved"],
                                                    fontsize=10.5)
    axL.set_yticks([0.5, 1.5]); axL.set_yticklabels(["straight", "curved"],
                                                    fontsize=10.5, rotation=90,
                                                    va="center")
    axL.set_xlabel("the half that squeezes  (encoder)", fontsize=10.5)
    axL.set_ylabel("the half that unfolds  (decoder)", fontsize=10.5)
    axL.set_title(f"reconstruction error    (PCA floor {pca:.5f})", fontsize=11)
    for s in axL.spines.values():
        s.set_visible(False)
    axL.tick_params(length=0)

    # --- 오른쪽: 누가 벌었나 ------------------------------------------------
    flat = cells[("linear", "linear")]
    full = cells[("curved", "curved")]
    tot = flat - full
    enc_only = (flat - cells[("curved", "linear")]) / tot * 100
    dec_only = (flat - cells[("linear", "curved")]) / tot * 100
    bars = axR.bar(["curving the\nencoder alone", "curving the\ndecoder alone"],
                   [enc_only, dec_only], color=["#b91c1c", "tab:green"],
                   width=0.55)
    for b, v in zip(bars, [enc_only, dec_only]):
        axR.text(b.get_x() + b.get_width() / 2,
                 v + (3 if v > 0 else -9), f"{v:.0f}%", ha="center",
                 fontsize=13)
    axR.axhline(0, color="0.3", lw=1.1)
    axR.axhline(100, color="0.6", lw=1.0, ls="--")
    axR.text(-0.42, 103, "all of it", ha="left", fontsize=9.5, color="0.45")
    axR.set_ylim(-22, 118)
    axR.set_ylabel("share of the whole gain it accounts for", fontsize=10.5)
    axR.set_title("the page curves both - only one of them pays",
                  fontsize=11)
    axR.spines[["top", "right"]].set_visible(False)

    fig.text(0.5, 0.015, "four models trained here with this page's own "
             "settings; 'curved both' reproduces its 0.05462.",
             ha="center", fontsize=8.5, color="0.45")
    fig.subplots_adjust(left=0.085, right=0.985, top=0.91, bottom=0.165,
                        wspace=0.28)
    fig.savefig("where_nonlinearity.svg", transparent=True)
    plt.close(fig)
    print(f"wrote where_nonlinearity.svg  (encoder {enc_only:.0f}%, "
          f"decoder {dec_only:.0f}%)")


def fig_cifar_provenance():
    """CIFAR-10 이 어디서 잘려 나왔는가, 그리고 그 어미에게 무슨 일이 있었는가.

    CIFAR-10 쪽 수(60,000 장, 열 가지, 한 가지에 6,000 장, 32x32x3)는
    `data/cifar-10-batches-py` 를 직접 세어 적었다. 어미 자료의 수(약 8천만
    장, 워드넷 명사 약 7만 5천 개)는 **그 논문에 적힌 값을 옮긴 것**이며
    여기서 세지 않았다 — 그림 안에 그렇게 밝혀 둔다.
    """
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.6, 4.4),
                                   gridspec_kw={"width_ratios": [1, 1.25]})

    # --- 왼쪽: 크기 ---------------------------------------------------------
    names = ["the parent set\n(reported: ~80,000,000)",
             "CIFAR-10\n(counted here: 60,000)"]
    vals = [80_000_000, 60_000]
    axL.barh(names[::-1], vals[::-1], color=["tab:blue", "0.65"], height=0.5)
    axL.set_xscale("log")
    axL.text(80_000_000, 1.33, " 80 million", va="center", fontsize=10,
             color="0.4")
    axL.text(60_000, 0.33, " 60 thousand", va="center", fontsize=10,
             color="0.4")
    axL.annotate("CIFAR-10 is 0.075% of it", xy=(60_000, 0.0),
                 xytext=(2.2e6, -0.42), fontsize=10, color="tab:blue",
                 arrowprops=dict(arrowstyle="->", color="tab:blue", lw=1.1))
    axL.set_xlim(1e4, 4e8); axL.set_ylim(-0.75, 1.75)
    axL.set_xlabel("number of pictures", fontsize=10)
    axL.set_title("a hand-picked corner of something much larger", fontsize=11)
    axL.spines[["top", "right", "left"]].set_visible(False)
    axL.tick_params(axis="y", length=0, labelsize=9.5)

    # --- 오른쪽: 무슨 일이 있었나 -------------------------------------------
    ev = [(2008, "the parent set is built\nby searching the web for\n"
                 "every noun in a dictionary\n(~75,000 of them), each\n"
                 "picture shrunk to 32x32", "tab:blue", 1),
          (2009, "CIFAR-10 is cut out of it:\n10 ordinary classes,\n"
                 "6,000 pictures each,\nlabelled by hand", "0.35", -1),
          (2020, "the parent set is taken down\nby the people who made it.\n"
                 "nobody had checked what\nthe dictionary's uglier nouns\n"
                 "had pulled in", "#b91c1c", 1)]
    axR.axhline(0, color="0.35", lw=1.2)
    for i, (yr, txt, c, up) in enumerate(ev):
        axR.plot([i], [0], "o", color=c, ms=10, zorder=3)
        axR.plot([i, i], [0, 0.20 * up], color=c, lw=1.1)
        axR.text(i, 0.26 * up, f"{yr}", ha="center",
                 va="bottom" if up > 0 else "top", fontsize=11.5, color=c,
                 weight="bold")
        axR.text(i, 0.56 * up, txt, ha="center",
                 va="bottom" if up > 0 else "top", fontsize=8.8, color=c)
    axR.set_xlim(-0.6, 2.6); axR.set_ylim(-1.65, 1.75)
    axR.set_yticks([]); axR.set_xticks([])
    axR.spines[:].set_visible(False)
    axR.set_title("where these 60,000 pictures came from", fontsize=11)

    fig.text(0.5, 0.015, "CIFAR-10's counts were read off the copy in this "
             "repo; the parent's counts are quoted from its paper, not "
             "measured here.", ha="center", fontsize=8.5, color="0.45")
    fig.text(0.72, 0.075, "CIFAR-10 itself was never withdrawn, and is still "
             "used - this page uses it.", ha="center", fontsize=9,
             color="0.35")
    fig.subplots_adjust(left=0.175, right=0.985, top=0.9, bottom=0.18,
                        wspace=0.1)
    fig.savefig("cifar_provenance.svg", transparent=True)
    plt.close(fig)
    print("wrote cifar_provenance.svg")


def fig_eps_units(T=1000):
    """잡음을 맞히라는 로스가 1,000스텝을 얼마나 고르지 않게 다루는가.

    둘 다 **이 쪽이 쓰는 일정**으로 여기서 셈한 것이고 논문의 수가 아니다.

    왼쪽: 학습이 전혀 없는 예측기 eps = x_t / sqrt(1 - abar_t) 가 내는 MSE.
    목표 eps 의 분산이 1 이므로 1.0 은 「그냥 0이라고 답하기」의 값이다.
    E[x0^2] = 1 로 두었다(정규화한 자료라 대략 그렇다).

    오른쪽: eps 를 조금 틀렸을 때 그것이 원본 그림으로 돌아가며 곱해지는
    배수 sqrt((1 - abar_t) / abar_t). x0 를 x_t 에서 되짚을 때 sqrt(abar_t)
    로 나누기 때문이다.
    """
    b = np.linspace(1e-4, 0.02, T)
    ab = np.cumprod(1 - b)
    t = np.arange(1, T + 1)
    triv = ab / (1 - ab)
    amp = np.sqrt((1 - ab) / ab)
    n_easy = int((triv < 0.1).sum())

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.4, 4.6))

    axL.plot(t, triv, color="#c2410c", lw=2.1)
    axL.axhline(1.0, color="0.45", lw=1.1, ls="--")
    axL.text(990, 1.5, "what you get by just answering zero", ha="right",
             fontsize=9.3, color="0.45")
    axL.axvspan(T - n_easy, T, color="#c2410c", alpha=0.08)
    axL.annotate(f"on {n_easy} of the 1,000 steps a formula\nwith nothing to "
                 "learn already gets\nwithin a tenth of the target",
                 xy=(700, triv[699]), xytext=(330, 2e-4), fontsize=9.3,
                 color="#c2410c",
                 arrowprops=dict(arrowstyle="->", color="#c2410c", lw=1.1))
    axL.set_yscale("log")
    axL.set_xlabel("step  (1 = almost clean,  1000 = pure noise)", fontsize=10)
    axL.set_ylabel("error left for a predictor\nthat learns nothing",
                   fontsize=10)
    axL.set_title("how much is actually left to learn", fontsize=11)
    axL.spines[["top", "right"]].set_visible(False)

    axR.plot(t, amp, color="tab:blue", lw=2.1)
    axR.axhline(1.0, color="0.45", lw=1.1, ls="--")
    for tt in (1, 1000):
        axR.annotate(f"{amp[tt-1]:.2f}x" if tt == 1 else f"{amp[tt-1]:.0f}x",
                     xy=(tt, amp[tt - 1]), xytext=(14 if tt == 1 else -14, 8),
                     textcoords="offset points",
                     ha="left" if tt == 1 else "right", fontsize=10.5,
                     color="tab:blue")
    axR.text(500, 0.035, f"the same slip is punished\n"
             f"{amp[-1]/amp[0]:,.0f} times harder at one end\nthan at the other",
             ha="center", fontsize=9.3, color="tab:blue")
    axR.set_yscale("log")
    axR.set_xlabel("step  (1 = almost clean,  1000 = pure noise)", fontsize=10)
    axR.set_ylabel("how much a small slip in the answer\n"
                   "is magnified in the picture", fontsize=10)
    axR.set_title("what getting it slightly wrong costs", fontsize=11)
    axR.spines[["top", "right"]].set_visible(False)

    fig.text(0.5, 0.015, "the loss on this page gives all 1,000 steps the "
             "same weight. computed here from this page's own schedule.",
             ha="center", fontsize=8.8, color="0.45")
    fig.subplots_adjust(left=0.095, right=0.985, top=0.91, bottom=0.175,
                        wspace=0.3)
    fig.savefig("eps_units.svg", transparent=True)
    plt.close(fig)
    print(f"wrote eps_units.svg  ({n_easy} easy steps, "
          f"{amp[-1]/amp[0]:,.0f}x spread)")


def fig_outside_measured():
    """「바깥에 떨어진다」를 두 가지 자로 재면 서로 다른 말을 한다.

    저장해 둔 `mnist_ae_mlp64.pt` 와 `mnist_ae_mlp2_latent.pt` 로 잰다.
    앞엣것의 복원 MSE 가 0.05462 로 5.1절 값과 같아 같은 모델임을 확인했다.

    왼쪽 곡선은 **64차원 잠재 표현에서 좌표를 d개만 떼어** 잰 것이다.
    d차원 오토인코더를 따로 학습한 것이 아니므로 그렇게 읽지 말 것 —
    「좌표 d개 사이에 서로 묶임이 얼마나 쌓여 있는가」를 보는 것이다.

    부피 배수는 상관행렬 R 의 행렬식으로 det(R)^(-1/2) 이다. 좌표마다 따로
    정규분포를 두고 곱한 것이 참 분포보다 몇 배 넓은 자리를 덮는지를 뜻한다.
    """
    from torch.utils.data import DataLoader, TensorDataset
    import torch.nn.functional as Fn
    import torch.optim as optim

    root = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                        "..", "..", ".."))
    tf = transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize((0.1307,), (0.3081,))])
    te = torchvision.datasets.MNIST(DATA, train=False, download=True,
                                    transform=tf)
    X = torch.cat([x for x, _ in DataLoader(te, batch_size=2000)]).flatten(1)

    def build(k):
        return nn.ModuleDict({
            "enc": nn.Sequential(nn.Linear(784, 256), nn.ReLU(),
                                 nn.Linear(256, k)),
            "dec": nn.Sequential(nn.Linear(k, 256), nn.ReLU(),
                                 nn.Linear(256, 784))})

    def codes(k):
        """저장된 체크포인트가 있으면 쓰고, 없으면 그 쪽의 규약으로 학습한다.

        `*.pt` 는 .gitignore 에 걸려 저장소에 없으므로, 내려받아 쓰는 사람은
        여기서 100 에포크를 돌게 된다(잠재 2와 64를 합쳐 십여 분).
        """
        lat = os.path.join(root, f"mnist_ae_mlp{k}_latent.pt")
        if os.path.exists(lat):                 # 잠재 표현만 저장해 둔 것이 있다
            return torch.load(lat, weights_only=True,
                              map_location="cpu")[0].double()
        p = os.path.join(root, f"mnist_ae_mlp{k}.pt")
        m = build(k)
        if os.path.exists(p):
            m.load_state_dict(torch.load(p, weights_only=True,
                                         map_location="cpu"))
        else:
            tr = torchvision.datasets.MNIST(DATA, train=True, download=True,
                                            transform=tf)
            Xtr = torch.cat([x for x, _ in DataLoader(tr, batch_size=2000)]
                            ).flatten(1)
            torch.manual_seed(42)
            m = build(k)
            o = optim.Adam(m.parameters(), lr=1e-3)
            dl = DataLoader(TensorDataset(Xtr), batch_size=100, shuffle=True)
            for _ in range(100):
                for (xb,) in dl:
                    o.zero_grad()
                    Fn.mse_loss(m["dec"](m["enc"](xb)), xb).backward()
                    o.step()
            torch.save(m.state_dict(), p)
        m.eval()
        with torch.no_grad():
            return m["enc"](X).double()

    z64, z2 = codes(64), codes(2)

    def inflation(z):
        s = z.std(0)
        R = torch.cov(z.T) / (s[:, None] * s[None, :])
        return float(torch.exp(-0.5 * torch.linalg.slogdet(R)[1]))

    def nn_ratio(z):
        z = z.double()
        g = torch.Generator().manual_seed(0)
        s = torch.randn(z.shape, generator=g).double() * z.std(0) + z.mean(0)
        d = torch.cdist(z, z); d.fill_diagonal_(1e9)
        return float(torch.cdist(s, z).min(1).values.median() /
                     d.min(1).values.median())

    ds = [2, 4, 8, 16, 32, 48, 64]
    infl = [inflation(z64[:, :d]) for d in ds]
    i2, i64 = inflation(z2.double()), infl[-1]
    r2, r64 = nn_ratio(z2), nn_ratio(z64)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.4, 4.6),
                                   gridspec_kw={"width_ratios": [1.1, 1]})

    axL.plot(ds, infl, "o-", color="#c2410c", lw=2.1, ms=6.5)
    axL.annotate(f"{i64:,.0f}x", xy=(64, i64), xytext=(-12, 6),
                 textcoords="offset points", ha="right", fontsize=11,
                 color="#c2410c")
    axL.annotate(f"{infl[0]:.2f}x", xy=(2, infl[0]), xytext=(7, 13),
                 textcoords="offset points", ha="left", fontsize=10.5,
                 color="#c2410c")
    axL.axhline(1.0, color="0.5", lw=1.0, ls="--")
    axL.text(62, 1.14, "a perfect fit would sit here", ha="right", fontsize=9.3,
             color="0.45")
    axL.set_yscale("log")
    axL.set_xlabel("how many of the 64 coordinates you look at", fontsize=10)
    axL.set_ylabel("how much roomier the fitted bell is\n"
                   "than the codes really are", fontsize=10)
    axL.set_title("the mismatch is flat, then it explodes", fontsize=11)
    axL.spines[["top", "right"]].set_visible(False)

    x = np.arange(2)
    axR.bar(x - 0.19, [r2, i2], 0.37, color="0.65", label="latent 2")
    axR.bar(x + 0.19, [r64, i64], 0.37, color="#c2410c", label="latent 64")
    for xi, (a, b) in zip(x, [(r2, r64), (i2, i64)]):
        axR.text(xi - 0.19, a * 1.18, f"{a:.2f}x", ha="center", fontsize=10)
        axR.text(xi + 0.19, b * 1.18,
                 f"{b:,.0f}x" if b > 10 else f"{b:.2f}x", ha="center",
                 fontsize=10)
    axR.set_yscale("log"); axR.set_ylim(0.6, 4000)
    axR.set_xticks(x)
    axR.set_xticklabels(["how FAR the draws land\n(nearest real code)",
                         "which DIRECTION they point\n(roominess above)"],
                        fontsize=9.6)
    axR.axhline(1.0, color="0.5", lw=1.0, ls="--")
    axR.text(0, 170, "this ruler says\nthe two are alike", ha="center",
             fontsize=9.4, color="0.4")
    axR.text(1, 2200, "this one does not", ha="center", fontsize=9.4,
             color="#c2410c")
    axR.legend(fontsize=9.3, frameon=False, loc="upper left")
    axR.set_title("two rulers, two answers", fontsize=11)
    axR.spines[["top", "right"]].set_visible(False)

    fig.text(0.5, 0.015, "measured on the saved autoencoders from this "
             "section - nothing was retrained.", ha="center", fontsize=8.8,
             color="0.45")
    fig.subplots_adjust(left=0.085, right=0.985, top=0.91, bottom=0.185,
                        wspace=0.26)
    fig.savefig("outside_measured.svg", transparent=True)
    plt.close(fig)
    print(f"wrote outside_measured.svg  (k2 {i2:.2f}x, k64 {i64:,.0f}x; "
          f"dist {r2:.2f} vs {r64:.2f})")


def fig_nested_code():
    """PCA 의 64개 좌표에는 순서가 있고 선형 AE 의 64개에는 없다.

    수는 02_ae_linear.md 의 규약(씨앗 42, Adam 1e-3, 배치 100, 100 에포크,
    잠재 64)으로 선형 AE 한 벌을 따로 학습해 잰 것이다. 64개를 다 쥐었을 때
    0.09639 가 나왔는데 그 쪽이 적은 0.09582 보다 0.6% 높다 — MPS 의 실행 간
    흔들림으로 보이며, 아래 견줌은 **같은 한 벌 안에서** 이루어지므로 그
    차이에 흔들리지 않는다.

    재는 법: 64개 가운데 d개만 남기고 나머지는 그 좌표의 평균으로 채운 뒤
    되살린다. PCA 는 앞에서부터 d개가 곧 최선이다. AE 는 고를 순서가 없어
    무작위로 20벌 골랐고, 띠가 그 최소~최대다.
    """
    d = np.array([1, 2, 4, 8, 16, 32, 48, 64])
    pca = np.array([0.63980, 0.58645, 0.50464, 0.39442, 0.28296, 0.17728,
                    0.12552, 0.09530])
    ae = np.array([0.70278, 0.69196, 0.66633, 0.63035, 0.53723, 0.38138,
                   0.23437, 0.09639])
    lo = np.array([0.69020, 0.67318, 0.64421, 0.60582, 0.51352, 0.34809,
                   0.21575, 0.09639])
    hi = np.array([0.70582, 0.70237, 0.68671, 0.64789, 0.56923, 0.44083,
                   0.26485, 0.09639])

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.4, 4.6),
                                   gridspec_kw={"width_ratios": [1.15, 1]})

    axL.plot(d, pca, "o-", color="tab:green", lw=2.1, ms=6,
             label="PCA - keep the first d")
    axL.plot(d, ae, "s-", color="#c2410c", lw=2.1, ms=6,
             label="linear AE - keep any d")
    axL.fill_between(d, lo, hi, color="#c2410c", alpha=0.17)
    axL.annotate("they meet only here,\nwith all 64 in hand",
                 xy=(64, 0.0958), xytext=(3.1, 0.255), fontsize=9.6,
                 color="0.3",
                 arrowprops=dict(arrowstyle="->", color="0.45", lw=1.1))
    axL.set_xscale("log", base=2)
    axL.set_xticks(d); axL.set_xticklabels(d)
    axL.set_xlabel("how many of the 64 numbers you keep", fontsize=10)
    axL.set_ylabel("reconstruction error", fontsize=10)
    axL.set_title("one of these codes can be cut short", fontsize=11)
    axL.legend(fontsize=9.5, frameon=False, loc="upper right")
    axL.spines[["top", "right"]].set_visible(False)

    ratio = ae / pca
    axR.plot(d, ratio, "o-", color="#c2410c", lw=2.1, ms=6)
    axR.axhline(1.0, color="0.5", lw=1.0, ls="--")
    k = int(ratio.argmax())
    axR.annotate(f"{ratio[k]:.2f}x", xy=(d[k], ratio[k]), xytext=(0, 10),
                 textcoords="offset points", ha="center", fontsize=11,
                 color="#c2410c")
    axR.text(1.1, 1.03, "same cost as PCA", fontsize=9.3, color="0.45")
    axR.set_xscale("log", base=2)
    axR.set_xticks(d); axR.set_xticklabels(d)
    axR.set_ylim(0.9, 2.45)
    axR.set_xlabel("how many of the 64 numbers you keep", fontsize=10)
    axR.set_ylabel("how many times worse the AE is", fontsize=10)
    axR.set_title("the penalty for having no order", fontsize=11)
    axR.spines[["top", "right"]].set_visible(False)

    fig.text(0.5, 0.015, "both measured here on one linear autoencoder "
             "trained with this page's settings. dropped numbers are replaced "
             "by their average.", ha="center", fontsize=8.5, color="0.45")
    fig.subplots_adjust(left=0.08, right=0.985, top=0.91, bottom=0.175,
                        wspace=0.24)
    fig.savefig("nested_code.svg", transparent=True)
    plt.close(fig)
    print(f"wrote nested_code.svg  (worst {ratio.max():.2f}x at d={d[k]})")


def fig_shuffled_pixels():
    """화소 자리를 한 번 섞으면 누가 무너지는가.

    네 수는 04_ae_cnn.md 의 규약(씨앗 42, Adam 1e-3, 배치 100, 100 에포크,
    잠재 64)으로 네 벌을 따로 학습해 잰 것이다. 섞지 않은 두 벌이 그 쪽의
    0.0243 과 5.1절의 0.0546 을 되살린다(0.02465, 0.05448 - MPS 흔들림 안).

    섞기는 784자리의 고정 순열 하나이고 학습·시험 그림에 똑같이 건다.
    순열은 전단사라 화소값의 모임이 그대로이므로 MSE 를 견줄 수 있다.
    """
    mlp_n, mlp_s = 0.05448, 0.05460
    cnv_n, cnv_s = 0.02465, 0.04705
    edge_n = (mlp_n - cnv_n) / mlp_n * 100
    edge_s = (mlp_s - cnv_s) / mlp_s * 100

    tf = transforms.ToTensor()
    te = torchvision.datasets.MNIST(DATA, train=False, download=True,
                                    transform=tf)
    from torch.utils.data import DataLoader
    X = torch.cat([x for x, _ in DataLoader(te, batch_size=2000)])
    perm = torch.randperm(784, generator=torch.Generator().manual_seed(7))
    P = X.flatten(1)[:, perm].reshape(-1, 1, 28, 28)

    fig = plt.figure(figsize=(12.6, 4.7))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.25],
                          left=0.045, right=0.985, top=0.86, bottom=0.13,
                          wspace=0.2, hspace=0.42)

    pick = [3, 2, 1, 18, 4]
    for row, (src, lab, c) in enumerate([(X, "what the networks normally see",
                                          "0.3"),
                                         (P, "the same digits, pixels shuffled "
                                          "once and for all", "#b91c1c")]):
        sub = gs[row, 0].subgridspec(1, len(pick), wspace=0.08)
        for k, idx in enumerate(pick):
            ax = fig.add_subplot(sub[0, k])
            ax.imshow(src[idx, 0], cmap="gray_r", vmin=0, vmax=1)
            ax.set_xticks([]); ax.set_yticks([]); ax.spines[:].set_visible(False)
        fig.text(0.045, 0.875 - row * 0.425, lab, fontsize=9.6, color=c)

    axR = fig.add_subplot(gs[:, 1])
    x = np.arange(2)
    axR.bar(x - 0.19, [mlp_n, cnv_n], 0.37, color="0.65", label="normal")
    axR.bar(x + 0.19, [mlp_s, cnv_s], 0.37, color="#b91c1c",
            label="pixels shuffled")
    for xi, (a, b) in zip(x, [(mlp_n, mlp_s), (cnv_n, cnv_s)]):
        axR.text(xi - 0.19, a + 0.0016, f"{a:.4f}", ha="center", fontsize=9.4)
        axR.text(xi + 0.19, b + 0.0016, f"{b:.4f}", ha="center", fontsize=9.4)
        axR.text(xi, max(a, b) + 0.0075, f"x{b/a:.2f}", ha="center",
                 fontsize=12.5, color="0.3" if b / a < 1.2 else "#b91c1c")
    axR.set_xticks(x)
    axR.set_xticklabels(["the MLP autoencoder",
                         "the convolutional one"], fontsize=10.5)
    axR.set_ylabel("reconstruction error", fontsize=10)
    axR.set_ylim(0, 0.078)
    axR.legend(fontsize=9.5, frameon=False, loc="upper left")
    axR.set_title("only one of them was relying on where the pixels were",
                  fontsize=11)
    axR.spines[["top", "right"]].set_visible(False)
    axR.text(1.0, 0.0705, f"convolution's edge over the MLP:\n"
             f"{edge_n:.0f}%  ->  {edge_s:.0f}%", ha="center", va="top",
             fontsize=10, color="#b91c1c")

    fig.text(0.5, 0.015, "four autoencoders trained here with this page's own "
             "settings; the unshuffled pair reproduces its 0.0243 and 0.0546.",
             ha="center", fontsize=8.5, color="0.45")
    fig.savefig("shuffled_pixels.svg", transparent=True)
    plt.close(fig)
    print(f"wrote shuffled_pixels.svg  (conv x{cnv_s/cnv_n:.2f}, "
          f"edge {edge_n:.0f}%->{edge_s:.0f}%)")


# 아래 fig_samples 는 main 에서 부르지 않는다 — 학습해 둔 다섯 모델이
# 있어야 돌아가는데 그 .pt 들이 저장소에 없기 때문이다. 쓸 때는 이
# 함수만 따로 import 해서 부른다. 다만 정의는 반드시 main 보다 **위**에
# 둔다 — main 아래의 정의는 스크립트를 직접 돌릴 때 실행 순서상 늦어,
# main 에서 부르려 하면 NameError 가 난다.
# === 그림 8: 네 모델이 만들어 낸 표본 =======================================
@torch.no_grad()
def fig_samples():
    """AE, VAE, GAN, DCGAN, DDPM이 뽑아 낸 그림을 나란히 둔다.

    표로는 전할 수 없는 것이 있다. 흐릿함과 선명함은 보아야 안다.
    """
    import torch.nn.functional as F

    class GMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(64, 256), nn.LeakyReLU(0.2),
                nn.Linear(256, 512), nn.LeakyReLU(0.2),
                nn.Linear(512, 784), nn.Tanh())
        def forward(self, z): return self.net(z).reshape(-1, 1, 28, 28)

    class GConv(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Sequential(nn.Linear(64, 128 * 7 * 7),
                                    nn.BatchNorm1d(128 * 7 * 7), nn.ReLU())
            self.net = nn.Sequential(
                nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(),
                nn.ConvTranspose2d(64, 1, 4, 2, 1), nn.Tanh())
        def forward(self, z): return self.net(self.fc(z).reshape(-1, 128, 7, 7))

    n = 10
    g = torch.Generator().manual_seed(7)
    rows = []

    # 오토인코더 / VAE 는 심판 정규화 공간에서 내놓는다
    ae = load(MLPAE, 64, "ae_mlp64")
    (Xtr, _), (Xte, _) = load_flat()
    zr = ae.enc(Xte)
    z = torch.randn(n, 64, generator=g) * zr.std(0) + zr.mean(0)
    rows.append(("AE_MLP-64", (ae.dec(z) * STD + MEAN).clamp(0, 1)))

    vae = VAEModel(64)
    vae.load_state_dict(torch.load(f"{MODELS}/mnist_vae64_beta1.0.pt",
                                   map_location="cpu", weights_only=True)); vae.eval()
    z = torch.randn(n, 64, generator=g)
    rows.append(("VAE-64", (vae.dec(z) * STD + MEAN).clamp(0, 1)))

    # GAN 들은 [-1, 1] 로 내놓는다
    for tag, cls, label in [("gan_mlp", GMLP, "GAN (MLP)"), ("dcgan", GConv, "DCGAN")]:
        G = cls()
        G.load_state_dict(torch.load(f"{MODELS}/mnist_{tag}_G.pt",
                                     map_location="cpu", weights_only=True))
        G.eval()
        z = torch.randn(n, 64, generator=g)
        rows.append((label, ((G(z).flatten(1) + 1) / 2).clamp(0, 1)))

    # DDPM 은 뽑는 데 그물을 1,000번 지나므로 여기서 다시 뽑지 않는다.
    # 5.4절이 재어 둔 5,000장 가운데 앞의 것을 그대로 읽어 쓴다
    ddpm = torch.load(f"{MODELS}/ddpm_samples.pt", map_location="cpu", weights_only=True)
    rows.append(("DDPM", ((ddpm.flatten(1) + 1) / 2).clamp(0, 1)))

    fig, axes = plt.subplots(len(rows), n, figsize=(11, 6.0))
    for r, (label, imgs) in enumerate(rows):
        for c in range(n):
            axes[r, c].imshow(imgs[c].reshape(28, 28), cmap="gray", vmin=0, vmax=1)
            axes[r, c].axis("off")
        axes[r, 0].text(-0.18, 0.5, label, transform=axes[r, 0].transAxes,
                        ha="right", va="center", fontsize=9)
    fig.subplots_adjust(left=0.13, right=0.995, top=0.995, bottom=0.005,
                        wspace=0.06, hspace=0.10)
    fig.savefig("samples.svg", transparent=True)
    plt.close(fig)
    print("wrote samples.svg")


def fig_who_wins():
    """자 넷을 들이대면 금메달이 갈린다.

    수는 모두 5장 개요의 표에서 옮긴 것이고 여기서 다시 재지 않는다.
    차례는 그 수로 **코드가 매긴다** — 손으로 적지 않으므로 표를 고치면
    시상대도 따라 바뀐다.

    값싸기 칸만 꼴이 다르다. 넷이 순전파 1회로 **똑같이 1등**이라 한 칸에
    묶어 적고, 두 자에서 1등을 한 DDPM 이 여기서 꼴찌라는 것을 함께 보인다.
    억지로 하나를 세우면(이름 차례 따위로) 없는 우열을 지어내게 된다.
    """
    models = ["AE", "VAE", "GAN", "DCGAN", "DDPM", "DDIM"]
    conf = [0.737, 0.758, 0.908, 0.909, 0.940, 0.904]
    bal = [0.580, 0.887, 0.945, 0.970, 0.986, 0.971]
    var = [0.640, 0.628, 0.953, 1.085, 1.047, 0.979]       # 진짜가 1.00
    fwd = [1, 1, 1, 1, 1000, 10]

    quality = [("does it still look\nlike a digit?", [-c for c in conf]),
               ("are all ten made\nequally often?", [-b for b in bal]),
               ("as varied as real\nhandwriting?", [abs(v - 1.0) for v in var])]

    fig, axes = plt.subplots(1, 4, figsize=(13.2, 4.4),
                             gridspec_kw={"width_ratios": [1, 1, 1, 1.25]})
    GOLD, GREY = "#c89b1e", "0.74"
    winners = []

    for ax, (name, key) in zip(axes, quality):
        order = sorted(range(6), key=lambda i: (key[i], models[i]))[:3]
        winners.append(models[order[0]])
        for slot, (rank, h) in enumerate(zip([1, 0, 2], [0.66, 1.0, 0.44])):
            i = order[rank]
            gold = rank == 0
            ax.add_patch(plt.Rectangle((slot - 0.45, 0), 0.90, h,
                                       facecolor=GOLD if gold else GREY,
                                       edgecolor="white", lw=2))
            ax.text(slot, h / 2, models[i], ha="center", va="center",
                    fontsize=11.5 if gold else 10,
                    color="white" if gold else "0.2",
                    weight="bold" if gold else "normal")
            ax.text(slot, h + 0.05, str(rank + 1), ha="center", fontsize=9.5,
                    color="0.45")
        ax.set_title(name, fontsize=10.5, color="0.2")
        ax.set_xlim(-0.8, 2.8); ax.set_ylim(0, 1.45)
        ax.set_xticks([]); ax.set_yticks([]); ax.spines[:].set_visible(False)

    # --- 값싸기: 시상대가 아니라 값 그대로 ---------------------------------
    # 넷이 순전파 1회로 똑같다. 억지로 하나를 세우면(이름 차례 따위로) 없는
    # 우열을 지어내게 되므로, 여기만 꼴을 바꾸어 값을 그대로 늘어놓는다.
    ax = axes[3]
    o = sorted(range(6), key=lambda i: (fwd[i], models[i]))
    y = np.arange(6)[::-1]
    cols = ["#b91c1c" if fwd[i] == max(fwd) else GOLD if fwd[i] == 1 else GREY
            for i in o]
    ax.barh(y, [fwd[i] for i in o], color=cols, height=0.62)
    for yy, i in zip(y, o):
        ax.text(fwd[i] * 1.35, yy, f"{fwd[i]:,}", va="center", fontsize=9.5,
                color="#b91c1c" if fwd[i] == max(fwd) else "0.35")
    ax.set_yticks(y); ax.set_yticklabels([models[i] for i in o], fontsize=9.5)
    ax.set_xscale("log"); ax.set_xlim(0.7, 6000)
    ax.set_xticks([1, 10, 100, 1000]); ax.set_xticklabels(["1", "10", "100", "1,000"],
                                                          fontsize=8.5)
    ax.set_xlabel("passes through the network", fontsize=9)
    ax.text(3.4, 3.45, "four tie\nat one", fontsize=9, color="#9a7714")
    ax.set_title("how cheap is\none picture?", fontsize=10.5, color="0.2")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)

    uniq = len(set(winners))
    fig.suptitle("the same six machines, judged four ways — and the gold "
                 "medal keeps moving", fontsize=12.5, y=0.985)
    fig.text(0.5, 0.025, "ranked by the code straight from chapter 5's own "
             "table; nothing here was re-measured.", ha="center",
             fontsize=8.8, color="0.45")
    fig.subplots_adjust(left=0.02, right=0.975, top=0.76, bottom=0.145,
                        wspace=0.18)
    fig.savefig("who_wins.svg", transparent=True)
    plt.close(fig)
    print(f"wrote who_wins.svg  (quality winners: {winners} -> {uniq} distinct)")


if __name__ == "__main__":
    fig_pca_reconstructions()
    fig_recon_ladder()
    fig_latent_2d()
    fig_ae_vs_vae_2d()
    fig_class_share()
    fig_ae_linear_descent()
    fig_cvae_label_sweep()
    fig_two_rulers()
    fig_forward_diffusion()
    fig_kl_tradeoff()
    fig_quality_vs_cost()
    fig_gan_samples()
    fig_gan_vs_dcgan()
    fig_ddim_steps()
    fig_rulers_disagree()
    fig_step2_two_ladders()
    fig_pearson_two_lines()
    fig_diffusion_attention()
    fig_gan_saturation()
    fig_dcgan_bundle()
    fig_ddim_invertible()
    fig_cvae_condition()
    fig_variety_ruler()
    fig_where_nonlinearity()
    fig_cifar_provenance()
    fig_eps_units()
    fig_outside_measured()
    fig_nested_code()
    fig_shuffled_pixels()
    fig_who_wins()
