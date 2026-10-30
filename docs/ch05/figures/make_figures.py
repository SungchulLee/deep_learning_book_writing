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
