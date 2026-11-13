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
    dcgan_walk.svg            잠재 공간의 직선 위를 16칸 걸으며 뽑은 그림과,
                              칸 사이의 화소 거리
    dcgan_nearest.svg         뽑은 것 가운데 학습 그림에 가장 가까운 여덟 짝
    dcgan_thickness.svg       굵기 벡터를 더해 가며 — 굵어지는데 숫자가 바뀐다
    dcgan_init_arms.svg       N(0, 0.02) 초기화를 넣고 뺀 세 갈래, 시드 다섯 벌
                              (수는 experiments/ch05_dcgan_init.json 에서 읽는다)
    dcgan_shapes.svg          널리 쓰이는 64x64 DCGAN 과 이 쪽의 28x28 을 나란히
    gan_code_lines.svg        ReLU 와 LeakyReLU, tanh 와 시그모이드, Adam 의 beta1
    dcgan_bn_modes.svg        같은 z 를 배치 정규화의 평가 모드와 학습 모드로 그린 것
    cvae_ten_planes.svg       VAE-2 와 cVAE-2 의 잠재 평면을 숫자별로 칠한 것
                              (mnist_cvae2.pt 는 03_cvae.md 의 블록이 저장한다)

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
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe
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

    **이 독스트링은 한때 「빈 곳을 복호하면 숫자가 아닌 것이 나온다」고
    적고 있었는데, 이 그림 자체가 그것을 반박한다.** 잠재 차원이 2 일 때는
    빈 칸도 멀쩡한 숫자를 내놓는다. 재어 보면 점이 하나도 없는 칸의 확신도가
    0.889 로 점이 있는 칸의 0.883 과 같고, 최근접 거리와 확신도의 상관이
    +0.005 다. 까닭은 06_sampling 이 재어 둔 대로 **2차원에는 빗나갈 자리가
    거의 없기** 때문이다(덮는 자리가 참 분포의 1.01배. 64차원에서는 576배).

    빈 곳이 말썽이 되는 것은 잠재 차원이 커진 뒤이고, VAE 가 고치려는 것도
    그쪽이다. 격자는 2~98 분위 상자만 훑는다는 점도 함께 적어 둔다.
    """
    (Xtr, _), (Xte, yte) = load_flat()
    mu, evecs, _ = pca_basis(Xtr)

    pca_z = (Xte - mu) @ evecs[:, :2]
    ae = load(MLPAE, 2, "ae_mlp2")
    ae_z = ae.enc(Xte)

    # 격자를 먼저 세운다. 가운데 칸에 겹쳐 그려야 어느 점이 어느 그림인지
    # 독자가 짚을 수 있다 (오른쪽 칸의 왼쪽 위 = 격자의 왼쪽 위)
    lo = ae_z.quantile(0.02, dim=0); hi = ae_z.quantile(0.98, dim=0)
    n = 16
    gx = torch.linspace(lo[0], hi[0], n); gy = torch.linspace(lo[1], hi[1], n)
    grid = torch.stack([torch.stack([x, y]) for y in gy.flip(0) for x in gx])
    step = ((gx[1] - gx[0]) + (gy[1] - gy[0])) / 2
    # 반지름 반 칸 안에 시험 점이 하나도 없는 칸 — 3절이 재는 그 17칸이다
    empty = (torch.cdist(grid, ae_z) < step / 2).sum(dim=1) == 0

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4))
    for ax, z, title in [(axes[0], pca_z, "PCA-2 latent"),
                         (axes[1], ae_z, "AE_MLP-2 latent")]:
        # rasterized=True — 점 1만 개를 벡터로 두면 SVG가 3MB를 넘는다.
        # 흩어진 점은 래스터로 묻어도 읽는 데 지장이 없다
        sc = ax.scatter(z[:, 0], z[:, 1], c=yte, cmap="tab10", s=1.2,
                        alpha=0.55, linewidths=0, rasterized=True)
        ax.set_title(title, fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])

    g = axes[1]
    g.scatter(grid[:, 0], grid[:, 1], marker="+", c="black", s=16,
              linewidths=0.55, alpha=0.9, zorder=3)
    g.scatter(grid[empty, 0], grid[empty, 1], marker="o", s=52,
              facecolors="none", edgecolors="#b91c1c", linewidths=1.2,
              zorder=4)
    g.add_patch(plt.Rectangle((lo[0], lo[1]), hi[0] - lo[0], hi[1] - lo[1],
                              fill=False, ec="0.35", lw=0.9, ls=":", zorder=2))
    g.set_xlabel(f"+  the 16x16 grid decoded on the right\n"
                 f"o  no test point within half a cell "
                 f"({int(empty.sum())} of {n * n})", fontsize=9)

    cb = fig.colorbar(sc, ax=axes[1], fraction=0.046, ticks=range(10))
    cb.ax.tick_params(labelsize=8)

    dec = ae.dec(grid).reshape(n, n, 28, 28)
    canvas = torch.cat([torch.cat([dec[i, j] for j in range(n)], dim=1)
                        for i in range(n)], dim=0)
    axes[2].imshow((canvas * STD + MEAN).clamp(0, 1), cmap="gray", vmin=0, vmax=1)
    axes[2].set_title("AE_MLP-2 decoded over a grid", fontsize=11)
    axes[2].axis("off")

    fig.subplots_adjust(left=0.01, right=0.99, top=0.93, bottom=0.11, wspace=0.12)
    fig.savefig("latent_2d.svg", transparent=True)
    plt.close(fig)
    print("wrote latent_2d.svg")


@torch.no_grad()
def fig_no_noise():
    """상자를 열 배로 넓혀도 디코더는 잡음을 내지 못한다.

    디코더가 닿을 수 있는 그림 전체가 잠재 차원과 같은 2차원 곡면이므로,
    무엇을 넣든 그 곡면 위에 떨어진다. 그래서 바깥으로 나가도 「잡음」이
    아니라 **더 과장된 숫자꼴**이 나온다 - 진짜 숫자에서는 멀어지는데
    심판은 오히려 더 자신 있어 한다. 자를 의심해야 하는 자리다.

    수는 docs/ch05/dimreduction/03_ae_mlp.md 3절이 싣는 것과 같다.
    """
    (_, _), (Xte, _) = load_flat()
    ae = load(MLPAE, 2, "ae_mlp2")
    judge = JudgeCNN()
    judge.load_state_dict(torch.load(f"{MODELS}/mnist_judge.pt",
                                     map_location="cpu", weights_only=True))
    judge.eval()

    z = ae.enc(Xte)
    lo, hi = z.quantile(0.02, dim=0), z.quantile(0.98, dim=0)
    mid, half = (lo + hi) / 2, (hi - lo) / 2
    n = 16

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 5.0))
    for ax, sc in zip(axes, (1.0, 10.0)):
        L, H = mid - half * sc, mid + half * sc
        gx = torch.linspace(L[0], H[0], n); gy = torch.linspace(L[1], H[1], n)
        pts = torch.stack([torch.stack([x, y]) for y in gy.flip(0) for x in gx])
        img = ae.dec(pts)
        canvas = img.reshape(n, n, 28, 28)
        canvas = torch.cat([torch.cat([canvas[i, j] for j in range(n)], dim=1)
                            for i in range(n)], dim=0)
        ax.imshow((canvas * STD + MEAN).clamp(0, 1), cmap="gray",
                  vmin=0, vmax=1)
        conf = judge(img.reshape(-1, 1, 28, 28)).softmax(dim=1).max(1).values
        px = torch.cdist(img, Xte).min(dim=1).values
        ax.set_title(f"the same box, {sc:g}x wider" if sc > 1
                     else "the box the page shows", fontsize=11.5)
        ax.set_xlabel(f"nearest real digit: {px.mean():.2f}      "
                      f"judge: {conf.mean():.3f}", fontsize=10.5)
        ax.set_xticks([]); ax.set_yticks([])
    fig.text(0.5, 0.045,
             "real digits sit 14.93 from their own nearest neighbour.  "
             "ten times out it is still a digit shape - never noise.",
             ha="center", fontsize=10, color="0.35")
    fig.subplots_adjust(left=0.02, right=0.98, top=0.93, bottom=0.14,
                        wspace=0.08)
    fig.savefig("no_noise.svg", transparent=True)
    plt.close(fig); print("wrote no_noise.svg")


def fig_conv_flip():
    """컨볼루션 걸음이 잠재 64 에서는 벌고 2 에서는 잃는다.

    5.1 의 lead_flip 과 같은 꼴 - 화살표가 가리키는 방향이 곧 결론이다.
    가로는 복원 오차(작을수록 좋다), 세로는 고름(클수록 좋다)이므로
    **왼쪽 위로 가면 양쪽 다 좋아진 것**이다. 두 칸의 가로 눈금이 열 배
    다르므로 칸을 나누었다. 수는 04_conv_vae 쪽의 출력 그대로다.
    """
    import matplotlib.pyplot as plt
    panels = [("latent 64", (0.07499, 0.877), (0.04228, 0.927),
               (0.030, 0.082), (0.855, 0.945)),
              ("latent 2", (0.42744, 0.950), (0.46375, 0.919),
               (0.415, 0.478), (0.905, 0.962))]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.3))
    for ax, (title, a, b, xl, yl) in zip(axes, panels):
        good = b[0] < a[0] and b[1] > a[1]
        col = "#15803d" if good else "#b91c1c"
        # VAE 를 기준으로 「둘 다 좋아진 칸」과 「둘 다 나빠진 칸」을 칠한다.
        # 화살표만으로는 두 칸이 같은 대각선으로 보여 뒤집힘이 드러나지 않는다
        ax.add_patch(plt.Rectangle((xl[0], a[1]), a[0] - xl[0], yl[1] - a[1],
                                   color="#15803d", alpha=0.085, zorder=0))
        ax.add_patch(plt.Rectangle((a[0], yl[0]), xl[1] - a[0], a[1] - yl[0],
                                   color="#b91c1c", alpha=0.085, zorder=0))
        # 이름표는 축의 두 먼 귀퉁이에 둔다 - 칠한 칸이 늘 왼위/오른아래이므로
        ax.text(0.015, 0.975, "better on both", transform=ax.transAxes,
                ha="left", va="top", fontsize=9.5, color="#15803d")
        ax.text(0.985, 0.025, "worse on both", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=9.5, color="#b91c1c")
        ax.annotate("", xy=b, xytext=a,
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=2.6,
                                    mutation_scale=22))
        ax.plot(*a, "o", ms=11, color="0.55")
        ax.plot(*b, "o", ms=11, color=col)
        ax.annotate("VAE", xy=a, xytext=(0, -20), textcoords="offset points",
                    ha="center", fontsize=10.5, color="0.4")
        ax.annotate("conv VAE", xy=b, xytext=(0, -22 if good else 14),
                    textcoords="offset points", ha="center", fontsize=10.5,
                    color=col)
        ax.set_xlim(*xl); ax.set_ylim(*yl)
        ax.set_title(f"{title}   -   {'both rulers improve' if good else 'both rulers worsen'}",
                     fontsize=11, color=col)
        ax.set_xlabel("reconstruction error   (lower is better)", fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("class balance of samples\n(higher is better)", fontsize=10)
    fig.text(0.5, 0.025, "up and to the left = better on both.  "
                         "the arrow flips when the bottleneck narrows.",
             ha="center", fontsize=10, color="0.35")
    fig.subplots_adjust(left=0.10, right=0.98, top=0.90, bottom=0.21, wspace=0.22)
    fig.savefig("conv_flip.svg", transparent=True)
    plt.close(fig); print("wrote conv_flip.svg")


def fig_conv_obedience():
    """컨볼루션이 되살리기는 크게 고치고 말 듣는 일은 못 고친다.

    네 자를 한 그림에 포개지 않는다 - 1 의 뜻이 저마다 다르고, 작은 변화
    둘은 이 절이 재어 둔 샘플링 스프레드(확신도 0.010, 고름 0.013) 안이라
    부를 수 없다. 그래서 **부를 수 있는 둘만** 칸을 나누어 보인다.
    수는 05_conv_cvae 쪽의 출력 그대로다.
    """
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.3))

    axL, axR = axes
    v = [0.07618, 0.04215]
    axL.bar([0, 1], v, width=0.44, color=["0.6", "#15803d"])
    for x, y in zip([0, 1], v):
        axL.text(x, y + 0.0022, f"{y:.5f}", ha="center", fontsize=11.5,
                 color="0.25")
    axL.set_xticks([0, 1]); axL.set_xticklabels(["cVAE", "conv cVAE"],
                                                fontsize=11)
    axL.set_ylim(0, 0.094)
    axL.set_ylabel("reconstruction error\n(lower is better)", fontsize=10.5)
    axL.set_title("the convolution step works", fontsize=11.5, color="#15803d")
    axL.text(0.5, 0.086, "45% less error", ha="center", fontsize=11,
             color="#15803d")

    o = [73.64, 72.38]
    axR.bar([0, 1], o, width=0.44, color=["0.6", "#b91c1c"])
    for x, y in zip([0, 1], o):
        axR.text(x, y + 1.3, f"{y:.2f}%", ha="center", fontsize=11.5,
                 color="0.25")
    axR.axhline(99.66, color="0.5", ls=":", lw=1.3)
    axR.text(0.97, 0.81, "the same model with a 2-number\nbottleneck reaches 99.66%",
             transform=axR.transAxes, ha="right", va="top", fontsize=9.5,
             color="0.45")
    axR.set_xticks([0, 1]); axR.set_xticklabels(["cVAE", "conv cVAE"],
                                                fontsize=11)
    axR.set_ylim(0, 112)
    axR.set_ylabel("asked for a digit, got that digit  (%)", fontsize=10.5)
    axR.set_title("but not on the thing we came for", fontsize=11.5,
                  color="#b91c1c")

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.text(0.5, 0.028,
             "the other two rulers moved too little to call: confidence "
             "+0.021 and balance +0.012, against sampling spreads of "
             "0.010 and 0.013.",
             ha="center", fontsize=9.5, color="0.4")
    fig.subplots_adjust(left=0.10, right=0.97, top=0.89, bottom=0.19,
                        wspace=0.30)
    fig.savefig("conv_obedience.svg", transparent=True)
    plt.close(fig); print("wrote conv_obedience.svg")


class VAEModel(nn.Module):
    def __init__(self, k, n_cond=0):
        super().__init__()
        self.k = k
        self.enc = nn.Sequential(nn.Linear(784 + n_cond, 256), nn.ReLU())
        self.mu = nn.Linear(256, k); self.logvar = nn.Linear(256, k)
        self.dec = nn.Sequential(nn.Linear(k + n_cond, 256), nn.ReLU(), nn.Linear(256, 784))


# === 컨볼루션 두 쪽이 본문에서 쓰는 그림 ===================================
def fig_two_steps_mirror():
    """05_conv_cvae 5절 - 두 걸음이 서로 거울상이다.

    컨볼루션으로 바꾸는 걸음과 조건을 다는 걸음을 잠재 64 와 2 에서
    모두 재면 2 x 2 가 둘 생긴다. 여덟 칸이 이미 다섯 쪽에 흩어져
    있어 새로 잰 것은 없다 - 01, 03, 04, 05 쪽의 출력에서 옮겼다.

    선 둘이 서로 겹치면 그 걸음은 아무것도 바꾸지 않은 것이고,
    벌어지면 바꾼 것이다. 왼쪽에서는 가로로 떨어지고 세로로 겹치며,
    오른쪽에서는 그 반대다.
    """
    #                        fully conn,  convolutional
    data = {
        64: {"plain": (0.07499, 0.04228), "cond": (0.07618, 0.04215)},
        2:  {"plain": (0.42744, 0.46375), "cond": (0.34787, 0.35275)},
    }
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.5))
    for ax, k in zip(axes, (64, 2)):
        for key, c, lab, mk in (("plain", "#1d4ed8", "no condition", "o"),
                                ("cond", "#b91c1c", "told the label", "s")):
            y = data[k][key]
            ax.plot([0, 1], y, mk + "-", color=c, lw=2.2, ms=10, label=lab)
            for x, v in zip((0, 1), y):
                dy = 13 if key == "cond" else -22   # 겹치지 않게 위아래로
                ax.annotate(f"{v:.5f}", (x, v), (0, dy),
                            textcoords="offset points", ha="center",
                            fontsize=10, color=c)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["fully connected", "convolutional"], fontsize=11)
        ax.set_xlim(-0.35, 1.35)
        ax.set_ylabel("reconstruction MSE   (lower is better)", fontsize=10.5)
        ax.set_title(f"latent {k}", fontsize=12.5)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.18)
    axes[0].set_ylim(0.030, 0.094)
    axes[1].set_ylim(0.315, 0.510)
    axes[0].text(0.5, 0.0905, "the convolution step does the work;\n"
                 "the two lines lie on top of each other",
                 ha="center", va="top", fontsize=10, color="0.25")
    axes[1].text(0.5, 0.504, "the condition does the work;\n"
                 "the convolution step slightly hurts",
                 ha="center", va="top", fontsize=10, color="0.25")
    axes[0].legend(fontsize=10.5, frameon=False, loc="lower left")
    fig.subplots_adjust(left=0.08, right=0.98, top=0.86, bottom=0.14,
                        wspace=0.26)
    fig.savefig("two_steps_mirror.svg", transparent=True)
    plt.close(fig); print("wrote two_steps_mirror.svg")


def fig_spread_vs_gap():
    """index 4절 - 주장한 틈이 흔들림보다 큰가.

    왼쪽은 5.3절의 「잘린 규약」에서 잰 두 범위다. 이 절의 표와 규약이
    다르므로 섞지 않고 그 규약 안에서만 그린다. 오른쪽은 같은 자를 두
    가지 흔들림으로 잰 것 - 다시 뽑기와 다시 학습하기다.

    수는 index 4절의 두 표에서 그대로 옮겼다.
    """
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.2, 4.1),
                                   gridspec_kw={"width_ratios": [1.15, 1]})

    # --- 왼쪽: 두 범위가 겹치는가 ---
    bands = [("AE_MLP-64", 0.567, 0.637, "#b91c1c"),
             ("VAE-64", 0.884, 0.899, "#15803d")]
    for i, (name, lo, hi, c) in enumerate(bands):
        axL.barh(i, hi - lo, 0.34, left=lo, color=c, alpha=0.85)
        axL.text(hi + 0.012, i, f"{lo:.3f} - {hi:.3f}", va="center",
                 fontsize=10.5, color=c)
    axL.annotate("", (0.637, 0.5), (0.884, 0.5),
                 arrowprops=dict(arrowstyle="<->", color="0.3", lw=1.6))
    axL.text(0.760, 0.60, "0.247 apart,\nand they do not overlap",
             ha="center", fontsize=10.5, color="0.2")
    axL.set_yticks(range(2)); axL.set_yticklabels([b[0] for b in bands],
                                                  fontsize=11.5)
    axL.set_ylim(-0.6, 1.6); axL.set_xlim(0.52, 1.00)
    axL.set_xlabel("balance of the samples, over five training seeds",
                   fontsize=10.5)
    axL.set_title("the claim this section rests on", fontsize=11.5)
    axL.spines[["top", "right", "left"]].set_visible(False)
    axL.tick_params(left=False)
    axL.text(0.52, -0.52, "measured under 5.3's clipped protocol",
             fontsize=9, color="0.45")

    # --- 오른쪽: 다시 뽑기 대 다시 학습하기 ---
    rows = [("VAE-64\nconfidence", 0.010, 0.010),
            ("VAE-64\nbalance", 0.013, 0.015),
            ("AE_MLP-64\nconfidence", 0.005, 0.029),
            ("AE_MLP-64\nbalance", 0.022, 0.070)]
    W = 0.34
    for i, (name, samp, trn) in enumerate(rows):
        axR.bar(i - W / 2, samp, W * 0.92, color="#1d4ed8",
                label="resample z" if i == 0 else None)
        axR.bar(i + W / 2, trn, W * 0.92, color="#ea580c",
                label="retrain the model" if i == 0 else None)
        axR.text(i + W / 2, trn + 0.0022, f"{trn / samp:.0f}x",
                 ha="center", fontsize=10.5, color="#ea580c")
    axR.axvline(1.5, color="0.75", ls="--", lw=1.1)
    axR.set_xticks(range(4)); axR.set_xticklabels([r[0] for r in rows],
                                                  fontsize=9.5)
    axR.set_ylim(0, 0.085)
    axR.set_ylabel("spread of the ruler", fontsize=10.5)
    axR.set_title("the KL term also buys repeatability", fontsize=11.5)
    axR.legend(fontsize=10, frameon=False, loc="upper left")
    axR.spines[["top", "right"]].set_visible(False)

    fig.subplots_adjust(left=0.11, right=0.985, top=0.89, bottom=0.21,
                        wspace=0.30)
    fig.savefig("spread_vs_gap.svg", transparent=True)
    plt.close(fig); print("wrote spread_vs_gap.svg")


def fig_beta_exchange_rate():
    """02_beta_vae 2절 - 같은 것을 사는 값이 걸음마다 달라진다.

    쪽의 표에 있는 두 열을 나눈 것이다 (오른 고름 / 내준 복원 %).
    나눈 수이므로 쪽에서도 그렇게 밝힌다. 값이 47배 떨어지므로
    세로를 로그로 둔다.
    """
    steps = [("b=0 -> 0.5", 0.243, 5.7), ("b=0.5 -> 1.0", 0.093, 30.0),
             ("b=1.0 -> 4.0", 0.073, 80.0)]
    rate = [g / c for _, g, c in steps]

    fig, ax = plt.subplots(figsize=(7.8, 4.4))
    cols, labs = ["#15803d", "#ca8a04", "#b91c1c"], []
    for i, ((name, g, c), r) in enumerate(zip(steps, rate)):
        ax.bar(i, r, 0.52, color=cols[i])
        ax.text(i, r * 1.15, f"{r:.4f}", ha="center", fontsize=12)
        labs.append(f"{name}\n+{g:.3f} balance\nfor +{c:g}% recon")
    ax.set_yscale("log")
    ax.set_ylim(5e-4, 0.12)
    ax.set_xticks(range(3)); ax.set_xticklabels(labs, fontsize=10.5)
    ax.set_ylabel("balance bought per 1% of recon given up\n(log scale)",
                  fontsize=10.5)
    ax.set_title("the first step buys 47 times more than the last",
                 fontsize=11.5)
    ax.annotate("", (2.42, rate[0]), (2.42, rate[2]),
                arrowprops=dict(arrowstyle="<->", color="0.35", lw=1.6))
    ax.text(2.48, (rate[0] * rate[2]) ** 0.5,
            f"{rate[0] / rate[2]:.0f}x", fontsize=12, va="center",
            color="0.2")
    ax.set_xlim(-0.6, 2.95)
    ax.grid(axis="y", alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig("beta_exchange_rate.svg", transparent=True)
    plt.close(fig); print("wrote beta_exchange_rate.svg")


def fig_conv_off_tradeoff():
    """04_conv_vae 7절 - 맞바꿈 선이 하나가 아니라 둘이다.

    앞서는 컨볼루션 점이 하나뿐이라 "선 바깥의 점" 으로 보였다. 6절이
    같은 기계에 beta 손잡이를 걸어 둘째 점을 얻으니, 그 둘이 완전 연결
    셋과 나란한 **또 하나의 선**을 그린다. 구조는 선을 옮기고 beta 는
    선 위에서 움직인다.

    수는 모두 이 절 다섯 쪽의 출력에서 그대로 옮긴 것이다 (잠재 64).
    가로는 복원 오차라 왼쪽이 좋고, 세로는 뽑은 그림의 균형도라 위가 좋다.
    """
    GR, RD = "#6b7280", "#b91c1c"
    # 라벨 자리를 점마다 따로 준다 - 가운데 아래로 통일하면 VAE 라벨이
    # 가파르게 오르는 선에 그대로 겹친다.
    mlp = [("autoencoder", 0.0546, 0.541, (0, -34)),
           ("VAE", 0.0750, 0.877, (-30, -40)),
           ("beta-VAE (b=4)", 0.1351, 0.950, (0, -34))]
    conv = [("conv VAE", 0.0423, 0.927, (0, -34)),
            ("conv beta-VAE (b=4)", 0.0905, 0.965, (16, -36))]

    fig, ax = plt.subplots(figsize=(7.8, 5.0))
    ax.axhspan(0.90, 1.02, 0.0, 0.30, color="#15803d", alpha=0.07)
    ax.text(0.0305, 1.012, "good at both", fontsize=10.5, color="#15803d",
            ha="left", va="top")

    for pts, c, lw in [(mlp, GR, 1.6), (conv, RD, 2.2)]:
        s = sorted(pts, key=lambda q: q[1])
        ax.plot([q[1] for q in s], [q[2] for q in s], "-", color=c, lw=lw,
                zorder=1)
    ax.text(0.112, 0.830, "fully connected:\nbalance is paid for with recon",
            fontsize=10, color=GR, ha="center", va="top")
    ax.text(0.078, 1.013, "convolution: the same trade,\non a better line",
            fontsize=10, color=RD, ha="center", va="top")

    for pts, c, big in [(mlp, GR, False), (conv, RD, True)]:
        for name, x, y, off in pts:
            ax.plot(x, y, "o", ms=13 if big else 9, color=c, zorder=3)
            ax.annotate(f"{name}\n{x:.4f} / {y:.3f}", (x, y), off,
                        textcoords="offset points", ha="center",
                        fontsize=10.5 if big else 9.5, color=c,
                        fontweight="bold" if big else "normal")

    ax.set_xlabel("reconstruction MSE   (left is better)", fontsize=11)
    ax.set_ylabel("balance of the samples   (up is better)", fontsize=11)
    ax.set_xlim(0.028, 0.160); ax.set_ylim(0.47, 1.04)
    ax.set_title("two trade-off lines - structure moves the line, "
                 "beta moves along it", fontsize=11.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.18)
    fig.tight_layout()
    fig.savefig("conv_off_tradeoff.svg", transparent=True)
    plt.close(fig); print("wrote conv_off_tradeoff.svg")


def fig_obedience_split():
    """05_conv_cvae 3절 - 가르는 것은 구조가 아니라 잠재 차원이다.

    수는 그 쪽의 네 줄짜리 표에서 그대로 옮긴 것이다. 막대 넷을 둘씩
    묶어 두면, 묶음 안의 틈(1%p 남짓)과 묶음 사이의 틈(26%p)이 한눈에
    견주어진다.
    """
    rows = [("latent 2",  99.66, 98.80), ("latent 64", 73.64, 72.38)]
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    W, gaps = 0.3, []
    for i, (grp, fc, cv) in enumerate(rows):
        for j, (v, c, lab) in enumerate(((fc, "#1d4ed8", "fully connected"),
                                         (cv, "#b91c1c", "convolutional"))):
            x = i + (j - 0.5) * W
            ax.bar(x, v, W * 0.92, color=c,
                   label=lab if i == 0 else None)
            ax.text(x, v + 1.1, f"{v:.2f}%", ha="center", fontsize=10.5)
        gaps.append(f"{grp}\nstructure moves it {abs(fc - cv):.2f} points")

    lo, hi = rows[1][1], rows[0][1]                     # 묶음 사이의 틈
    xb = 1.42
    ax.annotate("", (xb, lo), (xb, hi),
                arrowprops=dict(arrowstyle="<->", color="#15803d", lw=1.8))
    ax.text(xb + 0.05, (lo + hi) / 2,
            f"changing the latent\ndimension moves it\n{hi - lo:.2f} points",
            fontsize=10.5, color="#15803d", va="center")

    ax.set_xticks(range(2)); ax.set_xticklabels(gaps, fontsize=11)
    ax.set_ylim(0, 112); ax.set_xlim(-0.55, 2.05)
    ax.set_yticks(range(0, 101, 20))
    ax.set_ylabel("asked for a digit, got that digit (%)", fontsize=11)
    ax.set_title("what decides obedience is the width of the channel, "
                 "not the layers", fontsize=11.5)
    ax.legend(fontsize=10.5, frameon=False, loc="center left",
              bbox_to_anchor=(0.70, 0.30))
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig("obedience_split.svg", transparent=True)
    plt.close(fig); print("wrote obedience_split.svg")


# =============================================================================
# 장 개요 - 두 사다리 / 5.3 개요 - 시드 다섯 벌의 범위
# =============================================================================
def fig_two_ladders():
    """장 개요 - 이 장이 두 번 오른다는 것을 그림으로.

    표로 적으면 네 줄이지만, 「인코더가 있느냐」로 갈리는 구조가 안 보인다.
    왼쪽과 오른쪽으로 갈라 두면 5.1 만 따로 서 있는 까닭이 드러난다.
    지어낸 수는 없다 - 절 번호와 모델 이름뿐이다.
    """
    fig, ax = plt.subplots(figsize=(10.6, 5.4))
    ax.set_xlim(0, 100); ax.set_ylim(0, 62); ax.axis("off")

    cols = [(4, "#1d4ed8", "5.1  reconstruction",
             "has an encoder", ["PCA", "linear AE", "AE_MLP", "AE_CNN"],
             "squeeze it, then put it back"),
            (28, "#15803d", "5.2  latent generative", "has an encoder",
             ["VAE", "beta-VAE", "cVAE", "conv VAE", "conv cVAE"],
             "draw new ones from N(0, I)"),
            (54, "#b45309", "5.3  adversarial", "no encoder",
             ["GAN", "DCGAN"], "draw sharper ones"),
            (77, "#b91c1c", "5.4  diffusion", "no encoder",
             ["DDPM", "DDIM"], "draw better ones, pay in time")]
    for x, c, title, kind, steps, gain in cols:
        w = 19
        ax.text(x + w / 2, 58, title, ha="center", fontsize=12, color=c)
        ax.text(x + w / 2, 54.4, kind, ha="center", fontsize=9.5,
                color="0.45")
        for i, st in enumerate(steps):
            y = 48 - i * 6.4
            ax.add_patch(Rectangle((x, y - 2.5), w, 5, facecolor=c,
                                   alpha=0.14, edgecolor=c, lw=1.2))
            ax.text(x + w / 2, y, st, ha="center", va="center", fontsize=11)
            if i:
                ax.annotate("", (x + w / 2, y + 2.7), (x + w / 2, y + 3.9),
                            arrowprops=dict(arrowstyle="-|>", color=c, lw=1.4))
        ax.text(x + w / 2, 48 - len(steps) * 6.4 - 0.4, gain, ha="center",
                va="top", fontsize=9.5, color=c, style="italic")

    ax.plot([49, 49], [4, 56], ls=":", color="0.6", lw=1.4)
    ax.text(26.5, 1.0, "an encoder exists - you can ask what a real\n"
            "image turns into", ha="center", fontsize=10, color="0.35")
    ax.text(74.0, 1.0, "no encoder - you can only look at what\n"
            "comes out", ha="center", fontsize=10, color="0.35")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.02)
    fig.savefig("two_ladders.svg", transparent=True)
    plt.close(fig); print("wrote two_ladders.svg")


def fig_gen_seed_spread():
    """5.3 개요 5절 - 시드 다섯 벌의 범위와, 시드 42 가 어디 떨어지는가.

    수는 그 절의 두 표 그대로다. 막대가 다섯 벌의 범위이고 점이 시드 42 다.
    점이 막대 밖으로 나가면 그 수는 한 번 뽑은 운이었다는 뜻이다.
    """
    rulers = ["confidence", "class balance", "largest class",
              "within-class variety"]
    models = ["AE_MLP-64", "VAE-64", "GAN (MLP)", "DCGAN"]
    rng = {  # (low, high) x 4 rulers
        "AE_MLP-64": [(.720, .749), (.567, .637), (.564, .632), (.635, .648)],
        "VAE-64":    [(.752, .762), (.884, .899), (.324, .349), (.621, .630)],
        "GAN (MLP)": [(.903, .911), (.913, .937), (.194, .210), (.831, .905)],
        "DCGAN":     [(.899, .906), (.966, .978), (.147, .163), (1.055, 1.121)],
    }
    s42 = {"AE_MLP-64": [.737, .580, .619, .640],
           "VAE-64":    [.758, .887, .343, .628],
           "GAN (MLP)": [.908, .945, .214, .953],
           "DCGAN":     [.909, .970, .162, 1.085]}
    cols = ["#6b7280", "#1d4ed8", "#b45309", "#b91c1c"]

    fig, axes = plt.subplots(1, 4, figsize=(12.6, 4.3))
    for a, (ax, ruler) in enumerate(zip(axes, rulers)):
        for i, m in enumerate(models):
            lo, hi = rng[m][a]
            y = len(models) - 1 - i
            ax.barh(y, hi - lo, 0.42, left=lo, color=cols[i], alpha=0.55)
            v = s42[m][a]
            out = not (lo <= v <= hi)
            ax.plot(v, y, "o", ms=9, color="black" if not out else "white",
                    markeredgecolor="black", markeredgewidth=1.6, zorder=3)
            if out:
                ax.annotate("", (v, y + 0.38), (v, y + 0.72),
                            arrowprops=dict(arrowstyle="-|>", color="black",
                                            lw=1.3))
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels(models[::-1] if a == 0 else [""] * len(models),
                           fontsize=10.5)
        ax.set_title(ruler, fontsize=11)
        ax.grid(axis="x", alpha=0.2)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(left=False)
        if ruler == "within-class variety":
            ax.axvline(1.0, color="0.3", ls=":", lw=1.3)
            ax.text(1.0, -0.62, " real = 1.0", fontsize=9.5, color="0.3")
    fig.suptitle("bars = range over five training seeds,  dot = seed 42   "
                 "(hollow dot = seed 42 fell outside)", fontsize=11.5)
    fig.subplots_adjust(left=0.085, right=0.99, top=0.80, bottom=0.07,
                        wspace=0.12)
    fig.savefig("gen_seed_spread.svg", transparent=True)
    plt.close(fig); print("wrote gen_seed_spread.svg")


@torch.no_grad()
def fig_aggregate_posterior():
    """VAE 가 요구한 대로 되었는가 — mu 만 보면 속는다.

    01_vae 3절의 두 표를 한 그림에 담는다. 왼쪽은 좌표 64개의 퍼짐을
    두 가지로 잰 것이고(mu 만 / q(z) 전체), 오른쪽은 그 둘의 차이가
    어디서 오는지를 쌓은 막대로 보인다 — Var(mu) + E[sigma^2] = 1 이다.

    요점은 **왼쪽에서 아래로 처진 선이 위로 올라붙는 것**이다. 정보를
    싣지 않는 좌표일수록 sigma 가 1 에 가까워 그 자리를 메운다.
    """
    (_, _), (Xte, _) = load_flat()
    vae = load(VAEModel, 64, "vae64_beta1.0")
    h = vae.enc(Xte)
    mu, sig2 = vae.mu(h), vae.logvar(h).exp()
    sd_mu = mu.std(0)
    e_s2 = sig2.mean(0)
    sd_q = (mu.var(0) + e_s2).sqrt()
    order = sd_mu.argsort()                      # mu 가 작은 좌표부터

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.2, 4.3))

    x = range(64)
    axL.plot(x, sd_q[order], "o-", color="#15803d", lw=2, ms=4,
             label="q(z) as a whole")
    axL.plot(x, sd_mu[order], "o-", color="#b91c1c", lw=2, ms=4,
             label="only the means mu(x)")
    axL.axhline(1.0, color="0.5", ls=":", lw=1.3)
    axL.text(1, 1.045, "1.0 - what N(0,I) would give", fontsize=9.5,
             color="0.45")
    dead = int((sd_mu < 0.1).sum())
    axL.axvspan(-0.5, dead - 0.5, color="#b91c1c", alpha=0.07)
    axL.annotate(f"{dead} coordinates carry\nalmost nothing in mu",
                 (dead / 2, 0.05), (2.0, 0.60), ha="left", fontsize=9.5,
                 color="#b91c1c",
                 arrowprops=dict(arrowstyle="->", color="#b91c1c", lw=1.2))
    axL.set_xlabel("the 64 coordinates, sorted by how much mu moves",
                   fontsize=10)
    axL.set_ylabel("standard deviation", fontsize=10.5)
    axL.set_ylim(-0.05, 1.22); axL.set_xlim(-1, 64)
    axL.legend(fontsize=9.5, frameon=False, loc="lower right")
    axL.set_title("measured one way it looks collapsed, "
                  "the other way it does not", fontsize=11)
    axL.spines[["top", "right"]].set_visible(False)

    g = [("the %d quiet ones" % dead, sd_mu < 0.1),
         ("the other %d" % (64 - dead), sd_mu >= 0.1)]
    for i, (name, m) in enumerate(g):
        a, b = float(mu.var(0)[m].mean()), float(e_s2[m].mean())
        axR.bar(i, a, 0.72, color="#b91c1c", label="Var[mu]" if i == 0 else None)
        axR.bar(i, b, 0.72, bottom=a, color="#1d4ed8",
                label="E[sigma^2]" if i == 0 else None)
        for v, bot in ((a, 0.0), (b, a)):          # 얇은 칸에는 안 적는다
            if v > 0.10:
                axR.text(i, bot + v / 2, f"{v:.2f}", ha="center",
                         va="center", color="white", fontsize=11.5)
        if a <= 0.10:                                # 칸이 얇아 밑에 적는다
            axR.text(i, 0.05, f"Var[mu] = {a:.2f}", ha="center",
                     fontsize=9.5, color="white")
        axR.text(i, a + b + 0.03, f"total {a + b:.2f}", ha="center",
                 fontsize=11)
    axR.axhline(1.0, color="0.5", ls=":", lw=1.3)
    axR.set_xticks(range(2)); axR.set_xticklabels([n for n, _ in g],
                                                  fontsize=10.5)
    axR.set_ylim(0, 1.32); axR.set_xlim(-0.6, 1.6)
    axR.set_ylabel("variance of q(z), split in two", fontsize=10.5)
    axR.set_title("both add up to 1 - the quiet ones do it with sigma",
                  fontsize=11)
    axR.legend(fontsize=9.5, frameon=False, loc="upper center", ncol=2,
               bbox_to_anchor=(0.5, 1.0))
    axR.spines[["top", "right"]].set_visible(False)

    fig.subplots_adjust(left=0.07, right=0.98, top=0.90, bottom=0.14,
                        wspace=0.26)
    fig.savefig("aggregate_posterior.svg", transparent=True)
    plt.close(fig); print("wrote aggregate_posterior.svg")


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

    # 칸마다 자동으로 눈금을 맞추면 둘이 똑같이 퍼져 보인다. 눈금 수를 읽어야만
    # 차이를 아는 그림은 그림이 할 일을 글자에 떠넘긴 것이다. 같은 자리에 묶는다.
    # 가장 바깥 점에 맞추면 몇 개 때문에 축이 늘어나 VAE 쪽이 점으로 뭉친다.
    # 가운데 96% 상자를 기준으로 잡고 조금 넓힌다 (바깥 점 일부는 잘린다).
    lim = 1.18 * max(float(z.quantile(q, dim=0).abs().max())
                     for z in (ae_z, vae_z) for q in (0.02, 0.98))
    cm = plt.get_cmap("tab10")

    fig, axes = plt.subplots(2, 2, figsize=(9.5, 9.4))
    for col, (z, dec, title) in enumerate([
            (ae_z, ae.dec, "AE_MLP-2"),
            (vae_z, vae.dec, "VAE-2  (beta=1)")]):
        lo, hi = z.quantile(0.02, dim=0), z.quantile(0.98, dim=0)

        # 눈금을 지우면 두 잠재 공간의 크기가 전혀 다르다는 것이 보이지 않는다.
        # AE 는 제 마음대로 퍼지고 VAE 는 N(0, I) 쪽으로 밀려 있으므로, 축의
        # 수가 곧 "KL 항이 한 일"이다. 두 칸의 자리를 억지로 맞추지 않는다.
        ax = axes[0, col]
        # vmin/vmax 를 반 칸씩 넓혀 라벨 d 가 tab10 의 d 번 색과 정확히 맞게 한다
        ax.scatter(z[:, 0], z[:, 1], c=yte, cmap="tab10", vmin=-0.5, vmax=9.5,
                   s=1.2, alpha=0.55, linewidths=0, rasterized=True)
        # 범위를 제목 한 줄에 넣으면 오른쪽 칸에서 잘린다. 두 줄로 나눈다.
        ax.set_title(f"{title} latent\nmiddle 96%:  x {lo[0]:+.1f} to "
                     f"{hi[0]:+.1f}   y {lo[1]:+.1f} to {hi[1]:+.1f}",
                     fontsize=10)
        ax.set_aspect("equal")
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)      # 두 칸이 같은 자
        for d in range(10):                 # 범례를 찾아가지 않게 무리 위에 바로
            g = z[yte == d]
            ax.text(float(g[:, 0].median()), float(g[:, 1].median()), str(d),
                    fontsize=12, fontweight="bold", ha="center", va="center",
                    color=cm(d), zorder=5,
                    path_effects=[pe.withStroke(linewidth=2.6,
                                                foreground="white")])
        ax.tick_params(labelsize=9)
        ax.grid(alpha=0.18, lw=0.6)
        ax.axhline(0, color="0.4", lw=0.8); ax.axvline(0, color="0.4", lw=0.8)

        n = 16
        gx = torch.linspace(lo[0], hi[0], n); gy = torch.linspace(lo[1], hi[1], n)
        grid = torch.stack([torch.stack([x, y]) for y in gy.flip(0) for x in gx])
        d = dec(grid).reshape(n, n, 28, 28)
        canvas = torch.cat([torch.cat([d[i, j] for j in range(n)], 1) for i in range(n)], 0)
        # 칸의 가운데가 격자점이므로 바깥으로 반 칸씩 넓혀야 축의 수가 맞는다
        hw, hh = (hi[0] - lo[0]) / (2 * (n - 1)), (hi[1] - lo[1]) / (2 * (n - 1))
        axes[1, col].imshow((canvas * STD + MEAN).clamp(0, 1), cmap="gray",
                            vmin=0, vmax=1,
                            extent=(float(lo[0] - hw), float(hi[0] + hw),
                                    float(lo[1] - hh), float(hi[1] + hh)))
        axes[1, col].set_title(f"{title} decoded over its own middle 96%",
                               fontsize=10)
        axes[1, col].set_aspect("auto")
        axes[1, col].tick_params(labelsize=9)

    fig.legend(handles=[Line2D([], [], marker="o", ls="", ms=7, color=cm(d),
                               label=str(d)) for d in range(10)],
               loc="lower center", ncol=10, frameon=False, fontsize=10,
               handletextpad=0.15, columnspacing=1.1,
               title="which digit the point came from", title_fontsize=10)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.95, bottom=0.115,
                        wspace=0.14, hspace=0.16)
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


# =========================================================================
#  들머리 그림 — 쪽의 물음을 세우는 자리. 그 쪽 수만 쓰고 다시 재지 않는다.
# =========================================================================

def fig_lead_howmany():
    """784개 가운데 몇 개면 되는가. 5.1 개요 표의 수를 그대로 쓴다."""
    ks = [2, 16, 32, 64, 784]
    ident = [35.66, 83.69, 94.39, 97.81, 98.83]
    fig, ax = plt.subplots(figsize=(8.6, 4.0))
    xs = np.arange(len(ks))
    cols = ["#b91c1c", "0.72", "0.72", "tab:green", "0.45"]
    ax.bar(xs, ident, color=cols, width=0.6)
    for x, k, v in zip(xs, ks, ident):
        ax.text(x, v + 1.6, f"{v:.2f}%", ha="center", fontsize=10)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{k}\nnumbers" for k in ks], fontsize=10)
    ax.set_ylim(0, 112)
    ax.set_ylabel("the chapter-3 CNN still reads\nthe same digit  (%)",
                  fontsize=10)
    ax.axhline(98.83, color="0.45", lw=1.0, ls="--")
    # 설명 문장도 이름표도 적지 않는다. 바로 아래 본문이 같은 말을 하고,
    # 점선은 맨 오른쪽 784 막대의 꼭대기와 같은 높이라 저 혼자 읽힌다.
    ax.set_title("how many of a digit's 784 numbers do you actually need?",
                 fontsize=11.5)
    # 어느 방법으로 줄인 것인지 밝히는 이름표. 이 절은 방법을 넷 시험하는데
    # 이 막대는 그 가운데 첫째(PCA)로 잰 것이다.
    ax.text(-0.42, 105, "PCA", fontsize=11, color="0.45")
    # 맨 오른쪽 막대는 줄이지 않은 원본이다. 100 이 아닌 까닭은 심판 자신이
    # 그만큼밖에 못 맞히기 때문이고, 점선이 그 천장이다.
    ax.text(4.46, 105, "ceiling = the judge itself", ha="right", fontsize=9.5,
            color="0.55")
    ax.spines[["top", "right"]].set_visible(False)
    fig.subplots_adjust(left=0.10, right=0.98, top=0.88, bottom=0.14)
    fig.savefig("lead_howmany.svg", transparent=True)
    plt.close(fig); print("wrote lead_howmany.svg")


def fig_pca_funnel():
    """PCA-2 는 숫자를 뒤섞지 않고 네 곳으로 빨아들인다.

    들머리에 있던 것을 3절로 옮겼다 — 그 절이 적는 「4와 9는 둘 다 7」이
    바로 이 그림이다. 짝은 그 쪽 그림이 쓰는 열 장(숫자마다 첫 장)으로
    PCA-2 를 돌려 심판에게 읽혀 얻은 것이고 이 저장소에서 확인했다.
    """
    dest = {0: 0, 1: 1, 2: 3, 3: 3, 4: 7, 5: 0, 6: 0, 7: 7, 8: 0, 9: 7}
    fig, ax = plt.subplots(figsize=(8.8, 4.3))
    ypos = {d: 9 - d for d in range(10)}
    tpos = {t: i for i, t in enumerate([7, 3, 1, 0])}
    cmap = {0: "#2563eb", 1: "#16a34a", 3: "#c2410c", 7: "#9333ea"}
    for d in range(10):
        t = dest[d]
        y0, y1 = ypos[d], tpos[t] * 3 + 1.5
        kept = d == t
        ax.annotate("", xy=(2.62, y1), xytext=(0.38, y0),
                    arrowprops=dict(arrowstyle="-|>", color=cmap[t],
                                    lw=2.4 if not kept else 1.1,
                                    alpha=0.85 if not kept else 0.35,
                                    connectionstyle="arc3,rad=0.12"))
        ax.text(0.1, y0, str(d), ha="center", va="center", fontsize=13,
                color="0.25" if not kept else "0.6")
    for t, i in tpos.items():
        ax.text(2.9, i * 3 + 1.5, str(t), ha="center", va="center", fontsize=17,
                color="white", weight="bold",
                bbox=dict(boxstyle="circle,pad=0.42", fc=cmap[t], ec="none"))
    ax.text(0.1, 11.6, "you wrote", ha="center", fontsize=10, color="0.35")
    ax.text(2.9, 11.6, "it comes back as", ha="center", fontsize=10, color="0.35")
    ax.set_xlim(-0.35, 3.45); ax.set_ylim(-0.7, 12.3)
    ax.set_xticks([]); ax.set_yticks([]); ax.spines[:].set_visible(False)
    ax.set_title("squeeze a digit down to 2 numbers and back: "
                 "ten digits land on four", fontsize=11.5)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.91, bottom=0.075)
    fig.savefig("pca_funnel.svg", transparent=True)
    plt.close(fig); print("wrote pca_funnel.svg")


def fig_lead_projections(n=120):
    """PCA 와 회귀가 갈리는 까닭 — 무엇까지의 거리를 줄이느냐가 다르다.

    통계에서 쓰는 표준 그림이다. 같은 점에 두 직선을 긋고, 각 점에서
    **자기 직선까지 내리는 선**을 함께 그린다. 왼쪽은 직선에 수직으로
    내리고(PCA), 오른쪽은 세로로 곧장 내린다(최소제곱 회귀).

    점은 이 쪽 상자의 `fig_pearson_two_lines` 와 **같은 자료**다 — 같은
    생성기에 같은 씨앗 0, 같은 n = 120. 그래서 기울기도 그 그림과 똑같이
    나오고(PCA 1.00, 회귀 0.75) 두 그림이 서로를 받친다.
    """
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, n)
    y = 0.8 * x + rng.normal(0, 0.6, n)
    mx, my = x.mean(), y.mean()

    b_ols, a_ols = np.polyfit(x, y, 1)
    Xc = np.c_[x - mx, y - my]
    Vt = np.linalg.svd(Xc, full_matrices=False)[2]
    b_pca = Vt[0, 1] / Vt[0, 0]
    a_pca = my - b_pca * mx

    def perp_foot(bb, aa):
        t = (x + bb * (y - aa)) / (1 + bb ** 2)
        return t, aa + bb * t

    # 각자 제 자로 재면 제가 이긴다 — 그림이 거짓말하지 않도록 못박는다
    def perp_sse(bb, aa):
        return float((((y - (aa + bb * x)) ** 2) / (1 + bb ** 2)).sum())
    def vert_sse(bb, aa):
        return float((((y - (aa + bb * x)) ** 2)).sum())
    assert perp_sse(b_pca, a_pca) < perp_sse(b_ols, a_ols)
    assert vert_sse(b_ols, a_ols) < vert_sse(b_pca, a_pca)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.4, 4.9), sharex=True,
                                   sharey=True)
    xs = np.array([x.min() - 0.45, x.max() + 0.45])

    # --- 왼쪽: PCA — 직선까지 수직으로 ------------------------------------
    fx, fy = perp_foot(b_pca, a_pca)
    for i in range(n):
        axL.plot([x[i], fx[i]], [y[i], fy[i]], color="#c2410c", lw=0.9,
                 alpha=0.45, zorder=2)
    axL.plot(xs, a_pca + b_pca * xs, color="#c2410c", lw=2.3, zorder=3)
    axL.scatter(x, y, s=15, color="0.35", zorder=4, linewidths=0)
    axL.set_title(f"PCA — the line that sits closest to the cloud\n"
                  f"slope {b_pca:.2f}", fontsize=11)
    axL.text(0.97, 0.05, "every point drops at a right angle",
             transform=axL.transAxes, fontsize=9.8, color="#c2410c",
             ha="right", va="bottom")

    # --- 오른쪽: 회귀 — 세로로 곧장 ---------------------------------------
    for i in range(n):
        axR.plot([x[i], x[i]], [y[i], a_ols + b_ols * x[i]], color="tab:blue",
                 lw=0.9, alpha=0.45, zorder=2)
    axR.plot(xs, a_ols + b_ols * xs, color="tab:blue", lw=2.3, zorder=3)
    axR.scatter(x, y, s=15, color="0.35", zorder=4, linewidths=0)
    axR.set_title(f"regression — the line that predicts y from x\n"
                  f"slope {b_ols:.2f}", fontsize=11)
    axR.text(0.97, 0.05, "every point drops straight down",
             transform=axR.transAxes, fontsize=9.8, color="tab:blue",
             ha="right", va="bottom")

    for ax in (axL, axR):
        ax.set_xlabel("x", fontsize=10)
        ax.set_aspect("equal", adjustable="box")
        ax.spines[["top", "right"]].set_visible(False)
    axL.set_ylabel("y", fontsize=10)

    fig.suptitle("same points, two different lines — because they shorten "
                 "different distances", fontsize=12.5, y=0.985)
    fig.subplots_adjust(left=0.065, right=0.985, top=0.80, bottom=0.11,
                        wspace=0.12)
    fig.savefig("lead_projections.svg", transparent=True)
    plt.close(fig)
    print(f"wrote lead_projections.svg  (PCA {b_pca:.2f}, OLS {b_ols:.2f})")


def fig_lead_tie():
    """같은 문제를 푸는 두 길. 방법이 다른 것이지 문제가 다른 것이 아니다.

    요점은 **같은 식을 푼다**는 데 있다. 둘 다 「784개를 64개로 줄였다
    되살렸을 때 제곱오차가 가장 작게」라는 한 문제를 풀고, 한쪽은 공식으로
    한 번에, 한쪽은 조금씩 고쳐 가며 푼다. 그림이 그 관계를 보이도록
    문제를 위에 한 번만 적고 두 길이 거기서 갈라져 나오게 했다.
    """
    fig = plt.figure(figsize=(9.6, 4.6))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis("off")

    # --- 위: 둘이 함께 푸는 하나의 문제 ------------------------------------
    ax.add_patch(plt.Rectangle((0.14, 0.76), 0.72, 0.17, facecolor="0.94",
                               edgecolor="0.70", lw=1.2))
    ax.text(0.5, 0.875, "one and the same problem", ha="center",
            fontsize=11.5, weight="bold", color="0.2")
    ax.text(0.5, 0.805,
            "keep only 64 numbers, and make the squared error of the "
            "rebuilt picture as small as it can be",
            ha="center", fontsize=10, color="0.35")

    # --- 갈라지는 두 길 -----------------------------------------------------
    for x0, x1 in ((0.5, 0.29), (0.5, 0.71)):
        ax.annotate("", xy=(x1, 0.655), xytext=(x0, 0.755),
                    arrowprops=dict(arrowstyle="-|>", color="0.55", lw=1.6))
    ax.text(0.325, 0.705, "solve it", ha="right", fontsize=9.5, color="0.45")
    ax.text(0.675, 0.705, "search for it", ha="left", fontsize=9.5,
            color="0.45")

    routes = [(0.29, "tab:green", "a formula, once",
               "eigen-decomposition — no seed, no epochs", 0.09530),
              (0.71, "#c2410c", "gradient descent",
               "100 epochs x 600 batches = 60,000 updates", 0.09582)]
    for cx, col, name, how, val in routes:
        ax.add_patch(plt.Rectangle((cx - 0.195, 0.47), 0.39, 0.185,
                                   facecolor=col, alpha=0.13,
                                   edgecolor=col, lw=1.4))
        ax.text(cx, 0.595, name, ha="center", fontsize=11.5, weight="bold",
                color=col)
        ax.text(cx, 0.523, how, ha="center", fontsize=9.3, color="0.4")
        ax.annotate("", xy=(cx, 0.345), xytext=(cx, 0.455),
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=1.6))
        ax.text(cx, 0.265, f"{val:.5f}", ha="center", fontsize=19, color=col)

    # --- 아래: 같은 자리에 닿는다 -------------------------------------------
    ax.annotate("", xy=(0.60, 0.205), xytext=(0.40, 0.205),
                arrowprops=dict(arrowstyle="<->", color="0.45", lw=1.3))
    ax.text(0.5, 0.165, "0.00052 apart — and the search ends up on the "
            "wrong side of it", ha="center", fontsize=10, color="0.35")
    ax.text(0.5, 0.065, "reconstruction error on the test digits "
            "(lower is better)", ha="center", fontsize=9, color="0.5")

    fig.savefig("lead_tie.svg", transparent=True)
    plt.close(fig); print("wrote lead_tie.svg")


def fig_lead_curved(n=110):
    """같은 규칙, 다른 꼴 — 곧은 선과 굽은 선에 저마다 수직으로 내린다.

    01_pca 들머리의 왼쪽 칸을 그대로 이어받는다. 거기서는 곧은 선에
    수직으로 내렸는데, 여기서는 **굽은 선**에 수직으로 내린다. 내리는
    규칙은 글자 그대로 같고 **허락된 꼴만 다르다.**

    자료는 굽은 길 위에 흩뿌린 점이다(씨앗 1). 오른쪽 곡선은 그 점들에
    **맞춘** 것이지 만들 때 쓴 곡선을 그대로 보여 주는 것이 아니다 —
    주성분 방향으로 펴서 3차 다항식을 최소제곱으로 맞춘다. 두 손해는
    촘촘히 찍은 점까지의 거리를 재서 여기서 셈한다.

    2차원 장난감이다. 이 쪽이 실제로 하는 일은 784차원에서 64차원짜리
    굽은 면을 찾는 것이라 그림으로 그릴 수 없다.
    """
    rng = np.random.default_rng(1)
    t = rng.uniform(-2.0, 2.0, n)
    cx, cy = t, 0.40 * t ** 3 - 1.05 * t      # 굽은 길 (S자)
    dydt = 1.20 * t ** 2 - 1.05
    nx, ny = -dydt, np.ones_like(t)           # 그 길의 법선
    nn_ = np.hypot(nx, ny); nx, ny = nx / nn_, ny / nn_
    r = rng.normal(0, 0.17, n)
    x, y = cx + r * nx, cy + r * ny

    mx, my = x.mean(), y.mean()
    Vt = np.linalg.svd(np.c_[x - mx, y - my], full_matrices=False)[2]
    d = Vt[0]                                  # 제1주성분 방향
    u = (x - mx) * d[0] + (y - my) * d[1]      # 선 위 좌표
    v = -(x - mx) * d[1] + (y - my) * d[0]     # 선에서 떨어진 거리
    sse_line = float((v ** 2).sum())

    coef = np.polyfit(u, v, 3)                 # 편 좌표에서 곡선을 맞춘다
    us = np.linspace(u.min() - 0.15, u.max() + 0.15, 1200)
    vs = np.polyval(coef, us)
    curve = np.c_[mx + us * d[0] - vs * d[1], my + us * d[1] + vs * d[0]]
    line = np.c_[mx + us * d[0], my + us * d[1]]

    def feet(target):
        k = np.argmin((target[None, :, 0] - x[:, None]) ** 2 +
                      (target[None, :, 1] - y[:, None]) ** 2, axis=1)
        return target[k]
    fl, fc = feet(line), feet(curve)
    sse_curve = float((((x - fc[:, 0]) ** 2 + (y - fc[:, 1]) ** 2)).sum())
    assert sse_curve < sse_line            # 굽은 쪽이 이겨야 그림이 참이다

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.4, 4.9), sharex=True,
                                   sharey=True)
    for ax, (foot, path, col, ttl, sub) in zip(
            (axL, axR),
            [(fl, line, "#c2410c", "a straight line",
              "the best straight line there is"),
             (fc, curve, "tab:green", "a curved line",
              "same rule, only the shape is freer")]):
        for i2 in range(n):
            ax.plot([x[i2], foot[i2, 0]], [y[i2], foot[i2, 1]], color=col,
                    lw=0.9, alpha=0.45, zorder=2)
        ax.plot(path[:, 0], path[:, 1], color=col, lw=2.4, zorder=3)
        ax.scatter(x, y, s=15, color="0.35", zorder=4, linewidths=0)
        ax.set_title(f"{ttl}\n{sub}", fontsize=11, color=col)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([]); ax.set_yticks([])
        ax.spines[:].set_visible(False)
    axL.text(0.5, -0.02, f"total squared distance  {sse_line:.1f}",
             transform=axL.transAxes, ha="center", fontsize=10.5,
             color="#c2410c")
    axR.text(0.5, -0.02, f"total squared distance  {sse_curve:.1f}",
             transform=axR.transAxes, ha="center", fontsize=10.5,
             color="tab:green")

    fig.suptitle("every point still drops at a right angle — "
                 "only what it drops onto has changed", fontsize=12.5,
                 y=0.975)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.78, bottom=0.09,
                        wspace=0.06)
    fig.savefig("lead_curved.svg", transparent=True)
    plt.close(fig)
    print(f"wrote lead_curved.svg  (line {sse_line:.1f} -> curve "
          f"{sse_curve:.1f}, {100*(1-sse_curve/sse_line):.0f}% less)")


def fig_lead_same_trick():
    """같은 수법이 숫자에서는 벌고 사진에서는 못 번다."""
    fig, ax = plt.subplots(figsize=(8.4, 4.0))
    sets = ["MNIST\n(handwritten digits)", "CIFAR-10\n(colour photographs)"]
    straight = [0.0953, 0.1337]
    curved = [0.0546, 0.1343]
    x = np.arange(2)
    ax.bar(x - 0.19, straight, 0.37, color="0.72", label="straight path")
    ax.bar(x + 0.19, curved, 0.37, color="tab:green", label="curved path")
    for xi, (a, b) in zip(x, zip(straight, curved)):
        ax.text(xi - 0.19, a + 0.004, f"{a:.4f}", ha="center", fontsize=9.5)
        ax.text(xi + 0.19, b + 0.004, f"{b:.4f}", ha="center", fontsize=9.5)
        gain = (a - b) / a * 100
        ax.text(xi, max(a, b) + 0.019,
                f"{gain:+.0f}%" if abs(gain) > 1 else "no gain",
                ha="center", fontsize=13,
                color="tab:green" if gain > 1 else "#b91c1c")
    ax.set_xticks(x); ax.set_xticklabels(sets, fontsize=10.5)
    ax.set_ylim(0, 0.175)
    ax.set_ylabel("reconstruction error", fontsize=10)
    ax.legend(fontsize=9.5, frameon=False, loc="upper left")
    ax.set_title("the very same change — bend the path — twice",
                 fontsize=11.5)
    ax.spines[["top", "right"]].set_visible(False)
    fig.subplots_adjust(left=0.11, right=0.98, top=0.88, bottom=0.14)
    fig.savefig("lead_same_trick.svg", transparent=True)
    plt.close(fig); print("wrote lead_same_trick.svg")


def fig_lead_flip():
    """뒤집힘은 과적합이 아니다 — 잠재 2 에서 conv 는 학습 오차마저 높다.

    가로는 **학습** 자료에서의 복원 오차, 세로는 **시험** 자료에서의 그것.
    점선이 둘이 같은 자리다. 점이 그 선 위로 멀리 뜨면 과적합이고,
    선 가까이면 외운 것이 아니라 못 맞힌 것이다.

    네 수는 이 쪽의 규약(씨앗 42, Adam 1e-3, 배치 100, 100 에포크)으로
    네 벌을 따로 학습해 여기서 쟀다. 시험값이 이 쪽 두 표의 0.0546·
    0.0243·0.422·0.474 를 되살리므로 학습값도 같은 자로 읽을 수 있다.

    파라메터 수를 축으로 쓰지 않는다. conv 인코더의 컨볼루션 부분은
    k 와 무관하게 4,800 개로 똑같고 달라지는 것은 짜내는 마지막 한 겹뿐이라,
    두 모델의 총수를 견주는 것은 서로 다른 일을 하는 부분끼리 더해 견주는
    꼴이 된다.

    그 쪽 1절이 쓰는 나머지 수도 같은 규약으로 잰 것이다. 잠재를 98 개로
    맞추고 짜내는 방식만 바꾸면 —

        conv 공간형 (2,7,7)  학습 0.03590  시험 0.03488  인코더   5,378
        conv 벡터  98        학습 0.01765  시험 0.01885  인코더 158,562
        MLP  벡터  98        학습 0.04669  시험 0.04611  인코더 226,146

    완전 연결을 치운 공간형이 **오히려 나쁘다.** 다만 공간형으로 바꾸면
    잠재 수·디코더·파라메터가 함께 달라지므로 한 가지만 바꾼 실험이 아니다.
    """
    cases = [(64, (0.05510, 0.05462), (0.02304, 0.02427)),
             (2,  (0.39441, 0.42230), (0.47117, 0.47423))]
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8))
    for ax, (k, mlp, cnn) in zip(axes, cases):
        worse = cnn[1] > mlp[1]
        col = "#b91c1c" if worse else "tab:green"
        lo = min(mlp[0], mlp[1], cnn[0], cnn[1])
        hi = max(mlp[0], mlp[1], cnn[0], cnn[1])
        pad = (hi - lo) * 0.42
        lim = (lo - pad, hi + pad)
        ax.plot(lim, lim, color="0.70", lw=1.1, ls="--", zorder=1)
        ax.annotate("", xy=cnn, xytext=mlp,
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=2.5,
                                    shrinkA=13, shrinkB=13))
        # 화살표가 왼쪽아래에서 오른쪽위로 가므로, 아래 점은 오른쪽아래에
        # 위 점은 왼쪽위에 적어야 겹치지 않는다
        lower = mlp if mlp[0] < cnn[0] else cnn
        for (px, py), name, c in ((mlp, "AE_MLP", "0.45"), (cnn, "AE_CNN", col)):
            is_low = (px, py) == lower
            dx, dy, ha = (16, -14, "left") if is_low else (-16, 12, "right")
            ax.plot([px], [py], "o", ms=13, color=c, zorder=3)
            ax.annotate(name, xy=(px, py), xytext=(dx, dy),
                        textcoords="offset points", ha=ha, fontsize=11,
                        color=c)
            ax.annotate(f"{px:.3f}  /  {py:.3f}", xy=(px, py),
                        xytext=(dx, dy - 15), textcoords="offset points",
                        ha=ha, fontsize=9.5, color="0.5")
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("error on the training digits", fontsize=10)
        ax.set_title(f"{k}", fontsize=22, color="0.25", pad=14)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("error on the test digits", fontsize=10)
    fig.text(0.5, 0.925, "numbers kept", ha="center", fontsize=10.5,
             color="0.45")
    fig.subplots_adjust(left=0.085, right=0.98, top=0.80, bottom=0.14,
                        wspace=0.30)
    fig.savefig("lead_flip.svg", transparent=True)
    plt.close(fig); print("wrote lead_flip.svg")


def fig_lead_worse_than_nothing():
    """애써 배운 표현이 아무것도 안 한 것을 넘지 못한다.

    왼쪽은 **같은 MLP 탐침**으로 견준 것이라 like-for-like 다(원시 화소
    50.94%, PCA 50.22, 선형 AE 51.54, 깊은 AE 50.84, conv AE 납작하게
    43.52). 오른쪽은 양쪽이 쓸 수 있는 가장 좋은 머리끼리 — 이 절의 최선
    58.67% 대 원시 화소에 CNN 을 얹은 72.58% 다. 모두 그 쪽의 표에서 옮겼다.
    """
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.4, 4.2),
                                   gridspec_kw={"width_ratios": [1.5, 1]})

    names = ["PCA", "linear AE", "deep AE", "conv AE\n(flattened)"]
    vals = [50.22, 51.54, 50.84, 43.52]
    base = 50.94
    axL.bar(range(4), vals, color=["0.72", "0.72", "0.72", "#b91c1c"],
            width=0.58)
    for i, v in enumerate(vals):
        axL.text(i, v + 0.9, f"{v:.2f}", ha="center", fontsize=10)
    axL.axhline(base, color="tab:green", lw=1.7, ls="--")
    axL.text(3.45, 59.5, f"raw pixels, nothing learned — {base}%",
             ha="right", fontsize=10, color="tab:green")
    axL.annotate("", xy=(3.3, base), xytext=(3.3, 58.6),
                 arrowprops=dict(arrowstyle="->", color="tab:green", lw=1.0))
    axL.set_xticks(range(4)); axL.set_xticklabels(names, fontsize=10)
    axL.set_ylim(0, 66)
    axL.set_ylabel("accuracy, same MLP probe on top", fontsize=10)
    axL.set_title("like for like: four steps against doing nothing",
                  fontsize=11)
    axL.spines[["top", "right"]].set_visible(False)

    axR.bar([0, 1], [58.67, 72.58], color=["#b91c1c", "tab:green"], width=0.5)
    for x, v in zip([0, 1], [58.67, 72.58]):
        axR.text(x, v + 1.4, f"{v:.2f}%", ha="center", fontsize=11.5)
    axR.annotate("", xy=(0.5, 58.67), xytext=(0.5, 72.58),
                 arrowprops=dict(arrowstyle="<->", color="0.4", lw=1.3))
    axR.text(0.56, 65.6, "13.91\npoints", fontsize=10, color="0.3")
    axR.set_xticks([0, 1])
    axR.set_xticklabels(["this section's best\n(conv AE + CNN head)",
                         "raw pixels\n+ CNN"], fontsize=10)
    axR.set_ylim(0, 88)
    axR.set_title("best against best", fontsize=11)
    axR.spines[["top", "right"]].set_visible(False)

    fig.suptitle("four careful steps, and none of them beats doing nothing",
                 fontsize=12.5, y=0.985)
    fig.text(0.5, 0.02, "CIFAR-10 - every number read off this page's own tables.",
             ha="center", fontsize=8.8, color="0.45")
    fig.subplots_adjust(left=0.075, right=0.98, top=0.80, bottom=0.17,
                        wspace=0.22)
    fig.savefig("lead_worse_than_nothing.svg", transparent=True)
    plt.close(fig); print("wrote lead_worse_than_nothing.svg")


def fig_lead_cliff():
    """되살리기는 되는데 만들기는 안 된다."""
    fig, ax = plt.subplots(figsize=(8.4, 4.0))
    names = ["real digits", "rebuilt from\ntheir own 64 numbers",
             "made from\n64 numbers we invented"]
    vals = [0.992, 0.988, 0.776]
    cols = ["0.55", "tab:green", "#b91c1c"]
    ax.bar(range(3), vals, color=cols, width=0.56)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.012, f"{v:.3f}", ha="center", fontsize=11.5)
    ax.set_xticks(range(3)); ax.set_xticklabels(names, fontsize=10)
    ax.set_ylim(0, 1.13)
    ax.set_ylabel("how sure the judge is it is a digit", fontsize=10)
    # 벼랑이 눈에 보이므로 문장을 덧붙이지 않는다.
    ax.set_title("the decoder will rebuild any digit — "
                 "so why can it not invent one?", fontsize=11.5)
    ax.spines[["top", "right"]].set_visible(False)
    fig.subplots_adjust(left=0.10, right=0.98, top=0.88, bottom=0.16)
    fig.savefig("lead_cliff.svg", transparent=True)
    plt.close(fig); print("wrote lead_cliff.svg")


def fig_lead_kl_buys():
    """5.2 개요 - 장 개요의 자 넷으로 모델 다섯을 본다.

    넷째 자 「값」(그림 한 장에 디코더를 몇 번 지나는가) 도 선으로 그린다.
    넷 다 한 번이라 1.0 에 평평하게 눕고, 목표선과 겹쳐 앉는다. 갈리지
    않는다는 것을 글로 적어 두는 것보다 평평한 선 하나가 바로 보인다.
    1,000 회까지 벌어지는 것은 5.4 뿐이고 그 사실만 아래에 적는다.

    수의 출처: 확신도·균형도는 5.2 개요 표. 다양함은 오토인코더 0.642 와
    VAE·beta-VAE 가 02_beta_vae 3절, 컨볼루션 VAE 0.833 은 같은 규약
    (씨앗 42, 100 에포크)으로 따로 재어 확신도 0.807·균형도 0.927 이
    표와 맞는 것을 확인했다.
    """
    models = ["autoencoder", "VAE", "beta-VAE", "conv VAE", "conv beta-VAE"]
    # 오른쪽 끝에서 세 자가 0.807·0.927·0.833 으로 붙어 있어 데이터 자리에
    # 이름을 적으면 겹친다. 이름의 세로 자리와 값 라벨의 방향을 따로 준다.
    # 오른쪽 이름표는 선이 **끝나는** 높이 차례로 놓는다. 끝값이
    # 1.0 / 0.965 / 0.828 / 0.716 이므로 이름표도 그 순서여야 눈이 선을
    # 되짚지 않는다.
    NL = chr(10)
    rulers = [("does it look" + NL + " like a digit?",
               [0.777, 0.769, 0.794, 0.807, 0.828], "0.6", 0.845, -0.058),
              ("are all ten made" + NL + " equally often?",
               [0.541, 0.877, 0.950, 0.927, 0.965], "tab:green", 0.955, -0.042),
              ("as varied as" + NL + " real handwriting?",
               [0.642, 0.631, 0.575, 0.833, 0.716], "#1d4ed8", 0.712, -0.047),
              # 넷째 자 - 넷 다 디코더를 한 번 지나므로 1.0 에 평평하다
              ("one decoder pass" + NL + " for every model",
               [1.0] * 5, "#b45309", 1.063, 0.021)]
    fig, ax = plt.subplots(figsize=(11.0, 5.1))
    x = list(range(5))
    ax.axhline(1.0, color="0.65", ls=":", lw=1.2)
    for name, v, c, lab_y, dy in rulers:
        flat = len(set(v)) == 1
        ax.plot(x, v, "o-", color=c, lw=2.4, ms=9)
        for xi, y in zip(x, v):
            ax.text(xi, y + dy, "1" if flat else f"{y:.3f}", ha="center",
                    fontsize=10, color=c)
        ax.text(4.16, lab_y, name, va="center", fontsize=10, color=c)

    ax.text(-0.33, 1.093, "1.00  the target for all four", fontsize=9.5,
            color="0.5")
    ax.set_xticks(x); ax.set_xticklabels(models, fontsize=11)
    ax.set_ylabel("score   1 = best for all four", fontsize=10.5)
    ax.set_xlim(-0.38, 5.75); ax.set_ylim(0.46, 1.13)
    ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.set_title("four rulers, five models - the last two are the same knob "
                 "on a different machine", fontsize=11.5)
    fig.text(0.5, 0.015,
             "pairs 2-3 and 4-5 are the same beta step on two machines - "
             "conv keeps more variety at either setting, and loses more to the step",
             ha="center", fontsize=9.5, color="0.45")
    fig.subplots_adjust(left=0.10, right=0.995, top=0.90, bottom=0.17)
    fig.savefig("lead_kl_buys.svg", transparent=True)
    plt.close(fig); print("wrote lead_kl_buys.svg")


def fig_lead_price_tag():
    """치른 값과 산 것. 복원은 나빠지고 쏠림은 풀린다."""
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(10.4, 4.0))
    axL.bar([0, 1], [0.0546, 0.0750], color=["0.72", "#b91c1c"], width=0.5)
    for x, v in zip([0, 1], [0.0546, 0.0750]):
        axL.text(x, v + 0.0026, f"{v:.4f}", ha="center", fontsize=11)
    axL.set_xticks([0, 1]); axL.set_xticklabels(["autoencoder", "VAE"],
                                                fontsize=10.5)
    axL.set_ylim(0, 0.098)
    axL.set_ylabel("reconstruction error", fontsize=10)
    axL.set_title("what it costs  (worse)", fontsize=11, color="#b91c1c")
    axL.spines[["top", "right"]].set_visible(False)

    axR.bar([0, 1], [66, 36], color=["#b91c1c", "tab:green"], width=0.5)
    for x, v in zip([0, 1], [66, 36]):
        axR.text(x, v + 2.2, f"{v}%", ha="center", fontsize=11)
    axR.set_xticks([0, 1]); axR.set_xticklabels(["autoencoder", "VAE"],
                                                fontsize=10.5)
    axR.set_ylim(0, 86)
    axR.set_ylabel("share taken by the commonest digit", fontsize=10)
    axR.set_title("what it buys  (better)", fontsize=11, color="tab:green")
    axR.spines[["top", "right"]].set_visible(False)

    fig.subplots_adjust(left=0.085, right=0.98, top=0.86, bottom=0.12,
                        wspace=0.28)
    fig.savefig("lead_price_tag.svg", transparent=True)
    plt.close(fig); print("wrote lead_price_tag.svg")


def fig_lead_dial():
    """손잡이를 돌리면 둘이 반대로 간다. 어디서 멈출지는 알려 주지 않는다."""
    betas = [0, 0.5, 1.0, 4.0]
    rec = [0.0546, 0.0577, 0.0750, 0.1351]
    bal = [0.541, 0.784, 0.877, 0.950]
    x = np.arange(4)
    fig, ax = plt.subplots(figsize=(9.0, 4.2))
    ax.plot(x, [r / rec[0] for r in rec], "s-", color="#b91c1c", lw=2.3, ms=8)
    ax.plot(x, [b / bal[0] for b in bal], "o-", color="tab:green", lw=2.3, ms=8)
    for xi, r, b in zip(x, rec, bal):
        # 빨강은 아래, 초록은 위로 갈라 둔다 — 같은 쪽에 두면 1.0 에서 겹친다
        ax.text(xi, r / rec[0] - 0.145, f"{r:.4f}", ha="center", fontsize=9.5,
                color="#b91c1c")
        ax.text(xi, b / bal[0] + 0.085, f"{b:.3f}", ha="center", fontsize=9.5,
                color="tab:green")
    ax.text(3.16, rec[3] / rec[0], "reconstruction\n(getting worse)",
            va="center", fontsize=10, color="#b91c1c")
    ax.text(3.16, bal[3] / bal[0], "all ten equally often\n(getting better)",
            va="center", fontsize=10, color="tab:green")
    ax.set_xticks(x)
    ax.set_xticklabels(["0\n(no term)", "0.5", "1.0", "4.0"], fontsize=10.5)
    ax.set_xlabel("the one knob", fontsize=10.5)
    ax.set_xlim(-0.35, 4.75); ax.set_ylim(0.5, 2.95)
    ax.set_yticks([])
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.set_title("both keep going — so where do you stop?", fontsize=11.5)
    fig.subplots_adjust(left=0.03, right=0.98, top=0.88, bottom=0.17)
    fig.savefig("lead_dial.svg", transparent=True)
    plt.close(fig); print("wrote lead_dial.svg")


def fig_lead_who_disobeys():
    """말을 안 듣는 것이 기계의 힘 때문이 아니다."""
    fig, ax = plt.subplots(figsize=(9.2, 4.1))
    names = ["small machine\n(2 numbers)", "big machine\n(64 numbers)",
             "the same big machine,\nhanded a real digit's numbers"]
    vals = [99.66, 73.64, 98.42]
    cols = ["tab:green", "#b91c1c", "tab:blue"]
    ax.bar(range(3), vals, color=cols, width=0.55)
    for i, v in enumerate(vals):
        ax.text(i, v + 1.6, f"{v:.2f}%", ha="center", fontsize=12)
    ax.set_xticks(range(3)); ax.set_xticklabels(names, fontsize=10)
    ax.set_ylim(0, 116)
    ax.set_ylabel("asked for a 7, got a 7", fontsize=10)
    ax.set_title("the big machine disobeys — but not because it cannot obey",
                 fontsize=11.5)
    ax.spines[["top", "right"]].set_visible(False)
    fig.subplots_adjust(left=0.085, right=0.98, top=0.87, bottom=0.17)
    fig.savefig("lead_who_disobeys.svg", transparent=True)
    plt.close(fig); print("wrote lead_who_disobeys.svg")


def fig_lead_average_blur():
    """제곱오차가 애매할 때 고르는 답은 여럿의 한가운데다.

    진짜 MNIST 만 쓴다. 3 을 여러 장 모아 평균 내면 아무도 그렇게 쓰지 않는
    흐릿한 3 이 나오는데, 화소 제곱오차로 보면 그것이 가장 점수가 높다.
    """
    tf = transforms.ToTensor()
    te = torchvision.datasets.MNIST(DATA, train=False, download=True,
                                    transform=tf)
    from torch.utils.data import DataLoader
    X = torch.cat([x for x, _ in DataLoader(te, batch_size=2000)])
    y = torch.cat([l for _, l in DataLoader(te, batch_size=2000)])
    threes = X[y == 3][:, 0]
    mean3 = threes.mean(0)
    # 평균이 어느 한 장보다도 제곱오차가 작다는 것을 셈으로 확인한다
    d_mean = ((threes - mean3) ** 2).mean().item()
    d_best = min(((threes - threes[k]) ** 2).mean().item() for k in range(60))

    fig = plt.figure(figsize=(10.6, 3.4))
    gs = fig.add_gridspec(1, 8, left=0.015, right=0.985, top=0.80, bottom=0.14,
                          wspace=0.10)
    for k in range(6):
        ax = fig.add_subplot(gs[0, k])
        ax.imshow(threes[k], cmap="gray_r", vmin=0, vmax=1)
        ax.set_xticks([]); ax.set_yticks([]); ax.spines[:].set_visible(False)
        if k == 0:
            ax.set_title("six real 3s", fontsize=10.5, loc="left", color="0.3")
    ax = fig.add_subplot(gs[0, 7])
    ax.imshow(mean3, cmap="gray_r", vmin=0, vmax=1)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color("#b91c1c"); sp.set_linewidth(2.0)
    ax.set_title("their average", fontsize=10.5, color="#b91c1c")
    fig.text(0.5, 0.035,
             f"nobody writes the one on the right — yet by squared error it "
             f"beats every real 3  ({d_mean:.4f} against {d_best:.4f})",
             ha="center", fontsize=9.6, color="0.35")
    fig.savefig("lead_average_blur.svg", transparent=True)
    plt.close(fig)
    print(f"wrote lead_average_blur.svg  (mean {d_mean:.4f} vs best real "
          f"{d_best:.4f})")


def fig_lead_loss_wanders():
    """로스가 내려가면 좋아지는 것이 아니다. 01_gan 3절 표의 수."""
    ep = [1, 10, 20, 30]
    D = [0.811, 0.589, 1.026, 1.037]
    G = [1.687, 2.607, 1.360, 1.310]
    fig, ax = plt.subplots(figsize=(8.8, 4.2))
    ax.plot(ep, D, "o-", color="tab:blue", lw=2.3, ms=8, label="the one that judges")
    ax.plot(ep, G, "s-", color="#c2410c", lw=2.3, ms=8, label="the one that draws")
    ax.axhline(2 * np.log(2), color="tab:blue", lw=1.0, ls=":")
    ax.axhline(np.log(2), color="#c2410c", lw=1.0, ls=":")
    ax.text(33.2, 2 * np.log(2), " balance 1.386", va="center", fontsize=9,
            color="tab:blue")
    ax.text(33.2, np.log(2), " balance 0.693", va="center", fontsize=9,
            color="#c2410c")
    for e, d, g in zip(ep, D, G):
        ax.text(e, d - 0.16, f"{d:.3f}", ha="center", fontsize=9, color="tab:blue")
        ax.text(e, g + 0.11, f"{g:.3f}", ha="center", fontsize=9, color="#c2410c")
    ax.set_xticks(ep); ax.set_xlabel("epoch", fontsize=10)
    ax.set_xlim(-1, 39); ax.set_ylim(0.3, 3.0)
    ax.set_ylabel("loss", fontsize=10)
    ax.legend(fontsize=9.5, frameon=False, loc="upper left")
    ax.set_title("neither loss settles — and that is normal here",
                 fontsize=11.5)
    ax.spines[["top", "right"]].set_visible(False)
    fig.subplots_adjust(left=0.085, right=0.90, top=0.88, bottom=0.14)
    fig.savefig("lead_loss_wanders.svg", transparent=True)
    plt.close(fig); print("wrote lead_loss_wanders.svg")


def fig_lead_scoreboard_lies():
    """성적표는 더 나쁘고 결과물은 더 좋다. 02_dcgan 의 표에서 옮긴 수."""
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.6, 4.2))
    gapG, gapD = abs(1.310 - np.log(2)), abs(3.362 - np.log(2))
    axL.bar([0, 1], [gapG, gapD], color=["0.72", "#b91c1c"], width=0.5)
    for x, v, raw in zip([0, 1], [gapG, gapD], [1.310, 3.362]):
        axL.text(x, v + 0.07, f"{v:.2f}", ha="center", fontsize=11)
        axL.text(x, 0.12, f"loss {raw}", ha="center", fontsize=9, color="white")
    axL.set_xticks([0, 1]); axL.set_xticklabels(["GAN", "DCGAN"], fontsize=11)
    axL.set_ylim(0, 3.3)
    axL.set_ylabel("how far the drawer's loss sits\nfrom the balance point",
                   fontsize=10)
    axL.set_title("the report card  (DCGAN looks worse)", fontsize=11,
                  color="#b91c1c")
    axL.spines[["top", "right"]].set_visible(False)

    x = np.arange(2)
    axR.bar(x - 0.19, [0.945, 0.953], 0.37, color="0.72", label="GAN")
    axR.bar(x + 0.19, [0.970, 1.085], 0.37, color="tab:green", label="DCGAN")
    for xi, (a, b) in zip(x, [(0.945, 0.970), (0.953, 1.085)]):
        axR.text(xi - 0.19, a + 0.022, f"{a:.3f}", ha="center", fontsize=9.5)
        axR.text(xi + 0.19, b + 0.022, f"{b:.3f}", ha="center", fontsize=9.5)
    axR.set_xticks(x)
    axR.set_xticklabels(["all ten made\nequally often", "as varied as\nreal handwriting"],
                        fontsize=10)
    axR.set_ylim(0, 1.32)
    axR.legend(fontsize=9.5, frameon=False, loc="upper left")
    axR.set_title("the pictures  (DCGAN is better)", fontsize=11,
                  color="tab:green")
    axR.spines[["top", "right"]].set_visible(False)

    fig.subplots_adjust(left=0.095, right=0.98, top=0.86, bottom=0.15,
                        wspace=0.3)
    fig.savefig("lead_scoreboard_lies.svg", transparent=True)
    plt.close(fig); print("wrote lead_scoreboard_lies.svg")


def fig_lead_price_in_time():
    """값이 품질이 아니라 시간으로 옮겨 간다.

    가로축은 **그림 한 장에 그물을 몇 번 지나는가**다. 이 수는 5장의 표에
    그대로 있다(1, 10, 1,000). 초는 쪽이 적어 둔 자리에만 붙인다 —
    한 번짜리 모델의 초는 이 책이 재지 않았으므로 지어내지 않는다.
    """
    names = ["GAN / DCGAN", "DDIM, 10 steps", "DDPM, 1,000 steps"]
    passes = [1, 10, 1000]
    secs = [None, 26, 2495]
    cols = ["0.72", "tab:green", "#b91c1c"]
    y = [2, 1, 0]                       # 위에서부터 GAN, DDIM, DDPM

    fig, ax = plt.subplots(figsize=(9.4, 3.8))
    ax.barh(y, passes, color=cols, height=0.52)
    for yy, pz, sc in zip(y, passes, secs):
        txt = f"{pz:,} pass" + ("es" if pz > 1 else "")
        if sc is not None:
            txt += f"   —   {sc:,} s" + ("  (42 min)" if sc > 1000 else "")
        ax.text(pz * 1.35, yy, txt, va="center", fontsize=10.5,
                color="#b91c1c" if pz == 1000 else "0.3")
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=10.5)
    ax.set_xscale("log"); ax.set_xlim(0.8, 250000)
    ax.set_xlabel("times through the network for one picture", fontsize=10)
    ax.text(1.0, -0.72, "the one-pass models are not timed in seconds on "
            "this page, so none is shown for them",
            fontsize=8.8, color="0.5")
    ax.set_ylim(-0.95, 2.6)
    ax.set_title("the bill moved — from how good to how long", fontsize=11.5)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.subplots_adjust(left=0.20, right=0.97, top=0.86, bottom=0.23)
    fig.savefig("lead_price_in_time.svg", transparent=True)
    plt.close(fig); print("wrote lead_price_in_time.svg")


def fig_lead_never_a_digit(t_show=400):
    """그물은 숫자를 그리라는 말을 들은 적이 없다.

    진짜 MNIST 한 장에 이 쪽의 일정으로 잡음을 섞어 t 시각 그림을 만들고,
    거기 섞인 잡음을 나란히 둔다. 그물이 보는 것도 맞혀야 하는 것도 숫자가
    아니다. 씨앗을 박아 두어 돌릴 때마다 같은 그림이 나온다.
    """
    T = 1000
    ab = np.cumprod(1 - np.linspace(1e-4, 0.02, T))
    tf = transforms.ToTensor()
    te = torchvision.datasets.MNIST(DATA, train=False, download=True,
                                    transform=tf)
    from torch.utils.data import DataLoader
    X = torch.cat([x for x, _ in DataLoader(te, batch_size=2000)])
    x0 = (X[0, 0] * 2 - 1).numpy()
    rng = np.random.default_rng(0)
    eps = rng.normal(size=x0.shape)
    a = ab[t_show - 1]
    xt = np.sqrt(a) * x0 + np.sqrt(1 - a) * eps

    fig, axes = plt.subplots(1, 4, figsize=(10.4, 3.2))
    panels = [(x0, "the real digit\n(the net never sees this)", "0.45"),
              (xt, f"what the net is shown\n(step {t_show} of 1,000)", "0.2"),
              (eps, "what it must name\n(the noise that was added)", "#c2410c"),
              (None, None, None)]
    for ax, (im, ttl, c) in zip(axes, panels):
        if im is None:
            ax.axis("off")
            ax.text(0.5, 0.52, "neither of the two\npictures it works with\n"
                    "is a digit", ha="center", va="center", fontsize=11,
                    color="#c2410c", transform=ax.transAxes)
            continue
        ax.imshow(im, cmap="gray_r")
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color(c); sp.set_linewidth(1.6 if c != "0.45" else 0.8)
        ax.set_title(ttl, fontsize=9.8, color=c)
    fig.suptitle("the machine is never once asked to draw a digit",
                 fontsize=12, y=0.975)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.70, bottom=0.03,
                        wspace=0.12)
    fig.savefig("lead_never_a_digit.svg", transparent=True)
    plt.close(fig); print("wrote lead_never_a_digit.svg")


def fig_lead_96x():
    """같은 웨이트로 뽑는 법만 바꾼다. 02_ddim 3절 표."""
    steps = [1000, 100, 50, 20, 10]
    secs = [2495, 290, 158, 69, 26]
    conf = [0.940, 0.924, 0.923, 0.916, 0.904]
    fig, ax = plt.subplots(figsize=(8.8, 4.2))
    ax.plot(secs, conf, "o-", color="#c2410c", lw=2.2, ms=8)
    for s_, c_, st in zip(secs, conf, steps):
        ax.annotate(f"{st} steps", xy=(s_, c_), xytext=(0, 11),
                    textcoords="offset points", ha="center", fontsize=9.5,
                    color="0.35")
        ax.annotate(f"{c_:.3f}", xy=(s_, c_), xytext=(0, -17),
                    textcoords="offset points", ha="center", fontsize=9.5,
                    color="#c2410c")
    ax.annotate("", xy=(26, 0.884), xytext=(2495, 0.884),
                arrowprops=dict(arrowstyle="<->", color="0.45", lw=1.2))
    ax.text(255, 0.879, "96x faster, and 0.036 of confidence given up",
            ha="center", va="top", fontsize=10, color="0.35")
    ax.set_xscale("log")
    ax.set_xlabel("seconds to draw 5,000 pictures", fontsize=10)
    ax.set_ylabel("does it still look like a digit?", fontsize=10)
    ax.set_ylim(0.868, 0.955)
    ax.set_title("same weights, never retrained — only the way of drawing changed",
                 fontsize=11.5)
    ax.spines[["top", "right"]].set_visible(False)
    fig.subplots_adjust(left=0.10, right=0.975, top=0.88, bottom=0.14)
    fig.savefig("lead_96x.svg", transparent=True)
    plt.close(fig); print("wrote lead_96x.svg")


def fig_linear_ae_plane():
    """선형 오토인코더가 찾는 것은 초평면 하나다.

    로스가 "되살린 것과의 어긋남" 이므로, 활성화 없는 두 레이어가 하는 일은
    점을 k 차원 아핀 부분공간에 **수직으로 내리는 것** 이다. 그 부분공간을
    정하는 것이 (A, b, A', b') 넷이고, 학습은 빨간 선이 가장 짧아지도록 그
    판을 돌리고 옮기는 일이다.

    보여야 할 것은 x, x', x'' 가 서로 다르다는 점이다.
      x   원래 점. 784 개의 수.
      x'  판 위의 어디인가. 64 개의 수. 인코더가 내놓는다.
      x'' 그 자리의 점. 다시 784 개지만 판 위에 있으므로 자유로운 것은 64 개.
          디코더가 내놓는다.

    수는 없다 - 얼개만 보이는 그림이라 그림 안에 그렇게 적는다.
    그림 규칙(CLAUDE.md): SVG, svg.fonttype='path', transparent=True, ASCII.
    """
    rng = np.random.default_rng(3)
    base = np.array([1.0, 0.34]); base /= np.hypot(*base)   # 판의 방향
    perp = np.array([-base[1], base[0]])
    org = np.array([-1.2, 0.30])                            # 판 위의 한 점 = b

    n = 62
    s = rng.normal(0, 1.45, n)
    off = rng.normal(0, 0.34, n)
    pts = org + s[:, None] * base + off[:, None] * perp

    fig, ax = plt.subplots(figsize=(9.2, 4.4))
    L = np.array([-3.5, 4.0])
    line = org + L[:, None] * base
    ax.plot(line[:, 0], line[:, 1], "-", color="#1d4ed8", lw=2.4, zorder=2)

    for p, si in zip(pts, s):                      # 모든 점에서 내린 수선
        f = org + si * base
        ax.plot([p[0], f[0]], [p[1], f[1]], "-", color="#b91c1c", lw=0.7,
                alpha=0.45, zorder=1)
    ax.plot(pts[:, 0], pts[:, 1], "o", ms=3.4, color="0.45", zorder=3)

    # 가운데쯤에서 수선이 긴 점 하나를 골라 크게 그린다
    # b 에서 충분히 떨어진 점을 골라야 x' 화살표가 길이를 갖는다
    ok = np.where((s > 1.3) & (s < 2.3) & (off > 0.35))[0]
    i = ok[np.argmax(off[ok])]
    p, si = pts[i], s[i]
    foot = org + si * base
    ax.plot([p[0], foot[0]], [p[1], foot[1]], "--", color="#b91c1c", lw=2.2,
            zorder=4)
    ax.plot(*p, "o", ms=11, color="#111111", zorder=5)
    ax.plot(*foot, "o", ms=11, color="#15803d", zorder=5)
    ax.plot(*org, "o", ms=7, color="#1d4ed8", zorder=5)

    ax.annotate("x   784 numbers", p, (0, 16), textcoords="offset points",
                ha="center", fontsize=11, color="#111111", fontweight="bold")
    ax.annotate("this is what the loss adds up", ((p + foot) / 2),
                (14, 2), textcoords="offset points", ha="left", fontsize=10.5,
                color="#b91c1c")
    ax.annotate("x''   back in 784,\nbut pinned to the plane", foot,
                (14, -16), textcoords="offset points", ha="left",
                fontsize=10.5, color="#15803d", fontweight="bold")
    ax.annotate("b", org, (-16, 10), textcoords="offset points", ha="center",
                fontsize=11, color="#1d4ed8")

    # x' = 판 위의 좌표. 선 아래로 떨어뜨려 겹치지 않게 그린다
    d = -0.62 * perp
    ax.annotate("", org + d, foot + d, arrowprops=dict(
        arrowstyle="<|-|>", color="#15803d", lw=1.8))
    for q in (org, foot):
        ax.plot([q[0], (q + d)[0]], [q[1], (q + d)[1]], ":", color="#15803d",
                lw=1.0, zorder=1)
    ax.text(*((org + foot) / 2 + 1.55 * d),
            "x'   how far along the plane   64 numbers",
            color="#15803d", fontsize=10.5, ha="center", va="center")

    ax.text(line[1, 0] + 0.15, line[1, 1] - 0.05,
            "the plane the two\nlayers define\n(A, b, A', b')",
            color="#1d4ed8", fontsize=10.5, ha="left", va="center")
    ax.set_title("the linear autoencoder is not learning a curve - "
                 "it is placing one flat plane", fontsize=11.5)
    ax.text(0.5, -0.07,
            "schematic - the plane is 64-dimensional and the points live in "
            "784; no numbers are plotted",
            transform=ax.transAxes, ha="center", fontsize=9.5, color="0.45")
    ax.set_xlim(-4.6, 5.2); ax.set_ylim(-2.0, 2.5)
    ax.set_aspect("equal"); ax.axis("off")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.90, bottom=0.09)
    fig.savefig("linear_ae_plane.svg", transparent=True)
    plt.close(fig); print("wrote linear_ae_plane.svg")


# ============================================================================
# 5.3절 DCGAN 쪽의 그림 넷 — 모두 `mnist_dcgan_g.pt` 하나에서 나온다
# ============================================================================
DC_SEED, DC_ZDIM, DC_NWALK, DC_NSTEP = 42, 64, 12, 16


def _dc_generator():
    """02_dcgan.md 의 GConv 를 그대로 세우고 저장해 둔 웨이트를 읽는다."""
    import torch.nn as nn

    class GConv(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Sequential(nn.Linear(DC_ZDIM, 128 * 7 * 7),
                                    nn.BatchNorm1d(128 * 7 * 7), nn.ReLU())
            self.net = nn.Sequential(
                nn.ConvTranspose2d(128, 64, 4, 2, 1),
                nn.BatchNorm2d(64), nn.ReLU(),
                nn.ConvTranspose2d(64, 1, 4, 2, 1), nn.Tanh())

        def forward(self, z):
            return self.net(self.fc(z).reshape(-1, 128, 7, 7))

    G = GConv()
    G.load_state_dict(torch.load(f"{MODELS}/mnist_dcgan_g.pt",
                                 map_location="cpu", weights_only=True))
    return G.eval()


def _dc_pm1_train():
    """DCGAN 쪽이 쓰는 [-1, 1] MNIST 학습 그림 (N, 784)."""
    tf = transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize((0.5,), (0.5,))])
    tr = torchvision.datasets.MNIST(DATA, train=True, download=True, transform=tf)
    return torch.cat([x.flatten(1) for x, _ in
                      torch.utils.data.DataLoader(tr, batch_size=2000)])


@torch.no_grad()
def fig_dcgan_walk():
    """4절 (나) 를 눈으로 — z 둘을 잇는 직선 위를 열여섯 칸 걷는다.

    쪽의 `walk()` 와 **같은 씨앗·같은 절차**로 길 12개를 만든 뒤, 한 걸음이
    가장 크게 튄 길 하나를 골라 그린다. 끊긴다면 그 길에서 끊길 것이기
    때문이다. 아래 막대가 칸 사이의 화소 거리이고, 두 가로선은 견줄 자리다
    — 한 걸음의 평균과, 서로 상관없는 두 그림 사이의 거리.
    """
    G = _dc_generator()
    g = torch.Generator().manual_seed(DC_SEED)
    a = torch.randn(DC_NWALK, DC_ZDIM, generator=g)
    b = torch.randn(DC_NWALK, DC_ZDIM, generator=g)
    w = torch.linspace(0, 1, DC_NSTEP)[:, None, None]
    z = (1 - w) * a[None] + w * b[None]
    im = G(z.reshape(-1, DC_ZDIM)).reshape(DC_NSTEP, DC_NWALK, 1, 28, 28)
    flat = im.reshape(DC_NSTEP, DC_NWALK, -1)
    step = (flat[1:] - flat[:-1]).pow(2).sum(-1).sqrt()      # (step-1, walk)

    far = G(torch.randn(400, DC_ZDIM,
                        generator=torch.Generator().manual_seed(DC_SEED + 1)))
    far = far.reshape(400, -1)
    indep = (far[:200] - far[200:]).pow(2).sum(-1).sqrt().median().item()

    k = int(step.max(0).values.argmax())                     # 가장 크게 튄 길
    s = step[:, k]

    fig, (axT, axB) = plt.subplots(
        2, 1, figsize=(11.6, 3.5), gridspec_kw={"height_ratios": [1.5, 1.0]})

    canvas = torch.cat([im[t, k, 0] for t in range(DC_NSTEP)], 1)
    axT.imshow((canvas + 1) / 2, cmap="gray", vmin=0, vmax=1)
    axT.set_xticks([]); axT.set_yticks([]); axT.set_anchor("S")
    for sp in axT.spines.values():
        sp.set_visible(False)
    axT.set_title(f"one straight line in z space, {DC_NSTEP} stops "
                  f"(walk {k + 1} of {DC_NWALK} - the one that jumps most)",
                  fontsize=11, pad=5)

    x = np.arange(DC_NSTEP - 1) + 0.5
    axB.bar(x, s.numpy(), width=0.74, color="#2563eb", zorder=3)
    axB.axhline(float(step.mean()), color="#15803d", lw=1.6, ls="--", zorder=4)
    axB.axhline(indep, color="#b91c1c", lw=1.8, zorder=4)
    axB.text(0.12, indep, f"two unrelated samples  {indep:.1f}",
             color="#b91c1c", fontsize=9.5, va="bottom")
    axB.text(0.12, float(step.mean()), f"mean step {step.mean():.2f}",
             color="#15803d", fontsize=9.5, va="bottom")
    axB.set_xlim(0, DC_NSTEP - 1); axB.set_ylim(0, indep * 1.22)
    axB.set_xticks([]); axB.set_ylabel("pixel distance", fontsize=9.5)
    axB.grid(axis="y", alpha=0.25, zorder=0)
    for sp in ("top", "right", "bottom"):
        axB.spines[sp].set_visible(False)

    fig.subplots_adjust(left=0.058, right=0.995, top=0.90, bottom=0.055,
                        hspace=-0.08)
    fig.savefig("dcgan_walk.svg", transparent=True)
    plt.close(fig)
    print(f"wrote dcgan_walk.svg   (walk {k + 1}, max step {s.max():.3f}, "
          f"mean {step.mean():.3f}, indep {indep:.3f})")


@torch.no_grad()
def fig_dcgan_nearest():
    """4절 (가) 를 눈으로 — 베꼈다면 여기서 들통난다.

    뽑은 2,000장 가운데 **학습 그림에 가장 가까운 여덟 장**을 골라, 그 짝이
    되는 학습 그림을 바로 아래 둔다. 가장 의심스러운 여덟을 고른 것이므로,
    여기서 베낀 것이 안 보이면 나머지에서도 안 보인다.
    """
    G = _dc_generator()
    Xtr = _dc_pm1_train()
    g = torch.Generator().manual_seed(DC_SEED)
    fake = G(torch.randn(2000, DC_ZDIM, generator=g)).reshape(2000, -1)

    d = torch.cdist(fake, Xtr)
    dist, idx = d.min(1)
    order = dist.argsort()[:8]

    fig, axes = plt.subplots(2, 8, figsize=(11.2, 3.15))
    for c, i in enumerate(order.tolist()):
        axes[0, c].imshow((fake[i].reshape(28, 28) + 1) / 2, cmap="gray",
                          vmin=0, vmax=1)
        axes[1, c].imshow((Xtr[idx[i]].reshape(28, 28) + 1) / 2, cmap="gray",
                          vmin=0, vmax=1)
        axes[0, c].set_title(f"{dist[i]:.1f}", fontsize=10, pad=3)
        for r in (0, 1):
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
            for sp in axes[r, c].spines.values():
                sp.set_visible(False)
    axes[0, 0].set_ylabel("generated", fontsize=10)
    axes[1, 0].set_ylabel("nearest in\ntraining set", fontsize=10)
    fig.suptitle("the 8 closest generated-to-training pairs out of 2000\n"
                 "number above = pixel distance   "
                 "(median over all 2000 = 10.8;  training image to training "
                 "image = 10.2)", fontsize=10.5, y=0.985)
    fig.subplots_adjust(left=0.10, right=0.99, top=0.73, bottom=0.02,
                        wspace=0.06, hspace=0.10)
    fig.savefig("dcgan_nearest.svg", transparent=True)
    plt.close(fig)
    print(f"wrote dcgan_nearest.svg   (closest {dist.min():.3f}, "
          f"median {dist.median():.3f})")


def _dc_judge():
    """5.1절이 학습해 둔 평가 분류기. [-1,1] 그림을 받아 (확신도, 숫자)."""
    import torch.nn as nn

    class JudgeCNN(nn.Module):
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

    J = JudgeCNN()
    J.load_state_dict(torch.load(f"{MODELS}/mnist_judge.pt", map_location="cpu",
                                 weights_only=True))
    J.eval()
    JM, JS = 0.1307, 0.3081

    @torch.no_grad()
    def read(x_pm1):
        xj = (((x_pm1.reshape(-1, 1, 28, 28) + 1) / 2) - JM) / JS
        p = torch.softmax(J(xj), 1)
        return p.max(1)

    return read


@torch.no_grad()
def fig_dcgan_thickness():
    """논문의 「잠재 공간 벡터 산수」를 이 쪽의 생성기로 다시 해 본다.

    논문은 「안경 쓴 남자」의 z 를 여럿 평균 내고 「안경 안 쓴 남자」의 평균을
    빼서 **안경 벡터**를 만든 뒤, 다른 얼굴의 z 에 더해 안경을 씌웠다. 여기에는
    안경이 없으니 **획의 굵기**로 한다.

    뽑은 5,000장의 잉크 양(화소 평균)을 재어 위 10%의 z 평균에서 아래 10%의
    z 평균을 뺀다. 그것을 **굵기 벡터** v 라 하고, 가운데쯤 굵기의 그림 여섯 개에
    0.5v, 1.0v, 1.5v 를 더해 간다. 굵어지는가, 그리고 **숫자는 그대로인가.**
    """
    G = _dc_generator()
    read = _dc_judge()
    N, ALPHAS = 5000, (0.0, 0.5, 1.0, 1.5)

    z = torch.randn(N, DC_ZDIM, generator=torch.Generator().manual_seed(DC_SEED))
    img = torch.cat([G(z[i:i + 1000]) for i in range(0, N, 1000)])
    ink = ((img + 1) / 2).flatten(1).mean(1)

    k = N // 10
    order = ink.argsort()
    v = z[order[-k:]].mean(0) - z[order[:k]].mean(0)

    conf0, cls0 = read(img)
    rows = []
    for a in ALPHAS:
        im = torch.cat([G(z[i:i + 1000] + a * v) for i in range(0, N, 1000)])
        c, lab = read(im)
        rows.append({"a": a, "img": im,
                     "ink": ((im + 1) / 2).flatten(1).mean(1),
                     "same": (lab == cls0).float().mean().item(),
                     "conf": c.mean().item()})

    # 보일 여섯 장: 가운데쯤 굵기이면서 평가 분류기가 또렷이 읽는 것
    mid = order[N // 2 - 400: N // 2 + 400]
    pick = mid[conf0[mid].argsort(descending=True)[:6]]

    fig, axes = plt.subplots(6, len(ALPHAS), figsize=(5.6, 6.6))
    for r, i in enumerate(pick.tolist()):
        for c, row in enumerate(rows):
            ax = axes[r, c]
            ax.imshow((row["img"][i, 0] + 1) / 2, cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if r == 0:
                ax.set_title(["z", "z+0.5v", "z+1.0v", "z+1.5v"][c], fontsize=10,
                             pad=4)
    fig.suptitle("adding a thickness vector\n"
                 "v = mean z of the inkiest 10%  -  mean z of the thinnest 10%",
                 fontsize=10.5, y=0.99)
    foot = ["over all 5000 samples:"]
    for r in rows:
        foot.append(f"{r['a']:.1f}v    ink {r['ink'].mean():.3f}    "
                    f"same digit as at 0v  {100 * r['same']:.0f}%    "
                    f"judge confidence {r['conf']:.3f}")
    fig.text(0.5, 0.013, "\n".join(foot), ha="center", fontsize=8.4,
             family="monospace")
    fig.subplots_adjust(left=0.02, right=0.98, top=0.875, bottom=0.145,
                        wspace=0.06, hspace=0.06)
    fig.savefig("dcgan_thickness.svg", transparent=True)
    plt.close(fig)
    for r in rows:
        print(f"  a={r['a']:.1f}  ink {r['ink'].mean():.4f}  "
              f"same digit {100 * r['same']:.1f}%  conf {r['conf']:.3f}")
    # 굵기와 숫자가 얽혀 있다는 것을 양 끝의 숫자 구성으로 보인다
    for name, sel in (("thinnest 10%", order[:k]), ("inkiest  10%", order[-k:])):
        c = torch.bincount(cls0[sel], minlength=10).float() / k * 100
        print(f"  {name}: " + "  ".join(f"{d}:{c[d]:.0f}%" for d in range(10)
                                        if c[d] >= 3))
    print("wrote dcgan_thickness.svg")


def fig_dcgan_init_arms():
    """논문의 N(0, 0.02) 초기화를 넣고 뺀 세 갈래, 자마다 시드 다섯 벌.

    수는 `experiments/ch05_dcgan_init.py` 가 남긴 JSON 에서 읽는다. 세 갈래를
    **같은 기계에서 한 번에** 돌렸으므로 갈래끼리는 견줄 수 있고, 다른 쪽의
    다섯 벌 범위와는 견주지 않는다.
    """
    import json
    path = os.path.join(MODELS, "experiments", "ch05_dcgan_init.json")
    res = json.loads(open(path).read())

    arms = ["none", "conv", "all"]
    label = {"none": "no init\n(the page)", "conv": "paper's init\n(conv + BN)",
             "all": "+ Linear too\n(not in the paper)"}
    cols = {"none": "#64748b", "conv": "#2563eb", "all": "#b45309"}
    panels = [("conf", "sample confidence"),
              ("balance", "class balance"),
              ("within", "within-class variety")]

    fig, axes = plt.subplots(1, 3, figsize=(11.4, 2.9))
    for ax, (key, title) in zip(axes, panels):
        for i, arm in enumerate(arms):
            v = np.array([r[key] for r in res[arm]])
            y = 2 - i
            ax.plot([v.min(), v.max()], [y, y], lw=7, solid_capstyle="round",
                    color=cols[arm], alpha=0.32, zorder=2)
            ax.plot(v, [y] * len(v), "o", ms=4.6, color=cols[arm], zorder=3)
            ax.plot([v.mean()], [y], "|", ms=16, mew=2.1, color=cols[arm],
                    zorder=4)
        ax.set_yticks([2, 1, 0])
        ax.set_yticklabels([label[a] for a in arms] if key == "conf" else [],
                           fontsize=9)
        ax.set_ylim(-0.6, 2.6)
        ax.set_title(title, fontsize=11, pad=6)
        ax.grid(axis="x", alpha=0.25)
        ax.tick_params(axis="x", labelsize=9)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)

    axes[1].set_xlabel("the only ruler where the three ranges never touch",
                       fontsize=9.5, labelpad=4)
    fig.suptitle("five training seeds per arm, all measured on one machine "
                 "(bar = range, dots = seeds, tick = mean)", fontsize=10.5,
                 y=0.985)
    fig.subplots_adjust(left=0.135, right=0.985, top=0.70, bottom=0.17,
                        wspace=0.12)
    fig.savefig("dcgan_init_arms.svg", transparent=True)
    plt.close(fig)
    print("wrote dcgan_init_arms.svg")


def fig_dcgan_shapes():
    """널리 쓰이는 64x64 DCGAN 생성기와 이 쪽의 28x28 생성기를 나란히.

    칸의 넓이가 그 자리에서 들고 있는 수의 개수(채널 x 높이 x 너비)에 비례하게
    그렸다. 두 기둥의 눈금이 같으므로 **이 쪽이 얼마나 작은지**가 바로 보인다.
    """
    left = [("z", 100, 1, 1), ("ConvT 4x4 s1", 1024, 4, 4),
            ("ConvT 4x4 s2", 512, 8, 8), ("ConvT 4x4 s2", 256, 16, 16),
            ("ConvT 4x4 s2", 128, 32, 32), ("ConvT 4x4 s2 + tanh", 3, 64, 64)]
    right = [("z", 64, 1, 1), ("Linear + reshape", 128, 7, 7),
             ("ConvT 4x4 s2", 64, 14, 14), ("ConvT 4x4 s2 + tanh", 1, 28, 28)]
    titles = ["the usual 64x64 DCGAN", "this page  (28x28 MNIST)"]

    big = max(c * h * w for _, c, h, w in left)
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.3))
    for ax, rows, title in zip(axes, (left, right), titles):
        for i, (name, c, h, w) in enumerate(rows):
            n = c * h * w
            frac = (n / big) ** 0.5
            y = len(left) - 1 - i
            ax.add_patch(Rectangle((0.5 - frac / 2, y - 0.3), frac, 0.6,
                                   fc="#2563eb" if rows is left else "#b45309",
                                   alpha=0.3, ec="none"))
            ax.text(0.5, y, f"{c} x {h} x {w}", ha="center", va="center",
                    fontsize=10.5)
            ax.text(-0.02, y, name, ha="right", va="center", fontsize=9.5)
            ax.text(1.02, y, f"{n:,}", ha="left", va="center", fontsize=9,
                    color="#555")
        ax.set_xlim(-0.52, 1.30); ax.set_ylim(-0.8, len(left) - 0.4)
        ax.axis("off")
        ax.set_title(title, fontsize=11.5, pad=8)
        ax.text(0.5, -0.65, f"output = {rows[-1][1] * rows[-1][2] * rows[-1][3]:,}"
                " numbers", ha="center", fontsize=9.5, color="#555")
    fig.text(0.5, 0.025, "box width is proportional to sqrt(numbers held at "
             "that point);  both columns use the same scale", ha="center",
             fontsize=9, color="#555")
    fig.subplots_adjust(left=0.10, right=0.97, top=0.88, bottom=0.10, wspace=0.42)
    fig.savefig("dcgan_shapes.svg", transparent=True)
    plt.close(fig)
    print("wrote dcgan_shapes.svg")


def _judge_cpu():
    """5.1절의 평가 분류기. [0,1] 그림을 받아 (확신도, 읽은 숫자)."""
    import torch.nn as nn

    class JudgeCNN(nn.Module):
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

    J = JudgeCNN()
    J.load_state_dict(torch.load(f"{MODELS}/mnist_judge.pt", map_location="cpu",
                                 weights_only=True))
    J.eval()

    @torch.no_grad()
    def read(x01):
        p = torch.softmax(J((x01.reshape(-1, 1, 28, 28) - MEAN) / STD), 1)
        return p.max(1)

    return read


@torch.no_grad()
def fig_judge_sees_three():
    """「왜 하필 3인가」 — 두 가설을 각각 잰다.

    왼쪽: 오토인코더를 아예 빼고, **애매한 그림**을 평가 분류기에 직접 보인다.
    잡음과 섞은 그림에서도 3이 1등이면 3은 모델이 만든 것이 아니라 자의 버릇이다.

    오른쪽: 숫자마다 잠재 표현의 가운데가 구름 전체의 가운데에서 얼마나
    떨어져 있는지. 3이 가장 가깝다면 「가운데에 있어서 많이 나온다」가 된다.
    """
    import torch.nn as nn
    read = _judge_cpu()

    tf01 = transforms.ToTensor()
    tr01 = torchvision.datasets.MNIST(DATA, train=True, download=True,
                                      transform=tf01)
    X01 = torch.cat([x for x, _ in
                     torch.utils.data.DataLoader(tr01, batch_size=2000)])
    g = torch.Generator().manual_seed(0)
    N = 5000

    def perm():
        return X01[torch.randperm(len(X01), generator=g)[:N]]

    cases = [("gaussian noise", (torch.randn(N, 1, 28, 28, generator=g) * STD
                                 + MEAN).clamp(0, 1)),
             ("uniform noise", torch.rand(N, 1, 28, 28, generator=g)),
             ("5 real digits\naveraged",
              torch.stack([perm() for _ in range(5)]).mean(0)),
             ("2 real digits\nblended", 0.5 * perm() + 0.5 * perm())]

    # 견줄 자리: AE_MLP-64 가 맞춘 정규분포에서 뽑은 그림 (5.1절 마지막 쪽)
    sd = torch.load(f"{MODELS}/mnist_ae_mlp64.pt", map_location="cpu",
                    weights_only=True)
    enc = nn.Sequential(nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 64))
    dec = nn.Sequential(nn.Linear(64, 256), nn.ReLU(), nn.Linear(256, 784))
    for m, k in ((enc[0], "enc.0"), (enc[2], "enc.2"),
                 (dec[0], "dec.0"), (dec[2], "dec.2")):
        m.weight.data = sd[f"{k}.weight"]; m.bias.data = sd[f"{k}.bias"]
    (Xtr, _), (Xte, yte) = load_flat()
    z = enc(Xte)
    zs = torch.randn(N, 64, generator=torch.Generator().manual_seed(0))
    zs = zs * z.std(0) + z.mean(0)
    ae = (dec(zs) * STD + MEAN).clamp(0, 1)
    cases.append(("AE-64 samples\n(5.1, for comparison)", ae))

    shares, confs = [], []
    for _, x in cases:
        c, lab = read(x)
        shares.append((torch.bincount(lab, minlength=10).float() / N * 100).numpy())
        confs.append(float(c.mean()))

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.6, 3.9),
                                   gridspec_kw={"width_ratios": [1.65, 1.0]})
    wd = 0.16
    xs = np.arange(10)
    cols = ["#64748b", "#94a3b8", "#2563eb", "#60a5fa", "#b45309"]
    for i, ((name, _), sh) in enumerate(zip(cases, shares)):
        axL.bar(xs + (i - 2) * wd, sh, width=wd * 0.92, color=cols[i],
                label=f"{name}  (conf {confs[i]:.2f})", zorder=3)
    axL.axhline(10, color="#111", lw=1.0, ls=":", zorder=4)
    axL.text(9.42, 11, "10% = even", fontsize=8.6, ha="right")
    axL.add_patch(Rectangle((3 - 0.5, 0), 1.0, 70, fc="#fde68a", alpha=0.45,
                            zorder=0))
    axL.set_xticks(xs); axL.set_xlabel("digit the judge reads", fontsize=10)
    axL.set_ylabel("% of 5000", fontsize=10)
    axL.set_title("ask the judge about images no autoencoder made",
                  fontsize=11.5, pad=6)
    axL.legend(fontsize=8.4, frameon=False, ncol=1, loc="upper right")
    axL.grid(axis="y", alpha=0.25, zorder=0)
    for sp in ("top", "right"):
        axL.spines[sp].set_visible(False)

    c0 = z.mean(0)
    dist = [( (z[yte == k].mean(0) - c0).norm().item(), k) for k in range(10)]
    dist.sort()
    axR.barh([str(k) for _, k in dist][::-1], [d for d, _ in dist][::-1],
             color=["#b45309" if k == 3 else "#cbd5e1" for _, k in dist][::-1],
             zorder=3)
    axR.set_xlabel("distance from the centre of the latent cloud", fontsize=10)
    axR.set_ylabel("digit", fontsize=10)
    axR.set_title("AE-64: is 3 the one in the middle?  (no)", fontsize=11.5,
                  pad=6)
    axR.grid(axis="x", alpha=0.25, zorder=0)
    for sp in ("top", "right"):
        axR.spines[sp].set_visible(False)

    fig.subplots_adjust(left=0.055, right=0.985, top=0.86, bottom=0.145,
                        wspace=0.22)
    fig.savefig("judge_sees_three.svg", transparent=True)
    plt.close(fig)
    for (name, _), sh, c in zip(cases, shares, confs):
        print(f"  {name.replace(chr(10),' '):34s} top {int(sh.argmax())} "
              f"({sh.max():.0f}%)  3 is {sh[3]:.0f}%  conf {c:.3f}")
    print("  centre distance: " + "  ".join(f"{k}:{d:.2f}" for d, k in dist))
    print("wrote judge_sees_three.svg")


@torch.no_grad()
def fig_point_vs_blob():
    """강의 칠판의 두 원을 재어 본다 — 끌려간 것은 자리이지 크기가 아니다.

    AE 는 잠재 표현을 점으로 내놓고 VAE 는 분포로 내놓으므로, 흔히 VAE 쪽을
    「작은 원이 N(0, I) 라는 큰 원 위로 끌려오는 그림」으로 그린다. 잠재 차원
    2 인 두 모델로 그 그림을 실제 수로 그려 보면 **왼쪽은 맞고 오른쪽은 다르다.**

    왼쪽: 두 구름을 같은 눈금에 겹쳐 놓는다. KL 항이 구름을 사전분포 쪽으로
    당긴 것이 보인다.
    오른쪽: 그러면 그 「원」의 반지름은 얼마인가. 표본마다의 sigma 를 모두
    모아 히스토그램으로 그린다. 1 이 아니라 0.03 언저리다.
    """
    import torch.nn as nn
    from matplotlib.patches import Circle

    (Xtr, ytr), (Xte, yte) = load_flat()

    ae = torch.load(f"{MODELS}/mnist_ae_mlp2.pt", map_location="cpu",
                    weights_only=True)
    enc_ae = nn.Sequential(nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 2))
    for m, k in ((enc_ae[0], "enc.0"), (enc_ae[2], "enc.2")):
        m.weight.data = ae[f"{k}.weight"]; m.bias.data = ae[f"{k}.bias"]

    va = torch.load(f"{MODELS}/mnist_vae2_beta1.0.pt", map_location="cpu",
                    weights_only=True)
    trunk = nn.Sequential(nn.Linear(784, 256), nn.ReLU())
    trunk[0].weight.data = va["enc.0.weight"]
    trunk[0].bias.data = va["enc.0.bias"]
    mu_l, lv_l = nn.Linear(256, 2), nn.Linear(256, 2)
    mu_l.weight.data = va["mu.weight"]; mu_l.bias.data = va["mu.bias"]
    lv_l.weight.data = va["logvar.weight"]; lv_l.bias.data = va["logvar.bias"]

    z_ae = enc_ae(Xte)
    h = trunk(Xte)
    mu, lv = mu_l(h), lv_l(h)
    sd = (0.5 * lv).exp()

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.0, 4.6),
                                   gridspec_kw={"width_ratios": [1.0, 1.05]})

    lim = 16.0
    axL.set_aspect("equal"); axL.set_xlim(-lim, lim); axL.set_ylim(-lim, lim)
    # 점을 전부 그리면 SVG 가 3 MB 가 된다. 퍼짐은 아래 글자로 적으므로
    # 보이는 데 필요한 만큼만 솎아 그린다.
    sub = torch.randperm(len(z_ae),
                         generator=torch.Generator().manual_seed(0))[:2500]
    axL.plot(z_ae[sub, 0], z_ae[sub, 1], ".", ms=1.6, color="#f97316",
             alpha=0.45, zorder=2,
             label=f"AE-2      (coord std {z_ae.std(0).mean():.2f})")
    axL.plot(mu[sub, 0], mu[sub, 1], ".", ms=1.6, color="#2563eb", alpha=0.5,
             zorder=3, label=f"VAE-2 mu  (coord std {mu.std(0).mean():.2f})")
    for r, a_ in ((1, 0.8), (2, 0.5)):
        axL.add_patch(Circle((0, 0), r, fill=False, ls=":", lw=1.6, ec="#111",
                             alpha=a_, zorder=5))
    axL.annotate("N(0, I)\n1 and 2 sigma", (1.7, 1.7), xytext=(6.0, 9.0),
                 fontsize=9.5, arrowprops=dict(arrowstyle="->", lw=1.0))
    axL.legend(fontsize=9, frameon=False, loc="lower left", markerscale=7)
    axL.set_title("the KL term pulls the cloud onto the prior", fontsize=11.5,
                  pad=7)
    axL.text(lim - 0.4, lim - 0.6, "2500 of the 10000\ntest images drawn",
             ha="right", va="top", fontsize=8.6, color="#555")
    axL.grid(alpha=0.2)
    for sp in ("top", "right"):
        axL.spines[sp].set_visible(False)

    axR.hist(sd.flatten().numpy(), bins=80, color="#2563eb", alpha=0.75,
             zorder=3)
    top = axR.get_ylim()[1]
    axR.axvline(1.0, color="#b91c1c", lw=1.8, zorder=4)
    axR.text(0.98, top * 0.95, "sigma = 1, the size the\n"
             "blackboard picture draws  ", color="#b91c1c", fontsize=9.5,
             va="top", ha="right")
    axR.axvline(float(sd.mean()), color="#15803d", lw=1.6, ls="--", zorder=4)
    axR.text(float(sd.mean()) + 0.03, top * 0.55, f"mean {sd.mean():.3f}",
             color="#15803d", fontsize=9.5)
    axR.set_xlim(0, 1.25)
    axR.set_xlabel("sigma of q(z|x), every test image and both coordinates",
                   fontsize=10)
    axR.set_ylabel("count", fontsize=10)
    axR.set_title("but the blob itself is almost a point", fontsize=11.5, pad=7)
    axR.grid(axis="y", alpha=0.25, zorder=0)
    for sp in ("top", "right"):
        axR.spines[sp].set_visible(False)

    fig.subplots_adjust(left=0.06, right=0.985, top=0.87, bottom=0.125,
                        wspace=0.22)
    fig.savefig("point_vs_blob.svg", transparent=True)
    plt.close(fig)
    print(f"wrote point_vs_blob.svg   AE coord std {z_ae.std(0).mean():.2f} "
          f"range [{z_ae.min():.1f}, {z_ae.max():.1f}] | VAE mu std "
          f"{mu.std(0).mean():.2f} range [{mu.min():.1f}, {mu.max():.1f}] | "
          f"sigma mean {sd.mean():.3f} max {sd.max():.3f} "
          f"above 0.5: {100 * (sd > 0.5).float().mean():.1f}%")


def fig_dcgan_recipe_gains():
    """요령을 하나씩 더할 때 클래스 균형도가 어디까지 가는가.

    두 기계의 수가 섞이지 않게 **출처를 칸 안에 적는다.** 가운데 두 줄이
    같은 모델(DCGAN, 초기화 없음)을 서로 다른 기계에서 다섯 벌씩 잰 것인데
    범위가 거의 겹치므로, 이 자에서는 기계를 건너 읽어도 된다는 뜻이다.
    """
    import json
    res = json.loads(open(os.path.join(MODELS, "experiments",
                                       "ch05_dcgan_init.json")).read())
    mine = {k: np.array([r["balance"] for r in res[k]]) for k in res}

    rows = [("GAN (MLP)", (0.913, 0.937), None, "5.3 page", "#94a3b8"),
            ("DCGAN", (0.966, 0.978), None, "5.3 page", "#2563eb"),
            ("DCGAN", (mine["none"].min(), mine["none"].max()), mine["none"],
             "re-run here", "#2563eb"),
            ("+ paper's init", (mine["conv"].min(), mine["conv"].max()),
             mine["conv"], "re-run here", "#15803d"),
            ("+ Linear too", (mine["all"].min(), mine["all"].max()),
             mine["all"], "re-run here", "#b45309")]

    fig, ax = plt.subplots(figsize=(10.6, 3.6))
    for i, (name, (lo, hi), pts, src, col) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.plot([lo, hi], [y, y], lw=9, solid_capstyle="round", color=col,
                alpha=0.30, zorder=2)
        if pts is not None:
            ax.plot(pts, [y] * len(pts), "o", ms=4.4, color=col, zorder=3)
        ax.plot([(lo + hi) / 2], [y], "|", ms=19, mew=2.2, color=col, zorder=4)
        ax.text(0.9075, y, name, ha="left", va="center", fontsize=10.5)
        ax.text(1.0045, y, src, ha="right", va="center", fontsize=8.6,
                color="#666", style="italic")
    ax.annotate("", xy=(0.9648, 2.3), xytext=(0.9648, 1.7),
                arrowprops=dict(arrowstyle="<->", lw=1.1, color="#555"))
    ax.text(0.9638, 2.0, "same model,\ntwo machines  ", ha="right",
            va="center", fontsize=8.8, color="#555")
    ax.set_xlim(0.9070, 1.0055); ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.set_yticks([])
    ax.set_xlabel("class balance of 5000 samples   "
                  "(bar = range over five training seeds)", fontsize=10)
    ax.set_title("what each step of the recipe bought, measured",
                 fontsize=12, pad=8)
    ax.grid(axis="x", alpha=0.25)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    fig.subplots_adjust(left=0.025, right=0.985, top=0.855, bottom=0.165)
    fig.savefig("dcgan_recipe_gains.svg", transparent=True)
    plt.close(fig)
    print("wrote dcgan_recipe_gains.svg")


# ============================================================================
# 64x64 얼굴 DCGAN — experiments/ch05_dcgan_faces.py 가 남긴 lfw_dcgan_g.pt 를 읽는다.
# 그 파일은 저장소에 담지 않으므로(뿌리의 *.pt 는 .gitignore 에 있다) 먼저
# 그 스크립트를 돌려야 한다.
# ============================================================================
LFW_Z = 100


def _lfw_generator():
    import torch.nn as nn

    def up(i, o):
        return [nn.ConvTranspose2d(i, o, 4, 2, 1, bias=False),
                nn.BatchNorm2d(o), nn.ReLU(True)]

    net = nn.Sequential(
        nn.ConvTranspose2d(LFW_Z, 512, 4, 1, 0, bias=False),
        nn.BatchNorm2d(512), nn.ReLU(True),
        *up(512, 256), *up(256, 128), *up(128, 64),
        nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False), nn.Tanh())

    class G64(nn.Module):
        def __init__(self):
            super().__init__(); self.net = net

        def forward(self, z):
            return self.net(z.reshape(-1, LFW_Z, 1, 1))

    G = G64()
    G.load_state_dict(torch.load(f"{MODELS}/lfw_dcgan_g.pt", map_location="cpu",
                                 weights_only=True))
    return G.eval()


def _rgb(x):
    """(N,3,64,64) in [-1,1] -> (N,64,64,3) in [0,1]."""
    return ((x + 1) / 2).clamp(0, 1).permute(0, 2, 3, 1).numpy()


@torch.no_grad()
def fig_lfw_samples():
    """얼굴 64장. 28x28 흑백에서만 되는 이야기가 아니라는 것을 보인다."""
    G = _lfw_generator()
    z = torch.randn(64, LFW_Z, generator=torch.Generator().manual_seed(7))
    im = _rgb(torch.cat([G(z[i:i + 32]) for i in range(0, 64, 32)]))
    fig, axes = plt.subplots(8, 8, figsize=(6.6, 6.9))
    for i, ax in enumerate(axes.flat):
        ax.imshow(im[i]); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
    fig.suptitle("DCGAN on LFW faces, 64x64 colour\n"
                 "same recipe, same loss, same optimiser as the MNIST page",
                 fontsize=10.5, y=0.985)
    fig.subplots_adjust(left=0.004, right=0.996, top=0.905, bottom=0.004,
                        wspace=0.04, hspace=0.04)
    fig.savefig("lfw_samples.svg", transparent=True)
    plt.close(fig)
    print("wrote lfw_samples.svg")


@torch.no_grad()
def fig_lfw_walk():
    """잠재 공간의 직선 위를 걷는다 — 얼굴에서도 끊기지 않는가."""
    G = _lfw_generator()
    NW, NS = 4, 10
    g = torch.Generator().manual_seed(3)
    a = torch.randn(NW, LFW_Z, generator=g)
    b = torch.randn(NW, LFW_Z, generator=g)
    w = torch.linspace(0, 1, NS)[:, None, None]
    z = ((1 - w) * a[None] + w * b[None]).reshape(-1, LFW_Z)
    im = _rgb(torch.cat([G(z[i:i + 20]) for i in range(0, len(z), 20)]))
    im = im.reshape(NS, NW, 64, 64, 3)

    flat = torch.from_numpy(im).reshape(NS, NW, -1)
    step = (flat[1:] - flat[:-1]).pow(2).sum(-1).sqrt()

    fig, axes = plt.subplots(NW, NS, figsize=(10.4, 4.5))
    for r in range(NW):
        for c in range(NS):
            axes[r, c].imshow(im[c, r])
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
            for sp in axes[r, c].spines.values():
                sp.set_visible(False)
    fig.suptitle(f"four straight lines in z space, {NS} stops each   "
                 f"(mean step {step.mean():.2f}, largest {step.max():.2f})",
                 fontsize=10.5, y=0.975)
    fig.subplots_adjust(left=0.004, right=0.996, top=0.90, bottom=0.004,
                        wspace=0.04, hspace=0.04)
    fig.savefig("lfw_walk.svg", transparent=True)
    plt.close(fig)
    print(f"wrote lfw_walk.svg   mean step {step.mean():.3f} "
          f"max {step.max():.3f}")


@torch.no_grad()
def fig_lfw_vector():
    """논문 6.3.2 의 벡터 산수를 얼굴에서. 라벨이 없으므로 **재어서** 고른다.

    라벨 없이 잴 수 있는 성질 둘을 쓴다.
      bright  그림 전체의 밝기
      pose    오른쪽 절반의 분산 빼기 왼쪽 절반의 분산. 고개를 한쪽으로
              돌리면 한쪽은 매끈한 볼이 되고 다른 쪽에 귀·머리가 들어오므로
              **부호가 있는** 값이 된다. 논문 그림 8 의 「돌리기 벡터」와
              같은 자리를 노린 것이다.

    성질마다 위 10% 의 z 평균에서 아래 10% 의 z 평균을 빼 벡터를 만들고,
    가운데쯤인 얼굴에 더해 가며 **그 성질만 움직이는지** 본다.
    """
    G = _lfw_generator()
    N = 4000
    z = torch.randn(N, LFW_Z, generator=torch.Generator().manual_seed(11))
    im = torch.cat([G(z[i:i + 200]) for i in range(0, N, 200)])
    lum = ((im + 1) / 2).mean(1)                       # (N, 64, 64)

    attrs = {"brightness": lum.flatten(1).mean(1),
             "pose": lum[:, :, 32:].flatten(1).var(1)
                     - lum[:, :, :32].flatten(1).var(1)}

    k = N // 10
    ALPHAS = (-1.5, -0.75, 0.0, 0.75, 1.5)
    fig = plt.figure(figsize=(6.2, 7.8))
    gs = fig.add_gridspec(7, len(ALPHAS), height_ratios=[1, 1, 1, 0.34, 1, 1, 1])
    axes = {}
    for rr in (0, 1, 2, 4, 5, 6):
        for cc in range(len(ALPHAS)):
            axes[(rr, cc)] = fig.add_subplot(gs[rr, cc])
    for ai, (name, s) in enumerate(attrs.items()):
        order = s.argsort()
        v = z[order[-k:]].mean(0) - z[order[:k]].mean(0)
        mid = order[N // 2 - 300: N // 2 + 300]
        pick = mid[torch.randperm(len(mid),
                                  generator=torch.Generator().manual_seed(2))[:3]]
        stats = []
        for al in ALPHAS:
            out = torch.cat([G(z[i:i + 200] + al * v) for i in range(0, N, 200)])
            L = ((out + 1) / 2).mean(1)
            val = (L.flatten(1).mean(1) if ai == 0 else
                   L[:, :, 32:].flatten(1).var(1) - L[:, :, :32].flatten(1).var(1))
            stats.append(float(val.mean()))
        for r, i in enumerate(pick.tolist()):
            row = ai * 4 + r
            for c, al in enumerate(ALPHAS):
                ax = axes[(row, c)]
                ax.imshow(_rgb(G(z[i:i + 1] + al * v))[0])
                ax.set_xticks([]); ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_visible(False)
                if r == 0:
                    ax.set_title(f"{al:+.2f}v".replace("+0.00v", "z"),
                                 fontsize=9, pad=3)
            if r == 1:
                axes[(row, 0)].set_ylabel(name, fontsize=10.5)
        fig.text(0.5, (0.902, 0.433)[ai], f"{name}:   " +
                 "   ".join(f"{a_:+.2f}v {s_:.3f}"
                            for a_, s_ in zip(ALPHAS, stats)),
                 ha="center", fontsize=8.1, family="monospace", color="#444")
        print(f"  {name}: " + "  ".join(f"{a:+.2f}v {s_:.4f}"
                                        for a, s_ in zip(ALPHAS, stats)))
    fig.suptitle("vector arithmetic on faces   "
                 "(v = mean z of the top 10% minus the bottom 10%, 4000 samples;\n"
                 "pose = variance of the right half minus the left half)",
                 fontsize=10, y=0.992)
    fig.subplots_adjust(left=0.085, right=0.995, top=0.838, bottom=0.008,
                        wspace=0.05, hspace=0.06)
    fig.savefig("lfw_vector.svg", transparent=True)
    plt.close(fig)
    print("wrote lfw_vector.svg")


# ============================================================================
# 11/11 강의 필기에 붙인 그림 셋 — 칠판에 그린 것을 재어서 다시 그린다
# ============================================================================
def fig_gan_code_lines():
    """01_gan.md 「코드를 한 줄씩 읽는다」 — 활성화 함수 둘과 Adam 의 beta1.

    왼쪽: ReLU 와 LeakyReLU(0.2). 음수 쪽에 기울기 0.2 가 남는다.
    가운데: tanh 와 시그모이드. tanh(x) 는 2*sigmoid(2x)-1 이고,
            2*sigmoid(x)-1 은 tanh(x/2) 라서 더 누운 곡선이다.
    오른쪽: Adam 이 k 스텝 전 그래디언트에 주는 가중치 (1-b1)*b1^k.
    """
    x = np.linspace(-4, 4, 801)
    sig = lambda v: 1 / (1 + np.exp(-v))
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(12.6, 3.9))

    a1.plot(x, np.maximum(x, 0), color="#111", lw=2.2, label="ReLU")
    a1.plot(x, np.where(x > 0, x, 0.2 * x), color="#dc2626", lw=2.0, ls="--",
            label="LeakyReLU(0.2)")
    a1.annotate("slope 0.2", (-2.5, -0.5), xytext=(-3.7, 1.5), fontsize=10,
                color="#dc2626", arrowprops=dict(arrowstyle="->", color="#dc2626"))
    a1.annotate("slope 0:\nno gradient", (-2.0, 0.0), xytext=(-1.3, 2.6),
                fontsize=10, arrowprops=dict(arrowstyle="->"))
    a1.set_ylim(-1.2, 4.1)
    a1.set_title("hidden layers: what a negative input gets", fontsize=11)
    a1.legend(fontsize=9.5, frameon=False, loc="lower right")

    a2.plot(x, sig(x), color="#6b7280", lw=1.8, label="sigmoid(x), in (0, 1)")
    a2.plot(x, np.tanh(x), color="#2563eb", lw=4.5, alpha=0.35)
    a2.plot(x, 2 * sig(2 * x) - 1, color="#2563eb", lw=1.6, ls="--",
            label="tanh(x) = 2 sigmoid(2x) - 1")
    a2.plot(x, 2 * sig(x) - 1, color="#f97316", lw=2.0,
            label="2 sigmoid(x) - 1 = tanh(x/2)")
    gap = np.abs(np.tanh(x) - (2 * sig(x) - 1)).max()
    a2.set_title(f"output layer: tanh, in (-1, 1)   (max gap {gap:.3f})",
                 fontsize=11)
    a2.set_ylim(-1.12, 1.8)
    a2.legend(fontsize=8.8, frameon=False, loc="upper left")
    a2.axhline(1, color="#bbb", lw=0.8, ls=":"); a2.axhline(-1, color="#bbb", lw=0.8, ls=":")

    ks = np.arange(0, 11)
    for b1, col, off in ((0.9, "#6b7280", -0.2), (0.5, "#dc2626", 0.2)):
        a3.bar(ks + off, (1 - b1) * b1 ** ks, width=0.38, color=col,
               label=f"beta1 = {b1}   (10 or more steps ago: "
                     f"{100 * b1 ** 10:.1f}%)")
    a3.set_ylim(0, 0.64)
    a3.set_xticks(ks)
    a3.set_xlabel("k steps ago", fontsize=10)
    a3.set_title("Adam: weight on the gradient from k steps ago", fontsize=11)
    a3.legend(fontsize=8.8, frameon=False)

    for ax in (a1, a2):
        ax.axhline(0, color="#999", lw=0.8); ax.axvline(0, color="#999", lw=0.8)
        ax.set_xlabel("x", fontsize=10)
    for ax in (a1, a2, a3):
        ax.grid(alpha=0.2)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig.subplots_adjust(left=0.04, right=0.99, top=0.9, bottom=0.14, wspace=0.2)
    fig.savefig("gan_code_lines.svg", transparent=True)
    plt.close(fig)
    print(f"wrote gan_code_lines.svg   max |tanh(x) - (2 sigmoid(x) - 1)| {gap:.3f}")


@torch.no_grad()
def fig_dcgan_bn_modes(n_rows=6):
    """02_dcgan.md 「생성기 코드를 한 줄씩 읽는다」 (다) — 배치 정규화의 두 모드.

    같은 z 를 평가 모드로 한 번, 학습 모드로 두 번(함께 넣는 99개만 바꾸어)
    그린다. 평가 모드는 저장해 둔 running 통계를 쓰므로 z 하나가 그림 하나로
    정해지고, 학습 모드는 그 미니배치의 통계를 쓰므로 이웃이 바뀌면 그림이 바뀐다.
    첫 줄의 z 와 두 묶음은 쪽의 코드가 쓰는 것과 같다(SEED 42).
    """
    import copy
    G = _dc_generator()
    gz = torch.Generator().manual_seed(DC_SEED)
    z0 = torch.randn(1, DC_ZDIM, generator=gz)
    mates = [torch.randn(99, DC_ZDIM, generator=gz) for _ in range(2)]
    zs = torch.cat([z0, torch.randn(n_rows - 1, DC_ZDIM, generator=gz)])

    Gt = copy.deepcopy(G).train()
    cols = ["eval mode\n(running statistics)", "train mode\nwith batch A",
            "train mode\nwith batch B", "|A - B|"]
    fig, axes = plt.subplots(n_rows, 4, figsize=(5.6, 1.3 * n_rows + 0.8))
    diffs = []
    for r in range(n_rows):
        z = zs[r:r + 1]
        ev = G(z)[0, 0]
        a = Gt(torch.cat([z, mates[0]]))[0, 0]
        b = Gt(torch.cat([z, mates[1]]))[0, 0]
        diffs.append((a - b).abs().mean().item())
        for c, img in enumerate((ev, a, b)):
            axes[r, c].imshow(img, cmap="gray", vmin=-1, vmax=1)
        axes[r, 3].imshow((a - b).abs(), cmap="magma", vmin=0, vmax=2)
        axes[r, 3].text(1.06, 0.5, f"mean\n{diffs[-1]:.3f}", fontsize=8.5,
                        transform=axes[r, 3].transAxes, va="center")
        for c in range(4):
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
            if r == 0:
                axes[r, c].set_title(cols[c], fontsize=9, pad=5)
        axes[r, 0].set_ylabel(f"z {r + 1}", fontsize=9)
    fig.suptitle("one z, three pictures: BatchNorm reads the other 99 in the batch",
                 fontsize=10.5, y=0.995)
    fig.subplots_adjust(left=0.07, right=0.88, top=0.9, bottom=0.008,
                        wspace=0.06, hspace=0.06)
    fig.savefig("dcgan_bn_modes.svg", transparent=True)
    plt.close(fig)
    print("wrote dcgan_bn_modes.svg   mean |A - B| per row: "
          + "  ".join(f"{d:.3f}" for d in diffs))


@torch.no_grad()
def fig_cvae_ten_planes():
    """03_cvae.md 「칠판 그림」 — 숫자마다 잠재 평면이 한 장씩인가.

    시험 그림 10,000장의 mu 를 숫자별로 칠한다. 왼쪽은 라벨 없는 VAE-2 로,
    한 평면을 열 숫자가 나눠 쓴다. 오른쪽은 cVAE-2 로, 열 숫자가 모두 같은
    자리(원점 둘레)에 겹쳐 쌓인다 — 칠판의 「평면 열 장」을 한 장에 포개어
    그린 것이다. mnist_cvae2.pt 는 03_cvae.md 의 블록이 저장한다.
    """
    import torch.nn.functional as F
    (_, _), (Xte, yte) = load_flat()

    def mus(path, cond):
        sd = torch.load(f"{MODELS}/{path}", map_location="cpu", weights_only=True)
        x = torch.cat([Xte, F.one_hot(yte, 10).float()], 1) if cond else Xte
        h = torch.relu(x @ sd["enc.0.weight"].T + sd["enc.0.bias"])
        return h @ sd["mu.weight"].T + sd["mu.bias"]

    panels = [("VAE-2: one plane, shared by ten digits",
               mus("mnist_vae2_beta1.0.pt", False)),
              ("cVAE-2: ten planes, stacked on the same spot",
               mus("mnist_cvae2.pt", True))]
    sub = torch.randperm(len(Xte), generator=torch.Generator().manual_seed(0))[:2500]
    cmap = plt.get_cmap("tab10")
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 5.3))
    out = []
    for ax, (title, mu) in zip(axes, panels):
        ax.set_aspect("equal"); ax.set_xlim(-4.2, 4.2); ax.set_ylim(-4.2, 4.2)
        ax.scatter(mu[sub, 0], mu[sub, 1], s=3, c=[cmap(int(d)) for d in yte[sub]],
                   alpha=0.55, linewidths=0, zorder=2)
        cen = torch.stack([mu[yte == d].mean(0) for d in range(10)])
        for d in range(10):
            ax.text(cen[d, 0], cen[d, 1], str(d), fontsize=12, fontweight="bold",
                    ha="center", va="center", color=cmap(d), zorder=5,
                    path_effects=[pe.withStroke(linewidth=3, foreground="white")])
        w = torch.bincount(yte, minlength=10).float() / len(yte)
        share = 100 * ((w[:, None] * (cen - mu.mean(0)) ** 2).sum()
                       / mu.var(0, unbiased=False).sum()).item()
        out.append(share)
        ax.set_title(title, fontsize=11.5, pad=7)
        ax.text(0.5, -0.105, f"digit centres (the numerals) explain {share:.1f}% "
                "of the variance of mu", transform=ax.transAxes, ha="center",
                fontsize=9.5, color="#333")
        ax.grid(alpha=0.2)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig.text(0.5, 0.012, "mu of the 10000 test images (2500 drawn), coloured by "
             "the true digit; cVAE-2 encodes each image together with its label",
             ha="center", fontsize=8.8, color="#555")
    fig.subplots_adjust(left=0.045, right=0.99, top=0.94, bottom=0.14, wspace=0.1)
    fig.savefig("cvae_ten_planes.svg", transparent=True)
    plt.close(fig)
    print(f"wrote cvae_ten_planes.svg   centre share VAE-2 {out[0]:.1f}%  "
          f"cVAE-2 {out[1]:.1f}%")


# ============================================================================
# 칠판 도식 셋 — 강의에서 그리는 흐름도를 이 책의 그림체로 다시 그린다.
# 상자와 화살표는 도식이고, 양 끝의 그림과 잠재 공간의 점은 저장해 둔
# 모델에서 실제로 계산한 것이다.
# ============================================================================
def _bd_box(ax, xy, w, h, text, fc="#eef2ff", ec="#4338ca", fs=11, **kw):
    from matplotlib.patches import FancyBboxPatch
    ax.add_patch(FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=fc, ec=ec, lw=1.4, zorder=3, **kw))
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center",
            fontsize=fs, zorder=4)


def _bd_arrow(ax, a, b, color="#111", lw=1.6, ls="-", rad=0.0, z=2):
    ax.annotate("", xy=b, xytext=a, zorder=z,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, ls=ls,
                                shrinkA=0, shrinkB=0,
                                connectionstyle=f"arc3,rad={rad}"))


def _bd_digit(fig, ax, cx, cy, flat, size=1.0, title=None, pm1=False):
    """자료 좌표 (cx, cy) 를 가운데로 하는 작은 축에 28x28 그림을 넣는다."""
    x0, y0 = ax.transData.transform((cx - size / 2, cy - size / 2))
    x1, y1 = ax.transData.transform((cx + size / 2, cy + size / 2))
    inv = fig.transFigure.inverted()
    (fx0, fy0), (fx1, fy1) = inv.transform((x0, y0)), inv.transform((x1, y1))
    a = fig.add_axes([fx0, fy0, fx1 - fx0, fy1 - fy0])
    img = flat.reshape(28, 28)
    img = (img + 1) / 2 if pm1 else (img * STD + MEAN)
    a.imshow(img.clamp(0, 1), cmap="gray", vmin=0, vmax=1)
    a.set_xticks([]); a.set_yticks([])
    for sp in a.spines.values():
        sp.set_edgecolor("#555")
    if title:
        a.set_title(title, fontsize=8.5, pad=2)
    return a


def _bd_plane(fig, ax, cx, cy, size, lim, pts=(), title=None):
    """잠재 평면 하나를 작은 축으로 그린다. pts 는 (x, y, color, marker, ms)."""
    from matplotlib.patches import Circle
    x0, y0 = ax.transData.transform((cx - size / 2, cy - size / 2))
    x1, y1 = ax.transData.transform((cx + size / 2, cy + size / 2))
    inv = fig.transFigure.inverted()
    (fx0, fy0), (fx1, fy1) = inv.transform((x0, y0)), inv.transform((x1, y1))
    a = fig.add_axes([fx0, fy0, fx1 - fx0, fy1 - fy0])
    a.set_xlim(-lim, lim); a.set_ylim(-lim, lim); a.set_aspect("equal")
    a.axhline(0, color="#999", lw=0.8); a.axvline(0, color="#999", lw=0.8)
    a.set_xticks([]); a.set_yticks([])
    for sp in a.spines.values():
        sp.set_edgecolor("#bbb")
    for (x, y, c, mk, ms) in pts:
        a.plot([x], [y], mk, color=c, ms=ms, zorder=5)
    if title:
        a.set_title(title, fontsize=8.5, pad=2)
    return a


@torch.no_grad()
def fig_board_ae_vae():
    """01_vae.md — 칠판의 AE 와 VAE 흐름도. 같은 시험 그림 7 한 장을 두 모델에 넣는다.

    위: AE-2 는 인코더가 점 하나를 내놓고 디코더가 그 점을 그림으로 되돌린다.
    아래: VAE-2 는 인코더가 mu 와 log sigma^2 을 내놓고, z = mu + sigma * eps 로
    뽑은 점을 디코더에 넣는다. 평면 위의 점과 오른쪽 그림은 모두 저장해 둔
    모델(mnist_ae_mlp2.pt, mnist_vae2_beta1.0.pt)이 실제로 낸 값이다.
    """
    from matplotlib.patches import Circle
    (_, _), (Xte, yte) = load_flat()
    x = Xte[int((yte == 7).nonzero()[0])][None]

    ae = torch.load(f"{MODELS}/mnist_ae_mlp2.pt", map_location="cpu", weights_only=True)
    lin = lambda sd, k, v: v @ sd[f"{k}.weight"].T + sd[f"{k}.bias"]
    z_ae = lin(ae, "enc.2", torch.relu(lin(ae, "enc.0", x)))
    x_ae = lin(ae, "dec.2", torch.relu(lin(ae, "dec.0", z_ae)))

    va = torch.load(f"{MODELS}/mnist_vae2_beta1.0.pt", map_location="cpu", weights_only=True)
    h = torch.relu(lin(va, "enc.0", x))
    mu, sd = lin(va, "mu", h), (0.5 * lin(va, "logvar", h)).exp()
    eps = torch.randn(1, 2, generator=torch.Generator().manual_seed(0))
    z_va = mu + sd * eps
    x_va = lin(va, "dec.2", torch.relu(lin(va, "dec.0", z_va)))

    fig, ax = plt.subplots(figsize=(10.6, 5.6))
    ax.set_xlim(0, 12); ax.set_ylim(0, 6.4); ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.canvas.draw()

    for y0, name in ((4.75, "AE"), (1.75, "VAE")):
        ax.text(0.25, y0, name, fontsize=17, fontweight="bold", va="center")
        _bd_box(ax, (2.75, y0 - 0.4), 1.1, 0.8, "E\nencoder", fs=10)
        _bd_box(ax, (8.55, y0 - 0.4), 1.1, 0.8, "D\ndecoder", fs=10,
                fc="#ecfdf5", ec="#047857")
        _bd_arrow(ax, (2.25, y0), (2.72, y0))
        _bd_arrow(ax, (9.68, y0), (10.2, y0))
        _bd_digit(fig, ax, 1.6, y0, x[0], 1.25, "input x")

    # --- AE: 점 하나 --------------------------------------------------------
    _bd_arrow(ax, (3.88, 4.75), (5.55, 4.75))
    ax.text(4.7, 4.9, f"z = ({z_ae[0, 0]:.1f}, {z_ae[0, 1]:.1f})", ha="center",
            fontsize=9.5)
    _bd_plane(fig, ax, 6.5, 4.75, 1.8, 16, [(z_ae[0, 0], z_ae[0, 1], "#f97316", "o", 6)],
              "latent space: one point")
    _bd_arrow(ax, (7.45, 4.75), (8.52, 4.75))
    _bd_digit(fig, ax, 10.9, 4.75, x_ae[0], 1.25, "output x'")
    ax.text(6.5, 3.55, "loss = ||x - x'||^2", ha="center", fontsize=10.5,
            family="monospace")

    # --- VAE: 분포에서 뽑은 점 ----------------------------------------------
    _bd_arrow(ax, (3.88, 1.75), (4.2, 1.75))
    _bd_box(ax, (4.22, 1.2), 1.15, 1.1,
            f"mu =\n({mu[0, 0]:.2f}, {mu[0, 1]:.2f})\nsigma =\n({sd[0, 0]:.3f}, {sd[0, 1]:.3f})",
            fc="#fff7ed", ec="#c2410c", fs=8.3)
    _bd_arrow(ax, (5.4, 1.75), (5.58, 1.75))
    pl = _bd_plane(fig, ax, 6.5, 1.75, 1.8, 4,
                   [(z_va[0, 0], z_va[0, 1], "#2563eb", "o", 6)],
                   "latent space: a draw from N(mu, sigma^2)")
    for r in (1, 2):
        pl.add_patch(Circle((0, 0), r, fill=False, ls=":", lw=1.1, ec="#111", alpha=0.6))
    pl.text(2.1, 2.3, "N(0, I)", fontsize=7.5)
    _bd_arrow(ax, (7.45, 1.75), (8.52, 1.75))
    _bd_digit(fig, ax, 10.9, 1.75, x_va[0], 1.25, "output x'")
    ax.text(6.5, 0.55, "z = mu + sigma * eps,   eps ~ N(0, I)", ha="center",
            fontsize=10.5, family="monospace")
    ax.text(6.5, 0.2, "loss = ||x - x'||^2 + beta * KL( N(mu, sigma^2) || N(0, I) )",
            ha="center", fontsize=10.5, family="monospace")
    ax.plot([0.2, 11.8], [3.2, 3.2], color="#ccc", lw=1)
    ax.text(11.8, 6.2, "boxes and arrows are a schematic; the points and the two outputs\n"
            "are what the trained 2-d models return for this test image",
            ha="right", va="top", fontsize=8.3, color="#555")
    fig.savefig("board_ae_vae.svg", transparent=True)
    plt.close(fig)
    print(f"wrote board_ae_vae.svg   AE z ({z_ae[0,0]:.2f}, {z_ae[0,1]:.2f}) | "
          f"VAE mu ({mu[0,0]:.2f}, {mu[0,1]:.2f}) sigma ({sd[0,0]:.3f}, {sd[0,1]:.3f})")


@torch.no_grad()
def fig_board_cvae():
    """03_cvae.md — 칠판의 cVAE 그림: 라벨이 평면과 디코더를 고른다.

    잠재 평면 열 장을 비스듬히 포개어 그리고 2 의 평면만 칠한다. 왼쪽 그림은
    시험 집합의 2 한 장, 평면 위의 점은 cVAE-2 가 그 그림에 낸 mu, 오른쪽
    위는 그 mu 를 라벨 2 와 함께 디코딩한 것이다. 오른쪽 아래 세 장은
    N(0, I) 에서 뽑은 점 셋을 라벨 2 와 함께 디코딩한 것(SEED 0)이다.
    """
    import torch.nn.functional as F
    from matplotlib.patches import Polygon
    (_, _), (Xte, yte) = load_flat()
    sd = torch.load(f"{MODELS}/mnist_cvae2.pt", map_location="cpu", weights_only=True)
    lin = lambda k, v: v @ sd[f"{k}.weight"].T + sd[f"{k}.bias"]
    c2 = F.one_hot(torch.tensor([2]), 10).float()
    dec = lambda z: lin("dec.2", torch.relu(lin("dec.0", torch.cat([z, c2.expand(len(z), -1)], 1))))
    x = Xte[int((yte == 2).nonzero()[0])][None]
    mu = lin("mu", torch.relu(lin("enc.0", torch.cat([x, c2], 1))))
    zs = torch.randn(3, 2, generator=torch.Generator().manual_seed(0))

    fig, ax = plt.subplots(figsize=(10.6, 5.4))
    ax.set_xlim(0, 12); ax.set_ylim(0, 6.2); ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.canvas.draw()

    # 평면 열 장 — 아래에서 위로 0..9
    px, w, hh, sk = 4.6, 2.5, 0.34, 0.55
    ys = [0.55 + 0.5 * d for d in range(10)]
    for d in range(10):
        y = ys[d]
        on = d == 2
        ax.add_patch(Polygon([(px, y), (px + w, y), (px + w + sk, y + hh), (px + sk, y + hh)],
                             closed=True, fc="#fde68a" if on else "#f3f4f6",
                             ec="#b45309" if on else "#9ca3af", lw=1.6 if on else 0.9,
                             zorder=3 + d))
        ax.text(px + w + sk + 0.12, y + hh / 2, f"label {d}", fontsize=8.5, va="center",
                color="#b45309" if on else "#6b7280",
                fontweight="bold" if on else "normal")
    y2 = ys[2] + hh / 2
    # mu 를 2 의 평면 위에 찍는다 (평면 좌표 [-3, 3] 을 평행사변형에 옮긴다)
    def on_plane(z, y):
        u, v = (z[0] + 3) / 6, (z[1] + 3) / 6
        return px + u * w + v * sk, y + v * hh
    mx, my = on_plane(mu[0].tolist(), ys[2])
    ax.plot([mx], [my], "o", color="#dc2626", ms=6, zorder=20)
    for z in zs.tolist():
        qx, qy = on_plane(z, ys[2])
        ax.plot([qx], [qy], "x", color="#1d4ed8", ms=6, mew=1.8, zorder=20)

    _bd_digit(fig, ax, 1.0, 3.9, x[0], 1.3, "input: a real 2")
    _bd_box(ax, (2.3, 3.5), 1.25, 0.8, "E\n+ label 2", fs=9.5)
    _bd_arrow(ax, (1.68, 3.9), (2.27, 3.9))
    _bd_arrow(ax, (3.58, 3.9), (mx - 0.08, my + 0.06), color="#dc2626", rad=-0.15, z=25)
    ax.text(3.3, 2.85, "E_2", fontsize=11, color="#dc2626", fontweight="bold")
    for d, yy in ((9, 5.0), (0, 0.9)):
        _bd_arrow(ax, (3.2, 4.33 if d == 9 else 3.47), (px + 0.2, ys[d] + hh / 2),
                  color="#9ca3af", lw=1.0, ls="--", rad=0.0)
        ax.text(3.55, yy, f"E_{d}", fontsize=9.5, color="#6b7280")

    _bd_box(ax, (8.75, 3.5), 1.25, 0.8, "D\n+ label 2", fs=9.5, fc="#ecfdf5", ec="#047857")
    _bd_arrow(ax, (mx + 0.1, my + 0.05), (8.72, 4.0), color="#dc2626", rad=-0.2, z=25)
    _bd_arrow(ax, (10.03, 3.9), (10.45, 3.9))
    _bd_digit(fig, ax, 11.15, 3.9, dec(mu)[0], 1.3, "D_2(mu): rebuilt")
    ax.text(8.2, 2.95, "D_2", fontsize=11, color="#dc2626", fontweight="bold")

    out = dec(zs)
    for i in range(3):
        _bd_digit(fig, ax, 8.95 + 1.1 * i, 1.35, out[i], 1.0)
    ax.text(10.05, 2.0, "generate a 2: draw z ~ N(0, I) on plane 2\n(the crosses), decode with D_2",
            ha="center", va="bottom", fontsize=8.6, color="#1d4ed8")
    ax.text(6.1, 5.95, "one latent plane per label: the label picks the plane, z picks the handwriting",
            ha="center", fontsize=11)
    ax.text(0.15, 0.12, "schematic; the red dot (mu of the input), the rebuilt 2 and the three generated 2s "
            "come from the trained cVAE-2", fontsize=8.3, color="#555")
    fig.savefig("board_cvae.svg", transparent=True)
    plt.close(fig)
    print(f"wrote board_cvae.svg   mu ({mu[0,0]:.2f}, {mu[0,1]:.2f})")


def fig_board_gan_steps():
    """01_gan.md 0절 — 학습 루프의 두 스텝과, 그래디언트가 거슬러 가는 길.

    위: 판별기 스텝. 진짜에 1, 가짜에 0 을 정답으로 주고 D 만 고친다.
        fake.detach() 가 역전파를 D 앞에서 끊는다.
    아래: 생성기 스텝. 가짜에 1 을 정답으로 주고 G 만 고친다. 역전파는 D 를
        지나 G 까지 간다. D 의 웨이트는 oG 가 쥐고 있지 않으므로 움직이지 않는다.
    수는 넣지 않았다 — 도식이다.
    """
    fig, ax = plt.subplots(figsize=(10.8, 6.0))
    ax.set_xlim(0, 12); ax.set_ylim(0, 6.6); ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    G = dict(fc="#ecfdf5", ec="#047857"); D = dict(fc="#eef2ff", ec="#4338ca")
    GRAY = dict(fc="#f3f4f6", ec="#9ca3af")
    red = "#dc2626"

    # ---------------- step 1 ------------------------------------------------
    ax.text(0.2, 6.25, "step 1   train D   (oD.step())", fontsize=12.5, fontweight="bold")
    _bd_box(ax, (0.4, 5.05), 1.9, 0.6, "real images xb", fs=9.5, fc="#fff", ec="#555")
    _bd_arrow(ax, (2.33, 5.35), (6.9, 5.0))
    _bd_box(ax, (0.4, 3.85), 1.0, 0.6, "noise z", fs=9.5, fc="#fff", ec="#555")
    _bd_arrow(ax, (1.43, 4.15), (1.95, 4.15))
    _bd_box(ax, (1.98, 3.8), 1.0, 0.7, "G", fs=13, **GRAY)
    _bd_arrow(ax, (3.01, 4.15), (3.55, 4.15))
    _bd_box(ax, (3.58, 3.85), 2.05, 0.6, "fake.detach()", fs=9.5, fc="#fff", ec="#555")
    _bd_arrow(ax, (5.66, 4.15), (6.9, 4.5))
    _bd_box(ax, (6.93, 4.3), 1.0, 0.9, "D", fs=13, **D)
    _bd_arrow(ax, (7.96, 4.95), (9.0, 5.3)); _bd_arrow(ax, (7.96, 4.55), (9.0, 4.2))
    _bd_box(ax, (9.03, 5.05), 2.6, 0.55, "target 1  (ones)", fs=9.5, fc="#fff", ec="#555")
    _bd_box(ax, (9.03, 3.9), 2.6, 0.55, "target 0  (zeros)", fs=9.5, fc="#fff", ec="#555")
    # 그래디언트
    _bd_arrow(ax, (9.0, 4.05), (7.99, 4.4), color=red, ls="--", lw=1.4)
    _bd_arrow(ax, (6.88, 4.38), (5.7, 4.02), color=red, ls="--", lw=1.4)
    ax.text(5.2, 3.62, "X", color=red, fontsize=15, fontweight="bold", ha="center", va="center")
    ax.text(5.45, 3.45, "backprop stops here:\nnothing reaches G", color=red, fontsize=8.8,
            va="top")
    ax.text(7.43, 5.33, "updated", fontsize=8.8, ha="center", color="#4338ca")
    ax.text(2.48, 3.6, "not updated", fontsize=8.8, ha="center", color="#6b7280")

    ax.plot([0.2, 11.8], [3.0, 3.0], color="#ccc", lw=1)

    # ---------------- step 2 ------------------------------------------------
    ax.text(0.2, 2.65, "step 2   train G   (oG.step())", fontsize=12.5, fontweight="bold")
    _bd_box(ax, (0.4, 1.3), 1.0, 0.6, "noise z", fs=9.5, fc="#fff", ec="#555")
    _bd_arrow(ax, (1.43, 1.6), (1.95, 1.6))
    _bd_box(ax, (1.98, 1.25), 1.0, 0.7, "G", fs=13, **G)
    _bd_arrow(ax, (3.01, 1.6), (3.55, 1.6))
    _bd_box(ax, (3.58, 1.3), 2.05, 0.6, "fake", fs=9.5, fc="#fff", ec="#555")
    _bd_arrow(ax, (5.66, 1.6), (6.9, 1.6))
    _bd_box(ax, (6.93, 1.15), 1.0, 0.9, "D", fs=13, **GRAY)
    _bd_arrow(ax, (7.96, 1.6), (9.0, 1.6))
    _bd_box(ax, (9.03, 1.32), 2.6, 0.55, "target 1  (ones)", fs=9.5, fc="#fff", ec="#555")
    for a, b in (((9.0, 1.42), (7.99, 1.42)), ((6.88, 1.42), (5.7, 1.42)),
                 ((3.52, 1.42), (3.04, 1.42))):
        _bd_arrow(ax, a, b, color=red, ls="--", lw=1.4)
    ax.text(2.48, 2.08, "updated", fontsize=8.8, ha="center", color="#047857")
    ax.text(7.43, 2.18, "not updated\n(no line freezes it)", fontsize=8.8, ha="center",
            color="#6b7280")
    ax.text(5.9, 0.62, "backprop passes through D and into G; D's weights stay put\n"
            "because oG holds only G's parameters", color=red, fontsize=8.8, ha="center")
    ax.text(11.8, 0.1, "black: forward pass     red dashed: gradient     schematic, no measured values",
            ha="right", fontsize=8.3, color="#555")
    fig.savefig("board_gan_steps.svg", transparent=True)
    plt.close(fig)
    print("wrote board_gan_steps.svg")


# 그림 함수가 모두 정의된 뒤에 부른다 — 이 블록이 파일 가운데 있으면
# 그 아래에서 정의한 함수를 부르는 줄이 NameError 로 멈춘다.
if __name__ == "__main__":
    fig_pca_reconstructions()
    fig_recon_ladder()
    fig_latent_2d()
    fig_no_noise()
    fig_aggregate_posterior()
    fig_conv_off_tradeoff()
    fig_obedience_split()
    fig_spread_vs_gap()
    fig_beta_exchange_rate()
    fig_two_steps_mirror()
    fig_conv_flip()
    fig_conv_obedience()
    fig_ae_vs_vae_2d()
    fig_class_share()
    fig_ae_linear_descent()
    fig_linear_ae_plane()
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
    fig_dcgan_walk()
    fig_dcgan_nearest()
    fig_dcgan_thickness()
    fig_dcgan_init_arms()
    fig_dcgan_shapes()
    fig_judge_sees_three()
    fig_point_vs_blob()
    fig_dcgan_recipe_gains()
    fig_lfw_samples()
    fig_lfw_walk()
    fig_lfw_vector()
    fig_who_wins()

    # --- 들머리 그림 (쪽의 물음을 세우는 자리) ---------------------------
    fig_lead_howmany()
    fig_lead_projections()
    fig_pca_funnel()
    fig_lead_tie()
    fig_lead_curved()
    fig_lead_same_trick()
    fig_lead_flip()
    fig_lead_worse_than_nothing()
    fig_lead_cliff()
    fig_lead_kl_buys()
    fig_lead_price_tag()
    fig_lead_dial()
    fig_lead_who_disobeys()
    fig_lead_average_blur()
    fig_lead_loss_wanders()
    fig_lead_scoreboard_lies()
    fig_lead_price_in_time()
    fig_lead_never_a_digit()
    fig_lead_96x()

    # --- 11/11 강의 필기에 붙인 그림 셋 ---------------------------------
    fig_gan_code_lines()
    fig_dcgan_bn_modes()
    fig_cvae_ten_planes()
    fig_board_ae_vae()
    fig_board_cvae()
    fig_board_gan_steps()
