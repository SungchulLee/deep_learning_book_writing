"""「PyTorch로 MNIST에 PCA 적용하기」 쪽의 그림.

쪽의 코드와 **같은 계산** 을 한다 — 같은 60,000장, 같은 전체 SVD,
같은 분모 (n-1). 그래서 그림에 적힌 수(82.5%, 154)가 쪽의 출력과 맞는다.

규칙은 CLAUDE.md 의 Figures 항에 적어 둔 대로다.
- SVG, fonttype='path', transparent
- 글자는 모두 아스키 (matplotlib 기본 폰트에 한글이 없다)

MNIST 를 읽으므로 **저장소 뿌리에서** 돌린다.

실행: python docs/ch24/pca/figures/make_figures.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["svg.fonttype"] = "path"

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import datasets, transforms

HERE = Path(__file__).resolve().parent
INK = "#2f3337"
ACCENT = "#8a4b2a"
K = 50


def fit():
    """쪽의 코드와 같은 적합. 한 번만 하고 네 그림이 나눠 쓴다."""
    tf = transforms.Compose([transforms.ToTensor()])
    ds = datasets.MNIST(root="./data", train=True, download=False, transform=tf)
    X = torch.stack([ds[i][0] for i in range(len(ds))]).view(len(ds), -1)
    y = torch.tensor([ds[i][1] for i in range(len(ds))])
    mu = X.mean(0, keepdim=True)
    U, S, Vt = torch.linalg.svd(X - mu, full_matrices=False)
    ev = (S ** 2) / (X.shape[0] - 1)
    return X, y, mu, Vt.T, ev / ev.sum()


def style(ax):
    ax.tick_params(labelsize=8, colors=INK)
    for s in ax.spines.values():
        s.set_color(INK)
        s.set_linewidth(0.7)


def imshow(ax, vec, vmin=None, vmax=None, cmap="gray"):
    ax.imshow(vec.reshape(28, 28), cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


# ========================================================================
# 1. 설명 분산 — 50개가 82.5% 라는 수가 어디서 나오는가
# ========================================================================

def make_variance(evr):
    cum = np.cumsum(evr.numpy())
    k95 = int((cum < 0.95).sum()) + 1
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.6, 3.3))

    ax1.semilogy(np.arange(1, 101), evr.numpy()[:100], "o-",
                 color=ACCENT, ms=2.6, lw=1.0)
    ax1.set_xlabel("component", fontsize=9)
    ax1.set_ylabel("explained variance ratio (log)", fontsize=9)
    ax1.set_title("the spectrum falls fast", fontsize=9.5, weight="bold",
                  color=INK, pad=7)

    ax2.plot(np.arange(1, 785), cum, color=ACCENT, lw=1.4)
    for k, lab in ((K, f"k = {K}\n{cum[K-1]*100:.1f}%"),
                   (k95, f"k = {k95}\n95%")):
        ax2.plot([k], [cum[k - 1]], "o", color=INK, ms=5, zorder=4)
        ax2.annotate(lab, xy=(k, cum[k - 1]),
                     xytext=(k + 90, cum[k - 1] - 0.16), fontsize=8,
                     color=INK,
                     arrowprops=dict(arrowstyle="->", color=INK, lw=0.8))
    ax2.set_ylim(0, 1.02)
    ax2.set_xlabel("number of components", fontsize=9)
    ax2.set_ylabel("cumulative explained variance", fontsize=9)
    ax2.set_title("50 of 784 carry 82.5%", fontsize=9.5, weight="bold",
                  color=INK, pad=7)

    for ax in (ax1, ax2):
        style(ax)
    fig.tight_layout()
    fig.savefig(HERE / "variance.svg", transparent=True, bbox_inches="tight")
    plt.close(fig)
    print("wrote variance.svg")


# ========================================================================
# 2. 고유숫자 — 주성분을 그림으로 보면
# ========================================================================

def make_eigendigits(mu, V):
    fig, axes = plt.subplots(2, 6, figsize=(8.4, 3.1))
    imshow(axes[0, 0], mu.numpy()[0])
    axes[0, 0].set_title("mean", fontsize=8.5, color=INK, pad=4)
    comps = V.numpy()
    lim = np.abs(comps[:, :11]).max()
    for i in range(11):
        ax = axes.flat[i + 1]
        imshow(ax, comps[:, i], vmin=-lim, vmax=lim, cmap="RdBu_r")
        ax.set_title(f"PC{i+1}", fontsize=8.5, color=INK, pad=4)
    fig.text(0.5, -0.02,
             "red and blue are opposite signs - a component is a direction to "
             "add to or subtract from the mean, not a digit",
             ha="center", fontsize=8, style="italic", color=INK)
    fig.tight_layout()
    fig.savefig(HERE / "eigendigits.svg", transparent=True,
                bbox_inches="tight")
    plt.close(fig)
    print("wrote eigendigits.svg")


# ========================================================================
# 3. k 를 바꿔 가며 되살려 보면 — MSE 0.011793 이 눈에 어떻게 보이는가
# ========================================================================

def make_reconstruction(X, mu, V, evr):
    ks = [1, 5, 20, 50, 154, 784]
    rows = [3, 7, 11]                       # 아무 세 장
    cum = np.cumsum(evr.numpy())
    fig, axes = plt.subplots(len(rows), len(ks) + 1,
                             figsize=(9.0, 1.45 * len(rows)))
    Xc = X - mu
    recs = {}
    for k in ks:
        Vk = V[:, :k]
        recs[k] = (Xc @ Vk) @ Vk.T + mu
    for r, idx in enumerate(rows):
        imshow(axes[r, 0], X[idx].numpy(), 0, 1)
        if r == 0:
            axes[r, 0].set_title("original", fontsize=8.5, color=INK, pad=5)
        for c, k in enumerate(ks):
            imshow(axes[r, c + 1], recs[k][idx].numpy(), 0, 1)
            if r == 0:
                axes[r, c + 1].set_title(f"k = {k}\n{cum[k-1]*100:.0f}%",
                                         fontsize=8.5, color=INK, pad=5)
    fig.text(0.5, -0.03,
             "k = 50 is the page's setting (MSE 0.011793): readable but "
             "speckled. The speckle IS the 17.5% that was dropped.",
             ha="center", fontsize=8, style="italic", color=INK)
    fig.tight_layout()
    fig.savefig(HERE / "reconstruction.svg", transparent=True,
                bbox_inches="tight")
    plt.close(fig)
    print("wrote reconstruction.svg")


# ========================================================================
# 4. 앞 두 점수로 그려 보면 — 겹친다
# ========================================================================

def make_scores(X, y, mu, V, evr):
    g = torch.Generator().manual_seed(0)
    sel = torch.randperm(len(X), generator=g)[:3000]    # SVG 가 커지지 않게
    Z = ((X - mu) @ V[:, :2])[sel].numpy()
    lab = y[sel].numpy()
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    sc = ax.scatter(Z[:, 0], Z[:, 1], c=lab, cmap="tab10", s=4, alpha=0.65,
                    linewidths=0, rasterized=True)   # 벡터로 두면 SVG 가 680K 가 된다
    cb = fig.colorbar(sc, ax=ax, ticks=range(10))
    cb.ax.tick_params(labelsize=8, colors=INK)
    cb.set_label("digit", fontsize=9, color=INK)
    ax.set_xlabel(f"PC1  ({evr[0]*100:.1f}%)", fontsize=9)
    ax.set_ylabel(f"PC2  ({evr[1]*100:.1f}%)", fontsize=9)
    ax.set_title("first two scores, 3000 of 60000 samples",
                 fontsize=9.5, weight="bold", color=INK, pad=7)
    style(ax)
    fig.text(0.5, -0.02,
             "these two carry only "
             f"{(evr[0]+evr[1])*100:.1f}% - overlap here is not overlap in 784 "
             "dimensions",
             ha="center", fontsize=8, style="italic", color=INK)
    fig.tight_layout()
    fig.savefig(HERE / "scores_2d.svg", transparent=True,
                bbox_inches="tight", dpi=150)
    plt.close(fig)
    print("wrote scores_2d.svg")




# ========================================================================
# 「sklearn」 쪽 — whiten 이 하는 일
#
# 그 쪽 4절은 "whiten=True 는 학습 표본 공분산이 단위 행렬이 되도록 한다"
# 고 적어 두었다. 수로만 적혀 있어 그림으로 보인다.
# ========================================================================

def make_whiten():
    from sklearn.decomposition import PCA

    rng = np.random.default_rng(42)
    n = 150
    X = rng.multivariate_normal([2.0, -1.0],
                                [[3.0, 2.2], [2.2, 2.0]], size=n)

    Z = PCA(n_components=2, whiten=False).fit_transform(X)
    W = PCA(n_components=2, whiten=True).fit_transform(X)

    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.5))
    for ax, data, title in (
            (axes[0], X, "original x"),
            (axes[1], Z, "PCA scores  (whiten=False)"),
            (axes[2], W, "PCA scores  (whiten=True)")):
        ax.scatter(data[:, 0], data[:, 1], s=14, alpha=0.5, color="#2a4b8a",
                   linewidths=0)
        c = np.cov(data.T)
        ax.set_title(title, fontsize=9.5, weight="bold", color=INK, pad=7)
        # 제목 아래에 적으면 가운데 판에서 넘쳐 나간다. 판 바깥 위에 둔다.
        ax.text(0.5, 1.30,
                f"cov = [{c[0,0]:.2f} {c[0,1]:.2f} ; "
                f"{c[1,0]:.2f} {c[1,1]:.2f}]",
                transform=ax.transAxes, ha="center", va="bottom",
                fontsize=7.5, family="monospace", color=ACCENT)
        ax.axhline(0, color=INK, lw=0.6, alpha=0.4)
        ax.axvline(0, color=INK, lw=0.6, alpha=0.4)
        ax.set_aspect("equal")
        style(ax)
    fig.text(0.5, -0.03,
             "rotate, then divide each axis by its standard deviation - "
             "whitening makes the sample covariance the identity",
             ha="center", fontsize=8, style="italic", color=INK)
    fig.tight_layout()
    fig.savefig(HERE / "whitening.svg", transparent=True, bbox_inches="tight")
    plt.close(fig)
    print("wrote whitening.svg")


# ========================================================================
# 「PyTorch」 쪽 — 재구성 오차를 방향의 함수로 보면
#
# 그 쪽 5절은 "재구성 오차의 그래디언트는 오차가 가장 빠르게 증가하는
# 방향" 이라고 적는다. 1차원 투영이라 방향을 각도 하나로 적을 수 있어
# 그 로스 지형을 통째로 그릴 수 있다.
# ========================================================================

def make_angle_loss():
    torch.manual_seed(42)
    n = 150
    cov = torch.tensor([[3.0, 2.2], [2.2, 2.0]])
    L = torch.linalg.cholesky(cov)
    X = torch.tensor([2.0, -1.0]) + torch.randn(n, 2) @ L.T
    Xc = X - X.mean(0, keepdim=True)

    U, S, Vh = torch.linalg.svd(Xc, full_matrices=False)
    pc1 = Vh.T[:, 0]
    best = float(torch.atan2(pc1[1], pc1[0]))

    th = torch.linspace(-np.pi / 2, np.pi / 2, 721)
    err = []
    for t in th:                       # 각 방향으로 투영했을 때의 MSE
        v = torch.stack([torch.cos(t), torch.sin(t)])
        err.append(((Xc - torch.outer(Xc @ v, v)) ** 2).mean().item())
    err = np.array(err)
    thn = th.numpy()
    b = best if best > -np.pi / 2 else best + np.pi

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.plot(np.degrees(thn), err, color=ACCENT, lw=1.6)
    ax.plot([np.degrees(b)], [err.min()], "o", color=INK, ms=6, zorder=4)
    ax.annotate(f"SVD answer\n{np.degrees(b):.1f} deg",
                xy=(np.degrees(b), err.min()),
                xytext=(np.degrees(b) + 22, err.min() + 0.28), fontsize=8.5,
                color=INK,
                arrowprops=dict(arrowstyle="->", color=INK, lw=0.8))
    # 최솟값 왼쪽의 내리막 한 점. 거기서 기울기 g 는 음수이므로,
    # **기울기 방향** 으로 가면 theta 가 줄고 오차는 늘어난다.
    i = int(np.argmin(np.abs(thn - (b - 0.6))))
    g = (err[i + 1] - err[i - 1]) / (np.degrees(thn[i + 1] - thn[i - 1]))
    dx = -16.0                                   # 기울기가 음수 -> 왼쪽
    ax.annotate("", xy=(np.degrees(thn[i]) + dx, err[i] + dx * g),
                xytext=(np.degrees(thn[i]), err[i]),
                arrowprops=dict(arrowstyle="-|>", color="#8a2a4b", lw=1.6))
    ax.plot([np.degrees(thn[i])], [err[i]], "o", color="#8a2a4b", ms=4,
            zorder=4)
    ax.text(np.degrees(thn[i]) + dx - 2, err[i] + dx * g + 0.10,
            "gradient points UP", fontsize=8, color="#8a2a4b", ha="right")
    ax.set_xlabel("projection direction (degrees)", fontsize=9)
    ax.set_ylabel("reconstruction MSE", fontsize=9)
    ax.set_title("one direction, one number - the whole loss landscape",
                 fontsize=9.5, weight="bold", color=INK, pad=7)
    style(ax)
    fig.text(0.5, -0.04,
             "autograd would walk DOWN this curve; the SVD jumps to the "
             "bottom in one step",
             ha="center", fontsize=8, style="italic", color=INK)
    fig.tight_layout()
    fig.savefig(HERE / "angle_loss.svg", transparent=True,
                bbox_inches="tight")
    plt.close(fig)
    print("wrote angle_loss.svg")


def main():
    make_whiten()
    make_angle_loss()
    X, y, mu, V, evr = fit()
    make_variance(evr)
    make_eigendigits(mu, V)
    make_reconstruction(X, mu, V, evr)
    make_scores(X, y, mu, V, evr)



if __name__ == "__main__":
    main()
