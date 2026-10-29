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




# ========================================================================
# 「그림으로 보는 AI 7」 — 피어슨과 호텔링이 같은 선을 두 이름으로 불렀다
#
# 쪽의 정리 2 가 말하는 등식을 한 장에 담는다. 왼쪽은 피어슨의 물음
# (거리를 가장 작게), 오른쪽은 호텔링 쪽의 물음(퍼짐을 가장 크게).
# 수는 모두 직접 계산한 값이고, 자료는 이 절의 다른 쪽들과 같다.
# ========================================================================

def make_pearson_hotelling():
    rng = np.random.default_rng(42)
    n = 150
    X = rng.multivariate_normal([2.0, -1.0],
                                [[3.0, 2.2], [2.2, 2.0]], size=n)
    mu = X.mean(axis=0)
    Xc = X - mu
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    v = Vt.T[:, 0]

    z = Xc @ v                       # 투영 점수
    proj = np.outer(z, v) + mu       # 축 위의 자리
    resid = X - proj

    # 정리 2 의 두 항 — 합이 전체 분산이다
    spread = (z ** 2).sum() / (n - 1)
    leftover = (resid ** 2).sum() / (n - 1)
    total = (Xc ** 2).sum() / (n - 1)

    t = np.linspace(-3.2, 3.2, 2)
    axis = mu + np.outer(t * S[0] / np.sqrt(n), v)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 4.0))

    # --- 왼쪽: 거리를 가장 작게 (피어슨) ---
    ax1.scatter(X[:, 0], X[:, 1], s=16, alpha=0.5, color="#2a4b8a",
                linewidths=0, zorder=2)
    for i in range(0, n, 3):
        ax1.plot([X[i, 0], proj[i, 0]], [X[i, 1], proj[i, 1]],
                 color="#8a2a4b", lw=0.9, alpha=0.75, zorder=3)
    ax1.plot(axis[:, 0], axis[:, 1], color=INK, lw=1.8, zorder=4)
    ax1.set_title("1901  Pearson\nmake these distances small",
                  fontsize=10, weight="bold", color=INK, pad=8)
    ax1.text(0.03, 0.03, f"sum of squares / (n-1)\n= {leftover:.3f}",
             transform=ax1.transAxes, fontsize=8.5, family="monospace",
             color="#8a2a4b",
             bbox=dict(fc="white", ec="none", alpha=0.85, pad=2.5))

    # --- 오른쪽: 퍼짐을 가장 크게 (호텔링) ---
    ax2.scatter(X[:, 0], X[:, 1], s=16, alpha=0.16, color="#2a4b8a",
                linewidths=0, zorder=2)
    ax2.plot(axis[:, 0], axis[:, 1], color=INK, lw=1.8, zorder=3)
    ax2.scatter(proj[:, 0], proj[:, 1], s=16, alpha=0.8, color="#2f6f4e",
                marker="|", linewidths=1.1, zorder=4)
    lo, hi = mu + np.outer([z.min(), z.max()], v)
    ax2.annotate("", xy=tuple(hi), xytext=tuple(lo),
                 arrowprops=dict(arrowstyle="<|-|>", color="#2f6f4e", lw=1.4))
    ax2.set_title("1933  Hotelling\nmake this spread big",
                  fontsize=10, weight="bold", color=INK, pad=8)
    ax2.text(0.03, 0.03, f"variance of the marks\n= {spread:.3f}",
             transform=ax2.transAxes, fontsize=8.5, family="monospace",
             color="#2f6f4e",
             bbox=dict(fc="white", ec="none", alpha=0.85, pad=2.5))

    for ax in (ax1, ax2):
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for s_ in ax.spines.values():
            s_.set_color(INK)
            s_.set_linewidth(0.7)

    fig.text(0.5, -0.04,
             f"{spread:.3f} + {leftover:.3f} = {total:.3f}. The total is "
             f"fixed, so making one big IS making the other small - "
             f"one line, two questions",
             ha="center", fontsize=8.5, style="italic", color=INK)
    fig.tight_layout()
    fig.savefig(HERE / "pearson_hotelling.svg", transparent=True,
                bbox_inches="tight")
    plt.close(fig)
    print("wrote pearson_hotelling.svg")


# === 그림으로 보는 AI: Oja 1982 의 규칙을 이 쪽의 자료에 =======================
#
# Oja 는 신경세포가 입력 **상관**행렬 E[xx^T] 의 으뜸 고유벡터로 간다고 보였다.
# 그것이 공분산의 으뜸 고유벡터, 곧 제1주성분이 되는 것은 자료의 평균이 0 일
# 때뿐이다. 그래서 중심화한 자료와 원자료에 같은 규칙을 돌려 나란히 그린다.
# 수는 모두 여기서 실제로 돌려 얻는다 — 그림에 적어 넣는 수는 지어내지 않는다.

def oja_rule(data, gamma=0.002, passes=60, seed=0):
    """m <- m + gamma * eta * (x - eta * m),  eta = m^T x.  명시적 정규화 없음."""
    g = torch.Generator().manual_seed(seed)
    m = torch.randn(2, generator=g)
    m = m / m.norm()
    angles = []
    for _ in range(passes):
        for i in torch.randperm(len(data), generator=g):
            x = data[i]
            eta = torch.dot(m, x)
            m = m + gamma * eta * (x - eta * m)
        angles.append(torch.rad2deg(torch.atan2(m[1], m[0])).item() % 180)
    return m, angles


def make_oja():
    torch.manual_seed(42)                 # 쪽의 코드와 같은 씨앗, 같은 자료
    n = 150
    mean_true = torch.tensor([2.0, -1.0])
    cov_true = torch.tensor([[3.0, 2.2], [2.2, 2.0]])
    X = mean_true + torch.randn(n, 2) @ torch.linalg.cholesky(cov_true).T
    mu = X.mean(dim=0, keepdim=True)
    Xc = X - mu

    pc1 = torch.linalg.svd(Xc, full_matrices=False)[2].T[:, 0]
    svd_deg = torch.rad2deg(torch.atan2(pc1[1], pc1[0])).item() % 180

    _, ang_c = oja_rule(Xc)
    m_r, ang_r = oja_rule(X)
    deg_c, deg_r = ang_c[-1], ang_r[-1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.2, 4.3))

    # --- 왼쪽: 자료와 두 방향 ---------------------------------------------
    ax1.scatter(X[:, 0], X[:, 1], s=13, color=INK, alpha=0.32,
                edgecolors="none")
    cx, cy = mu[0, 0].item(), mu[0, 1].item()

    t = np.linspace(-5.2, 5.2, 2)
    r_c = np.radians(deg_c)
    ax1.plot(cx + t * np.cos(r_c), cy + t * np.sin(r_c), color=ACCENT,
             linewidth=2.2, label=f"centered: {deg_c:.2f} deg")
    r_r = np.radians(deg_r)
    ax1.plot(t * np.cos(r_r) * 1.6, t * np.sin(r_r) * 1.6, color="#2d6a8f",
             linewidth=2.2, linestyle="--", label=f"raw: {deg_r:.2f} deg")

    ax1.plot([cx], [cy], marker="o", color=ACCENT, markersize=7,
             markeredgecolor="white", markeredgewidth=1.2, zorder=5)
    ax1.annotate("mean", (cx, cy), textcoords="offset points",
                 xytext=(9, 7), fontsize=8, color=ACCENT)
    ax1.plot([0], [0], marker="x", color="#2d6a8f", markersize=8,
             markeredgewidth=2, zorder=5)
    ax1.annotate("origin", (0, 0), textcoords="offset points",
                 xytext=(-42, -4), fontsize=8, color="#2d6a8f")

    ax1.set_title("same rule, two data sets", fontsize=10, color=INK)
    ax1.set_xlabel("x1", fontsize=9)
    ax1.set_ylabel("x2", fontsize=9)
    ax1.legend(fontsize=8, loc="upper left", framealpha=0.9)
    ax1.set_aspect("equal")
    ax1.grid(alpha=0.12)

    # --- 오른쪽: 지나간 각 -------------------------------------------------
    steps = np.arange(1, len(ang_c) + 1)
    ax2.axhline(svd_deg, color=INK, linestyle=":", linewidth=1.3)
    ax2.annotate(f"SVD: {svd_deg:.2f} deg", (len(ang_c) * 0.52, svd_deg),
                 textcoords="offset points", xytext=(0, 6), fontsize=8.5,
                 color=INK)
    ax2.plot(steps, ang_c, color=ACCENT, linewidth=1.8, label="centered")
    ax2.plot(steps, ang_r, color="#2d6a8f", linewidth=1.8, linestyle="--",
             label="raw")
    ax2.set_xlabel("pass over the 150 points", fontsize=9)
    ax2.set_ylabel("angle of m  (deg)", fontsize=9)
    ax2.set_title("no normalization step anywhere", fontsize=10, color=INK)
    ax2.legend(fontsize=8, loc="center right", framealpha=0.9)
    ax2.grid(alpha=0.12)
    ax2.set_ylim(0, 50)

    fig.text(0.5, -0.04,
             f"Oja's rule, gamma = 0.002.  |m| ends at {m_r.norm():.4f} "
             f"although the rule never normalizes.",
             ha="center", fontsize=8.5, style="italic", color=INK)
    fig.tight_layout()
    fig.savefig(HERE / "oja_rule.svg", transparent=True, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote oja_rule.svg   centered {deg_c:.2f}  raw {deg_r:.2f}  "
          f"svd {svd_deg:.2f}")


# === 그림으로 보는 AI: 고유그림이 누구에게서 나왔는가 ==========================
#
# 같은 셈을 1987 년에는 얼굴에 했다. 다른 것은 셈이 아니라 **모아 놓은 무리**다.
# 두 수치는 확인한 것만 적는다 — 1987: 115 장, 128x128 = 16,384 픽셀.
# 이 쪽: 60,000 장, 28x28 = 784 픽셀. 줄인 뒤의 차원은 이 쪽의 출력에서 온다.

def make_ensembles():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.0, 3.9))

    def grid(ax, rows, cols, w, h, x0, y0, color, alpha=0.55):
        for r in range(rows):
            for c in range(cols):
                ax.add_patch(plt.Rectangle((x0 + c * w * 1.25,
                                            y0 - r * h * 1.35),
                                           w, h, facecolor=color,
                                           alpha=alpha, edgecolor="none"))

    # --- 1987 — 네모 115 개가 사진 115 장이다 (5 x 23 = 115) ------------------
    grid(ax1, 5, 23, 0.028, 0.045, 0.04, 0.72, "#2d6a8f")
    ax1.text(0.5, 0.41, "one square = one picture", fontsize=7,
             color=INK, ha="center", alpha=0.7)
    ax1.text(0.5, 0.86, "1987", fontsize=11, color=INK, ha="center",
             weight="bold")
    ax1.text(0.5, 0.80, "Sirovich and Kirby, eigenpictures", fontsize=8.5,
             color=INK, ha="center", style="italic")
    ax1.text(0.5, 0.29, "115 pictures\n128 x 128 = 16,384 numbers each",
             fontsize=9, color=INK, ha="center")
    ax1.text(0.5, 0.11,
             '"a relatively homogeneous population\n'
             'of smooth-skinned Caucasian males"\n'
             "- undergraduates at one university",
             fontsize=8, color=ACCENT, ha="center", style="italic")

    # --- 이 쪽 — 같은 자리에 훨씬 촘촘하게. 눈금을 그림에 적는다 --------------
    # 네모 2,000 개를 따로 그리면 SVG 가 0.5MB 가 된다. 한 장의 래스터로 깐다.
    cell = np.zeros((20 * 3, 100 * 3))
    cell[::3, :] = 1
    cell[:, ::3] = 1
    ax2.imshow(1 - cell, cmap="pink_r", vmin=0, vmax=1.9,
               extent=(0.04, 0.04 + 100 * 0.0058 * 1.25, 0.458, 0.765),
               aspect="auto", interpolation="nearest", zorder=2)
    ax2.text(0.5, 0.41, "one square = 30 images", fontsize=7,
             color=INK, ha="center", alpha=0.7)
    ax2.text(0.5, 0.86, "this page", fontsize=11, color=INK, ha="center",
             weight="bold")
    ax2.text(0.5, 0.80, "MNIST, eigendigits", fontsize=8.5, color=INK,
             ha="center", style="italic")
    ax2.text(0.5, 0.29, "60,000 images\n28 x 28 = 784 numbers each",
             fontsize=9, color=INK, ha="center")
    ax2.text(0.5, 0.11,
             "two groups of writers:\n"
             "Census Bureau employees\nand high school students",
             fontsize=8, color=ACCENT, ha="center", style="italic")

    for ax in (ax1, ax2):
        ax.set_xlim(0, 1)
        ax.set_ylim(0.03, 1.0)
        ax.axis("off")

    fig.text(0.5, -0.02,
             "the arithmetic is the same. the basis is a property of the "
             "ensemble, not of faces or of digits.",
             ha="center", fontsize=9, style="italic", color=INK)
    fig.tight_layout()
    fig.savefig(HERE / "ensembles.svg", transparent=True, bbox_inches="tight")
    plt.close(fig)
    print("wrote ensembles.svg")


# === 그림으로 보는 AI: scikit-learn 이 어디서 왔는가 ==========================
#
# 확인한 것만 적는다 — 2007 년 Google Summer of Code, 교토 대학의 대학원생
# 한 사람, 본디 이름은 scikits.learn (SciPy Toolkits 의 작명법). 첫 공개 배포는
# 2010 년 2 월 1 일, INRIA 의 네 사람.

def make_sklearn_timeline():
    fig, ax = plt.subplots(figsize=(9.6, 2.9))

    ax.annotate("", xy=(9.7, 1.0), xytext=(0.3, 1.0),
                arrowprops=dict(arrowstyle="->", color=INK, linewidth=1.3))

    marks = [
        (1.3, "2007",
         "Google Summer of Code\none graduate student, Kyoto\nworking on audio signal processing",
         "scikits.learn"),
        (5.0, "1 Feb 2010",
         "first public release\nfour researchers at INRIA\ntake over the project",
         "scikit-learn"),
        (8.6, "this page",
         "PCA(n_components=1)\n.fit_transform(X)\nthree lines",
         ""),
    ]
    for x, when, what, name in marks:
        ax.plot([x], [1.0], marker="o", color=ACCENT, markersize=9,
                markeredgecolor="white", markeredgewidth=1.5, zorder=5)
        ax.text(x, 1.22, when, ha="center", fontsize=9.5, weight="bold",
                color=INK)
        ax.text(x, 0.82, what, ha="center", va="top", fontsize=7.8,
                color=INK, alpha=0.9)
        if name:
            ax.text(x, 0.30, name, ha="center", fontsize=8.5,
                    family="monospace", color=ACCENT)

    ax.set_xlim(0, 10)
    ax.set_ylim(0.1, 1.5)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(HERE / "sklearn_timeline.svg", transparent=True,
                bbox_inches="tight")
    plt.close(fig)
    print("wrote sklearn_timeline.svg")


def main():
    make_sklearn_timeline()
    make_ensembles()
    make_oja()
    make_pearson_hotelling()
    make_whiten()
    make_angle_loss()
    X, y, mu, V, evr = fit()
    make_variance(evr)
    make_eigendigits(mu, V)
    make_reconstruction(X, mu, V, evr)
    make_scores(X, y, mu, V, evr)




if __name__ == "__main__":
    main()
