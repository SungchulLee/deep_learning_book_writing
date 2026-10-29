"""26장 구조 절 쪽들의 그림.

규칙은 CLAUDE.md 의 Figures 항에 적어 둔 대로다.
- SVG, fonttype='path', transparent
- 글자는 모두 아스키 (matplotlib 기본 폰트에 한글이 없다)

**체크포인트를 읽지 않는다.** 필요한 학습은 이 스크립트 안에서 씨앗을 박고
처음부터 돌린다. 그래야 그림에 적힌 수를 언제든 다시 만들 수 있다.

실행: python docs/ch25/architecture/figures/make_figures.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["svg.fonttype"] = "path"

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
INK = "#2f3337"
ACCENT = "#8a4b2a"
BLUE = "#2d6a8f"


def style(ax):
    ax.tick_params(labelsize=8, colors=INK)
    for s in ax.spines.values():
        s.set_color(INK)
        s.set_linewidth(0.7)


# === 그림으로 보는 AI: 같은 평면, 다른 축 =====================================
#
# Bourlard-Kamp 1988 의 주장을 수로 보인다. 선형 AE 의 웨이트는 주부분공간을
# 펼치지만 로딩 벡터와 같지 않다. 부분공간(주각)과 축(각도)을 따로 잰다.

def _train_linear_ae(Xc, k, seed=0, steps=6000, lr=0.01):
    """활성화도 바이어스도 없는 AE. 자료는 이미 중심화되어 있다."""
    torch.manual_seed(seed)
    enc = nn.Linear(Xc.shape[1], k, bias=False)
    dec = nn.Linear(k, Xc.shape[1], bias=False)
    opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=lr)
    Xt = torch.tensor(Xc)
    for _ in range(steps):
        opt.zero_grad()
        loss = ((dec(enc(Xt)) - Xt) ** 2).sum(1).mean()
        loss.backward()
        opt.step()
    with torch.no_grad():
        err = ((dec(enc(Xt)) - Xt) ** 2).sum(1).mean().item()
    return enc.weight.detach().numpy(), err


def _principal_angles(A, B):
    Qa = np.linalg.qr(A.T)[0]
    Qb = np.linalg.qr(B.T)[0]
    s = np.linalg.svd(Qa.T @ Qb, compute_uv=False)
    return np.degrees(np.arccos(np.clip(s, -1, 1)))


def same_plane_other_axes():
    rng = np.random.default_rng(0)
    n = 800
    lat = rng.normal(size=(n, 2)) * np.array([3.0, 1.4])
    basis = np.linalg.qr(rng.normal(size=(3, 3)))[0][:, :2]
    X = lat @ basis.T + rng.normal(scale=0.25, size=(n, 3))
    X = (X - X.mean(0)).astype(np.float32)

    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    V = Vt[:2]
    err_pca = ((X - (X @ V.T) @ V) ** 2).sum(1).mean()
    W, err_ae = _train_linear_ae(X, 2)
    ang = _principal_angles(W, V)

    Z_pca = X @ V.T
    Z_ae = X @ W.T
    r_pca = np.corrcoef(Z_pca.T)[0, 1]
    r_ae = np.corrcoef(Z_ae.T)[0, 1]

    # 평면 위에서 보기 위해 PCA 기저로 좌표를 잡는다
    Wp = np.stack([V @ w for w in W])          # AE 축을 평면 좌표로
    Wp = np.stack([w / np.linalg.norm(w) for w in Wp])

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(11.6, 3.9))

    # --- 1. 같은 평면 위의 두 축 ------------------------------------------
    ax1.scatter(Z_pca[:, 0], Z_pca[:, 1], s=7, color=INK, alpha=0.18,
                edgecolors="none", rasterized=True)
    L = 5.2
    for j, (lab, col, sty) in enumerate(
            [("PCA axis", BLUE, "-"), ("AE axis", ACCENT, "--")]):
        for i in range(2):
            v = np.array([1.0, 0.0]) if i == 0 else np.array([0.0, 1.0])
            if j == 1:
                v = Wp[i]
            ax1.plot([-L * v[0], L * v[0]], [-L * v[1], L * v[1]],
                     color=col, linewidth=2.0, linestyle=sty,
                     label=lab if i == 0 else None, zorder=3)
    ax1.set_title("the same plane", fontsize=10, color=INK, weight="bold")
    # 각도는 축 **밖**에 적는다 — 안에 두면 AE 축선 위에 겹친다
    ax1.set_xlabel(f"principal angles between the two subspaces:\n"
                   f"{ang[0]:.4f} and {ang[1]:.4f} deg",
                   fontsize=8.3, color=INK)
    ax1.legend(fontsize=8, loc="upper right", framealpha=0.9)
    ax1.set_aspect("equal")
    ax1.set_xlim(-6, 6)
    ax1.set_ylim(-6, 6)
    style(ax1)

    # 두 판이 같은 눈금이라야 기울기를 눈으로 견줄 수 있다.
    # 가로세로를 따로 잡는다 — 억지로 정사각형을 만들면 빈 자리만 넓어진다.
    xl = 1.06 * max(np.abs(Z_pca[:, 0]).max(), np.abs(Z_ae[:, 0]).max())
    yl = 1.06 * max(np.abs(Z_pca[:, 1]).max(), np.abs(Z_ae[:, 1]).max())

    # --- 2. PCA 점수 -------------------------------------------------------
    ax2.scatter(Z_pca[:, 0], Z_pca[:, 1], s=8, color=BLUE, alpha=0.30,
                edgecolors="none", rasterized=True)
    ax2.set_title(f"PCA scores:  r = {r_pca:+.6f}", fontsize=10, color=INK,
                  weight="bold")
    ax2.set_xlabel("score 1", fontsize=8.5)
    ax2.set_ylabel("score 2", fontsize=8.5)

    # --- 3. AE 잠재 --------------------------------------------------------
    ax3.scatter(Z_ae[:, 0], Z_ae[:, 1], s=8, color=ACCENT, alpha=0.30,
                edgecolors="none", rasterized=True)
    ax3.set_title(f"AE latent:  r = {r_ae:+.6f}", fontsize=10, color=INK,
                  weight="bold")
    ax3.set_xlabel("latent 1", fontsize=8.5)
    ax3.set_ylabel("latent 2", fontsize=8.5)

    for ax in (ax2, ax3):
        ax.set_aspect("equal")
        ax.set_xlim(-xl, xl)
        ax.set_ylim(-yl, yl)
        style(ax)

    fig.text(0.5, -0.10,
             f"same reconstruction error ({err_pca:.8f} vs {err_ae:.8f}) and "
             f"the same plane -\nbut the axes are turned, and the latent "
             f"coordinates are not uncorrelated.",
             ha="center", fontsize=8.3, style="italic", color=INK)
    fig.tight_layout()
    fig.savefig(HERE / "same_plane_other_axes.svg", transparent=True,
                bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"wrote same_plane_other_axes.svg  pca {err_pca:.8f} ae {err_ae:.8f} "
          f"angles {ang[0]:.4f}/{ang[1]:.4f} r_pca {r_pca:+.6f} r_ae {r_ae:+.6f}")


# === 그림으로 보는 AI: 원래 디노이징 잡음은 가우시안이 아니었다 ================
#
# Vincent 2008 은 성분 몇 개를 골라 **0 으로 지웠다** (masking noise).
# 이 쪽 3절의 세 분기 가운데 dropout 분기가 그것이다.

def corruption_kinds():
    rng = np.random.default_rng(7)
    img = np.zeros((28, 28))
    img[6:22, 12:16] = 1.0                      # 숫자 1 비슷한 획
    img[18:22, 8:20] = 1.0

    nu = 0.3
    gauss = np.clip(img + rng.normal(scale=nu, size=img.shape), 0, 1)
    sp = img.copy()
    m = rng.random(img.shape)
    sp[m < nu / 2] = 1.0
    sp[(m >= nu / 2) & (m < nu)] = 0.0
    mask = img.copy()
    mask[rng.random(img.shape) < nu] = 0.0

    panels = [
        (img, "clean", ""),
        (gauss, "gaussian", "add noise, then clip"),
        (sp, "salt and pepper", "force to 1 or to 0"),
        (mask, "masking", "force a fraction to 0"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(10.0, 3.0))
    for ax, (im, title, note) in zip(axes, panels):
        ax.imshow(im, cmap="gray_r", vmin=0, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontsize=9.5, color=INK, weight="bold", pad=6)
        ax.text(13.5, 31.5, note, ha="center", fontsize=7.6, color=INK,
                alpha=0.85)
        for s in ax.spines.values():
            s.set_color(INK)
            s.set_linewidth(0.7)
    axes[3].set_title("masking", fontsize=9.5, color=ACCENT, weight="bold",
                      pad=6)
    for s in axes[3].spines.values():
        s.set_color(ACCENT)
        s.set_linewidth(1.8)

    fig.text(0.5, -0.20,
             "Vincent et al. 2008 used the one on the right: a fixed number of "
             "components chosen at random and forced to 0.\n"
             "schematic digit - not an MNIST sample.",
             ha="center", fontsize=8.3, style="italic", color=INK)
    fig.tight_layout()
    fig.savefig(HERE / "corruption_kinds.svg", transparent=True,
                bbox_inches="tight")
    plt.close(fig)
    print("wrote corruption_kinds.svg")


# === 그림으로 보는 AI: 고르지 않은 겹침 =======================================
#
# 전치 컨볼루션의 웨이트를 전부 1 로 두고 1 을 넣으면, 출력값이 곧 그 자리에
# **몇 번 더해졌는가**가 된다. 커널이 stride 로 나누어떨어지지 않으면 이 수가
# 자리마다 달라진다 — Odena 등이 "uneven overlap" 이라 부른 것이다.

def uneven_overlap():
    def counts(k, s, n=7):
        ct = nn.ConvTranspose2d(1, 1, kernel_size=k, stride=s, bias=False)
        with torch.no_grad():
            ct.weight.fill_(1.0)
            return ct(torch.ones(1, 1, n, n))[0, 0].numpy()[3:9, 3:9]

    cases = [
        (3, 2, "k=3, s=2", "3 is not divisible by 2"),
        (4, 2, "k=4, s=2", "4 is divisible by 2"),
        (2, 2, "k=2, s=2", "2 is divisible by 2"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.8))
    for ax, (k, s, title, note) in zip(axes, cases):
        c = counts(k, s)
        uneven = len(set(c.flatten().tolist())) > 1
        ax.imshow(c, cmap="pink_r", vmin=0, vmax=5)
        for i in range(c.shape[0]):
            for j in range(c.shape[1]):
                ax.text(j, i, f"{c[i, j]:.0f}", ha="center", va="center",
                        fontsize=10, color=INK,
                        weight="bold" if uneven else "normal")
        ax.set_xticks([])
        ax.set_yticks([])
        col = ACCENT if uneven else INK
        ax.set_title(f"{title}\n{note}", fontsize=9.5, color=col,
                     weight="bold")
        for sp in ax.spines.values():
            sp.set_color(col)
            sp.set_linewidth(1.8 if uneven else 0.8)
        ax.set_xlabel("1, 2 and 4 in one map" if uneven else "every cell the same",
                      fontsize=8.6, color=col)

    fig.text(0.5, -0.09,
             "how many times each output cell gets written, with all weights "
             "set to 1.\n"
             "an even kernel is not the point - 2 and 4 are both fine. "
             "divisibility by the stride is.",
             ha="center", fontsize=8.3, style="italic", color=INK)
    fig.tight_layout()
    fig.savefig(HERE / "uneven_overlap.svg", transparent=True,
                bbox_inches="tight")
    plt.close(fig)
    print("wrote uneven_overlap.svg  k3s2 uneven, k4s2 and k2s2 even")


# === 그림으로 보는 AI: BCE 를 우도로 읽으면 적분이 1 이 아니다 ================
#
# lam^x (1-lam)^(1-x) 를 x in [0,1] 에서 적분하면 (1-2lam)/log((1-lam)/lam) 이고
# lam=0.5 에서 극한이 0.5 다. 어디서도 1 이 되지 않는다. 더 중요한 것은 이 값이
# **lam 에 따라 달라진다**는 점이라, 상수로 치고 무시할 수가 없다.

def bce_not_a_density():
    lam = np.linspace(1e-4, 1 - 1e-4, 2000)
    with np.errstate(divide="ignore", invalid="ignore"):
        integral = (1 - 2 * lam) / np.log((1 - lam) / lam)
    integral[np.abs(lam - 0.5) < 1e-9] = 0.5

    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    ax.plot(lam, integral, color=ACCENT, lw=2.2)
    ax.axhline(1.0, color=INK, ls="--", lw=1.3)
    ax.text(0.5, 1.02, "what a density has to integrate to", ha="center",
            fontsize=8.6, color=INK)

    for v in (0.01, 0.5, 0.99):
        y = 0.5 if v == 0.5 else (1 - 2 * v) / np.log((1 - v) / v)
        ax.plot([v], [y], "o", color=INK, ms=6, zorder=5)
        ax.annotate(f"{y:.3f}", xy=(v, y), xytext=(0, -16),
                    textcoords="offset points", ha="center", fontsize=8.6,
                    color=INK)

    ax.fill_between(lam, integral, 1.0, color=ACCENT, alpha=0.10)
    ax.text(0.5, 0.74, "the missing mass", ha="center", fontsize=9,
            color=ACCENT, style="italic")

    ax.set_xlabel("lambda, the value the decoder predicts for a pixel",
                  fontsize=9)
    ax.set_ylabel("integral over x in [0, 1]", fontsize=9)
    ax.set_title("BCE read as a likelihood, on grey pixels", fontsize=10.5,
                 color=INK, weight="bold", pad=8)
    ax.set_ylim(0, 1.16)
    ax.set_xlim(0, 1)
    ax.grid(alpha=0.13)
    ax.tick_params(labelsize=8.5, colors=INK)
    for s in ax.spines.values():
        s.set_color(INK)
        s.set_linewidth(0.7)

    fig.text(0.5, -0.04,
             "it never reaches 1, and it changes with lambda - so it is not a "
             "constant that drops out of the gradient.\n"
             "on genuinely binary targets the question does not arise.",
             ha="center", fontsize=8.3, style="italic", color=INK)
    fig.tight_layout()
    fig.savefig(HERE / "bce_not_a_density.svg", transparent=True,
                bbox_inches="tight")
    plt.close(fig)
    print(f"wrote bce_not_a_density.svg  at 0.5 -> 0.500, "
          f"at 0.01 -> {(1-0.02)/np.log(0.99/0.01):.3f}")


if __name__ == "__main__":
    same_plane_other_axes()
    corruption_kinds()
    uneven_overlap()
    bce_not_a_density()
