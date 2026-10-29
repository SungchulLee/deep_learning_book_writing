"""25.2 매니폴드 학습 쪽들의 그림.

규칙은 CLAUDE.md 의 Figures 항에 적어 둔 대로다.
- SVG, fonttype='path', transparent
- 글자는 모두 아스키 (matplotlib 기본 폰트에 한글이 없다)

데이터를 읽지 않으므로 어디서 돌려도 된다.

실행: python docs/ch24/manifold/figures/make_figures.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["svg.fonttype"] = "path"

import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
INK = "#2f3337"
ACCENT = "#8a4b2a"
PAPER = "#f3efe7"


# === 그림으로 보는 AI: Science 290권 2323쪽 ==================================
#
# Isomap 은 2319-2323쪽, LLE 는 2323-2326쪽이다. 두 논문이 2323쪽을 나눠 쓴다.
# 단의 배치는 실물을 재지 않았으므로 그림 안에 그렇게 적는다.

def page_2323():
    fig, ax = plt.subplots(figsize=(5.0, 6.4))

    ax.add_patch(plt.Rectangle((0, 0), 10, 13, facecolor=PAPER,
                               edgecolor=INK, linewidth=1.1))

    ax.text(5, 12.5, "SCIENCE   VOL 290   22 DECEMBER 2000",
            ha="center", va="center", fontsize=6.5, color=INK, alpha=0.75)
    ax.plot([0.6, 9.4], [12.15, 12.15], color=INK, linewidth=0.6, alpha=0.5)

    # --- 위: Isomap 의 끝 -------------------------------------------------
    ax.text(0.8, 11.6, "Tenenbaum, de Silva, Langford", fontsize=7.5,
            color=INK, style="italic")
    ax.text(0.8, 11.1, "A Global Geometric Framework for", fontsize=9,
            color=INK, weight="bold")
    ax.text(0.8, 10.6, "Nonlinear Dimensionality Reduction", fontsize=9,
            color=INK, weight="bold")
    ax.text(9.2, 11.6, "pp. 2319 - 2323", fontsize=6.5, color=INK,
            alpha=0.7, ha="right", va="center")

    for i in range(6):                      # 본문을 가는 줄로만 나타낸다
        y = 10.0 - i * 0.22
        ax.plot([0.8, 4.6], [y, y], color=INK, linewidth=0.5, alpha=0.3)
        ax.plot([5.4, 9.2], [y, y], color=INK, linewidth=0.5, alpha=0.3)

    ax.add_patch(plt.Rectangle((0.7, 7.45), 8.6, 1.0, facecolor=ACCENT,
                               alpha=0.16, edgecolor=ACCENT, linewidth=0.8))
    ax.text(5.0, 8.22, '"We thank Y. LeCun for making available the MNIST',
            fontsize=6.8, color=INK, ha="center", family="monospace")
    ax.text(5.0, 7.93, 'database and S. Roweis and L. Saul for sharing',
            fontsize=6.8, color=INK, ha="center", family="monospace")
    ax.text(5.0, 7.64, 'related unpublished work."',
            fontsize=6.8, color=INK, ha="center", family="monospace")

    ax.text(0.8, 7.1, "10 August 2000; accepted 21 November 2000",
            fontsize=6.2, color=INK, alpha=0.7, style="italic")

    # --- 가르는 선 ---------------------------------------------------------
    ax.plot([0.6, 9.4], [6.6, 6.6], color=INK, linewidth=1.0)

    # --- 아래: LLE 의 시작 -------------------------------------------------
    ax.text(0.8, 6.05, "Roweis and Saul", fontsize=7.5, color=INK,
            style="italic")
    ax.text(0.8, 5.55, "Nonlinear Dimensionality Reduction by", fontsize=9,
            color=INK, weight="bold")
    ax.text(0.8, 5.05, "Locally Linear Embedding", fontsize=9, color=INK,
            weight="bold")
    ax.text(9.2, 6.05, "pp. 2323 - 2326", fontsize=6.5, color=INK,
            alpha=0.7, ha="right", va="center")

    for i in range(13):
        y = 4.45 - i * 0.22
        ax.plot([0.8, 4.6], [y, y], color=INK, linewidth=0.5, alpha=0.3)
        ax.plot([5.4, 9.2], [y, y], color=INK, linewidth=0.5, alpha=0.3)

    # --- 쪽 번호 -----------------------------------------------------------
    ax.add_patch(plt.Circle((5.0, 0.95), 0.52, facecolor="white",
                            edgecolor=ACCENT, linewidth=1.3, zorder=5))
    ax.text(5.0, 0.95, "2323", fontsize=9, color=ACCENT, ha="center",
            va="center", weight="bold", zorder=6)

    ax.text(5.0, -0.5, "schematic - column layout not reproduced",
            fontsize=6, color=INK, alpha=0.6, ha="center", style="italic")

    ax.set_xlim(-0.4, 10.4)
    ax.set_ylim(-1.0, 13.4)
    ax.axis("off")
    fig.savefig(HERE / "page_2323.svg", transparent=True, bbox_inches="tight")
    plt.close(fig)


# === 그림으로 보는 AI: 에크만 1954 의 색 실험 =================================
#
# 자극은 파장 하나 — 434 에서 674 nm 까지 곧은 선이다. 그런데 사람의 판단을
# 늘어놓으면 두 축이 필요하고, 두 끝이 되돌아와 거의 만난다.
#
# 원 논문의 14x14 평점 행렬을 구하지 못했으므로 **배치를 셈하지 않았다.**
# 그림 안에 그렇게 적는다 (CLAUDE.md: 지어낸 수를 그림에 넣지 않는다).
# 검증한 것만 글자로 적는다 — 색 14 개, 434-674 nm, 31 명, 0-4 의 다섯 눈금.

def _wl_to_rgb(wl):
    """파장(nm)을 눈에 보이는 색으로 어림한다 (Dan Bruton 의 흔한 어림식)."""
    if wl < 440:
        r, g, b = -(wl - 440) / (440 - 380), 0.0, 1.0
    elif wl < 490:
        r, g, b = 0.0, (wl - 440) / (490 - 440), 1.0
    elif wl < 510:
        r, g, b = 0.0, 1.0, -(wl - 510) / (510 - 490)
    elif wl < 580:
        r, g, b = (wl - 510) / (580 - 510), 1.0, 0.0
    elif wl < 645:
        r, g, b = 1.0, -(wl - 645) / (645 - 580), 0.0
    else:
        r, g, b = 1.0, 0.0, 0.0
    if wl > 700:
        f = 0.3 + 0.7 * (780 - wl) / (780 - 700)
    elif wl < 420:
        f = 0.3 + 0.7 * (wl - 380) / (420 - 380)
    else:
        f = 1.0
    return tuple(min(1.0, max(0.0, c * f)) ** 0.8 for c in (r, g, b))


def ekman_colors():
    import numpy as np

    lo, hi, k = 434, 674, 14
    wls = np.linspace(lo, hi, k)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(11.2, 3.9))

    # --- 1. 물어본 것 ------------------------------------------------------
    ax1.add_patch(plt.Rectangle((0.12, 0.60), 0.33, 0.26,
                                facecolor=_wl_to_rgb(472), edgecolor=INK,
                                linewidth=0.8))
    ax1.add_patch(plt.Rectangle((0.55, 0.60), 0.33, 0.26,
                                facecolor=_wl_to_rgb(504), edgecolor=INK,
                                linewidth=0.8))
    ax1.text(0.5, 0.50, "how similar?", ha="center", fontsize=9.5,
             color=INK, style="italic")
    for i in range(5):
        x = 0.14 + i * 0.18
        ax1.add_patch(plt.Circle((x, 0.33), 0.028, facecolor="white",
                                 edgecolor=INK, linewidth=0.9))
        ax1.text(x, 0.33, str(i), ha="center", va="center", fontsize=7.5,
                 color=INK)
    ax1.text(0.14, 0.22, "no\nsimilarity", ha="center", fontsize=6.8,
             color=INK, alpha=0.8)
    ax1.text(0.86, 0.22, "identical", ha="center", fontsize=6.8,
             color=INK, alpha=0.8)
    ax1.text(0.5, 0.06, "31 subjects, all 91 pairs of 14 colors",
             ha="center", fontsize=8, color=ACCENT)
    ax1.set_title("what was measured", fontsize=10, color=INK)

    # --- 2. 자극은 곧은 선 --------------------------------------------------
    for i, wl in enumerate(np.linspace(lo, hi, 240)):
        ax2.add_patch(plt.Rectangle((i / 240, 0.45), 1 / 240 + 0.002, 0.17,
                                    facecolor=_wl_to_rgb(wl), linewidth=0))
    ax2.add_patch(plt.Rectangle((0, 0.45), 1.0, 0.17, facecolor="none",
                                edgecolor=INK, linewidth=0.8))
    ax2.annotate("", xy=(1.0, 0.33), xytext=(0.0, 0.33),
                 arrowprops=dict(arrowstyle="->", color=INK, linewidth=1.1))
    ax2.text(0.0, 0.24, f"{lo} nm", fontsize=8, color=INK, ha="left")
    ax2.text(1.0, 0.24, f"{hi} nm", fontsize=8, color=INK, ha="right")
    ax2.text(0.5, 0.10, "one number. the ends are\nas far apart as it gets.",
             ha="center", fontsize=8, color=INK, alpha=0.85)
    ax2.set_title("the light", fontsize=10, color=INK)

    # --- 3. 판단은 되돌아온다 -----------------------------------------------
    gap = np.radians(38)
    ang = np.linspace(np.pi / 2 - gap / 2, -3 * np.pi / 2 + gap / 2, k)
    for a, wl in zip(ang, wls):
        ax3.add_patch(plt.Circle((0.5 + 0.33 * np.cos(a),
                                  0.5 + 0.33 * np.sin(a)), 0.052,
                                 facecolor=_wl_to_rgb(wl), edgecolor=INK,
                                 linewidth=0.7))
    ax3.annotate("", xy=(0.5 + 0.33 * np.cos(ang[-1]),
                         0.5 + 0.33 * np.sin(ang[-1]) + 0.085),
                 xytext=(0.5 + 0.33 * np.cos(ang[0]),
                         0.5 + 0.33 * np.sin(ang[0]) + 0.085),
                 arrowprops=dict(arrowstyle="<->", color=ACCENT,
                                 linewidth=1.3,
                                 connectionstyle="arc3,rad=-0.45"))
    ax3.text(0.5, 0.95, "the two ends come back", ha="center", fontsize=8,
             color=ACCENT)
    ax3.text(0.5, 0.47, "two axes\nare needed", ha="center", va="center",
             fontsize=8.5, color=INK, alpha=0.85)
    ax3.text(0.5, 0.02, "schematic - layout not computed\nfrom the original ratings",
             ha="center", fontsize=6.6, color=INK, alpha=0.6, style="italic")
    ax3.set_title("the judgments", fontsize=10, color=INK)

    for ax in (ax1, ax2, ax3):
        ax.set_xlim(-0.03, 1.03)
        ax.set_ylim(0, 1.0)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(HERE / "ekman_colors.svg", transparent=True,
                bbox_inches="tight")
    plt.close(fig)


# === 그림으로 보는 AI: 비좁음 문제와 꼬리가 두꺼운 분포 =======================
#
# m 차원에서 서로 같은 거리에 놓을 수 있는 점은 많아야 m+1 개다 (정단체).
# 그래서 2 차원에서는 셋이 끝이고 10 차원에서는 열하나가 된다. 줄이는 쪽에
# 자리가 모자란다는 것이 비좁음(crowding) 문제다.
# 오른쪽의 두 밀도는 실제로 셈해 그린다.

def crowding():
    import numpy as np

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.0, 4.0))

    # --- 왼쪽: 2 차원에서는 셋이 끝이다 -------------------------------------
    ang = np.radians([90, 210, 330])
    pts = np.c_[np.cos(ang), np.sin(ang)]
    d = [np.linalg.norm(pts[i] - pts[j]) for i, j in ((0, 1), (1, 2), (0, 2))]
    assert np.allclose(d, d[0]), "세 거리가 같아야 한다"

    for i, j in ((0, 1), (1, 2), (0, 2)):
        ax1.plot(*zip(pts[i], pts[j]), color=ACCENT, linewidth=1.8,
                 zorder=1)
    ax1.scatter(pts[:, 0], pts[:, 1], s=130, color=INK, zorder=3)
    ax1.text(0.0, -0.12, f"all three gaps\nequal ({d[0]:.3f})", ha="center",
             va="center", fontsize=8.5, color=ACCENT)
    ax1.text(0, 1.55, "2 dimensions: 3 points at most",
             ha="center", fontsize=10, color=INK, weight="bold")
    ax1.text(0, -1.62,
             "10 dimensions: 11 points can do it.\n"
             "the map has nowhere to put the other 8.",
             ha="center", fontsize=9, color=INK)
    ax1.set_xlim(-1.9, 1.9)
    ax1.set_ylim(-2.1, 1.9)
    ax1.set_aspect("equal")
    ax1.axis("off")

    # --- 오른쪽: 꼬리를 두껍게 ---------------------------------------------
    x = np.linspace(-6, 6, 600)
    gauss = np.exp(-x ** 2 / 2) / np.sqrt(2 * np.pi)
    cauchy = 1 / (np.pi * (1 + x ** 2))        # t, 자유도 1
    ax2.plot(x, gauss, color=INK, linewidth=1.9, label="Gaussian (SNE)")
    ax2.plot(x, cauchy, color=ACCENT, linewidth=2.2,
             label="Student t, 1 d.o.f. (t-SNE)")
    ax2.fill_between(x, gauss, cauchy, where=(cauchy > gauss),
                     color=ACCENT, alpha=0.13)
    ax2.annotate("the heavy tail is\nroom for the crowd",
                 xy=(3.6, cauchy[np.argmin(np.abs(x - 3.6))]),
                 xytext=(1.5, 0.19), fontsize=8.5, color=ACCENT,
                 arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.0))
    ax2.set_xlabel("distance in the map", fontsize=9)
    ax2.set_ylabel("weight given to a pair", fontsize=9)
    ax2.set_title("why a t-distribution, not a Gaussian", fontsize=10,
                  color=INK, weight="bold")
    ax2.legend(fontsize=8.5, framealpha=0.9)
    ax2.set_ylim(0, 0.45)
    ax2.grid(alpha=0.12)
    ax2.tick_params(labelsize=8, colors=INK)
    for s in ax2.spines.values():
        s.set_color(INK)
        s.set_linewidth(0.7)

    fig.tight_layout()
    fig.savefig(HERE / "crowding.svg", transparent=True, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote crowding.svg   (triangle gaps all {d[0]:.6f})")


# === 그림으로 보는 AI: 풀던 문제를 내려가는 문제로 바꾸면 ======================
#
# 왼쪽은 두 수(手)의 꼴만 그린 것이다 — 토거슨은 풀고, 크러스컬은 내려간다.
# 오른쪽은 **실제로 재어** 그린다. 이 쪽의 자료(씨앗 42, 200개의 3차원 점)에
# 같은 stress 를 일곱 군데에서 내려가 보고 닿은 값을 찍는다.

def kruskal_descent():
    import numpy as np
    import torch
    from scipy.spatial.distance import pdist, squareform

    np.random.seed(42)                    # 쪽의 코드와 같은 씨앗
    X = np.random.randn(200, 3)
    D = squareform(pdist(X))

    n = D.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * H @ (D ** 2) @ H
    w, V = np.linalg.eigh(B)
    idx = np.argsort(w)[::-1]
    cl = V[:, idx[:2]] @ np.diag(np.sqrt(np.maximum(w[idx[:2]], 0)))

    D_t = torch.tensor(D, dtype=torch.float32)
    mask = torch.triu(torch.ones_like(D_t), diagonal=1) > 0

    def descend(Y0, steps=600, lr=0.05):
        Y = torch.tensor(Y0, dtype=torch.float32, requires_grad=True)
        opt = torch.optim.Adam([Y], lr=lr)
        for _ in range(steps):
            opt.zero_grad()
            s = ((D_t[mask] - torch.cdist(Y, Y)[mask]) ** 2).sum()
            s.backward()
            opt.step()
        with torch.no_grad():
            return ((D_t[mask] - torch.cdist(Y, Y)[mask]) ** 2).sum().item()

    s_cl = descend(cl)
    rs = np.random.default_rng(0)
    s_rand = [descend(rs.standard_normal((200, 2)) * cl.std())
              for _ in range(6)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 4.0),
                                   gridspec_kw={"width_ratios": [1, 1.25]})

    # --- 왼쪽: 두 수 (꼴만) -------------------------------------------------
    ax1.text(0.5, 0.93, "1952  Torgerson", ha="center", fontsize=9.5,
             color=INK, weight="bold")
    ax1.annotate("", xy=(0.80, 0.78), xytext=(0.20, 0.78),
                 arrowprops=dict(arrowstyle="->", color=INK, linewidth=2.0))
    ax1.text(0.5, 0.70, "solve it. one shot, one answer.",
             ha="center", fontsize=8.2, color=INK, alpha=0.85)

    ax1.text(0.5, 0.50, "1964  Kruskal", ha="center", fontsize=9.5,
             color=ACCENT, weight="bold")
    t = np.linspace(0.04, 0.96, 300)
    land = (0.20 + 0.075 * np.sin(2.1 * np.pi * t + 0.6)
            + 0.050 * np.sin(4.3 * np.pi * t))
    ax1.plot(t, land, color=ACCENT, linewidth=1.8)

    # 공은 반드시 **실제 골짜기**에 놓는다 — 봉우리에 얹으면 그림이 거짓말을 한다
    dips = [i for i in range(1, len(t) - 1)
            if land[i] < land[i - 1] and land[i] < land[i + 1]]
    assert len(dips) >= 2, "골짜기가 둘은 있어야 한다"
    for i, col in zip(dips[:2], (INK, ACCENT)):
        ax1.plot([t[i]], [land[i] + 0.016], marker="o", color=col,
                 markersize=9, markeredgecolor="white", markeredgewidth=1.3,
                 zorder=4)
        top = t[i] - 0.085
        ax1.annotate("", xy=(t[i] - 0.015, land[i] + 0.055),
                     xytext=(top, land[np.argmin(np.abs(t - top))] + 0.085),
                     arrowprops=dict(arrowstyle="->", color=col, lw=1.1,
                                     alpha=0.75))

    ax1.text(0.5, 0.045, "define badness, then walk downhill.\n"
                         "where you start decides where you stop.",
             ha="center", fontsize=8.2, color=INK, alpha=0.85)
    ax1.text(0.95, 0.335, "schematic", ha="right", fontsize=6.5,
             color=INK, alpha=0.5, style="italic")
    ax1.set_xlim(-0.02, 1.0)
    ax1.set_ylim(-0.02, 1.02)
    ax1.axis("off")

    # --- 오른쪽: 실제로 닿은 자리 ------------------------------------------
    ax2.scatter(s_rand, [1] * 6, s=85, color=INK, alpha=0.6, zorder=3,
                label="random start")
    ax2.scatter([s_cl], [1], s=150, color=ACCENT, marker="D", zorder=4,
                label="classical MDS start\n(what the page's code uses)")
    ax2.set_yticks([])
    ax2.set_ylim(0.75, 1.5)
    ax2.set_xlabel("stress reached after 600 Adam steps", fontsize=9)
    ax2.set_title("seven starts, the same stress", fontsize=10, color=INK,
                  weight="bold")
    ax2.annotate(f"{min(s_rand):.0f}", xy=(min(s_rand), 1), xytext=(0, 26),
                 textcoords="offset points", ha="center", fontsize=8.5,
                 color=INK)
    ax2.annotate(f"{max(s_rand):.0f}", xy=(max(s_rand), 1), xytext=(0, 26),
                 textcoords="offset points", ha="center", fontsize=8.5,
                 color=INK)
    ax2.annotate(f"{s_cl:.0f}", xy=(s_cl, 1), xytext=(0, -30),
                 textcoords="offset points", ha="center", fontsize=8.5,
                 color=ACCENT, weight="bold")
    ax2.legend(fontsize=7.8, loc="upper center", framealpha=0.9)
    ax2.grid(alpha=0.13, axis="x")
    ax2.tick_params(labelsize=8, colors=INK)
    for s in ax2.spines.values():
        s.set_color(INK)
        s.set_linewidth(0.7)

    n_better = sum(1 for s in s_rand if s < s_cl)
    fig.text(0.5, -0.04,
             f"{n_better} of the 6 random starts end lower than the "
             f"classical one. measured here, not assumed.",
             ha="center", fontsize=8.5, style="italic", color=INK)
    fig.tight_layout()
    fig.savefig(HERE / "kruskal_descent.svg", transparent=True,
                bbox_inches="tight")
    plt.close(fig)
    print(f"wrote kruskal_descent.svg   classical {s_cl:.3f}   "
          f"random {min(s_rand):.3f}-{max(s_rand):.3f}   better: {n_better}/6")


if __name__ == "__main__":
    page_2323()
    ekman_colors()
    crowding()
    kruskal_descent()
