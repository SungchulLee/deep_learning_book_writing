"""2장 군집화 쪽들의 그림.

규칙은 CLAUDE.md 의 Figures 항에 적어 둔 대로다.
- SVG, fonttype='path', transparent
- 글자는 모두 아스키 (matplotlib 기본 폰트에 한글이 없다)

수는 모두 여기서 실제로 돌려 얻는다. 쪽의 코드와 같은 씨앗, 같은 자료를 쓴다.

실행: python docs/ch02/clustering/figures/make_figures.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["svg.fonttype"] = "path"

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
INK = "#2f3337"
ACCENT = "#8a4b2a"
BLUE = "#2d6a8f"


def style(ax):
    ax.tick_params(labelsize=8, colors=INK)
    for s in ax.spines.values():
        s.set_color(INK)
        s.set_linewidth(0.7)


# === 그림으로 보는 AI: 1996년이 없앤 손잡이와 남긴 손잡이 =====================
#
# 논문은 MinPts 를 2 차원 자료에서 4 로 못박아 손잡이 하나를 없앴다. 이 쪽의
# moons 자료로 재어 보면 그 말이 맞는다 — 2 부터 10 까지 라벨이 똑같다.
# 그런데 Eps 를 고르라고 준 k-dist 그래프는 k 를 키우면 무릎이 밀려 올라간다.

def dbscan_knobs():
    from sklearn.cluster import DBSCAN
    from sklearn.datasets import make_moons
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler

    np.random.seed(42)                       # 쪽의 코드와 같은 씨앗
    X, _ = make_moons(n_samples=1000, noise=0.05, random_state=42)
    X = StandardScaler().fit_transform(X)

    mps = list(range(2, 11))
    nclu, noise = [], []
    for mp in mps:
        lab = DBSCAN(eps=0.2, min_samples=mp).fit(X).labels_
        nclu.append(len(set(lab)) - (1 if -1 in lab else 0))
        noise.append(int((lab == -1).sum()))

    lab4 = DBSCAN(eps=0.2, min_samples=4).fit(X).labels_
    lab5 = DBSCAN(eps=0.2, min_samples=5).fit(X).labels_
    agree = int((lab4 == lab5).sum())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 4.0))

    # --- 왼쪽: MinPts 를 돌려도 꿈쩍하지 않는다 ----------------------------
    ax1.plot(mps, nclu, "o-", color=ACCENT, linewidth=2.0, ms=6,
             label="clusters found")
    ax1.plot(mps, noise, "s--", color=BLUE, linewidth=1.6, ms=5,
             label="points called noise")
    for x, tag, col in ((4, "the paper fixes it here", ACCENT),
                        (5, "sklearn's default,\nwhat this page uses", BLUE)):
        ax1.axvline(x, color=col, linestyle=":", linewidth=1.1, alpha=0.75)
    ax1.annotate("MinPts = 4\nthe paper fixes it here", xy=(4, 2),
                 xytext=(4.25, 2.95), fontsize=8, color=ACCENT,
                 arrowprops=dict(arrowstyle="->", color=ACCENT, lw=0.9))
    ax1.annotate("5 = sklearn default\n(this page)", xy=(5, 0),
                 xytext=(6.1, 0.62), fontsize=8, color=BLUE,
                 arrowprops=dict(arrowstyle="->", color=BLUE, lw=0.9))
    ax1.set_xlabel("min_samples (MinPts)", fontsize=9)
    ax1.set_ylabel("count", fontsize=9)
    ax1.set_ylim(-0.4, 4.2)
    ax1.set_title("one knob the paper removed", fontsize=10, color=INK,
                  weight="bold")
    ax1.legend(fontsize=8, loc="upper right", framealpha=0.9)
    ax1.grid(alpha=0.13)
    style(ax1)
    ax1.text(0.5, -0.30,
             f"4 and 5 label all {agree} points identically",
             ha="center", fontsize=8.5, color=INK, alpha=0.85,
             transform=ax1.transAxes)

    # --- 오른쪽: Eps 를 고르라고 준 그래프는 k 에 따라 움직인다 -------------
    knees = {}
    for k, col, lw in ((3, "#b9b2a6", 1.2), (4, ACCENT, 2.3),
                       (5, BLUE, 1.7), (6, "#7f9aa8", 1.2),
                       (8, "#a8b5a0", 1.2), (10, "#9a8f80", 1.2)):
        nn = NearestNeighbors(n_neighbors=k).fit(X)
        d, _ = nn.kneighbors(X)
        c = np.sort(d[:, -1])
        knees[k] = c[int(0.95 * len(X))]
        ax2.plot(c, color=col, linewidth=lw, label=f"k = {k}")
    ax2.axhline(0.2, color=INK, linestyle="--", linewidth=1.3)
    ax2.text(20, 0.207, "eps = 0.2, what the page picks", fontsize=8,
             color=INK)
    ax2.annotate(f"the knee drifts with k:\n"
                 f"{knees[4]:.3f} at k=4, {knees[10]:.3f} at k=10",
                 xy=(962, knees[10]), xytext=(215, 0.158), fontsize=8.5,
                 color=INK,
                 arrowprops=dict(arrowstyle="->", color=INK, lw=0.9))
    ax2.set_xlabel("points, sorted by k-th nearest distance", fontsize=9)
    ax2.set_ylabel("distance to k-th neighbour", fontsize=9)
    ax2.set_title("the knob it left behind", fontsize=10, color=INK,
                  weight="bold")
    ax2.legend(fontsize=7.6, loc="upper left", framealpha=0.9, ncol=2)
    ax2.grid(alpha=0.13)
    style(ax2)

    fig.tight_layout()
    fig.savefig(HERE / "dbscan_knobs.svg", transparent=True,
                bbox_inches="tight")
    plt.close(fig)
    print(f"wrote dbscan_knobs.svg   agree {agree}/1000   "
          f"knee {knees[4]:.4f} -> {knees[10]:.4f}")


# === 그림으로 보는 AI: 이름을 준 알고리즘과 실제로 도는 알고리즘 ==============
#
# MacQueen 1967 이 "k-means" 라는 이름을 지었지만 그의 것은 점 하나마다 중심을
# 곧바로 고치는 온라인 방식이다. 그래서 **점의 순서를 탄다.** 같은 초기 중심을
# 주고 순서만 섞어 재면 답이 크게 흔들린다. Lloyd 의 일괄 방식은 꿈쩍 않는다.

def _macqueen(X, k, init_seed, passes, order_seed):
    """MacQueen 1967 의 온라인 갱신. 중심은 달리는 평균으로 고친다."""
    rs = np.random.default_rng(init_seed)
    C = X[rs.choice(len(X), k, replace=False)].astype(float).copy()
    n = np.ones(k)
    for p in range(passes):
        order = np.random.default_rng(order_seed + p).permutation(len(X))
        for i in order:
            x = X[i]
            j = int(((C - x) ** 2).sum(1).argmin())
            n[j] += 1
            C[j] += (x - C[j]) / n[j]
    return C


def _inertia(X, C):
    return ((X[:, None, :] - C[None, :, :]) ** 2).sum(-1).min(1).sum()


def kmeans_two_parents():
    from sklearn.cluster import KMeans
    from sklearn.datasets import make_blobs

    np.random.seed(42)                   # 쪽의 코드와 같은 씨앗
    X, _ = make_blobs(n_samples=500, centers=5,
                      cluster_std=[1.0, 0.7, 1.2, 0.9, 1.5], random_state=42)
    K = 5

    lloyd = KMeans(n_clusters=K, random_state=42, n_init=10).fit(X).inertia_

    N = 40                                # 섞는 횟수
    one = [_inertia(X, _macqueen(X, K, 0, 1, 1000 + 7 * s)) for s in range(N)]

    passes = [1, 2, 5, 10, 20]
    lo, hi = [], []
    for p in passes:
        v = [_inertia(X, _macqueen(X, K, 0, p, 1000 + 7 * s))
             for s in range(12)]
        lo.append(min(v))
        hi.append(max(v))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 4.0))

    # --- 왼쪽: 순서만 바꿔도 답이 흔들린다 ---------------------------------
    ax1.hist(one, bins=18, color=ACCENT, alpha=0.75, edgecolor="white",
             linewidth=0.6)
    ax1.axvline(lloyd, color=INK, linewidth=2.0)
    ax1.annotate(f"Lloyd: {lloyd:.0f}\nsame for every ordering",
                 xy=(lloyd, N * 0.17), xytext=(lloyd + 230, N * 0.26),
                 fontsize=8.5, color=INK,
                 arrowprops=dict(arrowstyle="->", color=INK, lw=0.9))
    ax1.set_xlabel("inertia after one pass", fontsize=9)
    ax1.set_ylabel(f"how many of {N} orderings", fontsize=9)
    ax1.set_title("same starting centres, shuffled order", fontsize=10,
                  color=INK, weight="bold")
    ax1.grid(alpha=0.13)
    style(ax1)

    # --- 오른쪽: 더 훑어도 폭이 줄지 않는다 --------------------------------
    ax2.fill_between(passes, lo, hi, color=ACCENT, alpha=0.22,
                     label="MacQueen, range over 12 orderings")
    ax2.plot(passes, lo, "o-", color=ACCENT, linewidth=1.8, ms=5)
    ax2.plot(passes, hi, "o-", color=ACCENT, linewidth=1.8, ms=5)
    ax2.axhline(lloyd, color=INK, linewidth=2.0, label=f"Lloyd ({lloyd:.0f})")
    ax2.set_xlabel("passes over the 500 points", fontsize=9)
    ax2.set_ylabel("inertia", fontsize=9)
    ax2.set_title("more passes do not close the spread", fontsize=10,
                  color=INK, weight="bold")
    ax2.set_xticks(passes)
    ax2.legend(fontsize=8, loc="center right", framealpha=0.9)
    ax2.grid(alpha=0.13)
    style(ax2)

    fig.text(0.5, -0.04,
             "the best MacQueen runs reach Lloyd's answer; the worst never "
             "do, and nothing but the order of the points separates them.",
             ha="center", fontsize=8.5, style="italic", color=INK)
    fig.tight_layout()
    fig.savefig(HERE / "kmeans_two_parents.svg", transparent=True,
                bbox_inches="tight")
    plt.close(fig)
    print(f"wrote kmeans_two_parents.svg   lloyd {lloyd:.2f}   "
          f"macqueen 1 pass {min(one):.0f}-{max(one):.0f}")


if __name__ == "__main__":
    dbscan_knobs()
    kmeans_two_parents()
