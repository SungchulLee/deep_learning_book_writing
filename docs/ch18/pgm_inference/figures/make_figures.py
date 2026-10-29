"""19.2 추론 쪽들의 그림.

규칙은 CLAUDE.md 의 Figures 항에 적어 둔 대로다.
- SVG, fonttype='path', transparent
- 글자는 모두 아스키 (matplotlib 기본 폰트에 한글이 없다)

자료를 읽지 않으므로 어디서 돌려도 된다.

실행: python docs/ch18/pgm_inference/figures/make_figures.py
"""
import itertools
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["svg.fonttype"] = "path"

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
INK = "#2f3337"
ACCENT = "#8a4b2a"
MUTED = "#2d6a8f"

R, C = 5, 5            # 격자 크기
N_RANDOM = 3000        # 무작위로 시험해 볼 순서의 수


def _grid(r, c):
    adj = {(i, j): set() for i in range(r) for j in range(c)}
    for i in range(r):
        for j in range(c):
            if i + 1 < r:
                adj[(i, j)].add((i + 1, j)); adj[(i + 1, j)].add((i, j))
            if j + 1 < c:
                adj[(i, j)].add((i, j + 1)); adj[(i, j + 1)].add((i, j))
    return adj


def _induced_width(adj, order):
    """그 순서로 제거할 때의 유도 폭. 이웃을 서로 이어 가며 잰다."""
    g = {v: set(ns) for v, ns in adj.items()}
    w = 0
    for v in order:
        nb = g[v]
        w = max(w, len(nb))
        for a, b in itertools.combinations(nb, 2):
            g[a].add(b); g[b].add(a)
        for u in nb:
            g[u].discard(v)
        del g[v]
    return w


def _min_degree(adj):
    g = {v: set(ns) for v, ns in adj.items()}
    order = []
    while g:
        v = min(g, key=lambda u: len(g[u]))
        order.append(v)
        nb = g[v]
        for a, b in itertools.combinations(nb, 2):
            g[a].add(b); g[b].add(a)
        for u in nb:
            g[u].discard(v)
        del g[v]
    return order


# === 그림으로 보는 AI: 순서가 값을 가르지만, 바닥은 못 내린다 ==================
#
# 5x5 격자에서 제거 순서를 바꿔 가며 유도 폭을 잰다. 최소 차수 휴리스틱은
# 폭 5 에 닿고, 무작위 순서는 13 까지 올라간다 (표 크기로 256 배).
# 그런데 **아무리 잘 골라도 5 아래로는 못 내려간다** — 구조가 바닥을 정한다.

def elimination_order():
    adj = _grid(R, C)
    w_md = _induced_width(adj, _min_degree(adj))

    rng = random.Random(0)
    ws = []
    for _ in range(N_RANDOM):
        o = list(adj)
        rng.shuffle(o)
        ws.append(_induced_width(adj, o))
    w_lo, w_hi = min(ws), max(ws)

    assert w_md <= w_lo, f"휴리스틱이 무작위 최선보다 나쁘다: {w_md} > {w_lo}"
    assert w_hi >= w_md + 4, f"순서에 따른 차이가 작다: {w_hi} vs {w_md}"

    ratio = 2 ** (w_hi + 1) / 2 ** (w_md + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 4.2),
                                   gridspec_kw={"width_ratios": [1, 1.3]})

    # --- 왼쪽: 격자 --------------------------------------------------------
    for (i, j), nbs in adj.items():
        for (a, b) in nbs:
            ax1.plot([j, b], [-i, -a], color=INK, linewidth=1.0, alpha=0.45,
                     zorder=1)
    for (i, j) in adj:
        ax1.add_patch(plt.Circle((j, -i), 0.17, facecolor="white",
                                 edgecolor=INK, linewidth=1.2, zorder=3))
    ax1.set_title(f"{R} x {C} grid,  {R*C} binary variables", fontsize=10,
                  color=INK, weight="bold")
    ax1.text(0.5, -0.09, "every ordering computes the same answer",
             transform=ax1.transAxes, ha="center", fontsize=8.6, color=INK)
    ax1.set_aspect("equal")
    ax1.set_xlim(-0.6, C - 0.4)
    ax1.set_ylim(-R + 0.4, 0.6)
    ax1.axis("off")

    # --- 오른쪽: 순서마다의 유도 폭 ----------------------------------------
    bins = np.arange(w_lo - 0.5, w_hi + 1.5, 1)
    ax2.hist(ws, bins=bins, color=INK, alpha=0.35, edgecolor=INK,
             linewidth=0.6)
    ax2.axvline(w_md, color=ACCENT, linewidth=2.2)
    ax2.annotate(f"min-degree\nheuristic: {w_md}", xy=(w_md, 0),
                 xytext=(w_md + 0.45, ax2.get_ylim()[1] * 0.74),
                 fontsize=8.8, color=ACCENT,
                 arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.0))
    ax2.annotate(f"worst seen: {w_hi}", xy=(w_hi, 0),
                 xytext=(w_hi - 0.25, ax2.get_ylim()[1] * 0.30),
                 fontsize=8.8, color=INK, ha="right",
                 arrowprops=dict(arrowstyle="->", color=INK, lw=1.0))
    ax2.set_xlabel("induced width of the elimination order", fontsize=9)
    ax2.set_ylabel(f"how many of {N_RANDOM:,} random orders", fontsize=9)
    ax2.set_title(f"the table is {ratio:,.0f}x bigger at the worst",
                  fontsize=10, color=INK, weight="bold")
    ax2.grid(alpha=0.13, axis="y")
    ax2.tick_params(labelsize=8, colors=INK)
    for s in ax2.spines.values():
        s.set_color(INK)
        s.set_linewidth(0.7)

    fig.text(0.5, -0.04,
             f"but nothing gets below {w_md}. the grid itself sets that floor, "
             "and no ordering lowers it.",
             ha="center", fontsize=8.8, style="italic", color=ACCENT)
    fig.tight_layout()
    fig.savefig(HERE / "elimination_order.svg", transparent=True,
                bbox_inches="tight")
    plt.close(fig)
    print(f"wrote elimination_order.svg   min-degree {w_md}   "
          f"random {w_lo}-{w_hi}   ratio {ratio:,.0f}x")


if __name__ == "__main__":
    elimination_order()
