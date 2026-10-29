"""19.1 확률 그래프 모델 기초 쪽들의 그림.

규칙은 CLAUDE.md 의 Figures 항에 적어 둔 대로다.
- SVG, fonttype='path', transparent
- 글자는 모두 아스키 (matplotlib 기본 폰트에 한글이 없다)

자료를 읽지 않으므로 어디서 돌려도 된다.

실행: python docs/ch18/pgm_foundations/figures/make_figures.py
"""
import itertools
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

# 4-고리 위의 고전적 반례. 균등하게 8 개 배치에만 질량을 둔다.
SUPPORT = [(0, 0, 0, 0), (1, 0, 0, 0), (1, 1, 0, 0), (1, 1, 1, 0),
           (1, 1, 1, 1), (0, 1, 1, 1), (0, 0, 1, 1), (0, 0, 0, 1)]
EDGES = [(0, 1), (1, 2), (2, 3), (3, 0)]


def _joint():
    P = np.zeros((2, 2, 2, 2))
    for s in SUPPORT:
        P[s] = 1 / 8
    return P


def _ci_violation(P, a, b, g1, g2):
    """a ⟂ b | g1,g2 가 깨지는 최대 폭. 직접 열거한다."""
    worst = 0.0
    for v1, v2 in itertools.product((0, 1), (0, 1)):
        tab = np.zeros((2, 2))
        for xa, xb in itertools.product((0, 1), (0, 1)):
            idx = [None] * 4
            idx[a], idx[b], idx[g1], idx[g2] = xa, xb, v1, v2
            tab[xa, xb] = P[tuple(idx)]
        if tab.sum() <= 0:
            continue
        tab = tab / tab.sum()
        worst = max(worst, np.abs(tab - np.outer(tab.sum(1),
                                                 tab.sum(0))).max())
    return worst


# === 그림으로 보는 AI: 양수성이 가정이 아니라 벽인 까닭 ========================
#
# 이 분포는 4-고리의 마르코프 성질을 **정확히** 만족한다(어긋남 0).
# 그런데 클리크 인수분해로는 적을 수 없다. 까닭이 세는 것만으로 나온다 —
# 네 변이 저마다 2x2 네 쌍을 **전부** 쓰므로 어떤 psi 도 0 일 수 없고,
# 그러면 곱이 어디서나 양수가 되어 0 인 칸 여덟을 만들 수 없다.
#
# 그림에 적는 수는 모두 여기서 셈하며, 주장마다 assert 로 확인하고 그린다.

def positivity_wall():
    P = _joint()

    # --- 주장 1: 마르코프 성질이 정확히 성립한다 ---------------------------
    v1 = _ci_violation(P, 0, 2, 1, 3)
    v2 = _ci_violation(P, 1, 3, 0, 2)
    assert max(v1, v2) < 1e-12, f"마르코프 성질이 깨진다: {max(v1, v2)}"

    # --- 주장 2: 네 변이 저마다 네 쌍을 전부 쓴다 --------------------------
    used = {e: set() for e in EDGES}
    for x in SUPPORT:
        for e in EDGES:
            used[e].add((x[e[0]], x[e[1]]))
    assert all(len(u) == 4 for u in used.values()), "네 쌍을 다 쓰지 않는다"

    # --- 주장 3: 열여섯 칸 가운데 여덟이 0 이다 ----------------------------
    n_zero = int((P == 0).sum())
    assert n_zero == 8, f"0 인 칸이 8 이 아니다: {n_zero}"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.8, 4.3),
                                   gridspec_kw={"width_ratios": [1, 1.25]})

    # --- 왼쪽: 고리와 열여섯 칸 -------------------------------------------
    pos = {0: (0.30, 0.80), 1: (0.70, 0.80), 2: (0.70, 0.52), 3: (0.30, 0.52)}
    for a, b in EDGES:
        ax1.plot(*zip(pos[a], pos[b]), color=INK, linewidth=1.6, zorder=1)
    for i, (x, y) in pos.items():
        ax1.add_patch(plt.Circle((x, y), 0.062, facecolor="white",
                                 edgecolor=INK, linewidth=1.5, zorder=3))
        ax1.text(x, y, f"X{i+1}", ha="center", va="center", fontsize=9.5,
                 color=INK, zorder=4)
    ax1.text(0.5, 0.93, "a 4-cycle, all binary", ha="center", fontsize=9.2,
             color=INK, style="italic")

    for k, cfg in enumerate(sorted(itertools.product((0, 1), repeat=4))):
        r, c = divmod(k, 4)
        x0, y0 = 0.10 + c * 0.21, 0.30 - r * 0.082
        on = P[cfg] > 0
        ax1.add_patch(plt.Rectangle((x0, y0), 0.185, 0.068,
                                    facecolor=ACCENT if on else "white",
                                    alpha=0.30 if on else 1.0,
                                    edgecolor=INK if on else "#c8c8c8",
                                    linewidth=0.9))
        ax1.text(x0 + 0.0925, y0 + 0.034, "".join(map(str, cfg)),
                 ha="center", va="center", fontsize=7.6,
                 color=INK if on else "#b4b4b4", family="monospace")
    ax1.text(0.5, 0.405, f"{16 - n_zero} of 16 carry mass,  "
                         f"{n_zero} are exactly zero",
             ha="center", fontsize=8.4, color=ACCENT)
    ax1.text(0.5, -0.045, "so the positivity condition fails", ha="center",
             fontsize=8.6, color=INK)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(-0.09, 1.0)
    ax1.axis("off")

    # --- 오른쪽: 세기만으로 끝나는 논증 ------------------------------------
    ax2.text(0.5, 0.95, "every edge uses all four of its pairs",
             ha="center", fontsize=9.6, color=INK, weight="bold")
    for k, e in enumerate(EDGES):
        y = 0.80 - k * 0.088
        ax2.text(0.10, y, f"psi(X{e[0]+1}, X{e[1]+1})", fontsize=9,
                 color=INK, family="monospace", va="center")
        for j, pair in enumerate(sorted(used[e])):
            ax2.add_patch(plt.Rectangle((0.48 + j * 0.105, y - 0.028), 0.088,
                                        0.056, facecolor=ACCENT, alpha=0.28,
                                        edgecolor=ACCENT, linewidth=0.8))
            ax2.text(0.524 + j * 0.105, y, f"{pair[0]}{pair[1]}",
                     ha="center", va="center", fontsize=7.6, color=INK,
                     family="monospace")
        ax2.text(0.93, y, "4/4", fontsize=8.2, color=ACCENT, va="center")

    steps = [
        "none of the four psi can be zero anywhere",
        "so the product is positive everywhere",
        "so it cannot be zero on the other eight",
    ]
    for k, s in enumerate(steps):
        ax2.text(0.10, 0.37 - k * 0.085, f"{k+1}.  {s}", fontsize=8.8,
                 color=INK, va="center")
    ax2.plot([0.08, 0.96], [0.075, 0.075], color=ACCENT, linewidth=1.2)
    ax2.text(0.5, 0.012, "no clique factorization exists -- "
                         "yet the Markov property holds exactly",
             ha="center", fontsize=8.8, color=ACCENT, weight="bold")
    ax2.set_xlim(0, 1)
    ax2.set_ylim(-0.03, 1.02)
    ax2.axis("off")

    fig.tight_layout()
    fig.savefig(HERE / "positivity_wall.svg", transparent=True,
                bbox_inches="tight")
    plt.close(fig)
    print(f"wrote positivity_wall.svg   markov violation "
          f"{max(v1, v2):.1e}   zeros {n_zero}/16")


if __name__ == "__main__":
    positivity_wall()
