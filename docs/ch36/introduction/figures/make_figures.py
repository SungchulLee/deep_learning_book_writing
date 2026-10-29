"""37.0 들어가며 쪽들의 그림.

규칙은 CLAUDE.md 의 Figures 항에 적어 둔 대로다.
- SVG, fonttype='path', transparent
- 글자는 모두 아스키 (matplotlib 기본 폰트에 한글이 없다)

자료를 읽지 않으므로 어디서 돌려도 된다.

실행: python docs/ch36/introduction/figures/make_figures.py
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
MUTED = "#2d6a8f"

N = 401


def _pareto(pts):
    return [i for i, p in enumerate(pts)
            if not np.any((pts[:, 0] >= p[0]) & (pts[:, 1] >= p[1])
                          & ((pts[:, 0] > p[0]) | (pts[:, 1] > p[1])))]


def _reachable(f1, f2):
    """어떤 가중치 w 로든 w*f1 + (1-w)*f2 를 최대화해 뽑히는 점."""
    return {int(np.argmax(w * f1 + (1 - w) * f2))
            for w in np.linspace(0, 1, 4001)}


# === 그림으로 보는 AI: 저울 하나로 닿지 못하는 자리 ===========================
#
# 두 목표를 동시에 높이고 싶을 때, **가중합** 으로 고를 수 있는 것은 도달 가능
# 집합의 볼록 껍질 위에 있는 점뿐이다. 안쪽으로 패인 구간의 파레토 최적점은
# 어떤 가중치로도 뽑히지 않는다. 오른쪽에 볼록한 경우를 대조로 둔다.

def scalar_reach():
    t = np.linspace(0, 1, N)
    dented = (t, 1 - t + 0.25 * np.sin(2 * np.pi * t))
    convex = (t, np.sqrt(np.clip(1 - t ** 2, 0, None)))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 4.3), sharey=True)
    out = {}
    for ax, (f1, f2), title in ((ax1, dented, "a dented frontier"),
                                (ax2, convex, "a convex one, for contrast")):
        pts = np.c_[f1, f2]
        pf = _pareto(pts)
        reach = set(pf) & _reachable(f1, f2)
        miss = sorted(set(pf) - reach)
        out[title] = (len(pf), len(reach), len(miss))

        ax.plot(f1, f2, color=INK, linewidth=1.0, alpha=0.30)
        if miss:
            ax.plot(f1[miss], f2[miss], "o", color=INK, ms=3.0, alpha=0.40,
                    label=f"never chosen  ({len(miss)})")
        rl = sorted(reach)
        ax.plot(f1[rl], f2[rl], "o", color=ACCENT, ms=4.2,
                label=f"chosen by some weighting  ({len(reach)})")
        ax.set_title(title, fontsize=10, color=INK, weight="bold")
        ax.set_xlabel("objective 1", fontsize=9)
        ax.legend(fontsize=8.0, loc="lower left", framealpha=0.92)
        ax.grid(alpha=0.13)
        ax.tick_params(labelsize=8, colors=INK)
        for s in ax.spines.values():
            s.set_color(INK)
            s.set_linewidth(0.7)
    ax1.set_ylabel("objective 2", fontsize=9)

    n_pf, n_re, n_mi = out["a dented frontier"]
    assert n_mi > 0, "패인 쪽에서 못 뽑는 점이 없다"
    assert out["a convex one, for contrast"][2] == 0, "볼록한 쪽에 구멍이 있다"

    f1, f2 = dented
    pts = np.c_[f1, f2]
    miss = sorted(set(_pareto(pts)) - set(_pareto(pts)).intersection(
        _reachable(f1, f2)))
    ax1.annotate("every one of these is Pareto-optimal\n"
                 "and no weighting reaches it",
                 xy=(f1[miss[len(miss) // 2]], f2[miss[len(miss) // 2]]),
                 xytext=(0.30, 0.18), fontsize=8.6, color=INK,
                 arrowprops=dict(arrowstyle="->", color=INK, lw=1.0))

    fig.text(0.5, -0.04,
             "a weighted sum can only ever pick points on the convex hull. "
             "a cleverer scalar could reach the rest - but not by weights.",
             ha="center", fontsize=8.6, style="italic", color=INK)
    fig.tight_layout()
    fig.savefig(HERE / "scalar_reach.svg", transparent=True,
                bbox_inches="tight")
    plt.close(fig)
    print(f"wrote scalar_reach.svg   dented: pareto {n_pf}, reachable "
          f"{n_re}, missed {n_mi}")


if __name__ == "__main__":
    scalar_reach()
