"""32장 장 소개의 그림.

규칙은 CLAUDE.md 의 Figures 항에 적어 둔 대로다.
- SVG, fonttype='path', transparent
- 글자는 모두 아스키 (matplotlib 기본 폰트에 한글이 없다)

자료를 읽지 않으므로 어디서 돌려도 된다.

실행: python docs/ch31/figures/make_figures.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["svg.fonttype"] = "path"

import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
INK = "#2f3337"
ACCENT = "#8a4b2a"
MUTED = "#2d6a8f"


# === 그림으로 보는 AI: 이 장의 연표는 물리에서 시작한다 =======================
#
# 적는 해는 모두 이 장의 개별 쪽에서 원문을 확인한 것만 쓴다.
# 파랑 = 물리, 호박 = 기계학습. 앞의 둘이 물리이고 사이가 비어 있다.

EVENTS = [
    (1925, "Ising", "one dimension:\nno transition", "phys"),
    (1944, "Onsager", "two dimensions:\nthere is one", "phys"),
    (1982, "Hopfield", "memories as\nenergy minima", "ml"),
    (1985, "Ackley, Hinton,\nSejnowski", "Boltzmann\nmachine", "ml"),
    (1986, "Smolensky", "Harmonium\n(the RBM)", "ml"),
    (2002, "Hinton", "contrastive\ndivergence", "ml"),
    (2005, "Hyvarinen", "score\nmatching", "ml"),
]


def lineage():
    # 1982/1985/1986 이 몰려 있어 **실제 연도 축**에서는 글자가 겹친다.
    # 그래서 칸을 고르게 두고 연도를 글자로 적되, 비어 있는 38년은
    # 가운데를 벌려 따로 표시한다.
    xs = [0.0, 1.0, 3.3, 4.3, 5.3, 6.6, 7.6]
    assert len(xs) == len(EVENTS)

    fig, ax = plt.subplots(figsize=(10.8, 4.2))
    ax.annotate("", xy=(xs[-1] + 0.75, 0), xytext=(xs[0] - 0.75, 0),
                arrowprops=dict(arrowstyle="->", color=INK, linewidth=1.3))

    for k, ((yr, who, what, kind), x) in enumerate(zip(EVENTS, xs)):
        col = MUTED if kind == "phys" else ACCENT
        up = k % 2 == 0
        sgn = 1.0 if up else -1.0
        ax.plot([x, x], [0, sgn * 0.16], color=col, linewidth=1.0, alpha=0.6)
        ax.plot([x], [0], marker="o", color=col, markersize=8,
                markeredgecolor="white", markeredgewidth=1.4, zorder=5)
        va = "bottom" if up else "top"
        ax.text(x, sgn * 0.20, str(yr), ha="center", va=va, fontsize=9.6,
                color=col, weight="bold")
        ax.text(x, sgn * 0.40, who, ha="center", va=va, fontsize=8.4,
                color=INK)
        ax.text(x, sgn * (0.62 if up else 0.70), what, ha="center", va=va,
                fontsize=7.8, color=INK, alpha=0.82)

    # 비어 있는 38 년
    mid = (xs[1] + xs[2]) / 2
    ax.annotate("", xy=(xs[2] - 0.30, 0), xytext=(xs[1] + 0.30, 0),
                arrowprops=dict(arrowstyle="<->", color=INK, linewidth=1.0,
                                alpha=0.5))
    ax.text(mid, 0.075, "38 years", ha="center", fontsize=8.4, color=INK,
            alpha=0.75, style="italic")

    ax.text(0.5, 1.30, "physics", fontsize=10.5, color=MUTED, weight="bold",
            ha="center")
    ax.text(5.45, 1.30, "machine learning", fontsize=10.5, color=ACCENT,
            weight="bold", ha="center")

    ax.set_xlim(xs[0] - 1.0, xs[-1] + 1.0)
    ax.set_ylim(-1.25, 1.45)
    ax.axis("off")

    fig.text(0.5, 0.005,
             "spacing is even, not to scale - every date here was checked "
             "against the source on this chapter's own pages.",
             ha="center", fontsize=8.4, style="italic", color=INK)
    fig.tight_layout()
    fig.savefig(HERE / "lineage.svg", transparent=True, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote lineage.svg   {EVENTS[0][0]} - {EVENTS[-1][0]}")


if __name__ == "__main__":
    lineage()
