"""2장 학습 패러다임 쪽들의 그림.

규칙은 CLAUDE.md 의 Figures 항에 적어 둔 대로다.
- SVG, fonttype='path', transparent
- 글자는 모두 아스키 (matplotlib 기본 폰트에 한글이 없다)

실행: python docs/ch02/paradigms/figures/make_figures.py
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


# === 그림으로 보는 AI: 널리 전해지는 이야기와 날짜 ============================
#
# 벨먼은 자서전에서 국방장관 윌슨의 눈을 피하려 "동적 계획법"이라는 이름을
# 골랐다고 적었다. 그런데 그 이름을 쓴 논문이 윌슨의 취임보다 먼저 나왔다.
# 날짜만 나란히 놓는다 — 어느 쪽이 옳은지 판정하지 않는다.

def bellman_dates():
    fig, ax = plt.subplots(figsize=(9.2, 2.9))

    ax.annotate("", xy=(10.0, 1.0), xytext=(0.4, 1.0),
                arrowprops=dict(arrowstyle="->", color=INK, linewidth=1.3))

    marks = [
        (2.4, "15 August 1952", ACCENT, 1,
         "Bellman, PNAS 38(8) 716-719\n"
         '"On the Theory of\nDynamic Programming"'),
        (7.0, "28 January 1953", BLUE, -1,
         "Charles E. Wilson becomes\nSecretary of Defense\n"
         "(serves to October 1957)"),
    ]
    for x, when, col, side, what in marks:
        ax.plot([x], [1.0], marker="o", color=col, markersize=10,
                markeredgecolor="white", markeredgewidth=1.6, zorder=5)
        ax.text(x, 1.0 + side * 0.17, when, ha="center", fontsize=9.5,
                weight="bold", color=col,
                va="bottom" if side > 0 else "top")
        ax.text(x, 1.0 + side * 0.40, what, ha="center", fontsize=8,
                color=INK, alpha=0.9,
                va="bottom" if side > 0 else "top")

    ax.annotate("", xy=(7.0, 0.72), xytext=(2.4, 0.72),
                arrowprops=dict(arrowstyle="<->", color=INK, lw=1.0,
                                alpha=0.6))
    ax.text(4.7, 0.59, "the name is already in print, five months early",
            ha="center", fontsize=8.5, color=INK, style="italic")

    ax.set_xlim(0, 10.4)
    ax.set_ylim(0.1, 1.95)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(HERE / "bellman_dates.svg", transparent=True,
                bbox_inches="tight")
    plt.close(fig)
    print("wrote bellman_dates.svg")


if __name__ == "__main__":
    bellman_dates()
