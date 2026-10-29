"""2장 장 소개의 그림.

규칙은 CLAUDE.md 의 Figures 항에 적어 둔 대로다.
- SVG, fonttype='path', transparent
- 글자는 모두 아스키 (matplotlib 기본 폰트에 한글이 없다)

이 그림은 꼴만 그린다 — 새뮤얼 프로그램의 성질을 세 갈래에 견주어 놓을 뿐
수를 재지 않는다. 그림 안에 그렇게 적는다.

실행: python docs/ch02/figures/make_figures.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["svg.fonttype"] = "path"

import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
INK = "#2f3337"
ACCENT = "#8a4b2a"
BLUE = "#2d6a8f"
GREY = "#8c8880"


def samuel_straddles():
    fig, ax = plt.subplots(figsize=(9.6, 4.6))

    cols = [(1.7, "supervised", "an answer key\nfor every example"),
            (5.0, "unsupervised", "no answer key,\nfind the structure"),
            (8.3, "reinforcement", "no answer key,\nonly later reward")]
    for x, name, what in cols:
        ax.add_patch(plt.Rectangle((x - 1.35, 3.25), 2.7, 1.25,
                                   facecolor="white", edgecolor=INK,
                                   linewidth=1.1))
        ax.text(x, 4.19, name, ha="center", fontsize=10.5, weight="bold",
                color=INK)
        ax.text(x, 3.72, what, ha="center", va="center", fontsize=8.2,
                color=INK, alpha=0.85)

    traits = [
        (1.7, "kept a table of\npositions it had seen\n(rote learning)"),
        (5.0, "was given no labelled\ngames to copy"),
        (8.3, "learned from the\noutcome of its own\nself-play"),
    ]
    for x, t in traits:
        ax.annotate("", xy=(x, 3.18), xytext=(x, 2.35),
                    arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.4))
        ax.text(x, 2.22, t, ha="center", va="top", fontsize=8.2,
                color=ACCENT)

    ax.add_patch(plt.Rectangle((0.2, 0.42), 9.6, 0.92, facecolor=ACCENT,
                               alpha=0.12, edgecolor=ACCENT, linewidth=1.1))
    ax.text(5.0, 0.88, "Samuel's checkers program, 1959 - one program, "
                       "all three columns",
            ha="center", va="center", fontsize=10, color=INK, weight="bold")

    ax.text(5.0, 0.05, "schematic - the three columns are a teaching split, "
                       "drawn here to show what it does not contain",
            ha="center", fontsize=7, color=INK, alpha=0.6, style="italic")

    ax.set_xlim(0, 10)
    ax.set_ylim(-0.15, 4.75)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(HERE / "samuel_straddles.svg", transparent=True,
                bbox_inches="tight")
    plt.close(fig)
    print("wrote samuel_straddles.svg")


if __name__ == "__main__":
    samuel_straddles()
