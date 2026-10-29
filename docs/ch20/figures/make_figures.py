"""21장 장 소개의 그림.

규칙은 CLAUDE.md 의 Figures 항에 적어 둔 대로다.
- SVG, fonttype='path', transparent
- 글자는 모두 아스키 (matplotlib 기본 폰트에 한글이 없다)

여기 적힌 연도는 모두 이 장의 다른 쪽 상자에서 원문을 확인한 것들이다.
확인하지 않은 연도는 넣지 않는다.

가로 자리는 **차례**이지 연도에 비례하지 않는다 — 1774 와 1907 사이가
비면 나머지가 다 뭉개진다. 그림 안에 그렇게 적는다.

실행: python docs/ch20/figures/make_figures.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["svg.fonttype"] = "path"

import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
INK = "#2f3337"
ACCENT = "#8a4b2a"
COOL = "#2d6a8f"

# (연도, 누가, 무엇, 이 장의 어느 쪽에서 확인했는가)
EVENTS = [
    (1774, "Laplace", "first asymptotic\napproximation to\na posterior"),
    (1907, "Weiss", "the molecular field:\nneighbours replaced\nby their average"),
    (1977, "Dempster, Laird\nand Rubin", "names EM, and\nproves it for GEM"),
    (1990, "Hecht-Nielsen", "weight-space\nsymmetries form\na group"),
    (1992, "MacKay", "the evidence as\na practical framework\nfor networks"),
    (1993, "Hinton and\nVan Camp", "weight noise as\ndescription length"),
    (1995, "Dayan, Hinton,\nNeal and Zemel", "an explicit\nrecognition model"),
    (1995, "Neal", "infinite width\nis a Gaussian\nprocess"),
    (1998, "Neal and Hinton", "EM is coordinate\nascent on one\nfunction"),
    (1999, "Jordan, Ghahramani,\nJaakkola and Saul", "variational methods\nfrom convex duality"),
    (2013, "Kingma and\nWelling", "reparameterize,\nthen differentiate"),
    (2015, "Blundell et al.", "Bayes by Backprop"),
]


def timeline():
    n = len(EVENTS)
    fig, ax = plt.subplots(figsize=(12.4, 4.6))

    ax.plot([-0.4, n - 0.6], [0, 0], color=INK, linewidth=1.4, zorder=1)

    for i, (year, who, what) in enumerate(EVENTS):
        up = (i % 2 == 0)
        y_lab = 0.30 if up else -0.30
        y_txt = 0.62 if up else -0.62
        va = "bottom" if up else "top"
        ax.plot([i, i], [0, y_lab * 0.72], color=INK, linewidth=0.8,
                alpha=0.55, zorder=1)
        ax.plot([i], [0], marker="o", color=ACCENT, markersize=8,
                markeredgecolor="white", markeredgewidth=1.4, zorder=5)
        ax.text(i, y_lab, str(year), ha="center", va=va, fontsize=9.5,
                color=ACCENT, weight="bold")
        ax.text(i, y_txt, who, ha="center", va=va, fontsize=7.6, color=INK)
        ax.text(i, y_txt + (0.30 if up else -0.30), what, ha="center", va=va,
                fontsize=6.8, color=INK, alpha=0.78)

    ax.set_xlim(-0.8, n - 0.2)
    ax.set_ylim(-1.75, 1.75)
    ax.axis("off")
    ax.text(0.5, 0.015,
            "placed in order, not to scale - the gap from 1774 to 1907 would "
            "flatten everything else",
            transform=ax.transAxes, ha="center", fontsize=7.4, color=INK,
            alpha=0.6, style="italic")

    fig.tight_layout()
    fig.savefig(HERE / "chapter_timeline.svg", transparent=True,
                bbox_inches="tight")
    plt.close(fig)
    span = EVENTS[-1][0] - EVENTS[0][0]
    print(f"wrote chapter_timeline.svg   {n} entries spanning {span} years")


if __name__ == "__main__":
    timeline()
