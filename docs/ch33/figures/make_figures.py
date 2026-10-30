"""34장 장 소개의 그림.

규칙은 CLAUDE.md 의 Figures 항에 적어 둔 대로다.
- SVG, fonttype='path', transparent
- 글자는 모두 아스키 (matplotlib 기본 폰트에 한글이 없다)

실행: python docs/ch33/figures/make_figures.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["svg.fonttype"] = "path"

import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
INK = "#2f3337"
ACCENT = "#8a4b2a"
MUTED = "#2d6a8f"


# === 그림으로 보는 AI: 이 장이 모아 놓은 해들 =================================
#
# 날짜는 전부 이 장의 상자에서 가져온다. 상자가 있는 쪽이 실제로 있는지
# 먼저 확인하고, 없으면 그리지 않는다 — 외운 날짜를 올리지 않기 위해서다.
#
# 해의 간격이 매우 고르지 않아 진짜 눈금에 올리면 1968 과 1977 이 왼쪽 끝에
# 붙어 버린다. 고르게 놓고 "not to scale" 이라고 적는다.

ENTRIES = [
    (1968, "Weisfeiler and Leman", "canonical form of a graph,\nand an algebra",
     "convolutions/gin.md", False),
    (1977, "Zachary", "an anthropologist watches\na karate club split",
     "node_link/community_detection.md", False),
    (2017, "Kipf and Welling", "the renormalization trick",
     "convolutions/gcn.md", True),
    (2017, "Gilmer and others", "eight models, one frame",
     "message_passing/mpnn.md", True),
    (2018, "Li, Han and Wu", "smoothing names its own\nfailure",
     "advanced/over_smoothing.md", True),
    (2019, "Xu and others", "the ceiling is the 1968 one",
     "convolutions/gin.md", True),
]


def chapter_years():
    # --- 날짜마다 그 상자가 정말 있는지 확인한다 ---------------------------
    for year, who, _, page, _ in ENTRIES:
        p = ROOT / page
        assert p.exists(), f"{page} 가 없다"
        assert "그림으로 보는 AI" in p.read_text(encoding="utf-8"), \
            f"{page} 에 상자가 없는데 {year} 를 시간선에 올릴 수 없다"

    fig, ax = plt.subplots(figsize=(10.6, 4.3))

    n = len(ENTRIES)
    xs = [i * (10.0 / (n - 1)) for i in range(n)]
    ax.annotate("", xy=(10.7, 1.0), xytext=(-0.7, 1.0),
                arrowprops=dict(arrowstyle="->", color=INK, linewidth=1.3))

    for x, (year, who, what, _, is_ml) in zip(xs, ENTRIES):
        col = ACCENT if is_ml else MUTED
        ax.plot([x], [1.0], marker="o", color=col, markersize=10,
                markeredgecolor="white", markeredgewidth=1.6, zorder=5)
        ax.text(x, 1.30, str(year), ha="center", fontsize=11, color=col,
                weight="bold")
        ax.text(x, 1.15, who, ha="center", fontsize=8.2, color=INK,
                alpha=0.9)
        ax.text(x, 0.84, what, ha="center", va="top", fontsize=8.0,
                color=INK, alpha=0.85)

    split = (xs[1] + xs[2]) / 2
    ax.plot([split, split], [0.42, 1.42], color=INK, linestyle=":",
            linewidth=1.1, alpha=0.6)
    ax.text(split - 0.3, 0.50, "not machine learning", ha="right",
            fontsize=9.2, color=MUTED, weight="bold")
    ax.text(split + 0.3, 0.50, "machine learning", ha="left",
            fontsize=9.2, color=ACCENT, weight="bold")

    ax.text(5.0, 0.26, "the ceiling and the benchmark both come from the left",
            ha="center", fontsize=9.4, color=INK, style="italic")
    ax.text(10.7, 0.08, "years evenly spaced - not to scale", ha="right",
            fontsize=6.6, color=INK, alpha=0.55, style="italic")

    ax.set_xlim(-1.2, 11.2)
    ax.set_ylim(0.05, 1.5)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(HERE / "chapter_years.svg", transparent=True,
                bbox_inches="tight")
    plt.close(fig)
    print(f"wrote chapter_years.svg   {ENTRIES[0][0]}-{ENTRIES[-1][0]}, "
          f"{sum(1 for e in ENTRIES if not e[4])} from outside ML")


if __name__ == "__main__":
    chapter_years()
