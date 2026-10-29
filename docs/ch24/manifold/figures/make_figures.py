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


if __name__ == "__main__":
    page_2323()
    print("page_2323.svg")
