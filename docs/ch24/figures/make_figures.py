"""25장 장 소개의 그림.

규칙은 CLAUDE.md 의 Figures 항에 적어 둔 대로다.
- SVG, fonttype='path', transparent
- 글자는 모두 아스키 (matplotlib 기본 폰트에 한글이 없다)

수는 전부 여기서 셈한다 — 축마다 칸 열 개로 자르면 d 차원에 10^d 칸이고,
MNIST 의 60,000 장과 어디서 엇갈리는지도 셈해서 찍는다.

실행: python docs/ch24/figures/make_figures.py
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

BINS = 10          # 축마다 칸 열 개
MNIST = 60_000     # 이 책이 내내 쓰는 자료의 크기


def curse():
    d = np.arange(1, 13)
    cells = float(BINS) ** d
    cross = np.log10(MNIST) / np.log10(BINS)      # 10^d = 60000 인 d

    fig, ax = plt.subplots(figsize=(7.6, 4.3))

    ax.semilogy(d, cells, "o-", color=ACCENT, linewidth=2.0, ms=5.5,
                label=f"cells, {BINS} bins per axis")
    ax.axhline(MNIST, color=INK, linestyle="--", linewidth=1.4,
               label=f"every image in MNIST ({MNIST:,})")

    ax.axvline(cross, color=INK, linestyle=":", linewidth=1.0, alpha=0.6)
    ax.annotate(f"from {int(np.ceil(cross))} dimensions on,\n"
                f"more cells than images",
                xy=(cross, MNIST), xytext=(cross + 0.45, MNIST / 1200),
                fontsize=8.5, color=INK,
                arrowprops=dict(arrowstyle="->", color=INK, lw=0.9))

    ax.set_xlabel("number of dimensions", fontsize=9.5)
    ax.set_ylabel("cells to fill (log scale)", fontsize=9.5)
    ax.set_title("what Bellman was counting", fontsize=10.5, color=INK,
                 weight="bold", pad=8)
    ax.legend(fontsize=8.5, loc="upper left", framealpha=0.92)
    ax.grid(alpha=0.13, which="both")
    ax.tick_params(labelsize=8.5, colors=INK)
    for s in ax.spines.values():
        s.set_color(INK)
        s.set_linewidth(0.7)

    fig.text(0.5, -0.04,
             f"an MNIST image has 784 dimensions, so this count would be "
             f"10^784 - but nothing in this chapter builds a grid.",
             ha="center", fontsize=8.5, style="italic", color=INK)
    fig.tight_layout()
    fig.savefig(HERE / "curse.svg", transparent=True, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote curse.svg   crossing at d = {cross:.3f}")


if __name__ == "__main__":
    curse()
