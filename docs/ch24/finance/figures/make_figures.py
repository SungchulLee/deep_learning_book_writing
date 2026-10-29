"""25.3 수익률 곡선 쪽의 그림.

규칙은 CLAUDE.md 의 Figures 항에 적어 둔 대로다.
- SVG, fonttype='path', transparent
- 글자는 모두 아스키 (matplotlib 기본 폰트에 한글이 없다)

이 쪽은 실제 수익률 자료를 받아 돌리지 않는다. 그래서 **로딩을 적합해 그리지
않고 꼴만 그린다.** 그림 안에 그렇게 적는다 (CLAUDE.md: 지어낸 수를 그림에
넣지 않는다). 세로축에 눈금을 매기지 않는 것도 같은 까닭이다 — 어떤 적합에서
나온 수가 아니므로 읽을 값이 없다.

실행: python docs/ch24/finance/figures/make_figures.py
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


def three_factors():
    m = np.linspace(0, 1, 60)                 # 짧은 만기에서 긴 만기로

    shapes = [
        ("level", np.ones_like(m) * 0.62,
         "every maturity moves\nthe same way"),
        ("steepness", 0.78 * (m - 0.5) * 2 / 2 * 1.0,
         "short and long\nmove opposite ways"),
        ("curvature", 0.80 * (1.5 * (2 * m - 1) ** 2 - 0.62),
         "the middle moves against\nboth ends"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(9.8, 3.1), sharey=True)
    for ax, (name, y, note) in zip(axes, shapes):
        ax.axhline(0, color=INK, linewidth=0.7, alpha=0.45)
        ax.plot(m, y, color=ACCENT, linewidth=2.4)
        ax.fill_between(m, 0, y, color=ACCENT, alpha=0.11)
        ax.set_title(name, fontsize=10.5, color=INK, weight="bold", pad=7)
        ax.text(0.5, -1.02, note, ha="center", va="top", fontsize=8,
                color=INK, alpha=0.85, transform=ax.transData)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["short", "long"], fontsize=8.5, color=INK)
        ax.set_yticks([])                     # 적합한 수가 아니므로 눈금을 안 매긴다
        ax.set_ylim(-1.05, 1.05)
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)

    axes[0].set_ylabel("loading", fontsize=9, color=INK)
    fig.text(0.5, -0.17,
             "shapes only - not loadings fitted to any data.  "
             "the names are read off observed loadings; PCA does not impose them.",
             ha="center", fontsize=8.2, style="italic", color=INK)
    fig.tight_layout()
    fig.savefig(HERE / "three_factors.svg", transparent=True,
                bbox_inches="tight")
    plt.close(fig)
    print("wrote three_factors.svg")


if __name__ == "__main__":
    three_factors()
