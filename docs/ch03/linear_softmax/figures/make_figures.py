"""3.2절의 그림을 그리는 스크립트.

「그림으로 보는 AI」 상자의 그림은 사진을 빌려 오지 않고 형식만 빌려 이
책의 그림체로 다시 그린다. 그래야 라이선스 문제가 없고, 이 책의 다른 그림과
결이 같다. 규칙은 CLAUDE.md 의 「그림으로 보는 AI」 항에 적어 두었다.

    python3 make_figures.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle, Circle, FancyArrowPatch
from pathlib import Path

plt.rcParams["svg.fonttype"] = "path"      # 글자를 외곽선으로 — 폰트에 기대지 않는다
HERE = Path(__file__).parent
INK = "#2b2b2b"


def save(fig, name):
    fig.savefig(HERE / name, transparent=True, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {name}")


# ===========================================================================
def draw_mark_i():
    """Mark I 퍼셉트론의 구조를 다시 그린다.

    왼쪽부터 20x20 광전지(S), 고정된 무작위 배선 플러그보드, 연합 유닛(A),
    모터가 돌리는 가변저항, 응답 유닛(R)이다. 오른쪽에 이 쪽의 모델을 나란히
    두어 「배우는 부분이 어디까지인가」를 견준다.
    """
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(10.6, 4.3))
    ax.set_xlim(0, 13.2); ax.set_ylim(0, 5.6); ax.axis("off")

    # --- S: 광전지 20x20 --------------------------------------------------------
    for i in range(8):
        for j in range(8):
            ax.add_patch(Circle((0.45 + 0.26 * j, 1.5 + 0.26 * i), 0.075,
                                fill=False, ec=INK, lw=0.6))
    ax.add_patch(Rectangle((0.28, 1.32), 2.1, 2.1, fill=False, ec=INK, lw=1.2))
    ax.text(1.33, 3.62, "400 photocells", ha="center", fontsize=9, color=INK)
    ax.text(1.33, 1.02, "20 x 20 retina", ha="center", fontsize=8.5, color=INK)
    ax.text(1.33, 0.62, "S-units", ha="center", fontsize=8.5, color=INK, style="italic")

    # --- 플러그보드: 고정된 무작위 배선 ----------------------------------------
    ax.add_patch(Rectangle((3.3, 1.32), 1.5, 2.1, fill=False, ec=INK, lw=1.2))
    for _ in range(16):
        y0, y1 = rng.uniform(1.45, 3.3, 2)
        ax.plot([3.35, 4.75], [y0, y1], color=INK, lw=0.5, alpha=0.75)
    ax.text(4.05, 3.62, "plugboard", ha="center", fontsize=9, color=INK)
    ax.text(4.05, 1.02, "random, fixed", ha="center", fontsize=8.5, color=INK)
    ax.text(4.05, 0.62, "never adjusted", ha="center", fontsize=8.5, color=INK, style="italic")
    ax.add_patch(FancyArrowPatch((2.45, 2.37), (3.25, 2.37), arrowstyle="-|>",
                                 mutation_scale=10, color=INK, lw=1.0))

    # --- A 유닛 -----------------------------------------------------------------
    for i in range(9):
        ax.add_patch(Circle((5.75, 1.5 + 0.24 * i), 0.09, fill=False, ec=INK, lw=0.8))
    ax.text(5.75, 3.62, "512 A-units", ha="center", fontsize=9, color=INK)

    # --- 배우는 층: 모터가 돌리는 가변저항 --------------------------------------
    for i in range(9):
        for k in range(3):
            ax.plot([5.9, 8.9], [1.5 + 0.24 * i, 1.75 + 0.6 * k],
                    color=INK, lw=0.35, alpha=0.5)
    kx, ky = 7.4, 4.35
    ax.add_patch(Circle((kx, ky), 0.33, fill=False, ec=INK, lw=1.3))
    ax.plot([kx, kx + 0.23], [ky, ky + 0.23], color=INK, lw=1.5)
    ax.add_patch(Rectangle((kx - 0.26, ky - 0.95), 0.52, 0.42, fill=False, ec=INK, lw=1.1))
    ax.text(kx, ky - 0.76, "M", ha="center", va="center", fontsize=8, color=INK)
    ax.plot([kx, kx], [ky - 0.53, ky - 0.33], color=INK, lw=1.0)
    ax.text(kx + 0.55, ky + 0.05, "one weight = one potentiometer,", ha="left",
            fontsize=8.5, color=INK)
    ax.text(kx + 0.55, ky - 0.35, "turned by a motor", ha="left", fontsize=8.5, color=INK)
    ax.text(7.4, 1.02, "512 x 8 = 4,096 weights", ha="center", fontsize=8.5, color=INK)
    ax.text(7.4, 0.62, "the only part that learns", ha="center", fontsize=8.5,
            color=INK, style="italic")

    # --- R 유닛 -----------------------------------------------------------------
    for k in range(3):
        ax.add_patch(Circle((9.05, 1.75 + 0.6 * k), 0.12, fill=False, ec=INK, lw=0.9))
    ax.text(9.05, 3.62, "8 R-units", ha="center", fontsize=9, color=INK)

    # --- 오늘의 쪽과 견주기 ------------------------------------------------------
    ax.plot([10.1, 10.1], [0.95, 4.0], color=INK, lw=0.6, ls=(0, (3, 3)))
    ax.text(11.6, 3.62, "this page", ha="center", fontsize=9, color=INK)
    rows = [("input", "784"), ("output", "10"), ("learned", "7,850")]
    for n, (a, b) in enumerate(rows):
        y = 2.9 - 0.52 * n
        ax.text(10.5, y, a, ha="left", fontsize=8.5, color=INK)
        ax.text(12.75, y, b, ha="right", fontsize=8.5, color=INK)
    ax.text(11.6, 1.02, "all of it learns", ha="center", fontsize=8.5, color=INK)

    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    return fig


# ===========================================================================
if __name__ == "__main__":
    save(draw_mark_i(), "mark_i_perceptron.svg")
