"""「cat 시각화」 쪽의 그림을 그린다.

규칙은 CLAUDE.md 의 Figures 항에 적어 둔 대로다.
- SVG 로만 저장하고 svg.fonttype='path' 로 글리프를 윤곽선으로 굽는다
- transparent=True — 테마가 밝은 쪽만 쓰므로 배경을 비워 둔다
- 그림 안의 글자는 모두 ASCII. matplotlib 기본 폰트에 한글이 없어
  한글을 넣으면 두부 상자로 나온다

실행: python make_figures.py   (이 디렉터리에서)
"""
import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["svg.fonttype"] = "path"

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

A_FILL = "#cfe3f7"      # A 에서 온 칸
B_FILL = "#f7dfcf"      # B 에서 온 칸
EDGE = "#44484d"


def grid(ax, x0, y0, values, fills, cell=0.52, z=1):
    """values 는 행의 리스트. 왼쪽 위에서 오른쪽 아래로 그린다.

    z 를 주는 까닭: matplotlib 은 글자를 모든 도형 위에 그린다. 두 장을
    겹쳐 그릴 때 그대로 두면 뒷장의 숫자가 앞장을 뚫고 보인다.
    """
    rows, cols = len(values), len(values[0])
    for i in range(rows):
        for j in range(cols):
            ax.add_patch(Rectangle((x0 + j * cell, y0 - (i + 1) * cell),
                                   cell, cell,
                                   facecolor=fills[i][j], edgecolor=EDGE,
                                   linewidth=0.9, zorder=z))
            ax.text(x0 + (j + 0.5) * cell, y0 - (i + 0.5) * cell,
                    str(values[i][j]), ha="center", va="center", fontsize=8.5,
                    zorder=z + 1)
    return x0 + cols * cell, y0 - rows * cell


def label(ax, x, y, text, size=9.5, weight="normal"):
    ax.text(x, y, text, ha="left", va="bottom", fontsize=size, weight=weight)


A = [[11, 12, 13], [14, 15, 16]]
B = [[21, 22, 23], [24, 25, 26]]
AF = [[A_FILL] * 3 for _ in range(2)]
BF = [[B_FILL] * 3 for _ in range(2)]


def make_cat_axes():
    """A, B 와 dim=0 / dim=1 결과를 한 장에 나란히 둔다."""
    fig, ax = plt.subplots(figsize=(7.6, 3.1))
    ax.set_axis_off()
    ax.set_xlim(0, 9.6)
    ax.set_ylim(-3.1, 0.6)

    # 왼쪽: 입력 둘
    label(ax, 0.0, 0.18, "A  (2, 3)", weight="bold")
    grid(ax, 0.0, 0.0, A, AF)
    label(ax, 0.0, -1.5, "B  (2, 3)", weight="bold")
    grid(ax, 0.0, -1.68, B, BF)

    # 가운데: dim=0 — 세로로 길어진다
    label(ax, 3.1, 0.18, "cat(dim=0)  ->  (4, 3)", weight="bold")
    grid(ax, 3.1, 0.0, A + B, AF + BF)
    ax.annotate("", xy=(3.0, -1.1), xytext=(3.0, 0.0),
                arrowprops=dict(arrowstyle="-|>", color=EDGE, lw=1.0))
    ax.text(2.92, -0.55, "axis 0 grows", ha="right", va="center",
            fontsize=8, rotation=90)

    # 오른쪽: dim=1 — 가로로 길어진다
    label(ax, 6.2, 0.18, "cat(dim=1)  ->  (2, 6)", weight="bold")
    grid(ax, 6.2, 0.0, [A[0] + B[0], A[1] + B[1]],
         [AF[0] + BF[0], AF[1] + BF[1]])
    ax.annotate("", xy=(9.32, -1.22), xytext=(6.2, -1.22),
                arrowprops=dict(arrowstyle="-|>", color=EDGE, lw=1.0))
    ax.text(7.76, -1.34, "axis 1 grows", ha="center", va="top", fontsize=8)

    ax.text(0.0, -2.95, "12 numbers either way - only the shape differs",
            fontsize=8.5, style="italic")
    fig.tight_layout()
    fig.savefig("cat_axes.svg", transparent=True, bbox_inches="tight")
    plt.close(fig)
    print("wrote cat_axes.svg")


def make_cat_vs_stack():
    """cat 은 있는 축을 늘리고 stack 은 없던 축을 만든다."""
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    ax.set_axis_off()
    ax.set_xlim(0, 10.5)
    ax.set_ylim(-3.0, 1.0)

    label(ax, 0.0, 0.18, "cat(dim=0)  ->  (4, 3)", weight="bold")
    grid(ax, 0.0, 0.0, A + B, AF + BF)
    ax.text(0.0, -2.45, "one block, 2 axes", fontsize=8.5, style="italic")

    # stack: 두 장을 겹쳐 보이게 살짝 엇갈려 그린다
    label(ax, 4.6, 0.18, "stack(dim=0)  ->  (2, 2, 3)", weight="bold")
    grid(ax, 5.05, -0.42, B, BF, z=1)   # 뒤쪽 장
    grid(ax, 4.6, 0.0, A, AF, z=3)      # 앞쪽 장
    ax.annotate("", xy=(6.95, -0.70), xytext=(6.45, -0.20),
                arrowprops=dict(arrowstyle="-|>", color=EDGE, lw=1.0),
                zorder=5)
    ax.text(7.1, -0.78, "new axis", fontsize=8, ha="left", va="center")
    ax.text(4.6, -2.45, "two blocks, 3 axes", fontsize=8.5, style="italic")

    fig.tight_layout()
    fig.savefig("cat_vs_stack.svg", transparent=True, bbox_inches="tight")
    plt.close(fig)
    print("wrote cat_vs_stack.svg")


if __name__ == "__main__":
    make_cat_axes()
    make_cat_vs_stack()
