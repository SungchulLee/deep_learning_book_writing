"""「그림으로 보는 AI」 — 리나인마의 1970년 석사 논문을 그린다.

규칙은 CLAUDE.md 의 Figures 항과 「그림으로 보는 AI」 항에 적어 둔 대로다.
글자는 모두 아스키라야 한다 (matplotlib 기본 폰트에 한글이 없다).

실행: python docs/ch06/autograd/figures/make_figures.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["svg.fonttype"] = "path"

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = Path(__file__).resolve().parent
INK = "#2f3337"
ACCENT = "#8a4b2a"


def box(ax, x, y, label, w=0.9, h=0.44):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle="round,pad=0.04",
                                fc="none", ec=INK, lw=1.0))
    ax.text(x, y, label, ha="center", va="center", fontsize=9.5,
            family="monospace", color=INK)


def arrow(ax, x0, y0, x1, y1, color, lw=1.4):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                 mutation_scale=12, color=color, lw=lw))


def make_linnainmaa():
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.6, 4.3))

    xs = [0.6, 2.4, 4.2, 6.0]
    names = ["u1", "u2", "u3", "y"]

    # --- 위: 앞으로 가며 반올림 오차가 생긴다 ---
    for x, n in zip(xs, names):
        box(ax1, x, 0.52, n)
    for a, b in zip(xs[:-1], xs[1:]):
        arrow(ax1, a + 0.48, 0.52, b - 0.48, 0.52, INK)
    for x in xs:
        arrow(ax1, x, 1.12, x, 0.78, ACCENT, lw=1.1)
        ax1.text(x, 1.22, "e", ha="center", va="bottom", fontsize=9,
                 family="monospace", color=ACCENT)
    ax1.text(-0.35, 0.52, "forward", ha="center", va="center", fontsize=9,
             weight="bold", color=INK, rotation=90)
    ax1.text(3.3, 1.62, "each step adds its own rounding error",
             ha="center", fontsize=9, color=ACCENT)
    ax1.set_xlim(-0.8, 6.8)
    ax1.set_ylim(0.1, 1.95)

    # --- 아래: 거꾸로 가며 민감도를 모은다 ---
    for x, n in zip(xs, names):
        box(ax2, x, 0.95, n)
    for a, b in zip(xs[1:], xs[:-1]):
        arrow(ax2, a - 0.48, 0.95, b + 0.48, 0.95, ACCENT)
    for i, x in enumerate(xs):
        lab = "dy/dy = 1" if i == len(xs) - 1 else f"dy/du{i+1}"
        ax2.text(x, 0.45, lab, ha="center", va="center", fontsize=8.5,
                 family="monospace", color=ACCENT)
    ax2.text(-0.35, 0.95, "reverse", ha="center", va="center", fontsize=9,
             weight="bold", color=INK, rotation=90)
    ax2.text(3.3, 0.09, "one backward sweep gives every sensitivity at once",
             ha="center", fontsize=9, color=ACCENT)
    ax2.set_xlim(-0.8, 6.8)
    ax2.set_ylim(-0.05, 1.45)

    for ax in (ax1, ax2):
        ax.set_axis_off()

    fig.text(0.5, 0.005,
             "the 1970 thesis asks about rounding error - it contains no "
             "neural network",
             ha="center", fontsize=8, style="italic", color=INK)
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    fig.savefig(HERE / "linnainmaa_1970.svg", transparent=True,
                bbox_inches="tight")
    plt.close(fig)
    print("wrote linnainmaa_1970.svg")


if __name__ == "__main__":
    make_linnainmaa()
