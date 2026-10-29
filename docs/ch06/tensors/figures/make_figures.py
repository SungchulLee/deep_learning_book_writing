"""7.1 텐서 만들기 쪽들의 그림.

규칙은 CLAUDE.md 의 Figures 항에 적어 둔 대로다.
- SVG, fonttype='path', transparent
- 글자는 모두 아스키 (matplotlib 기본 폰트에 한글이 없다)

자료를 읽지 않으므로 어디서 돌려도 된다.

실행: python docs/ch06/tensors/figures/make_figures.py
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
MUTED = "#2d6a8f"


# === 그림으로 보는 AI: tensor 라는 말의 세 뜻 =================================
#
# 1846 해밀턴 — 크기 하나. 수 하나다.
# 1898 포크트 — 탄성체 한 점의 응력. 좌표를 돌리면 정해진 규칙으로 바뀐다.
# 오늘 PyTorch — 모양과 자료형을 가진 다차원 배열. 변환 규칙은 어디에도 없다.
#
# 그림에 적는 수는 지어내지 않는다. 가운데 판의 화살표는 응력의 **꼴**만
# 보이는 것이라 그림 안에 그렇게 적는다.

def tensor_word():
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10.6, 3.9))

    # --- 1846 해밀턴: 크기 하나 -------------------------------------------
    ax1.annotate("", xy=(0.78, 0.74), xytext=(0.16, 0.26),
                 arrowprops=dict(arrowstyle="->", color=INK, linewidth=2.2))
    ax1.plot([0.16, 0.78], [0.26, 0.26], color=INK, linewidth=0.7, alpha=0.35)
    ax1.plot([0.78, 0.78], [0.26, 0.74], color=INK, linewidth=0.7, alpha=0.35)
    ax1.annotate("", xy=(0.74, 0.80), xytext=(0.12, 0.32),
                 arrowprops=dict(arrowstyle="<->", color=ACCENT, linewidth=1.4))
    ax1.text(0.33, 0.62, "tensor", fontsize=11, color=ACCENT, weight="bold",
             rotation=37.7, ha="center")
    ax1.text(0.5, 0.11, "the magnitude. one number.", ha="center",
             fontsize=8.6, color=INK)
    ax1.set_title("1846   Hamilton", fontsize=10, color=INK, weight="bold")

    # --- 1898 포크트: 한 점의 응력 ----------------------------------------
    ax2.add_patch(plt.Rectangle((0.30, 0.34), 0.40, 0.36, facecolor="none",
                                edgecolor=INK, linewidth=1.4))
    for (x, y, dx, dy) in ((0.50, 0.78, 0.0, -0.09), (0.50, 0.26, 0.0, 0.09),
                           (0.19, 0.52, 0.09, 0.0), (0.81, 0.52, -0.09, 0.0)):
        ax2.annotate("", xy=(x + dx, y + dy), xytext=(x, y),
                     arrowprops=dict(arrowstyle="->", color=ACCENT,
                                     linewidth=1.6))
    for (x, y, dx) in ((0.36, 0.745, 0.13), (0.64, 0.295, -0.13)):
        ax2.annotate("", xy=(x + dx, y), xytext=(x, y),
                     arrowprops=dict(arrowstyle="->", color=MUTED,
                                     linewidth=1.3))
    ax2.text(0.5, 0.16, "stress at a point.\n"
                        "turn the axes and it changes\nby a fixed rule.",
             ha="center", va="top", fontsize=8.6, color=INK)
    ax2.text(0.97, 0.90, "schematic", ha="right", fontsize=6.4, color=INK,
             alpha=0.55, style="italic")
    ax2.set_title("1898   Voigt", fontsize=10, color=INK, weight="bold")

    # --- 오늘: 다차원 배열 -------------------------------------------------
    vals = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    for r in range(2):
        for c in range(3):
            ax3.add_patch(plt.Rectangle((0.22 + c * 0.19, 0.60 - r * 0.19),
                                        0.17, 0.17, facecolor=ACCENT,
                                        alpha=0.13, edgecolor=ACCENT,
                                        linewidth=0.9))
            ax3.text(0.305 + c * 0.19, 0.685 - r * 0.19, f"{vals[r, c]:.0f}",
                     ha="center", va="center", fontsize=9, color=INK)
    ax3.text(0.5, 0.33, "shape (2, 3)   dtype float32", ha="center",
             fontsize=8.6, color=INK, family="monospace")
    ax3.text(0.5, 0.16, "no transformation law\nanywhere in the library.",
             ha="center", va="top", fontsize=8.6, color=ACCENT)
    ax3.set_title("today   torch.Tensor", fontsize=10, color=INK,
                  weight="bold")

    for ax in (ax1, ax2, ax3):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

    fig.text(0.5, -0.03,
             "three different things wear the same word. "
             "torch.tensor(42) is the 1846 one.",
             ha="center", fontsize=8.8, style="italic", color=INK)
    fig.tight_layout()
    fig.savefig(HERE / "tensor_word.svg", transparent=True,
                bbox_inches="tight")
    plt.close(fig)
    print("wrote tensor_word.svg")


if __name__ == "__main__":
    tensor_word()
