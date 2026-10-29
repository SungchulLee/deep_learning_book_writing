"""11.4절(어텐션)의 그림.

그림에 적히는 수는 모두 그 쪽의 점수에서 여기서 다시 계산한 것이다.
지어낸 수는 없다.

- scaling_softmax.svg  <- scaled_dot_product.md 3절의 점수 (20, 22, 18, 21)

규칙은 CLAUDE.md 의 Figures 항에 적어 둔 대로다.
- SVG, fonttype='path', transparent=True
- 글자는 모두 아스키 (matplotlib 기본 폰트에 한글이 없다)

실행: python docs/ch11/attention/figures/make_figures.py
      python docs/ch11/attention/figures/make_figures.py --png   <- 눈으로 볼 PNG 도 함께
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["svg.fonttype"] = "path"

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
INK = "#2f3337"
ACCENT = "#8a4b2a"
COOL = "#2f5f8a"
MUTED = "#9aa0a6"


def style(ax):
    ax.tick_params(labelsize=8, colors=INK)
    for s in ax.spines.values():
        s.set_color(INK)
        s.set_linewidth(0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save(fig, name):
    fig.savefig(HERE / f"{name}.svg", transparent=True, bbox_inches="tight")
    if "--png" in sys.argv:          # 글자가 겹치지 않는지 눈으로 보려고 쓴다
        fig.savefig(HERE / f"{name}.png", transparent=False, dpi=170,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {name}.svg")


# ========================================================================
# 1. sqrt(d_k) 로 나누기 전과 뒤의 어텐션 웨이트
#    점수는 scaled_dot_product.md 3절의 (20, 22, 18, 21), d_k = 512.
# ========================================================================

def softmax(s):
    e = np.exp(s - s.max())
    return e / e.sum()


def make_scaling_softmax():
    scores = np.array([20.0, 22.0, 18.0, 21.0])
    d_k = 512
    scaled = scores / np.sqrt(d_k)

    a_raw, a_scaled = softmax(scores), softmax(scaled)

    # 쪽이 적어 둔 값과 같은지 여기서 확인하고 간다.
    assert np.allclose(np.round(a_raw, 4), [0.0889, 0.6572, 0.0120, 0.2418])
    assert np.allclose(np.round(a_scaled, 4), [0.2467, 0.2695, 0.2259, 0.2579])

    keys = [f"key {i + 1}" for i in range(4)]
    x = np.arange(4)
    top = int(scores.argmax())

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 4.0), sharey=True)
    panels = (
        (axes[0], a_raw, scores, "no scaling", "{:.0f}"),
        (axes[1], a_scaled, scaled, "divided by sqrt(512) = 22.627", "{:.3f}"),
    )

    for ax, w, s, title, fmt in panels:
        title = (f"{title}\nkey 2 takes {100 * w.max():.1f}%,   "
                 f"largest minus smallest = {w.max() - w.min():.4f}")
        colors = [ACCENT if i == top else COOL for i in range(4)]
        ax.bar(x, w, width=0.56, color=colors, alpha=0.9, edgecolor=INK,
               linewidth=0.6, zorder=3)
        ax.axhline(0.25, color=MUTED, lw=1.0, ls=":", zorder=2)
        for i, v in enumerate(w):
            ax.text(i, v + 0.012, f"{v:.4f}", ha="center", va="bottom",
                    fontsize=8.6, color=colors[i], weight="bold")
            ax.text(i, -0.035, fmt.format(s[i]), ha="center", va="top",
                    fontsize=8, color=INK)
        ax.set_xticks(x)
        ax.set_xticklabels(keys, fontsize=8.5, color=INK)
        ax.tick_params(axis="x", length=0, pad=16)
        ax.set_xlim(-0.65, 3.65)
        ax.set_title(title, fontsize=9.5, color=INK, pad=8)
        style(ax)

    axes[0].set_ylim(0, 0.74)
    axes[0].set_ylabel("attention weight", fontsize=9, color=INK)
    axes[0].text(-0.6, 0.262, "uniform, 0.25", fontsize=7.8, color=MUTED,
                 ha="left", va="bottom")

    fig.suptitle("the same four scores, before and after dividing by "
                 "sqrt(d_k)",
                 fontsize=10.5, weight="bold", color=INK, y=1.0)
    fig.text(0.5, -0.055,
             "softmax of (20, 22, 18, 21) and of the same scores over "
             "sqrt(512); the scale shrinks every gap between scores by the "
             "same factor 22.627,\nwhich is why this row flattens - it does "
             "not pin the weights to any range, and a different row of scores "
             "would flatten by a different amount.",
             ha="center", fontsize=7.6, style="italic", color=INK)
    fig.tight_layout()
    save(fig, "scaling_softmax")


if __name__ == "__main__":
    make_scaling_softmax()
