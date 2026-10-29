"""37.1 마르코프 결정 과정 쪽들의 그림.

규칙은 CLAUDE.md 의 Figures 항에 적어 둔 대로다.
- SVG, fonttype='path', transparent
- 글자는 모두 아스키 (matplotlib 기본 폰트에 한글이 없다)

자료를 읽지 않으므로 어디서 돌려도 된다.

실행: python docs/ch36/mdp/figures/make_figures.py
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

SMALL, BIG, N_STEPS = 1.0, 20.0, 10


# === 그림으로 보는 AI: 감마가 폴리시를 고른다 =================================
#
# 복도 하나. 왼쪽으로 가면 지금 1 을 받고 끝, 오른쪽으로 10 칸 걸으면 20 을
# 받고 끝. 어느 쪽이 나은지는 **오직 감마가 정한다** — 갈림 지점이
# gamma* = (1/20)^(1/10) 로 정확히 떨어진다.

def discount_threshold():
    thr = (SMALL / BIG) ** (1 / N_STEPS)
    gs = np.linspace(0.40, 0.995, 400)
    pv = (gs ** N_STEPS) * BIG

    # 갈림 지점이 수치로도 맞는지 확인하고 그린다
    i = int(np.argmin(np.abs(pv - SMALL)))
    assert abs(gs[i] - thr) < 0.01, f"갈림 지점이 안 맞는다: {gs[i]} vs {thr}"
    assert (gs[0] ** N_STEPS) * BIG < SMALL, "낮은 감마에서 먼 쪽이 이긴다"

    fig, ax = plt.subplots(figsize=(7.8, 4.4))
    ax.axvspan(gs[0], thr, color=MUTED, alpha=0.10)
    ax.axvspan(thr, gs[-1], color=ACCENT, alpha=0.10)

    ax.plot(gs, pv, color=ACCENT, linewidth=2.4,
            label=f"the far reward, discounted  ({BIG:.0f} after "
                  f"{N_STEPS} steps)")
    ax.axhline(SMALL, color=INK, linestyle="--", linewidth=1.6,
               label=f"the reward right here  ({SMALL:.0f})")
    ax.axvline(thr, color=INK, linewidth=1.0, alpha=0.55)

    ax.plot([thr], [SMALL], "o", color=INK, ms=8, zorder=5)
    ax.annotate(f"they cross at gamma = {thr:.4f}", xy=(thr, SMALL),
                xytext=(thr - 0.03, SMALL + 4.2), fontsize=9, color=INK,
                ha="right",
                arrowprops=dict(arrowstyle="->", color=INK, lw=1.0))
    ax.text((gs[0] + thr) / 2, 16.5, "take the near one", ha="center",
            fontsize=9.5, color=MUTED, weight="bold")
    ax.text((thr + gs[-1]) / 2, 16.5, "walk for the far one", ha="center",
            fontsize=9.5, color=ACCENT, weight="bold")

    ax.set_xlabel("gamma", fontsize=9.5)
    ax.set_ylabel("value now", fontsize=9.5)
    ax.set_title("one corridor, two answers", fontsize=10.5, color=INK,
                 weight="bold")
    ax.legend(fontsize=8.4, loc="upper left", framealpha=0.92)
    ax.set_ylim(0, 20.5)
    ax.grid(alpha=0.13)
    ax.tick_params(labelsize=8, colors=INK)
    for s in ax.spines.values():
        s.set_color(INK)
        s.set_linewidth(0.7)

    fig.text(0.5, -0.04,
             "the rewards never change. only gamma does, and it decides "
             "which way the agent walks.",
             ha="center", fontsize=8.6, style="italic", color=INK)
    fig.tight_layout()
    fig.savefig(HERE / "discount_threshold.svg", transparent=True,
                bbox_inches="tight")
    plt.close(fig)
    print(f"wrote discount_threshold.svg   threshold {thr:.4f}")


if __name__ == "__main__":
    discount_threshold()
