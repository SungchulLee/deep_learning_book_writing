"""39.4 벗어난 폴리시 쪽들의 그림.

규칙은 CLAUDE.md 의 Figures 항에 적어 둔 대로다.
- SVG, fonttype='path', transparent
- 글자는 모두 아스키 (matplotlib 기본 폰트에 한글이 없다)

표본을 많이 뽑아 기댓값을 어림한다. 씨앗을 박아 다시 돌려도 같다.

실행: python docs/ch38/off_policy/figures/make_figures.py
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


# === 그림으로 보는 AI: 고르기만 해도 값이 올라간다 ============================
#
# 참 Q 가 모두 같아도 잡음 섞인 어림에서 최댓값을 고르면 위로 치우친다.
# 비평자 둘을 두고 작은 쪽을 쓰면 줄기는 하지만 **없어지지는 않고**, 액션이
# 적을 때는 오히려 아래로 지나친다. 둘 다 그대로 세어 적는다.

def max_of_noise_is_biased():
    rng = np.random.default_rng(0)
    Ks = [2, 3, 5, 10, 20, 50]
    SD, TRIALS = 0.5, 200000
    single, twin = [], []
    for K in Ks:
        n1 = rng.normal(0.0, SD, (TRIALS, K))
        n2 = rng.normal(0.0, SD, (TRIALS, K))
        single.append(float(n1.max(axis=1).mean()))
        twin.append(float(np.minimum(n1, n2).max(axis=1).mean()))
    single, twin = np.array(single), np.array(twin)
    # 참값은 0 이다 — 하나짜리는 반드시 위로 치우친다
    assert (single > 0).all(), "최댓값이 위로 치우치지 않으면 이 그림의 이야기가 없다"
    assert (twin < single).all(), "둘 중 작은 쪽이 더 낮아야 한다"

    fig, ax = plt.subplots(figsize=(8.8, 4.4))
    ax.axhline(0.0, color=INK, linestyle="--", linewidth=1.4)
    ax.text(Ks[0], 0.035, " the true value", fontsize=8.6, color=INK)
    ax.plot(Ks, single, "o-", color=ACCENT, linewidth=2.2, ms=7,
            label="one critic, take the max")
    ax.plot(Ks, twin, "s-", color=MUTED, linewidth=2.0, ms=6,
            label="two critics, take the smaller")
    ax.fill_between(Ks, twin, single, color=ACCENT, alpha=0.10)

    neg = [(k, t) for k, t in zip(Ks, twin) if t < 0]
    if neg:
        k, t = neg[0]
        ax.annotate(f"below the truth\n{t:+.4f}", xy=(k, t), xytext=(26, 20),
                    textcoords="offset points", fontsize=8.5, color=MUTED,
                    arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.9))
    ax.annotate(f"{single[-1]:+.4f}", xy=(Ks[-1], single[-1]), xytext=(-10, -16),
                textcoords="offset points", fontsize=9, color=ACCENT,
                ha="right")
    ax.set_ylim(min(twin) - 0.22, max(single) * 1.14)

    ax.set_xscale("log")
    ax.set_xticks(Ks)
    ax.set_xticklabels([str(k) for k in Ks])
    ax.set_xlabel("how many actions the max is taken over", fontsize=9.6)
    ax.set_ylabel("average estimate minus the truth", fontsize=9.6)
    ax.set_title("every true value here is zero", fontsize=10.5, color=INK,
                 weight="bold", pad=7)
    ax.legend(fontsize=8.6, framealpha=0.92, loc="upper left")
    ax.grid(alpha=0.13, which="both")
    ax.tick_params(labelsize=8.6, colors=INK)
    for sp in ax.spines.values():
        sp.set_color(INK)
        sp.set_linewidth(0.7)

    fig.text(0.5, 0.005,
             f"noise of {SD}, {TRIALS:,} draws per point. taking the smaller "
             f"of two shrinks the gap without closing it.",
             ha="center", fontsize=8.7, style="italic", color=INK)
    fig.tight_layout()
    fig.savefig(HERE / "max_is_biased.svg", transparent=True,
                bbox_inches="tight")
    plt.close(fig)
    print(f"wrote max_is_biased.svg   single {single[0]:+.4f}..{single[-1]:+.4f}"
          f"   twin {twin[0]:+.4f}..{twin[-1]:+.4f}")


if __name__ == "__main__":
    max_of_noise_is_biased()
