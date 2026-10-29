"""21장 베이즈 뉴럴 넷트웍 절의 그림.

규칙은 CLAUDE.md 의 Figures 항에 적어 둔 대로다.
- SVG, fonttype='path', transparent
- 글자는 모두 아스키 (matplotlib 기본 폰트에 한글이 없다)

수는 전부 여기서 실제로 돌려 얻는다.

실행: python docs/ch20/bnn/figures/make_figures.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["svg.fonttype"] = "path"

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
INK = "#2f3337"
ACCENT = "#8a4b2a"
COOL = "#2d6a8f"


def style(ax):
    ax.tick_params(labelsize=8, colors=INK)
    for s in ax.spines.values():
        s.set_color(INK)
        s.set_linewidth(0.7)


# === 그림으로 보는 AI: 폭을 늘리면 사전이 어디로 가는가 ======================
#
# 은닉층 하나짜리 tanh 넷에서 사전 함수값을 직접 뽑는다.
#   f(x) = sum_j v_j tanh(w_j x + b_j)
# v 의 분산을 그대로 두면 f 의 크기가 sqrt(H) 로 자라고, 1/H 로 줄이면
# 제자리에 머무르며 주변분포가 가우스로 간다. 닐(1995)의 극한이 그것이다.

def _prior_f(H, n_draw, scale_with_H, seed, x0=0.0):
    rng = np.random.default_rng(seed)
    ow = (1.0 / H) if scale_with_H else 1.0
    W = rng.standard_normal((n_draw, H))
    b = rng.standard_normal((n_draw, H))
    v = rng.standard_normal((n_draw, H)) * np.sqrt(ow)
    return (v * np.tanh(W * x0 + b)).sum(axis=1)


def width_limit():
    Hs = np.array([1, 2, 8, 32, 128, 512, 2048])
    fixed = np.array([_prior_f(H, 20000, False, 7).std() for H in Hs])
    scaled = np.array([_prior_f(H, 20000, True, 7).std() for H in Hs])

    ratio = fixed[2:] / np.sqrt(Hs[2:])
    assert ratio.std() < 0.01, "고정 스케일이면 sqrt(H) 로 자라야 한다"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 4.2))

    # --- 왼쪽: 폭에 따라 함수의 크기 ---------------------------------------
    ax1.loglog(Hs, fixed, "o-", color=INK, linewidth=1.9, ms=5,
               label="same weight variance")
    ax1.loglog(Hs, scaled, "s--", color=ACCENT, linewidth=1.9, ms=5,
               label="variance scaled as 1/H")
    ax1.loglog(Hs, fixed[0] * np.sqrt(Hs), "-", color=COOL, alpha=0.85,
               linewidth=5.0, zorder=0, label="sqrt(H), drawn underneath")
    ax1.set_xlabel("hidden units H", fontsize=9)
    ax1.set_ylabel("prior std of f(0)", fontsize=9)
    ax1.set_title("widen the net and the prior moves", fontsize=9.8,
                  color=INK, weight="bold")
    ax1.legend(fontsize=8, framealpha=0.92, loc="upper left")
    ax1.grid(alpha=0.13, which="both")
    style(ax1)
    ax1.text(0.97, 0.30,
             f"at H = {Hs[-1]}:  {fixed[-1]:.1f}  vs  {scaled[-1]:.2f}",
             transform=ax1.transAxes, ha="right", fontsize=8.2, color=ACCENT)

    # --- 오른쪽: 1/H 로 맞추면 가우스로 간다 --------------------------------
    for H, col, a in ((1, COOL, 0.55), (2048, ACCENT, 0.55)):
        f = _prior_f(H, 20000, True, 7)
        ax2.hist(f, bins=90, density=True, color=col, alpha=a,
                 label=f"H = {H}", edgecolor="none")
    g = np.linspace(-2.6, 2.6, 300)
    s = _prior_f(2048, 20000, True, 7).std()
    ax2.plot(g, np.exp(-g ** 2 / (2 * s ** 2)) / np.sqrt(2 * np.pi * s ** 2),
             color=INK, linewidth=1.8, label="Gaussian")
    p_small = stats.normaltest(_prior_f(1, 5000, True, 7)).pvalue
    p_big = stats.normaltest(_prior_f(2048, 5000, True, 7)).pvalue
    ax2.set_xlabel("f(0) under the prior", fontsize=9)
    ax2.set_ylabel("density", fontsize=9)
    ax2.set_title("and with 1/H it becomes Gaussian", fontsize=9.8,
                  color=INK, weight="bold")
    ax2.legend(fontsize=8, framealpha=0.92)
    ax2.set_xlim(-2.6, 2.6)
    ax2.grid(alpha=0.12)
    style(ax2)
    ax2.set_ylim(0, 1.72)
    ax2.text(0.03, 0.955,
             f"normality test p\n  {p_small:.3f} at H = 1\n"
             f"  {p_big:.2f} at H = 2048",
             transform=ax2.transAxes, ha="left", va="top", fontsize=8,
             color=ACCENT)

    fig.text(0.5, -0.03,
             "the width is not a free knob. holding the weight prior fixed "
             "changes what you believe about functions.",
             ha="center", fontsize=8.6, style="italic", color=INK)
    fig.tight_layout()
    fig.savefig(HERE / "width_limit.svg", transparent=True,
                bbox_inches="tight")
    plt.close(fig)
    print(f"wrote width_limit.svg   fixed {fixed[-1]:.2f} vs scaled "
          f"{scaled[-1]:.3f} at H={Hs[-1]}   sqrt-law ratio sd {ratio.std():.4f}")


if __name__ == "__main__":
    width_limit()
