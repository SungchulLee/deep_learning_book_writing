"""26장 한계 절 쪽의 그림.

규칙은 CLAUDE.md 의 Figures 항에 적어 둔 대로다.
- SVG, fonttype='path', transparent
- 글자는 모두 아스키 (matplotlib 기본 폰트에 한글이 없다)

오른쪽 판의 수는 **여기서 직접 재어** 얻는다. 씨앗을 박고 두 추정량을 각각
20,000 번 뽑아 표준편차를 잰다.

실행: python docs/ch25/limits/figures/make_figures.py
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
BLUE = "#2d6a8f"

# 장난감 목적함수.  E_{z ~ N(mu, sigma^2)} [ (z - C)^2 ] 의 mu 에 대한 미분은
# 정확히 2(mu - C) 다. 두 추정량이 이 값을 얼마나 흔들리며 맞히는지 잰다.
MU, SIGMA, C = 1.0, 1.0, 3.0
TRUE_GRAD = 2 * (MU - C)
TRIALS = 20_000


def _estimators(L, rng):
    """(점수함수 추정량, 리파라메리제이션 추정량) 각각 TRIALS 개."""
    eps = rng.standard_normal((TRIALS, L))
    z = MU + SIGMA * eps

    # 점수함수(REINFORCE): f(z) * d/dmu log q(z) = f(z) * (z - mu) / sigma^2
    score = ((z - C) ** 2 * (z - MU) / SIGMA ** 2).mean(axis=1)

    # 리파라메리제이션: z = mu + sigma*eps 이므로 d/dmu f(z) = 2(z - C)
    reparam = (2 * (z - C)).mean(axis=1)
    return score, reparam


def gradient_variance():
    rng = np.random.default_rng(0)
    Ls = [1, 2, 5, 10, 20, 50, 100]
    sd_score, sd_rep = [], []
    s1 = r1 = None
    for L in Ls:
        s, r = _estimators(L, rng)
        if L == 1:                 # 왼쪽 판과 오른쪽 판이 **같은 뽑기**를 쓴다
            s1, r1 = s, r
        sd_score.append(s.std())
        sd_rep.append(r.std())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 4.0))

    # --- 왼쪽: 한 표본일 때 두 추정량이 흩어지는 꼴 -------------------------
    bins = np.linspace(-14, 8, 70)
    ax1.hist(s1, bins=bins, color=BLUE, alpha=0.55,
             label=f"score function  (sd {s1.std():.2f})")
    ax1.hist(r1, bins=bins, color=ACCENT, alpha=0.75,
             label=f"reparameterised  (sd {r1.std():.2f})")
    ax1.axvline(TRUE_GRAD, color=INK, linestyle="--", linewidth=1.4)
    ax1.annotate(f"true gradient\n{TRUE_GRAD:.0f}", xy=(TRUE_GRAD, 0),
                 xytext=(TRUE_GRAD - 5.6, ax1.get_ylim()[1] * 0.62),
                 fontsize=8.5, color=INK,
                 arrowprops=dict(arrowstyle="->", color=INK, lw=0.9))
    ax1.set_xlabel("gradient estimate from one sample", fontsize=9)
    ax1.set_ylabel(f"count out of {TRIALS:,}", fontsize=9)
    ax1.set_title("both are unbiased. one is usable.", fontsize=10,
                  color=INK, weight="bold")
    ax1.legend(fontsize=8, framealpha=0.9)

    # --- 오른쪽: 표본을 늘려도 --------------------------------------------
    ax2.loglog(Ls, sd_score, "o-", color=BLUE, linewidth=1.8, ms=5,
               label="score function")
    ax2.loglog(Ls, sd_rep, "s--", color=ACCENT, linewidth=1.8, ms=5,
               label="reparameterised")
    ax2.set_xlabel("samples drawn per gradient step", fontsize=9)
    ax2.set_ylabel("standard deviation of the estimate", fontsize=9)
    ax2.set_title("the gap does not close", fontsize=10, color=INK,
                  weight="bold")
    ax2.legend(fontsize=8, framealpha=0.9)
    ax2.grid(alpha=0.13, which="both")
    ratio = sd_score[0] / sd_rep[0]
    ax2.annotate(f"{ratio:.1f}x wider\nat one sample",
                 xy=(Ls[0], sd_score[0]), xytext=(2.2, sd_score[0] * 0.95),
                 fontsize=8.5, color=INK,
                 arrowprops=dict(arrowstyle="->", color=INK, lw=0.9))

    for ax in (ax1, ax2):
        ax.tick_params(labelsize=8.5, colors=INK)
        for s in ax.spines.values():
            s.set_color(INK)
            s.set_linewidth(0.7)

    fig.text(0.5, -0.06,
             "toy objective E[(z-3)^2] with z ~ N(mu, 1), whose gradient in mu "
             "is exactly -4.\n"
             f"measured over {TRIALS:,} draws - both estimators are unbiased; "
             "they differ only in spread.",
             ha="center", fontsize=8.3, style="italic", color=INK)
    fig.tight_layout()
    fig.savefig(HERE / "gradient_variance.svg", transparent=True,
                bbox_inches="tight")
    plt.close(fig)
    print(f"wrote gradient_variance.svg  L=1: score sd {sd_score[0]:.3f}  "
          f"reparam sd {sd_rep[0]:.3f}  ratio {ratio:.2f}  "
          f"means {s1.mean():+.4f} / {r1.mean():+.4f}")


if __name__ == "__main__":
    gradient_variance()
