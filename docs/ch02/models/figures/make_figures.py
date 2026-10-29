"""2장 고전 모델 쪽들의 그림.

규칙은 CLAUDE.md 의 Figures 항에 적어 둔 대로다.
- SVG, fonttype='path', transparent
- 글자는 모두 아스키 (matplotlib 기본 폰트에 한글이 없다)

수는 모두 여기서 실제로 돌려 얻는다.

실행: python docs/ch02/models/figures/make_figures.py
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


def style(ax):
    ax.tick_params(labelsize=8, colors=INK)
    for s in ax.spines.values():
        s.set_color(INK)
        s.set_linewidth(0.7)


# === 그림으로 보는 AI: Cover-Hart 1967 의 복도 ===============================
#
# R* <= R <= 2R*. 베이즈 오차를 아는 문제를 만들어 1-NN 이 어디에 앉는지 잰다.
# 두 정규분포, 공분산 단위행렬, 평균 (+-delta/2, 0) 이므로 R* = Phi(-delta/2).

def cover_hart_corridor():
    from scipy.stats import norm
    from sklearn.neighbors import KNeighborsClassifier

    delta = 2.0
    r_star = float(norm.cdf(-delta / 2))

    rng = np.random.default_rng(0)
    M = 400_000
    lab = rng.integers(0, 2, M)
    x = rng.standard_normal((M, 2)) + np.where(
        lab[:, None] == 1, [delta / 2, 0], [-delta / 2, 0])
    eta = 1 / (1 + np.exp(-delta * x[:, 0]))
    r_inf = float((2 * eta * (1 - eta)).mean())

    lte = rng.integers(0, 2, 20000)
    Xte = rng.standard_normal((20000, 2)) + np.where(
        lte[:, None] == 1, [delta / 2, 0], [-delta / 2, 0])

    ns = [50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
    mean, sd = [], []
    for n in ns:
        e = []
        for rep in range(5):
            r = np.random.default_rng(1000 + rep)
            ltr = r.integers(0, 2, n)
            Xtr = r.standard_normal((n, 2)) + np.where(
                ltr[:, None] == 1, [delta / 2, 0], [-delta / 2, 0])
            kn = KNeighborsClassifier(n_neighbors=1).fit(Xtr, ltr)
            e.append(1 - kn.score(Xte, lte))
        mean.append(np.mean(e))
        sd.append(np.std(e))
    mean, sd = np.array(mean), np.array(sd)

    fig, ax = plt.subplots(figsize=(7.8, 4.5))

    ax.fill_between([ns[0] * 0.75, ns[-1] * 1.35], r_star, 2 * r_star,
                    color=ACCENT, alpha=0.10)
    ax.axhline(r_star, color=BLUE, linewidth=1.7, linestyle="--")
    ax.axhline(2 * r_star, color=ACCENT, linewidth=1.7, linestyle="--")
    ax.axhline(r_inf, color=INK, linewidth=1.2, linestyle=":")

    ax.errorbar(ns, mean, yerr=sd, fmt="o-", color=INK, linewidth=1.8,
                ms=5.5, capsize=3, label="1-NN, measured")

    ax.text(ns[-1] * 1.4, r_star, f"  Bayes error R*\n  {r_star:.4f}",
            fontsize=8.5, color=BLUE, va="center")
    ax.text(ns[-1] * 1.4, 2 * r_star, f"  the ceiling 2R*\n  {2*r_star:.4f}",
            fontsize=8.5, color=ACCENT, va="center")
    ax.text(ns[-1] * 1.4, r_inf, f"  predicted limit\n  {r_inf:.4f}",
            fontsize=8.5, color=INK, va="center")

    ax.annotate("already there by n = 100",
                xy=(100, mean[1]), xytext=(260, 0.175), fontsize=8.5,
                color=INK,
                arrowprops=dict(arrowstyle="->", color=INK, lw=0.9))

    ax.set_xscale("log")
    ax.set_xlim(ns[0] * 0.75, ns[-1] * 1.35)
    ax.set_ylim(0.13, 0.345)
    ax.set_xlabel("training points", fontsize=9.5)
    ax.set_ylabel("test error", fontsize=9.5)
    ax.set_title("the corridor Cover and Hart proved in 1967",
                 fontsize=10.5, color=INK, weight="bold", pad=8)
    ax.grid(alpha=0.13, which="both")
    style(ax)

    fig.text(0.5, -0.04,
             "two Gaussians, unit covariance, means 2.0 apart - so the Bayes "
             "error is known exactly, not estimated.",
             ha="center", fontsize=8.5, style="italic", color=INK)
    fig.tight_layout()
    fig.savefig(HERE / "cover_hart.svg", transparent=True,
                bbox_inches="tight")
    plt.close(fig)
    print(f"wrote cover_hart.svg   R* {r_star:.4f}  R_inf {r_inf:.4f}  "
          f"measured {mean[-1]:.4f}  ceiling {2*r_star:.4f}")


if __name__ == "__main__":
    cover_hart_corridor()
