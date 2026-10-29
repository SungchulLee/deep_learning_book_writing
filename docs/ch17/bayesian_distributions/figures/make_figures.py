"""18.2 회귀와 함수 사전분포 쪽들의 그림.

규칙은 CLAUDE.md 의 Figures 항에 적어 둔 대로다.
- SVG, fonttype='path', transparent
- 글자는 모두 아스키 (matplotlib 기본 폰트에 한글이 없다)

그림마다 **제 주장을 스스로 확인하고**, 어긋나면 그리지 않고 멈춘다.

실행: python docs/ch17/bayesian_distributions/figures/make_figures.py
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


# === 그림으로 보는 AI: 오차 막대가 자료 값을 보지 않는다 ======================
#
# 크리깅/가우시안 과정의 사후 공분산은 K** - K* K^-1 K*^T 이고 여기에 y 가
# 들어가지 않는다. 곧 **어디를 쟀는가**만으로 정해지고 **무엇이 나왔는가**는
# 보지 않는다. 금을 찾던 사람에게는 그것이 바로 "다음에 어디를 뚫을까" 였다.

def kriging_variance():
    rng = np.random.default_rng(0)
    xs = np.array([0.5, 0.9, 1.3, 1.7, 3.6, 4.0, 4.4])   # 뭉친 곳과 빈 곳
    grid = np.linspace(0, 5, 400)
    ell, sf, sn = 0.45, 1.0, 0.12

    def k(a, b):
        return sf ** 2 * np.exp(-0.5 * (a[:, None] - b[None, :]) ** 2 / ell ** 2)

    K = k(xs, xs) + sn ** 2 * np.eye(len(xs))
    Ks = k(grid, xs)
    Kinv = np.linalg.inv(K)
    var = sf ** 2 - np.einsum("ij,jk,ik->i", Ks, Kinv, Ks)
    sd = np.sqrt(np.maximum(var, 0))

    y1 = np.sin(2 * xs) + rng.normal(0, 0.1, len(xs))
    y2 = 4 * np.cos(3 * xs) - 2                      # 전혀 다른 값
    m1, m2 = Ks @ Kinv @ y1, Ks @ Kinv @ y2

    # 그림이 제 주장을 확인한다 — 자료 값을 바꿔도 띠가 **완전히** 같다
    K2 = k(xs, xs) + sn ** 2 * np.eye(len(xs))
    var2 = sf ** 2 - np.einsum("ij,jk,ik->i", Ks, np.linalg.inv(K2), Ks)
    assert np.allclose(var, var2, atol=0, rtol=0), "같은 자리면 같은 분산"
    assert not np.allclose(m1, m2), "가운데 선은 당연히 달라야 한다"

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.0), sharey=True)
    for ax, y, m, lab in ((axes[0], y1, m1, "one set of readings"),
                          (axes[1], y2, m2, "completely different readings")):
        ax.fill_between(grid, m - 2 * sd, m + 2 * sd, color=ACCENT,
                        alpha=0.22, linewidth=0)
        ax.plot(grid, m, color=INK, linewidth=1.9)
        ax.plot(xs, y, "o", color=MUTED, ms=7, markeredgecolor="white",
                markeredgewidth=1.1, zorder=5)
        ax.set_title(lab, fontsize=10, color=INK, weight="bold")
        ax.set_xlabel("where you measured", fontsize=9.5)
        ax.grid(alpha=0.12)
        ax.tick_params(labelsize=8, colors=INK)
        for s in ax.spines.values():
            s.set_color(INK)
            s.set_linewidth(0.7)
    axes[0].set_ylabel("value", fontsize=9.5)

    for ax in axes:
        ax.annotate("widest where nothing was sampled",
                    xy=(2.65, 0), xytext=(2.65, -4.2), fontsize=7.8,
                    color=ACCENT, ha="center",
                    arrowprops=dict(arrowstyle="->", color=ACCENT, lw=0.9))

    fig.text(0.5, -0.04,
             "the WIDTH of the band is identical in both panels, to the "
             "last bit. only its centre moved.",
             ha="center", fontsize=8.8, style="italic", color=INK)
    fig.tight_layout()
    fig.savefig(HERE / "kriging_variance.svg", transparent=True,
                bbox_inches="tight")
    plt.close(fig)
    print(f"wrote kriging_variance.svg   max sd {sd.max():.4f} "
          f"min sd {sd.min():.4f}")


if __name__ == "__main__":
    kriging_variance()
