"""20.2 몬테카를로 쪽들의 그림.

규칙은 CLAUDE.md 의 Figures 항에 적어 둔 대로다.
- SVG, fonttype='path', transparent
- 글자는 모두 아스키 (matplotlib 기본 폰트에 한글이 없다)

자료를 읽지 않는다. 씨앗을 박아 두어 다시 돌려도 같은 수가 나온다.

실행: python docs/ch19/monte_carlo/figures/make_figures.py
"""
import itertools
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["svg.fonttype"] = "path"

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
INK = "#2f3337"
ACCENT = "#8a4b2a"


# === 그림으로 보는 AI: 차원이 올라가면 자리가 뒤바뀐다 ========================
#
# 같은 수의 점으로 [0,1]^d 에서 exp(sum x) 를 적분한다. 참값은 (e-1)^d 다.
# 격자(중점 규칙)는 낮은 차원에서 압도적이지만 차원이 오르면 무너지고,
# 무작위 점은 거의 제자리다. 무작위 쪽 오차는 확률변수이므로 씨앗 스무 개의
# 제곱평균으로 적는다 — 한 번만 돌리면 들쭉날쭉해서 읽을 수 없다.

def grid_vs_random():
    def f(X):
        return np.exp(X.sum(axis=-1))

    target = 10_000
    dims = [1, 2, 3, 4, 5, 6, 7]
    g_err, m_err, n_pts = [], [], []

    for d in dims:
        true = (math.e - 1) ** d
        G = max(2, int(round(target ** (1 / d))))
        n = G ** d
        pts = (np.arange(G) + 0.5) / G
        grid = np.array(list(itertools.product(pts, repeat=d)))
        g_err.append(abs(f(grid).mean() - true) / true)
        errs = [abs(f(np.random.default_rng(s).random((n, d))).mean() - true)
                / true for s in range(20)]
        m_err.append(float(np.sqrt(np.mean(np.square(errs)))))
        n_pts.append(n)

    # 이 그림이 주장하는 것
    assert g_err[0] < m_err[0] / 100, "낮은 차원에서는 격자가 크게 이겨야 한다"
    assert g_err[-1] > m_err[-1], "높은 차원에서는 자리가 뒤바뀌어야 한다"
    cross = next(d for d, g, m in zip(dims, g_err, m_err) if g > m)

    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    ax.semilogy(dims, g_err, "s--", color=INK, lw=1.9, ms=7,
                label="an even grid")
    ax.semilogy(dims, m_err, "o-", color=ACCENT, lw=2.2, ms=7,
                label="points thrown at random")
    ax.axvline(cross, color=ACCENT, ls=":", lw=1.3, alpha=0.8)
    ax.annotate(f"they swap places\nat d = {cross}", xy=(cross, m_err[-2]),
                xytext=(-116, -40), textcoords="offset points", fontsize=8.8,
                color=ACCENT,
                arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.0))
    ax.annotate(f"{g_err[0]:.0e}", xy=(dims[0], g_err[0]), xytext=(8, 2),
                textcoords="offset points", fontsize=8.4, color=INK)
    ax.annotate(f"{g_err[-1]:.0e}", xy=(dims[-1], g_err[-1]), xytext=(-40, 6),
                textcoords="offset points", fontsize=8.4, color=INK)

    ax.set_xlabel("dimensions of the integral", fontsize=9.5)
    ax.set_ylabel("relative error (log)", fontsize=9.5)
    ax.set_title("the same budget of points, spent two ways", fontsize=10.5,
                 color=INK, weight="bold", pad=8)
    ax.set_xticks(dims)
    ax.legend(fontsize=9, framealpha=0.92)
    ax.grid(alpha=0.13, which="both")
    ax.tick_params(labelsize=8.5, colors=INK)
    for s in ax.spines.values():
        s.set_color(INK)
        s.set_linewidth(0.7)

    fig.text(0.5, -0.04,
             f"about {target:,} points each ({min(n_pts):,} to {max(n_pts):,}; "
             f"a grid needs a whole number per side). over this range the grid's "
             f"error grows {g_err[-1] / g_err[0]:.0e} times; the random "
             f"points' about {m_err[-1] / m_err[0]:.0f}x.",
             ha="center", fontsize=8.4, style="italic", color=INK)
    fig.tight_layout()
    fig.savefig(HERE / "grid_vs_random.svg", transparent=True,
                bbox_inches="tight")
    plt.close(fig)
    print(f"wrote grid_vs_random.svg   grid {g_err[0]:.1e}->{g_err[-1]:.1e}   "
          f"mc {m_err[0]:.1e}->{m_err[-1]:.1e}   cross d={cross}")


if __name__ == "__main__":
    grid_vs_random()
