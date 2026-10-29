"""18.1 베이즈 기초 쪽들의 그림.

규칙은 CLAUDE.md 의 Figures 항에 적어 둔 대로다.
- SVG, fonttype='path', transparent
- 글자는 모두 아스키 (matplotlib 기본 폰트에 한글이 없다)

자료를 읽지 않으므로 어디서 돌려도 된다.
그림마다 **제 주장을 스스로 확인하고**, 어긋나면 그리지 않고 멈춘다.

실행: python docs/ch17/bayesian_foundations/figures/make_figures.py
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
MUTED = "#2d6a8f"


# === 그림으로 보는 AI: 베이즈가 실제로 그린 것 ================================
#
# 1763년 글의 Postulate 1 은 "square table or plane ABCD" 다. 당구대가 아니다
# (원문에 billiard 는 한 번도 나오지 않는다). 공 W 를 먼저 던져 AD 에 나란한
# 선 os 를 긋고, 공 O 를 n 번 던져 AD 와 os 사이에 떨어지는 횟수를 센다.
#
# 오른쪽은 그 설정을 그대로 돌린 결과다. 균등 사전분포(Postulate 1)에서
# 사후분포는 Beta(p+1, q+1) 이고, 글이 요구한 것은 **두 점 사이의 확률**이다.

def bayes_table():
    rng = np.random.default_rng(1763)
    n = 12
    x_true = 0.62                      # 공 W 가 멈춘 자리
    throws = rng.random(n)
    p = int((throws < x_true).sum())   # AD 와 os 사이 = 사건 M
    q = n - p
    assert p + q == n, "앞뒤 횟수의 합이 n 이어야 한다"

    post = stats.beta(p + 1, q + 1)    # 균등 사전분포에서의 사후분포
    lo, hi = 0.40, 0.80                # "any two degrees of probability"
    prob = post.cdf(hi) - post.cdf(lo)

    # 그림이 제 주장을 스스로 확인한다 — 칠한 넓이가 그 확률과 같은가
    grid = np.linspace(lo, hi, 20001)
    area = np.trapz(post.pdf(grid), grid)      # 이 기계의 numpy 는 1.x 다
    assert abs(area - prob) < 1e-6, f"칠한 넓이 {area} != 확률 {prob}"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.6, 4.2),
                                   gridspec_kw={"width_ratios": [1, 1.12]})

    # --- 왼쪽: square table ABCD ------------------------------------------
    ax1.add_patch(plt.Rectangle((0, 0), 1, 1, facecolor="none",
                                edgecolor=INK, linewidth=1.5))
    for (x, y, lab, ha, va) in ((0, 0, "A", "right", "top"),
                                (1, 0, "B", "left", "top"),
                                (1, 1, "C", "left", "bottom"),
                                (0, 1, "D", "right", "bottom")):
        ax1.text(x, y, f" {lab} ", ha=ha, va=va, fontsize=10, color=INK,
                 weight="bold")

    ax1.plot([x_true, x_true], [0, 1], color=ACCENT, linewidth=2.0)
    ax1.text(x_true, 1.04, "s", ha="center", fontsize=9.5, color=ACCENT,
             weight="bold")
    ax1.text(x_true, -0.07, "o", ha="center", fontsize=9.5, color=ACCENT,
             weight="bold")
    ax1.plot([x_true], [0.52], marker="o", color=ACCENT, markersize=10,
             markeredgecolor="white", markeredgewidth=1.3, zorder=5)
    ax1.text(x_true - 0.03, 0.52, "W ", ha="right", va="center", fontsize=9,
             color=ACCENT)

    ys = np.linspace(0.12, 0.90, n)
    for t, y in zip(throws, ys):
        left = t < x_true
        ax1.plot([t], [y], marker="o", markersize=5.5,
                 color=MUTED if left else INK,
                 alpha=0.9 if left else 0.45, zorder=4)

    ax1.annotate("", xy=(x_true - 0.015, 0.045), xytext=(0.015, 0.045),
                 arrowprops=dict(arrowstyle="<->", color=MUTED, linewidth=1.2))
    ax1.text(x_true / 2, 0.085, f"event M  ({p} of {n})", ha="center",
             fontsize=8.4, color=MUTED)

    ax1.set_title('"the square table or plane ABCD"', fontsize=10, color=INK,
                  weight="bold")
    ax1.text(0.5, -0.20, "Postulate 1, 1763.  the word billiard\n"
                         "appears nowhere in the essay.",
             ha="center", va="top", fontsize=8.4, color=INK, alpha=0.85)
    ax1.set_xlim(-0.1, 1.1)
    ax1.set_ylim(-0.34, 1.14)
    ax1.set_aspect("equal")
    ax1.axis("off")

    # --- 오른쪽: 글이 요구한 것 --------------------------------------------
    xs = np.linspace(0, 1, 800)
    ax2.plot(xs, post.pdf(xs), color=INK, linewidth=1.9)
    shade = np.linspace(lo, hi, 400)
    ax2.fill_between(shade, 0, post.pdf(shade), color=ACCENT, alpha=0.26)
    # 눈금에 이미 0.4 와 0.8 이 있으므로 수를 겹쳐 적지 않는다
    for v in (lo, hi):
        ax2.axvline(v, color=ACCENT, linestyle="--", linewidth=1.1)
    ax2.annotate("", xy=(hi, post.pdf(post.mean()) * 1.06),
                 xytext=(lo, post.pdf(post.mean()) * 1.06),
                 arrowprops=dict(arrowstyle="<->", color=ACCENT,
                                 linewidth=1.2))
    ax2.text((lo + hi) / 2, post.pdf(post.mean()) * 1.09, f"{prob:.3f}",
             ha="center", va="bottom", fontsize=13, color=ACCENT,
             weight="bold")
    ax2.set_xlabel("the unknown chance", fontsize=9)
    ax2.set_ylabel("density", fontsize=9)
    ax2.set_title(f"what the Problem asks for  (p={p}, q={q})", fontsize=10,
                  color=INK, weight="bold")
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, post.pdf(post.mean()) * 1.30)
    ax2.grid(alpha=0.12)
    ax2.tick_params(labelsize=8, colors=INK)
    for s in ax2.spines.values():
        s.set_color(INK)
        s.set_linewidth(0.7)

    fig.text(0.5, -0.03,
             '"the chance that the probability ... lies somewhere between any '
             'two degrees of probability that can be named"',
             ha="center", fontsize=8.6, style="italic", color=INK)
    fig.tight_layout()
    fig.savefig(HERE / "bayes_table.svg", transparent=True,
                bbox_inches="tight")
    plt.close(fig)
    print(f"wrote bayes_table.svg   p={p} q={q}  P({lo},{hi})={prob:.4f}")


if __name__ == "__main__":
    bayes_table()
