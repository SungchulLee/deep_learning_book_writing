"""18.4 모델 비교 쪽들의 그림.

규칙은 CLAUDE.md 의 Figures 항에 적어 둔 대로다.
- SVG, fonttype='path', transparent
- 글자는 모두 아스키 (matplotlib 기본 폰트에 한글이 없다)

자료를 읽지 않으므로 어디서 돌려도 된다.
그림마다 **제 주장을 스스로 확인하고**, 어긋나면 그리지 않고 멈춘다.

실행: python docs/ch17/model_comparison/figures/make_figures.py
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

# 제프리스의 등급 경계는 10^(n/2) 다 — 1, 3.16, 10, 31.6, 100
JEFFREYS = [(1.0, 10 ** 0.5, "bare mention"),
            (10 ** 0.5, 10.0, "substantial"),
            (10.0, 10 ** 1.5, "strong"),
            (10 ** 1.5, 100.0, "very strong"),
            (100.0, 1e4, "decisive")]


# === 그림으로 보는 AI: p 를 붙박아 두고 n 만 키우면 ===========================
#
# Lindley 1957. z 를 1.96 에 고정하면 p 는 어느 n 에서나 정확히 0.05 인데,
# 영가설을 지지하는 베이즈 인자는 sqrt(n) 처럼 끝없이 자란다. 같은 자료를 두고
# 한쪽은 늘 기각하고 다른 쪽은 끝내 "결정적으로 영가설" 이라고 말한다.

def lindley():
    z = stats.norm.ppf(0.975)          # p = 0.05 를 정확히 고정
    sig = tau = 1.0
    ns = np.array([10 ** k for k in range(1, 8)], dtype=float)
    r = ns * tau ** 2 / sig ** 2
    bf01 = np.sqrt(1 + r) * np.exp(-z ** 2 / 2 * r / (1 + r))

    # 그림이 제 주장을 확인한다
    p = 2 * (1 - stats.norm.cdf(z))
    assert abs(p - 0.05) < 1e-12, "p 가 0.05 로 고정되어 있어야 한다"
    assert np.all(np.diff(bf01) > 0), "BF01 은 n 과 함께 단조증가해야 한다"
    assert bf01[0] < 1 < bf01[-1], "작은 n 에서는 H1, 큰 n 에서는 H0"
    assert bf01[-1] > 100, "끝에서는 제프리스 기준 decisive 여야 한다"

    fig, ax = plt.subplots(figsize=(8.4, 4.6))

    for lo, hi, lab in JEFFREYS:
        ax.axhspan(lo, hi, color=ACCENT, alpha=0.055)
        ax.axhline(lo, color=ACCENT, linewidth=0.7, alpha=0.45)
        mid = (lo * min(hi, 1800)) ** 0.5          # decisive 띠도 반드시 적는다
        ax.text(ns[0] * 0.78, mid, lab, fontsize=7.6, color=ACCENT,
                va="center", ha="right", alpha=0.95)

    ax.axhline(1.0, color=INK, linewidth=1.2, linestyle=":")
    ax.text(ns[-1] * 0.9, 0.80, "even odds", fontsize=8, color=INK,
            va="top", ha="right")

    ax.loglog(ns, bf01, "o-", color=INK, linewidth=2.0, ms=6, zorder=5)
    for n, b in zip(ns, bf01):
        if n in (ns[0], ns[3], ns[-1]):
            ax.annotate(f"{b:.2f}", xy=(n, b), xytext=(4, -13),
                        textcoords="offset points", fontsize=8.4, color=INK)

    ax.set_xlabel("sample size", fontsize=9.5)
    ax.set_ylabel("Bayes factor for the null", fontsize=9.5)
    ax.set_title("p = 0.0500 at every point on this curve", fontsize=10.5,
                 color=INK, weight="bold", pad=8)
    ax.set_xlim(ns[0] * 0.33, ns[-1] * 3)
    ax.set_ylim(0.3, 2000)
    ax.grid(alpha=0.12, which="both")
    ax.tick_params(labelsize=8, colors=INK)
    for s in ax.spines.values():
        s.set_color(INK)
        s.set_linewidth(0.7)

    fig.text(0.5, -0.04,
             "the frequentist rejects at every n. on Jeffreys's own scale the "
             "same data end up decisive FOR the null.",
             ha="center", fontsize=8.8, style="italic", color=INK)
    fig.tight_layout()
    fig.savefig(HERE / "lindley.svg", transparent=True, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote lindley.svg   BF01 {bf01[0]:.3f} -> {bf01[-1]:.1f}")


if __name__ == "__main__":
    lindley()
