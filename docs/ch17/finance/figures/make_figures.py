"""18.5 금융 응용 쪽들의 그림.

규칙은 CLAUDE.md 의 Figures 항에 적어 둔 대로다.
- SVG, fonttype='path', transparent
- 글자는 모두 아스키 (matplotlib 기본 폰트에 한글이 없다)

자료를 읽지 않으므로 어디서 돌려도 된다.
그림마다 **제 주장을 스스로 확인하고**, 어긋나면 그리지 않고 멈춘다.

실행: python docs/ch17/finance/figures/make_figures.py
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


# === 그림으로 보는 AI: 최적화기가 극대화하는 것은 추정 오차다 =================
#
# Michaud 1989. 평균-분산 최적화는 기대수익이 우연히 높게 추정된 자산에
# 몰린다. 표본 안에서는 눈부시고 표본 밖에서는 **아무 생각 없는 1/N 에
# 두 배 가까이 진다.** 수는 모두 여기서 셈한다.

def error_maximization():
    rng = np.random.default_rng(0)
    N, T, R = 10, 60, 2000                   # 자산 10, 월별 5년
    mu_true = np.full(N, 0.008)
    mu_true[:3] += 0.002
    S = np.full((N, N), 0.0016)
    np.fill_diagonal(S, 0.0036)
    L = np.linalg.cholesky(S)

    ins, outs, eq, mx = [], [], [], []
    for _ in range(R):
        X = mu_true + rng.normal(0, 1, (T, N)) @ L.T
        w = np.linalg.solve(np.cov(X.T), X.mean(0))
        w = w / np.abs(w).sum()
        mx.append(np.abs(w).max())
        Y = mu_true + rng.normal(0, 1, (T, N)) @ L.T      # 새 기간
        ins.append(X @ w)
        outs.append(Y @ w)
        eq.append(Y @ (np.ones(N) / N))
    ins, outs, eq = np.array(ins), np.array(outs), np.array(eq)

    def sr(a):
        return a.mean() / a.std() * np.sqrt(12)

    s_in, s_out, s_eq = sr(ins.ravel()), sr(outs.ravel()), sr(eq.ravel())
    beat = float(np.mean(outs.mean(1) > eq.mean(1)))

    # 그림이 제 주장을 확인한다
    assert s_in > s_out * 3, "표본 안이 표본 밖보다 훨씬 좋아 보여야 한다"
    assert s_out < s_eq, "최적화가 1/N 에 져야 이 이야기가 선다"
    assert beat < 0.25, "이기는 판이 드물어야 한다"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.6, 4.2),
                                   gridspec_kw={"width_ratios": [1, 1.2]})

    bars = [("optimised,\nin sample", s_in, ACCENT),
            ("optimised,\nout of sample", s_out, ACCENT),
            ("equal weights,\nout of sample", s_eq, MUTED)]
    for i, (lab, v, col) in enumerate(bars):
        ax1.bar(i, v, 0.62, color=col, alpha=0.5 if i == 0 else 0.85)
        ax1.text(i, v + 0.04, f"{v:.3f}", ha="center", fontsize=9.6,
                 color=col, weight="bold")
    ax1.set_xticks(range(3))
    ax1.set_xticklabels([b[0] for b in bars], fontsize=8.4)
    ax1.set_ylabel("Sharpe ratio", fontsize=9.5)
    ax1.set_title("what it looks like, and what it is", fontsize=10,
                  color=INK, weight="bold")
    ax1.set_ylim(0, s_in * 1.18)
    ax1.grid(alpha=0.12, axis="y")

    diff = outs.mean(1) - eq.mean(1)
    ax2.hist(diff * 1200, bins=60, color=ACCENT, alpha=0.78,
             edgecolor="white", linewidth=0.4)
    ax2.axvline(0, color=INK, linewidth=2.0)
    ax2.text(0, ax2.get_ylim()[1] * 0.96, "  equal weights win to the left",
             fontsize=8.4, color=INK, va="top")
    ax2.set_xlabel("optimised minus equal, annualised return (%)",
                   fontsize=9.5)
    ax2.set_ylabel("runs", fontsize=9.5)
    ax2.set_title(f"the optimiser wins {beat*100:.1f}% of the time",
                  fontsize=10, color=INK, weight="bold")
    ax2.grid(alpha=0.12)

    for ax in (ax1, ax2):
        ax.tick_params(labelsize=8, colors=INK)
        for s in ax.spines.values():
            s.set_color(INK)
            s.set_linewidth(0.7)

    fig.text(0.5, -0.04,
             f"{R:,} runs, {N} assets, {T} months. every asset truly has "
             f"almost the same mean - the optimiser is chasing noise.",
             ha="center", fontsize=8.8, style="italic", color=INK)
    fig.tight_layout()
    fig.savefig(HERE / "error_maximization.svg", transparent=True,
                bbox_inches="tight")
    plt.close(fig)
    print(f"wrote error_maximization.svg   in {s_in:.3f} out {s_out:.3f} "
          f"eq {s_eq:.3f}  beat {beat:.3f}  maxw {np.mean(mx):.3f}")


if __name__ == "__main__":
    error_maximization()
