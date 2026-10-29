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


# === 그림으로 보는 AI: 시장에 기대수익을 물어보는 대신 거꾸로 읽는다 =========
#
# Black-Litterman 의 출발점은 역최적화다. 수익을 예측하는 대신 **시가총액
# 비중이 최적이 되려면 기대수익이 얼마여야 하는가**를 거꾸로 푼다. 그 값을
# 다시 넣으면 시장 비중이 그대로 나온다 (기계 정밀도까지). 과거 표본평균을
# 넣으면 비중이 폭주한다.

def reverse_optimization():
    rng = np.random.default_rng(3)
    N, T, delta = 6, 60, 2.5
    names = ["A", "B", "C", "D", "E", "F"]
    w_mkt = np.array([.35, .22, .16, .12, .09, .06])
    S = np.full((N, N), 0.0012)
    np.fill_diagonal(S, 0.0040)

    pi = delta * S @ w_mkt                     # 시장이 함축하는 기대수익
    X = np.full(N, 0.008) + rng.normal(0, 1, (T, N)) @ np.linalg.cholesky(S).T
    mh = X.mean(0)                             # 과거 표본평균

    def weights(mu):
        return np.linalg.solve(delta * S, mu)

    w_back, w_hist = weights(pi), weights(mh)

    # 그림이 제 주장을 확인한다
    assert np.abs(w_back - w_mkt).max() < 1e-12, "되먹이면 시장 비중이 나온다"
    assert np.abs(w_hist).max() > 1.0, "표본평균을 넣으면 비중이 폭주한다"
    assert (w_hist < 0).sum() >= 1, "공매도가 나타난다"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.8, 4.2))

    xp = np.arange(N)
    ax1.bar(xp - 0.19, pi * 1200, 0.36, color=MUTED, alpha=0.88,
            label="implied by the market")
    ax1.bar(xp + 0.19, mh * 1200, 0.36, color=ACCENT, alpha=0.88,
            label="historical sample mean")
    ax1.axhline(0.008 * 1200, color=INK, linestyle=":", linewidth=1.4)
    ax1.text(N - 0.5, 0.008 * 1200, " the truth, equal for all",
             fontsize=7.8, color=INK, va="bottom", ha="right")
    ax1.set_xticks(xp)
    ax1.set_xticklabels(names, fontsize=9)
    ax1.set_ylabel("expected return, annualised (%)", fontsize=9.5)
    ax1.set_title("two ways to fill the same input", fontsize=10, color=INK,
                  weight="bold")
    ax1.legend(fontsize=8.2, loc="upper left", framealpha=0.93)
    ax1.grid(alpha=0.12, axis="y")

    ax2.bar(xp - 0.19, w_back * 100, 0.36, color=MUTED, alpha=0.88,
            label="feed the implied returns back")
    ax2.bar(xp + 0.19, w_hist * 100, 0.36, color=ACCENT, alpha=0.88,
            label="feed the sample means")
    ax2.plot(xp - 0.19, w_mkt * 100, "k_", ms=16, mew=2.2,
             label="the market weights")
    ax2.axhline(0, color=INK, linewidth=1.2)
    ax2.set_xticks(xp)
    ax2.set_xticklabels(names, fontsize=9)
    ax2.set_ylabel("portfolio weight (%)", fontsize=9.5)
    ax2.set_title("what the optimiser then does", fontsize=10, color=INK,
                  weight="bold")
    ax2.legend(fontsize=8.2, loc="upper left", framealpha=0.93)
    ax2.grid(alpha=0.12, axis="y")
    ax2.set_ylim(min(w_hist.min() * 100 - 20, -55),
                 max(w_hist.max() * 100 + 35, 130))
    ax2.text(0.97, 0.04,
             f"largest difference from the market weights: "
             f"{np.abs(w_back - w_mkt).max():.0e}",
             transform=ax2.transAxes, ha="right", fontsize=7.8, color=MUTED)

    for ax in (ax1, ax2):
        ax.set_xlabel("asset", fontsize=9.5)
        ax.tick_params(labelsize=8, colors=INK)
        for s in ax.spines.values():
            s.set_color(INK)
            s.set_linewidth(0.7)

    fig.text(0.5, -0.04,
             "every asset truly has the same expected return. the market's "
             "implied numbers stay tame; the sample means do not.",
             ha="center", fontsize=8.8, style="italic", color=INK)
    fig.tight_layout()
    fig.savefig(HERE / "reverse_optimization.svg", transparent=True,
                bbox_inches="tight")
    plt.close(fig)
    print(f"wrote reverse_optimization.svg   implied "
          f"{pi.min()*1200:.2f}-{pi.max()*1200:.2f}%  sample "
          f"{mh.min()*1200:.2f}-{mh.max()*1200:.2f}%  weights "
          f"{w_hist.min()*100:.1f}-{w_hist.max()*100:.1f}%")


# === 그림으로 보는 AI: 국면이 없어도 필터는 국면을 본다 ======================
#
# Hamilton 1989. 국면은 관측되지 않으므로 확률로 **추론**한다. 그런데 국면이
# 하나도 없는 자료를 넣어도 필터는 확신에 찬 전환을 수십 번 선언한다.
# 확신의 정도는 국면이 있을 때보다 오히려 **조금 높다.**

def regime_illusion():
    mus = np.array([-0.02, 0.01])
    sd = 0.04
    P = np.array([[0.90, 0.10], [0.05, 0.95]])
    T, R = 400, 200

    def hamilton(y):
        f = np.zeros(len(y))
        p = np.full(2, 0.5)
        for t, yt in enumerate(y):
            pred = P.T @ p
            lik = np.exp(-0.5 * ((yt - mus) / sd) ** 2) / sd
            post = pred * lik
            post /= post.sum()
            f[t] = post[0]
            p = post
        return f

    def draw_regime(rng):
        s = np.zeros(T, int)
        for t in range(1, T):
            s[t] = 0 if rng.random() < P[s[t - 1], 0] else 1
        return mus[s] + rng.normal(0, sd, T), s

    def draw_flat(rng):
        return np.full(T, 0.004) + rng.normal(0, sd, T), None

    stats_ = {}
    for key, gen in (("regime", draw_regime), ("flat", draw_flat)):
        rng = np.random.default_rng(0)
        sw, conf, true_sw = [], [], []
        for _ in range(R):
            y, s = gen(rng)
            f = hamilton(y)
            call = (f > 0.5).astype(int)
            sw.append(int((np.diff(call) != 0).sum()))
            conf.append(float(np.mean((f > 0.8) | (f < 0.2))))
            true_sw.append(0 if s is None else int((np.diff(s) != 0).sum()))
        stats_[key] = (np.mean(sw), np.mean(conf), np.mean(true_sw))

    # 그림이 제 주장을 확인한다
    assert stats_["flat"][0] > 20, "국면이 없어도 전환을 많이 선언한다"
    assert stats_["flat"][1] >= stats_["regime"][1] * 0.9, \
        "확신의 정도가 국면이 있을 때와 비슷하거나 높다"

    rng = np.random.default_rng(0)
    y1, s1 = draw_regime(rng)
    rng = np.random.default_rng(0)
    y2, _ = draw_flat(rng)
    f1, f2 = hamilton(y1), hamilton(y2)

    fig, axes = plt.subplots(2, 1, figsize=(9.2, 5.4), sharex=True)
    for ax, f, s, title, key in (
            (axes[0], f1, s1, "data that really does switch regime",
             "regime"),
            (axes[1], f2, None, "data with no regimes at all - one mean "
             "throughout", "flat")):
        if s is not None:
            ax.fill_between(np.arange(T), 0, 1, where=(s == 0),
                            color=MUTED, alpha=0.16, step="mid",
                            label="the true bad regime")
        ax.plot(f, color=ACCENT, linewidth=1.3)
        ax.axhline(0.5, color=INK, linestyle=":", linewidth=1.0)
        ax.set_ylim(-0.02, 1.02)
        ax.set_ylabel("filtered P(bad)", fontsize=9)
        ax.set_title(title, fontsize=10, color=INK, weight="bold")
        m_sw, m_cf, m_true = stats_[key]
        ax.text(0.985, 0.93,
                f"declares {m_sw:.1f} switches   (truth: {m_true:.1f})\n"
                f"confident {m_cf*100:.1f}% of the time",
                transform=ax.transAxes, ha="right", va="top", fontsize=8.2,
                color=INK,
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                          edgecolor=INK, alpha=0.9))
        if s is not None:
            ax.legend(fontsize=7.8, loc="lower right", framealpha=0.9)
        ax.tick_params(labelsize=8, colors=INK)
        ax.grid(alpha=0.12)
        for sp in ax.spines.values():
            sp.set_color(INK)
            sp.set_linewidth(0.7)
    axes[1].set_xlabel("period", fontsize=9.5)

    fig.text(0.5, -0.03,
             f"averages over {R} runs of {T} periods. the lower panel has no "
             f"regimes to find, and the filter is no less sure of itself.",
             ha="center", fontsize=8.8, style="italic", color=INK)
    fig.tight_layout()
    fig.savefig(HERE / "regime_illusion.svg", transparent=True,
                bbox_inches="tight")
    plt.close(fig)
    print(f"wrote regime_illusion.svg   regime {stats_['regime']}   "
          f"flat {stats_['flat']}")


if __name__ == "__main__":
    error_maximization()
    reverse_optimization()
    regime_illusion()
