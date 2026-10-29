"""21장 EM 절의 그림.

규칙은 CLAUDE.md 의 Figures 항에 적어 둔 대로다.
- SVG, fonttype='path', transparent
- 글자는 모두 아스키 (matplotlib 기본 폰트에 한글이 없다)

수는 전부 여기서 실제로 돌려 얻는다. 지어낸 수를 그림에 넣지 않는다.

실행: python docs/ch20/em/figures/make_figures.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["svg.fonttype"] = "path"

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal

HERE = Path(__file__).resolve().parent
INK = "#2f3337"
ACCENT = "#8a4b2a"


def style(ax):
    ax.tick_params(labelsize=8, colors=INK)
    for s in ax.spines.values():
        s.set_color(INK)
        s.set_linewidth(0.7)


# === 겹치는 성분 다섯으로 만든 자료 =========================================

def make_data(n=300, seed=0):
    rng = np.random.default_rng(seed)
    mu = np.array([[0., 0.], [1.3, .3], [.6, 1.4], [2.2, 1.8], [-1.1, 1.]])
    cov = [np.array([[.8, .45], [.45, .6]]),
           np.array([[.7, -.4], [-.4, .8]]),
           np.array([[.6, .1], [.1, .55]]),
           np.array([[.9, .3], [.3, .5]]),
           np.array([[.5, -.2], [-.2, .9]])]
    w = np.array([.28, .24, .2, .16, .12])
    comp = rng.choice(5, size=n, p=w)
    return np.vstack([rng.multivariate_normal(mu[c], cov[c]) for c in comp])


def em(X, K=5, iters=120, seed=0, reg=1e-6):
    """표준 EM. 걸음마다 관측 로그 우도를 기록한다."""
    r = np.random.default_rng(seed)
    n, d = X.shape
    mu = X[r.choice(n, K, replace=False)]
    cov = [np.cov(X.T) + reg * np.eye(d) for _ in range(K)]
    pi = np.ones(K) / K
    hist = []
    for _ in range(iters):
        logp = np.column_stack([
            np.log(pi[k] + 1e-300)
            + multivariate_normal.logpdf(X, mu[k], cov[k], allow_singular=True)
            for k in range(K)])
        m = logp.max(1, keepdims=True)
        hist.append((m.ravel() + np.log(np.exp(logp - m).sum(1))).sum())
        R = np.exp(logp - m)
        R /= R.sum(1, keepdims=True)
        Nk = R.sum(0) + 1e-12
        pi = Nk / n
        mu = (R.T @ X) / Nk[:, None]
        cov = [((R[:, k, None] * (X - mu[k])).T @ (X - mu[k]) / Nk[k]
                + reg * np.eye(d)) for k in range(K)]
    return np.array(hist)


# === 증명한 것과 증명하지 않은 것 ===========================================

def monotone_but_not_global():
    X = make_data()
    runs = [em(X, seed=s) for s in range(12)]
    finals = np.array([h[-1] for h in runs])
    steps = np.concatenate([np.diff(h) for h in runs])
    viol = int((steps < -1e-8).sum())
    assert viol == 0, "단조 증가가 깨지면 그림의 주장이 틀린다"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 4.1),
                                   gridspec_kw={"width_ratios": [1.3, 1]})

    for h in runs:
        ax1.plot(h, color=ACCENT, linewidth=1.0, alpha=0.55)
    ax1.set_xlabel("EM iteration", fontsize=9)
    ax1.set_ylabel("observed log-likelihood", fontsize=9)
    ax1.set_title("every step goes up - 12 runs, 0 violations",
                  fontsize=10, color=INK, weight="bold")
    ax1.set_ylim(finals.min() - 42, finals.max() + 4)
    ax1.grid(alpha=0.13)
    style(ax1)
    ax1.text(0.97, 0.06,
             f"smallest single step: {steps.min():+.1e}",
             transform=ax1.transAxes, ha="right", fontsize=8, color=INK,
             alpha=0.85)

    ax2.scatter(finals, np.arange(12), s=46, color=INK, alpha=0.75, zorder=3)
    ax2.set_ylim(-2.2, 13.4)        # 라벨이 테두리에 걸리지 않게 자리를 둔다
    ax2.set_yticks([])
    ax2.set_xlabel("log-likelihood it settled at", fontsize=9)
    ax2.set_ylabel("run", fontsize=9)
    ax2.set_title(f"and all 12 stop somewhere different",
                  fontsize=10, color=INK, weight="bold")
    ax2.grid(alpha=0.13, axis="x")
    style(ax2)
    ax2.text(0.5, 0.055,
             f"spread {finals.min():.2f} to {finals.max():.2f}"
             f"   ({finals.max() - finals.min():.2f} apart)",
             transform=ax2.transAxes, ha="center", fontsize=8.5,
             color=ACCENT)

    fig.text(0.5, -0.04,
             "the 1977 theorem is the left panel. the right panel is what it "
             "does not say.",
             ha="center", fontsize=8.8, style="italic", color=INK)
    fig.tight_layout()
    fig.savefig(HERE / "em_monotone.svg", transparent=True,
                bbox_inches="tight")
    plt.close(fig)
    print(f"wrote em_monotone.svg   violations {viol}   "
          f"distinct {len(set(np.round(finals, 2)))}/12   "
          f"spread {finals.max() - finals.min():.3f}")




# === 그림으로 보는 AI: 최대 우도인데 최대가 없다 ==============================
#
# 성분 하나의 평균을 관측점에 붙이고 그 분산을 줄이면 로그 우도가 -log(sigma)
# 로 끝없이 자란다. 십년자리마다 +ln10 = +2.303 이다. 분산을 묶으면 반대로
# 내려간다. 수는 모두 여기서 직접 셈한다.

def _gmm_ll(x, mu, sig, w):
    comp = np.array([wk * np.exp(-0.5 * ((x - mk) / sk) ** 2)
                     / (np.sqrt(2 * np.pi) * sk)
                     for mk, sk, wk in zip(mu, sig, w)])
    return np.log(comp.sum(0) + 1e-300).sum()


def unbounded_likelihood():
    rng = np.random.default_rng(0)
    x = np.concatenate([rng.normal(-2, 1, 60), rng.normal(2, 1, 60)])
    x0, xm, xs = x[0], x.mean(), x.std()

    es = np.arange(0, -61, -2.0)
    sig = 10.0 ** es
    free = np.array([_gmm_ll(x, [x0, xm], [s, xs], [.5, .5]) for s in sig])
    tied = np.array([_gmm_ll(x, [x0, xm], [s, s], [.5, .5]) for s in sig])

    # 기울기가 십년자리마다 ln10 인가 (멀리서)
    slope = (free[-1] - free[-6]) / (es[-6] - es[-1])
    assert abs(slope - np.log(10)) < 1e-6, "기울기가 -log(sigma) 여야 한다"
    base = free[0]
    cross = sig[np.argmax(free > base)]

    fig, ax = plt.subplots(figsize=(7.8, 4.4))
    ax.plot(-es, free, "-", color=ACCENT, linewidth=2.2,
            label="each component its own variance")
    ax.plot(-es, tied, "--", color=INK, linewidth=1.9,
            label="one shared variance")
    ax.axhline(base, color=INK, linestyle=":", linewidth=1.1, alpha=0.7)
    ax.annotate(f"value at sigma = 1", xy=(3, base), xytext=(6, 14),
                textcoords="offset points", fontsize=8, color=INK, alpha=0.8)
    ax.annotate(f"crosses it at sigma = {cross:.0e},\nand keeps going",
                xy=(-np.log10(cross), base), xytext=(-78, 44),
                textcoords="offset points", ha="center", fontsize=8.2,
                color=ACCENT,
                arrowprops=dict(arrowstyle="->", color=ACCENT, lw=0.9))
    ax.text(0.97, 0.10, f"slope {slope:.3f} per decade  (= ln 10)",
            transform=ax.transAxes, ha="right", fontsize=8.4, color=ACCENT)

    ax.set_xlabel("how far the variance is driven down  "
                  "(decades below 1)", fontsize=9)
    ax.set_ylabel("log-likelihood", fontsize=9)
    ax.set_title("pin one mean to a data point and shrink its variance",
                 fontsize=10, color=INK, weight="bold")
    # 묶은 쪽은 곧바로 바닥을 뚫고 내려간다. 보이는 데까지만 두고 간 곳을 적는다.
    ax.set_ylim(base - 220, free.max() + 30)
    deep = tied[np.argmin(np.abs(-es - 4))]
    ax.annotate(f"off the bottom:\n{deep:,.0f} at 4 decades",
                xy=(4, base - 200), xytext=(34, 28),
                textcoords="offset points", fontsize=8, color=INK,
                arrowprops=dict(arrowstyle="->", color=INK, lw=0.9,
                                alpha=0.8))
    ax.legend(fontsize=8.4, framealpha=0.92, loc="lower left")
    ax.grid(alpha=0.13)
    style(ax)

    fig.text(0.5, -0.04,
             "there is no maximum to find. the useful estimate is a local "
             "one, not the largest.",
             ha="center", fontsize=8.6, style="italic", color=INK)
    fig.tight_layout()
    fig.savefig(HERE / "unbounded_likelihood.svg", transparent=True,
                bbox_inches="tight")
    plt.close(fig)
    print(f"wrote unbounded_likelihood.svg   slope {slope:.4f}  "
          f"crosses at {cross:.0e}")


if __name__ == "__main__":
    monotone_but_not_global()
    unbounded_likelihood()
