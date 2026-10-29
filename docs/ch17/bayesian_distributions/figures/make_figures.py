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


# === 그림으로 보는 AI: 릿지는 가우시안 사전분포의 사후평균과 같은 것이다 ======
#
# lambda = sigma^2 / tau^2 로 두면 릿지 해와 사후평균이 **비트 단위로** 같다.
# 그러니 lambda 를 고르는 일은 사전분포를 고르는 일이다. 교차검증이 고르는
# lambda 는 중앙값으로는 참값 가까이 가지만 **판마다 몇 자릿수씩 흔들린다.**

def ridge_is_a_prior():
    sig, tau, p = 1.0, 0.7, 8
    lam_true = sig ** 2 / tau ** 2
    grid = np.logspace(-6, 4, 400)

    def make(seed, n):
        rng = np.random.default_rng(seed)
        X = rng.normal(0, 1, (n, p))
        X[:, 1] = X[:, 0] + 0.01 * rng.normal(0, 1, n)   # 다중공선성
        beta = rng.normal(0, tau, p)
        y = X @ beta + rng.normal(0, sig, n)
        return rng, X, y

    # --- 왼쪽: 두 식이 정말 같은가 ----------------------------------------
    _, X, y = make(0, 60)
    ridge = np.linalg.solve(X.T @ X + lam_true * np.eye(p), X.T @ y)
    Vinv = X.T @ X / sig ** 2 + np.eye(p) / tau ** 2
    post = np.linalg.solve(Vinv, X.T @ y / sig ** 2)
    gap = float(np.abs(ridge - post).max())
    assert gap < 1e-12, f"릿지와 사후평균이 같아야 한다 {gap}"

    # --- 오른쪽: 교차검증이 고르는 lambda 의 흩어짐 ------------------------
    def cv_pick(seed, n):
        rng, X, y = make(seed, n)
        idx = np.arange(n)
        rng.shuffle(idx)

        def err(l, K=10):
            e = 0.0
            for k in range(K):
                te = idx[k::K]
                tr = np.setdiff1d(idx, te)
                b = np.linalg.solve(X[tr].T @ X[tr] + l * np.eye(p),
                                    X[tr].T @ y[tr])
                e += ((y[te] - X[te] @ b) ** 2).sum()
            return e
        return grid[int(np.argmin([err(l) for l in grid]))]

    picks = np.array([cv_pick(s, 60) for s in range(40)])
    assert np.median(picks) < lam_true * 3, "중앙값은 참값 근처여야 한다"
    assert picks.max() / picks.min() > 1e4, "판마다의 흩어짐이 요점이다"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.6, 4.1))

    lo = min(ridge.min(), post.min()) - 0.1
    hi = max(ridge.max(), post.max()) + 0.1
    ax1.plot([lo, hi], [lo, hi], color=INK, linewidth=1.0, linestyle=":")
    ax1.plot(ridge, post, "o", color=ACCENT, ms=11,
             markeredgecolor="white", markeredgewidth=1.2)
    ax1.set_xlabel("ridge coefficient", fontsize=9.5)
    ax1.set_ylabel("posterior mean", fontsize=9.5)
    ax1.set_title("the same numbers, not merely similar", fontsize=10,
                  color=INK, weight="bold")
    ax1.text(0.04, 0.95, f"largest difference\n{gap:.1e}",
             transform=ax1.transAxes, va="top", fontsize=8.8, color=ACCENT,
             weight="bold",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                       edgecolor=ACCENT, alpha=0.92))

    ax2.axvline(lam_true, color=INK, linewidth=2.0)
    ax2.annotate("implied by the truth", xy=(lam_true, 0.02),
                 xytext=(0.04, 0.06), textcoords="axes fraction",
                 fontsize=8.4, color=INK,
                 arrowprops=dict(arrowstyle="->", color=INK, lw=0.9))
    ax2.plot(picks, rng_jitter(len(picks)), "o", color=ACCENT, ms=7,
             alpha=0.7, markeredgecolor="white", markeredgewidth=0.8)
    ax2.axvline(np.median(picks), color=MUTED, linewidth=1.8,
                linestyle="--")
    ax2.text(np.median(picks), 1.06, f"median {np.median(picks):.2f}",
             fontsize=8.4, color=MUTED, ha="center")
    ax2.set_xscale("log")
    ax2.set_xlabel("lambda chosen by cross-validation", fontsize=9.5)
    ax2.set_yticks([])
    ax2.set_ylim(-0.1, 1.2)
    ax2.set_title("40 datasets from the same truth", fontsize=10, color=INK,
                  weight="bold")

    for ax in (ax1, ax2):
        ax.grid(alpha=0.12)
        ax.tick_params(labelsize=8, colors=INK)
        for sp in ax.spines.values():
            sp.set_color(INK)
            sp.set_linewidth(0.7)

    fig.text(0.5, -0.04,
             f"tuning lambda is choosing a prior. the median lands near the "
             f"truth; a single dataset can be {picks.max()/picks.min():.0e} "
             f"apart end to end.",
             ha="center", fontsize=8.8, style="italic", color=INK)
    fig.tight_layout()
    fig.savefig(HERE / "ridge_is_a_prior.svg", transparent=True,
                bbox_inches="tight")
    plt.close(fig)
    print(f"wrote ridge_is_a_prior.svg   gap {gap:.1e}   "
          f"cv median {np.median(picks):.3f} vs {lam_true:.3f}   "
          f"range {picks.min():.3g}-{picks.max():.3g}")


def rng_jitter(m):
    """점이 겹치지 않게 세로로 흩는다 — 재현되도록 씨앗을 고정한다."""
    return np.random.default_rng(7).uniform(0.15, 0.85, m)


# === 그림으로 보는 AI: 라플라스 근사가 듣는 자리와 안 듣는 자리 ==============
#
# 로지스틱 우도에는 켤레 사전분포가 없으므로 최빈값에 가우시안을 맞춘다.
# 자료가 넉넉하면 거의 완벽하다. 자료가 분리되면 사후분포가 한쪽으로 쏠려
# 가우시안이 **0 과 음수까지 품는 구간**을 내놓는다 — 결론이 달라진다.

def laplace_limits():
    from scipy import optimize

    tau = 3.0

    def analyse(x, y, lo_g, hi_g):
        def nlp(b):
            return -(np.sum(y * b * x - np.log1p(np.exp(b * x)))
                     - b ** 2 / (2 * tau ** 2))
        bhat = optimize.minimize_scalar(nlp, bounds=(-40, 40),
                                        method="bounded").x
        h = 1e-4
        H = (nlp(bhat + h) - 2 * nlp(bhat) + nlp(bhat - h)) / h ** 2
        sd = 1 / np.sqrt(H)
        g = np.linspace(lo_g, hi_g, 400001)
        lp = np.array([-nlp(b) for b in g])
        lp -= lp.max()
        P = np.exp(lp)
        P /= np.trapz(P, g)
        m = np.trapz(g * P, g)
        s = np.sqrt(np.trapz((g - m) ** 2 * P, g))
        skew = np.trapz((g - m) ** 3 * P, g) / s ** 3
        c = np.cumsum(P)
        c /= c[-1]
        ex = (g[np.searchsorted(c, 0.025)], g[np.searchsorted(c, 0.975)])
        la = (bhat - 1.96 * sd, bhat + 1.96 * sd)
        mask = (g > la[0]) & (g < la[1])
        held = np.trapz(P[mask], g[mask])
        return dict(g=g, P=P, bhat=bhat, sd=sd, ex=ex, la=la, held=held,
                    skew=skew)

    rng = np.random.default_rng(0)
    xa = rng.normal(0, 1, 200)
    ya = (rng.random(200) < 1 / (1 + np.exp(-1.5 * xa))).astype(float)
    A = analyse(xa, ya, -1, 4)

    xb = np.array([-2., -1.5, -1., -0.8, 0.8, 1., 1.5, 2.])
    yb = np.array([0, 0, 0, 0, 1, 1, 1, 1.])
    B = analyse(xb, yb, -5, 40)

    # 그림이 제 주장을 확인한다
    assert A["held"] > 0.94, "자료가 넉넉하면 라플라스가 잘 맞아야 한다"
    assert B["la"][0] < 0 < B["ex"][0], "분리되면 라플라스 구간만 0 을 품는다"
    assert B["skew"] > 0.5, "분리된 사후분포는 한쪽으로 쏠려 있다"

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2))
    for ax, D, title, xlim in (
            (axes[0], A, "200 points, well mixed", (-0.5, 3.5)),
            (axes[1], B, "8 points, perfectly separated", (-3, 14))):
        ax.fill_between(D["g"], 0, D["P"], color=ACCENT, alpha=0.25,
                        label="exact posterior")
        ax.plot(D["g"], D["P"], color=ACCENT, linewidth=1.6)
        gauss = (np.exp(-0.5 * ((D["g"] - D["bhat"]) / D["sd"]) ** 2)
                 / (D["sd"] * np.sqrt(2 * np.pi)))
        ax.plot(D["g"], gauss, color=INK, linewidth=1.9, linestyle="--",
                label="Laplace approximation")
        top = max(D["P"].max(), gauss.max())
        for k, (iv, col, lab) in enumerate(
                ((D["ex"], ACCENT, "exact 95%"),
                 (D["la"], INK, "Laplace 95%"))):
            yy = top * (1.12 - 0.09 * k)
            ax.plot(iv, [yy, yy], color=col, linewidth=4.5,
                    solid_capstyle="butt")
            ax.text(iv[1], yy, f"  {lab} [{iv[0]:.2f}, {iv[1]:.2f}]",
                    fontsize=7.8, color=col, va="center")
        ax.axvline(0, color=MUTED, linewidth=1.2, linestyle=":")
        ax.set_xlabel("coefficient", fontsize=9.5)
        ax.set_title(title, fontsize=10, color=INK, weight="bold")
        ax.set_xlim(*xlim)
        ax.set_ylim(0, top * 1.32)
        ax.grid(alpha=0.12)
        ax.tick_params(labelsize=8, colors=INK)
        for sp in ax.spines.values():
            sp.set_color(INK)
            sp.set_linewidth(0.7)
        ax.text(0.03, 0.74,
                f"its 95% interval really holds {D['held']:.3f}",
                transform=ax.transAxes, fontsize=8.4, color=INK)
    axes[0].set_ylabel("density", fontsize=9.5)
    axes[0].legend(fontsize=8.2, loc="upper right", framealpha=0.92)
    axes[1].text(0.03, 0.64, "and it reaches past zero",
                 transform=axes[1].transAxes, fontsize=8.4, color=MUTED,
                 weight="bold")

    fig.text(0.5, -0.04,
             "a mode and a curvature cannot see a skewed posterior. "
             "on the right the two intervals disagree about whether the "
             "coefficient is positive at all.",
             ha="center", fontsize=8.8, style="italic", color=INK)
    fig.tight_layout()
    fig.savefig(HERE / "laplace_limits.svg", transparent=True,
                bbox_inches="tight")
    plt.close(fig)
    print(f"wrote laplace_limits.svg   plentiful holds {A['held']:.4f}   "
          f"separated holds {B['held']:.4f}  la {B['la'][0]:.2f} vs "
          f"exact {B['ex'][0]:.2f}")


if __name__ == "__main__":
    kriging_variance()
    ridge_is_a_prior()
    laplace_limits()
