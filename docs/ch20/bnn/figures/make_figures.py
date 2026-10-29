"""21장 베이즈 뉴럴 넷트웍 절의 그림.

규칙은 CLAUDE.md 의 Figures 항에 적어 둔 대로다.
- SVG, fonttype='path', transparent
- 글자는 모두 아스키 (matplotlib 기본 폰트에 한글이 없다)

수는 전부 여기서 실제로 돌려 얻는다.

실행: python docs/ch20/bnn/figures/make_figures.py
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
COOL = "#2d6a8f"


def style(ax):
    ax.tick_params(labelsize=8, colors=INK)
    for s in ax.spines.values():
        s.set_color(INK)
        s.set_linewidth(0.7)


# === 그림으로 보는 AI: 폭을 늘리면 사전이 어디로 가는가 ======================
#
# 은닉층 하나짜리 tanh 넷에서 사전 함수값을 직접 뽑는다.
#   f(x) = sum_j v_j tanh(w_j x + b_j)
# v 의 분산을 그대로 두면 f 의 크기가 sqrt(H) 로 자라고, 1/H 로 줄이면
# 제자리에 머무르며 주변분포가 가우스로 간다. 닐(1995)의 극한이 그것이다.

def _prior_f(H, n_draw, scale_with_H, seed, x0=0.0):
    rng = np.random.default_rng(seed)
    ow = (1.0 / H) if scale_with_H else 1.0
    W = rng.standard_normal((n_draw, H))
    b = rng.standard_normal((n_draw, H))
    v = rng.standard_normal((n_draw, H)) * np.sqrt(ow)
    return (v * np.tanh(W * x0 + b)).sum(axis=1)


def width_limit():
    Hs = np.array([1, 2, 8, 32, 128, 512, 2048])
    fixed = np.array([_prior_f(H, 20000, False, 7).std() for H in Hs])
    scaled = np.array([_prior_f(H, 20000, True, 7).std() for H in Hs])

    ratio = fixed[2:] / np.sqrt(Hs[2:])
    assert ratio.std() < 0.01, "고정 스케일이면 sqrt(H) 로 자라야 한다"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 4.2))

    # --- 왼쪽: 폭에 따라 함수의 크기 ---------------------------------------
    ax1.loglog(Hs, fixed, "o-", color=INK, linewidth=1.9, ms=5,
               label="same weight variance")
    ax1.loglog(Hs, scaled, "s--", color=ACCENT, linewidth=1.9, ms=5,
               label="variance scaled as 1/H")
    ax1.loglog(Hs, fixed[0] * np.sqrt(Hs), "-", color=COOL, alpha=0.85,
               linewidth=5.0, zorder=0, label="sqrt(H), drawn underneath")
    ax1.set_xlabel("hidden units H", fontsize=9)
    ax1.set_ylabel("prior std of f(0)", fontsize=9)
    ax1.set_title("widen the net and the prior moves", fontsize=9.8,
                  color=INK, weight="bold")
    ax1.legend(fontsize=8, framealpha=0.92, loc="upper left")
    ax1.grid(alpha=0.13, which="both")
    style(ax1)
    ax1.text(0.97, 0.30,
             f"at H = {Hs[-1]}:  {fixed[-1]:.1f}  vs  {scaled[-1]:.2f}",
             transform=ax1.transAxes, ha="right", fontsize=8.2, color=ACCENT)

    # --- 오른쪽: 1/H 로 맞추면 가우스로 간다 --------------------------------
    for H, col, a in ((1, COOL, 0.55), (2048, ACCENT, 0.55)):
        f = _prior_f(H, 20000, True, 7)
        ax2.hist(f, bins=90, density=True, color=col, alpha=a,
                 label=f"H = {H}", edgecolor="none")
    g = np.linspace(-2.6, 2.6, 300)
    s = _prior_f(2048, 20000, True, 7).std()
    ax2.plot(g, np.exp(-g ** 2 / (2 * s ** 2)) / np.sqrt(2 * np.pi * s ** 2),
             color=INK, linewidth=1.8, label="Gaussian")
    p_small = stats.normaltest(_prior_f(1, 5000, True, 7)).pvalue
    p_big = stats.normaltest(_prior_f(2048, 5000, True, 7)).pvalue
    ax2.set_xlabel("f(0) under the prior", fontsize=9)
    ax2.set_ylabel("density", fontsize=9)
    ax2.set_title("and with 1/H it becomes Gaussian", fontsize=9.8,
                  color=INK, weight="bold")
    ax2.legend(fontsize=8, framealpha=0.92)
    ax2.set_xlim(-2.6, 2.6)
    ax2.grid(alpha=0.12)
    style(ax2)
    ax2.set_ylim(0, 1.72)
    ax2.text(0.03, 0.955,
             f"normality test p\n  {p_small:.3f} at H = 1\n"
             f"  {p_big:.2f} at H = 2048",
             transform=ax2.transAxes, ha="left", va="top", fontsize=8,
             color=ACCENT)

    fig.text(0.5, -0.03,
             "the width is not a free knob. holding the weight prior fixed "
             "changes what you believe about functions.",
             ha="center", fontsize=8.6, style="italic", color=INK)
    fig.tight_layout()
    fig.savefig(HERE / "width_limit.svg", transparent=True,
                bbox_inches="tight")
    plt.close(fig)
    print(f"wrote width_limit.svg   fixed {fixed[-1]:.2f} vs scaled "
          f"{scaled[-1]:.3f} at H={Hs[-1]}   sqrt-law ratio sd {ratio.std():.4f}")




# === 그림으로 보는 AI: 웨이트 하나에 수 하나 대신 분포 하나 ===================
#
# 블런델 일행(2015)의 생각은 간단하다 — 모든 웨이트를 수 대신 분포로 둔다.
# 그 값은 파라메터 수가 두 배가 되는 것으로 치른다. 그림의 D 는 직접 센다.

def weights_as_distributions():
    layers = [2, 3, 1]
    D = sum(a * b + b for a, b in zip(layers[:-1], layers[1:]))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.0, 3.9))

    def draw_net(ax, distributional):
        pos = {}
        for li, nl in enumerate(layers):
            for ni in range(nl):
                pos[(li, ni)] = (li * 2.4, ni - (nl - 1) / 2)
        for li in range(len(layers) - 1):
            for a in range(layers[li]):
                for b in range(layers[li + 1]):
                    x0, y0 = pos[(li, a)]
                    x1, y1 = pos[(li + 1, b)]
                    ax.plot([x0, x1], [y0, y1], color=INK, alpha=0.3,
                            linewidth=0.9, zorder=1)
                    mx, my = (x0 + x1) / 2, (y0 + y1) / 2
                    if distributional:
                        t = np.linspace(-1, 1, 40)
                        ax.plot(mx + t * 0.42,
                                my + 0.30 * np.exp(-(t * 2.1) ** 2),
                                color=ACCENT, linewidth=1.5, zorder=3)
                        ax.plot([mx - 0.42, mx + 0.42], [my, my],
                                color=ACCENT, linewidth=0.7, alpha=0.6,
                                zorder=3)
                    else:
                        ax.plot([mx], [my], marker="o", color=INK,
                                markersize=4.5, zorder=3)
        for (li, ni), (x, y) in pos.items():
            ax.add_patch(plt.Circle((x, y), 0.19, facecolor="white",
                                    edgecolor=INK, linewidth=1.1, zorder=4))
        ax.set_xlim(-0.7, 5.1)
        ax.set_ylim(-2.0, 1.8)
        ax.set_aspect("equal")
        ax.axis("off")

    draw_net(ax1, False)
    ax1.set_title("one number per weight", fontsize=10, color=INK,
                  weight="bold")
    ax1.text(2.2, -1.5, f"{D} numbers to learn", ha="center", fontsize=9,
             color=INK)
    ax1.text(2.2, -1.76, "(weights and biases; biases not drawn)",
             ha="center", fontsize=7, color=INK, alpha=0.65)

    draw_net(ax2, True)
    ax2.set_title("one distribution per weight", fontsize=10, color=ACCENT,
                  weight="bold")
    ax2.text(2.2, -1.5, f"{2 * D} numbers to learn", ha="center",
             fontsize=9, color=ACCENT)
    ax2.text(2.2, -1.76, "(a mean and a scale for each of the same 13)",
             ha="center", fontsize=7, color=ACCENT, alpha=0.8)

    fig.text(0.5, -0.02,
             "Hinton and Van Camp 1993  ->  Graves 2011  ->  Blundell et al. "
             "2015.  the price is exactly double.",
             ha="center", fontsize=8.6, style="italic", color=INK)
    fig.tight_layout()
    fig.savefig(HERE / "weights_as_distributions.svg", transparent=True,
                bbox_inches="tight")
    plt.close(fig)
    print(f"wrote weights_as_distributions.svg   D={D}  2D={2*D}")




# === 그림으로 보는 AI: 증거는 복잡함에 제동을 건다 ============================
#
# 맥케이(1992)의 증거 틀이 오컴의 면도날을 저절로 담는다는 주장을 잰다.
# 가우스 선형 모델이라 로그 주변우도에 닫힌 식이 있다.
# 한 번만 돌리면 씨앗 하나의 우연을 보여 주게 되므로 30 개 씨앗으로 잰다.

_TRUE_DEG, _NOISE, _NOBS = 3, 0.25, 14
_ALPHA, _MAXDEG, _SEEDS = 1.0, 10, 30


def _evidence_curve(seed):
    rng = np.random.default_rng(seed)
    x = np.sort(rng.uniform(-1, 1, _NOBS))
    y = (0.6 * x ** 3 - 0.9 * x ** 2 + 0.2 * x + 0.3
         + rng.normal(0, _NOISE, _NOBS))
    beta = 1.0 / _NOISE ** 2
    ev, mse = [], []
    for d in range(_MAXDEG):
        Phi = np.vander(x, d + 1, increasing=True)
        M = d + 1
        A = _ALPHA * np.eye(M) + beta * Phi.T @ Phi
        mN = beta * np.linalg.solve(A, Phi.T @ y)
        E = beta / 2 * np.sum((y - Phi @ mN) ** 2) + _ALPHA / 2 * mN @ mN
        _, ldA = np.linalg.slogdet(A)
        ev.append(M / 2 * np.log(_ALPHA) + _NOBS / 2 * np.log(beta)
                  - E - ldA / 2 - _NOBS / 2 * np.log(2 * np.pi))
        mse.append(np.mean((y - Phi @ mN) ** 2))
    return np.array(ev), np.array(mse)


def evidence_brakes():
    curves = [_evidence_curve(s) for s in range(_SEEDS)]
    evs = np.array([c[0] for c in curves])
    mses = np.array([c[1] for c in curves])
    picks = evs.argmax(axis=1)
    deg = np.arange(_MAXDEG)

    frac_le = (picks <= _TRUE_DEG).mean()
    # 사전분포가 있으므로 MAP 적합이고, 씨앗 하나하나는 아주 조금 출렁일 수
    # 있다(여기서 가장 큰 증가가 1e-3 이하). 중앙값 곡선은 단조 감소다.
    med_mse = np.median(mses, axis=0)
    assert np.all(np.diff(med_mse) <= 1e-12), "훈련 오차 중앙값은 단조 감소여야 한다"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 4.2))

    # --- 왼쪽: 30 개 씨앗의 띠 ---------------------------------------------
    ev_c = evs - evs.max(axis=1, keepdims=True)        # 씨앗마다 최고점을 0 으로
    lo, md, hi = np.percentile(ev_c, [10, 50, 90], axis=0)
    ax1.fill_between(deg, lo, hi, color=ACCENT, alpha=0.18,
                     label="log evidence, 10-90% of 30 seeds")
    ax1.plot(deg, md, "-o", color=ACCENT, linewidth=2.1, ms=4.5,
             label="median log evidence")
    ax1.axvline(_TRUE_DEG, color=INK, linestyle=":", linewidth=1.2, alpha=0.75)
    ax1.annotate("true degree", xy=(_TRUE_DEG, md.max()),
                 xytext=(9, -16), textcoords="offset points", fontsize=8,
                 color=INK)
    ax1.set_xlabel("polynomial degree", fontsize=9)
    ax1.set_ylabel("log evidence, relative to each seed's best", fontsize=8.8)
    ax1.set_title("the evidence stops paying for complexity", fontsize=9.8,
                  color=INK, weight="bold")

    axb = ax1.twinx()
    axb.plot(deg, med_mse, "--s", color=COOL, linewidth=1.8,
             ms=4, label="median training MSE")
    axb.set_ylabel("training MSE", fontsize=9, color=COOL)
    axb.tick_params(labelsize=8, colors=COOL)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = axb.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, fontsize=7.6, framealpha=0.92,
               loc="lower right")
    ax1.grid(alpha=0.13)
    style(ax1)

    # --- 오른쪽: 실제로 고른 차수 ------------------------------------------
    counts = np.bincount(picks, minlength=_MAXDEG)
    ax2.bar(deg, counts, color=[ACCENT if d <= _TRUE_DEG else INK
                                for d in deg], alpha=0.85)
    ax2.axvline(_TRUE_DEG, color=INK, linestyle=":", linewidth=1.2, alpha=0.75)
    ax2.set_xlabel("degree the evidence picked", fontsize=9)
    ax2.set_ylabel(f"out of {_SEEDS} seeds", fontsize=9)
    ax2.set_title("but it is not an oracle", fontsize=9.8, color=INK,
                  weight="bold")
    ax2.set_xticks(deg)
    ax2.grid(alpha=0.13, axis="y")
    style(ax2)
    ax2.text(0.97, 0.95,
             f"picks <= {_TRUE_DEG}:  {frac_le*100:.0f}%\n"
             f"most common:  {np.argmax(counts)}\n"
             f"true degree:  {_TRUE_DEG}",
             transform=ax2.transAxes, ha="right", va="top", fontsize=8.2,
             color=INK)

    fig.text(0.5, -0.03,
             "training error always wants more. the evidence does not - but "
             "it tends to stop one short.",
             ha="center", fontsize=8.6, style="italic", color=INK)
    fig.tight_layout()
    fig.savefig(HERE / "evidence_brakes.svg", transparent=True,
                bbox_inches="tight")
    plt.close(fig)
    print(f"wrote evidence_brakes.svg   <=3: {frac_le*100:.0f}%   "
          f"mode {np.argmax(counts)}   true {_TRUE_DEG}")




# === 그림으로 보는 AI: 웨이트는 멀고 함수는 같다 ==============================
#
# 헥트닐슨(1990)이 보인 대칭 — 은닉 유닛 자리바꿈과(tanh 가 홀함수이므로)
# 부호 뒤집기는 입출력 함수를 바꾸지 않는다. 직접 옮겨 보고 두 거리를 잰다.

def weight_space_symmetry():
    from math import factorial

    rng = np.random.default_rng(0)
    H, D = 6, 3
    W1 = rng.standard_normal((H, D))
    b1 = rng.standard_normal(H)
    w2 = rng.standard_normal(H)
    b2 = 0.4

    def f(X, A, c, v):
        return np.tanh(X @ A.T + c) @ v + b2

    def vec(A, c, v):
        return np.concatenate([A.ravel(), c, v])

    X = rng.standard_normal((500, D))
    p = rng.permutation(H)
    s = rng.choice([-1, 1], H)

    variants = [
        ("original", W1, b1, w2),
        ("permuted", W1[p], b1[p], w2[p]),
        ("signs flipped", W1 * s[:, None], b1 * s, w2 * s),
        ("both", (W1 * s[:, None])[p], (b1 * s)[p], (w2 * s)[p]),
    ]
    y0 = f(X, W1, b1, w2)
    th0 = vec(W1, b1, w2)
    wdist, fdist = [], []
    for _, A, c, v in variants:
        wdist.append(np.linalg.norm(vec(A, c, v) - th0))
        fdist.append(np.abs(f(X, A, c, v) - y0).max())
    assert max(fdist) < 1e-12, "대칭 변환은 함수를 바꾸지 않아야 한다"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 4.2))

    # --- 왼쪽: 한 줄을 따라 그린 함수 넷 ------------------------------------
    t = np.linspace(-3, 3, 300)
    direction = rng.standard_normal(D)
    direction /= np.linalg.norm(direction)
    Xline = t[:, None] * direction
    styles = [("-", ACCENT, 3.4), ("--", INK, 2.0), (":", COOL, 2.0),
              ("-.", "#6b8f2d", 1.6)]
    for (name, A, c, v), (ls, col, lw) in zip(variants, styles):
        ax1.plot(t, f(Xline, A, c, v), ls, color=col, linewidth=lw,
                 label=name)
    ax1.set_xlabel("position along a line through input space", fontsize=8.8)
    ax1.set_ylabel("network output", fontsize=9)
    ax1.set_title("four different weight vectors", fontsize=9.8, color=INK,
                  weight="bold")
    ax1.legend(fontsize=8.2, framealpha=0.92)
    ax1.grid(alpha=0.13)
    style(ax1)

    # --- 오른쪽: 두 거리 ----------------------------------------------------
    idx = np.arange(1, 4)
    w = 0.34
    ax2.bar(idx - w / 2, [wdist[i] for i in idx], w, color=INK, alpha=0.85,
            label="distance in weight space")
    ax2.bar(idx + w / 2, [max(fdist[i], 1e-18) for i in idx], w,
            color=ACCENT, alpha=0.9, label="largest change in output")
    ax2.set_yscale("log")
    ax2.set_ylim(1e-18, 1e3)
    ax2.set_xticks(idx)
    ax2.set_xticklabels([variants[i][0] for i in idx], fontsize=8.5)
    ax2.set_ylabel("size (log)", fontsize=9)
    ax2.set_title("far apart, and identical", fontsize=9.8, color=INK,
                  weight="bold")
    ax2.legend(fontsize=8.2, framealpha=0.92, loc="upper right")
    ax2.grid(alpha=0.13, axis="y", which="major")
    style(ax2)
    # 정확히 0 이면 로그 축에 막대가 안 그려진다 — 빠진 것이 아니라고 적어 둔다
    for i in idx:
        if fdist[i] == 0.0:
            ax2.annotate("exactly 0\n(no bar to draw)", xy=(i + w / 2, 1e-17),
                         xytext=(0, 30), textcoords="offset points",
                         ha="center", fontsize=7.6, color=ACCENT,
                         arrowprops=dict(arrowstyle="->", color=ACCENT,
                                         lw=0.8))

    n_sym = factorial(H) * 2 ** H
    fig.text(0.5, -0.04,
             f"with {H} hidden units at least {factorial(H)} x 2^{H} = "
             f"{n_sym:,} weight vectors give this one function.",
             ha="center", fontsize=8.6, style="italic", color=INK)
    fig.tight_layout()
    fig.savefig(HERE / "weight_symmetry.svg", transparent=True,
                bbox_inches="tight")
    plt.close(fig)
    print(f"wrote weight_symmetry.svg   max |df| {max(fdist):.1e}   "
          f"weight dists {np.round(wdist[1:],2)}   symmetries {n_sym:,}")


if __name__ == "__main__":
    width_limit()
    weights_as_distributions()
    evidence_brakes()
    weight_space_symmetry()
