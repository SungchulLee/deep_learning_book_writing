"""32.2 고전 이진 모델 쪽들의 그림.

규칙은 CLAUDE.md 의 Figures 항에 적어 둔 대로다.
- SVG, fonttype='path', transparent
- 글자는 모두 아스키 (matplotlib 기본 폰트에 한글이 없다)

자료를 읽지 않으므로 어디서 돌려도 된다. 수는 전부 여기서 실제로 돌려 얻는다.

실행: python docs/ch31/classical_ebm/figures/make_figures.py
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

ALPHA_C = 0.138          # Amit-Gutfreund-Sompolinsky 1985
ALPHAS = (0.05, 0.08, 0.10, 0.12, 0.138, 0.16, 0.20, 0.25)


# === 그림으로 보는 AI: 두 수는 서로 다른 물음에 답한다 ========================
#
# 홉필드 1982 가 적은 수는 0.15N 이고 모의실험(N=30, N=100)에서 나왔다.
# 흔히 인용하는 0.138 은 세 해 뒤 AGS 1985 가 복제법으로 얻은 N->무한 문턱이다.
# 두 기준을 **각각** 재어 나란히 놓으면, 문턱이 있는 것도 없는 것도 사실이다.
#   왼쪽  m : 수렴한 상태가 저장 무늬와 얼마나 겹치는가 (AGS 의 순서변수)
#   오른쪽  : 저장 무늬가 **한 비트도 틀리지 않고** 고정점인 비율

def _hopfield_sweep(N, p, trials, rng, sweeps=8, max_patterns=12):
    """m 과 '완전 안정' 비율을 한 번에 잰다."""
    ms, perfect = [], []
    for _ in range(trials):
        X = rng.choice([-1.0, 1.0], size=(p, N))
        W = (X.T @ X) / N
        np.fill_diagonal(W, 0.0)

        # 완전 안정: 저장 무늬의 모든 비트가 국소장의 부호와 같은가
        perfect.append(((np.sign(X @ W.T) == X).all(axis=1)).mean())

        for k in range(min(p, max_patterns)):
            s = X[k].copy()
            for _ in range(sweeps):
                for i in rng.permutation(N):
                    h = W[i] @ s
                    if h != 0:
                        s[i] = np.sign(h)
            ms.append(abs(s @ X[k]) / N)
    return float(np.mean(ms)), float(np.mean(perfect))


def capacity_two_questions():
    rng = np.random.default_rng(0)
    sizes = ((100, 8), (400, 3))
    m = {N: [] for N, _ in sizes}
    perf = {N: [] for N, _ in sizes}
    for a in ALPHAS:
        for N, tr in sizes:
            mm, pp = _hopfield_sweep(N, max(1, int(round(a * N))), tr, rng)
            m[N].append(mm)
            perf[N].append(pp)

    a = np.array(ALPHAS)
    ic = list(ALPHAS).index(ALPHA_C)

    # --- 그림이 주장하는 바를 그림이 스스로 확인한다 ----------------------
    assert m[400][ic] > 0.95, (
        f"문턱 아래에서 m 이 1 에 가까워야 한다: {m[400][ic]:.3f}")
    assert m[400][-1] < 0.65, (
        f"문턱 위에서 m 이 무너져야 한다: {m[400][-1]:.3f}")
    assert perf[400][ic] < 0.6, (
        f"같은 자리에서 완전 복구는 이미 반 아래여야 한다: {perf[400][ic]:.3f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.6, 4.1), sharex=True)

    for ax, data, title, ylab in (
            (ax1, m, "is the recalled state essentially right?",
             "overlap m after settling"),
            (ax2, perf, "is every stored pattern perfect?",
             "fraction stable in all N bits")):
        for (N, _), col, mk in zip(sizes, (MUTED, ACCENT), ("o-", "s-")):
            ax.plot(a, data[N], mk, color=col, ms=5, linewidth=1.8,
                    label=f"N = {N}")
        ax.axvline(ALPHA_C, color=INK, linestyle="--", linewidth=1.2,
                   alpha=0.7)
        ax.set_xlabel("alpha  =  patterns / N", fontsize=9)
        ax.set_ylabel(ylab, fontsize=9)
        ax.set_title(title, fontsize=10, color=INK, weight="bold")
        ax.set_ylim(0, 1.06)
        ax.grid(alpha=0.13)
        ax.legend(fontsize=8.5, loc="lower left", framealpha=0.9)
        ax.tick_params(labelsize=8, colors=INK)
        for s in ax.spines.values():
            s.set_color(INK)
            s.set_linewidth(0.7)

    ax1.annotate("0.138\n(AGS 1985)", xy=(ALPHA_C, 0.42),
                 xytext=(ALPHA_C + 0.028, 0.30), fontsize=8.2, color=INK,
                 arrowprops=dict(arrowstyle="->", color=INK, lw=0.9))
    ax2.annotate(f"only {perf[400][ic]:.2f}\nat the threshold",
                 xy=(ALPHA_C, perf[400][ic]),
                 xytext=(ALPHA_C + 0.045, perf[400][ic] + 0.30), fontsize=8.2,
                 color=ACCENT, ha="left",
                 arrowprops=dict(arrowstyle="->", color=ACCENT, lw=0.9))

    fig.text(0.5, -0.03,
             "same networks, same runs, two criteria. the threshold governs "
             "the left; on the right the decay starts well before it.",
             ha="center", fontsize=8.8, style="italic", color=INK)
    fig.tight_layout()
    fig.savefig(HERE / "capacity_two_questions.svg", transparent=True,
                bbox_inches="tight")
    plt.close(fig)
    print(f"wrote capacity_two_questions.svg   "
          f"m(400)@0.138 = {m[400][ic]:.3f}   "
          f"perfect(400)@0.138 = {perf[400][ic]:.3f}")




# === 그림으로 보는 AI: 제한이 사 주는 것과 사 주지 않는 것 =====================
#
# 이분 제한은 **조건부** 독립을 정확히 만들어 준다. 그러나 주변분포 p(v) 는
# 여전히 독립이 아니다 — 그 쪽 2 절이 적어 둔 그대로다. 가시 10 개면 1,024 개
# 상태를 전부 더해 p(v) 를 정확히 구할 수 있으므로 그 말을 수로 보인다.

def restriction_buys():
    import itertools

    rng = np.random.default_rng(3)
    nv, nh = 10, 5
    V = np.array(list(itertools.product([0, 1], repeat=nv)), dtype=float)
    W = rng.normal(0, 1.2, (nh, nv))
    a = rng.normal(0, 0.4, nv)
    b = rng.normal(0, 0.4, nh)

    free = -(V @ a) - np.logaddexp(0.0, V @ W.T + b).sum(1)
    p = np.exp(-free - (-free).max())
    p /= p.sum()

    marg = (V * p[:, None]).sum(0)
    q = np.prod(np.where(V == 1, marg, 1 - marg), axis=1)
    kl = float((p * np.log(p / q)).sum())

    C = np.cov(V.T, aweights=p)
    sd = np.sqrt(np.diag(C))
    R = C / np.outer(sd, sd)
    np.fill_diagonal(R, 0.0)
    rmax = float(np.abs(R).max())

    # --- 조건부 독립은 **정확히** 성립하는지도 확인한다 --------------------
    h0 = rng.integers(0, 2, size=nh).astype(float)
    pv_given_h = 1.0 / (1.0 + np.exp(-(a + h0 @ W)))
    joint = np.prod(np.where(V == 1, pv_given_h, 1 - pv_given_h), axis=1)
    assert abs(joint.sum() - 1.0) < 1e-10, "조건부는 곱으로 정확히 분해돼야 한다"
    assert kl > 1e-3, f"주변분포는 독립이 아니어야 한다: {kl}"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 4.1),
                                   gridspec_kw={"width_ratios": [1, 1.15]})

    # --- 왼쪽: 무엇을 지웠는가 --------------------------------------------
    vx = np.linspace(0.12, 0.88, 6)
    hx = np.linspace(0.26, 0.74, 3)
    for i in vx:
        for j in hx:
            ax1.plot([i, j], [0.70, 0.30], color=INK, linewidth=0.6,
                     alpha=0.30, zorder=1)
    ax1.scatter(vx, [0.70] * 6, s=150, color=ACCENT, zorder=3,
                edgecolors="white", linewidths=1.2)
    ax1.scatter(hx, [0.30] * 3, s=150, color=MUTED, zorder=3,
                edgecolors="white", linewidths=1.2)
    for y, lab, col in ((0.70, "visible", ACCENT), (0.30, "hidden", MUTED)):
        ax1.text(0.5, y + 0.105, lab, fontsize=9, color=col, va="center",
                 ha="center", weight="bold")

    # 지운 연결을 점선으로 표시한다
    ax1.plot([vx[1], vx[2]], [0.70, 0.70], color=INK, linewidth=1.4,
             linestyle=":", alpha=0.6)
    ax1.plot([hx[0], hx[1]], [0.30, 0.30], color=INK, linewidth=1.4,
             linestyle=":", alpha=0.6)
    ax1.text(0.5, 0.95, "no visible-visible, no hidden-hidden",
             ha="center", fontsize=8.6, color=INK, style="italic")
    ax1.text(0.5, 0.12,
             "fix one layer and the other\nfactorises exactly.",
             ha="center", fontsize=8.8, color=INK)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0.03, 1.00)
    ax1.axis("off")
    ax1.set_title("what the restriction removes", fontsize=10, color=INK,
                  weight="bold")

    # --- 오른쪽: 그래도 주변분포는 독립이 아니다 ---------------------------
    ax2.loglog(q, p, "o", color=ACCENT, ms=2.6, alpha=0.45,
               rasterized=True)          # 1,024 점을 벡터로 두면 SVG 가 커진다
    lim = [min(q.min(), p.min()) * 0.7, max(q.max(), p.max()) * 1.4]
    ax2.plot(lim, lim, color=INK, linestyle="--", linewidth=1.2)
    ax2.text(0.97, 0.06,
             "if the visible units were independent,\n"
             "every point would sit on the line",
             transform=ax2.transAxes, ha="right", va="bottom",
             fontsize=8.2, color=INK, alpha=0.85)
    ax2.set_xlim(lim)
    ax2.set_ylim(lim)
    ax2.set_xlabel("product of the marginals", fontsize=9)
    ax2.set_ylabel("true p(v)", fontsize=9)
    ax2.set_title("but the marginal is not factorial", fontsize=10,
                  color=INK, weight="bold")
    ax2.text(0.035, 0.955,
             f"KL = {kl:.3f} nats\nlargest pair |corr| = {rmax:.3f}",
             transform=ax2.transAxes, va="top", fontsize=8.6, color=INK,
             bbox=dict(boxstyle="round,pad=0.42", facecolor="white",
                       edgecolor=INK, alpha=0.92))
    ax2.grid(alpha=0.13, which="both")
    ax2.tick_params(labelsize=8, colors=INK)
    for s in ax2.spines.values():
        s.set_color(INK)
        s.set_linewidth(0.7)

    fig.text(0.5, -0.03,
             f"all {len(V):,} visible states summed exactly, "
             f"{nv} visible and {nh} hidden units.",
             ha="center", fontsize=8.6, style="italic", color=INK)
    fig.tight_layout()
    fig.savefig(HERE / "restriction_buys.svg", transparent=True,
                bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"wrote restriction_buys.svg   KL = {kl:.4f}   maxcorr = {rmax:.3f}")



# === 그림으로 보는 AI: 규칙은 단순한데 값은 비싸다 =============================
#
# AHS 1985 의 학습 규칙은 **상관 둘의 차**뿐이고 결합마다 제 두 단위만 보면 된다.
# 그런데 그 둘이 모두 **평형 평균**이라, 갱신 한 번마다 사슬을 충분히 돌려야 한다.
# 작은 BM 에서 정확한 값을 전부 열거해 놓고, 깁스 표본 수를 바꿔 가며 재어 본다.

def bm_price():
    import itertools

    rng = np.random.default_rng(0)
    n, nvis = 10, 6
    S = np.array(list(itertools.product([-1, 1], repeat=n)), dtype=float)
    W = rng.normal(0, 0.35, (n, n))
    W = (W + W.T) / 2
    np.fill_diagonal(W, 0.0)

    E = -0.5 * np.einsum("si,ij,sj->s", S, W, S)
    p = np.exp(-E - (-E).min())
    p /= p.sum()
    exact_neg = np.einsum("s,si,sj->ij", p, S, S)

    vclamp = S[rng.integers(len(S))][:nvis]
    mask = (S[:, :nvis] == vclamp).all(1)
    pc = p[mask] / p[mask].sum()
    exact_pos = np.einsum("s,si,sj->ij", pc, S[mask], S[mask])
    g = exact_pos - exact_neg
    iu = np.triu_indices(n, 1)

    def gibbs(nsamp, burn, rr, clamp=None):
        s = rr.choice([-1.0, 1.0], n)
        acc = np.zeros((n, n))
        for t in range(burn + nsamp):
            for i in range(n):
                if clamp is not None and i < nvis:
                    s[i] = clamp[i]
                    continue
                s[i] = 1.0 if rr.random() < 1 / (1 + np.exp(-2 * (W[i] @ s))) \
                    else -1.0
            if t >= burn:
                acc += np.outer(s, s)
        return acc / nsamp

    counts = (50, 200, 1000, 5000)
    err, sign = [], []
    for ns in counts:
        es, ss = [], []
        for r in range(8):
            rr = np.random.default_rng(100 + r)
            gh = gibbs(ns, 50, rr, clamp=vclamp) - gibbs(ns, 50, rr)
            es.append(np.linalg.norm(gh[iu] - g[iu]) / np.linalg.norm(g[iu]))
            ss.append((np.sign(gh[iu]) == np.sign(g[iu])).mean())
        err.append(float(np.mean(es)))
        sign.append(float(np.mean(ss)))

    assert err[0] > err[-1], "표본을 늘리면 오차가 줄어야 한다"
    assert sign[0] < 0.9, f"짧은 사슬에서는 부호부터 틀려야 한다: {sign[0]:.3f}"
    assert sign[-1] > 0.97, "길게 돌리면 부호는 맞아야 한다"

    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    ax2 = ax.twinx()

    l1 = ax.semilogx(counts, err, "o-", color=ACCENT, ms=6, linewidth=2.0,
                     label="relative error of the update")
    l2 = ax2.semilogx(counts, [s * 100 for s in sign], "s--", color=MUTED,
                      ms=6, linewidth=2.0,
                      label="weights pushed the right way")

    ax.set_xlabel("Gibbs samples per phase, per single weight update",
                  fontsize=9)
    ax.set_ylabel("relative error", fontsize=9, color=ACCENT)
    ax2.set_ylabel("correct sign  (%)", fontsize=9, color=MUTED)
    ax.set_ylim(0, max(err) * 1.25)
    ax2.set_ylim(80, 101)
    ax.tick_params(labelsize=8, colors=INK)
    ax2.tick_params(labelsize=8, colors=INK)
    ax.grid(alpha=0.13, which="both")

    ax2.annotate(f"{(1-sign[0])*100:.0f}% of weights move\nthe wrong way",
                 xy=(counts[0], sign[0] * 100),
                 xytext=(170, 83.5), fontsize=8.4, color=MUTED,
                 arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.9))

    ax.set_title("the rule is two correlations. the price is equilibrium.",
                 fontsize=10, color=INK, weight="bold", pad=9)
    ax.legend(handles=l1 + l2, fontsize=8.2, loc="center right",
              framealpha=0.92)
    for s_ in list(ax.spines.values()) + list(ax2.spines.values()):
        s_.set_color(INK)
        s_.set_linewidth(0.7)

    fig.text(0.5, -0.04,
             f"{n} units ({nvis} visible), exact values enumerated over all "
             f"{2**n:,} states.",
             ha="center", fontsize=8.6, style="italic", color=INK)
    fig.tight_layout()
    fig.savefig(HERE / "bm_price.svg", transparent=True, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote bm_price.svg   err {err[0]:.3f}->{err[-1]:.3f}   "
          f"sign {sign[0]:.3f}->{sign[-1]:.3f}")

if __name__ == "__main__":
    capacity_two_questions()
    restriction_buys()
    bm_price()
