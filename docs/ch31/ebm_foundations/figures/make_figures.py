"""32.1/32.3 에너지 기반 모델 기초 쪽들의 그림.

규칙은 CLAUDE.md 의 Figures 항에 적어 둔 대로다.
- SVG, fonttype='path', transparent
- 글자는 모두 아스키 (matplotlib 기본 폰트에 한글이 없다)

자료를 읽지 않으므로 어디서 돌려도 된다. 수는 전부 여기서 실제로 돌려 얻는다.

실행: python docs/ch31/ebm_foundations/figures/make_figures.py
"""
import itertools
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

NV, NH = 8, 4                      # 2^8 = 256 가시 상태를 전부 열거한다


def _sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def _all_visible():
    return np.array(list(itertools.product([0, 1], repeat=NV)), dtype=float)


def _exact_negative(V, W, b, c):
    """E_model[v h^T] 를 2^NV 를 전부 더해 **정확히** 셈한다."""
    free = -(V @ b) - np.logaddexp(0.0, V @ W + c).sum(1)
    p = np.exp(-free - (-free).max())
    p /= p.sum()
    return (V * p[:, None]).T @ _sig(V @ W + c)


def _cd_negative(D, W, b, c, k, rng):
    v = D.copy()
    for _ in range(k):
        h = (rng.random((len(v), NH)) < _sig(v @ W + c)).astype(float)
        v = (rng.random((len(v), NV)) < _sig(h @ W.T + b)).astype(float)
    return v.T @ _sig(v @ W + c) / len(v)


def _setup():
    rng = np.random.default_rng(0)
    base = rng.integers(0, 2, size=(4, NV)).astype(float)
    D = np.repeat(base, 64, axis=0)
    D = (D + (rng.random(D.shape) < 0.08)) % 2
    return _all_visible(), D


# === 그림으로 보는 AI: 힌턴이 스스로 물은 "얼마나 좋은 근사인가" ===============
#
# CD 는 우도 그래디언트의 한 항을 **버린** 것이고, 힌턴 2002 의 10 절 제목이
# 그대로 "How Good Is the Approximation?" 이다. 그 답은 모의실험이었다.
# 같은 실험을 다시 돌려 본다 — 작은 RBM 에서 정확한 그래디언트를 전부 열거해
# 셈하고 CD-k 와 견준다. 버린 항이 작은 것은 **어디에 있느냐에 달렸다.**

def cd_bias():
    V, D = _setup()
    scales = (0.5, 1.0, 2.0, 3.0, 4.0, 6.0)
    cos, ratio = [], []
    for sc in scales:
        W = np.random.default_rng(7).normal(0, sc, (NV, NH))
        b = np.random.default_rng(8).normal(0, 0.3, NV)
        c = np.random.default_rng(9).normal(0, 0.3, NH)
        pos = D.T @ _sig(D @ W + c) / len(D)
        g_exact = pos - _exact_negative(V, W, b, c)
        g = np.mean([pos - _cd_negative(D, W, b, c, 1,
                                        np.random.default_rng(1000 + r))
                     for r in range(60)], axis=0)
        cos.append(float((g * g_exact).sum()
                         / np.linalg.norm(g) / np.linalg.norm(g_exact)))
        ratio.append(float(np.linalg.norm(g) / np.linalg.norm(g_exact)))

    # 가장 나쁜 자리에서 k 를 늘리면 돌아오는가
    sc = scales[-1]
    W = np.random.default_rng(7).normal(0, sc, (NV, NH))
    b = np.random.default_rng(8).normal(0, 0.3, NV)
    c = np.random.default_rng(9).normal(0, 0.3, NH)
    pos = D.T @ _sig(D @ W + c) / len(D)
    g_exact = pos - _exact_negative(V, W, b, c)
    ks = (1, 2, 5, 10, 25, 50)
    cos_k = []
    for k in ks:
        g = np.mean([pos - _cd_negative(D, W, b, c, k,
                                        np.random.default_rng(2000 + r))
                     for r in range(40)], axis=0)
        cos_k.append(float((g * g_exact).sum()
                           / np.linalg.norm(g) / np.linalg.norm(g_exact)))

    # --- 그림이 주장하는 바를 그림이 스스로 확인한다 ----------------------
    assert cos[0] > 0.99, f"작은 웨이트에서는 CD-1 이 거의 정확해야 한다: {cos[0]:.3f}"
    assert cos[-1] < 0.9, f"큰 웨이트에서는 무너져야 한다: {cos[-1]:.3f}"
    assert all(r < 1.0 for r in ratio), f"CD-1 은 늘 모자라야 한다: {ratio}"
    assert cos_k[-1] > cos_k[0], "k 를 늘리면 나아져야 한다"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.6, 4.1),
                                   gridspec_kw={"width_ratios": [1.15, 1]})

    # --- 왼쪽: 어디에 있느냐에 달렸다 -------------------------------------
    ax1.plot(scales, cos, "o-", color=ACCENT, ms=5.5, linewidth=2.0,
             label="direction  (cosine)")
    ax1.plot(scales, ratio, "s--", color=MUTED, ms=5.5, linewidth=2.0,
             label="size  (|CD-1| / |exact|)")
    ax1.axhline(1.0, color=INK, linestyle=":", linewidth=1.0, alpha=0.55)
    ax1.text(scales[0], 1.012, "exact", fontsize=7.8, color=INK, alpha=0.7)
    ax1.set_xlabel("weight scale  (training moves rightward)", fontsize=9)
    ax1.set_ylabel("CD-1 against the exact gradient", fontsize=9)
    ax1.set_title("the dropped term is small - where it is small",
                  fontsize=10, color=INK, weight="bold")
    ax1.set_ylim(0.5, 1.08)
    ax1.legend(fontsize=8.5, loc="lower left", framealpha=0.9)
    ax1.grid(alpha=0.13)

    # --- 오른쪽: 걸음을 늘리면 돌아온다 -----------------------------------
    ax2.semilogx(ks, cos_k, "o-", color=ACCENT, ms=5.5, linewidth=2.0)
    ax2.axhline(1.0, color=INK, linestyle=":", linewidth=1.0, alpha=0.55)
    ax2.set_xlabel("k  in CD-k", fontsize=9)
    ax2.set_ylabel("cosine with the exact gradient", fontsize=9)
    ax2.set_title(f"at the hardest point (scale {scales[-1]:.0f})",
                  fontsize=10, color=INK, weight="bold")
    ax2.set_ylim(0.78, 1.02)
    ax2.grid(alpha=0.13, which="both")
    ax2.annotate(f"k=1: {cos_k[0]:.3f}", xy=(ks[0], cos_k[0]),
                 xytext=(2.6, cos_k[0] + 0.012), fontsize=8.4, color=ACCENT,
                 arrowprops=dict(arrowstyle="->", color=ACCENT, lw=0.9))

    for ax in (ax1, ax2):
        ax.tick_params(labelsize=8, colors=INK)
        for s in ax.spines.values():
            s.set_color(INK)
            s.set_linewidth(0.7)

    fig.text(0.5, -0.03,
             f"RBM with {NV} visible and {NH} hidden units, so the exact "
             f"gradient is enumerable over all {2**NV} states.",
             ha="center", fontsize=8.6, style="italic", color=INK)
    fig.tight_layout()
    fig.savefig(HERE / "cd_bias.svg", transparent=True, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote cd_bias.svg   cos {cos[0]:.4f} -> {cos[-1]:.4f}   "
          f"ratio {ratio[0]:.3f} -> {ratio[-1]:.3f}   "
          f"k-recovery {cos_k[0]:.3f} -> {cos_k[-1]:.3f}")




# === 그림으로 보는 AI: 한 차원에서 본 것을 세 차원으로 옮겨 적은 일 ============
#
# 1925 이징: 1 차원 사슬에는 유한 온도에서 자발 자화가 없다 — 옳다.
# 그리고 논문 끝에 3 차원도 그러리라 **추측**했다 — 틀렸다.
# 1944 온사거가 2 차원을 정확히 풀어 전이가 있음을 보였다.
# 두 곡선 모두 닫힌 식이 있으므로 지어낼 것 없이 그대로 그린다.

TC_2D = 2.0 / np.log(1.0 + np.sqrt(2.0))      # 온사거의 임계 온도


def _onsager_m(T):
    """2 차원 정사각 격자의 자발 자화 (온사거). T >= Tc 에서는 0."""
    T = np.asarray(T, dtype=float)
    m = np.zeros_like(T)
    low = T < TC_2D
    m[low] = (1.0 - np.sinh(2.0 / T[low]) ** -4) ** 0.125
    return m


def ising_dimensions():
    T = np.linspace(0.4, 4.0, 700)
    m2 = _onsager_m(T)
    m1 = np.zeros_like(T)                      # 1 차원: 모든 T>0 에서 0

    assert m2[0] > 0.999, "T->0 에서 2 차원은 포화해야 한다"
    assert m2[T >= TC_2D].max() == 0.0, "Tc 위에서는 0 이어야 한다"
    assert m2[(T > 2.0) & (T < TC_2D)].min() > 0.0, "Tc 아래에서는 양수여야 한다"

    fig, ax = plt.subplots(figsize=(7.2, 4.3))

    ax.plot(T, m2, color=ACCENT, linewidth=2.4,
            label="two dimensions  (Onsager, 1944)")
    ax.plot(T, m1, color=MUTED, linewidth=2.4, linestyle="--",
            label="one dimension  (Ising, 1925)")
    ax.axvline(TC_2D, color=INK, linestyle=":", linewidth=1.2, alpha=0.7)
    ax.annotate(f"Tc = {TC_2D:.3f}", xy=(TC_2D, 0.52),
                xytext=(TC_2D + 0.34, 0.60), fontsize=8.6, color=INK,
                arrowprops=dict(arrowstyle="->", color=INK, lw=0.9))

    ax.text(0.62, 0.17, "flat at zero, at every\ntemperature above zero",
            fontsize=8.4, color=MUTED)
    ax.text(2.75, 0.30,
            "Ising solved the line, found\nno transition, and supposed\n"
            "three dimensions behaved\nthe same way.",
            fontsize=8.4, color=INK, alpha=0.9)

    ax.set_xlabel("temperature", fontsize=9)
    ax.set_ylabel("spontaneous magnetisation", fontsize=9)
    ax.set_title("the same model, one dimension apart", fontsize=10.5,
                 color=INK, weight="bold", pad=9)
    ax.set_xlim(0.4, 4.0)
    ax.set_ylim(-0.03, 1.08)
    ax.legend(fontsize=8.6, loc="upper right", framealpha=0.92)
    ax.grid(alpha=0.13)
    ax.tick_params(labelsize=8, colors=INK)
    for s in ax.spines.values():
        s.set_color(INK)
        s.set_linewidth(0.7)

    fig.text(0.5, -0.04,
             "both curves are closed-form, not simulated.",
             ha="center", fontsize=8.6, style="italic", color=INK)
    fig.tight_layout()
    fig.savefig(HERE / "ising_dimensions.svg", transparent=True,
                bbox_inches="tight")
    plt.close(fig)
    print(f"wrote ising_dimensions.svg   Tc = {TC_2D:.4f}")



# === 그림으로 보는 AI: 사라지는 것이 둘인데, 사라지려면 조건이 있다 ============
#
# 스코어 매칭에서 Z 가 사라지는 것은 로그를 x 로 미분하기 때문이고,
# **모르는 자료 스코어**까지 사라지는 것은 부분적분 때문이다 (Hyvarinen 2005,
# 정리 1). 부분적분이 성립하려면 경계항이 0 이어야 한다 — 그 논문 각주 1 의
# 조건이다. 조건이 맞는 경우와 깨지는 경우를 나란히 재어 보인다.

def score_matching_boundary():
    rng = np.random.default_rng(0)

    # --- 조건이 맞는 경우: 실수 전체의 가우시안 --------------------------
    mu_t, s_t = 1.3, 0.8
    x = rng.normal(mu_t, s_t, 200_000)
    mus = np.linspace(0.3, 2.3, 400)
    s2 = s_t ** 2
    J_mu = np.array([np.mean(-1 / s2 + 0.5 * ((x - m) / s2) ** 2) for m in mus])
    mu_hat = mus[np.argmin(J_mu)]

    # --- 깨지는 경우: 반직선의 지수분포 ----------------------------------
    lam_t = 2.0
    lams = np.linspace(0.0, 4.0, 400)
    J_lam = 0.5 * lams ** 2            # psi = -lam 상수, dpsi/dy = 0
    lam_hat = lams[np.argmin(J_lam)]

    assert abs(mu_hat - mu_t) < 0.02, f"가우시안은 맞아야 한다: {mu_hat:.3f}"
    assert lam_hat < 1e-9, f"지수분포는 0 으로 무너져야 한다: {lam_hat}"
    assert np.all(np.diff(J_lam) > 0), "J 가 단조증가여야 한다"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.6, 4.1))

    ax1.plot(mus, J_mu, color=ACCENT, linewidth=2.2)
    ax1.axvline(mu_t, color=INK, linestyle="--", linewidth=1.2, alpha=0.75)
    ax1.plot([mu_hat], [J_mu.min()], "o", color=ACCENT, ms=9,
             markeredgecolor="white", markeredgewidth=1.3, zorder=4)
    ax1.annotate(f"minimum at {mu_hat:.3f}\ntrue value {mu_t}",
                 xy=(mu_hat, J_mu.min()), xytext=(mu_hat + 0.26,
                                                  J_mu.min() + 0.33 * J_mu.ptp()),
                 fontsize=8.4, color=INK,
                 arrowprops=dict(arrowstyle="->", color=INK, lw=0.9))
    ax1.set_xlabel("mu", fontsize=9)
    ax1.set_ylabel("score matching objective", fontsize=9)
    ax1.set_title("Gaussian on the whole line - it works", fontsize=10,
                  color=INK, weight="bold")

    ax2.plot(lams, J_lam, color=ACCENT, linewidth=2.2)
    ax2.axvline(lam_t, color=INK, linestyle="--", linewidth=1.2, alpha=0.75)
    ax2.plot([lam_hat], [J_lam.min()], "o", color=ACCENT, ms=9,
             markeredgecolor="white", markeredgewidth=1.3, zorder=4)
    ax2.annotate("minimum at 0", xy=(lam_hat, J_lam.min()),
                 xytext=(0.45, 2.1), fontsize=8.6, color=ACCENT,
                 arrowprops=dict(arrowstyle="->", color=ACCENT, lw=0.9))
    ax2.annotate(f"true lambda = {lam_t:.0f}", xy=(lam_t, 1.2),
                 xytext=(lam_t + 0.25, 1.0), fontsize=8.6, color=INK,
                 arrowprops=dict(arrowstyle="->", color=INK, lw=0.9))
    ax2.set_xlabel("lambda", fontsize=9)
    ax2.set_ylabel("score matching objective", fontsize=9)
    ax2.set_title("exponential on a half line - it does not", fontsize=10,
                  color=INK, weight="bold")

    for ax in (ax1, ax2):
        ax.grid(alpha=0.13)
        ax.tick_params(labelsize=8, colors=INK)
        for s in ax.spines.values():
            s.set_color(INK)
            s.set_linewidth(0.7)

    fig.text(0.5, -0.04,
             "the trick needs p(x) psi(x) to vanish at the boundary. "
             "on the half line it does not, and the estimate collapses.",
             ha="center", fontsize=8.6, style="italic", color=INK)
    fig.tight_layout()
    fig.savefig(HERE / "score_matching_boundary.svg", transparent=True,
                bbox_inches="tight")
    plt.close(fig)
    print(f"wrote score_matching_boundary.svg   mu_hat={mu_hat:.4f}  "
          f"lam_hat={lam_hat:.4f}")



# === 그림으로 보는 AI: Z 는 "상태를 모두 더한 것"이라는 뜻이다 ================
#
# 그 쪽 3 절이 "겹침이 나쁘면 소수 웨이트가 지배해 분산이 커진다" 고 적어 둔
# 것을 수로 보인다. 목표는 정규화 안 한 exp(-|x|^2/2) 라 참 Z=(2pi)^{d/2} 를
# 정확히 아는 경우이고, 제안은 폭만 1.5 배로 어긋난 가우시안이다.
# 추정값보다 **유효표본수가 먼저** 무너진다는 것이 요점이다.

def partition_overlap():
    s, N = 1.5, 200_000
    dims = (1, 2, 5, 10, 20, 50, 80, 120, 200)
    ratio, ess, top = [], [], []
    for d in dims:
        rr, ee, tt = [], [], []
        for r in range(5):
            rng = np.random.default_rng(r)
            x = rng.normal(0, s, (N, d))
            lw = (-0.5 * (x ** 2).sum(1)) - (
                -0.5 * (x ** 2).sum(1) / s ** 2
                - d * np.log(s * np.sqrt(2 * np.pi)))
            m = lw.max()
            w = np.exp(lw - m)
            rr.append(np.exp(m) * w.mean() / (2 * np.pi) ** (d / 2))
            ee.append(w.sum() ** 2 / (w ** 2).sum())
            tt.append(w.max() / w.sum())
        ratio.append(np.array(rr))
        ess.append(float(np.mean(ee)))
        top.append(float(np.mean(tt)))

    med = np.array([r.mean() for r in ratio])
    lo = np.array([r.min() for r in ratio])
    hi = np.array([r.max() for r in ratio])

    assert abs(med[0] - 1.0) < 0.02, "낮은 차원에서는 맞아야 한다"
    assert med[-1] < 0.05, f"높은 차원에서는 무너져야 한다: {med[-1]}"
    assert ess[-1] < 10, f"유효표본수가 무너져야 한다: {ess[-1]}"
    assert ess[0] > 1e5, "낮은 차원에서는 표본이 거의 다 살아야 한다"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.6, 4.1), sharex=True)

    ax1.fill_between(dims, lo, hi, color=ACCENT, alpha=0.18)
    ax1.semilogy(dims, med, "o-", color=ACCENT, ms=5.5, linewidth=2.0)
    ax1.axhline(1.0, color=INK, linestyle="--", linewidth=1.2, alpha=0.75)
    ax1.text(1.5, 1.25, "truth", fontsize=8.2, color=INK, alpha=0.8)
    ax1.set_xlabel("dimension", fontsize=9)
    ax1.set_ylabel("estimated Z  /  true Z", fontsize=9)
    ax1.set_title("the estimate looks fine, then it is not", fontsize=10,
                  color=INK, weight="bold")
    ax1.annotate(f"off by {1/med[-1]:.0f}x", xy=(dims[-1], med[-1]),
                 xytext=(96, med[-1] * 9), fontsize=8.5, color=ACCENT,
                 arrowprops=dict(arrowstyle="->", color=ACCENT, lw=0.9))

    ax2.semilogy(dims, ess, "s-", color=MUTED, ms=5.5, linewidth=2.0)
    ax2.axhline(N, color=INK, linestyle="--", linewidth=1.2, alpha=0.75)
    ax2.text(1.5, N * 1.25, f"all {N:,} draws", fontsize=8.2, color=INK,
             alpha=0.8)
    ax2.set_xlabel("dimension", fontsize=9)
    ax2.set_ylabel("effective sample size", fontsize=9)
    ax2.set_title("but this collapsed first", fontsize=10, color=INK,
                  weight="bold")
    ax2.annotate(f"{ess[-1]:.1f} of {N:,}\nlargest weight {top[-1]*100:.0f}%",
                 xy=(dims[-1], ess[-1]), xytext=(84, ess[-1] * 14),
                 fontsize=8.5, color=MUTED,
                 arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.9))

    for ax in (ax1, ax2):
        ax.grid(alpha=0.13, which="both")
        ax.tick_params(labelsize=8, colors=INK)
        for s_ in ax.spines.values():
            s_.set_color(INK)
            s_.set_linewidth(0.7)

    fig.text(0.5, -0.03,
             "target with a known Z, proposal only 1.5x too wide. "
             "band is the spread over 5 runs.",
             ha="center", fontsize=8.6, style="italic", color=INK)
    fig.tight_layout()
    fig.savefig(HERE / "partition_overlap.svg", transparent=True,
                bbox_inches="tight")
    plt.close(fig)
    print(f"wrote partition_overlap.svg   ratio {med[0]:.3f}->{med[-1]:.4f}   "
          f"ESS {ess[0]:.0f}->{ess[-1]:.1f}")



# === 그림으로 보는 AI: 재구성 오차는 매끄럽고, 우도는 그렇지 않다 ==============
#
# 힌턴 2010 의 실천 지침이 재구성 오차를 "a very poor measure of the progress
# of learning" 이라 적고 "Use it but don't trust it" 로 닫는다. 가시 12 개면
# 4,096 상태를 전부 더해 **정확한** 로그 우도를 셈할 수 있으므로, 둘을 나란히
# 기록해 정말 어긋나는지 본다.

def recon_vs_likelihood():
    import itertools

    rng = np.random.default_rng(0)
    nv, nh = 12, 8
    V = np.array(list(itertools.product([0, 1], repeat=nv)), dtype=float)

    def sig(x):
        return 1.0 / (1.0 + np.exp(-x))

    def free(X, W, b, c):
        return -(X @ b) - np.logaddexp(0.0, X @ W + c).sum(1)

    def exact_ll(W, b, c, D):
        logZ = np.logaddexp.reduce(-free(V, W, b, c))
        return float(np.mean(-free(D, W, b, c) - logZ))

    base = rng.integers(0, 2, size=(4, nv)).astype(float)
    D = np.repeat(base, 80, axis=0)
    D = (D + (rng.random(D.shape) < 0.05)) % 2

    W = rng.normal(0, 0.01, (nv, nh))
    b = np.zeros(nv)
    c = np.zeros(nh)
    lr = 0.5

    eps, rec, ll, wn = [], [], [], []
    for ep in range(1, 1201):
        idx = rng.permutation(len(D))
        for s in range(0, len(D), 20):
            v0 = D[idx[s:s + 20]]
            ph0 = sig(v0 @ W + c)
            h0 = (rng.random(ph0.shape) < ph0).astype(float)
            v1 = (rng.random((len(v0), nv)) < sig(h0 @ W.T + b)).astype(float)
            ph1 = sig(v1 @ W + c)
            W += lr * (v0.T @ ph0 - v1.T @ ph1) / len(v0)
            b += lr * (v0 - v1).mean(0)
            c += lr * (ph0 - ph1).mean(0)
        if ep % 20 == 0:
            vr = sig(sig(D @ W + c) @ W.T + b)
            eps.append(ep)
            rec.append(float(np.mean((D - vr) ** 2)))
            ll.append(exact_ll(W, b, c, D))
            wn.append(float(np.abs(W).mean()))

    rec_a, ll_a = np.array(rec), np.array(ll)
    worst = int(np.argmin(np.diff(ll_a))) + 1        # 우도가 가장 크게 꺾인 자리

    rec_gain = rec_a[0] - rec_a[-1]
    rec_worst_up = float(np.diff(rec_a).max())
    ll_gain = ll_a[-1] - ll_a[0]
    ll_worst_down = float(np.diff(ll_a).min())
    both = int(np.sum((np.diff(rec_a) < 0) & (np.diff(ll_a) < 0)))

    # --- 그림이 주장하는 바를 그림이 스스로 확인한다 ----------------------
    # 재구성 오차는 **단조가 아니다** — 성기게 찍으면 그렇게 보일 뿐이다.
    # 참인 대비는 뒷걸음의 **크기**다.
    assert rec_gain > 0, "재구성 오차는 전체적으로 줄어야 한다"
    assert rec_worst_up / rec_gain < 0.15, (
        f"재구성 오차의 뒷걸음은 작아야 한다: {rec_worst_up/rec_gain:.2f}")
    assert abs(ll_worst_down) > ll_gain, (
        f"우도의 한 번 하락이 전체 상승보다 커야 한다: "
        f"{ll_worst_down:.3f} vs {ll_gain:.3f}")
    assert both > len(rec_a) // 5, (
        f"둘이 어긋나는 구간이 충분해야 한다: {both}")
    assert wn[-1] > 2 * wn[0], "웨이트가 자라야 한다 (혼합이 느려지는 기제)"

    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    ax2 = ax.twinx()

    l1 = ax.plot(eps, rec, color=ACCENT, linewidth=2.2,
                 label="reconstruction error  (what the loop logs)")
    l2 = ax2.plot(eps, ll, color=MUTED, linewidth=1.6, alpha=0.9,
                  label="exact log-likelihood  (all 4,096 states)")

    ax2.plot([eps[worst]], [ll[worst]], "o", color=MUTED, ms=9,
             markeredgecolor="white", markeredgewidth=1.3, zorder=5)
    ax2.annotate(f"one step costs {abs(ll_worst_down):.2f} nats -\n"
                 f"more than the whole run gained",
                 xy=(eps[worst], ll[worst]),
                 xytext=(eps[worst] - 620, ll[worst] + 0.30),
                 fontsize=8.4, color=MUTED,
                 arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.0))

    ax.set_xlabel("epoch", fontsize=9)
    ax.set_ylabel("reconstruction error", fontsize=9, color=ACCENT)
    ax2.set_ylabel("exact log-likelihood", fontsize=9, color=MUTED)
    ax.set_title("one of these curves is reassuring. it is not the one that "
                 "matters.", fontsize=10, color=INK, weight="bold", pad=9)
    ax.grid(alpha=0.13)
    ax.legend(handles=l1 + l2, fontsize=8.2, loc="center right",
              framealpha=0.93)
    ax.tick_params(labelsize=8, colors=INK)
    ax2.tick_params(labelsize=8, colors=INK)
    for s_ in list(ax.spines.values()) + list(ax2.spines.values()):
        s_.set_color(INK)
        s_.set_linewidth(0.7)

    fig.text(0.5, -0.04,
             f"in {both} of {len(rec_a)-1} intervals the logged error improves "
             f"while the likelihood gets worse.  mean |W| grows "
             f"{wn[0]:.2f} -> {wn[-1]:.2f}.",
             ha="center", fontsize=8.5, style="italic", color=INK)
    fig.tight_layout()
    fig.savefig(HERE / "recon_vs_likelihood.svg", transparent=True,
                bbox_inches="tight")
    plt.close(fig)
    print(f"wrote recon_vs_likelihood.svg   recon {rec[0]:.4f}->{rec[-1]:.4f}  "
          f"worst ll step {ll_worst_down:+.3f} vs gain {ll_gain:+.3f}  "
          f"disagree {both}/{len(rec_a)-1}")



# === 그림으로 보는 AI: 이산화가 물려받지 못하는 것 ============================
#
# 그 쪽 3 절이 "유한 이산화의 추가 바이어스도 있다" 고 적어 둔 그 바이어스를,
# 닫힌 식이 있는 한 경우에서 정확히 보인다. 목표가 N(0,1) 이면 쪽의 관례
# x <- x - h*gradE + sqrt(2h)*xi 는 AR(1) 이고 정상분산이 1/(1-h/2) 다.
# h=2 에서 발산하는데, 로버츠-트위디 1996 이 gamma*h>2 에서 transient 임을
# 증명한 자리와 같다.

def ula_bias():
    rng = np.random.default_rng(0)

    hs = np.array([0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 1.5])
    theory = 1.0 / (1.0 - hs / 2.0)
    meas = []
    for h in hs:
        x = np.zeros(60_000)
        for _ in range(4000):
            x = x - h * x + np.sqrt(2 * h) * rng.normal(size=x.shape)
        meas.append(float(x.var()))
    meas = np.array(meas)

    # 발산 경계
    traj = {}
    for h in (1.99, 2.05):
        x = np.zeros(2000)
        path = []
        for _ in range(400):
            x = x - h * x + np.sqrt(2 * h) * rng.normal(size=x.shape)
            with np.errstate(over="ignore", invalid="ignore"):
                path.append(float(np.nanmax(np.abs(x))))
        traj[h] = np.array(path)

    assert np.allclose(meas, theory, rtol=0.03), (
        f"닫힌 식과 실측이 맞아야 한다: {meas} vs {theory}")
    assert traj[1.99][-1] < 1e3, "h<2 는 머물러야 한다"
    assert traj[2.05][-1] > 1e6 or not np.isfinite(traj[2.05][-1]), (
        "h>2 는 발산해야 한다")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.6, 4.1))

    hh = np.linspace(0.005, 1.95, 300)
    ax1.plot(hh, 1.0 / (1.0 - hh / 2.0), color=ACCENT, linewidth=2.2,
             label="exact:  1 / (1 - h/2)")
    ax1.plot(hs, meas, "o", color=INK, ms=6, zorder=4,
             label="measured")
    ax1.axhline(1.0, color=INK, linestyle="--", linewidth=1.2, alpha=0.7)
    ax1.text(2.10, 1.03, "the target", fontsize=8.2, color=INK,
             alpha=0.8, ha="right")
    ax1.axvline(2.0, color=MUTED, linestyle=":", linewidth=1.4)
    ax1.annotate("this page's h = 0.01\n(0.5% too wide)", xy=(0.01, 1.005),
                 xytext=(0.34, 1.95), fontsize=8.3, color=INK,
                 arrowprops=dict(arrowstyle="->", color=INK, lw=0.9))
    ax1.set_xlabel("step size h", fontsize=9)
    ax1.set_ylabel("stationary variance", fontsize=9)
    ax1.set_title("the chain settles, but not on the target", fontsize=10,
                  color=INK, weight="bold")
    ax1.set_xlim(0, 2.15)
    ax1.set_ylim(0.9, 4.4)
    ax1.legend(fontsize=8.4, loc="upper left", framealpha=0.92)
    ax1.grid(alpha=0.13)

    for h, col, lab in ((1.99, ACCENT, "h = 1.99"), (2.05, MUTED, "h = 2.05")):
        y = np.clip(traj[h], 1e-3, 1e300)
        ax2.semilogy(np.arange(len(y)), y, color=col, linewidth=1.8,
                     label=lab)
    ax2.set_xlabel("step", fontsize=9)
    ax2.set_ylabel("largest |x| in the ensemble", fontsize=9)
    ax2.set_title("and above h = 2 it leaves altogether", fontsize=10,
                  color=INK, weight="bold")
    ax2.legend(fontsize=8.6, loc="upper left", framealpha=0.92)
    ax2.grid(alpha=0.13, which="both")

    for ax in (ax1, ax2):
        ax.tick_params(labelsize=8, colors=INK)
        for s_ in ax.spines.values():
            s_.set_color(INK)
            s_.set_linewidth(0.7)

    fig.text(0.5, -0.04,
             "target is N(0,1), so gamma = 1 and Roberts and Tweedie's "
             "transience condition gamma*h > 2 lands exactly on the pole.",
             ha="center", fontsize=8.5, style="italic", color=INK)
    fig.tight_layout()
    fig.savefig(HERE / "ula_bias.svg", transparent=True, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote ula_bias.svg   h=0.01 -> {theory[0]:.4f}   "
          f"h=1.5 -> {theory[-1]:.3f}   1.99 ok, 2.05 diverges")



# === 그림으로 보는 AI: 가장 높은 밀도의 점을 모델은 뽑지 않는다 ===============
#
# 그 쪽 2 절이 "고차원에서는 밀도와 질량이 집중되는 영역이 다를 수 있다" 고
# 적어 두었다. N(0, I_d) 는 그것을 정확히 셈할 수 있는 경우다 — 밀도의 최대는
# 원점인데 질량은 반지름 sqrt(d) 의 껍질에 몰린다.

def density_vs_typicality():
    rng = np.random.default_rng(0)
    dims = (1, 2, 10, 100, 784)
    n = 20_000

    radii, drops = {}, []
    for d in dims:
        x = rng.normal(size=(n, d))
        r = np.linalg.norm(x, axis=1)
        radii[d] = r
        drops.append(float(np.median(-0.5 * r ** 2)))   # 원점 대비 로그밀도

    r784 = radii[784]
    assert abs(np.median(r784) - np.sqrt(784)) < 0.5, "껍질이 sqrt(d) 에 있어야 한다"
    assert r784.min() > 20, f"원점 근처 표본이 없어야 한다: min |x| = {r784.min():.1f}"
    assert drops[-1] < -300, f"로그밀도 차가 커야 한다: {drops[-1]:.1f}"
    assert drops[0] > -1, "낮은 차원에서는 차이가 작아야 한다"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.6, 4.1),
                                   gridspec_kw={"width_ratios": [1.2, 1]})

    cols = plt.cm.copper(np.linspace(0.15, 0.78, len(dims)))
    for d, c in zip(dims, cols):
        ax1.hist(radii[d], bins=90, density=True, histtype="step",
                 linewidth=1.9, color=c, label=f"d = {d}")
    ax1.axvline(0, color=ACCENT, linestyle="--", linewidth=1.6)
    ax1.annotate("the single highest-density\npoint sits here, at the origin",
                 xy=(0, 0.14), xytext=(4.5, 0.30), fontsize=8.3, color=ACCENT,
                 arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.0))
    ax1.set_xlabel("distance from the origin", fontsize=9)
    ax1.set_ylabel("density of samples", fontsize=9)
    ax1.set_title("where the samples actually are", fontsize=10, color=INK,
                  weight="bold")
    ax1.set_xlim(-1.5, 34)
    ax1.legend(fontsize=8.2, loc="upper right", framealpha=0.92)

    ax2.plot(dims, [-d for d in drops], "o-", color=ACCENT, ms=6,
             linewidth=2.0)
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("dimension", fontsize=9)
    ax2.set_ylabel("nats below the maximum", fontsize=9)
    ax2.set_title("how far a typical sample is from the peak",
                  fontsize=10, color=INK, weight="bold")
    ax2.annotate(f"{-drops[-1]:.0f} nats at d = 784", xy=(784, -drops[-1]),
                 xytext=(1.6, -drops[-1] * 0.30), fontsize=8.4, color=INK,
                 arrowprops=dict(arrowstyle="->", color=INK, lw=0.9))
    ax2.grid(alpha=0.13, which="both")

    for ax in (ax1, ax2):
        ax.tick_params(labelsize=8, colors=INK)
        for s in ax.spines.values():
            s.set_color(INK)
            s.set_linewidth(0.7)

    fig.text(0.5, -0.04,
             f"at d = 784 none of {n:,} draws came within "
             f"{r784.min():.0f} of the origin.",
             ha="center", fontsize=8.5, style="italic", color=INK)
    fig.tight_layout()
    fig.savefig(HERE / "density_vs_typicality.svg", transparent=True,
                bbox_inches="tight")
    plt.close(fig)
    print(f"wrote density_vs_typicality.svg   d=784 median |x| "
          f"{np.median(r784):.1f}, min {r784.min():.1f}, drop {drops[-1]:.0f}")

if __name__ == "__main__":
    cd_bias()
    ising_dimensions()
    score_matching_boundary()
    partition_overlap()
    recon_vs_likelihood()
    ula_bias()
    density_vs_typicality()
