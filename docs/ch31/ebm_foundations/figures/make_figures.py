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

if __name__ == "__main__":
    cd_bias()
    ising_dimensions()
    score_matching_boundary()
