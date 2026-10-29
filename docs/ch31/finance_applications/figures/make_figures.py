"""32.5 신용 네트워크 쪽의 그림.

규칙은 CLAUDE.md 의 Figures 항에 적어 둔 대로다.
- SVG, fonttype='path', transparent
- 글자는 모두 아스키 (matplotlib 기본 폰트에 한글이 없다)

자료를 읽지 않으므로 어디서 돌려도 된다. 수는 전부 닫힌 식으로 정확히 셈한다.

실행: python docs/ch31/finance_applications/figures/make_figures.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["svg.fonttype"] = "path"

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import betabinom, binom

HERE = Path(__file__).resolve().parent
INK = "#2f3337"
ACCENT = "#8a4b2a"
MUTED = "#2d6a8f"

N_FIRMS, P_DEF, RHO = 100, 0.02, 0.30


# === 그림으로 보는 AI: 쌍 상관은 꼬리를 정하지 않는다 =========================
#
# 교환가능 이진 모형은 S_i | Q ~ Bern(Q) 로 적을 수 있고, 이때 개별 부도확률과
# 쌍 상관은 E[Q] 와 Var(Q) **만** 정한다. 평균과 분산이 같은 서로 다른 Q 를
# 고르면 주변분포와 쌍 상관이 **똑같으면서** 동시부도 꼬리가 전혀 다른 두
# 모형이 나온다. 두 분포 모두 닫힌 식이라 어림이 섞이지 않는다.

def tail_not_pinned():
    n, p, rho = N_FIRMS, P_DEF, RHO
    var_q = rho * p * (1 - p)

    s = p * (1 - p) / var_q - 1.0          # A: 베타 혼합
    a, b = p * s, (1 - p) * s
    q2 = (var_q + p * p) / p               # B: 두 점 (Q = q2 또는 0)
    w = p / q2

    ks = np.arange(n + 1)
    pA = betabinom.pmf(ks, n, a, b)
    pB = w * binom.pmf(ks, n, q2)
    pB[0] += 1.0 - w

    def pair_corr(pk):
        es = float(np.sum(ks * pk)) / n
        ess = float(np.sum(ks * (ks - 1) * pk)) / (n * (n - 1))
        return (ess - es ** 2) / (es * (1 - es))

    mA, mB = float(np.sum(ks * pA)), float(np.sum(ks * pB))
    cA, cB = pair_corr(pA), pair_corr(pB)

    # --- 그림이 주장하는 바를 그림이 스스로 확인한다 ----------------------
    assert abs(mA - n * p) < 1e-6 and abs(mB - n * p) < 1e-6, (
        f"평균이 같아야 한다: {mA} {mB}")
    assert abs(cA - rho) < 1e-6 and abs(cB - rho) < 1e-6, (
        f"쌍 상관이 같아야 한다: {cA} {cB}")

    tailA = np.array([pA[k:].sum() for k in ks])
    tailB = np.array([pB[k:].sum() for k in ks])
    k50 = 50
    ratio = tailA[k50] / tailB[k50]
    assert ratio > 100, f"꼬리가 크게 갈려야 한다: {ratio:.1f}"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.8, 4.2),
                                   gridspec_kw={"width_ratios": [1, 1.2]})

    # --- 왼쪽: 두 모형이 똑같이 맞추는 것 ---------------------------------
    labs = ["each firm's\ndefault probability", "every pairwise\ncorrelation"]
    for i, (lab, va, vb, tgt) in enumerate(
            zip(labs, (mA / n, cA), (mB / n, cB), (p, rho))):
        y = 0.72 - i * 0.34
        ax1.text(0.5, y + 0.14, lab, ha="center", fontsize=9, color=INK)
        ax1.text(0.28, y, f"{va:.4f}", ha="center", fontsize=12,
                 color=ACCENT, family="monospace")
        ax1.text(0.5, y, "=", ha="center", fontsize=12, color=INK)
        ax1.text(0.72, y, f"{vb:.4f}", ha="center", fontsize=12,
                 color=MUTED, family="monospace")
    ax1.text(0.28, 0.93, "model A", ha="center", fontsize=9.5, color=ACCENT,
             weight="bold")
    ax1.text(0.72, 0.93, "model B", ha="center", fontsize=9.5, color=MUTED,
             weight="bold")
    ax1.text(0.5, 0.06, "identical to every digit shown", ha="center",
             fontsize=8.6, color=INK, style="italic")
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.axis("off")
    ax1.set_title("what the two models agree on", fontsize=10, color=INK,
                  weight="bold")

    # --- 오른쪽: 꼬리는 전혀 다르다 ---------------------------------------
    ax2.semilogy(ks[1:], tailA[1:], color=ACCENT, linewidth=2.3,
                 label="model A  (beta mixture)")
    ax2.semilogy(ks[1:], tailB[1:], color=MUTED, linewidth=2.3,
                 linestyle="--", label="model B  (two states)")
    ax2.axvline(k50, color=INK, linestyle=":", linewidth=1.2, alpha=0.6)
    ax2.annotate(f"at {k50} defaults\nthey differ {ratio:,.0f}-fold",
                 xy=(k50, tailB[k50]), xytext=(21, 3e-6),
                 fontsize=8.5, color=INK,
                 arrowprops=dict(arrowstyle="->", color=INK, lw=0.9))
    ax2.set_xlabel("number of simultaneous defaults, k", fontsize=9)
    ax2.set_ylabel("P(at least k)", fontsize=9)
    ax2.set_title("what they do not", fontsize=10, color=INK, weight="bold")
    ax2.set_ylim(1e-8, 1.4)
    ax2.set_xlim(0, 70)
    ax2.legend(fontsize=8.4, loc="upper right", framealpha=0.93)
    ax2.grid(alpha=0.13, which="both")
    ax2.tick_params(labelsize=8, colors=INK)
    for s_ in ax2.spines.values():
        s_.set_color(INK)
        s_.set_linewidth(0.7)

    fig.text(0.5, -0.04,
             f"{n} firms, each defaulting with probability {p}, every pair "
             f"correlated {rho}. both curves are closed-form.",
             ha="center", fontsize=8.5, style="italic", color=INK)
    fig.tight_layout()
    fig.savefig(HERE / "tail_not_pinned.svg", transparent=True,
                bbox_inches="tight")
    plt.close(fig)
    print(f"wrote tail_not_pinned.svg   mean {mA:.4f}/{mB:.4f}  "
          f"corr {cA:.4f}/{cB:.4f}  P(>={k50}) ratio {ratio:,.0f}")


if __name__ == "__main__":
    tail_not_pinned()
