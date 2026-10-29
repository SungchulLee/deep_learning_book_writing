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


if __name__ == "__main__":
    cd_bias()
