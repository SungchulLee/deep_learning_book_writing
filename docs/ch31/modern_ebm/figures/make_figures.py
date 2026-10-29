"""32.4 에너지 기반 생성 쪽들의 그림.

규칙은 CLAUDE.md 의 Figures 항에 적어 둔 대로다.
- SVG, fonttype='path', transparent
- 글자는 모두 아스키 (matplotlib 기본 폰트에 한글이 없다)

자료를 읽지 않으므로 어디서 돌려도 된다. 수는 전부 여기서 실제로 돌려 얻는다.

실행: python docs/ch31/modern_ebm/figures/make_figures.py
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

A = 16.0                       # 장벽 높이. 이 값에서 HMC 는 단독으로 못 넘는다
BETAS = (1.0, 0.5, 0.25, 0.12, 0.06)
N = 20_000


def _E(x):
    return A * (x ** 2 - 1.0) ** 2


def _dE(x):
    return 4.0 * A * x * (x ** 2 - 1.0)


def _hmc_step(x, beta, rng, h=0.08, L=25):
    r = rng.normal()
    xo = x
    H0 = beta * _E(x) + 0.5 * r * r
    r = r - 0.5 * h * beta * _dE(x)
    for l in range(L):
        x = x + h * r
        if l < L - 1:
            r = r - h * beta * _dE(x)
    r = r - 0.5 * h * beta * _dE(x)
    H1 = beta * _E(x) + 0.5 * r * r
    return x if rng.random() < np.exp(np.clip(H0 - H1, -50, 50)) else xo


# === 그림으로 보는 AI: 장벽 앞에서 갈리는 두 길 ===============================
#
# 그 쪽 3 절이 "HMC 가 먼 모드를 가르는 큰 에너지 장벽을 항상 잘 넘는 것은
# 아니다" 라고 적어 두었다. 장벽을 충분히 높이면 그 말이 수로 드러난다 —
# HMC 단독은 20,000 걸음 동안 한 번도 못 넘고, 4 절의 병렬 템퍼링은 넘는다.

def barrier():
    rng = np.random.default_rng(0)
    x = 1.0
    plain = np.empty(N)
    for t in range(N):
        x = _hmc_step(x, 1.0, rng)
        plain[t] = x

    rng = np.random.default_rng(0)
    xs = [1.0] * len(BETAS)
    tempered = np.empty(N)
    swaps = 0
    for t in range(N):
        for i, b in enumerate(BETAS):
            xs[i] = _hmc_step(xs[i], b, rng)
        i = int(rng.integers(len(BETAS) - 1))
        d = (BETAS[i] - BETAS[i + 1]) * (_E(xs[i]) - _E(xs[i + 1]))
        if rng.random() < np.exp(min(d, 50)):
            xs[i], xs[i + 1] = xs[i + 1], xs[i]
            swaps += 1
        tempered[t] = xs[0]

    cr_p = int(np.sum(np.diff(np.sign(plain)) != 0))
    cr_t = int(np.sum(np.diff(np.sign(tempered)) != 0))
    frac_p = float(np.mean(plain > 0))
    frac_t = float(np.mean(tempered > 0))

    # --- 그림이 주장하는 바를 그림이 스스로 확인한다 ----------------------
    assert cr_p == 0, f"HMC 단독은 한 번도 못 넘어야 한다: {cr_p}"
    assert cr_t > 500, f"병렬 템퍼링은 자주 넘어야 한다: {cr_t}"
    assert abs(frac_t - 0.5) < 0.08, f"대칭이 회복돼야 한다: {frac_t:.3f}"
    assert frac_p > 0.99, f"단독은 한쪽에만 있어야 한다: {frac_p:.3f}"

    grid = np.linspace(-1.9, 1.9, 600)
    dens = np.exp(-_E(grid))
    dens /= np.trapz(dens, grid)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.8, 4.2),
                                   gridspec_kw={"width_ratios": [1, 1.25]})

    # --- 왼쪽: 어디에 머물렀나 --------------------------------------------
    ax1.plot(grid, dens, color=INK, linewidth=1.6, linestyle="--",
             label="target density")
    ax1.hist(plain, bins=70, range=(-1.9, 1.9), density=True, color=MUTED,
             alpha=0.55, label=f"HMC alone  ({frac_p*100:.0f}% on one side)")
    ax1.hist(tempered, bins=70, range=(-1.9, 1.9), density=True,
             color=ACCENT, alpha=0.55,
             label=f"tempered  ({frac_t*100:.0f}% / {100-frac_t*100:.0f}%)")
    ax1.set_xlabel("x", fontsize=9)
    ax1.set_ylabel("density", fontsize=9)
    ax1.set_title("one sampler never saw the other well", fontsize=10,
                  color=INK, weight="bold")
    ax1.legend(fontsize=7.8, loc="upper center", framealpha=0.93)

    # --- 오른쪽: 궤적 ------------------------------------------------------
    show = 4000
    ax2.plot(np.arange(show), plain[:show], color=MUTED, linewidth=0.7,
             alpha=0.9, rasterized=True,
             label=f"HMC alone - {cr_p} crossings in {N:,}")
    ax2.plot(np.arange(show), tempered[:show], color=ACCENT, linewidth=0.7,
             alpha=0.75, rasterized=True,
             label=f"tempered - {cr_t:,} crossings")
    ax2.axhline(0, color=INK, linewidth=0.9, alpha=0.5)
    ax2.set_xlabel(f"step  (first {show:,} shown)", fontsize=9)
    ax2.set_ylabel("x", fontsize=9)
    ax2.set_title("the barrier is the whole difference", fontsize=10,
                  color=INK, weight="bold")
    ax2.legend(fontsize=7.8, loc="lower right", framealpha=0.93)

    for ax in (ax1, ax2):
        ax.grid(alpha=0.13)
        ax.tick_params(labelsize=8, colors=INK)
        for s in ax.spines.values():
            s.set_color(INK)
            s.set_linewidth(0.7)

    fig.text(0.5, -0.04,
             f"E(x) = {A:.0f}(x^2 - 1)^2, same leapfrog settings for both. "
             f"the ladder adds {len(BETAS)-1} hotter chains and "
             f"{swaps:,} accepted swaps.",
             ha="center", fontsize=8.5, style="italic", color=INK)
    fig.tight_layout()
    fig.savefig(HERE / "barrier.svg", transparent=True,
                bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"wrote barrier.svg   plain {cr_p} crossings ({frac_p:.3f}), "
          f"tempered {cr_t} ({frac_t:.3f}), swaps {swaps}")


if __name__ == "__main__":
    barrier()
