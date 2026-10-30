"""20.3 MCMC 쪽들의 그림.

규칙은 CLAUDE.md 의 Figures 항에 적어 둔 대로다.
- SVG, fonttype='path', transparent
- 글자는 모두 아스키 (matplotlib 기본 폰트에 한글이 없다)

자료를 읽지 않으므로 어디서 돌려도 된다. 씨앗을 박아 두어 다시 돌려도
같은 수가 나온다.

실행: python docs/ch19/mcmc/figures/make_figures.py
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


# === 그림으로 보는 AI: 거절한 자리도 표본이다 =================================
#
# 1953년 논문이 직접 적어 둔 규칙이다 — 움직였든 아니든 새 배치로 쳐서 평균에
# 넣는다. 버리면 어떻게 되는지를 재어 보인다. 평균은 대칭이라 거의 안 움직이고
# **분산이 위로 치우친다** — 거절은 사슬이 오래 머무는 가운데서 더 자주 나므로,
# 버리면 바깥쪽이 과하게 실린다.

def rejected_samples_count():
    rng = np.random.default_rng(0)

    n, step = 200_000, 2.4
    x = 0.0
    keep = np.empty(n)
    acc = np.empty(n, dtype=bool)
    for i in range(n):
        y = x + step * rng.standard_normal()
        if np.log(rng.random()) < -0.5 * (y * y - x * x):
            x = y
            acc[i] = True
        else:
            acc[i] = False
        keep[i] = x

    moved = keep[np.r_[True, np.diff(keep) != 0]]
    v_all, v_drop = keep.var(), moved.var()
    err_all, err_drop = abs(v_all - 1.0), abs(v_drop - 1.0)
    # 이 그림이 주장하는 것 — 버리면 분산이 눈에 띄게 나빠진다
    assert err_drop > err_all * 3, "버린 쪽이 더 나빠야 이 그림의 이야기가 산다"
    assert v_drop > v_all, "버리면 분산이 위로 치우쳐야 한다"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.8, 4.0),
                                   gridspec_kw={"width_ratios": [1.15, 1]})

    # --- 왼쪽: 두 히스토그램 ----------------------------------------------
    grid = np.linspace(-4.2, 4.2, 400)
    truth = np.exp(-grid ** 2 / 2) / np.sqrt(2 * np.pi)
    bins = np.linspace(-4.2, 4.2, 90)
    ax1.hist(keep, bins=bins, density=True, color=INK, alpha=0.30,
             label="every step, as the rule says")
    ax1.hist(moved, bins=bins, density=True, histtype="step", lw=1.9,
             color=ACCENT, label="rejections thrown away")
    ax1.plot(grid, truth, color=INK, lw=1.4, ls=":", label="the target")
    ax1.set_xlabel("x", fontsize=9)
    ax1.set_ylabel("density", fontsize=9)
    ax1.set_title("the same chain, counted two ways", fontsize=10,
                  color=INK, weight="bold")
    ax1.legend(fontsize=8, framealpha=0.9)
    ax1.grid(alpha=0.12)

    # --- 오른쪽: 분산이 어긋난 정도 ----------------------------------------
    ax2.bar([0, 1], [err_all, err_drop], color=[INK, ACCENT], width=0.52,
            alpha=0.85)
    for i, (v, e) in enumerate(((v_all, err_all), (v_drop, err_drop))):
        ax2.text(i, e + err_drop * 0.04, f"variance {v:.4f}", ha="center",
                 fontsize=9, color=INK)
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(["every step", "rejections\nthrown away"],
                        fontsize=9)
    ax2.set_ylabel("error in the variance", fontsize=9)
    ax2.set_ylim(0, err_drop * 1.22)
    ax2.set_title(f"{err_drop / err_all:.0f}x further from 1.0000",
                  fontsize=10, color=INK, weight="bold")
    ax2.grid(alpha=0.13, axis="y")

    for ax in (ax1, ax2):
        ax.tick_params(labelsize=8, colors=INK)
        for s in ax.spines.values():
            s.set_color(INK)
            s.set_linewidth(0.7)

    fig.text(0.5, -0.03,
             f"acceptance {acc.mean():.3f}. the mean barely moves - the target "
             f"is symmetric. the spread is what goes wrong.",
             ha="center", fontsize=8.8, style="italic", color=INK)
    fig.tight_layout()
    fig.savefig(HERE / "rejected_samples.svg", transparent=True,
                bbox_inches="tight")
    plt.close(fig)
    print(f"wrote rejected_samples.svg   var all {v_all:.4f} drop {v_drop:.4f}"
          f"   {err_drop / err_all:.1f}x")


# === 그림으로 보는 AI: 수락률 1 이 말해 주지 않는 것 ==========================
#
# 깁스는 거절이 없다. 수락률은 언제나 1.000 이고, 그래서 그 수로는 사슬이 잘
# 도는지 알 수 없다. 상관 rho 를 올려 가며 유효표본수를 재면 1% 까지 주저앉는다.
# 쪽의 5절이 적어 둔 rho^2 를 수로 보이는 셈이다.

def _ess(v):
    """자기상관을 더해 유효표본수를 센다 (초기 양수 구간까지)."""
    v = v - v.mean()
    n = len(v)
    f = np.fft.rfft(v, 2 * n)
    ac = np.fft.irfft(f * np.conj(f))[:n].real
    ac /= ac[0]
    s = 0.0
    for k in range(1, n):
        if ac[k] <= 0:
            break
        s += ac[k]
    return n / (1 + 2 * s)


def gibbs_acceptance_tells_nothing():
    rng = np.random.default_rng(0)

    def gibbs(rho, n=100_000):
        x = y = 0.0
        sd = np.sqrt(1 - rho * rho)
        out = np.empty(n)
        for i in range(n):
            x = rho * y + sd * rng.standard_normal()
            y = rho * x + sd * rng.standard_normal()
            out[i] = y
        return out

    rhos = [0.0, 0.5, 0.9, 0.99]
    n = 100_000
    frac = [_ess(gibbs(r, n)) / n for r in rhos]
    # 이 그림이 주장하는 것 — 수락률은 그대로인데 쓸모는 무너진다
    assert frac[0] > 0.9 and frac[-1] < 0.05, \
        "상관이 커질 때 유효표본 비율이 무너져야 이 그림의 이야기가 산다"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.8, 4.0))

    # --- 왼쪽: 두 수를 나란히 ----------------------------------------------
    xs = np.arange(len(rhos))
    ax1.plot(xs, [1.0] * len(rhos), "s--", color=INK, lw=1.8, ms=7,
             label="acceptance rate")
    ax1.plot(xs, frac, "o-", color=ACCENT, lw=2.2, ms=7,
             label="effective samples / draws")
    for i, f in enumerate(frac):
        ax1.annotate(f"{f:.4f}", xy=(i, f), xytext=(0, -17),
                     textcoords="offset points", ha="center", fontsize=8.4,
                     color=ACCENT)
    ax1.set_xticks(xs)
    ax1.set_xticklabels([f"{r:g}" for r in rhos])
    ax1.set_xlabel("correlation of the target", fontsize=9)
    ax1.set_ylabel("fraction", fontsize=9)
    ax1.set_ylim(-0.08, 1.12)
    ax1.set_title("one number never moves", fontsize=10, color=INK,
                  weight="bold")
    ax1.legend(fontsize=8.4, loc="upper right", framealpha=0.95)
    ax1.grid(alpha=0.13)

    # --- 오른쪽: 10만 번 걸어 얻은 것 --------------------------------------
    worth = [f * n for f in frac]
    ax2.bar(xs, worth, color=[INK if i < len(xs) - 1 else ACCENT
                              for i in range(len(xs))], width=0.55,
            alpha=0.85)
    for i, w in enumerate(worth):
        ax2.text(i, w + n * 0.025, f"{w:,.0f}", ha="center", fontsize=9,
                 color=INK)
    ax2.set_xticks(xs)
    ax2.set_xticklabels([f"{r:g}" for r in rhos])
    ax2.set_xlabel("correlation of the target", fontsize=9)
    ax2.set_ylabel(f"what {n:,} draws are worth", fontsize=9)
    ax2.set_ylim(0, n * 1.16)
    ax2.set_title("what they are actually worth", fontsize=10, color=INK,
                  weight="bold")
    ax2.grid(alpha=0.13, axis="y")

    for ax in (ax1, ax2):
        ax.tick_params(labelsize=8, colors=INK)
        for s in ax.spines.values():
            s.set_color(INK)
            s.set_linewidth(0.7)

    fig.text(0.5, -0.03,
             "nothing was ever rejected. the acceptance rate is 1.000 in "
             "every one of these runs.",
             ha="center", fontsize=8.8, style="italic", color=INK)
    fig.tight_layout()
    fig.savefig(HERE / "gibbs_acceptance.svg", transparent=True,
                bbox_inches="tight")
    plt.close(fig)
    print(f"wrote gibbs_acceptance.svg   ESS/n {[round(f, 4) for f in frac]}")


if __name__ == "__main__":
    rejected_samples_count()
    gibbs_acceptance_tells_nothing()
