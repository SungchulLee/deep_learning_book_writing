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


if __name__ == "__main__":
    rejected_samples_count()
