"""
================================================================================
make_figures.py - 11.2절(잔차 연결)의 그림을 만든다
================================================================================

만드는 그림:
    residual_vs_plain.svg   시드 셋에서 잰 평가 정확도. 두 띠가 겹치지 않고
                            평균 차이가 시드 사이의 폭보다 크다는 것을 보인다

그림 규칙(CLAUDE.md):
    - SVG로 저장하고 svg.fonttype='path'로 글자를 외곽선으로 만든다
    - transparent=True
    - 그림 안의 글자는 모두 ASCII (기본 글꼴에 한글이 없다)

실행:
    python make_figures.py
================================================================================
"""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams["svg.fonttype"] = "path"


# === 그림 1: 평범한 쪽과 잔차 쪽, 시드 셋 ===================================
# 03_training_comparison.md 출력 블록의 평가 정확도(val) 열을 그대로 옮긴 것이다.
# 줄 순서는 시드 0, 1, 2이고 열은 에포크 1~5다. 학습이 필요 없다.
VAL = {
    "plain":    [[8.79, 15.62, 17.38, 19.92, 16.60],
                 [10.94, 19.53, 21.29, 22.85, 22.85],
                 [10.16, 15.43, 16.41, 18.36, 18.16]],
    "residual": [[20.51, 28.71, 36.91, 36.13, 39.26],
                 [17.19, 25.59, 29.49, 32.23, 35.74],
                 [15.23, 29.10, 35.35, 31.84, 32.03]],
}


def fig_residual_vs_plain():
    """3.3절의 판정을 그림으로 보인다.

    그 절이 적는 것은 "평균 차이 16.47%포인트, 가장 넓은 시드 폭 7.23%포인트,
    그래서 차이가 폭보다 크다"이다. 표로는 열다섯 줄을 눈으로 따라가야 하지만
    띠로 그리면 두 무리가 겹치지 않는 것이 바로 보인다.
    """
    ep = np.arange(1, 6)
    style = {"plain": ("#777777", "plain"), "residual": ("tab:blue", "residual")}

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ends = {}
    for key, (color, label) in style.items():
        runs = np.array(VAL[key])
        lo, hi, mean = runs.min(0), runs.max(0), runs.mean(0)
        ax.fill_between(ep, lo, hi, color=color, alpha=0.16, linewidth=0)
        for r in runs:                      # 시드 셋을 가는 선으로 함께 둔다
            ax.plot(ep, r, color=color, lw=0.9, alpha=0.55)
        ax.plot(ep, mean, color=color, lw=2.4, label=f"{label}  (mean of 3 seeds)")
        ends[key] = (mean[-1], lo[-1], hi[-1])

    # 마지막 에포크에서 차이와 폭을 나란히 잰다
    pm, plo, phi = ends["plain"]
    rm, rlo, rhi = ends["residual"]
    x = 5.42
    ax.annotate("", xy=(x, rm), xytext=(x, pm),
                arrowprops=dict(arrowstyle="<->", color="black", lw=1.3))
    ax.text(x + 0.06, (pm + rm) / 2, f"gap of means\n{rm - pm:.2f} points",
            fontsize=9, va="center")
    xs = x - 0.20
    ax.annotate("", xy=(xs, rhi), xytext=(xs, rlo),
                arrowprops=dict(arrowstyle="<->", color="tab:blue", lw=1.3))
    ax.text(xs, rhi + 1.0, f"widest seed spread {rhi - rlo:.2f}",
            fontsize=9, color="tab:blue", ha="center", va="bottom")

    ax.set_xticks(ep)
    ax.set_xlim(0.85, 6.6)
    ax.set_ylim(5, 47)
    ax.set_xlabel("epoch", fontsize=10)
    ax.set_ylabel("validation accuracy (%)", fontsize=10)
    ax.legend(fontsize=9, frameon=False, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.subplots_adjust(left=0.095, right=0.985, top=0.97, bottom=0.125)
    fig.savefig("residual_vs_plain.svg", transparent=True)
    plt.close(fig)
    print("wrote residual_vs_plain.svg")


if __name__ == "__main__":
    fig_residual_vs_plain()
