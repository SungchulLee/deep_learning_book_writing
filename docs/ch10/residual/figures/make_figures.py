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


# === 그림 2: ReLU가 어디 있느냐로 항등 경로가 갈린다 ========================
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

SKIP = "#c2410c"          # 항등 경로를 따라갈 수 있게 한 가지 색으로 칠한다
BOX = "#eef1f5"


def _box(ax, y, label, w=1.5, h=0.34):
    ax.add_patch(FancyBboxPatch((-w / 2, y - h / 2), w, h,
                                boxstyle="round,pad=0.02", linewidth=1.0,
                                edgecolor="#555555", facecolor=BOX, zorder=3))
    ax.text(0, y, label, ha="center", va="center", fontsize=9, zorder=4)


def _down(ax, y0, y1, color="#555555", lw=1.1):
    ax.add_patch(FancyArrowPatch((0, y0), (0, y1), arrowstyle="-|>",
                                 mutation_scale=11, color=color, lw=lw, zorder=2))


def fig_pre_vs_post_activation():
    """두 블록에서 항등 경로가 어디를 지나는지 한 색으로 따라가게 그린다.

    요점은 더하기와 마지막 ReLU의 **순서**다. 사후 활성화에서는 더한 뒤에
    ReLU가 있어 스킵으로 들어온 신호까지 함께 눌린다. 사전 활성화에서는
    더하기가 마지막이라 스킵이 아무것도 거치지 않고 다음 블록으로 간다.
    """
    post = ["Conv 3x3", "BatchNorm", "ReLU", "Conv 3x3", "BatchNorm"]
    pre = ["BatchNorm", "ReLU", "Conv 3x3", "BatchNorm", "ReLU", "Conv 3x3"]

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 6.6))
    for ax, stack, is_post, title in (
            (axes[0], post, True, "post-activation  (original ResNet)"),
            (axes[1], pre, False, "pre-activation")):
        n = len(stack)
        ys = [-(i + 1) * 0.75 for i in range(n)]
        y_in, y_add = 0.0, ys[-1] - 0.85
        ax.text(0, y_in, "$x_l$", ha="center", va="center", fontsize=11, zorder=4)

        _down(ax, y_in - 0.12, ys[0] + 0.19)
        for i, (y, lab) in enumerate(zip(ys, stack)):
            _box(ax, y, lab)
            nxt = ys[i + 1] + 0.19 if i + 1 < n else y_add + 0.16
            _down(ax, y - 0.19, nxt)

        # 스킵 연결 — 입력에서 오른쪽으로 돌아 더하기로 들어온다
        xr = 1.55
        ax.add_patch(FancyArrowPatch((0.06, y_in), (xr, y_in), arrowstyle="-",
                                     color=SKIP, lw=2.2, zorder=2))
        ax.add_patch(FancyArrowPatch((xr, y_in), (xr, y_add), arrowstyle="-",
                                     color=SKIP, lw=2.2, zorder=2))
        ax.add_patch(FancyArrowPatch((xr, y_add), (0.16, y_add), arrowstyle="-|>",
                                     mutation_scale=12, color=SKIP, lw=2.2, zorder=2))
        ax.add_patch(plt.Circle((0, y_add), 0.14, facecolor="white",
                                edgecolor="#555555", lw=1.0, zorder=4))
        ax.text(0, y_add, "+", ha="center", va="center", fontsize=12, zorder=5)

        if is_post:                       # 더한 뒤에 ReLU가 하나 더 있다
            y_relu = y_add - 0.85
            _down(ax, y_add - 0.16, y_relu + 0.19, color=SKIP, lw=2.2)
            ax.add_patch(FancyBboxPatch((-0.75, y_relu - 0.17), 1.5, 0.34,
                                        boxstyle="round,pad=0.02", linewidth=1.6,
                                        edgecolor=SKIP, facecolor="#fde7dc", zorder=3))
            ax.text(0, y_relu, "ReLU", ha="center", va="center", fontsize=9, zorder=4)
            y_out = y_relu - 0.72
            _down(ax, y_relu - 0.19, y_out + 0.13, color=SKIP, lw=2.2)
            ax.text(1.72, y_relu, "the skip passes\nthrough here too",
                    fontsize=8.5, color=SKIP, va="center")
            formula = r"$x_{l+1} = \mathrm{ReLU}(x_l + F(x_l)) \geq 0$"
        else:
            y_out = y_add - 0.72
            _down(ax, y_add - 0.16, y_out + 0.13, color=SKIP, lw=2.2)
            ax.text(1.72, (y_add + y_out) / 2, "the skip reaches\nthe next block\nuntouched",
                    fontsize=8.5, color=SKIP, va="center")
            formula = r"$x_{l+1} = x_l + F(x_l)$"
        ax.text(0, y_out, "$x_{l+1}$", ha="center", va="center", fontsize=11, zorder=4)
        # 두 칸의 식을 같은 높이에 둔다 — 사후 쪽 기둥이 한 칸 길어 y_out이 다르다
        ax.text(0.1, -6.75, formula, ha="center", va="center", fontsize=10)

        ax.set_title(title, fontsize=11, pad=8)
        ax.set_xlim(-1.5, 3.5)
        ax.set_ylim(-7.15, 0.45)
        ax.axis("off")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.01, wspace=0.02)
    fig.savefig("pre_vs_post_activation.svg", transparent=True)
    plt.close(fig)
    print("wrote pre_vs_post_activation.svg")


if __name__ == "__main__":
    fig_residual_vs_plain()
    fig_pre_vs_post_activation()
