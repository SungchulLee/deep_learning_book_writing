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


# === 그림 3: 지름길의 두 모습 ===============================================
def _tensor(ax, x, y, label, color="#555555", w=1.25, h=0.30):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle="round,pad=0.02", linewidth=1.0,
                                edgecolor=color, facecolor=BOX, zorder=3))
    ax.text(x, y, label, ha="center", va="center", fontsize=8.5, zorder=4)


def fig_shortcut_two_shapes():
    """모양이 맞으면 지름길이 공짜이고, 어긋나면 값을 치른다.

    수는 모두 쪽이 적은 것이다. 항등 블록은 18C^2+4C로 C=64에서 73,984개,
    투영 블록은 221,696 + 8,448 = 230,144개다. 그림 스크립트가 nn.Conv2d와
    nn.BatchNorm2d로 다시 세어 같은 값이 나오는지 확인한다.
    """
    import torch.nn as nn

    def count(cin, cout, s):
        mods = [nn.Conv2d(cin, cout, 3, s, 1, bias=False), nn.BatchNorm2d(cout),
                nn.Conv2d(cout, cout, 3, 1, 1, bias=False), nn.BatchNorm2d(cout)]
        main = sum(q.numel() for m in mods for q in m.parameters())
        proj = 0
        if s != 1 or cin != cout:
            sc = nn.Sequential(nn.Conv2d(cin, cout, 1, s, bias=False),
                               nn.BatchNorm2d(cout))
            proj = sum(q.numel() for q in sc.parameters())
        return main, proj

    m_id, p_id = count(64, 64, 1)
    m_pr, p_pr = count(64, 128, 2)
    assert (m_id, p_id) == (73984, 0) and (m_pr, p_pr) == (221696, 8448)

    panels = [("shapes match", "32x32x64", "32x32x64", m_id, p_id,
               "identity - no parameters", "#777777"),
              ("shapes differ", "32x32x64", "16x16x128", m_pr, p_pr,
               "1x1 conv + BN", SKIP)]

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.6))
    for ax, (title, shin, shout, main, proj, sclab, col) in zip(axes, panels):
        _tensor(ax, 0, 0.0, shin)
        for i, lab in enumerate(["Conv 3x3", "BatchNorm", "ReLU",
                                 "Conv 3x3", "BatchNorm"]):
            _box(ax, -(i + 1) * 0.62, lab, w=1.25)
        y_add = -3.95
        _down(ax, -0.15, -0.62 + 0.15)
        for i in range(5):
            y = -(i + 1) * 0.62
            _down(ax, y - 0.15, (y - 0.62 + 0.15) if i < 4 else y_add + 0.14)

        xr = 1.5
        ax.add_patch(FancyArrowPatch((0.05, 0.0), (xr, 0.0), arrowstyle="-",
                                     color=col, lw=2.0, zorder=2))
        if proj:                            # 투영 지름길은 상자를 하나 지난다
            ax.add_patch(FancyBboxPatch((xr - 0.62, -2.1), 1.24, 0.34,
                                        boxstyle="round,pad=0.02", linewidth=1.4,
                                        edgecolor=col, facecolor="#fde7dc", zorder=3))
            ax.text(xr, -1.93, sclab, ha="center", va="center", fontsize=8.5, zorder=4)
            ax.add_patch(FancyArrowPatch((xr, 0.0), (xr, -1.76), arrowstyle="-",
                                         color=col, lw=2.0, zorder=2))
            ax.add_patch(FancyArrowPatch((xr, -2.1 - 0.17), (xr, y_add),
                                         arrowstyle="-", color=col, lw=2.0, zorder=2))
            ax.text(xr + 0.12, -2.6, f"{proj:,} params", fontsize=8.5, color=col)
        else:
            ax.add_patch(FancyArrowPatch((xr, 0.0), (xr, y_add), arrowstyle="-",
                                         color=col, lw=2.0, zorder=2))
            ax.text(xr + 0.12, -2.0, sclab, fontsize=8.5, color=col)
        ax.add_patch(FancyArrowPatch((xr, y_add), (0.15, y_add), arrowstyle="-|>",
                                     mutation_scale=12, color=col, lw=2.0, zorder=2))
        ax.add_patch(plt.Circle((0, y_add), 0.13, facecolor="white",
                                edgecolor="#555555", lw=1.0, zorder=4))
        ax.text(0, y_add, "+", ha="center", va="center", fontsize=12, zorder=5)
        _down(ax, y_add - 0.14, -4.45 + 0.16)
        _box(ax, -4.45, "ReLU", w=1.25)
        _down(ax, -4.45 - 0.16, -4.95 + 0.16)
        _tensor(ax, 0, -4.95, shout)

        ax.set_title(f"{title}   ({shin} -> {shout})", fontsize=10.5, pad=8)
        ax.text(0.1, -5.7, f"block total  {main + proj:,} parameters",
                ha="center", fontsize=9.5)
        ax.set_xlim(-1.35, 3.3)
        ax.set_ylim(-6.1, 0.45)
        ax.axis("off")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.93, bottom=0.02, wspace=0.03)
    fig.savefig("shortcut_two_shapes.svg", transparent=True)
    plt.close(fig)
    print("wrote shortcut_two_shapes.svg")


# === 그림 4: 스템 하나가 해상도를 어디까지 삼키는가 =========================
def fig_stem_resolution():
    """같은 ResNet-18에 스템만 바꾸었을 때 단계별로 남는 공간 크기.

    파라메터는 0.07%밖에 다르지 않은데(7x7 스템 9,408개, 3x3 스템 1,728개)
    32x32 입력에서 남는 크기는 한 축마다 네 배씩 어긋나고, ImageNet 스템 쪽은
    마지막 단계에서 1x1까지 줄어든다. 그 자리에서는 전역 평균 풀링이 한
    위치만 모은다. 크기는 모두 쪽의 크기 공식으로 다시 셈한 값이다.
    """
    def out(h, k, p, s):
        return (h + 2 * p - k) // s + 1

    cifar = [32] + [32 // 2 ** i for i in range(1, 4)]
    after = out(out(32, 7, 3, 2), 3, 1, 2)             # 32 -> 16 -> 8
    imnet = [after] + [after // 2 ** i for i in range(1, 4)]
    assert cifar == [32, 16, 8, 4] and imnet == [8, 4, 2, 1]
    assert 7 * 7 * 3 * 64 - 3 * 3 * 3 * 64 == 7680

    fig, ax = plt.subplots(figsize=(10.0, 4.4))
    rows = [("cifar stem  3x3 / 1", cifar, "tab:blue", 1.55),
            ("imagenet stem  7x7 / 2 + pool", imnet, SKIP, 0.0)]
    unit = 0.030                                        # 화소 하나의 그림 단위
    x0, dx = 2.75, 1.42
    for lab, sizes, col, y0 in rows:
        ax.text(0.04, y0, lab, ha="left", va="center", fontsize=9.5, color=col)
        for i, s_ in enumerate(sizes):
            x = x0 + i * dx
            side = s_ * unit
            ax.add_patch(plt.Rectangle((x - side / 2, y0 - side / 2), side, side,
                                       facecolor=col, alpha=0.22,
                                       edgecolor=col, lw=1.3, zorder=3))
            ax.text(x, y0 - side / 2 - 0.22, f"{s_}x{s_}", ha="center",
                    fontsize=9, color=col)
            if i < 3:
                ax.annotate("", xy=(x + dx - side / 2 - 0.08, y0),
                            xytext=(x + side / 2 + 0.08, y0),
                            arrowprops=dict(arrowstyle="->", color="0.65", lw=1.0))
    for i in range(4):
        ax.text(x0 + i * dx, 2.55, f"stage {i + 1}", ha="center", fontsize=9,
                color="0.4")
    ax.text(x0 + 3 * dx, -0.62, "global average pooling\ncombines one position",
            ha="center", va="top", fontsize=8.5, color=SKIP)
    ax.text(4.3, 3.05,
            "same ResNet-18, 11.18 M parameters either way - the two stems "
            "differ by 7,680 weights, 0.07%",
            ha="center", fontsize=9)
    ax.set_xlim(0, 8.3); ax.set_ylim(-1.35, 3.35)
    ax.axis("off")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    fig.savefig("stem_resolution.svg", transparent=True)
    plt.close(fig)
    print("wrote stem_resolution.svg")


if __name__ == "__main__":
    fig_residual_vs_plain()
    fig_pre_vs_post_activation()
    fig_shortcut_two_shapes()
    fig_stem_resolution()
