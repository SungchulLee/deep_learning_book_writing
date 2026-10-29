"""11.2절(LSTM과 GRU)의 그림.

그림에 적히는 수는 모두 그 쪽의 식을 여기서 다시 계산한 것이거나, PyTorch 에
실제로 물어본 값이다. 지어낸 수는 없다.

- lstm_cell_flow.svg      <- lstm_architecture.md 3절 「정보 흐름 그려 보기」
- param_count_vs_hidden.svg <- gru_vs_lstm.md 3절 parameter_scaling_analysis()

규칙은 CLAUDE.md 의 Figures 항에 적어 둔 대로다.
- SVG, fonttype='path', transparent=True
- 글자는 모두 아스키 (matplotlib 기본 폰트에 한글이 없다)

실행: python docs/ch11/lstm_gru/figures/make_figures.py
      python docs/ch11/lstm_gru/figures/make_figures.py --png   <- 눈으로 볼 PNG 도 함께
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["svg.fonttype"] = "path"

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

HERE = Path(__file__).resolve().parent
INK = "#2f3337"
ACCENT = "#8a4b2a"
COOL = "#2f5f8a"
MUTED = "#9aa0a6"
HIGHWAY = "#f0e2d8"


def style(ax):
    ax.tick_params(labelsize=8, colors=INK)
    for s in ax.spines.values():
        s.set_color(INK)
        s.set_linewidth(0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save(fig, name):
    fig.savefig(HERE / f"{name}.svg", transparent=True, bbox_inches="tight")
    if "--png" in sys.argv:          # 글자가 겹치지 않는지 눈으로 보려고 쓴다
        fig.savefig(HERE / f"{name}.png", transparent=False, dpi=170,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {name}.svg")


# ========================================================================
# 1. LSTM 셀의 정보 흐름 — 3절의 아스키 그림을 제대로 그린 것
#    위층은 셀 상태 c, 아래층은 게이트 넷, 오른쪽은 은닉 상태 h.
# ========================================================================

def _arrow(ax, p0, p1, color=INK, lw=1.4, head=7.0, ls="-"):
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle=f"-|>,head_width={head / 24:.3f},"
                           f"head_length={head / 16:.3f}",
        mutation_scale=14, color=color, lw=lw, linestyle=ls,
        shrinkA=0, shrinkB=0, zorder=4))


def _line(ax, p0, p1, color=INK, lw=1.4):
    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=color, lw=lw,
            solid_capstyle="round", zorder=3)


def _op(ax, xy, sym, color=INK):
    """성분별 곱(x)과 덧셈(+) 마디."""
    ax.add_patch(Circle(xy, 0.22, facecolor="white", edgecolor=color,
                        lw=1.3, zorder=5))
    ax.text(xy[0], xy[1], sym, ha="center", va="center", fontsize=11,
            color=color, zorder=6, fontweight="bold")


def _box(ax, xy, w, h, top, bottom, color):
    ax.add_patch(FancyBboxPatch(
        (xy[0] - w / 2, xy[1] - h / 2), w, h,
        boxstyle="round,pad=0.0,rounding_size=0.08",
        facecolor="white", edgecolor=color, lw=1.3, zorder=5))
    ax.text(xy[0], xy[1] + 0.13, top, ha="center", va="center",
            fontsize=9.5, color=color, fontweight="bold", zorder=6)
    ax.text(xy[0], xy[1] - 0.16, bottom, ha="center", va="center",
            fontsize=7.6, color=INK, zorder=6)


def make_lstm_flow():
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    ax.set_xlim(-0.1, 10.2)
    ax.set_ylim(-0.75, 6.35)
    ax.axis("off")

    Y_C = 5.0                     # 셀 상태 통로의 높이
    Y_H = 2.35                    # 은닉 상태 통로의 높이
    Y_G = 0.95                    # 게이트 상자의 높이
    X_F, X_I, X_G, X_O = 2.85, 4.45, 5.95, 7.45

    # --- 고속도로 띠 ---------------------------------------------------
    ax.add_patch(Rectangle((0.15, Y_C - 0.45), 9.85, 0.9, facecolor=HIGHWAY,
                           edgecolor="none", zorder=0))
    ax.text(0.3, Y_C + 0.62, "cell state: the only operations on this lane "
                             "are one elementwise multiply and one add",
            fontsize=8.4, color=ACCENT, style="italic", ha="left")

    # --- 셀 상태 통로 ---------------------------------------------------
    ax.text(0.28, Y_C, r"$c_{t-1}$", fontsize=11, color=INK, ha="center",
            va="center", fontweight="bold")
    _arrow(ax, (0.62, Y_C), (X_F - 0.24, Y_C), color=ACCENT, lw=1.8)
    _op(ax, (X_F, Y_C), "x", ACCENT)
    _arrow(ax, (X_F + 0.24, Y_C), (5.2 - 0.24, Y_C), color=ACCENT, lw=1.8)
    _op(ax, (5.2, Y_C), "+", ACCENT)
    _arrow(ax, (5.2 + 0.24, Y_C), (9.05, Y_C), color=ACCENT, lw=1.8)
    ax.text(9.42, Y_C, r"$c_t$", fontsize=11, color=INK, ha="center",
            va="center", fontweight="bold")
    _arrow(ax, (9.68, Y_C), (10.1, Y_C), color=ACCENT, lw=1.8)
    ax.text(9.9, Y_C + 0.33, "to  t+1", fontsize=7.6, color=INK, ha="center")

    # --- 게이트 넷 ------------------------------------------------------
    _box(ax, (X_F, Y_G), 1.18, 0.72, r"$f_t$  sigmoid", "forget", ACCENT)
    _box(ax, (X_I, Y_G), 1.18, 0.72, r"$i_t$  sigmoid", "input", COOL)
    _box(ax, (X_G, Y_G), 1.18, 0.72, r"$g_t$  tanh", "candidate", COOL)
    _box(ax, (X_O, Y_G), 1.18, 0.72, r"$o_t$  sigmoid", "output", COOL)

    # --- 입력 버스 [h_{t-1}, x_t] ---------------------------------------
    Y_BUS = -0.18
    ax.text(0.28, 1.95, r"$h_{t-1}$", fontsize=11, color=INK, ha="center",
            va="center", fontweight="bold")
    _arrow(ax, (0.62, 1.95), (1.85, 1.95), color=MUTED, lw=1.3)
    ax.text(0.28, Y_BUS, r"$x_t$", fontsize=11, color=INK, ha="center",
            va="center", fontweight="bold")
    _arrow(ax, (0.56, Y_BUS), (1.85, Y_BUS), color=MUTED, lw=1.3)
    _line(ax, (1.9, 1.95), (1.9, Y_BUS), color=MUTED, lw=1.3)
    _line(ax, (1.9, Y_BUS), (X_O, Y_BUS), color=MUTED, lw=1.3)
    for gx in (X_F, X_I, X_G, X_O):
        _arrow(ax, (gx, Y_BUS), (gx, Y_G - 0.36), color=MUTED, lw=1.3)
    ax.text(1.18, 1.02, "concat\n" + r"$[h_{t-1}, x_t]$", fontsize=8,
            color=MUTED, ha="center", va="center")

    # --- f 게이트 -> 곱 마디 --------------------------------------------
    _arrow(ax, (X_F, Y_G + 0.36), (X_F, Y_C - 0.24), color=ACCENT, lw=1.4)
    ax.text(X_F - 0.16, 3.2, "how much of\nthe old state\nsurvives",
            fontsize=7.6, color=ACCENT, ha="right", va="center")

    # --- i * g -> 더하기 마디 -------------------------------------------
    Y_IG = 2.95
    _arrow(ax, (X_I, Y_G + 0.36), (X_I, Y_IG), color=COOL, lw=1.4)
    _arrow(ax, (X_G, Y_G + 0.36), (X_G, Y_IG), color=COOL, lw=1.4)
    _line(ax, (X_I, Y_IG), (X_G, Y_IG), color=COOL, lw=1.4)
    _line(ax, (5.2, Y_IG), (5.2, Y_IG + 0.26), color=COOL, lw=1.4)
    _op(ax, (5.2, Y_IG + 0.48), "x", COOL)
    _arrow(ax, (5.2, Y_IG + 0.7), (5.2, Y_C - 0.24), color=COOL, lw=1.4)
    ax.text(5.62, 4.05, "how much of\nthe new candidate\nis written in",
            fontsize=7.6, color=COOL, ha="left", va="center")

    # --- 셀 상태 -> 은닉 상태 -------------------------------------------
    X_TAP = 8.3
    ax.plot([X_TAP], [Y_C], "o", color=ACCENT, ms=5, zorder=6)
    _line(ax, (X_TAP, Y_C), (X_TAP, 3.72), color=INK, lw=1.4)
    _arrow(ax, (X_TAP, 3.72), (X_TAP, 3.46), color=INK, lw=1.4)
    _box(ax, (X_TAP, 3.1), 0.95, 0.56, "tanh", "", INK)
    _arrow(ax, (X_TAP, 2.82), (X_TAP, Y_H + 0.24), color=INK, lw=1.4)
    _op(ax, (X_TAP, Y_H), "x", INK)
    _line(ax, (X_O, Y_G + 0.36), (X_O, Y_H), color=COOL, lw=1.4)
    _arrow(ax, (X_O, Y_H), (X_TAP - 0.24, Y_H), color=COOL, lw=1.4)
    _arrow(ax, (X_TAP + 0.24, Y_H), (9.05, Y_H), color=INK, lw=1.6)
    ax.text(9.42, Y_H, r"$h_t$", fontsize=11, color=INK, ha="center",
            va="center", fontweight="bold")
    _arrow(ax, (9.68, Y_H), (10.1, Y_H), color=INK, lw=1.6)
    ax.text(9.9, Y_H + 0.33, "output,\nand  t+1", fontsize=7.6, color=INK,
            ha="center", va="bottom")

    ax.text(X_TAP - 0.58, 3.12, "filter the state\nbefore showing it",
            fontsize=7.6, color=INK, ha="right", va="center")

    ax.set_title("why the cell state is called a highway: on the top lane the "
                 "Jacobian is just diag(f_t)",
                 fontsize=10.5, weight="bold", color=INK, pad=10)

    fig.text(0.5, -0.075,
             "schematic - PyTorch stacks the four groups in the order i, f, g, o "
             "and splits each bias in two; both are folded away here.\n"
             "diag(f_t) is the direct path only: the full derivative also leaves "
             "through h_t and comes back into the gates along the grey bus, "
             "where a matrix multiply does sit.",
             ha="center", fontsize=7.6, style="italic", color=INK)
    fig.tight_layout()
    save(fig, "lstm_cell_flow")


# ========================================================================
# 2. 은닉 차원에 따른 파라메터 수 — gru_vs_lstm.md 3절의 코드 그대로
#    PyTorch 규약(묶음마다 바이어스 둘): k * n * (d + n + 2), k = 1, 3, 4
# ========================================================================

def torch_counts(d, n):
    """nn.RNN / nn.GRU / nn.LSTM 에 직접 물어본 파라메터 수."""
    import torch.nn as nn
    return tuple(sum(p.numel() for p in m(d, n).parameters())
                 for m in (nn.RNN, nn.GRU, nn.LSTM))


def make_param_count():
    configs = [(64, 128), (128, 256), (256, 512), (512, 1024), (1024, 2048)]

    # 공식이 PyTorch 와 정말 같은지 여기서 확인하고 간다.
    for d, n in configs:
        assert torch_counts(d, n) == tuple(k * n * (d + n + 2)
                                           for k in (1, 3, 4)), (d, n)

    n = np.logspace(np.log10(100), np.log10(2600), 400)
    d = n / 2                                  # 쪽의 설정이 모두 d = n/2 이다
    base = n * (d + n + 2)

    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(9.2, 4.0), gridspec_kw=dict(width_ratios=[1.3, 1.0]))

    # --- 왼쪽: 절대 개수 -------------------------------------------------
    axL.fill_between(n, 3 * base, 4 * base, color=ACCENT, alpha=0.18, zorder=1)
    axL.plot(n, 4 * base, color=ACCENT, lw=1.9,
             label="LSTM   4 groups: i, f, g, o")
    axL.plot(n, 3 * base, color=COOL, lw=1.9, label="GRU    3 groups: r, z, n")
    axL.plot(n, base, color=MUTED, lw=1.4, ls="--", label="plain RNN   1 group")

    for d_i, n_i in configs:
        _, g, l = torch_counts(d_i, n_i)
        axL.plot([n_i], [l], "o", color=ACCENT, ms=5, zorder=5)
        axL.plot([n_i], [g], "o", color=COOL, ms=5, zorder=5)

    axL.annotate("n = 512, d = 256\nLSTM  1,576,960\nGRU   1,182,720",
                 xy=(512, 1.18e6), xytext=(580, 2.2e5), fontsize=8.2,
                 color=INK, ha="left", va="center",
                 arrowprops=dict(arrowstyle="-", color=INK, lw=0.7))
    axL.annotate("n = 2048, d = 1024\nLSTM  25,182,208\nGRU   18,886,656",
                 xy=(2048, 1.95e7), xytext=(112, 3.6e7), fontsize=8.2,
                 color=INK, ha="left", va="center",
                 arrowprops=dict(arrowstyle="-", color=INK, lw=0.7))

    axL.set_xscale("log")
    axL.set_yscale("log")
    axL.set_xlim(100, 2600)
    axL.set_ylim(3e4, 9e7)
    axL.set_xticks([128, 256, 512, 1024, 2048])
    axL.set_xticklabels(["128", "256", "512", "1024", "2048"])
    axL.set_xlabel("hidden size n   (input size d = n/2)", fontsize=9, color=INK)
    axL.set_ylabel("parameters, one layer one direction (log)",
                   fontsize=9, color=INK)
    axL.set_title("the counts", fontsize=9.5, color=INK, pad=6)
    axL.legend(fontsize=8.0, frameon=False, loc="lower right",
               bbox_to_anchor=(1.02, -0.02), handlelength=1.6,
               labelspacing=0.3)
    style(axL)

    # --- 오른쪽: 같은 수를 n(d+n+2) 로 나눈 것 ----------------------------
    axR.fill_between([100, 2600], 3, 4, color=ACCENT, alpha=0.18, zorder=1)
    axR.axhline(4, color=ACCENT, lw=1.9)
    axR.axhline(3, color=COOL, lw=1.9)
    axR.axhline(1, color=MUTED, lw=1.4, ls="--")

    for d_i, n_i in configs:
        r, g, l = torch_counts(d_i, n_i)
        b = n_i * (d_i + n_i + 2)
        for val, col in ((l / b, ACCENT), (g / b, COOL), (r / b, MUTED)):
            axR.plot([n_i], [val], "o", color=col, ms=5, zorder=5)

    axR.text(2500, 4.13, "LSTM", fontsize=9, color=ACCENT, ha="right",
             weight="bold")
    axR.text(2500, 3.13, "GRU", fontsize=9, color=COOL, ha="right",
             weight="bold")
    axR.text(2500, 1.13, "plain RNN", fontsize=9, color=MUTED, ha="right",
             weight="bold")
    axR.annotate("one group, 25% of the LSTM,\nand the band never narrows",
                 xy=(300, 3.5), xytext=(150, 2.1), fontsize=8.4, color=INK,
                 ha="left", va="center",
                 arrowprops=dict(arrowstyle="-", color=INK, lw=0.7))

    axR.set_xscale("log")
    axR.set_xlim(100, 2600)
    axR.set_ylim(0, 4.8)
    axR.set_xticks([128, 256, 512, 1024, 2048])
    axR.set_xticklabels(["128", "256", "512", "1024", "2048"])
    axR.set_yticks([0, 1, 2, 3, 4])
    axR.set_xlabel("hidden size n", fontsize=9, color=INK)
    axR.set_ylabel("parameters / n(d + n + 2)", fontsize=9, color=INK)
    axR.set_title("the same numbers, divided by one group",
                  fontsize=9.5, color=INK, pad=6)
    style(axR)

    fig.suptitle("three gate groups instead of four: a flat 25% off, "
                 "whatever the hidden size",
                 fontsize=10.5, weight="bold", color=INK, y=1.02)
    fig.text(0.5, -0.07,
             "PyTorch bias convention - two bias vectors per gate group, so the "
             "count is k*n*(d+n+2) with k = 1, 3, 4; the dots are what "
             "nn.RNN / nn.GRU / nn.LSTM actually report. One bias per group "
             "gives k*n*(d+n+1) instead, and the 3/4 ratio is unchanged.",
             ha="center", fontsize=7.6, style="italic", color=INK)
    fig.tight_layout()
    save(fig, "param_count_vs_hidden")


if __name__ == "__main__":
    make_lstm_flow()
    make_param_count()
