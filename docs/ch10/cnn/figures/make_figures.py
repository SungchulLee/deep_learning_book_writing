"""ch10/cnn 의 그림을 그린다.

규칙은 CLAUDE.md 의 Figures 항에 적어 둔 대로다.
그림 안의 글자는 모두 아스키라야 한다 (matplotlib 기본 폰트에 한글이 없다).

그리는 그림
  convolution_sliding.svg     convolution.md 2절 — 5x5 입력에 3x3 커널이 미끄러진다
  transposed_conv_steps.svg   transposed_conv.md 2절 — 0 끼우기 / 덧대기 / 보통 컨볼루션
  dilated_vs_downsample.svg   dilated_convolutions.md 6절 — 두 경로의 해상도
  gridding_coverage.svg       dilated_convolutions.md 5절 — d=2,2 와 d=1,2,4 의 덮개
  rf_strategies.svg           receptive_field.md 4절 — 커널·깊이·팽창 세 전략
  depthwise_decomposition.svg depthwise_separable.md 4절 — 표준 대 깊이별+점별
  advanced_training_curve.svg 07_cifar10_advanced.md 2.3절 — 열네 에포크의 정체

그림에 적는 수는 모두 쪽의 코드·출력에서 확인한 것이다. check_numbers() 가
그 확인을 다시 돌린다.

실행: python docs/ch10/cnn/figures/make_figures.py
      python docs/ch10/cnn/figures/make_figures.py --png   ← 눈으로 볼 PNG 도 함께
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["svg.fonttype"] = "path"

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

HERE = Path(__file__).resolve().parent

INK = "#2f3337"
ACCENT = "#8a4b2a"      # 호박빛 — 눈여겨볼 자리
COOL = "#2f6f8f"        # 청록 — 견주는 쪽
MUTED = "#9aa1a6"
FILL = "#f3ece4"
FILL_COOL = "#e6eff4"

# === 공통 도구 ====================================================


def cell(ax, x, y, s, label=None, fc="none", ec=INK, lw=0.8,
         fontsize=8, color=INK, ls="-"):
    """왼쪽 아래 모서리가 (x, y)인 한 칸."""
    ax.add_patch(Rectangle((x, y), s, s, fc=fc, ec=ec, lw=lw, ls=ls,
                           zorder=1))
    if label is not None:
        ax.text(x + s / 2, y + s / 2, label, ha="center", va="center",
                fontsize=fontsize, family="monospace", color=color, zorder=3)


def grid(ax, x0, ytop, values, s=1.0, fontsize=8, fc="none", ec=INK,
         lw=0.8, color=INK, ls="-"):
    """values[r][c] 를 왼쪽 위가 (x0, ytop)인 격자로 그린다. None 은 빈 칸."""
    for r, row in enumerate(values):
        for c, v in enumerate(row):
            cell(ax, x0 + c * s, ytop - (r + 1) * s, s,
                 None if v is None else str(v),
                 fc=fc, ec=ec, lw=lw, fontsize=fontsize, color=color, ls=ls)


def frame(ax, x0, ytop, rows, cols, s=1.0, ec=ACCENT, lw=2.0, ls="-",
          inset=0.0):
    """격자 위에 창을 하나 두른다."""
    ax.add_patch(Rectangle((x0 + inset, ytop - rows * s + inset),
                           cols * s - 2 * inset, rows * s - 2 * inset,
                           fc="none", ec=ec, lw=lw, ls=ls, zorder=4))


def arrow(ax, x0, y0, x1, y1, color=INK, lw=1.3):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                 mutation_scale=11, color=color, lw=lw,
                                 zorder=2))


def caption(ax, x, y, text, color=INK, fontsize=8.5, weight=None,
            style=None, ha="center", va="center", family=None):
    ax.text(x, y, text, ha=ha, va=va, fontsize=fontsize, color=color,
            weight=weight, style=style, family=family)


# === 그림 1. 미끄러지는 커널 ======================================

INPUT_5x5 = [[1, 2, 3, 0, 1],
             [0, 1, 2, 3, 1],
             [1, 2, 1, 0, 0],
             [0, 0, 1, 2, 1],
             [1, 1, 0, 1, 0]]

KERNEL_3x3 = [[1, 0, -1],
              [1, 0, -1],
              [1, 0, -1]]


def correlate(image, kernel):
    """스트라이드 1 · 패딩 없음 상호상관. 쪽의 정의 그대로다."""
    kh, kw = len(kernel), len(kernel[0])
    oh, ow = len(image) - kh + 1, len(image[0]) - kw + 1
    return [[sum(image[i + m][j + n] * kernel[m][n]
                 for m in range(kh) for n in range(kw))
             for j in range(ow)] for i in range(oh)]


def make_convolution_sliding():
    out = correlate(INPUT_5x5, KERNEL_3x3)
    win = [row[0:3] for row in INPUT_5x5[0:3]]
    prod = [[win[r][c] * KERNEL_3x3[r][c] for c in range(3)] for r in range(3)]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.8, 5.9),
                                   gridspec_kw={"height_ratios": [1.45, 1]})

    # --- 위: 5x5 입력, 세 자리의 창, 3x3 출력 ---
    grid(ax1, 0, 5, INPUT_5x5, fontsize=9)
    frame(ax1, 0, 5, 3, 3, ec=ACCENT, lw=2.2)
    frame(ax1, 2, 5, 3, 3, ec=COOL, lw=1.6, ls=(0, (4, 2)), inset=0.09)
    frame(ax1, 2, 3, 3, 3, ec=MUTED, lw=1.6, ls=(0, (2, 2)), inset=0.18)
    caption(ax1, 2.5, 5.45, "Input 5x5", fontsize=9.5, weight="bold")
    caption(ax1, 1.5, -0.42, "window (0,0)", color=ACCENT, fontsize=8)
    caption(ax1, 3.9, -0.42, "(0,2)", color=COOL, fontsize=8)
    caption(ax1, 5.18, 1.5, "(2,2)", color=MUTED, fontsize=8, ha="left")
    arrow(ax1, 3.15, 4.0, 4.35, 4.0, color=COOL, lw=1.1)

    caption(ax1, 6.55, 2.5, "*", fontsize=15, color=INK)

    grid(ax1, 7.2, 4.0, KERNEL_3x3, fontsize=9, fc=FILL)
    caption(ax1, 8.7, 4.45, "Kernel 3x3", fontsize=9.5, weight="bold")

    caption(ax1, 10.75, 2.5, "=", fontsize=15, color=INK)

    grid(ax1, 11.4, 4.0, out, fontsize=9)
    frame(ax1, 11.4, 4.0, 1, 1, ec=ACCENT, lw=2.2)
    caption(ax1, 12.9, 4.45, "Output 3x3", fontsize=9.5, weight="bold")
    caption(ax1, 12.9, 0.55, "5 - 3 + 1 = 3 on each side", fontsize=8,
            color=MUTED)

    ax1.set_xlim(-0.7, 14.9)
    ax1.set_ylim(-0.95, 5.95)

    # --- 아래: 창 (0,0) 한 자리의 성분별 곱 ---
    grid(ax2, 0, 3, win, fontsize=9)
    frame(ax2, 0, 3, 3, 3, ec=ACCENT, lw=2.2)
    caption(ax2, 1.5, 3.4, "window at (0,0)", fontsize=9)

    caption(ax2, 3.6, 1.5, "x", fontsize=13, color=INK)
    grid(ax2, 4.2, 3, KERNEL_3x3, fontsize=9, fc=FILL)
    caption(ax2, 5.7, 3.4, "kernel", fontsize=9)

    caption(ax2, 7.8, 1.5, "=", fontsize=15, color=INK)
    grid(ax2, 8.4, 3, prod, fontsize=9, fc=FILL)
    caption(ax2, 9.9, 3.4, "elementwise products", fontsize=9)

    flat = [prod[r][c] for r in range(3) for c in range(3)]
    terms = " ".join(("+ " if v >= 0 else "- ") + str(abs(v)) for v in flat)
    terms = terms[2:] if terms.startswith("+ ") else terms
    caption(ax2, 5.7, -0.55,
            "sum = " + terms + "  =  " + str(out[0][0]),
            fontsize=9.5, family="monospace", color=ACCENT)
    caption(ax2, 5.7, -1.45,
            "slide the window one step and repeat - nine windows, "
            "nine output values",
            fontsize=8.5, style="italic", color=MUTED)

    ax2.set_xlim(-0.7, 12.3)
    ax2.set_ylim(-1.95, 3.8)

    for ax in (ax1, ax2):
        ax.set_aspect("equal")
        ax.set_axis_off()

    fig.tight_layout()
    fig.savefig(HERE / "convolution_sliding.svg", transparent=True,
                bbox_inches="tight")
    plt.close(fig)
    print("wrote convolution_sliding.svg")


# === 그림 2. 전치 컨볼루션의 세 걸음 ==============================


def _zero_inserted(i, s):
    return s * (i - 1) + 1


def make_transposed_conv_steps():
    i, k, s, p = 2, 3, 2, 1
    n = _zero_inserted(i, s)                 # 3
    pad = k - 1 - p                          # 1
    padded = n + 2 * pad                     # 5
    out0 = padded - k + 1                    # 3
    out1 = (padded + 1) - k + 1              # 4  (output_padding = 1)

    small = [["a", "b"], ["c", "d"]]
    ins = [["a", 0, "b"], [0, 0, 0], ["c", 0, "d"]]
    pad5 = [[0] * 5,
            [0, "a", 0, "b", 0],
            [0] * 5,
            [0, "c", 0, "d", 0],
            [0] * 5]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.8, 6.6))

    cs = 0.62   # 칸 크기
    gap = 2.3   # 격자와 격자 사이

    def place(ax, x0, ymid, values, title, fc="none", fontsize=7.5):
        rows, cols = len(values), len(values[0])
        grid(ax, x0, ymid + rows * cs / 2, values, s=cs, fontsize=fontsize,
             fc=fc)
        caption(ax, x0 + cols * cs / 2, ymid + rows * cs / 2 + 0.22, title,
                fontsize=8.5, weight="bold", va="bottom")
        return x0 + cols * cs

    def step(ax, x, ymid, text, color=COOL):
        arrow(ax, x + 0.25, ymid, x + gap - 0.25, ymid, color=color)
        caption(ax, x + gap / 2, ymid + 0.14, text, fontsize=7.5,
                color=color, va="bottom")

    # --- 위: output_padding = 0 ---
    ym = 0.0
    x = place(ax1, 0.0, ym, small, "Input 2x2", fc=FILL)
    step(ax1, x, ym, "insert\n" + f"s - 1 = {s - 1} zero")
    x = place(ax1, x + gap, ym, ins, f"Zeros in {n}x{n}")
    step(ax1, x, ym, "pad\n" + f"k - 1 - p = {pad}\neach side")
    x = place(ax1, x + gap, ym, pad5, f"Padded {padded}x{padded}")
    step(ax1, x, ym, "convolve\n" + f"{k}x{k}, stride 1\nvalid")
    x0_out = x + gap
    x = place(ax1, x0_out, ym, [[""] * out0 for _ in range(out0)],
              f"Output {out0}x{out0}", fc=FILL)
    caption(ax1, x0_out + out0 * cs / 2, ym - out0 * cs / 2 - 0.22,
            f"{padded} - {k} + 1 = {out0}", fontsize=8, color=ACCENT,
            family="monospace", va="top")

    caption(ax1, 0.0, 2.65, "s = 2, p = 1, k = 3, output_padding = 0",
            fontsize=9.5, weight="bold", ha="left", color=INK)
    caption(ax1, 0.0, -2.25,
            "L_out = (2 - 1)*2 - 2*1 + 1*(3 - 1) + 0 + 1 = 3",
            fontsize=8.5, family="monospace", ha="left", va="top", color=INK)

    ax1.set_xlim(-0.4, 15.6)
    ax1.set_ylim(-2.95, 3.1)

    # --- 아래: output_padding = 1 ---
    ym2 = 0.0
    pad6 = [row + [0] for row in pad5] + [[0] * 6]
    x = place(ax2, 0.0, ym2, pad6, "Padded 6x6")
    # 덧붙은 마지막 행·열만 점선으로 다시 그린다
    top6 = ym2 + 6 * cs / 2
    for r in range(6):
        cell(ax2, 5 * cs, top6 - (r + 1) * cs, cs, "0", fc=FILL, ec=ACCENT,
             lw=1.2, ls=(0, (2, 1.6)), fontsize=7.5, color=ACCENT)
    for c in range(5):
        cell(ax2, c * cs, top6 - 6 * cs, cs, "0", fc=FILL, ec=ACCENT,
             lw=1.2, ls=(0, (2, 1.6)), fontsize=7.5, color=ACCENT)

    arrow(ax2, x + 0.3, ym2, x + 2.9, ym2, color=ACCENT)
    caption(ax2, x + 1.6, ym2 + 0.14,
            "convolve\n3x3, stride 1, valid", fontsize=7.5, color=ACCENT,
            va="bottom")
    x0_out = x + 3.2
    place(ax2, x0_out, ym2, [[""] * out1 for _ in range(out1)],
          f"Output {out1}x{out1}", fc=FILL)
    caption(ax2, x0_out + out1 * cs / 2, ym2 - out1 * cs / 2 - 0.22,
            f"6 - {k} + 1 = {out1}", fontsize=8, color=ACCENT,
            family="monospace", va="top")

    caption(ax2, 0.0, 2.65,
            "same s, p, k - but output_padding = 1", fontsize=9.5,
            weight="bold", ha="left", color=ACCENT)
    caption(ax2, 0.0, -2.25,
            "L_out = (2 - 1)*2 - 2*1 + 1*(3 - 1) + 1 + 1 = 4",
            fontsize=8.5, family="monospace", ha="left", va="top",
            color=ACCENT)
    caption(ax2, 10.1, 1.1,
            "output_padding grows the padded map\n"
            "by one zero row and column, bottom and\n"
            "right only - the extra output row is\n"
            "still computed from a, b, c, d, not zero",
            fontsize=8, color=MUTED, ha="left", va="top")

    ax2.set_xlim(-0.4, 15.6)
    ax2.set_ylim(-2.95, 3.1)

    for ax in (ax1, ax2):
        ax.set_aspect("equal")
        ax.set_axis_off()

    fig.tight_layout()
    fig.savefig(HERE / "transposed_conv_steps.svg", transparent=True,
                bbox_inches="tight")
    plt.close(fig)
    print("wrote transposed_conv_steps.svg")


# === 그림 3. 두 경로의 해상도 =====================================


def make_dilated_vs_downsample():
    fig, ax = plt.subplots(figsize=(9.6, 4.4))

    BIG = 1.15      # 224x224 의 변
    PITCH = 2.65    # 상자 중심 사이

    def fmap(x, y, res, label, color=INK, fc="none", ref=224.0):
        side = BIG * (res / ref) ** 0.5
        ax.add_patch(Rectangle((x - side / 2, y - side / 2), side, side,
                               fc=fc, ec=color, lw=1.2, zorder=2))
        ax.text(x, y - BIG / 2 - 0.2, label, ha="center", va="top",
                fontsize=8, family="monospace", color=color)

    xs = [0.0 + i * PITCH for i in range(5)]

    # --- 위: 표준 CNN ---
    ytop = 2.5
    res = [224, 112, 56, 7, 224]
    labels = ["224x224", "112x112", "56x56", "7x7", "224x224"]
    steps = ["Conv/Pool", "Conv/Pool", "... Conv/Pool", "Upsample"]
    for x, r, lab in zip(xs, res, labels):
        fmap(x, ytop, r, lab, color=INK, fc=FILL if r == 224 else "none")
    for a, b, st in zip(xs[:-1], xs[1:], steps):
        arrow(ax, a + BIG / 2 + 0.16, ytop, b - BIG / 2 - 0.16, ytop,
              color=INK, lw=1.2)
        caption(ax, (a + b) / 2, ytop + 0.14, st, fontsize=7.5, color=INK,
                va="bottom")
    caption(ax, -1.25, ytop, "standard", fontsize=9.5, weight="bold",
            color=INK, ha="center")
    caption(ax, xs[-1] + 1.0, ytop,
            "what the 7x7 map\ndropped does not\ncome back",
            fontsize=8, color=ACCENT, ha="left")

    # --- 아래: 팽창 CNN ---
    ybot = 0.0
    steps2 = ["Conv d=1", "Conv d=2", "Conv d=4", "Conv d=8"]
    rfs = ["RF 3", "RF 7", "RF 15", "RF 31"]
    for x in xs:
        fmap(x, ybot, 224, "224x224", color=COOL, fc=FILL_COOL)
    for a, b, st, rf in zip(xs[:-1], xs[1:], steps2, rfs):
        arrow(ax, a + BIG / 2 + 0.16, ybot, b - BIG / 2 - 0.16, ybot,
              color=COOL, lw=1.2)
        caption(ax, (a + b) / 2, ybot + 0.14, st, fontsize=7.5, color=COOL,
                va="bottom")
        caption(ax, (a + b) / 2, ybot - 0.16, rf, fontsize=7.5, color=COOL,
                family="monospace", va="top")
    caption(ax, -1.25, ybot, "dilated", fontsize=9.5, weight="bold",
            color=COOL, ha="center")
    caption(ax, xs[-1] + 1.0, ybot,
            "resolution kept, so\nevery layer pays the\nfull-resolution cost",
            fontsize=8, color=COOL, ha="left")

    caption(ax, xs[2], -1.55,
            "box size is schematic; the printed sizes are the real ones",
            fontsize=8, style="italic", color=MUTED)

    ax.set_xlim(-2.3, xs[-1] + 4.2)
    ax.set_ylim(-1.95, 3.5)
    ax.set_aspect("equal")
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(HERE / "dilated_vs_downsample.svg", transparent=True,
                bbox_inches="tight")
    plt.close(fig)
    print("wrote dilated_vs_downsample.svg")


# === 그림 4. 격자 무늬와 덮개 =====================================


def reachable(dilations, k=3):
    """커널 k·스트라이드 1 레이어를 쌓았을 때 한 축에서 닿는 입력 오프셋."""
    offsets = {0}
    half = k // 2
    for d in dilations:
        offsets = {o + d * j for o in offsets for j in range(-half, half + 1)}
    return sorted(offsets)


def _coverage_panel(ax, x0, dilations, title, color, fc, note):
    """한 축의 닿는 자리를 두 축으로 넓혀 덮개를 그린다. 칸 크기는 둘이 같다."""
    offs = reachable(dilations)
    lo, hi = offs[0], offs[-1]
    n = hi - lo + 1
    hit = set(offs)
    for r in range(n):
        for c in range(n):
            on = (lo + r) in hit and (lo + c) in hit
            cell(ax, x0 + c, n - r - 1, 1.0,
                 fc=fc if on else "none",
                 ec=color if on else MUTED,
                 lw=0.9 if on else 0.4)
            if on:
                ax.plot([x0 + c + 0.5], [n - r - 0.5], marker="o", ms=2.8,
                        color=color, zorder=3)
    caption(ax, x0 + n / 2, n + 0.95, title, fontsize=10, weight="bold",
            color=color, va="bottom")
    caption(ax, x0 + n / 2, n + 0.22, f"RF {n}x{n}", fontsize=9,
            family="monospace", color=color, va="bottom")
    sampled = len(hit) ** 2
    caption(ax, x0 + n / 2, -0.55,
            f"{sampled} of {n * n} positions sampled", fontsize=9.5,
            family="monospace", color=color, va="top")
    caption(ax, x0 + n / 2, -1.75, note, fontsize=9, color=color, va="top")
    return n


def make_gridding_coverage():
    fig, ax = plt.subplots(figsize=(9.6, 6.6))

    n1 = _coverage_panel(ax, 0.0, (2, 2), "two layers, both d = 2",
                         ACCENT, FILL,
                         "the other 56 are never seen,\n"
                         "however deep the stack goes")
    _coverage_panel(ax, n1 + 2.6, (1, 2, 4), "three layers, d = 1, 2, 4",
                    COOL, FILL_COOL,
                    "no holes - mixing the rates\ncloses the grid")

    ax.set_xlim(-1.4, n1 + 2.6 + 15 + 1.4)
    ax.set_ylim(-3.8, 17.2)
    ax.set_aspect("equal")
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(HERE / "gridding_coverage.svg", transparent=True,
                bbox_inches="tight")
    plt.close(fig)
    print("wrote gridding_coverage.svg")


# === 선그래프용 공통 도구 =========================================


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


# === 그림 5. 리셉티브 필드를 넓히는 세 전략 ========================
#     receptive_field.md 4절. 점이든 선이든 모두 그 쪽 3절의
#     compute_receptive_field() 를 그대로 돌려 얻은 값이다.


def compute_receptive_field(layers):
    """receptive_field.md 3절의 함수와 글자 그대로 같다."""
    rf = 1
    jump = 1
    for layer in layers:
        k = layer.get("kernel", 1)
        s = layer.get("stride", 1)
        d = layer.get("dilation", 1)
        k_eff = d * (k - 1) + 1
        rf = rf + (k_eff - 1) * jump
        jump = jump * s
    return rf, jump


def _deeper(n):                      # 전략 2: 3x3 을 더 쌓는다
    return [{"kernel": 3, "stride": 1} for _ in range(n)]


def _wider(n):                       # 전략 1: 커널을 7x7 로 키운다
    return [{"kernel": 7, "stride": 1} for _ in range(n)]


def _dilated(n):                     # 전략 4: 팽창률을 1,2,4,8,... 로 키운다
    return [{"kernel": 3, "stride": 1, "dilation": 2 ** i} for i in range(n)]


def weight_proxy(layers):
    """쪽의 compare_architectures() 가 쓰는 파라메터 대용치 — k^2 의 합."""
    return sum(layer.get("kernel", 1) ** 2 for layer in layers)


DEPTHS = list(range(0, 11))

STRATEGIES = (
    ("more 3x3 layers (k=3, d=1)", _deeper, COOL, "o", "-"),
    ("larger kernel (k=7, d=1)", _wider, MUTED, "s", "--"),
    ("dilation d = 1, 2, 4, 8, ... (k=3)", _dilated, ACCENT, "^", "-"),
)


def make_rf_strategies():
    curves = {name: [compute_receptive_field(build(n))[0] for n in DEPTHS]
              for name, build, _, _, _ in STRATEGIES}
    costs = {name: [weight_proxy(build(n)) for n in DEPTHS]
             for name, build, _, _, _ in STRATEGIES}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.6, 4.3))

    # --- 왼쪽: 깊이에 대한 리셉티브 필드 ---
    for name, build, color, marker, ls in STRATEGIES:
        ax1.plot(DEPTHS, curves[name], color=color, lw=1.7, ls=ls,
                 marker=marker, ms=4.2, label=name, zorder=3)

    ax1.set_yscale("log")
    ax1.set_xlabel("number of stacked layers", fontsize=9, color=INK)
    ax1.set_ylabel("receptive field width (px)", fontsize=9, color=INK)
    ax1.set_xticks(DEPTHS)
    ax1.set_yticks([1, 3, 10, 30, 100, 300, 1000, 3000])
    ax1.set_yticklabels(["1", "3", "10", "30", "100", "300", "1000", "3000"])
    ax1.set_ylim(0.8, 6000)
    ax1.grid(alpha=0.25, lw=0.6, zorder=0)
    ax1.legend(fontsize=8, loc="upper left", frameon=False)
    style(ax1)

    d5 = {name: curves[name][5] for name, *_ in STRATEGIES}
    rf5_dil = d5["dilation d = 1, 2, 4, 8, ... (k=3)"]
    rf5_std = d5["more 3x3 layers (k=3, d=1)"]
    ax1.annotate(f"5 layers: RF {rf5_dil}\n"
                 f"({rf5_dil / rf5_std:.1f}x the k=3 stack)",
                 xy=(5, rf5_dil), xytext=(5.4, 700), fontsize=8, color=ACCENT,
                 ha="left",
                 arrowprops=dict(arrowstyle="-", color=ACCENT, lw=0.8))
    ax1.annotate(f"5 layers: RF {d5['more 3x3 layers (k=3, d=1)']}",
                 xy=(5, d5["more 3x3 layers (k=3, d=1)"]),
                 xytext=(5.4, 2.0), fontsize=8, color=COOL, ha="left",
                 arrowprops=dict(arrowstyle="-", color=COOL, lw=0.8))
    ax1.annotate(f"10 layers: RF {curves['more 3x3 layers (k=3, d=1)'][10]}",
                 xy=(10, curves["more 3x3 layers (k=3, d=1)"][10]),
                 xytext=(8.2, 4.4), fontsize=8, color=COOL, ha="left",
                 arrowprops=dict(arrowstyle="-", color=COOL, lw=0.8))
    ax1.set_title("(a) how fast the width grows", fontsize=9.5, color=INK,
                  loc="left")

    # --- 오른쪽: 그 폭에 치르는 웨이트 값 ---
    for name, build, color, marker, ls in STRATEGIES:
        ax2.plot(costs[name][1:], curves[name][1:], color=color, lw=1.7,
                 ls=ls, marker=marker, ms=4.2, label=name, zorder=3)

    ax2.set_yscale("log")
    ax2.set_xlabel("sum of k^2 over the stack (the page's weight proxy)",
                   fontsize=9, color=INK)
    ax2.set_ylabel("receptive field width (px)", fontsize=9, color=INK)
    ax2.set_yticks([3, 10, 30, 100, 300, 1000, 3000])
    ax2.set_yticklabels(["3", "10", "30", "100", "300", "1000", "3000"])
    ax2.set_ylim(2.0, 6000)
    ax2.grid(alpha=0.25, lw=0.6, zorder=0)
    style(ax2)
    c10 = costs["more 3x3 layers (k=3, d=1)"][10]
    lo10 = curves["more 3x3 layers (k=3, d=1)"][10]
    hi10 = curves["dilation d = 1, 2, 4, 8, ... (k=3)"][10]
    assert c10 == costs["dilation d = 1, 2, 4, 8, ... (k=3)"][10]
    ax2.plot([c10, c10], [lo10, hi10], color=INK, lw=0.9, ls=":", zorder=2)
    ax2.annotate(f"the same {c10} weights, ten layers of k=3:\n"
                 f"RF {lo10} without dilation, RF {hi10} with",
                 xy=(c10, 220), xytext=(125, 220), fontsize=8, color=INK,
                 ha="left", va="center",
                 arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    ax2.annotate("k=7 pays 49 weights a layer\nfor a width k=3 reaches too",
                 xy=(costs["larger kernel (k=7, d=1)"][4],
                     curves["larger kernel (k=7, d=1)"][4]),
                 xytext=(170, 60), fontsize=8, color=MUTED, ha="left",
                 va="center",
                 arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    ax2.set_title("(b) what that width costs in weights", fontsize=9.5,
                  color=INK, loc="left")

    fig.tight_layout()
    save(fig, "rf_strategies")


# === 그림 6. 표준 컨볼루션을 깊이별과 점별로 쪼갠다 ================
#     depthwise_separable.md 4절. 셈은 모두 그 절의 식에서 나온다.

DS_CIN, DS_COUT, DS_K = 64, 128, 3


def ds_counts(c_in=DS_CIN, c_out=DS_COUT, k=DS_K):
    """4절 표의 웨이트 수와 바이어스 수를 식 그대로 센다."""
    std_w = c_in * c_out * k * k
    dw_w = c_in * k * k
    pw_w = c_in * c_out
    return {
        "std_w": std_w, "std_b": c_out,
        "dw_w": dw_w, "dw_b": c_in,
        "pw_w": pw_w, "pw_b": c_out,
        "sep_w": dw_w + pw_w,
        "sep_b": c_in + c_out,
    }


def make_depthwise_decomposition():
    n = ds_counts()
    ratio_w = n["std_w"] / n["sep_w"]
    ratio_p = (n["std_w"] + n["std_b"]) / (n["sep_w"] + n["sep_b"])

    fig, ax = plt.subplots(figsize=(9.8, 5.4))

    H = 32          # 4절 코드가 쓰는 입력 크기
    UNIT = 1.0 / 80.0

    def tensor(x, y, c, label, color=INK, fc="none"):
        h = c * UNIT
        ax.add_patch(Rectangle((x, y - h / 2), 0.62, h, fc=fc, ec=color,
                               lw=1.2, zorder=2))
        ax.text(x + 0.31, y + h / 2 + 0.11, label, ha="center", va="bottom",
                fontsize=8, family="monospace", color=color)
        return x + 0.62

    def op(x, y, width, title, shape, count, color):
        ax.add_patch(FancyArrowPatch((x + 0.12, y), (x + width - 0.12, y),
                                     arrowstyle="-|>", mutation_scale=12,
                                     color=color, lw=1.4, zorder=2))
        ax.text(x + width / 2, y + 0.16, title, ha="center", va="bottom",
                fontsize=8.5, color=color, weight="bold")
        ax.text(x + width / 2, y - 0.17, shape, ha="center", va="top",
                fontsize=8, color=color, family="monospace")
        ax.text(x + width / 2, y - 0.46, count, ha="center", va="top",
                fontsize=9, color=color, family="monospace")

    # --- 위: 표준 컨볼루션 ---
    ytop = 1.75
    x = tensor(0.0, ytop, DS_CIN, f"{DS_CIN} x {H} x {H}", INK, FILL)
    op(x, ytop, 3.4, "Standard conv",
       f"{DS_COUT} x {DS_CIN} x {DS_K} x {DS_K}",
       f"{n['std_w']:,} weights", INK)
    tensor(x + 3.4, ytop, DS_COUT, f"{DS_COUT} x {H} x {H}", INK, FILL)
    caption(ax, -0.15, ytop, "standard", fontsize=10, weight="bold",
            color=INK, ha="right")

    # --- 아래: 깊이별 + 점별 ---
    ybot = -1.35
    x = tensor(0.0, ybot, DS_CIN, f"{DS_CIN} x {H} x {H}", ACCENT, FILL)
    op(x, ybot, 1.7, "Depthwise 3x3", f"{DS_CIN} x 1 x {DS_K} x {DS_K}",
       f"{n['dw_w']:,}", ACCENT)
    x = tensor(x + 1.7, ybot, DS_CIN, f"{DS_CIN} x {H} x {H}", ACCENT, FILL)
    op(x, ybot, 1.7, "Pointwise 1x1", f"{DS_COUT} x {DS_CIN} x 1 x 1",
       f"{n['pw_w']:,}", ACCENT)
    tensor(x + 1.7, ybot, DS_COUT, f"{DS_COUT} x {H} x {H}", ACCENT, FILL)
    caption(ax, -0.15, ybot, "depthwise\nseparable", fontsize=10,
            weight="bold", color=ACCENT, ha="right")
    caption(ax, 2.63, ybot - 1.05,
            f"{n['dw_w']:,} + {n['pw_w']:,} = {n['sep_w']:,} weights",
            fontsize=9.5, family="monospace", color=ACCENT)

    # --- 비 ---
    caption(ax, 2.63, 0.38,
            f"{n['std_w']:,} / {n['sep_w']:,} = {ratio_w:.2f}x",
            fontsize=12.5, family="monospace", color=ACCENT, weight="bold")
    caption(ax, 2.63, -0.06,
            f"1 / (1/{DS_COUT} + 1/{DS_K}^2) = {ratio_w:.2f}",
            fontsize=9, family="monospace", color=MUTED)

    caption(ax, 2.63, ybot - 1.52,
            "weights only (bias=False, as in section 4). With PyTorch's default "
            f"bias=True the layer totals are\n{n['std_w'] + n['std_b']:,} and "
            f"{n['sep_w'] + n['sep_b']:,}, and the ratio falls to "
            f"{ratio_p:.2f}x - the {n['sep_b']} extra biases do not shrink.",
            fontsize=8.5, color=MUTED, ha="center", va="top")

    ax.set_xlim(-1.6, 5.5)
    ax.set_ylim(-3.6, 3.0)
    ax.set_axis_off()
    fig.tight_layout()
    save(fig, "depthwise_decomposition")


# === 그림 7. 열네 에포크가 어디서 멈추는가 =========================
#     07_cifar10_advanced.md 2.3절의 표를 그대로 옮긴 것이다.
#     (그 표 자체가 1절 출력의 "Epoch k: Avg Loss ..., Acc ..." 줄이다)

EPOCH_LOG = [
    # (에포크, 평균 로스, 학습 정확도 %)
    (1, 2.1525, 20.22),
    (2, 1.7721, 36.20),
    (3, 1.5800, 42.64),
    (4, 1.4875, 46.11),
    (5, 1.4361, 47.99),
    (6, 1.4026, 49.36),
    (7, 1.3820, 50.19),
    (8, 1.3681, 50.84),
    (9, 1.3520, 51.31),
    (10, 1.3478, 51.21),
    (11, 1.3417, 51.75),
    (12, 1.3349, 52.03),
    (13, 1.3348, 51.99),
    (14, 1.3331, 51.99),
]
LR0, GAMMA = 0.01, 0.7          # 1절의 --lr 과 --gamma 기본값
TEST_ACC = 54.82                # 1절 마지막 줄
PLATEAU_FROM = 9                # 2.3절이 짚는 자리


def make_advanced_training_curve():
    ep = [e for e, _, _ in EPOCH_LOG]
    loss = [l for _, l, _ in EPOCH_LOG]
    acc = [a for _, _, a in EPOCH_LOG]
    lr = [LR0 * GAMMA ** (e - 1) for e in ep]

    tail = [a for e, _, a in EPOCH_LOG if e >= PLATEAU_FROM]
    lo, hi = min(tail), max(tail)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.0, 6.2), sharex=True,
                                   gridspec_kw={"height_ratios": [1, 1.25]})

    for ax in (ax1, ax2):
        ax.axvspan(PLATEAU_FROM - 0.3, 14.3, color=FILL, zorder=0)

    # --- 위: 평균 로스 ---
    ax1.plot(ep, loss, color=ACCENT, lw=1.8, marker="o", ms=4.5, zorder=3)
    ax1.set_ylabel("average training loss", fontsize=9, color=INK)
    ax1.set_ylim(1.2, 2.3)
    ax1.grid(alpha=0.25, lw=0.6, zorder=1)
    ax1.annotate(f"{loss[0]:.4f} -> {loss[-1]:.4f}\nbut {loss[8]:.4f} -> "
                 f"{loss[-1]:.4f} over the last six",
                 xy=(13, loss[12]), xytext=(7.0, 1.78), fontsize=8,
                 color=ACCENT, ha="left", va="center",
                 arrowprops=dict(arrowstyle="-", color=ACCENT, lw=0.8))
    style(ax1)

    # --- 아래: 학습 정확도와 러닝레이트 ---
    ax2.plot(ep, acc, color=COOL, lw=1.8, marker="o", ms=4.5, zorder=4,
             label="train accuracy")
    ax2.hlines([lo, hi], PLATEAU_FROM - 0.3, 14.3, color=COOL, lw=0.8,
               ls=":", zorder=2)
    ax2.set_ylabel("train accuracy (%)", fontsize=9, color=COOL)
    ax2.set_xlabel("epoch", fontsize=9, color=INK)
    ax2.set_ylim(15, 60)
    ax2.set_xticks(ep)
    ax2.set_xlim(0.4, 14.6)
    ax2.grid(alpha=0.25, lw=0.6, zorder=1)
    style(ax2)
    ax2.tick_params(axis="y", colors=COOL)

    ax2.annotate(f"+{acc[1] - acc[0]:.2f}%p", xy=(2, acc[1]),
                 xytext=(2.3, 28.5), fontsize=8.5, color=COOL, ha="left",
                 va="center",
                 arrowprops=dict(arrowstyle="-", color=COOL, lw=0.8))
    ax2.annotate(f"epochs {PLATEAU_FROM}-14 stay inside a "
                 f"{hi - lo:.2f}%p band\n({lo:.2f}% to {hi:.2f}%)",
                 xy=(11.5, hi), xytext=(6.1, 40.5), fontsize=8.5, color=COOL,
                 ha="left", va="center",
                 arrowprops=dict(arrowstyle="-", color=COOL, lw=0.8))

    ax3 = ax2.twinx()
    ax3.plot(ep, lr, color=MUTED, lw=1.5, ls="--", marker="s", ms=3.4,
             zorder=3)
    ax3.set_yscale("log")
    ax3.set_ylabel("learning rate (log scale)", fontsize=9, color=MUTED)
    ax3.set_ylim(5e-5, 3e-2)
    ax3.tick_params(axis="y", labelsize=8, colors=MUTED)
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_color(MUTED)
    ax3.spines["right"].set_linewidth(0.7)
    ax3.annotate(f"lr = {LR0} x {GAMMA}^(k-1), stepped every epoch\n"
                 f"epoch 9: {lr[8]:.6f}   epoch 14: {lr[13]:.8f}",
                 xy=(9, lr[8]), xytext=(3.3, 1.25e-4), fontsize=8,
                 color=MUTED, ha="left", va="center",
                 arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))

    ax1.set_title(f"CNN_CIFAR10, seed 1, 14 epochs - test accuracy "
                  f"{TEST_ACC}% at the end", fontsize=9.5, color=INK,
                  loc="left")

    fig.tight_layout()
    save(fig, "advanced_training_curve")


# === 그림에 적은 수를 다시 확인한다 ===============================


def check_numbers():
    out = correlate(INPUT_5x5, KERNEL_3x3)
    assert len(out) == 3 and len(out[0]) == 3, out
    assert out[0][0] == -4, out
    assert out == [[-4, 2, 4], [-3, -2, 2], [0, 0, 1]], out

    # 전치 컨볼루션: i=2, k=3, s=2, p=1
    n = _zero_inserted(2, 2)
    assert n == 3
    pad = 3 - 1 - 1
    assert pad == 1
    padded = n + 2 * pad
    assert padded == 5
    assert padded - 3 + 1 == 3                      # output_padding = 0
    assert (padded + 1) - 3 + 1 == 4                # output_padding = 1
    # 쪽의 일반 공식과도 맞는지
    for op, want in ((0, 3), (1, 4)):
        assert (2 - 1) * 2 - 2 * 1 + 1 * (3 - 1) + op + 1 == want

    # 리셉티브 필드 점화식 r_l = r_{l-1} + (K-1) d_l
    rf, got = 1, []
    for d in (1, 2, 4, 8):
        rf += 2 * d
        got.append(rf)
    assert got == [3, 7, 15, 31], got

    # 덮개
    assert len(reachable((2, 2))) == 5
    assert reachable((2, 2))[-1] - reachable((2, 2))[0] + 1 == 9
    assert len(reachable((1, 2, 4))) == 15
    assert reachable((1, 2, 4))[-1] - reachable((1, 2, 4))[0] + 1 == 15

    # --- rf_strategies: receptive_field.md 가 찍은 값과 맞는지 ---
    # 3절 compare_receptive_fields() 출력: 표준 5레이어 11, 팽창 5레이어 63
    assert compute_receptive_field(_deeper(5))[0] == 11
    assert compute_receptive_field(_dilated(5))[0] == 63
    assert round(63 / 11, 1) == 5.7
    # 연습문제 3: 표준 10레이어 21
    assert compute_receptive_field(_deeper(10))[0] == 21
    # 2절의 닫힌 식 r_L = 1 + L(k-1) 과도 맞는다
    for L in range(11):
        assert compute_receptive_field(_deeper(L))[0] == 1 + 2 * L
        assert compute_receptive_field(_wider(L))[0] == 1 + 6 * L
        assert compute_receptive_field(_dilated(L))[0] == 1 + 2 * (2 ** L - 1)
    # 6절 compare_architectures() 의 파라메터 대용치
    assert weight_proxy(_dilated(5)) == 45

    # --- depthwise_decomposition: depthwise_separable.md 4·8절 ---
    n = ds_counts()
    assert (n["std_w"], n["dw_w"], n["pw_w"]) == (73728, 576, 8192)
    assert n["sep_w"] == 8768                       # 4절 출력
    assert round(n["std_w"] / n["sep_w"], 2) == 8.41
    assert n["std_w"] + n["std_b"] == 73856         # 8절 출력
    assert n["sep_w"] + n["sep_b"] == 8960          # 8절 출력
    assert round((n["std_w"] + n["std_b"]) / (n["sep_w"] + n["sep_b"]), 2) == 8.24
    assert round(1 / (1 / 128 + 1 / 9), 2) == 8.41  # 4절의 닫힌 식

    # --- advanced_training_curve: 07_cifar10_advanced.md 2.3절 ---
    assert len(EPOCH_LOG) == 14
    tail = [a for e, _, a in EPOCH_LOG if e >= PLATEAU_FROM]
    assert (min(tail), max(tail)) == (51.21, 52.03)
    assert round(max(tail) - min(tail), 2) == 0.82
    assert round(EPOCH_LOG[1][2] - EPOCH_LOG[0][2], 2) == 15.98
    assert round(LR0 * GAMMA ** 8, 7) == 0.0005765
    assert f"{LR0 * GAMMA ** 13:.8f}" == "0.00009689"
    print("numbers check out")


if __name__ == "__main__":
    check_numbers()
    make_convolution_sliding()
    make_transposed_conv_steps()
    make_dilated_vs_downsample()
    make_gridding_coverage()
    make_rf_strategies()
    make_depthwise_decomposition()
    make_advanced_training_curve()
