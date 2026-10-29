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
  mnist_balance_and_scale.svg 01_mnist_dataset.md 4·5절 — 부류 균형과 두 눈금
  fashion_class_means.svg     02_fashion_mnist_dataset.md 연습문제 2 — 밝기 차례
  cifar10_class_colour.svg    03_cifar10_dataset.md 연습문제 3 — 부류별 채널 평균
  fashion_classifier_errors.svg 05_fashion_mnist_classifier.md 2.1절 — 오류의 자리
  binary_pair_curves.svg      08_binary_classification.md 4절 — 45개 쌍의 학습 곡선
  feature_map_depths.svg      feature_maps.md 5절 — 세 깊이의 피처 맵
  conv1d_sliding.svg          conv1d.md 1절 — 시퀀스 위를 미끄러지는 커널
  cifar10_basic_per_class.svg 06_cifar10_basic.md 2.2절 — 부류별 재현율

그림에 적는 수는 모두 쪽의 코드·출력에서 확인한 것이다. check_numbers() 가
그 확인을 다시 돌린다.

실행: python docs/ch10/cnn/figures/make_figures.py
      python docs/ch10/cnn/figures/make_figures.py --png   ← 눈으로 볼 PNG 도 함께
      python docs/ch10/cnn/figures/make_figures.py rf_strategies   ← 그 하나만

이름을 주지 않으면 모두 다시 그린다. 내용이 한 점도 바뀌지 않아도 SVG 안의
타임스탬프와 클립 경로 id 는 바뀌므로, 손댈 그림의 이름만 주는 편이 낫다.
데이터셋에서 바로 재는 그림(mnist_·fashion_·cifar10_class_)은 저장소 뿌리의
./data 에 MNIST·FashionMNIST·CIFAR-10 이 이미 있어야 한다. 내려받지 않는다.
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


# === 데이터셋 쪽 그림의 공통 도구 =================================
#     01·02·03 쪽의 그림은 지어낸 수가 아니라 저장소 뿌리의 ./data 에 있는
#     진짜 데이터셋에서 바로 잰다. 쪽의 코드가 읽는 바로 그 경로다.
#     잰 값이 쪽에 실린 수와 어긋나면 그리기 전에 assert 가 먼저 걸린다.

DATA = HERE.parents[3] / "data"

CH_R, CH_G, CH_B = "#b3453a", "#4a8a4a", "#3a6fb3"      # RGB 세 채널


def load_train_set(name):
    """쪽의 코드와 같은 학습 집합을 읽는다. 내려받지는 않는다."""
    from torchvision import datasets

    return getattr(datasets, name)(root=str(DATA), train=True, download=False)


def image_axes(ax):
    """눈금은 지우되 축은 남긴다 — 맨 왼쪽 칸에 이름을 붙여야 한다."""
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def row_label(ax, text):
    ax.set_ylabel(text, fontsize=8.5, color=INK, rotation=0,
                  ha="right", va="center", labelpad=6)


# === 그림. MNIST: 고르기는 해도 똑같지는 않다, 그리고 눈금 ========
#     01_mnist_dataset.md 4·5절과 연습문제 1·4.
#     막대의 수는 4절 출력의 부류 분포 표, 평균 0.1307 과 표준편차 0.3081 은
#     연습문제 4 의 값이다. 모두 학습셋에서 다시 재어 맞는지 확인한다.

MNIST_COUNTS = (5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949)
MNIST_RAW_MEAN, MNIST_RAW_STD = 0.1307, 0.3081


def make_mnist_balance_and_scale():
    import numpy as np
    import torch

    ds = load_train_set("MNIST")
    x = ds.data.float() / 255.0            # ToTensor() 와 같은 [0, 1] 눈금
    counts = tuple(torch.bincount(ds.targets, minlength=10).tolist())
    assert counts == MNIST_COUNTS, counts
    assert f"{x.mean():.4f}" == f"{MNIST_RAW_MEAN:.4f}", x.mean()
    assert f"{x.std(unbiased=False):.4f}" == f"{MNIST_RAW_STD:.4f}", x.std()

    total = sum(counts)
    assert total == 60000
    zero_share = 100.0 * (x == 0).double().mean().item()

    # 부류마다 학습셋에서 처음 나오는 한 장. 난수를 쓰지 않으므로 씨앗도 없다.
    first = [int((ds.targets == d).nonzero()[0, 0]) for d in range(10)]

    nbins = 80
    hist = torch.histc(x, bins=nbins, min=0.0, max=1.0).numpy()
    raw_centers = (np.arange(nbins) + 0.5) / nbins
    centers = 2 * raw_centers - 1          # 같은 막대를 [-1, 1] 눈금에 얹는다
    mean_n = 2 * MNIST_RAW_MEAN - 1
    std_n = 2 * MNIST_RAW_STD

    fig = plt.figure(figsize=(9.6, 7.6))
    gs = fig.add_gridspec(3, 10, height_ratios=[0.55, 1.25, 1.75],
                          hspace=0.62, wspace=0.1,
                          left=0.105, right=0.98, top=0.95, bottom=0.085)

    # --- 위: 부류마다 한 장 ---
    for d in range(10):
        ax = fig.add_subplot(gs[0, d])
        ax.imshow(x[first[d]].numpy(), cmap="gray", vmin=0, vmax=1)
        image_axes(ax)
        if d == 0:
            row_label(ax, "one image")

    # --- 가운데: 부류별 장 수 ---
    axb = fig.add_subplot(gs[1, :])
    colors = [ACCENT if c == max(counts) else COOL if c == min(counts) else FILL
              for c in counts]
    axb.bar(range(10), counts, color=colors, edgecolor=INK, lw=0.7, width=0.72,
            zorder=3)
    axb.axhline(total / 10, color=MUTED, ls="--", lw=0.9, zorder=2)
    for d, c in enumerate(counts):
        axb.text(d, c + 110, f"{c:,}", fontsize=7.5, color=INK, ha="center",
                 va="bottom")
    axb.set_ylim(0, 7900)
    axb.set_xlim(-0.62, 9.62)
    axb.set_xticks(range(10))
    axb.set_ylabel("training images", fontsize=9, color=INK)
    axb.set_xlabel("digit", fontsize=9, color=INK)
    axb.set_title(f"digit 1 has {max(counts):,} images (11.24%) and digit 5 has "
                  f"{min(counts):,} (9.04%) - a spread of "
                  f"{max(counts) - min(counts):,}\n"
                  f"dashed line: {total // 10:,} = exactly balanced",
                  fontsize=9, color=INK, loc="left")
    style(axb)

    # --- 아래: 같은 화소를 두 자로 잰다 ---
    axh = fig.add_subplot(gs[2, :])
    axh.fill_between(centers, 1.0, hist, step="mid", color=FILL_COOL, zorder=2)
    axh.step(centers, hist, where="mid", color=COOL, lw=1.1, zorder=3)
    axh.set_yscale("log")
    axh.set_ylim(3e2, 2e8)
    axh.set_xlim(-1.03, 1.03)
    axh.axvline(mean_n, color=ACCENT, lw=1.4, zorder=4)
    axh.axvline(0.0, color=MUTED, lw=1.0, ls=":", zorder=4)
    axh.text(mean_n - 0.022, 3.0e3, f"mean {mean_n:.4f}", fontsize=8.5,
             color=ACCENT, ha="right", va="bottom", rotation=90)
    axh.text(0.03, 1.1e3, "0 on the normalized scale", fontsize=8,
             color=MUTED, ha="left", va="bottom")
    axh.text(0.995, 0.95,
             f"the same {total * 784:,} pixels, counted once and labelled "
             f"twice.\nNormalize((0.5,), (0.5,)) changes the ruler, not the "
             f"shape:\nmean {MNIST_RAW_MEAN} -> {mean_n:.4f},  "
             f"std {MNIST_RAW_STD} -> {std_n:.4f}.\n"
             f"Neither is the standardized mean 0, std 1.",
             transform=axh.transAxes, fontsize=8.5, color=INK, ha="right",
             va="top")
    axh.annotate(f"{zero_share:.1f}% of all pixels are exactly background",
                 xy=(-0.99, 3.4e7), xytext=(-0.70, 5.5e6), fontsize=8,
                 color=COOL, ha="left", va="center",
                 arrowprops=dict(arrowstyle="-", color=COOL, lw=0.8))
    axh.set_ylabel("pixels (log scale)", fontsize=9, color=INK)
    axh.set_xlabel("pixel value after Normalize((0.5,), (0.5,)):  [-1, 1]",
                   fontsize=9, color=INK)
    style(axh)
    sec = axh.secondary_xaxis("top", functions=(lambda v: v / 2 + 0.5,
                                                lambda u: 2 * u - 1))
    sec.set_xlabel("raw pixel value after ToTensor():  [0, 1]", fontsize=9,
                   color=INK)
    sec.tick_params(labelsize=8, colors=INK)
    sec.spines["top"].set_color(INK)
    sec.spines["top"].set_linewidth(0.7)

    save(fig, "mnist_balance_and_scale")


# === 그림. Fashion-MNIST: 밝기 차례와 평균 이미지 =================
#     02_fashion_mnist_dataset.md 연습문제 2 의 표를 그림으로 읽는다.
#     부류마다 6,000장을 평균한 이미지를 밝은 쪽부터 늘어놓았다.

FASHION_NAMES = ("T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
                 "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot")
FASHION_MEAN_NORM = (-0.3488, -0.5542, -0.2466, -0.4822, -0.2293,
                     -0.7265, -0.3364, -0.6646, -0.2929, -0.3976)


def make_fashion_class_means():
    import torch

    ds = load_train_set("FashionMNIST")
    x = ds.data.float() / 255.0
    y = ds.targets
    counts = torch.bincount(y, minlength=10).tolist()
    assert counts == [6000] * 10, counts

    means = []
    mean_imgs = []
    for c in range(10):
        sel = x[y == c]
        mean_imgs.append(sel.mean(dim=0).numpy())
        means.append((sel.mean() * 2 - 1).item())        # [-1, 1] 눈금
    for c, v in enumerate(means):
        assert f"{v:.4f}" == f"{FASHION_MEAN_NORM[c]:.4f}", (c, v)

    first = [int((y == c).nonzero()[0, 0]) for c in range(10)]
    order = sorted(range(10), key=lambda c: -means[c])   # 밝은 쪽부터

    base = -0.80
    fig = plt.figure(figsize=(9.8, 6.9))
    gs = fig.add_gridspec(3, 10, height_ratios=[0.9, 0.9, 1.75],
                          hspace=0.16, wspace=0.1,
                          left=0.125, right=0.98, top=0.86, bottom=0.185)

    for k, c in enumerate(order):
        ax = fig.add_subplot(gs[0, k])
        ax.imshow(x[first[c]].numpy(), cmap="gray", vmin=0, vmax=1)
        image_axes(ax)
        if k == 0:
            row_label(ax, "one image")

        ax = fig.add_subplot(gs[1, k])
        ax.imshow(mean_imgs[c], cmap="gray", vmin=0, vmax=1)
        image_axes(ax)
        if k == 0:
            row_label(ax, "mean of 6,000")

    axb = fig.add_subplot(gs[2, :])
    vals = [means[c] for c in order]
    axb.bar(range(10), [v - base for v in vals], bottom=base, width=0.72,
            color=FILL, edgecolor=INK, lw=0.7, zorder=3)
    for k, v in enumerate(vals):
        axb.text(k, v + 0.012, f"{v:.4f}", fontsize=7.5, color=INK,
                 ha="center", va="bottom")
    axb.set_ylim(base, -0.13)
    axb.set_xlim(-0.5, 9.5)                  # 위 두 줄의 칸과 자리를 맞춘다
    axb.set_xticks(range(10))
    axb.set_xticklabels([FASHION_NAMES[c] for c in order], fontsize=8,
                        rotation=30, ha="right")
    axb.set_ylabel("class mean pixel value,\nthe [-1, 1] scale the loader "
                   "returns", fontsize=9, color=INK)
    axb.grid(axis="y", alpha=0.25, lw=0.6, zorder=1)
    style(axb)

    fig.suptitle("Fashion-MNIST: every class has exactly 6,000 training "
                 "images, ordered brightest to darkest\nbrighter means the "
                 "garment covers more of the 28x28 frame - Trouser is dark "
                 "because it is narrow, not because it is small",
                 fontsize=9.5, color=INK, x=0.105, ha="left", y=0.98,
                 va="top")

    save(fig, "fashion_class_means")


# === 그림. CIFAR-10: 부류마다 다른 색 ============================
#     03_cifar10_dataset.md 연습문제 3 의 두 표(데이터셋 채널 평균과
#     부류별 채널 평균)를 그림으로 읽는다. 차례도 그 표와 같다.

CIFAR_NAMES = ("airplane", "automobile", "bird", "cat", "deer", "dog",
               "frog", "horse", "ship", "truck")
CIFAR_SET_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_CLASS_RGB = {                      # 연습문제 3 의 표, [0, 1] 눈금
    "airplane": (0.5257, 0.5603, 0.5889),
    "ship": (0.4902, 0.5254, 0.5547),
    "truck": (0.4987, 0.4853, 0.4781),
    "bird": (0.4893, 0.4915, 0.4240),
    "horse": (0.5020, 0.4799, 0.4169),
    "dog": (0.4999, 0.4646, 0.4165),
    "automobile": (0.4712, 0.4545, 0.4472),
    "cat": (0.4955, 0.4564, 0.4155),
    "deer": (0.4716, 0.4652, 0.3782),
    "frog": (0.4701, 0.4384, 0.3452),
}


def make_cifar10_class_colour():
    import torch

    ds = load_train_set("CIFAR10")
    x = torch.tensor(ds.data).float() / 255.0        # (50000, 32, 32, 3)
    y = torch.tensor(ds.targets)
    counts = torch.bincount(y, minlength=10).tolist()
    assert counts == [5000] * 10, counts
    set_mean = x.mean(dim=(0, 1, 2))
    for ch in range(3):
        assert f"{set_mean[ch]:.4f}" == f"{CIFAR_SET_MEAN[ch]:.4f}", set_mean

    rgb = {}
    mean_imgs = {}
    for c in range(10):
        sel = x[y == c]
        mean_imgs[CIFAR_NAMES[c]] = sel.mean(dim=0).numpy()
        m = sel.mean(dim=(0, 1, 2))
        rgb[CIFAR_NAMES[c]] = tuple(round(float(v), 4) for v in m)
        for ch in range(3):
            assert f"{m[ch]:.4f}" == f"{CIFAR_CLASS_RGB[CIFAR_NAMES[c]][ch]:.4f}", \
                (CIFAR_NAMES[c], m)

    first = {CIFAR_NAMES[c]: int((y == c).nonzero()[0, 0]) for c in range(10)}
    order = sorted(CIFAR_NAMES, key=lambda n: -sum(rgb[n]) / 3)
    assert order[0] == "airplane" and order[-1] == "frog", order

    fig = plt.figure(figsize=(10.2, 6.9))
    gs = fig.add_gridspec(3, 10, height_ratios=[0.9, 0.9, 1.9],
                          hspace=0.16, wspace=0.1,
                          left=0.1, right=0.86, top=0.87, bottom=0.175)

    for k, name in enumerate(order):
        ax = fig.add_subplot(gs[0, k])
        ax.imshow(x[first[name]].numpy())
        image_axes(ax)
        if k == 0:
            row_label(ax, "one image")

        ax = fig.add_subplot(gs[1, k])
        ax.imshow(mean_imgs[name].clip(0, 1))
        image_axes(ax)
        if k == 0:
            row_label(ax, "mean of 5,000")

    axb = fig.add_subplot(gs[2, :])
    w = 0.26
    for ch, (color, label) in enumerate(((CH_R, "R"), (CH_G, "G"),
                                         (CH_B, "B"))):
        axb.bar([k + (ch - 1) * w for k in range(10)],
                [rgb[n][ch] for n in order], width=w, color=color,
                edgecolor="none", zorder=3, label=label)
        axb.axhline(CIFAR_SET_MEAN[ch], color=color, ls="--", lw=0.8,
                    alpha=0.8, zorder=2)
        axb.text(9.62, CIFAR_SET_MEAN[ch], f" dataset {label} "
                 f"{CIFAR_SET_MEAN[ch]:.4f}", fontsize=7.5, color=color,
                 ha="left", va="center")
    axb.set_ylim(0.32, 0.615)
    axb.set_xlim(-0.5, 9.5)                  # 위 두 줄의 칸과 자리를 맞춘다
    axb.set_xticks(range(10))
    axb.set_xticklabels(order, fontsize=8, rotation=30, ha="right")
    axb.set_ylabel("class mean per channel, [0, 1] scale", fontsize=9,
                   color=INK)
    axb.grid(axis="y", alpha=0.25, lw=0.6, zorder=1)
    style(axb)
    for k, name in enumerate(order):
        if rgb[name][2] > rgb[name][0]:
            axb.text(k, max(rgb[name]) + 0.008, "B > R", fontsize=8,
                     color=CH_B, ha="center", va="bottom")
    axb.legend(loc="upper right", fontsize=8, frameon=False, ncol=3,
               bbox_to_anchor=(1.0, 1.02))

    fig.suptitle("CIFAR-10: 5,000 training images per class, ordered by "
                 "overall brightness\nthe mean of 5,000 keeps only what the "
                 "class has in common - sky for airplane, sea for ship",
                 fontsize=9.5, color=INK, x=0.1, ha="left", y=0.98, va="top")

    save(fig, "cifar10_class_colour")


# === 그림. 틀린 1,416건은 어디에 있는가 ===========================
#     05_fashion_mnist_classifier.md 2.1절. 부류별 정확도 열 줄과
#     셔츠의 451건 갈래는 모두 그 절에 실린 수다.

FMC_CLASS_ACC = (("Shirt", 54.90), ("Pullover", 74.00), ("T-shirt/top", 82.70),
                ("Coat", 82.80), ("Dress", 87.80), ("Sneaker", 93.80),
                ("Sandal", 94.00), ("Ankle boot", 95.00), ("Bag", 96.40),
                ("Trouser", 97.00))
FMC_SLEEVED = ("Shirt", "Pullover", "T-shirt/top", "Coat")
FMC_SHIRT_ERRORS = (("T-shirt/top", 185), ("Coat", 123), ("Pullover", 92),
                   ("the other six", 51))
FMC_TEST_CORRECT, FMC_TEST_TOTAL = 8584, 10000
FMC_ERRORS_AMONG_TOPS = 878


def make_fashion_classifier_errors():
    errors = FMC_TEST_TOTAL - FMC_TEST_CORRECT
    per_class = [round((100.0 - a) / 100.0 * 1000) for _, a in FMC_CLASS_ACC]
    assert sum(per_class) == errors, (sum(per_class), errors)
    shirt_errors = round((100.0 - FMC_CLASS_ACC[0][1]) / 100.0 * 1000)
    assert sum(v for _, v in FMC_SHIRT_ERRORS) == shirt_errors

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 4.5),
                                   gridspec_kw={"width_ratios": [1.45, 1]})

    # --- 왼쪽: 부류별 테스트 정확도 ---
    names = [n for n, _ in FMC_CLASS_ACC][::-1]
    accs = [a for _, a in FMC_CLASS_ACC][::-1]
    colors = [ACCENT if n in FMC_SLEEVED else COOL for n in names]
    ax1.barh(range(10), accs, color=colors, edgecolor="none", height=0.68,
             zorder=3)
    for k, (n, a) in enumerate(zip(names, accs)):
        ax1.text(a + 0.7, k, f"{a:.2f}%", fontsize=8, color=INK, ha="left",
                 va="center")
    ax1.axvline(100.0 * FMC_TEST_CORRECT / FMC_TEST_TOTAL, color=MUTED, ls="--",
                lw=0.9, zorder=2)
    ax1.text(100.0 * FMC_TEST_CORRECT / FMC_TEST_TOTAL - 0.8, -0.75,
             "overall 85.84%", fontsize=8, color=MUTED, ha="right", va="center")
    ax1.set_yticks(range(10))
    ax1.set_yticklabels(names, fontsize=8.5)
    ax1.set_xlim(50, 104)
    ax1.set_ylim(-1.1, 9.8)
    ax1.set_xlabel("test accuracy (%), 1,000 images per class", fontsize=9,
                   color=INK)
    ax1.grid(axis="x", alpha=0.25, lw=0.6, zorder=1)
    style(ax1)
    ax1.set_title(f"the four worst classes are the four sleeved upper-body "
                  f"garments;\n{FMC_ERRORS_AMONG_TOPS} of the {errors:,} errors "
                  f"are confusions among just those four",
                  fontsize=9, color=INK, loc="left")

    # --- 오른쪽: 셔츠가 틀린 451건의 갈래 ---
    labels = [n for n, _ in FMC_SHIRT_ERRORS][::-1]
    vals = [v for _, v in FMC_SHIRT_ERRORS][::-1]
    cols = [MUTED if n == "the other six" else ACCENT for n in labels]
    ax2.barh(range(4), vals, color=cols, edgecolor="none", height=0.6,
             zorder=3)
    for k, v in enumerate(vals):
        ax2.text(v + 4, k, f"{v}", fontsize=8.5, color=INK, ha="left",
                 va="center")
    ax2.set_yticks(range(4))
    ax2.set_yticklabels(labels, fontsize=8.5)
    ax2.set_xlim(0, 215)
    ax2.set_ylim(-0.9, 3.8)
    ax2.set_xlabel("misclassified Shirt images", fontsize=9, color=INK)
    ax2.grid(axis="x", alpha=0.25, lw=0.6, zorder=1)
    style(ax2)
    share = 100.0 * sum(v for _, v in FMC_SHIRT_ERRORS[:3]) / shirt_errors
    ax2.set_title(f"Shirt gets {FMC_CLASS_ACC[0][1]:.2f}% right; of its "
                  f"{shirt_errors} errors\n"
                  f"{sum(v for _, v in FMC_SHIRT_ERRORS[:3])} "
                  f"({share:.1f}%) land on another sleeved top",
                  fontsize=9, color=INK, loc="left")

    fig.tight_layout()
    save(fig, "fashion_classifier_errors")


# === 그림. 45개 쌍의 학습 곡선 =====================================
#     08_binary_classification.md 4절이 그리는 10x10 격자다. 곡선의 값은
#     그 쪽 출력 블록의 "loss ..." 줄 45개를 그대로 옮긴 것이고, 로그 축과
#     1e-6..1 범위도 그 절의 코드가 정한 그대로다.

BINARY_TRACES = {
    (0, 1): (0.032393, 0.003089, 0.003016, 0.001516, 0.002052, 0.001460, 0.000663, 0.000379, 0.000167, 0.000173),
    (0, 2): (0.078485, 0.013402, 0.008397, 0.004665, 0.002318, 0.001945, 0.001911, 0.001372, 0.001321, 0.000380),
    (0, 3): (0.070485, 0.007552, 0.004139, 0.002698, 0.001764, 0.001243, 0.001368, 0.000467, 0.000150, 0.000060),
    (0, 4): (0.047532, 0.007825, 0.004021, 0.002596, 0.002723, 0.000973, 0.000249, 0.000064, 0.000042, 0.000031),
    (0, 5): (0.073260, 0.012755, 0.006494, 0.006198, 0.003700, 0.002488, 0.001917, 0.000809, 0.000318, 0.000120),
    (0, 6): (0.061560, 0.013753, 0.010517, 0.005914, 0.006262, 0.004419, 0.003179, 0.003779, 0.005392, 0.002538),
    (0, 7): (0.052641, 0.005469, 0.002517, 0.001655, 0.000467, 0.000182, 0.000078, 0.000056, 0.000038, 0.000027),
    (0, 8): (0.074121, 0.014819, 0.008427, 0.007036, 0.006231, 0.002896, 0.002482, 0.001359, 0.000388, 0.000331),
    (0, 9): (0.058303, 0.012442, 0.008701, 0.005904, 0.003238, 0.003320, 0.001578, 0.001470, 0.000560, 0.000282),
    (1, 2): (0.068073, 0.017471, 0.011291, 0.007909, 0.006300, 0.004940, 0.004322, 0.003362, 0.003808, 0.002363),
    (1, 3): (0.071319, 0.010101, 0.005260, 0.001650, 0.001511, 0.000474, 0.000316, 0.001043, 0.000716, 0.000190),
    (1, 4): (0.057436, 0.009861, 0.007417, 0.005261, 0.003698, 0.002675, 0.002744, 0.002641, 0.001686, 0.000841),
    (1, 5): (0.054885, 0.007360, 0.002723, 0.001425, 0.000732, 0.000257, 0.000208, 0.000204, 0.000068, 0.000039),
    (1, 6): (0.049770, 0.004099, 0.002481, 0.001179, 0.001512, 0.002893, 0.000254, 0.000607, 0.000178, 0.000109),
    (1, 7): (0.054830, 0.013194, 0.008910, 0.006940, 0.005725, 0.006865, 0.003627, 0.005104, 0.003393, 0.002791),
    (1, 8): (0.066208, 0.011110, 0.009450, 0.006240, 0.004005, 0.005848, 0.002219, 0.002448, 0.001730, 0.001350),
    (1, 9): (0.054709, 0.010092, 0.007295, 0.004359, 0.003470, 0.002395, 0.001196, 0.001245, 0.000952, 0.000278),
    (2, 3): (0.113350, 0.028250, 0.017437, 0.010322, 0.007409, 0.007396, 0.003248, 0.002638, 0.001094, 0.000447),
    (2, 4): (0.073368, 0.009565, 0.006013, 0.003859, 0.003054, 0.002395, 0.001703, 0.000742, 0.000600, 0.000235),
    (2, 5): (0.085433, 0.007024, 0.002854, 0.002211, 0.000707, 0.000307, 0.000153, 0.000077, 0.000058, 0.000045),
    (2, 6): (0.076312, 0.005438, 0.004194, 0.001228, 0.001365, 0.000855, 0.000173, 0.000081, 0.000055, 0.000041),
    (2, 7): (0.080861, 0.025920, 0.016630, 0.013734, 0.011873, 0.009500, 0.006444, 0.005753, 0.004876, 0.002274),
    (2, 8): (0.098119, 0.023497, 0.015144, 0.009260, 0.006888, 0.005188, 0.002386, 0.003505, 0.002843, 0.002161),
    (2, 9): (0.080824, 0.016877, 0.009033, 0.005474, 0.003561, 0.002405, 0.001732, 0.002453, 0.000231, 0.000115),
    (3, 4): (0.072044, 0.009092, 0.004500, 0.002301, 0.000816, 0.001016, 0.001715, 0.000114, 0.000051, 0.000039),
    (3, 5): (0.144390, 0.026336, 0.019941, 0.013380, 0.009353, 0.008643, 0.006654, 0.006307, 0.006818, 0.005721),
    (3, 6): (0.066734, 0.005208, 0.001998, 0.001815, 0.000295, 0.000133, 0.000112, 0.000055, 0.000037, 0.000027),
    (3, 7): (0.077247, 0.018474, 0.010298, 0.006538, 0.005074, 0.005717, 0.003410, 0.002085, 0.000605, 0.000321),
    (3, 8): (0.124614, 0.022258, 0.009254, 0.006004, 0.006049, 0.002417, 0.002951, 0.001656, 0.003333, 0.000490),
    (3, 9): (0.092637, 0.019176, 0.009591, 0.007668, 0.005028, 0.002828, 0.002053, 0.003086, 0.001437, 0.000527),
    (4, 5): (0.068632, 0.005336, 0.004062, 0.002554, 0.000734, 0.003479, 0.001493, 0.000212, 0.000054, 0.000040),
    (4, 6): (0.064811, 0.008751, 0.005402, 0.003171, 0.003521, 0.001235, 0.002065, 0.001083, 0.000283, 0.000122),
    (4, 7): (0.077894, 0.012509, 0.006790, 0.005120, 0.003373, 0.003196, 0.002933, 0.001995, 0.001194, 0.001113),
    (4, 8): (0.077317, 0.014525, 0.010547, 0.007649, 0.004974, 0.005211, 0.004546, 0.002613, 0.001150, 0.000553),
    (4, 9): (0.183971, 0.029795, 0.021497, 0.017828, 0.015283, 0.013028, 0.011840, 0.010084, 0.011959, 0.008196),
    (5, 6): (0.086559, 0.021631, 0.013369, 0.010868, 0.006288, 0.004952, 0.006677, 0.002231, 0.001246, 0.002496),
    (5, 7): (0.068226, 0.008000, 0.003909, 0.002315, 0.001305, 0.001413, 0.000582, 0.000204, 0.000072, 0.000042),
    (5, 8): (0.116046, 0.024139, 0.015555, 0.012343, 0.009408, 0.009304, 0.006778, 0.004849, 0.003911, 0.004137),
    (5, 9): (0.100244, 0.020203, 0.013908, 0.009291, 0.007315, 0.005887, 0.005138, 0.002646, 0.002069, 0.001494),
    (6, 7): (0.036739, 0.002392, 0.001053, 0.000200, 0.000083, 0.000253, 0.000541, 0.000028, 0.000012, 0.000009),
    (6, 8): (0.097057, 0.018833, 0.012004, 0.007634, 0.006091, 0.005165, 0.004591, 0.002729, 0.001113, 0.000708),
    (6, 9): (0.061716, 0.005549, 0.003213, 0.001560, 0.000886, 0.000486, 0.000101, 0.000058, 0.000046, 0.000040),
    (7, 8): (0.098721, 0.012813, 0.011696, 0.006185, 0.006317, 0.003999, 0.003304, 0.001973, 0.001203, 0.001310),
    (7, 9): (0.133957, 0.025467, 0.017293, 0.012486, 0.009580, 0.007802, 0.006302, 0.005175, 0.004225, 0.001813),
    (8, 9): (0.108190, 0.024777, 0.016567, 0.013194, 0.011417, 0.009197, 0.005796, 0.004172, 0.003013, 0.002990),
}
BINARY_YLIM = (1e-6, 1.0)        # 4절 코드의 set_ylim
BINARY_TICKS = (1e-5, 1e-3, 1e-1)


def _pair_panel(ax, trace, color):
    ax.plot(range(1, len(trace) + 1), trace, color=color, lw=1.2, zorder=3)
    ax.set_yscale("log")
    ax.set_ylim(*BINARY_YLIM)
    ax.set_xlim(0.4, 10.6)
    ax.set_yticks(BINARY_TICKS)
    ax.set_xticks([])
    ax.set_yticklabels([])
    ax.tick_params(length=0)
    ax.grid(axis="y", alpha=0.3, lw=0.5, zorder=1)
    for s in ax.spines.values():
        s.set_color(MUTED)
        s.set_linewidth(0.5)


def make_binary_pair_curves():
    finals = {p: t[-1] for p, t in BINARY_TRACES.items()}
    hardest = max(finals, key=finals.get)
    easiest = min(finals, key=finals.get)
    assert hardest == (4, 9) and easiest == (6, 7)
    rising = sum(1 for t in BINARY_TRACES.values()
                 if any(b > a for a, b in zip(t, t[1:])))

    fig = plt.figure(figsize=(10.6, 12.2))
    gs = fig.add_gridspec(11, 10, height_ratios=[1] * 10 + [2.2],
                          hspace=0.3, wspace=0.18,
                          left=0.075, right=0.975, top=0.9, bottom=0.055)

    for i in range(10):
        for j in range(10):
            if i == j:
                continue
            ax = fig.add_subplot(gs[i, j])
            pair = (min(i, j), max(i, j))
            # 위 삼각형은 빨강, 아래 삼각형은 같은 자취를 파랑으로 되비친 것이다
            _pair_panel(ax, BINARY_TRACES[pair], "#c0392b" if i < j else COOL)
            if pair in (hardest, easiest):
                for s in ax.spines.values():
                    s.set_color(ACCENT)
                    s.set_linewidth(1.4)
            if i == 0:                       # 맨 윗줄에 열 이름
                ax.set_title(str(j), fontsize=10, color=INK, pad=4)
            if j == 0:                       # 맨 왼쪽 줄에 행 이름
                ax.set_ylabel(str(i), fontsize=10, color=INK, rotation=0,
                              ha="right", va="center", labelpad=7)
        # 대각선은 같은 숫자끼리라 문제가 없다. 그 자리에 숫자를 적어 두면
        # 행 이름과 열 이름 노릇을 한꺼번에 한다.
        axd = fig.add_subplot(gs[i, i])
        axd.set_axis_off()
        axd.text(0.5, 0.5, str(i), fontsize=15, color=MUTED, ha="center",
                 va="center", transform=axd.transAxes)

    # --- 아래 왼쪽: 눈금을 읽는 자 겸 양 끝 두 쌍 ---
    axk = fig.add_subplot(gs[10, 0:5])
    axk.plot(range(1, 11), BINARY_TRACES[hardest], color="#c0392b", lw=1.6,
             marker="o", ms=3.2,
             label=f"{hardest[0]} vs {hardest[1]} (hardest, "
                   f"{finals[hardest]:.6f})")
    axk.plot(range(1, 11), BINARY_TRACES[easiest], color=COOL, lw=1.6,
             marker="s", ms=3.2,
             label=f"{easiest[0]} vs {easiest[1]} (easiest, "
                   f"{finals[easiest]:.6f})")
    axk.set_yscale("log")
    axk.set_ylim(*BINARY_YLIM)
    axk.set_xlim(0.4, 10.6)
    axk.set_xticks(range(1, 11))
    axk.set_yticks((1e-6, 1e-4, 1e-2, 1.0))
    axk.set_xlabel("epoch", fontsize=9, color=INK)
    axk.set_ylabel("average training loss", fontsize=9, color=INK)
    axk.grid(alpha=0.25, lw=0.6)
    axk.legend(fontsize=8, frameon=False, loc="lower left")
    style(axk)
    axk.set_title("what one panel above holds", fontsize=9, color=INK,
                  loc="left")

    # --- 아래 오른쪽: 45개 쌍의 마지막 로스 ---
    axd = fig.add_subplot(gs[10, 6:10])
    vals = sorted(finals.values())
    axd.plot(vals, [0] * 45, "o", color=COOL, ms=5, alpha=0.6, mec="none")
    for pair, mark, dy in ((hardest, "o", 1), (easiest, "s", 1)):
        axd.plot([finals[pair]], [0], mark, color=ACCENT, ms=6, mec="none")
        axd.annotate(f"{pair[0]} vs {pair[1]}\n{finals[pair]:.6f}",
                     xy=(finals[pair], 0), xytext=(finals[pair], 0.55),
                     fontsize=8, color=ACCENT, ha="center", va="bottom",
                     arrowprops=dict(arrowstyle="-", color=ACCENT, lw=0.8))
    axd.set_xscale("log")
    axd.set_xlim(5e-6, 2e-2)
    axd.set_ylim(-0.6, 1.6)
    axd.set_yticks([])
    axd.set_xlabel("final-epoch loss of all 45 pairs (log scale)", fontsize=9,
                   color=INK)
    axd.spines["left"].set_visible(False)
    axd.grid(axis="x", alpha=0.25, lw=0.6)
    style(axd)
    axd.set_title("every pair is easy, and the spread inside them still runs "
                  "over three orders of magnitude",
                  fontsize=9, color=INK, loc="left")

    fig.suptitle("MNIST binary classification: 45 pairs, 10 epochs each, "
                 "loss on a log axis from 1e-6 to 1\n"
                 "row i, column j is the pair i vs j; the lower triangle "
                 f"mirrors the upper one, so 45 runs fill 90 panels; "
                 f"{rising} of the 45 traces rise at least once",
                 fontsize=10, color=INK, x=0.075, ha="left", y=0.975)

    save(fig, "binary_pair_curves")


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

    # --- mnist_balance_and_scale: 01_mnist_dataset.md 4·5절 ---
    assert sum(MNIST_COUNTS) == 60000
    assert f"{100 * max(MNIST_COUNTS) / 60000:.2f}" == "11.24"   # 숫자 1
    assert f"{100 * min(MNIST_COUNTS) / 60000:.2f}" == "9.04"    # 숫자 5
    assert max(MNIST_COUNTS) - min(MNIST_COUNTS) == 1321
    # 연습문제 1 의 되돌리기: 2 x 0.1307 - 1 = -0.7386
    assert f"{2 * MNIST_RAW_MEAN - 1:.4f}" == "-0.7386"
    assert f"{2 * MNIST_RAW_STD:.4f}" == "0.6162"

    # --- fashion_class_means: 02_fashion_mnist_dataset.md 연습문제 2 ---
    assert len(FASHION_NAMES) == len(FASHION_MEAN_NORM) == 10
    bright = sorted(zip(FASHION_MEAN_NORM, FASHION_NAMES), reverse=True)
    assert bright[0] == (-0.2293, "Coat"), bright[0]
    assert bright[-1] == (-0.7265, "Sandal"), bright[-1]

    # --- cifar10_class_colour: 03_cifar10_dataset.md 연습문제 3 ---
    assert set(CIFAR_CLASS_RGB) == set(CIFAR_NAMES)
    blue = [n for n, (r, _, b) in CIFAR_CLASS_RGB.items() if b > r]
    assert sorted(blue) == ["airplane", "ship"], blue
    ranked = sorted(CIFAR_CLASS_RGB, key=lambda n: -sum(CIFAR_CLASS_RGB[n]) / 3)
    assert ranked[0] == "airplane" and ranked[-1] == "frog", ranked

    # --- fashion_classifier_errors: 05_fashion_mnist_classifier.md 2.1절 ---
    assert len(FMC_CLASS_ACC) == 10
    errs = [round((100 - a) / 100 * 1000) for _, a in FMC_CLASS_ACC]
    assert sum(errs) == FMC_TEST_TOTAL - FMC_TEST_CORRECT == 1416
    assert errs[0] == 451                                   # 셔츠
    assert sum(v for _, v in FMC_SHIRT_ERRORS) == 451
    assert sum(v for _, v in FMC_SHIRT_ERRORS[:3]) == 400
    assert f"{100 * 400 / 451:.1f}" == "88.7"
    assert f"{100 * FMC_ERRORS_AMONG_TOPS / 1416:.1f}" == "62.0"
    assert [n for n, _ in FMC_CLASS_ACC[:4]] == list(FMC_SLEEVED)

    # --- binary_pair_curves: 08_binary_classification.md 4·5절 ---
    assert len(BINARY_TRACES) == 45
    assert all(len(t) == 10 for t in BINARY_TRACES.values())
    fin = {p: t[-1] for p, t in BINARY_TRACES.items()}
    assert max(fin, key=fin.get) == (4, 9) and max(fin.values()) == 0.008196
    assert min(fin, key=fin.get) == (6, 7) and min(fin.values()) == 0.000009
    assert sum(1 for t in BINARY_TRACES.values()
               if any(b > a for a, b in zip(t, t[1:]))) == 29
    print("numbers check out")




# === 그림 8. 깊이에 따라 달라지는 피처 맵 ==========================
#     feature_maps.md 5절. 모델은 그 쪽 3절의 named_model 을 글자 그대로
#     다시 세운 것이고, 웨이트는 학습하지 않은 무작위 초기값이다
#     (torch.manual_seed(0)). 입력은 CIFAR-10 시험셋 1번 이미지다.

FM_SEED = 0
FM_INDEX = 1                      # CIFAR-10 test[1] — 'ship'
FM_TAPS = ("relu1_1", "relu2_2", "relu3_2")
FM_NCHAN = 6                      # 깊이마다 앞에서 여섯 채널


def _vgg_feature_extractor():
    """feature_maps.md 3절의 named_model 과 같은 레이어·같은 이름이다."""
    import torch
    import torch.nn as nn

    torch.manual_seed(FM_SEED)
    layers = [
        nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(inplace=True),
        nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True),
        nn.MaxPool2d(2, 2),
        nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True),
        nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(inplace=True),
        nn.MaxPool2d(2, 2),
        nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(inplace=True),
        nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True),
        nn.MaxPool2d(2, 2),
    ]
    names = ["conv1_1", "relu1_1", "conv1_2", "relu1_2", "pool1",
             "conv2_1", "relu2_1", "conv2_2", "relu2_2", "pool2",
             "conv3_1", "relu3_1", "conv3_2", "relu3_2", "pool3"]
    model = nn.Sequential()
    for name, layer in zip(names, layers):
        model.add_module(name, layer)
    return model.eval()


def _feature_maps():
    """세 깊이의 피처 맵을 훅으로 받아 온다. 쪽 5절의 훅 코드 그대로다."""
    import torch
    from torchvision import datasets, transforms

    root = HERE.parents[3] / "data"
    ds = datasets.CIFAR10(root=str(root), train=False, download=False,
                          transform=transforms.ToTensor())
    classes = ("plane", "car", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck")
    image, label = ds[FM_INDEX]

    model = _vgg_feature_extractor()
    taken = {}

    def hook(name):
        def fn(module, inp, out):
            taken[name] = out.detach().clone()
        return fn

    handles = [getattr(model, n).register_forward_hook(hook(n))
               for n in FM_TAPS]
    with torch.no_grad():
        model(image.unsqueeze(0))
    for h in handles:
        h.remove()

    return image, classes[label], {n: taken[n][0] for n in FM_TAPS}


def make_feature_map_depths():
    image, class_name, feats = _feature_maps()

    # 그림에 적을 수 — 모두 위에서 계산한 텐서에서 바로 읽는다
    shapes = {n: tuple(feats[n].shape) for n in FM_TAPS}
    dead = {n: int((feats[n].amax((1, 2)) - feats[n].amin((1, 2)) == 0).sum())
            for n in FM_TAPS}
    assert shapes == {"relu1_1": (64, 32, 32),
                      "relu2_2": (128, 16, 16),
                      "relu3_2": (256, 8, 8)}, shapes
    assert class_name == "ship", class_name

    fig = plt.figure(figsize=(9.6, 6.1))
    gs = fig.add_gridspec(4, FM_NCHAN + 1, height_ratios=[1.25, 1, 1, 1],
                          width_ratios=[1.35] + [1] * FM_NCHAN,
                          hspace=0.38, wspace=0.12)

    # --- 맨 위: 입력 한 장 ---
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(image.permute(1, 2, 0).numpy())
    ax0.set_xticks([])
    ax0.set_yticks([])
    for s in ax0.spines.values():
        s.set_color(INK)
        s.set_linewidth(0.8)
    ax0.set_title(f"input  3 x 32 x 32\nCIFAR-10 test[{FM_INDEX}], '{class_name}'",
                  fontsize=8.5, color=INK)

    ax_note = fig.add_subplot(gs[0, 1:])
    ax_note.set_axis_off()
    ax_note.text(
        0.0, 0.92,
        "The stack is the one built in section 3, with no training at all:\n"
        f"every weight is the PyTorch default initialisation under "
        f"torch.manual_seed({FM_SEED}).\n"
        "So read the three rows below for geometry, not for meaning - the grid\n"
        "gets coarser, the channel count grows, and more channels fall silent.\n"
        "Each panel is scaled on its own (viridis, min to max), exactly as the\n"
        "page's visualize_feature_maps() does with a bare imshow.",
        fontsize=8.2, color=MUTED, ha="left", va="top", linespacing=1.5)

    # --- 아래 세 줄: 깊이마다 채널 여섯 ---
    for r, name in enumerate(FM_TAPS):
        f = feats[name]
        c, h, w = f.shape
        axl = fig.add_subplot(gs[r + 1, 0])
        axl.set_axis_off()
        axl.text(1.0, 0.56, f"{name}\n{c} x {h} x {w}", fontsize=9,
                 color=ACCENT if r == 0 else INK, ha="right", va="center",
                 family="monospace", weight="bold", linespacing=1.6)
        axl.text(1.0, 0.14, f"{dead[name]} of {c} channels\nare all zero",
                 fontsize=7.8, color=MUTED, ha="right", va="center",
                 linespacing=1.5)

        for j in range(FM_NCHAN):
            ax = fig.add_subplot(gs[r + 1, j + 1])
            panel = f[j].numpy()
            ax.imshow(panel, cmap="viridis", interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_color(MUTED)
                s.set_linewidth(0.5)
            ax.set_title(f"ch {j}", fontsize=7.5, color=INK, pad=2.5)
            if panel.max() == panel.min():
                ax.text(0.5, 0.5, "all zero", transform=ax.transAxes,
                        ha="center", va="center", fontsize=7,
                        color="#ffffff")

    save(fig, "feature_map_depths")


# === 그림 9. 1차원에서 미끄러지는 커널 =============================
#     conv1d.md 1절. 왼쪽은 그 절의 손 계산 x = [1,2,3,4,5],
#     w = [1,0,-1] 그대로이고, 오른쪽은 그림 코드가 식으로 정의한
#     길이 32짜리 신호에 같은 커널을 돌린 것이다.

C1D_X = [1.0, 2.0, 3.0, 4.0, 5.0]
C1D_W = [1.0, 0.0, -1.0]
C1D_N = 32
C1D_SLOPE = 0.03
C1D_LO, C1D_HI = 8, 20


def conv1d_valid(x, w):
    """conv1d.md 1절의 y[i] = sum_j x[i+j] w[j]. 커널을 뒤집지 않는다."""
    k = len(w)
    return [sum(x[i + j] * w[j] for j in range(k))
            for i in range(len(x) - k + 1)]


def _c1d_step_signal():
    """x[i] = slope*i + (1 if lo <= i < hi else 0). 그림 안에 그대로 적는다."""
    return [C1D_SLOPE * i + (1.0 if C1D_LO <= i < C1D_HI else 0.0)
            for i in range(C1D_N)]


def make_conv1d_sliding():
    y_small = conv1d_valid(C1D_X, C1D_W)
    assert y_small == [-2.0, -2.0, -2.0], y_small      # 1절의 출력
    assert len(y_small) == len(C1D_X) - len(C1D_W) + 1

    xs = _c1d_step_signal()
    ys = conv1d_valid(xs, C1D_W)
    assert len(ys) == C1D_N - 2 == 30
    # 평평한 구간에서는 x[i] - x[i+2] = -2*slope 로 일정하다
    assert abs(ys[0] - (-2 * C1D_SLOPE)) < 1e-12, ys[0]
    i_dn, i_up = ys.index(min(ys)), ys.index(max(ys))
    assert i_dn in (C1D_LO - 2, C1D_LO - 1), i_dn
    assert i_up in (C1D_HI - 2, C1D_HI - 1), i_up

    fig, axes = plt.subplots(2, 2, figsize=(10.4, 5.6), sharex="col",
                             gridspec_kw={"width_ratios": [1, 1.5],
                                          "hspace": 0.28, "wspace": 0.22})
    (ax_a, ax_b), (ax_c, ax_d) = axes

    # --- (a) 1절의 다섯 표본 위를 미끄러지는 창 ---
    ax_a.stem(range(len(C1D_X)), C1D_X, linefmt="-", markerfmt="o",
              basefmt=" ")
    for art, col in ((ax_a.lines[0], COOL),):
        art.set_color(col)
    for coll in ax_a.containers:
        coll.markerline.set_color(COOL)
        coll.markerline.set_markersize(5)
        coll.stemlines.set_color(COOL)
        coll.stemlines.set_linewidth(1.4)

    wins = ((0, ACCENT, "-", 1.9), (1, "#b2763f", (0, (4, 2)), 1.4),
            (2, MUTED, (0, (1.6, 1.8)), 1.4))
    for start, col, ls, lw in wins:
        inset = 0.10 * start
        ax_a.add_patch(Rectangle((start - 0.45 + inset, 0.18 + inset),
                                 2.9 - 2 * inset, 5.5 - 2 * inset,
                                 fc="none", ec=col, lw=lw, ls=ls, zorder=4))
        ax_a.text(start + 1.0, 5.85 + 0.62 * start, f"window {start}",
                  ha="center", va="bottom", fontsize=7.5, color=col)
    for i, v in enumerate(C1D_X):
        ax_a.text(i + 0.17, v, f"{v:.0f}", ha="left", va="center",
                  fontsize=8.5, color=COOL, family="monospace")
    for j, wv in enumerate(C1D_W):
        ax_a.text(j, -0.55, f"{wv:+.0f}", ha="center", va="top",
                  fontsize=9, color=ACCENT, family="monospace")
    ax_a.text(1.0, -1.45, "the three kernel taps, window 0", ha="center",
              va="top", fontsize=7.5, color=ACCENT)
    ax_a.set_ylim(-2.4, 8.2)
    ax_a.set_ylabel("x[i]", fontsize=9, color=INK)
    ax_a.set_title("(a) the five samples of section 1\n"
                   "x = [1, 2, 3, 4, 5],  w = [1, 0, -1]",
                   fontsize=9, color=INK, loc="left")
    style(ax_a)

    # --- (c) 그 출력 세 개 ---
    for i, v in enumerate(y_small):
        ax_c.plot([i, i], [0, v], color=ACCENT, lw=1.8, zorder=3)
        ax_c.plot([i], [v], marker="o", ms=5.5, color=ACCENT, zorder=4)
        ax_c.text(i, v - 0.25, f"{v:.0f}", ha="center", va="top",
                  fontsize=9, color=ACCENT, family="monospace")
    ax_c.axhline(0, color=INK, lw=0.7)
    ax_c.set_xticks(range(len(C1D_X)))
    ax_c.set_xlim(-0.7, 4.7)
    ax_c.set_ylim(-4.6, 1.0)
    ax_c.set_xlabel("position along the sequence", fontsize=9, color=INK)
    ax_c.set_ylabel("y[i]", fontsize=9, color=ACCENT)
    ax_c.text(-0.6, 0.55,
              "no padding: 5 - 3 + 1 = 3 outputs",
              ha="left", va="bottom", fontsize=8, color=MUTED)
    ax_c.text(-0.6, -3.55,
              "y[i] sits at its window's first tap,\n"
              "as in y[i] = sum_j x[i+j] w[j]",
              ha="left", va="top", fontsize=8, color=MUTED)
    style(ax_c)

    # --- (b) 같은 커널, 긴 신호 ---
    ax_b.step(range(C1D_N), xs, where="mid", color=COOL, lw=1.6, zorder=3)
    ax_b.plot(range(C1D_N), xs, marker="o", ms=2.6, ls="none", color=COOL,
              zorder=4)
    for start in (C1D_LO - 2, C1D_HI - 2):
        ax_b.add_patch(Rectangle((start - 0.45, -0.14), 2.9, 1.92, fc="none",
                                 ec=ACCENT, lw=1.5, zorder=5))
    ax_b.text((C1D_LO + C1D_HI) / 2 - 1.0, 1.98,
              "the two windows that straddle a step", ha="center",
              va="bottom", fontsize=8, color=ACCENT)
    ax_b.set_ylim(-0.4, 2.45)
    ax_b.set_ylabel("x[i]", fontsize=9, color=INK)
    ax_b.set_title("(b) the same kernel on a longer signal\n"
                   f"x[i] = {C1D_SLOPE}*i + (1 if {C1D_LO} <= i < {C1D_HI} "
                   f"else 0),  i = 0..{C1D_N - 1}",
                   fontsize=9, color=INK, loc="left")
    style(ax_b)

    # --- (d) 그 출력 ---
    ax_d.axhline(0, color=INK, lw=0.7)
    ax_d.plot(range(len(ys)), ys, color=ACCENT, lw=1.5, marker="o", ms=3.0,
              zorder=3)
    ax_d.axhline(-2 * C1D_SLOPE, color=MUTED, lw=0.9, ls=":", zorder=2)
    ax_d.annotate(f"on the flat stretches x[i] - x[i+2] = -2 x {C1D_SLOPE} "
                  f"= {-2 * C1D_SLOPE:.2f},\nnot 0 - the slow drift never "
                  "leaves the output",
                  xy=(13, -2 * C1D_SLOPE), xytext=(13.5, 1.88), fontsize=8,
                  color=MUTED, ha="center", va="center",
                  arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    ax_d.annotate(f"step up at i = {C1D_LO}\ny = {min(ys):.2f}",
                  xy=(i_dn, min(ys)), xytext=(2.5, -1.35), fontsize=8,
                  color=ACCENT, ha="center", va="center",
                  arrowprops=dict(arrowstyle="-", color=ACCENT, lw=0.8))
    ax_d.annotate(f"step down at i = {C1D_HI}\ny = +{max(ys):.2f}",
                  xy=(i_up, max(ys)), xytext=(26.0, 1.18), fontsize=8,
                  color=ACCENT, ha="center", va="center",
                  arrowprops=dict(arrowstyle="-", color=ACCENT, lw=0.8))
    ax_d.set_xlim(-1.0, C1D_N)
    ax_d.set_ylim(-1.95, 2.45)
    ax_d.set_xlabel("position along the sequence (time)", fontsize=9,
                    color=INK)
    ax_d.set_ylabel("y[i]", fontsize=9, color=ACCENT)
    ax_d.text(C1D_N - 0.5, -1.85,
              f"{C1D_N} - 3 + 1 = {len(ys)} outputs",
              ha="right", va="bottom", fontsize=8, color=MUTED,
              family="monospace")
    style(ax_d)

    fig.tight_layout()
    save(fig, "conv1d_sliding")


# === 그림 10. 10.70% 가 가리고 있는 것 ==============================
#     06_cifar10_basic.md 1절과 2.2절의 출력 블록을 그대로 옮긴 것이다.

CIFAR_CLASSES = ("plane", "car", "bird", "cat", "deer",
                 "dog", "frog", "horse", "ship", "truck")
# (부류별 맞힌 장 수) — 부류마다 시험 이미지가 1,000장씩이다
BASE_CORRECT = (0, 0, 0, 90, 0, 0, 980, 0, 0, 0)        # 1절 출력
MOM_CORRECT = (410, 556, 185, 260, 261, 261, 603, 550, 576, 445)  # 2.2절 출력
PER_CLASS = 1000


def make_cifar10_basic_per_class():
    base = [100.0 * c / PER_CLASS for c in BASE_CORRECT]
    mom = [100.0 * c / PER_CLASS for c in MOM_CORRECT]
    acc_base = 100.0 * sum(BASE_CORRECT) / (10 * PER_CLASS)
    acc_mom = 100.0 * sum(MOM_CORRECT) / (10 * PER_CLASS)
    assert round(acc_base, 2) == 10.70, acc_base       # 1절 마지막 줄
    assert round(acc_mom, 2) == 41.07, acc_mom         # 2.2절 둘째 줄
    assert round(acc_mom - acc_base, 2) == 30.37
    assert sum(1 for v in base if v == 0.0) == 8
    assert (min(mom), max(mom)) == (18.5, 60.3)

    fig, ax = plt.subplots(figsize=(9.4, 4.6))
    idx = range(10)
    width = 0.38

    ax.bar([i - width / 2 for i in idx], base, width, color=MUTED,
           edgecolor=INK, lw=0.6, zorder=3,
           label=f"momentum = 0.0  (overall {acc_base:.2f}%)")
    ax.bar([i + width / 2 for i in idx], mom, width, color=FILL,
           edgecolor=ACCENT, lw=1.0, zorder=3,
           label=f"momentum = 0.9  (overall {acc_mom:.2f}%)")

    ax.hlines(acc_base, -0.75, 9.45, color=MUTED, lw=1.0, ls="--", zorder=2)
    ax.hlines(acc_mom, -0.75, 9.45, color=ACCENT, lw=1.0, ls="--", zorder=2)
    ax.text(9.6, acc_base, f"{acc_base:.2f}%", fontsize=8, color=MUTED,
            va="center", ha="left", family="monospace")
    ax.text(9.6, acc_mom, f"{acc_mom:.2f}%", fontsize=8, color=ACCENT,
            va="center", ha="left", family="monospace")

    for i, v in enumerate(base):
        if v > 0:
            ax.text(i - width / 2, v + 1.5, f"{v:.0f}", ha="center",
                    va="bottom", fontsize=7.5, color=INK,
                    family="monospace")
    ax.text(2.0, 72.0,
            "eight of the ten classes are never once correct;\n"
            "the 10.70% is almost entirely 980 frogs",
            fontsize=8.5, color=INK, ha="center", va="center")
    ax.annotate("", xy=(6 - width / 2, 94.0), xytext=(3.4, 76.5),
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=0.9,
                                mutation_scale=10))

    ax.set_xticks(list(idx))
    ax.set_xticklabels(CIFAR_CLASSES, fontsize=8.5)
    ax.set_ylim(0, 105)
    ax.set_xlim(-0.75, 10.65)
    ax.set_ylabel("per-class recall (%)", fontsize=9, color=INK)
    ax.set_title("SimpleCNN, seed 0, 5 epochs - same 62,006 parameters, "
                 "one word of difference in the optimiser",
                 fontsize=9.5, color=INK, loc="left")
    ax.legend(fontsize=8.5, loc="upper left", frameon=False)
    ax.grid(axis="y", alpha=0.25, lw=0.6, zorder=0)
    style(ax)

    fig.tight_layout()
    save(fig, "cifar10_basic_per_class")


# === 모두 그린다 ==================================================
#     그림 함수는 이 블록보다 **위에** 두어야 한다. 모듈은 위에서 아래로
#     실행되므로 아래에 정의한 함수는 여기에서 아직 이름이 없다.


FIGURES = {
    "convolution_sliding": make_convolution_sliding,
    "transposed_conv_steps": make_transposed_conv_steps,
    "dilated_vs_downsample": make_dilated_vs_downsample,
    "gridding_coverage": make_gridding_coverage,
    "rf_strategies": make_rf_strategies,
    "depthwise_decomposition": make_depthwise_decomposition,
    "advanced_training_curve": make_advanced_training_curve,
    "feature_map_depths": make_feature_map_depths,
    "conv1d_sliding": make_conv1d_sliding,
    "cifar10_basic_per_class": make_cifar10_basic_per_class,
    "mnist_balance_and_scale": make_mnist_balance_and_scale,
    "fashion_class_means": make_fashion_class_means,
    "cifar10_class_colour": make_cifar10_class_colour,
    "fashion_classifier_errors": make_fashion_classifier_errors,
    "binary_pair_curves": make_binary_pair_curves,
}


if __name__ == "__main__":
    # 이름을 주면 그 그림만 다시 그린다. 전부 다시 그리면 손대지 않은 SVG 도
    # 타임스탬프와 클립 경로 id 만 바뀐 채 diff 에 끼어든다.
    #   python docs/ch10/cnn/figures/make_figures.py mnist_balance_and_scale --png
    wanted = [a for a in sys.argv[1:] if not a.startswith("--")]
    unknown = [w for w in wanted if w not in FIGURES]
    assert not unknown, unknown
    check_numbers()
    for name, draw in FIGURES.items():
        if not wanted or name in wanted:
            draw()
