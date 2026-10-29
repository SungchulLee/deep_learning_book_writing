"""ch10/cnn 의 그림을 그린다.

규칙은 CLAUDE.md 의 Figures 항에 적어 둔 대로다.
그림 안의 글자는 모두 아스키라야 한다 (matplotlib 기본 폰트에 한글이 없다).

그리는 그림
  convolution_sliding.svg     convolution.md 2절 — 5x5 입력에 3x3 커널이 미끄러진다
  transposed_conv_steps.svg   transposed_conv.md 2절 — 0 끼우기 / 덧대기 / 보통 컨볼루션
  dilated_vs_downsample.svg   dilated_convolutions.md 6절 — 두 경로의 해상도
  gridding_coverage.svg       dilated_convolutions.md 5절 — d=2,2 와 d=1,2,4 의 덮개

그림에 적는 수는 모두 쪽의 코드·출력에서 확인한 것이다. check_numbers() 가
그 확인을 다시 돌린다.

실행: python docs/ch10/cnn/figures/make_figures.py
"""
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
    print("numbers check out")


if __name__ == "__main__":
    check_numbers()
    make_convolution_sliding()
    make_transposed_conv_steps()
    make_dilated_vs_downsample()
    make_gridding_coverage()
