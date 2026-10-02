"""4.6절 전이 companion 쪽들의 등뼈 그림을 만든다.

VGG16 그림(vgg16_layers.svg)과 같은 꼴이다. 왼쪽에 층을 아래에서 위로 쌓고,
오른쪽에 텐서 모양을 **세 가지 입력 크기**로 적는다. 가로로 읽으면 층마다
모양이 어떻게 바뀌는지가 보이고, 세 열을 견주면 크기를 바꿨을 때 그 모델이
버티는지 터지는지가 보인다.

층은 묶지 않고 **하나씩 다 그린다.** "Inception block x5" 처럼 묶어 두면
정작 그림이 답해야 할 물음(어느 층에서 무엇이 바뀌는가)이 가려진다.

**흐름은 어디서나 아래에서 위로 간다.** 큰 줄기도, 옆에 펼친 블록 속도,
그 블록 안 상자의 줄 차례도 그렇다. 한 군데라도 거꾸로 두면 읽는 이가
방향을 다시 잡아야 하므로, 새 인셋을 더할 때도 이 규칙을 지킬 것.

모양은 손으로 적지 않는다. trace_backbones.py 가 실제로 재어 둔
backbone_shapes.json 을 읽는다.

그림 규칙(CLAUDE.md):
    - SVG, svg.fonttype='path', transparent=True
    - 그림 안의 글자는 모두 ASCII (기본 글꼴에 한글이 없다)

실행:
    python trace_backbones.py && python make_backbone_figures.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

matplotlib.rcParams["svg.fonttype"] = "path"

CONV = ("#fce4d2", "#d4813f")
POOL = ("#dbe7f7", "#4a72b8")
FCL  = ("#dfeadb", "#4e7a43")
INP  = ("#e6e6e6", "#333333")
SPEC = ("#fdf0c8", "#c9a227")
NORM = ("#efe7f7", "#8e6bb5")
HL   = "#fff3cd"

BW, BH, GAP = 6.2, 0.46, 0.10
COL = 3.5                       # 모양 열 사이 간격

SHAPES = json.loads(Path("backbone_shapes.json").read_text())


def colour_of(text):
    t = text.lower()
    if t.startswith("input"):            return INP
    if "pool" in t and "conv" not in t:  return POOL
    if t.startswith("fc") or "fc " in t: return FCL
    if "layernorm" in t:                 return NORM
    return CONV


def node(ax, x, y, w, h, text, colors=CONV, fs=7.0):
    fill, edge = colors
    ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=fill, edgecolor=edge,
                               linewidth=1.0, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color="#26323c", zorder=3)


def arrow(ax, p, q, color="#8a97a4", lw=0.9, rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="->", color=color, lw=lw,
                                 mutation_scale=8,
                                 connectionstyle=f"arc3,rad={rad}", zorder=3))


def title(ax, ix, iy, iw, ih, t, sub=None):
    ax.text(ix + iw / 2, iy + ih - 0.28, t, ha="center", va="top",
            fontsize=8.4, weight="bold", color="#8a6d1a", zorder=3)
    if sub:
        ax.text(ix + iw / 2, iy + ih - 0.72, sub, ha="center", va="top",
                fontsize=6.9, color="#8a6d1a", zorder=3)


def draw(name, svg, spec_row, inset_fn, heads, caption, iw=6.0, ih=5.2):
    """층 목록만 그린다. 블록 속은 draw_block() 이 따로, 같은 크기로 그린다."""
    d = SHAPES[name]
    sizes = [str(s) for s in d["sizes"]]
    rows, tr = d["rows"], d["trace"]

    fig, ax = plt.subplots(figsize=(17.0, 1.2 + 0.42 * len(rows)))
    spec_y = None
    for i, (lab, text) in enumerate(rows):
        y = i * (BH + GAP)                          # rows[0] 이 입력이므로 맨 아래로 간다
        is_spec = (lab == spec_row)
        if is_spec:
            ax.add_patch(plt.Rectangle((-2.9, y - GAP / 2), BW + 3.0 + 3 * COL,
                                       BH + GAP, facecolor=HL, edgecolor="none",
                                       zorder=0))
            spec_y = y
        fill, edge = SPEC if is_spec else colour_of(text)
        ax.add_patch(plt.Rectangle((0, y), BW, BH, facecolor=fill,
                                   edgecolor=edge, linewidth=1.0))
        ax.text(BW / 2, y + BH / 2, text, ha="center", va="center",
                fontsize=7.4, color="#26323c")
        ax.text(-0.22, y + BH / 2, lab, ha="right", va="center",
                fontsize=7.0, color="#54626e")
        for c, s in enumerate(sizes):
            if lab == "input":
                sh, dead = f"3x{s}x{s}", False
            else:
                sh = tr[s]["shapes"].get(lab)
                dead = sh is None
            ax.text(BW + 0.3 + c * COL, y + BH / 2, sh or "--",
                    ha="left", va="center", fontsize=6.9,
                    color="#c44f4f" if dead else "#6b7883")

    top = len(rows) * (BH + GAP)
    for c, (h, sub, col) in enumerate(heads):
        ax.text(BW + 0.3 + c * COL, top + 0.42, h, ha="left", va="bottom",
                fontsize=8.0, weight="bold", color="#3c4b58")
        ax.text(BW + 0.3 + c * COL, top + 0.08, sub, ha="left", va="bottom",
                fontsize=6.8, color=col)
    ax.text(BW + 0.3, top + 1.05, "shape after the layer  (PyTorch, NCHW)",
            ha="left", va="bottom", fontsize=7.6, color="#8a97a4")

    ax.text(-2.9, -1.05, caption, ha="left", va="bottom",
            fontsize=7.4, color="#8a6d1a")
    ax.set_xlim(-3.2, BW + 3 * COL + 0.6)
    ax.set_ylim(-1.3, top + 1.9)
    ax.axis("off")
    fig.savefig(svg, transparent=True, bbox_inches="tight")
    plt.close(fig)
    print("  " + svg)


# =============================================================================
def inset_resnet(ax, ix, iy, iw, ih):
    title(ax, ix, iy, iw, ih, "BasicBlock")
    cx = ix + iw / 2
    # 큰 그림과 같이 아래에서 위로 흐른다. x 가 아래로 들어와 위로 나간다.
    ax.text(cx - 0.16, iy + 0.62, "x", ha="right", va="center", fontsize=8.2,
            color="#26323c", zorder=3)
    node(ax, cx - 1.7, iy + 1.05, 3.4, 0.46, "3x3 conv - BN - ReLU")
    node(ax, cx - 1.7, iy + 1.85, 3.4, 0.46, "3x3 conv - BN")
    node(ax, cx - 0.5, iy + 2.65, 1.0, 0.46, "+", SPEC, fs=10)
    node(ax, cx - 0.9, iy + 3.45, 1.8, 0.46, "ReLU", NORM)
    for a, b in ((0.72, 1.02), (1.53, 1.82), (2.33, 2.62), (3.13, 3.42)):
        arrow(ax, (cx, iy + a), (cx, iy + b))
    arrow(ax, (cx + 1.75, iy + 0.66), (cx + 0.55, iy + 2.80),
          color="#c9a227", lw=1.8, rad=0.40)
    ax.text(cx + 2.05, iy + 1.75, "identity", ha="left", va="center",
            fontsize=7.0, color="#8a6d1a", zorder=3)


def inset_inception(ax, ix, iy, iw, ih):
    title(ax, ix, iy, iw, ih, "Inception block",
          "(Mixed_5b)   number = output channels")
    cx = ix + iw / 2
    w = 1.26
    # 층 하나에 한 줄. "3x3 x2" 처럼 묶어 적으면 3x3x2 로 읽혀 텐서 모양처럼 보인다.
    # 아래에서 위로 읽는다. 큰 그림이 아래에서 위로 흐르므로 상자 안도 같아야
    # 한다 -- 맨 아랫줄이 먼저 지나는 층, 맨 윗줄이 이어 붙기 직전의 층이다.
    branches = ["1x1 -> 64",
                "5x5 -> 64\n1x1 -> 48",
                "3x3 -> 96\n3x3 -> 96\n1x1 -> 64",
                "1x1 -> 32\navgpool 3x3"]
    for k, txt in enumerate(branches):
        bx = ix + 0.30 + k * (w + 0.16)
        node(ax, bx, iy + 1.40, w, 1.90, txt, CONV, fs=6.2)
        arrow(ax, (cx, iy + 0.98), (bx + w / 2, iy + 1.37), rad=0.12)
        arrow(ax, (bx + w / 2, iy + 3.34), (cx, iy + 3.70), rad=0.12)
    node(ax, cx - 1.7, iy + 3.72, 3.4, 0.46, "concat -> 64+64+96+32 = 256", SPEC, fs=6.6)
    ax.text(cx, iy + 0.68, "192 channels in", ha="center", fontsize=6.9,
            color="#6b7883", zorder=3)


def inset_mobilenet(ax, ix, iy, iw, ih):
    title(ax, ix, iy, iw, ih, "InvertedResidual", "channels mixed only by the 1x1s")
    cx = ix + iw / 2
    for k, (txt, col) in enumerate([("1x1 conv  (expand)", CONV),
                                    ("3x3 depthwise  (space)", SPEC),
                                    ("Squeeze-Excitation", NORM),
                                    ("1x1 conv  (project)", CONV)]):
        node(ax, cx - 2.0, iy + 0.85 + k * 0.80, 4.0, 0.50, txt, col, fs=6.9)
        if k:
            # 아래 상자 위쪽에서 이 상자 아래쪽으로 — 위를 향한다
            arrow(ax, (cx, iy + 0.55 + k * 0.80), (cx, iy + 0.83 + k * 0.80))


def inset_efficientnet(ax, ix, iy, iw, ih):
    title(ax, ix, iy, iw, ih, "compound scaling", "scaled together, not one at a time")
    for k, (nm, why) in enumerate([("depth  d", "more MBConv blocks"),
                                   ("width  w", "more channels"),
                                   ("resolution  r", "larger input")]):
        y = iy + 1.15 + k * 0.90
        node(ax, ix + 0.4, y, 2.0, 0.55, nm, SPEC, fs=7.2)
        ax.text(ix + 2.6, y + 0.28, why, ha="left", va="center",
                fontsize=6.9, color="#6b7883", zorder=3)
    ax.text(ix + iw / 2, iy + 0.55, "B0 -> B7 : one architecture, seven sizes",
            ha="center", fontsize=6.9, color="#8a6d1a", zorder=3)


def inset_vit(ax, ix, iy, iw, ih):
    title(ax, ix, iy, iw, ih, "patchify = strided convolution",
          "196 + 1 = 197 tokens, 768 wide")
    gx, gy, cell = ix + 0.55, iy + 1.55, 0.34
    for r in range(4):
        for c in range(4):
            node(ax, gx + c * cell, gy + r * cell, cell * 0.9, cell * 0.9, "",
                 ("#e9eef5", "#6b7883"))
    ax.text(gx + 2 * cell, gy - 0.26, "224x224", ha="center", fontsize=6.8,
            color="#6b7883", zorder=3)
    ax.text(gx + 2 * cell, gy + 4 * cell + 0.10, "16x16 patches", ha="center",
            fontsize=6.8, color="#6b7883", zorder=3)
    tx = ix + 3.2
    for k, t in enumerate(["token 196", "...", "token 2", "token 1"]):
        node(ax, tx, iy + 1.25 + k * 0.52, 2.1, 0.42, t, NORM, fs=6.6)
    node(ax, tx, iy + 0.68, 2.1, 0.42, "class token", SPEC, fs=6.6)
    arrow(ax, (gx + 4 * cell + 0.12, gy + 2 * cell), (tx - 0.10, iy + 1.85))



def inset_densenet(ax, ix, iy, iw, ih):
    title(ax, ix, iy, iw, ih, "DenseBlock", "concatenate, do not add")
    cx = ix + iw / 2
    # 아래에서 위로. 층마다 앞의 것을 모두 이어 붙여 받는다.
    for k in range(4):
        y = iy + 0.70 + k * 0.72
        node(ax, cx - 1.9, y, 2.5, 0.46, f"layer {k+1}", CONV if k else SPEC, fs=6.9)
        ax.text(cx + 0.78, y + 0.23, f"in {64 + 32*k}", ha="left", va="center",
                fontsize=6.5, color="#6b7883", zorder=3)
        if k:
            arrow(ax, (cx - 0.65, y - 0.26), (cx - 0.65, y - 0.02))
    # 건너뛰어 올라가는 이음들
    for k in range(3):
        y0 = iy + 0.93 + k * 0.72
        arrow(ax, (cx - 2.05, y0), (cx - 2.05, iy + 0.93 + (k + 1.6) * 0.72),
              color="#c9a227", lw=1.2, rad=-0.45)
    ax.text(cx - 2.35, iy + 2.1, "every earlier\nfeature map,\nconcatenated",
            ha="right", va="center", fontsize=6.6, color="#8a6d1a", zorder=3)
    ax.text(cx, iy + 3.70, "growth rate 32: each layer adds 32 channels",
            ha="center", fontsize=6.7, color="#8a6d1a", zorder=3)


def inset_convnext(ax, ix, iy, iw, ih):
    title(ax, ix, iy, iw, ih, "ConvNeXt block", "a CNN rebuilt with transformer habits")
    cx = ix + iw / 2
    for k, (txt, col) in enumerate([("7x7 depthwise conv", SPEC),
                                    ("LayerNorm", NORM),
                                    ("1x1 conv -> 4x wider", CONV),
                                    ("GELU", NORM),
                                    ("1x1 conv -> back", CONV)]):
        node(ax, cx - 2.0, iy + 0.62 + k * 0.62, 4.0, 0.44, txt, col, fs=6.6)
        if k:
            arrow(ax, (cx, iy + 0.44 + k * 0.62), (cx, iy + 0.60 + k * 0.62))
    ax.text(cx, iy + 3.86, "big kernel, LayerNorm, GELU, one activation",
            ha="center", fontsize=6.6, color="#8a6d1a", zorder=3)



# =============================================================================
# 블록 그림 — 층 목록과 **같은 크기**로 따로 그린다.
#
# 이 블록이 곧 그 모델이 내놓은 생각이므로, 층 목록 옆에 작게 끼워 넣으면
# 순서가 뒤바뀐다. 따로 떼어 크게 그리고, 안을 지나는 동안 모양이 어떻게
# 바뀌는지를 함께 적는다 -- 그것이 층 목록이 대신 말해 줄 수 없는 것이다.
# =============================================================================
RW, RH, RG = 6.2, 0.62, 0.34     # 블록 그림의 상자 크기


def draw_block(svg, title_text, subtitle, rows, caption, note=None, skip=None):
    """rows: (글, 색, 들어온 모양, 나간 모양). 아래에서 위로 쌓는다.

    skip: (아래 칸 번호, 위 칸 번호, 라벨) — 건너뛰는 이음을 그린다.
    """
    n = len(rows)
    fig, ax = plt.subplots(figsize=(11.0, 1.9 + 0.62 * n))
    x0 = 0.0
    for k, (txt, col, shp_in, shp_out) in enumerate(rows):
        y = k * (RH + RG)
        fill, edge = col
        ax.add_patch(plt.Rectangle((x0, y), RW, RH, facecolor=fill,
                                   edgecolor=edge, linewidth=1.3))
        ax.text(x0 + RW / 2, y + RH / 2, txt, ha="center", va="center",
                fontsize=9.0, color="#26323c")
        if shp_out:
            ax.text(x0 + RW + 0.35, y + RH / 2, shp_out, ha="left", va="center",
                    fontsize=8.2, color="#6b7883")
        if k == 0 and shp_in:
            ax.text(x0 + RW + 0.35, y - RG / 2 - 0.02, shp_in, ha="left",
                    va="center", fontsize=8.2, color="#6b7883")
            ax.text(x0 - 0.3, y - RG / 2 - 0.02, "in", ha="right", va="center",
                    fontsize=8.2, color="#54626e")
        if k:
            ax.add_patch(FancyArrowPatch((x0 + RW / 2, y - RG + 0.04),
                                         (x0 + RW / 2, y - 0.04),
                                         arrowstyle="->", color="#8a97a4", lw=1.0,
                                         mutation_scale=10, zorder=3))
    top = n * (RH + RG) - RG
    # 지름길은 하나일 수도 여럿일 수도 있다. ViT 블록은 더하는 자리가 둘이라
    # 둘 다 그려야 한다 -- 들어오는 줄이 하나뿐인 "+" 는 아무 말도 하지 않는다.
    skips = [skip] if (skip and isinstance(skip[0], int)) else list(skip or [])
    for i, (a, b, lab) in enumerate(skips):
        off = 0.55 + i * 1.15
        ya = a * (RH + RG) - RG / 2
        yb = b * (RH + RG) + RH / 2
        ax.add_patch(FancyArrowPatch((x0 - off, ya), (x0 - off, yb),
                                     arrowstyle="->", color="#c9a227", lw=2.0,
                                     mutation_scale=11,
                                     connectionstyle="arc3,rad=-0.45", zorder=3))
        ax.text(x0 - off - 1.25, (ya + yb) / 2, lab, ha="center", va="center",
                fontsize=8.4, color="#8a6d1a")
    ax.text(x0 + RW / 2, top + 0.95, title_text, ha="center", va="bottom",
            fontsize=12.5, weight="bold", color="#8a6d1a")
    ax.text(x0 + RW / 2, top + 0.45, subtitle, ha="center", va="bottom",
            fontsize=8.8, color="#8a6d1a")
    ax.text(x0 + RW + 0.35, top + 0.45, "shape", ha="left", va="bottom",
            fontsize=8.2, color="#8a97a4")
    if note:
        ax.text(-3.4, -1.05, note, ha="left", va="bottom", fontsize=8.2,
                color="#6b7883")
    ax.text(-3.4, -1.65, caption, ha="left", va="bottom", fontsize=8.2,
            color="#8a6d1a")
    ax.set_xlim(-3.6 - 1.15 * max(0, len(skips) - 1), RW + 4.6)
    ax.set_ylim(-2.0, top + 1.9)
    ax.axis("off")
    fig.savefig(svg, transparent=True, bbox_inches="tight")
    plt.close(fig)
    print("  " + svg)


# =============================================================================
# 어텐션 한 칸도 따로 그린다.
#
# 블록 그림에서 "Multi-Head Attention, 12 heads" 는 상자 하나였다. 그 안에
# Q·K·V 세 사영, 머리 쪼개기, 197x197 짜리 어텐션 행렬, 소프트맥스, 다시
# 이어 붙이기, 출력 사영이 전부 들어 있다. 블록에서 가장 많은 일이 일어나는
# 칸인데 그림에서는 가장 말이 없었다.
#
# 층 목록에서 블록을 떼어 따로 그린 것과 같은 까닭으로, 그 칸을 떼어 따로
# 그린다. 특히 **197x197** 이 눈에 보여야 한다 -- 토큰 수의 제곱으로 자라는
# 유일한 자리이고, ViT 가 큰 그림에서 비싼 까닭이 거기 있다.
# =============================================================================
AW, AH, AVG, ACG = 3.25, 0.64, 0.52, 0.45     # 어텐션 그림의 상자와 사이


def draw_attention(svg, title_text, subtitle, caption, note=None):
    pitch_x, pitch_y = AW + ACG, AH + AVG
    W = 3 * AW + 2 * ACG
    cx = [i * pitch_x + AW / 2 for i in range(3)]
    fig, ax = plt.subplots(figsize=(11.0, 6.6))

    def box(x, y, w, txt, col, shp=None, fs=8.4):
        fill, edge = col
        ax.add_patch(plt.Rectangle((x, y), w, AH, facecolor=fill, edgecolor=edge,
                                   linewidth=1.3, zorder=2))
        ax.text(x + w / 2, y + AH / 2, txt, ha="center", va="center",
                fontsize=fs, color="#26323c", zorder=3)
        if shp:
            ax.text(W + 0.30, y + AH / 2, shp, ha="left", va="center",
                    fontsize=8.0, color="#6b7883")

    def up(x, y0, y1, col="#8a97a4", lw=1.0):
        ax.add_patch(FancyArrowPatch((x, y0 + 0.04), (x, y1 - 0.04),
                                     arrowstyle="->", color=col, lw=lw,
                                     mutation_scale=10, zorder=3))

    y = [i * pitch_y for i in range(7)]
    box(0, y[0], W, "input  (after LayerNorm)", INP, "197x768")
    for i, t in enumerate(["W_Q   768 -> 768", "W_K   768 -> 768", "W_V   768 -> 768"]):
        box(i * pitch_x, y[1], AW, t, CONV)
        up(cx[i], y[0] + AH, y[1])
    ax.text(W + 0.30, y[1] + AH / 2, "197x768 -> 12 x (197x64)", ha="left",
            va="center", fontsize=8.0, color="#6b7883")

    span = 2 * AW + ACG                      # Q 와 K 두 칸을 덮는다
    box(0, y[2], span, "Q K^T  /  8", SPEC, None)
    ax.text(W + 0.30, y[2] + AH / 2, "197x197  <- tokens squared", ha="left",
            va="center", fontsize=8.0, color="#c4792f")
    up(cx[0], y[1] + AH, y[2]); up(cx[1], y[1] + AH, y[2])
    box(0, y[3], span, "softmax  (each row)", NORM, None)
    ax.text(W + 0.30, y[3] + AH / 2, "197x197", ha="left", va="center",
            fontsize=8.0, color="#6b7883")
    up(span / 2, y[2] + AH, y[3])

    box(0, y[4], W, "x V   (weighted average of V)", SPEC, "12 x (197x64)")
    up(span / 2, y[3] + AH, y[4])
    # V 는 어텐션 행렬을 거치지 않고 옆으로 올라와 여기서 합류한다
    ax.plot([cx[2], cx[2]], [y[1] + AH, y[4] - 0.04], color="#c9a227", lw=1.7,
            zorder=2, solid_capstyle="round")
    up(cx[2], y[4] - 0.30, y[4], col="#c9a227", lw=1.7)
    ax.text(cx[2] + 0.18, (y[1] + y[4]) / 2, "V skips the matrix", ha="left",
            va="center", fontsize=7.8, color="#8a6d1a", rotation=90)

    box(0, y[5], W, "concat   12 heads", SPEC, "197x768")
    up(W / 2, y[4] + AH, y[5])
    box(0, y[6], W, "out_proj   768 -> 768", CONV, "197x768")
    up(W / 2, y[5] + AH, y[6])

    top = y[6] + AH
    ax.text(W / 2, top + 0.95, title_text, ha="center", va="bottom",
            fontsize=12.5, weight="bold", color="#8a6d1a")
    ax.text(W / 2, top + 0.45, subtitle, ha="center", va="bottom",
            fontsize=8.8, color="#8a6d1a")
    ax.text(W + 0.30, top + 0.45, "shape", ha="left", va="bottom",
            fontsize=8.2, color="#8a97a4")
    if note:
        ax.text(0, -0.95, note, ha="left", va="top", fontsize=8.2, color="#6b7883")
    ax.text(0, -1.55, caption, ha="left", va="top", fontsize=8.2, color="#8a6d1a")
    ax.set_xlim(-0.4, W + 5.6)
    ax.set_ylim(-2.4, top + 1.9)
    ax.axis("off")
    fig.savefig(svg, transparent=True, bbox_inches="tight")
    plt.close(fig)
    print("  " + svg)


# =============================================================================
# DenseBlock 도 따로 그린다.
#
# draw_block 으로 그리면 층 넷이 화살표로 이어진 **사슬**이 된다. 그러면 둘째
# 층이 첫째 층의 출력만 받는 것처럼 보이는데, DenseBlock 에서 둘째 층은 그
# 아래 것을 **모두 이어 붙여** 받는다(denselayer2 의 conv1 은 입력 채널이
# 96 이다 -- 들어온 64 에 첫 층이 만든 32 를 더한 값이다). 이어 붙이기가
# 곧 이 블록의 이름인데 사슬 그림에는 그것이 한 군데도 없다.
#
# 그래서 채널 띠가 자라는 모습으로 그린다. 먼저 온 채널은 층을 **비켜 지나**
# 그대로 남고, 층은 32 칸을 새로 만들어 오른쪽에 덧붙일 뿐이다.
# =============================================================================
DCH = 0.034                        # 채널 하나가 차지하는 가로 길이
DBARH, DLH, DGAP = 0.46, 0.58, 0.46


def draw_dense_block(svg, title_text, subtitle, x_ch, growth, nlayer, inner,
                     caption, note=None):
    pitch = DBARH + DLH + 2 * DGAP
    total = x_ch + growth * nlayer
    DW = total * DCH
    fig, ax = plt.subplots(figsize=(11.0, 1.9 + 0.74 * nlayer))

    def bar(y, chans, mark_new):
        x = 0.0
        for i, c in enumerate(chans):
            w = c * DCH
            if i == 0:
                col = INP
            elif mark_new and i == len(chans) - 1:
                col = SPEC
            else:
                col = CONV
            ax.add_patch(plt.Rectangle((x, y), w, DBARH, facecolor=col[0],
                                       edgecolor=col[1], linewidth=1.1, zorder=2))
            ax.text(x + w / 2, y + DBARH / 2, str(c), ha="center", va="center",
                    fontsize=7.4, color="#26323c", zorder=3)
            x += w
        ax.text(x + 0.20, y + DBARH / 2, f"{sum(chans)}ch", ha="left",
                va="center", fontsize=8.0, color="#6b7883")

    for k in range(nlayer + 1):
        bar(k * pitch, [x_ch] + [growth] * k, k > 0)

    for k in range(nlayer):
        yb = k * pitch                              # 들어가는 띠
        yl = yb + DBARH + DGAP                      # 층
        yn = (k + 1) * pitch                        # 나오는 띠
        win = (x_ch + growth * k) * DCH             # 층이 받는 띠의 너비
        ax.add_patch(plt.Rectangle((0, yl), DW, DLH, facecolor=CONV[0],
                                   edgecolor=CONV[1], linewidth=1.3, zorder=2))
        ax.text(DW / 2, yl + DLH / 2, f"layer {k + 1}    {inner}", ha="center",
                va="center", fontsize=8.4, color="#26323c", zorder=3)
        # 받는 것은 그 아래 띠 **전체**다
        ax.add_patch(FancyArrowPatch((win / 2, yb + DBARH + 0.04),
                                     (win / 2, yl - 0.04), arrowstyle="->",
                                     color="#8a97a4", lw=1.0, mutation_scale=10,
                                     zorder=3))
        # 내놓는 32 칸은 오른쪽 끝에 새로 붙는다
        xn = win + growth * DCH / 2
        ax.add_patch(FancyArrowPatch((xn, yl + DLH + 0.04), (xn, yn - 0.04),
                                     arrowstyle="->", color="#8a97a4", lw=1.0,
                                     mutation_scale=10, zorder=3))
        # 먼저 온 채널은 층을 비켜 그대로 올라간다 -- 이것이 이어 붙이기다
        ax.add_patch(FancyArrowPatch((-0.42, yb + DBARH / 2),
                                     (-0.42, yn + DBARH / 2), arrowstyle="->",
                                     color="#c9a227", lw=1.7, mutation_scale=11,
                                     connectionstyle="arc3,rad=-0.30", zorder=3))

    top = nlayer * pitch + DBARH
    ax.text(DW / 2, top + 0.95, title_text, ha="center", va="bottom",
            fontsize=12.5, weight="bold", color="#8a6d1a")
    ax.text(DW / 2, top + 0.45, subtitle, ha="center", va="bottom",
            fontsize=8.8, color="#8a6d1a")
    ax.text(-1.75, (nlayer * pitch) / 2, "carried through\nuntouched",
            ha="center", va="center", fontsize=8.0, color="#8a6d1a")
    if note:
        ax.text(-1.95, -0.95, note, ha="left", va="top", fontsize=8.2,
                color="#6b7883")
    ax.text(-1.95, -1.55, caption, ha="left", va="top", fontsize=8.2,
            color="#8a6d1a")
    ax.set_xlim(-3.1, DW + 2.2)
    ax.set_ylim(-2.5, top + 1.9)
    ax.axis("off")
    fig.savefig(svg, transparent=True, bbox_inches="tight")
    plt.close(fig)
    print("  " + svg)


# =============================================================================
# 잔차 블록도 따로 그린다.
#
# draw_block 으로 그리면 틀린 말을 하지는 않는다 -- 본줄기는 정말 한 줄로
# 이어지고, 지름길은 옆에 호를 하나 그려 두었다. 다만 셋이 아쉬웠다.
#
#   1. "3x3 conv - BN - ReLU" 처럼 세 연산을 한 칸에 눌러 담았다. 더하는 자리
#      앞에 ReLU 가 없다는 것이 이 블록의 요점인데, 눌러 담으면 그 자리가
#      보이지 않는다.
#   2. 더하는 자리가 다른 칸과 똑같은 네모였다. 들어오는 줄이 둘인 곳은
#      그것 하나뿐이므로 모양이 달라야 한다.
#   3. **모양이 맞지 않는 블록을 그리지 않았다.** 그림 설명이 "the add needs
#      matching shapes" 라고 물음을 띄워 놓고 답을 그리지 않은 꼴이다.
#      ResNet18 에서 layer2/3/4 의 첫 블록은 채널이 두 배가 되고 한 변이
#      반으로 줄어 x 를 그대로 더할 수 없다. 그 자리에 1x1 합성곱이 선다.
#
# 그래서 두 갈래를 나란히 그린다 -- 왼쪽은 x 를 그대로 더하는 layer1.0,
# 오른쪽은 1x1 로 x 를 맞춰 주는 layer2.0.
# =============================================================================
MW, MH, MV = 4.0, 0.60, 0.46      # 본줄기 상자: 너비, 높이, 사이
SHPW, SKIPX, SKIPW = 2.35, 0.62, 3.70   # 모양 칸, 지름길 간격, 지름길 상자
DASH = 0.28                       # 점선 테두리가 상자에서 떨어지는 거리
PANEL = SHPW + MW + SKIPX + SKIPW + 1.5


def draw_residual_block(svg, panels, caption, note=None):
    """panels: [(제목, 밑글, 본줄기 [(글,색,모양)], 지름길 글 또는 None, 나간 모양)]

    본줄기는 아래에서 위로, 지름길은 x 에서 오른쪽으로 빠져 더하는 자리로
    올라간다. 더하는 자리는 동그라미이고 ReLU 는 그 **뒤**에 온다.
    """
    nrow = max(len(p[2]) for p in panels)
    pitch = MH + MV
    y_x = -1.45                            # x 띠
    y_add = nrow * pitch + 0.30            # 동그라미 복판
    y_relu = y_add + 0.62                  # 마지막 ReLU 상자 밑변
    fig, ax = plt.subplots(figsize=(11.0, 2.0 + 0.60 * (nrow + 3)))

    for pi, (ttl, sub, rows, skip_txt, shp_out) in enumerate(panels):
        px = pi * PANEL
        bx = px + SHPW                     # 본줄기 상자 왼쪽 끝
        cx = bx + MW / 2
        sx = bx + MW + SKIPX + SKIPW / 2   # 지름길 복판

        # x 띠는 본줄기와 지름길을 함께 받친다
        ax.add_patch(plt.Rectangle((bx, y_x), MW, MH, facecolor=INP[0],
                                   edgecolor=INP[1], linewidth=1.3))
        ax.text(cx, y_x + MH / 2, "x", ha="center", va="center",
                fontsize=10.5, style="italic", color="#26323c")
        ax.text(bx - DASH - 0.16, y_x + MH / 2, rows[0][2], ha="right",
                va="center", fontsize=8.0, color="#6b7883")

        # F(x) 를 점선으로 두른다. 모양 글씨는 점선 **바깥**에 둔다 --
        # 안쪽에 두면 테두리가 글씨를 가로지른다.
        dash_top = nrow * pitch - MV + DASH
        ax.add_patch(plt.Rectangle((bx - DASH, -DASH), MW + 2 * DASH,
                                   dash_top + DASH, fill=False,
                                   edgecolor="#8a97a4", linewidth=1.0,
                                   linestyle=(0, (4, 3)), zorder=1))
        ax.text(bx + MW + DASH + 0.10, dash_top, "F(x)", ha="left", va="top",
                fontsize=8.6, color="#54626e", style="italic")

        for r, (txt, col, shp) in enumerate(rows):
            y = r * pitch
            fill, edge = col
            ax.add_patch(plt.Rectangle((bx, y), MW, MH, facecolor=fill,
                                       edgecolor=edge, linewidth=1.3, zorder=2))
            ax.text(cx, y + MH / 2, txt, ha="center", va="center",
                    fontsize=8.4, color="#26323c", zorder=3)
            if shp:
                ax.text(bx - DASH - 0.16, y + MH / 2, shp, ha="right",
                        va="center", fontsize=8.0, color="#6b7883")
            lo = y_x + MH if r == 0 else y - MV
            ax.add_patch(FancyArrowPatch((cx, lo + 0.04), (cx, y - 0.04),
                                         arrowstyle="->", color="#8a97a4",
                                         lw=1.0, mutation_scale=10, zorder=3))

        # 더하는 자리 -- 들어오는 줄이 둘인 유일한 곳이므로 동그라미로 둔다
        ax.add_patch(FancyArrowPatch((cx, nrow * pitch - MV + 0.04),
                                     (cx, y_add - 0.26), arrowstyle="->",
                                     color="#8a97a4", lw=1.0,
                                     mutation_scale=10, zorder=3))
        ax.add_patch(plt.Circle((cx, y_add), 0.24, facecolor=SPEC[0],
                                edgecolor=SPEC[1], linewidth=1.3, zorder=4))
        ax.text(cx, y_add, "+", ha="center", va="center", fontsize=11,
                color="#26323c", zorder=5)

        # 지름길: x 에서 오른쪽으로 빠져 올라가 더하는 자리로 들어온다
        ax.plot([cx, sx], [y_x + MH / 2, y_x + MH / 2], color="#c9a227",
                lw=1.6, zorder=2, solid_capstyle="round")
        if skip_txt:
            sh2 = 2 * MH
            sy = (nrow * pitch) / 2 - sh2 / 2
            ax.plot([sx, sx], [y_x + MH / 2, sy], color="#c9a227", lw=1.6, zorder=2)
            ax.add_patch(plt.Rectangle((sx - SKIPW / 2, sy), SKIPW, sh2,
                                       facecolor=SPEC[0], edgecolor=SPEC[1],
                                       linewidth=1.3, zorder=3))
            ax.text(sx, sy + sh2 / 2, skip_txt, ha="center", va="center",
                    fontsize=8.0, color="#26323c", zorder=4)
            ax.plot([sx, sx], [sy + sh2, y_add], color="#c9a227", lw=1.6, zorder=2)
        else:
            ax.plot([sx, sx], [y_x + MH / 2, y_add], color="#c9a227",
                    lw=1.6, zorder=2)
            ax.text(sx + 0.16, (y_x + y_add) / 2, "identity", ha="left",
                    va="center", fontsize=8.0, color="#8a6d1a", rotation=90)
        ax.add_patch(FancyArrowPatch((sx, y_add), (cx + 0.26, y_add),
                                     arrowstyle="->", color="#c9a227", lw=1.6,
                                     mutation_scale=11, zorder=3))

        # 더한 **뒤** 의 ReLU
        ax.add_patch(FancyArrowPatch((cx, y_add + 0.26), (cx, y_relu - 0.04),
                                     arrowstyle="->", color="#8a97a4", lw=1.0,
                                     mutation_scale=10, zorder=3))
        fill, edge = NORM
        ax.add_patch(plt.Rectangle((bx, y_relu), MW, MH, facecolor=fill,
                                   edgecolor=edge, linewidth=1.3, zorder=2))
        ax.text(cx, y_relu + MH / 2, "ReLU", ha="center", va="center",
                fontsize=8.4, color="#26323c", zorder=3)
        ax.text(bx - DASH - 0.16, y_relu + MH / 2, shp_out, ha="right",
                va="center", fontsize=8.0, color="#6b7883")

        ax.text(bx + MW / 2, y_relu + MH + 0.95, ttl, ha="center", va="bottom",
                fontsize=11.5, weight="bold", color="#8a6d1a")
        ax.text(bx + MW / 2, y_relu + MH + 0.48, sub, ha="center", va="bottom",
                fontsize=8.4, color="#8a6d1a")

    if note:
        ax.text(0, y_x - 0.75, note, ha="left", va="top", fontsize=8.2,
                color="#6b7883")
    ax.text(0, y_x - 1.35, caption, ha="left", va="top", fontsize=8.2,
            color="#8a6d1a")
    # 지름길 상자는 본줄기 오른쪽 끝에서 SKIPX + SKIPW 만큼 더 나간다.
    # patch 는 기본으로 axes 에 잘리므로 xlim 이 그것을 다 담아야 한다.
    ax.set_xlim(-0.4, (len(panels) - 1) * PANEL + SHPW + MW + SKIPX
                + SKIPW + 0.6)
    ax.set_ylim(y_x - 2.3, y_relu + MH + 2.0)
    ax.axis("off")
    fig.savefig(svg, transparent=True, bbox_inches="tight")
    plt.close(fig)
    print("  " + svg)


RESIDUAL_PANELS = [
    ("x is added untouched", "layer1.0 -- shapes already match",
     [("3x3 conv, stride 1", CONV, "64x56x56"),
      ("BatchNorm",          NORM, None),
      ("ReLU",               NORM, None),
      ("3x3 conv, stride 1", CONV, None),
      ("BatchNorm",          NORM, "64x56x56")],
     None, "64x56x56"),
    ("x has to be reshaped first", "layer2.0 -- channels double, side halves",
     [("3x3 conv, stride 2", CONV, "64x56x56"),
      ("BatchNorm",          NORM, "128x28x28"),
      ("ReLU",               NORM, None),
      ("3x3 conv, stride 1", CONV, None),
      ("BatchNorm",          NORM, "128x28x28")],
     "1x1 conv, stride 2\n+ BatchNorm", "128x28x28"),
]


# =============================================================================
# Inception 블록만 그리는 법이 다르다.
#
# draw_block 은 칸마다 화살표를 이어 붙이는 **한 줄 쌓기**다. 그것으로 Inception
# 을 그리면 네 갈래가 차례로 이어진 것처럼 보인다 -- 첫 갈래가 둘째 갈래로
# 들어가는 것처럼. 실제로는 넷이 같은 입력을 나란히 받아 concat 에서야 처음
# 만나므로, 그 그림은 없는 차례를 하나 지어내는 셈이다. 블록의 생각 자체가
# "크기를 고르지 않고 다 해 본다" 인데 그것이 바로 가려진다.
#
# 그래서 갈래를 **가로로** 늘어놓는 함수를 따로 둔다.
# =============================================================================
BW, BH, BGAP, CGAP = 3.3, 0.72, 0.52, 0.42   # 갈래 상자 크기와 사이

# 갈래를 지나는 중간 칸은 옅게, 그 갈래가 내놓는 칸은 짙게 칠한다. 더해서
# 256 이 되는 네 수가 어느 칸에서 나오는지 눈으로 짚을 수 있어야 한다.
MID = ("#fdf3ea", "#e0a877")


def draw_inception_block(svg, title_text, subtitle, branches, caption,
                         shp_in, shp_out, note=None):
    """branches: [[(글, 색), ...], ...] -- 갈래마다 아래에서 위로 쌓는다.

    맨 윗칸이 그 갈래가 내놓는 것이고, 그 채널 수가 concat 에서 더해진다.
    """
    ncol = len(branches)
    nrow = max(len(b) for b in branches)
    pitch_x, pitch_y = BW + CGAP, BH + BGAP
    width = ncol * BW + (ncol - 1) * CGAP
    fig, ax = plt.subplots(figsize=(11.0, 2.2 + 0.72 * (nrow + 2)))

    y_in, y_cat = -pitch_y, nrow * pitch_y

    def bar(y, txt, col, shp):
        fill, edge = col
        ax.add_patch(plt.Rectangle((0, y), width, BH, facecolor=fill,
                                   edgecolor=edge, linewidth=1.3))
        ax.text(width / 2, y + BH / 2, txt, ha="center", va="center",
                fontsize=9.5, color="#26323c")
        ax.text(width + 0.3, y + BH / 2, shp, ha="left", va="center",
                fontsize=8.2, color="#6b7883")

    bar(y_in, "Input", INP, shp_in)
    bar(y_cat, "Concatenation", SPEC, shp_out)

    def arrow(xc, y0, y1):
        ax.add_patch(FancyArrowPatch((xc, y0 + 0.04), (xc, y1 - 0.04),
                                     arrowstyle="->", color="#8a97a4", lw=1.0,
                                     mutation_scale=10, zorder=3))

    for k, boxes in enumerate(branches):
        x = k * pitch_x
        xc = x + BW / 2
        for r, (txt, col) in enumerate(boxes):
            y = r * pitch_y
            fill, edge = col
            ax.add_patch(plt.Rectangle((x, y), BW, BH, facecolor=fill,
                                       edgecolor=edge, linewidth=1.3))
            ax.text(xc, y + BH / 2, txt, ha="center", va="center",
                    fontsize=8.4, color="#26323c")
            if r:
                arrow(xc, y - BGAP, y)           # 갈래 안에서 위로
        arrow(xc, y_in + BH, 0.0)                # 입력에서 갈라진다
        arrow(xc, (len(boxes) - 1) * pitch_y + BH, y_cat)   # concat 으로 모인다

    ax.text(width / 2, y_cat + BH + 0.95, title_text, ha="center", va="bottom",
            fontsize=12.5, weight="bold", color="#8a6d1a")
    ax.text(width / 2, y_cat + BH + 0.45, subtitle, ha="center", va="bottom",
            fontsize=8.8, color="#8a6d1a")
    ax.text(width + 0.3, y_cat + BH + 0.45, "shape", ha="left", va="bottom",
            fontsize=8.2, color="#8a97a4")
    if note:
        ax.text(0, y_in - 0.75, note, ha="left", va="top", fontsize=8.2,
                color="#6b7883")
    ax.text(0, y_in - 1.35, caption, ha="left", va="top", fontsize=8.2,
            color="#8a6d1a")
    ax.set_xlim(-0.5, width + 3.9)
    ax.set_ylim(y_in - 2.3, y_cat + BH + 1.9)
    ax.axis("off")
    fig.savefig(svg, transparent=True, bbox_inches="tight")
    plt.close(fig)
    print("  " + svg)


# Mixed_5b = InceptionA(192, pool_features=32). 갈래 차례는 torchvision 이
# concat 하는 차례 그대로다: branch1x1, branch5x5, branch3x3dbl, branch_pool.
INCEPTION_BRANCHES = [
    [("1x1 conv, 64", CONV)],
    [("1x1 conv, 48", MID),
     ("5x5 conv, pad 2, 64", CONV)],
    [("1x1 conv, 64", MID),
     ("3x3 conv, pad 1, 96", MID),
     ("3x3 conv, pad 1, 96", CONV)],
    [("3x3 avgpool, pad 1", POOL),
     ("1x1 conv, 32", CONV)],
]


BLOCKS = [
 ("mobilenet_block.svg", "InvertedResidual", "inside features.12 of MobileNetV3-L, at input 224",
  [("1x1 conv  (expand)",         CONV, "112x14x14", "672x14x14"),
   ("3x3 depthwise  (space)",     SPEC, None,        "672x14x14"),
   ("SE gate   672 -> 168 -> 672",  NORM, None,       "672x14x14"),
   ("1x1 conv  (project)",        CONV, None,        "112x14x14"),
   ("+   (add x)",                SPEC, None,        "112x14x14")],
  "112 -> 672 -> 112 : the middle swells, which is what inverted means",
  "space is seen only by the 3x3 depthwise; channels are mixed only by the two 1x1s",
  (0, 4, "identity x")),

 ("convnext_block.svg", "ConvNeXt block", "inside features.5 of ConvNeXt-T, at input 224",
  [("7x7 depthwise conv", SPEC, "384x14x14", "384x14x14"),
   ("LayerNorm",          NORM, None,        "384x14x14"),
   ("1x1  (Linear)  4x",  CONV, None,        "1536x14x14"),
   ("GELU",               NORM, None,        "1536x14x14"),
   ("1x1  (Linear)  back",CONV, None,        "384x14x14"),
   ("+   (add x)",        SPEC, None,        "384x14x14")],
  "384 -> 1536 -> 384 : the same swollen waist as MobileNet, but a 7x7 kernel and LayerNorm",
  "one activation per block, not one per convolution",
  (0, 5, "identity x")),

 ("efficientnet_block.svg", "MBConv", "inside features.5[1] of EfficientNet-B0, at input 224",
  [("1x1 conv  (expand)",         CONV, "112x14x14", "672x14x14"),
   ("5x5 depthwise  (space)",     SPEC, None,        "672x14x14"),
   ("SE gate   672 -> 28 -> 672",   NORM, None,       "672x14x14"),
   ("1x1 conv  (project)",        CONV, None,        "112x14x14"),
   ("+   (add x)",                SPEC, None,        "112x14x14")],
  "112 -> 672 -> 112 : same skeleton as MobileNet, but the kernel and the SE waist differ",
  "compound scaling decides how many of these and how wide -- the block itself is borrowed",
  (0, 4, "identity x")),
 ("vit_block.svg", "Transformer block", "one block of ViT-B/16, at input 224",
  [("LayerNorm",                  NORM, "197x768", "197x768"),
   ("Multi-Head Attention, 12 heads", SPEC, None,   "197x768"),
   ("+   (add x)",                SPEC, None,      "197x768"),
   ("LayerNorm",                  NORM, None,      "197x768"),
   ("MLP  768 -> 3072 -> 768",    CONV, None,      "197x768"),
   ("+   (add h)",                SPEC, None,      "197x768")],
  "still 197x768 after twelve of these -- nothing here shrinks",
  "where a CNN would pool the space down, this has no such step",
  [(0, 2, "x"), (3, 5, "h")]),
]

RUNS = [
 ("resnet18", "resnet18_layers.svg", "layer1.0", inset_resnet,
  [("input 224", "the usual", "#2f6b3a"), ("input 32", "runs, silently", "#a06a10"),
   ("input 448", "runs too", "#5b6b7d")],
  "every stride-2 block halves the side; global pooling then flattens whatever arrives"),
 ("inception", "inception_layers.svg", "Mixed_5b", inset_inception,
  [("input 299", "the usual", "#2f6b3a"), ("input 32", "dies at Mixed_6a", "#c44f4f"),
   ("input 598", "runs", "#5b6b7d")],
  "at 32 the stem already bottoms out at 1x1; three blocks survive, the fourth raises"),
 ("mobilenet", "mobilenet_layers.svg", "features.12", inset_mobilenet,
  [("input 224", "the usual", "#2f6b3a"), ("input 32", "runs, silently", "#a06a10"),
   ("input 448", "runs too", "#5b6b7d")],
  "depthwise holds 3% of the weights; the 1x1s hold 97%"),
 ("efficientnet", "efficientnet_layers.svg", "features.5", inset_efficientnet,
  [("input 224", "the usual", "#2f6b3a"), ("input 32", "runs, silently", "#a06a10"),
   ("input 448", "runs too", "#5b6b7d")],
  "4.4 measured depth alone stalling: 4 layers 76.74%, 6 layers 75.74%"),
 ("densenet", "densenet_layers.svg", "denseblock1", inset_densenet,
  [("input 224", "the usual", "#2f6b3a"), ("input 32", "runs, silently", "#a06a10"),
   ("input 448", "runs too", "#5b6b7d")],
  "6.95M parameters, yet 552s to extract -- 3.2x MobileNetV3-L, which is smaller still"),
 ("convnext", "convnext_layers.svg", "features.5", inset_convnext,
  [("input 224", "the usual", "#2f6b3a"), ("input 32", "runs, silently", "#a06a10"),
   ("input 448", "runs too", "#5b6b7d")],
  "no attention anywhere, and it covers 88% of the climb from VGG16 to ViT"),
 ("vit", "vit_layers.svg", "conv_proj", inset_vit,
  [("input 224", "the only one that runs", "#2f6b3a"),
   ("input 32", "AssertionError", "#c44f4f"), ("input 448", "AssertionError", "#c44f4f")],
  "the shape never changes after patchify: 197x768 all the way up"),
]

if __name__ == "__main__":
    for args in RUNS:
        draw(*args)
    for args in BLOCKS:
        draw_block(*args[:5], note=args[5], skip=args[6] if len(args) > 6 else None)
    draw_attention(
        "vit_attention.svg", "Multi-Head Attention",
        "inside one Transformer block of ViT-B/16, at input 224",
        "the only shape here that is not 197x768 is the 197x197 matrix, "
        "and it is the one that grows as tokens squared",
        note="every token looks at every token; 12 heads do it side by side "
             "on 64 dimensions each, then the results are concatenated back to 768")
    draw_dense_block(
        "densenet_block.svg", "DenseBlock",
        "inside denseblock1 of DenseNet121, all six layers, at input 224",
        64, 32, 6, "1x1 conv, 128   ->   3x3 conv, 32",
        "64 + 6x32 = 256, which is exactly the denseblock1 row in the layer list",
        note="every layer reads all the channels below it and appends 32 more; "
             "nothing is summed, so the bar only grows. the 1x1 always hands the "
             "3x3 exactly 128 channels, however long the bar has grown")
    draw_residual_block(
        "resnet18_block.svg", RESIDUAL_PANELS,
        "F(x) is what the block learns; x reaches the add either untouched or "
        "through one 1x1",
        note="the add is the only place two lines meet, and ReLU comes after "
             "it, not before -- so nothing clips x on its way through")
    draw_inception_block(
        "inception_block.svg", "Inception block",
        "inside Mixed_5b of Inception v3, at input 299",
        INCEPTION_BRANCHES,
        "the four meet only here: 64 + 64 + 96 + 32 = 256 channels",
        "192x35x35", "256x35x35",
        note="all four read the same input at the same time; "
             "pale boxes are on the way, solid boxes are what a branch hands over")
