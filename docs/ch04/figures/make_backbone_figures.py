"""4.6절 전이 companion 쪽들의 등뼈 그림을 만든다.

VGG16 그림(vgg16_layers.svg)과 같은 꼴이다. 왼쪽에 층을 아래에서 위로 쌓고,
오른쪽에 텐서 모양을 적는다. 다른 점은 **모델마다 그 모델을 그 모델이게 하는
블록 하나를 노란색으로 짚고, 옆에 그 속을 펼쳐 그린다**는 것이다.

그림 규칙(CLAUDE.md):
    - SVG, svg.fonttype='path', transparent=True
    - 그림 안의 글자는 모두 ASCII (기본 글꼴에 한글이 없다)

실행:
    python make_backbone_figures.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

matplotlib.rcParams["svg.fonttype"] = "path"

# 색은 vgg16_layers.svg 와 같은 갈래를 쓴다
CONV = ("#fce4d2", "#d4813f")      # 합성곱
POOL = ("#dbe7f7", "#4a72b8")      # 풀링
FCL  = ("#dfeadb", "#4e7a43")      # 완전 연결
SMAX = ("#fbdada", "#c44f4f")      # 소프트맥스
INP  = ("#e6e6e6", "#333333")      # 입력
SPEC = ("#fdf0c8", "#c9a227")      # 그 모델의 고갱이 블록
NORM = ("#efe7f7", "#8e6bb5")      # 정규화·어텐션 따위
HL   = "#fff3cd"                   # 짚는 칸 바탕

BW, BH, GAP = 6.0, 0.70, 0.16


def stack(ax, rows, x0=0.0):
    """rows: (왼쪽 이름, 상자 글, 오른쪽 모양, 색, 고갱이인가)

    아래에서 위로 쌓고, 고갱이 줄은 바탕을 칠한다. 고갱이 줄의 y를 돌려준다.
    """
    spec_y = None
    for i, (name, mid, shape, (fill, edge), is_spec) in enumerate(rows):
        y = i * (BH + GAP)
        if is_spec:
            ax.add_patch(plt.Rectangle((x0 - 2.7, y - GAP / 2), BW + 6.6, BH + GAP,
                                       facecolor=HL, edgecolor="none", zorder=0))
            spec_y = y
        ax.add_patch(plt.Rectangle((x0, y), BW, BH, facecolor=fill,
                                   edgecolor=edge, linewidth=1.2))
        ax.text(x0 + BW / 2, y + BH / 2, mid, ha="center", va="center",
                fontsize=8.4, color="#26323c")
        if name:
            ax.text(x0 - 0.26, y + BH / 2, name, ha="right", va="center",
                    fontsize=8.2, color="#26323c")
        ax.text(x0 + BW + 0.26, y + BH / 2, shape, ha="left", va="center",
                fontsize=7.8, color="#6b7883")
    return spec_y, len(rows) * (BH + GAP)


def inset(ax, x, y, w, h, title):
    """고갱이 블록의 속을 그릴 판을 둔다."""
    ax.add_patch(plt.Rectangle((x, y), w, h, facecolor="#fffdf5",
                               edgecolor="#c9a227", linewidth=1.3, zorder=1))
    ax.text(x + w / 2, y + h - 0.32, title, ha="center", va="top",
            fontsize=8.6, weight="bold", color="#8a6d1a", zorder=2)


def node(ax, x, y, w, h, text, colors=CONV, fs=7.2):
    fill, edge = colors
    ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=fill, edgecolor=edge,
                               linewidth=1.0, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color="#26323c", zorder=3)


def arrow(ax, p, q, color="#8a97a4", style="->", lw=0.9, rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, color=color, lw=lw,
                                 mutation_scale=9,
                                 connectionstyle=f"arc3,rad={rad}", zorder=3))


def finish(ax, fig, name, xlim, ylim, caption=None):
    if caption:
        ax.text(xlim[0] + 0.3, ylim[0] + 0.35, caption, ha="left", va="bottom",
                fontsize=7.6, color="#8a6d1a")
    ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.axis("off")
    fig.savefig(name, transparent=True, bbox_inches="tight")
    plt.close(fig)
    print("  " + name)


# =============================================================================
# ResNet18 — 지름길
# =============================================================================
def fig_resnet():
    rows = [
        ("",        "Input",                 "3x224x224",   INP,  False),
        ("conv1",   "7x7 conv, 64, stride 2","64x112x112",  CONV, False),
        ("",        "MaxPool 3x3, stride 2", "64x56x56",    POOL, False),
        ("layer1",  "BasicBlock x2, 64",     "64x56x56",    SPEC, True),
        ("layer2",  "BasicBlock x2, 128",    "128x28x28",   CONV, False),
        ("layer3",  "BasicBlock x2, 256",    "256x14x14",   CONV, False),
        ("layer4",  "BasicBlock x2, 512",    "512x7x7",     CONV, False),
        ("",        "Global AvgPool",        "512x1x1",     POOL, False),
        ("fc",      "FC 1000",               "1000",        FCL,  False),
    ]
    fig, ax = plt.subplots(figsize=(11.6, 6.2))
    sy, top = stack(ax, rows)

    ix, iy, iw, ih = BW + 4.6, sy - 2.1, 5.0, 4.6
    inset(ax, ix, iy, iw, ih, "BasicBlock")
    cx = ix + iw / 2
    node(ax, cx - 1.5, iy + 3.05, 3.0, 0.5, "3x3 conv - BN - ReLU")
    node(ax, cx - 1.5, iy + 2.25, 3.0, 0.5, "3x3 conv - BN")
    node(ax, cx - 0.7, iy + 1.35, 1.4, 0.5, "+", ("#fdf0c8", "#c9a227"), fs=10)
    node(ax, cx - 1.0, iy + 0.55, 2.0, 0.5, "ReLU", NORM)
    arrow(ax, (cx, iy + 3.90), (cx, iy + 3.58))
    arrow(ax, (cx, iy + 3.03), (cx, iy + 2.77))
    arrow(ax, (cx, iy + 2.23), (cx, iy + 1.87))
    arrow(ax, (cx, iy + 1.33), (cx, iy + 1.07))
    ax.text(cx - 0.18, iy + 3.90, "x", ha="right", va="center", fontsize=8.5,
            color="#26323c", zorder=3)
    # 지름길
    arrow(ax, (cx + 1.6, iy + 3.95), (cx + 0.75, iy + 1.62),
          color="#c9a227", lw=1.9, rad=-0.42)
    ax.text(cx + 2.35, iy + 2.7, "identity\nshortcut", ha="center", va="center",
            fontsize=7.4, color="#8a6d1a", zorder=3)
    arrow(ax, (BW + 1.9, sy + BH / 2), (ix - 0.15, sy + BH / 2), color="#c9a227", lw=1.2)

    finish(ax, fig, "resnet18_layers.svg", (-3.0, ix + iw + 1.2), (-1.5, top + 0.6),
           caption="the block learns F(x); x is added back unchanged")


# =============================================================================
# Inception v3 — 여러 크기를 한 층에서
# =============================================================================
def fig_inception():
    rows = [
        ("",         "Input",                "3x299x299",  INP,  False),
        ("stem",     "conv x5 + pool x2",    "192x35x35",  CONV, False),
        ("Mixed_5b", "Inception block x3",   "288x35x35",  SPEC, True),
        ("Mixed_6a", "Inception block x5",   "768x17x17",  CONV, False),
        ("Mixed_7a", "Inception block x3",   "2048x8x8",   CONV, False),
        ("",         "Global AvgPool",       "2048x1x1",   POOL, False),
        ("fc",       "FC 1000",              "1000",       FCL,  False),
    ]
    fig, ax = plt.subplots(figsize=(12.6, 5.6))
    sy, top = stack(ax, rows)

    ix, iy, iw, ih = BW + 4.6, sy - 1.7, 6.4, 4.2
    inset(ax, ix, iy, iw, ih, "Inception block (Mixed_5b)")
    cx = ix + iw / 2
    ax.text(cx, iy + 0.45, "192 channels in", ha="center", fontsize=7.4,
            color="#6b7883", zorder=3)
    branches = [("1x1\n64", 64), ("1x1 48\n5x5 64", 64), ("1x1 64\n3x3 x2 96", 96),
                ("pool\n1x1 32", 32)]
    w = 1.35
    for k, (txt, ch) in enumerate(branches):
        bx = ix + 0.3 + k * (w + 0.19)
        node(ax, bx, iy + 1.15, w, 1.55, txt, CONV, fs=6.8)
        arrow(ax, (cx, iy + 0.72), (bx + w / 2, iy + 1.12), rad=0.12)
        arrow(ax, (bx + w / 2, iy + 2.74), (cx, iy + 3.12), rad=0.12)
    node(ax, cx - 1.7, iy + 3.15, 3.4, 0.5, "concat -> 256", ("#fdf0c8", "#c9a227"))
    ax.text(cx, iy + 3.05, "", ha="center")
    arrow(ax, (BW + 1.9, sy + BH / 2), (ix - 0.15, sy + BH / 2), color="#c9a227", lw=1.2)

    finish(ax, fig, "inception_layers.svg", (-3.0, ix + iw + 1.2), (-1.5, top + 0.6),
           caption="four filter sizes run in parallel; 1x1 shrinks channels first")


# =============================================================================
# MobileNetV3 — 합성곱을 둘로 쪼갠다
# =============================================================================
def fig_mobilenet():
    rows = [
        ("",       "Input",                    "3x224x224",   INP,  False),
        ("stem",   "3x3 conv, 16, stride 2",   "16x112x112",  CONV, False),
        ("blocks", "InvertedResidual x15",     "160x7x7",     SPEC, True),
        ("",       "1x1 conv, 960",            "960x7x7",     CONV, False),
        ("",       "Global AvgPool",           "960x1x1",     POOL, False),
        ("",       "FC 1280 - Hardswish",      "1280",        FCL,  False),
        ("cls[3]", "FC 1000",                  "1000",        FCL,  False),
    ]
    fig, ax = plt.subplots(figsize=(11.8, 5.6))
    sy, top = stack(ax, rows)

    ix, iy, iw, ih = BW + 4.6, sy - 1.8, 5.2, 4.4
    inset(ax, ix, iy, iw, ih, "InvertedResidual")
    cx = ix + iw / 2
    steps = [("1x1 conv  (expand)", CONV), ("3x3 depthwise  (space)", SPEC),
             ("Squeeze-Excitation", NORM), ("1x1 conv  (project)", CONV)]
    for k, (txt, col) in enumerate(steps):
        node(ax, cx - 1.9, iy + 0.55 + k * 0.72, 3.8, 0.52, txt, col, fs=7.0)
        if k:
            arrow(ax, (cx, iy + 0.53 + k * 0.72), (cx, iy + 0.30 + k * 0.72))
    ax.text(cx, iy + 3.62, "channels mixed only by the 1x1s",
            ha="center", fontsize=7.2, color="#8a6d1a", zorder=3)
    arrow(ax, (BW + 1.9, sy + BH / 2), (ix - 0.15, sy + BH / 2), color="#c9a227", lw=1.2)

    finish(ax, fig, "mobilenet_layers.svg", (-3.0, ix + iw + 1.2), (-1.5, top + 0.6),
           caption="depthwise holds 3% of the weights; the 1x1s hold 97%")


# =============================================================================
# EfficientNet-B0 — 세 축
# =============================================================================
def fig_efficientnet():
    rows = [
        ("",       "Input",                  "3x224x224",   INP,  False),
        ("stem",   "3x3 conv, 32, stride 2", "32x112x112",  CONV, False),
        ("s1-s2",  "MBConv x3",              "24x56x56",    CONV, False),
        ("s3-s5",  "MBConv x9",              "112x14x14",   SPEC, True),
        ("s6-s7",  "MBConv x4",              "320x7x7",     CONV, False),
        ("",       "1x1 conv, 1280",         "1280x7x7",    CONV, False),
        ("",       "Global AvgPool",         "1280",        POOL, False),
        ("cls[1]", "FC 1000",                "1000",        FCL,  False),
    ]
    fig, ax = plt.subplots(figsize=(12.2, 6.0))
    sy, top = stack(ax, rows)

    ix, iy, iw, ih = BW + 4.6, sy - 2.0, 5.6, 4.8
    inset(ax, ix, iy, iw, ih, "compound scaling")
    cx = ix + iw / 2
    axes_ = [("depth  d", "more MBConv blocks", 0),
             ("width  w", "more channels", 1),
             ("resolution  r", "larger input", 2)]
    for k, (nm, why, i) in enumerate(axes_):
        y = iy + 0.75 + k * 0.95
        node(ax, ix + 0.35, y, 1.9, 0.6, nm, SPEC, fs=7.4)
        ax.text(ix + 2.45, y + 0.3, why, ha="left", va="center",
                fontsize=7.0, color="#6b7883", zorder=3)
    ax.text(cx, iy + 3.72, "scaled together, not one at a time",
            ha="center", fontsize=7.4, color="#8a6d1a", zorder=3)
    ax.text(cx, iy + 0.42, "B0 -> B7 : one architecture, seven sizes",
            ha="center", fontsize=7.0, color="#8a6d1a", zorder=3)
    arrow(ax, (BW + 1.9, sy + BH / 2), (ix - 0.15, sy + BH / 2), color="#c9a227", lw=1.2)

    finish(ax, fig, "efficientnet_layers.svg", (-3.0, ix + iw + 1.2), (-1.5, top + 0.6),
           caption="4.4 measured depth alone stalling: 4 layers 76.74%, 6 layers 75.74%")


# =============================================================================
# ViT-B/16 — 조각과 어텐션
# =============================================================================
def fig_vit():
    rows = [
        ("",         "Input",                    "3x224x224",  INP,  False),
        ("conv_proj","16x16 conv, 768, stride 16","768x14x14", SPEC, True),
        ("",         "flatten + class token",    "197x768",    NORM, False),
        ("",         "+ position embedding",     "197x768",    NORM, False),
        ("encoder",  "Transformer block x12",    "197x768",    CONV, False),
        ("",         "LayerNorm, take token 0",  "768",        NORM, False),
        ("heads",    "FC 1000",                  "1000",       FCL,  False),
    ]
    fig, ax = plt.subplots(figsize=(12.4, 5.8))
    sy, top = stack(ax, rows)

    ix, iy, iw, ih = BW + 4.6, sy - 1.5, 6.0, 4.4
    inset(ax, ix, iy, iw, ih, "patchify  =  strided convolution")
    cx = ix + iw / 2
    # 왼쪽: 그림을 조각으로
    gx, gy, cell = ix + 0.5, iy + 1.5, 0.36
    for r in range(4):
        for c in range(4):
            node(ax, gx + c * cell, gy + r * cell, cell * 0.92, cell * 0.92, "",
                 ("#e9eef5", "#6b7883"))
    ax.text(gx + 2 * cell, gy - 0.28, "224x224", ha="center", fontsize=7.0,
            color="#6b7883", zorder=3)
    ax.text(gx + 2 * cell, gy + 4 * cell + 0.12, "16x16 patches", ha="center",
            fontsize=7.0, color="#6b7883", zorder=3)
    # 오른쪽: 토큰 줄
    tx = ix + 3.1
    for k in range(4):
        node(ax, tx, iy + 1.1 + k * 0.52, 2.1, 0.42,
             ["token 196", "...", "token 2", "token 1"][k], NORM, fs=6.8)
    node(ax, tx, iy + 0.55, 2.1, 0.42, "class token", ("#fdf0c8", "#c9a227"), fs=6.8)
    arrow(ax, (gx + 4 * cell + 0.15, gy + 2 * cell), (tx - 0.12, iy + 1.7))
    ax.text(cx + 0.4, iy + 3.62, "196 + 1 = 197 tokens, 768 wide",
            ha="center", fontsize=7.4, color="#8a6d1a", zorder=3)
    arrow(ax, (BW + 1.9, sy + BH / 2), (ix - 0.15, sy + BH / 2), color="#c9a227", lw=1.2)

    finish(ax, fig, "vit_layers.svg", (-3.0, ix + iw + 1.2), (-1.5, top + 0.6),
           caption="no locality is built in; attention sees all 197 tokens from block 1")


if __name__ == "__main__":
    fig_resnet()
    fig_inception()
    fig_mobilenet()
    fig_efficientnet()
    fig_vit()
