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

    ix = BW + 3 * COL + 0.9
    iy = max(0.0, spec_y - ih / 2 + BH / 2)
    ax.add_patch(plt.Rectangle((ix, iy), iw, ih, facecolor="#fffdf5",
                               edgecolor="#c9a227", linewidth=1.3, zorder=1))
    inset_fn(ax, ix, iy, iw, ih)
    arrow(ax, (BW + 3 * COL - 0.4, spec_y + BH / 2), (ix - 0.12, spec_y + BH / 2),
          color="#c9a227", lw=1.2)

    ax.text(-2.9, -1.05, caption, ha="left", va="bottom",
            fontsize=7.4, color="#8a6d1a")
    ax.set_xlim(-3.2, ix + iw + 0.6)
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
            arrow(ax, (cx, iy + 0.83 + k * 0.80), (cx, iy + 0.60 + k * 0.80))


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


RUNS = [
 ("resnet18", "resnet18_layers.svg", "layer1.0", inset_resnet,
  [("input 224", "the usual", "#2f6b3a"), ("input 32", "runs, silently", "#a06a10"),
   ("input 448", "runs too", "#5b6b7d")],
  "every stride-2 block halves the side; global pooling then flattens whatever arrives"),
 ("inception", "inception_layers.svg", "Mixed_5b", inset_inception,
  [("input 299", "the usual", "#2f6b3a"), ("input 32", "dies at Mixed_6a", "#c44f4f"),
   ("input 598", "runs", "#5b6b7d")],
  "at 32 the stem already bottoms out at 1x1; three blocks survive, the fourth raises"),
 ("mobilenet", "mobilenet_layers.svg", "features.7", inset_mobilenet,
  [("input 224", "the usual", "#2f6b3a"), ("input 32", "runs, silently", "#a06a10"),
   ("input 448", "runs too", "#5b6b7d")],
  "depthwise holds 3% of the weights; the 1x1s hold 97%"),
 ("efficientnet", "efficientnet_layers.svg", "features.5", inset_efficientnet,
  [("input 224", "the usual", "#2f6b3a"), ("input 32", "runs, silently", "#a06a10"),
   ("input 448", "runs too", "#5b6b7d")],
  "4.4 measured depth alone stalling: 4 layers 76.74%, 6 layers 75.74%"),
 ("vit", "vit_layers.svg", "conv_proj", inset_vit,
  [("input 224", "the only one that runs", "#2f6b3a"),
   ("input 32", "AssertionError", "#c44f4f"), ("input 448", "AssertionError", "#c44f4f")],
  "the shape never changes after patchify: 197x768 all the way up"),
]

if __name__ == "__main__":
    for args in RUNS:
        draw(*args)
