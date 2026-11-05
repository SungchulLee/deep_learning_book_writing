"""1.6 자동 미분의 그림 — 계산 그래프를 한 가지 시각 언어로 그린다.

이 절은 29쪽이 모두 같은 그림의 변형을 그린다. 그래서 좌표를 쪽마다 손으로
찍지 않고 아래의 어휘를 돌려 쓴다.

  node(ax, x, y, label, kind)   잎 / 중간 텐서 / 연산 / 끊긴 것
  fwd(ax, a, b)                 순전파 — 실선, 잉크색, 왼쪽에서 오른쪽
  bwd(ax, a, b, label)          역전파 — 파선, 강조색, 오른쪽에서 왼쪽
  badge(ax, n, text, got)       그 텐서가 .grad 를 받았는가

색이 곧 갈래다.

  INK     구조와 순전파
  ACCENT  경사가 흐르는 자리 (리나인마 그림의 reverse 와 같은 색이다)
  GRAY    따라가지 않는 것, 끊긴 것, 못 받는 것

규칙은 CLAUDE.md 의 Figures 항에 적어 둔 대로다. 글자는 **모두 아스키**라야
한다 — matplotlib 기본 폰트에 한글이 없어 한글은 두부 상자로 나온다.
fonttype='path' 가 글자를 외곽선으로 구워 버리므로 SVG 를 grep 해도 안 잡힌다.

실행: python docs/ch06/autograd/figures/make_figures.py [그림이름 ...]
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["svg.fonttype"] = "path"

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = Path(__file__).resolve().parent

INK = "#2f3337"
ACCENT = "#8a4b2a"
GRAY = "#9aa3a9"
LEAF_FILL = "#eceff1"
GOT_FILL = "#f3e7de"

MONO = "monospace"


# ============================================================================
# 어휘 — 이 절의 그림은 모두 이 넷으로 그린다
# ============================================================================

class Node:
    """그림 위의 한 칸. 화살표는 좌표가 아니라 이 칸의 변을 받는다."""

    def __init__(self, x, y, w, h):
        self.x, self.y, self.w, self.h = x, y, w, h

    @property
    def left(self):
        return self.x - self.w / 2, self.y

    @property
    def right(self):
        return self.x + self.w / 2, self.y

    @property
    def top(self):
        return self.x, self.y + self.h / 2

    @property
    def bottom(self):
        return self.x, self.y - self.h / 2


KINDS = {
    # 잎 — 내가 만든 것. 테두리가 굵고 안이 채워져 있다.
    "leaf": dict(fc=LEAF_FILL, ec=INK, lw=1.7, ls="solid", tc=INK),
    # 중간 텐서 — 계산으로 생긴 것. 테두리만 있다.
    "tensor": dict(fc="none", ec=INK, lw=1.0, ls="solid", tc=INK),
    # 연산 — grad_fn 이 가리키는 것.
    "op": dict(fc="none", ec=ACCENT, lw=1.0, ls="solid", tc=ACCENT),
    # 따라가지 않는 것 — requires_grad=False, no_grad, detach 뒤.
    "off": dict(fc="none", ec=GRAY, lw=1.0, ls=(0, (3, 2)), tc=GRAY),
    # 바라던 것 — badge 의 받은 꼴과 같은 칠이다.
    "ok": dict(fc=GOT_FILL, ec=ACCENT, lw=1.2, ls="solid", tc=ACCENT),
}


def node(ax, x, y, label, kind="tensor", w=None, h=0.46, fontsize=9.5):
    """칸 하나를 그리고 Node 를 돌려준다. w 를 안 주면 글자 길이로 잡는다."""
    st = KINDS[kind]
    if w is None:
        # 여러 줄이면 가장 긴 줄로 잰다. len() 을 그대로 쓰면 줄바꿈까지
        # 세어 칸이 쓸데없이 넓어진다.
        w = max(0.62, 0.135 * max(len(s) for s in label.split("\n")) + 0.34)
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle="round,pad=0.035",
                                fc=st["fc"], ec=st["ec"], lw=st["lw"],
                                linestyle=st["ls"], zorder=3))
    ax.text(x, y, label, ha="center", va="center", fontsize=fontsize,
            family=MONO, color=st["tc"], zorder=4)
    return Node(x, y, w, h)


def row(ax, x0, y, items, gap=0.5, h=0.46, fontsize=9.5):
    """(label, kind) 목록을 왼쪽에서 오른쪽으로 놓는다.

    칸 너비가 저마다 다르므로 자리를 고정 간격으로 찍으면 긴 칸끼리 붙는다.
    너비에서 다음 자리를 계산해 간격을 일정하게 지킨다.
    """
    ns, x = [], x0
    for label, kind in items:
        w = max(0.62, 0.135 * len(label) + 0.34)
        ns.append(node(ax, x + w / 2, y, label, kind, w=w, h=h,
                       fontsize=fontsize))
        x += w + gap
    return ns


def _arrow(ax, p0, p1, color, ls, lw, rad=0.0):
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle="-|>", mutation_scale=11, color=color, lw=lw,
        linestyle=ls, connectionstyle=f"arc3,rad={rad}", zorder=2,
        shrinkA=2, shrinkB=2))


def fwd(ax, a, b, label=None, lw=1.3):
    """순전파 — 실선, a 의 오른쪽 변에서 b 의 왼쪽 변으로."""
    _arrow(ax, a.right, b.left, INK, "solid", lw)
    if label:
        ax.text((a.x + b.x) / 2, a.y + 0.2, label, ha="center", va="bottom",
                fontsize=8.2, family=MONO, color=INK)


def dead(ax, a, b):
    """끊긴 자리 — 파선 화살표에 x 표를 얹는다."""
    _arrow(ax, a.right, b.left, GRAY, (0, (3, 2)), 1.1)
    # 칸 중심이 아니라 **변**의 중점에 찍는다. 너비가 다른 두 칸 사이에서
    # 중심의 중점을 쓰면 x 표가 화살표 가운데를 벗어난다.
    mx, my = (a.right[0] + b.left[0]) / 2, a.y
    for dx, dy in ((-1, -1), (-1, 1)):
        ax.plot([mx - 0.07 * dx, mx + 0.07 * dx],
                [my - 0.07 * dy, my + 0.07 * dy],
                color=GRAY, lw=1.6, zorder=5, solid_capstyle="round")


def bwd(ax, a, b, label=None, drop=0.56, lw=1.3):
    """역전파 — 파선, a 의 아래에서 b 의 아래로 (오른쪽에서 왼쪽으로).

    칸 아래로 drop 만큼 내려 그린다. 순전파 줄과 겹치지 않게 하려는 것이다.
    """
    y = a.y - drop
    ax.plot([a.x, a.x], [a.bottom[1], y + 0.02], color=ACCENT, lw=0.8,
            linestyle=(0, (2, 2)), zorder=1)
    _arrow(ax, (a.x, y), (b.x, y), ACCENT, (0, (4, 2)), lw)
    ax.plot([b.x, b.x], [y, b.bottom[1] - 0.02], color=ACCENT, lw=0.8,
            linestyle=(0, (2, 2)), zorder=1)
    if label:
        ax.text((a.x + b.x) / 2, y - 0.1, label, ha="center", va="top",
                fontsize=8.2, family=MONO, color=ACCENT)


def loopback(ax, a, b, drop=0.42, color=None, label=None):
    """a 의 아래로 내려가 b 의 아래로 돌아오는 화살표 — 「한 바퀴」를 보인다.

    곡선 화살표로 바로 이으면 사이에 놓인 칸들을 가로지른다. 줄 아래로
    빼내어 ㄷ 자로 돌린다.
    """
    c = color or INK
    y = min(a.y - a.h / 2, b.y - b.h / 2) - drop
    ax.plot([a.x, a.x], [a.y - a.h / 2, y], color=c, lw=1.0,
            linestyle=(0, (3, 2)), zorder=1)
    ax.plot([a.x, b.x], [y, y], color=c, lw=1.0, linestyle=(0, (3, 2)),
            zorder=1)
    _arrow(ax, (b.x, y), (b.x, b.y - b.h / 2), c, (0, (3, 2)), 1.0)
    if label:
        ax.text((a.x + b.x) / 2, y - 0.1, label, ha="center", va="top",
                fontsize=8.2, color=c)


def badge(ax, n, text, got, dy=-0.52, fontsize=8.5):
    """그 텐서가 .grad 를 받았는가. 받으면 채운 칸, 못 받으면 파선 칸."""
    w = max(0.8, 0.118 * len(text) + 0.3)
    y = n.y + dy
    ax.add_patch(FancyBboxPatch((n.x - w / 2, y - 0.17), w, 0.34,
                                boxstyle="round,pad=0.03",
                                fc=GOT_FILL if got else "none",
                                ec=ACCENT if got else GRAY,
                                lw=1.2 if got else 1.0,
                                linestyle="solid" if got else (0, (3, 2)),
                                zorder=3))
    ax.text(n.x, y, text, ha="center", va="center", fontsize=fontsize,
            family=MONO, color=ACCENT if got else GRAY, zorder=4)


def flag(ax, n, value, dy=-0.4, name="requires_grad", size=7.8):
    """칸 아래에 깃발 값을 적는다 — 켜져 있으면 강조색, 꺼져 있으면 회색.

    `.grad` 를 받았는가를 말하는 badge 와는 다른 것이다. 이쪽은 **따라갈
    것인가**이고, 그래서 아래 칸이 아니라 글자 한 줄로 둔다.
    """
    ax.text(n.x, n.y + dy, f"{name}={value}", ha="center", va="center",
            fontsize=size, family=MONO, color=ACCENT if value else GRAY)


def region(ax, nodes, label, pad=0.26, color=None, below=False, size=8.4):
    """칸 몇을 한 테두리로 묶는다.

    `with torch.no_grad():` 블록, 얼린 층 묶음, 「그래프와 저장된 중간값」처럼
    **범위가 있는 것**을 그릴 때 쓴다. 범위를 말로만 적으면 어디까지인지가
    그림에 안 나온다.
    """
    c = color or INK
    x0 = min(n.x - n.w / 2 for n in nodes) - pad
    x1 = max(n.x + n.w / 2 for n in nodes) + pad
    y0 = min(n.y - n.h / 2 for n in nodes) - pad
    y1 = max(n.y + n.h / 2 for n in nodes) + pad
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                                boxstyle="round,pad=0.02", fc="none", ec=c,
                                lw=1.0, linestyle=(0, (4, 3)), zorder=1))
    if label:
        ax.text((x0 + x1) / 2, y0 - 0.12 if below else y1 + 0.12, label,
                ha="center", va="top" if below else "bottom", fontsize=size,
                color=c)
    return Node((x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0)


def note(ax, x, y, text, color=None, size=8.6, ha="center", va="center",
         italic=False):
    ax.text(x, y, text, ha=ha, va=va, fontsize=size, color=color or INK,
            style="italic" if italic else "normal")


def axes_style(ax, xlabel=None, ylabel=None):
    """재어서 그린 그래프의 축 — 도형 그림과 달리 눈금이 있어야 한다.

    위·오른쪽 테두리를 없애고 색과 글자 크기를 이 절의 것으로 맞춘다.
    """
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK)
        ax.spines[s].set_linewidth(0.9)
    ax.tick_params(colors=INK, labelsize=8.2, width=0.9)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=8.8, color=INK)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=8.8, color=INK)
    ax.grid(True, lw=0.5, color=GRAY, alpha=0.35)
    ax.set_axisbelow(True)


def frame(ax, xlim, ylim):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_axis_off()


def save(fig, name, foot=None, foot_color=None, foot_size=8.0, rect=0.045):
    if foot:
        fig.text(0.5, 0.005, foot, ha="center", fontsize=foot_size,
                 style="italic" if foot_color is None else "normal",
                 color=foot_color or INK)
        fig.tight_layout(rect=(0, rect, 1, 1))
    else:
        fig.tight_layout()
    fig.savefig(HERE / f"{name}.svg", transparent=True, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {name}.svg")


# ============================================================================
# 01_basic_scalar_backward — 네 속성, 잎과 중간, retain_grad 의 때
# ============================================================================

def fig_four_attributes():
    """네 속성이 한 그래프 위에서 각자 다른 일을 하는 것을 보인다.

    쪽 맨 앞의 표가 말로 적은 것을 그림으로 겹쳐 둔 것이다.
    """
    fig, ax = plt.subplots(figsize=(7.8, 3.5))

    x = node(ax, 0.0, 0.0, "x", "leaf")
    p = node(ax, 1.55, 0.0, "pow", "op")
    q = node(ax, 3.1, 0.0, "x**2", "tensor")
    s = node(ax, 4.65, 0.0, "sum", "op")
    L = node(ax, 6.2, 0.0, "loss", "tensor")
    for a, b in ((x, p), (p, q), (q, s), (s, L)):
        fwd(ax, a, b)

    note(ax, 3.1, 0.52, "forward builds the graph as it goes", INK, 8.8)

    rows = [
        ("requires_grad", ["True", "True", "True"]),
        ("is_leaf", ["True", "False", "False"]),
        ("grad_fn", ["None", "PowBackward0", "SumBackward0"]),
    ]
    cols = [x.x, q.x, L.x]
    y0 = -0.62
    for i, (name, vals) in enumerate(rows):
        y = y0 - 0.3 * i
        note(ax, -1.25, y, name, INK, 8.4, ha="right")
        for cx, v in zip(cols, vals):
            ax.text(cx, y, v, ha="center", va="center", fontsize=8.4,
                    family=MONO, color=INK)

    y = y0 - 0.3 * len(rows) - 0.1
    note(ax, -1.25, y, ".grad", ACCENT, 8.4, ha="right")
    for cx, v, got in zip(cols, ["2x", "None", "None"], [True, False, False]):
        badge(ax, Node(cx, y, 0, 0), v, got, dy=0)

    note(ax, 3.1, y - 0.42,
         "only the leaf keeps a .grad - the other two are computed and dropped",
         ACCENT, 8.6)

    frame(ax, (-3.0, 6.95), (y - 0.62, 0.78))
    save(fig, "four_attributes")


def fig_leaf_vs_nonleaf():
    """y 는 역전파에 참여하지만 .grad 를 받지 않는다 — 5절."""
    fig, ax = plt.subplots(figsize=(7.8, 3.2))

    x = node(ax, 0.0, 0.0, "x", "leaf")
    m = node(ax, 1.7, 0.0, "mul 2", "op")
    y = node(ax, 3.4, 0.0, "y", "tensor")
    o = node(ax, 5.1, 0.0, "pow, sum", "op")
    z = node(ax, 6.9, 0.0, "z", "tensor")
    for a, b in ((x, m), (m, y), (y, o), (o, z)):
        fwd(ax, a, b)
    note(ax, 3.45, 0.5, "x = [1, 2, 3]      y = 2x      z = sum(y**2)",
         INK, 8.8)

    bwd(ax, z, y, "dz/dy = 2y = [4, 8, 12]", drop=0.62)
    bwd(ax, y, x, "dz/dx = 2y * 2 = [8, 16, 24]", drop=0.62)
    note(ax, 7.55, -0.62, "dz/dz = 1", ACCENT, 8.2, ha="left")

    badge(ax, x, "x.grad = [8, 16, 24]", True, dy=-1.32)
    badge(ax, y, "y.grad = None", False, dy=-1.32)

    note(ax, 3.45, -1.78,
         "the gradient flows through y; it is simply not stored there",
         ACCENT, 8.8)

    frame(ax, (-1.7, 9.4), (-2.05, 0.75))
    save(fig, "leaf_vs_nonleaf")


def fig_retain_grad_timing():
    """retain_grad() 는 역전파 전에 불러야 한다 — 6절."""
    fig, ax = plt.subplots(figsize=(7.8, 2.6))

    for yy, steps, ok, out, msg in [
        (0.0, ["forward", "y.retain_grad()", "backward"], True,
         "y.grad = [4, 8, 12]",
         "asked before the sweep - the value is kept"),
        (-1.3, ["forward", "backward", "w.retain_grad()"], False,
         "w.grad = None",
         "asked after the sweep - nothing left to keep"),
    ]:
        kinds = [("off" if not ok else "op") if "retain" in s else "tensor"
                 for s in steps]
        ns = row(ax, 0.0, yy, list(zip(steps, kinds)), gap=0.52, h=0.42,
                 fontsize=9.0)
        for a, b in zip(ns, ns[1:]):
            fwd(ax, a, b, lw=1.1)
        tail = row(ax, ns[-1].right[0] + 0.52, yy, [(out, "tensor" if ok
                                                     else "off")],
                   h=0.42, fontsize=9.0)[0]
        _arrow(ax, ns[-1].right, tail.left,
               ACCENT if ok else GRAY, "solid" if ok else (0, (3, 2)), 1.2)
        note(ax, (ns[0].x + tail.x) / 2, yy + (0.47 if ok else -0.47), msg,
             ACCENT if ok else GRAY, 8.6)

    note(ax, -0.92, -0.65, "time", INK, 8.6, ha="right")
    _arrow(ax, (-0.6, -0.42), (-0.6, -0.9), INK, "solid", 1.1)

    frame(ax, (-1.85, 9.75), (-1.95, 0.72))
    save(fig, "retain_grad_timing")


# ============================================================================
# 02_gradient_accumulation — 그릇, 쪼개기, 지우는 두 길, 그리고 반대인 격자
# ============================================================================

def fig_grad_is_a_bucket():
    """`.grad` 는 덮어쓰는 칸이 아니라 더해 담는 그릇이다 — 2절."""
    fig, ax = plt.subplots(figsize=(7.8, 2.0))

    b0 = node(ax, 0.6, 0.0, "None", "off", w=1.1)
    b1 = node(ax, 3.5, 0.0, "4.0", "ok", w=1.1)
    b2 = node(ax, 6.4, 0.0, "7.0", "ok", w=1.1)
    for a, b, call, add in ((b0, b1, "loss1.backward()", "+ 2x = 4"),
                            (b1, b2, "loss2.backward()", "+ 3")):
        _arrow(ax, a.right, b.left, ACCENT, "solid", 1.3)
        note(ax, (a.x + b.x) / 2, 0.26, call, INK, 8.4)
        note(ax, (a.x + b.x) / 2, -0.28, add, ACCENT, 8.4)

    note(ax, -0.35, 0.0, "x.grad", INK, 9.0, ha="right")
    note(ax, 3.5, -0.72,
         "the second sweep does not erase the first - it adds to it",
         ACCENT, 8.8)

    frame(ax, (-1.6, 7.3), (-0.95, 0.5))
    save(fig, "grad_is_a_bucket")


def fig_micro_batching():
    """40개 한 번과 10개씩 네 번이 같아지는 조건 — 4절.

    mean 로스에서 /k 를 빼면 정확히 k 배가 된다. 그 빠진 칸을 그림에 둔다.
    """
    fig, ax = plt.subplots(figsize=(7.8, 3.8))

    g = "[0.582795, 0.494237, -1.425693]"
    XB, XG, YB = 5.0, 8.6, -0.275          # backward 칸, grad 칸, 아래 묶음의 가운데

    one = node(ax, 1.2, 1.2, "40 samples", "tensor", w=1.9)
    bw1 = node(ax, XB, 1.2, "backward()", "op", w=1.7)
    gr1 = node(ax, XG, 1.2, g, "ok", w=4.2, fontsize=8.4)
    fwd(ax, one, bw1)
    _arrow(ax, bw1.right, gr1.left, ACCENT, "solid", 1.3)
    note(ax, 0.1, 1.2, "one batch", INK, 8.8, ha="right")

    bw2 = node(ax, XB, YB, "backward()", "op", w=1.7)
    for yy in (0.25, -0.1, -0.45, -0.8):
        m = node(ax, 1.2, yy, "10", "tensor", w=0.8, h=0.27, fontsize=8.2)
        k = node(ax, 2.9, yy, "/ 4", "ok", w=0.7, h=0.27, fontsize=8.2)
        _arrow(ax, m.right, k.left, INK, "solid", 1.0)
        _arrow(ax, k.right, bw2.left, ACCENT, "solid", 1.0)
    gr2 = node(ax, XG, YB, g, "ok", w=4.2, fontsize=8.4)
    _arrow(ax, bw2.right, gr2.left, ACCENT, "solid", 1.3)
    note(ax, 0.1, YB, "four batches", INK, 8.8, ha="right")

    note(ax, 11.05, 0.46, "same", ACCENT, 9.5)
    for yy in (0.68, 0.24):
        ax.plot([10.8, 11.3], [yy, yy], color=ACCENT, lw=1.1)

    node(ax, 2.9, -1.52, "/ 4", "off", w=0.7, h=0.27, fontsize=8.2)
    note(ax, 2.35, -1.52, "drop it", GRAY, 8.6, ha="right")
    note(ax, 6.8, -1.52, "and the gradient comes out k = 4 times too large",
         GRAY, 8.6)

    note(ax, 5.1, -2.08,
         "sum loss needs no divider; mean loss does - this is the trap",
         ACCENT, 8.8)

    frame(ax, (-1.7, 12.0), (-2.4, 1.6))
    save(fig, "micro_batching")


def fig_zero_vs_none():
    """지우는 두 길이 걸음을 다르게 만든다 — 3절."""
    fig, ax = plt.subplots(figsize=(7.8, 2.4))

    # 회색은 이 절 내내 "아무것도 흐르지 않는다"는 뜻이다. 여기서 아무것도
    # 흐르지 않는 쪽은 None 이고, 그것이 파이토치 2.0 의 기본값이다.
    for yy, start, cut, verdict, tail in [
        (0.0, "p.grad = None", True, "step() is handed nothing and skips it",
         "-0.021486, +0.282214, -0.422172"),
        (-1.2, "p.grad = 0", False,
         "step() reads 'move by 0' - momentum keeps pushing",
         "-0.036934, +0.257463, -0.374460"),
    ]:
        a = node(ax, 1.15, yy, start, "off" if cut else "tensor", w=2.3)
        b = node(ax, 4.2, yy, "step()", "op", w=1.3)
        c = node(ax, 7.4, yy, tail, "tensor", w=4.4, fontsize=8.2)
        if cut:
            dead(ax, a, b)
        else:
            fwd(ax, a, b)
        _arrow(ax, b.right, c.left, ACCENT, "solid", 1.2)
        note(ax, 4.2, yy + 0.42, verdict, GRAY if cut else ACCENT, 8.4)

    note(ax, 5.0, -1.82,
         "one empty step apart, with momentum on: the two differ by 0.047712",
         INK, 8.8)
    note(ax, 5.0, -2.18,
         "pytorch 2.0 made the upper row the default - an erased gradient "
         "should mean 'nothing to say'", ACCENT, 8.6)

    frame(ax, (-0.3, 9.9), (-2.45, 0.68))
    save(fig, "zero_vs_none")


def fig_mean_sum_grid():
    """나누어야 하는가 — mean 과 sum 이 정확히 반대다 (연습문제 2)."""
    fig, ax = plt.subplots(figsize=(6.4, 2.7))

    cells = [  # (행, 열, 배수, 맞는가)
        (0, 0, "x 4.0", False), (0, 1, "x 1.0", True),
        (1, 0, "x 1.0", True), (1, 1, "x 0.25", False),
    ]
    cx = [1.55, 3.6]
    cy = [0.55, -0.35]
    for r, c, txt, ok in cells:
        node(ax, cx[c], cy[r], txt, "ok" if ok else "off", w=1.6, h=0.58,
             fontsize=10)

    for c, lab in enumerate(["no divider", "divide by k"]):
        note(ax, cx[c], 1.18, lab, INK, 9.0)
    for r, lab in enumerate(['reduction="mean"', 'reduction="sum"']):
        ax.text(0.55, cy[r], lab, ha="right", va="center", fontsize=9.0,
                family=MONO, color=INK)

    note(ax, 2.55, -1.02,
         "mean needs the divider and sum must not have it - exact opposites",
         ACCENT, 8.8)
    note(ax, 2.55, -1.42,
         "pytorch defaults to mean, so in practice you almost always divide",
         INK, 8.6)

    frame(ax, (-1.8, 4.7), (-1.65, 1.45))
    save(fig, "mean_sum_grid")


# ============================================================================
# 03_retain_graph — 해제되는 것은 그래프가 아니라 저장해 둔 값이다
# ============================================================================

def fig_saved_buffers():
    """두 번째 backward() 가 터지는 쪽과 안 터지는 쪽 — 2절.

    가르는 것은 식의 복잡함이 아니라 **역전파가 저장해 두어야 하는 값이
    있는가**다. 그래서 가운데 칸이 이 그림의 주인공이다.
    """
    fig, ax = plt.subplots(figsize=(7.8, 3.0))

    cols = [0.0, 3.3, 5.7, 8.3]
    for x, lab in zip(cols[1:], ["saved for backward", "after the 1st sweep",
                                 "2nd backward()"]):
        note(ax, x, 1.05, lab, INK, 8.6)

    rows = [
        ("x.sum()", "nothing", "off", "nothing", "off", "ok", "ok", False),
        ("(x**2).sum()", "x", "ok", "freed", "off", "RuntimeError", "off",
         True),
        ("x.exp().sum()", "exp(x)", "ok", "freed", "off", "RuntimeError",
         "off", True),
    ]
    for i, (expr, s, sk, a, ak, v, vk, cut) in enumerate(rows):
        y = 0.4 - 0.72 * i
        ax.text(cols[0], y, expr, ha="left", va="center", fontsize=9.2,
                family=MONO, color=INK)
        n1 = node(ax, cols[1], y, s, sk, w=1.5, h=0.42, fontsize=9.0)
        n2 = node(ax, cols[2], y, a, ak, w=1.5, h=0.42, fontsize=9.0)
        node(ax, cols[3], y, v, vk, w=2.4, h=0.42, fontsize=9.0)
        if cut:
            dead(ax, n1, n2)
        else:
            _arrow(ax, n1.right, n2.left, GRAY, (0, (3, 2)), 1.0)

    note(ax, 4.3, -1.85,
         "what the first sweep releases is the saved value, not the graph",
         ACCENT, 8.8)
    note(ax, 4.3, -2.22,
         "sum stores nothing to differentiate, so it survives a second sweep",
         INK, 8.6)

    frame(ax, (-0.5, 9.9), (-2.5, 1.35))
    save(fig, "saved_buffers")


def fig_retain_graph_vs_zero_grad():
    """둘은 다른 것을 맡는다 — 4절에서 가장 자주 섞이는 자리."""
    fig, ax = plt.subplots(figsize=(7.4, 2.9))

    x = node(ax, 0.0, 0.0, "x", "leaf")
    p = node(ax, 1.9, 0.0, "pow", "op")
    L = node(ax, 3.9, 0.0, "loss", "tensor")
    fwd(ax, x, p)
    fwd(ax, p, L)
    badge(ax, x, "x.grad", True, dy=-0.95)

    region(ax, [p, L],
           "retain_graph=True keeps this - the path and its saved values",
           color=ACCENT, size=8.8)
    region(ax, [Node(x.x, x.y - 0.95, 1.1, 0.34)],
           "zero_grad() clears this - the gradient on the leaf",
           color=INK, below=True, size=8.8)

    note(ax, 1.95, -2.1,
         "two different things, one name each - neither touches the other",
         ACCENT, 8.8)

    frame(ax, (-1.6, 5.9), (-2.4, 1.15))
    save(fig, "retain_graph_vs_zero_grad")


def fig_retain_graph_chain():
    """여러 번 걸 때는 마지막 호출에만 빼 준다 — 3절."""
    fig, ax = plt.subplots(figsize=(7.8, 2.2))

    steps = [
        ("backward(retain_graph=True)", "op", "x.grad = 2x", "ok"),
        ("backward()", "op", "x.grad = 4x", "ok"),
        ("backward()", "off", "RuntimeError", "off"),
    ]
    xs = [1.5, 5.3, 8.6]
    prev = None
    for (lab, k, res, rk), cx in zip(steps, xs):
        n = node(ax, cx, 0.3, lab, k, fontsize=8.8)
        r = node(ax, cx, -0.55, res, rk, w=2.1, h=0.4, fontsize=8.8)
        _arrow(ax, n.bottom, r.top, ACCENT if rk == "ok" else GRAY,
               "solid" if rk == "ok" else (0, (3, 2)), 1.1)
        if prev:
            _arrow(ax, prev.right, n.left, INK, "solid", 1.1)
        prev = n

    note(ax, 7.4, 1.34, "the 2nd call did not ask to keep them,", GRAY, 8.4)
    note(ax, 7.4, 1.06, "so the buffers were released here", GRAY, 8.4)
    note(ax, 4.3, -1.22,
         "2x accumulates to 4x - keeping the graph and clearing the gradient "
         "are separate chores", ACCENT, 8.8)

    frame(ax, (-0.95, 10.6), (-1.55, 1.58))
    save(fig, "retain_graph_chain")


# ============================================================================
# 04_mini_training_step — 다섯 걸음의 차례
# ============================================================================

def fig_training_loop_order():
    """지우기가 역전파 **앞**에 와야 하는 까닭 — 3절.

    뒤에 두면 오류도 안 나고 로스도 계산된다. 모델만 안 움직인다.
    """
    fig, ax = plt.subplots(figsize=(7.8, 3.4))

    for yy, steps, tail, tk, cut_at in [
        (0.6, ["forward", "loss", "zero_grad()", "backward()", "update"],
         "w = +0.654094", "ok", None),
        (-1.3, ["forward", "loss", "backward()", "zero_grad()", "update"],
         "w = +1.540996", "off", 3),
    ]:
        ns = row(ax, 0.0, yy, [(s, "op" if "(" in s else "tensor")
                               for s in steps], gap=0.46, h=0.44,
                 fontsize=8.8)
        for i, (a, b) in enumerate(zip(ns, ns[1:])):
            if cut_at is not None and i == cut_at:
                dead(ax, a, b)
            else:
                fwd(ax, a, b, lw=1.1)
        t = node(ax, ns[-1].right[0] + 2.0, yy, tail, tk, w=2.6, h=0.44,
                 fontsize=8.8)
        _arrow(ax, ns[-1].right, t.left, ACCENT if tk == "ok" else GRAY,
               "solid" if tk == "ok" else (0, (3, 2)), 1.2)

        loopback(ax, ns[-1], ns[0], drop=0.34, label="next step")

    note(ax, 4.6, 1.3, "zero before the sweep - the usual order", ACCENT, 8.8)
    note(ax, 4.6, -0.72, "zero after the sweep - nothing is wrong at runtime",
         GRAY, 8.8)
    note(ax, 4.6, -2.72,
         "the second loop raises nothing and prints a loss every step; "
         "w simply never leaves its starting value", ACCENT, 8.8)

    frame(ax, (-0.6, 13.0), (-3.0, 1.6))
    save(fig, "training_loop_order")


def fig_no_grad_update():
    """갱신은 학습 바깥의 일이라 그래프에 적지 않는다 — 4절."""
    fig, ax = plt.subplots(figsize=(7.6, 3.2))

    w = node(ax, 0.0, 0.75, "w", "leaf")
    f = node(ax, 1.9, 0.75, "forward", "op", w=1.6)
    L = node(ax, 4.0, 0.75, "loss", "tensor")
    fwd(ax, w, f)
    fwd(ax, f, L)
    region(ax, [w, f, L], "recorded on the graph", color=INK)

    # 갱신 칸을 w 바로 아래에 두어, 되돌려 쓰는 화살표가 곧게 서게 한다.
    u = node(ax, 0.0, -1.15, "w -= lr * w.grad", "op", w=3.0, h=0.42,
             fontsize=9.0)
    g = node(ax, 4.0, -1.15, "w.grad", "ok", w=1.5, h=0.42, fontsize=9.0)
    _arrow(ax, g.left, u.right, ACCENT, "solid", 1.2)
    region(ax, [u, g], "with torch.no_grad():  -  deliberately not recorded",
           color=ACCENT, below=True, size=8.6)

    _arrow(ax, L.bottom, g.top, ACCENT, (0, (4, 2)), 1.2)
    note(ax, 4.75, -0.2, "backward", ACCENT, 8.4, ha="left")

    _arrow(ax, u.top, w.bottom, INK, "solid", 1.2)
    note(ax, 0.18, -0.2, "writes into w", INK, 8.4, ha="left")

    note(ax, 6.3, 0.75, "the model's own computation", INK, 8.6, ha="left")
    note(ax, 6.3, -1.15, "what we do about it", ACCENT, 8.6, ha="left")

    note(ax, 2.0, -2.35,
         "drop the no_grad and torch refuses: a leaf Variable that requires "
         "grad", INK, 8.6)
    note(ax, 2.0, -2.68,
         "is being used in an in-place operation", INK, 8.6)

    frame(ax, (-2.0, 10.4), (-3.0, 1.6))
    save(fig, "no_grad_update")


def fig_three_steps():
    """세 걸음이 간 거리 — 2절. 줄어들기는 하지만 아직 멀다."""
    fig, ax = plt.subplots(figsize=(7.4, 2.1))

    ws = [1.540996, 1.202197, 0.908600, 0.654094]
    target = -1.0

    ax.annotate("", xy=(1.95, 0), xytext=(-1.35, 0),
                arrowprops=dict(arrowstyle="-", color=INK, lw=1.0))
    for v in (-1, 0, 1):
        ax.plot([v, v], [-0.06, 0.06], color=INK, lw=1.0)
        ax.text(v, -0.1, f"{v}", ha="center", va="top", fontsize=8.0,
                family=MONO, color=GRAY)

    ax.plot([target], [0], marker="o", ms=8, color=ACCENT, zorder=5)
    ax.text(target, 0.26, "target  w = -1", ha="center", va="bottom",
            fontsize=8.8, color=ACCENT)

    for i, v in enumerate(ws):
        ax.plot([v], [0], marker="|", ms=14, color=INK, lw=1.6, zorder=4)
        ax.text(v, -0.54 if i % 2 else -0.84, f"{v:.4f}", ha="center",
                va="center", fontsize=8.0, family=MONO, color=INK)
        ax.text(v, 0.12, f"{i}", ha="center", va="bottom", fontsize=8.0,
                color=GRAY)
    for a, b in zip(ws, ws[1:]):
        _arrow(ax, (a, 0.52), (b, 0.52), ACCENT, "solid", 1.2)

    moved, need = ws[0] - ws[-1], ws[0] - target
    note(ax, 0.3, 1.0,
         f"three steps moved {moved:.4f} of the {need:.4f} it has to cover "
         f"- {100 * moved / need:.0f}%", ACCENT, 8.8)

    frame(ax, (-1.6, 2.2), (-1.12, 1.3))
    save(fig, "three_steps")


# ============================================================================
# 05_complete_linear_regression — 셋은 어디에 있고, 무엇이 거리를 정하는가
# ============================================================================

def fig_three_numbers():
    """참값 · 최소제곱해 · 경사 하강이 놓인 자리 — 3절.

    경사 하강이 닿은 곳은 참값이 아니라 해석해다. 두 거리의 자릿수가
    네 자리 차이라는 것을 보이려면 로그 축이 있어야 한다.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.8, 2.9),
                                   gridspec_kw=dict(width_ratios=[1, 1.15]))

    # --- 왼쪽: (w, b) 평면 위의 두 자리 ---
    truth = (2.0, 1.0)
    land = (2.002363, 1.007134)       # 경사 하강 = 해석해 (이 눈금에서는 한 점)
    ax1.plot(*truth, marker="o", ms=7, color=INK, zorder=4)
    ax1.plot(*land, marker="s", ms=7, color=ACCENT, zorder=4)
    ax1.annotate("", xy=land, xytext=truth,
                 arrowprops=dict(arrowstyle="-|>", color=GRAY, lw=1.1,
                                 shrinkA=7, shrinkB=7))
    ax1.text(truth[0] - 0.0004, truth[1] - 0.0012, "truth\n(2, 1)",
             ha="right", va="top", fontsize=8.4, color=INK)
    ax1.text(land[0] + 0.0003, land[1] + 0.0008,
             "least squares\nand gradient descent\n(one dot here)",
             ha="left", va="bottom", fontsize=8.4, color=ACCENT)
    ax1.set_xlim(1.9985, 2.0048)
    ax1.set_ylim(0.9955, 1.0118)
    axes_style(ax1, "w", "b")

    # --- 오른쪽: 두 거리를 로그 축에 ---
    # 이름표는 축 **바깥** 눈금으로 뺀다. 안에 두면 점을 가린다.
    pairs = [("to the least squares\nsolution", 7.15e-07, 4.77e-07, ACCENT),
             ("to the true\nparameters", 2.363e-03, 7.134e-03, INK)]
    for i, (_, dw, db, c) in enumerate(pairs):
        y = 1 - i
        ax2.plot([dw], [y + 0.14], marker="o", ms=6, color=c)
        ax2.plot([db], [y - 0.14], marker="D", ms=5, color=c)
        ax2.text(dw * 1.7, y + 0.14, "w", ha="left", va="center",
                 fontsize=8.2, family=MONO, color=c)
        ax2.text(db * 1.7, y - 0.14, "b", ha="left", va="center",
                 fontsize=8.2, family=MONO, color=c)
    ax2.set_xscale("log")
    ax2.set_xlim(2.2e-07, 4e-02)
    ax2.set_ylim(-0.6, 1.6)
    ax2.set_yticks([1, 0])
    ax2.set_yticklabels([p[0] for p in pairs], fontsize=8.4)
    axes_style(ax2, "distance from where gradient descent landed")
    ax2.spines["left"].set_visible(False)
    ax2.tick_params(axis="y", length=0)
    # axes_style 이 눈금 색을 INK 로 덮으므로 **그 뒤에** 칠한다.
    for lab, (_, _, _, c) in zip(ax2.get_yticklabels(), pairs):
        lab.set_color(c)

    save(fig, "three_numbers",
         foot="three to four orders of magnitude apart - the leftover gap "
              "is the noise in the data, not the optimiser",
         foot_color=ACCENT, foot_size=8.4, rect=0.06)


def fig_noise_to_solution():
    """잡음을 줄이면 해석해가 참값으로 간다 — 4절."""
    fig, ax = plt.subplots(figsize=(6.4, 2.7))

    noise = [0.40, 0.20, 0.05, 0.00]
    err = [0.004724, 0.002362, 0.000591, 0.000000]

    ax.plot(noise, err, marker="o", ms=6, color=ACCENT, lw=1.4, zorder=4)
    for n, e in zip(noise, err):
        ax.annotate(f"{e:.6f}", (n, e), textcoords="offset points",
                    xytext=(0, 9), ha="center", fontsize=7.8, family=MONO,
                    color=INK)

    ax.set_xlim(-0.03, 0.45)
    ax.set_ylim(-0.0004, 0.0059)
    axes_style(ax, "noise standard deviation", "|w of least squares  -  2|")

    ax.text(0.205, 0.0012,
            "one seed: the same noise pattern, only rescaled,\n"
            "so the error tracks it exactly - other seeds would scatter",
            fontsize=8.2, color=GRAY, ha="left", va="center")

    save(fig, "noise_to_solution",
         foot="at zero noise the least squares solution is the true one - "
              "'not trained enough' and 'the data looks like that' are "
              "different questions",
         foot_color=ACCENT, rect=0.075)


def fig_learning_rate_window():
    """너무 작으면 못 닿고 너무 크면 튄다 — 5절. 가운데가 넓다."""
    fig, ax = plt.subplots(figsize=(6.6, 2.9))

    lrs = [0.001, 0.01, 0.1, 0.5]
    dist = [9.65e-01, 2.10e-02, 1.19e-06, 3.58e-07]

    ax.plot(lrs, dist, marker="o", ms=6, color=ACCENT, lw=1.4, zorder=4)
    for lr, d in zip(lrs, dist):
        ax.annotate(f"{d:.2e}", (lr, d), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=7.8, family=MONO,
                    color=INK)

    ax.axvline(1.1, color=GRAY, lw=1.2, ls=(0, (4, 3)))
    ax.text(1.02, 2e-04, "lr = 1.1\ndiverges", fontsize=8.4, color=GRAY,
            ha="right", va="center")

    ax.axvspan(0.1, 0.5, color=ACCENT, alpha=0.08, zorder=0)
    ax.text(0.22, 2e-05, "the floor is wide", fontsize=8.4, color=ACCENT,
            ha="center")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(6e-04, 2.4)
    ax.set_ylim(8e-08, 1.1e01)
    axes_style(ax, "learning rate", "distance to least squares")

    save(fig, "learning_rate_window",
         foot="200 steps at each rate - between 0.1 and 0.5 the answer is "
              "the same to six decimals",
         foot_color=INK, foot_size=8.2, rect=0.065)


# ============================================================================
# 06_linear_regression_with_module — 같은 알고리즘의 다른 표기
# ============================================================================

def fig_hand_vs_module():
    """손으로 짠 셋과 라이브러리 셋이 서로 맞바꿔지고, 결과가 비트까지 같다."""
    fig, ax = plt.subplots(figsize=(7.8, 3.5))

    pairs = [
        ("w, b  made by hand", "nn.Linear(1, 1)"),
        ("((pred - y)**2).mean()", "nn.MSELoss()"),
        ("no_grad(): w -= lr * w.grad", "torch.optim.SGD"),
    ]
    LX, RX = 0.0, 6.3
    for i, (l, r) in enumerate(pairs):
        y = 1.0 - 0.78 * i
        a = node(ax, LX, y, l, "tensor", w=4.6, h=0.5, fontsize=8.8)
        b = node(ax, RX, y, r, "op", w=3.2, h=0.5, fontsize=8.8)
        _arrow(ax, a.right, b.left, INK, "solid", 1.1)

    note(ax, LX, 1.72, "by hand", INK, 9.4)
    note(ax, RX, 1.72, "with the library", ACCENT, 9.4)

    # 한 줄에 다 적으면 칸을 넘는다. 세 줄로 접는다.
    res = "w    2.00236273\nb    1.00713444\nloss 0.03121442"
    node(ax, LX, -1.7, res, "ok", w=2.9, h=1.0, fontsize=8.4)
    node(ax, RX, -1.7, res, "ok", w=2.9, h=1.0, fontsize=8.4)
    note(ax, 3.15, -1.56, "difference", ACCENT, 8.8)
    note(ax, 3.15, -1.88, "0.00e+00", ACCENT, 9.4)

    note(ax, 3.15, -2.72,
         "200 steps from the same starting values - identical to the last bit",
         ACCENT, 8.8)
    note(ax, 3.15, -3.06,
         "not a different algorithm, the same one written differently",
         INK, 8.6)

    frame(ax, (-2.8, 9.3), (-3.35, 2.05))
    save(fig, "hand_vs_module")


def fig_what_step_hides():
    """`opt.step()` 안에는 뺄셈 한 줄이 있고, reduction 은 경사를 N 배 한다."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.8, 2.7),
                                   gridspec_kw=dict(width_ratios=[1.15, 1]))

    # --- 왼쪽: step() 한 걸음 ---
    a = node(ax1, 0.0, 0.8, "w = -0.775056", "tensor", w=3.0, h=0.44,
             fontsize=8.8)
    s = node(ax1, 0.0, 0.0, "opt.step()", "op", w=2.0, h=0.44, fontsize=8.8)
    h = node(ax1, 0.0, -0.8, "w - lr * w.grad", "op", w=3.0, h=0.44,
             fontsize=8.8)
    r = node(ax1, 3.9, -0.4, "-0.215242", "ok", w=2.0, h=0.44, fontsize=8.8)
    _arrow(ax1, a.bottom, s.top, INK, "solid", 1.1)
    _arrow(ax1, s.right, r.left, ACCENT, "solid", 1.2)
    _arrow(ax1, h.right, r.left, ACCENT, "solid", 1.2)
    note(ax1, 1.95, 1.3, "one step() is one subtraction", ACCENT, 8.8)
    note(ax1, 0.0, -1.35, "done by hand", INK, 8.4)
    frame(ax1, (-1.9, 5.3), (-1.7, 1.6))

    # --- 오른쪽: reduction 이 바꾸는 것 ---
    for i, (lab, g, k) in enumerate([('reduction="mean"', "-4.4662", "ok"),
                                     ('reduction="sum"', "-446.6163",
                                      "tensor")]):
        y = 0.5 - 1.0 * i
        ax2.text(-0.3, y, lab, ha="right", va="center", fontsize=8.8,
                 family=MONO, color=INK)
        node(ax2, 1.5, y, g, k, w=2.2, h=0.44, fontsize=9.0)
    note(ax2, 3.5, 0.0, "x 100", ACCENT, 9.5, ha="left")
    note(ax2, 3.5, -0.34, "= the sample count", ACCENT, 8.4, ha="left")
    note(ax2, 1.2, -1.35, "weight.grad after one backward", INK, 8.6)
    frame(ax2, (-3.3, 6.2), (-1.7, 1.6))

    save(fig, "what_step_hides",
         foot="the abstraction is thin - the parts it hides are a subtraction "
              "and a choice of divider",
         foot_color=ACCENT, foot_size=8.6, rect=0.07)


# ============================================================================
# 07_when_to_stop_tracking_grads — 넷이 건드리는 것이 서로 다르다
# ============================================================================

def fig_four_ways_to_stop():
    """추적을 끄는 네 길이 **무엇을** 건드리는지 — 2절.

    넷을 섞는 까닭은 결과가 비슷해서다 (requires_grad 가 False 가 된다).
    다른 것은 결과가 아니라 **무엇이 바뀌는가**이므로, 원본이 그대로인지를
    칸마다 보여야 한다.
    """
    fig, axs = plt.subplots(2, 2, figsize=(7.8, 4.6))

    # --- (가) 깃발을 내린다 — 원본이 바뀐다 ---
    ax = axs[0][0]
    a = node(ax, 0.0, 0.0, "w", "leaf")
    b = node(ax, 2.4, 0.0, "w", "off")
    flag(ax, a, True)
    flag(ax, b, False)
    _arrow(ax, a.right, b.left, INK, "solid", 1.2)
    note(ax, 1.2, 0.34, "the same tensor", INK, 8.2)
    note(ax, 1.2, 0.92, "w.requires_grad_(False)", INK, 9.2)
    note(ax, 1.2, -1.08, "the only one that changes the original",
         ACCENT, 8.2)
    note(ax, 1.2, -1.38, "reversible", GRAY, 8.0)
    frame(ax, (-0.9, 3.3), (-1.62, 1.2))

    # --- (나) 블록 안에서는 기록하지 않는다 — 원본은 그대로 ---
    ax = axs[0][1]
    a = node(ax, 0.0, 0.0, "w", "leaf")
    op = node(ax, 1.9, 0.0, "matmul", "op", w=1.5)
    y = node(ax, 3.6, 0.0, "y", "off")
    fwd(ax, a, op)
    fwd(ax, op, y)
    region(ax, [op, y], "", color=ACCENT, pad=0.2)
    # 깃발은 테두리 **밖**에 적는다. 안에 두면 점선에 걸린다.
    flag(ax, a, True, dy=-0.66)
    flag(ax, y, False, dy=-0.66)
    note(ax, 2.75, 0.92, "with torch.no_grad():", ACCENT, 9.2)
    note(ax, 1.8, -1.08, "only what is made inside is unflagged", ACCENT, 8.2)
    note(ax, 1.8, -1.38, "w itself is untouched", GRAY, 8.0)
    frame(ax, (-0.9, 4.5), (-1.62, 1.2))

    # --- (다) 새 손잡이를 내준다 — 같은 데이터 ---
    ax = axs[1][0]
    a = node(ax, 0.0, 0.0, "w", "leaf")
    d = node(ax, 2.4, 0.0, "w.detach()", "off", w=1.9)
    flag(ax, a, True)
    flag(ax, d, False)
    _arrow(ax, a.right, d.left, INK, "solid", 1.2)
    note(ax, 1.3, 0.34, "a new handle", INK, 8.2)
    note(ax, 1.2, 0.92, "w.detach()", INK, 9.2)
    note(ax, 1.3, -1.08, "same storage - writing to one writes both",
         ACCENT, 8.2)
    note(ax, 1.3, -1.38, "detach().clone() if you mean to edit it",
         GRAY, 8.0)
    frame(ax, (-0.9, 3.7), (-1.62, 1.2))

    # --- (라) 표가 박힌다 — 학습으로 되돌아갈 수 없다 ---
    ax = axs[1][1]
    a = node(ax, 0.0, 0.0, "w", "leaf")
    op = node(ax, 1.9, 0.0, "matmul", "op", w=1.5)
    y = node(ax, 3.6, 0.0, "y", "off")
    back = node(ax, 6.0, 0.0, "training", "tensor", w=1.7)
    fwd(ax, a, op)
    fwd(ax, op, y)
    dead(ax, y, back)
    region(ax, [op, y], "", color=ACCENT, pad=0.2)
    flag(ax, a, True, dy=-0.66)
    flag(ax, y, False, dy=-0.66)
    note(ax, 2.75, 0.92, "with torch.inference_mode():", ACCENT, 9.2)
    note(ax, 2.9, -1.08, "y is stamped and cannot re-enter a graph",
         ACCENT, 8.2)
    note(ax, 2.9, -1.38, "saves more; not reversible", GRAY, 8.0)
    frame(ax, (-0.9, 7.0), (-1.62, 1.2))

    save(fig, "four_ways_to_stop",
         foot="all four end with requires_grad=False - they differ in what "
              "they changed to get there",
         foot_color=ACCENT, foot_size=8.6, rect=0.055)


def fig_graph_or_not():
    """그래프를 안 세우면 저장해 둘 중간값도 없다 — 3절.

    아끼는 메모리의 정체가 바로 그 중간값이다. 03 쪽의 saved_buffers 와
    같은 칸을 쓴다.
    """
    fig, ax = plt.subplots(figsize=(7.6, 2.9))

    for yy, title, fn, saved in [
        (0.55, "default", "grad_fn = AddmmBackward0", True),
        (-1.05, "no_grad / inference_mode", "grad_fn = None", False),
    ]:
        ns = row(ax, 0.0, yy, [("x", "tensor"), ("linear", "op"),
                               ("relu", "op"), ("linear", "op"),
                               ("out", "tensor" if saved else "off")],
                 gap=0.44, h=0.42, fontsize=8.8)
        for a, b in zip(ns, ns[1:]):
            fwd(ax, a, b, lw=1.1)
        note(ax, -0.55, yy, title, INK, 8.6, ha="right")
        for n in ns[1:4]:
            node(ax, n.x, yy - 0.62, "saved" if saved else "nothing",
                 "ok" if saved else "off", w=1.15, h=0.3, fontsize=7.8)
        note(ax, ns[-1].right[0] + 0.5, yy, fn,
             ACCENT if saved else GRAY, 8.4, ha="left")

    note(ax, 3.2, -0.44, "kept for the backward sweep", ACCENT, 8.4)
    note(ax, 3.2, -2.04, "nothing to keep - this is the memory that is saved",
         GRAY, 8.4)

    frame(ax, (-3.4, 11.4), (-2.35, 1.1))
    save(fig, "graph_or_not")


# ============================================================================
# 08_turn_off_gradient_tracking — 얼린 층은 길을 막지 않는다
# ============================================================================

def fig_frozen_passes_through():
    """얼린 층을 경사가 통과해 앞쪽까지 간다 — 3절.

    「얼린다」를 「막는다」로 읽으면 전이학습 코드가 왜 도는지 설명이 안 된다.
    막는 것이 아니라 **제 몫을 안 쌓는** 것이다.
    """
    fig, ax = plt.subplots(figsize=(7.6, 2.8))

    a = node(ax, 0.0, 0.0, "layer 0", "leaf", w=1.7)
    b = node(ax, 2.6, 0.0, "layer 1", "off", w=1.7)
    c = node(ax, 5.2, 0.0, "loss", "tensor", w=1.4)
    fwd(ax, a, b)
    fwd(ax, b, c)
    flag(ax, a, True, dy=0.46)
    flag(ax, b, False, dy=0.46)

    bwd(ax, c, b, drop=0.72)
    bwd(ax, b, a, drop=0.72)
    note(ax, 3.9, -0.94, "passes straight through", ACCENT, 8.4)

    badge(ax, a, "grad", True, dy=-1.42)
    badge(ax, b, "None", False, dy=-1.42)

    note(ax, 7.0, -0.1, "the path is intact", INK, 8.6, ha="left")
    note(ax, 7.0, -0.46, "only the middle box", GRAY, 8.6, ha="left")
    note(ax, 7.0, -0.78, "keeps nothing", GRAY, 8.6, ha="left")

    note(ax, 2.0, -2.0,
         "frozen means 'does not accumulate', not 'blocks the way'",
         ACCENT, 8.8)

    frame(ax, (-1.2, 10.6), (-2.3, 1.0))
    save(fig, "frozen_passes_through")


def fig_three_tools_on_a_net():
    """가운데 층에 세 가지를 걸면 **앞쪽 층의 운명이 갈린다** — 4절."""
    fig, ax = plt.subplots(figsize=(7.8, 4.0))

    # 가운데 칸을 점선으로 그리는 것은 「따라가지 않는다」는 뜻이다.
    # (b) 는 가운데 층의 깃발이 그대로 서 있고 받을 것이 없을 뿐이라 실선이다.
    rows = [
        ("(a)  net[1].requires_grad_(False)", False, "off",
         [True, False, True], "layer 0 still gets its gradient"),
        ("(b)  detach the output of net[1]", True, "tensor",
         [False, False, True], "the path is cut - layer 0 gets nothing"),
        ("(c)  wrap net[1] in no_grad", True, "off",
         [False, False, True], "the path is cut - layer 0 gets nothing"),
    ]
    for i, (title, cut, mid, grads, verdict) in enumerate(rows):
        y = 1.1 - 1.55 * i
        ns = [node(ax, 0.0, y, "layer 0", "tensor", w=1.5, h=0.42,
                   fontsize=8.6),
              node(ax, 2.3, y, "layer 1", mid, w=1.5, h=0.42, fontsize=8.6),
              node(ax, 4.6, y, "layer 2", "tensor", w=1.5, h=0.42,
                   fontsize=8.6),
              node(ax, 6.7, y, "loss", "tensor", w=1.2, h=0.42,
                   fontsize=8.6)]
        fwd(ax, ns[0], ns[1], lw=1.1)
        if cut:
            dead(ax, ns[1], ns[2])
        else:
            fwd(ax, ns[1], ns[2], lw=1.1)
        fwd(ax, ns[2], ns[3], lw=1.1)
        note(ax, -0.8, y + 0.52, title, INK, 8.8, ha="left")
        for n, g in zip(ns[:3], grads):
            badge(ax, n, "grad" if g else "None", g, dy=-0.46,
                  fontsize=8.0)
        note(ax, 8.0, y - 0.22, verdict, ACCENT if grads[0] else GRAY, 8.4,
             ha="left")

    note(ax, 4.0, -3.5,
         "all three stop layer 1 from learning; only (a) leaves the way open "
         "to what sits before it",
         ACCENT, 8.8)

    frame(ax, (-1.0, 14.6), (-3.75, 1.9))
    save(fig, "three_tools_on_a_net")


# ============================================================================
# 09_detach_vs_detach_ — 새 손잡이냐 제자리냐
# ============================================================================

def _pair(ax, x, yc, y_kind, saved_kind, gap=0.72):
    """`y` 와 역전파가 저장해 둔 칸을 「같은 객체」로 묶어 그린다.

    이 쪽의 함정이 성립하는 까닭이 바로 둘이 같은 객체라는 데 있다.
    """
    a = node(ax, x, yc + gap / 2, "y", y_kind, w=1.1, h=0.4, fontsize=9.0)
    b = node(ax, x, yc - gap / 2, "saved", saved_kind, w=1.5, h=0.4,
             fontsize=8.6)
    ax.plot([x, x], [a.y - 0.2, b.y + 0.2],
            color=GRAY, lw=1.0, linestyle=(0, (2, 2)), zorder=1)
    return a, b


def fig_detach_trap():
    """detach_ 는 저장된 그 객체를 끊는다 — 3절.

    detach() 는 새 객체를 만들 뿐이라 저장된 칸이 멀쩡하다. 한 글자 차이가
    여기서 갈린다.
    """
    fig, ax = plt.subplots(figsize=(7.8, 3.4))

    BEFORE, AFTER, NEW, VERDICT = 0.0, 3.4, 5.7, 8.7
    for yc, call, after_y, after_s, extra, verdict, vk in [
        (1.0, "y.detach_()", "off", "off", False, "RuntimeError", "off"),
        (-1.4, "y.detach()", "tensor", "ok", True, "ok   dloss/dx = 8x",
         "ok"),
    ]:
        _pair(ax, BEFORE, yc, "tensor", "ok")
        _pair(ax, AFTER, yc, after_y, after_s)
        _arrow(ax, (0.9, yc), (2.5, yc), INK, "solid", 1.2)
        note(ax, 1.7, yc + 0.18, call, INK, 9.0)
        if extra:
            d = node(ax, NEW, yc + 0.36, "d", "off", w=0.9, h=0.4,
                     fontsize=9.0)
            _arrow(ax, (3.98, yc + 0.36), (5.2, yc + 0.36), GRAY, "solid",
                   1.0)
            note(ax, NEW, yc + 0.78, "a new object", GRAY, 7.8)
        node(ax, VERDICT, yc, verdict, vk, w=3.1, h=0.44, fontsize=8.6)
        _arrow(ax, (4.2, yc - 0.36), (7.1, yc), ACCENT if vk == "ok"
               else GRAY, "solid" if vk == "ok" else (0, (3, 2)), 1.2)
        note(ax, VERDICT, yc + 0.46, "backward()", INK, 8.0)

    note(ax, BEFORE, 1.86, "before", INK, 8.6)
    note(ax, AFTER, 1.86, "after", INK, 8.6)
    note(ax, -1.0, 1.0, "the same", GRAY, 8.0, ha="right")
    note(ax, -1.0, 0.72, "object", GRAY, 8.0, ha="right")

    note(ax, 3.6, -2.46,
         "'saved' is the tensor the backward sweep is holding - and y is "
         "that same object",
         INK, 8.6)
    note(ax, 3.6, -2.8,
         "detach_ reaches into it; detach makes a second one and leaves it "
         "alone",
         ACCENT, 8.8)

    frame(ax, (-2.6, 10.6), (-3.1, 2.15))
    save(fig, "detach_trap")


# ============================================================================
# 10_with_torch_no_grad — 끄는 것은 역전파가 아니라 기록이다
# ============================================================================

def fig_no_grad_scope():
    """범위는 텐서가 아니라 코드 블록이다 — 2절."""
    fig, ax = plt.subplots(figsize=(7.6, 2.6))

    a = node(ax, 0.9, 0.55, "a = x * 2", "tensor", w=2.1, h=0.44,
             fontsize=8.8)
    b = node(ax, 4.4, 0.55, "b = x * 2", "off", w=2.1, h=0.44, fontsize=8.8)
    c = node(ax, 7.9, 0.55, "c = x * 2", "tensor", w=2.1, h=0.44,
             fontsize=8.8)
    for n, v in ((a, True), (b, False), (c, True)):
        flag(ax, n, v, dy=-0.4)

    region(ax, [b], "with torch.no_grad():", color=ACCENT, pad=0.72,
           size=9.0)

    # 깃발은 블록을 가로질러 그대로다 — 그래서 선을 블록 너머까지 긋는다.
    ax.plot([-0.5, 9.3], [-1.35, -1.35], color=ACCENT, lw=1.4)
    for x in (-0.5, 9.3):
        ax.plot([x, x], [-1.45, -1.25], color=ACCENT, lw=1.4)
    note(ax, 4.4, -1.62, "w.requires_grad = True, the whole way across",
         ACCENT, 8.8)

    note(ax, 4.4, 2.08, "the block governs what is made, not what exists",
         INK, 8.8)

    frame(ax, (-1.3, 10.1), (-2.0, 2.4))
    save(fig, "no_grad_scope")


def fig_what_is_blocked():
    """막히는 것은 그래프 **세우기**지 걷기가 아니다 — 3절."""
    fig, ax = plt.subplots(figsize=(7.6, 2.9))

    for yc, where, kind, call, res, rk in [
        (0.7, "graph built outside", "tensor",
         "backward() called inside", "[2.0, 2.0]", "ok"),
        (-1.2, "graph built inside", "off",
         "backward() on it", "does not require grad", "off"),
    ]:
        g = node(ax, 0.0, yc, where, kind, w=3.4, h=0.44, fontsize=8.8)
        r = node(ax, 5.6, yc, res, rk, w=3.4, h=0.44, fontsize=8.8)
        if rk == "ok":
            _arrow(ax, g.right, r.left, ACCENT, "solid", 1.2)
        else:
            dead(ax, g, r)
        note(ax, 2.8, yc + 0.32, call, INK, 8.4)

    note(ax, 2.8, -2.1,
         "what no_grad stops is the building; walking a graph that already "
         "exists is fine",
         ACCENT, 8.8)

    frame(ax, (-2.0, 7.6), (-2.45, 1.45))
    save(fig, "what_is_blocked")


def fig_eval_vs_no_grad():
    """eval() 은 무엇을 셈할지, no_grad 는 기록할지 — 5절. 서로 다른 손잡이다."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.8, 2.6))

    note(ax1, 1.4, 1.5, "eval() decides what is computed", INK, 9.0)
    for i, (lab, res, k) in enumerate([("train() + no_grad", "False", "off"),
                                       ("eval()  + no_grad", "True", "ok")]):
        y = 0.55 - 1.0 * i
        ax1.text(-0.35, y, lab, ha="right", va="center", fontsize=8.8,
                 family=MONO, color=INK)
        node(ax1, 1.6, y, res, k, w=1.3, h=0.42, fontsize=9.0)
    note(ax1, 1.4, -1.25, "two runs give the same numbers?", INK, 8.4)
    note(ax1, 1.4, -1.6, "dropout is still on in train()", GRAY, 8.2)
    frame(ax1, (-3.4, 3.1), (-1.95, 1.85))

    note(ax2, 1.4, 1.5, "no_grad decides whether it is recorded", ACCENT, 9.0)
    for i, (lab, res, k) in enumerate([("eval()  + no_grad", "None", "off"),
                                       ("eval()  alone",
                                        "AddmmBackward0", "tensor")]):
        y = 0.55 - 1.0 * i
        ax2.text(-0.35, y, lab, ha="right", va="center", fontsize=8.8,
                 family=MONO, color=INK)
        node(ax2, 2.1, y, res, k, w=2.6, h=0.42, fontsize=8.6)
    note(ax2, 1.6, -1.25, "grad_fn on the output", INK, 8.4)
    note(ax2, 1.6, -1.6, "eval() alone still holds a graph", GRAY, 8.2)
    frame(ax2, (-3.4, 4.2), (-1.95, 1.85))

    save(fig, "eval_vs_no_grad",
         foot="two switches, two different jobs - an evaluation loop wants "
              "both",
         foot_color=ACCENT, foot_size=8.6, rect=0.07)


# ============================================================================
# 11_optimizer_like_updates — step() 이 숨기고 있는 세 줄
# ============================================================================

def fig_three_optimizers():
    """셋을 손으로 적어도 파이토치와 같은 수가 나온다 — 2~4절."""
    fig, ax = plt.subplots(figsize=(7.8, 2.9))

    rows = [
        ("SGD", "w -= lr * g", "exact", "ok"),
        ("momentum", "b = mu * b + g ;  w -= lr * b", "1.79e-07", "ok"),
        ("Adam", "m, v, bias correction ;  w -= lr * mhat / sqrt(vhat)",
         "0.00e+00", "ok"),
    ]
    for i, (name, rule, diff, k) in enumerate(rows):
        y = 0.9 - 0.85 * i
        ax.text(-0.3, y, name, ha="right", va="center", fontsize=9.2,
                family=MONO, color=INK)
        node(ax, 3.9, y, rule, "op", w=7.8, h=0.46, fontsize=8.6)
        node(ax, 9.3, y, diff, k, w=2.0, h=0.46, fontsize=8.8)

    note(ax, 3.9, 1.52, "written by hand", INK, 8.8)
    note(ax, 9.3, 1.52, "largest gap from torch", ACCENT, 8.8)

    note(ax, 4.4, -1.5,
         "momentum differs in the last two digits because torch seeds the "
         "buffer with g",
         GRAY, 8.4)
    note(ax, 4.4, -1.86,
         "on the first step rather than with zero - a convention, not a "
         "different method",
         GRAY, 8.4)

    frame(ax, (-2.6, 10.6), (-2.15, 1.9))
    save(fig, "three_optimizers")


def fig_new_tensor_breaks_the_leaf():
    """제자리를 피하려고 새 텐서를 만들면 다음 걸음이 사라진다 — 4절."""
    fig, ax = plt.subplots(figsize=(7.8, 3.0))

    w = node(ax, 0.0, 0.85, "w", "leaf")
    bad = node(ax, 4.0, 0.85, "RuntimeError", "off", w=2.8, h=0.44,
               fontsize=8.8)
    dead(ax, w, bad)
    note(ax, 2.1, 1.22, "w -= lr * g   without no_grad", INK, 8.6)

    w2 = node(ax, 0.0, -0.75, "w", "leaf")
    w3 = node(ax, 3.5, -0.75, "w3", "tensor", w=0.9)
    nxt = node(ax, 6.6, -0.75, "w3.grad = None", "off", w=3.0, h=0.44,
               fontsize=8.8)
    fwd(ax, w2, w3)
    note(ax, 1.75, -0.38, "w = w - lr * g   a new tensor", INK, 8.6)
    flag(ax, w3, False, dy=-0.42, name="is_leaf")
    flag(ax, w3, True, dy=-0.72, name="requires_grad")
    dead(ax, w3, nxt)
    note(ax, 5.1, -0.38, "second backward", ACCENT, 8.4)

    note(ax, 3.2, -1.96,
         "it passes, and then there is no second step - a leaf is the only "
         "thing that collects a gradient",
         ACCENT, 8.8)
    note(ax, 3.2, -2.32,
         "which is why the update has to be in place, and in place needs "
         "no_grad",
         INK, 8.6)

    frame(ax, (-1.2, 9.4), (-2.6, 1.6))
    save(fig, "new_tensor_breaks_the_leaf")


# ============================================================================
# 12_converting_to_numpy — 터져 주는 둘과 조용한 하나
# ============================================================================

def fig_numpy_three_questions():
    """`.numpy()` 가 묻는 셋 가운데 둘은 거절하고 하나는 말이 없다 — 2절."""
    fig, ax = plt.subplots(figsize=(7.8, 3.0))

    gates = [
        ("requires_grad?", "Can't call numpy() on\nTensor that requires grad",
         True),
        ("on the cpu?", "can't convert mps:0\ndevice type tensor", True),
        ("share the memory?", "(no message at all)", False),
    ]
    xs = [0.6, 4.0, 7.4]
    for x, (q, msg, loud) in zip(xs, gates):
        g = node(ax, x, 0.8, q, "op" if loud else "off", w=2.7, h=0.44,
                 fontsize=8.8)
        m = node(ax, x, -0.5, msg, "tensor" if loud else "off", w=2.9,
                 h=0.72, fontsize=8.0)
        _arrow(ax, g.bottom, m.top, ACCENT if loud else GRAY,
               "solid" if loud else (0, (3, 2)), 1.1)
        note(ax, x, -1.1, "refuses" if loud else "just does it",
             ACCENT if loud else GRAY, 8.4)

    node(ax, 2.3, -1.95, "x.detach().cpu().numpy()", "ok", w=5.0, h=0.44,
         fontsize=9.0)
    note(ax, 2.3, -2.42, "the idiom answers the first two", ACCENT, 8.6)
    note(ax, 7.4, -1.95, "nothing in the idiom", GRAY, 8.6)
    note(ax, 7.4, -2.28, "answers the third", GRAY, 8.6)

    note(ax, 4.0, 1.6, "what .numpy() asks", INK, 9.2)

    frame(ax, (-1.2, 9.4), (-2.8, 2.0))
    save(fig, "numpy_three_questions")


def fig_shared_storage_bite():
    """공유가 물어뜯는 자리 — 4절. 역전파가 넘파이가 써 놓은 수를 읽는다."""
    fig, ax = plt.subplots(figsize=(7.8, 3.2))

    for yc, mode, shared, g0, gk, extra in [
        (0.9, ".numpy()", True, "594.0", "off", "numpy wrote into it"),
        (-1.4, ".numpy().copy()", False, "18.0", "ok",
         "the array got its own"),
    ]:
        t = node(ax, 0.0, yc + 0.45, "w   (tensor)", "leaf", w=2.3, h=0.4,
                 fontsize=8.6)
        a = node(ax, 0.0, yc - 0.45, "a   (ndarray)", "tensor", w=2.3, h=0.4,
                 fontsize=8.6)
        note(ax, 1.75, yc + 0.84, mode, INK, 8.8)

        if shared:
            # 한 칸으로 화살표 둘이 모인다 — 그것이 공유다.
            s = node(ax, 3.9, yc, "one storage", "off", w=2.3, h=0.42,
                     fontsize=8.6)
            _arrow(ax, t.right, s.left, GRAY, "solid", 1.0)
            _arrow(ax, a.right, s.left, GRAY, "solid", 1.0)
            reader = s
        else:
            # 칸이 둘로 갈린다. 화살표도 따로 간다.
            s = node(ax, 3.9, yc + 0.45, "storage", "ok", w=2.3, h=0.4,
                     fontsize=8.6)
            s2 = node(ax, 3.9, yc - 0.45, "a copy", "tensor", w=2.3, h=0.4,
                      fontsize=8.6)
            _arrow(ax, t.right, s.left, GRAY, "solid", 1.0)
            _arrow(ax, a.right, s2.left, GRAY, "solid", 1.0)
            reader = s
        note(ax, 3.9, yc - 1.0 if shared else yc - 0.9, extra, GRAY, 8.0)

        r = node(ax, 8.2, reader.y, f"w.grad = [{g0}, 36.0]", gk, w=3.6,
                 h=0.44, fontsize=8.8)
        _arrow(ax, reader.right, r.left, ACCENT, (0, (4, 2)), 1.2)
        note(ax, 6.65, reader.y + 0.44, "backward reads it", ACCENT, 8.2)

    note(ax, 4.2, -2.72,
         "the true value is 18w = [18.0, 36.0] - the first entry is wrong "
         "because the backward sweep",
         INK, 8.6)
    note(ax, 4.2, -3.07,
         "read the number numpy had written over it, and nothing raised",
         ACCENT, 8.8)

    frame(ax, (-1.4, 10.6), (-3.4, 1.95))
    save(fig, "shared_storage_bite")


# ============================================================================
# 13_print_layers — 세어지는 것과 저장되는 것이 다르다
# ============================================================================

def fig_params_vs_buffers():
    """학습되는 것은 저장되는 것의 **부분집합**이다 — 3절.

    파라미터 수를 세면 버퍼가 빠지는데 저장 파일에는 들어간다. 두 테두리를
    겹쳐 그리면 그 어긋남이 한눈에 보인다.
    """
    fig, ax = plt.subplots(figsize=(7.4, 2.9))

    p = node(ax, 1.9, 0.0, "8 tensors   202 numbers", "ok", w=4.0, h=0.5,
             fontsize=8.8)
    region(ax, [p], "parameters - what the optimizer moves", color=ACCENT,
           pad=0.34, size=8.6)

    b = node(ax, 7.0, 0.0, "1 buffer:  steps", "tensor", w=2.8, h=0.5,
             fontsize=8.8)
    region(ax, [p, b], "state_dict - what gets written to the file",
           color=INK, pad=0.78, below=True, size=8.8)

    note(ax, 7.0, 0.84, "counted by nobody,", GRAY, 8.2)
    note(ax, 7.0, 0.58, "saved all the same", GRAY, 8.2)

    note(ax, 4.2, -1.9,
         "count the parameters and the buffer is missing; load the file and "
         "it is there",
         ACCENT, 8.8)

    frame(ax, (-0.9, 9.4), (-2.2, 1.45))
    save(fig, "params_vs_buffers")


def fig_batchnorm_buffers():
    """버퍼가 실제로 쓰이는 자리 — 4절. 순전파가 고치고 역전파는 안 준다."""
    fig, ax = plt.subplots(figsize=(7.4, 2.9))

    ps = ["weight", "bias"]
    bs = ["running_mean", "running_var", "num_batches_tracked"]
    for i, n in enumerate(ps):
        node(ax, 1.3, 0.55 - 0.62 * i, n, "ok", w=2.4, h=0.44, fontsize=8.8)
    for i, n in enumerate(bs):
        node(ax, 5.6, 0.86 - 0.62 * i, n, "tensor", w=3.4, h=0.44,
             fontsize=8.8)

    note(ax, 1.3, 1.34, "parameters", ACCENT, 9.2)
    note(ax, 5.6, 1.62, "buffers", INK, 9.2)

    note(ax, 3.4, -1.28, "forward changes them after one pass:  True",
         ACCENT, 8.8)
    note(ax, 3.4, -1.62, "backward gives them nothing:  grad is None",
         GRAY, 8.8)
    note(ax, 3.4, -2.08,
         "changed by the model, not by the optimizer - that is what a buffer "
         "is",
         INK, 8.8)

    frame(ax, (-0.9, 8.0), (-2.3, 1.95))
    save(fig, "batchnorm_buffers")


# ============================================================================
# 14_layer_naming_with_ordereddict — 번호는 밀리고 이름은 안 밀린다
# ============================================================================

def fig_numbers_shift():
    """층 하나를 끼우면 번호 키가 전부 밀린다 — 2절."""
    fig, ax = plt.subplots(figsize=(7.8, 3.2))

    for yc, title, before, after, ak, verdict, vk in [
        (0.9, "numbered", "0.weight\n2.weight", "0.weight\n3.weight", "off",
         "Missing  3.weight\nUnexpected  2.weight", "off"),
        (-1.3, "named", "encoder.weight\nhead.weight",
         "encoder.weight\nhead.weight", "ok", "loads", "ok"),
    ]:
        ax.text(-1.5, yc, title, ha="right", va="center", fontsize=9.4,
                family=MONO, color=INK)
        b = node(ax, 0.0, yc, before, "tensor", w=2.9, h=0.78, fontsize=8.6)
        a = node(ax, 4.0, yc, after, ak, w=2.9, h=0.78, fontsize=8.6)
        _arrow(ax, b.right, a.left, INK, "solid", 1.2)
        v = node(ax, 8.6, yc, verdict, vk, w=4.0, h=0.78, fontsize=8.6)
        _arrow(ax, a.right, v.left, ACCENT if vk == "ok" else GRAY,
               "solid" if vk == "ok" else (0, (3, 2)), 1.2)

    note(ax, 2.0, 1.72, "a layer is inserted in the middle", INK, 8.8)
    note(ax, 8.6, 1.72, "loading the old checkpoint", INK, 8.8)

    note(ax, 3.6, -2.28,
         "the numbers are positions, so inserting anything renames every key "
         "after it",
         ACCENT, 8.8)
    note(ax, 3.6, -2.62,
         "the names are names - they do not move",
         INK, 8.6)

    frame(ax, (-3.8, 11.0), (-2.9, 2.05))
    save(fig, "numbers_shift")


# ============================================================================
# 「그림으로 보는 AI 6」 — 리나인마 1970
# ============================================================================

def make_linnainmaa():
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.6, 4.3))

    xs = [0.6, 2.4, 4.2, 6.0]
    names = ["u1", "u2", "u3", "y"]

    # --- 위: 앞으로 가며 반올림 오차가 생긴다 ---
    for x, n in zip(xs, names):
        node(ax1, x, 0.52, n, "tensor", w=0.9, h=0.44)
    for a, b in zip(xs[:-1], xs[1:]):
        _arrow(ax1, (a + 0.48, 0.52), (b - 0.48, 0.52), INK, "solid", 1.4)
    for x in xs:
        _arrow(ax1, (x, 1.12), (x, 0.78), ACCENT, "solid", 1.1)
        ax1.text(x, 1.22, "e", ha="center", va="bottom", fontsize=9,
                 family=MONO, color=ACCENT)
    ax1.text(-0.35, 0.52, "forward", ha="center", va="center", fontsize=9,
             weight="bold", color=INK, rotation=90)
    ax1.text(3.3, 1.62, "each step adds its own rounding error",
             ha="center", fontsize=9, color=ACCENT)
    frame(ax1, (-0.8, 6.8), (0.1, 1.95))

    # --- 아래: 거꾸로 가며 민감도를 모은다 ---
    for x, n in zip(xs, names):
        node(ax2, x, 0.95, n, "tensor", w=0.9, h=0.44)
    for a, b in zip(xs[1:], xs[:-1]):
        _arrow(ax2, (a - 0.48, 0.95), (b + 0.48, 0.95), ACCENT, "solid", 1.4)
    for i, x in enumerate(xs):
        lab = "dy/dy = 1" if i == len(xs) - 1 else f"dy/du{i+1}"
        ax2.text(x, 0.45, lab, ha="center", va="center", fontsize=8.5,
                 family=MONO, color=ACCENT)
    ax2.text(-0.35, 0.95, "reverse", ha="center", va="center", fontsize=9,
             weight="bold", color=INK, rotation=90)
    ax2.text(3.3, 0.09, "one backward sweep gives every sensitivity at once",
             ha="center", fontsize=9, color=ACCENT)
    frame(ax2, (-0.8, 6.8), (-0.05, 1.45))

    save(fig, "linnainmaa_1970",
         foot="the 1970 thesis asks about rounding error - it contains no "
              "neural network")


# ============================================================================

FIGURES = {
    # 01_basic_scalar_backward
    "four_attributes": fig_four_attributes,
    "leaf_vs_nonleaf": fig_leaf_vs_nonleaf,
    "retain_grad_timing": fig_retain_grad_timing,
    # 06_linear_regression_with_module
    "hand_vs_module": fig_hand_vs_module,
    "what_step_hides": fig_what_step_hides,
    # 14_layer_naming_with_ordereddict
    "numbers_shift": fig_numbers_shift,
    # 13_print_layers
    "params_vs_buffers": fig_params_vs_buffers,
    "batchnorm_buffers": fig_batchnorm_buffers,
    # 12_converting_to_numpy
    "numpy_three_questions": fig_numpy_three_questions,
    "shared_storage_bite": fig_shared_storage_bite,
    # 11_optimizer_like_updates
    "three_optimizers": fig_three_optimizers,
    "new_tensor_breaks_the_leaf": fig_new_tensor_breaks_the_leaf,
    # 10_with_torch_no_grad
    "no_grad_scope": fig_no_grad_scope,
    "what_is_blocked": fig_what_is_blocked,
    "eval_vs_no_grad": fig_eval_vs_no_grad,
    # 09_detach_vs_detach_
    "detach_trap": fig_detach_trap,
    # 08_turn_off_gradient_tracking
    "frozen_passes_through": fig_frozen_passes_through,
    "three_tools_on_a_net": fig_three_tools_on_a_net,
    # 07_when_to_stop_tracking_grads
    "four_ways_to_stop": fig_four_ways_to_stop,
    "graph_or_not": fig_graph_or_not,
    # 05_complete_linear_regression
    "three_numbers": fig_three_numbers,
    "noise_to_solution": fig_noise_to_solution,
    "learning_rate_window": fig_learning_rate_window,
    # 04_mini_training_step
    "training_loop_order": fig_training_loop_order,
    "no_grad_update": fig_no_grad_update,
    "three_steps": fig_three_steps,
    # 03_retain_graph
    "saved_buffers": fig_saved_buffers,
    "retain_graph_vs_zero_grad": fig_retain_graph_vs_zero_grad,
    "retain_graph_chain": fig_retain_graph_chain,
    # 02_gradient_accumulation
    "grad_is_a_bucket": fig_grad_is_a_bucket,
    "micro_batching": fig_micro_batching,
    "zero_vs_none": fig_zero_vs_none,
    "mean_sum_grid": fig_mean_sum_grid,
    # 그림으로 보는 AI 6
    "linnainmaa_1970": make_linnainmaa,
}


def main(argv):
    want = argv or list(FIGURES)
    for name in want:
        if name not in FIGURES:
            raise SystemExit(f"그런 그림이 없다: {name}\n"
                             f"있는 것: {', '.join(FIGURES)}")
        FIGURES[name]()


if __name__ == "__main__":
    import sys

    main(sys.argv[1:])
