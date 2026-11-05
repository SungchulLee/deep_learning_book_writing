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
        w = max(0.62, 0.135 * len(label) + 0.34)
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


def note(ax, x, y, text, color=None, size=8.6, ha="center", va="center",
         italic=False):
    ax.text(x, y, text, ha=ha, va=va, fontsize=size, color=color or INK,
            style="italic" if italic else "normal")


def frame(ax, xlim, ylim):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_axis_off()


def save(fig, name, foot=None):
    if foot:
        fig.text(0.5, 0.005, foot, ha="center", fontsize=8, style="italic",
                 color=INK)
        fig.tight_layout(rect=(0, 0.045, 1, 1))
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
