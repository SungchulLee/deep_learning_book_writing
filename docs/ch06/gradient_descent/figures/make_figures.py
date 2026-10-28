"""「그림으로 보는 AI」 — 코시의 1847년 노트를 그린다.

규칙은 CLAUDE.md 의 Figures 항과 「그림으로 보는 AI」 항에 적어 둔 대로다.
- 사진을 빌려 오지 않고 직접 그린다
- 글자는 모두 ASCII (matplotlib 기본 폰트에 한글이 없다)
- 지어낸 수를 넣지 않는다. 꼴만 옮긴 그림에는 schematic 이라 적는다
- SVG, fonttype='path', transparent

`learning_rate.py` 의 탐색기가 MNIST 를 읽으므로 **저장소 뿌리에서** 돌린다.
갈무리는 이 파일이 있는 자리에 하므로 어디서 돌리든 그림은 제자리에 간다.

실행: python docs/ch06/gradient_descent/figures/make_figures.py
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["svg.fonttype"] = "path"

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))      # learning_rate.py 를 읽으려고

INK = "#2f3337"
ACCENT = "#8a4b2a"


def make_cauchy():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.4, 3.2),
                                   gridspec_kw={"width_ratios": [1, 1.15]})

    # --- 왼쪽: 식 여러 개를 하나로 ---
    ax1.set_axis_off()
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.text(0.5, 0.95, "many equations", ha="center", va="top",
             fontsize=10, weight="bold", color=INK)
    for i, lab in enumerate(("f1(x, y) = 0", "f2(x, y) = 0", "f3(x, y) = 0")):
        ax1.text(0.5, 0.78 - 0.1 * i, lab, ha="center", va="center",
                 fontsize=9.5, family="monospace", color=INK)
    ax1.add_patch(FancyArrowPatch((0.5, 0.44), (0.5, 0.31),
                                  arrowstyle="-|>", mutation_scale=13,
                                  color=ACCENT, lw=1.4))
    ax1.text(0.5, 0.22, "one number to push down", ha="center", va="center",
             fontsize=10, weight="bold", color=INK)
    ax1.text(0.5, 0.09, "u = f1^2 + f2^2 + f3^2", ha="center", va="center",
             fontsize=10, family="monospace", color=ACCENT)

    # --- 오른쪽: 등고선과 한 걸음 ---
    g = np.linspace(-2.2, 2.2, 400)
    X, Y = np.meshgrid(g, g)
    Z = 0.6 * X ** 2 + 2.0 * Y ** 2          # 늘어난 사발
    ax2.contour(X, Y, Z, levels=10, colors=INK, linewidths=0.6, alpha=0.55)

    px, py = -1.8, 1.1
    gx, gy = 1.2 * px, 4.0 * py              # grad of 0.6x^2 + 2y^2
    n = (gx ** 2 + gy ** 2) ** 0.5
    sx, sy = -gx / n * 1.25, -gy / n * 1.25
    ax2.plot([px], [py], "o", color=ACCENT, ms=6, zorder=5)
    ax2.add_patch(FancyArrowPatch((px, py), (px + sx, py + sy),
                                  arrowstyle="-|>", mutation_scale=14,
                                  color=ACCENT, lw=1.8, zorder=5))
    # 화살 끝 아래, 가장 안쪽 등고선 안의 빈 자리에 둔다
    ax2.text(px + sx + 0.14, py + sy - 0.30,
             "-grad u", fontsize=9.5, color=ACCENT, family="monospace")
    ax2.plot([0], [0], "x", color=INK, ms=8, mew=1.6, zorder=5)
    ax2.text(0.12, -0.26, "u = 0", fontsize=9, color=INK, family="monospace")
    ax2.set_xticks([]); ax2.set_yticks([])
    for s in ax2.spines.values():
        s.set_color(INK); s.set_linewidth(0.7)
    ax2.set_title("step against the gradient", fontsize=10, weight="bold",
                  color=INK, pad=8)

    fig.text(0.5, 0.015,
             "schematic - not Cauchy's own figure; his 1847 note carries no diagram",
             ha="center", fontsize=8, style="italic", color=INK)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(HERE / "cauchy_1847.svg", transparent=True, bbox_inches="tight")
    plt.close(fig)
    print("wrote cauchy_1847.svg")


def make_contours():
    """등고선 · 경사 화살 · 접선. 수는 steepest_direction.py 가 찍는 것과 같다."""
    fig, ax = plt.subplots(figsize=(4.6, 4.2))

    g = np.linspace(-3.2, 3.2, 400)
    X, Y = np.meshgrid(g, g)
    Z = X ** 2 + 3 * Y ** 2
    ax.contour(X, Y, Z, levels=[1, 3, 6, 10, 13, 18, 24, 31],
               colors=INK, linewidths=0.7, alpha=0.6, zorder=1)
    # 등고선에 값을 적지 않는다. 자동으로 붙는 자리가 접선 글자와 겹치고,
    # 어느 곡선의 값인지는 이 그림이 말하려는 바가 아니다.

    # 경사 화살 — grad = (2x, 6y)
    for px, py in ((1.0, 2.0), (-2.0, 1.0), (2.0, -1.0)):
        gx, gy = 2 * px, 6 * py
        n = (gx ** 2 + gy ** 2) ** 0.5
        ax.add_patch(FancyArrowPatch((px, py),
                                     (px + gx / n * 0.95, py + gy / n * 0.95),
                                     arrowstyle="-|>", mutation_scale=12,
                                     color=ACCENT, lw=1.6, zorder=3))
        ax.plot([px], [py], "o", color=ACCENT, ms=4.5, zorder=4)

    # (1, 2) 에서 등위곡선의 접선 — 경사에 수직임을 눈으로 본다
    gx, gy = 2.0, 12.0
    n = (gx ** 2 + gy ** 2) ** 0.5
    tx, ty = -gy / n, gx / n
    ax.plot([1 - 1.3 * tx, 1 + 1.3 * tx], [2 - 1.3 * ty, 2 + 1.3 * ty],
            "--", color=INK, lw=1.2, zorder=2)
    ax.text(1.35, 2.52, "grad", fontsize=9, color=ACCENT, family="monospace")
    ax.text(1.46, 1.44, "tangent", fontsize=8.5, color=INK,
            family="monospace")
    ax.plot([1.0], [2.0], marker=(4, 2, 45), color=INK, ms=7, mew=1.0,
            zorder=5)

    ax.set_xlim(-3.2, 3.2)
    ax.set_ylim(-3.2, 3.2)
    ax.set_xlabel("x", fontsize=9)
    ax.set_ylabel("y", fontsize=9)
    ax.set_title("f(x, y) = x^2 + 3y^2", fontsize=10, weight="bold",
                 color=INK, pad=8)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=8, colors=INK)
    for s in ax.spines.values():
        s.set_color(INK); s.set_linewidth(0.7)

    fig.tight_layout()
    fig.savefig(HERE / "contours_gradients.svg", transparent=True,
                bbox_inches="tight")
    plt.close(fig)
    print("wrote contours_gradients.svg")


def make_lr():
    """러닝레이트 두 장. 수는 learning_rate.py 가 찍는 것과 같은 계산이다."""
    import learning_rate as LR                      # noqa: N814

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.6, 3.5))

    # --- 왼쪽: L = (w-3)^2 위의 걸음, 러닝레이트 넷 ---
    w = np.linspace(-2.5, 8.5, 300)
    ax1.plot(w, (w - 3) ** 2, color=INK, lw=0.9, alpha=0.6, zorder=1)
    colors = ["#2f6f4e", "#8a4b2a", "#8a2a4b", "#2a4b8a"]
    for (eta, c) in zip((0.1, 0.5, 0.9, 1.1), colors):
        x, traj = 7.0, [7.0]
        for _ in range(12):
            x = x - eta * 2 * (x - 3)
            traj.append(x)
        traj = np.array(traj)
        ax1.plot(traj, (traj - 3) ** 2, "o-", color=c, ms=3.4, lw=1.1,
                 label=f"eta = {eta}", zorder=3)
    ax1.axvline(3, color=INK, ls="--", lw=0.8, alpha=0.5, zorder=2)
    ax1.set_ylim(-1.5, 30)
    ax1.set_xlim(-2.5, 8.5)
    ax1.set_xlabel("w", fontsize=9)
    ax1.set_ylabel("loss", fontsize=9)
    ax1.set_title("L(w) = (w - 3)^2, 12 steps from w = 7",
                  fontsize=9.5, weight="bold", color=INK, pad=7)
    ax1.legend(fontsize=7.5, frameon=False, loc="upper center", ncol=2)

    # --- 오른쪽: 러닝레이트 탐색기 곡선 ---
    lrs, losses, start = LR.lr_finder()
    ax2.semilogx(lrs, losses, "o-", color=ACCENT, ms=3.4, lw=1.2, zorder=3)
    ax2.axhline(start, color=INK, ls="--", lw=0.8, alpha=0.6, zorder=2)
    ax2.text(1.3e-6, start + 0.07, "loss before training", fontsize=7.5,
             color=INK)
    best = int(np.argmin(losses))
    ax2.plot([lrs[best]], [losses[best]], "o", color=INK, ms=6, zorder=4)
    ax2.annotate(f"best  lr = {lrs[best]:.2f}",
                 xy=(lrs[best], losses[best]),
                 xytext=(lrs[best] * 0.03, losses[best] + 0.55),
                 fontsize=8, color=INK,
                 arrowprops=dict(arrowstyle="->", color=INK, lw=0.8))
    ax2.set_xlabel("learning rate (log scale)", fontsize=9)
    ax2.set_ylabel("loss after 20 steps", fontsize=9)
    ax2.set_title("LR range test, MNIST 1000 images",
                  fontsize=9.5, weight="bold", color=INK, pad=7)

    for ax in (ax1, ax2):
        ax.tick_params(labelsize=8, colors=INK)
        for s in ax.spines.values():
            s.set_color(INK)
            s.set_linewidth(0.7)

    fig.tight_layout()
    fig.savefig(HERE / "learning_rate.svg", transparent=True, bbox_inches="tight")
    plt.close(fig)
    print("wrote learning_rate.svg")


def main():
    make_cauchy()
    make_contours()
    make_lr()
    make_convexity()


if __name__ == "__main__":
    main()


def make_convexity():
    """볼록 집합·볼록 함수와 그 반례. 아스키 그림을 대신한다."""
    fig, axes = plt.subplots(1, 4, figsize=(10.4, 2.9))

    # --- 1. 볼록 집합 ---
    ax = axes[0]
    th = np.linspace(0, 2 * np.pi, 200)
    ax.fill(1.6 * np.cos(th), 1.1 * np.sin(th), color=INK, alpha=0.12)
    ax.plot(1.6 * np.cos(th), 1.1 * np.sin(th), color=INK, lw=1.1)
    p, q = (-1.0, -0.55), (1.1, 0.6)
    ax.plot([p[0], q[0]], [p[1], q[1]], "-", color=ACCENT, lw=1.6, zorder=3)
    ax.plot(*zip(p, q), "o", color=ACCENT, ms=5, zorder=4)
    ax.set_title("convex set", fontsize=9.5, weight="bold", color=INK, pad=6)
    ax.text(0, -1.52, "the segment stays inside", ha="center", fontsize=7.5,
            color=INK)
    ax.set_ylim(-1.85, 1.35)

    # --- 2. 볼록하지 않은 집합 (ㄱ 자 모양) ---
    ax = axes[1]
    poly = np.array([[0, 0], [2, 0], [2, 0.9], [0.9, 0.9], [0.9, 2], [0, 2]])
    ax.fill(poly[:, 0], poly[:, 1], color=INK, alpha=0.12)
    ax.plot(np.append(poly[:, 0], poly[0, 0]),
            np.append(poly[:, 1], poly[0, 1]), color=INK, lw=1.1)
    p, q = (1.75, 0.45), (0.45, 1.75)
    ax.plot([p[0], q[0]], [p[1], q[1]], "-", color=ACCENT, lw=1.6, zorder=3)
    ax.plot(*zip(p, q), "o", color=ACCENT, ms=5, zorder=4)
    ax.set_title("not convex", fontsize=9.5, weight="bold", color=INK, pad=6)
    ax.text(1.0, -0.42, "part of it steps outside", ha="center", fontsize=7.5,
            color=INK)
    ax.set_xlim(-0.35, 2.35)
    ax.set_ylim(-0.75, 2.35)

    for ax in axes[:2]:
        ax.set_aspect("equal")
        ax.set_axis_off()

    # --- 3. 볼록 함수와 현 ---
    ax = axes[2]
    x = np.linspace(-2, 2, 300)
    ax.plot(x, x ** 2, color=INK, lw=1.3)
    a, b = -1.5, 1.2
    ax.plot([a, b], [a ** 2, b ** 2], "-", color=ACCENT, lw=1.6, zorder=3)
    ax.plot([a, b], [a ** 2, b ** 2], "o", color=ACCENT, ms=5, zorder=4)
    ax.set_title("convex: chord above", fontsize=9.5, weight="bold",
                 color=INK, pad=6)

    # --- 4. 볼록하지 않은 함수와 현 ---
    ax = axes[3]
    x = np.linspace(-2.1, 2.1, 400)
    y = x ** 4 - 3 * x ** 2
    ax.plot(x, y, color=INK, lw=1.3)
    a, b = -1.6, 1.6
    ax.plot([a, b], [a ** 4 - 3 * a ** 2, b ** 4 - 3 * b ** 2], "-",
            color=ACCENT, lw=1.6, zorder=3)
    ax.plot([a, b], [a ** 4 - 3 * a ** 2, b ** 4 - 3 * b ** 2], "o",
            color=ACCENT, ms=5, zorder=4)
    ax.set_title("not convex: chord cuts under", fontsize=9.5, weight="bold",
                 color=INK, pad=6)

    for ax in axes[2:]:
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color(INK)
            s.set_linewidth(0.7)

    fig.tight_layout()
    fig.savefig(HERE / "convexity.svg", transparent=True, bbox_inches="tight")
    plt.close(fig)
    print("wrote convexity.svg")
