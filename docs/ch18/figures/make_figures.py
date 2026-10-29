"""19장 장 소개의 그림.

규칙은 CLAUDE.md 의 Figures 항에 적어 둔 대로다.
- SVG, fonttype='path', transparent
- 글자는 모두 아스키 (matplotlib 기본 폰트에 한글이 없다)

자료를 읽지 않으므로 어디서 돌려도 된다.

실행: python docs/ch18/figures/make_figures.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["svg.fonttype"] = "path"

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
INK = "#2f3337"
ACCENT = "#8a4b2a"

A, B, C = 0.6, 0.3, 0.5     # X->Y, X->Z, Y->Z  (경로 계수)
N = 2_000_000


# === 그림으로 보는 AI: 라이트의 경로 추적 =====================================
#
# 1921 년 라이트의 기본 정리 — 두 변수의 상관은 그 둘을 잇는 **경로를 따라
# 계수를 곱하고 모두 더하면** 나온다. 표준화한 선형 모형에서 직접 재어 확인한다.

def wright_paths():
    rng = np.random.default_rng(0)
    X = rng.standard_normal(N)
    Y = A * X + np.sqrt(1 - A ** 2) * rng.standard_normal(N)
    eZ = np.sqrt(1 - (B ** 2 + C ** 2 + 2 * A * B * C))
    Z = B * X + C * Y + eZ * rng.standard_normal(N)
    for v, nm in ((X, "X"), (Y, "Y"), (Z, "Z")):
        assert abs(v.std() - 1) < 0.01, f"{nm} 가 표준화되지 않았다"

    emp = np.corrcoef(np.vstack([X, Y, Z]))
    traced = {("X", "Y"): A, ("X", "Z"): B + A * C, ("Y", "Z"): C + A * B}
    measured = {("X", "Y"): emp[0, 1], ("X", "Z"): emp[0, 2],
                ("Y", "Z"): emp[1, 2]}
    worst = max(abs(traced[k] - measured[k]) for k in traced)
    assert worst < 0.005, f"경로 추적이 맞지 않는다: {worst}"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.2, 3.9),
                                   gridspec_kw={"width_ratios": [1, 1.1]})

    # --- 왼쪽: 경로 그림 ---------------------------------------------------
    pos = {"X": (0.16, 0.74), "Y": (0.76, 0.74), "Z": (0.46, 0.26)}
    arcs = [("X", "Y", f"{A}"), ("X", "Z", f"{B}"), ("Y", "Z", f"{C}")]
    for s, t, lab in arcs:
        x0, y0 = pos[s]
        x1, y1 = pos[t]
        dx, dy = x1 - x0, y1 - y0
        L = np.hypot(dx, dy)
        ux, uy = dx / L, dy / L
        ax1.annotate("", xy=(x1 - ux * 0.085, y1 - uy * 0.085),
                     xytext=(x0 + ux * 0.085, y0 + uy * 0.085),
                     arrowprops=dict(arrowstyle="->", color=INK,
                                     linewidth=1.8))
        ax1.text((x0 + x1) / 2 + (0.0 if s == "X" and t == "Y" else 0.055),
                 (y0 + y1) / 2 + (0.055 if s == "X" and t == "Y" else 0.0),
                 lab, fontsize=9.5, color=ACCENT, ha="center", va="center",
                 weight="bold")
    for k, (x, y) in pos.items():
        ax1.add_patch(plt.Circle((x, y), 0.085, facecolor="white",
                                 edgecolor=INK, linewidth=1.6, zorder=3))
        ax1.text(x, y, k, ha="center", va="center", fontsize=11, color=INK,
                 zorder=4)
    ax1.text(0.5, 0.97, "a path diagram, 1921", ha="center", fontsize=9.4,
             color=INK, style="italic")
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0.05, 1.05)
    ax1.axis("off")

    # --- 오른쪽: 경로를 따라 더하면 상관이 나온다 --------------------------
    ax2.text(0.5, 0.95, "trace the paths, add the products", ha="center",
             fontsize=9.6, color=INK, weight="bold")
    rows = [("r(X,Y)", "a", traced[("X", "Y")], measured[("X", "Y")]),
            ("r(X,Z)", "b + a c", traced[("X", "Z")], measured[("X", "Z")]),
            ("r(Y,Z)", "c + a b", traced[("Y", "Z")], measured[("Y", "Z")])]
    ax2.text(0.08, 0.78, "", fontsize=9)
    ax2.text(0.06, 0.78, "correlation", fontsize=8.4, color=INK, alpha=0.75)
    ax2.text(0.36, 0.78, "paths", fontsize=8.4, color=INK, alpha=0.75)
    ax2.text(0.63, 0.78, "traced", fontsize=8.4, color=ACCENT, alpha=0.95)
    ax2.text(0.85, 0.78, "measured", fontsize=8.4, color=INK, alpha=0.75)
    for k, (name, expr, tr, me) in enumerate(rows):
        y = 0.60 - k * 0.165
        ax2.text(0.06, y, name, fontsize=9.5, color=INK, family="monospace")
        ax2.text(0.36, y, expr, fontsize=9.5, color=INK, family="monospace")
        ax2.text(0.63, y, f"{tr:.4f}", fontsize=9.5, color=ACCENT,
                 family="monospace", weight="bold")
        ax2.text(0.85, y, f"{me:.4f}", fontsize=9.5, color=INK,
                 family="monospace")
    ax2.plot([0.04, 0.98], [0.10, 0.10], color=ACCENT, linewidth=1.1)
    ax2.text(0.5, 0.02, f"largest gap {worst:.4f}  -  "
                        f"{N:,} samples", ha="center", fontsize=8.4,
             color=ACCENT)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1.02)
    ax2.axis("off")

    fig.tight_layout()
    fig.savefig(HERE / "wright_paths.svg", transparent=True,
                bbox_inches="tight")
    plt.close(fig)
    print(f"wrote wright_paths.svg   largest gap {worst:.5f}")


if __name__ == "__main__":
    wright_paths()
