"""34.6 그래프 수준 과제 쪽들의 그림.

규칙은 CLAUDE.md 의 Figures 항에 적어 둔 대로다.
- SVG, fonttype='path', transparent
- 글자는 모두 아스키 (matplotlib 기본 폰트에 한글이 없다)

자료를 내려받지 않으므로 어디서 돌려도 된다.

실행: python docs/ch33/graph_tasks/figures/make_figures.py
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
MUTED = "#2d6a8f"


# === 그림으로 보는 AI: 평균과 최대는 크기를 지운다 ============================
#
# 같은 노드 피처를 가진 그래프와 그것을 그대로 복제해 키운 그래프를 읽어 낸다.
# 합만 둘을 가르고 평균·최대는 사실상 같은 값을 돌려준다 (최대는 정확히 0,
# 평균은 나누는 수가 달라 float32 반올림만 남는다). 직접 잰다.

def size_blindness():
    import torch

    torch.manual_seed(0)
    h = torch.randn(1, 4)
    small = h.repeat(3, 1)          # 노드 3
    big = h.repeat(12, 1)           # 같은 피처, 노드 12

    reads = {"sum": lambda X: X.sum(0),
             "mean": lambda X: X.mean(0),
             "max": lambda X: X.max(0).values}
    gaps = {k: float((f(small) - f(big)).abs().max()) for k, f in reads.items()}

    torch.manual_seed(1)
    X1 = torch.randn(5, 4)
    X2 = torch.cat([X1, X1])        # 똑같은 다중집합을 두 배로
    gaps2 = {k: float((f(X1) - f(X2)).abs().max()) for k, f in reads.items()}

    # --- 그림이 주장하는 것을 먼저 확인한다 -------------------------------
    # 평균은 **정확히** 0 이 아니다 — 3 으로 나누느냐 12 로 나누느냐에서
    # float32 반올림이 남는다. 최대만 정확히 0 이다. 그대로 적는다.
    assert gaps["mean"] < 1e-5 and gaps["max"] == 0.0, \
        "평균과 최대는 크기를 못 보아야 한다"
    assert gaps["sum"] > 1.0, "합은 크기를 보아야 한다"
    assert gaps2["mean"] < 1e-5 and gaps2["max"] == 0.0

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.6, 4.0),
                                   gridspec_kw={"width_ratios": [1, 1.15]})

    # --- 왼쪽: 같은 피처, 다른 크기 ---------------------------------------
    for k, (cnt, cx, lab) in enumerate(((3, 0.26, "3 nodes"),
                                        (12, 0.74, "12 nodes"))):
        cols = 2 if cnt == 3 else 4
        for i in range(cnt):
            r, c = divmod(i, cols)
            ax1.add_patch(plt.Circle((cx - 0.055 * (cols - 1) / 2 + c * 0.055,
                                      0.62 - r * 0.085), 0.021,
                                     facecolor=ACCENT, alpha=0.85,
                                     edgecolor="white", linewidth=0.9))
        ax1.text(cx, 0.80, lab, ha="center", fontsize=10, color=INK,
                 weight="bold")
    ax1.text(0.5, 0.20, "every node carries the\nsame feature vector",
             ha="center", fontsize=9, color=INK)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0.05, 0.92)
    ax1.set_aspect("equal")
    ax1.axis("off")
    ax1.set_title("one graph, and a bigger copy of it", fontsize=10,
                  color=INK, weight="bold")

    # --- 오른쪽: 세 readout 이 본 차이 ------------------------------------
    names = list(reads)
    vals = [gaps[k] for k in names]
    cols = [ACCENT if v > 1e-5 else MUTED for v in vals]
    ax2.barh(range(3), [max(v, 0.0) for v in vals], color=cols, height=0.5,
             alpha=0.85)
    ax2.set_yticks(range(3))
    ax2.set_yticklabels(names, fontsize=10)
    ax2.invert_yaxis()
    for i, v in enumerate(vals):
        if v > 1e-5:
            ax2.text(v * 0.98, i, f"{v:.2f} ", va="center", ha="right",
                     fontsize=9.5, color="white", weight="bold")
        else:
            tag = "  exactly 0" if v == 0.0 else f"  {v:.0e}  (round-off)"
            ax2.text(max(vals) * 0.02, i, tag, va="center",
                     fontsize=9.5, color=MUTED, weight="bold")
    ax2.set_xlabel("difference between the two readouts", fontsize=9)
    ax2.set_xlim(0, max(vals) * 1.12)
    ax2.set_title("only one of them notices", fontsize=10, color=INK,
                  weight="bold")
    ax2.grid(alpha=0.13, axis="x")
    ax2.tick_params(labelsize=8, colors=INK)
    for s in ax2.spines.values():
        s.set_color(INK)
        s.set_linewidth(0.7)

    fig.text(0.5, -0.03,
             "mean is not exactly zero - dividing by 3 or by 12 leaves "
             "float32 round-off. max is.",
             ha="center", fontsize=8.6, style="italic", color=INK)
    fig.tight_layout()
    fig.savefig(HERE / "size_blindness.svg", transparent=True,
                bbox_inches="tight")
    plt.close(fig)
    print(f"wrote size_blindness.svg   {gaps}  duplicate {gaps2}")


if __name__ == "__main__":
    size_blindness()
