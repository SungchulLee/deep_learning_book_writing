"""14장 장 소개의 그림.

규칙은 CLAUDE.md 의 Figures 항에 적어 둔 대로다.
- SVG, fonttype='path', transparent
- 글자는 모두 아스키 (matplotlib 기본 폰트에 한글이 없다)

자료를 읽지 않고 학습도 하지 않는다. **논문이 보고한 수**를 옮겨 그리고,
옮긴 것임을 그림 안에 적는다.

실행: python docs/ch13/figures/make_figures.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["svg.fonttype"] = "path"

import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
INK = "#2f3337"
ACCENT = "#8a4b2a"
MUTED = "#2d6a8f"

# Richemond 외 2020 (arXiv:2010.10241) 이 보고한 수. 여기서 재지 않았다.
BARS = (
    ("vanilla BYOL\n(with batch norm)", 74.3, INK),
    ("group norm +\nweight standardisation", 73.9, ACCENT),
    ("the no-batch-norm\ncase they tested", 65.7, MUTED),
)


def collapse_debate():
    labels = [b[0] for b in BARS]
    vals = [b[1] for b in BARS]
    cols = [b[2] for b in BARS]

    # 그림이 주장하는 것을 먼저 확인한다
    assert abs(vals[0] - vals[1]) < 1.0, \
        "배치 통계를 뺀 쪽이 비슷하지 않으면 이 그림의 이야기가 없다"
    assert vals[2] < vals[1], "무너지지 않았다는 쪽이 더 낮아야 이야기가 맞다"

    fig, ax = plt.subplots(figsize=(8.8, 4.2))
    bars = ax.bar(range(3), vals, color=cols, width=0.56, alpha=0.88)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.6, f"{v}", ha="center", fontsize=11, color=INK,
                weight="bold")

    ax.annotate("", xy=(1, vals[1] + 2.4), xytext=(0, vals[0] + 2.4),
                arrowprops=dict(arrowstyle="<->", color=ACCENT, lw=1.4))
    ax.text(0.5, vals[0] + 3.2, "0.4 apart", ha="center", fontsize=9.5,
            color=ACCENT, weight="bold")

    ax.set_xticks(range(3))
    ax.set_xticklabels(labels, fontsize=8.8)
    ax.set_ylabel("ImageNet top-1 (%)   - reported, not measured here",
                  fontsize=9)
    ax.set_ylim(60, 80)
    ax.set_title("the batch-statistics explanation did not survive testing",
                 fontsize=10.5, color=INK, weight="bold", pad=10)
    ax.grid(alpha=0.14, axis="y")
    ax.tick_params(labelsize=8, colors=INK)
    for s in ax.spines.values():
        s.set_color(INK)
        s.set_linewidth(0.7)

    fig.text(0.5, -0.06,
             "if batch statistics were the hidden negatives, removing them "
             "should have caused collapse. it did not.",
             ha="center", fontsize=8.7, style="italic", color=INK)
    fig.tight_layout()
    fig.savefig(HERE / "collapse_debate.svg", transparent=True,
                bbox_inches="tight")
    plt.close(fig)
    print(f"wrote collapse_debate.svg   {vals}")


if __name__ == "__main__":
    collapse_debate()
