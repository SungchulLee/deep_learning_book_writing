"""17장 장 소개 — 「그림으로 보는 AI」 그림.

규칙은 CLAUDE.md 의 Figures 항에 적어 둔 대로다.
- SVG, fonttype='path', transparent
- 글자는 모두 아스키 (matplotlib 기본 폰트에 한글이 없다)

실행: python docs/ch16/figures/make_figures.py
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

# === 그림으로 보는 AI: 이 장의 연표 ==========================================
#
# 여기 적는 날짜는 **모두 이 장의 다른 쪽 상자에서 확인한 것**이다. 기억으로
# 적은 것은 하나도 없다. 쪽 이름을 함께 들고 있다가, 그 쪽이 실제로 있고
# 상자가 들어 있는지 확인한 뒤에만 그린다.

ENTRIES = [
    (1989, "catastrophic\ninterference named",
     "continual_learning/catastrophic_forgetting.md", "problem"),
    (1995, "rehearsal and\npseudorehearsal",
     "replay_methods/experience_replay.md", "replay"),
    (2015, "distillation,\nand the T^2",
     "distillation_methods/lwf.md", "distil"),
    (2016, "Learning without\nForgetting",
     "distillation_methods/lwf.md", "distil"),
    (2017, "EWC", "regularization_methods/ewc.md", "penalty"),
    (2017, "GEM, and the\nBWT/FWT metrics",
     "continual_learning/evaluation_metrics.md", "replay"),
    (2018, "the note, and\nthe reply",
     "regularization_methods/ewc.md", "penalty"),
    (2019, "A-GEM", "replay_methods/agem.md", "replay"),
    (2019, "supermask",
     "architecture_methods/supermask.md", "structure"),
]

COLOURS = {"problem": INK, "replay": ACCENT, "distil": MUTED,
           "penalty": "#6b6f73", "structure": "#9a6b4f"}


def chapter_timeline():
    docs = HERE.parent
    for _, _, page, _ in ENTRIES:
        p = docs / page
        assert p.exists(), f"연표가 가리키는 쪽이 없다: {page}"
        assert 'class="storybox"' in p.read_text(encoding="utf-8"), (
            f"그 쪽에 상자가 없다 — 기억으로 적은 날짜일 수 있다: {page}")

    fig, ax = plt.subplots(figsize=(11.0, 4.0))
    years = [e[0] for e in ENTRIES]
    span = max(years) - min(years)

    ax.annotate("", xy=(len(ENTRIES) - 0.4, 0), xytext=(-0.6, 0),
                arrowprops=dict(arrowstyle="->", color=INK, linewidth=1.3))

    for i, (yr, label, _, kind) in enumerate(ENTRIES):
        col = COLOURS[kind]
        up = i % 2 == 0
        y = 0.30 if up else -0.30
        ax.plot([i], [0], marker="o", color=col, markersize=9,
                markeredgecolor="white", markeredgewidth=1.5, zorder=5)
        ax.plot([i, i], [0, y * 0.62], color=col, linewidth=1.0, alpha=0.6)
        ax.text(i, y, str(yr), ha="center",
                va="bottom" if up else "top", fontsize=10.5, color=col,
                weight="bold")
        ax.text(i, y + (0.16 if up else -0.16), label, ha="center",
                va="bottom" if up else "top", fontsize=8.2, color=INK,
                alpha=0.9)

    ax.text((len(ENTRIES) - 1) / 2, -0.93,
            f"{span} years, and the first entry is the problem, "
            f"not a method",
            ha="center", fontsize=9, color=INK, style="italic")

    ax.set_xlim(-1.0, len(ENTRIES) - 0.2)
    ax.set_ylim(-1.05, 0.95)
    ax.axis("off")
    ax.text(0.5, 1.0, "not to scale - spacing is even, not by year",
            transform=ax.transAxes, ha="center", fontsize=6.6,
            color=INK, alpha=0.55, style="italic")

    fig.tight_layout()
    fig.savefig(HERE / "chapter_timeline.svg", transparent=True,
                bbox_inches="tight")
    plt.close(fig)
    print(f"wrote chapter_timeline.svg   {min(years)}-{max(years)}, "
          f"{len(ENTRIES)} entries, all verified against pages")


if __name__ == "__main__":
    chapter_timeline()
