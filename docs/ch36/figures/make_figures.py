"""37장 장 소개의 그림.

규칙은 CLAUDE.md 의 Figures 항에 적어 둔 대로다.
- SVG, fonttype='path', transparent
- 글자는 모두 아스키 (matplotlib 기본 폰트에 한글이 없다)

여기 적힌 열 개의 날짜는 모두 **이 장의 다른 쪽에 실제로 쓴 상자**에서 왔다.
외워서 적은 것이 하나도 없어야 한다 — SOURCES 의 쪽 이름이 그 보증이다.

실행: python docs/ch36/figures/make_figures.py
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
GREEN = "#3f6b52"
OLIVE = "#8a6f2a"

# (해, 표시할 해, 누구, 이 장에 준 것, 갈래, 그 상자가 있는 쪽)
EVENTS = [
    (1937, "1937", "Samuelson", "discounting", "economics",
     "mdp/discount.md"),
    (1957, "1957", "Bellman", "the optimality principle", "operations research",
     "value_functions/bellman.md"),
    (1960, "1960", "Howard", "policy iteration", "operations research",
     "dynamic_programming/policy_iteration.md"),
    (1961, "1961", "Minsky", "credit assignment", "artificial intelligence",
     "value_functions/state_value.md"),
    (1972, "1972", "Klopf", "eligibility", "neuroscience",
     "n_step/eligibility_traces.md"),
    (1988, "1988", "Sutton", "TD(lambda)", "reinforcement learning",
     "n_step/td_lambda.md"),
    (1989, "1989", "Watkins", "Q-learning", "reinforcement learning",
     "value_functions/action_value.md"),
    (1990, "~1990", "Sutton and Littman", "the reward hypothesis",
     "reinforcement learning", "introduction/reward_hypothesis.md"),
    (1992, "1992", "Watkins and Dayan", "the convergence proof",
     "reinforcement learning", "value_functions/action_value.md"),
    (2004, "2004", "Sutton", "the value-function hypothesis",
     "reinforcement learning", "value_functions/state_value.md"),
]

COLOURS = {"economics": MUTED, "operations research": INK,
           "artificial intelligence": GREEN, "neuroscience": OLIVE,
           "reinforcement learning": ACCENT}


# === 그림으로 보는 AI: 이 장은 67 년에 걸쳐 모였다 =============================

def chapter_timeline():
    # 날짜마다 출처 쪽이 실제로 있는지 확인하고 그린다
    root = HERE.parent
    for _, _, _, _, _, page in EVENTS:
        assert (root / page).exists(), f"출처 쪽이 없다: {page}"
    years = [e[0] for e in EVENTS]
    assert years == sorted(years), "해가 차례대로가 아니다"
    span = years[-1] - years[0]

    fig, ax = plt.subplots(figsize=(11.0, 4.6))
    ax.annotate("", xy=(2010, 0), xytext=(1931, 0),
                arrowprops=dict(arrowstyle="->", color=INK, linewidth=1.4))

    # 1988~1992 가 붙어 있어 층을 나눠 적는다
    # 1957/1961 과 1988~1992 가 가까워 층을 벌려 적는다
    levels = [1.30, -1.30, 1.30, -2.15, 1.30, 1.72, -1.30, 0.80, -2.05, 1.30]
    for (yr, lab, who, gave, field, _), lv in zip(EVENTS, levels):
        col = COLOURS[field]
        ax.plot([yr, yr], [0, lv * 0.78], color=col, linewidth=1.0,
                alpha=0.55)
        ax.plot([yr], [0], marker="o", color=col, markersize=7,
                markeredgecolor="white", markeredgewidth=1.2, zorder=4)
        va = "bottom" if lv > 0 else "top"
        ax.text(yr, lv * 0.84, f"{lab}\n{who}\n{gave}", ha="center", va=va,
                fontsize=8.0, color=col, linespacing=1.45)

    ax.text(1931.5, 2.30, f"{span} years, five fields", fontsize=10.5,
            color=INK, weight="bold")
    for i, (field, col) in enumerate(COLOURS.items()):
        ax.text(1931.5 + i * 15.5, -2.78, field, fontsize=7.8, color=col)

    ax.set_xlim(1930, 2011)
    ax.set_ylim(-3.05, 2.55)
    ax.axis("off")

    fig.text(0.5, -0.02,
             "every date here comes from a box written on a page of this "
             "chapter, not from memory.",
             ha="center", fontsize=8.6, style="italic", color=INK)
    fig.tight_layout()
    fig.savefig(HERE / "chapter_timeline.svg", transparent=True,
                bbox_inches="tight")
    plt.close(fig)
    print(f"wrote chapter_timeline.svg   {years[0]}-{years[-1]}, "
          f"{span} years, {len(EVENTS)} dates")


if __name__ == "__main__":
    chapter_timeline()
