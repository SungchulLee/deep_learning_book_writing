"""2장 학습 패러다임 쪽들의 그림.

규칙은 CLAUDE.md 의 Figures 항에 적어 둔 대로다.
- SVG, fonttype='path', transparent
- 글자는 모두 아스키 (matplotlib 기본 폰트에 한글이 없다)

실행: python docs/ch02/paradigms/figures/make_figures.py
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
BLUE = "#2d6a8f"


def style(ax):
    ax.tick_params(labelsize=8, colors=INK)
    for s in ax.spines.values():
        s.set_color(INK)
        s.set_linewidth(0.7)


# === 그림으로 보는 AI: 널리 전해지는 이야기와 날짜 ============================
#
# 벨먼은 자서전에서 국방장관 윌슨의 눈을 피하려 "동적 계획법"이라는 이름을
# 골랐다고 적었다. 그런데 그 이름을 쓴 논문이 윌슨의 취임보다 먼저 나왔다.
# 날짜만 나란히 놓는다 — 어느 쪽이 옳은지 판정하지 않는다.

def bellman_dates():
    fig, ax = plt.subplots(figsize=(9.2, 2.9))

    ax.annotate("", xy=(10.0, 1.0), xytext=(0.4, 1.0),
                arrowprops=dict(arrowstyle="->", color=INK, linewidth=1.3))

    marks = [
        (2.4, "15 August 1952", ACCENT, 1,
         "Bellman, PNAS 38(8) 716-719\n"
         '"On the Theory of\nDynamic Programming"'),
        (7.0, "28 January 1953", BLUE, -1,
         "Charles E. Wilson becomes\nSecretary of Defense\n"
         "(serves to October 1957)"),
    ]
    for x, when, col, side, what in marks:
        ax.plot([x], [1.0], marker="o", color=col, markersize=10,
                markeredgecolor="white", markeredgewidth=1.6, zorder=5)
        ax.text(x, 1.0 + side * 0.17, when, ha="center", fontsize=9.5,
                weight="bold", color=col,
                va="bottom" if side > 0 else "top")
        ax.text(x, 1.0 + side * 0.40, what, ha="center", fontsize=8,
                color=INK, alpha=0.9,
                va="bottom" if side > 0 else "top")

    ax.annotate("", xy=(7.0, 0.72), xytext=(2.4, 0.72),
                arrowprops=dict(arrowstyle="<->", color=INK, lw=1.0,
                                alpha=0.6))
    ax.text(4.7, 0.59, "the name is already in print, five months early",
            ha="center", fontsize=8.5, color=INK, style="italic")

    ax.set_xlim(0, 10.4)
    ax.set_ylim(0.1, 1.95)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(HERE / "bellman_dates.svg", transparent=True,
                bbox_inches="tight")
    plt.close(fig)
    print("wrote bellman_dates.svg")


# === 그림으로 보는 AI: 딜레마는 용량 대 자료의 이야기다 ======================
#
# 정리 3 의 분해를 수로 확인하고, 같은 분해를 표본 수만 바꿔 다시 그린다.
# n=20 에서는 차수를 올리면 분산이 폭발하고, n=200 에서는 거의 공짜가 된다.

def bias_variance_depends_on_n():
    SIG = 0.30
    f = lambda t: np.sin(2 * np.pi * t)
    x0 = 0.35
    degs = [1, 2, 3, 5, 7, 9]

    def decompose(n, reps=600):
        b2, va = [], []
        for d in degs:
            preds = []
            for r in range(reps):
                rng = np.random.default_rng(10_000 * d + r)
                x = rng.uniform(0, 1, n)
                y = f(x) + rng.normal(0, SIG, n)
                preds.append(np.polyval(np.polyfit(x, y, d), x0))
            preds = np.array(preds)
            b2.append((preds.mean() - f(x0)) ** 2)
            va.append(preds.var())
        return np.array(b2), np.array(va)

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.2))
    store = {}
    for ax, n in zip(axes, (20, 200)):
        b2, va = decompose(n)
        store[n] = (b2, va)
        total = b2 + va + SIG ** 2
        xs = np.arange(len(degs))
        ax.bar(xs - 0.22, b2, 0.4, color=BLUE, alpha=0.8, label="bias squared")
        ax.bar(xs + 0.22, va, 0.4, color=ACCENT, alpha=0.85, label="variance")
        ax.plot(xs, total, "o-", color=INK, linewidth=1.8, ms=5,
                label="total (incl. noise)")
        ax.axhline(SIG ** 2, color=INK, linestyle=":", linewidth=1.1,
                   alpha=0.7)
        ax.text(len(degs) - 1.1, SIG ** 2 * 1.25, "noise floor", fontsize=8,
                color=INK, alpha=0.8)
        ax.set_xticks(xs)
        ax.set_xticklabels([str(d) for d in degs])
        ax.set_xlabel("polynomial degree", fontsize=9.5)
        ax.set_yscale("log")
        ax.set_ylim(1e-5, 4)
        ax.set_title(f"n = {n} training points", fontsize=10.5, color=INK,
                     weight="bold")
        ax.legend(fontsize=8, loc="upper left", framealpha=0.92)
        ax.grid(alpha=0.13, which="both")
        style(ax)
    axes[0].set_ylabel("error contribution (log scale)", fontsize=9.5)

    ratio = store[20][1][-1] / store[200][1][-1]
    axes[1].annotate(f"variance at degree 9\nis {ratio:.0f}x smaller here",
                     xy=(5, store[200][1][-1]), xytext=(2.1, 0.25),
                     fontsize=8.5, color=ACCENT,
                     arrowprops=dict(arrowstyle="->", color=ACCENT, lw=0.9))

    fig.text(0.5, -0.04,
             "the same capacity is reckless at n = 20 and nearly free at "
             "n = 200. the dilemma is about capacity relative to data.",
             ha="center", fontsize=8.5, style="italic", color=INK)
    fig.tight_layout()
    fig.savefig(HERE / "bias_variance_n.svg", transparent=True,
                bbox_inches="tight")
    plt.close(fig)
    print(f"wrote bias_variance_n.svg   var@deg9  n=20 {store[20][1][-1]:.5f}"
          f"  n=200 {store[200][1][-1]:.5f}  ratio {ratio:.0f}")


if __name__ == "__main__":
    bellman_dates()
    bias_variance_depends_on_n()
