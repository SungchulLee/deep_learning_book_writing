"""20.1 마르코프 사슬 쪽들의 그림.

규칙은 CLAUDE.md 의 Figures 항에 적어 둔 대로다.
- SVG, fonttype='path', transparent
- 글자는 모두 아스키 (matplotlib 기본 폰트에 한글이 없다)

자료를 읽지 않는다. 쓰는 수는 1913년 마르코프가 발표한 집계뿐이고,
비교값은 그 집계에서 여기서 셈한다.

실행: python docs/ch19/markov_chains/figures/make_figures.py
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

# 마르코프가 1913년에 발표한 집계 (『예브게니 오네긴』 첫 2만 글자)
N_LETTERS = 20_000
N_VOWEL = 8_638
N_CONS = 11_362
N_VV = 1_104          # 모음-모음 이웃 쌍


# === 그림으로 보는 AI: 독립이 아니어도 큰 수의 법칙은 선다 ====================
#
# 마르코프가 센 수만 쓰고, 독립이었다면 몇 쌍이 나왔을지는 여기서 셈한다.
# 그 둘의 차이가 그가 보이려던 것이다 — 글자는 서로 기대어 있다.

def markov_onegin():
    assert N_VOWEL + N_CONS == N_LETTERS, "집계가 맞아떨어져야 한다"

    pairs = N_LETTERS - 1
    p_v = N_VOWEL / N_LETTERS
    exp_vv = pairs * p_v * p_v                 # 독립이라면
    p_v_after_v = N_VV / N_VOWEL
    p_v_after_c = (N_VOWEL - N_VV) / N_CONS    # 대략 — 경계 효과는 무시한다
    # 이 그림이 주장하는 것 — 앞 글자를 알면 확률이 크게 달라진다
    assert N_VV < exp_vv * 0.5, "셈한 쌍이 독립 예측보다 훨씬 적어야 한다"
    assert p_v_after_v < p_v < p_v_after_c, "조건부가 양쪽으로 갈라져야 한다"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.6, 4.0))

    # --- 왼쪽: 독립이라면 몇 쌍이었을까 ------------------------------------
    ax1.bar([0, 1], [exp_vv, N_VV], color=[INK, ACCENT], width=0.52,
            alpha=0.85)
    for i, v in enumerate((exp_vv, N_VV)):
        ax1.text(i, v + exp_vv * 0.035, f"{v:,.0f}", ha="center",
                 fontsize=10, color=INK, weight="bold")
    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(["if letters were\nindependent",
                         "what Markov\ncounted"], fontsize=9)
    ax1.set_ylabel("vowel-vowel neighbours", fontsize=9)
    ax1.set_ylim(0, exp_vv * 1.22)
    ax1.set_title(f"{N_VV / exp_vv * 100:.0f}% of what independence predicts",
                  fontsize=10, color=INK, weight="bold")
    ax1.grid(alpha=0.13, axis="y")

    # --- 오른쪽: 앞 글자를 알면 ---------------------------------------------
    vals = [p_v_after_v, p_v, p_v_after_c]
    cols = [ACCENT, INK, ACCENT]
    ax2.bar(range(3), vals, color=cols, width=0.55, alpha=0.85)
    for i, v in enumerate(vals):
        ax2.text(i, v + 0.025, f"{v:.4f}", ha="center", fontsize=9.5,
                 color=INK)
    ax2.axhline(p_v, color=INK, ls=":", lw=1.2, alpha=0.6)
    ax2.set_xticks(range(3))
    ax2.set_xticklabels(["after a\nvowel", "knowing\nnothing",
                         "after a\nconsonant"], fontsize=9)
    ax2.set_ylabel("probability the letter is a vowel", fontsize=9)
    ax2.set_ylim(0, max(vals) * 1.22)
    ax2.set_title("one letter back changes everything", fontsize=10,
                  color=INK, weight="bold")
    ax2.grid(alpha=0.13, axis="y")

    for ax in (ax1, ax2):
        ax.tick_params(labelsize=8, colors=INK)
        for s in ax.spines.values():
            s.set_color(INK)
            s.set_linewidth(0.7)

    fig.text(0.5, -0.05,
             f"Markov's own tallies: {N_VOWEL:,} vowels, {N_CONS:,} consonants, "
             f"{N_VV:,} vowel-vowel pairs in the first {N_LETTERS:,} letters.",
             ha="center", fontsize=8.6, style="italic", color=INK)
    fig.tight_layout()
    fig.savefig(HERE / "markov_onegin.svg", transparent=True,
                bbox_inches="tight")
    plt.close(fig)
    print(f"wrote markov_onegin.svg   exp {exp_vv:,.0f} vs {N_VV:,}   "
          f"P(v) {p_v:.4f} -> {p_v_after_v:.4f} / {p_v_after_c:.4f}")


if __name__ == "__main__":
    markov_onegin()
