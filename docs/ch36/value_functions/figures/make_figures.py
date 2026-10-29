"""37.2 밸류 함수 쪽들의 그림.

규칙은 CLAUDE.md 의 Figures 항에 적어 둔 대로다.
- SVG, fonttype='path', transparent
- 글자는 모두 아스키 (matplotlib 기본 폰트에 한글이 없다)

자료를 읽지 않으므로 어디서 돌려도 된다.

실행: python docs/ch36/value_functions/figures/make_figures.py
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

NS, NA, GAMMA = 60, 4, 0.9


def _mdp(seed=0):
    rng = np.random.default_rng(seed)
    P = rng.random((NS, NA, NS))
    P /= P.sum(2, keepdims=True)
    R = rng.standard_normal((NS, NA))
    return P, R, rng


# === 그림으로 보는 AI: 되짚는 식 하나가 왜 수렴하는가 =========================
#
# 벨먼 최적 연산자 T 는 sup-노름에서 gamma-축약이다. 두 가지를 재어 그린다.
# (1) 아무 두 밸류 함수에 T 를 먹였을 때의 축약 비율 — gamma 를 넘지 않는다.
# (2) v_k 가 v* 로 가는 속도 — 결국 gamma^k 꼴로 준다.

def contraction():
    P, R, rng = _mdp()

    def T(v):
        return (R + GAMMA * P @ v).max(1)

    ratios = []
    for _ in range(400):
        v = rng.standard_normal(NS) * 5
        w = rng.standard_normal(NS) * 5
        ratios.append(float(np.abs(T(v) - T(w)).max() / np.abs(v - w).max()))
    assert max(ratios) <= GAMMA + 1e-12, f"축약이 깨졌다: {max(ratios)}"

    vstar = np.zeros(NS)
    for _ in range(600):
        vstar = T(vstar)
    v, errs = np.zeros(NS), []
    for _ in range(45):
        errs.append(float(np.abs(v - vstar).max()))
        v = T(v)
    errs = np.array(errs)
    rate = (errs[40] / errs[10]) ** (1 / 30)
    assert errs[-1] < errs[0] / 10, "줄어들지 않는다"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 4.0))

    # --- 왼쪽: 한 번 먹일 때의 축약 비율 -----------------------------------
    ax1.hist(ratios, bins=35, color=INK, alpha=0.35, edgecolor=INK,
             linewidth=0.6)
    ax1.axvline(GAMMA, color=ACCENT, linewidth=2.2)
    ax1.text(GAMMA - 0.02, ax1.get_ylim()[1] * 0.92,
             f"gamma = {GAMMA}\nthe guarantee", fontsize=8.8, color=ACCENT,
             ha="right", va="top")
    ax1.text(max(ratios) + 0.02, ax1.get_ylim()[1] * 0.45,
             f"worst actually seen\n{max(ratios):.3f}", fontsize=8.4,
             color=INK, ha="left", va="center")
    ax1.set_xlim(0, 1.0)
    ax1.set_xlabel("how much one application of T shrinks the gap",
                   fontsize=9)
    ax1.set_ylabel("out of 400 random pairs", fontsize=9)
    ax1.set_title("T never expands", fontsize=10, color=INK, weight="bold")
    ax1.grid(alpha=0.13, axis="y")

    # --- 오른쪽: 그래서 v_k 가 v* 로 간다 ----------------------------------
    ks = np.arange(len(errs))
    ax2.semilogy(ks, errs, "o-", color=ACCENT, linewidth=2.0, ms=4,
                 label="distance to the fixed point")
    ax2.semilogy(ks, errs[0] * GAMMA ** ks, "--", color=INK, linewidth=1.6,
                 label=f"gamma^k")
    ax2.set_xlabel("sweeps of the Bellman operator", fontsize=9)
    ax2.set_ylabel("max error over states", fontsize=9)
    ax2.set_title(f"the late rate settles at {rate:.3f}", fontsize=10,
                  color=INK, weight="bold")
    ax2.legend(fontsize=8.4, framealpha=0.92)
    ax2.grid(alpha=0.13, which="both")

    for ax in (ax1, ax2):
        ax.tick_params(labelsize=8, colors=INK)
        for s in ax.spines.values():
            s.set_color(INK)
            s.set_linewidth(0.7)

    fig.text(0.5, -0.04,
             f"a random MDP with {NS} states and {NA} actions. random pairs "
             "shrink faster than promised - the iteration itself settles "
             "onto exactly gamma.",
             ha="center", fontsize=8.6, style="italic", color=INK)
    fig.tight_layout()
    fig.savefig(HERE / "contraction.svg", transparent=True,
                bbox_inches="tight")
    plt.close(fig)
    print(f"wrote contraction.svg   worst ratio {max(ratios):.4f}   "
          f"late rate {rate:.4f}")


if __name__ == "__main__":
    contraction()
