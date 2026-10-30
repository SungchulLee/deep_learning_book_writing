"""20.5 시뮬레이티드 어닐링 쪽들의 그림.

규칙은 CLAUDE.md 의 Figures 항에 적어 둔 대로다.
- SVG, fonttype='path', transparent
- 글자는 모두 아스키 (matplotlib 기본 폰트에 한글이 없다)

자료를 읽지 않는다. 씨앗을 박아 두어 다시 돌려도 같은 수가 나온다.

실행: python docs/ch19/mcmc/simulated_annealing/figures/make_figures.py
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

# 한 줄짜리 풍경 — 전역 최소(자리 0)와 국소 최소(자리 6) 사이에 벽이 있다
ENERGY = np.array([0.0, 1.5, 3.0, 4.5, 3.2, 1.8, 0.6, 2.4, 4.0, 2.6, 1.2, 2.0])


# === 그림으로 보는 AI: 이론이 맞는 온도와 사슬이 닿는 온도 ====================
#
# 고정 온도에서 메트로폴리스를 돌리면 볼츠만 분포를 표본한다 — **높은 온도에서는**.
# 온도를 낮추면 이론은 전역 최소에 몰리라고 하는데, 정작 사슬은 벽을 못 넘어
# 국소 최소에 갇힌다. 어닐링이 "식혀 가며" 도는 까닭이 그 틈이다.

def theory_versus_chain():
    n = len(ENERGY)
    gmin, lmin = int(np.argmin(ENERGY)), 6

    def run(T, steps=200_000, seed=0, start=6):
        r = np.random.default_rng(seed)
        x = start
        occ = np.zeros(n)
        for _ in range(steps):
            y = (x + r.choice([-1, 1])) % n
            if r.random() < np.exp(-(ENERGY[y] - ENERGY[x]) / T):
                x = y
            occ[x] += 1
        return occ / steps

    Ts = [3.0, 1.0, 0.5, 0.2, 0.05]
    measured, theory = [], []
    for T in Ts:
        occ = run(T)
        b = np.exp(-ENERGY / T)
        b /= b.sum()
        measured.append(occ[gmin])
        theory.append(b[gmin])

    # 이 그림이 주장하는 것
    assert abs(measured[0] - theory[0]) < 0.02, "높은 온도에서는 이론과 맞아야 한다"
    assert measured[-1] < 0.01 < theory[-1], \
        "낮은 온도에서 이론은 몰리라는데 사슬은 못 가야 한다"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.8, 4.0),
                                   gridspec_kw={"width_ratios": [1, 1.2]})

    # --- 왼쪽: 풍경 --------------------------------------------------------
    ax1.plot(range(n), ENERGY, "o-", color=INK, lw=1.8, ms=5)
    ax1.plot([gmin], [ENERGY[gmin]], "o", color=ACCENT, ms=11, zorder=5)
    ax1.plot([lmin], [ENERGY[lmin]], "o", color=MUTED, ms=11, zorder=5)
    ax1.annotate("the answer", xy=(gmin, ENERGY[gmin]), xytext=(14, 26),
                 textcoords="offset points", fontsize=8.8, color=ACCENT,
                 arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.0))
    ax1.annotate("where it starts,\nand stays", xy=(lmin, ENERGY[lmin]),
                 xytext=(-10, 30), textcoords="offset points", fontsize=8.8,
                 color=MUTED, ha="center",
                 arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.0))
    ax1.set_xlabel("position", fontsize=9)
    ax1.set_ylabel("energy", fontsize=9)
    ax1.set_title("a wall between two valleys", fontsize=10, color=INK,
                  weight="bold")
    ax1.grid(alpha=0.13)

    # --- 오른쪽: 이론과 실제 ----------------------------------------------
    xs = np.arange(len(Ts))
    w = 0.38
    ax2.bar(xs - w / 2, theory, width=w, color=INK, alpha=0.75,
            label="what Boltzmann says")
    ax2.bar(xs + w / 2, measured, width=w, color=ACCENT, alpha=0.9,
            label="where the chain actually was")
    for i, (t, m) in enumerate(zip(theory, measured)):
        if m < 0.02:
            ax2.text(i + w / 2, 0.03, f"{m:.4f}", ha="center", fontsize=8.2,
                     color=ACCENT, weight="bold", rotation=90)
    ax2.set_xticks(xs)
    ax2.set_xticklabels([f"T={t:g}" for t in Ts], fontsize=8.6)
    ax2.set_ylabel("time spent at the global minimum", fontsize=9)
    ax2.set_ylim(0, 1.12)
    ax2.set_title("they agree, until they matter", fontsize=10, color=INK,
                  weight="bold")
    ax2.legend(fontsize=8.2, loc="upper left", framealpha=0.92)
    ax2.grid(alpha=0.13, axis="y")

    for ax in (ax1, ax2):
        ax.tick_params(labelsize=8, colors=INK)
        for s in ax.spines.values():
            s.set_color(INK)
            s.set_linewidth(0.7)

    fig.text(0.5, -0.04,
             f"at T={Ts[-1]:g} the theory puts {theory[-1]:.4f} of the time at "
             f"the answer. the chain spent {measured[-1]:.4f}.",
             ha="center", fontsize=8.8, style="italic", color=INK)
    fig.tight_layout()
    fig.savefig(HERE / "theory_vs_chain.svg", transparent=True,
                bbox_inches="tight")
    plt.close(fig)
    print(f"wrote theory_vs_chain.svg   theory {[round(t,4) for t in theory]}"
          f"   measured {[round(m,4) for m in measured]}")


if __name__ == "__main__":
    theory_versus_chain()
