"""39.2 액터-크리틱 쪽들의 그림.

규칙은 CLAUDE.md 의 Figures 항에 적어 둔 대로다.
- SVG, fonttype='path', transparent
- 글자는 모두 아스키 (matplotlib 기본 폰트에 한글이 없다)

작은 복도에서 값을 닫힌 식으로 셈한다. 무작위가 없어 다시 돌려도 같다.

실행: python docs/ch38/actor_critic/figures/make_figures.py
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


# === 그림으로 보는 AI: 비평자가 처음 맡은 일 ==================================
#
# 1983 년의 ACE 는 분산을 줄이려고 만든 것이 아니라, 끝에서 한 번만 오는 신호를
# **걸음마다 쓸 수 있는 신호로 바꾸려고** 만든 것이다. 끝에서만 리워드가 오는
# 복도에서 몬테카를로와 비평자의 신호를 나란히 셈해 그 차이를 보인다.

def critic_localises_blame():
    N, GAMMA = 6, 0.9
    V = np.array([GAMMA ** (N - 1 - s) for s in range(N)])
    traj = [0, 1, 2, 3, 2, 3, 4, 5]                  # 네 번째 걸음에서 뒤로 샜다
    T = len(traj) - 1

    G = 1.0 * GAMMA ** T                             # 끝에서만 오는 리턴
    td = []
    for t in range(T):
        s, s2 = traj[t], traj[t + 1]
        r = 1.0 if s2 == N - 1 else 0.0
        td.append(r + (0.0 if s2 == N - 1 else GAMMA * V[s2]) - V[s])
    td = np.array(td)

    bad = int(np.argmin(td))
    assert td[bad] < -1e-9, "잘못한 걸음에 음의 TD 오차가 있어야 한다"
    assert np.sum(np.abs(td) > 1e-9) < T, "TD 오차가 모든 걸음에 있으면 요지가 없다"

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.4, 5.0), sharex=True,
                                   gridspec_kw={"height_ratios": [1, 1.25]})

    xs = np.arange(T)
    ax1.bar(xs, np.full(T, G), color=MUTED, alpha=0.65, width=0.62)
    ax1.set_ylabel("Monte Carlo\ncredit", fontsize=9.2)
    ax1.set_title("the same episode, scored two ways", fontsize=10.5,
                  color=INK, weight="bold", pad=8)
    ax1.text(T - 0.5, G * 1.04, f"every step gets {G:.4f}", fontsize=8.6,
             color=MUTED, ha="right", va="bottom", weight="bold")
    ax1.set_ylim(0, G * 1.32)

    cols = [ACCENT if t == bad else (INK if abs(v) > 1e-9 else "#c8c8c8")
            for t, v in enumerate(td)]
    ax2.bar(xs, td, color=cols, alpha=0.85, width=0.62)
    ax2.axhline(0, color=INK, linewidth=0.9)
    ax2.annotate(f"the one wrong step\n{td[bad]:+.4f}", xy=(bad, td[bad]),
                 xytext=(0, -42), textcoords="offset points", ha="center",
                 fontsize=8.8, color=ACCENT,
                 arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.0))
    ax2.set_ylabel("critic's\nTD error", fontsize=9.2)
    ax2.set_xticks(xs)
    ax2.set_xticklabels([f"{traj[t]}->{traj[t+1]}" for t in range(T)],
                        fontsize=8.4)
    ax2.set_xlabel("each step of the episode", fontsize=9.4)
    ax2.set_ylim(td.min() * 1.9, max(td.max() * 1.5, 0.05))

    for ax in (ax1, ax2):
        ax.grid(alpha=0.13, axis="y")
        ax.tick_params(labelsize=8.4, colors=INK)
        for sp in ax.spines.values():
            sp.set_color(INK)
            sp.set_linewidth(0.7)

    fig.text(0.5, -0.03,
             "reward arrives only at the end, so Monte Carlo cannot tell the "
             "mistake from the six good steps.",
             ha="center", fontsize=8.8, style="italic", color=INK)
    fig.tight_layout()
    fig.savefig(HERE / "critic_localises.svg", transparent=True,
                bbox_inches="tight")
    plt.close(fig)
    print(f"wrote critic_localises.svg   MC {G:.4f} for all {T} steps   "
          f"TD worst {td[bad]:+.4f} at step {bad}")


if __name__ == "__main__":
    critic_localises_blame()
