"""37.3 동적 계획법 쪽들의 그림.

규칙은 CLAUDE.md 의 Figures 항에 적어 둔 대로다.
- SVG, fonttype='path', transparent
- 글자는 모두 아스키 (matplotlib 기본 폰트에 한글이 없다)

자료를 읽지 않으므로 어디서 돌려도 된다.

실행: python docs/ch36/dynamic_programming/figures/make_figures.py
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

NS, NA, GAMMA = 60, 4, 0.9
TOL = 1e-10


# === 그림으로 보는 AI: 왜 폴리시 반복은 몇 걸음이면 끝나는가 ===================
#
# 같은 MDP 에 두 방법을 돌려 고정점까지의 오차를 되풀이마다 잰다.
# 폴리시 반복은 한 걸음이 비싼 대신 몇 걸음이면 끝나고, 값 되풀이는 한 걸음이
# 싼 대신 gamma^k 로 기어간다. 수는 모두 여기서 재어 얻는다.

def pi_vs_vi():
    rng = np.random.default_rng(0)
    P = rng.random((NS, NA, NS))
    P /= P.sum(2, keepdims=True)
    R = rng.standard_normal((NS, NA))

    def T(v):
        return (R + GAMMA * P @ v).max(1)

    vstar = np.zeros(NS)
    for _ in range(2000):
        vstar = T(vstar)

    # --- 폴리시 반복 -------------------------------------------------------
    pi = np.zeros(NS, dtype=int)
    pi_err = []
    while True:
        Ppi = P[np.arange(NS), pi]
        Rpi = R[np.arange(NS), pi]
        vpi = np.linalg.solve(np.eye(NS) - GAMMA * Ppi, Rpi)
        pi_err.append(float(np.abs(vpi - vstar).max()))
        newpi = (R + GAMMA * P @ vpi).argmax(1)
        if (newpi == pi).all():
            break
        pi = newpi

    # --- 값 되풀이 ---------------------------------------------------------
    v = np.zeros(NS)
    vi_err = []
    while True:
        vi_err.append(float(np.abs(v - vstar).max()))
        nv = T(v)
        if np.abs(nv - v).max() < TOL:
            break
        v = nv

    assert len(pi_err) * 10 < len(vi_err), "두 방법의 걸음 수가 비슷하다"
    assert pi_err[-1] < 1e-8, f"폴리시 반복이 최적에 닿지 않았다: {pi_err[-1]}"

    fig, ax = plt.subplots(figsize=(7.8, 4.4))
    ax.semilogy(np.arange(1, len(vi_err) + 1),
                np.maximum(vi_err, 1e-16), "--", color=INK, linewidth=1.8,
                label=f"value iteration  ({len(vi_err)} sweeps)")
    ax.semilogy(np.arange(1, len(pi_err) + 1),
                np.maximum(pi_err, 1e-16), "o-", color=ACCENT, linewidth=2.4,
                ms=8, label=f"policy iteration  ({len(pi_err)} rounds)")

    ax.annotate("exact, and it stops",
                xy=(len(pi_err), max(pi_err[-1], 1e-16)),
                xytext=(len(pi_err) + 14, 1e-7), fontsize=8.8, color=ACCENT,
                arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.0))
    ax.set_xlabel("iterations", fontsize=9.5)
    ax.set_ylabel("max error against the optimal values", fontsize=9.5)
    ax.set_title("the same MDP, solved two ways", fontsize=10.5, color=INK,
                 weight="bold")
    ax.legend(fontsize=8.8, loc="upper right", framealpha=0.92)
    ax.grid(alpha=0.13, which="both")
    ax.tick_params(labelsize=8, colors=INK)
    for s in ax.spines.values():
        s.set_color(INK)
        s.set_linewidth(0.7)

    fig.text(0.5, -0.04,
             f"{NS} states, {NA} actions, gamma {GAMMA}. one of them pays "
             "a linear solve per round and buys the whole policy with it.",
             ha="center", fontsize=8.6, style="italic", color=INK)
    fig.tight_layout()
    fig.savefig(HERE / "pi_vs_vi.svg", transparent=True, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote pi_vs_vi.svg   policy iteration {len(pi_err)}   "
          f"value iteration {len(vi_err)}")


if __name__ == "__main__":
    pi_vs_vi()
