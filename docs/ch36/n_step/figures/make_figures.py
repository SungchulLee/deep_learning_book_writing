"""37.4 n-스텝 쪽들의 그림.

규칙은 CLAUDE.md 의 Figures 항에 적어 둔 대로다.
- SVG, fonttype='path', transparent
- 글자는 모두 아스키 (matplotlib 기본 폰트에 한글이 없다)

자료를 읽지 않으므로 어디서 돌려도 된다.

실행: python docs/ch36/n_step/figures/make_figures.py
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

TRUE = np.arange(1, 6) / 6.0        # 서턴 1988 무작위 걸음의 참값
N_EPISODE, N_SEED = 100, 60
LAMBDAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
ALPHAS = np.arange(0.01, 0.65, 0.02)


def _episode(rng):
    s, traj = 2, []
    while 0 <= s <= 4:
        traj.append(s)
        s += 1 if rng.random() < 0.5 else -1
    return traj, (1.0 if s > 4 else 0.0)


def _run(lam, alpha, seed):
    rng = np.random.default_rng(seed)
    V = np.full(5, 0.5)
    for _ in range(N_EPISODE):
        traj, R = _episode(rng)
        e = np.zeros(5)
        for i, s in enumerate(traj):
            last = i == len(traj) - 1
            delta = (R if last else 0.0) \
                + (0.0 if last else V[traj[i + 1]]) - V[s]
            e *= lam
            e[s] += 1
            V += alpha * delta * e
    return float(np.sqrt(((V - TRUE) ** 2).mean()))


# === 그림으로 보는 AI: 가운데가 가장 좋다 =====================================
#
# 서턴 1988 의 무작위 걸음을 그대로 돌린다. lambda 마다 alpha 를 최적화해
# 가장 좋은 RMSE 를 재면 **가운데 lambda** 가 이긴다 — 다만 lambda=0 과의
# 차이는 작고 lambda=1 과의 차이는 크다. 둘 다 그림에 적는다.

def lambda_sweep():
    best_rmse, best_alpha = [], []
    for lam in LAMBDAS:
        curve = [np.mean([_run(lam, a, s) for s in range(N_SEED)])
                 for a in ALPHAS]
        i = int(np.argmin(curve))
        best_rmse.append(float(curve[i]))
        best_alpha.append(float(ALPHAS[i]))

    best_rmse = np.array(best_rmse)
    i_best = int(np.argmin(best_rmse))
    lam_best = LAMBDAS[i_best]
    assert 0.0 < lam_best < 1.0, f"가운데가 이기지 않았다: lambda={lam_best}"

    gain_td = best_rmse[0] / best_rmse[i_best]
    gain_mc = best_rmse[-1] / best_rmse[i_best]

    fig, ax = plt.subplots(figsize=(7.8, 4.5))
    ax.plot(LAMBDAS, best_rmse, "o-", color=ACCENT, linewidth=2.2, ms=6)
    ax.plot([lam_best], [best_rmse[i_best]], "o", color=ACCENT, ms=13,
            markerfacecolor="white", markeredgewidth=2.4, zorder=5)

    ax.annotate(f"best at lambda = {lam_best}", xy=(lam_best,
                                                    best_rmse[i_best]),
                xytext=(lam_best + 0.10, best_rmse[i_best] + 0.009),
                fontsize=9, color=ACCENT,
                arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.0))
    ax.annotate(f"TD(0)\n{gain_td:.2f}x worse", xy=(0.0, best_rmse[0]),
                xytext=(0.02, best_rmse[0] + 0.012), fontsize=8.4, color=INK)
    ax.annotate(f"Monte Carlo\n{gain_mc:.2f}x worse", xy=(1.0, best_rmse[-1]),
                xytext=(0.80, best_rmse[-1] - 0.016), fontsize=8.4,
                color=INK, ha="center")

    ax.set_xlabel("lambda", fontsize=9.5)
    ax.set_ylabel("RMSE after 100 episodes, best alpha for each lambda",
                  fontsize=9)
    ax.set_title("Sutton's 1988 random walk, run again", fontsize=10.5,
                 color=INK, weight="bold")
    ax.grid(alpha=0.13)
    ax.tick_params(labelsize=8, colors=INK)
    for s in ax.spines.values():
        s.set_color(INK)
        s.set_linewidth(0.7)

    fig.text(0.5, -0.04,
             f"averaged over {N_SEED} seeds. the win over Monte Carlo is "
             "large; the win over TD(0) is slight.",
             ha="center", fontsize=8.6, style="italic", color=INK)
    fig.tight_layout()
    fig.savefig(HERE / "lambda_sweep.svg", transparent=True,
                bbox_inches="tight")
    plt.close(fig)
    print(f"wrote lambda_sweep.svg   best lambda {lam_best}   "
          f"vs TD(0) {gain_td:.3f}x   vs MC {gain_mc:.3f}x")


if __name__ == "__main__":
    lambda_sweep()
