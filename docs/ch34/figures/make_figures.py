"""35장 장 소개의 그림.

규칙은 CLAUDE.md 의 Figures 항에 적어 둔 대로다.
- SVG, fonttype='path', transparent
- 글자는 모두 아스키 (matplotlib 기본 폰트에 한글이 없다)

바깥 자료를 읽지 않는다. 참값을 알아야 "참 최적 목록"을 만들 수 있으므로
**만든 자료**를 쓴다. 넷플릭스에서 실제로 일어난 일을 재현한 것이 아니라,
매김 오차가 조금 줄 때 목록이 얼마나 바뀌는지를 보이는 장치다.
그림 안에 만든 자료임을 적는다.

실행: python docs/ch34/figures/make_figures.py
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

N_U, N_I = 2000, 500
KS = (5, 10, 20)


# === 그림으로 보는 AI: 매김을 잘 맞히는 일과 목록을 잘 짜는 일 ================
#
# RMSE 를 10% 줄이면 추천 목록이 얼마나 좋아지는가. 그리고 얼마나 바뀌는가.
# 둘은 다른 물음이고, 둘 다 재어 보아야 한다. 수는 전부 여기서 셈한다.

def rmse_versus_ranking():
    rng = np.random.default_rng(0)
    truth = rng.normal(3.6, 1.0, (N_U, N_I))
    worse = truth + rng.normal(0, 1.00, (N_U, N_I))
    better = truth + rng.normal(0, 0.90, (N_U, N_I))

    def rmse(p):
        return float(np.sqrt(((p - truth) ** 2).mean()))

    r_w, r_b = rmse(worse), rmse(better)
    gain = (1 - r_b / r_w) * 100

    def topk(p, k):
        return np.argsort(-p, 1)[:, :k]

    hit_w, hit_b, churn = [], [], []
    for k in KS:
        tw, tb, tt = topk(worse, k), topk(better, k), topk(truth, k)
        hit_w.append(np.mean([len(set(a) & set(c)) for a, c in zip(tw, tt)]) / k)
        hit_b.append(np.mean([len(set(a) & set(c)) for a, c in zip(tb, tt)]) / k)
        churn.append(1 - np.mean([len(set(a) & set(b_)) for a, b_ in
                                  zip(tw, tb)]) / k)

    # 그림이 주장하는 것을 먼저 확인한다
    assert 8 < gain < 12, "RMSE 차이가 10% 언저리가 아니면 이 그림의 이야기가 없다"
    assert all(b > w for b, w in zip(hit_b, hit_w)), \
        "더 낮은 RMSE 가 목록을 낫게 하지 않으면 왼쪽 이야기가 없다"
    assert min(churn) > 0.7, "목록이 크게 바뀌지 않으면 오른쪽 이야기가 없다"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.6, 4.2))
    x = np.arange(len(KS))

    ax1.bar(x - 0.19, hit_w, width=0.36, color=INK, alpha=0.8,
            label=f"RMSE {r_w:.3f}")
    ax1.bar(x + 0.19, hit_b, width=0.36, color=ACCENT, alpha=0.88,
            label=f"RMSE {r_b:.3f}")
    for i in range(len(KS)):
        ax1.text(x[i], max(hit_w[i], hit_b[i]) + 0.016,
                 f"+{(hit_b[i] - hit_w[i]) * 100:.1f}", ha="center",
                 fontsize=8.8, color=ACCENT, weight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"top {k}" for k in KS], fontsize=9)
    ax1.set_ylabel("share of the truly best items found", fontsize=9)
    ax1.set_ylim(0, max(hit_b) * 1.35)
    ax1.set_title(f"a {gain:.1f}% better RMSE helps the list",
                  fontsize=10, color=INK, weight="bold")
    ax1.legend(fontsize=8.4, framealpha=0.92)

    ax2.bar(x, churn, width=0.5, color=MUTED, alpha=0.85)
    for i, c in enumerate(churn):
        ax2.text(x[i], c + 0.016, f"{c * 100:.0f}%", ha="center",
                 fontsize=9.5, color=INK, weight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"top {k}" for k in KS], fontsize=9)
    ax2.set_ylabel("share of the list that changed", fontsize=9)
    ax2.set_ylim(0, 1.0)
    ax2.set_title("and replaces most of it", fontsize=10, color=INK,
                  weight="bold")

    for ax in (ax1, ax2):
        ax.grid(alpha=0.13, axis="y")
        ax.tick_params(labelsize=8, colors=INK)
        for s in ax.spines.values():
            s.set_color(INK)
            s.set_linewidth(0.7)

    fig.text(0.5, -0.06,
             "made-up ratings, so a truly best list exists to score against. "
             "not a reconstruction of any real system.",
             ha="center", fontsize=8.5, style="italic", color=INK)
    fig.tight_layout()
    fig.savefig(HERE / "rmse_vs_ranking.svg", transparent=True,
                bbox_inches="tight")
    plt.close(fig)
    print(f"wrote rmse_vs_ranking.svg   gain {gain:.1f}%   "
          f"hit {hit_w[1]:.3f}->{hit_b[1]:.3f}   churn {churn[1]:.3f}")


if __name__ == "__main__":
    rmse_versus_ranking()
