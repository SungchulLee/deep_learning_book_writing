"""35.1 기초 쪽들의 그림.

규칙은 CLAUDE.md 의 Figures 항에 적어 둔 대로다.
- SVG, fonttype='path', transparent
- 글자는 모두 아스키 (matplotlib 기본 폰트에 한글이 없다)

바깥 자료를 읽지 않는다. 낮은 계수 행렬을 **직접 만들어** 쓴다 — MovieLens 가
이 저장소에 없기도 하고, 참값을 알아야 가린 칸의 오차를 잴 수 있기 때문이다.
그림 안에 만든 자료임을 적는다.

⚠️ 관측률 네 자리에서 SGD 를 돌리므로 이 기계에서 3 분 안팎 걸린다.

실행: python docs/ch34/foundations/figures/make_figures.py
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

N_U, N_I, RANK = 300, 200, 5
FRACS = (0.05, 0.10, 0.20, 0.40)


# === 그림으로 보는 AI: 모든 칸을 보느냐 본 칸만 보느냐 ========================
#
# 쪽의 표 첫 줄 — 잘라 낸 SVD 는 **모든** 칸을, 배운 MF 는 **본** 칸만 맞춘다 —
# 이 말주변이 아니라 되느냐 안 되느냐를 가른다는 것을 잰다. 그리고 관측이 너무
# 성기면 MF 가 평균 찍기보다도 못해지는 자리가 있다. 수는 전부 여기서 셈한다.

def _one(frac, seed=0, epochs=600, lr=0.01, lam=0.05):
    rng = np.random.default_rng(seed)
    truth = rng.normal(0, 1, (N_U, RANK)) @ rng.normal(0, 1, (N_I, RANK)).T
    mask = rng.random((N_U, N_I)) < frac
    obs = np.argwhere(mask)
    hid = np.argwhere(~mask)

    def rmse(x):
        return float(np.sqrt(((x[hid[:, 0], hid[:, 1]]
                               - truth[hid[:, 0], hid[:, 1]]) ** 2).mean()))

    mu = truth[mask].mean()
    u, s, vt = np.linalg.svd(np.where(mask, truth, mu), full_matrices=False)
    svd = rmse((u[:, :RANK] * s[:RANK]) @ vt[:RANK])

    p = rng.normal(0, 0.1, (N_U, RANK))
    q = rng.normal(0, 0.1, (N_I, RANK))
    for _ in range(epochs):
        rng.shuffle(obs)
        for a, b in obs:
            err = truth[a, b] - p[a] @ q[b]
            pa = p[a].copy()
            p[a] += lr * (err * q[b] - lam * p[a])
            q[b] += lr * (err * pa - lam * q[b])
    return svd, rmse(p @ q.T), rmse(np.full_like(truth, mu)), int(mask.sum())


def filled_versus_observed():
    svd, mf, base, counts = [], [], [], []
    for f in FRACS:
        a, b, c, n = _one(f)
        svd.append(a)
        mf.append(b)
        base.append(c)
        counts.append(n)

    # 그림이 주장하는 두 가지를 먼저 확인한다
    assert mf[-1] < svd[-1] / 5, "빽빽할 때 MF 가 크게 이기지 않으면 이야기가 없다"
    assert mf[0] > base[0], "가장 성길 때 MF 가 평균보다 낫다면 절벽 이야기가 없다"

    x = np.arange(len(FRACS))
    fig, ax = plt.subplots(figsize=(8.6, 4.4))

    ax.semilogy(x, base, "s:", color=INK, alpha=0.55, linewidth=1.5, ms=6,
                label="just predict the mean")
    ax.semilogy(x, svd, "^--", color=MUTED, linewidth=1.9, ms=7,
                label="fill the blanks, then SVD  (all cells)")
    ax.semilogy(x, mf, "o-", color=ACCENT, linewidth=2.2, ms=7,
                label="factorise the observed cells only")

    for i, (m, b) in enumerate(zip(mf, base)):
        ax.annotate(f"{m:.3f}", (i, m), textcoords="offset points",
                    xytext=(0, -17), ha="center", fontsize=8.4, color=ACCENT)

    bad = [i for i, (m, b) in enumerate(zip(mf, base)) if m > b]
    for i in bad:
        ax.annotate("worse than guessing", (i, mf[i]),
                    textcoords="offset points", xytext=(14, 10),
                    fontsize=8.4, color=ACCENT,
                    arrowprops=dict(arrowstyle="->", color=ACCENT, lw=0.9))

    lo = min(mf + svd + base) * 0.72          # 아래 라벨이 잘리지 않게
    hi = max(mf + svd + base) * 1.35
    ax.set_ylim(lo, hi)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{f:.0%}\n{n:,} seen" for f, n in zip(FRACS, counts)],
                       fontsize=8.6)
    ax.set_xlabel("share of the matrix that is observed", fontsize=9.5)
    ax.set_ylabel("RMSE on the hidden cells  (log)", fontsize=9.5)
    ax.set_title("the first row of the table is not a technicality",
                 fontsize=10.5, color=INK, weight="bold", pad=8)
    ax.legend(fontsize=8.6, framealpha=0.92, loc="lower left")
    ax.grid(alpha=0.14, which="both")
    ax.tick_params(labelsize=8, colors=INK)
    for s in ax.spines.values():
        s.set_color(INK)
        s.set_linewidth(0.7)

    fig.text(0.5, -0.06,
             f"made-up rank-{RANK} matrix, {N_U} x {N_I}, so the hidden cells "
             f"have a known answer to score against.",
             ha="center", fontsize=8.6, style="italic", color=INK)
    fig.tight_layout()
    fig.savefig(HERE / "filled_vs_observed.svg", transparent=True,
                bbox_inches="tight")
    plt.close(fig)
    print("wrote filled_vs_observed.svg   "
          + "  ".join(f"{f:.0%}: svd {a:.3f} mf {b:.3f} base {c:.3f}"
                      for f, a, b, c in zip(FRACS, svd, mf, base)))


if __name__ == "__main__":
    filled_versus_observed()
