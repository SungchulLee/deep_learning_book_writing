"""18장 장 소개의 그림.

규칙은 CLAUDE.md 의 Figures 항에 적어 둔 대로다.
- SVG, fonttype='path', transparent
- 글자는 모두 아스키 (matplotlib 기본 폰트에 한글이 없다)

그림마다 **제 주장을 스스로 확인하고**, 어긋나면 그리지 않고 멈춘다.

실행: python docs/ch17/figures/make_figures.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["svg.fonttype"] = "path"

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import betaln

HERE = Path(__file__).resolve().parent
INK = "#2f3337"
ACCENT = "#8a4b2a"
MUTED = "#2d6a8f"


# === 그림으로 보는 AI: 사전분포는 잊히지만 증거는 잊지 않는다 =================
#
# 추정(1-3절)에서는 자료가 쌓이면 사전분포가 씻겨 나간다. 모델 비교(4절)에서는
# 씻기지 않는다 — 주변우도에 사전분포의 폭이 그대로 남기 때문이다.
# 같은 자료에 같은 두 사전분포를 두고 두 양을 나란히 잰다.

def prior_washout():
    TRUE = 0.3
    rng = np.random.default_rng(0)
    A, B = (1.0, 1.0), (0.5, 0.5)
    C = (40.0, 4.0)                     # 아주 틀린 사전분포
    ns = np.array([5, 20, 100, 1000, 10000, 100000])

    gap, ev = [], []
    for n in ns:
        x = rng.binomial(n, TRUE)
        mean = lambda P: (P[0] + x) / (P[0] + P[1] + n)
        lml = lambda P: betaln(P[0] + x, P[1] + n - x) - betaln(*P)
        gap.append(max(mean(A), mean(B), mean(C))
                   - min(mean(A), mean(B), mean(C)))
        ev.append(float(np.exp(lml(A) - lml(B))))
    gap, ev = np.array(gap), np.array(ev)

    # 그림이 제 주장을 확인한다
    assert gap[-1] < gap[0] / 100, "사후평균 격차는 씻겨 나가야 한다"
    assert gap[-1] < 1e-3, "끝에서는 사실상 0 이어야 한다"
    assert abs(ev[-1] - 1) > 0.2, "증거 비는 1 로 가지 않는다"
    assert abs(ev[-1] - ev[0]) < 0.5, "증거 비는 대체로 제자리다"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.6, 4.1))

    ax1.loglog(ns, np.maximum(gap, 1e-7), "o-", color=ACCENT, linewidth=2.1,
               ms=6)
    ax1.set_xlabel("sample size", fontsize=9.5)
    ax1.set_ylabel("spread of posterior means", fontsize=9.5)
    ax1.set_title("estimation forgets the prior", fontsize=10, color=INK,
                  weight="bold")
    ax1.annotate(f"{gap[0]:.3f}", xy=(ns[0], gap[0]), xytext=(8, 4),
                 textcoords="offset points", fontsize=8.6, color=INK)
    ax1.annotate(f"{gap[-1]:.6f}", xy=(ns[-1], gap[-1]), xytext=(-62, 6),
                 textcoords="offset points", fontsize=8.6, color=INK)

    ax2.semilogx(ns, ev, "o-", color=MUTED, linewidth=2.1, ms=6)
    ax2.axhline(1.0, color=INK, linestyle=":", linewidth=1.2)
    ax2.text(ns[0], 1.0, " equal evidence", fontsize=8.2, color=INK,
             va="bottom")
    ax2.set_xlabel("sample size", fontsize=9.5)
    ax2.set_ylabel("evidence ratio between two priors", fontsize=9.5)
    ax2.set_title("model comparison does not", fontsize=10, color=INK,
                  weight="bold")
    ax2.set_ylim(0.9, 1.7)
    ax2.annotate(f"still {ev[-1]:.3f}", xy=(ns[-1], ev[-1]), xytext=(-70, 8),
                 textcoords="offset points", fontsize=8.6, color=MUTED)

    for ax in (ax1, ax2):
        ax.grid(alpha=0.12, which="both")
        ax.tick_params(labelsize=8, colors=INK)
        for s in ax.spines.values():
            s.set_color(INK)
            s.set_linewidth(0.7)

    fig.text(0.5, -0.04,
             "the same data and the same priors, measured two ways. "
             "one quantity washes the prior out; the other keeps it.",
             ha="center", fontsize=8.8, style="italic", color=INK)
    fig.tight_layout()
    fig.savefig(HERE / "prior_washout.svg", transparent=True,
                bbox_inches="tight")
    plt.close(fig)
    print(f"wrote prior_washout.svg   gap {gap[0]:.4f}->{gap[-1]:.6f}   "
          f"evidence {ev[0]:.3f}->{ev[-1]:.3f}")


if __name__ == "__main__":
    prior_washout()
