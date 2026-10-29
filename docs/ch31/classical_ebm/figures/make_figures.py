"""32.2 고전 이진 모델 쪽들의 그림.

규칙은 CLAUDE.md 의 Figures 항에 적어 둔 대로다.
- SVG, fonttype='path', transparent
- 글자는 모두 아스키 (matplotlib 기본 폰트에 한글이 없다)

자료를 읽지 않으므로 어디서 돌려도 된다. 수는 전부 여기서 실제로 돌려 얻는다.

실행: python docs/ch31/classical_ebm/figures/make_figures.py
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

ALPHA_C = 0.138          # Amit-Gutfreund-Sompolinsky 1985
ALPHAS = (0.05, 0.08, 0.10, 0.12, 0.138, 0.16, 0.20, 0.25)


# === 그림으로 보는 AI: 두 수는 서로 다른 물음에 답한다 ========================
#
# 홉필드 1982 가 적은 수는 0.15N 이고 모의실험(N=30, N=100)에서 나왔다.
# 흔히 인용하는 0.138 은 세 해 뒤 AGS 1985 가 복제법으로 얻은 N->무한 문턱이다.
# 두 기준을 **각각** 재어 나란히 놓으면, 문턱이 있는 것도 없는 것도 사실이다.
#   왼쪽  m : 수렴한 상태가 저장 무늬와 얼마나 겹치는가 (AGS 의 순서변수)
#   오른쪽  : 저장 무늬가 **한 비트도 틀리지 않고** 고정점인 비율

def _hopfield_sweep(N, p, trials, rng, sweeps=8, max_patterns=12):
    """m 과 '완전 안정' 비율을 한 번에 잰다."""
    ms, perfect = [], []
    for _ in range(trials):
        X = rng.choice([-1.0, 1.0], size=(p, N))
        W = (X.T @ X) / N
        np.fill_diagonal(W, 0.0)

        # 완전 안정: 저장 무늬의 모든 비트가 국소장의 부호와 같은가
        perfect.append(((np.sign(X @ W.T) == X).all(axis=1)).mean())

        for k in range(min(p, max_patterns)):
            s = X[k].copy()
            for _ in range(sweeps):
                for i in rng.permutation(N):
                    h = W[i] @ s
                    if h != 0:
                        s[i] = np.sign(h)
            ms.append(abs(s @ X[k]) / N)
    return float(np.mean(ms)), float(np.mean(perfect))


def capacity_two_questions():
    rng = np.random.default_rng(0)
    sizes = ((100, 8), (400, 3))
    m = {N: [] for N, _ in sizes}
    perf = {N: [] for N, _ in sizes}
    for a in ALPHAS:
        for N, tr in sizes:
            mm, pp = _hopfield_sweep(N, max(1, int(round(a * N))), tr, rng)
            m[N].append(mm)
            perf[N].append(pp)

    a = np.array(ALPHAS)
    ic = list(ALPHAS).index(ALPHA_C)

    # --- 그림이 주장하는 바를 그림이 스스로 확인한다 ----------------------
    assert m[400][ic] > 0.95, (
        f"문턱 아래에서 m 이 1 에 가까워야 한다: {m[400][ic]:.3f}")
    assert m[400][-1] < 0.65, (
        f"문턱 위에서 m 이 무너져야 한다: {m[400][-1]:.3f}")
    assert perf[400][ic] < 0.6, (
        f"같은 자리에서 완전 복구는 이미 반 아래여야 한다: {perf[400][ic]:.3f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.6, 4.1), sharex=True)

    for ax, data, title, ylab in (
            (ax1, m, "is the recalled state essentially right?",
             "overlap m after settling"),
            (ax2, perf, "is every stored pattern perfect?",
             "fraction stable in all N bits")):
        for (N, _), col, mk in zip(sizes, (MUTED, ACCENT), ("o-", "s-")):
            ax.plot(a, data[N], mk, color=col, ms=5, linewidth=1.8,
                    label=f"N = {N}")
        ax.axvline(ALPHA_C, color=INK, linestyle="--", linewidth=1.2,
                   alpha=0.7)
        ax.set_xlabel("alpha  =  patterns / N", fontsize=9)
        ax.set_ylabel(ylab, fontsize=9)
        ax.set_title(title, fontsize=10, color=INK, weight="bold")
        ax.set_ylim(0, 1.06)
        ax.grid(alpha=0.13)
        ax.legend(fontsize=8.5, loc="lower left", framealpha=0.9)
        ax.tick_params(labelsize=8, colors=INK)
        for s in ax.spines.values():
            s.set_color(INK)
            s.set_linewidth(0.7)

    ax1.annotate("0.138\n(AGS 1985)", xy=(ALPHA_C, 0.42),
                 xytext=(ALPHA_C + 0.028, 0.30), fontsize=8.2, color=INK,
                 arrowprops=dict(arrowstyle="->", color=INK, lw=0.9))
    ax2.annotate(f"only {perf[400][ic]:.2f}\nat the threshold",
                 xy=(ALPHA_C, perf[400][ic]),
                 xytext=(ALPHA_C + 0.045, perf[400][ic] + 0.30), fontsize=8.2,
                 color=ACCENT, ha="left",
                 arrowprops=dict(arrowstyle="->", color=ACCENT, lw=0.9))

    fig.text(0.5, -0.03,
             "same networks, same runs, two criteria. the threshold governs "
             "the left; on the right the decay starts well before it.",
             ha="center", fontsize=8.8, style="italic", color=INK)
    fig.tight_layout()
    fig.savefig(HERE / "capacity_two_questions.svg", transparent=True,
                bbox_inches="tight")
    plt.close(fig)
    print(f"wrote capacity_two_questions.svg   "
          f"m(400)@0.138 = {m[400][ic]:.3f}   "
          f"perfect(400)@0.138 = {perf[400][ic]:.3f}")


if __name__ == "__main__":
    capacity_two_questions()
