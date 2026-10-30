"""39.1 폴리시 경사 기초 쪽들의 그림.

규칙은 CLAUDE.md 의 Figures 항에 적어 둔 대로다.
- SVG, fonttype='path', transparent
- 글자는 모두 아스키 (matplotlib 기본 폰트에 한글이 없다)

자료를 읽지 않고 작은 밴딧을 **정확히** 셈하므로 어디서 돌려도 된다.
기댓값을 표본으로 어림하지 않고 닫힌 식으로 더하기 때문에 표본 잡음이 없다.

실행: python docs/ch38/foundations/figures/make_figures.py
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


# === 작은 밴딧 — 기댓값을 닫힌 식으로 =========================================

def _softmax(t):
    e = np.exp(t - t.max())
    return e / e.sum()


def _score(p, i, K):
    """d log pi(a) / d theta_i = 1[a=i] - p_i,  액션 a 마다."""
    return (np.arange(K) == i).astype(float) - p[i]


def _var(p, s, r, b):
    """베이스라인 b 를 쓸 때 그 파라메터 추정량의 분산 (정확한 값)."""
    x = (r - b) * s
    return float((p * x ** 2).sum() - ((p * x).sum()) ** 2)


def _b_star(p, s, r):
    """분산을 가장 작게 하는 베이스라인.

    **값함수가 아니다.** 리턴을 점수의 제곱으로 웨이트 주어 평균 낸 것이다 —
    E[g^2 r] / E[g^2]. 파라메터마다 다른 값이 나온다.
    """
    w = p * s ** 2
    return float((w * r).sum() / w.sum())


# === 그림으로 보는 AI: 평균을 빼는 것이 최선이 아니다 ==========================
#
# 윌리엄스가 1992년 논문 8.3절에 적어 둔 대로, 분산을 최소로 만드는 베이스라인은
# "평균 리워드"가 아니라 다른 양이다. 그것을 정확히 셈해 견주고, 무작위로 뽑은
# 배치 여럿에서 얼마나 자주·얼마나 벌어지는지를 **세어서 적는다.**
# 21.6% 같은 수는 뽑은 배치에 달려 있으므로 단언하지 않고 보고한다.

def baseline_not_the_mean():
    # --- 보기 하나: 파라메터마다 b* 가 다르고 평균과도 다르다 ----------------
    theta = np.array([0.3, -0.7, 0.2])
    r = np.array([1.0, 0.4, 2.0])
    K = len(r)
    p = _softmax(theta)
    mean_r = float((p * r).sum())

    rows = []
    for i in range(K):
        s = _score(p, i, K)
        bs = _b_star(p, s, r)
        rows.append((bs, _var(p, s, r, bs), _var(p, s, r, mean_r),
                     _var(p, s, r, 0.0)))
        # b* 는 정의상 최소다 — 언제나 참이므로 단언한다
        assert rows[-1][1] <= rows[-1][2] + 1e-12
        assert rows[-1][1] <= rows[-1][3] + 1e-12

    # --- 무작위 배치를 훑어 얼마나 흔한 일인지 센다 -------------------------
    rng = np.random.default_rng(0)          # 씨앗 고정 — 다시 돌려도 같은 수
    n_cfg, worse, tot, ratios = 2000, 0, 0, []
    for _ in range(n_cfg):
        k = int(rng.integers(2, 6))
        th = rng.normal(0.0, 1.0, k)
        rr = rng.normal(1.0, 1.0, k)
        pp = _softmax(th)
        er = float((pp * rr).sum())
        for i in range(k):
            ss = _score(pp, i, k)
            v_mean = _var(pp, ss, rr, er)
            v_zero = _var(pp, ss, rr, 0.0)
            v_star = _var(pp, ss, rr, _b_star(pp, ss, rr))
            tot += 1
            if v_mean > v_zero:
                worse += 1
            if v_star > 0:
                ratios.append(v_mean / v_star)
    frac = worse / tot
    ratios = np.array(ratios)
    med = float(np.median(ratios))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.1),
                                   gridspec_kw={"width_ratios": [1, 1.15]})

    # --- 왼쪽: 셋이 다 다르다 ----------------------------------------------
    ax1.axvline(mean_r, color=INK, linestyle="--", linewidth=1.4)
    ax1.text(mean_r, 2.72, " mean return", fontsize=8.6, color=INK,
             va="center")
    for i, (bs, v_s, v_m, v_0) in enumerate(rows):
        y = 2.0 - i * 0.8
        ax1.plot([bs], [y], marker="o", color=ACCENT, markersize=11,
                 markeredgecolor="white", markeredgewidth=1.3, zorder=4)
        # 평균선과 라벨이 겹치지 않게 가까우면 반대쪽으로 민다
        off = -0.035 if bs > mean_r - 0.12 and bs < mean_r else 0.0
        ha = "right" if off else "center"
        ax1.text(bs + off, y + 0.26, f"b* = {bs:.3f}", fontsize=8.6,
                 color=ACCENT, ha=ha, weight="bold")
        ax1.text(bs, y - 0.30, f"parameter {i}", fontsize=8.0, color=INK,
                 ha="center", alpha=0.85)
    ax1.set_xlabel("baseline value", fontsize=9.5)
    ax1.set_yticks([])
    ax1.set_ylim(-0.6, 3.0)
    ax1.set_title("one setting, three parameters", fontsize=10, color=INK,
                  weight="bold")
    ax1.grid(alpha=0.13, axis="x")
    ax1.tick_params(labelsize=8.5, colors=INK)
    for sp in ax1.spines.values():
        sp.set_color(INK)
        sp.set_linewidth(0.7)
    ax1.text(0.5, -0.30,
             "the variance-minimising baseline is a score-squared\n"
             "weighted average of returns, not the value function,\n"
             "and it differs from parameter to parameter.",
             transform=ax1.transAxes, ha="center", va="top",
             fontsize=8.0, color=INK, alpha=0.9)

    # --- 오른쪽: 얼마나 벌어지는가 (센 값을 적는다) -------------------------
    CAP = 1e4                      # 꼬리가 길어 자른다 — 자른 개수를 적는다
    over = int((ratios > CAP).sum())
    lg = np.log10(ratios[ratios <= CAP])
    ax2.hist(lg, bins=48, color=ACCENT, alpha=0.55, edgecolor="none")
    ax2.text(0.97, 0.56, f"{over:,} beyond 10^4, not drawn",
             transform=ax2.transAxes, ha="right", fontsize=7.8,
             color=INK, alpha=0.8, style="italic")
    ax2.axvline(np.log10(med), color=INK, linestyle="--", linewidth=1.5)
    ax2.annotate(f"median {med:.1f}x", xy=(np.log10(med), 0),
                 xytext=(16, 128), textcoords="offset points",
                 fontsize=9, color=INK,
                 arrowprops=dict(arrowstyle="->", color=INK, lw=0.9))
    ax2.set_xlabel("variance with the mean baseline, over variance with b*"
                   "   (log10)", fontsize=9)
    ax2.set_ylabel("how many parameters", fontsize=9)
    ax2.set_title(f"{n_cfg:,} random settings, {tot:,} parameters",
                  fontsize=10, color=INK, weight="bold")
    ax2.text(0.97, 0.94,
             f"the mean baseline is WORSE\nthan no baseline at all in\n"
             f"{worse:,} of {tot:,}  ({frac * 100:.1f}%)",
             transform=ax2.transAxes, ha="right", va="top", fontsize=8.8,
             color=ACCENT,
             bbox=dict(boxstyle="round,pad=0.45", facecolor="white",
                       edgecolor=ACCENT, alpha=0.93))
    ax2.grid(alpha=0.13)
    ax2.tick_params(labelsize=8.5, colors=INK)
    for sp in ax2.spines.values():
        sp.set_color(INK)
        sp.set_linewidth(0.7)

    fig.text(0.5, -0.04,
             "counted over settings drawn here, not a theorem - resample and "
             "the two numbers move.",
             ha="center", fontsize=8.7, style="italic", color=INK)
    fig.tight_layout()
    fig.savefig(HERE / "baseline_not_the_mean.svg", transparent=True,
                bbox_inches="tight")
    plt.close(fig)
    print(f"wrote baseline_not_the_mean.svg   b* {[round(r[0], 3) for r in rows]}"
          f"  mean {mean_r:.3f}   worse {worse}/{tot} ({frac*100:.1f}%)"
          f"   median {med:.2f}x")


if __name__ == "__main__":
    baseline_not_the_mean()
