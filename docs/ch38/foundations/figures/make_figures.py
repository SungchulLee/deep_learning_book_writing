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


# === 그림으로 보는 AI: 정리 1 이 약속하는 것 ==================================
#
# 윌리엄스의 정리 1 은 "그래디언트와 같다"가 아니라 "그래디언트와의 **내적이
# 음이 아니다**"이다. 같아지는 것은 alpha 가 모든 i 에 대해 같을 때뿐이다.
# 두 벡터는 언제나 한 평면에 놓이므로, 그래디언트를 x 축에 두고 고침 벡터를
# 실제 각도로 그리면 **투영 손실 없이 정확히** 그릴 수 있다.

def theorem_one_promises():
    theta = np.array([0.3, -0.7, 0.2])
    r = np.array([1.0, 0.4, 2.0])
    p = _softmax(theta)
    g = p * (r - float((p * r).sum()))          # 정확한 그래디언트

    cases = [("same alpha for every i", np.array([0.10, 0.10, 0.10]), MUTED),
             ("alpha differs by i", np.array([0.30, 0.02, 0.10]), ACCENT)]
    drawn = []
    for label, alpha, col in cases:
        u = alpha * g                            # E{dW} = diag(alpha) g
        ip = float(u @ g)
        cos = ip / (np.linalg.norm(u) * np.linalg.norm(g))
        ang = float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))
        drawn.append((label, ang, ip, col))
        # 정리 1 — 내적은 음이 아니다. 언제나 참이므로 단언한다.
        assert ip >= -1e-15, "정리 1 이 깨지면 이 그림의 이야기가 없다"
    assert drawn[0][1] < 1e-6, "같은 alpha 면 나란해야 한다"
    assert drawn[1][1] > 1.0, "다른 alpha 면 어긋나야 한다"

    fig, ax = plt.subplots(figsize=(8.6, 4.6))

    # 내적이 양수인 반평면
    th = np.linspace(-90, 90, 200)
    ax.fill(np.r_[0, np.cos(np.radians(th))] * 1.28,
            np.r_[0, np.sin(np.radians(th))] * 1.28,
            color=INK, alpha=0.055, linewidth=0)
    ax.plot([0, 0], [-1.28, 1.28], color=INK, linewidth=0.8, alpha=0.35,
            linestyle=":")
    ax.text(0.04, 1.20, "inner product > 0 to the right of this line",
            fontsize=8.0, color=INK, alpha=0.75)

    ax.annotate("", xy=(1.0, 0.0), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color=INK, linewidth=2.6))
    ax.text(1.03, 0.0, "the true gradient", fontsize=9.4, color=INK,
            va="center", weight="bold")

    for label, ang, ip, col in drawn:
        rad = np.radians(ang)
        L = 0.80
        ax.annotate("", xy=(L * np.cos(rad), L * np.sin(rad)), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color=col,
                                    linewidth=2.2))
        ax.text(L * np.cos(rad) * 1.06 + 0.02, L * np.sin(rad) * 1.06 + 0.03,
                f"{label}\n{ang:.3f} deg off", fontsize=8.6, color=col,
                va="bottom", weight="bold")

    ax.set_xlim(-0.30, 1.72)
    ax.set_ylim(-0.32, 1.34)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("what Theorem 1 actually guarantees", fontsize=10.5,
                 color=INK, weight="bold", pad=6)
    fig.text(0.5, 0.015,
             "both updates climb. only the uniform one is the gradient.",
             ha="center", fontsize=9.0, style="italic", color=INK)
    fig.tight_layout()
    fig.savefig(HERE / "theorem_one.svg", transparent=True,
                bbox_inches="tight")
    plt.close(fig)
    print(f"wrote theorem_one.svg   "
          + "   ".join(f"{l}: {a:.3f} deg" for l, a, _, _ in drawn))


# === 그림으로 보는 AI: 더 잘 맞히는 비평자가 더 나쁜 방향을 준다 ==============
#
# 서튼 외 1999 의 정리 2 는 비평자가 **호환 조건**을 지킬 때만 등호를 준다.
# 호환은 "잘 맞힌다"가 아니라 "어느 함수 공간에 사는가"의 조건이다. 그래서
# Q 를 훨씬 잘 맞히는 비평자가 그래디언트는 크게 어긋나게 만들 수 있다.
# 작은 밴딧에서 둘을 정확히 셈해 견준다.

def compatible_not_accurate():
    theta = np.array([0.3, -0.7, 0.2])
    r = np.array([1.0, 0.4, 2.0])
    K = len(r)
    p = _softmax(theta)
    # S[a, i] = d log pi(a) / d theta_i  — 호환 특징
    S = np.stack([_score(p, i, K) for i in range(K)], axis=1)
    g = S.T @ (p * r)                               # 정확한 그래디언트

    rootW = np.diag(np.sqrt(p))

    def fit(basis):
        w = np.linalg.lstsq(rootW @ basis, rootW @ r, rcond=None)[0]
        f = basis @ w
        return f, float((p * (r - f) ** 2).sum()), S.T @ (p * f)

    f_c, err_c, g_c = fit(S)                        # 호환 — 점수 특징으로 선형
    basis_i = np.stack([np.ones(K), np.arange(K, dtype=float)], axis=1)
    f_i, err_i, g_i = fit(basis_i)                  # 비호환 — 다른 기저

    def ang(u, v):
        c = float(u @ v) / (np.linalg.norm(u) * np.linalg.norm(v))
        return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))

    a_c, a_i = ang(g_c, g), ang(g_i, g)
    # 정리 2 — 호환이면 등호다. 언제나 참이므로 단언한다.
    assert a_c < 1e-6, "호환 비평자가 그래디언트를 정확히 주어야 한다"
    # 이 그림의 요지 — 더 잘 맞히는 쪽이 더 어긋난다
    assert err_i < err_c and a_i > a_c, "요지가 성립하지 않으면 그리지 않는다"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.2),
                                   gridspec_kw={"width_ratios": [1.1, 1]})

    # --- 왼쪽: 비평자가 리턴을 얼마나 잘 맞히는가 --------------------------
    x = np.arange(K)
    ax1.plot(x, r, "o-", color=INK, linewidth=2.2, ms=9, label="true return")
    ax1.plot(x, f_i, "s--", color=MUTED, linewidth=1.9, ms=7,
             label=f"incompatible critic   err {err_i:.3f}")
    ax1.plot(x, f_c, "^--", color=ACCENT, linewidth=1.9, ms=7,
             label=f"compatible critic   err {err_c:.3f}")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"action {i}" for i in x], fontsize=8.6)
    ax1.set_ylabel("value", fontsize=9.5)
    ax1.set_title("how well each critic fits", fontsize=10, color=INK,
                  weight="bold")
    ax1.legend(fontsize=8.2, framealpha=0.92)
    ax1.grid(alpha=0.13)
    ax1.tick_params(labelsize=8.5, colors=INK)
    for sp in ax1.spines.values():
        sp.set_color(INK)
        sp.set_linewidth(0.7)

    # --- 오른쪽: 그래서 어느 쪽을 가리키는가 -------------------------------
    ax2.annotate("", xy=(1.0, 0.0), xytext=(0, 0),
                 arrowprops=dict(arrowstyle="-|>", color=INK, linewidth=2.6))
    ax2.text(1.03, 0.0, "true gradient", fontsize=9.2, color=INK,
             va="center", weight="bold")
    for a, col, lab in ((a_c, ACCENT, "compatible"),
                        (a_i, MUTED, "incompatible")):
        rad = np.radians(a)
        ax2.annotate("", xy=(0.78 * np.cos(rad), 0.78 * np.sin(rad)),
                     xytext=(0, 0),
                     arrowprops=dict(arrowstyle="-|>", color=col,
                                     linewidth=2.2))
        ax2.text(0.80 * np.cos(rad) + 0.03, 0.80 * np.sin(rad) + 0.04,
                 f"{lab}\n{a:.3f} deg off", fontsize=8.7, color=col,
                 va="bottom", weight="bold")
    ax2.set_xlim(-0.22, 1.62)
    ax2.set_ylim(-0.26, 0.92)
    ax2.set_aspect("equal")
    ax2.axis("off")
    ax2.set_title("where each one points", fontsize=10, color=INK,
                  weight="bold")

    fig.text(0.5, -0.03,
             f"the critic that fits {err_c / err_i:.0f}x better is the one "
             f"that misses the gradient.",
             ha="center", fontsize=8.9, style="italic", color=INK)
    fig.tight_layout()
    fig.savefig(HERE / "compatible_not_accurate.svg", transparent=True,
                bbox_inches="tight")
    plt.close(fig)
    print(f"wrote compatible_not_accurate.svg   compat {a_c:.3f}deg err "
          f"{err_c:.4f}   incompat {a_i:.3f}deg err {err_i:.4f}")


# === 그림으로 보는 AI: 가우스 폴리시의 sigma 가 스스로 좁아진다 ===============
#
# 윌리엄스 8.2 절이 적은 대로, sigma 는 "그 y 가 다시 나오기 쉽게" 움직인다.
# 평균 가까이에서 좋은 값이 나오면 좁아지고, 멀리서 나오면 넓어진다.
# 그리고 r 이 늘 양수인데 베이스라인이 없으면 탐색이 너무 일찍 닫힌다.
# 씨앗 스무 개를 돌려 **퍼짐과 함께** 보고한다 — 강화학습 수는 흔들린다.

def gaussian_sigma_closes():
    PEAK, T, LR, SEEDS = 3.0, 30000, 2e-3, 20

    def run(use_baseline, seed):
        rng = np.random.default_rng(seed)
        mu, sig, b = 0.0, 1.0, 0.0
        ever = False
        for _ in range(T):
            y = rng.normal(mu, sig)
            r = float(np.exp(-0.5 * (y - PEAK) ** 2))   # 늘 양수
            adv = r - (b if use_baseline else 0.0)
            if use_baseline:
                b += 0.01 * (r - b)
            # sigma 가 작아지면 1/sigma^2 이 터지므로 스텝을 묶는다
            mu += LR * float(np.clip(adv * (y - mu) / sig ** 2, -10, 10))
            sig += LR * float(np.clip(
                adv * ((y - mu) ** 2 - sig ** 2) / sig ** 3, -10, 10))
            sig = float(np.clip(sig, 1e-3, 10.0))
            if not ever and abs(mu - PEAK) < 0.5:
                ever = True
        return mu, sig, ever

    out = {}
    for lab, ub in (("no baseline", False), ("with a baseline", True)):
        rows = [run(ub, s) for s in range(SEEDS)]
        mus = np.array([m for m, _, _ in rows])
        sgs = np.array([s for _, s, _ in rows])
        ever = int(sum(e for _, _, e in rows))
        stay = int((np.abs(mus - PEAK) < 0.5).sum())
        out[lab] = (mus, sgs, ever, stay)

    (m0, s0, e0, k0) = out["no baseline"]
    (m1, s1, e1, k1) = out["with a baseline"]
    # 이 그림의 요지 — 베이스라인 없이는 머물지 못한다
    assert e0 == SEEDS and e1 == SEEDS, "양쪽 다 한 번은 닿아야 비교가 된다"
    assert k1 > k0, "베이스라인 쪽이 더 많이 머물러야 이 그림이 성립한다"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.2),
                                   gridspec_kw={"width_ratios": [1, 1.2]})

    # --- 왼쪽: sigma 가 움직이는 규칙 (윌리엄스 8.2) -----------------------
    xs = np.linspace(-3.2, 3.2, 300)
    ax1.plot(xs, np.exp(-0.5 * xs ** 2), color=INK, linewidth=1.8)
    ax1.axvspan(-1, 1, color=ACCENT, alpha=0.11)
    ax1.text(0, 0.52, "within one sigma", ha="center", fontsize=8.4,
             color=ACCENT)
    ax1.annotate("good point here\n-> narrow", xy=(0.55, 0.86),
                 xytext=(0.0, 1.14), textcoords="data", ha="center",
                 fontsize=8.5, color=ACCENT,
                 arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.1))
    ax1.annotate("good point here\n-> broaden", xy=(2.25, 0.08),
                 xytext=(2.35, 0.62), textcoords="data", ha="center",
                 fontsize=8.5, color=INK,
                 arrowprops=dict(arrowstyle="->", color=INK, lw=1.1))
    ax1.set_ylim(0, 1.42)
    ax1.set_xlabel("sampled action, in units of sigma from the mean",
                   fontsize=9)
    ax1.set_yticks([])
    ax1.set_title("how sigma moves", fontsize=10, color=INK, weight="bold")
    ax1.tick_params(labelsize=8.5, colors=INK)
    for sp in ax1.spines.values():
        sp.set_visible(False)
    ax1.text(0.5, -0.30, "sigma always changes so the sampled action\n"
                         "becomes more likely to happen again.",
             transform=ax1.transAxes, ha="center", va="top", fontsize=8.3,
             color=INK, alpha=0.9)

    # --- 오른쪽: 씨앗 스무 개가 어디서 멈췄나 -------------------------------
    ax2.axhline(PEAK, color=INK, linestyle="--", linewidth=1.4)
    ax2.text(0.5, PEAK + 0.16, "the peak", fontsize=8.6, color=INK)
    rng = np.random.default_rng(7)
    for k, (lab, mus, col) in enumerate(((f"no baseline\n{k0} of {SEEDS} stay",
                                          m0, ACCENT),
                                         (f"with a baseline\n{k1} of {SEEDS} stay",
                                          m1, MUTED))):
        xj = k + 1 + rng.uniform(-0.13, 0.13, len(mus))
        ax2.plot(xj, mus, "o", color=col, ms=7, alpha=0.75,
                 markeredgecolor="white", markeredgewidth=0.8)
    ax2.set_xticks([1, 2])
    ax2.set_xticklabels([f"no baseline\n{k0} of {SEEDS} stay",
                         f"with a baseline\n{k1} of {SEEDS} stay"],
                        fontsize=8.8)
    ax2.set_ylabel("where the mean ended up", fontsize=9.5)
    ax2.set_xlim(0.4, 2.6)
    ax2.set_title(f"every run reached the peak at least once "
                  f"({e0}/{SEEDS} and {e1}/{SEEDS})",
                  fontsize=9.6, color=INK, weight="bold")
    ax2.grid(alpha=0.13, axis="y")
    ax2.tick_params(labelsize=8.5, colors=INK)
    for sp in ax2.spines.values():
        sp.set_color(INK)
        sp.set_linewidth(0.7)

    fig.text(0.5, -0.05,
             "reward is positive everywhere, so without a baseline every "
             "sample reinforces itself.",
             ha="center", fontsize=8.8, style="italic", color=INK)
    fig.tight_layout()
    fig.savefig(HERE / "gaussian_sigma.svg", transparent=True,
                bbox_inches="tight")
    plt.close(fig)
    print(f"wrote gaussian_sigma.svg   no-baseline stay {k0}/{SEEDS} "
          f"sigma~{np.median(s0):.4f}   baseline stay {k1}/{SEEDS} "
          f"sigma~{np.median(s1):.4f}")


if __name__ == "__main__":
    baseline_not_the_mean()
    theorem_one_promises()
    compatible_not_accurate()
    gaussian_sigma_closes()
