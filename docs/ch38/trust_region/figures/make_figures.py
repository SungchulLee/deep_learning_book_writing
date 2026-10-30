"""39.3 신뢰 영역 쪽들의 그림.

규칙은 CLAUDE.md 의 Figures 항에 적어 둔 대로다.
- SVG, fonttype='path', transparent
- 글자는 모두 아스키 (matplotlib 기본 폰트에 한글이 없다)

자료를 읽지 않으므로 어디서 돌려도 된다. 무작위를 쓰는 곳은 모두 씨앗을 박는다.

실행: python docs/ch38/trust_region/figures/make_figures.py
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


def _softmax(Z):
    E = np.exp(Z - Z.max(axis=1, keepdims=True))
    return E / E.sum(axis=1, keepdims=True)


def _kl_rows(P, Q):
    return (P * (np.log(P + 1e-30) - np.log(Q + 1e-30))).sum(axis=1)


# === 그림으로 보는 AI: 정리와 알고리즘 사이에 난 두 자리 ======================
#
# TRPO 논문 스스로 두 군데를 바꿔 썼다고 적는다 — (1) 이론이 권하는 벌점 대신
# 하드 제약, (2) 모든 상태에서의 최대 KL 대신 평균 KL. 둘째가 얼마나 느슨한지를
# 상태 수를 늘려 가며 **직접 잰다.** 씨앗을 박아 다시 돌려도 같은 수가 나온다.

def theory_to_algorithm():
    rng = np.random.default_rng(0)
    sizes = [10, 100, 1000, 10000]
    ratios = []
    for S in sizes:
        Z = rng.normal(0.0, 1.0, (S, 6))
        D = rng.normal(0.0, 0.30, (S, 6))
        k = _kl_rows(_softmax(Z), _softmax(Z + D))
        ratios.append(float(k.max() / k.mean()))
    # 최대는 평균보다 작을 수 없다 — 언제나 참이므로 단언한다
    assert all(r >= 1.0 for r in ratios)
    assert ratios[-1] > ratios[0], "상태가 늘수록 벌어져야 이 그림이 성립한다"

    # 평균 KL 을 0.01 에 맞췄을 때의 최대/평균, 씨앗 열 개
    pinned = []
    for s in range(10):
        r2 = np.random.default_rng(s)
        Z = r2.normal(0.0, 1.0, (1000, 6))
        D = r2.normal(0.0, 1.0, (1000, 6))
        lo, hi = 0.0, 5.0
        for _ in range(60):
            m = (lo + hi) / 2
            if _kl_rows(_softmax(Z), _softmax(Z + m * D)).mean() < 0.01:
                lo = m
            else:
                hi = m
        k = _kl_rows(_softmax(Z), _softmax(Z + lo * D))
        pinned.append(float(k.max() / k.mean()))
    pinned = np.array(pinned)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.2, 4.1),
                                   gridspec_kw={"width_ratios": [1.05, 1]})

    # --- 왼쪽: 논문이 스스로 적은 두 번의 바꿔 쓰기 -------------------------
    swaps = [("a penalty on KL", "a hard constraint with delta",
              "the coefficient the theory\nrecommends makes the steps\ntoo small"),
             ("the KL at every state", "the average KL",
              "one constraint per state is\nnot solvable, so this is called\n"
              "a heuristic approximation")]
    for k, (a, b, why) in enumerate(swaps):
        y = 0.74 - k * 0.46
        ax1.text(0.03, y + 0.10, a, fontsize=9.6, color=INK, va="center")
        ax1.annotate("", xy=(0.60, y + 0.10), xytext=(0.44, y + 0.10),
                     arrowprops=dict(arrowstyle="-|>", color=ACCENT, lw=1.8))
        ax1.text(0.63, y + 0.10, b, fontsize=9.6, color=ACCENT,
                 va="center", weight="bold")
        ax1.text(0.03, y - 0.10, why, fontsize=7.9, color=INK, va="top",
                 alpha=0.85)
    ax1.text(0.5, 0.96, "what the theorem asks for  ->  what the algorithm does",
             ha="center", fontsize=9.0, color=INK, style="italic")
    ax1.set_xlim(0, 1.32)
    ax1.set_ylim(0.02, 1.04)
    ax1.axis("off")
    ax1.set_title("the paper says both of these itself", fontsize=10,
                  color=INK, weight="bold")

    # --- 오른쪽: 평균으로 바꾸면 얼마나 느슨해지나 -------------------------
    ax2.semilogx(sizes, ratios, "o-", color=ACCENT, linewidth=2.1, ms=7)
    for S, r in zip(sizes, ratios):
        ax2.annotate(f"{r:.1f}x", xy=(S, r), xytext=(0, 10),
                     textcoords="offset points", ha="center", fontsize=8.6,
                     color=INK)
    ax2.axhline(1.0, color=INK, linestyle=":", linewidth=1.1, alpha=0.6)
    ax2.text(sizes[0], 1.08, "max = mean", fontsize=8.0, color=INK,
             alpha=0.75)
    ax2.set_xlabel("number of states", fontsize=9.5)
    ax2.set_ylabel("largest KL, over the average KL", fontsize=9.5)
    ax2.set_title("the substitution loosens as states multiply",
                  fontsize=10, color=INK, weight="bold")
    ax2.set_ylim(0, max(ratios) * 1.30)
    ax2.grid(alpha=0.13, which="both")
    ax2.tick_params(labelsize=8.5, colors=INK)
    for sp in ax2.spines.values():
        sp.set_color(INK)
        sp.set_linewidth(0.7)

    fig.text(0.5, -0.04,
             f"pin the average KL at 0.01 over 1,000 states and the largest "
             f"is a median {np.median(pinned):.1f}x bigger "
             f"({pinned.min():.1f}-{pinned.max():.1f} over ten seeds).",
             ha="center", fontsize=8.7, style="italic", color=INK)
    fig.tight_layout()
    fig.savefig(HERE / "theory_to_algorithm.svg", transparent=True,
                bbox_inches="tight")
    plt.close(fig)
    print(f"wrote theory_to_algorithm.svg   ratios {[round(r,1) for r in ratios]}"
          f"   pinned median {np.median(pinned):.1f}x")


# === 그림으로 보는 AI: 자르기는 비를 묶어 두지 않는다 =========================
#
# 자르기는 그 액션을 **더 밀 이유**를 없앨 뿐, 이미 넘어간 비를 끌어오지 않는다.
# 같은 배치로 에폭을 거듭하면 비가 eps 를 지나 계속 간다. 씨앗 여덟 개로 재어
# 중앙값을 적는다.

def clipping_does_not_bound():
    def _sm(z):
        e = np.exp(z - z.max())
        return e / e.sum()

    def _kl(p, q):
        return float((p * (np.log(p + 1e-30) - np.log(q + 1e-30))).sum())

    EPS, K, LR = 0.2, 6, 0.05

    def run(epochs, seed):
        rng = np.random.default_rng(seed)
        z0 = rng.normal(0, 1, K)
        p0 = _sm(z0)
        A = rng.normal(0, 1, K)
        z = z0.copy()
        for _ in range(epochs):
            p = _sm(z)
            r = p / p0
            active = (r * A) <= (np.clip(r, 1 - EPS, 1 + EPS) * A)
            g = np.zeros(K)
            for i in range(K):
                if active[i]:
                    dr = -p[i] * p
                    dr[i] = p[i] * (1 - p[i])
                    g += A[i] / p0[i] * dr
            z += LR * g
        p = _sm(z)
        return _kl(p0, p), float(np.abs(p / p0 - 1).max())

    eps_list = [1, 2, 5, 10, 20, 50, 100, 200]
    kls, rats = [], []
    for ep in eps_list:
        out = [run(ep, s) for s in range(8)]
        kls.append(float(np.median([k for k, _ in out])))
        rats.append(float(np.median([r for _, r in out])))
    # 이 그림의 요지 — 비가 eps 를 넘어간다
    assert rats[-1] > EPS, "비가 eps 안에 머물면 이 그림의 이야기가 없다"
    assert rats[0] < EPS, "처음에는 안에 있어야 대비가 된다"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.1),
                                   gridspec_kw={"width_ratios": [1, 1.15]})

    # --- 왼쪽: 자른 목표의 생김새 ------------------------------------------
    r = np.linspace(0.4, 1.8, 400)
    for A, col, lab, sgn in ((1.0, ACCENT, "advantage > 0", 1),
                             (-1.0, MUTED, "advantage < 0", -1)):
        obj = np.minimum(r * A, np.clip(r, 1 - EPS, 1 + EPS) * A)
        ax1.plot(r, obj * sgn, color=col, linewidth=2.1, label=lab)
    for v in (1 - EPS, 1 + EPS):
        ax1.axvline(v, color=INK, linestyle=":", linewidth=1.1, alpha=0.65)
    ax1.text(1 + EPS, ax1.get_ylim()[1] * 0.96, "  1 + eps", fontsize=8.2,
             color=INK, va="top")
    ax1.text(1 - EPS, ax1.get_ylim()[1] * 0.96, "1 - eps  ", fontsize=8.2,
             color=INK, va="top", ha="right")
    ax1.set_xlabel("probability ratio r", fontsize=9.5)
    ax1.set_ylabel("objective (sign-aligned)", fontsize=9.5)
    ax1.set_title("flat means no gradient, not no motion", fontsize=10,
                  color=INK, weight="bold")
    ax1.legend(fontsize=8.3, framealpha=0.92)
    ax1.grid(alpha=0.13)
    ax1.tick_params(labelsize=8.5, colors=INK)
    for sp in ax1.spines.values():
        sp.set_color(INK)
        sp.set_linewidth(0.7)

    # --- 오른쪽: 에폭을 거듭하면 ------------------------------------------
    ax2.semilogx(eps_list, rats, "o-", color=ACCENT, linewidth=2.1, ms=6,
                 label="largest |r - 1|")
    ax2.axhline(EPS, color=ACCENT, linestyle="--", linewidth=1.4)
    ax2.text(eps_list[0], EPS + 0.012, " the clip, eps = 0.2", fontsize=8.4,
             color=ACCENT)
    ax2b = ax2.twinx()
    ax2b.semilogx(eps_list, kls, "s--", color=INK, linewidth=1.8, ms=5,
                  label="KL(old || new)")
    ax2b.set_ylabel("KL(old || new)", fontsize=9.3, color=INK)
    ax2b.tick_params(labelsize=8.3, colors=INK)
    ax2.set_xlabel("epochs over the same batch", fontsize=9.5)
    ax2.set_ylabel("largest |r - 1|", fontsize=9.5, color=ACCENT)
    ax2.set_title("the ratio walks straight past it", fontsize=10,
                  color=INK, weight="bold")
    ax2.grid(alpha=0.13, which="both")
    ax2.tick_params(labelsize=8.5, colors=INK)
    for sp in list(ax2.spines.values()) + list(ax2b.spines.values()):
        sp.set_color(INK)
        sp.set_linewidth(0.7)

    fig.text(0.5, -0.04,
             f"after {eps_list[-1]} epochs the worst ratio sits at "
             f"{rats[-1]:.3f}, about {rats[-1] / EPS:.1f}x the clip, with "
             f"KL {kls[-1]:.3f}.",
             ha="center", fontsize=8.7, style="italic", color=INK)
    fig.tight_layout()
    fig.savefig(HERE / "clipping_does_not_bound.svg", transparent=True,
                bbox_inches="tight")
    plt.close(fig)
    print(f"wrote clipping_does_not_bound.svg   ratio {rats[0]:.3f} -> "
          f"{rats[-1]:.3f} (eps {EPS})   KL -> {kls[-1]:.4f}")


if __name__ == "__main__":
    theory_to_algorithm()
    clipping_does_not_bound()
