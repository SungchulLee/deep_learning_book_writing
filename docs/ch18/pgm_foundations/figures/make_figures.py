"""19.1 확률 그래프 모델 기초 쪽들의 그림.

규칙은 CLAUDE.md 의 Figures 항에 적어 둔 대로다.
- SVG, fonttype='path', transparent
- 글자는 모두 아스키 (matplotlib 기본 폰트에 한글이 없다)

자료를 읽지 않으므로 어디서 돌려도 된다.

실행: python docs/ch18/pgm_foundations/figures/make_figures.py
"""
import itertools
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

# 4-고리 위의 고전적 반례. 균등하게 8 개 배치에만 질량을 둔다.
SUPPORT = [(0, 0, 0, 0), (1, 0, 0, 0), (1, 1, 0, 0), (1, 1, 1, 0),
           (1, 1, 1, 1), (0, 1, 1, 1), (0, 0, 1, 1), (0, 0, 0, 1)]
EDGES = [(0, 1), (1, 2), (2, 3), (3, 0)]


def _joint():
    P = np.zeros((2, 2, 2, 2))
    for s in SUPPORT:
        P[s] = 1 / 8
    return P


def _ci_violation(P, a, b, g1, g2):
    """a ⟂ b | g1,g2 가 깨지는 최대 폭. 직접 열거한다."""
    worst = 0.0
    for v1, v2 in itertools.product((0, 1), (0, 1)):
        tab = np.zeros((2, 2))
        for xa, xb in itertools.product((0, 1), (0, 1)):
            idx = [None] * 4
            idx[a], idx[b], idx[g1], idx[g2] = xa, xb, v1, v2
            tab[xa, xb] = P[tuple(idx)]
        if tab.sum() <= 0:
            continue
        tab = tab / tab.sum()
        worst = max(worst, np.abs(tab - np.outer(tab.sum(1),
                                                 tab.sum(0))).max())
    return worst


# === 그림으로 보는 AI: 양수성이 가정이 아니라 벽인 까닭 ========================
#
# 이 분포는 4-고리의 마르코프 성질을 **정확히** 만족한다(어긋남 0).
# 그런데 클리크 인수분해로는 적을 수 없다. 까닭이 세는 것만으로 나온다 —
# 네 변이 저마다 2x2 네 쌍을 **전부** 쓰므로 어떤 psi 도 0 일 수 없고,
# 그러면 곱이 어디서나 양수가 되어 0 인 칸 여덟을 만들 수 없다.
#
# 그림에 적는 수는 모두 여기서 셈하며, 주장마다 assert 로 확인하고 그린다.

def positivity_wall():
    P = _joint()

    # --- 주장 1: 마르코프 성질이 정확히 성립한다 ---------------------------
    v1 = _ci_violation(P, 0, 2, 1, 3)
    v2 = _ci_violation(P, 1, 3, 0, 2)
    assert max(v1, v2) < 1e-12, f"마르코프 성질이 깨진다: {max(v1, v2)}"

    # --- 주장 2: 네 변이 저마다 네 쌍을 전부 쓴다 --------------------------
    used = {e: set() for e in EDGES}
    for x in SUPPORT:
        for e in EDGES:
            used[e].add((x[e[0]], x[e[1]]))
    assert all(len(u) == 4 for u in used.values()), "네 쌍을 다 쓰지 않는다"

    # --- 주장 3: 열여섯 칸 가운데 여덟이 0 이다 ----------------------------
    n_zero = int((P == 0).sum())
    assert n_zero == 8, f"0 인 칸이 8 이 아니다: {n_zero}"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.8, 4.3),
                                   gridspec_kw={"width_ratios": [1, 1.25]})

    # --- 왼쪽: 고리와 열여섯 칸 -------------------------------------------
    pos = {0: (0.30, 0.80), 1: (0.70, 0.80), 2: (0.70, 0.52), 3: (0.30, 0.52)}
    for a, b in EDGES:
        ax1.plot(*zip(pos[a], pos[b]), color=INK, linewidth=1.6, zorder=1)
    for i, (x, y) in pos.items():
        ax1.add_patch(plt.Circle((x, y), 0.062, facecolor="white",
                                 edgecolor=INK, linewidth=1.5, zorder=3))
        ax1.text(x, y, f"X{i+1}", ha="center", va="center", fontsize=9.5,
                 color=INK, zorder=4)
    ax1.text(0.5, 0.93, "a 4-cycle, all binary", ha="center", fontsize=9.2,
             color=INK, style="italic")

    for k, cfg in enumerate(sorted(itertools.product((0, 1), repeat=4))):
        r, c = divmod(k, 4)
        x0, y0 = 0.10 + c * 0.21, 0.30 - r * 0.082
        on = P[cfg] > 0
        ax1.add_patch(plt.Rectangle((x0, y0), 0.185, 0.068,
                                    facecolor=ACCENT if on else "white",
                                    alpha=0.30 if on else 1.0,
                                    edgecolor=INK if on else "#c8c8c8",
                                    linewidth=0.9))
        ax1.text(x0 + 0.0925, y0 + 0.034, "".join(map(str, cfg)),
                 ha="center", va="center", fontsize=7.6,
                 color=INK if on else "#b4b4b4", family="monospace")
    ax1.text(0.5, 0.405, f"{16 - n_zero} of 16 carry mass,  "
                         f"{n_zero} are exactly zero",
             ha="center", fontsize=8.4, color=ACCENT)
    ax1.text(0.5, -0.045, "so the positivity condition fails", ha="center",
             fontsize=8.6, color=INK)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(-0.09, 1.0)
    ax1.axis("off")

    # --- 오른쪽: 세기만으로 끝나는 논증 ------------------------------------
    ax2.text(0.5, 0.95, "every edge uses all four of its pairs",
             ha="center", fontsize=9.6, color=INK, weight="bold")
    for k, e in enumerate(EDGES):
        y = 0.80 - k * 0.088
        ax2.text(0.10, y, f"psi(X{e[0]+1}, X{e[1]+1})", fontsize=9,
                 color=INK, family="monospace", va="center")
        for j, pair in enumerate(sorted(used[e])):
            ax2.add_patch(plt.Rectangle((0.48 + j * 0.105, y - 0.028), 0.088,
                                        0.056, facecolor=ACCENT, alpha=0.28,
                                        edgecolor=ACCENT, linewidth=0.8))
            ax2.text(0.524 + j * 0.105, y, f"{pair[0]}{pair[1]}",
                     ha="center", va="center", fontsize=7.6, color=INK,
                     family="monospace")
        ax2.text(0.93, y, "4/4", fontsize=8.2, color=ACCENT, va="center")

    steps = [
        "none of the four psi can be zero anywhere",
        "so the product is positive everywhere",
        "so it cannot be zero on the other eight",
    ]
    for k, s in enumerate(steps):
        ax2.text(0.10, 0.37 - k * 0.085, f"{k+1}.  {s}", fontsize=8.8,
                 color=INK, va="center")
    ax2.plot([0.08, 0.96], [0.075, 0.075], color=ACCENT, linewidth=1.2)
    ax2.text(0.5, 0.012, "no clique factorization exists -- "
                         "yet the Markov property holds exactly",
             ha="center", fontsize=8.8, color=ACCENT, weight="bold")
    ax2.set_xlim(0, 1)
    ax2.set_ylim(-0.03, 1.02)
    ax2.axis("off")

    fig.tight_layout()
    fig.savefig(HERE / "positivity_wall.svg", transparent=True,
                bbox_inches="tight")
    plt.close(fig)
    print(f"wrote positivity_wall.svg   markov violation "
          f"{max(v1, v2):.1e}   zeros {n_zero}/16")


# === 그림으로 보는 AI: 충돌자를 조건으로 두면 =================================
#
# 독립인 두 병 A, B 를 만들고, 둘 다 입원 확률을 높이게 한다 (A -> H <- B).
# 모집단 오즈비는 1 인데 입원한 사람만 보면 1 보다 한참 아래로 내려간다.
# 입원 유발력을 키워 가며 재서, 치우침이 **정도의 문제**임을 보인다.

def berkson():
    rng = np.random.default_rng(0)
    n = 3_000_000
    A = rng.random(n) < 0.10
    B = rng.random(n) < 0.10
    u = rng.random(n)

    def odds_ratio(a, b):
        n11 = int((a & b).sum()); n10 = int((a & ~b).sum())
        n01 = int((~a & b).sum()); n00 = int((~a & ~b).sum())
        return (n11 * n00) / (n10 * n01) if n10 and n01 else float("nan")

    or_pop = odds_ratio(A, B)
    assert abs(or_pop - 1.0) < 0.02, f"모집단에서 독립이 아니다: {or_pop}"

    strengths = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    ors, rates = [], []
    for s in strengths:
        H = u < (0.02 + s * A + s * B)
        ors.append(odds_ratio(A[H], B[H]))
        rates.append(float(H.mean()))
    assert ors[-1] < 0.2, f"치우침이 나타나지 않았다: {ors[-1]}"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 4.0),
                                   gridspec_kw={"width_ratios": [1, 1.25]})

    # --- 왼쪽: 충돌자 ------------------------------------------------------
    pos = {"A": (0.20, 0.72), "B": (0.80, 0.72), "H": (0.50, 0.36)}
    for src in ("A", "B"):
        ax1.annotate("", xy=(pos["H"][0] + (-0.07 if src == "A" else 0.07),
                             pos["H"][1] + 0.055),
                     xytext=pos[src],
                     arrowprops=dict(arrowstyle="->", color=INK, linewidth=1.8))
    labels = {"A": "disease A", "B": "disease B", "H": "admitted"}
    for k, (x, y) in pos.items():
        ax1.add_patch(plt.Circle((x, y), 0.072, facecolor="white",
                                 edgecolor=ACCENT if k == "H" else INK,
                                 linewidth=1.8, zorder=3))
        ax1.text(x, y, k, ha="center", va="center", fontsize=11, color=INK,
                 zorder=4)
        ax1.text(x, y - 0.115, labels[k], ha="center", fontsize=8,
                 color=INK, alpha=0.8)
    ax1.text(0.5, 0.93, "a collider", ha="center", fontsize=9.6, color=INK,
             weight="bold")
    ax1.text(0.5, 0.13, "A and B independent\n"
                        "until you look only at H = 1",
             ha="center", va="center", fontsize=8.8, color=ACCENT)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0.02, 1.0)
    ax1.axis("off")

    # --- 오른쪽: 재어 보면 -------------------------------------------------
    ax2.axhline(1.0, color=INK, linestyle="--", linewidth=1.3)
    ax2.text(0.58, 0.72, f"population odds ratio {or_pop:.2f}", fontsize=8.4,
             color=INK, ha="right")
    ax2.plot(strengths, ors, "o-", color=ACCENT, linewidth=2.0, ms=6)
    ax2.annotate(f"{ors[-1]:.3f}", xy=(strengths[-1], ors[-1]),
                 xytext=(-6, 16), textcoords="offset points", fontsize=9,
                 color=ACCENT, ha="right")
    ax2.set_yscale("log")
    ax2.set_xlabel("how strongly each disease drives admission", fontsize=9)
    ax2.set_ylabel("odds ratio among the admitted", fontsize=9)
    ax2.set_title("the same two independent diseases", fontsize=10,
                  color=INK, weight="bold")
    ax2.grid(alpha=0.13, which="both")
    ax2.tick_params(labelsize=8, colors=INK)
    for s in ax2.spines.values():
        s.set_color(INK)
        s.set_linewidth(0.7)

    fig.text(0.5, -0.04,
             "3,000,000 people, A and B built independent. "
             "nothing about them changes - only who is looked at.",
             ha="center", fontsize=8.6, style="italic", color=INK)
    fig.tight_layout()
    fig.savefig(HERE / "berkson.svg", transparent=True, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote berkson.svg   population OR {or_pop:.3f}   "
          f"admitted OR {ors[1]:.3f} -> {ors[-1]:.3f}")


# === 그림으로 보는 AI: 화살표는 자료가 정해 주지 않는다 =======================
#
# 사슬 X->Y->Z 에서 만든 결합분포를 네 가지 DAG 로 적합해 본다. 사슬·포크·
# 뒤집은 사슬 셋은 **같은 분포**를 돌려주고(KL 이 기계 오차), 충돌점만 다르다.
# v-구조가 같은 DAG 끼리는 관측 자료로 구별되지 않는다는 뜻이다.

def arrows_undetermined():
    pX = np.array([0.7, 0.3])
    pY_X = np.array([[0.8, 0.2], [0.35, 0.65]])
    pZ_Y = np.array([[0.9, 0.1], [0.25, 0.75]])
    P = np.zeros((2, 2, 2))
    for x, y, z in itertools.product((0, 1), repeat=3):
        P[x, y, z] = pX[x] * pY_X[x, y] * pZ_Y[y, z]
    assert abs(P.sum() - 1) < 1e-12, "참 분포가 1 로 안 더해진다"

    def fit(parents):
        Q = np.ones((2, 2, 2))
        for v, pa in parents.items():
            drop = tuple(i for i in range(3) if i != v and i not in pa)
            jv = P.sum(axis=drop) if drop else P.copy()
            ax = sorted([v] + list(pa))
            mp = jv.sum(axis=ax.index(v), keepdims=True)
            cond = np.divide(jv, mp, out=np.zeros_like(jv), where=mp > 0)
            shape = [1, 1, 1]
            for a in ax:
                shape[a] = 2
            Q = Q * cond.reshape(shape)
        return Q / Q.sum()

    def kl(p, q):
        m = p > 0
        return float((p[m] * np.log(p[m] / np.clip(q[m], 1e-300, None))).sum())

    dags = [
        ("X -> Y -> Z", {0: (), 1: (0,), 2: (1,)}, [(0, 1), (1, 2)]),
        ("X <- Y -> Z", {1: (), 0: (1,), 2: (1,)}, [(1, 0), (1, 2)]),
        ("X <- Y <- Z", {2: (), 1: (2,), 0: (1,)}, [(2, 1), (1, 0)]),
        ("X -> Y <- Z", {0: (), 2: (), 1: (0, 2)}, [(0, 1), (2, 1)]),
    ]
    kls = [kl(P, fit(pa)) for _, pa, _ in dags]
    assert max(abs(k) for k in kls[:3]) < 1e-12, "셋이 같지 않다"
    assert kls[3] > 1e-3, "충돌점이 구별되지 않는다"

    fig, axes = plt.subplots(1, 4, figsize=(11.0, 3.5))
    xs = {0: 0.16, 1: 0.50, 2: 0.84}
    for ax, (name, _, arcs), k in zip(axes, dags, kls):
        same = abs(k) < 1e-12
        for a, b in arcs:
            d = 0.10 if xs[b] > xs[a] else -0.10
            ax.annotate("", xy=(xs[b] - d, 0.60), xytext=(xs[a] + d, 0.60),
                        arrowprops=dict(arrowstyle="->", color=INK,
                                        linewidth=1.7))
        for i, lab in enumerate("XYZ"):
            ax.add_patch(plt.Circle((xs[i], 0.60), 0.085, facecolor="white",
                                    edgecolor=INK, linewidth=1.5, zorder=3))
            ax.text(xs[i], 0.60, lab, ha="center", va="center", fontsize=10,
                    color=INK, zorder=4)
        ax.text(0.5, 0.90, name, ha="center", fontsize=9.6, color=INK,
                family="monospace")
        ax.text(0.5, 0.30, "KL to the data", ha="center", fontsize=7.8,
                color=INK, alpha=0.75)
        ax.text(0.5, 0.17, "0" if same else f"{k:.3f}", ha="center",
                fontsize=13 if same else 12,
                color=ACCENT if not same else INK, weight="bold")
        ax.text(0.5, 0.02,
                "indistinguishable" if same else "distinguishable",
                ha="center", fontsize=8.2,
                color=INK if same else ACCENT,
                style="italic" if same else "normal",
                weight="normal" if same else "bold")
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.03, 1.0)
        ax.axis("off")

    fig.text(0.5, -0.04,
             "the first three fit the very same distribution to machine "
             "precision. only the collider differs.",
             ha="center", fontsize=8.8, style="italic", color=INK)
    fig.tight_layout()
    fig.savefig(HERE / "arrows_undetermined.svg", transparent=True,
                bbox_inches="tight")
    plt.close(fig)
    print(f"wrote arrows_undetermined.svg   KLs "
          f"{[f'{k:.1e}' for k in kls]}")


if __name__ == "__main__":
    positivity_wall()
    berkson()
    arrows_undetermined()
