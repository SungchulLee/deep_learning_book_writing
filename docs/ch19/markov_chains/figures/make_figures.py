"""20.1 마르코프 사슬 쪽들의 그림.

규칙은 CLAUDE.md 의 Figures 항에 적어 둔 대로다.
- SVG, fonttype='path', transparent
- 글자는 모두 아스키 (matplotlib 기본 폰트에 한글이 없다)

자료를 읽지 않는다. 쓰는 수는 1913년 마르코프가 발표한 집계뿐이고,
비교값은 그 집계에서 여기서 셈한다.

실행: python docs/ch19/markov_chains/figures/make_figures.py
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

# 마르코프가 1913년에 발표한 집계 (『예브게니 오네긴』 첫 2만 글자)
N_LETTERS = 20_000
N_VOWEL = 8_638
N_CONS = 11_362
N_VV = 1_104          # 모음-모음 이웃 쌍


# === 그림으로 보는 AI: 독립이 아니어도 큰 수의 법칙은 선다 ====================
#
# 마르코프가 센 수만 쓰고, 독립이었다면 몇 쌍이 나왔을지는 여기서 셈한다.
# 그 둘의 차이가 그가 보이려던 것이다 — 글자는 서로 기대어 있다.

def markov_onegin():
    assert N_VOWEL + N_CONS == N_LETTERS, "집계가 맞아떨어져야 한다"

    pairs = N_LETTERS - 1
    p_v = N_VOWEL / N_LETTERS
    exp_vv = pairs * p_v * p_v                 # 독립이라면
    p_v_after_v = N_VV / N_VOWEL
    p_v_after_c = (N_VOWEL - N_VV) / N_CONS    # 대략 — 경계 효과는 무시한다
    # 이 그림이 주장하는 것 — 앞 글자를 알면 확률이 크게 달라진다
    assert N_VV < exp_vv * 0.5, "셈한 쌍이 독립 예측보다 훨씬 적어야 한다"
    assert p_v_after_v < p_v < p_v_after_c, "조건부가 양쪽으로 갈라져야 한다"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.6, 4.0))

    # --- 왼쪽: 독립이라면 몇 쌍이었을까 ------------------------------------
    ax1.bar([0, 1], [exp_vv, N_VV], color=[INK, ACCENT], width=0.52,
            alpha=0.85)
    for i, v in enumerate((exp_vv, N_VV)):
        ax1.text(i, v + exp_vv * 0.035, f"{v:,.0f}", ha="center",
                 fontsize=10, color=INK, weight="bold")
    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(["if letters were\nindependent",
                         "what Markov\ncounted"], fontsize=9)
    ax1.set_ylabel("vowel-vowel neighbours", fontsize=9)
    ax1.set_ylim(0, exp_vv * 1.22)
    ax1.set_title(f"{N_VV / exp_vv * 100:.0f}% of what independence predicts",
                  fontsize=10, color=INK, weight="bold")
    ax1.grid(alpha=0.13, axis="y")

    # --- 오른쪽: 앞 글자를 알면 ---------------------------------------------
    vals = [p_v_after_v, p_v, p_v_after_c]
    cols = [ACCENT, INK, ACCENT]
    ax2.bar(range(3), vals, color=cols, width=0.55, alpha=0.85)
    for i, v in enumerate(vals):
        ax2.text(i, v + 0.025, f"{v:.4f}", ha="center", fontsize=9.5,
                 color=INK)
    ax2.axhline(p_v, color=INK, ls=":", lw=1.2, alpha=0.6)
    ax2.set_xticks(range(3))
    ax2.set_xticklabels(["after a\nvowel", "knowing\nnothing",
                         "after a\nconsonant"], fontsize=9)
    ax2.set_ylabel("probability the letter is a vowel", fontsize=9)
    ax2.set_ylim(0, max(vals) * 1.22)
    ax2.set_title("one letter back changes everything", fontsize=10,
                  color=INK, weight="bold")
    ax2.grid(alpha=0.13, axis="y")

    for ax in (ax1, ax2):
        ax.tick_params(labelsize=8, colors=INK)
        for s in ax.spines.values():
            s.set_color(INK)
            s.set_linewidth(0.7)

    fig.text(0.5, -0.05,
             f"Markov's own tallies: {N_VOWEL:,} vowels, {N_CONS:,} consonants, "
             f"{N_VV:,} vowel-vowel pairs in the first {N_LETTERS:,} letters.",
             ha="center", fontsize=8.6, style="italic", color=INK)
    fig.tight_layout()
    fig.savefig(HERE / "markov_onegin.svg", transparent=True,
                bbox_inches="tight")
    plt.close(fig)
    print(f"wrote markov_onegin.svg   exp {exp_vv:,.0f} vs {N_VV:,}   "
          f"P(v) {p_v:.4f} -> {p_v_after_v:.4f} / {p_v_after_c:.4f}")


# === 그림으로 보는 AI: 가장 그럴듯한 자리와 가장 그럴듯한 거리 ================
#
# 피어슨의 1905년 결론("취한 사람을 찾을 가장 그럴듯한 자리는 출발점 근처")은
# **자리마다의 확률**에 대한 말이라 참이다. 그런데 **거리**의 최빈값은 0 이
# 아니다 — 멀리 있는 둘레에는 자리가 훨씬 많기 때문이다. 둘 다 재어 보인다.

def pearson_walk():
    rng = np.random.default_rng(0)
    n, trials = 400, 400_000

    steps = rng.integers(0, 4, size=(trials, n))
    dx = ((steps == 0).astype(np.int8) - (steps == 1).astype(np.int8)).sum(1)
    dy = ((steps == 2).astype(np.int8) - (steps == 3).astype(np.int8)).sum(1)
    r = np.hypot(dx, dy)

    mean_r = r.mean()
    theory = np.sqrt(np.pi * n / 4)

    # 거리의 분포. 격자 홀짝 때문에 한 칸짜리 통은 들쭉날쭉하므로 두 칸으로 묶는다.
    edges = np.arange(0, int(r.max()) + 3, 2)
    counts, _ = np.histogram(r, bins=edges)
    mids = (edges[:-1] + edges[1:]) / 2
    mode_r = mids[counts.argmax()]

    # **자리 하나당** 확률은 둘레를 어림하지 말고, x 축을 따라 한 줄로 베어
    # 실제 자리의 확률을 그대로 센다. 400 걸음 뒤 (x,0) 은 x 가 짝수일 때만 닿는다.
    xs_cut = np.arange(0, 41, 2)
    at_y0 = dy == 0
    site_p = np.array([((dx == x) & at_y0).mean() for x in xs_cut])

    assert abs(mean_r - theory) / theory < 0.01, "평균 거리가 이론과 맞아야 한다"
    assert mode_r > 0, "거리의 최빈값이 0 이 아니어야 이 그림의 이야기가 산다"
    # 원점이 맨 꼭대기라고 단언하지 않는다 — 가까운 몇 자리의 차이는 이 표본
    # 크기에서 잡음에 묻힌다(참값 차이가 1% 남짓이라 6천만 번쯤 돌려야 갈린다).
    # 단언할 수 있는 것은 가까운 쪽과 먼 쪽의 차이다.
    near, far = site_p[0], site_p[-1]
    assert near > far * 3, "가까운 자리가 먼 자리보다 뚜렷이 높아야 한다"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.8, 4.0))

    # --- 왼쪽: 거리의 분포 -------------------------------------------------
    ax1.bar(mids, counts / trials, width=1.8, color=ACCENT, alpha=0.55,
            edgecolor="none")
    ax1.axvline(mode_r, color=ACCENT, lw=1.6)
    ax1.annotate(f"most common distance\nr = {mode_r:.0f}",
                 xy=(mode_r, (counts / trials).max()), xytext=(14, -6),
                 textcoords="offset points", fontsize=8.6, color=ACCENT)
    ax1.axvline(0, color=INK, ls=":", lw=1.4)
    ax1.annotate("back at the start", xy=(0, 0), xytext=(10, 40),
                 textcoords="offset points", fontsize=8.4, color=INK,
                 arrowprops=dict(arrowstyle="->", color=INK, lw=0.9))
    ax1.set_xlabel("distance from the start", fontsize=9)
    ax1.set_ylabel("fraction of walks", fontsize=9)
    ax1.set_xlim(-2, mode_r * 3)
    ax1.set_title("how far they end up", fontsize=10, color=INK,
                  weight="bold")
    ax1.grid(alpha=0.13)

    # --- 오른쪽: 자리 하나당 확률 ------------------------------------------
    ax2.semilogy(xs_cut, site_p, "o-", color=INK, lw=1.9, ms=4)
    ax2.plot(xs_cut[:3], site_p[:3], "o", color=ACCENT, ms=9, zorder=5)
    ax2.annotate("near the start - these three are\nthe same to within noise",
                 xy=(3, site_p[1]), xytext=(42, -34),
                 textcoords="offset points", fontsize=8.4, color=ACCENT,
                 arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.0))
    ax2.annotate(f"{near / far:.0f}x lower", xy=(xs_cut[-1], far),
                 xytext=(-66, 4), textcoords="offset points", fontsize=8.6,
                 color=INK,
                 arrowprops=dict(arrowstyle="->", color=INK, lw=0.9))
    ax2.set_xlabel("one spot, x steps along a line from the start",
                   fontsize=9)
    ax2.set_ylabel("chance of ending on that spot (log)", fontsize=9)
    ax2.set_title("but which single spot is likeliest", fontsize=10,
                  color=INK, weight="bold")
    ax2.grid(alpha=0.13, which="both")

    for ax in (ax1, ax2):
        ax.tick_params(labelsize=8, colors=INK)
        for s in ax.spines.values():
            s.set_color(INK)
            s.set_linewidth(0.7)

    fig.text(0.5, -0.04,
             f"{trials:,} walks of {n} steps. mean distance {mean_r:.3f}, "
             f"against sqrt(pi*n/4) = {theory:.3f}. both panels are the same "
             f"data. the peak itself is too flat to resolve at this many walks.",
             ha="center", fontsize=8.6, style="italic", color=INK)
    fig.tight_layout()
    fig.savefig(HERE / "pearson_walk.svg", transparent=True,
                bbox_inches="tight")
    plt.close(fig)
    print(f"wrote pearson_walk.svg   mean {mean_r:.3f} vs {theory:.3f}   "
          f"mode r={mode_r:.0f}")


# === 그림으로 보는 AI: 0.85 가 사는 것은 그럴듯함이 아니라 유일성 ============
#
# 떨어진 섬이 있는 그래프에서 감쇠 없이(alpha=1) 거듭제곱법을 돌리면 **출발점에
# 따라 답이 달라진다** — 정상분포가 유일하지 않다. 감쇠를 넣으면 어디서 출발해도
# 같은 답이 나오고, 둘째 고윳값의 크기가 정확히 alpha 라 수렴 속도까지 정해진다.

def pagerank_damping():
    n = 5
    links = {0: [1, 2], 1: [2], 2: [0], 3: [4], 4: [3]}   # {3,4} 는 떨어진 섬
    M = np.zeros((n, n))
    for i, outs in links.items():
        for j in outs:
            M[j, i] = 1 / len(outs)

    def matrix(alpha):
        return alpha * M + (1 - alpha) * np.ones((n, n)) / n

    def power(alpha, v0, iters=20_000, tol=1e-13):
        A = matrix(alpha)
        v = v0 / v0.sum()
        for k in range(iters):
            w = A @ v
            if np.abs(w - v).sum() < tol:
                return w, k
            v = w
        return v, iters

    starts = {"uniform": np.ones(n),
              "the mainland": np.array([1.0, 0, 0, 0, 0]),
              "the island": np.array([0.0, 0, 0, 1, 0])}
    none = {k: power(1.0, v.copy())[0] for k, v in starts.items()}
    damped = {k: power(0.85, v.copy())[0] for k, v in starts.items()}

    alphas = [0.5, 0.7, 0.85, 0.95, 0.99]
    iters, lam2 = [], []
    for a in alphas:
        iters.append(power(a, np.ones(n))[1])
        ev = np.sort(np.abs(np.linalg.eigvals(matrix(a))))[::-1]
        lam2.append(ev[1])

    vals = list(none.values())
    assert max(np.abs(vals[0] - vals[1])) > 0.1, \
        "감쇠가 없으면 출발점에 따라 답이 달라야 한다"
    dv = list(damped.values())
    assert max(np.abs(dv[0] - dv[1])) < 1e-6, \
        "감쇠를 넣으면 출발점과 무관해야 한다"
    assert max(abs(l - a) for l, a in zip(lam2, alphas)) < 1e-6, \
        "둘째 고윳값의 크기가 alpha 와 같아야 한다"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.0),
                                   gridspec_kw={"width_ratios": [1.3, 1]})

    # --- 왼쪽: 출발점을 바꿔 보면 ------------------------------------------
    idx = np.arange(n)
    w = 0.26
    for k, (name, v) in enumerate(none.items()):
        ax1.bar(idx + (k - 1) * w, v, width=w, alpha=0.85,
                color=[ACCENT, INK, MUTED][k], label=f"started at {name}")
    ax1.set_xticks(idx)
    ax1.set_xticklabels([f"page {i}" for i in idx], fontsize=8.4)
    ax1.set_ylabel("score", fontsize=9)
    ax1.set_title("no damping: three starts, three answers", fontsize=10,
                  color=INK, weight="bold")
    ax1.legend(fontsize=8, framealpha=0.92)
    ax1.grid(alpha=0.13, axis="y")
    ax1.text(0.5, -0.30,
             "with damping at 0.85 all three agree to four decimals",
             transform=ax1.transAxes, ha="center", fontsize=8.6,
             color=ACCENT)

    # --- 오른쪽: alpha 가 정하는 수렴 속도 ---------------------------------
    ax2.plot(alphas, iters, "o-", color=ACCENT, lw=2.1, ms=7)
    for a, it in zip(alphas, iters):
        ax2.annotate(f"{it}", xy=(a, it), xytext=(0, 9),
                     textcoords="offset points", ha="center", fontsize=8.4,
                     color=INK)
    ax2.set_xlabel("damping factor", fontsize=9)
    ax2.set_ylabel("iterations to settle", fontsize=9)
    ax2.set_title("and it sets the cost", fontsize=10, color=INK,
                  weight="bold")
    ax2.grid(alpha=0.13)
    ax2.text(0.5, 0.06, "the second eigenvalue is exactly the damping factor",
             transform=ax2.transAxes, ha="center", fontsize=8.4, color=INK,
             style="italic")

    for ax in (ax1, ax2):
        ax.tick_params(labelsize=8, colors=INK)
        for s in ax.spines.values():
            s.set_color(INK)
            s.set_linewidth(0.7)

    fig.tight_layout()
    fig.savefig(HERE / "pagerank_damping.svg", transparent=True,
                bbox_inches="tight")
    plt.close(fig)
    print(f"wrote pagerank_damping.svg   iters {iters}   lam2 "
          f"{[round(l, 4) for l in lam2]}")


if __name__ == "__main__":
    markov_onegin()
    pearson_walk()
    pagerank_damping()
