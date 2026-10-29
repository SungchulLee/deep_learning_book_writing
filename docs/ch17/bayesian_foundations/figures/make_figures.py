"""18.1 베이즈 기초 쪽들의 그림.

규칙은 CLAUDE.md 의 Figures 항에 적어 둔 대로다.
- SVG, fonttype='path', transparent
- 글자는 모두 아스키 (matplotlib 기본 폰트에 한글이 없다)

자료를 읽지 않으므로 어디서 돌려도 된다.
그림마다 **제 주장을 스스로 확인하고**, 어긋나면 그리지 않고 멈춘다.

실행: python docs/ch17/bayesian_foundations/figures/make_figures.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["svg.fonttype"] = "path"

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
INK = "#2f3337"
ACCENT = "#8a4b2a"
MUTED = "#2d6a8f"


# === 그림으로 보는 AI: 베이즈가 실제로 그린 것 ================================
#
# 1763년 글의 Postulate 1 은 "square table or plane ABCD" 다. 당구대가 아니다
# (원문에 billiard 는 한 번도 나오지 않는다). 공 W 를 먼저 던져 AD 에 나란한
# 선 os 를 긋고, 공 O 를 n 번 던져 AD 와 os 사이에 떨어지는 횟수를 센다.
#
# 오른쪽은 그 설정을 그대로 돌린 결과다. 균등 사전분포(Postulate 1)에서
# 사후분포는 Beta(p+1, q+1) 이고, 글이 요구한 것은 **두 점 사이의 확률**이다.

def bayes_table():
    rng = np.random.default_rng(1763)
    n = 12
    x_true = 0.62                      # 공 W 가 멈춘 자리
    throws = rng.random(n)
    p = int((throws < x_true).sum())   # AD 와 os 사이 = 사건 M
    q = n - p
    assert p + q == n, "앞뒤 횟수의 합이 n 이어야 한다"

    post = stats.beta(p + 1, q + 1)    # 균등 사전분포에서의 사후분포
    lo, hi = 0.40, 0.80                # "any two degrees of probability"
    prob = post.cdf(hi) - post.cdf(lo)

    # 그림이 제 주장을 스스로 확인한다 — 칠한 넓이가 그 확률과 같은가
    grid = np.linspace(lo, hi, 20001)
    area = np.trapz(post.pdf(grid), grid)      # 이 기계의 numpy 는 1.x 다
    assert abs(area - prob) < 1e-6, f"칠한 넓이 {area} != 확률 {prob}"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.6, 4.2),
                                   gridspec_kw={"width_ratios": [1, 1.12]})

    # --- 왼쪽: square table ABCD ------------------------------------------
    ax1.add_patch(plt.Rectangle((0, 0), 1, 1, facecolor="none",
                                edgecolor=INK, linewidth=1.5))
    for (x, y, lab, ha, va) in ((0, 0, "A", "right", "top"),
                                (1, 0, "B", "left", "top"),
                                (1, 1, "C", "left", "bottom"),
                                (0, 1, "D", "right", "bottom")):
        ax1.text(x, y, f" {lab} ", ha=ha, va=va, fontsize=10, color=INK,
                 weight="bold")

    ax1.plot([x_true, x_true], [0, 1], color=ACCENT, linewidth=2.0)
    ax1.text(x_true, 1.04, "s", ha="center", fontsize=9.5, color=ACCENT,
             weight="bold")
    ax1.text(x_true, -0.07, "o", ha="center", fontsize=9.5, color=ACCENT,
             weight="bold")
    ax1.plot([x_true], [0.52], marker="o", color=ACCENT, markersize=10,
             markeredgecolor="white", markeredgewidth=1.3, zorder=5)
    ax1.text(x_true - 0.03, 0.52, "W ", ha="right", va="center", fontsize=9,
             color=ACCENT)

    ys = np.linspace(0.12, 0.90, n)
    for t, y in zip(throws, ys):
        left = t < x_true
        ax1.plot([t], [y], marker="o", markersize=5.5,
                 color=MUTED if left else INK,
                 alpha=0.9 if left else 0.45, zorder=4)

    ax1.annotate("", xy=(x_true - 0.015, 0.045), xytext=(0.015, 0.045),
                 arrowprops=dict(arrowstyle="<->", color=MUTED, linewidth=1.2))
    ax1.text(x_true / 2, 0.085, f"event M  ({p} of {n})", ha="center",
             fontsize=8.4, color=MUTED)

    ax1.set_title('"the square table or plane ABCD"', fontsize=10, color=INK,
                  weight="bold")
    ax1.text(0.5, -0.20, "Postulate 1, 1763.  the word billiard\n"
                         "appears nowhere in the essay.",
             ha="center", va="top", fontsize=8.4, color=INK, alpha=0.85)
    ax1.set_xlim(-0.1, 1.1)
    ax1.set_ylim(-0.34, 1.14)
    ax1.set_aspect("equal")
    ax1.axis("off")

    # --- 오른쪽: 글이 요구한 것 --------------------------------------------
    xs = np.linspace(0, 1, 800)
    ax2.plot(xs, post.pdf(xs), color=INK, linewidth=1.9)
    shade = np.linspace(lo, hi, 400)
    ax2.fill_between(shade, 0, post.pdf(shade), color=ACCENT, alpha=0.26)
    # 눈금에 이미 0.4 와 0.8 이 있으므로 수를 겹쳐 적지 않는다
    for v in (lo, hi):
        ax2.axvline(v, color=ACCENT, linestyle="--", linewidth=1.1)
    ax2.annotate("", xy=(hi, post.pdf(post.mean()) * 1.06),
                 xytext=(lo, post.pdf(post.mean()) * 1.06),
                 arrowprops=dict(arrowstyle="<->", color=ACCENT,
                                 linewidth=1.2))
    ax2.text((lo + hi) / 2, post.pdf(post.mean()) * 1.09, f"{prob:.3f}",
             ha="center", va="bottom", fontsize=13, color=ACCENT,
             weight="bold")
    ax2.set_xlabel("the unknown chance", fontsize=9)
    ax2.set_ylabel("density", fontsize=9)
    ax2.set_title(f"what the Problem asks for  (p={p}, q={q})", fontsize=10,
                  color=INK, weight="bold")
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, post.pdf(post.mean()) * 1.30)
    ax2.grid(alpha=0.12)
    ax2.tick_params(labelsize=8, colors=INK)
    for s in ax2.spines.values():
        s.set_color(INK)
        s.set_linewidth(0.7)

    fig.text(0.5, -0.03,
             '"the chance that the probability ... lies somewhere between any '
             'two degrees of probability that can be named"',
             ha="center", fontsize=8.6, style="italic", color=INK)
    fig.tight_layout()
    fig.savefig(HERE / "bayes_table.svg", transparent=True,
                bbox_inches="tight")
    plt.close(fig)
    print(f"wrote bayes_table.svg   p={p} q={q}  P({lo},{hi})={prob:.4f}")


# === 그림으로 보는 AI: 켤레족이 말할 수 없는 믿음 =============================
#
# "0.2 쯤이거나 0.8 쯤이다" 는 믿음을 베타 하나로는 적을 수 없다. 평균과 분산을
# 맞춘 베타는 **안쪽 봉우리가 아예 없고** 끝에서 발산한다 — 곧 "거의 늘 앞면
# 이거나 거의 늘 뒷면" 이라는 다른 믿음이다. 수는 모두 여기서 셈한다.

def conjugate_limit():
    from scipy.special import betaln

    w, A, B = 0.5, (2.0, 8.0), (8.0, 2.0)

    def mix_pdf(x, P1, P2, ww):
        return ww * stats.beta(*P1).pdf(x) + (1 - ww) * stats.beta(*P2).pdf(x)

    def mix_cdf(x, P1, P2, ww):
        return ww * stats.beta(*P1).cdf(x) + (1 - ww) * stats.beta(*P2).cdf(x)

    m1, v1 = stats.beta(*A).mean(), stats.beta(*A).var()
    m2, v2 = stats.beta(*B).mean(), stats.beta(*B).var()
    mean = w * m1 + (1 - w) * m2
    var = w * (v1 + m1 ** 2) + (1 - w) * (v2 + m2 ** 2) - mean ** 2
    k = mean * (1 - mean) / var - 1
    a, b = mean * k, (1 - mean) * k
    single = stats.beta(a, b)

    # 그림이 제 주장을 확인한다 — 평균과 분산이 정말 같은가
    assert abs(single.mean() - mean) < 1e-9, "평균이 안 맞는다"
    assert abs(single.var() - var) < 1e-9, "분산이 안 맞는다"

    # 그리고 안쪽 봉우리가 정말 없는가
    xs = np.linspace(1e-4, 1 - 1e-4, 20001)
    def interior_modes(y):
        return [xs[i] for i in range(1, len(xs) - 1)
                if y[i] > y[i - 1] and y[i] > y[i + 1]]
    assert len(interior_modes(mix_pdf(xs, A, B, w))) == 2, "혼합은 봉우리 둘"
    assert len(interior_modes(single.pdf(xs))) == 0, "단일 베타는 안쪽 봉우리 없음"

    def deterministic(cdf):
        return cdf(0.05) + (1 - cdf(0.95))

    d_mix = deterministic(lambda x: mix_cdf(x, A, B, w))
    d_one = deterministic(single.cdf)

    h, n = 1, 4
    def lml(P):
        return betaln(P[0] + h, P[1] + n - h) - betaln(*P)
    w1 = w * np.exp(lml(A))
    w2 = (1 - w) * np.exp(lml(B))
    w1, w2 = w1 / (w1 + w2), w2 / (w1 + w2)
    A2, B2 = (A[0] + h, A[1] + n - h), (B[0] + h, B[1] + n - h)
    p_mix = deterministic(lambda x: mix_cdf(x, A2, B2, w1))
    p_one = deterministic(stats.beta(a + h, b + n - h).cdf)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.6, 4.1), sharey=True)

    for ax, (P1, P2, ww), one, title, dm, do in (
            (ax1, (A, B, w), single, "the belief", d_mix, d_one),
            (ax2, (A2, B2, w1), stats.beta(a + h, b + n - h),
             f"after {h} head in {n}", p_mix, p_one)):
        ax.plot(xs, mix_pdf(xs, P1, P2, ww), color=ACCENT, linewidth=2.1,
                label="mixture of two Betas")
        ax.plot(xs, one.pdf(xs), color=INK, linewidth=1.9, linestyle="--",
                label="one Beta, same mean and variance" if ax is ax1
                else "one Beta")
        ax.axvspan(0, 0.05, color=MUTED, alpha=0.13)
        ax.axvspan(0.95, 1, color=MUTED, alpha=0.13)
        ax.set_title(title, fontsize=10, color=INK, weight="bold")
        ax.set_xlabel("the unknown chance", fontsize=9)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 3.6)
        ax.grid(alpha=0.12)
        ax.legend(fontsize=7.8, loc="upper center", framealpha=0.9)
        ax.text(0.5, 0.40,
                f"P(below 0.05 or above 0.95)\n"
                f"mixture {dm:.4f}      one Beta {do:.4f}",
                transform=ax.transAxes, ha="center", fontsize=8.4,
                color=INK,
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                          edgecolor=INK, alpha=0.9))
        ax.tick_params(labelsize=8, colors=INK)
        for s in ax.spines.values():
            s.set_color(INK)
            s.set_linewidth(0.7)

    ax1.set_ylabel("density", fontsize=9)
    fig.text(0.5, -0.03,
             f"identical mean ({mean:.3f}) and variance ({var:.4f}). "
             f"the shaded ends are where they disagree.",
             ha="center", fontsize=8.6, style="italic", color=INK)
    fig.tight_layout()
    fig.savefig(HERE / "conjugate_limit.svg", transparent=True,
                bbox_inches="tight")
    plt.close(fig)
    print(f"wrote conjugate_limit.svg   Beta({a:.3f},{b:.3f})  "
          f"prior {d_mix:.4f} vs {d_one:.4f}   post {p_mix:.4f} vs {p_one:.4f}")


# === 그림으로 보는 AI: 최빈값은 좌표를 타고 평균은 분포의 성질이다 =============
#
# Beta(3,7) 사후분포에서 theta 좌표의 MAP 은 (a-1)/(a+b-2) = 0.25 다.
# 로그오즈 좌표로 옮겨 최대점을 찾고 되돌리면 a/(a+b) = 0.30 — 사후평균이다.
# 같은 분포, 다른 좌표, 다른 "가장 그럴듯한 값". 둘 다 정확히 맞아떨어진다.

def map_not_invariant():
    a, b = 3.0, 7.0
    post = stats.beta(a, b)
    map_theta = (a - 1) / (a + b - 2)
    map_phi_back = a / (a + b)          # 로그오즈 좌표 MAP 을 되돌린 값

    # 그림이 제 주장을 스스로 확인한다
    th = np.linspace(1e-6, 1 - 1e-6, 2000001)
    assert abs(th[np.argmax(post.pdf(th))] - map_theta) < 1e-5, "theta MAP"
    logit = np.log(th / (1 - th))
    dens_phi = post.pdf(th) * th * (1 - th)      # 야코비안
    assert abs(th[np.argmax(dens_phi)] - map_phi_back) < 1e-5, "logit MAP"
    assert abs(post.mean() - map_phi_back) < 1e-12, "되돌린 값 = 사후평균"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.6, 4.0))

    xs = np.linspace(0, 1, 800)
    ax1.plot(xs, post.pdf(xs), color=INK, linewidth=2.0)
    ax1.axvline(map_theta, color=ACCENT, linewidth=1.8)
    ax1.plot([map_theta], [post.pdf(map_theta)], marker="o", color=ACCENT,
             markersize=9, markeredgecolor="white", markeredgewidth=1.2,
             zorder=5)
    ax1.text(map_theta, post.pdf(map_theta) * 1.04, f"  {map_theta:.2f}",
             fontsize=11, color=ACCENT, weight="bold", ha="left")
    ax1.set_xlabel("theta", fontsize=9)
    ax1.set_ylabel("density", fontsize=9)
    ax1.set_title("in theta", fontsize=10, color=INK, weight="bold")
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, post.pdf(map_theta) * 1.22)

    ph = np.linspace(-6, 4, 800)
    t_of_p = 1 / (1 + np.exp(-ph))
    dp = post.pdf(t_of_p) * t_of_p * (1 - t_of_p)
    map_phi = np.log(map_phi_back / (1 - map_phi_back))
    ax2.plot(ph, dp, color=INK, linewidth=2.0)
    ax2.axvline(map_phi, color=ACCENT, linewidth=1.8)
    peak = post.pdf(map_phi_back) * map_phi_back * (1 - map_phi_back)
    ax2.plot([map_phi], [peak], marker="o", color=ACCENT, markersize=9,
             markeredgecolor="white", markeredgewidth=1.2, zorder=5)
    ax2.text(map_phi, peak * 1.04,
             f"  {map_phi:.2f}   ->  theta = {map_phi_back:.2f}",
             fontsize=10.5, color=ACCENT, weight="bold", ha="left")
    ax2.set_xlabel("log odds  log(theta / (1 - theta))", fontsize=9)
    ax2.set_ylabel("density", fontsize=9)
    ax2.set_title("the same posterior, in log odds", fontsize=10, color=INK,
                  weight="bold")
    ax2.set_xlim(-6, 4)
    ax2.set_ylim(0, peak * 1.22)

    for ax in (ax1, ax2):
        ax.grid(alpha=0.12)
        ax.tick_params(labelsize=8, colors=INK)
        for s in ax.spines.values():
            s.set_color(INK)
            s.set_linewidth(0.7)

    fig.text(0.5, -0.03,
             f"one posterior, two coordinates, two most-probable values: "
             f"{map_theta:.2f} and {map_phi_back:.2f}.  "
             f"nothing was approximated.",
             ha="center", fontsize=8.8, style="italic", color=INK)
    fig.tight_layout()
    fig.savefig(HERE / "map_not_invariant.svg", transparent=True,
                bbox_inches="tight")
    plt.close(fig)
    print(f"wrote map_not_invariant.svg   theta MAP {map_theta:.4f}   "
          f"logit MAP back {map_phi_back:.4f}   mean {post.mean():.4f}")


# === 그림으로 보는 AI: 두 구간은 서로의 시험을 통과하지 못한다 ================
#
# 신용구간은 "자료를 고정했을 때 그 안에 있을 확률 0.95" 를 지키고,
# 신뢰구간은 "theta 가 무엇이든 포함률이 0.95 아래로 안 떨어진다" 를 지킨다.
# 각자 제 시험은 통과하고 상대의 시험은 떨어진다. 수는 모두 여기서 셈한다.

def two_intervals():
    n = 10

    def cred(x):
        po = stats.beta(x + 1, n - x + 1)
        return po.ppf(0.025), po.ppf(0.975)

    def cp(x):
        lo = 0.0 if x == 0 else stats.beta(x, n - x + 1).ppf(0.025)
        hi = 1.0 if x == n else stats.beta(x + 1, n - x).ppf(0.975)
        return lo, hi

    grid = np.linspace(0.01, 0.99, 99)

    def coverage(f):
        out = []
        for t in grid:
            pm = stats.binom(n, t).pmf(np.arange(n + 1))
            out.append(sum(pm[x] for x in range(n + 1)
                           if f(x)[0] <= t <= f(x)[1]))
        return np.array(out)

    cov_b, cov_f = coverage(cred), coverage(cp)

    # 그림이 제 주장을 확인한다 — 하나는 0.95 아래로 떨어지고 하나는 안 떨어진다
    assert cov_b.min() < 0.95, "신용구간은 포함률이 떨어져야 이야기가 선다"
    assert cov_f.min() >= 0.95, "Clopper-Pearson 은 0.95 아래로 안 떨어진다"

    x = 2
    po = stats.beta(x + 1, n - x + 1)
    ib, ifz = cred(x), cp(x)
    pb = po.cdf(ib[1]) - po.cdf(ib[0])
    pf = po.cdf(ifz[1]) - po.cdf(ifz[0])
    assert abs(pb - 0.95) < 1e-9, "신용구간의 사후확률은 정확히 0.95"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.8, 4.1),
                                   gridspec_kw={"width_ratios": [1.15, 1]})

    # --- 왼쪽: theta 를 훑으며 포함률 ---------------------------------------
    ax1.axhline(0.95, color=INK, linestyle=":", linewidth=1.3)
    ax1.plot(grid, cov_b, color=ACCENT, linewidth=1.9,
             label=f"credible, uniform prior   min {cov_b.min():.3f}")
    ax1.plot(grid, cov_f, color=MUTED, linewidth=1.9, linestyle="--",
             label=f"Clopper-Pearson   min {cov_f.min():.3f}")
    ax1.set_xlabel("the true theta", fontsize=9)
    ax1.set_ylabel("frequentist coverage", fontsize=9)
    ax1.set_title("Neyman's test: sweep theta", fontsize=10, color=INK,
                  weight="bold")
    ax1.set_ylim(0.78, 1.01)
    ax1.set_yticks([0.80, 0.85, 0.90, 0.95, 1.00])   # 약속한 0.95 를 눈금으로
    ax1.legend(fontsize=8, loc="lower right", framealpha=0.92)
    ax1.grid(alpha=0.12)

    # --- 오른쪽: 자료를 고정했을 때 ----------------------------------------
    xs = np.linspace(0, 1, 800)
    ax2.plot(xs, po.pdf(xs), color=INK, linewidth=1.8)
    for k, (lab, iv, pp, col) in enumerate(
            (("credible", ib, pb, ACCENT),
             ("Clopper-Pearson", ifz, pf, MUTED))):
        y = po.pdf(po.mean()) * (0.30 - 0.11 * k)
        ax2.plot(iv, [y, y], color=col, linewidth=5, solid_capstyle="butt",
                 alpha=0.85)
        ax2.text(iv[1] + 0.02, y, f"  {lab}\n  posterior prob {pp:.4f}",
                 va="center", fontsize=8.2, color=col)
    ax2.set_xlabel("theta", fontsize=9)
    ax2.set_ylabel("posterior density", fontsize=9)
    ax2.set_title(f"the Bayesian test: fix the data  ({x} of {n})",
                  fontsize=10, color=INK, weight="bold")
    ax2.set_xlim(0, 1)
    ax2.grid(alpha=0.12)

    for ax in (ax1, ax2):
        ax.tick_params(labelsize=8, colors=INK)
        for s in ax.spines.values():
            s.set_color(INK)
            s.set_linewidth(0.7)

    fig.text(0.5, -0.03,
             "each one passes its own test and fails the other's. "
             "they are answers to different questions.",
             ha="center", fontsize=8.8, style="italic", color=INK)
    fig.tight_layout()
    fig.savefig(HERE / "two_intervals.svg", transparent=True,
                bbox_inches="tight")
    plt.close(fig)
    print(f"wrote two_intervals.svg   cov min {cov_b.min():.3f} / "
          f"{cov_f.min():.3f}   post {pb:.4f} / {pf:.4f}")


if __name__ == "__main__":
    bayes_table()
    conjugate_limit()
    map_not_invariant()
    two_intervals()
    thompson_1933()
    james_stein()
    efron_morris()


# === 그림으로 보는 AI: 1933년의 동기를 그대로 재어 본다 =======================
#
# Thompson 1933 은 임상시험을 두고 쓴 글이다 — 나쁜 치료를 받는 환자 수를
# 줄이자는 것. 그 동기를 그대로 재면, 반반 배정 500명이 62명 언저리로 준다.
# 다만 **흩어짐이 크다**(8-411). 평균의 이야기이지 한 번의 보장이 아니다.

def thompson_1933():
    rng = np.random.default_rng(0)
    pA, pB, N, R = 0.30, 0.45, 1000, 600

    def run():
        a, b, bad = np.ones(2), np.ones(2), 0
        for _ in range(N):
            k = int(np.argmax(rng.beta(a, b)))
            r = rng.random() < (pA if k == 0 else pB)
            a[k] += r
            b[k] += 1 - r
            bad += (k == 0)
        return bad

    bad = np.array([run() for _ in range(R)])
    fixed = N // 2

    assert bad.mean() < fixed * 0.25, "톰프슨이 훨씬 적게 배정해야 한다"
    assert bad.max() > fixed * 0.5, "그래도 나쁜 판이 나온다는 것을 보여야 한다"

    fig, ax = plt.subplots(figsize=(8.6, 4.3))
    ax.hist(bad, bins=46, color=ACCENT, alpha=0.75, edgecolor="white",
            linewidth=0.5)
    ax.axvline(fixed, color=INK, linewidth=2.0)
    ax.text(fixed, ax.get_ylim()[1] * 0.93, "  fixed 50/50 split", fontsize=9,
            color=INK, va="top")
    ax.axvline(bad.mean(), color=MUTED, linewidth=2.0, linestyle="--")
    ax.text(bad.mean(), ax.get_ylim()[1] * 0.70,
            f"  Thompson mean {bad.mean():.0f}", fontsize=9, color=MUTED,
            va="top")
    ax.annotate(f"worst of {R} runs: {bad.max()}",
                xy=(bad.max(), 2), xytext=(-150, 46),
                textcoords="offset points", fontsize=8.4, color=INK,
                arrowprops=dict(arrowstyle="->", color=INK, lw=0.9))
    ax.set_xlabel(f"patients given the worse treatment, out of {N}",
                  fontsize=9.5)
    ax.set_ylabel("runs", fontsize=9.5)
    ax.set_title("what Thompson was trying to reduce", fontsize=10.5,
                 color=INK, weight="bold", pad=8)
    ax.grid(alpha=0.12)
    ax.tick_params(labelsize=8, colors=INK)
    for s in ax.spines.values():
        s.set_color(INK)
        s.set_linewidth(0.7)
    fig.text(0.5, -0.04,
             f"success rates {pA} and {pB}, {R} simulated trials. "
             f"the spread is the point: {bad.min()} to {bad.max()}.",
             ha="center", fontsize=8.8, style="italic", color=INK)
    fig.tight_layout()
    fig.savefig(HERE / "thompson_1933.svg", transparent=True,
                bbox_inches="tight")
    plt.close(fig)
    print(f"wrote thompson_1933.svg   mean {bad.mean():.1f}  "
          f"range {bad.min()}-{bad.max()}")


# === 그림으로 보는 AI: 차원 3 에서 갑자기 생기는 경계 =========================
#
# Stein 1956 / James-Stein 1961. k >= 3 이면 표본값(최대우도)이 비허용적이다.
# k = 1, 2 에서는 수축 인자가 0 이라 두 추정량이 **완전히 같고**, k = 3 부터
# 갈라진다. 경계가 눈에 보이도록 두 곡선을 함께 그린다.

def james_stein():
    rng = np.random.default_rng(0)
    R = 20000
    ks = [1, 2, 3, 4, 6, 10, 20, 50]
    e_mle, e_js, win = [], [], []
    for k in ks:
        mu = np.full((R, k), 2.0 / np.sqrt(k))
        x = mu + rng.normal(0, 1, size=(R, k))
        em = ((x - mu) ** 2).sum(1)
        if k > 2:
            s2 = (x ** 2).sum(1)
            sh = np.maximum(0.0, 1 - (k - 2) / np.where(s2 > 0, s2, 1e-12))
            js = x * sh[:, None]
        else:
            js = x
        ej = ((js - mu) ** 2).sum(1)
        e_mle.append(em.mean()); e_js.append(ej.mean())
        win.append(float(np.mean(ej < em)))

    # 그림이 제 주장을 확인한다 — 경계가 정확히 k = 3 에 있다
    assert abs(e_js[0] - e_mle[0]) < 1e-12, "k=1 에서는 완전히 같아야 한다"
    assert abs(e_js[1] - e_mle[1]) < 1e-12, "k=2 에서도 완전히 같아야 한다"
    assert e_js[2] < e_mle[2] * 0.95, "k=3 부터 갈라져야 한다"
    assert e_js[-1] < e_mle[-1] * 0.2, "큰 k 에서는 크게 벌어져야 한다"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.6, 4.1))

    ax1.plot(ks, e_mle, "s--", color=INK, linewidth=2.0, ms=6,
             label="the sample mean")
    ax1.plot(ks, e_js, "o-", color=ACCENT, linewidth=2.0, ms=6,
             label="James-Stein")
    ax1.axvline(3, color=MUTED, linewidth=1.3, linestyle=":")
    ax1.text(3, max(e_mle) * 0.93, " k = 3", fontsize=9, color=MUTED)
    ax1.set_xlabel("number of means estimated at once", fontsize=9.5)
    ax1.set_ylabel("total squared error", fontsize=9.5)
    ax1.set_title("one estimator stops growing", fontsize=10, color=INK,
                  weight="bold")
    ax1.legend(fontsize=8.6, loc="upper left", framealpha=0.92)

    ax2.plot(ks, win, "o-", color=ACCENT, linewidth=2.0, ms=6)
    ax2.axhline(0.5, color=INK, linestyle=":", linewidth=1.1)
    ax2.axvline(3, color=MUTED, linewidth=1.3, linestyle=":")
    ax2.set_xlabel("number of means estimated at once", fontsize=9.5)
    ax2.set_ylabel("runs where James-Stein wins", fontsize=9.5)
    ax2.set_title("nothing at all below 3", fontsize=10, color=INK,
                  weight="bold")
    ax2.set_ylim(-0.04, 1.06)
    ax2.annotate(f"{win[2]:.3f}", xy=(3, win[2]), xytext=(10, -16),
                 textcoords="offset points", fontsize=8.6, color=ACCENT)

    for ax in (ax1, ax2):
        ax.grid(alpha=0.12)
        ax.tick_params(labelsize=8, colors=INK)
        for s in ax.spines.values():
            s.set_color(INK)
            s.set_linewidth(0.7)

    fig.text(0.5, -0.04,
             f"{R:,} repetitions per point. at k = 1 and 2 the two estimators "
             f"are identical, not merely close.",
             ha="center", fontsize=8.8, style="italic", color=INK)
    fig.tight_layout()
    fig.savefig(HERE / "james_stein.svg", transparent=True,
                bbox_inches="tight")
    plt.close(fig)
    print(f"wrote james_stein.svg   k=3 {e_mle[2]:.3f}->{e_js[2]:.3f}  "
          f"k=50 {e_mle[-1]:.2f}->{e_js[-1]:.2f}")


# === 그림으로 보는 AI: 그 유명한 야구 결과는 한 번의 추출이다 =================
#
# Efron-Morris 1975 는 1970년 시즌 첫 45타석, 선수 18명으로 수축이 16/18 에서
# 더 가깝고 총 제곱오차가 원값의 1/3 미만이라고 보고했다. 같은 **설계**를
# 되풀이해 시늉하면 전형적인 판은 그보다 덜 극적이고, 더러 지기도 한다.
# 원 자료를 구하지 못했으므로 그림은 시늉이며, 그 사실을 그림 안에 적는다.

def efron_morris():
    rng = np.random.default_rng(1970)
    P, AB, R, GRAND = 18, 45, 4000, 0.265
    ratio, closer = [], []
    for _ in range(R):
        theta = rng.beta(GRAND * 90, (1 - GRAND) * 90, P)
        y = rng.binomial(AB, theta) / AB
        m, = (y.mean(),)
        sh = max(0.0, 1 - (P - 3) * (m * (1 - m) / AB) / ((y - m) ** 2).sum())
        js = m + sh * (y - m)
        ratio.append(((js - theta) ** 2).sum() / ((y - theta) ** 2).sum())
        closer.append(int((np.abs(js - theta) < np.abs(y - theta)).sum()))
    ratio = np.array(ratio)
    closer = np.array(closer)

    lose = float(np.mean(ratio > 1))
    assert ratio.mean() < 0.7, "수축이 평균적으로는 뚜렷이 나아야 한다"
    assert lose > 0.0, "지는 판이 있다는 것이 이 그림의 요점이다"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.6, 4.1))

    ax1.hist(ratio, bins=60, color=ACCENT, alpha=0.75, edgecolor="white",
             linewidth=0.4)
    ax1.axvline(1.0, color=INK, linewidth=1.8)
    ax1.text(1.0, ax1.get_ylim()[1] * 0.95, "  shrinkage loses", fontsize=8.6,
             color=INK, va="top")
    ax1.axvline(1 / 3, color=MUTED, linewidth=1.8, linestyle="--")
    ax1.text(1 / 3, ax1.get_ylim()[1] * 0.72,
             " the 1975 paper\n reported below this", fontsize=8.2,
             color=MUTED, va="top")
    ax1.set_xlabel("total squared error, shrunk / raw", fontsize=9.5)
    ax1.set_ylabel("simulated seasons", fontsize=9.5)
    ax1.set_title(f"median {np.median(ratio):.3f}", fontsize=10, color=INK,
                  weight="bold")

    ax2.hist(closer, bins=np.arange(-0.5, P + 1.5), color=ACCENT, alpha=0.75,
             edgecolor="white", linewidth=0.5)
    ax2.axvline(16, color=MUTED, linewidth=1.8, linestyle="--")
    ax2.text(16, ax2.get_ylim()[1] * 0.95, " 16 of 18\n in 1970",
             fontsize=8.4, color=MUTED, va="top", ha="right")
    ax2.axvline(closer.mean(), color=INK, linewidth=1.8)
    ax2.text(closer.mean(), ax2.get_ylim()[1] * 0.60,
             f" typical {closer.mean():.1f}", fontsize=8.6, color=INK,
             va="top")
    ax2.set_xlabel(f"players the shrunk estimate got closer, of {P}",
                   fontsize=9.5)
    ax2.set_ylabel("simulated seasons", fontsize=9.5)
    ax2.set_title("one season is one draw", fontsize=10, color=INK,
                  weight="bold")

    for ax in (ax1, ax2):
        ax.grid(alpha=0.12)
        ax.tick_params(labelsize=8, colors=INK)
        for s in ax.spines.values():
            s.set_color(INK)
            s.set_linewidth(0.7)

    fig.text(0.5, -0.04,
             f"simulation of the 1975 design, not the original data. "
             f"shrinkage loses outright in {lose*100:.1f}% of seasons.",
             ha="center", fontsize=8.8, style="italic", color=INK)
    fig.tight_layout()
    fig.savefig(HERE / "efron_morris.svg", transparent=True,
                bbox_inches="tight")
    plt.close(fig)
    print(f"wrote efron_morris.svg   ratio med {np.median(ratio):.3f}  "
          f"closer {closer.mean():.1f}/18  loses {lose*100:.1f}%")
