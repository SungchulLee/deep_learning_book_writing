"""2장 고전 모델 쪽들의 그림.

규칙은 CLAUDE.md 의 Figures 항에 적어 둔 대로다.
- SVG, fonttype='path', transparent
- 글자는 모두 아스키 (matplotlib 기본 폰트에 한글이 없다)

수는 모두 여기서 실제로 돌려 얻는다.

실행: python docs/ch02/models/figures/make_figures.py
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
BLUE = "#2d6a8f"


def style(ax):
    ax.tick_params(labelsize=8, colors=INK)
    for s in ax.spines.values():
        s.set_color(INK)
        s.set_linewidth(0.7)


# === 그림으로 보는 AI: Cover-Hart 1967 의 복도 ===============================
#
# R* <= R <= 2R*. 베이즈 오차를 아는 문제를 만들어 1-NN 이 어디에 앉는지 잰다.
# 두 정규분포, 공분산 단위행렬, 평균 (+-delta/2, 0) 이므로 R* = Phi(-delta/2).

def cover_hart_corridor():
    from scipy.stats import norm
    from sklearn.neighbors import KNeighborsClassifier

    delta = 2.0
    r_star = float(norm.cdf(-delta / 2))

    rng = np.random.default_rng(0)
    M = 400_000
    lab = rng.integers(0, 2, M)
    x = rng.standard_normal((M, 2)) + np.where(
        lab[:, None] == 1, [delta / 2, 0], [-delta / 2, 0])
    eta = 1 / (1 + np.exp(-delta * x[:, 0]))
    r_inf = float((2 * eta * (1 - eta)).mean())

    lte = rng.integers(0, 2, 20000)
    Xte = rng.standard_normal((20000, 2)) + np.where(
        lte[:, None] == 1, [delta / 2, 0], [-delta / 2, 0])

    ns = [50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
    mean, sd = [], []
    for n in ns:
        e = []
        for rep in range(5):
            r = np.random.default_rng(1000 + rep)
            ltr = r.integers(0, 2, n)
            Xtr = r.standard_normal((n, 2)) + np.where(
                ltr[:, None] == 1, [delta / 2, 0], [-delta / 2, 0])
            kn = KNeighborsClassifier(n_neighbors=1).fit(Xtr, ltr)
            e.append(1 - kn.score(Xte, lte))
        mean.append(np.mean(e))
        sd.append(np.std(e))
    mean, sd = np.array(mean), np.array(sd)

    fig, ax = plt.subplots(figsize=(7.8, 4.5))

    ax.fill_between([ns[0] * 0.75, ns[-1] * 1.35], r_star, 2 * r_star,
                    color=ACCENT, alpha=0.10)
    ax.axhline(r_star, color=BLUE, linewidth=1.7, linestyle="--")
    ax.axhline(2 * r_star, color=ACCENT, linewidth=1.7, linestyle="--")
    ax.axhline(r_inf, color=INK, linewidth=1.2, linestyle=":")

    ax.errorbar(ns, mean, yerr=sd, fmt="o-", color=INK, linewidth=1.8,
                ms=5.5, capsize=3, label="1-NN, measured")

    ax.text(ns[-1] * 1.4, r_star, f"  Bayes error R*\n  {r_star:.4f}",
            fontsize=8.5, color=BLUE, va="center")
    ax.text(ns[-1] * 1.4, 2 * r_star, f"  the ceiling 2R*\n  {2*r_star:.4f}",
            fontsize=8.5, color=ACCENT, va="center")
    ax.text(ns[-1] * 1.4, r_inf, f"  predicted limit\n  {r_inf:.4f}",
            fontsize=8.5, color=INK, va="center")

    ax.annotate("already there by n = 100",
                xy=(100, mean[1]), xytext=(260, 0.175), fontsize=8.5,
                color=INK,
                arrowprops=dict(arrowstyle="->", color=INK, lw=0.9))

    ax.set_xscale("log")
    ax.set_xlim(ns[0] * 0.75, ns[-1] * 1.35)
    ax.set_ylim(0.13, 0.345)
    ax.set_xlabel("training points", fontsize=9.5)
    ax.set_ylabel("test error", fontsize=9.5)
    ax.set_title("the corridor Cover and Hart proved in 1967",
                 fontsize=10.5, color=INK, weight="bold", pad=8)
    ax.grid(alpha=0.13, which="both")
    style(ax)

    fig.text(0.5, -0.04,
             "two Gaussians, unit covariance, means 2.0 apart - so the Bayes "
             "error is known exactly, not estimated.",
             ha="center", fontsize=8.5, style="italic", color=INK)
    fig.tight_layout()
    fig.savefig(HERE / "cover_hart.svg", transparent=True,
                bbox_inches="tight")
    plt.close(fig)
    print(f"wrote cover_hart.svg   R* {r_star:.4f}  R_inf {r_inf:.4f}  "
          f"measured {mean[-1]:.4f}  ceiling {2*r_star:.4f}")


# === 그림으로 보는 AI: 버려도 되는 점과 버리면 안 되는 점 =====================
#
# 1992년 초록의 말로 "supporting patterns" 만 있으면 해가 그대로다. 이 쪽의
# 자료로 확인한다. 서포트 벡터만 남기면 결정함수가 거의 그대로이고, 나머지만
# 남기면 크게 움직인다.
#
# 주의: gamma="scale" 은 넣어 준 자료의 분산으로 다시 셈한다. 부분집합으로
# 다시 학습하면 모형 자체가 바뀌어 버려 비교가 성립하지 않는다. 그래서
# gamma 를 전체 학습 집합에서 한 번 셈해 못박고 쓴다.

def svm_what_matters():
    from sklearn.svm import SVC
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    X, y = make_classification(n_samples=500, n_features=10, random_state=42)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                              random_state=42)
    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr)
    X_te_s = sc.transform(X_te)

    g = 1.0 / (X_tr_s.shape[1] * X_tr_s.var())      # 못박은 gamma
    full = SVC(kernel="rbf", C=1.0, gamma=g).fit(X_tr_s, y_tr)
    sv = full.support_
    non = np.setdiff1d(np.arange(len(X_tr_s)), sv)

    only = SVC(kernel="rbf", C=1.0, gamma=g).fit(X_tr_s[sv], y_tr[sv])
    alt = SVC(kernel="rbf", C=1.0, gamma=g).fit(X_tr_s[non], y_tr[non])

    d0 = full.decision_function(X_te_s)
    d1 = only.decision_function(X_te_s)
    d2 = alt.decision_function(X_te_s)
    e1, e2 = np.abs(d0 - d1).max(), np.abs(d0 - d2).max()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 4.2))

    # --- 왼쪽: 무엇을 버렸는가 --------------------------------------------
    ax1.barh([1], [len(sv)], color=ACCENT, height=0.5,
             label=f"supporting patterns ({len(sv)})")
    ax1.barh([1], [len(non)], left=[len(sv)], color="#cfc8bd", height=0.5,
             label=f"the rest ({len(non)})")
    ax1.set_yticks([])
    ax1.set_xlim(0, len(X_tr_s))
    ax1.set_xlabel("training points", fontsize=9.5)
    ax1.set_title(f"{len(non)} of {len(X_tr_s)} can go",
                  fontsize=10.5, color=INK, weight="bold")
    ax1.legend(fontsize=8.5, loc="lower center", framealpha=0.9)
    ax1.text(len(sv) / 2, 1, f"{len(sv)/len(X_tr_s)*100:.0f}%", ha="center",
             va="center", fontsize=11, color="white", weight="bold")
    ax1.set_ylim(0.5, 1.5)
    style(ax1)

    # --- 오른쪽: 경계가 움직였는가 ----------------------------------------
    lim = [min(d0.min(), d2.min()) - 0.2, max(d0.max(), d2.max()) + 0.2]
    ax2.plot(lim, lim, color=INK, linewidth=0.9, linestyle=":", zorder=1)
    ax2.scatter(d0, d2, s=26, color="#8c8880", alpha=0.75, zorder=2,
                label=f"keep only the rest   (max gap {e2:.2f})")
    ax2.scatter(d0, d1, s=26, color=ACCENT, zorder=3,
                label=f"keep only the {len(sv)}   (max gap {e1:.5f})")
    ax2.set_xlabel("decision value, trained on all 400", fontsize=9.5)
    ax2.set_ylabel("decision value, retrained on a subset", fontsize=9.5)
    ax2.set_title("one subset keeps the boundary, one does not",
                  fontsize=10.5, color=INK, weight="bold")
    ax2.legend(fontsize=8, loc="upper left", framealpha=0.92)
    ax2.set_xlim(lim)
    ax2.set_ylim(lim)
    ax2.grid(alpha=0.13)
    style(ax2)

    fig.text(0.5, -0.04,
             "gamma is pinned to the full training set - sklearn's "
             "gamma='scale' would otherwise be recomputed and change the model.",
             ha="center", fontsize=8.2, style="italic", color=INK)
    fig.tight_layout()
    fig.savefig(HERE / "svm_supporting.svg", transparent=True,
                bbox_inches="tight")
    plt.close(fig)
    print(f"wrote svm_supporting.svg   sv {len(sv)}/{len(X_tr_s)}   "
          f"gap sv-only {e1:.3e}   rest-only {e2:.3e}")


# === 그림으로 보는 AI: 따로 자란 두 갈래가 고른 두 불순도 ====================
#
# CART(Breiman 외 1984)는 지니를, Quinlan 의 ID3/C4.5 는 엔트로피를 골랐다.
# 따로 한 선택이 결과를 가르는지 이 쪽의 자료로 잰다. 가르지 않는다 —
# 두 측도의 차이가 폴드 사이의 흔들림보다 작다.

def gini_vs_entropy():
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split, cross_val_score

    X, y = make_classification(n_samples=500, n_features=10, n_informative=5,
                               random_state=42)
    X_tr, _, y_tr, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    depths = list(range(1, 13))
    gm, gs, em, es = [], [], [], []
    for d in depths:
        g = cross_val_score(DecisionTreeClassifier(
            max_depth=d, criterion="gini", random_state=42), X_tr, y_tr, cv=5)
        e = cross_val_score(DecisionTreeClassifier(
            max_depth=d, criterion="entropy", random_state=42),
            X_tr, y_tr, cv=5)
        gm.append(g.mean()); gs.append(g.std())
        em.append(e.mean()); es.append(e.std())
    gm, gs, em, es = map(np.array, (gm, gs, em, es))

    ga, ea = [], []
    for s in range(30):
        Xs, ys = make_classification(n_samples=500, n_features=10,
                                     n_informative=5, random_state=s)
        ga.append(cross_val_score(DecisionTreeClassifier(
            max_depth=5, criterion="gini", random_state=0), Xs, ys, cv=5).mean())
        ea.append(cross_val_score(DecisionTreeClassifier(
            max_depth=5, criterion="entropy", random_state=0), Xs, ys, cv=5).mean())
    ga, ea = np.array(ga), np.array(ea)
    wins_g = int((ga > ea).sum()); wins_e = int((ea > ga).sum())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 4.2))

    # --- 왼쪽: 깊이마다 재도 띠가 겹친다 ----------------------------------
    ax1.fill_between(depths, gm - gs, gm + gs, color=ACCENT, alpha=0.16)
    ax1.fill_between(depths, em - es, em + es, color=BLUE, alpha=0.16)
    ax1.plot(depths, gm, "o-", color=ACCENT, linewidth=1.9, ms=4.5,
             label="Gini  (CART, 1984)")
    ax1.plot(depths, em, "s--", color=BLUE, linewidth=1.9, ms=4.5,
             label="entropy  (ID3 / C4.5)")
    ax1.set_xlabel("max_depth", fontsize=9.5)
    ax1.set_ylabel("5-fold CV accuracy", fontsize=9.5)
    ax1.set_title("bands are the fold-to-fold spread", fontsize=10.5,
                  color=INK, weight="bold")
    ax1.legend(fontsize=8.5, loc="lower right", framealpha=0.92)
    ax1.grid(alpha=0.13)
    style(ax1)

    # --- 오른쪽: 문제를 바꿔 가며 맞붙여도 ---------------------------------
    lo = min(ga.min(), ea.min()) - 0.02
    hi = max(ga.max(), ea.max()) + 0.02
    ax2.plot([lo, hi], [lo, hi], color=INK, linewidth=1.0, linestyle=":")
    ax2.scatter(ga, ea, s=42, color=ACCENT, alpha=0.8, edgecolors="white",
                linewidth=0.7, zorder=3)
    ax2.set_xlim(lo, hi); ax2.set_ylim(lo, hi)
    ax2.set_xlabel("Gini accuracy", fontsize=9.5)
    ax2.set_ylabel("entropy accuracy", fontsize=9.5)
    ax2.set_title(f"30 different problems: {wins_g} - {wins_e}",
                  fontsize=10.5, color=INK, weight="bold")
    ax2.text(0.04, 0.93, "above the line = entropy won", fontsize=8,
             color=INK, alpha=0.75, transform=ax2.transAxes)
    ax2.text(0.37, 0.05, "below = Gini won", fontsize=8, color=INK,
             alpha=0.75, transform=ax2.transAxes)
    ax2.grid(alpha=0.13)
    style(ax2)

    fig.text(0.5, -0.04,
             "the gap between the two criteria is smaller than the spread "
             "within either one. p = 0.25 for the head-to-head.",
             ha="center", fontsize=8.5, style="italic", color=INK)
    fig.tight_layout()
    fig.savefig(HERE / "gini_vs_entropy.svg", transparent=True,
                bbox_inches="tight")
    plt.close(fig)
    print(f"wrote gini_vs_entropy.svg   {wins_g}-{wins_e}   "
          f"max gap {np.abs(gm-em).max():.4f}   typical spread {gs.mean():.4f}")


# === 그림으로 보는 AI: 가우스가 고른 가정이 무엇을 사 주는가 ==================
#
# 최소제곱이 좋은 것은 오차가 정규분포일 때다. 그 가정을 조금 깨면 어떻게 되는지
# 잰다. 참 기울기 2.0 을 알고 있으니 오차를 정확히 잴 수 있다.

def least_squares_assumption():
    from scipy.optimize import minimize

    TRUE_A, TRUE_B = 2.0, 1.0
    n, REP = 60, 300

    def _ls(x, y):
        return np.linalg.lstsq(np.c_[x, np.ones_like(x)], y, rcond=None)[0]

    def _lad(x, y):
        X = np.c_[x, np.ones_like(x)]
        return minimize(lambda p: np.abs(y - X @ p).sum(), _ls(x, y),
                        method="Nelder-Mead",
                        options=dict(xatol=1e-8, fatol=1e-8,
                                     maxiter=5000)).x

    def run(kind):
        a, b = [], []
        for r in range(REP):
            rng = np.random.default_rng(r)
            x = rng.uniform(-3, 3, n)
            noise = rng.standard_normal(n)
            if kind == "contam":
                k = max(1, int(0.05 * n))
                idx = rng.choice(n, k, replace=False)
                noise[idx] += rng.choice([-1, 1], k) * rng.uniform(12, 25, k)
            y = TRUE_A * x + TRUE_B + noise
            a.append(abs(_ls(x, y)[0] - TRUE_A))
            b.append(abs(_lad(x, y)[0] - TRUE_A))
        return np.array(a), np.array(b)

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2), sharey=True)
    titles = ["errors exactly Gaussian",
              "5% of points are outliers"]
    for ax, kind, title in zip(axes, ("gauss", "contam"), titles):
        a, b = run(kind)
        bp = ax.boxplot([a, b], labels=["least\nsquares",
                                             "least absolute\ndeviations"],
                        widths=0.45, patch_artist=True, showfliers=False)
        for patch, col in zip(bp["boxes"], (ACCENT, BLUE)):
            patch.set_facecolor(col)
            patch.set_alpha(0.45)
        for med in bp["medians"]:
            med.set_color(INK)
            med.set_linewidth(1.6)
        ax.set_title(title, fontsize=10.5, color=INK, weight="bold")
        ax.annotate(f"{np.median(a):.4f}", xy=(1, np.median(a)),
                    xytext=(16, 7), textcoords="offset points",
                    fontsize=8.5, color=ACCENT, weight="bold")
        ax.annotate(f"{np.median(b):.4f}", xy=(2, np.median(b)),
                    xytext=(16, 7), textcoords="offset points",
                    fontsize=8.5, color=BLUE, weight="bold")
        win = (a < b).mean() * 100
        ax.set_xlabel(f"least squares wins {win:.0f}% of runs", fontsize=9)
        ax.grid(alpha=0.13, axis="y")
        style(ax)
    axes[0].set_ylabel("error in the fitted slope", fontsize=9.5)
    axes[0].set_ylim(0, 0.45)

    fig.text(0.5, -0.05,
             "same true line (slope 2.0), same 60 points, 300 runs. only the "
             "shape of the noise changes.",
             ha="center", fontsize=8.5, style="italic", color=INK)
    fig.tight_layout()
    fig.savefig(HERE / "ls_assumption.svg", transparent=True,
                bbox_inches="tight")
    plt.close(fig)
    print("wrote ls_assumption.svg")


if __name__ == "__main__":
    cover_hart_corridor()
    svm_what_matters()
    gini_vs_entropy()
    least_squares_assumption()
