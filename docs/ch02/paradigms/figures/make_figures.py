"""2장 학습 패러다임 쪽들의 그림.

규칙은 CLAUDE.md 의 Figures 항에 적어 둔 대로다.
- SVG, fonttype='path', transparent
- 글자는 모두 아스키 (matplotlib 기본 폰트에 한글이 없다)

실행: python docs/ch02/paradigms/figures/make_figures.py
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


# === 그림으로 보는 AI: 널리 전해지는 이야기와 날짜 ============================
#
# 벨먼은 자서전에서 국방장관 윌슨의 눈을 피하려 "동적 계획법"이라는 이름을
# 골랐다고 적었다. 그런데 그 이름을 쓴 논문이 윌슨의 취임보다 먼저 나왔다.
# 날짜만 나란히 놓는다 — 어느 쪽이 옳은지 판정하지 않는다.

def bellman_dates():
    fig, ax = plt.subplots(figsize=(9.2, 2.9))

    ax.annotate("", xy=(10.0, 1.0), xytext=(0.4, 1.0),
                arrowprops=dict(arrowstyle="->", color=INK, linewidth=1.3))

    marks = [
        (2.4, "15 August 1952", ACCENT, 1,
         "Bellman, PNAS 38(8) 716-719\n"
         '"On the Theory of\nDynamic Programming"'),
        (7.0, "28 January 1953", BLUE, -1,
         "Charles E. Wilson becomes\nSecretary of Defense\n"
         "(serves to October 1957)"),
    ]
    for x, when, col, side, what in marks:
        ax.plot([x], [1.0], marker="o", color=col, markersize=10,
                markeredgecolor="white", markeredgewidth=1.6, zorder=5)
        ax.text(x, 1.0 + side * 0.17, when, ha="center", fontsize=9.5,
                weight="bold", color=col,
                va="bottom" if side > 0 else "top")
        ax.text(x, 1.0 + side * 0.40, what, ha="center", fontsize=8,
                color=INK, alpha=0.9,
                va="bottom" if side > 0 else "top")

    ax.annotate("", xy=(7.0, 0.72), xytext=(2.4, 0.72),
                arrowprops=dict(arrowstyle="<->", color=INK, lw=1.0,
                                alpha=0.6))
    ax.text(4.7, 0.59, "the name is already in print, five months early",
            ha="center", fontsize=8.5, color=INK, style="italic")

    ax.set_xlim(0, 10.4)
    ax.set_ylim(0.1, 1.95)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(HERE / "bellman_dates.svg", transparent=True,
                bbox_inches="tight")
    plt.close(fig)
    print("wrote bellman_dates.svg")


# === 그림으로 보는 AI: 딜레마는 용량 대 자료의 이야기다 ======================
#
# 정리 3 의 분해를 수로 확인하고, 같은 분해를 표본 수만 바꿔 다시 그린다.
# n=20 에서는 차수를 올리면 분산이 폭발하고, n=200 에서는 거의 공짜가 된다.

def bias_variance_depends_on_n():
    SIG = 0.30
    f = lambda t: np.sin(2 * np.pi * t)
    x0 = 0.35
    degs = [1, 2, 3, 5, 7, 9]

    def decompose(n, reps=600):
        b2, va = [], []
        for d in degs:
            preds = []
            for r in range(reps):
                rng = np.random.default_rng(10_000 * d + r)
                x = rng.uniform(0, 1, n)
                y = f(x) + rng.normal(0, SIG, n)
                preds.append(np.polyval(np.polyfit(x, y, d), x0))
            preds = np.array(preds)
            b2.append((preds.mean() - f(x0)) ** 2)
            va.append(preds.var())
        return np.array(b2), np.array(va)

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.2))
    store = {}
    for ax, n in zip(axes, (20, 200)):
        b2, va = decompose(n)
        store[n] = (b2, va)
        total = b2 + va + SIG ** 2
        xs = np.arange(len(degs))
        ax.bar(xs - 0.22, b2, 0.4, color=BLUE, alpha=0.8, label="bias squared")
        ax.bar(xs + 0.22, va, 0.4, color=ACCENT, alpha=0.85, label="variance")
        ax.plot(xs, total, "o-", color=INK, linewidth=1.8, ms=5,
                label="total (incl. noise)")
        ax.axhline(SIG ** 2, color=INK, linestyle=":", linewidth=1.1,
                   alpha=0.7)
        ax.text(len(degs) - 1.1, SIG ** 2 * 1.25, "noise floor", fontsize=8,
                color=INK, alpha=0.8)
        ax.set_xticks(xs)
        ax.set_xticklabels([str(d) for d in degs])
        ax.set_xlabel("polynomial degree", fontsize=9.5)
        ax.set_yscale("log")
        ax.set_ylim(1e-5, 4)
        ax.set_title(f"n = {n} training points", fontsize=10.5, color=INK,
                     weight="bold")
        ax.legend(fontsize=8, loc="upper left", framealpha=0.92)
        ax.grid(alpha=0.13, which="both")
        style(ax)
    axes[0].set_ylabel("error contribution (log scale)", fontsize=9.5)

    ratio = store[20][1][-1] / store[200][1][-1]
    axes[1].annotate(f"variance at degree 9\nis {ratio:.0f}x smaller here",
                     xy=(5, store[200][1][-1]), xytext=(2.1, 0.25),
                     fontsize=8.5, color=ACCENT,
                     arrowprops=dict(arrowstyle="->", color=ACCENT, lw=0.9))

    fig.text(0.5, -0.04,
             "the same capacity is reckless at n = 20 and nearly free at "
             "n = 200. the dilemma is about capacity relative to data.",
             ha="center", fontsize=8.5, style="italic", color=INK)
    fig.tight_layout()
    fig.savefig(HERE / "bias_variance_n.svg", transparent=True,
                bbox_inches="tight")
    plt.close(fig)
    print(f"wrote bias_variance_n.svg   var@deg9  n=20 {store[20][1][-1]:.5f}"
          f"  n=200 {store[200][1][-1]:.5f}  ratio {ratio:.0f}")


# === 그림으로 보는 AI: 자료를 늘리면 모두가 같이 좋아지는가 ==================
#
# Banko-Brill 2001 의 그림을 베껴 그리지 않는다. 그 대신 **같은 물음**을 여기서
# 직접 돌려 본다 — 자료를 열 배씩 늘릴 때 학습기들이 어떻게 갈라지는가.

def more_data_curves():
    import warnings
    from sklearn.datasets import make_classification
    from sklearn.naive_bayes import GaussianNB
    from sklearn.linear_model import Perceptron, LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import train_test_split

    warnings.filterwarnings("ignore")
    X, y = make_classification(n_samples=40000, n_features=30,
                               n_informative=12, n_redundant=6,
                               class_sep=0.8, flip_y=0.03, random_state=0)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=8000,
                                          random_state=0)

    learners = [
        ("memory-based (kNN)", lambda: KNeighborsClassifier(n_neighbors=5),
         ACCENT, "o-"),
        ("logistic", lambda: LogisticRegression(max_iter=2000), BLUE, "s--"),
        ("naive bayes", lambda: GaussianNB(), "#7a8b6f", "^--"),
        ("perceptron", lambda: Perceptron(max_iter=1000, tol=1e-3,
                                          random_state=0), "#9a8f80", "v--"),
    ]
    sizes = [100, 300, 1000, 3000, 10000, 32000]

    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    curves = {}
    for name, mk, col, fmt in learners:
        acc = []
        for n in sizes:
            reps = [mk().fit(Xtr[i], ytr[i]).score(Xte, yte)
                    for i in (np.random.default_rng(r).choice(
                        len(Xtr), n, replace=False) for r in range(3))]
            acc.append(float(np.mean(reps)))
        curves[name] = acc
        ax.semilogx(sizes, acc, fmt, color=col, linewidth=1.9, ms=5,
                    label=name)

    top = curves["memory-based (kNN)"]
    ax.annotate(f"still climbing at 32,000\n{top[0]:.3f} -> {top[-1]:.3f}",
                xy=(sizes[-1], top[-1]), xytext=(1700, 0.805),
                fontsize=8.5, color=ACCENT,
                arrowprops=dict(arrowstyle="->", color=ACCENT, lw=0.9))
    nb = curves["naive bayes"]
    ax.annotate("flat from about 3,000 on", xy=(10000, nb[4]),
                xytext=(220, 0.645), fontsize=8.5, color="#5f6b58",
                arrowprops=dict(arrowstyle="->", color="#5f6b58", lw=0.9))

    ax.set_xlabel("training examples (log scale)", fontsize=9.5)
    ax.set_ylabel("test accuracy", fontsize=9.5)
    ax.set_title("more data does not lift every learner alike",
                 fontsize=10.5, color=INK, weight="bold", pad=8)
    ax.legend(fontsize=8.5, loc="upper left", framealpha=0.92)
    ax.grid(alpha=0.13, which="both")
    style(ax)

    fig.text(0.5, -0.04,
             "run here, not copied from the 2001 paper. the ranking happened "
             "to hold at every size on this problem.",
             ha="center", fontsize=8.5, style="italic", color=INK)
    fig.tight_layout()
    fig.savefig(HERE / "more_data.svg", transparent=True, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote more_data.svg   knn {top[0]:.4f}->{top[-1]:.4f}   "
          f"nb {nb[0]:.4f}->{nb[-1]:.4f}")


# === 그림으로 보는 AI: 군집이 없는 자료에도 팔꿈치는 생긴다 ==================
#
# 정리 3 은 J 가 k 에 대해 단조 감소한다고 말한다. 팔꿈치법은 그 곡선이 꺾이는
# 자리를 고르자는 요령인데, **군집이 전혀 없는 균일 잡음**에도 꺾임이 보인다.

def elbow_on_noise():
    from sklearn.cluster import KMeans
    from sklearn.datasets import make_blobs

    ks = list(range(1, 11))
    cases = []
    Xb, _ = make_blobs(n_samples=600, centers=3, cluster_std=1.0,
                       random_state=42)
    cases.append(("three real clusters", Xb, 3))
    Xn = np.random.default_rng(0).uniform(0, 1, (600, 2))
    cases.append(("uniform noise - no clusters at all", Xn, None))

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.2))
    for ax, (title, X, truth) in zip(axes, cases):
        J = np.array([KMeans(n_clusters=k, n_init=10,
                             random_state=0).fit(X).inertia_ for k in ks])
        drop = (J[:-1] - J[1:]) / J[:-1] * 100
        # 줄어드는 폭이 가장 크게 꺾이는 자리를 "팔꿈치"로 읽는다
        # drop[i] 는 k=i+1 -> k=i+2 의 감소폭이므로, 꺾이는 자리의 k 는 i+2 다
        elbow = int(np.argmax(drop[:-1] - drop[1:])) + 2

        ax.plot(ks, J / J[0], "o-", color=ACCENT, linewidth=2.0, ms=5.5)
        ax.axvline(elbow, color=INK, linestyle=":", linewidth=1.2, alpha=0.8)
        ax.annotate(f"the rule reads k = {elbow}", xy=(elbow, J[elbow - 1] / J[0]),
                    xytext=(elbow + 1.1, 0.55), fontsize=8.8, color=INK,
                    arrowprops=dict(arrowstyle="->", color=INK, lw=0.9))
        if truth:
            ax.text(0.97, 0.93, f"true k = {truth}", transform=ax.transAxes,
                    ha="right", fontsize=9, color=BLUE, weight="bold")
        else:
            ax.text(0.97, 0.93, "there is no true k", transform=ax.transAxes,
                    ha="right", fontsize=9, color=BLUE, weight="bold")
        ax.set_xlabel("k", fontsize=9.5)
        ax.set_xticks(ks)
        ax.set_ylim(0, 1.05)
        ax.set_title(title, fontsize=10.5, color=INK, weight="bold")
        ax.grid(alpha=0.13)
        style(ax)
    axes[0].set_ylabel("within-cluster sum of squares\n(scaled to J at k=1)",
                       fontsize=9.5)

    fig.text(0.5, -0.04,
             "the curve bends on the right as convincingly as on the left. "
             "a bend is not evidence that groups exist.",
             ha="center", fontsize=8.5, style="italic", color=INK)
    fig.tight_layout()
    fig.savefig(HERE / "elbow_on_noise.svg", transparent=True,
                bbox_inches="tight")
    plt.close(fig)
    print("wrote elbow_on_noise.svg")


if __name__ == "__main__":
    bellman_dates()
    bias_variance_depends_on_n()
    more_data_curves()
    elbow_on_noise()
