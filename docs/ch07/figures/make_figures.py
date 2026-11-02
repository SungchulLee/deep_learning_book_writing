"""
================================================================================
make_figures.py - 2.5~2.6절의 그림을 만든다
================================================================================

만드는 그림:
    sigmoid_steepness.svg  시그모이드가 「부드러운 계단 함수」라는 말을 그린다.
                           z 에 c 를 곱하면 0.5 자리는 그대로인 채 전이만
                           가팔라지고, c 가 커지면 계단에 다가간다.
    poly_bias_variance.svg 차수를 1 에서 10 까지 올리며 학습 오차와 시험
                           오차를 잰다. 쪽의 5절 표가 「차수 3 이 가장 낫다」
                           를 말로만 적어 두었는데, 그것을 재어서 보인다.

수에 대하여:
    시그모이드 쪽은 식 하나뿐이라 잰 것이 없다.
    다항 쪽은 polynomial_features.md 2절이 정의한 그 데이터를 그대로 쓴다
    (rng seed 0, n=50, p=2, test_size=0.3, random_state=0). 그래서 쪽의
    수와 어긋나지 않는다.

그림 규칙(CLAUDE.md): SVG, svg.fonttype='path', transparent=True,
그림 안 글자는 모두 ASCII.
"""
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["svg.fonttype"] = "path"
import matplotlib.pyplot as plt
import numpy as np


def fig_sigmoid_steepness():
    z = np.linspace(-6, 6, 800)
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.0, 4.3))

    for c, col in ((0.5, "#93c5fd"), (1.0, "#1d4ed8"), (3.0, "#b45309"),
                   (10.0, "#b91c1c")):
        axL.plot(z, 1 / (1 + np.exp(-c * z)), lw=2.2, color=col,
                 label=f"c = {c:g}")
    axL.step(z, (z > 0).astype(float), where="mid", color="0.35", ls="--",
             lw=1.6, label="step function")
    axL.axhline(0.5, color="0.7", ls=":", lw=1.1)
    axL.axvline(0, color="0.7", ls=":", lw=1.1)
    axL.plot(0, 0.5, "o", ms=9, color="black", zorder=4)
    axL.annotate("every curve passes through (0, 0.5)", (0, 0.5), (18, -34),
                 textcoords="offset points", fontsize=10.5,
                 arrowprops=dict(arrowstyle="->", color="0.4", lw=1.1))
    axL.set_xlabel("z", fontsize=11)
    axL.set_ylabel("sigmoid(c z)", fontsize=11)
    axL.set_title("scaling the score steepens the transition, "
                  "it does not move it", fontsize=11.5)
    axL.legend(fontsize=10, frameon=False, loc="upper left")
    axL.grid(alpha=0.2)
    axL.spines[["top", "right"]].set_visible(False)

    # 오른쪽: 확률이 0.9 를 넘는 데 필요한 z
    cs = np.linspace(0.3, 12, 300)
    z90 = np.log(9) / cs
    axR.plot(cs, z90, color="#b91c1c", lw=2.4)
    for c in (0.5, 1, 3, 10):
        axR.plot(c, np.log(9) / c, "o", ms=9, color="#b91c1c")
        axR.annotate(f"c={c:g}:  z = {np.log(9) / c:.2f}",
                     (c, np.log(9) / c), (12, 8), textcoords="offset points",
                     fontsize=10, color="#b91c1c")
    axR.set_xlabel("c", fontsize=11)
    axR.set_ylabel("score z needed for probability 0.9", fontsize=11)
    axR.set_title("the bigger c is, the less evidence a confident "
                  "answer needs", fontsize=11.5)
    axR.set_ylim(0, 7.8)
    axR.grid(alpha=0.2)
    axR.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig("sigmoid_steepness.svg", transparent=True)
    plt.close(fig)
    print("wrote sigmoid_steepness.svg")
    for c in (0.5, 1, 3, 10):
        print(f"   c={c:<5g} z for p=0.9 is {np.log(9) / c:.4f}")


def fig_poly_bias_variance():
    """쪽 2절의 데이터를 그대로 쓴다."""
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.metrics import mean_squared_error

    rng = np.random.default_rng(0)
    n, p = 50, 2
    X = rng.normal(size=(n, p))
    y = 1.0 + 2.0 * X[:, 0] - 1.5 * X[:, 1] ** 2 + 0.3 * rng.normal(size=n)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3,
                                          random_state=0)

    degs = list(range(1, 9))
    tr, te, ncol = [], [], []
    for d in degs:
        m = make_pipeline(PolynomialFeatures(d), LinearRegression()).fit(Xtr, ytr)
        tr.append(mean_squared_error(ytr, m.predict(Xtr)))
        te.append(mean_squared_error(yte, m.predict(Xte)))
        ncol.append(PolynomialFeatures(d).fit(Xtr).n_output_features_)

    best = int(np.argmin(te))
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    ax.plot(degs, tr, "o-", color="#1d4ed8", lw=2.2, ms=8,
            label="training error")
    ax.plot(degs, te, "s-", color="#b91c1c", lw=2.4, ms=9,
            label="test error")
    ax.plot(degs[best], te[best], "o", ms=18, mfc="none", mec="#15803d",
            mew=2.4)
    ax.annotate(f"lowest test error\nat degree {degs[best]}",
                (degs[best], te[best]), (16, 24),
                textcoords="offset points", fontsize=11, color="#15803d",
                arrowprops=dict(arrowstyle="->", color="#15803d", lw=1.3))
    ax.set_yscale("log")
    ax.set_ylim(1e-3, 3e6)
    # 차수 7, 8 에서 학습 오차가 0 으로 떨어져 로그 축 밖으로 나간다.
    # 축을 잘라 두고 그 사실을 글로 적는다 - 안 그러면 나머지가 뭉개진다.
    for d, a in zip(degs, tr):
        if a < 1e-3:
            ax.annotate("0", (d, 1.4e-3), ha="center", fontsize=11,
                        color="#1d4ed8")
    ax.text(degs[-1], 2.2e-3, "training error is exactly 0 here  ",
            ha="right", fontsize=10, color="#1d4ed8")
    ax.set_xticks(degs)
    ax.set_xticklabels([f"{d}\n({c} cols)" for d, c in zip(degs, ncol)],
                       fontsize=9.5)
    ax.set_xlabel("polynomial degree   (columns after expansion)",
                  fontsize=11)
    ax.set_ylabel("mean squared error (log scale)", fontsize=11)
    ax.set_title("training error only ever falls; test error turns back up",
                 fontsize=11.5)
    ax.legend(fontsize=11, frameon=False)
    ax.grid(alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig("poly_bias_variance.svg", transparent=True)
    plt.close(fig)
    print("wrote poly_bias_variance.svg")
    print(f"   {'deg':>4}{'cols':>7}{'train':>12}{'test':>12}")
    for d, c, a, b in zip(degs, ncol, tr, te):
        print(f"   {d:>4}{c:>7}{a:>12.5f}{b:>12.5f}")


if __name__ == "__main__":
    fig_sigmoid_steepness()
    fig_poly_bias_variance()
