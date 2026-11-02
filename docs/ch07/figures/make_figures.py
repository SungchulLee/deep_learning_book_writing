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


def fig_ridge_path():
    """릿지 쪽 5.3절의 데이터와 6.1·6.2절의 코드를 그대로 돌린다.

    왼쪽은 6.2절이 코드로 적어 두고 그림은 보이지 않은 정규화 경로다.
    오른쪽은 6.1절의 교차 검증 곡선이며, 그 절의 출력 (Best alpha = 0.3728,
    CV MSE = 0.2753) 과 5.5절의 RidgeCV 출력 (0.1) 을 다시 계산한다.
    """
    import torch
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import cross_val_score

    torch.manual_seed(0)                       # 쪽 5.3절
    n, p = 200, 20
    X = torch.randn(n, p).numpy()
    true_theta = torch.randn(p, 1)
    y = (torch.from_numpy(X).float() @ true_theta
         + 0.5 * torch.randn(n, 1)).numpy().ravel()

    Xs = StandardScaler().fit_transform(X)

    # --- 왼쪽: 6.2절의 경로 ---------------------------------------------
    alphas = np.logspace(-2, 4, 100)
    coefs = np.array([Ridge(alpha=a, fit_intercept=True).fit(Xs, y).coef_
                      for a in alphas])

    # --- 오른쪽: 6.1절의 교차 검증 --------------------------------------
    a_cv = np.logspace(-3, 3, 50)
    cv = []
    for a in a_cv:
        pipe = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=a))])
        cv.append(-cross_val_score(pipe, X, y, cv=5,
                                   scoring="neg_mean_squared_error").mean())
    cv = np.array(cv)
    best = int(np.argmin(cv))
    norms = np.array([np.linalg.norm(Ridge(alpha=a).fit(Xs, y).coef_)
                      for a in a_cv])

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.4, 4.5))

    for j in range(p):
        axL.plot(alphas, coefs[:, j], lw=1.5, alpha=0.85)
    axL.axhline(0, color="0.3", lw=1.0)
    axL.axvline(a_cv[best], color="#15803d", ls="--", lw=1.6)
    axL.annotate(f"cross-validation picks alpha = {a_cv[best]:.4f},\n"
                 "where nothing has shrunk yet",
                 (a_cv[best], -1.70), (120, -16),
                 textcoords="offset points", fontsize=10, color="#15803d",
                 arrowprops=dict(arrowstyle="->", color="#15803d", lw=1.2))
    axL.set_xscale("log")
    axL.set_xlabel("alpha  (ridge penalty)", fontsize=11)
    axL.set_ylabel("coefficient value", fontsize=11)
    axL.set_title("all 20 coefficients slide to zero together, "
                  "none arrives early", fontsize=11)
    axL.grid(alpha=0.2)
    axL.spines[["top", "right"]].set_visible(False)

    axR.plot(a_cv, cv, color="#b91c1c", lw=2.4, label="5-fold CV mean squared error")
    axR.plot(a_cv[best], cv[best], "o", ms=10, color="#b91c1c")
    axR.annotate(f"minimum {cv[best]:.4f}\nat alpha = {a_cv[best]:.4f}",
                 (a_cv[best], cv[best]), (8, 62), textcoords="offset points",
                 fontsize=10, color="#b91c1c",
                 arrowprops=dict(arrowstyle="->", color="#b91c1c", lw=1.2))
    axR.set_xscale("log")
    axR.set_yscale("log")
    axR.set_xlabel("alpha  (ridge penalty)", fontsize=11)
    axR.set_ylabel("CV mean squared error (log scale)", fontsize=11,
                   color="#b91c1c")
    axR.tick_params(axis="y", labelcolor="#b91c1c")
    axR.grid(alpha=0.2)
    axR.spines[["top"]].set_visible(False)

    ax2 = axR.twinx()
    ax2.plot(a_cv, norms, color="#1d4ed8", lw=2.2, ls="--",
             label="length of the coefficient vector")
    ax2.set_ylabel("L2 norm of the coefficients", fontsize=11, color="#1d4ed8")
    ax2.tick_params(axis="y", labelcolor="#1d4ed8")
    ax2.spines[["top"]].set_visible(False)
    axR.set_title("the penalty that costs least is almost no penalty",
                  fontsize=11)

    fig.tight_layout()
    fig.savefig("ridge_path.svg", transparent=True)
    plt.close(fig)
    print("wrote ridge_path.svg")
    print(f"   best alpha {a_cv[best]:.4f}  CV MSE {cv[best]:.4f}")
    print(f"   coefficient norm: alpha=0.001 -> {norms[0]:.3f}, "
          f"alpha={a_cv[best]:.4f} -> {norms[best]:.3f}, "
          f"alpha=1000 -> {norms[-1]:.3f}")
    sv = np.linalg.svd(Xs, compute_uv=False)
    print(f"   singular values of the scaled design: {sv.min():.2f} .. "
          f"{sv.max():.2f}  (condition number {sv.max() / sv.min():.2f})")


def fig_lasso_path():
    """라쏘 쪽 5.0절의 데이터와 8절의 코드를 그대로 돌린다.

    왼쪽은 8절이 코드로만 적어 둔 라쏘 경로다. 참으로 쓰인 다섯 열
    (make_regression 이 돌려주는 coef 로 확인했다) 을 색으로, 나머지
    열다섯을 회색으로 그린다. 오른쪽은 같은 데이터에서 0 이 아닌 계수의
    수를 세어 릿지와 견준다. 5.1절의 출력 (alpha=0.1 에서 18/20,
    LassoCV 가 고른 alpha=0.7615) 을 다시 계산한다.

    8절의 코드는 alpha 를 10까지만 훑는데, 이 데이터에서 계수가 실제로
    죽는 자리는 그보다 뒤다. 그래서 축을 200까지 늘렸다.
    """
    from sklearn.datasets import make_regression
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import lasso_path, Lasso, LassoCV, Ridge
    from sklearn.pipeline import Pipeline

    X, y, coef_true = make_regression(n_samples=200, n_features=20,
                                      n_informative=5, noise=10.0,
                                      random_state=0, coef=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
    Xs = StandardScaler().fit_transform(Xtr)
    used = np.nonzero(coef_true)[0]

    alphas = np.logspace(-3, 2.3, 160)
    a_path, coefs, _ = lasso_path(Xs, ytr, alphas=alphas)

    pcv = Pipeline([("scaler", StandardScaler()),
                    ("lasso", LassoCV(cv=5, random_state=42))]).fit(Xtr, ytr)
    a_star = pcv.named_steps["lasso"].alpha_

    grid = np.logspace(-3, 2.3, 140)
    sel = [set(np.nonzero(np.abs(Lasso(alpha=a, max_iter=200000)
                                 .fit(Xs, ytr).coef_) > 1e-8)[0].tolist())
           for a in grid]
    nz_l = [len(s) for s in sel]
    nz_r = [int(np.sum(np.abs(Ridge(alpha=a).fit(Xs, ytr).coef_) > 1e-8))
            for a in grid]
    exact = [a for a, s in zip(grid, sel) if s == set(used.tolist())]
    lo, hi = min(exact), max(exact)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.8, 4.7))

    for ax in (axL, axR):
        ax.axvspan(lo, hi, color="#15803d", alpha=0.10, zorder=0)
        ax.axvline(0.1, color="#b45309", ls="--", lw=1.5)
        ax.axvline(a_star, color="#1d4ed8", ls="--", lw=1.5)
        ax.set_xscale("log")
        ax.grid(alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)

    for j in range(coefs.shape[0]):
        if j in used:
            axL.plot(a_path, coefs[j], lw=2.4, zorder=3,
                     label=f"column {j}  (truly used)")
        else:
            axL.plot(a_path, coefs[j], lw=1.1, color="0.6", alpha=0.8,
                     zorder=2)
    axL.axhline(0, color="0.3", lw=1.0)
    axL.set_xlabel("alpha  (lasso penalty)", fontsize=11)
    axL.set_ylabel("coefficient value", fontsize=11)
    axL.set_title("the fifteen grey columns die first, the five coloured "
                  "ones last", fontsize=11)
    axL.legend(fontsize=9, frameon=False, loc="upper right")

    axR.step(grid, nz_l, where="post", color="#b91c1c", lw=2.6, label="lasso")
    axR.step(grid, nz_r, where="post", color="#1d4ed8", lw=2.4, ls=":",
             label="ridge, same data")
    axR.axhline(5, color="0.4", ls=":", lw=1.4)
    axR.annotate("alpha = 0.1\n18 of 20 left", (0.1, 18), (-104, -26),
                 textcoords="offset points", fontsize=10, color="#b45309",
                 arrowprops=dict(arrowstyle="->", color="#b45309", lw=1.1))
    axR.annotate(f"cross-validation picks\nalpha = {a_star:.4f}\n"
                 "9 of 20 left", (a_star, 9), (-54, -104),
                 textcoords="offset points", fontsize=10, color="#1d4ed8",
                 arrowprops=dict(arrowstyle="->", color="#1d4ed8", lw=1.1))
    axR.annotate(f"exactly the 5 real columns\nfor alpha in "
                 f"[{lo:.2f}, {hi:.1f}]", (hi, 5), (-32, 52),
                 textcoords="offset points", fontsize=10, color="#15803d",
                 ha="center",
                 arrowprops=dict(arrowstyle="->", color="#15803d", lw=1.1))
    axR.set_xlabel("alpha  (penalty strength)", fontsize=11)
    axR.set_ylabel("number of nonzero coefficients", fontsize=11)
    axR.set_ylim(-0.8, 23)
    axR.set_title("ridge never reaches zero; lasso does, but not where "
                  "it was asked to", fontsize=11)
    axR.legend(fontsize=10, frameon=False, loc="lower left")

    fig.tight_layout()
    fig.savefig("lasso_path.svg", transparent=True)
    plt.close(fig)
    print("wrote lasso_path.svg")
    print(f"   columns truly used: {used.tolist()}")
    c01 = Lasso(alpha=0.1).fit(Xs, ytr).coef_
    ccv = Lasso(alpha=a_star).fit(Xs, ytr).coef_
    print(f"   alpha=0.1   nonzero {int(np.sum(np.abs(c01) > 1e-8))}/20")
    print(f"   alpha={a_star:.4f} nonzero {int(np.sum(np.abs(ccv) > 1e-8))}/20")
    print(f"   at the cross-validated alpha: {np.round(ccv, 2).tolist()}")
    print(f"   exactly the five real columns for alpha in "
          f"[{lo:.4f}, {hi:.4f}]")


def fig_gd_lr_bound():
    """경사 하강법 쪽 4절 1단계의 데이터를 그대로 쓴다.

    쪽 2.3절은 안정 조건을 eta < 1 / lambda_max(X^T X / n) 으로 적고,
    같은 절의 compute_max_learning_rate 는 2 / lambda_max 를 돌려준다.
    어느 쪽이 맞는지 이 쪽의 데이터에서 재어 본다. 바이어스도 함께
    맞추므로 1 열을 붙인 설계 행렬의 최대 고윳값을 쓴다.
    """
    import torch

    torch.manual_seed(42)                      # 쪽 4절 1단계
    n, p = 500, 3
    X = torch.randn(n, p)
    w_true = torch.tensor([2.0, -1.5, 0.5])
    y = X @ w_true + 3.0 + 0.3 * torch.randn(n)

    n_train = int(0.8 * n)
    Xt, yt = X[:n_train], y[:n_train].unsqueeze(1)
    Xa = torch.cat([Xt, torch.ones(n_train, 1)], dim=1)
    lam = torch.linalg.eigvalsh(Xa.T @ Xa / n_train).max().item()
    bound = 1.0 / lam

    def batch_gd(lr, iters=60):
        w, b, hist = torch.zeros(p, 1), torch.zeros(1), []
        for _ in range(iters):
            yp = Xt @ w + b
            hist.append(torch.mean((yt - yp) ** 2).item())
            e = yp - yt
            w = w - lr * (2.0 / n_train) * (Xt.T @ e)
            b = b - lr * (2.0 / n_train) * e.sum()
        return hist

    runs = {0.1: ("#93c5fd", "0.1 x bound  (slow)"),
            0.5: ("#15803d", "0.5 x bound  (fastest here)"),
            0.9: ("#1d4ed8", "0.9 x bound"),
            1.0: ("#b45309", "1.0 x bound  (stalls)"),
            1.2: ("#dc2626", "1.2 x bound  (diverges)"),
            2.0: ("#7f1d1d", "2.0 x bound = the helper's answer")}
    hist = {m: batch_gd(m * bound) for m in runs}
    tail = {m: h[-1] for m, h in hist.items()}

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.8, 4.8))

    for m in (0.1, 0.5, 0.9, 1.0):
        col, lab = runs[m]
        axL.plot(hist[m], color=col, lw=2.3, label=f"{lab}  "
                 f"-  eta = {m * bound:.3f}")
    axL.set_yscale("log")
    axL.set_ylim(0.05, 30)
    axL.axhline(tail[0.5], color="0.6", ls=":", lw=1.2)
    axL.text(59, tail[0.5] * 1.12, f"best loss reached: {tail[0.5]:.4f}",
             ha="right", va="bottom", fontsize=10, color="0.35")
    axL.annotate(f"at the bound exactly the loss\nsticks at {tail[1.0]:.4f}",
                 (42, tail[1.0]), (-24, 56), textcoords="offset points",
                 fontsize=10, color="#b45309",
                 arrowprops=dict(arrowstyle="->", color="#b45309", lw=1.2))
    axL.set_xlabel("batch gradient descent iteration", fontsize=11)
    axL.set_ylabel("mean squared error (log scale)", fontsize=11)
    axL.set_title("up to the bound: it converges", fontsize=11.5)
    axL.legend(fontsize=9.5, frameon=False, loc="upper right")
    axL.grid(alpha=0.2)
    axL.spines[["top", "right"]].set_visible(False)

    for m in (1.0, 1.2, 2.0):
        col, _ = runs[m]
        h = [v if np.isfinite(v) and v > 0 else np.nan for v in hist[m]]
        axR.plot(h, color=col, lw=2.3)
    axR.set_yscale("log")
    axR.set_ylim(1e-2, 1e21)
    axR.text(30, 2.2, f"1.0 x bound  -  eta = {bound:.3f}", fontsize=10,
             color="#b45309")
    axR.text(36, 1.5e6, f"1.2 x bound\neta = {1.2 * bound:.3f}", fontsize=10,
             color="#dc2626")
    axR.annotate("2.0 x bound = what the helper returns:\n"
                 f"eta = {2 * bound:.3f} overflows to infinity by step 25",
                 (15, 8e13), (34, 26), textcoords="offset points",
                 fontsize=10, color="#7f1d1d",
                 arrowprops=dict(arrowstyle="->", color="#7f1d1d", lw=1.2))
    axR.set_xlabel("batch gradient descent iteration", fontsize=11)
    axR.set_ylabel("mean squared error (log scale)", fontsize=11)
    axR.set_title("past the bound: it leaves", fontsize=11.5)
    axR.grid(alpha=0.2)
    axR.spines[["top", "right"]].set_visible(False)

    fig.suptitle(f"the stable step size on this page's data is "
                 f"{bound:.4f}; the helper returns {2 * bound:.4f}",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig("gd_lr_bound.svg", transparent=True)
    plt.close(fig)
    print("wrote gd_lr_bound.svg")
    print(f"   lambda_max(X^T X / n) = {lam:.5f}")
    print(f"   1 / lambda_max = {bound:.5f}   2 / lambda_max = {2 * bound:.5f}")
    for m, (_, lab) in runs.items():
        print(f"   {lab:36s} eta={m * bound:.4f}  "
              f"loss after 60 steps = {tail[m]:.6g}")


if __name__ == "__main__":
    fig_sigmoid_steepness()
    fig_poly_bias_variance()
    fig_ridge_path()
    fig_lasso_path()
    fig_gd_lr_bound()
