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


def fig_decision_boundary():
    """결정 경계 쪽 5절 3부의 모델을 그대로 다시 학습시킨다.

    같은 씨앗(42), 같은 데이터(make_classification 500개), 같은 최적화
    (Adam lr=0.1, 500에폭)이므로 쪽의 출력에 찍힌 계수가 그대로 나온다.
    beta0 = -0.324, beta1 = 0.456, beta2 = 3.412.

    왼쪽은 그 경계와 평행한 확률 등고선이고, 오른쪽은 경계에 수직인
    방향으로 점들을 사영해 2절의 「전이 폭」을 실제 거리로 보인 것이다.
    """
    import torch
    import torch.nn as nn
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    torch.manual_seed(42)
    np.random.seed(42)
    X, y = make_classification(n_samples=500, n_features=2, n_redundant=0,
                               n_informative=2, random_state=42,
                               n_clusters_per_class=1)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2,
                                          random_state=42)
    scaler = StandardScaler()
    Xtr = torch.FloatTensor(scaler.fit_transform(Xtr))
    ytr_t = torch.FloatTensor(ytr).reshape(-1, 1)

    class LogisticRegression(nn.Module):
        def __init__(self, n_features):
            super().__init__()
            self.linear = nn.Linear(n_features, 1)

        def forward(self, x):
            return torch.sigmoid(self.linear(x))

    model = LogisticRegression(2)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    for _ in range(500):
        loss = criterion(model(Xtr), ytr_t)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    w = model.linear.weight.data.numpy().flatten()
    b = float(model.linear.bias.data.numpy()[0])
    nrm = float(np.linalg.norm(w))
    unit = w / nrm
    gap90 = np.log(9) / nrm                    # 0.5 등고선에서 0.9 등고선까지

    P = Xtr.numpy()
    t = P @ unit + b / nrm                     # 경계로부터의 부호 있는 거리
    cls0, cls1 = ytr == 0, ytr == 1

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.0, 4.9))

    # --- 왼쪽: 경계와 평행한 등고선 -------------------------------------
    xs = np.linspace(P[:, 0].min() - 0.4, P[:, 0].max() + 0.4, 200)
    for p_star, style, col in ((0.1, ":", "#1d4ed8"),
                               (0.5, "-", "black"),
                               (0.9, ":", "#b91c1c")):
        c = np.log(p_star / (1 - p_star))
        axL.plot(xs, (c - b - w[0] * xs) / w[1], style, color=col,
                 lw=2.4 if p_star == 0.5 else 1.8)
    axL.scatter(P[cls0, 0], P[cls0, 1], s=18, c="#1d4ed8", alpha=0.45,
                edgecolors="none", label="class 0")
    axL.scatter(P[cls1, 0], P[cls1, 1], s=18, c="#b91c1c", alpha=0.45,
                edgecolors="none", label="class 1")

    foot = -(b / nrm) * unit                   # 원점에서 경계에 내린 발
    axL.annotate("", foot + unit * gap90, foot,
                 arrowprops=dict(arrowstyle="-|>", color="#15803d", lw=2.4))
    axL.annotate(f"from p = 0.5 to p = 0.9 is\n{gap90:.3f} along beta,\n"
                 f"which is log 9 / {nrm:.3f}",
                 foot + unit * gap90 * 0.5, (64, -136),
                 textcoords="offset points", fontsize=10, color="#15803d",
                 arrowprops=dict(arrowstyle="->", color="#15803d", lw=1.1))
    axL.text(xs[4], (np.log(1 / 9) - b - w[0] * xs[4]) / w[1] - 0.30,
             "p = 0.1", fontsize=10, color="#1d4ed8")
    axL.text(xs[4], (-b - w[0] * xs[4]) / w[1] - 0.30,
             "p = 0.5  (the decision boundary)", fontsize=10, color="black")
    axL.text(xs[4], (np.log(9) - b - w[0] * xs[4]) / w[1] + 0.14,
             "p = 0.9", fontsize=10, color="#b91c1c")
    axL.set_xlabel("feature 1 (standardized)", fontsize=11)
    axL.set_ylabel("feature 2 (standardized)", fontsize=11)
    axL.set_title("the contours are parallel; beta is the normal",
                  fontsize=11.5)
    axL.set_ylim(P[:, 1].min() - 0.5, P[:, 1].max() + 0.5)
    axL.legend(fontsize=10, frameon=False, loc="lower right")
    axL.grid(alpha=0.2)
    axL.spines[["top", "right"]].set_visible(False)

    # --- 오른쪽: 법선 방향으로 사영 -------------------------------------
    tt = np.linspace(t.min() - 0.2, t.max() + 0.2, 400)
    axR.plot(tt, 1 / (1 + np.exp(-nrm * tt)), color="black", lw=2.4)
    axR.axvspan(-gap90, gap90, color="#fde68a", alpha=0.45, zorder=0)
    axR.scatter(t[cls0], np.full(cls0.sum(), -0.055), s=14, c="#1d4ed8",
                alpha=0.4, edgecolors="none")
    axR.scatter(t[cls1], np.full(cls1.sum(), 1.055), s=14, c="#b91c1c",
                alpha=0.4, edgecolors="none")
    axR.axhline(0.5, color="0.7", ls=":", lw=1.1)
    axR.axvline(0, color="0.7", ls=":", lw=1.1)
    inside = int(np.sum(np.abs(t) < gap90))
    axR.annotate(f"{inside} of {len(t)} training points\nlie in the band "
                 f"where the model\nis less than 90% sure",
                 (0, 0.30), (58, -14), textcoords="offset points",
                 fontsize=10, color="#92400e",
                 arrowprops=dict(arrowstyle="->", color="#92400e", lw=1.1))
    axR.set_xlabel("signed distance from the boundary, along beta",
                   fontsize=11)
    axR.set_ylabel("predicted P(Y = 1 | x)", fontsize=11)
    axR.set_title(f"the transition is {2 * gap90:.2f} wide because "
                  f"||beta|| = {nrm:.2f}", fontsize=11.5)
    axR.set_ylim(-0.12, 1.12)
    axR.grid(alpha=0.2)
    axR.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig("decision_boundary.svg", transparent=True)
    plt.close(fig)
    print("wrote decision_boundary.svg")
    print(f"   beta0 = {b:.3f}  beta1 = {w[0]:.3f}  beta2 = {w[1]:.3f}")
    print(f"   boundary: x2 = {-b / w[1]:.3f} + {-w[0] / w[1]:.3f} x1")
    print(f"   ||beta_1:d|| = {nrm:.4f}")
    print(f"   origin to boundary = |beta0|/||beta|| = {abs(b) / nrm:.4f}")
    print(f"   0.5 -> 0.9 contour distance = {gap90:.4f}")
    print(f"   training points inside the |p-0.5| < 0.4 band: "
          f"{inside} / {len(t)}")


def _breast_cancer_loaders(batch_size=32):
    """정규화된 로지스틱 회귀 쪽 6절 1부의 prepare_data 와 같은 분할."""
    import torch
    from torch.utils.data import TensorDataset, DataLoader
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    data = load_breast_cancer()
    X, y = data.data, data.target
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp)

    scaler = StandardScaler().fit(X_train)

    def to_ds(a, b):
        return TensorDataset(torch.FloatTensor(scaler.transform(a)),
                             torch.FloatTensor(b).reshape(-1, 1))

    return (DataLoader(to_ds(X_train, y_train), batch_size=batch_size,
                       shuffle=True),
            DataLoader(to_ds(X_val, y_val), batch_size=batch_size),
            DataLoader(to_ds(X_test, y_test), batch_size=batch_size),
            X.shape[1])


def fig_regularization_strength():
    """정규화된 로지스틱 회귀 쪽 6절 6부의 lambda 조율을 다시 돌린다.

    씨앗 42 한 번만 돌린 쪽의 출력(λ=0 에서 AUC 0.9934·||β||=4.1285,
    λ=0.01 에서 0.9960·2.2926, λ=1 에서 0.9934·0.4598)을 재현하고,
    그 위에 씨앗 0~4 와 42 를 합한 여섯 번의 폭을 얹는다.
    「한 번 돌린 것은 잰 것이 아니다」(CLAUDE.md) 를 이 쪽의 수에
    적용한 것이다.
    """
    import torch
    import torch.nn as nn
    from sklearn.metrics import roc_auc_score

    train_loader, val_loader, test_loader, n_features = \
        _breast_cancer_loaders()

    class LogisticRegression(nn.Module):
        def __init__(self, n):
            super().__init__()
            self.linear = nn.Linear(n, 1)
            nn.init.xavier_uniform_(self.linear.weight)
            nn.init.zeros_(self.linear.bias)

        def logits(self, x):
            return self.linear(x)

    def fit(l2, seed, num_epochs=200, lr=0.01, patience=15):
        torch.manual_seed(seed)
        model = LogisticRegression(n_features)
        crit = nn.BCEWithLogitsLoss()
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2)
        best, best_state, wait = float("inf"), None, 0
        for _ in range(num_epochs):
            model.train()
            for bx, by in train_loader:
                loss = crit(model.logits(bx), by)
                opt.zero_grad()
                loss.backward()
                opt.step()
            model.eval()
            tot, n = 0.0, 0
            with torch.no_grad():
                for bx, by in val_loader:
                    tot += crit(model.logits(bx), by).item() * len(bx)
                    n += len(bx)
            v = tot / n
            if v < best:
                best, wait = v, 0
                best_state = {k: t.clone() for k, t in model.state_dict().items()}
            else:
                wait += 1
                if wait >= patience:
                    break
        model.load_state_dict(best_state)
        model.eval()
        probs, targets = [], []
        with torch.no_grad():
            for bx, by in test_loader:
                probs += torch.sigmoid(model.logits(bx)).numpy().ravel().tolist()
                targets += by.numpy().ravel().tolist()
        w = model.linear.weight.data.numpy().ravel()
        return roc_auc_score(targets, probs), float(np.linalg.norm(w))

    lambdas = [0.0, 0.0001, 0.001, 0.01, 0.1, 1.0]
    x = [1e-5 if lam == 0 else lam for lam in lambdas]
    page = [fit(lam, 42) for lam in lambdas]          # 쪽이 찍은 수
    seeds = (0, 1, 2, 3, 4, 42)      # 쪽이 쓴 42 도 폭 안에 넣는다
    sweep = [[fit(lam, s) if s != 42 else page[i]
              for s in seeds] for i, lam in enumerate(lambdas)]

    auc42 = [a for a, _ in page]
    nrm42 = [n for _, n in page]
    auc_lo = [min(a for a, _ in row) for row in sweep]
    auc_hi = [max(a for a, _ in row) for row in sweep]
    nrm_lo = [min(n for _, n in row) for row in sweep]
    nrm_hi = [max(n for _, n in row) for row in sweep]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.8, 4.6))

    axL.fill_between(x, auc_lo, auc_hi, color="#b91c1c", alpha=0.18,
                     label="spread over 6 seeds")
    axL.plot(x, auc42, "o-", color="#b91c1c", lw=2.4, ms=8,
             label="the run this page prints (seed 42)")
    axL.annotate(f"no penalty: {auc42[0]:.4f}\n"
                 f"best: {max(auc42):.4f} at lambda = 0.01\n"
                 f"the two seed bands do not overlap",
                 (0.01, max(auc42)), (24, -122), textcoords="offset points",
                 fontsize=10, color="#b91c1c",
                 arrowprops=dict(arrowstyle="->", color="#b91c1c", lw=1.2))
    axL.set_xscale("log")
    axL.set_xticks(x)
    axL.set_xticklabels(["0", "1e-4", "1e-3", "0.01", "0.1", "1"])
    axL.set_xlabel("lambda  (L2 weight decay)", fontsize=11)
    axL.set_ylabel("test AUC", fontsize=11)
    axL.set_title(f"the penalty buys {max(auc42) - auc42[0]:.4f} of AUC",
                  fontsize=11.5)
    axL.legend(fontsize=10, frameon=False, loc="lower left")
    axL.grid(alpha=0.2)
    axL.spines[["top", "right"]].set_visible(False)

    axR.fill_between(x, nrm_lo, nrm_hi, color="#1d4ed8", alpha=0.18,
                     label="spread over 6 seeds")
    axR.plot(x, nrm42, "o-", color="#1d4ed8", lw=2.4, ms=8,
             label="the run this page prints (seed 42)")
    axR.annotate(f"seed to seed the length varies by {nrm_hi[0] - nrm_lo[0]:.2f}",
                 (x[0], nrm_hi[0]), (66, -28), textcoords="offset points",
                 fontsize=10, color="#1d4ed8",
                 arrowprops=dict(arrowstyle="->", color="#1d4ed8", lw=1.1))
    axR.annotate(f"here by {nrm_hi[-1] - nrm_lo[-1]:.3f}",
                 (x[-1], nrm_hi[-1]), (-26, 52), textcoords="offset points",
                 fontsize=10, color="#1d4ed8", ha="right",
                 arrowprops=dict(arrowstyle="->", color="#1d4ed8", lw=1.1))
    axR.set_xscale("log")
    axR.set_xticks(x)
    axR.set_xticklabels(["0", "1e-4", "1e-3", "0.01", "0.1", "1"])
    axR.set_xlabel("lambda  (L2 weight decay)", fontsize=11)
    axR.set_ylabel("length of the weight vector", fontsize=11)
    axR.set_title("what the penalty really moves: the length, and how much\n"
                  "it wanders from run to run", fontsize=11.5)
    axR.legend(fontsize=10, frameon=False, loc="lower left")
    axR.grid(alpha=0.2)
    axR.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig("regularization_strength.svg", transparent=True)
    plt.close(fig)
    print("wrote regularization_strength.svg")
    print(f"   {'lambda':>8}{'AUC(42)':>10}{'AUC lo':>9}{'AUC hi':>9}"
          f"{'|b|(42)':>10}{'|b| lo':>9}{'|b| hi':>9}")
    for i, lam in enumerate(lambdas):
        print(f"   {lam:>8.4f}{auc42[i]:>10.4f}{auc_lo[i]:>9.4f}"
              f"{auc_hi[i]:>9.4f}{nrm42[i]:>10.4f}{nrm_lo[i]:>9.4f}"
              f"{nrm_hi[i]:>9.4f}")


def fig_newton_irls():
    """경사 계산 쪽 10절을 통째로 다시 돌린다.

    그 쪽의 출력은 세 방법이 같은 로스 0.488636 에 이르렀다고 찍고도
    「Newton ~ IRLS: False, Max difference: 1.27e+01」 이라고 적는다.
    까닭은 같은 출력의 윗줄에 이미 있다 - 헤세 행렬의 가장 작은 고윳값이
    정확히 0 이다. make_classification 의 n_redundant=1 이 다른 열들의
    선형 결합인 열을 하나 만들어 X 의 랭크가 6 이 아니라 5 다.

    왼쪽은 수렴, 오른쪽은 그 고윳값들이다. 쪽의 2부가 난수를 먼저 쓰므로
    그 부분까지 그대로 재현해야 같은 수가 나온다.
    """
    import torch
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    torch.manual_seed(42)
    np.random.seed(42)

    def sigmoid(z):
        return 1.0 / (1.0 + torch.exp(-z))

    def compute_loss(X, y, beta):
        p = sigmoid(X @ beta)
        eps = 1e-12
        return -(y * torch.log(p + eps) + (1 - y) * torch.log(1 - p + eps)).mean()

    def compute_gradient(X, y, beta):
        p = sigmoid(X @ beta)
        return (1.0 / len(y)) * X.T @ (p - y)

    def compute_hessian(X, beta):
        p = sigmoid(X @ beta)
        b = p * (1 - p)
        return (1.0 / len(p)) * (X * b).T @ X

    # 쪽 2부 - 여기서 난수를 쓰므로 건너뛰면 뒤의 수가 달라진다
    X0 = torch.randn(100, 5)
    y0 = torch.randint(0, 2, (100, 1)).float()
    beta0 = torch.randn(5, 1, requires_grad=True)
    compute_loss(X0, y0, beta0).backward()

    X_raw, y_raw = make_classification(n_samples=200, n_features=5,
                                       n_informative=4, n_redundant=1,
                                       random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X_raw, y_raw, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = torch.FloatTensor(scaler.fit_transform(X_train))
    y_train = torch.FloatTensor(y_train).reshape(-1, 1)
    X_train = torch.cat([torch.ones(len(X_train), 1), X_train], dim=1)
    n, d = X_train.shape

    beta_gd, gd_losses = torch.zeros(d, 1), []
    for _ in range(50):
        gd_losses.append(compute_loss(X_train, y_train, beta_gd).item())
        beta_gd = beta_gd - 1.0 * compute_gradient(X_train, y_train, beta_gd)

    beta_newton, newton_losses = torch.zeros(d, 1), []
    for _ in range(10):
        newton_losses.append(compute_loss(X_train, y_train, beta_newton).item())
        beta_newton = beta_newton - torch.linalg.solve(
            compute_hessian(X_train, beta_newton),
            compute_gradient(X_train, y_train, beta_newton))

    beta_irls, irls_losses = torch.zeros(d, 1), []
    for _ in range(10):
        irls_losses.append(compute_loss(X_train, y_train, beta_irls).item())
        p = sigmoid(X_train @ beta_irls)
        B = p * (1 - p)
        z = X_train @ beta_irls - (p - y_train) / B
        beta_irls = torch.linalg.solve((X_train * B).T @ X_train,
                                       (X_train * B).T @ z)

    star = min(newton_losses[-1], irls_losses[-1])
    eig = torch.linalg.eigvalsh(compute_hessian(X_train,
                                                torch.zeros(d, 1))).numpy()
    rank = int(torch.linalg.matrix_rank(X_train).item())
    bdiff = float((beta_newton - beta_irls).abs().max())
    pred = float((X_train @ (beta_newton - beta_irls)).norm())

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.8, 4.7))

    axL.plot(gd_losses, "-o", ms=3.5, color="#1d4ed8", lw=2.0,
             label="gradient descent, 50 steps")
    axL.plot(newton_losses, "-s", ms=9, color="#b91c1c", lw=2.4,
             label="Newton, 10 steps")
    axL.plot(irls_losses, "--^", ms=7, color="#15803d", lw=2.0,
             label="IRLS, 10 steps  (under the red line)")
    axL.axhline(star, color="0.6", ls=":", lw=1.2)
    axL.annotate(f"both are at {star:.6f}\nby step 3",
                 (3, newton_losses[3]), (34, 44),
                 textcoords="offset points", fontsize=10, color="#b91c1c",
                 arrowprops=dict(arrowstyle="->", color="#b91c1c", lw=1.2))
    axL.annotate(f"after 50 steps: {gd_losses[-1]:.6f}",
                 (49, gd_losses[-1]), (-16, 46),
                 textcoords="offset points", fontsize=10, color="#1d4ed8",
                 ha="right",
                 arrowprops=dict(arrowstyle="->", color="#1d4ed8", lw=1.2))
    axL.set_xlabel("iteration", fontsize=11)
    axL.set_ylabel("binary cross-entropy loss", fontsize=11)
    axL.set_title("Newton and IRLS are the same algorithm here",
                  fontsize=11.5)
    axL.legend(fontsize=10, frameon=False, loc="upper right")
    axL.grid(alpha=0.2)
    axL.spines[["top", "right"]].set_visible(False)

    tol = eig.max() * 1e-5            # float32 에서 0 은 2.9e-08 로 나온다
    cols = ["#b91c1c" if v < tol else "#1d4ed8" for v in eig]
    axR.bar(range(1, d + 1), eig, color=cols, width=0.62)
    axR.plot([1], [0], "x", ms=16, mew=3.4, color="#b91c1c", zorder=5,
             clip_on=False)
    for i, v in enumerate(eig):
        if v >= tol:           # 0 인 막대는 빨간 x 와 설명으로 대신한다
            axR.text(i + 1, v + 0.018, f"{v:.3f}", ha="center",
                     fontsize=9.5, color="0.25")
    axR.annotate(f"2.9e-08, which is 0 in float32:\n"
                 f"the design has {d} columns\n"
                 f"but rank {rank}, so the minimiser\n"
                 "is a line, not a point",
                 (1, 0.012), (24, 92), textcoords="offset points",
                 fontsize=10, color="#b91c1c",
                 arrowprops=dict(arrowstyle="->", color="#b91c1c", lw=1.2))
    axR.set_xticks(range(1, d + 1))
    axR.set_xlabel("eigenvalue of the Hessian at beta = 0, sorted",
                   fontsize=11)
    axR.set_ylabel("eigenvalue", fontsize=11)
    axR.set_ylim(0, 0.78)
    axR.set_title(f"which is why the two answers differ by {bdiff:.2f}",
                  fontsize=11.5)
    axR.grid(alpha=0.2, axis="y")
    axR.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig("newton_irls.svg", transparent=True)
    plt.close(fig)
    print("wrote newton_irls.svg")
    print(f"   n = {n}, d = {d}, rank(X) = {rank}")
    print(f"   Hessian eigenvalues: {np.round(eig, 6).tolist()}")
    print(f"   GD final {gd_losses[-1]:.6f}  Newton {newton_losses[-1]:.6f}"
          f"  IRLS {irls_losses[-1]:.6f}")
    print(f"   max |beta_newton - beta_irls| = {bdiff:.4f}")
    print(f"   || X (beta_newton - beta_irls) || = {pred:.3e}")
    print(f"   beta_newton = {beta_newton.numpy().ravel().round(3).tolist()}")
    print(f"   beta_irls   = {beta_irls.numpy().ravel().round(3).tolist()}")


def fig_capture_recapture_spread():
    """표지-재포획 쪽 2절의 설정(N=150, C=30, R=40)에서 추정값의 분포.

    쪽의 코드는 1000번 모의실험을 돌려 표본 변동을 본다(씨앗 0 으로
    고정되어 있다). 여기서는 모의실험 대신 초기하분포의 확률질량을
    그대로 계산했으므로 난수가 아예 들어가지 않는다 - 표본 오차 없이
    정확한 분포다. T ~ Hypergeometric(N=150, K=30, n=40) 이고
    링컨-피터슨 추정값은 N_hat = C*R/T = 1200/T 다.

    쪽이 관측한 T=8 은 이 분포의 최빈값이자 기댓값이다. 그래서 0.7%
    오차가 나왔다.
    """
    from math import comb

    N_true, C, R = 150, 30, 40
    ts = [t for t in range(0, min(C, R) + 1) if 0 <= R - t <= N_true - C]
    pmf = np.array([comb(C, t) * comb(N_true - C, R - t) / comb(N_true, R)
                    for t in ts])
    ts = np.array(ts)

    pos = ts > 0
    nhat = C * R / np.where(pos, ts, 1)
    p_pos, n_pos = pmf[pos], nhat[pos]
    within10 = p_pos[np.abs(n_pos - N_true) <= 0.1 * N_true].sum()
    over200 = p_pos[n_pos >= 200].sum()
    order = np.argsort(n_pos)
    cum = np.cumsum(p_pos[order]) / p_pos.sum()
    q05 = n_pos[order][int(np.searchsorted(cum, 0.05))]
    q95 = n_pos[order][int(np.searchsorted(cum, 0.95))]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.8, 4.6))

    cols = ["#b91c1c" if t == 8 else "#1d4ed8" for t in ts]
    axL.bar(ts, pmf, color=cols, width=0.72)
    axL.annotate(f"the survey saw T = 8\nwith probability {pmf[ts == 8][0]:.3f}"
                 f"\n- the most likely result there is",
                 (8, pmf[ts == 8][0]), (46, -34), textcoords="offset points",
                 fontsize=10, color="#b91c1c",
                 arrowprops=dict(arrowstyle="->", color="#b91c1c", lw=1.2))
    axL.set_xlim(-0.6, 18)
    axL.set_xticks(range(0, 18, 2))
    axL.set_xlabel("T, the number of marked deer among the 40 recaptured",
                   fontsize=10.5)
    axL.set_ylabel("probability", fontsize=11)
    axL.set_title("what the survey can see is well behaved", fontsize=11.5)
    axL.grid(alpha=0.2, axis="y")
    axL.spines[["top", "right"]].set_visible(False)

    axR.axvspan(0.9 * N_true, 1.1 * N_true, color="#15803d", alpha=0.13,
                zorder=0)
    for n_, p_, t_ in zip(n_pos, p_pos, ts[pos]):
        c_ = "#b91c1c" if t_ == 8 else "#1d4ed8"
        axR.plot([n_, n_], [0, p_], color=c_, lw=3.0, solid_capstyle="butt",
                 zorder=4 if t_ == 8 else 2)
        axR.plot(n_, p_, "o", ms=6, color=c_, zorder=5)
    axR.axvline(N_true, color="#15803d", ls="--", lw=1.6, zorder=1)
    axR.annotate(f"only {within10:.1%} of surveys land\nwithin 10% of the "
                 f"true 150", (N_true, 0.150), (46, 20),
                 textcoords="offset points", fontsize=10, color="#15803d",
                 arrowprops=dict(arrowstyle="->", color="#15803d", lw=1.2))
    axR.annotate(f"{over200:.1%} of surveys answer\n200 or more",
                 (200, 0.126), (44, -54), textcoords="offset points",
                 fontsize=10, color="#1d4ed8",
                 arrowprops=dict(arrowstyle="->", color="#1d4ed8", lw=1.1))
    axR.set_xscale("log")
    axR.set_xticks([75, 100, 150, 200, 300, 600, 1200])
    axR.set_xticklabels(["75", "100", "150", "200", "300", "600", "1200"])
    axR.set_xlim(60, 1600)
    axR.set_xlabel("the estimate 1200 / T  (log scale)", fontsize=11)
    axR.set_ylabel("probability", fontsize=11)
    axR.set_title(f"what it reports is not: 5% to 95% runs "
                  f"{q05:.0f} to {q95:.0f}", fontsize=11.5)
    axR.grid(alpha=0.2, axis="y")
    axR.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig("capture_recapture_spread.svg", transparent=True)
    plt.close(fig)
    print("wrote capture_recapture_spread.svg")
    print(f"   P(T=8) = {pmf[ts == 8][0]:.4f}, E[T] = {(ts * pmf).sum():.3f}")
    print(f"   P(|Nhat - 150| <= 15) = {within10:.4f}")
    print(f"   P(Nhat >= 200) = {over200:.4f}")
    print(f"   5% to 95% of Nhat: {q05:.1f} to {q95:.1f}")
    print(f"   P(T = 0), where the estimate does not exist: {pmf[0]:.2e}")


def fig_likelihood_sharpens():
    """최대가능도 추정 쪽 8절의 generate_coin_flips 를 그대로 쓴다.

    같은 함수, 같은 씨앗 42 로 n = 10, 100, 1000 을 뽑는다.
    n = 100 은 쪽이 출력에 찍은 바로 그 자료(앞면 67번, MLE 0.6700)다.

    왼쪽은 5절의 「피셔 정보가 크면 로그우도가 날카로운 봉우리를 이룬다」
    를 그대로 그린 것이고, 오른쪽은 6절의 「오차를 절반으로 줄이려면
    데이터가 4배 필요하다」를 로그-로그 자 위에 올린 것이다.
    """
    import torch

    def generate_coin_flips(n_flips, true_p, seed=42):
        torch.manual_seed(seed)
        return (torch.rand(n_flips) < true_p).float()

    TRUE_P = 0.7
    sizes = (10, 100, 1000)
    cols = ("#93c5fd", "#1d4ed8", "#b91c1c")
    p = np.linspace(0.005, 0.995, 1200)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.8, 4.6))

    mles = []
    for n, col in zip(sizes, cols):
        data = generate_coin_flips(n, TRUE_P)
        k = int(data.sum().item())
        mle = k / n
        mles.append(mle)
        ll = k * np.log(p) + (n - k) * np.log(1 - p)
        axL.plot(p, np.exp(ll - ll.max()), color=col, lw=2.4,
                 label=f"n = {n}:  {k} heads,  MLE = {mle:.3f}")
        axL.plot(mle, 1.0, "o", ms=8, color=col)

    axL.axvline(TRUE_P, color="#15803d", ls="--", lw=1.8)
    axL.annotate("the p that generated the data: 0.7", (TRUE_P, 0.62),
                 (-212, 10), textcoords="offset points", fontsize=10,
                 color="#15803d",
                 arrowprops=dict(arrowstyle="->", color="#15803d", lw=1.1))
    axL.set_xlim(0.1, 1.0)
    axL.set_xlabel("p", fontsize=11)
    axL.set_ylabel("likelihood, rescaled so the peak is 1", fontsize=11)
    axL.set_title("more data is a narrower peak, not a better guess",
                  fontsize=11.5)
    axL.legend(fontsize=10, frameon=False, loc="upper left")
    axL.grid(alpha=0.2)
    axL.spines[["top", "right"]].set_visible(False)

    ns = np.logspace(0.7, 3.6, 200)
    half = 1.96 * np.sqrt(TRUE_P * (1 - TRUE_P) / ns)
    axR.loglog(ns, half, color="#15803d", lw=2.4,
               label="95% half-width from the Fisher information")
    offs = ((16, 16, "left"), (-14, -30, "right"), (-14, -30, "right"))
    for n, m, col, (dx, dy, ha) in zip(sizes, mles, cols, offs):
        axR.plot(n, abs(m - TRUE_P), "o", ms=11, color=col)
        axR.annotate(f"n = {n}: off by {abs(m - TRUE_P):.3f}",
                     (n, abs(m - TRUE_P)), (dx, dy), ha=ha,
                     textcoords="offset points", fontsize=10, color=col)
    axR.set_xlabel("number of flips (log scale)", fontsize=11)
    axR.set_ylabel("distance from 0.7 (log scale)", fontsize=11)
    axR.set_title("the line has slope -1/2: four times the data, "
                  "half the error", fontsize=11.5)
    axR.legend(fontsize=10, frameon=False, loc="upper right")
    axR.grid(alpha=0.2, which="both")
    axR.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig("likelihood_sharpens.svg", transparent=True)
    plt.close(fig)
    print("wrote likelihood_sharpens.svg")
    for n, m in zip(sizes, mles):
        print(f"   n={n:>5}  MLE={m:.4f}  |MLE-0.7|={abs(m - TRUE_P):.4f}  "
              f"95% half-width={1.96 * np.sqrt(0.21 / n):.4f}")


def fig_mae_vs_mse():
    """회귀 MLE 쪽 4절의 mae_vs_mse_robustness() 를 그대로 돌린다.

    쪽은 이 함수를 정의만 하고 부르지 않아 출력이 없다. 같은 씨앗(42),
    같은 데이터(x 를 0~10 에 100개, y = 2x + 1 + 0.5 잡음, 10/30/50/70/90
    번째에 +15), 같은 학습(Adam lr=0.1, 1000스텝)으로 돌린 결과다.

    2절 표의 「MSE 민감함 / MAE 강건함」이 여기서는 기울기가 아니라
    절편에 나타난다. 이상치가 x 전체에 고르게 퍼져 있기 때문이다.
    """
    import torch
    import torch.nn as nn

    torch.manual_seed(42)
    n = 100
    x = torch.linspace(0, 10, n).reshape(-1, 1)
    y_clean = 2 * x + 1 + torch.randn(n, 1) * 0.5
    y = y_clean.clone()
    outlier_idx = [10, 30, 50, 70, 90]
    y[outlier_idx] += 15

    def fit(loss_fn):
        torch.manual_seed(42)
        model = nn.Linear(1, 1)
        opt = torch.optim.Adam(model.parameters(), lr=0.1)
        for _ in range(1000):
            loss = loss_fn(model(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        return model.weight.item(), model.bias.item()

    w_mse, b_mse = fit(nn.MSELoss())
    w_mae, b_mae = fit(nn.L1Loss())

    xn = x.numpy().ravel()
    yn = y.numpy().ravel()
    keep = np.ones(n, dtype=bool)
    keep[outlier_idx] = False

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.8, 4.7))

    axL.scatter(xn[keep], yn[keep], s=22, c="0.55", alpha=0.75,
                edgecolors="none", label="95 ordinary points")
    axL.scatter(xn[~keep], yn[~keep], s=90, marker="X", c="#b45309",
                label="5 outliers, each +15")
    grid = np.array([0.0, 10.0])
    axL.plot(grid, 2 * grid + 1, color="#15803d", lw=2.2, ls="--",
             label="the line the data came from:  2.00 x + 1.00")
    axL.plot(grid, w_mse * grid + b_mse, color="#b91c1c", lw=2.4,
             label=f"MSE fit:  {w_mse:.2f} x + {b_mse:.2f}")
    axL.plot(grid, w_mae * grid + b_mae, color="#1d4ed8", lw=2.4,
             label=f"MAE fit:  {w_mae:.2f} x + {b_mae:.2f}")
    axL.set_xlabel("x", fontsize=11)
    axL.set_ylabel("y", fontsize=11)
    axL.set_title("the slope survives; the intercept does not", fontsize=11.5)
    axR.legend(fontsize=9.5, frameon=False, loc="upper left")
    axL.grid(alpha=0.2)
    axL.spines[["top", "right"]].set_visible(False)

    bs = np.linspace(0.3, 2.6, 400)
    resid = yn[None, :] - (2.0 * xn[None, :] + bs[:, None])
    mse_b = (resid ** 2).mean(axis=1)
    mae_b = np.abs(resid).mean(axis=1)
    axR.plot(bs, mse_b / mse_b.min(), color="#b91c1c", lw=2.4,
             label="mean squared error (Gaussian NLL)")
    axR.plot(bs, mae_b / mae_b.min(), color="#1d4ed8", lw=2.4,
             label="mean absolute error (Laplace NLL)")
    txt_mean = "the mean of y - 2x"
    base = yn - 2.0 * xn
    b_mean, b_median = float(base.mean()), float(np.median(base))
    for val, col, txt in ((b_mean, "#b91c1c", "the mean of y - 2x"),
                          (b_median, "#1d4ed8", "the median of y - 2x")):
        axR.axvline(val, color=col, ls=":", lw=1.4)
    axR.annotate(f"MSE settles on {b_mean:.2f},\n{txt_mean}",
                 (b_mean, 1.0), (14, 62), textcoords="offset points",
                 fontsize=10, color="#b91c1c",
                 arrowprops=dict(arrowstyle="->", color="#b91c1c", lw=1.1))
    axR.annotate(f"MAE settles on {b_median:.2f},\nthe median of y - 2x",
                 (b_median, 1.0), (-14, 96), ha="right",
                 textcoords="offset points", fontsize=10, color="#1d4ed8",
                 arrowprops=dict(arrowstyle="->", color="#1d4ed8", lw=1.1))
    axR.axvline(1.0, color="#15803d", ls="--", lw=1.6)
    axR.text(1.06, 1.58, "true b = 1", fontsize=10, color="#15803d")
    axR.set_xlabel("intercept b, with the slope held at 2.0", fontsize=11)
    axR.set_ylabel("loss, divided by its own minimum", fontsize=11)
    axR.set_title("the two likelihoods disagree about where the line sits",
                  fontsize=11.5)
    axR.legend(fontsize=10, frameon=False, loc="upper left")
    axR.grid(alpha=0.2)
    axR.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig("mae_vs_mse.svg", transparent=True)
    plt.close(fig)
    print("wrote mae_vs_mse.svg")
    print(f"   true   w = 2.0000  b = 1.0000")
    print(f"   MSE    w = {w_mse:.4f}  b = {b_mse:.4f}")
    print(f"   MAE    w = {w_mae:.4f}  b = {b_mae:.4f}")
    print(f"   5 outliers of +15 over 100 points lift the mean of y by "
          f"{5 * 15 / 100:.2f}")
    print(f"   with the slope held at 2.0: mean(y - 2x) = {b_mean:.4f}, "
          f"median(y - 2x) = {b_median:.4f}")


def fig_softmax_limits():
    """소프트맥스 회귀의 한계 쪽 3절의 코드를 그대로 돌리고 그 결정
    영역을 그린다.

    make_data 와 train 을 쪽에 적힌 그대로 옮겼으므로 정확도도 쪽의
    출력과 같다 (blobs 88.83/87.50, moons 86.50/97.00,
    circles 52.17/99.00).

    2절이 「결정 영역은 볼록하다」고 증명한 것을 눈으로 보이는 그림이다.
    """
    import torch
    import torch.nn as nn
    from sklearn.datasets import make_blobs, make_moons, make_circles
    from sklearn.model_selection import train_test_split

    def make_data(name, n=2000, seed=0):
        if name == "blobs":
            X, y = make_blobs(n_samples=n, centers=3, n_features=2,
                              cluster_std=1.2, random_state=seed)
        elif name == "moons":
            X, y = make_moons(n_samples=n, noise=0.20, random_state=seed)
        else:
            X, y = make_circles(n_samples=n, noise=0.10, factor=0.5,
                                random_state=seed)
        X = (X - X.mean(0)) / X.std(0)
        return train_test_split(X.astype(np.float32), y.astype(np.int64),
                                test_size=0.3, random_state=seed)

    def train(model, Xtr, ytr, Xte, yte, epochs=400):
        opt = torch.optim.Adam(model.parameters(), lr=0.05)
        criterion = nn.CrossEntropyLoss()
        Xtr, ytr = torch.from_numpy(Xtr), torch.from_numpy(ytr)
        Xte, yte = torch.from_numpy(Xte), torch.from_numpy(yte)
        for _ in range(epochs):
            opt.zero_grad()
            criterion(model(Xtr), ytr).backward()
            opt.step()
        with torch.no_grad():
            return 100 * (model(Xte).argmax(1) == yte).float().mean().item()

    names = ("blobs", "moons", "circles")
    fig, axes = plt.subplots(2, 3, figsize=(13.0, 8.0))
    face = ("#dbeafe", "#fee2e2", "#dcfce7")
    edge = ("#1d4ed8", "#b91c1c", "#15803d")
    accs = {}

    for col, name in enumerate(names):
        Xtr, Xte, ytr, yte = make_data(name)
        k = len(np.unique(ytr))

        torch.manual_seed(0)
        linear = nn.Linear(2, k)
        accs[(name, "linear")] = train(linear, Xtr, ytr, Xte, yte)

        torch.manual_seed(0)
        mlp = nn.Sequential(nn.Linear(2, 32), nn.ReLU(),
                            nn.Linear(32, 32), nn.ReLU(),
                            nn.Linear(32, k))
        accs[(name, "mlp")] = train(mlp, Xtr, ytr, Xte, yte)

        lo = Xte.min(0) - 0.45
        hi = Xte.max(0) + 0.45
        gx, gy = np.meshgrid(np.linspace(lo[0], hi[0], 320),
                             np.linspace(lo[1], hi[1], 320))
        grid = torch.from_numpy(np.c_[gx.ravel(), gy.ravel()]
                                .astype(np.float32))

        for row, (model, tag) in enumerate(((linear, "linear"), (mlp, "mlp"))):
            ax = axes[row, col]
            with torch.no_grad():
                Z = model(grid).argmax(1).numpy().reshape(gx.shape)
            ax.contourf(gx, gy, Z, levels=np.arange(-0.5, k, 1.0),
                        colors=face[:k], alpha=0.95)
            ax.contour(gx, gy, Z, levels=np.arange(0.5, k - 0.5 + 1e-9, 1.0),
                       colors="0.25", linewidths=1.6)
            for c in range(k):
                m = yte == c
                ax.scatter(Xte[m, 0], Xte[m, 1], s=9, c=edge[c], alpha=0.62,
                           edgecolors="none")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"{accs[(name, tag)]:.2f}%", fontsize=13,
                         color="#b91c1c" if accs[(name, tag)] < 60 else "0.15")
            if col == 0:
                ax.set_ylabel("softmax regression\n(one linear map)"
                              if row == 0 else
                              "MLP\n(two hidden layers, ReLU)", fontsize=11.5)

    for col, name in enumerate(names):
        axes[0, col].text(0.5, 1.20, name, transform=axes[0, col].transAxes,
                          ha="center", fontsize=14)

    axes[1, 2].annotate("only a curved region can hold a ring",
                        (0.5, 0.06), xycoords="axes fraction", ha="center",
                        fontsize=11, color="#15803d")
    axes[0, 2].annotate("a straight cut through a ring is a coin flip",
                        (0.40, 0.05), xycoords="axes fraction", ha="center",
                        fontsize=11, color="#b91c1c")

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig("softmax_limits.svg", transparent=True)
    plt.close(fig)
    print("wrote softmax_limits.svg")
    for name in names:
        print(f"   {name:<8} linear {accs[(name, 'linear')]:.2f}%   "
              f"mlp {accs[(name, 'mlp')]:.2f}%")


def fig_least_squares_projection():
    """L2 로스 쪽 5절의 ASCII 그림을 실제 수로 다시 그린다.

    7절의 「닫힌 해와 경사 하강법이 같은 곳에 이르는지 보기」 데이터
    (torch seed 0, N=100, y = 2x + 1 + 0.1 잡음)를 그대로 쓴다.
    정규 방정식의 해는 그 절의 출력과 같은 [1.9881, 1.016] 이고
    X^T r 도 같은 -6.65e-05, -1.34e-05 가 나온다.

    왼쪽 그림의 **각도만** 과장했다 - 실제로 잰 각은 2.57도라서 그대로
    그리면 삼각형이 보이지 않는다. 그 사실을 그림 안에 적어 둔다.
    길이와 R^2 는 모두 잰 값이다.
    """
    import torch
    from matplotlib.patches import Polygon

    torch.manual_seed(0)
    N = 100
    x = torch.randn(N, 1)
    X = torch.cat([x, torch.ones(N, 1)], dim=1)
    theta_true = torch.tensor([[2.0], [1.0]])
    y = X @ theta_true + 0.1 * torch.randn(N, 1)

    theta = torch.linalg.solve(X.T @ X, X.T @ y)
    y_hat = X @ theta
    r = y - y_hat
    xtr = (X.T @ r).numpy().ravel()

    ybar = float(y.mean())
    sst = float(((y - ybar) ** 2).sum())
    ssr = float(((y_hat - ybar) ** 2).sum())
    sse = float((r ** 2).sum())
    r2 = ssr / sst
    ang = np.degrees(np.arccos(np.sqrt(r2)))

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.8, 4.9))

    # --- 왼쪽: 사영 그림 (각도만 과장) -----------------------------------
    draw = np.radians(25.0)
    L = 5.6
    foot = np.array([L * np.cos(draw), 0.0])
    tip = np.array([L * np.cos(draw), L * np.sin(draw)])

    axL.add_patch(Polygon([(-1.5, -0.95), (6.6, -0.95), (7.8, 0.75),
                           (-0.3, 0.75)], closed=True, facecolor="#dbeafe",
                          edgecolor="#1d4ed8", lw=1.6, alpha=0.7))
    axL.text(8.0, -1.45, "the shaded plane is the column space of X - "
             "every prediction the model can make", fontsize=10,
             color="#1d4ed8", ha="right", va="bottom")

    axL.annotate("", tip, (0, 0), arrowprops=dict(arrowstyle="-|>", lw=2.6,
                                                  color="#b91c1c"))
    axL.annotate("", foot, (0, 0), arrowprops=dict(arrowstyle="-|>", lw=2.6,
                                                   color="#15803d"))
    axL.annotate("", tip, foot, arrowprops=dict(arrowstyle="-|>", lw=2.6,
                                                color="#b45309"))
    axL.plot([foot[0] - 0.34, foot[0] - 0.34, foot[0]],
             [0.0, 0.34, 0.34], color="0.3", lw=1.4)

    axL.text(tip[0] * 0.46, tip[1] * 0.56 + 0.22,
             f"y, centred\nlength {np.sqrt(sst):.4f}", fontsize=10.5,
             color="#b91c1c", ha="center")
    axL.text(foot[0] * 0.40, -0.70,
             f"y-hat = X theta*\nlength {np.sqrt(ssr):.4f}", fontsize=10.5,
             color="#15803d", ha="center")
    axL.text(tip[0] + 0.18, tip[1] * 0.5,
             f"residual r\nlength {np.sqrt(sse):.4f}", fontsize=10.5,
             color="#b45309")
    axL.annotate(f"the angle is drawn at 25 degrees so the triangle is "
                 f"visible.\nmeasured on this data it is {ang:.2f} degrees, "
                 f"and R-squared = cos-squared = {r2:.6f}",
                 (0.5, -0.17), xycoords="axes fraction", ha="center",
                 fontsize=10, color="0.3")
    axL.set_xlim(-1.9, 8.6)
    axL.set_ylim(-1.55, 2.55)
    axL.set_aspect("equal")
    axL.axis("off")
    axL.set_title("least squares is a shadow cast straight down",
                  fontsize=11.5)

    # --- 오른쪽: 직교를 데이터로 ----------------------------------------
    xn = x.numpy().ravel()
    rn = r.numpy().ravel()
    axR.axhline(0, color="0.4", lw=1.2)
    axR.scatter(xn, rn, s=20, c="#b45309", alpha=0.7, edgecolors="none")
    axR.annotate(f"sum of residuals  =  {xtr[1]:.2e}\n"
                 f"sum of x times residual  =  {xtr[0]:.2e}\n"
                 "both columns of X are orthogonal to r",
                 (0.03, 0.04), xycoords="axes fraction", fontsize=10.5,
                 color="0.2")
    axR.set_xlabel("x", fontsize=11)
    axR.set_ylabel("residual", fontsize=11)
    axR.set_title("no tilt and no offset left over", fontsize=11.5)
    axR.grid(alpha=0.2)
    axR.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig("least_squares_projection.svg", transparent=True)
    plt.close(fig)
    print("wrote least_squares_projection.svg")
    print(f"   theta* = {[round(v, 4) for v in theta.flatten().tolist()]}")
    print(f"   X^T r = {xtr.tolist()}")
    print(f"   SST = {sst:.4f}  SSR = {ssr:.4f}  SSE = {sse:.4f}  "
          f"SSR+SSE = {ssr + sse:.4f}")
    print(f"   R^2 = {r2:.6f}  angle = {ang:.4f} degrees")


def fig_kl_both_directions():
    """KL 발산 쪽 3절의 손셈을 축 위에 올린다.

    f = N(1, 1) 을 고정하고 g = N(0, s) 의 분산 s 를 움직이며 8절의
    단변량 공식으로 두 방향을 모두 계산한다. 3절이 손으로 구한 네 수
    (s=1 에서 양쪽 0.5, s=2 에서 0.3466 과 0.6534) 가 그대로 나온다.

    잰 것이 아니라 공식을 그린 것이므로 데이터는 없다.
    """
    s = np.linspace(0.18, 5.0, 600)          # g 의 분산
    sig = np.sqrt(s)
    d_fg = np.log(sig) + 1.0 / s - 0.5        # D(f || g)
    d_gf = -np.log(sig) + s / 2.0             # D(g || f)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.8, 4.6))

    xx = np.linspace(-5, 6, 600)
    norm = lambda x, m, v: np.exp(-(x - m) ** 2 / (2 * v)) / np.sqrt(2 * np.pi * v)
    axL.plot(xx, norm(xx, 1, 1), color="#b91c1c", lw=2.6,
             label="f = N(1, 1), fixed")
    for v, st, lab in ((1.0, "--", "g = N(0, 1):  both directions 0.5000"),
                       (2.0, "-", "g = N(0, 2):  0.3466 and 0.6534")):
        axL.plot(xx, norm(xx, 0, v), color="#1d4ed8", lw=2.2, ls=st,
                 alpha=0.95 if v == 2 else 0.55, label=lab)
    axL.set_ylim(0, 0.52)
    axL.set_xlabel("x", fontsize=11)
    axL.set_ylabel("density", fontsize=11)
    axL.set_title("same width, or not", fontsize=11.5)
    axL.legend(fontsize=9.5, frameon=False, loc="upper right")
    axL.grid(alpha=0.2)
    axL.spines[["top", "right"]].set_visible(False)

    axR.plot(s, d_fg, color="#b91c1c", lw=2.6, label="D(f || g)")
    axR.plot(s, d_gf, color="#1d4ed8", lw=2.6, label="D(g || f)")
    axR.plot(1.0, 0.5, "o", ms=11, color="#15803d", zorder=5)
    axR.annotate("the one place they agree:\nvariance 1, both 0.5000",
                 (1.0, 0.5), (34, 54), textcoords="offset points",
                 fontsize=10, color="#15803d",
                 arrowprops=dict(arrowstyle="->", color="#15803d", lw=1.2))
    axR.axvline(2.0, color="0.55", ls=":", lw=1.4)
    axR.plot([2.0, 2.0], [np.log(np.sqrt(2)) + 0.5 - 0.5, 1 - np.log(np.sqrt(2))],
             "o", ms=8, color="0.35")
    axR.annotate("variance 2:\n0.3466 and 0.6534",
                 (2.0, np.log(np.sqrt(2))), (26, -36),
                 textcoords="offset points", fontsize=10, color="0.25",
                 arrowprops=dict(arrowstyle="->", color="0.35", lw=1.1))
    axR.set_xlabel("the variance of g  (f stays N(1, 1))", fontsize=11)
    axR.set_ylabel("KL divergence, in nats", fontsize=11)
    axR.set_ylim(0, 2.6)
    axR.set_title("the two directions cross exactly once", fontsize=11.5)
    axR.legend(fontsize=11, frameon=False, loc="upper left")
    axR.grid(alpha=0.2)
    axR.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig("kl_both_directions.svg", transparent=True)
    plt.close(fig)
    print("wrote kl_both_directions.svg")
    for v in (0.5, 1.0, 2.0, 4.0):
        a = np.log(np.sqrt(v)) + 1.0 / v - 0.5
        b = -np.log(np.sqrt(v)) + v / 2.0
        print(f"   var(g) = {v:<4} D(f||g) = {a:.4f}   D(g||f) = {b:.4f}")


def fig_leverage_vs_influence():
    """해석적 해 쪽 4.5절의 지렛값을 5.4절 데이터에서 재어 본다.

    5.4절의 make_regression(n_samples=200, n_features=3, noise=10,
    random_state=42) 과 같은 분할(test_size=0.2, random_state=42)을 쓰므로
    학습 표본은 160개, 바이어스를 포함한 열은 4개다.

    4.5절은 「지렛값이 큰 점은 적합에 불균형하게 큰 영향을 준다」고 적는다.
    점을 하나씩 빼고 다시 풀어 theta 가 얼마나 움직이는지 재면 그 상관이
    0.11 밖에 되지 않는다. 잔차와의 상관은 0.87 이다.
    """
    import numpy.linalg as la
    from sklearn.datasets import make_regression
    from sklearn.model_selection import train_test_split

    x, y = make_regression(n_samples=200, n_features=3, noise=10,
                           random_state=42)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42)
    X = np.hstack([np.ones((len(x_train), 1)), x_train])
    n, d = X.shape

    A = la.inv(X.T @ X)
    h = np.einsum("ij,jk,ik->i", X, A, X)
    theta = A @ X.T @ y_train
    res = y_train - X @ theta

    shift = np.empty(n)
    for i in range(n):
        m = np.arange(n) != i
        shift[i] = la.norm(la.solve(X[m].T @ X[m], X[m].T @ y_train[m])
                           - theta)

    thr = 2 * d / n
    flag = h > thr
    r_lev = np.corrcoef(shift, h)[0, 1]
    r_res = np.corrcoef(shift, np.abs(res))[0, 1]
    hi, big = int(np.argmax(h)), int(np.argmax(shift))

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.8, 4.7))

    for ax, xs, lab, rr in ((axL, h, "leverage h_ii", r_lev),
                            (axR, np.abs(res), "size of the residual",
                             r_res)):
        ax.scatter(xs[~flag], shift[~flag], s=26, c="#9ca3af", alpha=0.75,
                   edgecolors="none", label="not flagged")
        ax.scatter(xs[flag], shift[flag], s=40, c="#b45309", alpha=0.9,
                   edgecolors="none",
                   label=f"flagged by h > 2(p+1)/n = {thr:.3f}  "
                         f"({int(flag.sum())} points)")
        ax.plot(xs[big], shift[big], "o", ms=20, mfc="none", mec="#b91c1c",
                mew=2.4)
        ax.set_xlabel(lab, fontsize=11)
        ax.set_ylabel("how far theta moves when the point is dropped",
                      fontsize=10.5)
        ax.set_title(f"correlation with influence: {rr:.2f}", fontsize=11.5)
        ax.grid(alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)

    axL.axvline(thr, color="#b45309", ls="--", lw=1.4)
    axL.annotate(f"the most influential point\nhas middling leverage "
                 f"{h[big]:.3f}\nand a residual of {res[big]:.1f}",
                 (h[big], shift[big]), (40, -16), textcoords="offset points",
                 fontsize=10, color="#b91c1c",
                 arrowprops=dict(arrowstyle="->", color="#b91c1c", lw=1.2))
    axL.annotate(f"the highest leverage of all, {h[hi]:.3f},\nmoves theta "
                 f"by {shift[hi]:.3f} - below average",
                 (h[hi], shift[hi]), (-16, 56), ha="right",
                 textcoords="offset points", fontsize=10, color="0.25",
                 arrowprops=dict(arrowstyle="->", color="0.4", lw=1.1))
    axR.legend(fontsize=9.5, frameon=False, loc="upper left")

    fig.tight_layout()
    fig.savefig("leverage_vs_influence.svg", transparent=True)
    plt.close(fig)
    print("wrote leverage_vs_influence.svg")
    print(f"   n = {n}, columns = {d}, sum of h = {h.sum():.6f}")
    print(f"   h ranges {h.min():.5f} .. {h.max():.5f}  (1/n = {1 / n:.5f})")
    print(f"   rule of thumb {thr:.3f} flags {int(flag.sum())} points")
    print(f"   corr(shift, leverage)  = {r_lev:.4f}")
    print(f"   corr(shift, |residual|) = {r_res:.4f}")
    print(f"   highest leverage  h={h[hi]:.4f} r={res[hi]:+.3f} "
          f"shift={shift[hi]:.4f}")
    print(f"   most influential  h={h[big]:.4f} r={res[big]:+.3f} "
          f"shift={shift[big]:.4f}")
    print(f"   mean shift = {shift.mean():.4f}")
    top10 = np.argsort(shift)[-10:]
    print(f"   of the 10 most influential points, "
          f"{int(flag[top10].sum())} are flagged")


def fig_loglik_baselines():
    """이진 분류 쪽 9절의 4부를 그대로 다시 돌린다.

    쪽의 스크립트는 2부에서 nn.Linear(1, 1) 을 한 번 만든다. 그 초기화가
    난수를 쓰므로, 그 줄까지 재현해야 3부의 데이터가 쪽의 출력과 같아진다
    (양성 63 / 음성 37, 에폭 40 에서 로그우도 -47.354, 80 부터 -47.341,
    정확도 0.6375).

    8절의 표는 「무작위로 찍는 것」의 기준으로 -n log 2 하나만 든다.
    이 데이터는 한쪽이 많으므로 절편만 둔 모델도 함께 그려 견준다.
    """
    import torch
    import torch.nn as nn
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    torch.manual_seed(42)
    np.random.seed(42)

    glm = nn.Linear(1, 1)                 # 쪽 2부 - 난수를 쓴다
    glm.weight.data = torch.tensor([[2.0]])
    glm.bias.data = torch.tensor([-1.0])

    n_samples = 100
    X = torch.randn(n_samples, 2)
    true_beta = torch.tensor([0.5, 1.0, -0.5])
    X_with_bias = torch.cat([torch.ones(n_samples, 1), X], dim=1)
    y = torch.bernoulli(torch.sigmoid(X_with_bias @ true_beta))

    X_train, X_test, y_train, y_test = train_test_split(
        X.numpy(), y.numpy(), test_size=0.2, random_state=42)
    scaler = StandardScaler()
    Xtr_raw = X_train
    Xtr = torch.FloatTensor(scaler.fit_transform(X_train))
    ytr = torch.FloatTensor(y_train).reshape(-1, 1)
    n = len(ytr)
    base = float(ytr.mean())

    class LogisticRegressionML(nn.Module):
        def __init__(self, n_features):
            super().__init__()
            self.linear = nn.Linear(n_features, 1)

        def forward(self, x):
            return torch.sigmoid(self.linear(x))

        def log_likelihood(self, x, yy):
            z = self.linear(x)
            return (yy * -torch.log1p(torch.exp(-z))
                    + (1 - yy) * -torch.log1p(torch.exp(z))).sum()

    model = LogisticRegressionML(2)
    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
    ll_hist, acc_hist = [], []
    for _ in range(200):
        pred = model(Xtr)
        loss = criterion(pred, ytr)
        with torch.no_grad():
            ll_hist.append(model.log_likelihood(Xtr, ytr).item())
            acc_hist.append(float(((pred >= 0.5).float() == ytr)
                                  .float().mean()))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    ll_chance = -n * np.log(2)
    ll_base = n * (base * np.log(base) + (1 - base) * np.log(1 - base))
    Xraw = torch.cat([torch.ones(n, 1), torch.FloatTensor(Xtr_raw)], dim=1)
    z = Xraw @ true_beta
    yv = ytr.squeeze()
    ll_true = float((yv * -torch.log1p(torch.exp(-z))
                     + (1 - yv) * -torch.log1p(torch.exp(z))).sum())

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.8, 4.7))

    axL.plot(ll_hist, color="#1d4ed8", lw=2.6, label="the model being trained")
    for val, col, lab in (
            (ll_chance, "0.45", f"beta = 0, which is -n log 2 = {ll_chance:.2f}"),
            (ll_base, "#b45309", f"intercept only = {ll_base:.2f}"),
            (ll_true, "#15803d", f"the beta that made the data = {ll_true:.2f}")):
        axL.axhline(val, color=col, ls="--", lw=1.7, label=lab)
    axL.annotate(f"settles at {ll_hist[-1]:.2f}", (199, ll_hist[-1]),
                 (-12, -36), ha="right", textcoords="offset points",
                 fontsize=10, color="#1d4ed8",
                 arrowprops=dict(arrowstyle="->", color="#1d4ed8", lw=1.2))
    axL.set_xlabel("epoch", fontsize=11)
    axL.set_ylabel("log-likelihood on the 80 training rows", fontsize=11)
    axL.set_title("three lines worth measuring against", fontsize=11.5)
    axL.legend(fontsize=9.5, frameon=False, loc="lower right")
    axL.grid(alpha=0.2)
    axL.spines[["top", "right"]].set_visible(False)

    axR.plot(acc_hist, color="#1d4ed8", lw=2.6, label="the model")
    axR.axhline(max(base, 1 - base), color="#b45309", ls="--", lw=1.7,
                label=f"always answer the commoner class "
                      f"({max(base, 1 - base):.4f})")
    const = len(acc_hist)
    while const > 0 and abs(acc_hist[const - 1] - acc_hist[-1]) < 1e-9:
        const -= 1
    axR.annotate(f"frozen from epoch {const + 1} at {acc_hist[-1]:.4f},\n"
                 "while the log-likelihood is still climbing",
                 (120, acc_hist[-1]), (-10, -72), ha="center",
                 textcoords="offset points", fontsize=10, color="#1d4ed8",
                 arrowprops=dict(arrowstyle="->", color="#1d4ed8", lw=1.2))
    axR.set_ylim(0.4, 0.78)
    axR.set_xlabel("epoch", fontsize=11)
    axR.set_ylabel("training accuracy", fontsize=11)
    axR.set_title("accuracy is the coarser instrument", fontsize=11.5)
    axR.legend(fontsize=10, frameon=False, loc="lower right")
    axR.grid(alpha=0.2)
    axR.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig("loglik_baselines.svg", transparent=True)
    plt.close(fig)
    print("wrote loglik_baselines.svg")
    print(f"   class distribution: {int(y.sum())} positive, "
          f"{int((1 - y).sum())} negative")
    print(f"   train n = {n}, positives = {int(ytr.sum())}, "
          f"base rate = {base:.4f}")
    print(f"   LL at beta = 0        : {ll_chance:.3f}")
    print(f"   LL intercept only     : {ll_base:.3f}")
    print(f"   LL at the true beta   : {ll_true:.3f}")
    print(f"   LL fitted             : {ll_hist[-1]:.3f}")
    print(f"   accuracy fitted {acc_hist[-1]:.4f} vs majority "
          f"{max(base, 1 - base):.4f}; constant from epoch {const + 1}")


if __name__ == "__main__":
    fig_sigmoid_steepness()
    fig_poly_bias_variance()
    fig_ridge_path()
    fig_lasso_path()
    fig_gd_lr_bound()
    fig_decision_boundary()
    fig_regularization_strength()
    fig_newton_irls()
    fig_capture_recapture_spread()
    fig_likelihood_sharpens()
    fig_mae_vs_mse()
    fig_softmax_limits()
    fig_least_squares_projection()
    fig_kl_both_directions()
    fig_leverage_vs_influence()
    fig_loglik_baselines()
