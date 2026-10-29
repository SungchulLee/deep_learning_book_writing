"""18.3 계층 모델과 수축 쪽들의 그림.

규칙은 CLAUDE.md 의 Figures 항에 적어 둔 대로다.
- SVG, fonttype='path', transparent
- 글자는 모두 아스키 (matplotlib 기본 폰트에 한글이 없다)

그림마다 **제 주장을 스스로 확인하고**, 어긋나면 그리지 않고 멈춘다.

실행: python docs/ch17/hierarchical/figures/make_figures.py
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


# === 그림으로 보는 AI: 사전분포를 적지 않고 사후평균을 얻는다 =================
#
# Robbins 1956. 푸아송 관측에서 E[theta|x] = (x+1) f(x+1) / f(x) 인데 f 는
# **관측된 도수**다. 사전분포의 꼴을 쓰지 않는다. 여기서는 참 사전분포를
# 감마로 두어 정답을 알 수 있게 해 두고, 로빈스 공식이 그것을 맞히는지 본다.

def robbins():
    rng = np.random.default_rng(0)
    N = 200000
    a, b = 2.0, 1.5                       # 참 사전분포 — 공식은 이것을 모른다
    theta = rng.gamma(a, 1 / b, N)
    x = rng.poisson(theta)

    cnt = np.bincount(x, minlength=x.max() + 2).astype(float)
    ks = np.arange(0, 8)
    rob = np.array([(k + 1) * cnt[k + 1] / cnt[k] if cnt[k] > 0 else np.nan
                    for k in ks])
    true = (a + ks) / (b + 1)             # 감마-푸아송 켤레의 정확한 답

    # 그림이 제 주장을 확인한다
    assert np.nanmax(np.abs(rob[:5] - true[:5])) < 0.05, "앞쪽은 잘 맞아야 한다"
    assert abs(rob[6] - true[6]) > abs(rob[1] - true[1]), "꼬리에서 나빠진다"

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.4, 5.4), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1]})

    ax1.plot(ks, true, "s--", color=INK, linewidth=1.9, ms=7,
             label="the exact answer (needs the prior)")
    ax1.plot(ks, rob, "o-", color=ACCENT, linewidth=2.1, ms=7,
             label="Robbins, from the counts alone")
    for k in (0, 1, 6):
        ax1.annotate(f"{rob[k]:.4f}\nvs {true[k]:.4f}", xy=(k, rob[k]),
                     xytext=(-6, 16 if k < 6 else -40),
                     textcoords="offset points", fontsize=7.8, color=INK)
    ax1.set_ylabel("posterior mean of theta", fontsize=9.5)
    ax1.set_title("no prior was ever written down", fontsize=10.5, color=INK,
                  weight="bold", pad=8)
    ax1.legend(fontsize=8.4, loc="upper left", framealpha=0.92)
    ax1.grid(alpha=0.12)

    ax2.bar(ks, cnt[:len(ks)], color=MUTED, alpha=0.7, width=0.6)
    ax2.set_yscale("log")
    ax2.set_xlabel("observed count x", fontsize=9.5)
    ax2.set_ylabel("how many\nsuch cases", fontsize=9)
    ax2.grid(alpha=0.12, axis="y")
    ax2.annotate(f"only {int(cnt[6]):,}", xy=(6, cnt[6]), xytext=(-18, 16),
                 textcoords="offset points", fontsize=8, color=MUTED)

    for ax in (ax1, ax2):
        ax.tick_params(labelsize=8, colors=INK)
        for s in ax.spines.values():
            s.set_color(INK)
            s.set_linewidth(0.7)

    fig.text(0.5, -0.03,
             f"{N:,} draws. the formula tracks the truth while the counts are "
             f"thick, and drifts once they thin out.",
             ha="center", fontsize=8.8, style="italic", color=INK)
    fig.tight_layout()
    fig.savefig(HERE / "robbins.svg", transparent=True, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote robbins.svg   x=0 {rob[0]:.4f}/{true[0]:.4f}  "
          f"x=6 {rob[6]:.4f}/{true[6]:.4f}")


# === 그림으로 보는 AI: 여덟 학교에서 tau 의 최대우도 추정이 0 이다 ============
#
# Rubin 1981 의 자료. 하이퍼파라미터 tau 를 주변우도로 점추정하면 정확히 0 이
# 나오고, 그것을 대입하는 경험적 베이즈는 여덟 학교를 **같은 수 하나**로
# 뭉갠다. tau 의 불확실성을 적분하는 완전 베이즈는 그러지 않는다.

def eight_schools():
    y = np.array([28, 8, -3, 7, -1, 1, 18, 12.])
    s = np.array([15, 10, 16, 11, 9, 11, 10, 18.])
    names = [chr(65 + i) for i in range(8)]

    taus = np.linspace(1e-6, 40, 4000)

    def lml(t):                       # mu 를 평평한 사전분포로 적분해 없앤다
        v = s ** 2 + t ** 2
        w = 1 / v
        mu = (w * y).sum() / w.sum()
        return (-0.5 * np.sum(np.log(v)) - 0.5 * np.sum((y - mu) ** 2 / v)
                - 0.5 * np.log(w.sum()))

    L = np.array([lml(t) for t in taus])
    L -= L.max()
    P = np.exp(L)
    P /= np.trapz(P, taus)
    tau_hat = taus[int(np.argmax(L))]

    w = 1 / (s ** 2 + tau_hat ** 2)
    eb = np.full(8, (w * y).sum() / w.sum())     # tau_hat = 0 이면 완전 풀링

    full = np.zeros(8)
    dt = taus[1] - taus[0]
    for t, pr in zip(taus, P):
        ww = 1 / (s ** 2 + t ** 2)
        m = (ww * y).sum() / ww.sum()
        full += pr * dt * ((y / s ** 2 + m / t ** 2)
                           / (1 / s ** 2 + 1 / t ** 2))

    # 그림이 제 주장을 확인한다
    assert tau_hat < 1e-3, "tau 의 최대우도 추정은 0 이어야 이야기가 선다"
    assert np.ptp(eb) < 1e-9, "경험적 베이즈는 여덟을 같은 수로 뭉갠다"
    assert np.ptp(full) > 4, "완전 베이즈는 차이를 남긴다"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.8, 4.3),
                                   gridspec_kw={"width_ratios": [1, 1.25]})

    ax1.plot(taus, P, color=INK, linewidth=2.0)
    ax1.fill_between(taus, 0, P, color=ACCENT, alpha=0.18)
    ax1.axvline(tau_hat, color=ACCENT, linewidth=2.0)
    ax1.annotate("the maximising value\nis exactly zero",
                 xy=(tau_hat, P.max() * 0.55), xytext=(9, P.max() * 0.72),
                 fontsize=8.4, color=ACCENT,
                 arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.0))
    ax1.set_xlabel("tau, the spread between schools", fontsize=9.5)
    ax1.set_ylabel("marginal posterior", fontsize=9.5)
    ax1.set_title("what the data say about tau", fontsize=10, color=INK,
                  weight="bold")
    ax1.set_xlim(0, 35)
    ax1.set_ylim(bottom=0)

    ypos = np.arange(8)[::-1]
    for i in range(8):
        lo = min(y[i], full[i], eb[i])
        hi = max(y[i], full[i], eb[i])
        ax2.plot([lo, hi], [ypos[i], ypos[i]], color=INK, linewidth=0.8,
                 alpha=0.3, zorder=1)
    ax2.plot(y, ypos, "o", color=MUTED, ms=9, label="observed",
             markeredgecolor="white", markeredgewidth=1.1, zorder=5)
    ax2.plot(full, ypos, "D", color=INK, ms=7, label="full Bayes",
             markeredgecolor="white", markeredgewidth=1.0, zorder=5)
    ax2.plot(eb, ypos, "s", color=ACCENT, ms=8,
             label="empirical Bayes at tau = 0", markeredgecolor="white",
             markeredgewidth=1.0, zorder=6)
    ax2.set_yticks(ypos)
    ax2.set_yticklabels(names, fontsize=9)
    ax2.set_xlabel("estimated coaching effect", fontsize=9.5)
    ax2.set_title("every school gets the same number", fontsize=10,
                  color=INK, weight="bold")
    ax2.legend(fontsize=8.2, loc="lower left", framealpha=0.93)
    ax2.annotate(f"{eb[0]:.2f} for all eight",
                 xy=(eb[0], ypos[2]), xytext=(18, ypos[2] + 1.15),
                 fontsize=8.6, color=ACCENT, weight="bold",
                 arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.0))
    ax2.set_ylim(-0.9, 7.9)

    for ax in (ax1, ax2):
        ax.grid(alpha=0.12)
        ax.tick_params(labelsize=8, colors=INK)
        for sp in ax.spines.values():
            sp.set_color(INK)
            sp.set_linewidth(0.7)

    fig.text(0.5, -0.04,
             "Rubin's eight schools. plugging in a point estimate of tau "
             "throws away the only thing that keeps the schools apart.",
             ha="center", fontsize=8.8, style="italic", color=INK)
    fig.tight_layout()
    fig.savefig(HERE / "eight_schools.svg", transparent=True,
                bbox_inches="tight")
    plt.close(fig)
    print(f"wrote eight_schools.svg   tau_hat {tau_hat:.4f}  eb {eb[0]:.2f}  "
          f"full A {full[0]:.2f} E {full[4]:.2f}")


# === 그림으로 보는 AI: 교환가능성이 깨지면 누가 대가를 치르는가 ==============
#
# 교환가능성이 성립하면 부분 풀링이 모두를 조금씩 돕는다. 한 그룹만 딴 데서
# 왔다면 **그 그룹이 홀로 크게 손해**를 보고 나머지는 거의 이득이 없다.
# 부분 풀링에 참 tau 를 알려 주는 유리한 조건에서도 그렇다.

def exchangeability():
    rng = np.random.default_rng(0)
    K, n, sig, tau, R = 10, 8, 1.0, 1.0, 4000

    def trial(outlier):
        th = rng.normal(0, tau, K)
        if outlier:
            th[0] = 8.0                      # 한 그룹만 전혀 다른 데서 왔다
        y = th[:, None] + rng.normal(0, sig, (K, n))
        ybar = y.mean(1)
        se2 = sig ** 2 / n
        w = tau ** 2 / (tau ** 2 + se2)      # 참 tau 를 알려 준다 (유리하게)
        pp = ybar.mean() + w * (ybar - ybar.mean())
        return np.abs(ybar - th), np.abs(pp - th)

    out = {}
    for key in (False, True):
        a = np.array([trial(key) for _ in range(R)])
        out[key] = (a[:, 0, :], a[:, 1, :])

    ex_np, ex_pp = out[False]
    od_np, od_pp = out[True]

    # 그림이 제 주장을 확인한다
    assert ex_pp.mean() < ex_np.mean(), "교환가능하면 풀링이 도와야 한다"
    assert od_pp[:, 0].mean() > od_np[:, 0].mean() * 2, "딴 그룹은 크게 손해"
    assert od_pp[:, 1:].mean() < od_np[:, 1:].mean(), "나머지는 조금 이득"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.6, 4.1), sharey=True)
    for ax, (a_np, a_pp), title, first in (
            (ax1, (ex_np, ex_pp), "all groups exchangeable", "group 1"),
            (ax2, (od_np, od_pp), "one group came from elsewhere",
             "the odd group")):
        labels = [first, "the other nine"]   # 왼쪽 판에는 '딴' 그룹이 없다
        npv = [a_np[:, 0].mean(), a_np[:, 1:].mean()]
        ppv = [a_pp[:, 0].mean(), a_pp[:, 1:].mean()]
        xp = np.arange(2)
        ax.bar(xp - 0.19, npv, 0.36, color=MUTED, alpha=0.85,
               label="no pooling")
        ax.bar(xp + 0.19, ppv, 0.36, color=ACCENT, alpha=0.85,
               label="partial pooling")
        for x, (a, b) in enumerate(zip(npv, ppv)):
            ax.text(x - 0.19, a + 0.015, f"{a:.3f}", ha="center",
                    fontsize=8.2, color=INK)
            ax.text(x + 0.19, b + 0.015, f"{b:.3f}", ha="center",
                    fontsize=8.2, color=ACCENT,
                    weight="bold" if b > a else "normal")
        ax.set_xticks(xp)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_title(title, fontsize=10, color=INK, weight="bold")
        ax.grid(alpha=0.12, axis="y")
        ax.tick_params(labelsize=8, colors=INK)
        for sp in ax.spines.values():
            sp.set_color(INK)
            sp.set_linewidth(0.7)
    ax1.set_ylabel("mean absolute error", fontsize=9.5)
    ax1.legend(fontsize=8.4, loc="upper left", framealpha=0.93)
    ax1.set_ylim(0, 0.95)

    fig.text(0.5, -0.04,
             f"{R:,} runs, {K} groups of {n}. partial pooling was even given "
             f"the true tau, which flatters it.",
             ha="center", fontsize=8.8, style="italic", color=INK)
    fig.tight_layout()
    fig.savefig(HERE / "exchangeability.svg", transparent=True,
                bbox_inches="tight")
    plt.close(fig)
    print(f"wrote exchangeability.svg   odd {od_np[:,0].mean():.3f}->"
          f"{od_pp[:,0].mean():.3f}   rest {od_np[:,1:].mean():.3f}->"
          f"{od_pp[:,1:].mean():.3f}")


if __name__ == "__main__":
    robbins()
    eight_schools()
    exchangeability()
