"""21장 변분 추론 절의 그림.

규칙은 CLAUDE.md 의 Figures 항에 적어 둔 대로다.
- SVG, fonttype='path', transparent
- 글자는 모두 아스키 (matplotlib 기본 폰트에 한글이 없다)

수는 전부 여기서 실제로 셈하거나 돌려 얻는다.

실행: python docs/ch20/variational_inference/figures/make_figures.py
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
COOL = "#2d6a8f"


def style(ax):
    ax.tick_params(labelsize=8, colors=INK)
    for s in ax.spines.values():
        s.set_color(INK)
        s.set_linewidth(0.7)


def _ellipse(cov, nsig=2.0, n=240):
    """공분산의 등고선 하나를 점렬로."""
    w, V = np.linalg.eigh(cov)
    t = np.linspace(0, 2 * np.pi, n)
    circle = np.c_[np.cos(t), np.sin(t)].T
    return (V @ np.diag(nsig * np.sqrt(w)) @ circle)


# === 그림으로 보는 AI: 이웃을 평균으로 바꾸면 흔들림이 사라진다 ===============
#
# 바이스(1907)는 둘레의 스핀 하나하나 대신 그 **평균**이 만드는 장을 보게 했다.
# 평균장 변분 추론이 사후분포에 하는 일이 꼭 같고, 잃는 것도 같다 — 흔들림이다.
# 오른쪽 수는 닫힌 식 v = 1/Lambda_jj 로 직접 셈한다.

def mean_field_shrinks():
    rho = 0.9
    S = np.array([[1.0, rho], [rho, 1.0]])
    mf_var = 1.0 / np.diag(np.linalg.inv(S))
    assert np.allclose(mf_var, 1 - rho ** 2), "닫힌 식과 어긋나면 그림이 틀린다"
    Q = np.diag(mf_var)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.2, 4.3))

    # --- 왼쪽: 바이스의 바꿔치기 (꼴만) ------------------------------------
    rng = np.random.default_rng(3)
    xs, ys = np.meshgrid(np.arange(4), np.arange(4))
    ang = rng.uniform(0, 2 * np.pi, xs.size)
    ax1.quiver(xs.ravel(), ys.ravel(), np.cos(ang), np.sin(ang),
               color=INK, alpha=0.8, scale=18, width=0.010,
               pivot="mid")
    ax1.text(1.5, -1.05, "neighbours, each one fluctuating",
             ha="center", fontsize=8.2, color=INK)

    ax1.annotate("", xy=(6.1, 1.5), xytext=(4.0, 1.5),
                 arrowprops=dict(arrowstyle="->", color=ACCENT, linewidth=2.0))
    ax1.text(5.05, 1.95, "replace by\ntheir average", ha="center",
             fontsize=8.2, color=ACCENT)

    gx, gy = np.meshgrid(np.arange(4), np.arange(4))
    ax1.quiver(gx.ravel() + 6.8, gy.ravel(), np.ones(gx.size),
               np.zeros(gx.size), color=ACCENT, alpha=0.85, scale=18,
               width=0.010, pivot="mid")
    ax1.text(8.3, -1.05, "one uniform field. no fluctuation left.",
             ha="center", fontsize=8.2, color=ACCENT)

    ax1.set_title("Weiss 1907: the molecular field", fontsize=10,
                  color=INK, weight="bold")
    ax1.text(0.99, 0.02, "schematic", transform=ax1.transAxes, ha="right",
             fontsize=6.5, color=INK, alpha=0.5, style="italic")
    ax1.set_xlim(-1.2, 10.6)
    ax1.set_ylim(-1.8, 3.9)
    ax1.set_aspect("equal")
    ax1.axis("off")

    # --- 오른쪽: 같은 일이 사후분포에 ---------------------------------------
    for nsig, a in ((1.0, 0.95), (2.0, 0.5)):
        e = _ellipse(S, nsig)
        ax2.plot(e[0], e[1], color=INK, linewidth=1.9, alpha=a,
                 label="true posterior" if nsig == 1.0 else None)
        e = _ellipse(Q, nsig)
        ax2.plot(e[0], e[1], color=ACCENT, linewidth=1.9, alpha=a,
                 linestyle="--",
                 label="best mean-field fit" if nsig == 1.0 else None)

    ax2.axhline(0, color=INK, linewidth=0.5, alpha=0.3)
    ax2.axvline(0, color=INK, linewidth=0.5, alpha=0.3)
    ax2.set_title(f"the same trade, rho = {rho}", fontsize=10, color=INK,
                  weight="bold")
    ax2.set_xlabel("theta 1", fontsize=9)
    ax2.set_ylabel("theta 2", fontsize=9)
    ax2.legend(fontsize=8.2, loc="upper left", framealpha=0.92)
    ax2.set_aspect("equal")
    ax2.set_xlim(-3.4, 3.4)
    ax2.set_ylim(-3.4, 3.4)
    ax2.grid(alpha=0.12)
    style(ax2)
    ax2.text(0.5, 0.035,
             f"marginal variance {1.0:.2f} becomes {mf_var[0]:.2f}"
             f"  ({(1-mf_var[0])*100:.0f}% gone)",
             transform=ax2.transAxes, ha="center", fontsize=8.5,
             color=ACCENT)

    fig.text(0.5, -0.03,
             "both pictures drop the same thing: what the neighbours were "
             "doing. the confidence that is left is an artefact.",
             ha="center", fontsize=8.6, style="italic", color=INK)
    fig.tight_layout()
    fig.savefig(HERE / "mean_field_weiss.svg", transparent=True,
                bbox_inches="tight")
    plt.close(fig)
    print(f"wrote mean_field_weiss.svg   rho {rho}   "
          f"MF var {mf_var[0]:.4f} vs true 1.0")




# === 그림으로 보는 AI: 어느 경사가 덜 흔들리는가 — 고정 phi 에서 직접 =========
#
# 쪽 5절이 "학습 중 ELBO 의 이동 분산은 경사 분산이 아니다"라고 적어 두었고,
# 연습 3 이 바로 이 비교 절차를 설계하라고 한다. 여기서 그 절차대로 잰다 —
# **같은 phi, 같은 표본 수**, 독립 반복 4,000 번.
# 기준선은 현재 표본과 분리해서 따로 뽑는다 (쪽 3절의 조건).

_NOBS = 10


def _model(seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(1.0, 1.0, _NOBS)
    return x


def _log_joint(th, x):                      # 상수 생략 (쪽과 같음)
    return -0.5 * th ** 2 - 0.5 * np.sum((x[:, None] - th[None, :]) ** 2, axis=0)


def _log_q(th, mu, sig):
    return -0.5 * np.log(2 * np.pi * sig ** 2) - 0.5 * ((th - mu) / sig) ** 2


def _variances(x, mu, log_sig, S, rep, seed):
    rng = np.random.default_rng(seed)
    sig = np.exp(log_sig)
    out = {k: np.zeros((rep, 2)) for k in ("reparam", "score", "score_bl")}
    for r in range(rep):
        eps = rng.standard_normal(S)
        th = mu + sig * eps
        f = _log_joint(th, x) - _log_q(th, mu, sig)
        dlogp = -th - np.sum(th[None, :] - x[:, None], axis=0)
        out["reparam"][r] = [dlogp.mean(), (dlogp * sig * eps).mean() + 1.0]
        s_mu = (th - mu) / sig ** 2
        s_ls = ((th - mu) ** 2 / sig ** 2) - 1.0
        out["score"][r] = [(f * s_mu).mean(), (f * s_ls).mean()]
        th_b = mu + sig * rng.standard_normal(max(S, 8))
        b = (_log_joint(th_b, x) - _log_q(th_b, mu, sig)).mean()
        out["score_bl"][r] = [((f - b) * s_mu).mean(), ((f - b) * s_ls).mean()]
    return {k: out[k].var(axis=0).sum() for k in out}


def gradient_variance():
    x = _model()
    xbar = x.mean()
    opt = (xbar * _NOBS / (_NOBS + 1), 0.5 * np.log(1 / (_NOBS + 1)))
    off = (0.0, 0.0)
    S = 10

    v_opt = _variances(x, *opt, S, 4000, 1)
    v_off = _variances(x, *off, S, 4000, 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 4.2))

    # --- 왼쪽: 잰 분산 -----------------------------------------------------
    names = ["reparam", "score", "score_bl"]
    labels = ["pathwise", "score function", "score + baseline"]
    xs = np.arange(3)
    w = 0.36
    floor = 1e-6
    a = [max(v_off[k], floor) for k in names]
    b = [max(v_opt[k], floor) for k in names]
    ax1.bar(xs - w / 2, a, w, color=INK, alpha=0.8, label="away from the optimum")
    ax1.bar(xs + w / 2, b, w, color=ACCENT, alpha=0.85, label="at the exact posterior")
    ax1.set_yscale("log")
    ax1.set_xticks(xs)
    ax1.set_xticklabels(labels, fontsize=8.5)
    ax1.set_ylabel("total gradient variance (log)", fontsize=9)
    ax1.set_title(f"same phi, same budget (S = {S}), 4000 repeats",
                  fontsize=9.8, color=INK, weight="bold")
    ax1.set_ylim(floor / 3, max(a + b) * 60)   # 범례가 막대를 덮지 않게
    ax1.legend(fontsize=8, framealpha=0.92, loc="upper left", ncol=1)
    ax1.grid(alpha=0.13, axis="y", which="both")
    style(ax1)
    ax1.annotate("exactly zero", xy=(2 + w / 2, floor),
                 xytext=(-52, 48), textcoords="offset points", ha="right",
                 fontsize=8, color=ACCENT,
                 arrowprops=dict(arrowstyle="->", color=ACCENT, lw=0.9))

    # --- 오른쪽: 왜 그런가 — f 가 평평해진다 --------------------------------
    for (mu, ls), col, lab, st in ((off, INK, "away from the optimum", "-"),
                                   (opt, ACCENT, "at the exact posterior", "-")):
        sig = np.exp(ls)
        th = np.linspace(mu - 3 * sig, mu + 3 * sig, 200)
        f = _log_joint(th, x) - _log_q(th, mu, sig)
        ax2.plot(th, f, color=col, linewidth=2.1, linestyle=st, label=lab)
    ax2.set_xlabel("theta", fontsize=9)
    ax2.set_ylabel("f(theta) = log p(x,theta) - log q(theta)", fontsize=8.6)
    ax2.set_title("the reason: f goes flat", fontsize=9.8, color=INK,
                  weight="bold")
    ax2.legend(fontsize=8, framealpha=0.92, loc="lower left")
    ax2.grid(alpha=0.13)
    style(ax2)

    sig_o = np.exp(opt[1])
    th_o = np.linspace(opt[0] - 3 * sig_o, opt[0] + 3 * sig_o, 200)
    f_o = _log_joint(th_o, x) - _log_q(th_o, opt[0], sig_o)
    spread = f_o.max() - f_o.min()
    assert spread < 1e-10, "정확한 사후에서 f 는 상수여야 한다"
    ax2.text(0.97, 0.52, f"flat to {spread:.0e}\n"
                         "a constant has no variance\nfor a baseline to miss",
             transform=ax2.transAxes, ha="right", va="top", fontsize=7.8,
             color=ACCENT)

    fig.text(0.5, -0.03,
             "pathwise wins by about 7x away from the optimum, and loses "
             "outright at it. neither is always better.",
             ha="center", fontsize=8.6, style="italic", color=INK)
    fig.tight_layout()
    fig.savefig(HERE / "gradient_variance.svg", transparent=True,
                bbox_inches="tight")
    plt.close(fig)
    print(f"wrote gradient_variance.svg   off: reparam {v_off['reparam']:.2f} "
          f"vs score+bl {v_off['score_bl']:.2f}  "
          f"({v_off['score_bl']/v_off['reparam']:.1f}x)   "
          f"opt: reparam {v_opt['reparam']:.4f} vs score+bl {v_opt['score_bl']:.2e}")




# === 그림으로 보는 AI: 두 단계가 한 함수를 번갈아 오른다 ======================
#
# 닐과 힌턴(1998)의 눈으로 EM 을 보면 F(q, theta) 하나뿐이고 E 단계는 q 에 대해,
# M 단계는 theta 에 대해 그것을 최대화한다. 간격이 곧 KL 이고 E 단계가 0 으로
# 만든다. 수는 여기서 직접 돌려 얻는다.

def em_staircase():
    from scipy.stats import norm

    rng = np.random.default_rng(1)
    x = np.concatenate([rng.normal(-2, 1, 80), rng.normal(2.5, 1, 80)])
    SIG, W = 1.0, 0.5

    def loglik(mu):
        return np.log(W * norm.pdf(x, mu[0], SIG)
                      + W * norm.pdf(x, mu[1], SIG)).sum()

    def free_energy(R, mu):
        lj = np.column_stack([np.log(W) + norm.logpdf(x, m, SIG) for m in mu])
        return (R * lj).sum() - (R * np.log(R + 1e-300)).sum()

    mu = np.array([-0.5, 0.6])
    lls, els, kinds = [], [], []
    for _ in range(4):
        lj = np.column_stack([np.log(W) + norm.logpdf(x, m, SIG) for m in mu])
        m_ = lj.max(1, keepdims=True)
        R = np.exp(lj - m_)
        R /= R.sum(1, keepdims=True)
        lls.append(loglik(mu)); els.append(free_energy(R, mu)); kinds.append("E")
        mu = (R * x[:, None]).sum(0) / R.sum(0)
        lls.append(loglik(mu)); els.append(free_energy(R, mu)); kinds.append("M")

    lls, els = np.array(lls), np.array(els)
    gaps = lls - els
    e_gaps = gaps[[i for i, k in enumerate(kinds) if k == "E"]]
    assert np.all(np.abs(e_gaps) < 1e-9), "E 단계 뒤에는 간격이 0 이어야 한다"

    t = np.arange(len(lls))
    fig, ax = plt.subplots(figsize=(8.4, 4.5))
    ax.plot(t, lls, "-", color=INK, linewidth=2.1, label="log p(x)  (the evidence)")
    ax.plot(t, els, "-", color=ACCENT, linewidth=2.1,
            label="F(q, theta)  (the bound)")
    for i, k in enumerate(kinds):
        ax.plot([i], [els[i]], marker="o" if k == "E" else "s",
                color=ACCENT, markersize=6, zorder=4)
        if gaps[i] > 0.05:
            ax.annotate("", xy=(i, lls[i]), xytext=(i, els[i]),
                        arrowprops=dict(arrowstyle="<->", color=COOL, lw=1.3))
            ax.text(i + 0.14, (lls[i] + els[i]) / 2, f"KL = {gaps[i]:.1f}",
                    fontsize=8, color=COOL, va="center")
    ax.set_xticks(t)
    ax.set_xticklabels(kinds, fontsize=8.5)
    ax.set_xlabel("E step closes the gap, M step raises the bound", fontsize=9)
    ax.set_ylabel("value", fontsize=9)
    ax.set_title("one function, two coordinates", fontsize=10, color=INK,
                 weight="bold")
    ax.legend(fontsize=8.6, framealpha=0.92, loc="lower right")
    ax.grid(alpha=0.13)
    style(ax)
    ax.text(0.98, 0.42,
            f"after every E step\nthe gap is 0\n"
            f"(largest seen {np.abs(e_gaps).max():.0e})",
            transform=ax.transAxes, va="top", ha="right", fontsize=8.2,
            color=ACCENT)

    fig.text(0.5, -0.03,
             "the bound touches the evidence only where the E step put it. "
             "a tight bound at one theta says nothing at the next.",
             ha="center", fontsize=8.6, style="italic", color=INK)
    fig.tight_layout()
    fig.savefig(HERE / "em_staircase.svg", transparent=True,
                bbox_inches="tight")
    plt.close(fig)
    print(f"wrote em_staircase.svg   max E-step gap {np.abs(e_gaps).max():.1e}  "
          f"first M gap {gaps[1]:.2f}")


if __name__ == "__main__":
    mean_field_shrinks()
    gradient_variance()
    em_staircase()
