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


if __name__ == "__main__":
    mean_field_shrinks()
