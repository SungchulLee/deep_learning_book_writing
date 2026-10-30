"""35.5 금융 응용 쪽들의 그림.

규칙은 CLAUDE.md 의 Figures 항에 적어 둔 대로다.
- SVG, fonttype='path', transparent
- 글자는 모두 아스키 (matplotlib 기본 폰트에 한글이 없다)

바깥 시장 자료를 읽지 않는다. 참 분포를 알아야 재앙 확률을 셀 수 있으므로
**만든 자산**을 쓴다. 그림 안에 만든 자료임을 적는다.

실행: python docs/ch34/finance/figures/make_figures.py
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

N_DRAW = 600_000
DISASTER = -0.15


# === 그림으로 보는 AI: 흔들림이 못 보는 꼬리 =================================
#
# 로이 1952 의 안전제일 기준(재앙 수준 아래로 떨어질 확률을 가장 작게)과
# 평균-분산 기준이 **다른 꾸러미**를 고르는 자리를 만든다. B 는 평소 아주
# 얌전한데 드물게 크게 무너진다. 수는 전부 여기서 셈한다.

def safety_first():
    rng = np.random.default_rng(1)
    mu = np.array([0.08, 0.11, 0.07])
    sd = np.array([0.14, 0.05, 0.12])
    x = rng.normal(mu, sd, (N_DRAW, 3))
    crash = rng.random(N_DRAW) < 0.03
    x[crash, 1] -= 0.70

    best_s = best_r = None
    for w1 in np.arange(0, 1.001, 0.02):
        for w2 in np.arange(0, 1.001 - w1 + 1e-9, 0.02):
            w = np.array([w1, w2, 1 - w1 - w2])
            r = x @ w
            sh = r.mean() / r.std()
            pr = float((r < DISASTER).mean())
            if best_s is None or sh > best_s[0]:
                best_s = (sh, w, pr, r.mean())
            if best_r is None or pr < best_r[0]:
                best_r = (pr, w, sh, r.mean())

    sh_s, w_s, p_s, m_s = best_s
    p_r, w_r, sh_r, m_r = best_r

    # 그림이 주장하는 것을 먼저 확인한다
    assert w_s[1] > w_r[1] * 1.5, "두 기준이 B 비중을 비슷하게 고르면 이야기가 없다"
    assert p_r < p_s / 2, "재앙 확률이 크게 줄지 않으면 이야기가 없다"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.8, 4.2),
                                   gridspec_kw={"width_ratios": [1.15, 1]})

    idx = np.arange(3)
    ax1.bar(idx - 0.19, w_s, width=0.36, color=INK, alpha=0.82,
            label="mean-variance (best Sharpe)")
    ax1.bar(idx + 0.19, w_r, width=0.36, color=ACCENT, alpha=0.88,
            label="Roy's safety first")
    ax1.set_xticks(idx)
    ax1.set_xticklabels(["A", "B\n(calm, then a cliff)", "C"], fontsize=9)
    ax1.set_ylabel("weight in the portfolio", fontsize=9.5)
    ax1.set_title("the two criteria pick differently", fontsize=10,
                  color=INK, weight="bold")
    ax1.legend(fontsize=8.4, framealpha=0.92)
    ax1.grid(alpha=0.13, axis="y")

    bars = [p_s, p_r]
    ax2.bar([0, 1], bars, color=[INK, ACCENT], width=0.5, alpha=0.88)
    for i, v in enumerate(bars):
        ax2.text(i, v + max(bars) * 0.03, f"{v * 100:.2f}%", ha="center",
                 fontsize=10.5, color=INK, weight="bold")
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(["mean-variance", "safety first"], fontsize=9)
    ax2.set_ylabel(f"chance of losing more than {abs(DISASTER):.0%}",
                   fontsize=9.5)
    ax2.set_ylim(0, max(bars) * 1.3)
    ax2.set_title(f"{(1 - p_r / p_s) * 100:.0f}% less often", fontsize=10,
                  color=ACCENT, weight="bold")
    ax2.grid(alpha=0.13, axis="y")

    for ax in (ax1, ax2):
        ax.tick_params(labelsize=8, colors=INK)
        for s in ax.spines.values():
            s.set_color(INK)
            s.set_linewidth(0.7)

    fig.text(0.5, -0.06,
             f"made-up assets.  the price is expected return "
             f"{m_s:.4f} -> {m_r:.4f} and Sharpe {sh_s:.3f} -> {sh_r:.3f}.",
             ha="center", fontsize=8.5, style="italic", color=INK)
    fig.tight_layout()
    fig.savefig(HERE / "safety_first.svg", transparent=True,
                bbox_inches="tight")
    plt.close(fig)
    print(f"wrote safety_first.svg   B {w_s[1]:.2f}->{w_r[1]:.2f}   "
          f"disaster {p_s:.4f}->{p_r:.4f}   sharpe {sh_s:.3f}->{sh_r:.3f}")


if __name__ == "__main__":
    safety_first()
