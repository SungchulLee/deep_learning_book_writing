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


if __name__ == "__main__":
    robbins()
