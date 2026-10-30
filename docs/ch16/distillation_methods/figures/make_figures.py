"""17.5 증류 기반 방법 — 「그림으로 보는 AI」 그림.

규칙은 CLAUDE.md 의 Figures 항에 적어 둔 대로다.
- SVG, fonttype='path', transparent
- 글자는 모두 아스키 (matplotlib 기본 폰트에 한글이 없다)

자료를 내려받지 않으므로 어디서 돌려도 된다.

실행: python docs/ch16/distillation_methods/figures/make_figures.py
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


# === 그림으로 보는 AI: T^2 는 관례가 아니라 보정이다 ==========================
#
# 부드러운 목표에서 오는 경사는 1/T^2 로 작아진다 (Hinton 2015). 그래서 T^2 을
# 곱한다. **크기는 되돌아오지만 방향은 되돌아오지 않는다** — 그것이 이 쪽이
# "서로 다른 T 의 최적화가 완전히 같아지는 것은 아니다" 라고 적은 까닭이다.

def temperature_scaling():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    torch.manual_seed(0)
    kl = nn.KLDivLoss(reduction="batchmean")
    n, c = 256, 10
    teacher = torch.randn(n, c) * 2.0
    ts = [1, 2, 3, 4, 6, 8]

    def grad(t, use_t2):
        z = torch.randn(n, c, generator=torch.Generator().manual_seed(7),
                        requires_grad=True)
        with torch.no_grad():
            p = torch.softmax(teacher / t, 1)
        loss = kl(torch.log_softmax(z / t, 1), p)
        if use_t2:
            loss = loss * t * t
        loss.backward()
        return z.grad.clone()

    raw = [grad(float(t), False).norm().item() for t in ts]
    fixed = [grad(float(t), True).norm().item() for t in ts]
    g1 = grad(1.0, True)
    cos = [F.cosine_similarity(g1.flatten(), grad(float(t), True).flatten(),
                               dim=0).item() for t in ts]

    # 1/T^2 를 실제로 따르는가
    ratio = [r / raw[0] * t * t for r, t in zip(raw, ts)]
    assert all(0.8 < v < 1.3 for v in ratio), (
        f"T^2 을 안 곱하면 경사가 1/T^2 로 줄어야 한다 (ratio={ratio})")
    assert max(fixed) / min(fixed) < 1.5, "T^2 을 곱하면 크기가 고르게 남아야 한다"
    assert cos[-1] < 0.9, "그래도 방향은 달라져야 한다"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.6, 4.1))

    # --- 왼쪽: 크기 ---------------------------------------------------------
    ax1.loglog(ts, raw, "s--", color=INK, linewidth=1.9, ms=6,
               label="the soft-target gradient")
    ax1.loglog(ts, [raw[0] / (t * t) for t in ts], ":", color=INK,
               linewidth=1.2, alpha=0.6, label="exactly 1 / T^2")
    ax1.loglog(ts, fixed, "o-", color=ACCENT, linewidth=2.2, ms=6.5,
               label="after multiplying by T^2")
    ax1.set_xlabel("temperature T", fontsize=9.5)
    ax1.set_ylabel("gradient magnitude", fontsize=9.5)
    ax1.set_title("the size comes back", fontsize=10, color=INK,
                  weight="bold")
    ax1.set_xticks(ts)
    ax1.set_xticklabels([str(t) for t in ts])
    ax1.legend(fontsize=8.2, loc="lower left", framealpha=0.9)
    ax1.grid(alpha=0.13, which="both")

    # --- 오른쪽: 방향 -------------------------------------------------------
    ax2.plot(ts, cos, "o-", color=ACCENT, linewidth=2.2, ms=7)
    ax2.axhline(1.0, color=INK, linestyle=":", linewidth=1.2, alpha=0.7)
    ax2.text(ts[-1], 1.012, "same direction", fontsize=8.4, color=INK,
             alpha=0.8, ha="right")
    ax2.annotate(f"{cos[-1]:.2f}", xy=(ts[-1], cos[-1]), xytext=(-8, 8),
                 textcoords="offset points", fontsize=11, color=ACCENT,
                 weight="bold", ha="right")
    ax2.set_xlabel("temperature T", fontsize=9.5)
    ax2.set_ylabel("cosine with the T = 1 gradient", fontsize=9.5)
    ax2.set_title("the direction does not", fontsize=10, color=ACCENT,
                  weight="bold")
    ax2.set_xticks(ts)
    ax2.set_ylim(min(cos) - 0.08, 1.06)
    ax2.grid(alpha=0.13)

    for ax in (ax1, ax2):
        ax.tick_params(labelsize=8.5, colors=INK)
        for sp in ax.spines.values():
            sp.set_color(INK)
            sp.set_linewidth(0.7)

    fig.text(0.5, -0.04,
             "same teacher logits and same student logits throughout. "
             "T^2 restores the scale, not the objective.",
             ha="center", fontsize=8.5, style="italic", color=INK)
    fig.tight_layout()
    fig.savefig(HERE / "temperature_t2.svg", transparent=True,
                bbox_inches="tight")
    plt.close(fig)
    print(f"wrote temperature_t2.svg   raw {raw[0]:.4f}->{raw[-1]:.5f}, "
          f"fixed {fixed[0]:.4f}->{fixed[-1]:.4f}, cos {cos[-1]:.3f}")


if __name__ == "__main__":
    temperature_scaling()
