"""26장 변종 절 쪽의 그림.

규칙은 CLAUDE.md 의 Figures 항에 적어 둔 대로다.
- SVG, fonttype='path', transparent
- 글자는 모두 아스키 (matplotlib 기본 폰트에 한글이 없다)

오른쪽 판의 두 수는 **여기서 직접 학습해 재어** 얻는다. 체크포인트를 읽지
않는다 — 씨앗을 박고 작은 오토인코더를 처음부터 돌린다.

실행: python docs/ch25/variants/figures/make_figures.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["svg.fonttype"] = "path"

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
INK = "#2f3337"
ACCENT = "#8a4b2a"
BLUE = "#2d6a8f"


# === 그림으로 보는 AI: 두 야코비안은 같은 것이 아니다 =========================
#
# CAE 가 벌하는 것은 표현의 야코비안 J_f 이고, 작은 가우시안 잡음의 DAE 를
# 펼치면 나오는 것은 재구성의 야코비안 J_{g.f} 다. 두 수를 실제로 재어
# 나란히 둔다.

def _train_small_ae(seed=0, steps=3000):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    n, d, k = 512, 8, 3
    lat = rng.normal(size=(n, k))
    basis = np.linalg.qr(rng.normal(size=(d, d)))[0][:, :k]
    X = np.tanh(lat @ basis.T * 1.4) + rng.normal(scale=0.1, size=(n, d))
    X = torch.tensor(X, dtype=torch.float32)

    enc = nn.Sequential(nn.Linear(d, 16), nn.Sigmoid(), nn.Linear(16, k),
                        nn.Sigmoid())
    dec = nn.Sequential(nn.Linear(k, 16), nn.Sigmoid(), nn.Linear(16, d))
    opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()),
                           lr=0.01)
    for _ in range(steps):
        opt.zero_grad()
        loss = ((dec(enc(X)) - X) ** 2).sum(1).mean()
        loss.backward()
        opt.step()
    return enc, dec, X


def _frob(fn, X, m=128):
    """표본 m 개에서 야코비안 프로베니우스 제곱 노름의 평균."""
    tot = 0.0
    for i in range(m):
        x = X[i:i + 1].clone().requires_grad_(True)
        J = torch.autograd.functional.jacobian(lambda t: fn(t).squeeze(0),
                                               x, vectorize=True)
        tot += (J ** 2).sum().item()
    return tot / m


def two_jacobians():
    enc, dec, X = _train_small_ae()
    jf = _frob(lambda t: enc(t), X)
    jgf = _frob(lambda t: dec(enc(t)), X)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 3.8),
                                   gridspec_kw={"width_ratios": [1.45, 1]})

    # --- 왼쪽: 어느 구간의 야코비안인가 ------------------------------------
    box = dict(boxstyle="round,pad=0.45", linewidth=1.2)
    xs = [0.10, 0.50, 0.90]
    labels = ["x", "z = f(x)", "g(f(x))"]
    for x, lab in zip(xs, labels):
        ax1.text(x, 0.66, lab, ha="center", va="center", fontsize=12,
                 color=INK, transform=ax1.transAxes,
                 bbox=dict(facecolor="white", edgecolor=INK, **box))
    for a, b, lab in ((xs[0], xs[1], "encoder f"), (xs[1], xs[2], "decoder g")):
        ax1.annotate("", xy=(b - 0.09, 0.66), xytext=(a + 0.09, 0.66),
                     xycoords=ax1.transAxes, textcoords=ax1.transAxes,
                     arrowprops=dict(arrowstyle="->", color=INK, lw=1.4))
        ax1.text((a + b) / 2, 0.735, lab, ha="center", fontsize=8.5,
                 color=INK, alpha=0.8, transform=ax1.transAxes)

    # CAE 가 보는 구간
    ax1.annotate("", xy=(xs[1], 0.44), xytext=(xs[0], 0.44),
                 xycoords=ax1.transAxes, textcoords=ax1.transAxes,
                 arrowprops=dict(arrowstyle="-", color=ACCENT, lw=2.6))
    ax1.text((xs[0] + xs[1]) / 2, 0.355, "CAE penalises this one\n"
             "Jacobian of the representation", ha="center", va="top",
             fontsize=8.6, color=ACCENT, transform=ax1.transAxes)

    # 작은 잡음 DAE 를 펼치면 나오는 구간
    ax1.annotate("", xy=(xs[2], 0.16), xytext=(xs[0], 0.16),
                 xycoords=ax1.transAxes, textcoords=ax1.transAxes,
                 arrowprops=dict(arrowstyle="-", color=BLUE, lw=2.6))
    ax1.text((xs[0] + xs[2]) / 2, 0.075, "small-noise DAE expands to this one\n"
             "Jacobian of the reconstruction", ha="center", va="top",
             fontsize=8.6, color=BLUE, transform=ax1.transAxes)

    ax1.set_title("two different stretches of the same path", fontsize=10,
                  color=INK, weight="bold")
    ax1.axis("off")

    # --- 오른쪽: 재어 보면 -------------------------------------------------
    ax2.bar([0, 1], [jf, jgf], color=[ACCENT, BLUE], width=0.55, alpha=0.9)
    for i, v in enumerate((jf, jgf)):
        ax2.text(i, v * 1.04, f"{v:.3f}", ha="center", fontsize=10.5,
                 color=INK, weight="bold")
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(["$||J_f||_F^2$\nrepresentation",
                         "$||J_{g \\circ f}||_F^2$\nreconstruction"],
                        fontsize=9)
    ax2.set_ylabel("mean over 128 training points", fontsize=8.5)
    ax2.set_title("measured on one trained AE", fontsize=10, color=INK,
                  weight="bold")
    ax2.set_ylim(0, max(jf, jgf) * 1.22)
    ax2.grid(alpha=0.13, axis="y")
    ax2.set_axisbelow(True)
    ax2.tick_params(labelsize=8.5, colors=INK)
    for s in ax2.spines.values():
        s.set_color(INK)
        s.set_linewidth(0.7)

    fig.text(0.5, -0.07,
             f"the two numbers differ by a factor of "
             f"{max(jf, jgf) / min(jf, jgf):.1f} - penalising one is not "
             f"penalising the other.",
             ha="center", fontsize=8.4, style="italic", color=INK)
    fig.tight_layout()
    fig.savefig(HERE / "two_jacobians.svg", transparent=True,
                bbox_inches="tight")
    plt.close(fig)
    print(f"wrote two_jacobians.svg   J_f {jf:.4f}   J_gf {jgf:.4f}   "
          f"ratio {max(jf,jgf)/min(jf,jgf):.2f}")


if __name__ == "__main__":
    two_jacobians()
