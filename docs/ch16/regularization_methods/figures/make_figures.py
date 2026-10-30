"""17.3 정규화 기반 방법 — 「그림으로 보는 AI」 그림.

규칙은 CLAUDE.md 의 Figures 항에 적어 둔 대로다.
- SVG, fonttype='path', transparent
- 글자는 모두 아스키 (matplotlib 기본 폰트에 한글이 없다)

자료를 내려받지 않으므로 어디서 돌려도 된다. 30 초쯤 걸린다.

실행: python docs/ch16/regularization_methods/figures/make_figures.py
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


# === 그림으로 보는 AI: 벌점을 따로 두느냐 합치느냐 ============================
#
# EWC 의 벌점을 과제마다 하나씩 두는 꼴과, 피셔를 더해 하나로 두는 꼴은
# **과제가 둘일 때는 똑같다.** 셋부터 갈라진다. 2018 년 PNAS 의 노트와
# 답장이 다툰 자리가 바로 거기다.
#
# 여기서는 어느 쪽이 옳은지 가리지 않는다. **언제부터 달라지는지**만 잰다.

def _run(T, seed, d=12, lam=50.0, steps=500):
    """선형회귀 과제 T 개를 두 가지 벌점 꼴로 이어 배운다."""
    import torch

    torch.manual_seed(seed)
    xs, ys = [], []
    for k in range(T):
        x = torch.randn(60, d)
        x[:, :min(k * 2, d)] *= 0.05      # 앞 과제가 쓰던 좌표는 거의 안 쓴다
        w = torch.randn(d)
        xs.append(x)
        ys.append(x @ w + 0.01 * torch.randn(60))

    out = {}
    for mode in ("sum", "single"):
        th = torch.zeros(d, requires_grad=True)
        anchors, fishers = [], []
        for k in range(T):
            opt = torch.optim.Adam([th], lr=0.05)
            for _ in range(steps):
                opt.zero_grad()
                loss = ((xs[k] @ th - ys[k]) ** 2).mean()
                if anchors:
                    if mode == "sum":          # 논문 그대로: 과제마다 하나씩
                        for f, a in zip(fishers, anchors):
                            loss = loss + lam * (f * (th - a) ** 2).sum()
                    else:                      # 합친 꼴: 피셔를 더하고 기준점 하나
                        fs = torch.stack(fishers).sum(0)
                        loss = loss + lam * (fs * (th - anchors[-1]) ** 2).sum()
                loss.backward()
                opt.step()
            fishers.append((xs[k] ** 2).mean(0))
            anchors.append(th.detach().clone())
        out[mode] = th.detach().clone()

    gap = (out["sum"] - out["single"]).norm() / out["sum"].norm()
    return gap.item() * 100


def two_tasks_agree():
    ts = [2, 3, 4, 5, 6, 8]
    gaps = [float(np.mean([_run(t, s) for s in range(3)])) for t in ts]

    assert gaps[0] < 1e-6, (
        f"과제가 둘일 때는 두 꼴이 같아야 이 그림의 이야기가 성립한다 "
        f"(gap={gaps[0]})")
    assert gaps[1] > gaps[0], "셋부터는 갈라져야 한다"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.8, 4.0),
                                   gridspec_kw={"width_ratios": [1, 1.15]})

    # --- 왼쪽: 두 가지 적는 법 ---------------------------------------------
    ax1.text(0.5, 0.93, "two ways to write the penalty", ha="center",
             fontsize=9, color=INK, style="italic")

    ax1.add_patch(plt.Rectangle((0.04, 0.58), 0.92, 0.22, facecolor=ACCENT,
                                alpha=0.13, edgecolor=ACCENT, linewidth=1.0))
    ax1.text(0.5, 0.715, "one penalty per old task", ha="center",
             va="center", fontsize=9.5, color=ACCENT, weight="bold")
    ax1.text(0.5, 0.625, "sum over k  of  F_k (th - th_k)^2", ha="center",
             va="center", fontsize=9.5, color=INK, family="monospace")

    ax1.add_patch(plt.Rectangle((0.04, 0.24), 0.92, 0.22, facecolor=MUTED,
                                alpha=0.13, edgecolor=MUTED, linewidth=1.0))
    ax1.text(0.5, 0.375, "one penalty, Fishers added", ha="center",
             va="center", fontsize=9.5, color=MUTED, weight="bold")
    ax1.text(0.5, 0.285, "(sum of F_k) (th - th_last)^2", ha="center",
             va="center", fontsize=9.5, color=INK, family="monospace")

    ax1.text(0.5, 0.10, "with two tasks these are the same thing",
             ha="center", fontsize=9, color=INK)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0.02, 1.0)
    ax1.axis("off")

    # --- 오른쪽: 언제부터 갈라지는가 ---------------------------------------
    ax2.plot(ts, gaps, "o-", color=ACCENT, linewidth=2.1, ms=7)
    ax2.axhline(0, color=INK, linewidth=0.8, alpha=0.4)
    ax2.annotate("identical", xy=(2, 0), xytext=(2.35, max(gaps) * 0.30),
                 fontsize=9, color=MUTED, weight="bold",
                 arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.1))
    ax2.set_xlabel("number of tasks seen", fontsize=9)
    ax2.set_ylabel("difference between the two (%)", fontsize=9)
    ax2.set_title("they part company at three", fontsize=10, color=INK,
                  weight="bold")
    ax2.set_xticks(ts)
    ax2.set_ylim(-max(gaps) * 0.12, max(gaps) * 1.3)
    ax2.grid(alpha=0.13)

    for ax in (ax1, ax2):
        ax.tick_params(labelsize=8, colors=INK)
        for sp in ax.spines.values():
            sp.set_color(INK)
            sp.set_linewidth(0.7)

    fig.text(0.5, -0.04,
             "mean of 3 seeds on a linear toy. the gap is real but small - "
             "which is what the reply argued.",
             ha="center", fontsize=8.5, style="italic", color=INK)
    fig.tight_layout()
    fig.savefig(HERE / "two_penalty_forms.svg", transparent=True,
                bbox_inches="tight")
    plt.close(fig)
    print(f"wrote two_penalty_forms.svg   gaps "
          f"{[round(g, 2) for g in gaps]}")


if __name__ == "__main__":
    two_tasks_agree()
