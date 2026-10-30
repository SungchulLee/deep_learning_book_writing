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


# === 그림으로 보는 AI: 라벨을 보는 중요도와 안 보는 중요도 ===================
#
# EWC 의 피셔는 **라벨이 있어야** 셈한다. MAS 는 출력의 크기만 보므로 라벨을
# 쓰지 않는다. 라벨을 섞어 보면 둘이 얼마나 다른 것을 재는지 바로 드러난다.

def labels_or_not():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    def facts(k):
        s = set()
        for j in range(1, 10):
            s.add((k, j))
            s.add((j, k))
        return sorted(s)

    pairs = facts(1)
    x = torch.zeros(len(pairs), 20)
    y = torch.zeros(len(pairs), dtype=torch.long)
    for i, (a, b) in enumerate(pairs):
        x[i, a - 1] = 1
        x[i, 10 + b - 1] = 1
        y[i] = a + b

    lf = nn.CrossEntropyLoss()
    torch.manual_seed(0)
    net = nn.Sequential(nn.Linear(20, 50), nn.Sigmoid(), nn.Linear(50, 19))
    opt = torch.optim.SGD(net.parameters(), lr=0.5, momentum=0.9)
    for _ in range(400):
        opt.zero_grad()
        lf(net(x), y).backward()
        opt.step()
    ps = list(net.parameters())

    def flat():
        return torch.cat([p.grad.flatten() for p in ps]).clone()

    def fisher(labels):                 # 라벨이 필요하다
        acc = None
        for i in range(len(x)):
            net.zero_grad()
            lf(net(x[i:i + 1]), labels[i:i + 1]).backward()
            g = flat() ** 2
            acc = g if acc is None else acc + g
        return acc / len(x)

    def mas():                          # 라벨을 보지 않는다
        acc = None
        for i in range(len(x)):
            net.zero_grad()
            (net(x[i:i + 1]) ** 2).sum().backward()
            g = flat().abs()
            acc = g if acc is None else acc + g
        return acc / len(x)

    g = torch.Generator().manual_seed(1)
    y_shuf = y[torch.randperm(len(y), generator=g)]
    f_true, f_shuf = fisher(y), fisher(y_shuf)
    m1, m2 = mas(), mas()

    def cos(a, b):
        return F.cosine_similarity(a, b, dim=0).item()

    vals = [cos(f_true, f_shuf), cos(m1, m2), cos(f_true, m1)]
    assert vals[1] > 0.999, "MAS 는 라벨을 안 보므로 두 번 재도 같아야 한다"
    assert vals[0] < 0.9, "피셔는 라벨을 섞으면 달라져야 한다"

    fig, ax = plt.subplots(figsize=(8.8, 4.2))
    labels = ["EWC Fisher\ntrue labels vs shuffled",
              "MAS\nmeasured twice",
              "EWC vs MAS\non the same network"]
    cols = [INK, ACCENT, MUTED]
    bars = ax.bar(range(3), vals, color=cols, width=0.55, alpha=0.88)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.03, f"{v:.2f}", ha="center", fontsize=12,
                color=INK, weight="bold")
    ax.axhline(1.0, color=INK, linestyle=":", linewidth=1.2, alpha=0.6)
    ax.text(2.45, 1.015, "identical", fontsize=8.4, color=INK, alpha=0.75,
            ha="right")

    ax.set_xticks(range(3))
    ax.set_xticklabels(labels, fontsize=8.8)
    ax.set_ylabel("cosine between the two importance vectors", fontsize=9.5)
    ax.set_ylim(0, 1.18)
    ax.set_title("shuffle the labels and only one of them notices",
                 fontsize=10.5, color=INK, weight="bold", pad=10)
    ax.grid(alpha=0.13, axis="y")
    ax.tick_params(labelsize=8.5, colors=INK)
    for sp in ax.spines.values():
        sp.set_color(INK)
        sp.set_linewidth(0.7)

    fig.text(0.5, -0.05,
             "one trained network, same weights throughout. "
             "the third bar is why they are not interchangeable.",
             ha="center", fontsize=8.5, style="italic", color=INK)
    fig.tight_layout()
    fig.savefig(HERE / "labels_or_not.svg", transparent=True,
                bbox_inches="tight")
    plt.close(fig)
    print(f"wrote labels_or_not.svg   {[round(v,3) for v in vals]}")


if __name__ == "__main__":
    two_tasks_agree()
    labels_or_not()
