"""17.6 구조 기반 방법 — 「그림으로 보는 AI」 그림.

규칙은 CLAUDE.md 의 Figures 항에 적어 둔 대로다.
- SVG, fonttype='path', transparent
- 글자는 모두 아스키 (matplotlib 기본 폰트에 한글이 없다)

sklearn 의 digits (8x8) 를 쓰므로 내려받는 것이 없다.

실행: python docs/ch16/architecture_methods/figures/make_figures.py
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


# === 그림으로 보는 AI: 웨이트를 한 번도 안 고치고 ============================
#
# 고정된 무작위 웨이트에 **이진 마스크만** 배워도 우연보다 한참 높이 간다.
# Supermask 가 성립하는 바탕이고, 이 쪽의 과제별 마스크도 거기에 기댄다.
#
# 구현 주의: straight-through 로 마스크를 배울 때 점수를 0 으로 시작하면
# 마스크가 전부 꺼진 채 출발해 **그래디언트가 아예 흐르지 않는다**. 처음에
# 절반쯤 켜져 있도록 점수를 흩어 두어야 한다. 처음 시도가 우연 수준에서
# 멈춘 것이 그 때문이었다.

class _Masked:
    """고정 무작위 웨이트 + 배우는 이진 마스크."""

    def __init__(self, torch, nn, din, dh, dout, seed):
        g = torch.Generator().manual_seed(seed)
        self.torch, self.nn = torch, nn
        self.w1 = torch.randn(din, dh, generator=g) * (2 / din) ** 0.5
        self.w2 = torch.randn(dh, dout, generator=g) * (2 / dh) ** 0.5
        self.s1 = nn.Parameter(torch.randn(din, dh, generator=g) * 0.1)
        self.s2 = nn.Parameter(torch.randn(dh, dout, generator=g) * 0.1)

    @staticmethod
    def _m(sc):
        hard = (sc > 0).float()
        return hard.detach() + sc - sc.detach()      # straight-through

    def __call__(self, x):
        torch = self.torch
        h = torch.relu(x @ (self.w1 * self._m(self.s1)))
        return h @ (self.w2 * self._m(self.s2))

    def kept(self):
        a = (self.s1 > 0).float().mean().item()
        b = (self.s2 > 0).float().mean().item()
        return (a + b) / 2 * 100


def mask_only():
    import torch
    import torch.nn as nn
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split

    d = load_digits()
    x = torch.tensor(d.data, dtype=torch.float32) / 16.0
    y = torch.tensor(d.target)
    xtr, xte, ytr, yte = train_test_split(x, y, test_size=0.3,
                                          random_state=0, stratify=y)
    chance = 100.0 / len(set(d.target))
    lf = nn.CrossEntropyLoss()

    def acc(f, a, b):
        with torch.no_grad():
            return (f(a).argmax(1) == b).float().mean().item() * 100

    before, after, trained, kept = [], [], [], []
    for seed in range(3):
        net = _Masked(torch, nn, 64, 300, 10, seed)
        before.append(acc(net, xte, yte))
        opt = torch.optim.Adam([net.s1, net.s2], lr=0.02)
        for _ in range(800):
            opt.zero_grad()
            lf(net(xtr), ytr).backward()
            opt.step()
        after.append(acc(net, xte, yte))
        kept.append(net.kept())

        torch.manual_seed(1000 + seed)       # 견줄 그물도 씨앗을 고정한다.
        ref = nn.Sequential(nn.Linear(64, 300), nn.ReLU(), nn.Linear(300, 10))
        o2 = torch.optim.Adam(ref.parameters(), lr=0.01)
        for _ in range(800):
            o2.zero_grad()
            lf(ref(xtr), ytr).backward()
            o2.step()
        trained.append(acc(ref, xte, yte))

    vals = [chance, float(np.mean(before)), float(np.mean(after)),
            float(np.mean(trained))]
    assert vals[2] > 3 * vals[0], (
        f"마스크만 배워도 우연보다 한참 높아야 한다 ({vals[2]} vs {vals[0]})")
    assert vals[2] < vals[3], "그래도 제대로 배운 쪽보다는 낮아야 한다"

    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    labels = ["chance", "random mask,\nnothing learned",
              "mask only,\nweights frozen", "weights trained\nnormally"]
    cols = [INK, INK, ACCENT, MUTED]
    alphas = [0.28, 0.45, 0.9, 0.9]
    bars = ax.bar(range(4), vals, color=cols, width=0.56)
    for b, a in zip(bars, alphas):
        b.set_alpha(a)
    for i, v in enumerate(vals):
        ax.text(i, v + 2.0, f"{v:.1f}%", ha="center", fontsize=11,
                color=INK, weight="bold")

    ax.annotate("", xy=(2, vals[2] + 9), xytext=(3, vals[3] + 9),
                arrowprops=dict(arrowstyle="<->", color=ACCENT, lw=1.3))
    ax.text(2.5, vals[3] + 12.5, f"{vals[3] - vals[2]:.1f} points apart",
            ha="center", fontsize=9, color=ACCENT)

    ax.set_xticks(range(4))
    ax.set_xticklabels(labels, fontsize=8.8)
    ax.set_ylabel("test accuracy (%)", fontsize=9.5)
    ax.set_ylim(0, 118)
    ax.set_title("not one weight was changed in the third bar",
                 fontsize=10.5, color=INK, weight="bold", pad=10)
    ax.grid(alpha=0.13, axis="y")
    ax.tick_params(labelsize=8.5, colors=INK)
    for sp in ax.spines.values():
        sp.set_color(INK)
        sp.set_linewidth(0.7)

    fig.text(0.5, -0.04,
             f"8x8 digits, mean of 3 seeds. the learned mask keeps "
             f"{np.mean(kept):.0f}% of the connections.",
             ha="center", fontsize=8.5, style="italic", color=INK)
    fig.tight_layout()
    fig.savefig(HERE / "mask_only.svg", transparent=True,
                bbox_inches="tight")
    plt.close(fig)
    print(f"wrote mask_only.svg   {[round(v,1) for v in vals]}   "
          f"kept {np.mean(kept):.1f}%")


# === 그림으로 보는 AI: 분류기 대신 오토인코더로 고르기 ======================
#
# Expert Gate 는 어느 전문가를 쓸지 **분류기로 고르지 않는다.** 과제마다 한
# 레이어 오토인코더를 두고, 테스트 입력을 가장 잘 되살리는 쪽을 고른다.
# 오토인코더들은 서로의 자료를 본 적이 없고, 테스트 때 과제 이름도 필요 없다.

def autoencoder_gate():
    import torch
    import torch.nn as nn
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split

    d = load_digits()
    x = torch.tensor(d.data, dtype=torch.float32) / 16.0
    y = torch.tensor(d.target)
    xtr, xte, ytr, yte = train_test_split(x, y, test_size=0.3,
                                          random_state=0, stratify=y)
    tasks = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]

    def sub(xs, ys, t):
        m = torch.isin(ys, torch.tensor(t))
        return xs[m], ys[m]

    torch.manual_seed(0)
    aes = []
    for t in tasks:                       # 과제마다 한 레이어 오토인코더
        xt, _ = sub(xtr, ytr, t)
        ae = nn.Sequential(nn.Linear(64, 20), nn.Sigmoid(),
                           nn.Linear(20, 64), nn.Sigmoid())
        opt = torch.optim.Adam(ae.parameters(), lr=0.01)
        for _ in range(600):
            opt.zero_grad()
            ((ae(xt) - xt) ** 2).mean().backward()
            opt.step()
        aes.append(ae)

    k = len(tasks)
    conf = np.zeros((k, k))
    correct = total = 0
    for ti, t in enumerate(tasks):
        xt, _ = sub(xte, yte, t)
        with torch.no_grad():
            errs = torch.stack([((ae(xt) - xt) ** 2).mean(1) for ae in aes])
        pick = errs.argmin(0)
        for p in pick.tolist():
            conf[ti, p] += 1
        correct += (pick == ti).sum().item()
        total += len(xt)

    acc = 100.0 * correct / total
    chance = 100.0 / k
    assert acc > 4 * chance, (
        f"재구성 오차로 고르는 것이 우연보다 한참 나아야 한다 ({acc} vs {chance})")

    cn = conf / conf.sum(1, keepdims=True) * 100

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 4.2),
                                   gridspec_kw={"width_ratios": [1.15, 1]})

    im = ax1.imshow(cn, cmap="pink_r", vmin=0, vmax=100)
    for i in range(k):
        for j in range(k):
            ax1.text(j, i, f"{cn[i, j]:.0f}", ha="center", va="center",
                     fontsize=9,
                     color="white" if cn[i, j] > 55 else INK)
    ax1.set_xticks(range(k))
    ax1.set_yticks(range(k))
    ax1.set_xticklabels([f"T{j}" for j in range(k)], fontsize=9)
    ax1.set_yticklabels([f"T{i}" for i in range(k)], fontsize=9)
    ax1.set_xlabel("expert chosen", fontsize=9.5)
    ax1.set_ylabel("task the input came from", fontsize=9.5)
    ax1.set_title("lowest reconstruction error wins", fontsize=10,
                  color=INK, weight="bold")
    fig.colorbar(im, ax=ax1, label="% of inputs")

    ax2.bar([0, 1], [chance, acc], color=[INK, ACCENT], width=0.5,
            alpha=[0.4, 0.9][0])
    bars = ax2.bar([0, 1], [chance, acc], color=[INK, ACCENT], width=0.5)
    bars[0].set_alpha(0.4)
    bars[1].set_alpha(0.9)
    for i, v in enumerate([chance, acc]):
        ax2.text(i, v + 2.5, f"{v:.1f}%", ha="center", fontsize=12,
                 color=INK, weight="bold")
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(["picking at\nrandom",
                         "by reconstruction\nerror"], fontsize=9)
    ax2.set_ylabel("right expert chosen (%)", fontsize=9.5)
    ax2.set_ylim(0, 118)
    ax2.set_title("no task label at test time", fontsize=10, color=INK,
                  weight="bold")
    ax2.grid(alpha=0.13, axis="y")

    for ax in (ax1, ax2):
        ax.tick_params(labelsize=8.5, colors=INK)
        for sp in ax.spines.values():
            sp.set_color(INK)
            sp.set_linewidth(0.7)

    fig.text(0.5, -0.05,
             "8x8 digits in five pairs. the five autoencoders never saw each "
             "other's data and were never trained against each other.",
             ha="center", fontsize=8.5, style="italic", color=INK)
    fig.tight_layout()
    fig.savefig(HERE / "autoencoder_gate.svg", transparent=True,
                bbox_inches="tight")
    plt.close(fig)
    print(f"wrote autoencoder_gate.svg   gating {acc:.1f}% vs chance {chance:.0f}%")


if __name__ == "__main__":
    mask_only()
    autoencoder_gate()
