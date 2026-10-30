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


# === 그림으로 보는 AI: 감마가 정하는 것은 굳기의 천장이다 ====================
#
# 온라인 EWC 의 재귀 F~_t = g F~_{t-1} + F_t 에서 g=1 이면 중요도가 과제 수에
# 비례해 쌓인다 — 천장이 없다. g<1 이면 기하급수가 되어 F/(1-g) 에서 멈춘다.
# 정확한 순차 베이즈라면 g=1 이므로, g<1 은 유도에서 나온 값이 아니라 **고른
# 값**이다. 피셔는 실제로 과제를 차례로 배우며 잰다.

def gamma_ceiling():
    import torch
    import torch.nn as nn

    def facts(k):
        s = set()
        for j in range(1, 10):
            s.add((k, j))
            s.add((j, k))
        return sorted(s)

    def enc(pairs):
        x = torch.zeros(len(pairs), 20)
        y = torch.zeros(len(pairs), dtype=torch.long)
        for i, (a, b) in enumerate(pairs):
            x[i, a - 1] = 1
            x[i, 10 + b - 1] = 1
            y[i] = a + b
        return x, y

    n_tasks = 9
    data = [enc(facts(k)) for k in range(1, n_tasks + 1)]
    lf = nn.CrossEntropyLoss()

    torch.manual_seed(0)
    net = nn.Sequential(nn.Linear(20, 60), nn.Sigmoid(), nn.Linear(60, 19))
    ps = list(net.parameters())

    fishers = []
    for t in range(n_tasks):
        x, y = data[t]
        opt = torch.optim.SGD(ps, lr=0.3, momentum=0.9)
        for _ in range(250):
            opt.zero_grad()
            lf(net(x), y).backward()
            opt.step()
        f = torch.zeros(sum(p.numel() for p in ps))
        for i in range(len(x)):
            net.zero_grad()
            lf(net(x[i:i + 1]), y[i:i + 1]).backward()
            f += torch.cat([p.grad.flatten() for p in ps]) ** 2
        fishers.append(f / len(x))

    gammas = [1.0, 0.9, 0.7, 0.5]
    acc = {g: torch.zeros_like(fishers[0]) for g in gammas}
    curves = {g: [] for g in gammas}
    for t in range(n_tasks):
        for g in gammas:
            acc[g] = g * acc[g] + fishers[t]
            curves[g].append(acc[g].sum().item())

    mean_f = float(np.mean([f.sum().item() for f in fishers]))
    assert curves[1.0][-1] > curves[1.0][len(curves[1.0]) // 2], (
        "g=1 이면 계속 쌓여야 한다")
    for g in gammas[1:]:
        assert curves[g][-1] <= mean_f / (1 - g) * 1.05, (
            f"g={g} 는 F/(1-g) 를 넘지 않아야 한다")

    fig, ax = plt.subplots(figsize=(8.8, 4.3))
    t = np.arange(1, n_tasks + 1)
    cols = {1.0: ACCENT, 0.9: INK, 0.7: MUTED, 0.5: "#9a6b4f"}
    for g in gammas:
        ax.plot(t, curves[g], "o-", color=cols[g], linewidth=2.1, ms=5.5,
                label=f"gamma = {g}")
        if g < 1.0:
            cap = mean_f / (1 - g)
            ax.axhline(cap, color=cols[g], linestyle=":", linewidth=1.1,
                       alpha=0.65)
    ax.text(n_tasks, mean_f / (1 - 0.9) * 0.985,
            "dotted lines: the ceiling  F / (1 - gamma)",
            fontsize=8.2, color=INK, alpha=0.8, ha="right", va="top")

    ax.set_xlabel("tasks seen", fontsize=9.5)
    ax.set_ylabel("total accumulated importance", fontsize=9.5)
    ax.set_xticks(t)
    ax.set_title("gamma is a ceiling on how rigid the model may become",
                 fontsize=10.5, color=INK, weight="bold", pad=9)
    ax.legend(fontsize=8.6, loc="upper left", framealpha=0.9)
    ax.grid(alpha=0.13)
    ax.tick_params(labelsize=8.5, colors=INK)
    for sp in ax.spines.values():
        sp.set_color(INK)
        sp.set_linewidth(0.7)

    fig.text(0.5, -0.05,
             "Fishers measured by actually training the nine tasks in turn. "
             "gamma = 1 is the Bayesian value and the one with no ceiling.",
             ha="center", fontsize=8.5, style="italic", color=INK)
    fig.tight_layout()
    fig.savefig(HERE / "gamma_ceiling.svg", transparent=True,
                bbox_inches="tight")
    plt.close(fig)
    print(f"wrote gamma_ceiling.svg   g=1 ends {curves[1.0][-1]:.4f}, "
          f"g=0.5 ends {curves[0.5][-1]:.4f} (cap {mean_f/0.5:.4f})")


# === 그림으로 보는 AI: lambda 와 Omega 는 따로 정해지지 않는다 ===============
#
# 벌점은 lambda * Omega 로만 나타나므로 (lambda/c, c*Omega) 는 같은 목적함수다.
# 그래서 **같은 lambda 로 세 방법을 견주는 것** 은 방법이 아니라 Omega 의
# 눈금을 견주는 일이 된다. 세 눈금이 실제로 얼마나 다른지 잰다.

def lambda_omega_scale():
    import torch
    import torch.nn as nn

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
    ps = list(net.parameters())

    def flat():
        return torch.cat([p.grad.flatten() for p in ps]).clone()

    def params():
        return torch.cat([p.detach().flatten() for p in ps]).clone()

    n = sum(p.numel() for p in ps)
    w = torch.zeros(n)
    start = params()
    opt = torch.optim.SGD(ps, lr=0.3, momentum=0.9)
    for _ in range(400):                      # SI 는 학습 경로를 따라 쌓는다
        prev = params()
        opt.zero_grad()
        lf(net(x), y).backward()
        g = flat()
        opt.step()
        w += -g * (params() - prev)
    end = params()
    om_si = torch.clamp(w, min=0) / ((end - start) ** 2 + 1e-3)

    f = torch.zeros(n)
    for i in range(len(x)):
        net.zero_grad()
        lf(net(x[i:i + 1]), y[i:i + 1]).backward()
        f += flat() ** 2
    om_ewc = f / len(x)

    m = torch.zeros(n)
    for i in range(len(x)):
        net.zero_grad()
        (net(x[i:i + 1]) ** 2).sum().backward()
        m += flat().abs()
    om_mas = m / len(x)

    names = ["EWC", "SI", "MAS"]
    sums = [om_ewc.sum().item(), om_si.sum().item(), om_mas.sum().item()]
    ratio = max(sums) / min(sums)
    assert ratio > 1e4, f"눈금이 자릿수로 달라야 이야기가 성립한다 ({ratio})"

    th = torch.randn(n)
    base = 50.0 / 2 * (om_ewc * th ** 2).sum().item()
    cs = [1.0, 10.0, 1000.0]
    vals = [(50.0 / c) / 2 * ((c * om_ewc) * th ** 2).sum().item() for c in cs]
    assert max(abs(v - base) for v in vals) < abs(base) * 1e-6, (
        "lambda 를 나누고 Omega 를 곱하면 같은 값이어야 한다")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 4.1),
                                   gridspec_kw={"width_ratios": [1, 1.1]})

    cols = [INK, MUTED, ACCENT]
    ax1.bar(range(3), sums, color=cols, width=0.55, alpha=0.88)
    ax1.set_yscale("log")
    for i, v in enumerate(sums):
        ax1.text(i, v * 2.2, f"{v:,.4g}", ha="center", fontsize=10,
                 color=INK, weight="bold")
    ax1.set_xticks(range(3))
    ax1.set_xticklabels(names, fontsize=9.5)
    ax1.set_ylabel("total importance (log)", fontsize=9.5)
    ax1.set_ylim(top=max(sums) * 60)
    ax1.set_title(f"{ratio:,.0f}x apart on the same network",
                  fontsize=10, color=INK, weight="bold")
    ax1.grid(alpha=0.13, axis="y", which="both")

    ax2.axis("off")
    ax2.text(0.5, 0.93, "the penalty only ever uses the product",
             ha="center", fontsize=9.2, color=INK, style="italic")
    ax2.text(0.5, 0.80, "lambda  x  Omega", ha="center", fontsize=13,
             color=ACCENT, family="monospace", weight="bold")
    for k, (c, v) in enumerate(zip(cs, vals)):
        yy = 0.56 - k * 0.145
        ax2.add_patch(plt.Rectangle((0.04, yy - 0.055), 0.92, 0.115,
                                    facecolor=INK, alpha=0.06,
                                    edgecolor=INK, linewidth=0.6))
        ax2.text(0.09, yy, f"lambda/{c:<6g}  Omega x {c:<6g}", fontsize=9,
                 color=INK, family="monospace", va="center")
        ax2.text(0.95, yy, f"{v:.8f}", fontsize=9, color=ACCENT,
                 family="monospace", va="center", ha="right")
    ax2.text(0.5, 0.06,
             "same objective, eight decimals.\n"
             "so a shared lambda compares the scales, not the methods.",
             ha="center", va="bottom", fontsize=8.6, color=INK)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)

    ax1.tick_params(labelsize=8.5, colors=INK)
    for sp in ax1.spines.values():
        sp.set_color(INK)
        sp.set_linewidth(0.7)

    fig.tight_layout()
    fig.savefig(HERE / "lambda_omega.svg", transparent=True,
                bbox_inches="tight")
    plt.close(fig)
    print(f"wrote lambda_omega.svg   sums {[round(s,4) for s in sums]}  "
          f"ratio {ratio:,.0f}")


if __name__ == "__main__":
    two_tasks_agree()
    labels_or_not()
    gamma_ceiling()
    lambda_omega_scale()
