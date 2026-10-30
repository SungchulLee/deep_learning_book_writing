"""17.4 리플레이 기반 방법 — 「그림으로 보는 AI」 그림.

규칙은 CLAUDE.md 의 Figures 항에 적어 둔 대로다.
- SVG, fonttype='path', transparent
- 글자는 모두 아스키 (matplotlib 기본 폰트에 한글이 없다)

자료를 내려받지 않으므로 어디서 돌려도 된다.

실행: python docs/ch16/replay_methods/figures/make_figures.py
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


# === 그림으로 보는 AI: 리허설과 유사 리허설 ==================================
#
# 1995 년 Robins 는 둘을 한 논문에 담았다 — 옛 보기를 저장해 다시 쓰는 리허설,
# 그리고 저장하지 않고 **지어낸 입력에 옛 그물의 답을 붙여** 쓰는 유사 리허설.
#
# 처음 잰 값은 유사 리허설이 더 낫다고 나왔는데, 항목 수를 맞추지 않은 탓이었다
# (유사 34 개 대 진짜 4 개). **같은 개수로 맞추면 차례가 뒤집힌다.** 그림은
# 맞춘 쪽을 그린다.

def _facts(k):
    s = set()
    for j in range(1, 10):
        s.add((k, j))
        s.add((j, k))
    return sorted(s)


def _encode(pairs, torch):
    x = torch.zeros(len(pairs), 20)
    y = torch.zeros(len(pairs), dtype=torch.long)
    for i, (a, b) in enumerate(pairs):
        x[i, a - 1] = 1
        x[i, 10 + b - 1] = 1
        y[i] = a + b
    return x, y


def _trial(mode, n, seed, torch, nn, copy):
    xo, yo = _encode(_facts(1), torch)
    xt, yt = _encode(_facts(2), torch)
    lf = nn.CrossEntropyLoss()
    kl = nn.KLDivLoss(reduction="batchmean")

    torch.manual_seed(seed)
    net = nn.Sequential(nn.Linear(20, 50), nn.Sigmoid(), nn.Linear(50, 19))

    def acc(x, y):
        with torch.no_grad():
            return (net(x).argmax(1) == y).float().mean().item() * 100

    opt = torch.optim.SGD(net.parameters(), lr=0.5, momentum=0.9)
    for _ in range(4000):
        opt.zero_grad()
        lf(net(xo), yo).backward()
        opt.step()
        if acc(xo, yo) == 100:
            break
    old = copy.deepcopy(net)

    g = torch.Generator().manual_seed(seed + 99)
    xp = torch.zeros(n, 20)                      # 지어낸 입력
    for i in range(n):
        xp[i, torch.randint(0, 10, (1,), generator=g)] = 1
        xp[i, 10 + torch.randint(0, 10, (1,), generator=g)] = 1
    with torch.no_grad():
        pp = torch.softmax(old(xp), 1)           # 옛 그물이 붙인 답

    idx = torch.randperm(len(xo), generator=g)[:min(n, len(xo))]
    xb, yb = xo[idx], yo[idx]

    opt = torch.optim.SGD(net.parameters(), lr=0.5, momentum=0.9)
    for _ in range(60):
        opt.zero_grad()
        loss = lf(net(xt), yt)
        if mode == "rehearsal":
            loss = loss + lf(net(xb), yb)
        elif mode == "pseudo":
            loss = loss + kl(torch.log_softmax(net(xp), 1), pp)
        loss.backward()
        opt.step()
    return acc(xo, yo)


def rehearsal_vs_pseudo():
    import copy

    import torch
    import torch.nn as nn

    ns = [2, 4, 8, 17]
    seeds = range(8)
    real = [float(np.mean([_trial("rehearsal", n, s, torch, nn, copy)
                           for s in seeds])) for n in ns]
    pseudo = [float(np.mean([_trial("pseudo", n, s, torch, nn, copy)
                             for s in seeds])) for n in ns]
    none = float(np.mean([_trial("none", 1, s, torch, nn, copy)
                          for s in seeds]))

    assert all(r >= p for r, p in zip(real, pseudo)), (
        "개수를 맞추면 진짜 리허설이 항상 같거나 낫다 — 그렇지 않으면 "
        "이 그림의 이야기가 성립하지 않는다")
    assert pseudo[-1] > none, "유사 리허설도 아무것도 안 하는 것보다는 낫다"

    fig, ax = plt.subplots(figsize=(8.4, 4.3))
    ax.axhline(none, color=INK, linestyle=":", linewidth=1.3, alpha=0.7)
    ax.text(ns[-1], none - 5.5, "nothing kept at all", fontsize=8.4,
            color=INK, alpha=0.8, ha="right")

    ax.plot(ns, real, "o-", color=ACCENT, linewidth=2.2, ms=7,
            label="rehearsal - real examples stored")
    ax.plot(ns, pseudo, "s--", color=MUTED, linewidth=2.2, ms=6.5,
            label="pseudorehearsal - nothing stored")

    ax.annotate(f"{real[-1]:.0f}%", xy=(ns[-1], real[-1]),
                xytext=(-6, 7), textcoords="offset points",
                fontsize=10, color=ACCENT, weight="bold", ha="right")
    ax.annotate(f"{pseudo[-1]:.0f}%", xy=(ns[-1], pseudo[-1]),
                xytext=(-6, 7), textcoords="offset points",
                fontsize=10, color=MUTED, weight="bold", ha="right")

    ax.set_xlabel("items used for rehearsal (matched)", fontsize=9.5)
    ax.set_ylabel("old task accuracy after the new one (%)", fontsize=9.5)
    ax.set_xticks(ns)
    ax.set_ylim(0, 112)
    ax.set_title("both were in the same 1995 paper", fontsize=10.5,
                 color=INK, weight="bold", pad=9)
    ax.legend(fontsize=8.6, loc="upper left", framealpha=0.9)
    ax.grid(alpha=0.13)
    ax.tick_params(labelsize=8.5, colors=INK)
    for sp in ax.spines.values():
        sp.set_color(INK)
        sp.set_linewidth(0.7)

    fig.text(0.5, -0.04,
             "mean of 8 seeds. matching the item counts is what decides this - "
             "unmatched, the made-up items appear to win.",
             ha="center", fontsize=8.5, style="italic", color=INK)
    fig.tight_layout()
    fig.savefig(HERE / "rehearsal_vs_pseudo.svg", transparent=True,
                bbox_inches="tight")
    plt.close(fig)
    print(f"wrote rehearsal_vs_pseudo.svg   real {[round(v) for v in real]} "
          f"pseudo {[round(v) for v in pseudo]} none {none:.1f}")


if __name__ == "__main__":
    rehearsal_vs_pseudo()
