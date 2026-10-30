"""17장 지속 학습 — 「그림으로 보는 AI」 그림.

규칙은 CLAUDE.md 의 Figures 항에 적어 둔 대로다.
- SVG, fonttype='path', transparent
- 글자는 모두 아스키 (matplotlib 기본 폰트에 한글이 없다)

자료를 내려받지 않으므로 어디서 돌려도 된다.

실행: python docs/ch16/continual_learning/figures/make_figures.py
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


# === 그림으로 보는 AI: 1989 년의 덧셈 사실 실험 ===============================
#
# McCloskey-Cohen 은 MNIST 가 아니라 **한 자리 덧셈 사실**로 보였다.
# ones = 1+1..1+9 와 2+1..9+1 로 17 문제. 그것을 외운 그물에 twos 만 가르친다.
#
# 여기서 재는 것은 "언젠가 잊는다"가 아니라 **무너지는 속도**다. 익히는 데
# 걸린 횟수와 무너지는 데 걸린 횟수를 나란히 놓고, 무너진 그 순간 새 과제가
# 어디까지 왔는지를 함께 본다. 씨앗 12 개의 중앙값이다.

def _facts(k):
    """k 가 들어가는 한 자리 덧셈 사실. 1989 년 논문과 같은 17 문제."""
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


def _run(seed, torch, nn, n_trials):
    """ones 를 외운 뒤 twos 만 가르치며 두 과제 정확도를 좇는다."""
    xo, yo = _encode(_facts(1), torch)
    xt, yt = _encode(_facts(2), torch)
    lf = nn.CrossEntropyLoss()

    torch.manual_seed(seed)
    net = nn.Sequential(nn.Linear(20, 50), nn.Sigmoid(), nn.Linear(50, 19))

    def acc(x, y):
        with torch.no_grad():
            return (net(x).argmax(1) == y).float().mean().item() * 100

    opt = torch.optim.SGD(net.parameters(), lr=0.5, momentum=0.9)
    learn = None
    for e in range(4000):
        opt.zero_grad()
        lf(net(xo), yo).backward()
        opt.step()
        if acc(xo, yo) == 100:
            learn = e + 1
            break

    opt = torch.optim.SGD(net.parameters(), lr=0.5, momentum=0.9)
    old, new = [], []
    for _ in range(n_trials):
        opt.zero_grad()
        lf(net(xt), yt).backward()
        opt.step()
        old.append(acc(xo, yo))
        new.append(acc(xt, yt))
    return learn, np.array(old), np.array(new)


def forgetting_speed():
    import torch
    import torch.nn as nn

    n_trials, seeds = 60, range(12)
    learns, olds, news = [], [], []
    for s in seeds:
        l, o, n = _run(s, torch, nn, n_trials)
        learns.append(l)
        olds.append(o)
        news.append(n)
    olds, news = np.array(olds), np.array(news)
    med_old, med_new = np.median(olds, 0), np.median(news, 0)
    learn_med = float(np.median(learns))

    # **처음** 50% 아래로 내려가는 trial 은 일시적인 출렁임을 잡는다
    # (문제가 17 개뿐이라 정확도가 5.9%p 단위로 튄다). 여기서 쓰는 것은
    # 그 뒤로 다시는 올라오지 않는 자리 — 되돌아오지 않는 지점이다.
    above = np.where(med_old >= 50)[0]
    below = int(above.max()) + 2 if len(above) else 1
    new_then = med_new[below - 1]

    assert learn_med > below, (
        "익히는 데 걸린 횟수가 무너지는 데 걸린 횟수보다 커야 이 그림의 "
        f"이야기가 성립한다 (learn={learn_med}, break={below})")
    assert med_old[below - 1:].max() < 50, (
        "그 뒤로는 다시 50% 를 넘지 않아야 한다")
    assert new_then < 100, (
        f"무너질 때 새 과제가 아직 다 배워지지 않아야 한다 (twos={new_then})")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.8, 4.1),
                                   gridspec_kw={"width_ratios": [1, 1.35]})

    # --- 왼쪽: 익히는 데 36, 무너지는 데 7 ---------------------------------
    ax1.barh([1, 0], [learn_med, below], color=[MUTED, ACCENT], height=0.46,
             alpha=0.88)
    ax1.set_yticks([1, 0])
    ax1.set_yticklabels(["to learn\nthe ones facts",
                         "to lose\nthem again"], fontsize=9)
    for yy, v in ((1, learn_med), (0, below)):
        ax1.text(v + learn_med * 0.03, yy, f"{v:.0f}", va="center",
                 fontsize=12, color=INK, weight="bold")
    ax1.set_xlabel("trials over the 17 problems", fontsize=9)
    ax1.set_xlim(0, learn_med * 1.22)
    ax1.set_title(f"{learn_med / below:.1f}x faster to destroy",
                  fontsize=10, color=INK, weight="bold")
    ax1.grid(alpha=0.13, axis="x")

    # --- 오른쪽: 새 과제가 배워지기 전에 옛 과제가 사라진다 ----------------
    t = np.arange(1, n_trials + 1)
    ax2.fill_between(t, np.percentile(olds, 25, 0), np.percentile(olds, 75, 0),
                     color=ACCENT, alpha=0.16)
    ax2.plot(t, med_old, color=ACCENT, linewidth=2.2, label="ones (the old task)")
    ax2.fill_between(t, np.percentile(news, 25, 0), np.percentile(news, 75, 0),
                     color=MUTED, alpha=0.16)
    ax2.plot(t, med_new, color=MUTED, linewidth=2.2, linestyle="--",
             label="twos (the new task)")

    ax2.axvline(below, color=INK, linestyle=":", linewidth=1.2, alpha=0.7)
    ax2.annotate(f"ones never recovers\nfrom here. twos: {new_then:.0f}%",
                 xy=(below, new_then), xytext=(below + 11, 68),
                 fontsize=8.6, color=INK,
                 arrowprops=dict(arrowstyle="->", color=INK, lw=0.9))

    ax2.set_xlabel("trials of the twos facts", fontsize=9)
    ax2.set_ylabel("accuracy (%)", fontsize=9)
    ax2.set_ylim(-3, 108)
    ax2.set_title("the old one is gone first", fontsize=10, color=INK,
                  weight="bold")
    ax2.legend(fontsize=8.4, loc="lower right", framealpha=0.9)
    ax2.grid(alpha=0.13)

    for ax in (ax1, ax2):
        ax.tick_params(labelsize=8, colors=INK)
        for sp in ax.spines.values():
            sp.set_color(INK)
            sp.set_linewidth(0.7)

    fig.text(0.5, -0.04,
             "median of 12 seeds, shaded band is the quartile spread. "
             "the 17 addition facts are the 1989 setup, not MNIST.",
             ha="center", fontsize=8.5, style="italic", color=INK)
    fig.tight_layout()
    fig.savefig(HERE / "forgetting_speed.svg", transparent=True,
                bbox_inches="tight")
    plt.close(fig)
    print(f"wrote forgetting_speed.svg   learn {learn_med:.0f}, "
          f"break {below}, twos then {new_then:.1f}%")


if __name__ == "__main__":
    forgetting_speed()
