"""14.3 증류와 EMA 교사 쪽들의 그림.

규칙은 CLAUDE.md 의 Figures 항에 적어 둔 대로다.
- SVG, fonttype='path', transparent
- 글자는 모두 아스키 (matplotlib 기본 폰트에 한글이 없다)

MNIST 를 읽으므로 **저장소 뿌리에서** 돌린다.

⚠️ 교사를 2 에포크 학습시키므로 이 기계에서 20 초 안팎 걸린다.

실행: python docs/ch13/self_distillation/figures/make_figures.py
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


# === 그림으로 보는 AI: 틀린 답들 사이의 비율 ==================================
#
# 힌턴 외 2015 의 요지 — "The relative probabilities of incorrect answers tell
# us a lot about how the cumbersome model tends to generalize" — 를 숫자 자료로
# 보인다. 딱딱한 레이블은 "4" 하나만 말하지만, 교사는 "4이고, 아니라면 1보다
# 9일 가능성이 훨씬 크다"고 말한다. 수는 전부 여기서 학습시켜 셈한다.

def dark_knowledge(epochs=2):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torchvision import datasets, transforms

    torch.manual_seed(0)
    np.random.seed(0)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tf = transforms.ToTensor()
    tr = datasets.MNIST("./data", train=True, download=False, transform=tf)
    te = datasets.MNIST("./data", train=False, download=False, transform=tf)
    dl = torch.utils.data.DataLoader(tr, batch_size=256, shuffle=True,
                                     num_workers=0)

    net = nn.Sequential(
        nn.Conv2d(1, 32, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(), nn.Linear(64 * 7 * 7, 128), nn.ReLU(),
        nn.Linear(128, 10)).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    for _ in range(epochs):
        for x, y in dl:
            opt.zero_grad()
            F.cross_entropy(net(x.to(dev)), y.to(dev)).backward()
            opt.step()

    net.eval()
    logits, labels = [], []
    with torch.no_grad():
        for x, y in torch.utils.data.DataLoader(te, batch_size=512):
            logits.append(net(x.to(dev)).cpu())
            labels.append(y)
    logits = torch.cat(logits)
    labels = torch.cat(labels)
    acc = (logits.argmax(1) == labels).float().mean().item()

    def off_mass(temp):
        p = F.softmax(logits / temp, 1)
        m = torch.stack([p[labels == c].mean(0) for c in range(10)])
        m = m.clone()
        m.fill_diagonal_(0)
        return (m / m.sum(1, keepdim=True)).numpy()

    off1, off4 = off_mass(1.0), off_mass(4.0)
    ent1 = float(-(off1 * np.log(off1 + 1e-12)).sum(1).mean())
    ent4 = float(-(off4 * np.log(off4 + 1e-12)).sum(1).mean())
    unif = float(np.log(9))

    # 그림이 주장하는 것을 먼저 확인한다
    assert ent1 < unif * 0.95, "틀린 답 분포가 균등에 가까우면 이 그림의 이야기가 없다"
    assert ent4 > ent1, "온도를 올렸는데 평평해지지 않으면 오른쪽 이야기가 없다"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.3),
                                   gridspec_kw={"width_ratios": [1.1, 1]})

    # --- 왼쪽: 틀린 답들의 질량 ---------------------------------------------
    im = ax1.imshow(off1, cmap="pink_r", vmin=0, vmax=off1.max())
    ax1.set_xticks(range(10))
    ax1.set_yticks(range(10))
    ax1.set_xlabel("probability given to this wrong digit", fontsize=9)
    ax1.set_ylabel("true digit", fontsize=9)
    ax1.set_title("the teacher's wrong answers are not random",
                  fontsize=10, color=INK, weight="bold")
    for c in range(10):
        j = int(off1[c].argmax())
        ax1.text(j, c, f"{off1[c, j]:.2f}", ha="center", va="center",
                 fontsize=7.4, color=ACCENT, weight="bold")
    fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.03).ax.tick_params(
        labelsize=7, colors=INK)

    # --- 오른쪽: 가장 큰 오답 몫과 균등선 -----------------------------------
    tops = off1.max(1)
    ax2.bar(range(10), tops, color=ACCENT, alpha=0.85, width=0.62)
    ax2.axhline(1 / 9, color=INK, linestyle="--", linewidth=1.3)
    ax2.text(9.4, 1 / 9, "  if wrong answers\n  were uniform", fontsize=8,
             color=INK, va="center", ha="right")
    for c in range(10):
        ax2.text(c, tops[c] + 0.012, str(int(off1[c].argmax())), ha="center",
                 fontsize=8.4, color=INK)
    ax2.set_xticks(range(10))
    ax2.set_xlabel("true digit  (label above bar = its top wrong answer)",
                   fontsize=9)
    ax2.set_ylabel("share of the wrong-answer mass", fontsize=9)
    ax2.set_ylim(0, max(tops) * 1.3)
    ax2.set_title("one wrong answer takes most of it", fontsize=10,
                  color=INK, weight="bold")
    ax2.grid(alpha=0.13, axis="y")

    for ax in (ax1, ax2):
        ax.tick_params(labelsize=8, colors=INK)
        for s in ax.spines.values():
            s.set_color(INK)
            s.set_linewidth(0.7)

    fig.text(0.5, -0.05,
             f"MNIST teacher, test accuracy {acc:.4f}.  entropy of the wrong "
             f"answers {ent1:.3f} nats at T=1, {ent4:.3f} at T=4, "
             f"{unif:.3f} if uniform.",
             ha="center", fontsize=8.5, style="italic", color=INK)
    fig.tight_layout()
    fig.savefig(HERE / "dark_knowledge.svg", transparent=True,
                bbox_inches="tight")
    plt.close(fig)
    print(f"wrote dark_knowledge.svg   acc {acc:.4f}  "
          f"entropy T1 {ent1:.3f}  T4 {ent4:.3f}  uniform {unif:.3f}")


# === 그림으로 보는 AI: 평균이 가속이 되는 자리 ================================
#
# 폴랴크-유디츠키 1992 의 결과 — 잡음 섞인 반복값을 **평균내면** 반복값 자체보다
# 훨씬 정확해진다 — 를 잡음 섞인 이차식에서 잰다. 쪽의 9절이 EMA 와 이 평균은
# 같은 연산이 아니라고 못 박아 두었으므로, 여기서 재는 것은 **균등 평균** 쪽이다.

AVG_D, AVG_T = 20, 4000


def averaging_gain(lrs=(0.05, 0.15)):
    rng = np.random.default_rng(0)
    a = np.diag(np.linspace(0.3, 3.0, AVG_D))
    xstar = rng.normal(size=AVG_D)

    def run(lr):
        x = np.zeros(AVG_D)
        avg = np.zeros(AVG_D)
        ex, ea = [], []
        for t in range(1, AVG_T + 1):
            g = a @ (x - xstar) + rng.normal(0, 1.0, AVG_D)
            x = x - lr * g
            avg = avg + (x - avg) / t
            ex.append(np.linalg.norm(x - xstar))
            ea.append(np.linalg.norm(avg - xstar))
        return np.array(ex), np.array(ea)

    out = {lr: run(lr) for lr in lrs}
    tail = slice(AVG_T // 2, None)
    ratios = {lr: out[lr][0][tail].mean() / out[lr][1][tail].mean()
              for lr in lrs}
    assert min(ratios.values()) > 3, \
        "평균이 반복값보다 크게 낫지 않으면 이 그림의 이야기가 없다"

    big, small = max(lrs), min(lrs)
    assert out[big][0][tail].mean() > out[small][0][tail].mean(), \
        "큰 걸음의 반복값이 더 나쁘지 않으면 뒷이야기가 성립하지 않는다"

    fig, ax = plt.subplots(figsize=(8.8, 4.3))
    steps = np.arange(1, AVG_T + 1)
    ex, ea = out[big]
    ax.semilogy(steps, ex, color=INK, linewidth=0.9, alpha=0.45,
                label=f"the iterate itself  (step {big})")
    ax.semilogy(steps, ea, color=ACCENT, linewidth=2.2,
                label=f"the running average of it")
    ex2, _ = out[small]
    ax.semilogy(steps, ex2, color=INK, linewidth=0.9, alpha=0.8,
                linestyle="--", label=f"the iterate with a smaller step "
                                      f"({small})")

    ax.annotate(f"{ratios[big]:.1f}x lower", (AVG_T * 0.62, ea[int(AVG_T * 0.62)]),
                textcoords="offset points", xytext=(0, 20), ha="center",
                fontsize=9, color=ACCENT, weight="bold")
    ax.set_xlabel("step", fontsize=9.5)
    ax.set_ylabel("distance to the optimum  (log)", fontsize=9.5)
    ax.set_title("averaging buys accuracy the steps never had",
                 fontsize=10.5, color=INK, weight="bold", pad=8)
    ax.legend(fontsize=8.5, framealpha=0.92, loc="upper right")
    ax.grid(alpha=0.14, which="both")
    ax.tick_params(labelsize=8, colors=INK)
    for s in ax.spines.values():
        s.set_color(INK)
        s.set_linewidth(0.7)

    fig.text(0.5, -0.05,
             f"noisy quadratic in {AVG_D} dimensions. the big-step average "
             f"even beats the small-step iterate.",
             ha="center", fontsize=8.6, style="italic", color=INK)
    fig.tight_layout()
    fig.savefig(HERE / "averaging_gain.svg", transparent=True,
                bbox_inches="tight")
    plt.close(fig)
    print("wrote averaging_gain.svg   "
          + "  ".join(f"lr={lr}: {r:.1f}x" for lr, r in ratios.items()))


if __name__ == "__main__":
    dark_knowledge()
    averaging_gain()
