"""reconstruction_analysis 쪽의 그림 — 복원 오차를 이상 점수로 썼을 때.

make_figures.py 와 따로 둔 까닭은 이것만 MNIST 를 읽고 학습을 하기 때문이다.
**저장소 뿌리에서** 돌린다.

규칙은 CLAUDE.md 의 Figures 항에 적어 둔 대로다.
- SVG, fonttype='path', transparent
- 글자는 모두 아스키

실행: python docs/ch25/ae/figures/anomaly.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["svg.fonttype"] = "path"

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms

HERE = Path(__file__).resolve().parent
INK = "#2f3337"
ACCENT = "#8a4b2a"
BLUE = "#2d6a8f"

NORMAL = 0          # 이 숫자만 보고 배운다
EPOCHS = 30


def _auc(pos, neg):
    """P(s_anomaly > s_normal) + 0.5 P(=) — 쪽의 6절에 적힌 정의 그대로."""
    s = np.concatenate([pos, neg])
    r = s.argsort().argsort().astype(float) + 1          # 평균 순위
    # 동점 처리: 같은 값끼리 평균 순위로 바꾼다
    order = np.argsort(s)
    s_sorted = s[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            r[order[i:j + 1]] = np.arange(i + 1, j + 2).mean()
        i = j + 1
    n1, n0 = len(pos), len(neg)
    return (r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def anomaly_scores():
    torch.manual_seed(0)
    tf = transforms.ToTensor()
    tr = datasets.MNIST(root="./data", train=True, download=False, transform=tf)
    te = datasets.MNIST(root="./data", train=False, download=False, transform=tf)

    ytr = tr.targets
    Xtr = tr.data[ytr == NORMAL].float().div(255).view(-1, 784)
    Xte = te.data.float().div(255).view(-1, 784)
    yte = te.targets.numpy()

    net = nn.Sequential(
        nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 32), nn.ReLU(),
        nn.Linear(32, 128), nn.ReLU(), nn.Linear(128, 784), nn.Sigmoid())
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    lossf = nn.MSELoss()
    for _ in range(EPOCHS):
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), 128):
            xb = Xtr[perm[i:i + 128]]
            opt.zero_grad()
            lossf(net(xb), xb).backward()
            opt.step()

    with torch.no_grad():
        err = ((net(Xte) - Xte) ** 2).mean(1).numpy()

    normal = err[yte == NORMAL]
    aucs = {d: _auc(err[yte == d], normal) for d in range(10) if d != NORMAL}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.0),
                                   gridspec_kw={"width_ratios": [1.1, 1]})

    ds = sorted(aucs, key=lambda d: aucs[d])
    vals = [aucs[d] for d in ds]
    cols = [ACCENT if v < 0.9 else BLUE for v in vals]
    ax1.barh(range(9), vals, color=cols, alpha=0.9, height=0.62)
    ax1.set_yticks(range(9))
    ax1.set_yticklabels([f"digit {d}" for d in ds], fontsize=8.5)
    ax1.axvline(0.5, color=INK, ls=":", lw=1.1)
    ax1.text(0.505, 8.6, "chance", fontsize=8, color=INK, alpha=0.75)
    for i, v in enumerate(vals):
        ax1.text(v + 0.008, i, f"{v:.3f}", va="center", fontsize=8.2,
                 color=INK)
    ax1.set_xlim(0.4, 1.06)
    ax1.set_xlabel("ROC-AUC against the digit it was trained on", fontsize=9)
    ax1.set_title(f"trained only on {NORMAL}s, scored by reconstruction error",
                  fontsize=9.5, color=INK, weight="bold")
    ax1.grid(alpha=0.13, axis="x")
    ax1.set_axisbelow(True)

    # AUC 가 0.99 라도 **임계**는 따로다. 이상의 95% 를 잡는 자리에서
    # 정상이 얼마나 잘못 걸리는지 센다.
    anom = err[yte != NORMAL]
    thr = np.quantile(anom, 0.05)              # 이상의 95% 가 이보다 크다
    fpr = float((normal > thr).mean())

    worst = ds[0]
    bins = np.linspace(0, np.quantile(err, 0.995), 60)
    ax2.hist(normal, bins=bins, color=INK, alpha=0.40,
             label=f"{NORMAL}  (normal)")
    ax2.hist(anom, bins=bins, color=ACCENT, alpha=0.62,
             label="all other digits")
    ax2.axvline(thr, color=BLUE, lw=1.8, ls="--")
    ax2.annotate(f"catch 95% of anomalies here\n"
                 f"-> {fpr:.1%} of normal 0s flagged",
                 xy=(thr, ax2.get_ylim()[1] * 0.45),
                 xytext=(thr * 0.10, ax2.get_ylim()[1] * 0.62),
                 ha="left", fontsize=8.3, color=BLUE,
                 arrowprops=dict(arrowstyle="->", color=BLUE, lw=0.9))
    ax2.set_xlabel("reconstruction error", fontsize=9)
    ax2.set_ylabel("count", fontsize=9)
    ax2.set_title("a good AUC is not a threshold", fontsize=9.5,
                  color=INK, weight="bold")
    ax2.legend(fontsize=8.5, framealpha=0.9, loc="upper right")

    for ax in (ax1, ax2):
        ax.tick_params(labelsize=8.5, colors=INK)
        for s in ax.spines.values():
            s.set_color(INK)
            s.set_linewidth(0.7)

    fig.text(0.5, -0.06,
             f"AUC runs from {min(vals):.3f} to {max(vals):.3f} - the ranking "
             f"is nearly perfect.\n"
             f"the threshold is still a trade: catching 95% of anomalies "
             f"falsely flags {fpr:.1%} of the normal class.",
             ha="center", fontsize=8.3, style="italic", color=INK)
    fig.tight_layout()
    fig.savefig(HERE / "anomaly_auc.svg", transparent=True,
                bbox_inches="tight")
    plt.close(fig)
    print(f"wrote anomaly_auc.svg  AUC {min(vals):.3f}-{max(vals):.3f}  "
          f"worst digit {worst}  thr {thr:.5f}  fpr {fpr:.4f}")
    print("  ", {d: round(aucs[d], 3) for d in sorted(aucs)})


if __name__ == "__main__":
    anomaly_scores()
