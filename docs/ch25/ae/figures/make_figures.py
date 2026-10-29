"""26장 오토인코더 기본 절 쪽의 그림.

규칙은 CLAUDE.md 의 Figures 항에 적어 둔 대로다.
- SVG, fonttype='path', transparent
- 글자는 모두 아스키 (matplotlib 기본 폰트에 한글이 없다)

수는 **여기서 직접 학습해** 얻는다. 체크포인트를 읽지 않는다.
Rumelhart, Hinton, Williams (1986) 의 8-3-8 인코더 문제를 그대로 돌린다.

실행: python docs/ch25/ae/figures/make_figures.py
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


def _train_encoder(n_in, n_hid, seed=0, steps=20_000, lr=0.02):
    """N 개의 one-hot 패턴을 n_hid 개의 은닉 유닛으로 보냈다 되돌린다."""
    torch.manual_seed(seed)
    X = torch.eye(n_in)
    enc = nn.Sequential(nn.Linear(n_in, n_hid), nn.Sigmoid())
    dec = nn.Linear(n_hid, n_in)
    opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()),
                           lr=lr)
    lossf = nn.CrossEntropyLoss()
    target = torch.arange(n_in)
    for _ in range(steps):
        opt.zero_grad()
        loss = lossf(dec(enc(X)), target)
        loss.backward()
        opt.step()
    with torch.no_grad():
        h = enc(X)
        ok = int((dec(h).argmax(1) == target).sum())
    return h.numpy(), ok


def encoder_problem():
    h8, ok8 = _train_encoder(8, 3)
    h8_2, ok8_2 = _train_encoder(8, 2)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(11.4, 3.7),
                                        gridspec_kw={"width_ratios":
                                                     [1.25, 1, 1]})

    # --- 1. 구조 ----------------------------------------------------------
    for i in range(8):
        ax1.add_patch(plt.Circle((0.1, 0.92 - i * 0.115), 0.028,
                                 facecolor="white", edgecolor=INK, lw=0.9))
        ax1.add_patch(plt.Circle((0.9, 0.92 - i * 0.115), 0.028,
                                 facecolor="white", edgecolor=INK, lw=0.9))
    for j in range(3):
        ax1.add_patch(plt.Circle((0.5, 0.70 - j * 0.16), 0.034,
                                 facecolor=ACCENT, edgecolor=INK, lw=0.9,
                                 alpha=0.85))
    for i in range(8):
        for j in range(3):
            ax1.plot([0.128, 0.466], [0.92 - i * 0.115, 0.70 - j * 0.16],
                     color=INK, lw=0.28, alpha=0.30, zorder=0)
            ax1.plot([0.534, 0.872], [0.70 - j * 0.16, 0.92 - i * 0.115],
                     color=INK, lw=0.28, alpha=0.30, zorder=0)
    ax1.text(0.1, 1.02, "8 in", ha="center", fontsize=9, color=INK)
    ax1.text(0.5, 1.02, "3", ha="center", fontsize=9, color=ACCENT,
             weight="bold")
    ax1.text(0.9, 1.02, "8 out", ha="center", fontsize=9, color=INK)
    ax1.text(0.5, -0.04, "8 patterns through 3 units.\n"
                         "log2(8) = 3 exactly.",
             ha="center", va="top", fontsize=8.6, color=INK)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(-0.2, 1.1)
    ax1.axis("off")
    ax1.set_title("the encoder problem", fontsize=10, color=INK,
                  weight="bold")

    # --- 2. 8-3-8 이 찾은 코드 --------------------------------------------
    ax2.imshow(h8, cmap="gray_r", vmin=0, vmax=1, aspect="auto")
    for i in range(8):
        for j in range(3):
            v = h8[i, j]
            ax2.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7.6,
                     color="white" if v > 0.5 else INK)
    ax2.set_xticks(range(3))
    ax2.set_xticklabels([f"h{j+1}" for j in range(3)], fontsize=8.5)
    ax2.set_yticks(range(8))
    ax2.set_yticklabels([f"pattern {i+1}" for i in range(8)], fontsize=7.6)
    b8 = np.minimum(h8, 1 - h8).mean()
    ax2.set_title(f"3 units: {ok8} of 8 recovered\n"
                  f"mean distance from 0 or 1: {b8:.4f}", fontsize=9.5,
                  color=INK, weight="bold")

    # --- 3. 2 개로 줄이면 --------------------------------------------------
    ax3.imshow(h8_2, cmap="gray_r", vmin=0, vmax=1, aspect="auto")
    for i in range(8):
        for j in range(2):
            v = h8_2[i, j]
            ax3.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7.6,
                     color="white" if v > 0.5 else INK)
    ax3.set_xticks(range(2))
    ax3.set_xticklabels([f"h{j+1}" for j in range(2)], fontsize=8.5)
    ax3.set_yticks(range(8))
    ax3.set_yticklabels([f"pattern {i+1}" for i in range(8)], fontsize=7.6)
    b2 = np.minimum(h8_2, 1 - h8_2).mean()
    ax3.set_title(f"2 units: {ok8_2} of 8 recovered too\n"
                  f"mean distance from 0 or 1: {b2:.4f}", fontsize=9.5,
                  color=ACCENT, weight="bold")

    for ax in (ax2, ax3):
        ax.tick_params(length=0, colors=INK)
        for s in ax.spines.values():
            s.set_visible(False)

    fig.text(0.5, -0.11,
             "trained from a fixed seed inside this script. with 3 units the "
             "code goes binary on its own, unasked.\n"
             "with 2 it still works - by giving the binary code up. log2(N) "
             "binds only if the units are binary.",
             ha="center", fontsize=8.3, style="italic", color=INK)
    fig.tight_layout()
    fig.savefig(HERE / "encoder_problem.svg", transparent=True,
                bbox_inches="tight")
    plt.close(fig)
    rounded = np.round(h8).astype(int)
    uniq = len({tuple(r) for r in rounded})
    print(f"wrote encoder_problem.svg  8-3-8 ok {ok8}/8  8-2-8 ok {ok8_2}/8  "
          f"distinct rounded codes {uniq}/8")
    print("codes:", [''.join(map(str, r)) for r in rounded])


if __name__ == "__main__":
    encoder_problem()
