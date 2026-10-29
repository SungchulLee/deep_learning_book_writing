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


# === 그림으로 보는 AI: 같은 패널티, 다른 코드 =================================
#
# 베르누이 KL 희소성도 L1 도 **유닛별 평균**만 본다. 그래서 평균이 같으면
# 전혀 다른 두 코드에 같은 값을 매긴다. 쪽의 3절이 적어 둔 말을 수로 보인다.

RHO, N_SAMP, N_UNIT = 0.05, 200, 40


def _kl_penalty(h):
    r = np.clip(h.mean(0), 1e-12, 1 - 1e-12)
    return float((RHO * np.log(RHO / r)
                  + (1 - RHO) * np.log((1 - RHO) / (1 - r))).sum())


def _hoyer(h):
    """1 이면 완전히 성기고 0 이면 고르다."""
    x = h.flatten()
    m = x.size
    return float((np.sqrt(m) - np.abs(x).sum() / np.sqrt((x ** 2).sum()))
                 / (np.sqrt(m) - 1))


def same_penalty_other_code():
    rng = np.random.default_rng(0)

    # A: 유닛마다 정확히 10/200 자리만 1 -> 유닛별 평균이 꼭 0.05
    A = np.zeros((N_SAMP, N_UNIT))
    for j in range(N_UNIT):
        A[rng.choice(N_SAMP, int(RHO * N_SAMP), replace=False), j] = 1.0
    B = np.full((N_SAMP, N_UNIT), RHO)        # 모든 자리가 0.05

    rows = [("mean activation", f"{A.mean():.4f}", f"{B.mean():.4f}", True),
            ("KL penalty", f"{_kl_penalty(A):.1e}", f"{_kl_penalty(B):.1e}",
             True),
            ("L1 of activations", f"{np.abs(A).mean():.4f}",
             f"{np.abs(B).mean():.4f}", True),
            ("cells that are exactly 0", f"{(A == 0).mean():.0%}",
             f"{(B == 0).mean():.0%}", False),
            ("Hoyer sparsity", f"{_hoyer(A):.3f}", f"{_hoyer(B):.3f}", False)]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(11.6, 4.1),
                                        gridspec_kw={"width_ratios":
                                                     [1, 1, 1.5]})

    for ax, h, title in ((ax1, A, "A:  a few units fire, hard"),
                         (ax2, B, "B:  every unit, always, a little")):
        ax.imshow(h[:60], cmap="gray_r", vmin=0, vmax=1, aspect="auto",
                  interpolation="nearest")
        ax.set_title(title, fontsize=9.5, color=INK, weight="bold")
        ax.set_xlabel("40 latent units", fontsize=8.5)
        ax.set_ylabel("60 of 200 samples", fontsize=8.5)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color(INK)
            s.set_linewidth(0.8)

    ax3.axis("off")
    ax3.text(0.03, 0.95, "what the two penalties see", fontsize=9.5,
             color=INK, weight="bold", transform=ax3.transAxes)
    ax3.text(0.60, 0.865, "A", fontsize=9, color=INK, weight="bold",
             ha="center", transform=ax3.transAxes)
    ax3.text(0.86, 0.865, "B", fontsize=9, color=INK, weight="bold",
             ha="center", transform=ax3.transAxes)
    for i, (lab, va, vb, same) in enumerate(rows):
        y = 0.77 - i * 0.135
        col = ACCENT if same else INK
        ax3.text(0.03, y, lab, fontsize=8.4, color=col,
                 transform=ax3.transAxes)
        ax3.text(0.60, y, va, fontsize=8.6, color=col, ha="center",
                 family="monospace", transform=ax3.transAxes)
        ax3.text(0.86, y, vb, fontsize=8.6, color=col, ha="center",
                 family="monospace", transform=ax3.transAxes)
        if same:
            ax3.plot([0.50, 0.96], [y - 0.028, y - 0.028], color=ACCENT,
                     lw=0.6, alpha=0.45, transform=ax3.transAxes)
    ax3.text(0.03, 0.055, "amber rows are identical.\n"
                          "the penalties cannot tell these two apart.",
             fontsize=8.4, color=ACCENT, style="italic",
             transform=ax3.transAxes)

    fig.tight_layout()
    fig.savefig(HERE / "same_penalty_other_code.svg", transparent=True,
                bbox_inches="tight")
    plt.close(fig)
    print(f"wrote same_penalty_other_code.svg  KL {_kl_penalty(A):.2e} vs "
          f"{_kl_penalty(B):.2e}   hoyer {_hoyer(A):.3f} vs {_hoyer(B):.3f}")


if __name__ == "__main__":
    encoder_problem()
    same_penalty_other_code()
