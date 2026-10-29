"""11.3절(SE · 비국소 · CNN에서 ViT로)의 그림.

그림에 적히는 수는 모두 세 쪽이 이미 찍어 둔 출력에서 가져오거나, 그 쪽의
식을 여기서 그대로 다시 계산한 것이다. 지어낸 수는 없다.

- receptive_field_vs_depth.svg  ← cnn_to_vit_bridge.md 의 [3] 출력과 3절 점화식
- attention_cost_share.svg      ← cnn_to_vit_bridge.md 의 [4] 출력과 연습문제 5
- se_params_by_stage.svg        ← squeeze_excitation.md 의 [3] 출력
- nonlocal_mac_floor.svg        ← non_local.md 의 [4](b) 출력

규칙은 CLAUDE.md 의 Figures 항에 적어 둔 대로다.
- SVG, fonttype='path', transparent=True
- 글자는 모두 아스키 (matplotlib 기본 폰트에 한글이 없다)

실행: python docs/ch10/vit/figures/make_figures.py
      python docs/ch10/vit/figures/make_figures.py --png   ← 눈으로 볼 PNG 도 함께
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["svg.fonttype"] = "path"

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
INK = "#2f3337"
ACCENT = "#8a4b2a"
COOL = "#2f5f8a"
MUTED = "#9aa0a6"

IMG, PATCH, DIM = 224, 16, 768


def style(ax):
    ax.tick_params(labelsize=8, colors=INK)
    for s in ax.spines.values():
        s.set_color(INK)
        s.set_linewidth(0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save(fig, name):
    fig.savefig(HERE / f"{name}.svg", transparent=True, bbox_inches="tight")
    if "--png" in sys.argv:          # 글자가 겹치지 않는지 눈으로 보려고 쓴다
        fig.savefig(HERE / f"{name}.png", transparent=False, dpi=170,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {name}.svg")


# ========================================================================
# 1. 리셉티브 필드의 폭 대 깊이 — 쪽의 점화식 그대로
#    r_l = r_{l-1} + (K_l - 1) * d_l * prod(s_i)
# ========================================================================

def receptive_field(layers):
    """cnn_to_vit_bridge.md 6절 [3]의 함수와 같다."""
    rf, jump = 1, 1
    for k, s, d in layers:
        rf += (k - 1) * d * jump
        jump *= s
    return rf, jump


def make_receptive_field():
    depth_s1 = 112                      # [3]: 3x3 stride 1 만 쌓을 때
    depth_alt = 12                      # [3]: stride 1,2 를 번갈아 쌓을 때

    ls1 = np.arange(0, depth_s1 + 5)
    rf1 = np.array([receptive_field([(3, 1, 1)] * int(n))[0] for n in ls1])

    ls2 = np.arange(0, depth_alt + 3)
    alt = ([(3, 1, 1), (3, 2, 1)] * 20)
    rf2 = np.array([receptive_field(alt[:int(n)])[0] for n in ls2])

    fig, ax = plt.subplots(figsize=(7.4, 4.0))

    ax.axhline(IMG, color=MUTED, lw=1.0, ls="--", zorder=1)
    ax.text(116, IMG * 1.1, "224 px = the whole image", fontsize=8,
            color=INK, ha="right", va="bottom")

    ax.plot(ls1, rf1, color=ACCENT, lw=1.6, label="conv 3x3, all strides 1")
    ax.plot(ls2, rf2, color=COOL, lw=1.6, ls="-.",
            label="conv 3x3, strides 1,2 alternating")

    # ViT: 조각 하나가 16 px, 어텐션 한 레이어면 토큰 197 개가 서로를 다 본다.
    ax.plot([0, 1], [PATCH, IMG], color=INK, lw=1.8)
    ax.plot([1, 118], [IMG, IMG], color=INK, lw=1.8,
            label="ViT: patch embed, then global attention")

    for x, y, txt, col, tx, ty, ha in (
        (depth_s1, 225, f"{depth_s1} layers", ACCENT, 92, 26, "right"),
        (depth_alt, 253, f"{depth_alt} layers", COOL, 22, 760, "left"),
        (1, IMG, "1 layer", INK, 3.5, 640, "left"),
    ):
        ax.plot([x], [y], "o", color=col, ms=5, zorder=5)
        ax.annotate(txt, xy=(x, y), xytext=(tx, ty), fontsize=8.5,
                    color=col, weight="bold", ha=ha, va="center",
                    arrowprops=dict(arrowstyle="-", color=col, lw=0.7))

    ax.set_xlim(-3, 118)
    ax.set_ylim(1, 1400)
    ax.set_yscale("log")
    ax.set_yticks([1, 4, 16, 64, 224, 1024])
    ax.set_yticklabels(["1", "4", "16", "64", "224", "1024"])
    ax.set_xlabel("layers stacked", fontsize=9, color=INK)
    ax.set_ylabel("receptive field width (pixels, log)", fontsize=9, color=INK)
    ax.set_title("depth buys a CNN what attention has at layer 1",
                 fontsize=10, weight="bold", color=INK, pad=8)
    ax.legend(fontsize=8, frameon=False, loc="lower right",
              bbox_to_anchor=(1.0, 0.03))
    style(ax)

    fig.text(0.5, -0.045,
             "width criterion only - a 224-wide field is not corner-to-corner "
             "reach, which needs 223 layers at stride 1",
             ha="center", fontsize=7.6, style="italic", color=INK)
    fig.tight_layout()
    save(fig, "receptive_field_vs_depth")


# ========================================================================
# 2. 어텐션 행렬이 블록 비용에서 차지하는 몫 — N / (6d + N)
# ========================================================================

def block_cost(n_tokens, dim=DIM, mlp_mult=4):
    """cnn_to_vit_bridge.md 6절 [4]의 함수와 같다."""
    qkv_proj = 4 * n_tokens * dim * dim
    attn_mat = 2 * n_tokens * n_tokens * dim
    mlp = 2 * mlp_mult * n_tokens * dim * dim
    return qkv_proj, attn_mat, mlp


def make_cost_share():
    n = np.logspace(np.log10(60), np.log10(60000), 400)
    share = 100 * n / (6 * DIM + n)

    fig, ax = plt.subplots(figsize=(7.4, 3.9))

    ax.plot(n, share, color=ACCENT, lw=1.8)
    ax.axvline(6 * DIM, color=MUTED, lw=1.0, ls="--")
    ax.axhline(50, color=MUTED, lw=0.8, ls=":")

    marks = (
        (197, "224 px image, patch 16", 66, 20, "left"),
        (577, "384 px image, patch 16", 200, 36, "left"),
        (6 * DIM, "N = 6d: the two terms are equal\n(68x68 patches, 1088 px)",
         40000, 24, "right"),
    )
    for nn, label, tx, ty, ha in marks:
        a, q, m = block_cost(int(nn))
        pct = 100 * q / (a + q + m)
        ax.plot([nn], [pct], "o", color=INK, ms=5.5, zorder=5)
        ax.annotate(f"{label}\nN = {nn} -> {pct:.1f}%", xy=(nn, pct),
                    xytext=(tx, ty), fontsize=8, color=INK, ha=ha,
                    va="center",
                    arrowprops=dict(arrowstyle="-", color=INK, lw=0.7))

    ax.set_xscale("log")
    ax.set_xlim(60, 60000)
    ax.set_ylim(0, 100)
    ax.set_xlabel("tokens N (log)", fontsize=9, color=INK)
    ax.set_ylabel("attention matrix,\n% of block multiply-adds",
                  fontsize=9, color=INK)
    ax.set_title("quadratic, yes - but a minority of the arithmetic "
                 "until N reaches 6d",
                 fontsize=10, weight="bold", color=INK, pad=8)
    style(ax)

    fig.text(0.5, -0.05,
             "share = N / (6d + N) with d = 768 and MLP ratio 4; "
             "projections and MLP only, no layer norm or softmax",
             ha="center", fontsize=7.6, style="italic", color=INK)
    fig.tight_layout()
    save(fig, "attention_cost_share")


# ========================================================================
# 3. SE 가 ResNet-50 에 더하는 파라메터가 어느 단계에 쌓이는가
#    수는 squeeze_excitation.md 4절 [3] 출력 그대로.
# ========================================================================

def make_se_params():
    stages = [("layer1", 256, 3, 8464), ("layer2", 512, 4, 33312),
              ("layer3", 1024, 6, 132160), ("layer4", 2048, 3, 526464)]
    totals = [k * per for _, _, k, per in stages]
    grand = sum(totals)                                # 2,530,992
    assert grand == 2530992, grand

    fig, ax = plt.subplots(figsize=(7.4, 3.3))
    ypos = np.arange(len(stages))[::-1]

    ax.barh(ypos, totals, height=0.56, color=ACCENT, alpha=0.85,
            edgecolor=INK, linewidth=0.6)
    for y, (name, c, k, per), tot in zip(ypos, stages, totals):
        ax.text(tot + 30000, y, f"{tot:,}   ({100 * tot / grand:.1f}%)",
                va="center", fontsize=8.5, color=INK, weight="bold")
        ax.text(-40000, y, f"{name}\nC = {c},  {k} x {per:,}",
                va="center", ha="right", fontsize=8, color=INK)

    ax.set_yticks([])
    ax.set_xlim(0, 2350000)
    ax.set_ylim(-0.7, len(stages) - 0.3)
    ax.set_xlabel("parameters SE adds to the stage", fontsize=9, color=INK)
    ax.set_title("SE costs 2,530,992 parameters, and 62.4% of them "
                 "sit in layer4",
                 fontsize=10, weight="bold", color=INK, pad=8)
    ax.spines["left"].set_visible(False)
    style(ax)
    ax.xaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1e6:.1f} M"))

    fig.text(0.5, -0.11,
             "ResNet-50 with r = 16: 25,557,032 -> 28,088,024 parameters, "
             "+9.90%; per block the cost grows with C^2",
             ha="center", fontsize=7.6, style="italic", color=INK)
    fig.tight_layout()
    save(fig, "se_params_by_stage")


# ========================================================================
# 4. 비국소 블록의 MAC 바닥 — 2NC^2 은 단계가 바뀌어도 꿈쩍하지 않는다
#    수는 non_local.md 4절 [4](b) 출력과 같은 식으로 여기서 다시 센다.
# ========================================================================

def make_nonlocal_floor():
    stages = [("layer1", 256, 56), ("layer2", 512, 28),
              ("layer3", 1024, 14), ("layer4", 2048, 7)]
    names, pair, proj, conv3 = [], [], [], []
    for name, c, s in stages:
        n = s * s
        names.append(f"{name}\nC={c}, {s}x{s}")
        pair.append(n * n * c / 1e9)            # 2 * N^2 * (C/2)
        proj.append(2 * n * c * c / 1e9)        # 4 * N * C * (C/2)
        conv3.append(n * 9 * c * c / 1e9)       # 3x3 conv in the same place

    x = np.arange(len(stages))
    fig, ax = plt.subplots(figsize=(7.4, 3.9))

    ax.bar(x, pair, width=0.5, color=ACCENT, alpha=0.9, edgecolor=INK,
           linewidth=0.6, label="similarity + aggregation  2 N^2 d")
    ax.bar(x, proj, width=0.5, bottom=pair, color=COOL, alpha=0.9,
           edgecolor=INK, linewidth=0.6, label="four 1x1 convolutions  4 N C d")
    ax.axhline(conv3[0], color=MUTED, lw=1.1, ls="--")
    ax.text(0.62, conv3[0] + 0.07,
            "3x3 conv in the same place: 1.850 G at every stage",
            fontsize=8, color=INK, ha="left")

    for i in range(len(stages)):
        ax.text(i, pair[i] + proj[i] + 0.07, f"{pair[i] + proj[i]:.3f} G",
                ha="center", fontsize=8.5, color=INK, weight="bold")
        ax.text(i, pair[i] + proj[i] / 2, f"{proj[i]:.3f} G", ha="center",
                va="center", fontsize=7.8, color="white", weight="bold")
    ax.annotate("2 N C d never moves:\nresolution halves, C doubles",
                xy=(2.27, pair[2] + proj[2] / 2), xytext=(3.45, 1.18),
                fontsize=8, color=COOL, ha="right", va="center",
                arrowprops=dict(arrowstyle="-", color=COOL, lw=0.7))

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8, color=INK)
    ax.set_ylim(0, 3.45)
    ax.set_ylabel("multiply-adds per sample (G)", fontsize=9, color=INK)
    ax.set_title("the N^2 term collapses 512x; the projection floor does not",
                 fontsize=10, weight="bold", color=INK, pad=8)
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    style(ax)

    fig.text(0.5, -0.12,
             "d = C/2, one 224x224 sample; MAC only - softmax, bias and "
             "memory traffic are not counted",
             ha="center", fontsize=7.6, style="italic", color=INK)
    fig.tight_layout()
    save(fig, "nonlocal_mac_floor")


if __name__ == "__main__":
    make_receptive_field()
    make_cost_share()
    make_se_params()
    make_nonlocal_floor()
