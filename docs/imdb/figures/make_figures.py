"""
================================================================================
make_figures.py - 6장의 그림을 만든다
================================================================================

만드는 그림:
    rung23_curves.svg   2걸음과 3걸음을 30 에포크까지 늘려 나란히 그린다.
                        이 장의 규약인 5 에포크가 어디쯤인지 세로선으로 둔다
    lstm_curves.svg     4걸음 LSTM을 30 에포크까지 늘려 씨 다섯 벌의 곡선을
                        따로 그린다. 3걸음과 5걸음의 값을 가로선으로 둔다

왜 띠가 아니라 낱낱의 곡선인가:
    [4.2절](../../ch04/02_depth.md)의 깊이 곡선은 씨 다섯의 최소~최대를 띠로
    둘렀다. 여기서는 그렇게 하면 안 된다. 이 그림의 핵심이 **한 씨가 갑자기
    무너지는 사건**인데, 띠로 그리면 그것이 그저 넓은 띠로 뭉개져 보이고
    무너짐과 "씨마다 높이가 다름"을 구별할 수 없다.

그림 규칙(CLAUDE.md):
    - SVG로 저장하고 svg.fonttype='path'로 글자를 외곽선으로 만든다
    - transparent=True
    - 그림 안의 글자는 모두 ASCII (기본 글꼴에 한글이 없다)

실행:
    python make_figures.py
    LOG=/path/to/imdb_step4_e30.log python make_figures.py
================================================================================
"""

import os
import re

# 곡선 원자료(.npy)가 있는 곳. 실험 스크립트가 남긴 자리를 가리킨다.
NPY_DIR = os.environ.get("NPY_DIR", ".")
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

matplotlib.rcParams["svg.fonttype"] = "path"

LOG = os.environ.get("LOG", "imdb_step4_e30.log")
LOG_R2 = os.environ.get("LOG_R2", "ch6_rung2_e30.log")
LOG_R3 = os.environ.get("LOG_R3", "ch6_rung3_e30.log")

# 같은 규약으로 잰 다른 걸음들 (본문 표와 같은 값)
RUNG3 = 87.03          # 임베딩 + 평균
RUNG5 = 86.11          # 셀프 어텐션
RUNG2 = 78.80          # 선형 학습
PROTOCOL_EPOCHS = 5    # 이 장이 표준으로 삼은 예산


def load_curves(path):
    """로그에서 씨앗별 {에포크: 정확도}를 읽는다."""
    curves = defaultdict(dict)
    pat = re.compile(r"씨앗 (\d+)\s+에포크\s+(\d+)/\d+\s+([\d.]+)%")
    with open(path, encoding="utf-8") as f:
        for line in f:
            m = pat.search(line)
            if m:
                curves[int(m.group(1))][int(m.group(2))] = float(m.group(3))
    return curves


# === 그림: 씨 다섯 벌의 학습 곡선 ===========================================
def fig_lstm_curves():
    curves = load_curves(LOG)
    if not curves:
        raise SystemExit(f"no curve data in {LOG}")

    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]

    fig, ax = plt.subplots(figsize=(9, 4.4))

    # 견줄 값을 먼저 깔아 둔다
    ax.axhline(RUNG3, color="#000000", lw=1.2, ls="-", zorder=1)
    ax.axhline(RUNG5, color="#666666", lw=1.0, ls="--", zorder=1)
    ax.axhline(RUNG2, color="#999999", lw=0.9, ls=":", zorder=1)
    ax.axvline(PROTOCOL_EPOCHS, color="#bbbbbb", lw=0.9, ls="-", zorder=1)

    for i, seed in enumerate(sorted(curves)):
        eps = sorted(curves[seed])
        ax.plot(eps, [curves[seed][e] for e in eps],
                color=colors[i % len(colors)], lw=1.4, marker="o", ms=2.5,
                label=f"seed {seed}", zorder=3)

    xmax = max(e for s in curves.values() for e in s)
    ax.set_xlim(0.5, xmax + 0.5)
    ax.set_ylim(60, 89)
    ax.set_xticks(range(5, xmax + 1, 5))          # 정수 에포크만
    ax.set_xlabel("epoch")
    ax.set_ylabel("test accuracy (%)")

    # 가로선 이름표. 3걸음과 5걸음은 0.9밖에 안 떨어져 있어 서로 겹치므로,
    # 하나는 선 위, 하나는 선 아래에 두고 좌우로도 갈라 놓는다
    ax.text(0.8, RUNG3 + 0.3, "rung 3: embedding + mean  87.03",
            ha="left", va="bottom", fontsize=8, color="#000000")
    ax.text(xmax + 0.3, RUNG5 - 0.3, "rung 5: self-attention  86.11",
            ha="right", va="top", fontsize=8, color="#666666")
    # 이 선 언저리는 곡선이 무너지며 오르내리는 자리다. 뒤쪽 아래가 비어 있다
    ax.text(xmax * 0.78, RUNG2 - 0.5, "rung 2: linear  78.80",
            ha="center", va="top", fontsize=8, color="#999999")
    ax.text(PROTOCOL_EPOCHS + 0.3, 61.0, "chapter budget: 5 epochs",
            ha="left", va="bottom", fontsize=8, color="#888888")

    ax.legend(loc="lower right", fontsize=8, framealpha=0.0, ncol=5,
              bbox_to_anchor=(1.0, -0.02))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", lw=0.4, color="#dddddd", zorder=0)

    fig.tight_layout()
    fig.savefig("lstm_curves.svg", transparent=True)
    plt.close(fig)
    print("  lstm_curves.svg")

    # 그림과 본문의 수가 어긋나지 않도록, 그림을 만든 값을 함께 찍는다
    print("\n  seed   peak (epoch)   last (epoch)   worst-after-peak")
    for seed in sorted(curves):
        eps = sorted(curves[seed])
        vals = [curves[seed][e] for e in eps]
        pk = max(vals); pe = eps[vals.index(pk)]
        after = vals[vals.index(pk):]
        print(f"   {seed}     {pk:6.2f} ({pe:2d})   {vals[-1]:6.2f} ({eps[-1]:2d})   {min(after):6.2f}")


# === 그림: 2걸음과 3걸음을 같은 예산에서 ====================================
def fig_rung23_curves():
    """규약이 걸음들 사이에서 중립인가를 보이는 그림.

    5 에포크는 3걸음의 꼭대기 언저리이고 2걸음은 아직 오르는 중이다.
    그래서 그 자리에서 잰 차이가 가장 크게 나온다.
    """
    r2, r3 = load_curves(LOG_R2), load_curves(LOG_R3)
    if not r2 or not r3:
        raise SystemExit("need both rung2 and rung3 logs")

    def band(c):
        eps = sorted(set.intersection(*[set(v) for v in c.values()]))
        lo = [min(c[s][e] for s in c) for e in eps]
        hi = [max(c[s][e] for s in c) for e in eps]
        mid = [sum(c[s][e] for s in c) / len(c) for e in eps]
        return eps, lo, mid, hi

    fig, ax = plt.subplots(figsize=(9, 4.4))
    ax.axvline(PROTOCOL_EPOCHS, color="#bbbbbb", lw=1.0, zorder=1)
    for c, col, lab in ((r2, "#1f77b4", "rung 2: linear"),
                        (r3, "#d62728", "rung 3: embedding + mean")):
        eps, lo, mid, hi = band(c)
        ax.fill_between(eps, lo, hi, color=col, alpha=0.18, lw=0, zorder=2)
        ax.plot(eps, mid, color=col, lw=1.8, marker="o", ms=2.5, label=lab, zorder=3)

    xmax = max(max(v) for v in r2.values())
    ax.set_xlim(0.5, xmax + 0.5)
    ax.set_ylim(72, 90)
    ax.set_xticks(range(5, xmax + 1, 5))
    ax.set_xlabel("epoch")
    ax.set_ylabel("test accuracy (%)")
    ax.text(PROTOCOL_EPOCHS + 0.4, 72.6, "chapter budget: 5 epochs",
            ha="left", va="bottom", fontsize=8, color="#888888")
    ax.annotate("rung 3 peaks here (~ep 9)", xy=(9, 88.0), xytext=(13.5, 89.0),
                fontsize=8, color="#d62728",
                arrowprops=dict(arrowstyle="->", color="#d62728", lw=0.8))
    ax.annotate("rung 2 still rising at 30", xy=(30, 83.5), xytext=(20.5, 78.5),
                fontsize=8, color="#1f77b4",
                arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=0.8))
    ax.legend(loc="lower right", fontsize=8, framealpha=0.0)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.grid(axis="y", lw=0.4, color="#dddddd", zorder=0)
    fig.tight_layout()
    fig.savefig("rung23_curves.svg", transparent=True)
    plt.close(fig)
    print("  rung23_curves.svg")
    print("\n  epoch   rung2    rung3     gap")
    for e in (5, 10, 20, 30):
        a = [r2[s][e] for s in r2]; b = [r3[s][e] for s in r3]
        ma, mb = sum(a)/len(a), sum(b)/len(b)
        print(f"   {e:2d}    {ma:6.2f}   {mb:6.2f}   {mb-ma:+5.2f}")


def fig_stars_crossover():
    """여덟 갈래 별점에서 3걸음과 5걸음의 30에포크 곡선.

    이 장에서 가장 또렷한 예산 이야기다. 두 곡선이 **6에포크에서 교차**하고,
    이 장의 규약인 5에포크는 그 바로 한 에포크 앞이다. 어디서 멈추느냐에 따라
    같은 두 모델의 차이가 +7.75에서 -9.09까지 움직인다.
    """
    import numpy as np
    A = np.load(os.path.join(NPY_DIR, "curve_mean.npy"))   # (씨앗, 에포크)
    B = np.load(os.path.join(NPY_DIR, "curve_attn.npy"))
    ep = np.arange(1, A.shape[1] + 1)
    a, b = A.mean(0), B.mean(0)

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.fill_between(ep, A.min(0), A.max(0), color="#1f77b4", alpha=0.15, lw=0)
    ax.fill_between(ep, B.min(0), B.max(0), color="#d62728", alpha=0.15, lw=0)
    ax.plot(ep, a, color="#1f77b4", lw=1.8, label="rung 3  average")
    ax.plot(ep, b, color="#d62728", lw=1.8, label="rung 5  attention")

    ax.axvline(5, color="#888888", lw=0.9, ls="--")
    ax.text(5.3, 25.4, "chapter budget: 5 epochs", fontsize=8, color="#888888")
    ax.axvline(6, color="#444444", lw=0.9, ls=":")
    ax.text(6.3, 44.2, "curves cross: ep 6", fontsize=8, color="#444444")

    ia, ib = int(a.argmax()), int(b.argmax())
    ax.plot([ep[ia]], [a[ia]], "o", color="#1f77b4", ms=5)
    ax.plot([ep[ib]], [b[ib]], "o", color="#d62728", ms=5)
    ax.annotate(f"peak {a[ia]:.2f} (ep {ep[ia]})", xy=(ep[ia], a[ia]),
                xytext=(ep[ia] + 2.5, a[ia] + 1.6), fontsize=8, color="#1f77b4",
                arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=0.8))
    ax.annotate(f"peak {b[ib]:.2f} (ep {ep[ib]})", xy=(ep[ib], b[ib]),
                xytext=(ep[ib] + 3.0, b[ib] + 3.2), fontsize=8, color="#d62728",
                arrowprops=dict(arrowstyle="->", color="#d62728", lw=0.8))

    ax.axhline(20.09, color="#aaaaaa", lw=0.8, ls="-.")
    ax.text(21.0, 20.6, "majority class 20.09", fontsize=8, color="#888888")

    ax.set_xlabel("epoch"); ax.set_ylabel("test accuracy (%)")
    ax.set_title("IMDB, 8 star classes: attention is earlier, not better", fontsize=10)
    ax.set_xlim(1, 30); ax.set_ylim(19, 46)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.0)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.grid(axis="y", lw=0.4, color="#dddddd", zorder=0)
    fig.tight_layout()
    fig.savefig("stars_crossover.svg", transparent=True)
    plt.close(fig)
    print("  stars_crossover.svg")
    print("\n  epoch   rung3   rung5     gap")
    for e in (1, 3, 5, 6, 13, 20, 30):
        print(f"   {e:2d}    {a[e-1]:6.2f}  {b[e-1]:6.2f}   {b[e-1]-a[e-1]:+5.2f}")


# =========================================================================
#  들머리 그림 — 쪽의 물음을 세우는 자리. 그 쪽 수만 쓰고 다시 재지 않는다.
#  그림 안에는 설명 문장을 적지 않는다. 축 이름과 수까지다.
# =========================================================================

def fig_lead_two_ladders():
    """한 사다리는 끝까지 오르고, 한 사다리는 가운데서 꺾인다.

    MNIST 네 칸은 4.1절 표에서, IMDB 여섯 칸과 스프레드는 이 장 개요 표에서
    옮겼다. 여기서 다시 재지 않는다.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    mn = [82.03, 92.35, 97.90, 99.21]
    im = [65.16, 78.80, 87.03, 81.51, 86.11, 85.98]
    sp = [0.0, 0.23, 0.34, 4.74, 0.33, 0.17]      # 스텝1은 학습이 없어 스프레드가 없다

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.4, 4.5))

    axL.plot(range(1, 5), mn, "o-", color="tab:green", lw=2.4, ms=9)
    for x, v, (dx, dy, ha) in zip(range(1, 5), mn,
                                  [(0, -21, "center"), (14, -16, "left"),
                                   (14, -16, "left"), (0, 14, "center")]):
        axL.annotate(f"{v:.2f}", xy=(x, v), xytext=(dx, dy),
                     textcoords="offset points", ha=ha, fontsize=10,
                     color="tab:green")
    axL.set_xticks(range(1, 5)); axL.set_xlim(0.5, 4.5)
    axL.set_ylim(78, 103)
    axL.set_title("MNIST", fontsize=13, color="0.25")

    axR.errorbar(range(1, 7), im, yerr=sp, fmt="o-", color="#c2410c", lw=2.4,
                 ms=9, capsize=5, ecolor="#c2410c", elinewidth=1.6)
    # 점마다 비어 있는 쪽으로 — 4걸음은 오차막대가 아래로 길어 오른쪽에 둔다
    off = [(0, -21, "center"), (13, -16, "left"), (0, 17, "center"),
           (15, 0, "left"), (0, -21, "center"), (0, -21, "center")]
    for x, v, (dx, dy, ha) in zip(range(1, 7), im, off):
        axR.annotate(f"{v:.2f}", xy=(x, v), xytext=(dx, dy),
                     textcoords="offset points", ha=ha, fontsize=10,
                     color="#c2410c")
    axR.plot([3], [im[2]], "o", ms=16, mfc="none", mec="#c2410c", mew=2.2)
    axR.set_xticks(range(1, 7)); axR.set_xlim(0.5, 6.8)
    axR.set_ylim(61, 93)
    axR.set_title("IMDB", fontsize=13, color="0.25")

    for ax in (axL, axR):
        ax.set_xlabel("step", fontsize=10.5)
        ax.spines[["top", "right"]].set_visible(False)
    axL.set_ylabel("accuracy  (%)", fontsize=10.5)

    fig.subplots_adjust(left=0.08, right=0.98, top=0.89, bottom=0.14,
                        wspace=0.18)
    fig.savefig("lead_two_ladders.svg", transparent=True)
    plt.close(fig)
    print("wrote lead_two_ladders.svg")


def fig_lead_headroom():
    """같은 수법인데 남은 자리의 80% 를 먹고 30% 를 먹는다. 1걸음 쪽의 표."""
    import matplotlib.pyplot as plt
    rows = [("MNIST  (10 classes)", 10, 82.03, "tab:green"),
            ("IMDB  (2 classes)", 50, 65.16, "#c2410c")]
    fig, ax = plt.subplots(figsize=(9.2, 3.4))
    for y, (name, chance, got, col) in zip((1, 0), rows):
        ax.barh(y, 100 - chance, left=chance, height=0.46, color="0.88")
        ax.barh(y, got - chance, left=chance, height=0.46, color=col)
        ax.plot([chance, chance], [y - .3, y + .3], color="0.35", lw=1.6)
        ax.text(chance - 1.5, y, f"{chance}", ha="right", va="center",
                fontsize=10, color="0.35")
        ax.text(got + 1.5, y, f"{got:.2f}", ha="left", va="center",
                fontsize=11, color=col)
        pct = (got - chance) / (100 - chance) * 100
        ax.text(99, y + 0.34, f"{pct:.0f}%", ha="right", va="bottom",
                fontsize=13, color=col)
    ax.set_yticks([1, 0]); ax.set_yticklabels([r[0] for r in rows], fontsize=10.5)
    ax.set_xlim(0, 104); ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("accuracy  (%)", fontsize=10)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.subplots_adjust(left=0.20, right=0.98, top=0.93, bottom=0.22)
    fig.savefig("lead_headroom.svg", transparent=True)
    plt.close(fig); print("wrote lead_headroom.svg")


def fig_lead_words():
    """피처는 그대로인데 위로 올라오는 단어가 통째로 바뀐다. 2걸음 쪽의 표."""
    import matplotlib.pyplot as plt
    avg = ["and", "is", "of", "great", "in", "as",
           "the", "his", "very", "a", "best", "love"]
    learned = ["great", "excellent", "wonderful", "best", "love", "perfect",
               "amazing", "well", "beautiful", "loved", "highly", "favorite"]
    stop = {"and", "is", "of", "in", "as", "the", "his", "a", "very"}
    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    ax.set_xlim(0, 2); ax.set_ylim(0.3, 13.5); ax.axis("off")
    ax.text(0.5, 13.0, "averaged", ha="center", fontsize=12, color="0.4")
    ax.text(1.5, 13.0, "learned", ha="center", fontsize=12, color="#c2410c")
    for i2, (a, b) in enumerate(zip(avg, learned)):
        y = 12 - i2
        ax.text(0.5, y, a, ha="center", fontsize=12,
                color="0.72" if a in stop else "0.2")
        ax.text(1.5, y, b, ha="center", fontsize=12, color="#c2410c")
    ax.plot([1.0, 1.0], [0.6, 12.6], color="0.85", lw=1.0)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.97, bottom=0.02)
    fig.savefig("lead_words.svg", transparent=True)
    plt.close(fig); print("wrote lead_words.svg")


def fig_lead_embed_dim():
    """임베딩을 넓혀도 금세 평평해진다. 3걸음 쪽의 표."""
    import matplotlib.pyplot as plt
    d = [1, 4, 16, 64]
    a = [71.16, 80.05, 84.56, 86.96]
    fig, ax = plt.subplots(figsize=(8.4, 4.0))
    ax.plot(d, a, "o-", color="#c2410c", lw=2.4, ms=9)
    for x, v in zip(d, a):
        ax.annotate(f"{v:.2f}", xy=(x, v), xytext=(0, -21),
                    textcoords="offset points", ha="center", fontsize=10.5,
                    color="#c2410c")
    ax.set_xscale("log", base=2); ax.set_xticks(d); ax.set_xticklabels(d)
    ax.set_xlim(0.8, 90); ax.set_ylim(66, 91)
    ax.set_xlabel("numbers per word", fontsize=10.5)
    ax.set_ylabel("accuracy  (%)", fontsize=10.5)
    ax.spines[["top", "right"]].set_visible(False)
    fig.subplots_adjust(left=0.10, right=0.98, top=0.95, bottom=0.15)
    fig.savefig("lead_embed_dim.svg", transparent=True)
    plt.close(fig); print("wrote lead_embed_dim.svg")


def fig_lead_frozen_flip():
    """임베딩을 어떻게 다루느냐에 따라 3걸음->5걸음의 부호가 바뀐다."""
    import numpy as np
    import matplotlib.pyplot as plt
    names = ["learned", "learned,\nthen frozen", "borrowed,\nfrozen"]
    vals = [-0.83, 0.63, 12.22]
    cols = ["#b91c1c", "0.72", "tab:green"]
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    ax.bar(range(3), vals, color=cols, width=0.5)
    for i2, v in enumerate(vals):
        ax.text(i2, v + (0.5 if v > 0 else -1.1), f"{v:+.2f}", ha="center",
                fontsize=12.5, color=cols[i2])
    ax.axhline(0, color="0.35", lw=1.2)
    ax.set_xticks(range(3)); ax.set_xticklabels(names, fontsize=10.5)
    ax.set_ylim(-3.2, 14.2)
    ax.set_ylabel("step 3  ->  step 5   (%p)", fontsize=10.5)
    ax.spines[["top", "right"]].set_visible(False)
    fig.subplots_adjust(left=0.12, right=0.98, top=0.95, bottom=0.17)
    fig.savefig("lead_frozen_flip.svg", transparent=True)
    plt.close(fig); print("wrote lead_frozen_flip.svg")


def fig_lead_spread():
    """4걸음: 읽어야 할 것은 81.51 이 아니라 4.74 다."""
    import matplotlib.pyplot as plt
    rows = [("step 2   linear",    78.69, 78.92, 78.80, 0.23, "0.55"),
            ("step 3   average",   86.92, 87.26, 87.03, 0.34, "0.55"),
            ("step 4   LSTM",      78.76, 83.51, 81.51, 4.74, "#b91c1c")]
    fig, ax = plt.subplots(figsize=(9.0, 3.5))
    for i, (name, lo, hi, mu, sp, c) in enumerate(rows):
        y = 2 - i
        ax.plot([lo, hi], [y, y], lw=9, color=c, alpha=0.30,
                solid_capstyle="butt")
        ax.plot([mu], [y], "o", ms=10, color=c)
        ax.text(hi + 0.35, y, f"spread {sp:.2f}", va="center", fontsize=10.5,
                color=c)
    ax.axvline(78.80, color="0.55", ls=":", lw=1.3)
    ax.annotate("worst of five seeds: 78.76\nbelow step 2's average",
                xy=(78.76, 0), xytext=(79.6, -0.62), fontsize=10,
                color="#b91c1c",
                arrowprops=dict(arrowstyle="->", color="#b91c1c", lw=1.2))
    ax.set_yticks([2, 1, 0]); ax.set_yticklabels([r[0] for r in rows],
                                                 fontsize=10.5)
    ax.set_ylim(-0.95, 2.6); ax.set_xlim(78.0, 89.6)
    ax.set_xlabel("accuracy over five seeds, IMDB test set  (%)", fontsize=10.5)
    ax.set_title("the dot is the score; the bar is whether to believe it",
                 fontsize=11.5)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.subplots_adjust(left=0.19, right=0.97, top=0.86, bottom=0.26)
    fig.savefig("lead_spread.svg", transparent=True)
    plt.close(fig); print("wrote lead_spread.svg")


def fig_lead_path_vs_spread():
    """5걸음: 흔들린 하나만 경로가 길다."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8.6, 4.0))
    # 3걸음과 5걸음은 두 축 모두에서 사실상 같은 자리다. 겹쳐 그리는 것이
    # 이 그림의 요점이므로 하나를 테두리로 두어 둘 다 보이게 한다.
    ax.plot([1], [0.34], "o", ms=22, mfc="none", mec="0.45", mew=2.2)
    ax.plot([1], [0.33], "o", ms=12, color="#15803d")
    ax.annotate("step 3  average     spread 0.34\n"
                "step 5  attention   spread 0.33\n"
                "(the same place on both axes)",
                xy=(1, 0.33), xytext=(22, 6), textcoords="offset points",
                fontsize=10.5, color="0.25")
    ax.plot([400], [4.74], "o", ms=13, color="#b91c1c")
    ax.annotate("step 4  LSTM\nspread 4.74", xy=(400, 4.74), xytext=(-16, -4),
                textcoords="offset points", fontsize=10.5, color="#b91c1c",
                ha="right")
    ax.set_xscale("log"); ax.set_xticks([1, 10, 100, 400])
    ax.set_xticklabels(["1", "10", "100", "400"])
    ax.set_xlim(0.55, 1400); ax.set_ylim(-0.6, 6.0)
    ax.set_xlabel("steps a word must pass through to reach the output, "
                  "at most  (log)", fontsize=10.5)
    ax.set_ylabel("spread over five seeds\n(points)", fontsize=10.5)
    ax.set_title("the only model with a long path is the only unsteady one",
                 fontsize=11.5)
    ax.spines[["top", "right"]].set_visible(False)
    fig.subplots_adjust(left=0.13, right=0.97, top=0.87, bottom=0.17)
    fig.savefig("lead_path_vs_spread.svg", transparent=True)
    plt.close(fig); print("wrote lead_path_vs_spread.svg")


def fig_lead_steady_not_top():
    """6걸음: 가장 안 흔들리는데 가장 높지는 않다.

    흔들림이 0.17 에서 4.74 까지 28배 벌어져 있어 세로를 로그로 둔다.
    선형 눈금이면 LSTM 하나가 다 차지하고 나머지 넷이 바닥에 붙는다.
    """
    import matplotlib.pyplot as plt
    pts = [("step 2  linear",      78.80, 0.23, "0.55",   (0, 15), "center"),
           ("step 4  LSTM",        81.51, 4.74, "#b91c1c", (0, -24), "center"),
           ("step 6  transformer", 85.98, 0.17, "#1d4ed8", (0, -26), "center"),
           ("step 5  attention",   86.11, 0.33, "0.45",   (-8, 16), "center"),
           ("step 3  average",     87.03, 0.34, "#15803d", (13, -4), "left")]
    fig, ax = plt.subplots(figsize=(8.8, 4.2))
    for name, x, y, c, off, ha in pts:
        ax.plot([x], [y], "o", ms=12, color=c)
        ax.annotate(name, xy=(x, y), xytext=off, textcoords="offset points",
                    ha=ha, fontsize=10.5, color=c)
    ax.axvline(87.03, color="#15803d", ls=":", lw=1.3)
    ax.text(86.80, 3.4, "the ceiling is still step 3", ha="right",
            fontsize=10.5, color="#15803d")
    ax.axhline(0.17, color="#1d4ed8", ls=":", lw=1.3)
    ax.text(79.9, 0.185, "steadiest run of the chapter", fontsize=10.5,
            color="#1d4ed8")
    ax.set_yscale("log")
    ax.set_yticks([0.2, 0.5, 1, 2, 5])
    ax.set_yticklabels(["0.2", "0.5", "1", "2", "5"])
    ax.minorticks_off()
    ax.set_xlim(77.7, 90.6); ax.set_ylim(0.11, 9.0)
    ax.set_xlabel("accuracy, IMDB test set  (%)", fontsize=10.5)
    ax.set_ylabel("spread over five seeds\n(points, log)", fontsize=10.5)
    ax.set_title("steadiest is not highest", fontsize=11.5)
    ax.spines[["top", "right"]].set_visible(False)
    fig.subplots_adjust(left=0.13, right=0.97, top=0.88, bottom=0.16)
    fig.savefig("lead_steady_not_top.svg", transparent=True)
    plt.close(fig); print("wrote lead_steady_not_top.svg")


def fig_lead_protocol_flip():
    """별점: 같은 두 모델인데 재는 자리를 옮기면 부호가 바뀐다."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8.0, 4.0))
    vals = [0.80, -1.46]
    cols = ["#15803d", "#b91c1c"]
    labs = ["measured at epoch 5\n(this chapter's protocol)",
            "measured at each model's\nown best epoch"]
    ax.bar([0, 1], vals, width=0.42, color=cols)
    for x, v in zip([0, 1], vals):
        ax.text(x, v + (0.13 if v > 0 else -0.21), f"{v:+.2f} points",
                ha="center", fontsize=12, color=cols[0 if v > 0 else 1])
    ax.axhline(0, color="0.3", lw=1.2)
    ax.set_xticks([0, 1]); ax.set_xticklabels(labs, fontsize=10.5)
    ax.set_xlim(-0.6, 1.6); ax.set_ylim(-2.2, 1.5)
    ax.set_ylabel("attention minus average\n(8-class accuracy, points)",
                  fontsize=10.5)
    ax.set_title("same two models, same data - the ruler moved, the sign flipped",
                 fontsize=11.5)
    ax.spines[["top", "right", "bottom"]].set_visible(False)
    ax.tick_params(axis="x", length=0)
    fig.subplots_adjust(left=0.17, right=0.97, top=0.87, bottom=0.19)
    fig.savefig("lead_protocol_flip.svg", transparent=True)
    plt.close(fig); print("wrote lead_protocol_flip.svg")


if __name__ == "__main__":
    fig_lstm_curves()
    fig_rung23_curves()
    fig_stars_crossover()
    fig_lead_two_ladders()
    fig_lead_headroom()
    fig_lead_words()
    fig_lead_embed_dim()
    fig_lead_frozen_flip()
    fig_lead_spread()
    fig_lead_path_vs_spread()
    fig_lead_steady_not_top()
    fig_lead_protocol_flip()
