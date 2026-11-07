"""
================================================================================
make_figures.py - 6장의 그림을 만든다
================================================================================

만드는 그림:
    pca_2d.svg   64차원 낱말 벡터를 2차원으로 사영한 흩뿌림. 네 묶음에만
                 색과 이름을 붙이고 나머지는 회색 점으로 둔다
    keep_rate.svg 남긴 차원에 따라 「원래 이웃 열 가운데 몇이 남는가」

왜 이 둘인가:
    6.6 쪽의 요점은 **그림이 그럴듯해 보이는 것과 가까움이 지켜진 것은
    다르다** 는 것이다. 왼쪽 그림만 있으면 묶음이 갈려 보여 믿게 되고,
    오른쪽 그림이 그 믿음을 깎는다. 둘이 함께 있어야 쪽이 선다.

그림 규칙(CLAUDE.md):
    - SVG 로 저장하고 svg.fonttype='path' 로 글자를 외곽선으로 만든다
    - transparent=True
    - 그림 안의 글자는 모두 ASCII (기본 글꼴에 한글이 없다)

수는 6.6 쪽의 프로그램이 찍는 것과 같다. 벡터도 같은 씨앗으로 다시
학습하므로 쪽의 수와 어긋나지 않는다.

실행:
    python make_figures.py
"""
import re
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["svg.fonttype"] = "path"
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import torch
import torch.nn as nn
import torch.nn.functional as F

SEED, VOCAB, DIM, WINDOW = 42, 8000, 64, 4
STEPS, BATCH, NEG, LR = 6000, 1024, 5, 3e-3
ROOT = Path("./data/aclImdb")
TOKEN = re.compile(r"[a-z']+")
torch.set_num_threads(4)

GROUPS = {
    "positive": (["great", "excellent", "wonderful", "brilliant", "superb"],
                 "#15803d"),
    "negative": (["awful", "terrible", "horrible", "dreadful", "lousy"],
                 "#b91c1c"),
    "function": (["the", "of", "and", "to", "in"], "#1d4ed8"),
    "people":   (["he", "she", "him", "her", "they"], "#a16207"),
}


def load_tokens(n_docs=6000):
    out = []
    for name in ("pos", "neg"):
        for f in sorted((ROOT / "train" / name).glob("*.txt"))[: n_docs // 2]:
            out += TOKEN.findall(f.read_text(encoding="utf-8")
                                 .replace("<br />", " ").lower())
    return out


class SkipGram(nn.Module):
    def __init__(self, v, d):
        super().__init__()
        self.mid, self.ctx = nn.Embedding(v, d), nn.Embedding(v, d)
        nn.init.normal_(self.mid.weight, std=0.1)
        nn.init.zeros_(self.ctx.weight)


def train(ids, counts, words):
    freq = torch.tensor([counts[w] for w in words], dtype=torch.float)
    p = freq ** 0.75; p /= p.sum()
    torch.manual_seed(SEED)
    m = SkipGram(VOCAB, DIM)
    opt = torch.optim.Adam(m.parameters(), lr=LR)
    g = torch.Generator().manual_seed(SEED)
    for _ in range(STEPS):
        c = torch.randint(WINDOW, len(ids) - WINDOW, (BATCH,), generator=g)
        off = torch.randint(1, WINDOW + 1, (BATCH,), generator=g)
        off *= torch.where(torch.rand(BATCH, generator=g) < 0.5, -1, 1)
        cc, oo = ids[c], ids[c + off]
        v = m.mid(cc)
        pos = (v * m.ctx(oo)).sum(-1)
        ni = torch.multinomial(p, BATCH * NEG, True, generator=g)
        ng = torch.bmm(m.ctx(ni).view(BATCH, NEG, -1),
                       v.unsqueeze(-1)).squeeze(-1)
        loss = -(F.logsigmoid(pos).mean() + F.logsigmoid(-ng).sum(1).mean())
        opt.zero_grad(); loss.backward(); opt.step()
    return F.normalize(m.mid.weight.detach(), dim=1)


def pca(X, k):
    Xc = X - X.mean(0, keepdim=True)
    _, S, V = torch.linalg.svd(Xc, full_matrices=False)
    var = S ** 2 / (len(X) - 1)
    return Xc @ V[:k].T, float(var[:k].sum() / var.sum())


def keep_rate(A, B, k=10, n=2000):
    a, b = F.normalize(A[:n], dim=1), F.normalize(B[:n], dim=1)
    ta = (a @ a.T).fill_diagonal_(-2).topk(k).indices
    tb = (b @ b.T).fill_diagonal_(-2).topk(k).indices
    return sum(len(set(ta[i].tolist()) & set(tb[i].tolist()))
               for i in range(n)) / (n * k)


def fig_pca_2d(P2, ev2, stoi):
    """네 묶음만 色을 주고 나머지는 회색으로 깔아 둔다."""
    fig, ax = plt.subplots(figsize=(7.6, 5.6))
    sub = P2[:2000]
    ax.scatter(sub[:, 0], sub[:, 1], s=3, c="0.82", linewidths=0,
               rasterized=True, zorder=1)
    for name, (ws, col) in GROUPS.items():
        idx = [stoi[w] for w in ws if w in stoi]
        pts = P2[idx]
        ax.scatter(pts[:, 0], pts[:, 1], s=46, color=col, zorder=3,
                   edgecolors="white", linewidths=0.8)
        for w, (x, y) in zip(ws, pts.tolist()):
            ax.annotate(w, (x, y), xytext=(4, 4), textcoords="offset points",
                        fontsize=9, color=col, zorder=4)
        c = pts.mean(0)
        dy = -26 if name in ("positive", "negative") else -22
        ax.annotate(name, c.tolist(), xytext=(0, dy),
                    textcoords="offset points", ha="center", fontsize=11,
                    color=col, weight="bold", zorder=4,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white",
                              ec="none", alpha=0.75))
    ax.set_title(f"8,000 word vectors squeezed into 2 dimensions "
                 f"({ev2 * 100:.1f}% of the spread)", fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])
    ax.spines[:].set_visible(False)
    fig.text(0.5, 0.025, "the groups look separated here - "
                         "the other figure says how much to trust that",
             ha="center", fontsize=9.5, color="0.4")
    fig.subplots_adjust(left=0.03, right=0.97, top=0.93, bottom=0.10)
    fig.savefig("pca_2d.svg", transparent=True)
    plt.close(fig); print("wrote pca_2d.svg")


def fig_keep_rate(ks, keeps, evs):
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    ax.plot(ks, [k * 10 for k in keeps], "o-", color="#b91c1c", lw=2.4, ms=9,
            label="neighbours kept (of 10)")
    ax.plot(ks, [e * 10 for e in evs], "s--", color="0.55", lw=1.8, ms=7,
            label="spread kept (x10)")
    for x, y in zip(ks, [k * 10 for k in keeps]):
        ax.annotate(f"{y:.2f}", (x, y), xytext=(0, 9),
                    textcoords="offset points", ha="center", fontsize=10,
                    color="#b91c1c")
    ax.axhline(0.05, color="0.7", ls=":", lw=1.2)
    ax.text(2.1, -0.42, "chance = 0.05", fontsize=9.5, color="0.5")
    ax.set_xscale("log", base=2); ax.set_xticks(ks)
    ax.set_xticklabels([str(k) for k in ks])
    ax.set_xlabel("dimensions kept", fontsize=10.5)
    ax.set_ylabel("of the original 10 nearest\nneighbours, how many remain",
                  fontsize=10.5)
    ax.set_ylim(-0.6, 11.2)
    ax.set_title("the picture keeps 0.25 of 10", fontsize=11.5)
    ax.legend(fontsize=9.5, frameon=False, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.subplots_adjust(left=0.15, right=0.97, top=0.89, bottom=0.16)
    fig.savefig("keep_rate.svg", transparent=True)
    plt.close(fig); print("wrote keep_rate.svg")


# =============================================================================
# 6.1 조회는 행렬 곱이다 / 원핫을 실제로 만들면 드는 값
# =============================================================================
def fig_lookup_is_matmul():
    """6.1 2절 - 표에서 꺼내는 것과 원핫을 곱하는 것이 왜 같은가.

    식만으로는 안 보이는 것을 그린다. 원핫을 세로로 세워 행렬의 줄과
    나란히 두면, 1 인 칸 하나가 어느 줄을 살리는지가 바로 보인다.

    번호 47 은 쪽의 출력에 있는 good 의 번호 그대로다.
    """
    N, HOT = 9, 4
    CH, ROW_W, VEC_W = 3.4, 30.0, 5.0
    top, left = 45.0, 6.0
    ybot = lambda r: top - (r + 1) * CH

    fig, ax = plt.subplots(figsize=(10.0, 4.4))
    ax.set_xlim(0, 100); ax.set_ylim(0, 52); ax.axis("off")

    # --- 원핫 (세로로 세운다) ---
    for r in range(N):
        on = r == HOT
        ax.add_patch(Rectangle((left, ybot(r)), VEC_W, CH,
                               facecolor="#b91c1c" if on else "white",
                               edgecolor="#374151", lw=1))
        ax.text(left + VEC_W / 2, ybot(r) + CH / 2, "1" if on else "0",
                ha="center", va="center", fontsize=11,
                color="white" if on else "#9ca3af")
    ax.text(left + VEC_W / 2, top + 1.6, "one-hot", ha="center", fontsize=11)
    ax.text(left + VEC_W / 2, ybot(N - 1) - 2.4,
            "20,000 slots\nonly slot 47 is 1", ha="center", va="top",
            fontsize=9.5, color="0.45")

    ax.text(left + VEC_W + 3.0, top - N * CH / 2, "x", fontsize=16,
            va="center", ha="center")

    # --- 표 W (줄을 나란히) ---
    mx = left + VEC_W + 6.0
    for r in range(N):
        on = r == HOT
        ax.add_patch(Rectangle((mx, ybot(r)), ROW_W, CH,
                               facecolor="#fca5a5" if on else "white",
                               edgecolor="#374151", lw=1.4 if on else 1))
    ax.text(mx + ROW_W / 2, top + 1.6, "the table W   (20,000 x 64)",
            ha="center", fontsize=11)
    ax.text(mx + ROW_W / 2, ybot(HOT) + CH / 2, "row 47", ha="center",
            va="center", fontsize=10.5, color="#7f1d1d")
    ax.text(mx + ROW_W / 2, ybot(N - 1) - 2.4,
            "one row per word", ha="center", va="top",
            fontsize=9.5, color="0.45")

    ax.text(mx + ROW_W + 3.2, top - N * CH / 2, "=", fontsize=16,
            va="center", ha="center")

    # --- 나온 것: 줄 하나 ---
    rx = mx + ROW_W + 6.4
    ax.add_patch(Rectangle((rx, ybot(HOT)), ROW_W, CH, facecolor="#fca5a5",
                           edgecolor="#374151", lw=1.4))
    ax.text(rx + ROW_W / 2, ybot(HOT) + CH / 2, "row 47 of W", ha="center",
            va="center", fontsize=11)
    ax.text(rx + ROW_W / 2, top + 1.6, '64 numbers for "good"',
            ha="center", fontsize=11)

    ax.text(50, 1.2, "the single 1 keeps one row and zeroes every other - "
            "so multiplying IS looking up",
            ha="center", fontsize=11.5, color="#b91c1c")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.97, bottom=0.03)
    fig.savefig("lookup_is_matmul.svg", transparent=True)
    plt.close(fig); print("wrote lookup_is_matmul.svg")


def fig_cost_of_onehot():
    """6.1 3절 - 같은 식인데 실제로 만들면 값이 다르다.

    수는 쪽의 출력 그대로다. 시간은 이 글을 쓴 기계에서 잰 것이고,
    메모리는 102,400 x 20,000 x 4 바이트라 기계와 무관하다.
    """
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(10.4, 4.0))
    pairs = [(axL, "time to turn 102,400 words into vectors",
              [("lookup", 1.61, "#15803d"), ("one-hot matmul", 1914.39,
                                             "#b91c1c")],
              "milliseconds (log scale)", "1,193x", "{:,.2f} ms"),
             (axR, "memory just to hold the thing",
              [("the table W", 4.9, "#15803d"), ("one-hot matrix", 7812.0,
                                                 "#b91c1c")],
              "megabytes (log scale)", "1,594x", "{:,.1f} MB")]
    for ax, title, bars, ylab, ratio, fmt in pairs:
        for i, (name, v, c) in enumerate(bars):
            ax.bar(i, v, 0.52, color=c)
            ax.text(i, v * 1.25, fmt.format(v), ha="center", fontsize=12)
        ax.set_yscale("log")
        ax.set_ylim(1, bars[1][1] * 12)
        ax.set_xticks([0, 1]); ax.set_xticklabels([b[0] for b in bars],
                                                  fontsize=11.5)
        ax.set_xlim(-0.6, 1.9)
        ax.set_ylabel(ylab, fontsize=10.5)
        ax.set_title(title, fontsize=11.5)
        ax.annotate("", (1.55, bars[0][1]), (1.55, bars[1][1]),
                    arrowprops=dict(arrowstyle="<->", color="0.35", lw=1.6))
        ax.text(1.62, (bars[0][1] * bars[1][1]) ** 0.5, ratio, fontsize=13,
                va="center", color="0.2")
        ax.grid(axis="y", alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig("cost_of_onehot.svg", transparent=True)
    plt.close(fig); print("wrote cost_of_onehot.svg")


# =============================================================================
# 6.2 Skip-gram 이 푸는 문제 / 네거티브 샘플링이 버는 자리 / 0.75 제곱
# =============================================================================
def fig_skipgram_task():
    """6.2 2절 - 기계가 실제로 푸는 문제가 무엇인지 그린다.

    글로만 적으면 「가운데 낱말로 둘레를 맞힌다」가 추상적으로 들린다.
    글월 하나를 늘어놓고 창을 씌우면 짝이 어떻게 생기는지 바로 보인다.
    글월은 IMDB 영화평에 흔한 꼴로 골랐고 지어낸 수는 없다.
    """
    sent = ["this", "was", "a", "really", "terrible", "movie", "with", "bad",
            "acting"]
    centre = 4
    fig, ax = plt.subplots(figsize=(10.2, 3.9))
    ax.set_xlim(-0.5, len(sent) - 0.5); ax.set_ylim(-2.3, 2.0); ax.axis("off")

    for i, w in enumerate(sent):
        inwin = abs(i - centre) <= 4 and i != centre
        is_c = i == centre
        ax.text(i, 0, w, ha="center", va="center", fontsize=12.5,
                color="white" if is_c else ("#111827" if inwin else "#9ca3af"),
                bbox=dict(boxstyle="round,pad=0.34",
                          facecolor="#b91c1c" if is_c else
                          ("#fee2e2" if inwin else "#f3f4f6"),
                          edgecolor="#b91c1c" if is_c else "none"))
        if inwin:
            ax.annotate("", (i, -0.42), (centre, -0.42),
                        arrowprops=dict(arrowstyle="->", color="#b91c1c",
                                        lw=1.1, alpha=0.65,
                                        connectionstyle="arc3,rad=0.42"))

    ax.text(centre, 1.25, "centre word", ha="center", fontsize=11,
            color="#b91c1c")
    ax.text(centre, -1.65,
            "window of 4 on each side - every arrow is one training pair",
            ha="center", fontsize=11, color="#374151")
    ax.text(centre, -2.15,
            '"given terrible, was bad nearby?"   yes.   '
            '"given terrible, was kitchen nearby?"   no.',
            ha="center", fontsize=10.5, color="0.4")
    fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
    fig.savefig("skipgram_task.svg", transparent=True)
    plt.close(fig); print("wrote skipgram_task.svg")


def fig_negative_sampling_scaling():
    """6.2 4절 - 네거티브 샘플링의 값어치는 어휘를 키워야 보인다.

    수는 쪽의 출력 그대로다 (점수 매기는 비용만, 뽑는 일은 뺐다).
    """
    v = [8_000, 32_000, 128_000, 512_000]
    neg = [2.79, 2.37, 5.11, 14.58]
    full = [6.86, 23.90, 121.27, 507.30]

    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    ax.plot(v, full, "s-", color="#b91c1c", lw=2.2, ms=9,
            label="score the whole vocabulary")
    ax.plot(v, neg, "o-", color="#15803d", lw=2.2, ms=9,
            label="negative sampling (6 pairs)")
    for x, a, b in zip(v, neg, full):
        ax.annotate(f"{b:.0f}", (x, b), (0, 11), textcoords="offset points",
                    ha="center", fontsize=10, color="#b91c1c")
        ax.annotate(f"{a:.1f}", (x, a), (0, -17), textcoords="offset points",
                    ha="center", fontsize=10, color="#15803d")
        ax.annotate("", (x, a), (x, b),
                    arrowprops=dict(arrowstyle="-", color="0.75", lw=1))
    ax.text(512_000, (neg[-1] * full[-1]) ** 0.5, "  35x", fontsize=13,
            va="center", color="0.2")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks(v); ax.set_xticklabels([f"{x // 1000}k" for x in v],
                                         fontsize=11.5)
    ax.set_xlim(6_000, 1_150_000); ax.set_ylim(1.4, 1300)
    ax.set_xlabel("vocabulary size", fontsize=11)
    ax.set_ylabel("milliseconds per step   (log scale)", fontsize=11)
    ax.set_title("the green line is nearly flat - that is the whole point",
                 fontsize=11.5)
    ax.legend(fontsize=10.5, frameon=False, loc="upper left")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig("negative_sampling_scaling.svg", transparent=True)
    plt.close(fig); print("wrote negative_sampling_scaling.svg")


def fig_three_quarter_power(counts, words):
    """6.2 4절 - 0.75 제곱이 어느 사이를 노린 값인가.

    세 분포를 같은 축에 둔다. 빈도 그대로 뽑으면 the 가 거의 다 먹고,
    고르게 뽑으면 아무도 안 쓰는 낱말만 나온다. 0.75 제곱이 그 사이다.
    수는 이 장이 쓰는 말뭉치(영화평 6,000편, 어휘 8,000)에서 센 것이다.
    """
    f = torch.tensor([counts[w] for w in words], dtype=torch.double)
    dists = [("p ~ frequency", f / f.sum(), "#b91c1c"),
             ("p ~ frequency^0.75", f ** 0.75 / (f ** 0.75).sum(), "#15803d"),
             ("p ~ uniform", torch.ones_like(f) / len(f), "#1d4ed8")]
    rank = torch.arange(1, len(f) + 1)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.0, 4.3),
                                   gridspec_kw={"width_ratios": [1.3, 1]})
    for name, p, c in dists:
        axL.plot(rank, p, color=c, lw=2, label=name)
    axL.set_xscale("log"); axL.set_yscale("log")
    axL.set_xlabel("word, most frequent first (log scale)", fontsize=10.5)
    axL.set_ylabel("chance of being drawn as a negative", fontsize=10.5)
    axL.set_title("damping the common words without erasing them",
                  fontsize=11.5)
    axL.legend(fontsize=10.5, frameon=False)
    axL.grid(alpha=0.2)
    axL.spines[["top", "right"]].set_visible(False)

    top = 100
    for i, (name, p, c) in enumerate(dists):
        share = 100 * float(p[:top].sum())
        axR.bar(i, share, 0.5, color=c)
        axR.text(i, share + 1.6, f"{share:.1f}%", ha="center", fontsize=12)
    axR.set_xticks(range(3))
    axR.set_xticklabels(["frequency", "^0.75", "uniform"], fontsize=11)
    axR.set_ylim(0, 100)
    axR.set_ylabel(f"share taken by the top {top} words (%)", fontsize=10.5)
    axR.set_title("how much the commonest words take", fontsize=11.5)
    axR.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig("three_quarter_power.svg", transparent=True)
    plt.close(fig)
    print("wrote three_quarter_power.svg")
    for name, p, _ in dists:
        print(f"   {name:<22} top-{top} share {100 * float(p[:top].sum()):.1f}%"
              f"   'the' {100 * float(p[0]):.2f}%")


# =============================================================================
# 6.3 세는 길 - 횟수와 PPMI 는 서로 다른 줄을 세운다 / 세 걸음
# =============================================================================
def fig_count_vs_ppmi(C, M, words, stoi, probe="the"):
    """6.3 2절 - 같은 낱말에 두 가지 자를 대면 줄이 달라진다.

    가로는 probe 와 같이 나온 횟수, 세로는 PPMI 다. 오른쪽 아래에 몰린
    것이 「흔해서 많이 나온」 기능어이고, 왼쪽 위가 「드문데도 probe 를
    달고 나오는」 낱말이다. 두 자가 서로 다른 것을 재고 있음을 보인다.

    수는 이 장의 말뭉치에서 직접 센 것이고 쪽의 출력과 같은 표에서 온다.
    """
    i = stoi[probe]
    cnt, ppmi = C[i].clone(), M[i].clone()
    cnt[i] = 0
    keep = cnt > 0
    idx = torch.nonzero(keep).squeeze(1)

    by_cnt = idx[cnt[idx].argsort(descending=True)][:6].tolist()
    by_pmi = idx[ppmi[idx].argsort(descending=True)][:6].tolist()

    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    ax.scatter(cnt[idx], ppmi[idx], s=7, color="#9ca3af", alpha=0.35,
               edgecolors="none")
    for group, c, lab in ((by_cnt, "#b91c1c", "top 6 by raw count"),
                          (by_pmi, "#15803d", "top 6 by PPMI")):
        ax.scatter(cnt[group], ppmi[group], s=70, color=c, zorder=3,
                   label=lab)
        for k, j in enumerate(group):          # 라벨이 겹치지 않게 번갈아
            ax.annotate(words[j], (cnt[j], ppmi[j]),
                        (7, 6) if k % 2 == 0 else (7, -13),
                        textcoords="offset points", fontsize=11, color=c)
    ax.set_xscale("log")
    ax.set_xlabel(f'how often it appeared next to "{probe}"  (log scale)',
                  fontsize=11)
    ax.set_ylabel("PPMI with the same word", fontsize=11)
    ax.set_title("two rulers, two different rankings - "
                 "the red words are merely common", fontsize=11.5)
    ax.legend(fontsize=10.5, frameon=False, loc="upper right")
    ax.grid(alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig("count_vs_ppmi.svg", transparent=True)
    plt.close(fig); print("wrote count_vs_ppmi.svg")


def fig_counting_pipeline():
    """6.3 - 세는 길의 세 걸음을 크기와 함께 그린다.

    세고 -> 고치고 -> 줄인다. 어느 걸음에서 표가 어떻게 바뀌는지를
    네모의 크기와 글자로 보인다. 지어낸 수는 없다 - 8,000 과 64 는
    이 장이 쓰는 값이고 3.5% 는 쪽의 출력이다.
    """
    fig, ax = plt.subplots(figsize=(10.4, 3.9))
    ax.set_xlim(0, 100); ax.set_ylim(-9, 34); ax.axis("off")
    steps = [(6, 22, 22, "co-occurrence\ncounts",
              "8,000 x 8,000\n3.5% non-zero", "#9ca3af"),
             (39, 22, 22, "PPMI", "same shape,\ncommonness divided out",
              "#1d4ed8"),
             (72, 22, 7, "word vectors", "8,000 x 64\nby SVD, as in 5.1", "#15803d")]
    for x, w, h, title, sub, c in steps:
        ax.add_patch(Rectangle((x, 18 - h / 2), w, h, facecolor=c, alpha=0.16,
                               edgecolor=c, lw=1.6))
        ax.text(x + w / 2, 18, title, ha="center", va="center", fontsize=12.5,
                color=c)
        ax.text(x + w / 2, 18 - h / 2 - 2.0, sub, ha="center", va="top",
                fontsize=9.5, color="0.4")
    for x in (29.5, 62.5):
        ax.annotate("", (x + 7.5, 18), (x, 18),
                    arrowprops=dict(arrowstyle="-|>", color="#374151", lw=1.8))
    ax.text(33.2, 21.0, "divide out\ncommonness", ha="center",
            fontsize=10, color="#374151")
    ax.text(66.2, 20.2, "shrink", ha="center", fontsize=10, color="#374151")
    ax.text(50, -7.6, "no gradient descent anywhere - "
            "count once, divide once, factorize once",
            ha="center", fontsize=11, color="#15803d")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.98, bottom=0.02)
    fig.savefig("counting_pipeline.svg", transparent=True)
    plt.close(fig); print("wrote counting_pipeline.svg")


# =============================================================================
# 6.4 / 6.5 / 6.6 - 각 쪽의 결론을 그림 하나로
# =============================================================================
def fig_bias_split():
    """6.4 5절 - 로스는 합만 묶고 쪼개는 방식은 정해 주지 않는다.

    수는 그 쪽의 출력 그대로다. 왼쪽 막대(식을 얼마나 잘 맞추나)는
    셋이 거의 같은데 오른쪽 막대(내적이 PMI 를 따라가나)만 크게 흔들린다.
    그것이 「로스가 셋을 구별하지 못한다」는 말의 그림이다.
    """
    rows = [("biases learned\n(as in the paper)", 0.837, -0.248),
            ("biases pinned\nto log X_i", 0.777, 0.783),
            ("no biases", 0.833, -0.333)]
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    W = 0.34
    for i, (name, fit, pmi) in enumerate(rows):
        ax.bar(i - W / 2, fit, W * 0.92, color="#6b7280",
               label="fits the objective" if i == 0 else None)
        ax.bar(i + W / 2, pmi, W * 0.92,
               color="#15803d" if pmi > 0 else "#b91c1c",
               label="inner product vs PMI  (green +, red -)" if i == 0 else None)
        ax.text(i - W / 2, fit + 0.03, f"{fit:+.3f}", ha="center", fontsize=11)
        ax.text(i + W / 2, pmi + (0.03 if pmi > 0 else -0.09),
                f"{pmi:+.3f}", ha="center", fontsize=11)
    ax.axhline(0, color="0.3", lw=1)
    ax.set_xticks(range(3)); ax.set_xticklabels([r[0] for r in rows],
                                                fontsize=11)
    ax.set_ylim(-0.55, 1.08)
    ax.set_ylabel("correlation", fontsize=11)
    ax.set_title("all three fit the loss equally well - "
                 "only the interpretation moves", fontsize=11.5)
    ax.legend(fontsize=10.5, frameon=False, loc="upper center", ncol=2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig("bias_split.svg", transparent=True)
    plt.close(fig); print("wrote bias_split.svg")


def fig_antonyms_stay_close():
    """6.5 6절 / 6.6 5절 - 자료를 천 배 늘려도 반대말은 안 갈린다.

    수는 두 쪽의 출력 그대로다. 왼쪽이 영화평 6,000편으로 만든 것,
    오른쪽이 GloVe 60억 낱말판이다. 같은 자(코사인 닮음)로 쟀다.
    """
    pairs = [("movie - film", 0.801, 0.931, "same thing"),
             ("good - great", 0.748, 0.798, "synonym"),
             ("good - bad", 0.594, 0.796, "OPPOSITE"),
             ("good - terrible", 0.578, 0.595, "OPPOSITE"),
             ("movie - kitchen", 0.348, 0.363, "unrelated")]
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    y = list(range(len(pairs)))[::-1]
    for (name, ours, glove, kind), yy in zip(pairs, y):
        opp = kind == "OPPOSITE"
        ax.plot([ours, glove], [yy, yy], "-", lw=2,
                color="#b91c1c" if opp else "#9ca3af", zorder=1)
        ax.scatter([ours], [yy], s=70, color="#9ca3af", zorder=3)
        ax.scatter([glove], [yy], s=95,
                   color="#b91c1c" if opp else "#374151", zorder=3)
        ax.text(0.265, yy, name, ha="right", va="center", fontsize=11.5,
                color="#b91c1c" if opp else "#111827")
        ax.annotate(f"{glove:.3f}", (glove, yy), (8, 0),
                    textcoords="offset points", va="center", fontsize=10.5,
                    color="#b91c1c" if opp else "#374151")
        ax.annotate(f"{ours:.3f}", (ours, yy), (-8, 0), ha="right",
                    textcoords="offset points", va="center", fontsize=10,
                    color="#9ca3af")
    ax.axvspan(0.78, 0.82, color="#b91c1c", alpha=0.07)
    ax.text(0.80, len(pairs) - 0.45, "three of these land\nin the same band",
            ha="center", fontsize=10, color="#b91c1c")
    ax.scatter([], [], s=70, color="#9ca3af", label="our 6,000 reviews")
    ax.scatter([], [], s=95, color="#374151", label="GloVe, 6 billion words")
    ax.set_xlim(0.20, 1.02); ax.set_ylim(-0.7, len(pairs) - 0.1)
    ax.set_yticks([])
    ax.set_xlabel("cosine similarity", fontsize=11)
    ax.set_title("a thousand times more text does not separate "
                 "good from bad", fontsize=11.5)
    ax.legend(fontsize=10.5, frameon=False, loc="lower right")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig("antonyms_stay_close.svg", transparent=True)
    plt.close(fig); print("wrote antonyms_stay_close.svg")


# =============================================================================
# 6.4 GloVe - 얼음과 김 (「그림으로 보는 AI」)
# =============================================================================
def fig_ice_steam():
    """GloVe 가 태어난 자리 - Pennington 외 (2014) Table 1 을 그린다.

    수는 지어낸 것이 아니라 논문 Table 1 에 실린 그대로다 (60억 낱말
    말뭉치에서 잰 것). 요점은 「확률 두 개를 따로 보면 안 보이는 것이
    둘을 나누면 보인다」는 것이다.
    """
    ks = ["solid", "gas", "water", "fashion"]
    ice = [1.9e-4, 6.6e-5, 3.0e-3, 1.7e-5]
    steam = [2.2e-5, 7.8e-4, 2.2e-3, 1.8e-5]
    ratio = [8.9, 8.5e-2, 1.36, 0.96]

    fig, (axT, axB) = plt.subplots(
        2, 1, figsize=(7.6, 5.6), sharex=True,
        gridspec_kw={"height_ratios": [1.5, 1], "hspace": 0.42})

    x = range(4)
    W = 0.36
    for j, (v, c, lab) in enumerate(((ice, "#1d4ed8", "next to  ice"),
                                     (steam, "#b91c1c", "next to  steam"))):
        axT.bar([i + (j - 0.5) * W for i in x], v, W * 0.9, color=c, label=lab)
    axT.set_yscale("log")
    axT.set_ylim(5e-6, 2e-2)
    axT.set_ylabel("how often the word shows up\n(log scale)", fontsize=10.5)
    axT.legend(fontsize=11, frameon=False, loc="upper left")
    axT.set_title("Two words, four neighbours - the counts alone do not sort them",
                  fontsize=11.5)
    axT.spines[["top", "right"]].set_visible(False)

    for i, r in enumerate(ratio):
        far = r > 3 or r < 1 / 3
        axB.bar(i, r, 0.5, color="#15803d" if far else "#9ca3af")
        axB.text(i, r * (1.35 if r > 1 else 1.35), f"{r:g}", ha="center",
                 fontsize=12, color="#15803d" if far else "#4b5563")
    axB.axhline(1.0, color="0.4", ls=":", lw=1.3)
    axB.text(3.52, 1.0, "1.0 - equally likely\n       next to both",
             fontsize=9.5, color="0.4", va="center")
    axB.set_yscale("log")
    axB.set_ylim(4e-2, 40)
    axB.set_ylabel("one divided\nby the other", fontsize=10.5)
    axB.set_xticks(list(x)); axB.set_xticklabels(ks, fontsize=12.5)
    axB.set_xlim(-0.6, 4.9)
    axB.set_title("but dividing them does: green is a word that "
                  "belongs to one side", fontsize=11.5, pad=9)
    axB.spines[["top", "right"]].set_visible(False)

    fig.subplots_adjust(left=0.14, right=0.99, top=0.92, bottom=0.09)
    fig.savefig("ice_steam.svg", transparent=True)
    plt.close(fig)
    print("wrote ice_steam.svg")



# =============================================================================
# 장 개요 — 네 길이 어떻게 놓이는가
# =============================================================================
def fig_chapter_map():
    """사다리가 아니라는 것을 배치로 말한다.

    위 두 상자는 **나란히** 두고 등호로 잇는다 (같은 행렬, 두 길).
    GloVe 는 그 사이 아래. fastText 는 세로 점선 **바깥**에 두어 다른 축임을
    보인다 — 아래로 쌓으면 독자가 사다리로 읽는다.
    """
    fig, ax = plt.subplots(figsize=(9.2, 4.7))
    ax.set_xlim(-3, 103); ax.set_ylim(-3, 59); ax.axis("off")

    INK, ACC, GRAY = "#1f2937", "#b91c1c", "#6b7280"

    def box(x, y, w, h, title, sub, ec=INK, lw=1.6, ls="solid"):
        ax.add_patch(Rectangle((x, y), w, h, fill=False, ec=ec, lw=lw,
                               ls=ls, zorder=2))
        ax.text(x + w / 2, y + h * 0.63, title, ha="center", va="center",
                fontsize=10.5, color=ec, zorder=3, linespacing=1.35)
        ax.text(x + w / 2, y + h * 0.25, sub, ha="center", va="center",
                fontsize=8.6, color=GRAY, zorder=3)

    # --- 같은 목표에 닿는 두 길 — 나란히 ---
    box(1, 32, 25, 15, "Skip-gram +\nnegative sampling", "learn   (6.1)")
    box(35, 32, 25, 15, "PPMI + SVD", "count   (6.1)")
    for dy in (1.3, -1.3):
        ax.plot([27.5, 33.5], [39.5 + dy, 39.5 + dy], color=ACC, lw=2.0, zorder=2)
    ax.text(30.5, 49.6, "same matrix:  PMI - log k", ha="center", va="bottom",
            fontsize=10, color=ACC)
    ax.text(13.5, 30.6, "implicit", ha="center", va="top", fontsize=8.6, color=GRAY)
    ax.text(47.5, 30.6, "explicit", ha="center", va="top", fontsize=8.6, color=GRAY)

    # --- 그 사이 ---
    box(16, 13, 29, 12, "GloVe", "count first, then fit   (6.2)")
    ax.annotate("", xy=(25, 25.4), xytext=(13.5, 28.8),
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.3))
    ax.annotate("", xy=(36, 25.4), xytext=(47.5, 28.8),
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.3))

    # --- 점선 바깥이 다른 축 ---
    ax.plot([65, 65], [11, 53.5], color=GRAY, lw=1.1, ls=(0, (4, 3)), zorder=1)
    box(70, 27, 29, 12, "fastText", "same objective, smaller unit   (6.3)",
        ec=GRAY, lw=1.3, ls=(0, (4, 2)))
    ax.text(84.5, 41.2, "different axis", ha="center", va="bottom",
            fontsize=9.5, color=GRAY)
    ax.text(84.5, 23.6, "word  ->  character n-grams", ha="center", va="top",
            fontsize=8.6, color=GRAY)

    # --- 넷 모두가 가진 한계 — 6.4 의 성질이 아니라 이 갈래 전체의 성질이다 ---
    ax.add_patch(Rectangle((-1.5, 9.5), 103, 45.5, fill=False, ec=GRAY,
                           lw=0.9, ls=(0, (2, 3)), zorder=0))
    ax.annotate("", xy=(50, 4.2), xytext=(50, 9.0),
                arrowprops=dict(arrowstyle="-|>", color=ACC, lw=1.5))
    ax.text(48.5, 3.0, "all four:  one vector per word, fixed after training"
                       "   ->   ch.8   (6.4)",
            fontsize=9.3, color=ACC, ha="center", va="top")

    fig.tight_layout()
    fig.savefig("chapter_map.svg", transparent=True)
    plt.close(fig); print("wrote chapter_map.svg")

if __name__ == "__main__":
    toks = load_tokens()
    counts = Counter(toks)
    words = [w for w, _ in counts.most_common(VOCAB)]
    stoi = {w: i for i, w in enumerate(words)}
    ids = torch.tensor([stoi[w] for w in toks if w in stoi], dtype=torch.long)

    W = train(ids, counts, words)
    base, _ = pca(W, DIM)
    ks = [2, 3, 8, 32, 64]
    keeps, evs = [], []
    for k in ks:
        P, ev = pca(W, k)
        keeps.append(keep_rate(base, P)); evs.append(ev)
        if k == 2:
            P2, ev2 = P, ev

    fig_pca_2d(P2, ev2, stoi)
    fig_keep_rate(ks, keeps, evs)

    fig_lookup_is_matmul()
    fig_cost_of_onehot()
    fig_ice_steam()

    fig_chapter_map()
