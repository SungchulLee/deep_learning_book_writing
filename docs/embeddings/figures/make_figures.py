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
