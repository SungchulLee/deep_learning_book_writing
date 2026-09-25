"""어텐션이 왜 지는가 — 지나친 유연함인가, 자리 임베딩인가.

여덟 갈래 IMDB에서 어텐션은 4에포크에 41.32%로 꼭대기를 찍고 30에포크에
32.62%까지 무너진다. 평균은 13에포크에 42.78%까지 오른 뒤 39.89%로
천천히 내려온다. 어텐션은 **더 빨리 과적합하고, 가장 좋은 자리도 낮다.**

"어텐션이 이 자료에 견주어 너무 유연하다"는 읽기가 자연스럽다. 그런데
어텐션은 평균보다 매개변수가 **3.3%** 많을 뿐이고(1,322,888 대 1,280,520),
임베딩을 얼렸을 때는 어텐션이 오히려 이겼다. 그러니 "유연함"이 어디에
있는지를 짚어야 한다.

후보 둘을 잰다.

  1. 규제를 걸면 나아지는가 (드롭아웃 0.3)
     -> 나아지면 남아도는 유연함이 맞다.

  2. 자리 임베딩을 빼면 나아지는가
     -> self.pos는 400x64 = 25,600개로, 어텐션이 평균 위에 더하는
        42,368개의 60%다. 게다가 400칸을 다 채우는 평이 드물어
        뒤쪽 자리 벡터는 몇 편 안 되는 자료로 정해진다. 곧 **애초에
        제대로 정해질 수 없는** 유연함이다. 빼서 좋아지면 이쪽이 범인이다.

기준선은 이미 있다(어텐션 꼭대기 41.32, 평균 꼭대기 42.78).

    python imdb_stars_regularize.py 0     # 씨앗 0, 두 변종 모두
"""

import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

SP = Path("/private/tmp/claude-501/-Users-sungchul-Documents-book-deep-learning-in-korean"
          "/1eb6679d-9c60-43ad-992c-e397ba670b3b/scratchpad")
ROOT = SP / "data" / "aclImdb"

VOCAB_SIZE, MAX_LEN, EMB_DIM, HEADS = 20000, 400, 64, 4
EPOCHS, BATCH, LR, PAD = 20, 100, 1e-3, 0      # 어텐션 꼭대기가 4에포크라 20이면 넉넉하다
DROP = 0.3
STARS = [1, 2, 3, 4, 7, 8, 9, 10]
N_CLASS = len(STARS)
S2I = {s: i for i, s in enumerate(STARS)}
torch.set_num_threads(1)

TOKEN = re.compile(r"[a-z']+")


def tokenize(t):
    return TOKEN.findall(t.replace("<br />", " ").lower())


def load(split):
    docs, stars = [], []
    for pol in ("pos", "neg"):
        for f in sorted((ROOT / split / pol).glob("*.txt")):
            docs.append(tokenize(f.read_text(encoding="utf-8")))
            stars.append(int(f.stem.split("_")[1]))
    return docs, np.array([S2I[s] for s in stars])


train_docs, ytr_np = load("train")
test_docs, yte_np = load("test")
counts = Counter(w for d in train_docs for w in d)
vocab = ["<pad>"] + [w for w, _ in counts.most_common(VOCAB_SIZE - 1)]
index = {w: i for i, w in enumerate(vocab)}


def to_ids(docs):
    X = np.zeros((len(docs), MAX_LEN), dtype=np.int64)
    for r, d in enumerate(docs):
        ids = [index[w] for w in d if w in index][:MAX_LEN]
        X[r, :len(ids)] = ids
    return torch.from_numpy(X)


Xtr, Xte = to_ids(train_docs), to_ids(test_docs)
ytr = torch.from_numpy(ytr_np).long()
STAR_ARR = np.array(STARS)


def safe_mask(pad):
    m = pad.clone(); m[:, 0] = False
    return m


class Attention(nn.Module):
    """5걸음. use_pos와 drop만 바꾸어 세 벌을 만든다."""

    def __init__(self, use_pos=True, drop=0.0):
        super().__init__()
        self.emb = nn.Embedding(VOCAB_SIZE, EMB_DIM, padding_idx=PAD)
        self.use_pos = use_pos
        if use_pos:
            self.pos = nn.Parameter(torch.randn(1, MAX_LEN, EMB_DIM) * 0.02)
        self.attn = nn.MultiheadAttention(EMB_DIM, HEADS, batch_first=True)
        self.norm = nn.LayerNorm(EMB_DIM)
        self.drop = nn.Dropout(drop)
        self.fc = nn.Linear(EMB_DIM, N_CLASS)

    def forward(self, x):
        e = self.emb(x)
        if self.use_pos:
            e = e + self.pos
        pad = (x == PAD)
        a, _ = self.attn(e, e, e, key_padding_mask=safe_mask(pad))
        h = self.norm(e + self.drop(a))
        m = (~pad).unsqueeze(-1).float()
        pooled = (h * m).sum(1) / m.sum(1).clamp(min=1)
        return self.fc(self.drop(pooled))


VARIANTS = [
    ("5 어텐션 + 드롭아웃", lambda: Attention(use_pos=True, drop=DROP)),
    ("5 어텐션 − 자리임베딩", lambda: Attention(use_pos=False, drop=0.0)),
]


@torch.no_grad()
def evaluate(m):
    m.eval()
    p = torch.cat([m(Xte[i:i + 500]).argmax(1) for i in range(0, len(Xte), 500)]).numpy()
    return float(100.0 * (p == yte_np).mean()), float(np.abs(STAR_ARR[p] - STAR_ARR[yte_np]).mean())


def curve(make, seed):
    torch.manual_seed(seed)
    m = make()
    opt = optim.Adam(m.parameters(), lr=LR)
    crit = nn.CrossEntropyLoss()
    g = torch.Generator().manual_seed(seed)
    ld = DataLoader(TensorDataset(Xtr, ytr), batch_size=BATCH, shuffle=True, generator=g)
    accs, maes = [], []
    for _ in range(EPOCHS):
        m.train()
        for xb, yb in ld:
            loss = crit(m(xb), yb)
            if not torch.isfinite(loss):
                raise RuntimeError("손실이 NaN이다")
            opt.zero_grad(); loss.backward(); opt.step()
        a, e = evaluate(m)
        accs.append(a); maes.append(e)
    return accs, maes


if __name__ == "__main__":
    seed = int(sys.argv[1])
    out = SP / f"reg_seed{seed}.json"
    res = json.load(open(out)) if out.exists() else {}
    for name, make in VARIANTS:
        if name in res:
            print(f"  씨앗 {seed} {name} 이미 있음", flush=True); continue
        n_par = sum(p.numel() for p in make().parameters())
        t0 = time.time()
        accs, maes = curve(make, seed)
        res[name] = {"acc": accs, "mae": maes, "params": n_par}
        json.dump(res, open(out, "w"), indent=1)
        b = int(np.argmax(accs))
        print(f"  씨앗 {seed} {name}  매개변수 {n_par:,}  "
              f"꼭대기 {accs[b]:.2f}% ({b+1}에포크)  끝 {accs[-1]:.2f}%  "
              f"({time.time()-t0:.0f}초)", flush=True)
