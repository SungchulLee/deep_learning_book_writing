"""별 다섯 갈래 Yelp 2만 5천 편에서 3걸음과 5걸음의 30에포크 곡선.

Colab에서 5에포크로 재니 어텐션이 평균을 **+1.96** 이겼다. 그런데 같은
크기의 여덟 갈래 IMDB에서 30에포크 곡선을 그려 보니, 어텐션은 4에포크에서
꼭대기를 찍고 내려오고 평균은 12~14에포크까지 올랐다. 5에포크는 어텐션
편이었고, 저마다 가장 좋은 자리끼리 견주면 부호가 **뒤집혔다**(+0.80 →
−1.46).

같은 일이 여기서도 일어나는지 본다. +1.96이 참이면 어려운 일감에서는
작은 자료로도 차례가 값을 한다는 뜻이고, 규약 탓이면 IMDB와 같은
이야기가 된다.

자료는 파케이를 곧장 읽는다(datasets 꾸러미가 없어도 된다). 고르는
차례는 Colab 스크립트와 글자 하나 다르지 않아 같은 2만 5천 편이 나온다.

    python yelp5_curve_local.py 0     # 씨앗 0 한 벌
"""

import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

SP = Path("/private/tmp/claude-501/-Users-sungchul-Documents-book-deep-learning-in-korean"
          "/1eb6679d-9c60-43ad-992c-e397ba670b3b/scratchpad")
DATA = SP / "yelp5"

VOCAB_SIZE, MAX_LEN, EMB_DIM, HEADS = 20000, 400, 64, 4
EPOCHS, BATCH, LR, PAD = 30, 100, 1e-3, 0       # 5가 아니라 30
N_TRAIN, N_CLASS = 25_000, 5
torch.set_num_threads(1)

TOKEN = re.compile(r"[a-z']+")


def tokenize(t):
    return TOKEN.findall(t.replace("<br />", " ").lower())


tr = pq.read_table(DATA / "train-00000-of-00001.parquet")
te = pq.read_table(DATA / "test-00000-of-00001.parquet")
train_y_all = np.array(tr.column("label"))
test_txt = [str(x) for x in te.column("text")]
yte_np = np.array(te.column("label"))


def subsample(n, seed=0):
    """Colab 스크립트와 같은 차례 — 같은 부분집합이 나와야 한다."""
    rng = np.random.RandomState(seed)
    per = n // N_CLASS
    idx = [rng.choice(np.where(train_y_all == c)[0], per, replace=False)
           for c in range(N_CLASS)]
    idx = np.concatenate(idx); rng.shuffle(idx)
    return idx


sub = subsample(N_TRAIN)
train_txt = [str(tr.column("text")[int(i)]) for i in sub]
ytr_np = train_y_all[sub]

counts = Counter()
for t in train_txt:
    counts.update(tokenize(t))
vocab = ["<pad>"] + [w for w, _ in counts.most_common(VOCAB_SIZE - 1)]
index = {w: i for i, w in enumerate(vocab)}


def to_ids(texts):
    X = np.zeros((len(texts), MAX_LEN), dtype=np.int64)
    for r, t in enumerate(texts):
        ids = [index[w] for w in tokenize(t) if w in index][:MAX_LEN]
        X[r, :len(ids)] = ids
    return torch.from_numpy(X)


Xtr, Xte = to_ids(train_txt), to_ids(test_txt)
ytr = torch.from_numpy(ytr_np).long()
STAR = np.array([1, 2, 3, 4, 5])


def safe_mask(pad):
    m = pad.clone(); m[:, 0] = False
    return m


class MeanEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(VOCAB_SIZE, EMB_DIM, padding_idx=PAD)
        self.fc = nn.Linear(EMB_DIM, N_CLASS)

    def forward(self, x):
        e = self.emb(x)
        m = (x != PAD).unsqueeze(-1).float()
        return self.fc((e * m).sum(1) / m.sum(1).clamp(min=1))


class AttentionClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(VOCAB_SIZE, EMB_DIM, padding_idx=PAD)
        self.pos = nn.Parameter(torch.randn(1, MAX_LEN, EMB_DIM) * 0.02)
        self.attn = nn.MultiheadAttention(EMB_DIM, HEADS, batch_first=True)
        self.norm = nn.LayerNorm(EMB_DIM)
        self.fc = nn.Linear(EMB_DIM, N_CLASS)

    def forward(self, x):
        e = self.emb(x) + self.pos
        pad = (x == PAD)
        a, _ = self.attn(e, e, e, key_padding_mask=safe_mask(pad))
        h = self.norm(e + a)
        m = (~pad).unsqueeze(-1).float()
        return self.fc((h * m).sum(1) / m.sum(1).clamp(min=1))


@torch.no_grad()
def evaluate(m):
    m.eval()
    p = torch.cat([m(Xte[i:i + 1000]).argmax(1) for i in range(0, len(Xte), 1000)]).numpy()
    return float(100.0 * (p == yte_np).mean()), float(np.abs(STAR[p] - STAR[yte_np]).mean())


def curve(Model, seed):
    torch.manual_seed(seed)
    m = Model()
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
    out = SP / f"yelp5_curve_seed{seed}.json"
    res = json.load(open(out)) if out.exists() else {}
    for name, Model in (("3 평균", MeanEmbedding), ("5 어텐션", AttentionClassifier)):
        if name in res:
            print(f"  씨앗 {seed} {name} 이미 있음", flush=True); continue
        t0 = time.time()
        accs, maes = curve(Model, seed)
        res[name] = {"acc": accs, "mae": maes}
        json.dump(res, open(out, "w"), indent=1)
        b = int(np.argmax(accs))
        print(f"  씨앗 {seed} {name}  5에포크 {accs[4]:.2f}%  30에포크 {accs[-1]:.2f}%  "
              f"꼭대기 {accs[b]:.2f}% ({b+1}에포크)  ({time.time()-t0:.0f}초)", flush=True)
