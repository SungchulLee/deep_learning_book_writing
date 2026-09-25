"""여덟 갈래 IMDB에서 3걸음과 5걸음의 30에포크 곡선.

5에포크에서 어텐션이 평균을 +0.80 이겼다. 범위가 겹치지 않으니 참인
차이지만, **5에포크가 두 걸음에게 공평한 자리인가**는 다른 물음이다.

이진 일감에서는 공평하지 않았다. 3걸음은 9에포크에서 꼭대기를 찍고
내려왔고 2걸음은 30에포크까지 계속 올랐다. 그래서 5에포크는 3걸음
편이었다. 같은 일이 여기서도 일어난다면 +0.80은 실제보다 작거나 큰
값이다.

에포크마다 시험 정확도와 MAE를 재어 곡선을 그린다. 씨앗 하나를
인자로 받아 한 벌만 돌리므로, 씨앗마다 따로 띄워 나란히 돌릴 수 있다
(스레드가 하나씩이라 더하는 차례는 바뀌지 않는다).

    python imdb_stars_curve.py 0    # 씨앗 0
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
EPOCHS, BATCH, LR, PAD = 30, 100, 1e-3, 0      # 5가 아니라 30이다
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
    p = torch.cat([m(Xte[i:i + 500]).argmax(1) for i in range(0, len(Xte), 500)]).numpy()
    acc = 100.0 * (p == yte_np).mean()
    mae = float(np.abs(STAR_ARR[p] - STAR_ARR[yte_np]).mean())
    return float(acc), mae


def curve(Model, seed):
    """에포크마다 재어 곡선으로 돌려준다."""
    torch.manual_seed(seed)
    m = Model()
    opt = optim.Adam(m.parameters(), lr=LR)
    crit = nn.CrossEntropyLoss()
    g = torch.Generator().manual_seed(seed)
    ld = DataLoader(TensorDataset(Xtr, ytr), batch_size=BATCH, shuffle=True, generator=g)
    accs, maes = [], []
    for ep in range(EPOCHS):
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
    out = SP / f"curve_seed{seed}.json"
    res = json.load(open(out)) if out.exists() else {}
    for name, Model in (("3 평균", MeanEmbedding), ("5 어텐션", AttentionClassifier)):
        if name in res:
            print(f"  씨앗 {seed} {name} 이미 있음", flush=True); continue
        t0 = time.time()
        accs, maes = curve(Model, seed)
        res[name] = {"acc": accs, "mae": maes}
        json.dump(res, open(out, "w"), indent=1)
        best = int(np.argmax(accs))
        print(f"  씨앗 {seed} {name}  5에포크 {accs[4]:.2f}%  30에포크 {accs[-1]:.2f}%  "
              f"꼭대기 {accs[best]:.2f}% ({best+1}에포크)  ({time.time()-t0:.0f}초)", flush=True)
