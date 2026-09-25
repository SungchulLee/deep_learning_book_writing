"""별점을 맞히는 IMDB — 같은 평 2만 5천 편, 어려운 일감.

6장의 사다리는 뒤집혀 있다(3걸음이 꼭대기). 까닭 후보가 둘이었다.

  (가) 평 2만 5천 편에 견주어 128만 개짜리 임베딩이 너무 크다
  (나) 좋다/나쁘다를 가르는 일이 낱말 세기로 거의 풀려서 엮을 것이 없다

Yelp를 키워 보니 사다리가 바로 섰다 — (가)의 증거다. 그런데 Yelp는
자료도 바뀌고 크기도 바뀌었다. **크기를 그대로 두고 일감만 어렵게** 하면
(나)를 따로 잴 수 있다.

aclImdb의 파일 이름에 별점이 들어 있다(`id_rating.txt`). 그래서 같은
2만 5천 편으로 **여덟 갈래** 일감을 만들 수 있다. 5★과 6★은 원래
라벨에서 빠져 있으므로 1~4★과 7~10★, 모두 여덟이다.

  우연은 12.5%지만 갈래가 고르지 않다. 가장 많은 1★이 20.1%이므로
  **넘어야 할 바닥은 20.1%**다.

  뒤집힘이 풀리면 -> 원인은 (나), 일감이 쉬웠던 것
  그대로 뒤집히면 -> 원인은 (가), 자료가 작았던 것 (Yelp와 같은 결론)

규약은 6장 그대로다. 내놓는 갈래 수만 8이다.
"""

import json
import os
import re
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
ROOT = SP / "data" / "aclImdb"          # 없으면 aclImdb 경로로 바꾼다

VOCAB_SIZE, MAX_LEN, EMB_DIM, HIDDEN, HEADS = 20000, 400, 64, 64, 4
FF_DIM = 4 * EMB_DIM
EPOCHS, BATCH, LR, PAD = 5, 100, 1e-3, 0
SEEDS = range(5)
RESULTS = "imdb_stars_results.json"
STARS = [1, 2, 3, 4, 7, 8, 9, 10]       # 5★, 6★은 라벨이 없다
N_CLASS = len(STARS)
S2I = {s: i for i, s in enumerate(STARS)}

torch.set_num_threads(1)                # 6장과 같은 규약
device = torch.device("cpu")            # 6장이 CPU였으므로 맞춘다

TOKEN = re.compile(r"[a-z']+")


def tokenize(t):
    return TOKEN.findall(t.replace("<br />", " ").lower())


def load(split):
    """파일 이름 'id_rating.txt'에서 별점을 읽는다."""
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

maj = np.bincount(yte_np).max() / len(yte_np) * 100
print(f"자료 학습 {len(train_docs):,}편  시험 {len(test_docs):,}편  갈래 {N_CLASS}개")
print(f"  우연 {100/N_CLASS:.2f}%   가장 많은 갈래 {maj:.2f}%  <- 넘어야 할 바닥")
print(f"  갈래별 시험 편수 " +
      "  ".join(f"{STARS[i]}★:{c:,}" for i, c in enumerate(np.bincount(yte_np))), flush=True)


def to_counts(docs):
    X = np.zeros((len(docs), VOCAB_SIZE), dtype=np.float32)
    for r, d in enumerate(docs):
        for w in d:
            j = index.get(w)
            if j is not None:
                X[r, j] += 1
    n = X.sum(1, keepdims=True)
    return X / np.maximum(n, 1)


def to_ids(docs):
    X = np.zeros((len(docs), MAX_LEN), dtype=np.int64)
    for r, d in enumerate(docs):
        ids = [index[w] for w in d if w in index][:MAX_LEN]
        X[r, :len(ids)] = ids
    return torch.from_numpy(X)


# === 1걸음: 최근접 중심 (학습이 없다) ======================================
def rung1(Ctr, Cte):
    cent = np.vstack([Ctr[ytr_np == c].mean(0) for c in range(N_CLASS)])
    # ||x-c||^2 = ||x||^2 - 2x·c + ||c||^2 로 편다. 그냥 빼면
    # (25000, 8, 20000)짜리 배열이 생겨 16GB를 먹는다.
    d = (-2.0 * (Cte @ cent.T)) + (cent ** 2).sum(1)[None, :]
    return 100.0 * (d.argmin(1) == yte_np).mean()      # ||x||^2은 갈래마다 같아 빼도 된다


# === 2걸음: 낱말 세기 위의 선형 ============================================
def rung2(Ctr, Cte, seed):
    torch.manual_seed(seed)
    m = nn.Linear(VOCAB_SIZE, N_CLASS)
    opt = optim.Adam(m.parameters(), lr=LR)
    crit = nn.CrossEntropyLoss()
    g = torch.Generator().manual_seed(seed)
    Xt, yt = torch.from_numpy(Ctr), torch.from_numpy(ytr_np).long()
    ld = DataLoader(TensorDataset(Xt, yt), batch_size=BATCH, shuffle=True, generator=g)
    for _ in range(EPOCHS):
        m.train()
        for xb, yb in ld:
            opt.zero_grad(); crit(m(xb), yb).backward(); opt.step()
    m.eval()
    with torch.no_grad():
        p = m(torch.from_numpy(Cte)).argmax(1).numpy()
    return 100.0 * (p == yte_np).mean(), p


# === 3~6걸음: 6장 그대로, 내놓는 갈래만 8 ==================================
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


class LSTMClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(VOCAB_SIZE, EMB_DIM, padding_idx=PAD)
        self.lstm = nn.LSTM(EMB_DIM, HIDDEN, batch_first=True)
        self.fc = nn.Linear(HIDDEN, N_CLASS)

    def forward(self, x):
        e = self.emb(x)
        lengths = (x != PAD).sum(1).clamp(min=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(e, lengths, batch_first=True,
                                                   enforce_sorted=False)
        _, (h, _) = self.lstm(packed)
        return self.fc(h[-1])


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


class TransformerClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(VOCAB_SIZE, EMB_DIM, padding_idx=PAD)
        self.pos = nn.Parameter(torch.randn(1, MAX_LEN, EMB_DIM) * 0.02)
        self.attn = nn.MultiheadAttention(EMB_DIM, HEADS, batch_first=True)
        self.n1 = nn.LayerNorm(EMB_DIM)
        self.ff = nn.Sequential(nn.Linear(EMB_DIM, FF_DIM), nn.ReLU(),
                                nn.Linear(FF_DIM, EMB_DIM))
        self.n2 = nn.LayerNorm(EMB_DIM)
        self.fc = nn.Linear(EMB_DIM, N_CLASS)

    def forward(self, x):
        h = self.emb(x) + self.pos
        pad = (x == PAD)
        a, _ = self.attn(h, h, h, key_padding_mask=safe_mask(pad))
        h = self.n1(h + a)
        h = self.n2(h + self.ff(h))
        m = (~pad).unsqueeze(-1).float()
        return self.fc((h * m).sum(1) / m.sum(1).clamp(min=1))


DEEP = [("3 평균", MeanEmbedding), ("4 LSTM", LSTMClassifier),
        ("5 어텐션", AttentionClassifier), ("6 트랜스포머", TransformerClassifier)]


def train_deep(Model, seed, Xtr, ytr, Xte):
    torch.manual_seed(seed)
    m = Model()
    opt = optim.Adam(m.parameters(), lr=LR)
    crit = nn.CrossEntropyLoss()
    g = torch.Generator().manual_seed(seed)
    ld = DataLoader(TensorDataset(Xtr, ytr), batch_size=BATCH, shuffle=True, generator=g)
    for _ in range(EPOCHS):
        m.train()
        for xb, yb in ld:
            loss = crit(m(xb), yb)
            if not torch.isfinite(loss):
                raise RuntimeError("손실이 NaN이다")
            opt.zero_grad(); loss.backward(); opt.step()
    m.eval()
    with torch.no_grad():
        p = torch.cat([m(Xte[i:i + 500]).argmax(1) for i in range(0, len(Xte), 500)]).numpy()
    return 100.0 * (p == yte_np).mean(), p


def mae_stars(pred):
    """별점 사이 거리로 잰 평균 오차. 9★을 10★으로 본 것과 1★으로 본 것은 다르다."""
    s = np.array(STARS)
    return float(np.abs(s[pred] - s[yte_np]).mean())


def load_done():
    return json.load(open(RESULTS)) if os.path.exists(RESULTS) else {}


def save(d):
    json.dump(d, open(RESULTS + ".tmp", "w"), indent=1); os.replace(RESULTS + ".tmp", RESULTS)


if __name__ == "__main__":
    done = load_done()

    print("\n=== 1·2걸음: 낱말 세기 ===", flush=True)
    Ctr, Cte = to_counts(train_docs), to_counts(test_docs)
    if "1 최근접중심" not in done:
        t0 = time.time(); done["1 최근접중심"] = float(rung1(Ctr, Cte)); save(done)
        print(f"  1 최근접중심  {done['1 최근접중심']:.2f}%  ({time.time()-t0:.0f}초)", flush=True)
    for s in SEEDS:
        k = f"2 선형|{s}"
        if k in done: continue
        t0 = time.time(); acc, p = rung2(Ctr, Cte, s)
        done[k] = float(acc); done[f"{k}|mae"] = mae_stars(p); save(done)
        print(f"  2 선형  씨앗 {s}  {acc:.2f}%  MAE {done[f'{k}|mae']:.2f}★  "
              f"({time.time()-t0:.0f}초)", flush=True)
    del Ctr, Cte

    print("\n=== 3~6걸음 ===", flush=True)
    Xtr, Xte = to_ids(train_docs), to_ids(test_docs)
    ytr_t = torch.from_numpy(ytr_np).long()
    for name, Model in DEEP:
        for s in SEEDS:
            k = f"{name}|{s}"
            if k in done: continue
            t0 = time.time(); acc, p = train_deep(Model, s, Xtr, ytr_t, Xte)
            done[k] = float(acc); done[f"{k}|mae"] = mae_stars(p); save(done)
            print(f"  {name}  씨앗 {s}  {acc:.2f}%  MAE {done[f'{k}|mae']:.2f}★  "
                  f"({time.time()-t0:.0f}초)", flush=True)
        got = [done[f"{name}|{s}"] for s in SEEDS]
        print(f"  >> {name}  평균 {sum(got)/len(got):.2f}%  "
              f"퍼짐 {max(got)-min(got):.2f}\n", flush=True)

    print(f"\n{'='*64}\n=== 여덟 갈래 사다리 (바닥 {maj:.2f}%) ===\n{'='*64}")
    print(f"{'걸음':>14} | {'정확도':>8} | {'퍼짐':>6} | {'MAE(★)':>7}")
    rows = ["1 최근접중심", "2 선형"] + [r for r, _ in DEEP]
    for r in rows:
        if r == "1 최근접중심":
            print(f"{r:>14} | {done[r]:>8.2f} | {'—':>6} | {'—':>7}"); continue
        g = [done[f"{r}|{s}"] for s in SEEDS]
        m = [done[f"{r}|{s}|mae"] for s in SEEDS]
        print(f"{r:>14} | {sum(g)/len(g):>8.2f} | {max(g)-min(g):>6.2f} | {sum(m)/len(m):>7.2f}")
    print("\n3걸음이 4·5·6걸음을 여전히 이기면 원인은 자료 크기다(Yelp와 같은 답)."
          "\n뒤집힘이 풀리면 원인은 이진 일감이 쉬웠던 것이다.")
