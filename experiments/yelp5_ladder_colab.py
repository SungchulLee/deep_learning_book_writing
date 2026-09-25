"""별 다섯 갈래 Yelp로 사다리를 오른다 — 어려운 일감에서도 뒤집히는가.

두 갈래 Yelp(좋다/나쁘다)에서 사다리가 크기를 따라 바로 섰다. 그런데
두 갈래 일감은 **낱말을 세는 것만으로 거의 풀린다.** IMDB에서 매개변수
20,001개짜리 로지스틱 회귀가 85.93%를 냈고, 128만 개짜리 모델이
87.03%였다. 1.1%포인트 차이다. 그런 자리에서는 무엇을 얹어도 벌 것이 없다.

별 갯수를 맞히는 일은 다르다.

  * 우연이 20%다 (두 갈래의 50%가 아니라).
  * 4★과 5★을 가르려면 **세기**를 읽어야 한다. 낱말이 있느냐 없느냐가
    아니라 얼마나 센 말이냐다.
  * 3★이 가장 어렵다. "음식은 훌륭했지만 서비스가 끔찍했다"에는 센
    긍정과 센 부정이 함께 있다. 낱말만 세면 서로 지워진다. 어느 쪽이
    문장의 주인인지는 **엮어 보아야** 안다.

곧 차례와 짜임이 값을 할 자리가 실제로 있는 일감이다. 여기서도 사다리가
뒤집힌다면 그것은 자료 탓이 아니라 더 깊은 이야기다.

이 스크립트는 여섯 걸음을 모두 돈다. 두 갈래 쪽에서 빠뜨렸던
**1걸음과 2걸음(낱말 세기)**을 넣었다. 그것이 없으면 "깊은 모델이
번 것"을 잴 바닥이 없다.

Colab에서 쓰는 법
-----------------
    1. 런타임 > 런타임 유형 변경 > GPU
    2. 붙여넣고 실행. 끊기면 다시 실행하면 끝난 씨앗은 건너뛴다.
"""

import json
import os
import re
import time
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# ===========================================================================
# 설정
# ===========================================================================
# 별 다섯 갈래. 두 갈래짜리(yelp_polarity)와 **주인이 다르다** —
# polarity는 fancyzhx, full은 Yelp 아래에 있다. 차례로 시도한다.
CANDIDATES = ["Yelp/yelp_review_full", "yelp_review_full",
              "fancyzhx/yelp_review_full"]
VOCAB_SIZE, MAX_LEN, EMB_DIM, HIDDEN, HEADS = 20000, 400, 64, 64, 4
FF_DIM = 4 * EMB_DIM
EPOCHS, BATCH, LR, PAD = 5, 100, 1e-3, 0
SEEDS = range(5)
SIZES = [25_000, 100_000, 650_000]      # 작은 것부터
RESULTS = "yelp5_ladder_results.json"   # 드라이브에 남기려면 여기를 바꾼다

device = torch.device("cuda" if torch.cuda.is_available()
                      else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"장치 {device}", flush=True)

try:
    from datasets import load_dataset
except ImportError:
    os.system("pip install -q datasets")
    from datasets import load_dataset
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer

print("별 다섯 갈래 Yelp를 내려받는다", flush=True)
ds = None
for cand in CANDIDATES:
    try:
        ds = load_dataset(cand)
        print(f"  '{cand}' 에서 읽었다", flush=True)
        break
    except Exception as exc:
        print(f"  '{cand}' 실패 ({type(exc).__name__})", flush=True)
if ds is None:
    raise SystemExit("자료를 읽지 못했다.")
train_ds, test_ds = ds["train"], ds["test"]
train_y = np.array(train_ds["label"])
test_y = np.array(test_ds["label"])
test_txt = test_ds["text"]
N_CLASS = int(train_y.max()) + 1
CHANCE = 100.0 / N_CLASS
print(f"  학습 {len(train_ds):,}편  시험 {len(test_txt):,}편  "
      f"갈래 {N_CLASS}개  우연 {CHANCE:.2f}%", flush=True)

TOKEN = re.compile(r"[a-z']+")


def tokenize(t):
    return TOKEN.findall(t.replace("<br />", " ").lower())


def subsample(n, seed=0):
    """갈래마다 같은 수씩 고른다. 고르는 차례도 고정한다."""
    rng = np.random.RandomState(seed)
    per = n // N_CLASS
    idx = [rng.choice(np.where(train_y == c)[0], per, replace=False)
           for c in range(N_CLASS)]
    idx = np.concatenate(idx); rng.shuffle(idx)
    return idx


# ===========================================================================
# 자료 두 벌 — 1·2걸음은 낱말 세기, 3~6걸음은 낱말 번호
# ===========================================================================
def build_vocab(txt):
    counts = Counter()
    for t in txt:
        counts.update(tokenize(t))
    vocab = ["<pad>"] + [w for w, _ in counts.most_common(VOCAB_SIZE - 1)]
    return {w: i for i, w in enumerate(vocab)}


def build_counts(txt, index):
    """(문서 수, VOCAB_SIZE) 성긴 행렬. 65만 편을 빽빽하게 두면 52GB다."""
    vec = CountVectorizer(vocabulary=index, analyzer=tokenize)
    X = vec.transform(txt).astype(np.float32).tocsr()
    n = np.asarray(X.sum(1)).ravel()          # 줄마다 길이로 나눈다
    n[n == 0] = 1.0
    X.data /= np.repeat(n, np.diff(X.indptr))  # 대각 행렬 곱 대신 제자리에서
    return X


def build_ids(txt, index):
    X = np.zeros((len(txt), MAX_LEN), dtype=np.int64)
    for r, t in enumerate(txt):
        ids = [index[w] for w in tokenize(t) if w in index][:MAX_LEN]
        X[r, :len(ids)] = ids
    return torch.from_numpy(X)


# ===========================================================================
# 1걸음 — 최근접 중심 (학습이 없으므로 씨앗도 없다)
# ===========================================================================
def rung1(Xtr, ytr, Xte, yte):
    cent = np.vstack([np.asarray(Xtr[ytr == c].mean(0)).ravel()
                      for c in range(N_CLASS)]).astype(np.float32)
    # ||x-c||^2 = ||x||^2 - 2x·c + ||c||^2. ||x||^2은 갈래마다 같아 빼도 된다.
    # 그냥 빼면 (문서, 갈래, 낱말)짜리 배열이 생겨 기가바이트를 먹는다.
    d = (-2.0) * (Xte @ cent.T) + (cent ** 2).sum(1)[None, :]
    return 100.0 * (np.asarray(d).argmin(1) == yte).mean()


# ===========================================================================
# 2걸음 — 낱말 세기 위의 선형 학습 (성긴 행렬을 배치마다 푼다)
# ===========================================================================
def rung2(Xtr, ytr, Xte, yte, seed):
    torch.manual_seed(seed)
    model = nn.Linear(VOCAB_SIZE, N_CLASS).to(device)
    opt = optim.Adam(model.parameters(), lr=LR)
    crit = nn.CrossEntropyLoss()
    g = torch.Generator().manual_seed(seed)
    yt = torch.from_numpy(ytr).long()
    for _ in range(EPOCHS):
        model.train()
        perm = torch.randperm(Xtr.shape[0], generator=g).numpy()
        for i in range(0, len(perm), BATCH):
            j = perm[i:i + BATCH]
            xb = torch.from_numpy(Xtr[j].toarray()).to(device)
            opt.zero_grad()
            crit(model(xb), yt[j].to(device)).backward()
            opt.step()
    model.eval(); ok = 0
    with torch.no_grad():
        for i in range(0, Xte.shape[0], 2000):
            xb = torch.from_numpy(Xte[i:i + 2000].toarray()).to(device)
            ok += (model(xb).argmax(1).cpu().numpy() == yte[i:i + 2000]).sum()
    return 100.0 * ok / Xte.shape[0]


# ===========================================================================
# 3~6걸음 — 6장 그대로, 내놓는 갈래 수만 N_CLASS
# ===========================================================================
def safe_mask(pad):
    """줄이 통째로 비면 softmax가 NaN을 낸다. 한 자리를 열어 둔다."""
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


@torch.no_grad()
def accuracy(m, Xte, yte):
    m.eval(); ok = 0
    for i in range(0, len(Xte), 500):
        ok += (m(Xte[i:i + 500].to(device)).argmax(1).cpu() == yte[i:i + 500]).sum().item()
    return 100.0 * ok / len(Xte)


def train_deep(Model, seed, Xtr, ytr, Xte, yte):
    torch.manual_seed(seed)
    m = Model().to(device)
    opt = optim.Adam(m.parameters(), lr=LR)
    crit = nn.CrossEntropyLoss()
    g = torch.Generator().manual_seed(seed)
    ld = DataLoader(TensorDataset(Xtr, ytr), batch_size=BATCH, shuffle=True, generator=g)
    for _ in range(EPOCHS):
        m.train()
        for xb, yb in ld:
            loss = crit(m(xb.to(device)), yb.to(device))
            if not torch.isfinite(loss):
                raise RuntimeError(f"손실이 NaN이다 — 조용히 {CHANCE:.2f}%로 굳기 전에 멈춘다")
            opt.zero_grad(); loss.backward(); opt.step()
    return accuracy(m, Xte, yte)


def load_done():
    return json.load(open(RESULTS)) if os.path.exists(RESULTS) else {}


def save(d):
    tmp = RESULTS + ".tmp"
    json.dump(d, open(tmp, "w"), indent=1)
    os.replace(tmp, RESULTS)


# ===========================================================================
if __name__ == "__main__":
    done = load_done()
    for n in SIZES:
        want = [f"{n}|1 최근접중심|0"] + \
               [f"{n}|2 선형|{s}" for s in SEEDS] + \
               [f"{n}|{r}|{s}" for r, _ in DEEP for s in SEEDS]
        if all(k in done for k in want):
            print(f"\n=== {n:,}편 — 이미 끝났다 ===", flush=True); continue

        print(f"\n{'='*58}\n=== 학습 {n:,}편  (우연 {CHANCE:.2f}%) ===\n{'='*58}", flush=True)
        idx = subsample(n)
        txt = train_ds.select(idx.tolist())["text"]
        ytr = train_y[idx]
        index = build_vocab(txt)

        # --- 1·2걸음: 낱말 세기 -------------------------------------------
        t0 = time.time()
        Ctr, Cte = build_counts(txt, index), build_counts(test_txt, index)
        print(f"  세기 행렬 {Ctr.shape} 성김 {Ctr.nnz/np.prod(Ctr.shape):.4f} "
              f"({time.time()-t0:.0f}초)", flush=True)

        k = f"{n}|1 최근접중심|0"
        if k not in done:
            t0 = time.time(); done[k] = float(rung1(Ctr, ytr, Cte, test_y)); save(done)
            print(f"    1 최근접중심  {done[k]:.2f}%  (씨앗 없음, {time.time()-t0:.0f}초)", flush=True)
        for s in SEEDS:
            k = f"{n}|2 선형|{s}"
            if k in done: continue
            t0 = time.time(); done[k] = float(rung2(Ctr, ytr, Cte, test_y, s)); save(done)
            print(f"    2 선형  씨앗 {s}  {done[k]:.2f}%  ({time.time()-t0:.0f}초)", flush=True)
        got = [done[f"{n}|2 선형|{s}"] for s in SEEDS]
        print(f"  >> 2 선형  평균 {sum(got)/len(got):.2f}%  퍼짐 {max(got)-min(got):.2f}\n", flush=True)
        del Ctr, Cte

        # --- 3~6걸음: 낱말 번호 -------------------------------------------
        Xtr, Xte = build_ids(txt, index), build_ids(test_txt, index)
        ytr_t, yte_t = torch.from_numpy(ytr).long(), torch.from_numpy(test_y).long()
        print(f"  빈 줄  학습 {int((Xtr==PAD).all(1).sum())}  "
              f"시험 {int((Xte==PAD).all(1).sum())}", flush=True)
        for name, Model in DEEP:
            for s in SEEDS:
                k = f"{n}|{name}|{s}"
                if k in done: continue
                t0 = time.time()
                done[k] = float(train_deep(Model, s, Xtr, ytr_t, Xte, yte_t)); save(done)
                print(f"    {name}  씨앗 {s}  {done[k]:.2f}%  ({time.time()-t0:.0f}초)", flush=True)
            got = [done[f"{n}|{name}|{s}"] for s in SEEDS]
            print(f"  >> {name}  평균 {sum(got)/len(got):.2f}%  "
                  f"퍼짐 {max(got)-min(got):.2f}\n", flush=True)
        del Xtr, Xte

    # === 정리 ==============================================================
    print(f"\n{'='*70}\n=== 별 {N_CLASS}갈래 사다리 (우연 {CHANCE:.2f}%) ===\n{'='*70}")
    rows = ["1 최근접중심", "2 선형"] + [r for r, _ in DEEP]
    print(f"{'크기':>9} | " + " ".join(f"{r:>12}" for r in rows) + " |  깊은 것이 번 몫")
    for n in SIZES:
        try:
            avg = {}
            avg["1 최근접중심"] = done[f"{n}|1 최근접중심|0"]
            for r in rows[1:]:
                avg[r] = sum(done[f"{n}|{r}|{s}"] for s in SEEDS) / len(list(SEEDS))
        except KeyError:
            continue
        gain = max(avg[r] for r in rows[2:]) - avg["2 선형"]
        print(f"{n:>9,} | " + " ".join(f"{avg[r]:>12.2f}" for r in rows) + f" | {gain:>+8.2f}")
    print("\n마지막 칸이 요점이다. 낱말 세기(2걸음) 위로 깊은 모델이 얼마나 버는가."
          "\n두 갈래 IMDB에서는 이 값이 1.1%포인트였다 — 그래서 아무것도 가릴 수 없었다.")
