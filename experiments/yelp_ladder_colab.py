"""Yelp로 사다리를 다시 오른다 — 자료가 커지면 뒤집힘이 풀리는가.

6장의 사다리는 뒤집혀 있다. 차례를 버린 3걸음(87.03%)이 차례를 읽는
4걸음(81.51%)과 5걸음(86.11%)을 이긴다. 가설 하나는 이것이다.
**IMDB 2만 5천 편이 큰 모델을 학습시키기에 너무 작다.**

그 가설이 옳다면 자료를 키울수록 뒤집힘이 풀려야 한다. 그래서 같은
사다리를 Yelp에서 2만 5천 / 10만 / 56만으로 세 번 오른다.

규약은 6장과 글자 하나 다르지 않다. 바뀌는 것은 **자료의 크기뿐**이다.
2만 5천 칸이 IMDB와 같은 크기라 "Yelp라서 다른 것"을 걸러 준다.

Colab에서 쓰는 법
-----------------
    1. 런타임 > 런타임 유형 변경 > GPU
    2. 이 파일을 셀 하나에 붙여넣고 실행
    3. 끊기면 그냥 다시 실행한다. 끝난 씨앗은 건너뛴다.

결과는 씨앗 하나가 끝날 때마다 RESULTS에 쌓인다. 구글 드라이브에
남기려면 아래 RESULTS를 드라이브 경로로 바꾼다.
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
# 설정 — 크기 말고는 6장 그대로
# ===========================================================================
VOCAB_SIZE, MAX_LEN, EMB_DIM, HIDDEN, HEADS = 20000, 400, 64, 64, 4
FF_DIM = 4 * EMB_DIM
EPOCHS, BATCH, LR, PAD = 5, 100, 1e-3, 0
SEEDS = range(5)
SIZES = [25_000, 100_000, 560_000]      # 작은 것부터 — 흐름이 일찍 보인다

RESULTS = "yelp_ladder_results.json"    # 드라이브에 남기려면 여기를 바꾼다
# 예: RESULTS = "/content/drive/MyDrive/yelp_ladder_results.json"

device = torch.device("cuda" if torch.cuda.is_available()
                      else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"장치 {device}", flush=True)

# ===========================================================================
# 자료
# ===========================================================================
try:
    from datasets import load_dataset
except ImportError:
    os.system("pip install -q datasets")
    from datasets import load_dataset

print("Yelp를 내려받는다 (처음 한 번만, 몇 분 걸린다)", flush=True)

# 예전 이름 "yelp_polarity"는 새 huggingface_hub에서 풀리지 않는다.
# 이름이 'namespace/name' 꼴이라야 하며, 이 자료는 fancyzhx로 옮겨 갔다.
CANDIDATES = ["fancyzhx/yelp_polarity", "yelp_polarity"]
ds = None
for cand in CANDIDATES:
    try:
        ds = load_dataset(cand)
        print(f"  '{cand}' 에서 읽었다", flush=True)
        break
    except Exception as exc:
        print(f"  '{cand}' 실패 ({type(exc).__name__})", flush=True)
if ds is None:
    raise SystemExit("Yelp를 읽지 못했다. datasets를 올리거나 이름을 확인하라.")

# 글월은 미리 다 꺼내지 않는다. 56만 편을 파이썬 문자열로 들고 있으면
# 2만 5천 칸을 돌 때도 그만큼을 쥐고 있게 된다. 이름표만 먼저 꺼낸다.
train_ds, test_ds = ds["train"], ds["test"]
train_y = np.array(train_ds["label"])
test_y = np.array(test_ds["label"])
test_txt = test_ds["text"]                    # 3만 8천 편, 이쪽은 작다
print(f"  학습 {len(train_ds):,}편  시험 {len(test_txt):,}편  "
      f"이름표 {sorted(set(train_y.tolist()))}", flush=True)
assert set(train_y.tolist()) == {0, 1}, "이름표가 0/1이 아니다"

TOKEN = re.compile(r"[a-z']+")


def tokenize(t):
    return TOKEN.findall(t.replace("<br />", " ").lower())


def subsample(n, seed=0):
    """갈래를 반반으로 맞추어 n편을 고른다. 고르는 차례도 고정한다."""
    rng = np.random.RandomState(seed)
    idx = []
    for lab in (0, 1):
        pool = np.where(train_y == lab)[0]
        idx.append(rng.choice(pool, n // 2, replace=False))
    idx = np.concatenate(idx)
    rng.shuffle(idx)
    return idx


def build_ids(texts, index):
    """토큰 목록을 들고 있지 않고 곧장 번호 배열로 만든다 (메모리 때문이다)."""
    X = np.zeros((len(texts), MAX_LEN), dtype=np.int64)
    for r, t in enumerate(texts):
        ids = [index[w] for w in tokenize(t) if w in index][:MAX_LEN]
        X[r, :len(ids)] = ids
    return torch.from_numpy(X)


def prepare(n):
    """크기 n짜리 학습 집합과, 그 집합에서 만든 낱말집을 돌려준다.

    낱말집을 부분집합에서 만드는 것이 옳다. 자료가 적으면 낱말집도
    나빠지는 것이 자료가 적다는 말의 일부이기 때문이다.
    """
    idx = subsample(n)
    txt = train_ds.select(idx.tolist())["text"]        # 고른 것만 꺼낸다
    counts = Counter()
    for t in txt:
        counts.update(tokenize(t))
    vocab = ["<pad>"] + [w for w, _ in counts.most_common(VOCAB_SIZE - 1)]
    index = {w: i for i, w in enumerate(vocab)}
    Xtr = build_ids(txt, index)
    ytr = torch.from_numpy(train_y[idx]).long()
    Xte = build_ids(test_txt, index)          # 시험은 언제나 전부 쓴다
    yte = torch.from_numpy(test_y).long()
    return Xtr, ytr, Xte, yte


# ===========================================================================
# 네 걸음 — 6장 그대로
# ===========================================================================
class MeanEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(VOCAB_SIZE, EMB_DIM, padding_idx=PAD)
        self.fc = nn.Linear(EMB_DIM, 2)

    def forward(self, x):
        e = self.emb(x)
        mask = (x != PAD).unsqueeze(-1).float()
        return self.fc((e * mask).sum(1) / mask.sum(1).clamp(min=1))


class LSTMClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(VOCAB_SIZE, EMB_DIM, padding_idx=PAD)
        self.lstm = nn.LSTM(EMB_DIM, HIDDEN, batch_first=True)
        self.fc = nn.Linear(HIDDEN, 2)

    def forward(self, x):
        e = self.emb(x)
        lengths = (x != PAD).sum(1).clamp(min=1).cpu()      # 길이는 CPU에 있어야 한다
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
        self.fc = nn.Linear(EMB_DIM, 2)

    def forward(self, x):
        e = self.emb(x) + self.pos
        pad = (x == PAD)
        a, _ = self.attn(e, e, e, key_padding_mask=pad)
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
        self.fc = nn.Linear(EMB_DIM, 2)

    def forward(self, x):
        h = self.emb(x) + self.pos
        pad = (x == PAD)
        a, _ = self.attn(h, h, h, key_padding_mask=pad)
        h = self.n1(h + a)
        h = self.n2(h + self.ff(h))
        m = (~pad).unsqueeze(-1).float()
        return self.fc((h * m).sum(1) / m.sum(1).clamp(min=1))


RUNGS = [("3 평균", MeanEmbedding), ("4 LSTM", LSTMClassifier),
         ("5 어텐션", AttentionClassifier), ("6 트랜스포머", TransformerClassifier)]


# ===========================================================================
# 학습과 평가
# ===========================================================================
@torch.no_grad()
def accuracy(m, Xte, yte):
    m.eval()
    ok = 0
    for i in range(0, len(Xte), 1000):
        xb = Xte[i:i + 1000].to(device)
        ok += (m(xb).argmax(1).cpu() == yte[i:i + 1000]).sum().item()
    return 100.0 * ok / len(Xte)


def train_one(Model, seed, Xtr, ytr, Xte, yte):
    torch.manual_seed(seed)
    m = Model().to(device)
    opt = optim.Adam(m.parameters(), lr=LR)
    crit = nn.CrossEntropyLoss()
    g = torch.Generator().manual_seed(seed)
    ld = DataLoader(TensorDataset(Xtr, ytr), batch_size=BATCH,
                    shuffle=True, generator=g)
    for _ in range(EPOCHS):
        m.train()
        for xb, yb in ld:
            opt.zero_grad()
            crit(m(xb.to(device)), yb.to(device)).backward()
            opt.step()
    return accuracy(m, Xte, yte)


def load_done():
    if os.path.exists(RESULTS):
        with open(RESULTS) as f:
            return json.load(f)
    return {}


def save(done):
    tmp = RESULTS + ".tmp"
    with open(tmp, "w") as f:
        json.dump(done, f, indent=1)
    os.replace(tmp, RESULTS)            # 쓰다 끊겨도 파일이 깨지지 않는다


# ===========================================================================
if __name__ == "__main__":
    done = load_done()
    for n in SIZES:
        need = [f"{n}|{r}|{s}" for r, _ in RUNGS for s in SEEDS]
        if all(k in done for k in need):
            print(f"\n=== {n:,}편 — 이미 끝났다, 건너뛴다 ===", flush=True)
            continue

        print(f"\n{'='*58}\n=== 학습 {n:,}편 ===\n{'='*58}", flush=True)
        t0 = time.time()
        Xtr, ytr, Xte, yte = prepare(n)
        print(f"  자료 준비 {time.time()-t0:.0f}초  {tuple(Xtr.shape)}", flush=True)

        for name, Model in RUNGS:
            for s in SEEDS:
                key = f"{n}|{name}|{s}"
                if key in done:
                    continue
                t0 = time.time()
                acc = train_one(Model, s, Xtr, ytr, Xte, yte)
                done[key] = acc
                save(done)                       # 씨앗마다 남긴다
                print(f"    {name}  씨앗 {s}  {acc:.2f}%  "
                      f"({time.time()-t0:.0f}초)", flush=True)
            got = [done[f"{n}|{name}|{s}"] for s in SEEDS]
            print(f"  >> {name}  평균 {sum(got)/len(got):.2f}%  "
                  f"퍼짐 {max(got)-min(got):.2f}\n", flush=True)

        del Xtr, ytr, Xte, yte

    # === 정리: 뒤집힘이 풀리는가 ===========================================
    print(f"\n{'='*58}\n=== 사다리가 크기를 따라 어떻게 바뀌는가 ===\n{'='*58}")
    print(f"{'크기':>9} | {'3 평균':>8} {'4 LSTM':>8} {'5 어텐션':>8} {'6 트랜스':>8} | {'3→5':>7}")
    print("-" * 62)
    for n in SIZES:
        try:
            avg = {name: sum(done[f"{n}|{name}|{s}"] for s in SEEDS) / len(list(SEEDS))
                   for name, _ in RUNGS}
        except KeyError:
            continue
        gap = avg["5 어텐션"] - avg["3 평균"]
        print(f"{n:>9,} | {avg['3 평균']:>8.2f} {avg['4 LSTM']:>8.2f} "
              f"{avg['5 어텐션']:>8.2f} {avg['6 트랜스포머']:>8.2f} | {gap:>+7.2f}")
    print("\n3→5가 음수면 뒤집힌 것이고, 크기를 따라 0으로 올라가면"
          "\n'자료가 작아서'라는 가설이 서는 것이다.")
