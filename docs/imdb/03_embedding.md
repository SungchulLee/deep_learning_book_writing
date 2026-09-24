# 박아 넣기 — 낱말을 벡터로

[2걸음](02_linear.md)은 낱말마다 **수 하나**를 배웠다. `excellent`의 가중치, `the`의 가중치. 그것으로 78.80%를 얻었다.

이 절은 낱말마다 **벡터**를 배운다. 64개의 수다. 그 벡터들을 평균내어 평 하나를 나타내고, 그 위에 선형 층을 얹는다.

**차례는 여전히 버린다.** 평균은 순서를 보지 않으므로 "좋지 않다"와 "않다 좋지"는 아직 같다. 바뀌는 것은 낱말을 **어떻게 나타내는가**뿐이다.

---

## 1. 길이를 맞추는 일이 먼저다

[2걸음](02_linear.md)까지는 평을 세기 벡터로 바꾸었으므로 길이가 저절로 20,000으로 같았다. 이제는 낱말을 차례대로 넣어야 하므로 길이가 제각각인 것이 문제가 된다.

흔한 방법이 **채움과 자르기**다. 400칸으로 정하고, 짧으면 0으로 채우고 길면 뒤를 버린다.

값이 둘 든다. **평의 12.5%가 400칸을 다 채워 뒷부분이 잘린다.** 그리고 짧은 평에는 뜻 없는 0이 잔뜩 붙으므로, 평균을 낼 때 그 칸들을 빼야 한다. 코드의 `mask`가 하는 일이다.

---

## 2. 코드

```python
"""3걸음: 낱말을 배운 벡터로 바꾸고 평균낸다.

2걸음은 낱말마다 수 하나(가중치)를 배웠다. 이 절은 낱말마다 **벡터**를 배운다.
그 벡터들을 평균내어 평 하나를 나타내고, 그 위에 선형 층을 얹는다.

차례는 여전히 버린다. 평균은 순서를 보지 않는다. 바뀌는 것은
'낱말을 어떻게 나타내는가'뿐이다.
"""
import re
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path("./data/aclImdb")
VOCAB_SIZE, MAX_LEN, EMB_DIM = 20000, 400, 64
EPOCHS, BATCH, LR = 5, 100, 1e-3
PAD = 0

TOKEN = re.compile(r"[a-z']+")
def tokenize(t): return TOKEN.findall(t.replace("<br />"," ").lower())
def load(split):
    docs, labels = [], []
    for label, name in ((1,"pos"),(0,"neg")):
        for f in sorted((ROOT/split/name).glob("*.txt")):
            docs.append(tokenize(f.read_text(encoding="utf-8"))); labels.append(label)
    return docs, np.array(labels)

train_docs, train_y = load("train"); test_docs, test_y = load("test")
counts = Counter(w for d in train_docs for w in d)
# 0번은 채움(PAD)으로 비워 둔다
vocab = ["<pad>"] + [w for w,_ in counts.most_common(VOCAB_SIZE-1)]
index = {w:i for i,w in enumerate(vocab)}

def to_ids(docs):
    """평 하나를 길이 MAX_LEN의 번호 열로. 짧으면 0으로 채우고 길면 자른다."""
    X = np.zeros((len(docs), MAX_LEN), dtype=np.int64)
    for r,d in enumerate(docs):
        ids = [index[w] for w in d if w in index][:MAX_LEN]
        X[r,:len(ids)] = ids
    return X

Xtr = torch.from_numpy(to_ids(train_docs)); ytr = torch.from_numpy(train_y).long()
Xte = torch.from_numpy(to_ids(test_docs));  yte = torch.from_numpy(test_y).long()
lens = (Xtr != PAD).sum(1)
print(f"자료 학습 {len(train_docs):,}편  시험 {len(test_docs):,}편")
print(f"{MAX_LEN}칸으로 맞춘 뒤: 채움이 아닌 칸 가운뎃값 {int(lens.median())}, "
      f"{MAX_LEN}칸을 다 채운 평 {100.0*(lens==MAX_LEN).float().mean():.1f}%")

class MeanEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(VOCAB_SIZE, EMB_DIM, padding_idx=PAD)
        self.fc = nn.Linear(EMB_DIM, 2)
    def forward(self, x):
        e = self.emb(x)                          # (B, MAX_LEN, EMB_DIM)
        mask = (x != PAD).unsqueeze(-1).float()  # 채움 칸은 평균에서 뺀다
        mean = (e*mask).sum(1) / mask.sum(1).clamp(min=1)
        return self.fc(mean)

def run(seed):
    torch.manual_seed(seed)
    m = MeanEmbedding()
    opt = optim.Adam(m.parameters(), lr=LR); crit = nn.CrossEntropyLoss()
    g = torch.Generator().manual_seed(seed)
    ld = DataLoader(TensorDataset(Xtr,ytr), batch_size=BATCH, shuffle=True, generator=g)
    for _ in range(EPOCHS):
        m.train()
        for xb,yb in ld:
            opt.zero_grad(); crit(m(xb),yb).backward(); opt.step()
    m.eval()
    with torch.no_grad():
        acc = 100.0*torch.cat([m(Xte[i:i+1000]).argmax(1) for i in range(0,len(Xte),1000)]
                              ).eq(yte).float().mean().item()
    return acc, sum(p.numel() for p in m.parameters()), m

accs=[]
for s in range(5):
    a, npar, m = run(s); accs.append(a)
    print(f"  씨앗 {s}  {a:.2f}%", flush=True)
print(f"\n3걸음 박아 넣기 + 평균  평균 {sum(accs)/len(accs):.2f}%  "
      f"퍼짐 {max(accs)-min(accs):.2f} ({min(accs):.2f}~{max(accs):.2f})  매개변수 {npar:,}")

# 배운 벡터가 뜻을 담았는가 — 가까운 낱말을 본다
with torch.no_grad():
    E = m.emb.weight[1:]
    E = E/E.norm(dim=1, keepdim=True).clamp(min=1e-8)
    for w in ["great","bad","boring"]:
        i = index[w]-1
        sim = (E @ E[i]); sim[i] = -9
        print(f"  '{w}'과 가까운 낱말:", ", ".join(vocab[1+j] for j in sim.topk(6).indices))
```

**출력:**

```
자료 학습 25,000편  시험 25,000편
400칸으로 맞춘 뒤: 채움이 아닌 칸 가운뎃값 169, 400칸을 다 채운 평 12.5%
  씨앗 0  86.97%
  씨앗 1  86.96%
  씨앗 2  87.03%
  씨앗 3  87.26%
  씨앗 4  86.92%

3걸음 박아 넣기 + 평균  평균 87.03%  퍼짐 0.34 (86.92~87.26)  매개변수 1,280,130
  'great'과 가까운 낱말: perfect, ages, best, enjoyed, rare, malone
  'bad'과 가까운 낱말: money, dull, worst, below, stars', clone
  'boring'과 가까운 낱말: effort, lacks, pills, clone, poor, annoying
```

---

## 3. 87.03%

| 걸음 | 정확도 | 남은 오차 | 지운 몫 | 매개변수 |
|---|---|---|---|---|
| [1 최근접 중심](01_counting.md) | 65.16% | 34.84%p | — | 40,000 |
| [2 선형 학습](02_linear.md) | 78.80% | 21.20%p | 39% | 40,002 |
| 3 박아 넣기 + 평균 | **87.03%** | 12.97%p | **39%** | **1,280,130** |

**8.23%포인트를 벌었다.** 씨앗 퍼짐이 0.34이므로 의심할 여지가 없다. 그리고 두 걸음이 공교롭게도 남은 오차를 똑같이 39%씩 지운다.

그런데 매개변수가 **32배**로 늘었다.

---

## 4. 나타냄이 번 것인가, 크기가 번 것인가

앞의 두 걸음은 매개변수가 같았으므로 이 물음이 없었다. 여기서는 물어야 한다. 박아 넣기 차원만 바꾸어 재어 보면 갈린다.

| 박아 넣기 차원 | 정확도 | 매개변수 |
|---|---|---|
| 1 | 71.16% | 20,004 |
| 4 | 80.05% | 80,010 |
| 16 | 84.56% | 320,034 |
| 64 | **86.96%** | 1,280,130 |

**크기를 따라 고르게 오른다.** 어느 차원에서 갑자기 좋아지는 문턱 같은 것이 없다.

그리고 눈여겨볼 칸이 둘이다.

**차원 1은 2걸음보다 못하다**(71.16% 대 78.80%). 낱말마다 수 하나를 배우는 것은 2걸음과 같은데, 2걸음은 갈래마다 하나씩 둘을 배우고 이쪽은 하나만 배운 뒤 선형 층으로 넘긴다. 자유도가 적으니 못하는 것이 이치에 맞는다.

**차원 4는 2걸음을 겨우 넘는다**(80.05% 대 78.80%). 매개변수가 두 배인데 1.25%포인트다.

그러므로 이 걸음이 번 8.23%포인트를 "낱말의 뜻을 배웠다"로 읽으면 **과장이다.** 상당 부분은 **모델이 커진 몫**이다. 정직하게 적자면 이렇다. **나타냄을 벡터로 바꾸면 크기를 키울 길이 열리고, 그 길을 따라가면 좋아진다.** 벡터라는 형태 자체가 마법을 부린 것이 아니다.

---

## 5. 배운 벡터가 뜻을 담았는가

출력의 마지막 세 줄이 그 물음에 답한다.

> `great`과 가까운 낱말: **perfect**, ages, **best**, **enjoyed**, rare, malone
>
> `bad`과 가까운 낱말: money, **dull**, **worst**, below, stars', clone
>
> `boring`과 가까운 낱말: effort, **lacks**, pills, clone, **poor**, **annoying**

굵게 칠한 것은 그럴듯하다. `great`—`perfect`—`best`, `boring`—`lacks`—`poor`—`annoying`은 뜻이 닿는다.

**나머지는 그렇지 않다.** `ages`, `malone`, `pills`, `clone`, `stars'`가 왜 거기 있는지 설명하기 어렵다. `malone`은 사람 이름이다.

!!! warning "이 낱말들은 씨앗 하나의 것이다"
    위 목록은 마지막으로 학습한 모델(씨앗 4)에서 뽑은 것이다. 씨앗 0으로 다시 학습하면
    `great` 곁에 `vivid, wonderfully, ideal, wonderful, replies, today`가 오고
    `boring` 곁에는 `mess, poorly, unfortunately, spare, ridiculous, sucks`가 온다.

    **어느 낱말이 오는지는 씨앗마다 바뀐다.** 바뀌지 않는 것은 그 목록이
    '그럴듯한 것 반, 까닭을 알 수 없는 것 반'이라는 성질이다. 아래 논의는
    그 성질에 대한 것이지 특정 낱말에 대한 것이 아니다.

까닭은 이 벡터들이 **뜻을 배운 것이 아니라 이 일감을 배웠기** 때문이다. 목표는 오직 좋게 보았는지 나쁘게 보았는지를 맞히는 것이므로, 어떤 배우 이름이 나쁜 평에 자주 나왔다면 그 이름이 `bad` 쪽으로 끌려간다. 뜻이 닮아서가 아니라 **같은 갈래의 평에 함께 나와서**다.

!!! note "word2vec 같은 것과 다르다"
    낱말 벡터라 하면 흔히 "왕 − 남자 + 여자 = 여왕" 같은 것을 떠올린다. 그런 벡터는 **훨씬 많은 글**에서 **낱말 스스로를 맞히는 일**로 배운 것이다.

    여기의 벡터는 영화평 25,000편에서 **감정 하나**를 맞히는 일로 배웠다. 그러니 감정 방향으로는 줄이 서지만 그 밖의 뜻은 담기지 않는다. 같은 이름을 쓰지만 같은 것이 아니다.

---

## 6. 아직 차례를 모른다

87.03%는 좋은 값이지만 이 모델도 [2걸음](02_linear.md)과 같은 벽 앞에 있다. **평균은 순서를 보지 않는다.**

"훌륭하다고 말하고 싶지만 지루했다"와 "지루하다고 말하고 싶지만 훌륭했다"는 낱말이 같으므로 벡터의 평균도 같다. 판정은 반대인데 모델이 보는 것은 하나다.

[4걸음 LSTM](04_lstm.md)이 이 벽을 다룬다.

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff normal" title="중간"></span>
평의 12.5%가 400칸에서 잘린다. 잘린 뒷부분에 판정을 뒤집을 말이 있다면? `MAX_LEN`을 키우면 해결되는가?

</div>

??? success "연습문제 1 풀이"
    **키우면 잘림은 줄지만 값이 든다.** 800으로 두면 셈이 두 배가 되고, 평의 대부분은 400칸도 채우지 못하므로 그 셈의 대부분이 채움 칸에 쓰인다.

    그런데 이 절의 모델에서는 **잘림이 생각만큼 아프지 않다.** 평균을 내므로 낱말 하나하나의 몫이 $1/n$로 작고, 400낱말을 이미 읽었다면 뒤의 몇 낱말이 평균을 크게 움직이지 못한다.

    **차례를 읽는 모델에서는 다르다.** 영화평은 마지막에 판정을 적는 일이 잦아("결론적으로 추천하지 않는다") 뒤를 자르면 가장 중요한 부분을 버릴 수 있다. 그래서 자를 때 **앞이 아니라 뒤를 남기는** 쪽이 나은 경우가 있다.

    확인하려면 `[:MAX_LEN]`을 `[-MAX_LEN:]`으로 바꾸어 재어 보면 된다. 이 절의 모델에서는 거의 안 바뀌고, LSTM에서는 달라질 수 있다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff hard" title="어려움"></span>
본문은 이득의 상당 부분이 "모델이 커진 몫"이라고 했다. 그 둘을 더 깨끗하게 가르려면 어떤 실험을 해야 하는가?

</div>

??? success "연습문제 2 풀이"
    **2걸음 쪽의 크기를 키워 맞대어야 한다.** 지금 표는 박아 넣기 쪽만 크기를 바꾸었으므로, 크기가 같을 때 어느 나타냄이 나은지는 말하지 못한다.

    맞대는 방법이 둘 있다.

    **하나, 2걸음에 은닉층을 넣는다.** 세기 벡터 20,000개를 받아 64칸으로 줄였다가 2로 보내면 매개변수가 약 128만 개로 3걸음과 거의 같아진다. 그러면 "같은 크기에서 세기냐 박아 넣기냐"를 물을 수 있다.

    **둘, 3걸음의 낱말 목록을 줄인다.** 차원 64를 두고 낱말을 5,000개로 줄이면 32만 개가 되어 중간 크기와 견줄 수 있다.

    둘 다 이 절이 하지 않은 실험이다. 그러므로 본문의 "상당 부분은 크기의 몫"은 **차원을 키우면 좋아진다는 사실까지만** 말한 것이고, 같은 크기에서 어느 쪽이 나은지는 재지 않았다. [4.2절](../ch04/02_depth.md)이 깊이와 필터 크기를 가를 때 쓴 방식이 여기에도 그대로 필요하다.
