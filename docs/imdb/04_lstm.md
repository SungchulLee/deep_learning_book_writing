# 4걸음 LSTM — 차례를 읽었는데 뒤로 간다

[3걸음](03_embedding.md)은 낱말 벡터를 **평균**냈다. 평균은 순서를 보지 않으므로 "좋지 않다"와 "않다 좋지"가 같은 값이 된다. 그 절의 마지막 꼭지가 그것을 한계로 적었다.

이 절이 그 한계를 푼다. 왼쪽에서 오른쪽으로 한 낱말씩 읽으며 기억을 이어 가는 LSTM이다. 매개변수는 3걸음과 거의 같으므로(1,313,410 대 1,280,130, 2.6% 많다) 차이가 난다면 그것은 크기가 아니라 **차례를 읽은 몫**이다.

**그런데 값이 내려간다.** 이 절은 그 내려감을 어떻게 읽어야 하는지에 대한 것이다.

---

## 1. 마지막 숨은 상태 하나로

LSTM은 낱말을 하나씩 받아 **숨은 상태**를 고쳐 나간다. 평 하나를 다 읽고 나면 마지막 상태 하나가 남고, 그것이 평 전체를 나타내는 벡터가 된다. 그 위에 선형 층을 얹어 두 갈래로 가른다.

여기서 조심할 것이 하나 있다. [3걸음](03_embedding.md)과 같이 모든 평을 400칸으로 맞추었으므로 짧은 평 뒤에는 채움 칸이 수백 개 붙는다. 그대로 읽히면 **기억이 0으로 씻긴다.** 평을 다 읽은 뒤에도 빈 칸을 이백 번 더 읽으면 마지막 상태에는 아무것도 남지 않는다.

`pack_padded_sequence`가 그 일을 막는다. 평마다 진짜 길이를 재어 두고 거기까지만 읽게 한다.

---

## 2. 코드

```python
"""4걸음: LSTM. 낱말을 차례대로 읽는다.

3걸음은 낱말 벡터를 평균냈다. 평균은 순서를 보지 않으므로
"좋지 않다"와 "않다 좋지"가 같았다. 이 절은 왼쪽에서 오른쪽으로
한 낱말씩 읽으며 기억을 이어 간다.

매개변수가 3걸음과 거의 같다(1,313,410 대 1,280,130). 그러므로
차이가 나면 그것은 크기가 아니라 차례를 읽은 몫이다.
"""

import re, time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path("./data/aclImdb")
VOCAB_SIZE, MAX_LEN, EMB_DIM, HIDDEN = 20000, 400, 64, 64
EPOCHS, BATCH, LR, PAD = 5, 100, 1e-3, 0
torch.set_num_threads(1)          # 스레드 수가 바뀌면 더하는 차례가 바뀐다

TOKEN = re.compile(r"[a-z']+")


def tokenize(t):
    return TOKEN.findall(t.replace("<br />", " ").lower())


def load(split):
    docs, labels = [], []
    for label, name in ((1, "pos"), (0, "neg")):
        for f in sorted((ROOT / split / name).glob("*.txt")):
            docs.append(tokenize(f.read_text(encoding="utf-8")))
            labels.append(label)
    return docs, np.array(labels)


train_docs, train_y = load("train")
test_docs, test_y = load("test")
counts = Counter(w for d in train_docs for w in d)
vocab = ["<pad>"] + [w for w, _ in counts.most_common(VOCAB_SIZE - 1)]
index = {w: i for i, w in enumerate(vocab)}


def to_ids(docs):
    X = np.zeros((len(docs), MAX_LEN), dtype=np.int64)
    for r, d in enumerate(docs):
        ids = [index[w] for w in d if w in index][:MAX_LEN]
        X[r, :len(ids)] = ids
    return X


Xtr = torch.from_numpy(to_ids(train_docs)); ytr = torch.from_numpy(train_y).long()
Xte = torch.from_numpy(to_ids(test_docs));  yte = torch.from_numpy(test_y).long()
print(f"자료 학습 {len(train_docs):,}편  시험 {len(test_docs):,}편", flush=True)


# === 4걸음: 낱말을 차례대로 읽는다 ==========================================
class LSTMClassifier(nn.Module):
    """마지막 숨은 상태 하나로 평 전체를 나타낸다."""

    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(VOCAB_SIZE, EMB_DIM, padding_idx=PAD)
        self.lstm = nn.LSTM(EMB_DIM, HIDDEN, batch_first=True)
        self.fc = nn.Linear(HIDDEN, 2)

    def forward(self, x):
        e = self.emb(x)
        # 채움 칸을 빼고 진짜 길이만큼만 읽는다.
        # 안 그러면 짧은 평에서 0이 수백 칸 흘러 기억을 씻어 낸다
        lengths = (x != PAD).sum(1).clamp(min=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(e, lengths, batch_first=True,
                                                   enforce_sorted=False)
        _, (h, _) = self.lstm(packed)
        return self.fc(h[-1])


def run(seed):
    torch.manual_seed(seed)
    m = LSTMClassifier()
    opt = optim.Adam(m.parameters(), lr=LR)
    crit = nn.CrossEntropyLoss()
    g = torch.Generator().manual_seed(seed)
    ld = DataLoader(TensorDataset(Xtr, ytr), batch_size=BATCH, shuffle=True, generator=g)
    for _ in range(EPOCHS):
        m.train()
        for xb, yb in ld:
            opt.zero_grad(); crit(m(xb), yb).backward(); opt.step()
    m.eval()
    with torch.no_grad():
        pred = torch.cat([m(Xte[i:i + 500]).argmax(1) for i in range(0, len(Xte), 500)])
    return 100.0 * pred.eq(yte).float().mean().item(), sum(p.numel() for p in m.parameters())


accs = []
for seed in (0, 1, 2, 3, 4):
    t0 = time.time()
    acc, npar = run(seed)
    accs.append(acc)
    print(f"  씨앗 {seed}  {acc:.2f}%  ({time.time() - t0:.0f}s)", flush=True)

print(f"\n4걸음 LSTM  평균 {sum(accs) / len(accs):.2f}%  "
      f"퍼짐 {max(accs) - min(accs):.2f} ({min(accs):.2f}~{max(accs):.2f})  "
      f"매개변수 {npar:,}")
```

**출력:**

```
자료 학습 25,000편  시험 25,000편
  씨앗 0  83.51%
  씨앗 1  83.44%
  씨앗 2  81.41%
  씨앗 3  80.41%
  씨앗 4  78.76%

4걸음 LSTM  평균 81.51%  퍼짐 4.75 (78.76~83.51)  매개변수 1,313,410
```

---

## 3. 사다리가 처음으로 내려간다

| 걸음 | 정확도 | 남은 오차 | 지운 몫 | 매개변수 |
|---|---|---|---|---|
| [1 최근접 중심](01_counting.md) | 65.16% | 34.84%p | — | 40,000 |
| [2 선형 학습](02_linear.md) | 78.80% | 21.20%p | 39% | 40,002 |
| [3 박아 넣기 + 평균](03_embedding.md) | **87.03%** | 12.97%p | 39% | 1,280,130 |
| 4 LSTM | 81.51% | 18.49%p | **−43%** | 1,313,410 |

**5.52%포인트를 잃었다.** 남은 오차를 39%씩 지우며 올라오던 사다리가 여기서 지운 것을 43% 도로 뱉는다.

매개변수는 2.6% 늘었을 뿐이다. 그러므로 이것은 모델이 작아서 생긴 일이 아니다. **차례를 읽는 장치를 넣었더니 값이 내려갔다.**

세 장을 걸어오는 동안 이런 칸이 없었다. [4.5절 차원축소](../ch04/05_dimensionality_reduction.md)가 58.61%로 화소를 그냥 쓰느니만 못했던 자리가 가장 비슷한데, 그 절은 애초에 목표가 다른 사다리였다. 여기는 같은 사다리의 다음 칸이다.

---

## 4. 먼저 퍼짐을 보라

값만 보고 "차례는 쓸모없다"로 넘어가면 안 된다. 이 표의 마지막 열을 보라.

| 걸음 | 퍼짐 | 3걸음 대비 |
|---|---|---|
| 2 선형 학습 | 0.23 | — |
| 3 박아 넣기 + 평균 | 0.34 | 1배 |
| 4 LSTM | **4.75** | **14배** |

**앞의 세 걸음은 모두 3분의 1%포인트 안에 들어왔다.** 씨앗을 바꾸어도 값이 거의 움직이지 않았다는 뜻이고, 그래서 [4.1절](../ch04/01_two_ladders.md)의 규율대로 작은 차이까지 믿을 수 있었다.

이 걸음은 4.75%포인트를 오간다. 가장 좋은 씨앗과 가장 나쁜 씨앗 사이가 **3걸음이 번 것(8.23%포인트)의 절반이 넘는다.**

그리고 가장 나쁜 씨앗을 보라. **78.76%는 2걸음의 78.80%보다 낮다.** 매개변수 1,313,410개짜리 LSTM이 40,002개짜리 선형 모델에 진다. 다섯 번에 한 번 그렇게 된다.

**값이 이렇게 흔들리는 모델은 자리를 잡지 못한 모델이다.** 잘 학습된 모델은 씨앗을 바꾸어도 비슷한 곳에 내려앉는다. 앞의 세 걸음이 그랬다.

그러므로 여기서 읽어야 할 것은 "81.51%"라는 값이 아니라 **"퍼짐 4.75"라는 성질**이다. 앞 장들에서 퍼짐은 차이가 참인지를 가리는 문지기였다. 여기서는 그것이 **진단**이 된다.

---

## 5. 그래서 아직 결론이 아니다

[4장](../ch04/index.md)이 같은 함정을 세 번 적었다. 짧은 예산에서 재면 부호가 뒤집힌다.

| 절 | 5 에포크 | 30 에포크 |
|---|---|---|
| [4.4 증강](../ch04/04_augmentation.md) | −2.52%p | **+5.96%p** |
| [4.7 증류](../ch04/07_distillation.md) | −1.66%p | **+2.68%p** |

이 장도 5 에포크로 재고 있다. 그리고 **되돌이 그물은 그 함정에 가장 잘 걸리는 구조다.** 기억을 쓰는 법 자체를 배워야 기억이 값을 내기 시작하므로, 다른 구조보다 늦게 출발한다.

퍼짐 4.75가 바로 그 늦음의 모습일 수 있다. 다섯 벌 가운데 둘은 83%대에 닿았고 하나는 78%대에 머물렀다. **같은 설정에서 어떤 씨앗은 배우기 시작했고 어떤 씨앗은 아직 아니라는 뜻으로 읽힌다.**

그러므로 이 절이 지금 적을 수 있는 것은 여기까지다.

> **5 에포크에서 LSTM은 3걸음보다 못하고, 그 값은 믿을 만하지 않다.**

"차례가 도움이 되지 않는다"는 이보다 훨씬 큰 주장이며, 이 표만으로는 적을 수 없다. 다음 걸음이 같은 물음에 다른 구조로 답하고, 거기서 이 절의 내려감이 **구조 탓인지 예산 탓인지** 갈린다.

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
`pack_padded_sequence`를 빼고 그대로 400칸을 읽게 하면 정확도가 어떻게 될지 어림하고, 까닭을 적어라.

</div>

??? success "연습문제 1 풀이"
    **크게 떨어진다.** 평의 가운뎃값 길이가 169낱말이므로 절반이 넘는 평에서 채움 칸을 이백 번 넘게 읽는다.

    LSTM은 낱말을 받을 때마다 숨은 상태를 고친다. 받는 것이 `<pad>`의 벡터라도 고치기는 고친다. 같은 입력을 이백 번 연달아 받으면 상태는 그 입력이 끌고 가는 자리로 수렴하고, 앞서 읽은 평의 내용은 씻겨 나간다.

    곧 **마지막 상태가 나타내는 것이 평이 아니라 "채움 칸을 이백 번 읽은 뒤의 상태"가 된다.** 평이 길수록 덜 씻기므로, 길이에 따라 값이 달라지는 이상한 모델이 된다.

    [3걸음](03_embedding.md)의 평균에서도 같은 문제를 `mask`로 막았다. 다만 그쪽은 평균이라 채움 칸이 값을 **묽게** 만들 뿐이지만, 이쪽은 차례대로 읽으므로 **덮어쓴다.** 되돌이 그물에서 더 아프다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff normal" title="중간"></span>
본문은 퍼짐 4.75를 "자리를 잡지 못했다"는 진단으로 읽었다. 이 읽기가 맞는지 시험하려면 무엇을 재면 되는가? 두 가지를 적어라.

</div>

??? success "연습문제 2 풀이"
    **하나, 에포크를 늘려 다시 잰다.** 자리를 잡지 못한 것이라면 예산을 늘릴 때 값이 오르면서 **퍼짐이 함께 줄어야 한다.** 값만 오르고 퍼짐이 그대로라면 다른 까닭이 있는 것이다.

    **둘, 에포크마다 학습·시험 정확도를 이어서 그린다.** [4.2절](../ch04/02_depth.md)이 한 것과 같다. 씨앗 다섯 벌의 곡선을 겹쳐 그리면 다음이 갈린다.

    - 곡선들이 **아직 오르는 중에 끊겼다면** 예산 탓이다.
    - 곡선들이 이미 **평평해진 뒤에도 서로 다른 높이에 있다면** 예산 탓이 아니다. 씨앗마다 다른 곳에 갇혔다는 뜻이다.

    두 시험이 갈라놓는 것은 **덜 배운 것**과 **다르게 배운 것**이다. 값 하나로는 이 둘을 구별할 수 없고, 곡선을 보아야 갈린다. [4장](../ch04/index.md)이 "점을 늘리는 것과 곡선을 보는 것은 다른 일이다"라고 적은 것이 이 자리에도 그대로 걸린다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
이 절의 LSTM은 마지막 숨은 상태 하나만 쓴다. 400칸을 읽는 동안 나온 상태를 **모두** 평균내어 쓰면 어떻게 될지 어림하라. 그리고 그 모델이 [3걸음](03_embedding.md)과 무엇이 같고 무엇이 다른지 적어라.

</div>

??? success "연습문제 3 풀이"
    **아마 좋아진다.** 그리고 그 까닭이 이 절의 결과를 설명한다.

    마지막 상태 하나만 쓰면 기울기가 그 한 점을 지나 뒤로 흘러야 한다. 400칸을 거슬러 올라가는 동안 기울기가 약해지므로, 앞부분 낱말은 배우기 어렵다. 상태를 모두 평균내면 **모든 칸이 출력으로 가는 짧은 길을 하나씩 갖게 되어** 기울기가 고르게 퍼진다.

    3걸음과 견주면 이렇다.

    | | 3걸음 | 상태 평균 LSTM |
    |---|---|---|
    | 무엇을 평균내는가 | 낱말 **벡터** | 낱말까지 읽은 **상태** |
    | 차례를 보는가 | 아니다 | **본다** — $t$번째 상태는 앞의 $t$개에 딸려 있다 |
    | 출력까지의 길이 | 모든 칸이 1 | 모든 칸이 1 |

    **평균을 낸다는 뼈대는 같고, 평균내는 것이 달라진다.** 3걸음의 강점(짧은 길)을 지키면서 차례를 더하는 셈이다.

    이것이 알려 주는 것이 하나 더 있다. 3걸음이 잘한 까닭의 일부는 "낱말의 뜻을 배워서"가 아니라 **평균이라는 구조가 학습을 쉽게 만들어서**일 수 있다는 것이다. 다음 걸음의 어텐션도 이 성질을 지닌다. 어텐션은 무게를 준 평균이고, 평균인 한 모든 칸이 출력까지 짧은 길을 갖는다.
