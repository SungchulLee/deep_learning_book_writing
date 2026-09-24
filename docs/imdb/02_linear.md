# 선형 학습 — 같은 특징, 배운 가중치

[1걸음](01_counting.md)은 낱말 세기 벡터를 갈래마다 평균 냈다. 이 절은 **같은 벡터를 그대로 쓰되 가중치를 학습한다.**

바뀌는 것이 그것 하나뿐이다. 특징도 같고 매개변수 수도 사실상 같다.

| | 매개변수 | 그 값을 어떻게 정하는가 |
|---|---|---|
| [1걸음 최근접 중심](01_counting.md) | 40,000 | 갈래마다 **평균**을 낸다 |
| 2걸음 선형 학습 | 40,002 | 자료에 맞추어 **학습한다** |

둘 늘어난 것은 치우침(bias)이다. [3장 1→2걸음](../ch03/mnist/01_template_learning.md)이 7,850개로 똑같았던 것과 같은 자리이며, 묻는 것도 같다. **평균 대신 학습하면 얼마를 버는가.**

---

## 1. 코드

```python
"""2걸음: 같은 낱말 세기 벡터에 가중치를 학습시킨다.

1걸음과 특징이 같고 매개변수 수도 사실상 같다(40,000 대 40,002 — 치우침 둘).
다른 것은 그 수를 평균으로 못박느냐 자료에 맞추어 학습하느냐뿐이다.
3장 1→2걸음과 같은 실험이며, 자료 손질 부분은 1걸음과 같다.
"""
import re
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path("./data/aclImdb")   # 내려받아 풀어 둔 자리
VOCAB_SIZE = 20000

# =============================================================================
# 자료: 영화평을 읽어 낱말 목록으로 바꾼다
# =============================================================================
TOKEN = re.compile(r"[a-z']+")          # 소문자 낱말만. 숫자와 문장부호는 버린다

def tokenize(text):
    # HTML 줄바꿈이 원문에 그대로 남아 있다
    return TOKEN.findall(text.replace("<br />", " ").lower())

def load(split):
    """(낱말 목록들, 정답)을 돌려준다. 1 = 좋게 봄, 0 = 나쁘게 봄."""
    docs, labels = [], []
    for label, name in ((1, "pos"), (0, "neg")):
        for f in sorted((ROOT/split/name).glob("*.txt")):
            docs.append(tokenize(f.read_text(encoding="utf-8")))
            labels.append(label)
    return docs, np.array(labels)

train_docs, train_y = load("train")
test_docs,  test_y  = load("test")
print(f"자료 학습 {len(train_docs):,}편  시험 {len(test_docs):,}편")
print(f"평 하나의 낱말 수: 가운뎃값 {int(np.median([len(d) for d in train_docs]))}  "
      f"가장 짧은 것 {min(len(d) for d in train_docs)}  가장 긴 것 {max(len(d) for d in train_docs)}")

# =============================================================================
# 낱말 목록: 자주 나온 것 VOCAB_SIZE개만 남긴다
# =============================================================================
counts = Counter(w for d in train_docs for w in d)
vocab = [w for w, _ in counts.most_common(VOCAB_SIZE)]
index = {w: i for i, w in enumerate(vocab)}
print(f"낱말 {len(counts):,}가지 가운데 {VOCAB_SIZE:,}개를 쓴다")


def to_counts(docs):
    """평 하나를 낱말 세기 벡터로. (문서 수, VOCAB_SIZE)"""
    X = np.zeros((len(docs), VOCAB_SIZE), dtype=np.float32)
    for r, d in enumerate(docs):
        for w in d:
            j = index.get(w)
            if j is not None:
                X[r, j] += 1
    return X

Xtr, Xte = to_counts(train_docs), to_counts(test_docs)

# 평마다 길이가 다르므로 길이로 나눈다. 안 그러면 '긴 평'이 무조건 멀어진다.
# 3장에서는 그림 크기가 모두 같아 이 손질이 필요 없었다 — 글월의 첫 번째 차이다
def normalize(X):
    n = X.sum(1, keepdims=True)
    return X / np.maximum(n, 1)

Xtr_n, Xte_n = normalize(Xtr), normalize(Xte)

# =============================================================================
# 2걸음: 가중치를 학습한다
# =============================================================================
device = torch.device("cpu")          # 선형 모형 하나라 CPU로 넉넉하다
EPOCHS, BATCH, LR = 5, 100, 1e-3      # 3장과 같은 규약

Xtr_t = torch.from_numpy(Xtr_n); ytr_t = torch.from_numpy(train_y).long()
Xte_t = torch.from_numpy(Xte_n); yte_t = torch.from_numpy(test_y).long()


def run(seed):
    torch.manual_seed(seed)
    model = nn.Linear(VOCAB_SIZE, 2)          # 갈래가 둘이다
    opt = optim.Adam(model.parameters(), lr=LR)
    crit = nn.CrossEntropyLoss()
    g = torch.Generator().manual_seed(seed)
    loader = DataLoader(TensorDataset(Xtr_t, ytr_t), batch_size=BATCH,
                        shuffle=True, generator=g)
    for _ in range(EPOCHS):
        model.train()
        for xb, yb in loader:
            opt.zero_grad(); crit(model(xb), yb).backward(); opt.step()
    model.eval()
    with torch.no_grad():
        acc = 100.0*(model(Xte_t).argmax(1) == yte_t).float().mean().item()
    return acc, sum(p.numel() for p in model.parameters()), model


# 씨앗 다섯 개로 재어 퍼짐을 함께 적는다 (4.1절 규율)
accs = []
for s in range(5):
    acc, n_par, model = run(s)
    accs.append(acc)
    print(f"  씨앗 {s}  {acc:.2f}%", flush=True)

print(f"\n2걸음 선형 학습  평균 {sum(accs)/len(accs):.2f}%  퍼짐 {max(accs)-min(accs):.2f} "
      f"({min(accs):.2f}~{max(accs):.2f})  매개변수 {n_par:,}")

# 학습이 어떤 낱말을 골랐는가. 좋은 쪽 가중치에서 나쁜 쪽을 뺀다
W = model.weight.detach().numpy()
score = W[1] - W[0]
print("\n학습이 고른 좋은 쪽 낱말:", ", ".join(vocab[i] for i in np.argsort(score)[-12:][::-1]))
print("학습이 고른 나쁜 쪽 낱말:", ", ".join(vocab[i] for i in np.argsort(score)[:12]))
```

**출력:**

```
자료 학습 25,000편  시험 25,000편
평 하나의 낱말 수: 가운뎃값 173  가장 짧은 것 10  가장 긴 것 2462
낱말 85,680가지 가운데 20,000개를 쓴다
  씨앗 0  78.69%
  씨앗 1  78.85%
  씨앗 2  78.76%
  씨앗 3  78.92%
  씨앗 4  78.79%

2걸음 선형 학습  평균 78.80%  퍼짐 0.23 (78.69~78.92)  매개변수 40,002

학습이 고른 좋은 쪽 낱말: great, excellent, wonderful, best, love, perfect, amazing, well, beautiful, loved, highly, favorite
학습이 고른 나쁜 쪽 낱말: worst, bad, awful, waste, no, poor, worse, boring, nothing, terrible, even, money
```

---

## 2. 65.16% → 78.80%

**13.64%포인트를 번다.** 특징은 한 칸도 바꾸지 않았다.

| | 정확도 | 남은 오차 | 그 걸음이 지운 몫 |
|---|---|---|---|
| [1걸음 최근접 중심](01_counting.md) | 65.16% | 34.84%p | — |
| 2걸음 선형 학습 | **78.80%** | 21.20%p | **39%** |

씨앗 다섯 개의 퍼짐이 0.23이므로([4.1절](../ch04/01_two_ladders.md)의 규율) 13.64%포인트는 흔들림의 쉰 배가 넘는다. 의심할 여지가 없는 차이다.

3장에서 같은 걸음이 10.48%포인트를 벌었다. **글월에서 더 크게 번다.**

---

## 3. 무엇을 배웠는지 낱말이 말해 준다

이 절의 요점은 수가 아니라 아래 두 줄이다.

| | 위쪽 낱말 열둘 |
|---|---|
| [1걸음 평균](01_counting.md) | **and, is, of,** great, **in, as, the, his,** very, **a,** best, love |
| 2걸음 학습 | great, excellent, wonderful, best, love, perfect, amazing, well, beautiful, loved, highly, favorite |

**굵게 칠한 것이 사라졌다.** 평균이 위로 올린 `and`, `is`, `of`, `the`, `a`가 학습에서는 하나도 남지 않는다. 열두 자리가 전부 뜻을 가진 낱말이다. 나쁜 쪽도 마찬가지다 — `this`, `was`, `to`, `i`, `have`가 빠지고 `awful`, `waste`, `poor`, `boring`, `terrible`이 들어온다.

까닭은 학습이 **가중치를 낮출 수 있기** 때문이다. 평균은 모든 낱말을 똑같이 세어 더할 뿐이라, 자주 나오는 낱말이 판정을 좌우하는 것을 막을 방법이 없었다. 학습은 `the`의 가중치를 0 가까이 두고 `excellent`의 가중치를 키운다.

!!! note "3장에서도 같은 일이 있었다"
    [3장 2걸음](../ch03/linear_softmax/06_implementation.md)에서 학습된 가중치에 **음수**가 섞여 있는 것을 보았다. 클래스 평균은 "3은 이렇게 생겼다"만 담을 수 있는데 학습은 **"여기가 켜져 있으면 3이 아니다"**를 담는다.

    여기서도 같다. 평균은 "좋은 평에는 이런 낱말이 있다"만 담고, 학습은 **"이 낱말은 아무 평에나 있으니 무시하라"**를 담는다. 둘 다 평균이 표현할 수 없는 것이다.

---

## 4. 여기서 멈추는 것

78.80%는 우연 위로 28.80%포인트다. 그런데 이 모델은 아직 **차례를 전혀 모른다.**

"좋지 않다"와 "않다 좋지"가 같은 벡터이고, 더 나쁘게는 **"훌륭하다고 말하고 싶지만 지루했다"와 "지루하다고 말하고 싶지만 훌륭했다"가 같은 벡터다.** 두 평의 판정은 반대인데 낱말 자루가 같다.

낱말을 세는 한 이 벽을 넘을 수 없다. 다음 걸음이 먼저 낱말의 **나타냄**을 바꾸고, 그다음 4걸음 LSTM이 차례를 되찾는다.

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
1걸음과 2걸음의 매개변수 수가 40,000과 40,002로 거의 같다. 그런데 둘은 정말 같은 것을 담고 있는가?

</div>

??? success "연습문제 1 풀이"
    **아니다. 담을 수 있는 것의 범위가 다르다.**

    1걸음의 40,000개는 **두 중심의 좌표**다. 모두 0 이상이다. 낱말을 세어 평균을 냈으므로 음수가 나올 수 없다.

    2걸음의 40,002개는 **가중치**이며 음수가 될 수 있다. 그래서 "이 낱말이 있으면 나쁜 쪽"을 직접 담는다.

    수는 같지만 1걸음은 그 수의 **부분집합**만 쓸 수 있는 셈이다. 이것이 두 걸음의 차이를 설명하는 한 갈래이며, [3장 2걸음](../ch03/linear_softmax/06_implementation.md)이 같은 이야기를 그림으로 보인다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
본문은 "훌륭하다고 말하고 싶지만 지루했다"와 그 반대가 같은 벡터가 된다고 했다. 정말 같은가? 아니라면 어디가 다른가?

</div>

??? success "연습문제 2 풀이"
    **낱말 자루로는 정말 같다.** 두 월에 나오는 낱말이 같고 횟수도 같으므로 세기 벡터가 한 칸도 다르지 않다.

    그러므로 이 모델은 둘을 **원리적으로** 가를 수 없다. 더 오래 학습하거나 낱말을 더 많이 넣어도 안 된다. 두 입력이 같은 벡터이면 같은 출력이 나온다.

    이런 예가 자료에 얼마나 있는지가 실제 손해를 정한다. 확인하려면 세기 벡터가 같으면서 정답이 다른 평의 짝을 세어 보면 된다. IMDB에서 완전히 같은 짝은 드물지만, **거의 같은** 짝은 흔하다 — "재미없지 않다"와 "재미없다"는 낱말 하나 차이다.

    4걸음 LSTM이 이 자리를 다룬다. 다만 미리 짚어 두면, 차례를 읽을 수 있다고 해서 반드시 이기는 것은 아니다. **자료에 그 능력을 쓸 일이 얼마나 있는지**가 함께 정한다.
