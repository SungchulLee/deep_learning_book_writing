# IMDB 텍스트 분류 — 계수를 읽어 본다

앞 쪽들은 특징이 30개뿐인 자료(Breast Cancer)를 썼다. 이 쪽은 특징이 **20,000개**인 자료에 같은 모델을 건다. 영화평 5만 편을 좋게 보았는지 나쁘게 보았는지로 가르는 IMDB다.

로지스틱 회귀는 특징마다 수 하나를 배운다. 특징이 낱말이면 **낱말마다 수 하나**를 배우는 셈이고, 그 수는 읽을 수 있다. 이 쪽이 앞 쪽들과 다른 점이 거기에 있다. 정확도만 재고 끝내지 않고 **배운 계수를 들여다본다.**

!!! note "6장이 같은 자료로 사다리를 놓는다"
    [6장](../../imdb/index.md)이 이 자료 위에 다섯 걸음을 놓고 같은 규약으로 나란히 잰다. 낱말 세기에서 셀프 어텐션까지다. 이 쪽은 그 사다리의 한 칸을 **통계적 학습 쪽에서** 다시 보는 자리이며, 재는 것도 다르다. 6장은 정확도를, 이 쪽은 계수를 본다.

---

## 1. 코드

```python
"""IMDB에 로지스틱 회귀를 걸고, 배운 계수를 읽는다."""

import re
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path("./data/aclImdb")
VOCAB_SIZE, EPOCHS, BATCH, LR = 20000, 30, 256, 0.5
SEEDS = [0, 1, 2, 3, 4]

TOKEN = re.compile(r"[a-z']+")


def tokenize(t):
    return TOKEN.findall(t.replace("<br />", " ").lower())


def load(split):
    docs, labels = [], []
    for label, name in ((1, "pos"), (0, "neg")):
        for f in sorted((ROOT / split / name).glob("*.txt")):
            docs.append(tokenize(f.read_text(encoding="utf-8")))
            labels.append(label)
    return docs, np.array(labels, dtype=np.float32)


train_docs, train_y = load("train")
test_docs, test_y = load("test")
counts = Counter(w for d in train_docs for w in d)
vocab = [w for w, _ in counts.most_common(VOCAB_SIZE)]
index = {w: i for i, w in enumerate(vocab)}


def to_bow(docs):
    """평 하나를 20,000칸짜리 셈 벡터로. 길이로 나누어 길이 효과를 뺀다."""
    X = np.zeros((len(docs), VOCAB_SIZE), dtype=np.float32)
    for r, d in enumerate(docs):
        for w in d:
            j = index.get(w)
            if j is not None:
                X[r, j] += 1.0
        n = X[r].sum()
        if n:
            X[r] /= n
    return X


Xtr = torch.from_numpy(to_bow(train_docs)); ytr = torch.from_numpy(train_y).view(-1, 1)
Xte = torch.from_numpy(to_bow(test_docs));  yte = torch.from_numpy(test_y).view(-1, 1)
print(f"자료 학습 {len(train_docs):,}편  시험 {len(test_docs):,}편  낱말 {VOCAB_SIZE:,}개")


# ========================================================================
# 모델 — 선형 층 하나. 특징이 20,000개일 뿐 앞 쪽과 같은 로지스틱 회귀다
# ========================================================================
def run(seed):
    torch.manual_seed(seed)
    m = nn.Linear(VOCAB_SIZE, 1)
    opt = torch.optim.Adam(m.parameters(), lr=LR)
    crit = nn.BCEWithLogitsLoss()
    g = torch.Generator().manual_seed(seed)
    for _ in range(EPOCHS):
        perm = torch.randperm(len(Xtr), generator=g)
        for i in range(0, len(Xtr), BATCH):
            idx = perm[i:i + BATCH]
            opt.zero_grad()
            crit(m(Xtr[idx]), ytr[idx]).backward()
            opt.step()
    with torch.no_grad():
        acc = 100.0 * ((m(Xte) > 0).float() == yte).float().mean().item()
    return acc, m


accs, last = [], None
for s in SEEDS:
    a, m = run(s)
    accs.append(a); last = m
    print(f"  씨앗 {s}  {a:.2f}%", flush=True)
print(f"\n로지스틱 회귀 (낱말 주머니)  평균 {sum(accs)/len(accs):.2f}%  "
      f"퍼짐 {max(accs)-min(accs):.2f} ({min(accs):.2f}~{max(accs):.2f})  "
      f"매개변수 {VOCAB_SIZE+1:,}")

# === 배운 계수를 읽는다 =====================================================
w = last.weight.detach()[0]
top = torch.topk(w, 12).indices.tolist()
bot = torch.topk(-w, 12).indices.tolist()
print("\n좋은 평 쪽으로 가장 세게 미는 낱말")
print("  " + ", ".join(f"{vocab[i]}({w[i]:+.1f})" for i in top))
print("\n나쁜 평 쪽으로 가장 세게 미는 낱말")
print("  " + ", ".join(f"{vocab[i]}({w[i]:+.1f})" for i in bot))
```

**출력:**

```
자료 학습 25,000편  시험 25,000편  낱말 20,000개
  씨앗 0  85.90%
  씨앗 1  86.11%
  씨앗 2  85.70%
  씨앗 3  85.79%
  씨앗 4  86.14%

로지스틱 회귀 (낱말 주머니)  평균 85.93%  퍼짐 0.44 (85.70~86.14)  매개변수 20,001

좋은 평 쪽으로 가장 세게 미는 낱말
  refreshing(+275.5), flawless(+264.6), excellently(+243.2), caf(+242.1),
  vengeance(+238.8), kitty(+227.5), finely(+227.2), delightfully(+227.0),
  appreciated(+220.7), ankle(+220.1), nitpick(+217.9), moodiness(+217.0)

나쁜 평 쪽으로 가장 세게 미는 낱말
  disappointment(-276.2), poorly(-253.4), believer(-251.4), obnoxious(-244.6),
  unwatchable(-240.8), worst(-240.5), waste(-239.3), lacks(-231.5),
  stinker(-229.8), programming(-228.5), slightest(-228.4), wayans(-227.3)
```

---

## 2. 20,001개로 85.93%

특징이 30개에서 20,000개로 늘었는데 모델은 그대로다. 선형 층 하나에 `BCEWithLogitsLoss`이며, 바뀐 것은 입력 크기뿐이다.

퍼짐이 0.44이므로 다섯 씨앗이 좁게 모인다. 앞 쪽의 Breast Cancer에서는 시험 자료가 114개뿐이라 한 표본이 0.88%포인트였는데, 여기서는 25,000개라 한 표본이 0.004%포인트다. **자료가 크면 값이 안정된다.**

!!! warning "6장의 2걸음은 78.80%라고 적는다 — 같은 모델인데 왜 다른가"
    [6장 2걸음](../../imdb/02_linear.md)이 이것과 **같은 특징, 같은 모델**을 쓰고 78.80%를 적는다. 7%포인트 넘게 차이가 난다.

    까닭은 **학습 예산**이다.

    | | 6장 2걸음 | 이 쪽 |
    |---|---|---|
    | 에포크 | 5 | **30** |
    | 학습률 | $10^{-3}$ | **0.5** |

    6장은 다섯 걸음을 **하나의 규약으로** 재려고 3장에서 쓴 설정을 그대로 가져간다. 걸음들 사이를 견주는 것이 목적이므로 그것이 옳다. 이 쪽은 견줄 상대가 없으니 이 모델에 맞는 설정을 쓴다.

    곧 두 수는 서로 다른 물음의 답이다. 78.80%는 **"같은 규약에서 이 걸음이 어디쯤인가"**이고, 85.93%는 **"이 모델을 제대로 학습시키면 어디까지 가는가"**이다. 맞대어 놓고 어느 쪽이 맞다고 할 수 없다.

    다만 하나는 분명하다. **낱말 주머니에 로지스틱 회귀를 건 것이 생각보다 세다.** 6장이 1,280,130개짜리 모델로 얻은 87.03%에 20,001개로 1.1%포인트까지 다가선다.

---

## 3. 계수를 읽으면 절반만 말이 된다

로지스틱 회귀가 다른 모델과 갈리는 자리가 여기다. **가중치 하나가 낱말 하나에 곧바로 붙으므로 뜻을 물을 수 있다.**

좋은 쪽으로 미는 낱말부터 보자.

> **refreshing**, **flawless**, **excellently**, caf, vengeance, kitty, **finely**, **delightfully**, **appreciated**, ankle, nitpick, moodiness

굵게 칠한 여섯은 말이 된다. `flawless`, `excellently`, `delightfully`는 칭찬하는 낱말이다.

**나머지는 그렇지 않다.** `caf`, `kitty`, `ankle`, `nitpick`, `moodiness`가 왜 거기 있는지 설명하기 어렵다. 나쁜 쪽에서도 `believer`, `programming`, `wayans`(사람 이름)가 섞여 있다.

까닭은 두 가지다.

**하나, 드문 낱말이다.** 20,000번째 언저리의 낱말은 몇십 편에만 나온다. 그 몇십 편이 우연히 좋은 평이었다면 계수가 크게 붙는다. 자료가 적은 낱말일수록 계수가 극단으로 간다.

**둘, 이 모델에는 벌점이 없다.** 계수를 작게 붙들어 두는 항(L2 같은)이 없으므로 드문 낱말의 계수가 마음껏 커진다. 위의 수들이 ±270까지 가는 것이 그 표시다. 보통 로지스틱 회귀에 정칙화를 거는 까닭이 이것이다.

!!! note "6장도 같은 것을 겪는다"
    [6장 3걸음](../../imdb/03_embedding.md)이 낱말 벡터를 배운 뒤 `great`과 가까운 낱말을 뽑아 보는데, 거기에도 `ages`, `malone` 같은 것이 섞인다. 그 절이 적은 진단이 여기에도 그대로 걸린다.

    > 이 벡터들이 **뜻을 배운 것이 아니라 이 일감을 배웠기** 때문이다.

    낱말 주머니든 임베딩든, 목표가 "좋게 보았는지 맞히기" 하나뿐이면 **뜻이 닮은 낱말이 아니라 같은 갈래의 평에 함께 나온 낱말**이 가까워진다.

---

## 4. 이 모델이 못 보는 것

계수가 낱말마다 하나씩이라는 것은 **차례를 통째로 버린다**는 뜻이다. "좋지 않다"와 "않다 좋지"가 같은 벡터가 되고, "훌륭하다고 말하고 싶지만 지루했다"에서 `훌륭`과 `지루`가 각자 제 몫을 더할 뿐 **뒤집는 관계**를 잡지 못한다.

그런데도 85.93%가 나온다. 영화평의 감정은 **어떤 낱말이 나왔는가**로 거의 풀리기 때문이다.

차례를 읽는 장치를 붙이면 얼마나 더 버는지는 [6장](../../imdb/index.md)이 잰다. 답이 짐작과 다르다.

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
`to_bow`는 셈 벡터를 평의 길이로 나눈다. 이 줄을 빼면 무엇이 달라지겠는가?

</div>

??? success "연습문제 1 풀이"
    **긴 평이 더 큰 목소리를 갖게 된다.**

    나누지 않으면 낱말이 500개인 평의 벡터는 원소 합이 500이고, 50개인 평은 50이다. 로짓은 가중치와 입력의 안쪽곱이므로, 긴 평은 그것만으로 로짓의 크기가 열 배가 된다.

    그러면 모델이 **길이를 감정의 증거로 쓰기 시작한다.** IMDB에서 나쁜 평이 대체로 길다면 "길다 = 나쁘다"를 배우게 되고, 그것은 이 자료에서만 맞는 지름길이다.

    나누면 벡터가 **비율**이 된다. "이 평에서 `terrible`이 차지하는 몫"이 되어 길이와 무관해진다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
본문은 드문 낱말의 계수가 극단으로 간다고 적었다. 이것을 **재어서** 확인하라. 그리고 고칠 방법 하나를 코드로 적어라.

</div>

??? success "연습문제 2 풀이"
    **재는 법.** 낱말이 나온 평의 수와 그 낱말의 계수 크기를 짝지어 본다.

    ```python
    df = np.zeros(VOCAB_SIZE)                 # 낱말이 나온 평의 수
    for d in train_docs:
        for j in {index[w] for w in d if w in index}:
            df[j] += 1
    w = last.weight.detach()[0].numpy()
    for lo, hi in ((0, 50), (50, 500), (500, 5000), (5000, 30000)):
        sel = (df >= lo) & (df < hi)
        print(f"{lo:5d}~{hi:<6d}편에 나온 낱말 {sel.sum():5d}개  "
              f"계수 크기 평균 {np.abs(w[sel]).mean():.1f}")
    ```

    드문 쪽 칸의 평균이 훨씬 클 것이다.

    **고치는 법 하나 — 벌점을 건다.** `Adam`에 `weight_decay`를 주면 L2 벌점이 걸려 계수가 작게 유지된다.

    ```python
    opt = torch.optim.Adam(m.parameters(), lr=LR, weight_decay=1e-4)
    ```

    **다른 하나 — 드문 낱말을 아예 뺀다.** `most_common(VOCAB_SIZE)` 대신 나온 평이 5편 미만인 낱말을 버린다. 계수를 못 믿을 낱말은 애초에 특징으로 두지 않는 것이다.

    둘의 성격이 다르다. 벌점은 **모든 계수를 함께** 누르고, 낱말을 버리는 것은 **특정 낱말만** 없앤다. 앞엣것이 대개 낫지만 뒤엣것은 모델을 작게 만드는 덤이 있다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
본문은 이 모델이 "좋지 않다"와 "않다 좋지"를 구별하지 못한다고 적었다. 그런데도 85.93%가 나온다. **차례를 몰라서 틀리는 평**을 실제로 찾아내려면 어떻게 하겠는가?

</div>

??? success "연습문제 3 풀이"
    **틀린 평만 모아서 읽는 것으로는 부족하다.** 틀리는 까닭은 여럿이고(드문 낱말, 반어법, 줄거리 요약이 대부분인 평) 차례는 그중 하나일 뿐이다.

    **갈라내려면 차례를 아는 모델과 견주어야 한다.** [6장 5걸음](../../imdb/05_attention.md)의 어텐션 모델을 같이 돌려 두고, **이 모델은 틀렸는데 그쪽은 맞힌 평**만 모은다. 그 무리가 "차례를 알면 풀리는 평"의 후보다.

    그다음 그 무리에서 부정어(`not`, `n't`, `but`, `however`)가 나오는 비율을 전체 평과 견준다. 차례가 원인이라면 그 비율이 뚜렷이 높아야 한다.

    **더 곧은 시험 하나.** 평의 낱말을 무작위로 뒤섞어 다시 넣어 본다.

    - 낱말 주머니 모델은 **값이 하나도 안 바뀐다.** 벡터가 같기 때문이다.
    - 차례를 아는 모델은 값이 떨어져야 한다. 떨어지는 폭이 곧 **그 모델이 차례에서 얻고 있던 몫**이다.

    [3.4절](../../ch03/mnist/04_cnn.md)이 화소를 뒤섞어 합성곱이 이웃 관계에서 얻는 몫을 잰 것과 같은 수법이다. 자료를 망가뜨려 보고 값이 얼마나 떨어지는지로 **모델이 무엇에 기대고 있었는지**를 알아내는 것이다.
