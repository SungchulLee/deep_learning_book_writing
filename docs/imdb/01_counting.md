# 낱말 세기와 최근접 중심

[3장 1걸음](../ch03/mnist/01_template_learning.md)은 숫자마다 그림을 평균 내어 템플릿 열 장을 만들고, 새 그림을 가장 가까운 템플릿의 숫자로 불렀다. 학습이 없었다.

이 절은 **똑같은 규칙을 글월에 건다.** 좋게 본 평을 모두 평균 내고, 나쁘게 본 평을 모두 평균 내어 중심 둘을 만든다. 새 평은 가까운 쪽으로 부른다.

바뀌는 것은 무엇을 세느냐뿐이다. 화소 대신 **낱말**을 센다.

---

## 1. 글월은 그림과 두 가지가 다르다

**길이가 제각각이다.** MNIST는 언제나 28×28이라 벡터의 길이가 784로 고정이었다. 영화평은 짧은 것이 열 낱말, 긴 것이 2,462 낱말이다.

그래서 세기 벡터를 그대로 쓰면 **긴 평이 무조건 멀어진다.** 낱말이 많으면 모든 칸의 값이 커지므로, 뜻과 상관없이 중심에서 멀어진다. 이 절은 평마다 낱말 수로 나누어 이를 없앤다. 3장에서는 필요 없던 손질이다.

**차례가 뜻을 바꾼다.** 화소를 뒤섞으면 그림이 망가지지만 사람은 무엇이 망가졌는지 안다. 낱말을 뒤섞으면 "이 영화는 좋지 않다"가 "좋다, 이 영화는 않다"가 된다.

**이 절은 그 차례를 통째로 버린다.** 낱말을 세기만 하므로 "좋지 않다"와 "않다 좋지"가 같은 벡터가 된다. 이것을 자루 모형(bag of words)이라 한다. 사다리의 4걸음 LSTM이 되찾을 것이 바로 이것이다.

---

## 2. 코드

```python
"""1걸음: 낱말을 세어 갈래마다 평균을 내고, 가까운 쪽으로 부른다.

3장 1걸음의 템플릿 학습을 글월에 그대로 옮긴 것이다. 거기서는 그림을
갈래마다 평균 내어 템플릿 열 장을 만들었고, 여기서는 영화평을 갈래마다
평균 내어 '좋은 평의 평균'과 '나쁜 평의 평균' 둘을 만든다.

학습이 없다. 평균을 내고 가까운 쪽을 고를 뿐이다.
"""
import re, time
from collections import Counter
from pathlib import Path
import numpy as np

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

t0 = time.time()
train_docs, train_y = load("train")
test_docs,  test_y  = load("test")
print(f"자료 학습 {len(train_docs):,}편  시험 {len(test_docs):,}편  ({time.time()-t0:.0f}s)")
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
# 1걸음: 갈래마다 평균 — 학습이 없다
# =============================================================================
centroid_pos = Xtr_n[train_y == 1].mean(0)
centroid_neg = Xtr_n[train_y == 0].mean(0)

def classify(X):
    """가까운 중심을 고른다. 거리는 제곱 유클리드."""
    d_pos = ((X - centroid_pos) ** 2).sum(1)
    d_neg = ((X - centroid_neg) ** 2).sum(1)
    return (d_pos < d_neg).astype(int)

acc = 100.0 * (classify(Xte_n) == test_y).mean()
print(f"\n1걸음 최근접 중심  시험 정확도 {acc:.2f}%")
print(f"매개변수 {2*VOCAB_SIZE:,} (학습하지 않고 평균으로 못박은 값)")

# 어떤 낱말이 두 중심을 가장 크게 갈라놓는가
diff = centroid_pos - centroid_neg
top_pos = np.argsort(diff)[-12:][::-1]
top_neg = np.argsort(diff)[:12]
print("\n좋은 쪽으로 미는 낱말:", ", ".join(vocab[i] for i in top_pos))
print("나쁜 쪽으로 미는 낱말:", ", ".join(vocab[i] for i in top_neg))
```

**출력:**

```
자료 학습 25,000편  시험 25,000편  (21s)
평 하나의 낱말 수: 가운뎃값 173  가장 짧은 것 10  가장 긴 것 2462
낱말 85,680가지 가운데 20,000개를 쓴다

1걸음 최근접 중심  시험 정확도 65.16%
매개변수 40,000 (학습하지 않고 평균으로 못박은 값)

좋은 쪽으로 미는 낱말: and, is, of, great, in, as, the, his, very, a, best, love
나쁜 쪽으로 미는 낱말: bad, this, was, movie, to, no, just, even, not, i, worst, have
```

---

## 3. 65.16%

우연이 50%이므로 15.16%포인트를 번 셈이다.

[3장 1걸음](../ch03/mnist/01_template_learning.md)의 82.03%와 나란히 놓으면 겉보기와 다른 것이 보인다.

| | 정확도 | 우연 | 우연 위로 번 몫 | 남은 자리의 몇 %인가 |
|---|---|---|---|---|
| [MNIST 템플릿](../ch03/mnist/01_template_learning.md) | 82.03% | 10% | +72.03 | **80%** |
| IMDB 최근접 중심 | 65.16% | 50% | +15.16 | **30%** |

**같은 규칙인데 글월에서는 훨씬 덜 번다.** 갈래가 둘뿐이라 우연이 50%로 높은 것도 있지만, 우연 위의 자리를 얼마나 먹었는지로 보아도 80% 대 30%다.

---

## 4. 왜 덜 버는가 — 낱말 목록이 말해 준다

출력의 마지막 두 줄이 까닭을 드러낸다.

> 좋은 쪽으로 미는 낱말: **and, is, of,** great, **in, as, the, his,** very, **a,** best, love
>
> 나쁜 쪽으로 미는 낱말: bad, **this, was,** movie, **to,** no, **just, even, not, i,** worst, **have**

굵게 칠한 것들은 뜻이 없다. `and`, `is`, `of`, `the`, `a`는 어느 평에나 나오는 낱말이다. 그런데 **중심을 가장 크게 갈라놓는 낱말 열두 개 가운데 절반을 차지한다.**

까닭은 평균이 **자주 나오는 것에 끌려가기** 때문이다. `the`는 한 평에 열 번도 나오고 `excellent`는 나와야 한 번이다. 두 중심의 차이를 재면 `the`의 작은 빈도 차이가 `excellent`의 있고 없음보다 큰 값을 낸다.

3장에서 템플릿이 **평균 밝기**에 끌려갔던 것과 같은 일이다. 거기서도 숫자를 가르는 것은 획의 모양인데 평균이 잡은 것은 밝기였다.

---

## 5. 그래서 다음 걸음이 무엇을 고쳐야 하는가

`and`와 `excellent`를 **다르게 대접할 수 있어야 한다.** 평균은 그럴 수 없다. 모든 낱말을 똑같이 세어 더할 뿐이다.

[다음 걸음](02_linear.md)은 특징을 그대로 두고 **가중치만 학습한다.** 매개변수 수도 같다. 달라지는 것은 그 수를 평균으로 못박느냐 자료에 맞추어 고르느냐뿐이다.

3장에서 같은 걸음이 10.48%포인트를 벌었다.

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
이 절은 세기 벡터를 평의 낱말 수로 나눈다. 나누지 않으면 무엇이 잘못되는가?

</div>

??? success "연습문제 1 풀이"
    **긴 평이 두 중심 모두에서 멀어진다.**

    거리를 제곱 유클리드로 재므로, 모든 칸의 값이 커지면 거리도 커진다. 2,462 낱말짜리 평은 10 낱말짜리 평보다 원점에서 훨씬 먼 자리에 놓인다.

    그런데 중심 둘은 **평균**이므로 가운뎃값쯤에 있다. 그러면 아주 긴 평은 두 중심 모두에서 비슷하게 멀어지고, 판정이 낱말 수라는 상관없는 값에 휘둘린다.

    직접 확인하려면 `normalize`를 건너뛰고 다시 돌려 보면 된다. 그리고 평의 길이와 맞았는지를 나란히 그려 보면, 나누지 않았을 때 긴 평에서 정확도가 떨어지는 것이 보인다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff normal" title="중간"></span>
자주 나오는 낱말이 판정을 좌우하는 문제를 학습 없이 고치는 방법이 있다. 무엇인가?

</div>

??? success "연습문제 2 풀이"
    **낱말마다 무게를 달리 주면 된다.** 가장 널리 쓰이는 것이 TF-IDF다.

    어떤 낱말이 **몇 편의 평에 나타나는지**를 세어, 많은 평에 나타날수록 무게를 낮춘다. `the`는 25,000편 모두에 나오므로 무게가 거의 0이 되고, `excellent`는 일부에만 나오므로 무게가 남는다.

    $$
    w(\text{낱말}) = \log \frac{\text{전체 평 수}}{\text{그 낱말이 나온 평 수}}
    $$

    이것도 **학습이 아니다.** 자료를 한 번 세어 정하는 값이라 이 절의 규칙을 벗어나지 않는다. 그런데도 이 절의 65.16%를 꽤 끌어올린다.

    그러니 다음 걸음의 이득을 전부 "학습이 벌었다"로 읽으면 안 된다. 그 가운데 일부는 **무게를 다르게 준 것**만으로도 얻을 수 있었다. 학습은 그 무게를 손으로 정하지 않고 자료에 맡긴다는 점이 다르다.
