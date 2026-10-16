# IMDB 텍스트 분류 — 계수를 읽어 본다

앞 절의 수치 피처 대신 영화 리뷰의 단어 빈도를 사용해 로지스틱 회귀를 학습한다. IMDB의 학습·테스트 각 25,000개 리뷰를 사용하고 학습 텍스트로 만든 20,000단어 어휘에 웨이트를 연결한다. 코드와 기록된 출력을 읽으며 계수 해석의 범위를 살펴본다.

로지스틱 회귀는 피처마다 수 하나를 배운다. 피처가 단어이면 **단어마다 수 하나**를 배우는 계산이고, 그 수는 읽을 수 있다. 이 쪽이 앞 쪽들과 다른 점이 거기에 있다. 정확도만 재고 끝내지 않고 **배운 계수를 들여다본다.**

!!! note "6장이 같은 데이터로 사다리를 놓는다"
    [6장](../../imdb/index.md)이 이 데이터 위에 다섯 스텝을 놓고 같은 규약으로 나란히 잰다. 단어 세기에서 셀프 어텐션까지다. 이 쪽은 그 사다리의 한 칸을 **통계적 학습 쪽에서** 다시 보는 자리이며, 재는 것도 다르다. 6장은 정확도를, 이 쪽은 계수를 본다.

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
# 모델 — 선형 레이어 하나. 피처가 20,000개일 뿐 앞 쪽과 같은 로지스틱 회귀다
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
      f"스프레드 {max(accs)-min(accs):.2f} ({min(accs):.2f}~{max(accs):.2f})  "
      f"파라메터 {VOCAB_SIZE+1:,}")

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

로지스틱 회귀 (낱말 주머니)  평균 85.93%  스프레드 0.44 (85.70~86.14)  파라메터 20,001

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

피처가 30개에서 20,000개로 늘었는데 모델은 그대로다. 선형 레이어 하나에 `BCEWithLogitsLoss`이며, 바뀐 것은 입력 크기뿐이다.

이 기록의 “스프레드” 0.44는 다섯 초기값의 정확도 최댓값과 최솟값의 차이다. 테스트셋 25,000개에서는 개별 실행의 정답 하나가 0.004%포인트에 해당한다. 큰 테스트셋은 표본 오차를 줄이는 데 도움이 될 수 있지만 이 실행 범위가 표본 불확실성 전체나 신뢰구간을 나타내지는 않는다.

!!! warning "6장의 스텝2는 78.80%라고 적는다 — 같은 모델인데 왜 다른가"
    [6장 스텝2](../../imdb/02_linear.md)이 이것과 **같은 피처, 같은 모델**을 쓰고 78.80%를 적는다. 7%포인트 넘게 차이가 난다.

    까닭은 **학습 예산**이다.

    | | 6장 스텝2 | 이 쪽 |
    |---|---|---|
    | 에포크 | 5 | **30** |
    | 러닝레이트 | $10^{-3}$ | **0.5** |

    6장은 다섯 스텝을 **하나의 규약으로** 재려고 3장에서 쓴 설정을 그대로 가져간다. 스텝들 사이를 비교하는 것이 목적이므로 그것이 옳다. 이 쪽은 견줄 상대가 없으니 이 모델에 맞는 설정을 쓴다.

    두 수치는 러닝레이트와 반복 수가 다른 설정의 결과다. 따라서 같은 모델이라는 이유만으로 어느 한쪽이 틀렸다고 말할 수 없다. 반대로 이 페이지의 설정이 모델의 최대 성능을 달성했거나 테스트셋 튜닝 없이 선택되었다는 보장도 없다.

    다만 하나는 분명하다. **단어 주머니에 로지스틱 회귀를 건 것이 생각보다 세다.** 6장이 1,280,130개짜리 모델로 얻은 87.03%에 20,001개로 1.1%포인트까지 다가선다.

---

## 3. 계수를 읽으면 절반만 말이 된다

로지스틱 회귀가 다른 모델과 갈리는 자리가 여기다. **웨이트 하나가 단어 하나에 곧바로 붙으므로 뜻을 물을 수 있다.**

좋은 쪽으로 미는 단어부터 보자.

> **refreshing**, **flawless**, **excellently**, caf, vengeance, kitty, **finely**, **delightfully**, **appreciated**, ankle, nitpick, moodiness

굵게 칠한 여섯은 말이 된다. `flawless`, `excellently`, `delightfully`는 칭찬하는 단어다.

**나머지는 그렇지 않다.** `caf`, `kitty`, `ankle`, `nitpick`, `moodiness`가 왜 거기 있는지 설명하기 어렵다. 나쁜 쪽에서도 `believer`, `programming`, `wayans`(사람 이름)가 섞여 있다.

까닭은 두 가지다.

첫 번째 가설은 드문 단어의 표본 변동이다. 특정 단어가 나온 리뷰 수가 적으면 감정 라벨과의 표본 상관을 안정적으로 추정하기 어렵다. 다만 이 목록만으로 각 단어의 문서 빈도나 큰 계수의 원인을 확정할 수는 없으므로 실제 빈도와 계수를 함께 확인한다.

두 번째 요인은 L1·L2 같은 명시적 패널티가 없다는 점이다. 이는 큰 계수를 억제하는 직접 제약이 없다는 뜻이지만, 계수 크기에는 입력을 리뷰 길이로 나눈 스케일과 최적화 과정도 영향을 준다. 큰 절댓값 하나만으로 드문 단어의 과적합을 단정하지 않는다.

!!! note "6장도 같은 것을 겪는다"
    [6장 스텝3](../../imdb/03_embedding.md)이 단어 벡터를 배운 뒤 `great`과 가까운 단어를 뽑아 보는데, 거기에도 `ages`, `malone` 같은 것이 섞인다. 그 절이 적은 진단이 여기에도 그대로 걸린다.

    > 이 벡터들이 **뜻을 배운 것이 아니라 이 과제를 배웠기** 때문이다.

    단어 주머니든 임베딩든, 목표가 "좋게 보았는지 맞히기" 하나뿐이면 **뜻이 닮은 단어가 아니라 같은 클래스의 평에 함께 나온 단어**가 가까워진다.

---

## 4. 이 모델이 못 보는 것

계수가 단어마다 하나씩이라는 것은 **차례를 통째로 버린다**는 뜻이다. "좋지 않다"와 "않다 좋지"가 같은 벡터가 되고, "훌륭하다고 말하고 싶지만 지루했다"에서 `훌륭`과 `지루`가 각자 제 몫을 더할 뿐 **뒤집는 관계**를 잡지 못한다.

제공된 기록에서는 이 표현만으로 85.93%의 정확도를 얻었다. 이는 해당 분할에서 단어 빈도가 감정 분류에 유용한 정보를 담는다는 결과다. 모든 리뷰의 의미를 단어 등장만으로 설명할 수 있다거나 어순이 중요하지 않다는 결론은 아니다.

차례를 읽는 장치를 붙이면 얼마나 더 버는지는 [6장](../../imdb/index.md)이 잰다. 답이 짐작과 다르다.

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
`to_bow`는 계산 벡터를 평의 길이로 나눈다. 이 줄을 빼면 무엇이 달라지겠는가?

</div>

??? success "연습문제 1 풀이"
    **긴 평이 더 큰 목소리를 갖게 된다.**

    나누지 않은 벡터의 원소 합은 어휘 안에 포함된 토큰의 수와 연결된다. 단어 구성 비율이 같고 길이만 10배인 경우, 바이어스를 제외한 선형 점수도 10배가 된다. 서로 다른 내용의 두 리뷰에서 길이만으로 로짓이 정확히 10배라고 말할 수는 없다.

    길이와 라벨이 표본에서 관련되어 있으면 모델이 길이 정보를 활용할 수 있다. 그러나 IMDB의 특정 감정 리뷰가 더 길다는 사실은 별도로 측정해야 한다. 길이 정규화는 이런 척도 차이를 줄이지만 리뷰의 모든 길이 의존성을 제거하는 것은 아니다.

    나누면 벡터가 **비율**이 된다. "이 평에서 `terrible`이 차지하는 몫"이 되어 길이와 무관해진다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
본문은 드문 단어의 계수가 극단으로 간다고 적었다. 이것을 **재어서** 확인하라. 그리고 고칠 방법 하나를 코드로 적어라.

</div>

??? success "연습문제 2 풀이"
    **재는 법.** 단어가 나온 평의 수와 그 단어의 계수 크기를 짝지어 본다.

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

    빈도 구간별 계수 크기를 계산해 가설을 확인한다. 드문 단어의 평균이 실제로 큰지는 결과로 판단하며, 상관된 단어와 입력 정규화도 함께 고려한다.

    **고치는 법 하나 — 패널티를 건다.** `Adam`에 `weight_decay`를 주면 L2 패널티가 걸려 계수가 작게 유지된다.

    ```python
    opt = torch.optim.Adam(m.parameters(), lr=LR, weight_decay=1e-4)
    ```

    **다른 하나 — 드문 단어를 아예 뺀다.** `most_common(VOCAB_SIZE)` 대신 나온 평이 5편 미만인 단어를 버린다. 계수를 못 믿을 단어는 애초에 피처로 두지 않는 것이다.

    패널티는 계수의 크기에 비용을 부과하고, 빈도 기준은 피처 공간 자체를 바꾼다. 둘은 어휘 선택과 강도에 따라 다른 결과를 내므로 검증 데이터로 선택한다. 테스트셋을 이 선택에 반복 사용하지 않는다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
본문은 이 모델이 "좋지 않다"와 "않다 좋지"를 구별하지 못한다고 적었다. 그런데도 85.93%가 나온다. **차례를 몰라서 틀리는 평**을 실제로 찾아내려면 어떻게 하겠는가?

</div>

??? success "연습문제 3 풀이"
    **틀린 평만 모아서 읽는 것으로는 부족하다.** 틀리는 까닭은 여럿이고(드문 단어, 반어법, 줄거리 요약이 대부분인 평) 차례는 그중 하나일 뿐이다.

    **갈라내려면 차례를 아는 모델과 견주어야 한다.** [6장 스텝5](../../imdb/05_attention.md)의 어텐션 모델을 같이 돌려 두고, **이 모델은 틀렸는데 그쪽은 맞힌 평**만 모은다. 그 무리가 "차례를 알면 풀리는 평"의 후보다.

    후보 리뷰를 직접 읽고 부정·대조 표현의 비율을 비교한다. 다른 모델이 맞혔다는 사실만으로 원인이 어순이라고 확정할 수는 없다. 어휘, 용량과 최적화 차이도 통제해야 한다.

    **더 곧은 시험 하나.** 평의 단어를 무작위로 뒤섞어 다시 넣어 본다.

    - 단어 주머니 모델은 **값이 하나도 안 바뀐다.** 벡터가 같기 때문이다.
    - 어순을 활용하는 모델은 섞은 입력에서 예측이 달라질 수 있다. 성능이 줄었다면 어순 의존성의 단서지만, 감소 폭에는 입력 분포를 훼손한 효과도 함께 들어 있다. 감소가 반드시 나타나거나 그 폭이 어순의 순수한 기여와 같다는 보장은 없다.

    [3.4절](../../ch03/mnist/04_cnn.md)이 화소를 뒤섞어 컨볼루션이 이웃 관계에서 얻는 몫을 잰 것과 같은 수법이다. 데이터를 망가뜨려 보고 값이 얼마나 떨어지는지로 **모델이 무엇에 기대고 있었는지**를 알아내는 것이다.
