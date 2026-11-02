# 6.5 가져다 쓰는 임베딩 — GloVe 사용법

<div class="leadbox" markdown>

**들어가는 이야기** — 천 배 많이 읽어도 고쳐지지 않는 것

앞의 네 쪽은 영화평 **6,000편**으로 벡터를 손수 만들었다. 이 쪽은 남이 **60억 낱말**로 만들어 둔 것을 가져와 같은 자로 잰다. 천 배쯤 되는 자료다.

많이 읽은 값이 분명히 있다. 우리 벡터로는 물어볼 엄두도 못 냈던 것이 된다.

```
man : king  = woman : queen, throne, prince
paris : france = tokyo : japan, korea, japanese
walk : walked = go : got, went, put
```

**널리 알려진 그 유추가 1등으로 나온다.** `bank`도 또렷하다 — `banks, securities, banking, investment`. 영화평만 읽은 우리 벡터는 「은행 강도 영화」 쪽으로 기울었는데, 위키백과와 뉴스를 읽은 쪽은 금융기관으로 간다.

**그런데 고쳐지지 않은 것이 하나 있다.**

| | 닮음 |
|---|---|
| good — great | +0.798 |
| good — **bad** | **+0.796** |
| hot — **cold** | **+0.801** |

**`good`과 `bad`가 `good`과 `great`만큼 가깝다.** 소수 셋째 자리에서 0.002 차이다. `hot`과 `cold`는 그보다도 가깝다. **60억 낱말을 읽고도 반대말을 못 가린다.**

자료를 더 넣어서 될 일이 아니라는 뜻이다. **분포 가설이 재는 것이 애초에 그것**이기 때문이다.

</div>

---

## 1. 파일을 손에 넣는다

이 책에서 **바깥에서 받아와야 돌아가는 쪽은 여기뿐**이다. 다른 쪽은 저장소에 있는 것이나 `torchvision`이 알아서 받아 오는 것을 쓴다. 그러니 이 절만 따로 둔다.

```bash
curl -L -o data/glove-wiki-gigaword-50.gz \
  https://github.com/piskvorky/gensim-data/releases/download/glove-wiki-gigaword-50/glove-wiki-gigaword-50.gz
```

**제대로 받았는지 확인한다.** 중간에 끊기거나 100차원 판을 받아도 **아무 말 없이** 이 쪽의 수가 전부 달라진다.

```bash
$ ls -l data/glove-wiki-gigaword-50.gz
-rw-r--r--  1 user  staff  69182535  data/glove-wiki-gigaword-50.gz

$ gzcat data/glove-wiki-gigaword-50.gz | wc -l
  400001

$ shasum -a 256 data/glove-wiki-gigaword-50.gz
5c55f98957aa9fed8d2ac5fb1dcff57af3b23c5a3ee7af3f7945f8d49198eb24
```

### 안에 무엇이 들어 있나

임베딩이라고 하면 대단한 것 같지만 **글월 파일 한 장**이다. 첫 두 줄을 그대로 열어 보면 이렇다.

```text
400000 50
the 0.418 0.24968 -0.41242 0.1217 0.34527 -0.044457 -0.49688 -0.17862 ...
```

**첫 줄은 낱말이 아니다.** 「낱말 40만 개, 차원 50」이라고 알려 주는 머리글이다. 둘째 줄부터가 진짜 — 낱말 하나와 그 뒤에 수 50개다. 이것이 40만 줄 이어진다.

[6.1절](01_not_a_number.md)이 말한 표가 이 모양으로 저장된 것이다. 40만 × 50짜리 표 하나.

!!! note "코드가 머리글을 소리 없이 건너뛴다"
    아래 `load_glove`에 `if len(p) != DIM + 1: continue` 한 줄이 있다. 첫 줄은 토막이 둘뿐이라 51이 아니므로 여기서 걸러진다.

    덤으로 **스탠퍼드 원본도 같은 코드로 읽힌다.** 원본 `glove.6B.50d.txt`에는 머리글이 없는데, 그러면 이 줄이 한 번도 걸리지 않을 뿐 나머지는 똑같이 돈다.

### 이름과 출처

`glove-wiki-gigaword-50`은 세 토막이다.

| 토막 | 뜻 |
|---|---|
| `wiki-gigaword` | 읽은 말뭉치 — Wikipedia 2014 + Gigaword 5, **60억 낱말** |
| `50` | 차원. 같은 말뭉치로 50·100·200·300 네 가지가 있다 |

**정본은 [스탠퍼드 GloVe 프로젝트](https://nlp.stanford.edu/projects/glove/)다.** 거기서는 네 차원을 한데 묶은 `glove.6B.zip`(822MB)으로 내놓는다. 이 책이 gensim-data의 거울을 쓰는 까닭은 하나뿐이다 — **필요한 한 개만 66MB로 받을 수 있다.** 받은 파일의 머리글이 `400000 50`으로, 논문이 밝힌 어휘 크기 40만과 맞는다.

**50차원을 고른 것은 가장 작기 때문이다.** 좋아서가 아니다. 논문의 Figure 2(a)를 보면 유추 과제 정확도가 차원을 늘릴수록 가파르게 오르다 200 언저리에서 누우므로, **50은 그 비탈의 아래쪽**이다. 이 쪽의 목적은 가장 좋은 벡터를 쓰는 것이 아니라 **남이 만든 것을 열어 보는 것**이라 작은 쪽을 골랐다.

!!! note "`torchtext`로는 안 된다"
    인터넷의 예제는 대개 `torchtext.vocab.GloVe(name='6B', dim=50)` 한 줄을 보여 준다. **그 길은 막혔다** — `torchtext`는 더 이상 유지되지 않는다.

    그래서 이 쪽은 표준 라이브러리 `gzip`으로 직접 읽는다. 줄을 쪼개 수로 바꾸는 것이 전부라 **새로 깔 것이 없고**, 형식을 알고 나면 다른 임베딩 파일도 같은 방식으로 열린다.

---

## 2. 코드

```python
"""6.5 가져다 쓰는 임베딩 — 남이 배운 것을 열어 본다.

앞의 네 쪽은 영화평 6,000편으로 벡터를 손수 만들었다. 이 쪽은 남이
「60억 낱말」로 만들어 둔 것을 가져와 같은 자로 재어 본다.

쓰는 것은 GloVe 50차원이다. 위키백과와 뉴스 말뭉치에서 동시 출현을 세어
만들었으니 [6.3절](03_counting.md)이 걸은 길의 큰 판이다.

재는 것 넷.
  (가) 어휘가 얼마나 넓은가 — 우리 어휘를 얼마나 덮는가
  (나) 이웃이 우리 것과 어떻게 다른가
  (다) 유추가 더 잘 되는가 — king - man + woman 을 드디어 물어본다
  (라) 그리고 「더 큰 것이 늘 나은 것은 아니다」 — good 과 bad 를 본다

읽을 것: data/glove-wiki-gigaword-50.gz (66MB). 받는 법과 확인하는 법은
이 쪽 1 절에 적어 두었다. 7.3 절이 쓰는 그 파일이다.

실행: 이 쪽의 코드를 파일로 갈무리해 저장소 뿌리에서 돌린다.
"""
import gzip
import re
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F

VOCAB, DIM = 8000, 50
ROOT = Path("./data/aclImdb")
GLOVE = Path("./data/glove-wiki-gigaword-50.gz")
TOKEN = re.compile(r"[a-z']+")
torch.set_num_threads(4)


# === 우리 어휘 (앞의 네 쪽과 같다) ==========================================
def load_tokens(n_docs=6000):
    out = []
    for name in ("pos", "neg"):
        for f in sorted((ROOT / "train" / name).glob("*.txt"))[: n_docs // 2]:
            out += TOKEN.findall(f.read_text(encoding="utf-8")
                                 .replace("<br />", " ").lower())
    return out


counts = Counter(load_tokens())
mine = [w for w, _ in counts.most_common(VOCAB)]
mine_set = set(mine)


# === GloVe 를 읽는다 =======================================================
def load_glove():
    """한 줄에 「낱말 수 수 … 수」가 들어 있는 글월 파일이다."""
    words, vecs = [], []
    with gzip.open(GLOVE, "rt", encoding="utf-8") as f:
        for line in f:
            p = line.rstrip().split(" ")
            if len(p) != DIM + 1:
                continue
            words.append(p[0])
            vecs.append([float(x) for x in p[1:]])
    return words, F.normalize(torch.tensor(vecs), dim=1)


gw, GV = load_glove()
gi = {w: i for i, w in enumerate(gw)}


def near(vec, drop, k=6):
    s = GV @ F.normalize(vec, dim=0)
    for w in drop:
        if w in gi:
            s[gi[w]] = -2
    return [gw[i] for i in s.topk(k).indices.tolist()]


sim = lambda a, b: float(GV[gi[a]] @ GV[gi[b]])


if __name__ == "__main__":
    # === (가) 얼마나 넓은가 ==============================================
    hit = sum(w in gi for w in mine)
    used = sum(counts[w] for w in mine if w in gi)
    tot = sum(counts[w] for w in mine)
    print(f"  GloVe 낱말 {len(gw):,}개, 차원 {DIM}")
    print(f"  우리 어휘 {VOCAB:,}개 가운데 {hit:,}개를 덮는다 "
          f"({100 * hit / VOCAB:.1f}%)")
    print(f"  쓰임으로 세면 {100 * used / tot:.1f}%")
    print("  덮지 못한 낱말 열 개:")
    miss = [w for w in mine if w not in gi]
    print("    " + ", ".join(miss[:10]))

    # === (나) 이웃 ======================================================
    print("\n  이웃 여섯 — 60억 낱말로 배운 것")
    for w in ("good", "terrible", "movie", "she", "funny", "bank", "star"):
        if w in gi:
            print(f"    {w:<10s} {', '.join(near(GV[gi[w]], (w,)))}")

    # === (다) 유추 ======================================================
    print("\n  유추 — 드디어 물어볼 수 있다")
    tri = [("man", "king", "woman"), ("man", "woman", "he"),
           ("good", "better", "bad"), ("paris", "france", "tokyo"),
           ("walk", "walked", "go")]
    for a, b, c in tri:
        if all(x in gi for x in (a, b, c)):
            v = GV[gi[b]] - GV[gi[a]] + GV[gi[c]]
            print(f"    {a:7s}: {b:9s} = {c:7s}: "
                  f"{', '.join(near(v, (a, b, c), 3))}")

    # === (라) 그래도 반대말은 가깝다 ======================================
    print("\n  닮음 — 60억 낱말을 보고도")
    for a, b in [("good", "great"), ("good", "bad"), ("good", "terrible"),
                 ("hot", "cold"), ("movie", "film"), ("movie", "kitchen")]:
        if a in gi and b in gi:
            print(f"    {a:8s} {b:10s} {sim(a, b):+.3f}")
```

**출력:**

```
  GloVe 낱말 400,000개, 차원 50
  우리 어휘 8,000개 가운데 7,837개를 덮는다 (98.0%)
  쓰임으로 세면 98.3%
  덮지 못한 낱말 열 개:
    it's, don't, i'm, doesn't, didn't, that's, can't, i've, he's, isn't

  이웃 여섯 — 60억 낱말로 배운 것
    good       better, really, always, sure, something, think
    terrible   horrible, awful, tragic, dreadful, tragedy, horrific
    movie      movies, film, films, comedy, hollywood, drama
    she        her, herself, he, him, mother, his
    funny      hilarious, amusing, silly, joke, fun, weird
    bank       banks, securities, banking, investment, exchange, financial
    star       stars, superstar, player, plays, legend, actor

  유추 — 드디어 물어볼 수 있다
    man    : king      = woman  : queen, throne, prince
    man    : woman     = he     : she, her, his
    good   : better    = bad    : worse, too, unfortunately
    paris  : france    = tokyo  : japan, korea, japanese
    walk   : walked    = go     : got, went, put

  닮음 — 60억 낱말을 보고도
    good     great      +0.798
    good     bad        +0.796
    good     terrible   +0.595
    hot      cold       +0.801
    movie    film       +0.931
    movie    kitchen    +0.363
```

---

## 3. 덮지 못하는 낱말은 전부 줄임말이다

| | |
|---|---|
| GloVe 낱말 | 400,000개 |
| 우리 어휘 8,000개 가운데 덮는 것 | **7,837개 (98.0%)** |
| 쓰임으로 세면 | 98.3% |

```
덮지 못한 낱말 열 개:
  it's, don't, i'm, doesn't, didn't, that's, can't, i've, he's, isn't
```

**열 개가 모두 줄임말이다.** 뜻이 어려운 낱말이라서가 아니라 **낱말을 쪼개는 방식이 달라서** 생긴 일이다. 우리는 `[a-z']+`로 잘라 `it's`를 한 낱말로 두는데, GloVe를 만든 쪽은 `it`과 `'s`로 나누었다.

이것이 가져다 쓸 때 가장 흔히 걸리는 자리다. **벡터가 없는 것이 아니라 이름이 안 맞는 것**이고, 쪼개는 방식을 맞추면 거의 다 찾아진다. 2%가 작아 보이지만 `it's`·`don't`는 아주 흔한 낱말이라 그냥 두면 그때마다 벡터가 없다.

---

## 4. 말뭉치가 뜻을 정한다

같은 낱말을 두 말뭉치에서 만들면 어떻게 갈리는지 본다. 아래 윗줄은 [6.2절](02_context.md)과 같은 Skip-gram 으로 영화평에서 만든 벡터이고, [다음 쪽](06_what_it_holds.md)이 그 벡터를 자세히 뜯어본다.

| | `bank`의 이웃 |
|---|---|
| 영화평 6,000편 (우리 것) | sent, robbers, responsibility, fighters, jail, government |
| 위키백과·뉴스 60억 낱말 (GloVe) | **banks, securities, banking, investment, exchange, financial** |

**같은 낱말인데 다른 것이 되었다.** 우리 쪽은 「은행 강도 영화」의 `bank`였고, GloVe 쪽은 금융기관이다.

`star`도 갈린다. 우리 것은 `classic, romero, superhero`로 연예인 쪽만 보았는데, GloVe는 `stars, superstar, player, plays, legend, actor`로 — 역시 연예인 쪽이되 **운동선수까지** 끌어온다. 뉴스에서 `star player`가 흔하기 때문이다.

**어느 쪽이 맞는가. 둘 다 맞고, 둘 다 반쪽이다.** 벡터 하나에 뜻 하나이므로 말뭉치에서 흔한 쪽이 이기고, 흔한 쪽은 말뭉치가 정한다. **자료를 키워서 고칠 수 있는 것이 아니다** — 줄이 하나뿐인 한 어느 뜻이든 하나는 버려진다.

---

## 5. 유추는 확실히 좋아진다

| 물음 | 답 |
|---|---|
| man : king = woman : ? | **queen**, throne, prince |
| man : woman = he : ? | **she**, her, his |
| good : better = bad : ? | **worse**, too, unfortunately |
| paris : france = tokyo : ? | **japan**, korea, japanese |
| walk : walked = go : ? | got, **went**, put |

**다섯 가운데 넷이 1등으로 맞는다.** [다음 쪽](06_what_it_holds.md)이 우리 벡터에 같은 것을 물으면 다섯 가운데 셋이고, `king`–`queen` 같은 것은 어휘에 없어 물어보지도 못한다.

눈여겨볼 것은 **종류가 다양해졌다**는 점이다. 성별 바꾸기뿐 아니라 **나라 수도**(paris→france), **동사 시제**(walk→walked)까지 같은 벡터 셈으로 된다. 60억 낱말에서는 그런 바꿈들이 저마다 일정한 방향으로 자리 잡을 만큼 자료가 많다.

!!! warning "그래도 과하게 읽지 말 것"
    `walk : walked = go :` 에서 1등이 **`got`**이고 `went`는 2등이다. 그리고 `good : better = bad :` 의 2·3등은 `too`와 `unfortunately`로 유추와 상관없다.

    벡터를 더하고 빼는 것이 **뜻의 셈이라는 보장은 여기서도 없다.** 되는 쌍이 눈에 띌 뿐이다. [GloVe 프로젝트](https://nlp.stanford.edu/projects/glove/)의 원문도 유추를 **평가 과제의 하나**로 둘 뿐 의미 등식이라 주장하지 않는다.

---

## 6. 고쳐지지 않는 것

| | 우리 벡터 (영화평 6천 편) | GloVe (60억 낱말) |
|---|---|---|
| good — great | +0.748 | +0.798 |
| good — **bad** | +0.594 | **+0.796** |
| good — terrible | +0.578 | +0.595 |
| hot — **cold** | — | **+0.801** |
| movie — film | +0.801 | +0.931 |
| movie — kitchen | +0.348 | +0.363 |

**둘째 줄을 보라.** 우리 벡터에서는 `good`–`bad`(0.594)가 `good`–`great`(0.748)보다 0.154 낮았다. GloVe에서는 **0.796 대 0.798로 0.002 차이**다. 사실상 같다.

`hot`–`cold`가 0.801인 것도 같은 이야기다. 서로 반대인 두 낱말이 `movie`–`film`(0.931) 다음으로 가깝다.

**자료를 천 배 넣어서 좋아진 것과 좋아지지 않은 것이 갈린다.**

| 좋아진 것 | 좋아지지 않은 것 |
|---|---|
| 어휘 넓이 (8천 → 40만) | 반대말이 가까운 것 |
| 유추 (셋 → 넷, 종류도 늘었다) | 한 낱말에 벡터 하나 |

오른쪽 둘은 **자료의 문제가 아니라 방식의 문제**다. 「둘레가 같으면 가깝다」로 배우는 한, 같은 자리에 들어가는 `good`과 `bad`는 가까울 수밖에 없다. 그리고 벡터가 낱말마다 하나인 한, 문장을 보지 않는다.

이것이 [7.3절](../imdb/03_frozen.md)이 측정으로 확인하는 바이기도 하다. 거기서 IMDB 감정 분류에 GloVe를 얹으면 **IMDB에서 손수 배운 임베딩에 진다.** 60억 낱말이 2만 5천 편에 지는 것이고, 까닭이 바로 이 표의 둘째 줄이다 — 그 과제가 필요로 하는 것이 정확히 `good`과 `bad`를 **가르는** 일이기 때문이다.

---

## 7. 실제로 얹으려면

위 코드는 GloVe를 **읽어 보기만** 한다. 모델에 얹으려면 더 따질 것이 있다.

사전 학습 벡터를 사용할 때는 현재 어휘의 각 인덱스에 맞춰 행을 복사해야 한다. 벡터 파일의 행 순서와 현재 어휘의 인덱스가 같다고 가정하면 잘못된 단어에 벡터가 연결된다. 임베딩 차원, 토큰화와 대소문자 처리도 맞춰야 한다.

어휘 밖(OOV) 단어에는 무작위 벡터나 공통 미등록 단어 벡터를 줄 수 있다. 모든 사전 학습 벡터의 평균을 쓰면 처음에는 OOV 단어들의 표현이 같아진다. 이를 고정하면 그 단어들을 구분할 수 없다. FastText의 하위 단어 벡터를 쓰는 방법은 해당 모델과 하위 단어 정보가 필요하며, GloVe 파일만 읽어서 같은 처리를 할 수는 없다.

웨이트를 고정하면 갱신할 파라메터를 줄이고 초기 표현을 유지할 수 있지만, 나머지 모델의 과적합까지 막지는 않는다. 미세 조정은 과제에 적응할 여지를 주는 대신 러닝레이트와 데이터 규모에 따라 표현이 훼손될 수 있다. 고정 학습 후 일부 또는 전체를 해제하는 방식도 검증 성능으로 선택한다. 옵티마이저를 만들 때 고정 파라메터를 제외했다면 해제 후 갱신 대상에 추가해야 한다.

GloVe는 단어 동시 출현 횟수의 로그를 내적과 바이어스로 설명하는 가중 최소제곱 목적 함수를 사용한다. 단순한 행렬 분해나 모든 단어에 성립하는 의미 연산으로 이해해서는 안 된다. 벡터 유추의 성공 여부는 단어와 학습 데이터에 따라 다르며, 사전 학습 벡터에는 데이터의 바이어스도 반영될 수 있다. 원래 벡터와 설명은 [GloVe 프로젝트](https://nlp.stanford.edu/projects/glove/)에서 확인할 수 있다.

실제로 얹어서 **재어 보는 것**은 [7.3절 「빌려 온 벡터」](../imdb/03_frozen.md)가 한다. 거기서는 IMDB 사다리 전체를 GloVe로 다시 오르며 얼린 것과 미세 조정한 것을 견주는데, **부호가 뒤집히는 자리**가 나온다.

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span> 2절은 덮지 못한 낱말이 전부 줄임말이라고 했다. 그 2%를 되찾는 가장 싼 방법을 적으라.

</div>

??? success "풀이"

    **쪼개는 방식을 GloVe에 맞추면 된다.** `it's`를 만나면 `it`과 `'s`로 나누어 각각의 벡터를 찾고, 둘을 더하거나 평균 낸다. GloVe에 `'s`도 들어 있다.

    더 싼 길도 있다. 어차피 어포스트로피 뒤가 짧으므로 **앞부분만 쓰는 것**이다 — `don't` → `do`. 뜻이 조금 상하지만(부정이 사라진다) 벡터가 아예 없는 것보다는 낫다. **감정 분류에서는 그 「조금」이 치명적**일 수 있으니 과제를 보고 정해야 한다.

    가장 바른 길은 **처음부터 같은 쪼개기를 쓰는 것**이다. 빌려 올 벡터가 정해져 있다면 그 쪽의 쪼개기에 맞추어 어휘를 만든다.

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff medium" title="중간"></span> 5절은 GloVe에서 `good`–`bad`가 0.796으로 `good`–`great`의 0.798과 사실상 같다고 했다. 이것이 **GloVe의 흠인지 분포 가설의 성질인지** 가리는 실험을 설계하라.

</div>

??? success "풀이"

    **흠이라면 다른 방식에서는 덜해야 한다.** 그러니 같은 말뭉치를 다른 방식으로 배운 벡터와 견주면 된다.

    같은 위키백과·뉴스 자료로 만든 word2vec 벡터(공개되어 있다)를 받아 같은 쌍들의 닮음을 잰다. 거기서도 `good`–`bad`가 `good`–`great`에 맞먹으면 **방식이 아니라 분포 가설 자체의 성질**이다.

    이 쪽이 이미 반쯤 답을 갖고 있다. [6.2절](02_context.md)의 Skip-gram(배우는 길)과 [6.3절](03_counting.md)의 PPMI+SVD(세는 길)는 **서로 다른 방식**인데 둘 다 `good`과 `bad`를 가깝게 둔다. [6.4절](04_glove.md)이 같은 말뭉치로 돌린 GloVe도 마찬가지였다. **세 방식이 같은 성질을 보이면 방식 탓이 아니다.**

    결정적으로 가리려면 **분포 가설을 쓰지 않는** 벡터와 견주어야 한다 — 예를 들어 사전의 뜻풀이나 WordNet 같은 사람이 만든 관계망에서 만든 벡터다. 거기서 `good`과 `bad`가 멀면 답이 분명해진다.

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span> 3절은 같은 `bank`가 말뭉치에 따라 다른 것이 된다고 했다. 두 말뭉치의 벡터를 **합쳐서** 두 뜻을 모두 담을 수 있는가. 설계하고, 왜 어려운지 적으라.

</div>

??? success "풀이"

    **곧바로 더할 수 없다.** 두 벡터 공간은 서로 다른 학습에서 나왔으므로 축이 전혀 다르다. 우리 벡터의 3번 축과 GloVe의 3번 축 사이에 아무 관계가 없다. 더하면 뜻 없는 수가 나온다.

    **맞춰 붙이는 길은 있다.** 두 공간에 **함께 있는 낱말**(여기서는 7,837개)을 짝지어, 한쪽을 다른 쪽에 가장 잘 포개는 회전·크기 변환을 찾는다. 두 점구름을 겹치는 고전적인 문제이고 닫힌 식으로 풀린다. 그러고 나서 이어 붙이거나 평균 내면 된다.

    **그래도 두 뜻이 담기지는 않는다.** 맞춰 붙여 놓으면 우리 쪽 `bank`(강둑 기운)와 GloVe 쪽 `bank`(은행)가 **한 자리에 포개지고**, 평균을 내면 둘 사이 어딘가의 한 점이 될 뿐이다. 줄이 여전히 하나이기 때문이다.

    **뜻을 나누려면 줄을 나누어야 하는데**, 그러면 [다음 쪽 연습문제 3](06_what_it_holds.md)의 문제들이 그대로 돌아온다 — 몇 개로 나눌지, 쓸 때 어느 것을 고를지. [8장](../transformer/index.md)이 그 물음을 없애는 길이다.
