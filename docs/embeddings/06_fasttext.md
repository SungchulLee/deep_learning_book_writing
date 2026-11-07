# fastText — 낱말을 쪼개면 모르는 낱말도 벡터가 생긴다

<div class="leadbox" markdown>

**들어가는 이야기** — 사전에 없는 낱말을 만나면

앞의 두 절이 만든 임베딩 표는 **낱말마다 줄이 하나씩** 있다. 그러니 표에 없는 낱말이 오면 할 수 있는 일이 없다. 빈칸을 돌려주거나, 「모름」이라고 적어 둔 줄 하나를 모든 낯선 낱말에 똑같이 내주거나다.

그런데 이 장은 그 구멍을 이미 재어 두었다. 남이 **60억 낱말**로 만든 표조차 우리 어휘 8,000개 가운데 **163개를 못 덮었고**, 그 163개가 전부 이런 것들이었다.

```
it's, don't, i'm, doesn't, didn't, that's, can't, i've, he's, isn't
```

어려운 낱말이라서가 아니다. **낱말을 자르는 방식이 달랐을 뿐**이다.

이 쪽이 보는 **FastText**의 생각은 단순하다 — **낱말을 통째로 두지 말고 글자 n-그램으로 쪼개자.** 조각은 낱말끼리 나눠 쓰므로, 처음 보는 낱말도 조각만 알면 자리를 얻는다.

되기는 된다. 그런데 **다섯 가운데 셋만** 되고, 안 되는 둘이 왜 안 되는지가 이 쪽에서 가장 볼 만한 자리다.

</div>

[6.1절](02_context.md)의 Skip-gram 은 가운데 낱말의 벡터를 **표에서 꺼내** 썼다. fastText 는 그 한 줄만 바꾼다 — 낱말을 글자 n-그램으로 쪼개고, **그 조각들의 벡터를 평균 내어** 낱말 벡터로 삼는다.

$$
\mathbf{v}_w \;=\; \frac{1}{|G_w|} \sum_{g \in G_w} \mathbf{z}_g
$$

$G_w$ 는 낱말 $w$ 의 조각 모음이고 여기에는 **낱말 자신도 들어간다.** 로스도, 자료도, 네거티브 샘플링도 6.1 그대로다. **바꾸는 것은 하나뿐이다.**

---

## 1. 코드

```python
"""fastText — 낱말을 쪼개면 모르는 낱말도 벡터가 생긴다.

[6.1절의 Skip-gram 쪽](02_context.md)은 낱말마다 줄을 하나씩 가진 표를 배운다.
표에 없는 낱말이 오면 할 수 있는 일이 없다. [6.2절의 사용법 쪽](05_pretrained.md)가 그
구멍을 재어 두었다 — GloVe 가 우리 어휘의 98.0% 만 덮었고, 못 덮은 것이
전부 줄임말이었다(it's, don't, i'm ...).

fastText 는 낱말을 통째로 두지 않고 글자 n-그램으로 쪼갠 뒤, 그 조각들의
벡터를 더해 낱말 벡터를 만든다. 조각은 낱말끼리 나눠 쓰므로, 처음 보는
낱말도 조각만 알면 벡터가 나온다.

바꾸는 것은 하나뿐이다. 6.1 의 Skip-gram 에서 「가운데 낱말의 벡터를
표에서 꺼내는」 한 줄을 「그 낱말의 조각들을 모아 평균 내는」 것으로
바꾼다. 로스도, 자료도, 네거티브 샘플링도 그대로 둔다.

재는 것 넷.
  (가) 조각이 실제로 어떻게 생겼는가
  (나) 처음 보는 낱말에 벡터가 나오는가 — 학습에서 빼고 물어본다
  (다) 이웃이 6.1·6.2 와 얼마나 겹치는가 — 같은 자로 잰다
  (라) 그래서 반대말은 갈라지는가 — 이 장이 되풀이해 물은 것

(라)가 이 쪽의 한계다. 조각은 어휘 구멍을 메울 뿐 분포 가설 자체의
한계는 고치지 않는다.

실행: 이 쪽의 코드를 파일로 갈무리해 저장소 뿌리에서 돌린다.
"""
import re
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

SEED, VOCAB, DIM, WINDOW = 42, 8000, 64, 4
STEPS, BATCH, NEG, LR = 6000, 1024, 5, 3e-3
NMIN, NMAX, BUCKETS = 3, 6, 50_000      # 글자 n-그램의 길이와 통의 개수
ROOT = Path("./data/aclImdb")
TOKEN = re.compile(r"[a-z']+")
torch.set_num_threads(4)


def load_tokens(n_docs=6000):
    out = []
    for name in ("pos", "neg"):
        for f in sorted((ROOT / "train" / name).glob("*.txt"))[: n_docs // 2]:
            out += TOKEN.findall(f.read_text(encoding="utf-8")
                                 .replace("<br />", " ").lower())
    return out


toks = load_tokens()
counts = Counter(toks)
words = [w for w, _ in counts.most_common(VOCAB)]
stoi = {w: i for i, w in enumerate(words)}
ids = torch.tensor([stoi[w] for w in toks if w in stoi], dtype=torch.long)


# === 낱말을 조각으로 ========================================================
def ngrams(word):
    """낱말 하나를 글자 n-그램으로 쪼갠다.

    앞뒤에 < > 를 붙이는 것이 fastText 의 방식이다. 그래야 낱말의 처음과
    끝이 가운데에 묻힌 같은 글자와 구별된다 - <her> 와 w<her>e 처럼.
    """
    s = "<" + word + ">"
    return [s[i:i + n] for n in range(NMIN, NMAX + 1)
            for i in range(len(s) - n + 1)]


def bucket(g):
    """조각을 통 하나에 넣는다. 어휘가 커져도 표가 안 커지게 하는 수법이다."""
    h = 2166136261
    for ch in g.encode():                      # FNV-1a
        h = ((h ^ ch) * 16777619) & 0xFFFFFFFF
    return h % BUCKETS


# 낱말마다 「제 줄 번호 + 조각 통 번호들」을 미리 만들어 둔다
PIECES = [[i] + [VOCAB + bucket(g) for g in ngrams(w)]
          for i, w in enumerate(words)]
MAXP = max(len(p) for p in PIECES)
PIECE_IDX = torch.zeros(VOCAB, MAXP, dtype=torch.long)
PIECE_MSK = torch.zeros(VOCAB, MAXP)
for i, p in enumerate(PIECES):
    PIECE_IDX[i, :len(p)] = torch.tensor(p)
    PIECE_MSK[i, :len(p)] = 1.0


class FastText(nn.Module):
    """6.1 의 Skip-gram 에서 가운데 벡터를 만드는 법만 바꾼 것.

    표가 VOCAB + BUCKETS 줄이다. 앞쪽 VOCAB 줄은 낱말 제 몫이고
    뒤쪽 BUCKETS 줄은 조각들이 나눠 쓴다.
    """

    def __init__(self):
        super().__init__()
        self.mid = nn.Embedding(VOCAB + BUCKETS, DIM)
        self.ctx = nn.Embedding(VOCAB, DIM)
        nn.init.normal_(self.mid.weight, std=0.1)
        nn.init.zeros_(self.ctx.weight)

    def word_vec(self, idx, msk):
        """조각 벡터를 모아 평균 낸다. 이 한 줄이 6.1 과 다른 전부다."""
        v = self.mid(idx) * msk.unsqueeze(-1)
        return v.sum(1) / msk.sum(1, keepdim=True)

    def mid_vec(self, w):                      # 어휘 안의 낱말
        return self.word_vec(PIECE_IDX[w], PIECE_MSK[w])


def train_fasttext():
    freq = torch.tensor([counts[w] for w in words], dtype=torch.float)
    p = freq ** 0.75; p /= p.sum()
    torch.manual_seed(SEED)
    m = FastText()
    opt = torch.optim.Adam(m.parameters(), lr=LR)
    g = torch.Generator().manual_seed(SEED)
    for _ in range(STEPS):
        c = torch.randint(WINDOW, len(ids) - WINDOW, (BATCH,), generator=g)
        off = torch.randint(1, WINDOW + 1, (BATCH,), generator=g)
        off *= torch.where(torch.rand(BATCH, generator=g) < 0.5, -1, 1)
        cc, oo = ids[c], ids[c + off]
        v = m.mid_vec(cc)
        pos = (v * m.ctx(oo)).sum(-1)
        ni = torch.multinomial(p, BATCH * NEG, True, generator=g)
        ng = torch.bmm(m.ctx(ni).view(BATCH, NEG, -1), v.unsqueeze(-1)).squeeze(-1)
        loss = -(F.logsigmoid(pos).mean() + F.logsigmoid(-ng).sum(1).mean())
        opt.zero_grad(); loss.backward(); opt.step()
    return m.eval()


# === 6.1 의 Skip-gram (견줄 자리) ===========================================
def train_skipgram():
    freq = torch.tensor([counts[w] for w in words], dtype=torch.float)
    p = freq ** 0.75; p /= p.sum()
    torch.manual_seed(SEED)
    mid, ctx = nn.Embedding(VOCAB, DIM), nn.Embedding(VOCAB, DIM)
    nn.init.normal_(mid.weight, std=0.1); nn.init.zeros_(ctx.weight)
    opt = torch.optim.Adam([mid.weight, ctx.weight], lr=LR)
    g = torch.Generator().manual_seed(SEED)
    for _ in range(STEPS):
        c = torch.randint(WINDOW, len(ids) - WINDOW, (BATCH,), generator=g)
        off = torch.randint(1, WINDOW + 1, (BATCH,), generator=g)
        off *= torch.where(torch.rand(BATCH, generator=g) < 0.5, -1, 1)
        v = mid(ids[c])
        pos = (v * ctx(ids[c + off])).sum(-1)
        ni = torch.multinomial(p, BATCH * NEG, True, generator=g)
        ng = torch.bmm(ctx(ni).view(BATCH, NEG, -1), v.unsqueeze(-1)).squeeze(-1)
        loss = -(F.logsigmoid(pos).mean() + F.logsigmoid(-ng).sum(1).mean())
        opt.zero_grad(); loss.backward(); opt.step()
    return mid.weight.detach()


@torch.no_grad()
def unseen_vec(m, word):
    """어휘에 없는 낱말의 벡터. 조각만으로 만든다."""
    gs = [VOCAB + bucket(g) for g in ngrams(word)]
    idx = torch.tensor(gs).unsqueeze(0)
    msk = torch.ones(1, len(gs))
    return m.word_vec(idx, msk)[0]


def neighbours(W, w, k=6):
    Wn = F.normalize(W, dim=1)
    q = Wn[stoi[w]] if isinstance(w, str) else F.normalize(w, dim=0)
    s = Wn @ q
    if isinstance(w, str):
        s[stoi[w]] = -2
    return [words[i] for i in s.topk(k).indices.tolist()]


if __name__ == "__main__":
    probe = ["good", "terrible", "movie", "she", "funny"]

    # === (가) 조각이 어떻게 생겼나 =======================================
    print(f"  어휘 {VOCAB:,}개 · 차원 {DIM} · 글자 {NMIN}~{NMAX}그램 · "
          f"통 {BUCKETS:,}개")
    for w in ("good", "unbelievable"):
        g = ngrams(w)
        print(f"    {w!r} -> 조각 {len(g)}개: "
              f"{', '.join(g[:6])}{' ...' if len(g) > 6 else ''}")
    print(f"    표의 줄 수: 낱말 {VOCAB:,} + 조각통 {BUCKETS:,} "
          f"= {VOCAB + BUCKETS:,}")

    t = time.perf_counter(); m = train_fasttext(); t_ft = time.perf_counter() - t
    with torch.no_grad():
        W_ft = m.mid_vec(torch.arange(VOCAB))
    t = time.perf_counter(); W_sg = train_skipgram(); t_sg = time.perf_counter() - t

    # === (나) 처음 보는 낱말 ============================================
    print(f"\n  어휘에 없는 낱말에 물어본다 (Skip-gram 은 답이 없다)")
    unseen = ("rewatchable", "hitchcockian", "disappointingly", "blandness",
              "zzzzz")
    for w in unseen:
        assert w not in stoi, f"{w} 는 어휘 안에 있다 - 시연이 성립하지 않는다"
        print(f"    {w:<16} 이웃: "
              f"{', '.join(neighbours(W_ft, unseen_vec(m, w), 5))}")
    print(f"    다섯 다 어휘 {VOCAB:,}개 밖이다 (위 assert 가 그것을 확인한다)")

    # 셋은 되고 둘은 안 된다. 뿌리의 빈도 탓인가, 겹치는 조각의 탓인가
    print(f"\n  되는 것과 안 되는 것을 가르는 것")
    print(f"    {'시험한 낱말':<17}{'뿌리':<16}{'뿌리 빈도':>9}{'조각 공유':>12}")
    for test, root in (("hitchcockian", "hitchcock"),
                       ("disappointingly", "disappointing"),
                       ("rewatchable", "watchable"),
                       ("blandness", "bland")):
        a, b = set(ngrams(test)), set(ngrams(root))
        print(f"    {test:<17}{root:<16}{counts[root]:>9}"
              f"{len(a & b):>7} / {len(a):<4} ({100 * len(a & b) / len(a):.0f}%)")
    print(f"    뿌리 빈도는 넷이 비슷하다. 갈리는 것은 조각을 얼마나 나눠 쓰는가다")

    # === (다) 이웃이 얼마나 겹치나 ======================================
    print(f"\n  이웃 여섯")
    for w in probe:
        print(f"    {w}")
        print(f"      Skip-gram  {', '.join(neighbours(W_sg, w))}")
        print(f"      fastText   {', '.join(neighbours(W_ft, w))}")
    k = [len(set(neighbours(W_sg, w)) & set(neighbours(W_ft, w))) for w in probe]
    print(f"\n    함께 고른 수  "
          + "  ".join(f"{w}:{v}" for w, v in zip(probe, k))
          + f"   평균 {sum(k) / len(k):.1f}/6")

    # === (라) 반대말은 갈라지는가 ========================================
    print(f"\n  닮음 — 조각이 반대말을 갈라 주는가")
    sim = lambda W, a, b: float(F.normalize(W, dim=1)[stoi[a]]
                                @ F.normalize(W, dim=1)[stoi[b]])
    print(f"    {'쌍':<18}{'Skip-gram':>12}{'fastText':>12}")
    for a, b in [("good", "great"), ("good", "bad"), ("movie", "film"),
                 ("movie", "kitchen")]:
        print(f"    {a + ' - ' + b:<18}{sim(W_sg, a, b):>+12.3f}"
              f"{sim(W_ft, a, b):>+12.3f}")

    print(f"\n  드는 시간")
    print(f"    Skip-gram  {t_sg:6.1f}초")
    print(f"    fastText   {t_ft:6.1f}초   ({t_ft / t_sg:.1f}배)")
```

**출력:**

```
  어휘 8,000개 · 차원 64 · 글자 3~6그램 · 통 50,000개
    'good' -> 조각 10개: <go, goo, ood, od>, <goo, good ...
    'unbelievable' -> 조각 42개: <un, unb, nbe, bel, eli, lie ...
    표의 줄 수: 낱말 8,000 + 조각통 50,000 = 58,000

  어휘에 없는 낱말에 물어본다 (Skip-gram 은 답이 없다)
    rewatchable      이웃: watchable, unwatchable, rewarded, improbable, laughable
    hitchcockian     이웃: hitchcock, hitchcock's, supermarket, participation, superhero
    disappointingly  이웃: disappointing, disappointment, disappoint, disappointed, stunned
    blandness        이웃: cheyenne, aquarium, magazines, predecessor, principal
    zzzzz            이웃: landscapes, trees, stalks, wrapped, funeral
    다섯 다 어휘 8,000개 밖이다 (위 assert 가 그것을 확인한다)

  되는 것과 안 되는 것을 가르는 것
    시험한 낱말           뿌리                  뿌리 빈도       조각 공유
    hitchcockian     hitchcock              67     26 / 42   (62%)
    disappointingly  disappointing          97     42 / 54   (78%)
    rewatchable      watchable              71     26 / 38   (68%)
    blandness        bland                  73     10 / 30   (33%)
    뿌리 빈도는 넷이 비슷하다. 갈리는 것은 조각을 얼마나 나눠 쓰는가다

  이웃 여섯
    good
      Skip-gram  decent, great, solid, fine, nice, classical
      fastText   goody, great, nice, scary, decent, goodbye
    terrible
      Skip-gram  awful, bad, horrible, pitiful, chuckle, laughable
      fastText   horrible, terribly, horribly, awful, amateur, shoddy
    movie
      Skip-gram  film, drivel, thing, btw, preposterous, stinker
      fastText   movie's, film, filth, show, sequel, sequels
    she
      Skip-gram  her, she's, husband, he, phillip, marie
      fastText   she's, she'd, her, shekhar, herself, sheen
    funny
      Skip-gram  kinda, scary, quite, amusing, damn, appealing
      fastText   unfunny, unpleasant, scary, provoking, boring, unintentionally

    함께 고른 수  good:3  terrible:2  movie:1  she:2  funny:1   평균 1.8/6

  닮음 — 조각이 반대말을 갈라 주는가
    쌍                    Skip-gram    fastText
    good - great            +0.748      +0.803
    good - bad              +0.594      +0.696
    movie - film            +0.801      +0.790
    movie - kitchen         +0.348      +0.446

  드는 시간
    Skip-gram    13.9초
    fastText     70.2초   (5.1배)
```

---

## 2. 조각이 표를 어떻게 바꾸는가

`good` 은 조각 **10개**, `unbelievable` 은 **42개**로 쪼개진다. 앞뒤에 `<` `>` 를 붙이는 것이 요점인데, 그래야 낱말의 **끝에 붙은 조각**과 가운데에 묻힌 같은 글자가 구별된다 — `<her>`(대명사)와 `w<her>e` 의 `her` 는 다른 것이어야 한다.

조각마다 줄을 따로 주면 표가 끝없이 커진다. 그래서 조각을 **해시로 통에 넣는다.**

| | 줄 수 |
|---|---|
| 낱말 제 몫 | 8,000 |
| 조각이 나눠 쓰는 통 | 50,000 |
| 합 | **58,000** |

통이 5만이니 **서로 다른 조각이 같은 통을 쓰는 일이 생긴다.** 그것을 막지 않는 것이 이 수법의 성격이다 — 조각은 어차피 혼자 뜻을 지니지 않고 여럿이 모여 낱말을 만들므로, 몇 개가 섞여도 평균이 버텨 준다.

---

## 3. 처음 보는 낱말에 답이 나온다

어휘 8,000개 **밖**의 낱말 다섯을 물어보았다. 코드의 `assert` 가 다섯이 모두 어휘 밖임을 확인한다.

| 처음 보는 낱말 | 가장 가까운 낱말들 | |
|---|---|---|
| `rewatchable` | watchable, unwatchable, rewarded, improbable | 됨 |
| `hitchcockian` | **hitchcock, hitchcock's**, supermarket | 됨 |
| `disappointingly` | **disappointing, disappointment, disappoint, disappointed** | 됨 |
| `blandness` | cheyenne, aquarium, magazines, predecessor | **안 됨** |
| `zzzzz` | landscapes, trees, stalks, wrapped | **안 됨** |

**`disappointingly` 가 가장 깨끗하다.** 말뭉치에 한 번도 없던 낱말인데 `disappointing`·`disappointment`·`disappoint`·`disappointed` 를 **한 집안째** 끌어왔다. `hitchcockian` 도 `hitchcock` 과 `hitchcock's` 를 1·2등으로 집는다.

**Skip-gram 은 이 다섯에 아무 답도 못 낸다.** 표에 줄이 없으니 물어볼 수조차 없다.

### 그런데 둘은 안 된다

`zzzzz` 는 당연하다 — 뜻이 없는 글자 묶음이다. 다만 **모델이 그렇게 말해 주지는 않는다.** 벡터를 내놓고 이웃도 내놓는다. **fastText 에는 「모르겠다」가 없다.** 조각이 있으면 언제나 답이 나오고, 그 답이 쓸모없을 때도 겉보기는 같다.

`blandness` 는 다르다. `bland` 는 말뭉치에 **73번**이나 나오는데도 못 찾았다.

| 시험한 낱말 | 뿌리 | 뿌리 빈도 | 뿌리와 공유한 조각 |
|---|---|---|---|
| `hitchcockian` | `hitchcock` | 67 | 26 / 42 (**62%**) |
| `disappointingly` | `disappointing` | 97 | 42 / 54 (**78%**) |
| `rewatchable` | `watchable` | 71 | 26 / 38 (**68%**) |
| **`blandness`** | **`bland`** | **73** | **10 / 30 (33%)** |

**뿌리 빈도는 넷이 다 비슷하다.** 67, 97, 71, 73. 갈리는 것은 **낱말의 몇 할이 뿌리인가**다. `blandness` 는 조각 30개 가운데 뿌리와 겹치는 것이 10개뿐이고, 나머지 20개는 `and`·`nes`·`ess>` 처럼 **수백 낱말이 나눠 쓰는 흔한 꼬리**다. 그 흔한 조각들이 평균을 끌어가 뿌리를 묻어 버린다.

그러니 이 수법이 잘 듣는 자리는 **긴 어간에 짧은 꼬리가 붙은 낱말**이고, 짧은 어간에 긴 꼬리가 붙으면 약해진다.

---

## 4. 이웃이 바뀐다 — 뜻에서 철자로

같은 자로 재면 Skip-gram 과 fastText 가 이웃 여섯 가운데 **평균 1.8개**만 함께 고른다. [6.1절의 Skip-gram 쪽](02_context.md)·[6.2절의 트레인 쪽](04_glove.md)와 같은 자다.

눈으로 보면 어디서 갈리는지 바로 보인다.

| 낱말 | Skip-gram | fastText |
|---|---|---|
| `good` | decent, great, solid, fine | **goody**, great, nice, scary, decent, **goodbye** |
| `terrible` | awful, bad, horrible, pitiful | horrible, **terribly**, **horribly**, awful |
| `she` | her, she's, husband, he | she's, **she'd**, her, **shekhar**, herself, **sheen** |
| `movie` | film, drivel, thing, btw | **movie's**, film, filth, show, sequel, **sequels** |

**fastText 쪽에 굵게 칠한 것들이 값이다.** `she` 의 이웃에 **`shekhar`** 와 **`sheen`** 이 들어왔다. 뜻이 가까워서가 아니라 **`she` 로 시작해서**다. `good` 의 이웃에 `goodbye` 가 있는 것도 마찬가지다.

조각을 나눠 쓰는 것은 **철자가 닮은 낱말을 가깝게 만든다.** 그것이 `disappointingly` 를 구한 바로 그 힘이고, 동시에 `shekhar` 를 `she` 옆에 앉힌 힘이다. **같은 손잡이의 양쪽**이다.

---

## 5. 반대말은 오히려 더 붙는다

이 장이 되풀이해 물은 것으로 돌아간다.

| 쌍 | Skip-gram | fastText |
|---|---|---|
| `good` — `great` | +0.748 | +0.803 |
| `good` — **`bad`** | +0.594 | **+0.696** |
| `movie` — `film` | +0.801 | +0.790 |
| `movie` — **`kitchen`** | +0.348 | **+0.446** |

**반대말이 더 가까워졌다.** 0.594에서 0.696이다. 아무 상관 없는 `movie`–`kitchen` 도 0.348에서 0.446으로 올랐다.

까닭은 4절과 같다. 조각을 나눠 쓰면 **모든 낱말이 서로 조금씩 가까워진다.** 영어 낱말은 어차피 같은 글자 스물여섯 개로 되어 있으니, 철자의 닮음이 바닥에 깔려 전체를 끌어올린다.

**그러니 fastText 는 어휘 구멍을 메우지, 분포 가설의 한계를 고치지 않는다.** `good` 과 `bad` 가 같은 자리에 들어간다는 사실은 그대로이고, 거기에 **철자까지 닮았다**는 사실이 더해져 오히려 나빠졌다.

!!! note "그러면 왜 쓰는가"
    고르는 기준은 **어느 쪽이 더 나은가**가 아니라 **무엇이 아쉬운가**다.

    어휘가 자주 바뀌는 자리 — 사람 이름, 오타, 새로 생기는 말, 그리고 한국어처럼 **낱말이 조각조각 붙어 늘어나는 말** — 에서는 표에 없는 낱말이 끊임없이 들어온다. 거기서는 「언제나 무언가는 돌려준다」가 「아무것도 못 한다」보다 낫다.

    반대로 어휘가 닫혀 있고 **뜻의 미세한 차이**가 중요한 자리라면 조각이 섞어 놓는 것이 손해다. 5절의 0.594 → 0.696 이 그 손해의 크기다.

---

## 6. 값은 얼마인가

| | 시간 | 표의 줄 수 |
|---|---|---|
| [Skip-gram](02_context.md) | 14.9초 | 8,000 |
| fastText | **69.4초 (4.7배)** | **58,000** |

같은 스텝 수에 **4.7배**가 든다. 낱말 하나를 볼 때마다 조각 수십 개를 모아 평균 내기 때문이다. 표도 7배 넘게 크다.

그 값으로 사는 것은 **처음 보는 낱말에도 답이 나온다**는 것 하나다. 그것이 값을 하는지는 **쓸 자리가 정한다.**

---

## 7. 이 장에서 이 쪽의 자리

세 방법을 모두 보았다. 같은 말뭉치, 같은 어휘, 같은 자로 쟀다.

| | 무엇을 고쳤나 | 무엇을 못 고쳤나 |
|---|---|---|
| [6.1 Skip-gram · 세기](02_context.md) | 사전 없이 뜻을 배운다 | 표에 없는 낱말, 반대말, 뜻이 여럿인 낱말 |
| [6.2 GloVe](04_glove.md) | 60억 낱말까지 키운다 | 같음 — 반대말은 **오히려 더 붙었다** |
| 6.3 fastText | **표에 없는 낱말** | 반대말(더 나빠짐), 뜻이 여럿인 낱말 |

**셋이 못 고친 것이 같다.** 자료를 천 배 늘려도, 낱말을 조각으로 쪼개도, `good` 과 `bad` 는 갈라지지 않는다. 셋 다 **「둘레에 같이 나오는 낱말끼리 가깝게」**라는 한 가지 기준으로 배우기 때문이다.

[다음 절](07_what_it_holds.md)이 그 못 고친 것들을 한자리에 모아 재고, [8장](../transformer/index.md)이 기준 자체를 바꾼다 — 거기서는 낱말의 벡터가 **표에서 나오지 않고 그 문장 안에서 만들어진다.**

_확인한 곳 — P. Bojanowski, E. Grave, A. Joulin, T. Mikolov, "Enriching Word Vectors with Subword Information", *TACL* 5 (2017), arXiv:1607.04606. 글자 n-그램을 3~6으로 두는 것과 앞뒤에 `<` `>` 를 붙이는 것, 조각을 해시로 통에 넣는 것이 그 글의 방식이다. 이 쪽은 그 생각을 이 책의 말뭉치에 맞게 작게 다시 짠 것이고, 그 글이 보고한 수와는 무관한 이 책의 측정이다._

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span> `good` 이 조각 10개로 쪼개진다. $n$ 을 3에서 6까지 쓰고 앞뒤에 `<` `>` 를 붙인다는 규칙에서 그 10을 셈으로 보이라.

</div>

??? success "풀이"

    `<good>` 은 글자 6개다. 길이 $n$ 짜리 조각은 $6 - n + 1$ 개 나온다.

    $$
    \underbrace{4}_{n=3} + \underbrace{3}_{n=4} + \underbrace{2}_{n=5} + \underbrace{1}_{n=6} \;=\; 10
    $$

    여기에 **낱말 자신**이 따로 한 줄 더 들어가므로 평균을 내는 벡터는 11개다(코드의 `PIECES` 가 `[i] + 조각들`인 까닭).

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span> 3절은 `blandness` 가 실패한 까닭을 「조각의 33%만 뿌리와 겹친다」로 설명했다. 같은 설명이 맞는다면, **`blandly`** 는 어떻게 될지 예상하고 까닭을 적으라.

</div>

??? success "풀이"

    **더 잘될 것이라 예상한다.** `<blandly>` 는 9글자라 조각이 30개가 아니라 더 적고, `<bland` 를 품은 조각의 **비율이 높아진다.** 꼬리 `-ly` 는 두 글자뿐이라 `-ness` 보다 흔한 꼬리 조각을 덜 만든다.

    재어 보면 답이 나온다. 코드의 `ngrams` 를 두 낱말에 걸어 뿌리 `bland` 와 겹치는 비율을 세면 된다.

    **다만 「겹치는 비율이 높으면 반드시 된다」로 읽으면 안 된다.** 이 쪽이 본 것은 네 낱말뿐이고, 비율과 성공이 함께 움직인 것을 보았을 뿐 **비율이 원인임을 가린 것은 아니다.** 가리려면 비율만 다르고 뿌리 빈도·꼬리 흔함이 같은 쌍을 여럿 만들어야 한다.

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span> 5절은 fastText 에서 `movie`–`kitchen` 이 0.348에서 0.446으로 올랐다고 했다. 「조각을 나눠 쓰면 모든 낱말이 조금씩 가까워진다」가 맞는다면, **아무 낱말 쌍이나 많이 뽑아 닮음의 평균**을 두 모델에서 재면 무엇이 보여야 하는가. 그리고 그 결과가 나와도 **말할 수 없는 것**은 무엇인가.

</div>

??? success "풀이"

    **fastText 쪽의 평균이 더 높아야 한다.** 바닥이 올라갔다는 주장이므로, 특정 쌍이 아니라 **분포 전체가 오른쪽으로 밀렸는지**를 보아야 한다. 어휘에서 쌍을 만 개쯤 무작위로 뽑아 코사인 닮음의 평균과 히스토그램을 두 모델에서 그려 견주면 된다.

    **말할 수 없는 것** — 평균이 올라갔다고 해서 **쓸모가 떨어졌다**는 뜻은 아니다. 닮음을 쓸 때 보통 중요한 것은 절댓값이 아니라 **순위**다. 바닥이 고르게 올라갔을 뿐이라면 순위는 그대로이고, 이웃 목록도 안 바뀐다.

    그러니 4절처럼 **이웃 목록이 실제로 바뀌었는지**를 따로 보아야 한다. 이 쪽에서 그 둘이 함께 일어났지만(평균도 오르고 이웃도 바뀜), 둘은 **다른 주장**이고 따로 재야 한다.
