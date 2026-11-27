# 8.3 디코딩 방법이 결과를 바꾼다

<div class="leadbox" markdown>

**들어가는 이야기** — 가장 그럴듯한 것만 고르면 망가진다

모델이 내놓는 것은 글자 하나가 아니다. 다음 글자에 대한 **138칸짜리 확률 분포**다. 거기서 어느 글자를 고를지는 모델이 정하지 않는다 — **우리가 정한다.**

가장 그럴듯해 보이는 길은 **늘 확률이 가장 큰 글자를 고르는 것** — **그리디
디코딩**(greedy decoding) — 이다. 모델이 가장 믿는 답만 따라가니 가장 좋은 글이 나와야 할 것 같다. 해 보면 이렇다.

```
This movie is a see the story that the story that the story and the story
and the story and the story and the story of the film that the story of
the film that the story
```

**같은 구절을 반복한다.** 연속한 4글자 묶음(4-그램) 가운데 앞에 이미 나온 것의 비율, 곧 **반복률**을 재어 보면 **0.894**인데, 진짜 영화평을 같은 길이로 잘라 재면 **0.119**다.

그런데 그 글을 **모델에게 다시 채점시키면 가장 좋은 점수가 나온다.** 생성한 글을 같은 모델에 넣어 잰 교차 엔트로피 — 이 쪽은 **제 로스**라 부른다 — 가 0.801로, 진짜 영화평의 1.476보다 낮다. **모델이 가장 좋아하는 글이 사람이 보기에 가장 나쁜 글이다.**

[5장](../ch05/index.md)에서 이미 본 형태다. 거기서는 평가 분류기의 확신도가 진짜에서 멀어질수록 **올라갔다.** 여기서는 모델의 제 점수가 글이 망가질수록 **내려간다.** 둘 다 같은 말을 한다 — **모델이 제 출력을 채점하게 두면 안 된다.**

</div>

[앞 쪽](02_masking.md)이 지은 모델을 그대로 쓴다. **트레인은 한 번도 더 하지 않는다.** 바꾸는 것은 **디코딩 방법**(decoding strategy)뿐이다 — 모델이 내놓은 확률 분포에서 다음 글자를 고르는 규칙이다. 확률 분포에서 무작위로 하나를 뽑는 것을 **샘플링**이라 하고, 아래 셋 가운데 뒤의 둘이 샘플링이다.

| 디코딩 방법 | 하는 일 |
|---|---|
| **greedy** | 확률이 가장 큰 글자를 고른다. 무작위가 없다 |
| **temperature $T$** | 로짓을 $T$로 나눈 뒤 소프트맥스를 걸고 샘플링한다. $T$가 작으면 분포가 뾰족해지고 크면 평평해진다 |
| **top-$k$** | 확률이 높은 $k$개만 남기고 나머지의 확률을 0으로 둔 뒤 샘플링한다 |

모델이 내는 로짓을 $z_1, \dots, z_{138}$이라 하면 temperature $T$로 샘플링할 확률은

$$
p_i \;=\; \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}
$$

이다. $T = 1$이면 모델이 트레인된 그대로의 분포이고, $T < 1$이면 로짓의 차이가 벌어져 확률이 큰 글자가 더 커지며, $T > 1$이면 차이가 줄어 균등분포에 가까워진다.

!!! note "temperature 는 [8.1절](01_scaled_attention.md)에서 이미 쓴 연산이다"
    두 식을 나란히 놓으면 같은 것이다.

    $$
    \underbrace{\mathrm{softmax}\!\left(\frac{QK^{\top}}{\sqrt{d_k}}\right)}_{\text{8.1 절 — 어텐션}}
    \qquad\qquad
    \underbrace{\mathrm{softmax}\!\left(\frac{\text{로짓}}{T}\right)}_{\text{이 쪽 — 샘플링}}
    $$

    **$\sqrt{d_k}$가 곧 temperature다.** 소프트맥스에 넣기 전에 입력을 나누어 분포가 얼마나 뾰족한지를 조절하는, 글자 그대로 같은 연산이다.

    다른 것은 **왜 나누느냐**다. 8.1절은 **그래디언트를 살리려고** 나눈다 — 안 나누면 소프트맥스가 포화해 남은 그래디언트가 0.72에서 0.13으로 떨어졌다. 이 쪽은 **생성하는 글의 다양성을 조절하려고** 나눈다.

    그리고 두 자리에서 「좋은 값」이 반대다. 8.1절에서는 $d_k$가 무엇이든 분포가 **같은 온도에서 시작하도록** 나눈 것이라 고정값 하나가 옳다. 이 쪽에서는 **고를 수 있는 하이퍼파라메터**이고, 2절이 보인 대로 평가 지표마다 가장 좋은 값이 다르다.

---

## 1. 코드

```python
"""8.3 디코딩 방법이 결과를 바꾼다 — 같은 모델, 다른 디코딩.

모델은 자리마다 138칸짜리 확률 분포를 내놓을 뿐이다. 거기서 「어느 글자를
고를 것인가」는 모델이 정하지 않는다. 우리가 정한다.

[5.1절]이 그림에서 「어디서 샘플링할 것인가」를 물었던 것과 같은 물음이고,
답도 비슷하게 어긋난다 — 가장 그럴듯한 것만 고르면 오히려 망가진다.

평가 지표 셋:
  반복률    생성한 글에서 4글자 묶음(4-그램)이 앞에 또 나온 비율.
            「낮을수록 좋은 것이 아니다」 - 진짜 글에도 반복이 있다
  가짓수    쓴 글자가 몇 가지인가. 역시 진짜 글에 가까울수록 좋다
  제 로스   제가 생성한 글을 제가 다시 채점한 교차 엔트로피.
            낮다고 좋은 것이 아니다

읽는 것: 앞 쪽이 남긴 gpt_masked.pt 와 gpt_vocab.pt.
실행: 이 쪽의 코드를 파일로 갈무리해 저장소 뿌리에서 돌린다. 트레인이 없다.
"""
import re
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

SEED, CTX = 42, 128
D_MODEL, HEADS, LAYERS = 192, 4, 3
N_GEN, N_SAMPLE = 400, 8
ROOT = Path("./data/aclImdb")
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

chars = torch.load("gpt_vocab.pt", weights_only=False)["chars"]
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}
V = len(chars)
enc = lambda s: torch.tensor([stoi[c] for c in s if c in stoi], dtype=torch.long)
dec = lambda t: "".join(itos[int(i)] for i in t)


class Block(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, h, batch_first=True)
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x, m):
        q = self.n1(x)
        a, _ = self.attn(q, q, q, attn_mask=m, need_weights=False)
        x = x + a
        return x + self.ff(self.n2(x))


class TinyGPT(nn.Module):
    def __init__(self, v, d=D_MODEL, h=HEADS, n=LAYERS, masked=True):
        super().__init__()
        self.masked = masked
        self.tok = nn.Embedding(v, d)
        self.pos = nn.Parameter(torch.zeros(1, CTX, d))
        self.blocks = nn.ModuleList([Block(d, h) for _ in range(n)])
        self.nf = nn.LayerNorm(d)
        self.head = nn.Linear(d, v, bias=False)

    def forward(self, idx):
        L = idx.size(1)
        m = torch.triu(torch.full((L, L), float("-inf"), device=idx.device), 1)
        x = self.tok(idx) + self.pos[:, :L]
        for b in self.blocks:
            x = b(x, m)
        return self.head(self.nf(x))


model = TinyGPT(V).to(device)
model.load_state_dict(torch.load("gpt_masked.pt", map_location=device,
                                 weights_only=True))
model.eval()


# === 디코딩 방법 셋: greedy · temperature · top-k ==========================
@torch.no_grad()
def gen(prompt, n=N_GEN, temp=1.0, topk=None, greedy=False, seed=SEED):
    torch.manual_seed(seed)
    idx = enc(prompt).unsqueeze(0).to(device)
    for _ in range(n):
        logits = model(idx[:, -CTX:])[:, -1]
        if greedy:
            nxt = logits.argmax(-1, keepdim=True)
        else:
            logits = logits / temp
            if topk:
                v, _ = logits.topk(topk)
                logits = logits.masked_fill(logits < v[:, [-1]], float("-inf"))
            nxt = torch.multinomial(logits.softmax(-1), 1)
        idx = torch.cat([idx, nxt], 1)
    return dec(idx[0])


# === 평가 지표 =============================================================
def repeat_rate(s, k=4):
    grams = [s[i:i + k] for i in range(len(s) - k + 1)]
    return 1.0 - len(set(grams)) / len(grams)


@torch.no_grad()
def self_loss(s):
    x = enc(s).unsqueeze(0).to(device)[:, :CTX + 1]
    return float(F.cross_entropy(model(x[:, :-1]).reshape(-1, V),
                                 x[:, 1:].reshape(-1)))


def measure(name, **kw):
    rr, nc, sl = [], [], []
    for i in range(N_SAMPLE):
        s = gen("This movie", seed=SEED + i, **kw)[len("This movie"):]
        rr.append(repeat_rate(s)); nc.append(len(set(s))); sl.append(self_loss(s))
    f = lambda v: sum(v) / len(v)
    print(f"  {name:22s}{f(rr):10.3f}{f(nc):10.1f}{f(sl):10.3f}")
    return f(rr)


if __name__ == "__main__":
    # 기준선은 「같은 길이로」 재야 한다. 반복률도 글자 가짓수도
    # 글이 길수록 커지므로, 진짜 평도 400글자 조각 여덟 개로 끊어 잰다
    real = "".join(re.sub(r"\s+", " ", f.read_text(encoding="utf-8")
                          .replace("<br />", " "))
                   for f in sorted((ROOT / "test" / "pos").glob("*.txt"))[:40])
    cuts = [real[i * 1500: i * 1500 + N_GEN] for i in range(N_SAMPLE)]
    f = lambda v: sum(v) / len(v)
    print(f"  {'디코딩 방법':22s}{'반복률':>10}{'글자 가짓수':>10}{'제 로스':>10}")
    print(f"  {'진짜 영화평 (같은 길이)':22s}{f([repeat_rate(c) for c in cuts]):10.3f}"
          f"{f([len(set(c)) for c in cuts]):10.1f}"
          f"{f([self_loss(c) for c in cuts]):10.3f}")
    print("  " + "-" * 52)
    measure("greedy (가장 그럴듯)", greedy=True)
    for t in (0.5, 0.8, 1.0, 1.3):
        measure(f"temperature {t}", temp=t)
    for k in (5, 20, 60):
        measure(f"top-k {k}", topk=k)

    print("\n  === 같은 모델, 다른 디코딩 (앞 160글자) ===")
    for name, kw in [("greedy", dict(greedy=True)),
                     ("temperature 0.5", dict(temp=0.5)),
                     ("temperature 1.0", dict(temp=1.0)),
                     ("temperature 1.3", dict(temp=1.3)),
                     ("top-k 20", dict(topk=20))]:
        print(f"\n  -- {name}")
        print("     " + gen("This movie", n=160, **kw).replace("\n", " ⏎ "))
```

**출력:**

```
  디코딩 방법                       반복률    글자 가짓수      제 로스
  진짜 영화평 (같은 길이)             0.119      36.2     1.476
  ----------------------------------------------------
  greedy (가장 그럴듯)            0.894      15.0     0.801
  temperature 0.5            0.341      25.8     0.950
  temperature 0.8            0.134      31.1     1.278
  temperature 1.0            0.092      36.8     1.588
  temperature 1.3            0.030      46.2     2.131
  top-k 5                    0.204      28.2     1.189
  top-k 20                   0.085      32.1     1.501
  top-k 60                   0.096      37.1     1.556

  === 같은 모델, 다른 디코딩 (앞 160글자) ===

  -- greedy
     This movie is a see the story that the story that the story and the story and the story and the story and the story of the film that the story of the film that the story 

  -- temperature 0.5
     This movie is the movie movie is a movie that really be show the film that the comedy of the story with the film love of the movie and a completer of a horrible and the s

  -- temperature 1.0
     This movie Wattch modes movie. It is up like it both beest to speed. Through she pament . richt out was someone elech other on the convebout it is passted neanliatingly a

  -- temperature 1.3
     This movie W... ⏎ "PPEPT. ⏎ You Clibeting `He hit renough dest idesperate groups she pymone . richt ombinal sovel. Delugh okays one before about mudiek a Jiers's norching tha

  -- top-k 20
     This movie delightly bod. Everybody. But the grenatived. The beginning roles she paneten. End to get augh it didelogues as then begament of the film out of talking elades
```

---

## 2. 평가 지표가 서로 다른 설정을 고른다

| 디코딩 방법 | 반복률 | 글자 가짓수 | 제 로스 |
|---|---|---|---|
| **진짜 영화평** (같은 길이) | **0.119** | **36.2** | **1.476** |
| greedy | 0.894 | 15.0 | **0.801** |
| temperature 0.5 | 0.341 | 25.8 | 0.950 |
| temperature 0.8 | **0.134** | 31.1 | 1.278 |
| temperature 1.0 | 0.092 | **36.8** | 1.588 |
| temperature 1.3 | 0.030 | 46.2 | 2.131 |
| top-$k$ 5 | 0.204 | 28.2 | 1.189 |
| top-$k$ 20 | 0.085 | 32.1 | **1.501** |
| top-$k$ 60 | 0.096 | 37.1 | 1.556 |

**셋째 열만 보면 greedy가 1등이다.** 0.801로 가장 낮고, 진짜 영화평의 1.476보다도 낮다. 그런데 그 글이 「the story that the story that the story」다.

까닭은 단순하다. **제 로스는 「모델이 이 글에 얼마나 높은 확률을 주었는가」를 잴 뿐**이다. 자리마다 확률이 가장 높은 글자만 골라 이었으니 확률이 높은 글이 나오는 것이 당연하다. **순환 논증이다** — 생성한 모델과 채점한 모델이 같다.

### 「반복률이 낮을수록 좋다」도 틀렸다

반복률은 「낮을수록 좋은 지표」로 여기기 쉽다. 표를 보면 그 말이 틀렸다. **temperature 1.3이 0.030으로 가장 낮은데**, 진짜 영화평은 0.119다. 1.3은 **진짜보다 덜 반복한다** — 사람이 쓴 글에는 `the`, ` of `, `ing ` 같은 묶음이 당연히 반복되는데 그것까지 없애 버린 것이다.

그러니 이 지표도 **「진짜에 가까울수록 좋은」** 지표다. [5장의 다양성](../ch05/index.md)이 「1에 가까울수록」이었던 것과 같다. 그 기준으로 다시 읽으면 **temperature 0.8(0.134)이 진짜 0.119에 가장 가깝다.**

| 평가 지표 | 진짜에 가장 가까운 설정 |
|---|---|
| 반복률 | **temperature 0.8** (0.134 대 0.119) |
| 글자 가짓수 | **temperature 1.0** (36.8 대 36.2) |
| 제 로스 | **top-$k$ 20** (1.501 대 1.476) |

**셋이 서로 다른 설정을 고른다.** temperature로는 0.8과 1.0 사이, top-$k$로는 20과 60 사이 어디라고 말할 수는 있어도, 지표 하나로 「최적」을 정할 수는 없다.

!!! warning "샘플링한 줄은 셋째 자리까지 믿을 수 없다"
    표의 값은 설정마다 **시드 여덟 개**로 400글자씩 생성해 평균 낸 것이다. 표본이 적어 이웃한 설정의 차이가 우연일 수 있다. 글자 가짓수에서 temperature 1.0(36.8)과 top-$k$ 60(37.1)은 진짜(36.2)와의 거리가 0.6과 0.9로, 순위를 매길 만큼 벌어져 있지 않다.

    그리고 이 값들은 **체크포인트의 끝자리에 민감하다.** [8.4절](04_scale.md)이 적은 대로 MPS에서 트레인하면 로스의 넷째 자리가 달라질 수 있다. 그렇게 끝자리만 다른 체크포인트로 같은 표를 다시 재어 보면 greedy 줄과 진짜 영화평 줄은 그대로인데, 샘플링한 줄들은 반복률이 0.007, 글자 가짓수가 1.6, 제 로스가 0.02까지 달라졌다. 자기회귀 생성에서는 글자 하나가 다르게 뽑히면 그 뒤가 전부 달라지기 때문이다. 위에서 고른 세 설정(0.8 · 1.0 · top-$k$ 20)과 「셋이 서로 다르다」는 결론은 그 변동 안에서도 유지되었지만, **표의 셋째 자리로 설정의 우열을 가려서는 안 된다.**

---

## 3. 눈으로 보면 더 분명하다

**greedy** — 같은 구절을 반복한다.

```
This movie is a see the story that the story that the story and the story
```

**temperature 0.5** — 반복이 줄었지만 아직 같은 단어가 겹친다(`movie movie`, `the film that the`).

```
This movie is the movie movie is a movie that really be show the film that
the comedy of the story with the film love of the movie and a completer
```

**temperature 1.0** — 단어 묶음이 살아난다. 없는 단어가 섞이지만 영어처럼 읽힌다.

```
This movie Wattch modes movie. It is up like it both beest to speed.
Through she pament . richt out was someone elech other on the convebout
```

**temperature 1.3** — 확률이 낮은 글자가 너무 자주 나온다. 따옴표가 짝이 안 맞고 대문자가 아무 데나 서며, 없는 단어가 대부분이다.

```
This movie W... "PPEPT. You Clibeting `He hit renough dest idesperate groups
she pymone . richt ombinal sovel. Delugh okays one before about mudiek a
```

**top-$k$ 20** — 1.0과 비슷하되 이상한 글자가 덜 나온다. 분포의 **꼬리**(확률이 낮은 글자들)를 아예 잘라 버렸기 때문이다.

```
This movie delightly bod. Everybody. But the grenatived. The beginning
roles she paneten. End to get augh it didelogues as then begament of the
```

!!! note "두 하이퍼파라메터가 하는 일이 다르다"
    **temperature**는 꼬리의 확률을 **낮출 뿐 0으로 만들지 않는다.** $T = 0.5$에서도 확률이 아주 낮은 글자가 나올 길이 남아 있다. 그래서 긴 글을 생성하면 어쩌다 한 번 이상한 글자가 끼고, 자기회귀 생성이라 그 글자가 다음 입력이 되어 뒤의 글자들에 영향을 준다.

    **top-$k$**는 꼬리를 **잘라 낸다.** $k$개 밖은 확률이 0이 되어 절대 나오지 않는다. 대신 $k$가 고정이라 문맥이 뻔한 자리(예: `th` 다음)에서도 20가지를 열어 둔다.

    실제로 널리 쓰이는 **top-$p$ 샘플링**(nucleus sampling)은 그 사이를 노린다 — 개수가 아니라 **누적 확률**로 자르므로 뻔한 자리에서는 후보가 적고, 애매한 자리에서는 많다. 연습문제 3이 그것을 같은 지표로 잰다.

---

## 4. 5장과 같은 이야기

[5.1절](../ch05/dimreduction/06_sampling.md)은 그림에서 「어디서 샘플링할 것인가」를 물었다. 거기서도 답이 어긋났다 — 가장 그럴듯한 자리(잠재 표현 분포의 한가운데)에서만 샘플링하면 **숫자라 부르기 어려운 획 뭉치가 잔뜩 나왔고**, 평가 분류기는 그 셋에 둘을 3으로 읽었다.

여기서도 같다. **모델이 가장 믿는 것만 고르면 출력이 한 가지 형태로 쏠린다.** 글에서는 `the story`이고, 그림에서는 3으로 읽히는 획 뭉치였다.

그리고 [5장 개요](../ch05/index.md)가 적어 둔 지표의 한계가 여기서 또 드러난다. 5장의 평가 분류기는 **트레인 때 본 적 없는 종류의 입력**에서 믿을 수 없게 되는 지표였고, 제 로스는 **제가 생성한 입력**에서 믿을 수 없게 되는 지표다. 고장 난 까닭은 다르지만 교훈은 같다 — **평가하는 모델과 평가받는 모델이 같으면 그 수를 믿을 수 없다.**

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span> temperature $T \to 0$이면 무엇이 되는가. $T \to \infty$이면?

</div>

??? success "풀이"

    로짓을 $T$로 나눈 뒤 소프트맥스를 걸므로, $T \to 0$이면 가장 큰 로짓과 나머지의 차가 무한히 벌어져 **확률이 최댓값 한 칸에 몰린다.** 곧 **greedy와 같아진다.**

    $T \to \infty$이면 모든 로짓이 0으로 눌려 **균등분포**가 된다. 모델을 쓰지 않고 138가지에서 균등하게 샘플링하는 것과 같다.

    표의 흐름이 그 사이를 보인다 — $T$를 0.5에서 1.3으로 키울수록 반복률이 0.341에서 0.030으로 줄고 글자 가짓수가 25.8에서 46.2로 는다.

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span> 2절은 제 로스로 채점하는 것이 순환 논증이라 했다. 그렇다면 **쓸 수 있는** 자동 평가 지표는 무엇인가. 하나를 제안하고 그 지표의 약점도 적으라.

</div>

??? success "풀이"

    **다른 모델에게 채점시키면 된다.** 이 쪽의 모델이 생성한 글을 [8.4절](04_scale.md)의 다른 크기 모델에게 넣어 로스를 재면, 적어도 순환 논증은 아니다. 5장이 생성 모델을 따로 트레인한 평가 분류기로 잰 것과 같은 방법이다.

    **약점은 두 가지다.** 첫째, 채점하는 모델이 트레인 때 본 적 없는 종류의 글에 어떤 로스를 주는지 알 수 없다 — [5장](../ch05/index.md)이 거듭 적은 문제다. 둘째, 두 모델이 같은 데이터로 트레인했으면 **같은 편향을 공유한다.** 둘 다 `the story`를 좋아하면 그 글에 둘 다 낮은 로스를 준다.

    그래서 이 쪽은 자동 채점 하나로 결론 내지 않고 **진짜 글을 같은 길이로 잘라 기준선**으로 두었다. 「좋다」가 아니라 **「진짜와 얼마나 다른가」**를 묻는 쪽으로 바꾼 것이다.

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span> 2절의 세 지표가 서로 다른 설정을 골랐다. top-$p$(nucleus) 샘플링을 더해 같은 지표들로 재는 실험을 설계하고, **무엇이 나오면 top-$p$가 낫다고 말할 수 있는지** 미리 적으라.

</div>

??? success "풀이"

    **설계** — 확률을 내림차순으로 정렬해 **누적 확률이 $p$를 처음 넘는 글자까지만** 남기고 나머지의 확률을 0으로 둔 뒤 샘플링한다. $p = 0.7, 0.9, 0.95$를 본문과 같은 시드 여덟 개로 돌려 같은 세 지표를 잰다.

    **미리 적는 판정 기준** — top-$p$가 낫다고 말하려면 **세 지표가 동시에** 진짜에 더 가까워야 한다. 반복률이 0.119에, 가짓수가 36.2에, 제 로스가 1.476에 각각 더 가까운 설정이 top-$k$·temperature 어느 것보다 나아야 한다.

    ```python
    # 쪽의 프로그램 끝에 이어 붙여 돌린다 - model, enc, dec, repeat_rate, self_loss 를 그대로 쓴다
    @torch.no_grad()
    def gen_topp(prompt, topp, n=N_GEN, seed=SEED):
        torch.manual_seed(seed)
        idx = enc(prompt).unsqueeze(0).to(device)
        for _ in range(n):
            p = model(idx[:, -CTX:])[:, -1].softmax(-1)
            sp, si = p.sort(-1, descending=True)
            # 누적 확률이 topp 를 처음 넘는 글자까지만 남긴다
            sp = sp.masked_fill(sp.cumsum(-1) - sp >= topp, 0.0)
            nxt = si.gather(-1, torch.multinomial(sp / sp.sum(-1, keepdim=True), 1))
            idx = torch.cat([idx, nxt], 1)
        return dec(idx[0])

    f = lambda v: sum(v) / len(v)
    for topp in (0.7, 0.9, 0.95):
        rr, nc, sl = [], [], []
        for i in range(N_SAMPLE):
            s = gen_topp("This movie", topp, seed=SEED + i)[len("This movie"):]
            rr.append(repeat_rate(s)); nc.append(len(set(s))); sl.append(self_loss(s))
        print(f"  {'top-p ' + str(topp):22s}{f(rr):10.3f}{f(nc):10.1f}{f(sl):10.3f}")
    ```

    ```
      top-p 0.7                  0.251      25.9     1.053
      top-p 0.9                  0.118      29.9     1.246
      top-p 0.95                 0.112      33.2     1.362
    ```

    | 평가 지표 | 진짜 | top-$p$ 가운데 가장 가까운 값 | 본문 표에서 가장 가까운 값 |
    |---|---|---|---|
    | 반복률 | 0.119 | **0.118** ($p = 0.9$) | 0.134 (temperature 0.8) |
    | 글자 가짓수 | 36.2 | 33.2 ($p = 0.95$) | **36.8** (temperature 1.0) |
    | 제 로스 | 1.476 | 1.362 ($p = 0.95$) | **1.501** (top-$k$ 20) |

    **판정 기준을 채우지 못했다.** 반복률은 top-$p$ 0.9가 0.118로 진짜와 거의 같아 본문의 어느 설정보다 가깝다. 그러나 글자 가짓수와 제 로스는 세 $p$ 모두 진짜보다 낮은 쪽에 머물러, temperature 1.0과 top-$k$ 20이 더 가깝다. 셋 가운데 하나에서만 앞섰으므로 **「top-$p$가 낫다」고 말할 수 없다.**

    까닭으로 짐작할 수 있는 것은 어휘의 크기다. 이 모델은 글자 단위이고 어휘가 138가지뿐이라, top-$p$가 노리는 「자리마다 꼬리 길이가 크게 다른」 상황이 단어 단위만큼 뚜렷하지 않다. 그러니 결론은 「top-$p$가 나쁘다」가 아니라 **「이 설정에서는 우열을 가릴 수 없다」**이다.

    그리고 시드 여덟 개는 적다. 표의 top-$k$ 20과 60이 반복률 0.085와 0.096으로 가까운데, 그 차이가 참인지 가리려면 **같은 설정을 여러 번 돌린 스프레드**가 있어야 한다. [7장](../imdb/04_lstm.md)이 그것 하나로 쪽 하나를 썼다.
