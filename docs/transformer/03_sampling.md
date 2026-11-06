# 8.3 뽑는 법이 결과를 바꾼다

<div class="leadbox" markdown>

**들어가는 이야기** — 가장 그럴듯한 것만 고르면 망가진다

모델이 내놓는 것은 글자 하나가 아니다. **138가지 확률**이다. 거기서 어느 글자를 고를지는 모델이 정하지 않는다 — **우리가 정한다.**

가장 그럴듯해 보이는 길은 **늘 확률이 가장 큰 글자를 고르는 것**이다. 모델이 가장 믿는 답만 따라가니 가장 좋은 글이 나와야 할 것 같다. 해 보면 이렇다.

```
This movie is a see the story that the story that the story and the story
and the story and the story and the story of the film that the story of
the film that the story
```

**같은 자리에서 맴돈다.** 재어 보면 4글자 묶음의 되풀이가 **0.894**인데, 진짜 영화평을 같은 길이로 잘라 재면 **0.119**다.

그런데 그 글을 **모델에게 다시 채점시키면 가장 좋은 점수가 나온다.** 제 로스가 0.801로, 진짜 영화평의 1.476보다 낮다. **모델이 가장 좋아하는 글이 사람이 보기에 가장 나쁜 글이다.**

[5장](../ch05/index.md)에서 이미 본 꼴이다. 거기서는 평가 분류기의 확신도가 진짜에서 멀어질수록 **올라갔다.** 여기서는 모델의 제 점수가 글이 망가질수록 **내려간다.** 둘 다 같은 말을 한다 — **모델이 제 출력을 채점하게 두면 안 된다.**

</div>

[앞 쪽](02_masking.md)이 지은 모델을 그대로 쓴다. **학습은 한 번도 더 하지 않는다.** 바꾸는 것은 뽑는 법뿐이다.

| 뽑는 법 | 하는 일 |
|---|---|
| **greedy** | 확률이 가장 큰 글자를 고른다. 무작위가 없다 |
| **temperature $T$** | 확률을 $T$로 나눈 뒤 뽑는다. $T$가 작으면 뾰족해지고 크면 평평해진다 |
| **top-$k$** | 상위 $k$개만 남기고 나머지를 지운 뒤 뽑는다 |

!!! note "temperature 는 [8.1절](01_scaled_attention.md)에서 이미 쓴 손잡이다"
    두 식을 나란히 놓으면 같은 것이다.

    $$
    \underbrace{\mathrm{softmax}\!\left(\frac{QK^{\top}}{\sqrt{d_k}}\right)}_{\text{8.1 절 — 어텐션}}
    \qquad\qquad
    \underbrace{\mathrm{softmax}\!\left(\frac{\text{로짓}}{T}\right)}_{\text{이 쪽 — 뽑기}}
    $$

    **$\sqrt{d_k}$가 곧 temperature다.** 소프트맥스에 넣기 전에 수를 나누어 분포의 뾰족함을 조절하는, 글자 그대로 같은 연산이다.

    다른 것은 **왜 돌리느냐**다. 8.1절은 **기울기를 살리려고** 나눈다 — 안 나누면 소프트맥스가 굳어 남은 기울기가 0.72에서 0.13으로 떨어졌다. 이 쪽은 **내놓는 글의 결을 고르려고** 나눈다.

    그리고 두 자리에서 「좋은 값」이 반대다. 8.1절에서는 $d_k$가 무엇이든 분포가 **같은 온도에서 시작하도록** 나눈 것이라 고정값 하나가 옳다. 이 쪽에서는 **고를 수 있는 것**이고, 2절이 보인 대로 자마다 다른 값을 고른다.

---

## 1. 코드

```python
"""8.3 뽑는 법이 결과를 바꾼다 — 같은 모델, 다른 뽑기.

모델은 글자마다 138가지 확률을 내놓을 뿐이다. 거기서 「어느 글자를 고를
것인가」는 모델이 정하지 않는다. 우리가 정한다.

[5.1절]이 그림에서 「어디서 뽑을 것인가」를 물었던 것과 같은 물음이고,
답도 비슷하게 어긋난다 — 가장 그럴듯한 것만 고르면 오히려 망가진다.

재는 자 셋:
  되풀이    지어낸 글에서 4글자 묶음이 앞에 또 나온 비율.
            「낮을수록 좋은 것이 아니다」 - 진짜 글에도 되풀이가 있다
  가짓수    쓴 글자가 몇 가지인가. 역시 진짜 글에 가까울수록 좋다
  제 로스   제가 뽑은 글을 제가 다시 채점한 값. 낮다고 좋은 것이 아니다

읽는 것: 앞 쪽이 남긴 gpt_masked.pt 와 gpt_vocab.pt.
실행: 이 쪽의 코드를 파일로 갈무리해 저장소 뿌리에서 돌린다. 학습이 없다.
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


# === 뽑는 법 넷 ============================================================
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


# === 재는 자 ===============================================================
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
    # 기준선은 「같은 길이로」 재야 한다. 되풀이 비율도 글자 가짓수도
    # 글이 길수록 커지므로, 진짜 평도 400글자 조각 여덟 개로 끊어 잰다
    real = "".join(re.sub(r"\s+", " ", f.read_text(encoding="utf-8")
                          .replace("<br />", " "))
                   for f in sorted((ROOT / "test" / "pos").glob("*.txt"))[:40])
    cuts = [real[i * 1500: i * 1500 + N_GEN] for i in range(N_SAMPLE)]
    f = lambda v: sum(v) / len(v)
    print(f"  {'뽑는 법':22s}{'되풀이':>10}{'글자 가짓수':>10}{'제 로스':>10}")
    print(f"  {'진짜 영화평 (같은 길이)':22s}{f([repeat_rate(c) for c in cuts]):10.3f}"
          f"{f([len(set(c)) for c in cuts]):10.1f}"
          f"{f([self_loss(c) for c in cuts]):10.3f}")
    print("  " + "-" * 52)
    measure("greedy (가장 그럴듯)", greedy=True)
    for t in (0.5, 0.8, 1.0, 1.3):
        measure(f"temperature {t}", temp=t)
    for k in (5, 20, 60):
        measure(f"top-k {k}", topk=k)

    print("\n  === 같은 모델, 다른 뽑기 (앞 160글자) ===")
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
  뽑는 법                         되풀이    글자 가짓수      제 로스
  진짜 영화평 (같은 길이)             0.119      36.2     1.476
  ----------------------------------------------------
  greedy (가장 그럴듯)            0.894      15.0     0.801
  temperature 0.5            0.340      25.6     0.961
  temperature 0.8            0.128      31.1     1.276
  temperature 1.0            0.099      37.5     1.588
  temperature 1.3            0.026      47.8     2.113
  top-k 5                    0.202      28.6     1.190
  top-k 20                   0.088      31.9     1.496
  top-k 60                   0.093      36.8     1.567

  === 같은 모델, 다른 뽑기 (앞 160글자) ===

  -- greedy
     This movie is a see the story that the story that the story and the story and the story and the story and the story of the film that the story of the film that the story 

  -- temperature 0.5
     This movie is the movie movie is a movie that really be show the film that the comedy of the story with the film love of the movie and a completer of a horrible and the s

  -- temperature 1.0
     This movie Wattch modes movie. It is up like it both beest to speed. Through she pament . richt out was someone elech other on the convebout it is passted neanliatingly a

  -- temperature 1.3
     This movie W... ⏎ "PPEPT. ⏎ You Clibeting `He hit renough dest ides Arma who just her two eq. Enart one way sort dide, you kept on expecciover) wite who out of talking elades

  -- top-k 20
     This movie delightly bod. Everybody. But the grenatived. The beginning roles she paneten. End to get augh it didelogues as then begament of the film out of talking elades
```

---

## 2. 자가 서로 다른 쪽을 가리킨다

| 뽑는 법 | 되풀이 | 글자 가짓수 | 제 로스 |
|---|---|---|---|
| **진짜 영화평** (같은 길이) | **0.119** | **36.2** | **1.476** |
| greedy | 0.894 | 15.0 | **0.801** |
| temperature 0.5 | 0.340 | 25.6 | 0.961 |
| temperature 0.8 | **0.128** | 31.1 | 1.276 |
| temperature 1.0 | 0.099 | **37.5** | 1.588 |
| temperature 1.3 | 0.026 | 47.8 | 2.113 |
| top-$k$ 5 | 0.202 | 28.6 | 1.190 |
| top-$k$ 20 | 0.088 | 31.9 | 1.496 |
| top-$k$ 60 | 0.093 | 36.8 | 1.567 |

**셋째 칸만 보면 greedy가 1등이다.** 0.801로 가장 낮고, 진짜 영화평의 1.476보다도 낮다. 그런데 그 글이 「the story that the story that the story」다.

까닭은 단순하다. **제 로스는 「내가 이 글을 얼마나 예상했는가」를 잴 뿐**이다. 자기가 가장 예상한 글자만 골라 이었으니 가장 예상한 글이 나오는 것이 당연하다. **동어반복이다.**

### 「되풀이가 낮을수록 좋다」도 틀렸다

이 쪽을 설계할 때 되풀이를 「낮을수록 좋은 자」로 적었다. 표를 보면 그 말이 틀렸다. **temperature 1.3이 0.026으로 가장 낮은데**, 진짜 영화평은 0.119다. 1.3은 **진짜보다 덜 되풀이한다** — 사람 글에는 `the`, ` of `, `ing ` 같은 묶음이 당연히 되풀이되는데 거기까지 흩어 버린 것이다.

그러니 이 자도 **「진짜에 가까울수록」**이다. [5장의 다양함](../ch05/index.md)이 「1에 가까울수록」이었던 것과 같다. 그 자로 다시 읽으면 **temperature 0.8(0.128)이 진짜 0.119에 가장 가깝다.**

| 자 | 진짜에 가장 가까운 설정 |
|---|---|
| 되풀이 | **temperature 0.8** (0.128 대 0.119) |
| 글자 가짓수 | **temperature 1.0** (37.5 대 36.2) |
| 제 로스 | **top-$k$ 20** (1.496 대 1.476) |

**셋이 서로 다른 설정을 고른다.** 0.8과 1.0 사이 어디라고 말할 수는 있어도, 자 하나로 「최적」을 정할 수는 없다.

---

## 3. 눈으로 보면 더 분명하다

**greedy** — 맴돈다.

```
This movie is a see the story that the story that the story and the story
```

**temperature 0.5** — 덜 맴돌지만 아직 같은 말이 겹친다(`movie movie`, `the film that the`).

```
This movie is the movie movie is a movie that really be show the film that
the comedy of the story with the film love of the movie and a completer
```

**temperature 1.0** — 낱말 묶음이 살아난다. 없는 낱말이 섞이지만 영어처럼 읽힌다.

```
This movie Wattch modes movie. It is up like it both beest to speed.
Through she pament . richt out was someone elech other on the convebout
```

**temperature 1.3** — 너무 흩어진다. 따옴표와 괄호가 짝이 안 맞고 대문자가 아무 데나 선다.

```
This movie W... "PPEPT. You Clibeting `He hit renough dest ides Arma who
just her two eq. Enart one way sort dide, you kept on expecciover) wite
```

**top-$k$ 20** — 1.0과 비슷하되 이상한 글자가 덜 나온다. 꼬리를 아예 잘라 버렸기 때문이다.

```
This movie delightly bod. Everybody. But the grenatived. The beginning
roles she paneten. End to get augh it didelogues as then begament of the
```

!!! note "두 손잡이가 하는 일이 다르다"
    **temperature**는 꼬리를 **눌러 둘 뿐 없애지 않는다.** $T = 0.5$에서도 확률이 아주 낮은 글자가 나올 길이 남아 있다. 그래서 긴 글을 뽑으면 어쩌다 한 번 이상한 글자가 끼고, 그 뒤가 그 글자에 끌려간다.

    **top-$k$**는 꼬리를 **잘라 낸다.** $k$개 밖은 확률이 0이 되어 절대 나오지 않는다. 대신 $k$가 고정이라 문맥이 뻔한 자리(예: `th` 다음)에서도 20가지를 열어 둔다.

    실제로 쓰이는 top-$p$(nucleus)는 그 사이를 노린다 — 개수가 아니라 **누적 확률**로 자르므로 뻔한 자리에서는 좁게, 애매한 자리에서는 넓게 열린다. 이 쪽은 거기까지 가지 않는다.

---

## 4. 5장과 같은 이야기

[5.1절](../ch05/dimreduction/06_sampling.md)은 그림에서 「어디서 뽑을 것인가」를 물었다. 거기서도 답이 어긋났다 — 가장 그럴듯한 자리(잠재 표현 구름의 한가운데)에서만 뽑으면 **3만 잔뜩 나왔다.**

여기서도 같다. **모델이 가장 믿는 것만 고르면 모델이 가장 흔하게 본 것만 나온다.** 글에서는 `the story`이고, 그림에서는 3이었다.

그리고 [장 개요](index.md)가 적어 둔 자의 한계가 여기서 또 걸린다. 제 로스는 **본 적 없는 입력에서 무너지는 자**가 아니라 **제가 만든 입력에서 무너지는 자**다. 다른 고장이지만 교훈은 같다 — **재는 쪽과 재이는 쪽이 같으면 그 수를 믿을 수 없다.**

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span> temperature $T \to 0$이면 무엇이 되는가. $T \to \infty$이면?

</div>

??? success "풀이"

    로짓을 $T$로 나눈 뒤 소프트맥스를 걸므로, $T \to 0$이면 가장 큰 로짓과 나머지의 차가 무한히 벌어져 **확률이 최댓값 한 칸에 몰린다.** 곧 **greedy와 같아진다.**

    $T \to \infty$이면 모든 로짓이 0으로 눌려 **균등분포**가 된다. 모델을 쓰지 않고 138가지에서 아무거나 뽑는 것과 같다.

    표의 흐름이 그 사이를 보인다 — $T$를 0.5에서 1.3으로 키울수록 되풀이가 0.340에서 0.026으로 줄고 글자 가짓수가 25.6에서 47.8로 는다.

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span> 2절은 제 로스가 동어반복이라 했다. 그렇다면 **쓸 수 있는** 자동 채점 자는 무엇인가. 하나를 제안하고 그 자의 약점도 적으라.

</div>

??? success "풀이"

    **다른 모델에게 채점시키면 된다.** 이 쪽의 모델이 뽑은 글을 [8.4절](04_scale.md)의 다른 크기 모델에게 넣어 로스를 재면, 적어도 동어반복은 아니다. 5장이 생성 모델을 평가 분류기로 잰 것과 같은 수법이다.

    **약점은 두 가지다.** 첫째, 채점자가 본 적 없는 글에서 어떻게 반응하는지 알 수 없다 — [장 개요](index.md)와 [5장](../ch05/index.md)이 거듭 적은 문제다. 둘째, 두 모델이 같은 자료로 학습했으면 **같은 편향을 공유한다.** 둘 다 `the story`를 좋아하면 그 글에 둘 다 낮은 로스를 준다.

    그래서 이 쪽은 자동 채점 하나로 결론 내지 않고 **진짜 글을 같은 길이로 잘라 기준선**으로 두었다. 「좋다」가 아니라 **「진짜와 얼마나 다른가」**를 묻는 쪽으로 바꾼 것이다.

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span> 2절의 세 자가 서로 다른 설정을 골랐다. top-$p$(nucleus) 샘플링을 더해 같은 자들로 재는 실험을 설계하고, **무엇이 나오면 top-$p$가 낫다고 말할 수 있는지** 미리 적으라.

</div>

??? success "풀이"

    **설계** — `gen`에 `topp` 인수를 더한다. 로짓을 내림차순으로 정렬해 소프트맥스 누적합이 $p$를 넘는 지점까지만 남기고 나머지를 $-\infty$로 둔다. $p = 0.7, 0.9, 0.95$를 같은 씨앗 여덟 개로 돌려 표를 채운다.

    **미리 적는 판정 기준** — top-$p$가 낫다고 말하려면 **세 자가 동시에** 진짜에 더 가까워야 한다. 되풀이가 0.119에, 가짓수가 36.2에, 제 로스가 1.476에 각각 더 가까운 설정이 top-$k$·temperature 어느 것보다 나아야 한다.

    **그러지 않을 가능성이 크다는 것도 미리 적어 둔다.** 이 모델은 글자 단위이고 어휘가 138가지뿐이라, top-$p$가 노리는 「자리마다 꼬리 길이가 다른」 상황이 낱말 단위만큼 뚜렷하지 않다. 그러면 결론은 「top-$p$가 나쁘다」가 아니라 **「이 설정에서는 가릴 수 없다」**가 되고, 그 또한 적을 값어치가 있는 결과다.

    그리고 씨앗 여덟 개는 적다. 표의 top-$k$ 20과 60이 되풀이 0.088과 0.093으로 거의 같은데, 그 차이가 참인지 가리려면 **같은 설정을 여러 번 돌린 스프레드**가 있어야 한다. [7장](../imdb/04_lstm.md)이 그것 하나로 쪽 하나를 썼다.
