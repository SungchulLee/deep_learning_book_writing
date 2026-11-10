# 8.2 가르기에서 만들기로 — 마스킹이 하는 일

<div class="leadbox" markdown>

**들어가는 이야기** — 답을 보고 푸는 시험

[7장](../imdb/index.md)의 모델은 평 하나를 **통째로 보고** 좋다/나쁘다를 갈랐다. 통째로 보는 것이 당연했다 — 가르는 일이니 다 읽고 정하면 된다.

**글을 지어내는 일은 다르다.** 다음 글자를 맞히는 것이 할 일인데, 통째로 보게 두면 **맞혀야 할 그 글자가 입력 안에 들어 있다.** 답을 보고 푸는 시험이 된다.

그래서 논문의 디코더는 **마스크**를 건다. 자리 $t$는 $t$보다 뒤를 보지 못한다. 어텐션 점수판의 위쪽 삼각을 $-\infty$로 채우면 끝이다.

그러면 마스크를 빼면 어떻게 되는가. **로스가 0으로 떨어질 것 같다** — 베끼면 되니까. 재어 보니 그렇지 않았다.

| | 학습 로스 | 시험 로스 | 미리 보기를 막고 다시 재면 |
|---|---|---|---|
| 마스크 있음 | **1.5333** | **1.5902** | 1.5902 |
| 마스크 없음 | 2.2666 | 2.2710 | **2.9006** |

**마스크를 뺀 쪽이 제 시험에서도 더 못한다.** 베껴서 0이 되기는커녕 1.59를 못 따라간다. 다만 **미리 보기를 막고 다시 재면 2.2710이 2.9006으로 뛴다** — 조금은 앞을 훔쳐보고 있었다는 뜻이다.

그리고 글을 지어내게 하면 거기서 완전히 무너진다.

</div>

이 쪽은 작은 GPT 하나를 짓는다. 자료는 [7장](../imdb/index.md)과 같은 IMDB 영화평이고, 다만 **단어가 아니라 글자** 단위로 다룬다. 어휘를 138가지로 줄여 작은 모델로도 돌아가게 하려는 것이다.

같은 글로 **가르기에서 만들기로** 넘어가는 셈이다. [5장](../ch05/index.md)이 MNIST 하나를 붙들고 분류에서 생성으로 간 것과 같은 걸음이다.

---

## 1. 인코더로는 왜 지어낼 수 없는가

[7.6절](../imdb/06_transformer.md)의 블록을 그대로 두고 할 일만 바꿔 보자. 입력이 `This movi`이고 다음 글자 `e`를 맞혀야 한다. 어텐션은 모든 자리가 모든 자리를 본다. 자리 $t$의 출력으로 $x_{t+1}$을 맞히라고 하면 — **자리 $t+1$의 입력이 바로 그 $x_{t+1}$이다.**

$$
\underbrace{\text{자리 } t \text{ 가 보는 것}}_{x_1, \dots, x_t, \;\boxed{x_{t+1}}, \dots, x_n}
\quad\longrightarrow\quad
\underbrace{x_{t+1}}_{\text{맞혀야 할 것}}
$$

마스크는 그 상자를 지운다. 코드로는 한 줄이다.

```
m = torch.triu(torch.full((L, L), float("-inf")), 1)   # 위쪽 삼각을 -inf 로
```

소프트맥스에 $-\infty$가 들어가면 그 칸의 무게가 0이 된다. 자리 $t$는 $1$부터 $t$까지만 보게 된다.

---

## 2. 코드

```python
"""8.2 가르기에서 만들기로 — 마스킹을 빼면 무슨 일이 일어나는가.

7장의 트랜스포머는 평 하나를 통째로 보고 좋다/나쁘다를 갈랐다. 글을
지어내려면 그렇게 할 수 없다. 다음 글자를 맞히는 일인데 「그 글자를 미리
보고 있으면」 맞히는 것이 아니라 베끼는 것이다.

논문의 디코더가 그래서 마스킹을 건다. 자리 t 는 t 이하만 본다.

재는 법: 똑같은 모델을 둘 학습한다. 하나는 마스킹을 걸고, 하나는 뺀다.
바꾸는 것은 그 하나뿐이다. 그러고 나서 셋을 본다.
  (가) 학습 로스 — 마스킹을 빼면 얼마나 내려가는가
  (나) 다음 글자 맞히기 — 미리 보기를 막은 자리에서 다시 재면
  (다) 실제로 지어낸 글

자료는 7장과 같은 IMDB 영화평이고, 글자 단위로 다룬다.
실행: 이 쪽의 코드를 파일로 갈무리해 저장소 뿌리에서 돌린다.
"""
import math
import re
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

SEED, CTX, BATCH, STEPS, LR = 42, 128, 64, 3000, 3e-4
D_MODEL, HEADS, LAYERS = 192, 4, 3
ROOT = Path("./data/aclImdb")
SC = "."
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


# === 자료: 영화평을 글자 한 줄로 잇는다 ======================================
def load_text(split, n_docs=4000):
    out = []
    for name in ("pos", "neg"):
        for f in sorted((ROOT / split / name).glob("*.txt"))[: n_docs // 2]:
            t = f.read_text(encoding="utf-8").replace("<br />", " ")
            out.append(re.sub(r"\s+", " ", t).strip())
    return "\n".join(out)


train_text = load_text("train")
test_text = load_text("test", 400)
chars = sorted(set(train_text))
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}
V = len(chars)
enc = lambda s: torch.tensor([stoi[c] for c in s if c in stoi], dtype=torch.long)
dec = lambda t: "".join(itos[int(i)] for i in t)
Xtr, Xte = enc(train_text), enc(test_text)


# === 모델: 디코더 블록 하나짜리 GPT =========================================
class Block(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, h, batch_first=True)
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x, mask):
        a, _ = self.attn(self.n1(x), self.n1(x), self.n1(x), attn_mask=mask,
                         need_weights=False)
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
        m = None
        if self.masked:
            # 위쪽 삼각이 -inf — 자리 t 는 t 보다 뒤를 보지 못한다
            m = torch.triu(torch.full((L, L), float("-inf"), device=idx.device), 1)
        x = self.tok(idx) + self.pos[:, :L]
        for b in self.blocks:
            x = b(x, m)
        return self.head(self.nf(x))


def batch(data, g):
    i = torch.randint(len(data) - CTX - 1, (BATCH,), generator=g)
    x = torch.stack([data[j:j + CTX] for j in i])
    y = torch.stack([data[j + 1:j + CTX + 1] for j in i])
    return x.to(device), y.to(device)


def train(masked):
    torch.manual_seed(SEED)
    m = TinyGPT(V, masked=masked).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=LR)
    g = torch.Generator().manual_seed(SEED)
    m.train()
    for s in range(STEPS):
        x, y = batch(Xtr, g)
        loss = F.cross_entropy(m(x).reshape(-1, V), y.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
    return m.eval(), float(loss)


@torch.no_grad()
def eval_loss(m, data, force_mask=None):
    """force_mask=True 면 모델 설정과 무관하게 미리 보기를 막고 잰다."""
    g = torch.Generator().manual_seed(0)
    old = m.masked
    if force_mask is not None:
        m.masked = force_mask
    tot = 0.0
    for _ in range(40):
        x, y = batch(data, g)
        tot += float(F.cross_entropy(m(x).reshape(-1, V), y.reshape(-1)))
    m.masked = old
    return tot / 40


@torch.no_grad()
def generate(m, prompt, n=240, temp=1.0):
    m.masked = True                      # 지어낼 때는 늘 막는다
    idx = enc(prompt).unsqueeze(0).to(device)
    for _ in range(n):
        logits = m(idx[:, -CTX:])[:, -1] / temp
        nxt = torch.multinomial(logits.softmax(-1), 1)
        idx = torch.cat([idx, nxt], 1)
    return dec(idx[0])


if __name__ == "__main__":
    print(f"  글자 {V}가지, 학습 {len(Xtr):,}자, 시험 {len(Xte):,}자")
    npar = sum(p.numel() for p in TinyGPT(V).parameters())
    print(f"  모델 {LAYERS}겹 d_model {D_MODEL} 머리 {HEADS} — 파라메터 {npar:,}\n")

    res = {}
    for masked in (True, False):
        m, last = train(masked)
        tag = "마스킹 있음" if masked else "마스킹 없음"
        res[masked] = m
        print(f"  {tag}")
        print(f"    학습 로스(마지막 배치)        {last:.4f}")
        print(f"    시험 로스 — 제 설정 그대로     {eval_loss(m, Xte):.4f}")
        print(f"    시험 로스 — 미리 보기를 막고   {eval_loss(m, Xte, True):.4f}", flush=True)

    print("\n  === 지어낸 글 (prompt: 'This movie') ===")
    torch.manual_seed(SEED)
    for masked in (True, False):
        tag = "마스킹 있음" if masked else "마스킹 없음"
        print(f"\n  -- {tag}")
        print("     " + generate(res[masked], "This movie").replace("\n", " ⏎ "))

    torch.save(res[True].state_dict(), f"{SC}/gpt_masked.pt")
    torch.save({"chars": chars}, f"{SC}/gpt_vocab.pt")
```

**출력:**

```
  글자 138가지, 학습 5,115,854자, 시험 524,817자
  모델 3겹 d_model 192 머리 4 — 파라메터 1,412,544

  마스킹 있음
    학습 로스(마지막 배치)        1.5333
    시험 로스 — 제 설정 그대로     1.5902
    시험 로스 — 미리 보기를 막고   1.5902
  마스킹 없음
    학습 로스(마지막 배치)        2.2666
    시험 로스 — 제 설정 그대로     2.2710
    시험 로스 — 미리 보기를 막고   2.9006

  === 지어낸 글 (prompt: 'This movie') ===

  -- 마스킹 있음
     This movie Wattch modes movie. It is up like it both beest to speed. Through she pament . richt out was someone elech other on the convebout it is passted neanliatingly and seen went the seem ighos an Gils. Crecinned. ⏎ . to guave a may absolutely that

  -- 마스킹 없음
     This movie Thn s Thes Th This Thih hashithaisho hishas his has The is his hishishine sho h hisis hish ihihe hi Th hi his he hish. hise hohal his ey he hishe hishichehihhihim misheh he hhe hii hi hohhis ishii hahis ishichhhihohis hihh lihaishiohhhih h
```

---

## 3. 짐작이 틀렸다 — 베껴서 0이 되지 않는다

이 쪽을 설계할 때의 짐작은 이랬다. **마스크를 빼면 답이 입력에 있으니 로스가 0 가까이 떨어질 것이다.** 그러면 「학습 로스는 완벽한데 지어내기는 안 되는」 그림이 나오고, 그것이 이 쪽의 이야기가 될 참이었다.

**그렇게 되지 않았다.** 마스크 없는 쪽의 학습 로스가 2.2666으로, 마스크 있는 쪽의 1.5333보다 **나쁘다.**

까닭을 짚어 보면 이렇다. 베끼려면 자리 $t$가 **자리 $t+1$을 콕 집어** 보아야 하는데, 어텐션이 자리를 고르는 기준은 **내용**이다. 「내 바로 다음 자리」를 집으려면 자리 정보만으로 무게를 몰아주는 법을 따로 배워야 하고, 그것이 3,000 스텝 안에 저절로 되지는 않았다.

대신 일어난 일은 **엉뚱한 일을 배운 것**이다. 양쪽을 다 보는 표현을 만들어 놓고 그것으로 다음 글자를 맞히려 하니, 마스크를 걸고 쓰는 자리에서는 쓸모가 없다.

### 그래도 훔쳐보기는 하고 있었다

| 마스크 없이 학습한 모델을 | 로스 |
|---|---|
| 제 설정 그대로(앞뒤 다 보며) 재면 | 2.2710 |
| **미리 보기를 막고** 재면 | **2.9006** |

**0.63이 뛴다.** 뒤를 못 보게 하자 눈에 띄게 못해진다는 것은, 그 모델이 **뒤에 기대고 있었다**는 말이다. 완전히 베끼지는 못했어도 일부는 그러고 있었다.

이 두 수가 이 쪽의 요점이다. **마스크는 모델을 약하게 만드는 제약이 아니라, 지어내는 일과 배우는 일을 같은 것으로 맞추는 장치다.** 마스크가 없으면 배울 때와 쓸 때가 어긋나고, 그 어긋남이 0.63이다.

---

## 4. 지어낸 글을 보면 분명하다

수보다 글이 더 분명하다. 둘 다 `This movie`로 시작하게 하고 240글자를 뽑았다.

**마스크 있음**

```
This movie Wattch modes movie. It is up like it both beest to speed. Through
she pament . richt out was someone elech other on the convebout it is passted
neanliatingly and seen went the seem ighos an Gils. Crecinned.
```

**마스크 없음**

```
This movie Thn s Thes Th This Thih hashithaisho hishas his has The is his
hishishine sho h hisis hish ihihe hi Th hi his he hish. hise hohal his ey he
hishe hishichehihhihim misheh he hhe hii hi hohhis ishii hahis ishichhhihohis
```

**아래쪽은 같은 조각을 되풀이한다.** `his`, `hish`, `hihi`가 끝없이 돈다. 글자 단위 모델이 무너질 때 전형적으로 보이는 형태.

위쪽은 영어가 아니지만 **영어처럼 생겼다.** 띄어쓰기가 단어 길이에 맞고, 마침표 뒤에 대문자가 오고, `It is`, `like it`, `other on the`처럼 실제 단어 묶음이 나온다. `Wattch`, `beest`, `neanliatingly`는 없는 단어이지만 **영어 글자가 이어지는 규칙**은 따르고 있다.

!!! note "이 정도가 1.59 다"
    위 글이 「잘 지어낸 것」은 아니다. 로스 1.5902는 퍼플렉서티로 치면 $e^{1.5902} \approx 4.9$로, **다음 글자를 다섯 가지쯤으로 좁혔다**는 뜻이다. 138가지에서 다섯으로 줄인 것이니 배우기는 배웠는데, 뜻이 통하는 글을 쓰려면 한참 멀다.

    파라메터가 141만 개이고 3,000 스텝을 돌렸다. 논문의 base 모델이 6,500만 개에 P100 여덟 장으로 12시간이었다. **[다음 쪽](03_sampling.md)은 같은 모델을 그대로 두고 뽑는 법만 바꾸어** 얼마나 달라지는지 보고, **[그다음 쪽](04_scale.md)이 크기를 바꾼다.**

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span> `torch.triu(..., 1)`의 `1`을 `0`으로 바꾸면 무엇이 달라지는가. 모델이 무엇을 못 보게 되는지 말하라.

</div>

??? success "풀이"

    `triu(M, 1)`은 **주대각선 위쪽**만 남기고, `triu(M, 0)`은 **주대각선까지** 포함한다. $-\infty$로 채우는 자리가 늘어나는 것이므로, `0`으로 두면 자리 $t$가 **자기 자신도 보지 못한다.**

    그러면 자리 $t$의 출력은 $x_1, \dots, x_{t-1}$만으로 만들어지고, 그것으로 $x_{t+1}$을 맞혀야 한다. 한 글자 건너뛴 예측이 되어 훨씬 어려워진다. 첫 자리는 볼 것이 하나도 없어 소프트맥스가 전부 $-\infty$를 받고 **NaN**이 난다 — [7.5절](../imdb/05_attention.md)이 적어 둔 그 사고와 같은 형태.

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span> 3절은 마스크 없는 모델이 「완전히 베끼지는 못했다」고 했다. 정말 베낄 수 **있는지**를 가리는 실험을 설계하라.

</div>

??? success "풀이"

    베끼는 데 필요한 것은 **자리 기반 어텐션**이다. 그러니 그것을 쉽게 만들어 주고 다시 보면 된다.

    **(가) 더 오래 돌린다.** 3,000 스텝이 모자랐을 뿐일 수 있다. 3만 스텝까지 늘리며 학습 로스를 기록해 0 쪽으로 내려가는지 본다.

    **(나) 자리 정보를 세게 준다.** 자리 임베딩의 크기를 키우거나, 어텐션 점수에 상대 자리 항을 직접 더해 준다. 「바로 다음 자리」를 집기 쉬워진다.

    **(다) 손으로 심어 본다.** 학습하지 말고, 자리 $t$가 자리 $t+1$에만 무게 1을 주도록 어텐션 무게를 직접 짜 넣은 뒤 로스를 잰다. 그것이 0에 가깝게 나오면 **베끼는 해가 존재한다**는 것이 확인되고, 그러면 3절의 결과는 「못 하는 것」이 아니라 「못 찾은 것」이 된다.

    (다)가 가장 값싸고 결정적이다. 학습이 없으니 몇 초면 된다.

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span> 이 쪽은 글자 단위로 다루었다. [7장](../imdb/index.md)처럼 **단어 단위**로 바꾸면 무엇이 좋아지고 무엇이 나빠지는가. 로스 값을 두 설정에서 **직접 견줄 수 있는가?**

</div>

??? success "풀이"

    **좋아지는 것** — 한 스텝이 담는 뜻이 커진다. 글자 128개는 영어로 스무 단어 남짓인데, 단어 128개면 훨씬 긴 문맥을 같은 $n$으로 담는다. 맞춤법을 배우는 데 쓰던 용량도 아낀다.

    **나빠지는 것** — 어휘가 138에서 2만으로 커지니 임베딩과 출력 레이어가 그만큼 커진다. 이 쪽 모델은 141만 개 가운데 출력 레이어가 $192 \times 138 = 26{,}496$인데, 단어이면 $192 \times 20{,}000 = 3{,}840{,}000$으로 모델 전체보다 커진다. 그리고 목록에 없는 단어를 아예 쓸 수 없다.

    **직접 견줄 수 없다.** 로스는 「한 토큰을 맞히는」 교차 엔트로피인데 토큰이 다른 것이기 때문이다. 글자 1.59와 단어 5.0 가운데 어느 쪽이 나은지 말할 수 없다. 견주려면 **같은 단위로 환산**해야 한다 — 글월 하나의 전체 로그 가능도를 **글자 수로 나눈 값**(bits per character)으로 맞추는 것이 보통 쓰는 길이다. 이 쪽의 코드는 그 환산을 하지 않으므로, 여기 적힌 1.59는 **글자 단위 안에서만** 뜻이 있다.
