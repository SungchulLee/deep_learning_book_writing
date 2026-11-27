# 8.2 분류에서 생성으로 — 마스킹이 하는 일

<div class="leadbox" markdown>

**들어가는 이야기** — 답을 보고 푸는 시험

[7장](../imdb/index.md)의 모델은 평 하나를 **통째로 보고** 좋다/나쁘다를 분류했다. 통째로 보는 것이 당연했다 — 분류하는 일이니 다 읽고 정하면 된다.

**글을 생성하는 일은 다르다.** 다음 글자를 맞히는 것이 할 일인데, 통째로 보게 두면 **맞혀야 할 그 글자가 입력 안에 들어 있다.** 답을 보고 푸는 시험이 된다.

그래서 논문의 디코더는 **마스크**를 건다. 자리 $t$는 $t$보다 뒤를 보지 못한다. 어텐션 점수 행렬의 위쪽 삼각을 $-\infty$로 채우면 끝이다. 이 마스크를 **인과 마스크**(causal mask)라 한다.

그러면 마스크를 빼면 어떻게 되는가. **로스가 0으로 떨어질 것 같다** — 베끼면 되니까. 재어 보니 그렇지 않았다.

| | 트레인 로스 | 시험 로스 | 마스크를 걸고 다시 재면 |
|---|---|---|---|
| 마스크 있음 | **1.5333** | **1.5902** | 1.5902 |
| 마스크 없음 | 2.2666 | 2.2710 | **2.9006** |

**마스크를 뺀 쪽이 제 시험에서도 더 못한다.** 베껴서 0이 되기는커녕 1.59를 못 따라간다. 다만 **마스크를 걸어 뒤쪽 글자를 못 보게 하고 다시 재면 2.2710이 2.9006으로 뛴다** — 조금은 뒤쪽 글자에 의존하고 있었다는 뜻이다.

그리고 글을 생성하게 하면 거기서 완전히 무너진다.

</div>

이 쪽은 작은 GPT 하나를 짓는다. GPT는 트랜스포머의 **디코더만** 쌓아 다음 토큰을 예측하게 트레인한 모델이다. 데이터는 [7장](../imdb/index.md)과 같은 IMDB 영화평이고, 다만 토큰을 **단어가 아니라 글자** 단위로 잡는다. 어휘를 138가지로 줄여 작은 모델로도 돌아가게 하려는 것이다.

**무엇을 트레인하는가.** 글자열 $x_1, x_2, \dots, x_n$이 있을 때 모델이 배우는 것은 **앞의 글자들이 주어졌을 때 다음 글자의 조건부 확률**이다.

$$
p(x_{t+1} \mid x_1, \dots, x_t)
$$

이 확률을 내는 모델을 **언어 모델**(language model)이라 한다. 출력은 자리마다 138칸짜리 로짓이고, 로스는 [3장](../ch03/index.md)부터 써 온 **교차 엔트로피**다 — 정답이 「좋다/나쁘다」가 아니라 **실제로 다음에 온 글자**라는 것만 다르다. 정답을 사람이 붙일 필요가 없으므로 글만 있으면 트레인할 수 있다.

**어떻게 생성하는가.** 트레인된 모델에 `This movie`를 넣으면 다음 글자의 확률 분포가 나온다. 거기서 글자 하나를 샘플링해 입력 끝에 붙이고, 늘어난 입력을 다시 모델에 넣는다. 이렇게 제 출력을 다시 입력으로 받아 한 토큰씩 이어 가는 방식을 **자기회귀**(autoregressive) 생성이라 한다. 코드의 `generate`가 그 반복문이다.

같은 글로 **분류에서 생성으로** 넘어가는 셈이다. [5장](../ch05/index.md)이 MNIST 하나를 붙들고 분류에서 생성으로 간 것과 같은 걸음이다.

---

## 1. 인코더로는 왜 생성할 수 없는가

[7.6절](../imdb/06_transformer.md)의 블록을 그대로 두고 과제만 다음 글자 예측으로 바꿔 보자. 입력이 `This movi`이고 다음 글자 `e`를 맞혀야 한다. 어텐션은 모든 자리가 모든 자리를 본다. 자리 $t$의 출력으로 $x_{t+1}$을 맞히라고 하면 — **자리 $t+1$의 입력이 바로 그 $x_{t+1}$이다.**

$$
\underbrace{\text{자리 } t \text{ 가 보는 것}}_{x_1, \dots, x_t, \;\boxed{x_{t+1}}, \dots, x_n}
\quad\longrightarrow\quad
\underbrace{x_{t+1}}_{\text{맞혀야 할 것}}
$$

마스크는 그 상자를 지운다. 코드로는 한 줄이다.

```
m = torch.triu(torch.full((L, L), float("-inf")), 1)   # 위쪽 삼각을 -inf 로
```

소프트맥스에 $-\infty$가 들어가면 $e^{-\infty} = 0$이라 그 칸의 어텐션 웨이트가 0이 된다. 자리 $t$는 $1$부터 $t$까지만 보게 된다.

**마스크 덕에 트레인이 병렬로 된다.** 길이 128짜리 입력 하나를 넣으면 자리 1의 출력은 $x_1$만 보고 $x_2$를, 자리 2의 출력은 $x_1, x_2$만 보고 $x_3$을 예측한다. 곧 순전파 한 번에 **예측 문제 128개**를 동시에 푼다 — 코드의 `batch`가 입력 `x`와 그것을 한 칸 민 정답 `y`를 함께 만드는 까닭이다. RNN처럼 한 글자씩 순서대로 계산할 필요가 없다.

**이 쪽의 블록은 7.6절과 한 군데가 다르다.** 7.6절은 논문대로 잔차 연결 **뒤에** LayerNorm을 두었고(post-LN, `n1(h + a)`), 이 쪽은 GPT-2 이후의 관례대로 어텐션과 피드포워드 **앞에** 둔다(pre-LN, `x + attn(n1(x))`). 블록을 여러 개 쌓을 때 트레인이 더 안정적이라 요즘 모델은 대개 뒤쪽을 쓴다. 엑티베이션 함수도 ReLU 대신 GPT가 쓰는 GELU다.

---

## 2. 코드

```python
"""8.2 분류에서 생성으로 — 마스킹을 빼면 무슨 일이 일어나는가.

7장의 트랜스포머는 평 하나를 통째로 보고 좋다/나쁘다를 분류했다. 글을
생성하려면 그렇게 할 수 없다. 다음 글자를 예측하는 일인데 「그 글자를 미리
보고 있으면」 예측하는 것이 아니라 베끼는 것이다.

논문의 디코더가 그래서 인과 마스크(causal mask)를 건다. 자리 t 는 t 이하만 본다.

재는 법: 똑같은 모델을 둘 트레인한다. 하나는 마스킹을 걸고, 하나는 뺀다.
바꾸는 것은 그 하나뿐이다. 그러고 나서 셋을 본다.
  (가) 트레인 로스 — 마스킹을 빼면 얼마나 내려가는가
  (나) 다음 글자 예측 — 마스크를 걸고 다시 재면
  (다) 실제로 생성한 글

데이터는 7장과 같은 IMDB 영화평이고, 글자 단위로 다룬다.
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


# === 데이터: 영화평을 글자 한 줄로 잇는다 ====================================
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


# === 모델: 디코더 블록을 LAYERS 개 쌓은 GPT (pre-LN) =======================
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
    """force_mask=True 면 모델 설정과 무관하게 인과 마스크를 걸고 잰다."""
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
    m.masked = True                      # 생성할 때는 늘 마스크를 건다
    idx = enc(prompt).unsqueeze(0).to(device)
    for _ in range(n):
        logits = m(idx[:, -CTX:])[:, -1] / temp
        nxt = torch.multinomial(logits.softmax(-1), 1)
        idx = torch.cat([idx, nxt], 1)
    return dec(idx[0])


if __name__ == "__main__":
    print(f"  글자 {V}가지, 트레인 {len(Xtr):,}자, 시험 {len(Xte):,}자")
    npar = sum(p.numel() for p in TinyGPT(V).parameters())
    print(f"  모델 블록 {LAYERS}개 d_model {D_MODEL} 헤드 {HEADS} — 파라메터 {npar:,}\n")

    res = {}
    for masked in (True, False):
        m, last = train(masked)
        tag = "마스킹 있음" if masked else "마스킹 없음"
        res[masked] = m
        print(f"  {tag}")
        print(f"    트레인 로스(마지막 배치)        {last:.4f}")
        print(f"    시험 로스 — 제 설정 그대로     {eval_loss(m, Xte):.4f}")
        print(f"    시험 로스 — 마스크를 걸고      {eval_loss(m, Xte, True):.4f}", flush=True)

    print("\n  === 생성한 글 (prompt: 'This movie') ===")
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
  글자 138가지, 트레인 5,115,854자, 시험 524,817자
  모델 블록 3개 d_model 192 헤드 4 — 파라메터 1,412,544

  마스킹 있음
    트레인 로스(마지막 배치)        1.5333
    시험 로스 — 제 설정 그대로     1.5902
    시험 로스 — 마스크를 걸고      1.5902
  마스킹 없음
    트레인 로스(마지막 배치)        2.2666
    시험 로스 — 제 설정 그대로     2.2710
    시험 로스 — 마스크를 걸고      2.9006

  === 생성한 글 (prompt: 'This movie') ===

  -- 마스킹 있음
     This movie Wattch modes movie. It is up like it both beest to speed. Through she pament . richt out was someone elech other on the convebout it is passted neanliatingly and seen went the seem ighos an Gils. Crecinned. ⏎ . to guave a may absolutely that

  -- 마스킹 없음
     This movie Thn s Thes Th This Thih hashithaisho hishas his has The is his hishishine sho h hisis hish ihihe hi Th hi his he hish. hise hohal his ey he hishe hishichehihhihim misheh he hhe hii hi hohhis ishii hahis ishichhhihohis hihh lihaishiohhhih h
```

---

## 3. 짐작이 틀렸다 — 베껴서 0이 되지 않는다

이 쪽을 설계할 때의 짐작은 이랬다. **마스크를 빼면 답이 입력에 있으니 로스가 0 가까이 떨어질 것이다.** 그러면 「트레인 로스는 0에 가까운데 생성은 안 되는」 결과가 나올 것이라 예상했다.

**그렇게 되지 않았다.** 마스크 없는 쪽의 트레인 로스가 2.2666으로, 마스크 있는 쪽의 1.5333보다 **나쁘다.**

가능한 설명은 이렇다. 베끼려면 자리 $t$가 **자리 $t+1$ 하나에** 어텐션 웨이트를 몰아주어야 한다. 그런데 어텐션 점수는 쿼리와 키의 내적이고, 그 둘은 글자 임베딩과 위치 임베딩을 **더한 것**에서 나온다. 「내 바로 다음 자리」를 고르려면 글자가 무엇이든 위치 임베딩만으로 점수가 정해지도록 $W_Q$·$W_K$가 맞춰져야 하고, 그것이 3,000 스텝 안에 이루어지지는 않았다. 다만 이것은 **이 쪽이 재지 않은 설명**이다 — 어텐션 웨이트를 꺼내 어느 자리에 몰려 있는지 본 것은 아니다.

잰 것은 결과다. 마스크 없이 트레인한 모델은 앞뒤 글자를 다 볼 수 있을 때에도 2.27에 머물렀고, 아래 표가 보이듯 뒤쪽을 가리면 더 나빠진다.

### 그래도 뒤쪽 글자에 의존하고 있었다

| 마스크 없이 트레인한 모델을 | 로스 |
|---|---|
| 제 설정 그대로(앞뒤 다 보며) 재면 | 2.2710 |
| **인과 마스크를 걸고** 재면 | **2.9006** |

**0.63이 뛴다.** 뒤를 못 보게 하자 로스가 그만큼 오른다는 것은, 그 모델의 예측이 **뒤쪽 글자에 의존하고 있었다**는 말이다. 완전히 베끼지는 못했어도 뒤쪽 정보를 일부 쓰고 있었다.

이 두 수가 이 쪽의 요점이다. **마스크는 모델을 약하게 만드는 제약이 아니라, 트레인할 때의 조건을 생성할 때의 조건과 같게 맞추는 장치다.** 생성할 때에는 뒤쪽 글자가 아직 없으므로 볼 수가 없다. 마스크 없이 트레인하면 트레인 때에는 있던 입력이 생성 때에는 없어지고, 그 불일치의 크기가 로스로 0.63이다.

---

## 4. 생성한 글을 보면 분명하다

수보다 글이 더 분명하다. 둘 다 `This movie`를 프롬프트(생성을 시작하는 입력)로 주고 240글자를 생성했다.

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

**아래쪽은 같은 조각을 되풀이한다.** `his`, `hish`, `hihi`가 끝없이 반복된다.

위쪽은 영어가 아니지만 **영어처럼 생겼다.** 띄어쓰기가 단어 길이에 맞고, 마침표 뒤에 대문자가 오고, `It is`, `like it`, `other on the`처럼 실제 단어 묶음이 나온다. `Wattch`, `beest`, `neanliatingly`는 없는 단어이지만 **영어 글자가 이어지는 규칙**은 따르고 있다.

!!! note "이 정도가 1.59 다"
    위 글이 「잘 생성한 것」은 아니다. 로스에 지수 함수를 씌운 $e^{\text{로스}}$를 **퍼플렉서티**(perplexity)라 하고, 모델이 다음 글자를 평균 몇 가지 가운데서 고르는 셈인지를 나타낸다. 로스 1.5902는 퍼플렉서티로 $e^{1.5902} \approx 4.9$로, **다음 글자를 다섯 가지쯤으로 좁혔다**는 뜻이다. 138가지에서 다섯으로 줄인 것이니 배우기는 배웠는데, 뜻이 통하는 글을 쓰려면 한참 멀다.

    파라메터가 141만 개이고 3,000 스텝을 돌렸다. 논문의 base 모델이 6,500만 개에 P100 여덟 장으로 12시간이었다. **[다음 쪽](03_sampling.md)은 같은 모델을 그대로 두고 디코딩 방법만 바꾸어** 얼마나 달라지는지 보고, **[그다음 쪽](04_scale.md)이 크기를 바꾼다.**

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span> `torch.triu(..., 1)`의 `1`을 `0`으로 바꾸면 무엇이 달라지는가. 모델이 무엇을 못 보게 되는지 말하라.

</div>

??? success "풀이"

    `triu(M, 1)`은 **주대각선 위쪽**만 남기고, `triu(M, 0)`은 **주대각선까지** 포함한다. $-\infty$로 채우는 자리가 늘어나는 것이므로, `0`으로 두면 자리 $t$가 **자기 자신도 보지 못한다.**

    그러면 자리 $t$의 어텐션이 가져오는 것은 $x_1, \dots, x_{t-1}$의 밸류뿐이다. $x_t$ 자신의 벡터는 잔차 연결(`x + a`)로 출력에 더해지므로 아주 못 쓰는 것은 아니지만, 어텐션 안에서 $x_t$와 앞 글자들을 함께 섞을 수는 없게 된다. 더 큰 문제는 첫 자리다. 첫 자리는 볼 것이 하나도 없어 소프트맥스가 전부 $-\infty$를 받고 그 자리의 출력이 **NaN**이 된다. 로스는 모든 자리의 평균이므로 로스도 NaN이 되어 트레인이 되지 않는다 — [7.5절](../imdb/05_attention.md)이 적어 둔 그 사고와 같은 형태다.

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span> 3절은 마스크 없는 모델이 「완전히 베끼지는 못했다」고 했다. 정말 베낄 수 **있는지**를 가리는 실험을 설계하라.

</div>

??? success "풀이"

    베끼는 데 필요한 것은 내용이 아니라 **위치만으로 어텐션 웨이트를 정하는 것**이다. 그러니 그것을 쉽게 만들어 주고 다시 보면 된다.

    **(가) 더 오래 돌린다.** 3,000 스텝이 모자랐을 뿐일 수 있다. 3만 스텝까지 늘리며 트레인 로스를 기록해 0 쪽으로 내려가는지 본다.

    **(나) 위치 정보를 세게 준다.** 위치 임베딩의 크기를 키우거나, 어텐션 점수에 상대 위치 항을 직접 더해 준다. 「바로 다음 자리」를 집기 쉬워진다.

    **(다) 손으로 심어 본다.** 트레인하지 말고, 자리 $t$가 자리 $t+1$에만 어텐션 웨이트 1을 주도록 웨이트를 직접 짜 넣은 뒤 로스를 잰다. 그것이 0에 가깝게 나오면 **베끼는 해가 존재한다**는 것이 확인되고, 그러면 3절의 결과는 「못 하는 것」이 아니라 「못 찾은 것」이 된다.

    (다)가 가장 값싸다. 트레인이 없으니 계산이 금방 끝난다. 다만 (다)가 보이는 것은 **해가 있다**는 것까지이고, 경사 하강법이 그 해를 **찾는지**는 (가)와 (나)가 답한다.

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span> 이 쪽은 글자 단위로 다루었다. [7장](../imdb/index.md)처럼 **단어 단위**로 바꾸면 무엇이 좋아지고 무엇이 나빠지는가. 로스 값을 두 설정에서 **직접 견줄 수 있는가?**

</div>

??? success "풀이"

    **좋아지는 것** — 토큰 하나가 담는 뜻이 커진다. 글자 128개는 영어로 스무 단어 남짓인데, 단어 128개면 같은 문맥 길이로 훨씬 긴 글을 담는다. 철자를 배우는 데 쓰던 모델 용량도 아낀다.

    **나빠지는 것** — 어휘가 138에서 2만으로 커지니 임베딩과 출력 레이어가 그만큼 커진다. 이 쪽 모델은 141만 개 가운데 출력 레이어가 $192 \times 138 = 26{,}496$인데, 단어 단위면 $192 \times 20{,}000 = 3{,}840{,}000$으로 모델 전체보다 커진다. 그리고 어휘에 없는 단어(OOV)를 아예 쓸 수 없다.

    **직접 견줄 수 없다.** 로스는 「한 토큰을 맞히는」 교차 엔트로피인데 토큰이 다른 것이기 때문이다. 글자 단위의 로스 1.59와 단어 단위에서 얻은 로스를 나란히 놓고 어느 쪽이 나은지 말할 수 없다. 견주려면 **같은 단위로 환산**해야 한다 — 문장 하나의 전체 로그 가능도를 **글자 수로 나눈 값**(bits per character)으로 맞추는 것이 보통 쓰는 길이다. 이 쪽의 코드는 그 환산을 하지 않으므로, 여기 적힌 1.59는 **글자 단위 안에서만** 뜻이 있다.
