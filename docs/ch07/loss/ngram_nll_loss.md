# N-그램으로 보는 NLL 로스

`CrossEntropyLoss`와 `NLLLoss`는 서로 다른 로스가 아니다. **로그 소프트맥스를 어디서 거느냐**만 다르다 — 하나는 로스 안에서 걸고, 하나는 모델 안에서 건다.

$$\mathrm{CrossEntropyLoss}(\text{로짓}, y) \;=\; \mathrm{NLLLoss}(\mathrm{log\_softmax}(\text{로짓}), y)$$

이 쪽은 그 등식을 수로 확인하고, N-그램 말 모델을 세 API 로 각각 학습시켜 로스가 정말 같은지 본다. 그리고 `log(softmax(x))` 가 아니라 `log_softmax(x)` 여야 하는 까닭을 재어서 본다.

## 1. 코드

```python
"""2.8 N-그램으로 보는 NLL 로스 — 같은 로스를 부르는 세 가지 이름.

CrossEntropyLoss 와 NLLLoss 는 서로 다른 로스가 아니다. 하나는 로그
소프트맥스를 로스 안에서 걸고, 하나는 모델 안에서 건다. 걸리는 자리만
다를 뿐 값은 같다.

  CrossEntropyLoss(logits, y) = NLLLoss(log_softmax(logits), y)

재는 것 셋.
  (가) 식 한 줄로 끝나는 그 등식을 수로 확인한다
  (나) N-그램 말 모델을 세 API 로 각각 학습시켜 로스가 같은지 본다
       — 세 모델의 처음 웨이트를 같은 씨앗으로 맞춘다. 맞추지 않으면
         차이가 API 탓인지 초기값 탓인지 가릴 수 없다
  (다) 왜 log(softmax(x)) 가 아니라 log_softmax(x) 여야 하는지 재어서 본다

자료는 셰익스피어 소네트 2번의 앞부분이다. 작지만 되풀이가 있어 로스가
내려가는 것이 눈에 보인다.
실행: 이 쪽의 코드를 파일로 갈무리해 저장소 뿌리에서 돌린다.
"""
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

ARGS = SimpleNamespace(context_size=2, embedding_dim=10, hidden_dim=128,
                       lr=0.001, epochs=10, seed=42)

TEST_SENTENCE = """When forty winters shall besiege thy brow,
And dig deep trenches in thy beauty's field,
Thy youth's proud livery so gazed on now,
Will be a tattered weed of small worth held:
Then being asked where all thy beauty lies,
Where all the treasure of thy lusty days;
To say within thine own deep sunken eyes,
Were an all-eating shame and thriftless praise.""".split()

# (앞 단어 context_size 개, 다음 단어) 짝
NGRAMS = [([TEST_SENTENCE[i - j - 1] for j in range(ARGS.context_size)],
           TEST_SENTENCE[i])
          for i in range(ARGS.context_size, len(TEST_SENTENCE))]
VOCAB = sorted(set(TEST_SENTENCE))
WORD_TO_IX = {w: i for i, w in enumerate(VOCAB)}
ARGS.vocab_size = len(VOCAB)


# === 모델 둘 — 내놓는 것만 다르다 ==========================================
class NGramLogits(nn.Module):
    """날 로짓을 내놓는다. CrossEntropyLoss 에 그대로 넣는다."""

    def __init__(self):
        super().__init__()
        self.embeddings = nn.Embedding(ARGS.vocab_size, ARGS.embedding_dim)
        self.linear1 = nn.Linear(ARGS.context_size * ARGS.embedding_dim,
                                 ARGS.hidden_dim)
        self.linear2 = nn.Linear(ARGS.hidden_dim, ARGS.vocab_size)

    def forward(self, inputs):
        e = self.embeddings(inputs).view(inputs.shape[0], -1)
        return self.linear2(F.relu(self.linear1(e)))


class NGramLogProbs(NGramLogits):
    """로그 확률을 내놓는다. NLLLoss 에 넣는다.

    위 모델과 다른 것은 마지막 한 줄뿐이다.
    """

    def forward(self, inputs):
        return F.log_softmax(super().forward(inputs), dim=1)


def make_batch():
    """n-그램 전부를 텐서 한 쌍으로. 자료가 작아 배치를 나누지 않는다."""
    ctx = torch.tensor([[WORD_TO_IX[w] for w in c] for c, _ in NGRAMS],
                       dtype=torch.long)
    tgt = torch.tensor([WORD_TO_IX[t] for _, t in NGRAMS], dtype=torch.long)
    return ctx, tgt


def train(model_cls, loss_fn, epochs=None):
    """씨앗을 박고 모델을 새로 지어 학습한다.

    「씨앗을 모델마다 다시 박는 것」이 이 쪽의 핵심이다. 세 API 를 견주려면
    처음 웨이트가 같아야 한다. 다르면 마지막 로스의 차이가 API 탓인지
    초기값 탓인지 가릴 수 없다.
    """
    epochs = epochs or ARGS.epochs
    torch.manual_seed(ARGS.seed)
    model = model_cls()
    opt = torch.optim.SGD(model.parameters(), lr=ARGS.lr)
    ctx, tgt = make_batch()
    losses = []
    for _ in range(epochs):
        opt.zero_grad()
        loss = loss_fn(model(ctx), tgt)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses


if __name__ == "__main__":
    print(f"  단어 {len(TEST_SENTENCE)}개, 어휘 {ARGS.vocab_size}개, "
          f"n-그램 {len(NGRAMS)}개, 문맥 {ARGS.context_size}")

    # === (가) 등식을 수로 확인한다 =======================================
    print("\n  (가) CrossEntropyLoss = LogSoftmax + NLLLoss")
    logits = torch.tensor([[2.0, 5.0, 1.0]])
    target = torch.tensor([1])
    log_probs = F.log_softmax(logits, dim=1)

    ce = F.cross_entropy(logits, target)
    nll = F.nll_loss(log_probs, target)
    print(f"    로짓            {logits[0].tolist()}")
    print(f"    로그 확률        "
          f"{[round(v, 6) for v in log_probs[0].tolist()]}")
    print(f"    정답은 {target.item()}번이고 그 자리의 로그 확률은 "
          f"{log_probs[0, target].item():.6f}")
    print(f"    cross_entropy   {ce.item():.6f}")
    print(f"    nll_loss        {nll.item():.6f}")
    print(f"    차이            {abs(ce - nll).item():.2e}")
    print(f"    NLL 은 「정답 자리의 로그 확률에 음수를 붙인 것」이 전부다")

    # === (나) 세 API 를 같은 출발점에서 =================================
    print("\n  (나) 같은 씨앗으로 지어 세 API 로 학습한다")
    runs = [("nn.CrossEntropyLoss", NGramLogits, nn.CrossEntropyLoss()),
            ("F.cross_entropy", NGramLogits, F.cross_entropy),
            ("nn.NLLLoss + log_softmax", NGramLogProbs, nn.NLLLoss())]
    curves = {name: train(cls, fn) for name, cls, fn in runs}

    print(f"    {'에포크':>6}" + "".join(f"{n:>26}" for n in curves))
    for e in range(ARGS.epochs):
        print(f"    {e + 1:>6}"
              + "".join(f"{curves[n][e]:>26.10f}" for n in curves))

    base = curves["nn.CrossEntropyLoss"]
    print(f"\n    마지막 로스가 기준선과 얼마나 다른가")
    for n, c in curves.items():
        print(f"      {n:<26} {abs(c[-1] - base[-1]):.3e}")
    print(f"    셋이 소수점 아래 끝까지 같다. 같은 로스이기 때문이다 —")
    print(f"    달랐다면 그것은 API 가 아니라 초기값이 다르다는 뜻이다")

    # === (다) 왜 log_softmax 인가 ======================================
    print("\n  (다) log(softmax(x)) 와 log_softmax(x)")
    print(f"    먼저 흔한 오해 하나 — 로짓이 크다고 문제가 되지는 않는다.")
    for v in ([1000.0, 1001.0], [0.0, 1.0]):
        p = F.softmax(torch.tensor([v]), dim=1)[0]
        print(f"      softmax({v})  ->  "
              f"[{p[0]:.4f}, {p[1]:.4f}]")
    print(f"    소프트맥스는 같은 상수를 빼도 변하지 않으므로 둘이 같다.")
    print(f"    문제가 되는 것은 크기가 아니라 「차이」다.")

    big = torch.tensor([[0.0, -800.0]])
    p = F.softmax(big, dim=1)
    print(f"\n      logits          {big[0].tolist()}")
    print(f"      softmax         [{p[0, 0]:.1f}, {p[0, 1]:.1f}]"
          f"   <- 둘째 칸이 0 으로 내려앉았다")
    print(f"      log(softmax)    "
          f"[{torch.log(p)[0, 0]:.1f}, {torch.log(p)[0, 1]:.1f}]"
          f"   <- 0 의 로그라 -inf 다")
    ls = F.log_softmax(big, dim=1)
    print(f"      log_softmax     [{ls[0, 0]:.1f}, {ls[0, 1]:.1f}]"
          f"   <- 참값을 그대로 돌려준다")
    print(f"\n    확률을 먼저 만들면 작은 쪽이 0 으로 내려앉고, 그 뒤에 로그를")
    print(f"    취하면 되돌릴 수 없다. log_softmax 는 확률을 거치지 않는다.")
```

**출력:**

```
  단어 63개, 어휘 57개, n-그램 61개, 문맥 2

  (가) CrossEntropyLoss = LogSoftmax + NLLLoss
    로짓            [2.0, 5.0, 1.0]
    로그 확률        [-3.065884, -0.065884, -4.065884]
    정답은 1번이고 그 자리의 로그 확률은 -0.065884
    cross_entropy   0.065884
    nll_loss        0.065884
    차이            0.00e+00
    NLL 은 「정답 자리의 로그 확률에 음수를 붙인 것」이 전부다

  (나) 같은 씨앗으로 지어 세 API 로 학습한다
       에포크       nn.CrossEntropyLoss           F.cross_entropy  nn.NLLLoss + log_softmax
         1              4.0997972488              4.0997972488              4.0997972488
         2              4.0994930267              4.0994930267              4.0994930267
         3              4.0991878510              4.0991878510              4.0991878510
         4              4.0988836288              4.0988836288              4.0988836288
         5              4.0985789299              4.0985789299              4.0985789299
         6              4.0982742310              4.0982742310              4.0982742310
         7              4.0979695320              4.0979695320              4.0979695320
         8              4.0976657867              4.0976657867              4.0976657867
         9              4.0973615646              4.0973615646              4.0973615646
        10              4.0970568657              4.0970568657              4.0970568657

    마지막 로스가 기준선과 얼마나 다른가
      nn.CrossEntropyLoss        0.000e+00
      F.cross_entropy            0.000e+00
      nn.NLLLoss + log_softmax   0.000e+00
    셋이 소수점 아래 끝까지 같다. 같은 로스이기 때문이다 —
    달랐다면 그것은 API 가 아니라 초기값이 다르다는 뜻이다

  (다) log(softmax(x)) 와 log_softmax(x)
    먼저 흔한 오해 하나 — 로짓이 크다고 문제가 되지는 않는다.
      softmax([1000.0, 1001.0])  ->  [0.2689, 0.7311]
      softmax([0.0, 1.0])  ->  [0.2689, 0.7311]
    소프트맥스는 같은 상수를 빼도 변하지 않으므로 둘이 같다.
    문제가 되는 것은 크기가 아니라 「차이」다.

      logits          [0.0, -800.0]
      softmax         [1.0, 0.0]   <- 둘째 칸이 0 으로 내려앉았다
      log(softmax)    [0.0, -inf]   <- 0 의 로그라 -inf 다
      log_softmax     [0.0, -800.0]   <- 참값을 그대로 돌려준다

    확률을 먼저 만들면 작은 쪽이 0 으로 내려앉고, 그 뒤에 로그를
    취하면 되돌릴 수 없다. log_softmax 는 확률을 거치지 않는다.
```

---

## 2. 세 이름이 한 로스다

출력 (나)의 표에서 세 열이 **소수점 아래 열째 자리까지 같다.** 마지막 로스의 차이는 셋 다 `0.000e+00`이다.

이것이 당연해 보이지만, **처음 웨이트를 맞추지 않으면 볼 수 없는 결과**다. 세 모델을 따로 만들면 초기값이 달라 로스도 달라지고, 그 차이를 API 탓으로 읽게 된다. 그래서 `train()`은 모델을 받지 않고 **클래스를 받아 씨앗을 박고 직접 짓는다.**

```python
def train(model_cls, loss_fn, epochs=None):
    torch.manual_seed(ARGS.seed)     # <- 모델마다 다시 박는다
    model = model_cls()
```

바꾼 것이 하나뿐이어야 그 하나의 몫을 읽을 수 있다. 여기서 바꾼 하나는 **로스 API**이고, 답은 「아무것도 바뀌지 않는다」이다.

!!! note "로스가 거의 안 내려간 것은 따로 보아야 한다"
    4.0998에서 4.0971로, 열 에포크에 0.003밖에 안 내려갔다. 학습률이 0.001이고 에포크가 열이니 그럴 수밖에 없다.

    **이 쪽이 묻는 것은 모델이 잘 배우는가가 아니다.** 세 API 가 같은 값을 내는가다. 그 물음에는 로스가 내려가든 말든 답할 수 있고, 오히려 적게 움직이는 편이 세 열을 맞대기에 낫다.

### 어느 쪽을 쓸까

| | 모델이 내놓는 것 | 로스 |
|---|---|---|
| `nn.CrossEntropyLoss` · `F.cross_entropy` | 날 로짓 | 로스가 속에서 `log_softmax`를 건다 |
| `nn.NLLLoss` | 로그 확률 | 모델이 이미 걸어 둔 것을 받는다 |

분류라면 **앞의 것을 기본으로 둔다.** 모델에 활성화를 걸지 않아도 되니 「소프트맥스를 두 번 걸었다」는 흔한 실수가 생기지 않는다.

`NLLLoss`가 필요한 자리는 **로그 확률을 로스 말고 다른 데도 쓸 때**다. 표본을 뽑거나, 확률을 찍어 보거나, 다른 분포와 견주려면 모델이 그것을 내놓는 편이 낫다.

---

## 3. `log_softmax` 를 쓰는 까닭

흔히 「로짓이 크면 소프트맥스가 터진다」고 말하는데 **그렇지 않다.** 출력 (다)의 첫 두 줄이 그것을 보인다.

```
softmax([1000.0, 1001.0])  ->  [0.2689, 0.7311]
softmax([0.0, 1.0])        ->  [0.2689, 0.7311]
```

소프트맥스는 **모든 값에서 같은 상수를 빼도 변하지 않는다.** $1000$이든 $0$이든 차이가 $1$이면 결과가 같다. 구현도 그것을 이용해 최댓값을 먼저 빼 두므로 큰 로짓 자체는 문제가 아니다.

**문제는 크기가 아니라 차이다.**

```
logits          [0.0, -800.0]
softmax         [1.0, 0.0]        <- 둘째 칸이 0 으로 내려앉았다
log(softmax)    [0.0, -inf]       <- 0 의 로그라 -inf 다
log_softmax     [0.0, -800.0]     <- 참값을 그대로 돌려준다
```

$e^{-800}$은 배정밀도 실수로 표현할 수 있는 가장 작은 수보다 작아 **0이 된다.** 확률을 한 번 만들고 나면 그 0은 되돌릴 수 없고, 로그를 취하면 $-\infty$다. 로스가 $\infty$가 되고 기울기도 날아간다.

`log_softmax`는 **확률을 거치지 않는다.** $\log \frac{e^{x_i}}{\sum_j e^{x_j}} = x_i - \log\sum_j e^{x_j}$ 로 바꾸어 계산하므로, 언더플로가 일어날 자리를 아예 지나지 않는다. 답으로 나온 $-800$이 참값이다.

**그러니 `torch.log(F.softmax(x))` 는 쓰지 않는다.** 결과가 같아 보이는 자리에서도 쓰지 않는 편이 낫다 — 언제 차이가 벌어질지는 자료가 정하지 코드가 정하지 않는다.

---

## 4. 이 실험이 말해 주지 않는 것

**장기 의존성이나 언어 이해를 잰 것이 아니다.** 여기 있는 것은 앞 단어 두 개를 보고 다음 단어를 맞히는 **고정 길이 문맥의 순방향 모델**이다. 순환 상태도 어텐션도 없다.

**일반화를 잰 것도 아니다.** 소네트 한 토막 63단어를 통째로 학습에 쓰고 그 위에서 로스를 쟀다. 처음 보는 텍스트에서 어떤지는 이 수로 알 수 없다. 재려면 따로 떼어 둔 텍스트에서 단어당 음의 로그 가능도를 재야 한다.

이 쪽이 말해 주는 것은 좁다 — **세 API 가 같은 로스이고, 로그 확률은 `log_softmax`로 만들어야 한다.** 그 둘뿐이다.

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
어휘 크기 $V$, 임베딩 차원 $d$, 문맥 크기 $C$일 때 이 모델의 파라메터 수를 구하라.

</div>

??? success "풀이"
    임베딩은 $Vd$개, 첫 선형 레이어는 $128Cd+128$개, 출력 레이어는 $128V+V$개다. 합계는 $Vd+128Cd+129V+128$이다. 기본 $C=2,d=10$에서는 $139V+2688$개다. ReLU와 로그 소프트맥스는 파라메터를 추가하지 않는다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
모델 입력과 NLLLoss 정답에 필요한 형태·자료형·값의 범위를 설명하라.

</div>

??? success "풀이"
    모델 입력은 $(B,C)$ 형태의 정수 인덱스이며 각 값은 $0$ 이상 $V$ 미만이어야 한다. 이 예제에서는 `torch.long`을 사용한다. 정답은 $(B,)$ 형태의 long 텐서이며 클래스 범위에 있어야 한다. 로스에 전달하는 모델 출력은 $(B,V)$의 로그 확률이다. 입력을 float 이미지 텐서로 검사하는 것은 이 모델에 맞지 않는다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
로짓을 NLLLoss에 바로 넣는 경우와 학습 로스만으로 일반화를 판단하는 경우의 문제를 설명하라.

</div>

??? success "풀이"
    NLLLoss는 입력을 로그 확률로 해석하므로 정규화되지 않은 로짓을 넣으면 의도한 음의 로그 우도가 아니다. 출력에 log_softmax가 있는지 확인한다. 또한 작은 말뭉치의 학습 로스가 감소해도 처음 보는 문장에 대한 성능은 알 수 없으므로 분리된 검증 데이터에서 토큰별 음의 로그 우도 등을 평가한다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
아래 풀이의 `ModuleList` 코드가 원래 모델에 바로 적용될 수 있는지 검토하라. 입력·출력 차원, 활성화, 생성자와 `forward`에서 필요한 수정을 설명하라.

</div>

??? success "풀이"
    아래 코드는 깊이를 설정하는 아이디어를 위한 조각이다. 원래 모델의 생성자와 텐서 차원을 반영한 완성 구현은 아니다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    `ModuleList`는 하위 모듈을 등록하지만 실행 순서나 활성화를 자동으로 정하지 않는다. 입력 레이어와 출력 레이어는 과제의 차원을 유지하고, 은닉 레이어 사이에 필요한 비선형성을 넣어야 한다. 선형 레이어만 반복하면 아핀 함수 하나로 합쳐진다. 원래 클래스에 없는 `num_layers` 인수를 그대로 넘길 수 없으므로 생성자와 `forward`를 함께 수정해야 한다.

## 정리하며

`NLLLoss`는 로그 확률을 받고 `CrossEntropyLoss`는 날 로짓을 받는다. 그 차이뿐이고, 모델과 로스의 규약을 맞추면 **같은 목적 함수가 된다** — 이 쪽에서 소수점 아래 열째 자리까지 같았다.

그 결론이 나오려면 **처음 웨이트를 맞춰야 했다.** 맞추지 않은 비교는 API 의 차이가 아니라 초기값의 차이를 재게 된다. 하나만 바꾸는 규율이 여기서도 그대로 걸린다.

수치 안정성은 **로짓의 크기가 아니라 차이**의 문제다. 소프트맥스는 상수 이동에 변하지 않으므로 큰 로짓은 해롭지 않고, 차이가 벌어져 작은 확률이 0으로 내려앉는 것이 해롭다. `log_softmax`는 확률을 거치지 않아 그 자리를 지나간다.
