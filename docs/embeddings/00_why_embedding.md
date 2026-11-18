# 왜 임베딩인가 — 픽셀값은 잰 값이고 글자 코드는 약속이다

<div class="leadbox" markdown>

**들어가는 이야기** — 좋음 더하기 나쁨은 「off」다

앞의 세 장은 그림을 다루었다. 그림은 **처음부터 수였다** — 픽셀 하나가 0에서 255 사이의 수이고, 784개를 늘어놓으면 그대로 벡터다. 더하고 빼고 평균 낼 수 있었다.

글도 컴퓨터 안에서는 수다. 글자마다 코드가 있고, 단어에 번호를 매길 수도 있다. 영화평 2만 5천 편에 나오는 단어를 흔한 순서로 줄 세우면 `good`이 **47번**, `bad`가 **73번**이 된다. 그러면 물어보자 — **47 더하기 73은 무엇인가?**

120번이고, 거기 있는 단어는 **`off`**다. 좋음과 나쁨을 더했더니 「꺼짐」이 나왔다.

픽셀은 더하면 뜻이 남는데 번호는 왜 안 될까. 둘 다 수인데 무엇이 다른가. 그리고 이 번호를 **그대로 뉴럴 네트워크에 넣고 학습시키면** 네트워크가 알아서 뜻을 찾아내지 않을까 — 이 쪽은 그것을 재어 본다. 같은 영화평, 같은 학습 예산에서 번호를 그대로 넣은 모델은 긍정·부정을 **52%** 맞힌다. 동전 던지기가 50%다.

</div>

이 쪽이 6장의 출발점이다. 결론을 먼저 적으면, **픽셀값은 무언가를 잰 값이고 글자 코드와 단어 번호는 사람들이 정한 약속이다.** 잰 값에는 크기와 차이와 평균에 뜻이 있고, 약속에는 「같은가, 다른가」밖에 뜻이 없다. 그래서 단어에는 수를 **가져다 쓰지 않고 새로 정해 주어야** 하고, 그 새로 정한 수가 **임베딩**(embedding)이다 — 단어마다 수 여러 개(벡터)를 주되, 뜻이 가까운 단어끼리 가까워지도록 그 수를 배우는 것이다.

---

## 1. 그림의 수는 잰 값이다

MNIST의 픽셀값은 **그 칸에 잉크가 얼마나 묻었는가**를 잰 것이다. 0이 종이, 1이 가장 진한 잉크다. 잰 값이므로 셈에 뜻이 있다. 0.6은 0.3보다 두 배 진하고, 두 값의 평균은 그 사이의 진하기다.

그러니 그림에 셈을 해도 그림이 남는다. [5.1절](../ch05/dimreduction/index.md)의 평가 분류기 — 3장에서 만든, 시험 그림을 98.83% 맞히는 CNN — 에게 셈을 한 그림을 보여 주면 이렇다(아래 프로그램의 (가)).

| 그림에 한 일 | 같은 숫자로 읽는 비율 |
|---|---|
| 그대로 | 98.83% |
| 픽셀마다 잡음 (표준편차 0.2) | 98.43% |
| 픽셀마다 잡음 (표준편차 0.3) | 95.84% |
| 잉크를 0.5배로 옅게 | 98.51% |
| **같은 숫자 두 장을 픽셀마다 평균** | **99.61%** |
| 다른 숫자 두 장을 평균 | 89.38% (둘 가운데 하나로) |

**값을 조금 흔들면 그림도 조금 바뀐다.** 잡음을 얹어도, 잉크를 옅게 해도 숫자는 그대로다. 같은 숫자 두 장을 평균하면 오히려 더 잘 읽힌다. 서로 다른 숫자 둘을 평균하면 그 사이의 그림이 되고, 분류기는 열에 아홉 그 둘 가운데 하나로 읽는다. 아래 그림의 왼쪽 위 두 줄이 그것이다.

이 성질이 3·4·5장의 모든 것을 떠받쳤다. 경사 하강법은 웨이트를 **조금** 바꾸면 출력이 **조금** 바뀐다는 데 기대고, 컨볼루션은 이웃한 픽셀이 비슷한 값이라는 데 기대고, [5장의 PCA](../ch05/dimreduction/01_pca.md)와 오토인코더는 그림을 벡터로 보고 평균과 거리를 재는 데 기댄다. 모두 **수의 크기에 뜻이 있어야** 성립한다.

![칸이 셋인 그림. 왼쪽 위는 MNIST의 3 두 장과 그 픽셀 평균으로, 평균도 3으로 보인다. 그 아래는 3 한 장과 8 한 장과 그 평균으로, 평균은 둘이 겹친 그림이다. 왼쪽 아래는 cat의 아스키 코드 99, 97, 116과 dog의 100, 111, 103을 글자마다 평균한 100, 104, 110이 dhn이라는 뜻 없는 글자가 된다는 것이다. 오른쪽은 IMDB 시험 정확도 막대로, 단어 번호를 수 그대로 넣은 모델이 0.523에서 0.527, 번호를 섞으면 0.508에서 0.514, 빈칸인지만 넣으면 0.503에서 0.510으로 모두 0.5 점선 바로 옆에 있고, 임베딩 모델은 0.838에서 0.844이며 번호를 섞어도 똑같이 0.838에서 0.844다](figures/why_embedding.svg)

## 2. 글자의 수는 약속이다

글자도 컴퓨터 안에서는 수다. 그런데 그 수는 **무엇을 잰 것이 아니라 정한 것**이다. 1960년대에 미국의 표준 위원회가 `A`를 65로 정했고, 같은 시기 IBM은 자기 기계에서 `A`를 193으로 정했다. 둘 다 옳다. 약속이기 때문이다. (나)가 몇 가지 약속을 나란히 놓는다.

| 글자 | UTF-8 | EUC-KR | UTF-16 | EBCDIC |
|---|---|---|---|---|
| `A` | 41 | 41 | 00 41 | C1 |
| `가` | EA B0 80 | B0 A1 | AC 00 | — |

(16진수다. 41은 65, C1은 193이다.)

**뜻은 하나인데 수는 약속마다 다르다.** 픽셀에는 이런 일이 없다. 진하기를 0~1로 적든 0~255로 적든 한쪽이 다른 쪽의 255배일 뿐이라, 순서도 비율도 그대로다. 글자 코드는 약속을 바꾸면 순서조차 바뀐다.

그러니 코드에 셈을 하면 뜻이 남지 않는다.

| 셈 | 결과 |
|---|---|
| `boy` 의 글자마다 +1 | `cpz` |
| `good` 의 글자마다 +1 | `hppe` |
| `cat` 과 `dog` 를 글자마다 평균 | `dhn` |
| `good` 과 `bad` 를 글자마다 평균 | `dhj` |

그림 둘의 평균은 「그 사이의 그림」이었지만, 단어 둘의 평균은 **아무 단어도 아니다.**

### 한글 코드에는 구조가 있다 — 그런데 철자의 구조다

한글 음절의 유니코드는 아무렇게나 매긴 것이 아니라 공식으로 짜여 있다.

$$
\text{코드} = \texttt{0xAC00} + (\text{초성} \times 21 + \text{중성}) \times 28 + \text{종성}
$$

초성 19개, 중성 21개, 종성 28가지(받침 없음 포함)를 차례로 늘어놓은 것이다. 그래서 셈이 **어느 정도 통한다.** `가`에 1을 더하면 받침 ㄱ이 붙은 `각`이고, 28을 더하면 모음이 바뀐 `개`, 588을 더하면 초성이 바뀐 `까`, 1,176을 더하면 `나`다.

그런데 이 구조는 **철자의 구조이지 뜻의 구조가 아니다.** `가`와 `각`이 1 차이인 것은 받침 하나 차이이기 때문이지 뜻이 가까워서가 아니다. 다음 절의 아스키 거리와 같은 이야기다.

### 코드 사이의 거리는 철자를 잰다

그래도 글자 코드를 그대로 써 보고 싶을 수 있다. `boy`는 `b`, `o`, `y`이고 그것은 98, 111, 121이다. 여덟 자리 이진수로 적으면 이렇다.

```
boy   01100010 01101111 01111001
```

그러면 `boy`는 **픽셀 셋짜리 그림**이다. 앞의 세 장이 784픽셀짜리 그림을 다루었으니, 3픽셀짜리 그림쯤이야 똑같이 다루면 될 것 같다. (다)가 네 단어를 여덟 글자 칸에 넣고(짧으면 0으로 채운다) 사이 거리를 잰다.

![칸이 둘인 그림. 왼쪽은 boy, coy, toy, girl, boys 를 각각 여덟 칸짜리 줄로 그린 것으로 칸마다 아스키 코드가 적혀 있고 값에 따라 명암이 다르다. boy 와 coy 는 첫 칸만 98 과 99 로 거의 같아 보이고 boys 는 네 번째 칸에 115 가 더 붙어 있다. 오른쪽은 boy 에서의 거리를 가로로 세운 사다리로, coy 가 1.0 으로 가장 가깝고 toy 가 18.0, girl 이 108.5, 같은 단어의 복수형인 boys 가 115.0 으로 가장 멀다](figures/ascii_is_spelling.svg)

| 짝 | 아스키 거리 | 뜻으로는 |
|---|---|---|
| `boy` – `coy` | **1.0** | 아무 상관이 없다 (coy = 새침한) |
| `boy` – `toy` | 18.0 | 상관이 없다 |
| `boy` – `girl` | 108.5 | 가장 가까운 말이다 |
| `boy` – `boys` | **115.0** | **같은 단어의 복수형이다** |

**줄을 거꾸로 세운 표다.** 가장 가까운 것이 `coy`인데 뜻으로는 남남이고, 가장 먼 것이 `boys`인데 사실상 같은 단어다. `boy`와 `coy`가 1인 까닭은 첫 글자가 98과 99로 이웃이기 때문이고, `boy`와 `boys`가 115인 까닭은 네 번째 칸이 0에서 115(`s`)로 뛰었기 때문이다.

곧 **아스키 거리는 철자를 잰다.** 철자가 뜻을 담고 있다면 쓸 수 있겠지만 담고 있지 않다. 글자 하나만 바꾸어도 뜻이 통째로 달라지고(`boy`/`coy`), 글자 하나를 붙여도 뜻은 그대로다(`boy`/`boys`).

!!! note "그래도 철자가 쓸모없지는 않다"
    [fastText 쪽](06_fasttext.md)이 철자를 다시 꺼내 쓴다. 다만 거기서도 **글자를 그대로 수로 읽지 않는다** — 단어를 글자 n-그램으로 쪼갠 뒤 **조각마다 벡터를 배워** 그것들을 평균 낸다. 아스키 코드는 주어진 수이고 조각 벡터는 배운 수라는 것이 갈림길이다.

    그래서 그 쪽의 이웃 표에는 `movie` 곁에 `movie's`가, `she` 곁에 `she's`·`she'd`가 올라온다 — 글자를 나눠 쓰는 말들이다. **다만 같은 수법이 `funny` 곁에 `unfunny`를, `good` 곁에 `goodbye`를 올려 놓기도 한다.** 철자를 배워 쓰는 것이 철자를 그냥 읽는 것보다 나을 뿐, 뜻을 보장하지는 않는다.

## 3. 수의 네 가지 성격

통계학은 수가 **무엇을 허락하는가**에 따라 네 가지 척도로 나눈다(Stevens, 1946). 위의 두 절을 그 말로 정리하면 이렇다.

| 척도 | 뜻이 있는 셈 | 보기 |
|---|---|---|
| **명목** | 같은가, 다른가 | 글자 코드, 단어 번호, 우편번호 |
| **순서** | + 크고 작음 | 등수, 별점 |
| **구간** | + 차이 | 섭씨온도 |
| **비율** | + 몇 배인가, 평균 | 길이, 무게, **픽셀의 진하기** |

뉴럴 네트워크는 입력을 **비율 척도처럼** 다룬다. 웨이트를 곱하고, 더하고, 평균 낸다. 픽셀은 그 대접이 맞는 수다. 글자 코드와 단어 번호는 **명목 척도**라서 그 대접이 틀리다. 명목 척도의 수에 웨이트를 곱하는 것은 우편번호에 2를 곱하는 것과 같다.

## 4. 단어 번호도 이름이다

글자 대신 단어에 번호를 매겨도 마찬가지다. (라)가 영화평 2만 5천 편의 단어를 흔한 순서로 줄 세운다.

```
 2340  relatively      2343  danger
 2341  teenagers       2344  walks
 2342  haunted         2345  creatures
```

**여섯이 모두 213번씩 나온 단어들**이다. 같은 횟수라 순서를 매길 근거가 없어 아무렇게나 줄 세운 것뿐이다. 번호가 이웃이라고 뜻이 이웃이지 않다. 그리고 들머리에서 본 대로 `good`(47) + `bad`(73) = `off`(120)다. **번호는 이름이지 크기가 아니다.**

다만 이 책의 번호는 **완전한 약속은 아니다.** 흔한 순서로 매겼으므로 번호가 작을수록 흔한 단어다. 곧 번호가 **빈도의 순위**라는 진짜 양을 조금 품고 있다. 다음 절이 그 몫을 따로 떼어 잰다.

## 5. 번호를 그대로 넣고 배우면

그래도 뉴럴 네트워크는 힘이 세다. 번호를 그대로 넣어도 웨이트가 알아서 쓸모 있는 것을 뽑아내지 않을까. (마)가 IMDB 영화평 2만 5천 편으로 학습하고 다른 2만 5천 편으로 긍정·부정 정확도를 잰다. 두 모델은 학습 예산(5 에포크, Adam $10^{-3}$, 배치 128)이 같다.

- **번호 그대로** — 평의 앞 200단어 번호를 어휘 크기로 나눠 200칸짜리 입력으로 넣는다. 그 뒤는 64칸 은닉층 하나짜리 MLP다.
- **임베딩** — 번호로 2만 × 64짜리 표에서 줄을 꺼내 평균 내고, 선형 레이어 하나로 판정한다. 표 자체를 배운다. 표가 어떻게 생겼고 왜 「꺼낸다」가 곱셈인지는 [다음 쪽](01_not_a_number.md)이 다룬다.

그리고 둘 다 **번호 매기는 약속만 무작위로 바꿔서** 다시 학습한다. 단어도, 평도, 라벨도 그대로이고 `good`이 47번이던 것이 다른 번호가 될 뿐이다. 씨앗 셋으로 잰다.

| | 번호 그대로 | 번호를 섞으면 | 임베딩 | 번호를 섞으면 |
|---|---|---|---|---|
| 씨앗 0 | 0.5275 | 0.5084 | 0.8438 | **0.8438** |
| 씨앗 1 | 0.5228 | 0.5142 | 0.8376 | **0.8376** |
| 씨앗 2 | 0.5247 | 0.5098 | 0.8382 | **0.8382** |

읽을 것이 셋이다.

**번호 그대로는 동전 던지기를 겨우 넘는다.** 0.523~0.528이다. 번호에서 크기의 뜻을 찾으라고 했는데 찾을 것이 없다.

**그 겨우 넘는 몫마저 약속에서 온다.** 번호를 섞으면 0.508~0.514로 내려간다. 섞기 전에 더 있던 0.01~0.02는 4절에서 말한 빈도 순위였다 — 「흔한 단어가 많은 평」 같은 것을 조금 잡은 것이다. 섞은 뒤에 남은 0.01 언저리는 번호가 아니라 **빈칸**(0)에서 온다. 연습문제 3이 그것을 잰다.

**임베딩은 약속을 바꿔도 소수점 넷째 자리까지 같다.** 같은 수준이 아니라 **같다** — 학습이 끝난 표를 비교해도 모든 줄이 비트까지 같다(출력의 `True`). 섞은 쪽은 같은 단어가 같은 줄에서 출발하도록 줄을 새 번호 자리로 옮겨 두었을 뿐이다. 임베딩에서 번호는 **표의 몇째 줄인가**를 가리킬 뿐 셈에 들어가지 않으므로, 번호를 어떻게 매기든 학습이 똑같이 흘러간다. 명목 척도의 수를 명목 척도답게 쓴 것이다.

> 이 재기의 한계도 적어 둔다. 번호 그대로 쪽은 은닉층 하나짜리 MLP를 5 에포크 학습한 것이다. 훨씬 큰 모델을 훨씬 오래 학습하면 번호 하나하나를 외워 조금 더 올라갈 수는 있다. 그러나 그것은 「번호 → 뜻」을 2만 개 따로 외우는 일이고, 그렇게 외운 것을 담는 가장 곧은 그릇이 바로 단어마다 줄 하나를 두는 표, 곧 임베딩이다.

## 6. 그래서 임베딩

정리하면 이렇다.

- 그림의 수는 잰 값이라 **이미 쓸 수 있는 수**였다.
- 글자 코드와 단어 번호는 약속이라 **쓸 수 없는 수**다. 셈을 하면 뜻이 없고(`off`, `dhn`), 거리는 철자를 재고(`coy`), 학습에 넣으면 동전 던지기를 겨우 넘는다(0.52).
- 그러니 단어에는 **수를 새로 정해 주어야** 한다. 단어마다 수 여러 개를 두고, 그 수를 셈이 통하도록 — 뜻이 가까운 단어끼리 가깝도록 — 배운다. 그것이 임베딩이다. 번호는 그 표에서 **줄을 찾는 이름표**로만 쓴다.

남은 물음은 둘이다. 그 표를 **어떻게 들고 있는가**가 [다음 쪽](01_not_a_number.md)이고, 그 표를 **무엇을 기준으로 채우는가**가 그 뒤의 [특잇값 분해와 LSA](02_svd_lsa.md), [같이 나온 횟수](02_counting.md), [주위 단어](03_context.md) 쪽이다.

---

## 7. 코드

```python
"""왜 임베딩인가 — 픽셀값은 잰 값이고 글자 코드는 약속이다.

그림의 수와 글자의 수는 둘 다 컴퓨터 안의 수지만 성격이 다르다. 픽셀값은
잉크의 양을 잰 것이라 더하고 평균 내고 흔들어도 뜻이 남는다. 글자 코드와
단어 번호는 사람들이 정한 약속이라 그런 연산에 뜻이 없다.

  (가) 그림의 수 — 잡음을 얹고, 어둡게 하고, 두 장을 평균 내도 숫자가 읽히는가
  (나) 글자의 수 — 같은 글자가 약속마다 다른 수이고, 코드에 셈을 하면 무엇이 되는가
  (다) 글자 코드 사이의 거리는 무엇을 재는가
  (라) 단어 번호 — 이웃한 번호, 더한 번호
  (마) 번호를 수 그대로 넣고 배우면 — 그리고 번호 매기는 약속을 바꾸면

(가)는 5.1절이 학습해 둔 평가 분류기 mnist_judge.pt 를 읽는다.
실행: 이 쪽의 코드를 파일로 갈무리해 저장소 뿌리에서 돌린다.
"""
import re
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

SEED, VOCAB, DIM, LEN = 42, 20000, 64, 200
ROOT = Path("./data/aclImdb")
TOKEN = re.compile(r"[a-z']+")
torch.set_num_threads(4)


def header(title):
    print(f"\n=== {title} ===")


# === (가) 그림의 수 ===========================================================
class JudgeCNN(nn.Module):
    """5.1절의 평가 분류기와 같은 구조."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2); self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(64 * 7 * 7, 128); self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        return self.fc2(self.dropout(torch.relu(self.fc1(x.flatten(1)))))


def pixels_are_measurements():
    header("(가) 그림의 수는 잰 값이다 — 연산을 해도 숫자가 남는가")
    te = torchvision.datasets.MNIST("./data", train=False, download=True)
    X, y = te.data.float() / 255.0, te.targets          # 0 = 종이, 1 = 가장 진한 잉크
    judge = JudgeCNN()
    judge.load_state_dict(torch.load("mnist_judge.pt", map_location="cpu",
                                     weights_only=True))
    judge.eval()

    def read(x):
        with torch.no_grad():
            return torch.cat([judge(((x[i:i + 2000] - 0.1307) / 0.3081).unsqueeze(1))
                              .argmax(1) for i in range(0, len(x), 2000)])

    g = torch.Generator().manual_seed(SEED)
    print("  하는 일                              같은 숫자로 읽는 비율")
    print(f"  그대로                               {(read(X) == y).float().mean():.2%}")
    for s in (0.1, 0.2, 0.3):
        noisy = (X + s * torch.randn(X.shape, generator=g)).clamp(0, 1)
        print(f"  잡음 얹기 (표준편차 {s})              {(read(noisy) == y).float().mean():.2%}")
    for k in (0.8, 0.5):
        print(f"  잉크를 {k} 배로 옅게                   {(read(X * k) == y).float().mean():.2%}")
    idx = torch.randperm(len(X), generator=g)
    a, b = idx[:5000], idx[5000:10000]
    pred = read((X[a] + X[b]) / 2)                       # 두 장을 픽셀마다 평균
    same = y[a] == y[b]
    print(f"  같은 숫자 두 장의 평균 ({same.sum()}쌍)          "
          f"{(pred[same] == y[a][same]).float().mean():.2%}  (그 숫자로)")
    diff = ~same
    either = (pred[diff] == y[a][diff]) | (pred[diff] == y[b][diff])
    print(f"  다른 숫자 두 장의 평균 ({diff.sum()}쌍)        "
          f"{either.float().mean():.2%}  (둘 가운데 하나로)")


# === (나) 글자의 수 ===========================================================
def codes_are_agreements():
    header("(나) 글자의 수는 약속이다")
    print("  글자   UTF-8      EUC-KR   UTF-16   EBCDIC(cp500)")
    for ch in ("A", "가"):
        euc = ch.encode("euc-kr").hex(" ")
        ebc = ch.encode("cp500").hex(" ") if ch.isascii() else "-"
        print(f"  {ch:3s}   {ch.encode('utf-8').hex(' '):9s}  {euc:7s}  "
              f"{ch.encode('utf-16-be').hex(' '):7s}  {ebc}")
    print("\n  코드에 셈을 하면")
    for w in ("boy", "good"):
        print(f"    {w} 의 글자마다 +1            -> {''.join(chr(ord(c) + 1) for c in w)}")
    for u, v in (("cat", "dog"), ("good", "bad")):
        avg = "".join(chr(round((ord(p) + ord(q)) / 2)) for p, q in zip(u, v))
        print(f"    {u} 와 {v} 를 글자마다 평균  -> {avg}")
    print("\n  한글 음절 코드는 공식으로 짜여 있다: 0xAC00 + (초성 x 21 + 중성) x 28 + 종성")
    base = ord("가")
    for off in (1, 28, 588, 1176):
        print(f"    가 + {off:4d} = {chr(base + off)}")


# === (다) 거리 ================================================================
def ascii_vec(w, n=8):
    """단어를 글자의 아스키 코드 n 개로 적는다. 짧으면 0 으로 채운다."""
    return torch.tensor([ord(c) for c in w] + [0] * (n - len(w)), dtype=torch.float)


def ascii_distance():
    header("(다) 글자 코드 사이의 거리")
    print("  boy 를 화소 셋짜리 그림이라 치면")
    for w in ("boy", "coy", "toy", "girl", "boys"):
        print(f"    {w:<5s} {' '.join(f'{ord(c):08b}' for c in w)}")
    print("\n  그 수들 사이의 거리를 재면")
    for u, v in (("boy", "coy"), ("boy", "toy"), ("boy", "girl"), ("boy", "boys")):
        print(f"    {u:<5s}-{v:<5s} {(ascii_vec(u) - ascii_vec(v)).norm():8.1f}")


# === (라)(마) 단어 번호 =======================================================
def tokenize(text):
    return TOKEN.findall(text.replace("<br />", " ").lower())


def load(split):
    docs, labels = [], []
    for lab, name in ((1, "pos"), (0, "neg")):
        for f in sorted((ROOT / split / name).glob("*.txt")):
            docs.append(tokenize(f.read_text(encoding="utf-8")))
            labels.append(lab)
    return docs, torch.tensor(labels, dtype=torch.float)


def encode(docs, stoi):
    """평마다 앞 LEN 단어의 번호. 번호는 1 부터이고 0 은 빈칸(어휘 밖·짧은 평)이다."""
    X = torch.zeros(len(docs), LEN, dtype=torch.long)
    for i, d in enumerate(docs):
        ids = [stoi[w] + 1 for w in d if w in stoi][:LEN]
        X[i, :len(ids)] = torch.tensor(ids)
    return X


class RawIds(nn.Module):
    """번호를 수 그대로 받는다 — 앞 200 단어의 번호를 어휘 크기로 나눈 200 칸."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(LEN, DIM), nn.ReLU(), nn.Linear(DIM, 1))

    def forward(self, X):
        return self.net(X.float() / VOCAB).squeeze(1)


class Bag(nn.Module):
    """번호로 표에서 64 칸짜리 줄을 꺼내 평균 낸다 — 임베딩."""

    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(VOCAB + 1, DIM, padding_idx=0)
        self.out = nn.Linear(DIM, 1)

    def forward(self, X):
        m = (X > 0).float().unsqueeze(-1)
        e = (self.emb(X) * m).sum(1) / m.sum(1).clamp(min=1)
        return self.out(e).squeeze(1)


def fit(model, Xtr, ytr, Xte, yte, seed, epochs=5):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    g = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        order = torch.randperm(len(Xtr), generator=g)
        for i in range(0, len(order), 128):
            b = order[i:i + 128]
            loss = F.binary_cross_entropy_with_logits(model(Xtr[b]), ytr[b])
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        return ((model(Xte) > 0).float() == yte).float().mean().item()


if __name__ == "__main__":
    pixels_are_measurements()
    codes_are_agreements()
    ascii_distance()

    tr, ytr = load("train")
    te, yte = load("test")
    counts = Counter(w for d in tr for w in d)
    words = [w for w, _ in counts.most_common(VOCAB)]   # 흔한 순서로 번호를 매긴다
    stoi = {w: i for i, w in enumerate(words)}

    header("(라) 단어 번호는 이름이다")
    print(f"  영화평 25,000편에서 단어 {len(counts):,}가지, 그 가운데 흔한 {VOCAB:,}개를 쓴다\n")
    print("  번호가 이웃이라고 뜻이 이웃인 것은 아니다")
    for i in range(2340, 2346):
        print(f"    {i:5d}  {words[i]:<14s} ({counts[words[i]]:,}번 나옴)")
    a, b = stoi["good"], stoi["bad"]
    print(f"\n    good 은 {a}번, bad 는 {b}번이다. 둘을 더하면 {a + b}번인데")
    print(f"    거기 있는 단어는 '{words[a + b]}' 이다 — 아무 뜻이 없다")

    header("(마) 번호를 수 그대로 넣고 배우면 — IMDB 긍정·부정, 시험 25,000편")
    Xtr, Xte = encode(tr, stoi), encode(te, stoi)
    print("  씨앗   번호 그대로   번호를 섞으면   |   임베딩   번호를 섞으면   표가 같은가")
    for seed in range(3):
        # 번호 매기는 약속을 무작위로 바꾼다 (0 = 빈칸은 그대로 둔다)
        perm = torch.cat([torch.zeros(1, dtype=torch.long),
                          1 + torch.randperm(VOCAB, generator=torch.Generator()
                                             .manual_seed(100 + seed))])
        torch.manual_seed(seed)
        raw = fit(RawIds(), Xtr, ytr, Xte, yte, seed)
        torch.manual_seed(seed)
        raw_p = fit(RawIds(), perm[Xtr], ytr, perm[Xte], yte, seed)

        torch.manual_seed(seed)
        m1 = Bag()
        W0 = m1.emb.weight.detach().clone()
        out0 = {k: v.clone() for k, v in m1.out.state_dict().items()}
        bag = fit(m1, Xtr, ytr, Xte, yte, seed)
        # 섞은 쪽은 같은 단어가 같은 줄에서 출발하게 한다 — 줄이 새 번호 자리로 옮겨 갈 뿐이다
        m2 = Bag()
        with torch.no_grad():
            m2.emb.weight[perm] = W0
            m2.out.load_state_dict(out0)
        bag_p = fit(m2, perm[Xtr], ytr, perm[Xte], yte, seed)
        same = torch.equal(m2.emb.weight[perm], m1.emb.weight)
        print(f"  {seed:4d}   {raw:.4f}        {raw_p:.4f}          |   {bag:.4f}   "
              f"{bag_p:.4f}          {same}")
```

**출력:**

```
=== (가) 그림의 수는 잰 값이다 — 연산을 해도 숫자가 남는가 ===
  하는 일                              같은 숫자로 읽는 비율
  그대로                               98.83%
  잡음 얹기 (표준편차 0.1)              98.87%
  잡음 얹기 (표준편차 0.2)              98.43%
  잡음 얹기 (표준편차 0.3)              95.84%
  잉크를 0.8 배로 옅게                   98.79%
  잉크를 0.5 배로 옅게                   98.51%
  같은 숫자 두 장의 평균 (510쌍)          99.61%  (그 숫자로)
  다른 숫자 두 장의 평균 (4490쌍)        89.38%  (둘 가운데 하나로)

=== (나) 글자의 수는 약속이다 ===
  글자   UTF-8      EUC-KR   UTF-16   EBCDIC(cp500)
  A     41         41       00 41    c1
  가     ea b0 80   b0 a1    ac 00    -

  코드에 셈을 하면
    boy 의 글자마다 +1            -> cpz
    good 의 글자마다 +1            -> hppe
    cat 와 dog 를 글자마다 평균  -> dhn
    good 와 bad 를 글자마다 평균  -> dhj

  한글 음절 코드는 공식으로 짜여 있다: 0xAC00 + (초성 x 21 + 중성) x 28 + 종성
    가 +    1 = 각
    가 +   28 = 개
    가 +  588 = 까
    가 + 1176 = 나

=== (다) 글자 코드 사이의 거리 ===
  boy 를 화소 셋짜리 그림이라 치면
    boy   01100010 01101111 01111001
    coy   01100011 01101111 01111001
    toy   01110100 01101111 01111001
    girl  01100111 01101001 01110010 01101100
    boys  01100010 01101111 01111001 01110011

  그 수들 사이의 거리를 재면
    boy  -coy        1.0
    boy  -toy       18.0
    boy  -girl     108.5
    boy  -boys     115.0

=== (라) 단어 번호는 이름이다 ===
  영화평 25,000편에서 단어 85,680가지, 그 가운데 흔한 20,000개를 쓴다

  번호가 이웃이라고 뜻이 이웃인 것은 아니다
     2340  relatively     (213번 나옴)
     2341  teenagers      (213번 나옴)
     2342  haunted        (213번 나옴)
     2343  danger         (213번 나옴)
     2344  walks          (213번 나옴)
     2345  creatures      (213번 나옴)

    good 은 47번, bad 는 73번이다. 둘을 더하면 120번인데
    거기 있는 단어는 'off' 이다 — 아무 뜻이 없다

=== (마) 번호를 수 그대로 넣고 배우면 — IMDB 긍정·부정, 시험 25,000편 ===
  씨앗   번호 그대로   번호를 섞으면   |   임베딩   번호를 섞으면   표가 같은가
     0   0.5275        0.5084          |   0.8438   0.8438          True
     1   0.5228        0.5142          |   0.8376   0.8376          True
     2   0.5247        0.5098          |   0.8382   0.8382          True
```

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
2절의 공식으로 `한`의 코드를 구하라. ㅎ은 초성의 18번째(0부터), ㅏ는 중성의 0번째, 받침 ㄴ은 종성의 4번째다. 그리고 `하`와 `한`의 코드 차이가 4인 것이 두 글자의 뜻에 대해 무엇을 말해 주는지 답하라.

</div>

??? success "풀이"
    $\texttt{0xAC00} + (18 \times 21 + 0) \times 28 + 4 = 44032 + 10584 + 4 = 54620 = \texttt{0xD55C}$ 다.

    ```python
    print(hex(0xAC00 + (18 * 21 + 0) * 28 + 4), chr(0xAC00 + (18 * 21 + 0) * 28 + 4), ord("한") - ord("하"))
    ```

    ```
    0xd55c 한 4
    ```

    차이 4는 **받침 ㄴ이 붙었다**는 것만 말한다. `하`(동사 어간)와 `한`(관형형, 또는 「한국」의 한)이 뜻으로 가까운지는 이 수가 말해 주지 않는다. 같은 4가 `가`와 `간`, `바`와 `반` 사이에도 똑같이 있다. 코드의 구조는 글자를 어떻게 적는지를 담을 뿐이다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
5절은 번호를 섞어도 임베딩 모델의 학습이 **비트까지** 같다고 했다. 섞기를 순열 $\pi$ 로 적고(단어 $w$ 의 번호 $i$ 가 $\pi(i)$ 로 바뀐다), 섞은 쪽 표의 $\pi(i)$ 번째 줄을 원래 표의 $i$ 번째 줄로 두고 출발하면 순전파·역전파·Adam 갱신이 모두 같은 값을 냄을 보여라. 번호 그대로 모델에서는 왜 같은 논리가 안 되는가?

</div>

??? success "풀이"
    **순전파.** 평의 단어 번호가 $i_1, \dots, i_n$ 이면 원래 모델은 $W_{i_1}, \dots, W_{i_n}$ 을 평균하고, 섞은 모델은 $W'_{\pi(i_1)}, \dots, W'_{\pi(i_n)}$ 을 평균한다. $W'_{\pi(i)} = W_i$ 로 두었으니 **같은 벡터들을 같은 차례로** 평균한다. 출력 레이어는 두 모델이 같은 웨이트로 출발하므로 로짓도 로스도 같다.

    **역전파.** 그래디언트는 꺼내 쓴 줄로만 돌아간다. 원래 모델의 $i$ 번째 줄이 받는 그래디언트를 섞은 모델에서는 $\pi(i)$ 번째 줄이 받고, 값이 같다.

    **갱신.** Adam은 칸마다 따로 움직인다(그래디언트와 그 제곱의 이동평균을 칸마다 든다). 그래서 $\pi(i)$ 번째 줄의 갱신은 $i$ 번째 줄의 갱신과 같다. 다음 스텝에서도 $W'_{\pi(i)} = W_i$ 가 유지되니 귀납적으로 끝까지 같다.

    **번호 그대로 모델에서는** 번호가 **입력값**이다. 첫 레이어가 $\sum_j w_j \cdot i_j / 20000$ 을 계산하는데, $i_j$ 를 $\pi(i_j)$ 로 바꾸면 그 합이 달라지고, 웨이트를 어떻게 옮겨 놓아도 되돌릴 수 없다. 웨이트는 **자리**(앞에서 몇째 단어인가)마다 하나이지 **단어**마다 하나가 아니기 때문이다. 이 모델의 결과는 약속에 따라 바뀔 수밖에 없다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
번호를 섞은 「번호 그대로」 모델이 그래도 0.508~0.514로 0.5를 조금 넘는다. 번호에서 뜻을 다 지웠는데 무엇이 남았는가? 입력에 남은 정보가 무엇인지 짐작하고, 그것만 남긴 입력으로 같은 모델을 학습해 확인하라.

</div>

??? success "풀이"
    섞어도 남는 것은 **0인 칸**이다. 어휘 밖 단어는 건너뛰므로 0은 평의 끝에만 생긴다 — 어휘 안 단어가 200개가 안 되는 평의 뒷부분이다. 시험 평의 61.6%가 그렇다. 그러니 0이 몇째 칸부터 시작하는지가 곧 **평의 길이**를 알려 준다. 길이가 긍정·부정과 조금 상관이 있다면 그만큼은 맞힐 수 있다.

    「빈칸이 아닌가」만 남긴 입력(단어가 있으면 같은 수, 없으면 0)으로 확인한다. 위 프로그램에 이어 붙인다.

    ```python
    print("  씨앗   빈칸이 아닌가만 (0 / 1)")
    for seed in range(3):
        torch.manual_seed(seed)
        acc = fit(RawIds(), (Xtr > 0).long() * VOCAB, ytr, (Xte > 0).long() * VOCAB, yte, seed)
        print(f"  {seed:4d}   {acc:.4f}")
    print(f"  시험 평 가운데 단어가 {LEN} 개 안 되는 것  {((Xte > 0).sum(1) < LEN).float().mean():.1%}")
    ```

    ```
      씨앗   빈칸이 아닌가만 (0 / 1)
         0   0.5056
         1   0.5096
         2   0.5029
      시험 평 가운데 단어가 200 개 안 되는 것  61.6%
    ```

    **0.503~0.510이다.** 섞은 쪽의 0.508~0.514와 거의 겹친다. 섞은 뒤에 남은 몫은 대부분 길이였다. 그러니 번호 그대로 모델이 번 0.52 남짓을 쪼개면, 0.01 언저리는 길이, 0.01~0.02는 빈도 순위이고, **단어의 뜻에서 온 몫은 재어지지 않는다.**
