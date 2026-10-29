# 섞인 정밀도 — 왜 로스를 키워 두는가

`float16`은 `float32`의 절반만 쓴다. 메모리가 반이고, 셈도 (텐서 코어가 있는 하드웨어에서는) 빠르다. 그런데 그냥 바꾸면 **학습이 망가진다.**

이 쪽은 왜 망가지는지를 자료형의 성질에서 보고, `GradScaler`와 `autocast`가 각각 그 가운데 무엇을 막는지를 가른다. 여기 나오는 것은 모두 **수의 성질**이라 가속기가 없어도 그대로 돌아간다.

## 1. 코드

```python
"""튜토리얼 24: 섞인 정밀도 — 왜 로스를 키워 두는가.

여기 나오는 것은 모두 **자료형의 성질**이라 가속기가 없어도 그대로 돌아간다.
"""
import torch
import torch.nn as nn

torch.manual_seed(0)


def header(title):
    print(f"\n{'='*70}\n{title}\n{'='*70}")


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DEV = pick_device()


def main():
    header("1. 세 자료형이 담는 것")
    print(f"{'':10s}{'바이트':>7}{'가장 큰 수':>13}{'가장 작은 정규수':>18}{'eps':>12}")
    for name, dt in [("float32", torch.float32),
                     ("float16", torch.float16),
                     ("bfloat16", torch.bfloat16)]:
        i = torch.finfo(dt)
        b = torch.tensor([0], dtype=dt).element_size()
        print(f"{name:10s}{b:>7d}{i.max:>13.3e}{i.tiny:>18.3e}{i.eps:>12.3e}")
    print()
    print("float16 은 범위를 줄여 정밀도를 지켰고, bfloat16 은 반대로 했다.")
    print("bfloat16 의 범위는 float32 와 같고, 대신 eps 가 8 배 크다.")

    header("2. float16 에서 그래디언트가 사라진다")
    print(f"{'원래 값':>10}{'float16 으로':>16}{'0 이 되었나':>14}")
    for v in (1e-4, 1e-5, 1e-6, 1e-7, 1e-8):
        h = torch.tensor([v], dtype=torch.float32).to(torch.float16)
        print(f"{v:>10.0e}{h.item():>16.3e}{str(h.item() == 0):>14s}")
    print()
    print("float16 의 가장 작은 정규수는 6.1e-05 다. 그 아래는 비정규수로")
    print("버티다가 1e-08 에서 0 이 된다. 그래디언트가 그만큼 작아지는 일은 흔하다.")

    header("3. 그래서 로스를 키워 둔다 — GradScaler 의 속셈")
    g = 1e-8
    print(f"그래디언트가 {g:.0e} 일 때")
    print(f"{'배율':>8}{'float16 에서':>16}{'되돌린 값':>14}{'살았나':>9}")
    for s in (1, 2**8, 2**16):
        h = torch.tensor([g * s], dtype=torch.float32).to(torch.float16)
        back = h.to(torch.float32).item() / s
        print(f"{s:>8d}{h.item():>16.3e}{back:>14.3e}{str(back > 0):>9s}")
    print()
    print("로스에 배율을 곱해 역전파하면 그래디언트가 모두 그만큼 커져 살아남고,")
    print("갱신하기 전에 도로 나눈다. 이것이 GradScaler 가 하는 일의 전부다.")
    print("bfloat16 은 범위가 float32 와 같아 이 장치가 필요 없다.")

    header("4. autocast 는 모델을 half 로 바꾸는 것이 아니다")
    x = torch.randn(8, 16)
    w = torch.randn(16, 16)
    with torch.amp.autocast("cpu", dtype=torch.float16):
        rows = [("matmul", (x @ w).dtype),
                ("softmax", torch.softmax(x, -1).dtype),
                ("layer_norm", nn.functional.layer_norm(x, (16,)).dtype),
                ("cross_entropy",
                 nn.functional.cross_entropy(x[:, :10],
                                             torch.zeros(8, dtype=torch.long)).dtype),
                ("더하기", (x + x).dtype)]
    for name, dt in rows:
        print(f"  {name:15s} -> {dt}")
    print()
    half = nn.Sequential(nn.Linear(16, 16), nn.LayerNorm(16)).half()
    print(f"  모델을 .half() 로 바꾸면 layer_norm 출력 -> {half(x.half()).dtype}")
    print()
    print("autocast 는 연산마다 고른다 — 곱은 내리고, 합·정규화·소프트맥스는")
    print("float32 로 둔다. .half() 는 그런 구별 없이 모두 내린다.")

    header("5. 오늘의 API")
    print(f"고른 장치: {DEV}")
    print("torch.amp.autocast(device_type, dtype=...) 와 torch.amp.GradScaler(device_type)")
    print("가 지금 쓰는 꼴이다. torch.cuda.amp.* 는 예전 이름이다.")
    scaler = torch.amp.GradScaler(DEV)
    print(f"GradScaler 가 켜졌는가: {scaler.is_enabled()}")
    print("장치 이름을 넘기면 그 장치에 맞게 켜지고 꺼진다 — 직접 끌 일은 거의 없다.")


if __name__ == "__main__":
    main()
```

**출력:**

```
======================================================================
1. 세 자료형이 담는 것
======================================================================
              바이트       가장 큰 수         가장 작은 정규수         eps
float32         4    3.403e+38         1.175e-38   1.192e-07
float16         2    6.550e+04         6.104e-05   9.766e-04
bfloat16        2    3.390e+38         1.175e-38   7.812e-03

float16 은 범위를 줄여 정밀도를 지켰고, bfloat16 은 반대로 했다.
bfloat16 의 범위는 float32 와 같고, 대신 eps 가 8 배 크다.

======================================================================
2. float16 에서 그래디언트가 사라진다
======================================================================
      원래 값      float16 으로       0 이 되었나
     1e-04       1.000e-04         False
     1e-05       1.001e-05         False
     1e-06       1.013e-06         False
     1e-07       1.192e-07         False
     1e-08       0.000e+00          True

float16 의 가장 작은 정규수는 6.1e-05 다. 그 아래는 비정규수로
버티다가 1e-08 에서 0 이 된다. 그래디언트가 그만큼 작아지는 일은 흔하다.

======================================================================
3. 그래서 로스를 키워 둔다 — GradScaler 의 속셈
======================================================================
그래디언트가 1e-08 일 때
      배율      float16 에서         되돌린 값      살았나
       1       0.000e+00     0.000e+00    False
     256       2.563e-06     1.001e-08     True
   65536       6.552e-04     9.997e-09     True

로스에 배율을 곱해 역전파하면 그래디언트가 모두 그만큼 커져 살아남고,
갱신하기 전에 도로 나눈다. 이것이 GradScaler 가 하는 일의 전부다.
bfloat16 은 범위가 float32 와 같아 이 장치가 필요 없다.

======================================================================
4. autocast 는 모델을 half 로 바꾸는 것이 아니다
======================================================================
  matmul          -> torch.float16
  softmax         -> torch.float32
  layer_norm      -> torch.float32
  cross_entropy   -> torch.float32
  더하기             -> torch.float32

  모델을 .half() 로 바꾸면 layer_norm 출력 -> torch.float16

autocast 는 연산마다 고른다 — 곱은 내리고, 합·정규화·소프트맥스는
float32 로 둔다. .half() 는 그런 구별 없이 모두 내린다.

======================================================================
5. 오늘의 API
======================================================================
고른 장치: mps
torch.amp.autocast(device_type, dtype=...) 와 torch.amp.GradScaler(device_type)
가 지금 쓰는 꼴이다. torch.cuda.amp.* 는 예전 이름이다.
GradScaler 가 켜졌는가: True
장치 이름을 넘기면 그 장치에 맞게 켜지고 꺼진다 — 직접 끌 일은 거의 없다.
```

## 2. 두 가지를 반으로 줄이는 두 가지 방법

16비트에 담으려면 **범위**와 **정밀도** 가운데 하나를 내주어야 한다. 둘이 서로 다른 답을 골랐다.

| | 가장 큰 수 | 가장 작은 정규수 | eps |
|---|---|---|---|
| `float32` | $3.4 \times 10^{38}$ | $1.2 \times 10^{-38}$ | $1.2 \times 10^{-7}$ |
| `float16` | $6.6 \times 10^{4}$ | $6.1 \times 10^{-5}$ | $9.8 \times 10^{-4}$ |
| `bfloat16` | $3.4 \times 10^{38}$ | $1.2 \times 10^{-38}$ | $7.8 \times 10^{-3}$ |

**`bfloat16`의 범위는 `float32`와 같다.** 지수 자리를 그대로 두고 가수 자리를 깎았기 때문이고, 그 값으로 eps가 `float16`의 **8배**가 되었다. `float16`은 반대로 범위를 $10^{38}$에서 $10^{4}$로 줄이는 대신 정밀도를 지켰다.

이 한 줄이 뒤의 모든 것을 정한다.

## 3. 사라지는 그래디언트는 작아서가 아니라 범위 밖이라서

2절이 그것을 보인다. $10^{-4}$에서 $10^{-7}$까지는 `float16`이 버티는데 — $6.1 \times 10^{-5}$ 아래는 비정규수로 정밀도를 깎아 가며 버틴다 — $10^{-8}$에서 **0이 된다.**

학습 후반의 그래디언트가 그만큼 작아지는 일은 드물지 않다. 그리고 0이 된 그래디언트는 **오류를 내지 않는다.** 그냥 그 파라메터가 더 이상 움직이지 않을 뿐이다.

## 4. `GradScaler`는 범위 문제만 푼다

고치는 방법이 뜻밖에 단순하다. **로스에 큰 수를 곱해 역전파한다.** 연쇄법칙이 선형이므로 모든 그래디언트가 똑같이 그 배만큼 커지고, `float16`의 범위 안으로 들어온다. 갱신하기 전에 도로 나눈다.

3절의 표가 그것이다. 배율 $2^{8}$만 걸어도 $10^{-8}$이 살아남아 $1.001 \times 10^{-8}$로 돌아온다.

그러므로 **`bfloat16`을 쓰면 `GradScaler`가 필요 없다.** 범위가 `float32`와 같아 애초에 넘칠 일이 없기 때문이다. 둘 가운데 무엇을 고르느냐가 이 장치를 쓰느냐 마느냐를 정한다.

<div class="storybox" markdown>

**그림으로 보는 AI 1.** 세 가지가 한 논문에서 한꺼번에 왔다

`GradScaler`는 누가 따로 생각해 낸 요령이 아니라, **셋이 한 묶음으로** 나온 것 가운데 둘째다. 2017년에 NVIDIA와 바이두 리서치가 함께 낸 「Mixed Precision Training」이 그 논문이고, 초록에 셋이 나란히 적혀 있다 — 웨이트의 `float32` 사본을 따로 두고 거기에 그래디언트를 누적할 것, 작은 그래디언트를 살리려 **로스를 키워 둘 것**, 반정밀도로 곱하되 누적은 단정밀도로 할 것.

경쟁하는 두 회사가 같이 쓴 글이라는 점이 이 묶음의 성격을 말해 준다. 어느 하나가 알고리즘의 발견이라기보다, 반정밀도 하드웨어를 실제로 쓰려면 함께 갖춰야 하는 **조건 목록**에 가까웠다.

![왼쪽은 세 자료형의 표현 범위를 가로 막대로 그린 것으로, float32와 bfloat16의 막대 길이가 같고 float16은 가운데 짧은 토막이다. 오른쪽은 작은 신경망의 그래디언트 크기 분포를 그린 것으로, float16의 최소 정상수를 나타내는 세로 점선 왼쪽에 상당한 몫이 걸쳐 있고, 8을 곱해 옮긴 분포는 그 선 오른쪽으로 물러나 있다. 배율 1에서 12.49퍼센트, 8에서 1.54퍼센트, 1024에서 0.01퍼센트가 선 아래로 떨어진다고 적혀 있다.](figures/loss_scaling.svg)

얼마나 걸리는 문제인지 재어 보면 이렇다. 레이어 셋짜리 작은 신경망을 한 번 역전파시키면 그래디언트의 **12.49%** 가 `float16` 의 최소 정상수 아래에 있다. 8을 곱하면 1.54%, 1024를 곱하면 0.01%로 준다. 논문도 Multibox SSD 검출기에서 스케일 없이는 **학습이 발산했고** 8만 곱해도 `float32` 와 같은 정확도가 나왔다고 적었다. 다만 그 8은 그 신경망에서 나온 수이지 어디에나 맞는 수가 아니다.

**그런데 이 셋 가운데 하나는 오늘 조용히 사라지는 중이다.** 바로 위에서 적은 대로 `bfloat16` 에는 `GradScaler` 가 필요 없다 — 왼쪽 그림에서 보듯 범위가 `float32` 와 **같기** 때문이다. 로스 스케일링은 `float16` 의 좁은 범위를 소프트웨어로 메우던 장치였고, 지수 비트를 `float32` 와 똑같이 가져가는 형식이 나오자 **메울 구멍 자체가 없어졌다.** 고치는 자리가 알고리즘에서 수의 형식으로 옮겨 간 셈이다.

그러니 이 쪽에서 `GradScaler` 를 배우는 값어치는 그 호출을 외우는 데 있지 않다. **자료형을 고르는 일이 코드에 무엇을 더 써야 하는지까지 정한다**는 것, 그것이 다음 절로 이어지는 이야기다.

_확인한 곳 — P. Micikevicius 외, "Mixed Precision Training", ICLR 2018 (arXiv:1710.03740). 저자는 NVIDIA 와 Baidu Research 두 곳에 걸쳐 있다. 세 가지 기법은 초록에, SSD 검출기가 스케일 없이 발산했고 8을 곱해 회복했다는 것은 3.2절과 4.2절에 있다. 오른쪽 그림의 비율은 그림을 그리는 코드가 씨앗 0으로 직접 재어 얻는다._

</div>

## 5. `autocast`는 정밀도 문제를 푼다

`.half()`는 모델 전체를 내린다. `autocast`는 **연산마다 고른다.** 4절이 같은 입력에 둘을 걸어 보인다.

| | `autocast` 안 | `.half()` |
|---|---|---|
| 행렬곱 | `float16` | `float16` |
| 소프트맥스 | **`float32`** | `float16` |
| 레이어 정규화 | **`float32`** | `float16` |
| 교차 엔트로피 | **`float32`** | `float16` |

**곱은 내리고 합은 내리지 않는다.** 행렬곱은 큰 수들의 곱이라 16비트로도 버티지만, 소프트맥스·정규화·교차 엔트로피는 지수와 로그와 작은 차이를 다루어 eps가 $10^{-3}$인 자리에서 무너진다. `autocast`가 가진 목록이 그 구별이다.

그래서 둘은 **다른 일**을 한다 — `autocast`가 어떤 연산을 내릴지 고르고, `GradScaler`가 내린 연산의 그래디언트가 0으로 사라지지 않게 막는다. 섞인 정밀도 학습이 늘 둘을 함께 쓰는 까닭이다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
`float16`의 가장 큰 수는 $6.55 \times 10^{4}$다. 로스가 65,504를 넘으면 어떻게 되는가? `GradScaler`가 배율을 너무 크게 잡으면 같은 일이 생길 수 있다 — 그때 스케일러는 무엇을 하는가?

</div>

??? success "풀이"
    `inf`가 된다. 그리고 `inf`가 섞인 그래디언트로 갱신하면 파라메터가 통째로 `nan`이 된다.

    ```python
    import torch

    big = torch.tensor([70000.0], dtype=torch.float32)
    print("float32:", big.item())
    print("float16:", big.to(torch.float16).item())

    w = torch.tensor([1.0]); g = torch.tensor([float("inf")])
    print("inf 로 갱신하면:", (w - 0.1 * g).item())
    ```

    ```
    float32: 70000.0
    float16: inf
    inf 로 갱신하면: -inf
    ```

    **그래서 `GradScaler`는 배율을 고정해 두지 않는다.** 역전파한 그래디언트에 `inf`나 `nan`이 있으면 **그 스텝을 통째로 건너뛰고** 배율을 줄인다. 한동안 탈이 없으면 다시 올려 본다.

    곧 배율은 "넘치지 않는 가장 큰 값"을 **더듬어 찾는 값**이고, 학습 중에 오르내린다. 건너뛴 스텝이 몇 번 생기는 것은 고장이 아니라 그 탐색의 일부다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
`bfloat16`은 범위가 `float32`와 같아 `GradScaler`가 필요 없다. 그렇다면 `float16`보다 언제나 나은가? eps를 보고 답하라 — 어떤 셈에서 손해가 나겠는가?

</div>

??? success "풀이"
    언제나 낫지는 않다. **eps가 8배 크다**는 것은 같은 크기의 수를 8배 거칠게 적는다는 뜻이다.

    ```python
    import torch

    x = torch.tensor([1.0])
    for dt in (torch.float16, torch.bfloat16):
        eps = torch.finfo(dt).eps
        # 1 에 eps 의 절반을 더하면 사라진다 — 구별할 수 있는 가장 작은 차이
        print(f"{str(dt):17s} eps {eps:.3e}   "
              f"1 + eps/2 = {(x.to(dt) + eps/2).item():.6f}")
    ```

    ```
    torch.float16     eps 9.766e-04   1 + eps/2 = 1.000000
    torch.bfloat16    eps 7.812e-03   1 + eps/2 = 1.000000
    ```

    둘 다 사라지지만 **사라지는 문턱이 8배 다르다.** `bfloat16`은 1 언저리에서 0.4% 미만의 차이를 적지 못한다.

    손해가 나는 자리는 **비슷한 크기의 수를 여럿 더할 때**다. 긴 합을 누적하면 더해지는 양이 누적값에 비해 작아지고, 어느 순간부터 더해도 값이 변하지 않는다. 그래서 `autocast`는 어느 쪽을 쓰든 합·정규화·소프트맥스를 `float32`에 남겨 둔다.

    정리하면 — `bfloat16`은 **넘침을 걱정하지 않아도 되는 대신 정밀도를 더 내준** 선택이다. 범위가 문제이면 `bfloat16`, 정밀도가 문제이면 `float16`에 `GradScaler`다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
`autocast` 없이 모델을 `.half()`로 바꾸어 학습하면 흔히 로스가 `nan`이 된다. 소프트맥스가 그 범인이 되는 과정을 수로 보여라. 그리고 `float32`로 두면 왜 괜찮은지 설명하라.

</div>

??? success "풀이"
    소프트맥스는 지수를 쓴다. 로짓이 조금만 커도 $e^{x}$가 `float16`의 천장($6.55\times10^{4}$)을 넘는다.

    ```python
    import torch

    logits = torch.tensor([12.0, 0.0, -12.0])
    for dt in (torch.float16, torch.float32):
        x = logits.to(dt)
        naive = x.exp()                       # 안정화 없이 그대로
        print(f"{str(dt):15s} exp {naive.tolist()}  합 {naive.sum().item()}")
    print()
    for dt in (torch.float16, torch.float32):
        x = logits.to(dt)
        print(f"{str(dt):15s} softmax {torch.softmax(x, 0).tolist()}")
    ```

    ```
    torch.float16   exp [inf, 1.0, 6.139278411865234e-06]  합 inf
    torch.float32   exp [162754.796875, 1.0, 6.1442124206223525e-06]  합 162755.796875
    ```

    ```
    torch.float16   softmax [1.0, 6.139278411865234e-06, 0.0]
    torch.float32   softmax [0.9999938011169434, 6.144174221844878e-06, 3.775111126480546e-11]
    ```

    **$e^{12}$가 `float16`에서 곧바로 `inf`다.** 참값이 162,754인데 천장이 65,504이니 넘친다. 로짓 12는 학습 중에 쉽게 나오는 크기다.

    그런데 `torch.softmax` 자신은 `nan`을 내지 않는다. 안에서 최댓값을 빼는 안정화를 하므로 $e^{12-12}$부터 셈해 `inf`를 피한다. 터지는 것은 **안정화 없이 `exp`를 직접 부를 때**다.

    그래도 성하지는 않다. 셋째 값을 보라 — 참값이 $3.8 \times 10^{-11}$인데 `float16`에서 **0이 되었다.** `float16`의 가장 작은 비정규수가 $6 \times 10^{-8}$ 언저리라 그 아래는 담을 자리가 없다. 그 0에 로그를 씌우면 $-\infty$이고, 하필 그 클래스가 정답이면 교차 엔트로피가 그대로 터진다.

    `float32`에서는 둘 다 멀쩡하다. 지수의 천장이 $10^{38}$이라 $e^{12}$가 넉넉히 들어가고, 가장 작은 수가 $10^{-38}$이라 $3.8 \times 10^{-11}$도 살아남는다.

    그래서 `autocast`가 소프트맥스와 교차 엔트로피를 `float32`에 남기는 것이다. 모델을 통째로 `.half()`로 바꾸면 그 보호가 없어지고, 터지는 자리는 행렬곱이 아니라 **거의 언제나 이런 연산들**이다.

## 정리하며

16비트로 내려가며 내주는 것이 둘이고, 그것을 메우는 장치도 둘이다.

| 내준 것 | 생기는 일 | 막는 것 |
|---|---|---|
| 범위 ($10^{38} \to 10^{4}$) | 작은 그래디언트가 0이 된다 | `GradScaler` — 로스를 키웠다 되돌린다 |
| 정밀도 (eps $10^{-7} \to 10^{-3}$) | 지수·로그·긴 합이 무너진다 | `autocast` — 그 연산만 `float32`에 남긴다 |

둘은 **서로를 대신하지 못한다.** 그리고 `bfloat16`을 고르면 첫 줄이 통째로 사라지는 대신 둘째 줄이 나빠진다 — 어느 쪽을 고르든 공짜는 없고, 무엇을 내주었는지 알고 고르는 것이 전부다.
