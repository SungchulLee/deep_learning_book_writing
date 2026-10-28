# 형상, 자료형, 장치 — 연산이 되는지를 정하는 세 가지

텐서 둘을 더할 수 있는지는 속에 든 수가 아니라 이 세 속성이 정한다. 그런데 셋이
어긋났을 때 하는 일이 같지 않다.

| 어긋난 것 | 무슨 일이 나는가 |
|---|---|
| 모양 | 터진다 |
| 장치 | 터진다 |
| 자료형 | **조용히 한쪽으로 올라간다** |

앞의 둘은 터져 주니 고치면 끝이다. 자료형은 터지지 않으므로 **찾아야** 한다.
이 쪽은 그 셋을 읽는 법과, 어긋났을 때 갈리는 자리를 본다.

## 1. 코드

```python
"""모양·자료형·장치 — 연산이 되는지를 정하는 세 가지.

텐서 둘을 더할 수 있는지는 속에 든 수가 아니라 이 세 속성이 정한다.
그런데 셋이 어긋났을 때 하는 일이 같지 않다.

- 모양이 어긋나면 터진다
- 장치가 어긋나면 터진다
- 자료형이 어긋나면 **조용히 한쪽으로 올라간다**

그래서 앞의 둘은 고치면 끝이고, 자료형은 터지지 않기 때문에 찾아야
한다. 뒤이어 `.to()` 가 언제 같은 객체를 돌려주는지와, `T`·`mT`·`H`
가 2차원에서만 셋이 같은 것을 본다.

실행: 이 쪽의 코드를 파일로 갈무리해 돌린다
"""
import warnings

import torch


def header(title):
    print(f"\n=== {title} ===")


# ========================================================================
# 1. 세 속성을 읽는다
# ========================================================================

def reading():
    header("1. 속성 읽기")
    x = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    print("x =")
    print(x)
    print("  x.shape       :", x.shape, " (x.size() 와 같다)")
    print("  x.ndim        :", x.ndim, " = len(x.shape)")
    print("  x.numel()     :", x.numel())
    print("  x.dtype       :", x.dtype)
    print("  x.device      :", x.device)
    print("  x.layout      :", x.layout)
    print("  x.element_size:", x.element_size(), "바이트")
    print("  차지하는 메모리:", x.numel() * x.element_size(), "바이트")


# ========================================================================
# 2. 모양이 어긋나면 터진다 — 다만 브로드캐스트가 먼저 손을 쓴다
# ========================================================================

def shapes():
    header("2. 모양이 어긋나면")
    a = torch.ones(3, 4)
    try:
        a + torch.ones(4, 3)
    except RuntimeError as e:
        print("  (3,4) + (4,3) :", str(e).splitlines()[0])
    print("  (3,4) + (4,)  :", tuple((a + torch.ones(4)).shape),
          "— 브로드캐스트가 받아 준다")
    print("  (3,4) + (3,1) :", tuple((a + torch.ones(3, 1)).shape))
    try:
        a @ torch.ones(3, 4)
    except RuntimeError as e:
        print("  (3,4) @ (3,4) :", str(e).splitlines()[0])
    print(f"  (3,4) 와 (4,3) 은 원소가 둘 다 {a.numel()}개인데도 더할 수 없다")


# ========================================================================
# 3. 장치가 어긋나면 터진다
# ========================================================================

def devices():
    header("3. 장치가 어긋나면")
    dev = ("cuda" if torch.cuda.is_available()
           else "mps" if torch.backends.mps.is_available() else None)
    if dev is None:
        print("  가속기가 없어 cpu 뿐이다 — 이 절은 건너뛴다")
        return
    try:
        torch.ones(3) + torch.ones(3, device=dev)
    except RuntimeError as e:
        print(f"  cpu + {dev} :", str(e).splitlines()[0])
    print("  장치는 자료형과 달라서, 한쪽으로 자동으로 옮겨 주지 않는다")


# ========================================================================
# 4. 자료형이 어긋나면 조용히 올라간다
# ========================================================================

def dtypes():
    header("4. 자료형이 어긋나면")
    f32 = torch.ones(3, dtype=torch.float32)
    cases = (
        ("0 차원 float64", torch.tensor(1.0, dtype=torch.float64)),
        ("1 개짜리 float64", torch.ones(1, dtype=torch.float64)),
        ("3 개짜리 float64", torch.ones(3, dtype=torch.float64)),
        ("파이썬 float 1.0", 1.0),
    )
    for name, other in cases:
        pad = " " * max(0, 18 - sum(2 if "가" <= c <= "힣" else 1 for c in name))
        print(f"  float32 + {name}{pad}-> {(f32 + other).dtype}")
    print("  모양이 있는 float64 만 이긴다. 0 차원은 못 이긴다")

    i = torch.tensor([1, 2, 3])
    print("  int / 2       :", (i / 2).dtype, (i / 2).tolist(), "— 나누기는 실수로")
    print("  int // 2      :", (i // 2).dtype, (i // 2).tolist())
    try:
        i.mean()
    except RuntimeError as e:
        print("  int.mean()    :", str(e).splitlines()[0][:72])


# ========================================================================
# 5. .to() 는 바꿀 것이 없으면 그 텐서를 그대로 돌려준다
# ========================================================================

def conversion():
    header("5. .to() 가 언제 새것을 만드는가")
    x = torch.ones(3)
    print("  x.to(torch.float32) is x :", x.to(torch.float32) is x)
    print("  x.to('cpu')         is x :", x.to("cpu") is x)
    print("  x.to(torch.float64) is x :", x.to(torch.float64) is x)
    dev = ("cuda" if torch.cuda.is_available()
           else "mps" if torch.backends.mps.is_available() else None)
    if dev:
        y = x.to(dev)
        print(f"  x.to('{dev}')         is x :", y is x)
        print(f"  y.to('{dev}')         is y :", y.to(dev) is y)
    print("  그래서 x.to(...) 를 부르고 x 를 계속 쓰면 아무 일도 안 한 것이 된다")
    x.to(torch.float64)
    print("  x.to(torch.float64) 만 부른 뒤 x.dtype :", x.dtype)


# ========================================================================
# 6. T / mT / H — 2차원에서만 셋이 같다
# ========================================================================

def transposes():
    header("6. T / mT / H")
    m = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    print("  2차원에서는 셋이 같다:",
          torch.equal(m.T, m.mT) and torch.equal(m.mT, m.H))

    r3 = torch.arange(24).reshape(2, 3, 4)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        shape_T = tuple(r3.T.shape)
        note = str(caught[0].message).split(" and it will")[0] if caught else ""
    print("  3차원 .T  :", shape_T, "— 축을 전부 거꾸로 돌린다")
    print("    경고    :", note)
    print("  3차원 .mT :", tuple(r3.mT.shape), "— 마지막 두 축만 바꾼다")
    try:
        r3.H
    except RuntimeError as e:
        print("  3차원 .H  :", str(e).splitlines()[0][:66])

    c = torch.tensor([[1 + 2j, 3 - 1j]])
    print("  복소수 mT :", c.mT.flatten().tolist())
    print("  복소수 H  :", c.H.flatten().tolist(), "— 켤레까지 취한다")


def main():
    reading()
    shapes()
    devices()
    dtypes()
    conversion()
    transposes()


if __name__ == "__main__":
    main()
```

**출력:**

```
=== 1. 속성 읽기 ===
x =
tensor([[ 0.,  1.,  2.,  3.],
        [ 4.,  5.,  6.,  7.],
        [ 8.,  9., 10., 11.]])
  x.shape       : torch.Size([3, 4])  (x.size() 와 같다)
  x.ndim        : 2  = len(x.shape)
  x.numel()     : 12
  x.dtype       : torch.float32
  x.device      : cpu
  x.layout      : torch.strided
  x.element_size: 4 바이트
  차지하는 메모리: 48 바이트

=== 2. 모양이 어긋나면 ===
  (3,4) + (4,3) : The size of tensor a (4) must match the size of tensor b (3) at non-singleton dimension 1
  (3,4) + (4,)  : (3, 4) — 브로드캐스트가 받아 준다
  (3,4) + (3,1) : (3, 4)
  (3,4) @ (3,4) : mat1 and mat2 shapes cannot be multiplied (3x4 and 3x4)
  (3,4) 와 (4,3) 은 원소가 둘 다 12개인데도 더할 수 없다

=== 3. 장치가 어긋나면 ===
  cpu + mps : Expected all tensors to be on the same device, but found at least two devices, mps:0 and cpu!
  장치는 자료형과 달라서, 한쪽으로 자동으로 옮겨 주지 않는다

=== 4. 자료형이 어긋나면 ===
  float32 + 0 차원 float64    -> torch.float32
  float32 + 1 개짜리 float64  -> torch.float64
  float32 + 3 개짜리 float64  -> torch.float64
  float32 + 파이썬 float 1.0  -> torch.float32
  모양이 있는 float64 만 이긴다. 0 차원은 못 이긴다
  int / 2       : torch.float32 [0.5, 1.0, 1.5] — 나누기는 실수로
  int // 2      : torch.int64 [0, 1, 1]
  int.mean()    : mean(): could not infer output dtype. Input dtype must be either a float

=== 5. .to() 가 언제 새것을 만드는가 ===
  x.to(torch.float32) is x : True
  x.to('cpu')         is x : True
  x.to(torch.float64) is x : False
  x.to('mps')         is x : False
  y.to('mps')         is y : True
  그래서 x.to(...) 를 부르고 x 를 계속 쓰면 아무 일도 안 한 것이 된다
  x.to(torch.float64) 만 부른 뒤 x.dtype : torch.float32

=== 6. T / mT / H ===
  2차원에서는 셋이 같다: True
  3차원 .T  : (4, 3, 2) — 축을 전부 거꾸로 돌린다
    경고    : The use of `x.T` on tensors of dimension other than 2 to reverse their shape is deprecated
  3차원 .mT : (2, 4, 3) — 마지막 두 축만 바꾼다
  3차원 .H  : tensor.H is only supported on matrices (2-D tensors). Got 3-D tens
  복소수 mT : [(1+2j), (3-1j)]
  복소수 H  : [(1-2j), (3+1j)] — 켤레까지 취한다
```

`mps`가 찍힌 줄들은 이 쪽을 돌린 기계가 애플 실리콘이기 때문이다. CUDA 기계에서는
`cuda`로, 가속기가 없으면 해당 절을 건너뛴다.

## 2. 모양 — 원소 수가 같은 것과 모양이 같은 것은 다르다

`(3,4)`와 `(4,3)`은 둘 다 원소가 12개다. 그래도 더할 수 없다.

```
The size of tensor a (4) must match the size of tensor b (3)
at non-singleton dimension 1
```

메시지가 `non-singleton`이라고 말하는 데 뜻이 있다. 길이가 1인 축이라면
**브로드캐스트가 늘려 준다.** 그래서 `(3,4) + (3,1)`은 통과하고 `(3,4) + (4,)`도
통과한다. 뒤는 축이 하나 모자라는데, 모자라는 축은 앞에 1을 붙여 `(1,4)`로
보고 늘린다.

행렬곱은 또 다른 규칙이다. `(3,4) @ (3,4)`는 안 되고 `(3,4) @ (4,3)`은 된다.
더하기에서 어긋나는 짝이 곱하기에서는 맞물린다. 그러니 "모양이 맞다"는 말은
**어떤 연산에 대해** 맞는다는 뜻이고, 연산마다 다르다.

## 3. 장치 — 자동으로 옮겨 주지 않는다

```
Expected all tensors to be on the same device, but found at least
two devices, mps:0 and cpu!
```

자료형은 한쪽으로 올려 주면서 장치는 옮겨 주지 않는다. 일부러 그렇게 만들어
두었다 — 장치를 옮기는 일은 복사이고 그것도 느린 복사이므로, 몰래 해 주면 학습
루프가 까닭 없이 느려진다. 「GPU 연산」에서 본 전송 비용이 그 까닭이다.

그래서 장치는 코드가 **한 군데서** 정하고 입력과 모델을 둘 다 거기로 보내는 쪽이
맞다. 「실전 예제」의 `x.to(DEV)`가 매 배치마다 들어 있는 것이 그래서다.

## 4. 자료형 — 조용히 올라간다, 그런데 모양이 결정한다

자료형이 어긋나면 터지지 않고 **더 넓은 쪽으로 올라간다.** 그런데 누가 이기는지가
자료형만으로 정해지지 않는다.

| 식 | 결과 자료형 |
|---|---|
| `float32 + torch.tensor(1.0, dtype=float64)` | **float32** |
| `float32 + torch.ones(1, dtype=float64)` | **float64** |
| `float32 + torch.ones(3, dtype=float64)` | float64 |
| `float32 + 1.0` (파이썬 수) | float32 |

0차원 텐서와 파이썬 수는 **못 이긴다.** 원소가 하나뿐인 `(1,)` 텐서는 이긴다.
같은 float64 하나인데 `torch.tensor(1.0, dtype=float64)`는 지고
`torch.ones(1, dtype=float64)`는 이긴다 — 모양이 있느냐가 가른다.

파이토치가 0차원을 "덧붙은 수"로 취급해 상대 자료형을 따라가게 해 둔 것이다.
덕분에 `x * 0.5`가 float32를 유지하지만, 같은 0.5를 `torch.ones(1)`에 담아
넘기면 모델 전체가 float64로 넘어간다. 연습문제 2에서 그 값이 얼마인지 잰다.

정수에서는 또 다르다.

```
int / 2   -> torch.float32 [0.5, 1.0, 1.5]
int // 2  -> torch.int64   [0, 1, 1]
int.mean(): mean(): could not infer output dtype.
```

`/`는 정수끼리라도 실수를 돌려준다(파이썬 3과 같다). `//`는 정수를 지킨다.
그리고 `mean()`은 올려 주지 않고 **거절한다** — 평균의 자료형을 짐작하지 않겠다는
뜻이다. 자료형 가운데 터져 주는 드문 자리다.

## 5. `.to()`는 바꿀 것이 없으면 아무것도 만들지 않는다

```
x.to(torch.float32) is x : True      ← 이미 float32다
x.to('cpu')         is x : True      ← 이미 cpu다
x.to(torch.float64) is x : False     ← 바꿀 것이 있다
```

`is`가 `True`라는 것은 **같은 객체**를 돌려받았다는 뜻이다. 복사도 하지 않는다.
그래서 장치·자료형을 맞추는 코드에 `.to(...)`를 넉넉히 뿌려 두어도 공짜다. 이미
맞으면 아무 일도 일어나지 않는다.

반대쪽이 흔한 실수다. `.to()`는 **제자리에서 바꾸지 않는다.**

```
x.to(torch.float64) 만 부른 뒤 x.dtype : torch.float32
```

`nn.Module`의 `.to()`는 모듈을 제자리에서 바꾸므로 `model.to(dev)`만 써도 되지만,
텐서는 반환값을 받아야 한다. 같은 이름의 메서드가 한쪽은 제자리, 한쪽은 반환이라
섞이기 쉽다.

## 6. `T`·`mT`·`H`는 2차원에서만 셋이 같다

2차원 실수 행렬에서는 셋이 똑같은 것을 돌려준다. 차원이 늘거나 복소수가 되면
갈린다.

| | `(2,3,4)` 입력 | 복소수 `1+2j` |
|---|---|---|
| `.T` | `(4,3,2)` — 축을 **전부** 거꾸로, **deprecated** | 켤레 없음 |
| `.mT` | `(2,4,3)` — 마지막 두 축만 | `1+2j` 그대로 |
| `.H` | `RuntimeError` — 2차원만 | `1-2j` 켤레까지 |

`.T`는 3차원에서 축을 전부 뒤집고, 파이토치가 경고를 띄운다.

```
The use of `x.T` on tensors of dimension other than 2 to reverse
their shape is deprecated
```

배치 행렬을 다루는 코드에서 **원하는 것은 거의 언제나 `.mT`**다. 배치 축은 그대로
두고 행렬만 전치해야 하니까. `.T`를 쓰면 배치 축까지 끌려 들어가는데, 모양이 운
좋게 맞으면 **터지지 않고 틀린 값이 나온다.** 연습문제 3이 그 자리다.

## 7. 정리

| | 어긋나면 | 맞추는 법 |
|---|---|---|
| 모양 | 터진다 (브로드캐스트가 받는 경우는 빼고) | `reshape`·`view`·`unsqueeze` |
| 장치 | 터진다 | `.to(device)` — 반환값을 받아야 한다 |
| 자료형 | 조용히 올라간다 | `.to(dtype)`·`.float()`, 그리고 먼저 **알아채기** |

셋 가운데 둘은 터져 주므로 고치면 된다. 자료형만은 찾아내는 일이 먼저다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
`torch.arange(12)`를 `(12,)`·`(3,4)`·`(2,6)`·`(2,2,3)`·`(12,1)`로 바꿔 보고
`ndim`·`numel`·`stride`와 저장소 공유 여부를 적어라. 무엇이 달라지고 무엇이
그대로인가?

</div>

??? success "풀이"
    ```python
    x = torch.arange(12)
    for shape in ((12,), (3, 4), (2, 6), (2, 2, 3), (12, 1)):
        v = x.reshape(shape)
        print(f"  {str(shape):12s} ndim={v.ndim}  numel={v.numel()}  "
              f"stride={v.stride()}  저장소 공유={v.data_ptr() == x.data_ptr()}")
    a, b = x.reshape(3, 4), x.reshape(4, 3)
    try:
        a + b
    except RuntimeError as e:
        print("  (3,4)+(4,3):", str(e).splitlines()[0][:66])
    print("  (3,4)@(4,3):", tuple((a @ b).shape), "— 행렬곱은 맞물린다")
    ```

    ```
      (12,)        ndim=1  numel=12  stride=(1,)  저장소 공유=True
      (3, 4)       ndim=2  numel=12  stride=(4, 1)  저장소 공유=True
      (2, 6)       ndim=2  numel=12  stride=(6, 1)  저장소 공유=True
      (2, 2, 3)    ndim=3  numel=12  stride=(6, 3, 1)  저장소 공유=True
      (12, 1)      ndim=2  numel=12  stride=(1, 1)  저장소 공유=True
      (3,4)+(4,3): The size of tensor a (4) must match the size of tensor b (3) at no
      (3,4)@(4,3): (3, 3) — 행렬곱은 맞물린다
    ```

    **`numel`과 저장소는 다섯 모두 같다.** 바뀌는 것은 `ndim`과 `stride`뿐이다.
    `reshape`은 수를 하나도 옮기지 않고 **읽는 순서만 다시 적어** 준다. 12개가
    메모리에 한 줄로 누워 있는 것은 그대로고, `(3,4)`는 "네 칸마다 줄을 바꿔
    읽어라", `(2,6)`은 "여섯 칸마다"라고 적어 둔 것이다.

    `(12,1)`의 `stride=(1,1)`이 눈에 걸린다. 둘째 축의 길이가 1이라 그 축으로
    움직일 일이 없으니 보폭이 아무 값이어도 된다. 길이 1인 축의 보폭은 뜻이 없다.

    그리고 `numel`이 같아도 더할 수 없다. 모양이 맞는지는 원소 수가 아니라 축마다
    따져야 하고, 연산마다 따지는 방식이 다르다 — `(3,4)`와 `(4,3)`은 더하기에서는
    어긋나고 곱하기에서는 맞물린다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
float64 텐서 하나가 섞여 들면 그 뒤가 전부 float64로 올라간다. 값이 얼마인지
재어라. $1500 \times 1500$ 행렬곱의 시간과 메모리를 두 자료형에서 비교하고,
4절의 "0차원은 못 이긴다"를 확인하라.

</div>

??? success "풀이"
    ```python
    import time

    n = 1500
    for dt in (torch.float32, torch.float64):
        m = torch.randn(n, n, dtype=dt)
        got = []
        for _ in range(5):
            t0 = time.perf_counter(); m @ m; got.append(time.perf_counter() - t0)
        print(f"  {str(dt):15s} {min(got) * 1e3:6.1f} ~ {max(got) * 1e3:6.1f} ms   "
              f"{m.numel() * m.element_size() / 1e6:.0f} MB")

    f32 = torch.randn(4, dtype=torch.float32)
    print("  float32 * torch.tensor(0.5, float64) ->",
          (f32 * torch.tensor(0.5, dtype=torch.float64)).dtype)
    print("  float32 * torch.ones(1, float64)     ->",
          (f32 * torch.ones(1, dtype=torch.float64)).dtype)
    ```

    ```
      torch.float32      5.1 ~    6.6 ms   9 MB
      torch.float64     21.8 ~   22.2 ms   18 MB
      float32 * torch.tensor(0.5, float64) -> torch.float32
      float32 * torch.ones(1, float64)     -> torch.float64
    ```

    메모리는 정확히 두 배다(9 MB → 18 MB). 원소 하나가 4바이트에서 8바이트가
    되니 당연하다.

    시간은 **네 배쯤** 느리다(5.1 → 21.8 ms). 두 배가 아니다. 메모리를 두 배
    읽는 것에 더해, CPU의 벡터 명령이 한 번에 다루는 float64 개수가 float32의
    절반이기 때문이다. GPU에서는 격차가 더 벌어지는 경우가 많다 — 소비자용
    카드는 float64 연산기를 적게 넣어 둔다.

    그래서 float64 하나가 섞여 드는 것은 작은 일이 아니다. 그리고 그것이
    `torch.ones(1, ...)` 같은 길이-1 텐서로 들어오면 **터지지도 않고 느려지기만**
    한다. 학습이 왜 느린지 찾을 때 `print(x.dtype)`을 한 줄 넣어 볼 자리다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
`(3,3,3)` 텐서 둘을 배치 행렬곱 하려 한다. `A @ B.T`와 `A @ B.mT`를 둘 다
계산하라. **모양이 같게 나오는데 값이 다르다.** 어느 쪽이 맞는지 손으로 확인하고,
왜 이 모양에서는 터지지 않는지 설명하라.

</div>

??? success "풀이"
    ```python
    import warnings

    torch.manual_seed(0)
    A, B = torch.randn(3, 3, 3), torch.randn(3, 3, 3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")   # .T 의 deprecated 경고를 끈다
        bad = A @ B.T
    good = A @ B.mT
    print("  A @ B.T  모양:", tuple(bad.shape),
          "  A @ B.mT 모양:", tuple(good.shape))
    print("  두 결과가 같은가:", torch.allclose(bad, good))
    print("  A @ B.T  [0,0,0] =", round(bad[0, 0, 0].item(), 4))
    print("  A @ B.mT [0,0,0] =", round(good[0, 0, 0].item(), 4))
    print("  손으로 (A[0] @ B[0].T)[0,0] =", round((A[0] @ B[0].T)[0, 0].item(), 4))
    ```

    ```
      A @ B.T  모양: (3, 3, 3)   A @ B.mT 모양: (3, 3, 3)
      두 결과가 같은가: False
      A @ B.T  [0,0,0] = -1.5922
      A @ B.mT [0,0,0] = 0.0782
      손으로 (A[0] @ B[0].T)[0,0] = 0.0782
    ```

    **`.mT`가 맞다.** 손으로 첫 배치만 따로 계산한 `A[0] @ B[0].T`의 `[0,0]`이
    0.0782이고, `.mT` 쪽이 그 값이다. `.T` 쪽 −1.5922는 아무 뜻도 없는 수다.

    까닭은 `.T`가 **배치 축까지 뒤집기** 때문이다. `B`가 `(배치, 행, 열)`인데
    `.T`는 `(열, 행, 배치)`로 만들어 버린다. 그러고 나서 행렬곱을 하면 세 번째
    배치의 데이터가 첫 번째 배치의 행렬에 섞여 들어간다.

    이 모양에서 터지지 않는 까닭은 **세 축의 길이가 모두 3이라서**다. `(3,3,3)`을
    어떻게 뒤집어도 `(3,3,3)`이므로 행렬곱의 모양 검사를 통과한다. `(2,3,4)`였다면
    `.T`가 `(4,3,2)`를 만들어 배치 축 2와 4가 어긋나 터졌을 것이다.

    그래서 이것은 **정사각 배치에서만 조용한 버그**다. 개발하면서 `(2,3,4)`처럼
    길이가 다른 입력으로 돌려 보면 터져서 알게 되고, 하필 `(batch, d, d)` 꼴로
    돌리면 — 어텐션의 헤드별 행렬이 흔히 그렇다 — 숫자만 틀린 채 학습이 돌아간다.

    규칙은 짧다. **배치가 있으면 `.mT`를 쓴다.** `.T`는 2차원 전용이고, 그래서
    파이토치도 2차원 아닌 쓰임을 deprecated로 표시해 두었다.
