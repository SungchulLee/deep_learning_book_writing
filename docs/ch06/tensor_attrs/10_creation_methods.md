# 생성 방법 — 베끼는 것, 나눠 쓰는 것, 아무것도 안 하는 것

텐서를 만드는 함수가 많지만 외울 것은 몇 가지뿐이고, 그중 둘은 모르면 조용히
틀리는 종류다.

- `tensor`는 **베끼고** `as_tensor`·`from_numpy`는 **나눠 쓴다**
- `empty`는 **아무것도 하지 않는다**. 그 자리에 있던 값이 그대로 나온다
- `arange`는 끝을 뺀다고 적혀 있지만 **들어오는 자리가 있다**
- `*_like`는 `dtype`과 `device`는 따라가고 `requires_grad`는 안 따라간다

## 1. 코드

```python
"""텐서를 만드는 길들 — 베끼는 것, 나눠 쓰는 것, 아무것도 안 하는 것.

만드는 함수가 많지만 외울 것은 몇 가지뿐이다.

- `tensor` 는 베끼고 `as_tensor`·`from_numpy` 는 나눠 쓴다
- `empty` 는 **아무것도 하지 않는다**. 그 자리에 있던 값이 그대로 나온다
- `arange` 는 끝을 뺀다고 적혀 있지만 부동소수점 때문에 들어오는 자리가 있다.
  개수를 내가 정하려면 `linspace` 다
- `*_like` 는 dtype 과 device 는 따라가고 `requires_grad` 는 안 따라간다

실행: 이 쪽의 코드를 파일로 갈무리해 돌린다
"""
import numpy as np
import torch


def header(title):
    print(f"\n=== {title} ===")


# ========================================================================
# 1. 베끼는가 나눠 쓰는가
# ========================================================================

def copy_or_share():
    header("1. tensor / as_tensor / from_numpy")
    arr = np.array([[1, 2], [3, 4]])
    for name, fn in (("torch.tensor", torch.tensor),
                     ("torch.as_tensor", torch.as_tensor),
                     ("torch.from_numpy", torch.from_numpy)):
        t = fn(arr)
        print(f"  {name:18s} dtype {str(t.dtype):12s} "
              f"메모리 공유 {np.shares_memory(arr, t.numpy())}")
    arr[0, 0] = 999
    print("  배열을 999 로 고친 뒤")
    print("    torch.tensor 로 만든 것  :", torch.tensor([[1, 2], [3, 4]])[0, 0].item(),
          "(미리 베꼈다)")
    print("    torch.as_tensor 로 만든 것:", torch.as_tensor(arr)[0, 0].item())

    f32 = np.array([[1.0, 2.0]], dtype=np.float32)
    shared = np.shares_memory(f32, torch.as_tensor(f32, dtype=torch.float64).numpy())
    print("  as_tensor 에 dtype 을 바꿔 달라면 공유:", shared,
          "— 바꾸려면 베낄 수밖에 없다")


# ========================================================================
# 2. empty 는 아무것도 하지 않는다
# ========================================================================

def empty_is_garbage():
    header("2. empty 가 주는 것")
    a = torch.empty(5)
    print("  갓 잡은 empty(5):", a.tolist())
    print("  0 처럼 보인다. 운영체제가 새 쪽을 0 으로 비워 주기 때문이다")

    dirty = torch.empty(1000)
    dirty.fill_(12345.0)       # 그 자리를 더럽혀 놓고
    del dirty                  # 놓아 준다
    b = torch.empty(1000)      # 같은 자리를 다시 받는다
    print("  더럽힌 뒤 다시 잡은 empty(1000) 의 앞 다섯:",
          [round(v, 1) for v in b[:5].tolist()])
    print("  0 이 아닌 칸:", int((b != 0).sum()), "/ 1000")
    print("  방금 놓아 준 내 데이터가 그대로 나온다")
    print("  그러니 empty 로 만든 텐서는 반드시 채우고 나서 읽어야 한다")


# ========================================================================
# 3. 값을 정해 만들기
# ========================================================================

def constants():
    header("3. 값을 정해 만들기")
    print("  zeros(2,3)      :", torch.zeros(2, 3).flatten().tolist())
    print("  ones(2,3)       :", torch.ones(2, 3).flatten().tolist())
    print("  full((2,3), 7.5):", torch.full((2, 3), 7.5).flatten().tolist())
    print("  eye(3)          :", torch.eye(3).tolist())
    print("  eye(2,4) 정사각이 아니어도:", torch.eye(2, 4).tolist())

    v = torch.tensor([1.0, 2.0, 3.0])
    print("  diag(벡터) 는 행렬을 만든다:", torch.diag(v).shape)
    m = torch.arange(9).reshape(3, 3)
    print("  diag(행렬) 은 대각을 뽑는다:", torch.diag(m).tolist())
    print("  diag(행렬, 1) 한 칸 위  :", torch.diag(m, 1).tolist())
    print("  같은 이름인데 입력 차원에 따라 하는 일이 반대다")


# ========================================================================
# 4. 수열 — arange 와 linspace
# ========================================================================

def sequences():
    header("4. arange 와 linspace")
    for a, b, s in ((0, 1, 0.1), (0, 0.3, 0.1), (1, 1.3, 0.1)):
        r = torch.arange(a, b, s)
        inc = abs(r[-1].item() - b) < 1e-6
        print(f"  arange({a}, {b}, {s}) -> {r.numel()}개  "
              f"끝값 {b} 가 들어갔는가: {inc}")
    print("  셋 다 폭이 같은데 마지막만 끝값을 넣었다")
    print("  (1.3-1)/0.1 이 부동소수점에서 3.0000000000000004 가 되기 때문이다")
    print("  linspace(1, 1.2, 3):", torch.linspace(1, 1.2, 3).tolist())
    print("  linspace 는 개수를 내가 적으므로 이런 일이 없다")

    r64 = torch.arange(0, 1, 0.1, dtype=torch.float64)
    print("  그리고 값 자체도 정확하지 않다")
    print("    arange(0,1,0.1)[3] =", repr(r64[3].item()))
    print("    == 0.3 인가:", r64[3].item() == 0.3)


# ========================================================================
# 5. 난수 — 그리고 씨앗
# ========================================================================

def randoms():
    header("5. 난수")
    torch.manual_seed(42)
    print("  rand(2,3)   [0,1) 고른분포:",
          [round(v, 4) for v in torch.rand(2, 3).flatten().tolist()])
    print("  randn(2,3)  N(0,1)       :",
          [round(v, 4) for v in torch.randn(2, 3).flatten().tolist()])
    print("  randint(0,10,(5,))       :", torch.randint(0, 10, (5,)).tolist())
    print("  randperm(8)              :", torch.randperm(8).tolist())

    torch.manual_seed(42)
    first = torch.rand(3)
    torch.manual_seed(42)
    print("  같은 씨앗으로 두 번:", torch.equal(first, torch.rand(3)))

    g = torch.Generator().manual_seed(7)
    print("  Generator 를 따로 두면:",
          [round(v, 4) for v in torch.rand(3, generator=g).tolist()])
    print("  전역 씨앗을 건드리지 않고 재현할 수 있다")


# ========================================================================
# 6. _like 는 어디까지 따라가는가
# ========================================================================

def like_family():
    header("6. *_like")
    tpl = torch.randn(3, 4, dtype=torch.float64, requires_grad=True)
    z = torch.zeros_like(tpl)
    print(f"  본   dtype {tpl.dtype} device {tpl.device} "
          f"requires_grad {tpl.requires_grad}")
    print(f"  like dtype {z.dtype} device {z.device} "
          f"requires_grad {z.requires_grad}")
    print("  dtype 과 device 는 따라가고 requires_grad 는 안 따라간다")
    print("  dtype 만 바꿔 달라고 할 수도 있다:",
          torch.zeros_like(tpl, dtype=torch.int32).dtype)


# ========================================================================
# 7. 장치와 requires_grad
# ========================================================================

def device_and_grad():
    header("7. 만들 때 정하는 것")
    dev = ("cuda" if torch.cuda.is_available()
           else "mps" if torch.backends.mps.is_available() else "cpu")
    t = torch.randn(2, 2, device=dev)
    print(f"  device='{dev}' 로 바로 만들기:", t.device)
    print("  만들고 나서 옮기면 한 번 더 베낀다 — 처음부터 거기서 만드는 편이 낫다")
    x = torch.randn(2, 2, requires_grad=True)
    print("  requires_grad=True 로 만들기:", x.requires_grad, " 잎인가:", x.is_leaf)


def main():
    copy_or_share()
    empty_is_garbage()
    constants()
    sequences()
    randoms()
    like_family()
    device_and_grad()


if __name__ == "__main__":
    main()
```

**출력:**

```
=== 1. tensor / as_tensor / from_numpy ===
  torch.tensor       dtype torch.int64  메모리 공유 False
  torch.as_tensor    dtype torch.int64  메모리 공유 True
  torch.from_numpy   dtype torch.int64  메모리 공유 True
  배열을 999 로 고친 뒤
    torch.tensor 로 만든 것  : 1 (미리 베꼈다)
    torch.as_tensor 로 만든 것: 999
  as_tensor 에 dtype 을 바꿔 달라면 공유: False — 바꾸려면 베낄 수밖에 없다

=== 2. empty 가 주는 것 ===
  갓 잡은 empty(5): [0.0, 0.0, 0.0, 0.0, 0.0]
  0 처럼 보인다. 운영체제가 새 쪽을 0 으로 비워 주기 때문이다
  더럽힌 뒤 다시 잡은 empty(1000) 의 앞 다섯: [12345.0, 12345.0, 12345.0, 12345.0, 12345.0]
  0 이 아닌 칸: 1000 / 1000
  방금 놓아 준 내 데이터가 그대로 나온다
  그러니 empty 로 만든 텐서는 반드시 채우고 나서 읽어야 한다

=== 3. 값을 정해 만들기 ===
  zeros(2,3)      : [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
  ones(2,3)       : [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
  full((2,3), 7.5): [7.5, 7.5, 7.5, 7.5, 7.5, 7.5]
  eye(3)          : [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
  eye(2,4) 정사각이 아니어도: [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
  diag(벡터) 는 행렬을 만든다: torch.Size([3, 3])
  diag(행렬) 은 대각을 뽑는다: [0, 4, 8]
  diag(행렬, 1) 한 칸 위  : [1, 5]
  같은 이름인데 입력 차원에 따라 하는 일이 반대다

=== 4. arange 와 linspace ===
  arange(0, 1, 0.1) -> 10개  끝값 1 가 들어갔는가: False
  arange(0, 0.3, 0.1) -> 3개  끝값 0.3 가 들어갔는가: False
  arange(1, 1.3, 0.1) -> 4개  끝값 1.3 가 들어갔는가: True
  셋 다 폭이 같은데 마지막만 끝값을 넣었다
  (1.3-1)/0.1 이 부동소수점에서 3.0000000000000004 가 되기 때문이다
  linspace(1, 1.2, 3): [1.0, 1.100000023841858, 1.2000000476837158]
  linspace 는 개수를 내가 적으므로 이런 일이 없다
  그리고 값 자체도 정확하지 않다
    arange(0,1,0.1)[3] = 0.30000000000000004
    == 0.3 인가: False

=== 5. 난수 ===
  rand(2,3)   [0,1) 고른분포: [0.8823, 0.915, 0.3829, 0.9593, 0.3904, 0.6009]
  randn(2,3)  N(0,1)       : [1.1561, 0.3965, -2.4661, 0.3623, 0.3765, -0.1808]
  randint(0,10,(5,))       : [7, 6, 9, 6, 3]
  randperm(8)              : [7, 4, 3, 1, 5, 6, 0, 2]
  같은 씨앗으로 두 번: True
  Generator 를 따로 두면: [0.5349, 0.1988, 0.6592]
  전역 씨앗을 건드리지 않고 재현할 수 있다

=== 6. *_like ===
  본   dtype torch.float64 device cpu requires_grad True
  like dtype torch.float64 device cpu requires_grad False
  dtype 과 device 는 따라가고 requires_grad 는 안 따라간다
  dtype 만 바꿔 달라고 할 수도 있다: torch.int32

=== 7. 만들 때 정하는 것 ===
  device='mps' 로 바로 만들기: mps:0
  만들고 나서 옮기면 한 번 더 베낀다 — 처음부터 거기서 만드는 편이 낫다
  requires_grad=True 로 만들기: True  잎인가: True
```

## 2. 베끼는가 나눠 쓰는가

| | dtype | 메모리 공유 |
|---|---|---|
| `torch.tensor(arr)` | 그대로 | **False** — 베낀다 |
| `torch.as_tensor(arr)` | 그대로 | **True** |
| `torch.from_numpy(arr)` | 그대로 | **True** |

고르는 기준은 **그 배열을 뒤에 또 건드릴 작정인가**다. 전처리 버퍼를 돌려 쓰는
코드라면 `as_tensor`로 넘긴 텐서가 조용히 바뀐다. 한 번 만들고 버리는 배열이라면
공유가 공짜다.

`as_tensor`에 `dtype`을 바꿔 달라고 하면 공유가 깨진다. 당연하다 — 자료형을 바꾸려면
값을 새로 적어야 한다. 그래서 `as_tensor(arr, dtype=torch.float32)`는 이름과 달리
`tensor`와 같은 일을 한다.

## 3. `empty`는 아무것도 하지 않는다

```
갓 잡은 empty(5): [0.0, 0.0, 0.0, 0.0, 0.0]
```

0처럼 보인다. 그래서 "`empty`는 0을 준다"고 오해하기 쉽다. 운영체제가 **새로 주는
메모리 쪽을 0으로 비워서** 건네기 때문에 그렇게 보이는 것뿐이다.

같은 자리를 더럽혀 놓고 다시 받아 보면 정체가 드러난다.

```
더럽힌 뒤 다시 잡은 empty(1000) 의 앞 다섯: [12345.0, 12345.0, ...]
0 이 아닌 칸: 1000 / 1000
```

**방금 놓아 준 내 데이터가 그대로 나온다.** `empty`는 자리만 잡아 주고 값은 건드리지
않는다. 그러니 `empty`로 만든 텐서는 **반드시 채우고 나서 읽어야** 한다.

> 그래서 `empty`의 결과는 **쪽에 실을 수 없다.** 돌릴 때마다 다른 값이 나오므로
> 찍어 둔 숫자가 독자의 화면에 그대로 나올 까닭이 없다. 위에서 12345가 나온 것은
> 바로 앞 줄에서 그 자리를 그 값으로 채웠다 놓아 주었기 때문이고, 그렇게 **무엇이
> 들어갈지 정해 둔 경우에만** 보일 수 있다.

## 4. 같은 이름, 반대 일 — `diag`

```
diag(벡터) 는 행렬을 만든다: torch.Size([3, 3])
diag(행렬) 은 대각을 뽑는다: [0, 4, 8]
diag(행렬, 1) 한 칸 위  : [1, 5]
```

`torch.diag`는 **입력 차원에 따라 하는 일이 반대**다. 1차원을 주면 그것을 대각에
놓은 행렬을 만들고, 2차원을 주면 대각을 뽑아 1차원으로 준다. 헷갈리면
`torch.diagflat`(언제나 만든다)과 `torch.diagonal`(언제나 뽑는다)로 적는다.

## 5. `arange`는 끝을 뺀다 — 다만 언제나는 아니다

```
arange(0, 1,   0.1) -> 10개  끝값 1   가 들어갔는가: False
arange(0, 0.3, 0.1) ->  3개  끝값 0.3 가 들어갔는가: False
arange(1, 1.3, 0.1) ->  4개  끝값 1.3 가 들어갔는가: True
```

셋 다 폭이 0.3 또는 1.0이고 간격이 같은데, **마지막만 끝값을 넣었다.**

까닭은 `arange`가 개수를 $\lceil (\text{end} - \text{start}) / \text{step} \rceil$로
셈하기 때문이다. $1.3 - 1$이 부동소수점에서 정확히 $0.3$이 아니라
$0.30000000000000004$이고, 이것을 $0.1$로 나누면 $3.0000000000000004$가 되어
올림하면 **4**가 된다.

그리고 값 자체도 정확하지 않다.

```
arange(0,1,0.1)[3] = 0.30000000000000004
== 0.3 인가: False
```

그래서 규칙은 이렇다. **정수 간격이면 `arange`, 실수 간격이면 `linspace`.**
`linspace`는 개수를 내가 적으므로 몇 개가 나올지 세어 볼 필요가 없고, 끝값도
반드시 들어간다.

## 6. `*_like`는 어디까지 따라가는가

```
본   dtype float64  device cpu  requires_grad True
like dtype float64  device cpu  requires_grad False
```

`dtype`과 `device`는 따라가고 **`requires_grad`는 안 따라간다.** 대개 이것이 원하는
바다 — `zeros_like(w)`로 만든 것은 보통 버퍼나 마스크이지 학습할 파라미터가
아니다.

`*_like`를 쓰는 진짜 값은 **장치를 안 틀린다**는 데 있다. 모델이 GPU에 있을 때
`torch.zeros(x.shape)`로 만들면 CPU에 생겨 바로 터지지만, `torch.zeros_like(x)`는
언제나 `x`가 있는 곳에 생긴다.

## 7. 만들 때 정하는 것

```python
torch.randn(2, 2, device=dev)        # 처음부터 거기서 만든다
torch.randn(2, 2).to(dev)            # 만들고 옮긴다 — 한 번 더 베낀다
```

둘은 같은 결과를 주지만 뒤의 것은 CPU에 한 번 만들고 복사한다. 큰 텐서에서는
그 복사가 아깝고, 「GPU 연산」에서 본 전송 비용이 그대로 붙는다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
$3 \times 3$ 영행렬을 여섯 가지로 만들어라. 값은 모두 같은가? **`dtype`도 모두
같은가?**

</div>

??? success "풀이"
    ```python
    ways = (("torch.zeros(3,3)", lambda: torch.zeros(3, 3)),
            ("torch.full((3,3),0.)", lambda: torch.full((3, 3), 0.)),
            ("torch.empty(3,3).zero_()", lambda: torch.empty(3, 3).zero_()),
            ("torch.zeros_like(torch.ones(3,3))",
             lambda: torch.zeros_like(torch.ones(3, 3))),
            ("torch.tensor([[0.]*3]*3)", lambda: torch.tensor([[0.] * 3] * 3)),
            ("torch.arange(9).reshape(3,3)*0",
             lambda: torch.arange(9).reshape(3, 3) * 0))
    for n, f in ways:
        t = f()
        print(f"  {n:36s} dtype {t.dtype}  모두 0 {bool((t == 0).all())}")
    ```

    ```
      torch.zeros(3,3)                     dtype torch.float32  모두 0 True
      torch.full((3,3),0.)                 dtype torch.float32  모두 0 True
      torch.empty(3,3).zero_()             dtype torch.float32  모두 0 True
      torch.zeros_like(torch.ones(3,3))    dtype torch.float32  모두 0 True
      torch.tensor([[0.]*3]*3)             dtype torch.float32  모두 0 True
      torch.arange(9).reshape(3,3)*0       dtype torch.int64  모두 0 True
    ```

    값은 여섯 다 같은데 **마지막 하나만 `int64`**다. `arange`는 정수를 주고, 거기에
    0을 곱해도 정수다.

    이것이 조용히 틀리는 자리다. 영행렬을 만들어 모델에 넣으면 「형상, 자료형,
    장치」에서 본 `mat1 and mat2 must have the same dtype`가 나거나, 더 나쁘게는
    정수 나눗셈으로 바뀌어 값이 잘린다. 만드는 길이 여럿일 때는 `dtype`을 한 번
    찍어 보는 편이 싸다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
`empty`는 채우지 않으니 `zeros`보다 빠를 것이다. 얼마나 빠른지 재고, **`empty`로
만들어 0으로 채우면** 어떻게 되는지도 재어라.

</div>

??? success "풀이"
    ```python
    import time

    def spread(fn, rep=9):
        ts = []
        for _ in range(rep):
            t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
        return min(ts) * 1e3, max(ts) * 1e3

    for n in (10 ** 6, 10 ** 7):
        z = spread(lambda: torch.zeros(n))
        e = spread(lambda: torch.empty(n))
        ef = spread(lambda: torch.empty(n).fill_(0.0))
        print(f"  {n:>9,}개  zeros {z[0]:6.3f}~{z[1]:6.3f}   "
              f"empty {e[0]:6.3f}~{e[1]:6.3f}   "
              f"empty+fill_ {ef[0]:6.3f}~{ef[1]:6.3f} ms")
    ```

    ```
      1,000,000개  zeros  0.087~ 0.350   empty  0.025~ 0.030   empty+fill_  0.157~ 0.193 ms
      10,000,000개  zeros  1.067~ 1.608   empty  0.108~ 0.110   empty+fill_  1.060~ 1.711 ms
    ```

    `empty` 혼자는 1천만 개에서 **열 배 가까이 빠르다**(0.108 대 1.067 ms). 자리만
    잡고 아무것도 안 적으니 크기와 거의 무관하다.

    그런데 `empty` 뒤에 `fill_(0)`을 붙이면 **`zeros`와 같아진다**(1.060 대 1.067).
    당연하다 — 결국 1천만 칸에 0을 적는 일을 하기 때문이다.

    그래서 `empty`가 버는 것은 **채우지 않을 때만** 생긴다. 바로 뒤에 다른 값으로
    덮어쓸 자리(출력 버퍼 같은 것)라면 공짜고, 0이 필요한 자리라면 `zeros`를 쓰는
    편이 짧고 안전하다. 「연산」에서 본 `out=` 인자가 쓰이는 자리가 앞의 경우다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
`torch.manual_seed`를 부르면 재현된다고 한다. 그런데 **코드 사이에 난수를 하나
더 뽑는 줄이 끼어들면** 뒤의 값이 달라진다. 확인하고, `torch.Generator`로 그것을
막아라. 이 차이가 왜 실제로 문제가 되는지 적어라.

</div>

??? success "풀이"
    ```python
    torch.manual_seed(0)
    a = torch.rand(2)
    noise = torch.rand(2)       # 사이에 난수가 하나 끼었다
    b = torch.rand(2)

    torch.manual_seed(0)
    a2 = torch.rand(2)
    b2 = torch.rand(2)          # 끼지 않았다
    print("  사이에 난수가 끼면 b 가 달라지는가:", not torch.equal(b, b2))

    g1 = torch.Generator().manual_seed(0)
    g2 = torch.Generator().manual_seed(0)
    x1 = torch.rand(2, generator=g1)
    _ = torch.rand(100)         # 전역을 아무리 써도
    x2 = torch.rand(2, generator=g2)
    print("  Generator 를 쓰면 전역을 써도 같은가:", torch.equal(x1, x2))
    ```

    ```
      사이에 난수가 끼면 b 가 달라지는가: True
      Generator 를 쓰면 전역을 써도 같은가: True
    ```

    전역 씨앗은 **난수를 뽑는 순서 전체**를 정한다. 그래서 `manual_seed(0)`이
    보장하는 것은 "프로그램을 처음부터 똑같이 돌리면 똑같다"일 뿐, 중간의 어느
    값 하나가 고정되는 것이 아니다.

    실제로 걸리는 자리가 여럿이다.

    - **드롭아웃을 켜고 끄면** 뽑는 난수의 개수가 달라져, 그 뒤의 가중치
      초기화나 데이터 섞기가 전부 밀린다. 두 모델을 견주려던 것인데 데이터
      순서까지 달라진다.
    - **디버깅하려고 `print`에 `torch.rand`를 하나 끼워 넣으면** 그 뒤가 전부
      바뀐다. 고치려고 넣은 줄이 재현을 깬다.
    - **에폭 수를 바꾸면** 다음 실행의 섞기가 달라진다.

    `torch.Generator`를 따로 만들어 그 자리에만 넘기면 전역과 끊어진다. 위에서
    전역 난수를 100개 뽑아도 `g2`의 값은 그대로였다.

    파이토치의 `DataLoader`가 `generator=` 인자를 받는 것이 이 까닭이고,
    「데이터 적재」에서 일꾼마다 씨앗을 따로 주어야 했던 것도 같은 이야기다.
    **재현을 원하면 씨앗 하나가 아니라 난수의 출처를 나누어야 한다.**
