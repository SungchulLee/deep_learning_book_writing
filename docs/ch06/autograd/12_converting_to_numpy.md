# 텐서를 NumPy로 넘기기 — 거절당하는 두 자리와 조용히 공유되는 메모리

`.numpy()`는 세 가지를 한꺼번에 묻는다.

| 묻는 것 | 어긋나면 |
|---|---|
| 그래프에 매여 있는가 (`requires_grad`) | **거절한다** |
| CPU에 있는가 | **거절한다** |
| 메모리를 나눠 쓸 것인가 | 거절하지 않는다. **그냥 나눠 쓴다** |

앞의 둘은 터져 주니 메시지대로 고치면 끝이다. **셋째가 조용하다**, 그리고 그것이
이 쪽에서 볼 것이다.

## 1. 코드

```python
"""텐서를 NumPy 로 넘기기 — 거절당하는 두 자리와 공유되는 메모리.

`.numpy()` 는 세 가지를 한꺼번에 묻는다.

- **그래프에 매여 있는가** — `requires_grad=True` 면 거절한다
- **어느 장치에 있는가** — CPU 가 아니면 거절한다
- **메모리를 나눠 쓸 것인가** — 거절하지 않는다. 그냥 나눠 쓴다

앞의 둘은 터져 주니 고치면 끝이다. 셋째가 조용하다.

실행: 이 쪽의 코드를 파일로 갈무리해 돌린다
"""
import numpy as np
import torch


def header(title):
    print(f"\n=== {title} ===")


# ========================================================================
# 1. 거절당하는 두 자리
# ========================================================================

def refusals():
    header("1. 거절")
    g = torch.randn(3, requires_grad=True)
    try:
        g.numpy()
    except RuntimeError as e:
        print("  requires_grad=True :", str(e).splitlines()[0])

    dev = ("cuda" if torch.cuda.is_available()
           else "mps" if torch.backends.mps.is_available() else None)
    if dev:
        m = torch.randn(3, device=dev)
        try:
            m.numpy()
        except TypeError as e:
            print(f"  device={dev:5s}      :", str(e).splitlines()[0])
        print("  .cpu().numpy() 로는 된다:", m.cpu().numpy().shape)
    else:
        print("  가속기가 없어 장치 쪽은 건너뛴다")
    print("  두 메시지 모두 고칠 방법을 함께 적어 준다")
    print("  그래서 x.detach().cpu().numpy() 세 토막이 관용구가 되었다")


# ========================================================================
# 2. 메모리는 조용히 공유된다
# ========================================================================

def sharing():
    header("2. 공유")
    t = torch.tensor([1.0, 2.0, 3.0])
    a = t.numpy()
    print("  t      :", t.tolist())
    a[0] = 99.0
    print("  a[0]=99 뒤 t:", t.tolist(), " <- 텐서가 바뀌었다")
    t[1] = 77.0
    print("  t[1]=77 뒤 a:", a.tolist(), " <- 배열도 바뀐다")
    print("  np.shares_memory:", np.shares_memory(t.numpy(), a))

    d = torch.randn(3, requires_grad=True).detach().numpy()
    print("  detach().numpy() 도 공유한다 — detach 가 저장소를 안 끊으니까")

    safe = torch.tensor([1.0, 2.0, 3.0]).numpy().copy()
    print("  끊으려면 .copy() 나 텐서 쪽 .clone() 이 필요하다:", safe.tolist())


# ========================================================================
# 3. 거꾸로 — numpy 에서 텐서로
# ========================================================================

def back():
    header("3. 거꾸로")
    arr = np.array([1.0, 2.0, 3.0])
    shared = torch.from_numpy(arr)
    copied = torch.tensor(arr)
    arr[0] = 55.0
    print("  배열을 55 로 고친 뒤")
    print("    from_numpy  :", shared.tolist(), " dtype", shared.dtype)
    print("    torch.tensor:", copied.tolist(), " dtype", copied.dtype)
    print("  numpy 의 기본 실수형이 float64 라 dtype 도 따라온다")

    f32 = torch.from_numpy(np.array([1.0, 2.0], dtype=np.float32))
    print("  np.float32 로 만들면:", f32.dtype)


# ========================================================================
# 4. 공유가 물어뜯는 자리
# ========================================================================

def the_bite():
    header("4. 공유가 물어뜯는 자리")
    w = torch.tensor([1.0, 2.0], requires_grad=True)
    y = w * 3
    loss = (y ** 2).sum()
    arr = y.detach().numpy()         # 역전파가 저장해 둔 값을 꺼냈다
    arr[0] = 99.0                    # 그것을 numpy 쪽에서 고친다
    try:
        loss.backward()
        print("  numpy 로 고친 뒤 backward: 통과")
        print("    w.grad =", [round(v, 1) for v in w.grad.tolist()],
              " 참값 18w =", [18.0, 36.0])
    except RuntimeError as e:
        print("  numpy 로 고친 뒤 backward:", str(e).splitlines()[0][:62])

    w2 = torch.tensor([1.0, 2.0], requires_grad=True)
    y2 = w2 * 3
    loss2 = (y2 ** 2).sum()
    arr2 = y2.detach().numpy().copy()    # 베껴서 꺼낸다
    arr2[0] = 99.0
    loss2.backward()
    print("  .copy() 로 꺼낸 뒤 backward: w.grad =",
          [round(v, 1) for v in w2.grad.tolist()])
    print("  넘겨준 배열을 고칠 작정이면 반드시 베껴야 한다")


def main():
    refusals()
    sharing()
    back()
    the_bite()


if __name__ == "__main__":
    main()
```

**출력:**

```
=== 1. 거절 ===
  requires_grad=True : Can't call numpy() on Tensor that requires grad. Use tensor.detach().numpy() instead.
  device=mps        : can't convert mps:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.
  .cpu().numpy() 로는 된다: (3,)
  두 메시지 모두 고칠 방법을 함께 적어 준다
  그래서 x.detach().cpu().numpy() 세 토막이 관용구가 되었다

=== 2. 공유 ===
  t      : [1.0, 2.0, 3.0]
  a[0]=99 뒤 t: [99.0, 2.0, 3.0]  <- 텐서가 바뀌었다
  t[1]=77 뒤 a: [99.0, 77.0, 3.0]  <- 배열도 바뀐다
  np.shares_memory: True
  detach().numpy() 도 공유한다 — detach 가 저장소를 안 끊으니까
  끊으려면 .copy() 나 텐서 쪽 .clone() 이 필요하다: [1.0, 2.0, 3.0]

=== 3. 거꾸로 ===
  배열을 55 로 고친 뒤
    from_numpy  : [55.0, 2.0, 3.0]  dtype torch.float64
    torch.tensor: [1.0, 2.0, 3.0]  dtype torch.float64
  numpy 의 기본 실수형이 float64 라 dtype 도 따라온다
  np.float32 로 만들면: torch.float32

=== 4. 공유가 물어뜯는 자리 ===
  numpy 로 고친 뒤 backward: 통과
    w.grad = [594.0, 36.0]  참값 18w = [18.0, 36.0]
  .copy() 로 꺼낸 뒤 backward: w.grad = [18.0, 36.0]
  넘겨준 배열을 고칠 작정이면 반드시 베껴야 한다
```

## 2. 거절하는 두 자리

```
requires_grad=True : Can't call numpy() on Tensor that requires grad.
                     Use tensor.detach().numpy() instead.
device=mps         : can't convert mps:0 device type tensor to numpy.
                     Use Tensor.cpu() to copy the tensor to host memory first.
```

두 메시지 모두 **고칠 방법을 함께 적어 준다.** 그래서 흔히 보는 세 토막

```python
x.detach().cpu().numpy()
```

은 외울 주문이 아니라 **두 거절을 차례로 치우는 것**이다. `detach()`가 첫째를,
`cpu()`가 둘째를 치운다. CPU 텐서라면 `.cpu()`는 같은 객체를 돌려주므로 공짜고,
그래서 넣어 두어도 손해가 없다.

## 3. 메모리는 조용히 공유된다

```
t      : [1.0, 2.0, 3.0]
a[0]=99 뒤 t: [99.0, 2.0, 3.0]    <- 텐서가 바뀌었다
t[1]=77 뒤 a: [99.0, 77.0, 3.0]   <- 배열도 바뀐다
```

**양방향으로 공유한다.** `.numpy()`는 데이터를 베끼지 않고 같은 메모리에 numpy
머리를 씌운다.

`detach()`를 거쳐도 마찬가지다. 「데이터 관련 속성」에서 본 대로 `detach()`는
**그래프만 끊고 저장소는 그대로** 두기 때문이다. 그러니
`x.detach().cpu().numpy()`로 받은 배열도 원본과 한 몸이다.

끊으려면 어딘가에서 베껴야 한다 — numpy 쪽 `.copy()`나 텐서 쪽 `.clone()`.

## 4. 공유가 물어뜯는 자리

```python
y = w * 3
loss = (y ** 2).sum()        # 역전파가 y 를 저장해 둔다
arr = y.detach().numpy()     # 그 y 를 numpy 로 꺼낸다
arr[0] = 99.0                # numpy 쪽에서 고친다
loss.backward()
```

```
numpy 로 고친 뒤 backward: 통과
  w.grad = [594.0, 36.0]   참값 18w = [18.0, 36.0]
```

**터지지 않고 틀린 경사가 나온다.** 참값은 $18w = [18, 36]$인데 594다 —
$2 \times 99 \times 3$, 곧 99로 고쳐진 값으로 센 것이다.

「데이터 관련 속성」의 버전 카운터가 왜 못 잡는가. numpy 쪽에서 쓰는 것은
파이토치가 보지 못하는 길이기 때문이다. **텐서를 넘겨준 순간 파이토치의 안전
장치 바깥으로 나간 것**이고, 그 뒤에 일어나는 일은 아무도 지켜보지 않는다.

`.copy()`로 꺼내면 참값이 나온다.

```
.copy() 로 꺼낸 뒤 backward: w.grad = [18.0, 36.0]
```

규칙은 짧다. **넘겨준 배열을 고칠 작정이면 반드시 베낀다.** 읽기만 할 것이라면
공유가 공짜이므로 그대로 두는 편이 낫다.

## 5. 거꾸로

```
배열을 55 로 고친 뒤
  from_numpy  : [55.0, 2.0, 3.0]   dtype torch.float64
  torch.tensor: [1.0, 2.0, 3.0]    dtype torch.float64
```

`from_numpy`는 공유하고 `torch.tensor`는 베낀다. 그리고 **둘 다 `float64`**가
된다 — numpy의 기본 실수형이 그렇기 때문이다.

「형상, 자료형, 장치」에서 본 그 함정이다. 그대로 모델에 넣으면
`mat1 and mat2 must have the same dtype, but got Double and Float`로 끝난다.
`np.float32`로 만들거나 `.float()`를 붙여야 한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
파이토치의 자료형이 numpy에 모두 있는가? `float32`·`float64`·`int64`·`bool`·
`bfloat16`을 넘겨 보라.

</div>

??? success "풀이"
    ```python
    for name, maker in (("torch.float32", lambda: torch.ones(3)),
                        ("torch.float64", lambda: torch.ones(3, dtype=torch.float64)),
                        ("torch.int64  ", lambda: torch.ones(3, dtype=torch.int64)),
                        ("torch.bool   ", lambda: torch.ones(3, dtype=torch.bool)),
                        ("torch.bfloat16", lambda: torch.ones(3, dtype=torch.bfloat16))):
        t = maker()
        try:
            print(f"  {name} -> numpy dtype {t.numpy().dtype}")
        except TypeError as e:
            print(f"  {name} -> {str(e).splitlines()[0][:56]}")
    ```

    ```
      torch.float32 -> numpy dtype float32
      torch.float64 -> numpy dtype float64
      torch.int64   -> numpy dtype int64
      torch.bool    -> numpy dtype bool
      torch.bfloat16 -> Got unsupported ScalarType BFloat16
    ```

    넷은 그대로 넘어가고 **`bfloat16`만 막힌다.** numpy에 그 자료형이 없기
    때문이다.

    「혼합 정밀도」에서 본 `bfloat16`은 딥러닝을 위해 만들어진 형식이라 일반
    수치 계산 라이브러리에는 들어가 있지 않다. 그래서 학습 중간값을 그림으로
    그리거나 저장하려면 `.float()`으로 올려서 넘겨야 한다.

    `.float()`을 거치면 메모리가 두 배가 되고 공유도 깨진다 — 자료형을 바꾸는
    일은 언제나 베끼는 일이다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
비연속 텐서도 `.numpy()`가 되는가? 전치·건너뛰기·자르기를 넘겨 보고, numpy 쪽
`strides`를 보라.

</div>

??? success "풀이"
    ```python
    t = torch.arange(12).reshape(3, 4)
    for name, v in (("t", t), ("t.t()", t.t()),
                    ("t[::2]", t[::2]), ("t[:, 1:3]", t[:, 1:3])):
        a = v.numpy()
        print(f"  {name:10s} -> numpy 모양 {a.shape}  strides {a.strides}  "
              f"공유 {np.shares_memory(a, t.numpy())}")
    ```

    ```
      t          -> numpy 모양 (3, 4)  strides (32, 8)  공유 True
      t.t()      -> numpy 모양 (4, 3)  strides (8, 32)  공유 True
      t[::2]     -> numpy 모양 (2, 4)  strides (64, 8)  공유 True
      t[:, 1:3]  -> numpy 모양 (3, 2)  strides (32, 8)  공유 True
    ```

    **넷 다 되고 넷 다 공유한다.** 비연속이어도 막지 않는다.

    까닭은 numpy도 **보폭 모델을 쓰기 때문**이다. 「메모리 배치와 스트라이드」에서
    본 그 모델이 두 라이브러리에 공통이라, 보폭만 그대로 넘겨주면 된다.
    `t.t()`의 numpy strides가 `(8, 32)`로 뒤집혀 있는 것이 그 증거다.

    바이트 단위라는 점만 다르다. 파이토치의 `t.stride()`는 **원소 수**로
    `(4, 1)`이고, numpy의 `strides`는 **바이트 수**로 `(32, 8)`이다 —
    int64라 원소당 8바이트이므로 여덟 배다.

    공유되는 범위도 그대로다. `t[::2]`를 고치면 원본의 그 행들이 바뀐다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
공유가 공짜라면 **얼마나** 공짜인가? 천만 개짜리 텐서로 `.numpy()`와 베끼는
방법들을 재어라.

</div>

??? success "풀이"
    ```python
    import time

    def spread(fn, rep=7):
        ts = []
        for _ in range(rep):
            t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
        return min(ts) * 1e3, max(ts) * 1e3

    t = torch.randn(10 ** 7)
    for name, fn in (("t.numpy()", lambda: t.numpy()),
                     ("t.numpy().copy()", lambda: t.numpy().copy()),
                     ("t.clone().numpy()", lambda: t.clone().numpy()),
                     ("np.array(t)", lambda: np.array(t))):
        lo, hi = spread(fn)
        print(f"  {name:20s} {lo:8.3f} ~ {hi:8.3f} ms")
    ```

    ```
      t.numpy()               0.001 ~    0.016 ms
      t.numpy().copy()        1.983 ~    2.709 ms
      t.clone().numpy()       1.368 ~    2.479 ms
      np.array(t)             1.816 ~    4.351 ms
    ```

    **천 배 넘게 차이 난다.** `.numpy()`는 1마이크로초, 베끼는 것들은 2밀리초쯤.

    `.numpy()`가 하는 일은 **보폭과 포인터를 적은 머리 하나를 만드는 것**뿐이라
    크기와 무관하다. 천만 개든 한 개든 같다.

    베끼는 셋은 4천만 바이트를 실제로 옮기므로 크기에 비례한다. 셋 사이의
    차이는 의미 있게 말하기 어렵다 — 범위가 겹친다.

    그러니 고르는 기준은 속도가 아니라 **뒤에 고칠 것인가**다.

    | | 쓸 자리 |
    |---|---|
    | `.numpy()` | 읽기만 한다 (그림 그리기, 지표 계산) |
    | `.numpy().copy()` | 그 배열을 고칠 작정이다 |

    그림을 그리거나 평균을 내는 데 `.copy()`를 붙이는 것은 공짜인 일에 2밀리초를
    치르는 것이다. 매 배치마다 하면 쌓인다.
