# 데이터 관련 속성 — 꺼낸 값이 원본과 이어져 있는가

텐서에서 값을 꺼내는 길은 여럿이다. `item()`·`tolist()`·`clone()`·`detach()`·
`numpy()`. 이 길들은 두 가지에서 갈린다.

- **저장소를 나눠 쓰는가** — 꺼낸 것을 고치면 원본도 바뀌는가
- **그래프에 이어져 있는가** — 경사가 거슬러 올라가는가

이 둘은 따로 움직인다. `clone()`은 저장소를 끊고 그래프는 남기고, `detach()`는
그래프를 끊고 저장소는 남긴다. 둘을 하나로 묶어 "복사"라고 생각하면 어느 쪽에서든
틀린다.

그리고 `.data`는 둘 다 남긴 채 autograd의 **검사만** 끈다. 그래서 이 쪽에 나오는
것 가운데 가장 위험하다.

## 1. 코드

```python
"""데이터를 꺼내는 길 — 꺼낸 것이 원본과 아직 이어져 있는가.

텐서에서 값을 꺼내는 길은 여럿이다. `item()`·`tolist()`·`clone()`·
`detach()`·`numpy()`. 이 길들은 두 가지에서 갈린다.

- 저장소를 나눠 쓰는가 (고치면 원본도 바뀌는가)
- 그래프에 이어져 있는가 (경사가 거슬러 올라가는가)

둘은 따로 움직인다. `clone()` 은 저장소를 끊고 그래프는 남기며,
`detach()` 는 그래프를 끊고 저장소는 남긴다. 그리고 `.data` 는 둘 다
남긴 채로 autograd 의 검사만 지나간다 — 그래서 가장 위험하다.

실행: 이 쪽의 코드를 파일로 갈무리해 돌린다
"""
import numpy as np
import torch
import torch.nn as nn


def header(title):
    print(f"\n=== {title} ===")


# ========================================================================
# 1. 홑값과 리스트로 꺼내기
# ========================================================================

def extracting():
    header("1. item() 과 tolist()")
    s = torch.tensor(3.14159)
    print("  s          :", s)
    print("  s.item()   :", repr(s.item()))
    print("  s.item() == 3.14159 ?", s.item() == 3.14159)
    print("  float64 로 담으면:",
          repr(torch.tensor(3.14159, dtype=torch.float64).item()))

    v = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    print("  v.tolist() :", v.tolist())
    print("  꺼낸 리스트는 파이썬 객체라 저장소를 나눠 쓰지 않는다")


# ========================================================================
# 2. 복사 셋 — 무엇을 끊고 무엇을 남기는가
# ========================================================================

def copies():
    header("2. clone / detach / detach().clone()")
    torch.manual_seed(0)
    a = torch.randn(4, requires_grad=True)
    cases = (
        ("a (원본)", a),
        ("a.clone()", a.clone()),
        ("a.detach()", a.detach()),
        ("a.detach().clone()", a.detach().clone()),
    )
    print(f"  {'':20s}{'공유':>6s}{'requires_grad':>16s}{'grad_fn':>18s}")
    for name, t in cases:
        fn = type(t.grad_fn).__name__ if t.grad_fn is not None else "None"
        label = name + " " * max(0, 20 - sum(2 if "가" <= ch <= "힣"
                                             else 1 for ch in name))
        print(f"  {label}{str(t.data_ptr() == a.data_ptr()):>6s}"
              f"{str(t.requires_grad):>16s}{fn:>18s}")

    # clone() 은 그래프를 남기므로 경사가 원본까지 거슬러 간다
    b = torch.tensor([1.0, 2.0], requires_grad=True)
    (b.clone() ** 2).sum().backward()
    print("  clone() 을 지나 온 경사  b.grad =", b.grad.tolist(), "(2b)")

    # detach() 는 끊으므로 뒤로 걸을 길이 없다
    c = torch.tensor([1.0, 2.0], requires_grad=True)
    try:
        (c.detach() ** 2).sum().backward()
    except RuntimeError as e:
        print("  detach() 를 지나 뒤로 걸으면:", str(e).splitlines()[0])


# ========================================================================
# 3. numpy 는 메모리를 나눠 쓴다
# ========================================================================

def numpy_shares():
    header("3. numpy 와 메모리를 나눠 쓴다")
    t = torch.tensor([1.0, 2.0, 3.0])
    arr = t.numpy()
    arr[0] = 99.0
    print("  arr[0]=99 로 고치니 텐서 =", t.tolist())
    t[1] = 77.0
    print("  텐서를 고치니 배열       =", arr.tolist())
    print("  양쪽이 같은 메모리를 가리킨다. 복사가 아니다")

    b = np.array([1.0, 2.0])
    tb = torch.from_numpy(b)
    b[0] = 5.0
    print("  from_numpy 뒤 배열을 고치니 텐서 =", tb.tolist())
    print("  그리고 dtype 이", tb.dtype, "다 — numpy 의 기본값이 float64 라서다")

    lin = nn.Linear(2, 1)
    try:
        lin(torch.from_numpy(np.array([[1.0, 2.0]])))
    except RuntimeError as e:
        print("  그대로 모델에 넣으면:", str(e).splitlines()[0])
    ok = lin(torch.from_numpy(np.array([[1.0, 2.0]], dtype=np.float32)))
    print("  np.float32 로 만들면 통과:", tuple(ok.shape))


# ========================================================================
# 4. 꺼낼 수 없는 자리
# ========================================================================

def cannot_extract():
    header("4. 바로 꺼낼 수 없는 두 자리")
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
            print(f"  device={dev:4s}       :", str(e).splitlines()[0])
        print("  .cpu().numpy() 로는 된다:", m.cpu().numpy().shape)
    else:
        print("  가속기가 없어 장치 쪽은 건너뛴다 (cpu 뿐)")
    print("  두 메시지 모두 고칠 방법을 함께 적어 준다")


# ========================================================================
# 5. .data — 검사만 지나간다
# ========================================================================

def data_attribute():
    header("5. .data 는 버전 카운터를 지나간다")
    # 파이토치는 텐서마다 _version 을 세어 둔다. 역전파가 저장해 둔 값이
    # 그 뒤에 고쳐졌으면 숫자가 어긋나므로, 엔진이 거절한다.
    for label, use_data in (("y.add_(1)", False), ("y.data.add_(1)", True)):
        x = torch.tensor([1.0, 2.0], requires_grad=True)
        y = x.exp()               # exp 의 역전파는 결과 y 를 저장해 둔다
        (y.data if use_data else y).add_(1.0)
        print(f"  {label:16s} _version={y._version}", end="  ")
        try:
            y.sum().backward()
            print("경사 =", [round(v, 4) for v in x.grad.tolist()])
        except RuntimeError as e:
            print("RuntimeError:", str(e).splitlines()[0][:62])
    print("  참값 d(sum exp x)/dx = exp(x) =",
          [round(v, 4) for v in torch.tensor([1.0, 2.0]).exp().tolist()])
    print("  .data 쪽은 1 씩 크다 — 고쳐진 y 를 그대로 읽었다")
    print("  막아 주던 검사를 끈 셈이라, 터지는 대신 틀린 수가 나온다")


def main():
    extracting()
    copies()
    numpy_shares()
    cannot_extract()
    data_attribute()


if __name__ == "__main__":
    main()
```

**출력:**

```
=== 1. item() 과 tolist() ===
  s          : tensor(3.1416)
  s.item()   : 3.141590118408203
  s.item() == 3.14159 ? False
  float64 로 담으면: 3.14159
  v.tolist() : [[1.0, 2.0], [3.0, 4.0]]
  꺼낸 리스트는 파이썬 객체라 저장소를 나눠 쓰지 않는다

=== 2. clone / detach / detach().clone() ===
                          공유   requires_grad           grad_fn
  a (원본)              True            True              None
  a.clone()            False            True    CloneBackward0
  a.detach()            True           False              None
  a.detach().clone()   False           False              None
  clone() 을 지나 온 경사  b.grad = [2.0, 4.0] (2b)
  detach() 를 지나 뒤로 걸으면: element 0 of tensors does not require grad and does not have a grad_fn

=== 3. numpy 와 메모리를 나눠 쓴다 ===
  arr[0]=99 로 고치니 텐서 = [99.0, 2.0, 3.0]
  텐서를 고치니 배열       = [99.0, 77.0, 3.0]
  양쪽이 같은 메모리를 가리킨다. 복사가 아니다
  from_numpy 뒤 배열을 고치니 텐서 = [5.0, 2.0]
  그리고 dtype 이 torch.float64 다 — numpy 의 기본값이 float64 라서다
  그대로 모델에 넣으면: mat1 and mat2 must have the same dtype, but got Double and Float
  np.float32 로 만들면 통과: (1, 1)

=== 4. 바로 꺼낼 수 없는 두 자리 ===
  requires_grad=True : Can't call numpy() on Tensor that requires grad. Use tensor.detach().numpy() instead.
  device=mps        : can't convert mps:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.
  .cpu().numpy() 로는 된다: (3,)
  두 메시지 모두 고칠 방법을 함께 적어 준다

=== 5. .data 는 버전 카운터를 지나간다 ===
  y.add_(1)        _version=1  RuntimeError: one of the variables needed for gradient computation has been 
  y.data.add_(1)   _version=0  경사 = [3.7183, 8.3891]
  참값 d(sum exp x)/dx = exp(x) = [2.7183, 7.3891]
  .data 쪽은 1 씩 크다 — 고쳐진 y 를 그대로 읽었다
  막아 주던 검사를 끈 셈이라, 터지는 대신 틀린 수가 나온다
```

`device=mps` 줄은 이 쪽을 돌린 기계가 애플 실리콘이기 때문이다. CUDA 기계에서는
`device=cuda`로 찍히고 메시지도 그에 맞게 바뀌며, 가속기가 없으면 그 줄을
건너뛴다.

## 2. `item()`은 저장된 값을 그대로 보여 준다

`torch.tensor(3.14159)`를 찍으면 `tensor(3.1416)`이 나오고, `.item()`을 부르면
`3.141590118408203`이 나온다. 둘 다 같은 수다. 앞은 **보기 좋게 줄여 찍은 것**이고
뒤는 **float32에 실제로 담긴 값**이다.

```
s.item() == 3.14159 ?  False
float64 로 담으면: 3.14159
```

float32는 유효숫자를 24비트만 쥐므로 3.14159를 정확히 담을 수 없다. 그래서 꺼내
보면 원래 적어 준 파이썬 수(float64)와 같지 않다. float64로 담으면 같아진다.

이것이 모델의 출력을 상수와 `==`로 견주면 안 되는 까닭이다. 텐서에서 꺼낸 수는
`0.1`이 아니라 `0.10000000149011612`이고, 눈으로는 `0.1000`으로 보인다.
`torch.allclose`가 있는 이유가 여기 있다.

`tolist()`는 파이썬 리스트를 만들어 돌려주므로 저장소를 나눠 쓰지 않는다. 꺼낸
다음에는 원본과 아무 관계가 없다. 아래에 나오는 `numpy()`와 정반대다.

## 3. 복사 셋이 서로 다른 것을 끊는다

| | 저장소 공유 | `requires_grad` | `grad_fn` |
|---|---|---|---|
| `a` (원본) | — | True | None |
| `a.clone()` | **False** | True | `CloneBackward0` |
| `a.detach()` | **True** | False | None |
| `a.detach().clone()` | False | False | None |

`clone()`에 `grad_fn`이 붙은 것이 이 표의 핵심이다. **`clone()`은 미분을 끊지
않는다.** 복사본을 거쳐 온 경사가 원본까지 거슬러 올라간다.

```
clone() 을 지나 온 경사  b.grad = [2.0, 4.0] (2b)
```

$b$를 복사해 제곱하고 더해 역전파했는데 $b$가 경사를 받았다. 복사본에서 한 일이
원본의 학습에 그대로 섞여 들어간다는 뜻이다. 기록용 스냅숏을 `clone()`으로
떠 두면 이 길이 열린 채 남는다.

`detach()`는 반대다. 미분은 끊었으나 **저장소는 같다.** 그래서 거꾸로 걸으려 하면
길이 없다고 한다.

```
detach() 를 지나 뒤로 걸으면: element 0 of tensors does not require grad
and does not have a grad_fn
```

그러니 "학습과 무관한 독립 사본"은 둘 다 해야 얻어진다. `detach().clone()`이
표의 마지막 줄이고, 기록·비교·저장에 쓸 것은 이것이다.

## 4. `numpy()`는 복사가 아니다

`t.numpy()`는 배열을 새로 만드는 것처럼 보이지만 **같은 메모리를 가리킨다.** 양쪽
어디를 고쳐도 반대쪽이 따라 바뀐다.

```
arr[0]=99 로 고치니 텐서 = [99.0, 2.0, 3.0]
텐서를 고치니 배열       = [99.0, 77.0, 3.0]
```

`torch.from_numpy`도 같은 방향으로 공유한다. 그래서 numpy로 전처리를 돌려 놓고
배열을 재활용하면, 이미 넘긴 텐서가 조용히 바뀐다.

여기에 `dtype` 함정이 하나 더 붙는다. **numpy의 기본 실수형은 float64**이고
파이토치는 float32다. 그대로 모델에 넣으면 이렇게 끝난다.

```
mat1 and mat2 must have the same dtype, but got Double and Float
```

터져 주니 다행인 쪽이다. `np.float32`로 만들거나 `.float()`를 붙이면 통과한다.

## 5. 꺼낼 수 없는 두 자리

`numpy()`가 거절하는 경우가 둘 있고, 메시지가 둘 다 고칠 방법을 함께 적어 준다.

| 거절하는 까닭 | 메시지가 가리키는 길 |
|---|---|
| `requires_grad=True` | `tensor.detach().numpy()` |
| CPU가 아닌 장치 | `Tensor.cpu()`로 먼저 옮겨라 |

첫째는 넘긴 배열을 고치면 그래프가 저장해 둔 값이 바뀌기 때문에 막는다. 둘째는
numpy가 가속기 메모리를 읽을 수 없기 때문이다. 그래서 흔히 보는
`x.detach().cpu().numpy()` 세 토막은 이 두 거절을 차례로 치우는 주문이다.

## 6. `.data`는 검사를 끈다

파이토치는 텐서마다 `_version`을 센다. 역전파가 저장해 둔 값이 그 뒤에 고쳐졌으면
숫자가 어긋나므로, 엔진이 거절한다. 그 검사가 일하는 모습과, `.data`가 그것을
지나가는 모습을 나란히 두면 이렇다.

| | `_version` | 역전파 결과 |
|---|---|---|
| `y.add_(1)` | 0 → **1** | `RuntimeError` |
| `y.data.add_(1)` | 0 → **0** | `[3.7183, 8.3891]` |

참값은 $\exp(x) = [2.7183, 7.3891]$이다. `.data` 쪽은 **정확히 1씩 크다** —
`add_(1)`로 고쳐진 `y`를 그대로 읽었기 때문이다.

이 표가 `.data`를 쓰지 말아야 하는 까닭 전부다. 터져 주는 쪽은 숫자를 안 준다.
`.data`는 숫자를 주는데 그 숫자가 틀렸고, 아무도 알려 주지 않는다.
**`torch.no_grad()`는 기록을 끄는 것이고, `.data`는 검사를 끄는 것**이다. 이름이
비슷해 보여도 하는 일이 전혀 다르다.

## 7. 정리

| 꺼내는 길 | 저장소 | 그래프 | 쓸 자리 |
|---|---|---|---|
| `item()`, `tolist()` | 끊긴다 | 끊긴다 | 값을 적거나 찍을 때 |
| `clone()` | 끊긴다 | **남는다** | 그래프를 지나야 하는 사본 |
| `detach()` | **남는다** | 끊긴다 | 읽기만 할 때 |
| `detach().clone()` | 끊긴다 | 끊긴다 | 기록·비교·저장 |
| `numpy()` | **남는다** | 끊어야 한다 | 넘겨서 읽을 때 |
| `.data` | 남는다 | 남는데 검사가 없다 | 쓰지 않는다 |

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
`torch.tensor(v).item() == v`가 참이 되는 `v`와 거짓이 되는 `v`를 각각 찾아라.
`0.1`, `0.5`, `3.14159`, `1e8`, `1e8 + 1`을 넣어 보고 갈리는 까닭을 적어라.

</div>

??? success "풀이"
    ```python
    for v in (0.1, 0.5, 3.14159, 1e8, 1e8 + 1):
        t = torch.tensor(v)
        print(f"  {v!r:12s} -> item() {t.item()!r:22s} 같은가 {t.item() == v}")
    ```

    ```
      0.1          -> item() 0.10000000149011612    같은가 False
      0.5          -> item() 0.5                    같은가 True
      3.14159      -> item() 3.141590118408203      같은가 False
      100000000.0  -> item() 100000000.0            같은가 True
      100000001.0  -> item() 100000000.0            같은가 False
    ```

    `0.5`와 `1e8`만 같다. 둘은 **float32로 정확히 적히는 수**다. `0.5`는
    $2^{-1}$이고, `1e8`은 $5^8 \times 2^8$인데 $5^8 = 390{,}625$가 24비트
    안에 들어가므로 역시 정확하다.

    `0.1`과 `3.14159`는 이진 소수로 끝나지 않아 가장 가까운 float32로 반올림된다.
    `1e8 + 1`은 끝나기는 하지만 유효숫자가 24비트를 넘어서 `1e8`로 떨어진다 —
    **1을 더했는데 값이 그대로다.**

    그래서 텐서에서 꺼낸 수를 상수와 `==`로 견주는 코드는 거의 틀린다. 눈으로는
    `0.1000`으로 보이니 더 그렇다. `torch.allclose`나 `abs(a - b) < tol`을 쓴다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
numpy 배열에서 텐서를 만드는 길이 둘이다. `torch.from_numpy(a)`와
`torch.tensor(a)`. 어느 쪽이 메모리를 공유하는지 확인하고, 100만 개짜리 배열로
둘의 시간을 재어 그 차이가 어디서 오는지 설명하라.

</div>

??? success "풀이"
    ```python
    import time

    a = np.ones(3, dtype=np.float32)
    t = torch.from_numpy(a)
    t2 = torch.tensor(a)
    base = a.__array_interface__["data"][0]
    print("  from_numpy   공유:", t.data_ptr() == base)
    print("  torch.tensor 공유:", t2.data_ptr() == base)
    a[0] = 9.0
    print("  배열을 고친 뒤  from_numpy:", t.tolist(),
          "  torch.tensor:", t2.tolist())

    # 한 번 잰 시간은 측정이 아니다. 일곱 번 재어 가장 빠른 값과 가장
    # 느린 값을 함께 적는다.
    def spread(fn, n=7):
        got = []
        for _ in range(n):
            t0 = time.perf_counter(); fn(); got.append(time.perf_counter() - t0)
        return min(got) * 1e6, max(got) * 1e6

    big = np.ones((1000, 1000), dtype=np.float32)
    for name, fn in (("from_numpy  ", lambda: torch.from_numpy(big)),
                     ("torch.tensor", lambda: torch.tensor(big))):
        lo, hi = spread(fn)
        print(f"  {name}  가장 빠른 {lo:7.1f} us   가장 느린 {hi:7.1f} us")
    ```

    ```
      from_numpy   공유: True
      torch.tensor 공유: False
      배열을 고친 뒤  from_numpy: [9.0, 1.0, 1.0]   torch.tensor: [1.0, 1.0, 1.0]
      from_numpy    가장 빠른     0.5 us   가장 느린    71.9 us
      torch.tensor  가장 빠른   226.6 us   가장 느린  1842.8 us
    ```

    `from_numpy`는 **같은 메모리에 텐서 머리만 씌운다.** 그래서 배열을 고치면
    텐서도 바뀌고, 100만 개를 만드는 데 1 마이크로초도 안 걸린다. 크기와 거의
    무관한 시간이다.

    `torch.tensor(a)`는 **복사한다.** 그래서 배열을 고쳐도 끄떡없고, 100만 개에
    수백 마이크로초가 든다.

    시간 쪽은 퍼짐이 크다 — 둘 다 백 배 가까이 흔들린다. 그래도 **두 범위가
    겹치지 않는다**는 것은 말할 수 있다. 가장 느린 `from_numpy`(71.9)가 가장
    빠른 `torch.tensor`(226.6)보다 세 배 빠르다. 한 번씩만 재어 "200배"라고
    적으면 그 수는 다음 실행에서 바뀐다.

    고르는 기준은 속도가 아니라 **그 배열을 뒤에 또 건드릴 작정인가**다. 전처리
    버퍼를 돌려 쓰는 코드라면 `from_numpy`로 넘긴 텐서가 조용히 바뀌므로
    `torch.tensor`나 `from_numpy(a).clone()`이 맞다. 한 번 만들고 버리는
    배열이라면 `from_numpy`가 공짜다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
`.data`가 위험하다고 하면서도 옛 코드에 흔히 남아 있다. 가중치를 손으로 고치는
자리에서 `.data`와 `torch.no_grad()`가 **같은 결과를 주는 경우**와 **갈리는
경우**를 각각 만들어라. 무엇이 둘을 가르는가?

</div>

??? success "풀이"
    ```python
    print("가. 가중치를 고치는 데 쓰면")
    for label in ("no_grad", ".data  "):
        w = torch.tensor([2.0], requires_grad=True)
        for _ in range(3):
            if w.grad is not None:
                w.grad.zero_()
            (w ** 2).sum().backward()
            if label.strip() == "no_grad":
                with torch.no_grad():
                    w -= 0.1 * w.grad
            else:
                w.data -= 0.1 * w.grad
        print(f"  {label} 세 걸음 뒤 w = {[round(v, 6) for v in w.tolist()]}")

    print("\n나. 역전파가 저장해 둔 값을 고치면")
    for label in ("no_grad", ".data  "):
        x = torch.tensor([1.0, 2.0], requires_grad=True)
        y = x.exp()
        if label.strip() == "no_grad":
            with torch.no_grad():
                y.add_(1.0)
        else:
            y.data.add_(1.0)
        print(f"  {label} _version={y._version}", end="  ")
        try:
            y.sum().backward()
            print("경사 =", [round(v, 4) for v in x.grad.tolist()])
        except RuntimeError as e:
            print("RuntimeError:", str(e).splitlines()[0][:52])
    print("  참값 exp(x) =",
          [round(v, 4) for v in torch.tensor([1.0, 2.0]).exp().tolist()])
    ```

    ```
    가. 가중치를 고치는 데 쓰면
      no_grad 세 걸음 뒤 w = [1.024]
      .data   세 걸음 뒤 w = [1.024]

    나. 역전파가 저장해 둔 값을 고치면
      no_grad _version=1  RuntimeError: one of the variables needed for gradient computation
      .data   _version=0  경사 = [3.7183, 8.3891]
      참값 exp(x) = [2.7183, 7.3891]
    ```

    **가에서는 소수점 여섯째 자리까지 같다.** 가중치를 고치는 자리에서는 그 값을
    기다리는 역전파가 없으니, 기록을 끄든 검사를 끄든 결과가 같다. `.data`가 옛
    코드에 살아남은 까닭이 이것이다 — 그렇게 쓰면 멀쩡히 돌아간다.

    **나에서 갈린다.** `exp`의 역전파는 결과 $y$를 저장해 두는데 그것을 고쳤다.

    - `no_grad`로 고치면 `_version`이 1로 올라가고, 엔진이 어긋남을 알아채
      `RuntimeError`를 낸다. **`no_grad`는 버전 카운터를 끄지 않는다.**
    - `.data`로 고치면 `_version`이 0으로 남아 엔진이 모른 채 지나가고,
      고쳐진 값으로 센 경사가 나온다. 참값보다 정확히 1씩 크다.

    그래서 둘을 가르는 것은 **고치는 값이 아직 쓰일 데가 있는가**다. 그리고
    `no_grad`는 *기록*을 끄고, `.data`는 *검사*를 끈다. 앞의 것은 끄는 게 맞는
    자리가 있지만, 뒤의 것은 끄는 게 맞는 자리가 없다.

    `.data`가 들어 있는 옛 코드를 고칠 때는 그 자리가 가에 해당하는지 나에
    해당하는지 볼 것 없이 `no_grad`로 바꾸면 된다. 가라면 결과가 같고, 나라면
    숨어 있던 버그가 드러난다.
