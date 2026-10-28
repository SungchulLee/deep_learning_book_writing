# 메모리와 저장소 — 보폭 0으로 늘리기

이 절의 첫 쪽 「메모리 배치와 스트라이드」에서 저장소·보폭·오프셋 모델을 세우고,
뷰와 복사가 어디서 갈리는지 보았다. 이 쪽은 그 모델이 가장 멀리까지 밀린 자리를
본다 — **보폭이 0인 축.**

보폭 0은 "이 축으로는 메모리를 옮겨 가지 말라"는 뜻이다. 그래서 길이 1,000짜리
한 줄을 1,000×1,000으로 늘려도 저장소는 그대로다. 공짜처럼 보이고, 실제로
거의 공짜다. 대신 **같은 메모리 한 칸이 1,000군데에 나타난다**는 값을 치른다.

## 1. 코드

```python
"""복사 없이 늘리기 — expand 와 repeat, 그리고 메모리 형식.

이 절 첫 쪽 「메모리 배치와 스트라이드」에서 저장소·보폭·오프셋 모델과
뷰·복사의 갈림을 다루었다. 이 쪽은 그 모델이 가장 극단으로 쓰이는
자리를 본다 — 보폭 0.

- `expand` 는 보폭을 0 으로 두어 저장소를 한 줄도 늘리지 않는다
- `repeat` 는 정말로 베낀다
- 늘린 텐서에 쓰면 한 번 쓴 값이 여러 자리에 나타난다. 그런데 파이토치가
  막는 쓰기와 막지 않는 쓰기가 갈린다
- 모양이 같고 보폭만 다른 또 한 가지 — 메모리 형식(channels_last)

실행: python 04_memory_and_storage.py
"""
import time

import torch


def header(title):
    print(f"\n=== {title} ===")


def elems(t):
    """저장소에 실제로 들어 있는 원소 수. 모양이 아니라 쓰인 메모리다."""
    return t.untyped_storage().nbytes() // t.element_size()


# ========================================================================
# 1. 같은 값, 다른 저장소
# ========================================================================

def same_values():
    header("1. 같은 값인데 쓰인 메모리가 다르다")
    b = torch.tensor([1.0, 2.0, 3.0])
    e = b.expand(4, 3)
    r = b.repeat(4, 1)
    for name, t in (("b (원본)", b), ("b.expand(4,3)", e), ("b.repeat(4,1)", r)):
        pad = " " * max(0, 15 - sum(2 if "가" <= c <= "힣" else 1 for c in name))
        print(f"  {name}{pad}shape={tuple(t.shape)} stride={t.stride()} "
              f"저장소 원소={elems(t)}")
    print("  두 결과의 값이 같은가:", torch.equal(e, r))
    print("  expand 의 두 행이 같은 자리:", e[0].data_ptr() == e[1].data_ptr())
    print("  repeat 의 두 행이 같은 자리:", r[0].data_ptr() == r[1].data_ptr())
    print("  보폭 0 은 '이 축으로는 움직이지 말라' 는 뜻이다")


# ========================================================================
# 2. 값이 얼마나 되는가
# ========================================================================

def cost():
    header("2. 1000 줄로 늘리는 값")
    big = torch.randn(1000)
    # 시간과 메모리를 다른 줄에 찍는다. 한 줄에 같이 두면 시간이 흔들릴
    # 때마다 그 줄 전체가 못 믿을 줄이 되어, 메모리 수까지 확인할 수 없다.
    for name, fn in (("expand", lambda: big.expand(1000, 1000)),
                     ("repeat", lambda: big.repeat(1000, 1))):
        got = []
        for _ in range(5):          # 한 번 잰 시간은 측정이 아니다
            t0 = time.perf_counter()
            out = fn()
            got.append(time.perf_counter() - t0)
        print(f"  {name} 저장소 {elems(out)}개 = {elems(out) * 4 / 1e6:.3f} MB")
        print(f"  {name} 걸린 시간 {min(got) * 1e6:.1f} ~ "
              f"{max(got) * 1e6:.1f} us")
    print("  expand 는 모양만 적어 두므로 크기와 거의 무관하다")


# ========================================================================
# 3. 늘린 텐서에 쓰면 — 막는 쓰기와 막지 않는 쓰기
# ========================================================================

def writing():
    header("3. 늘린 텐서에 쓰기")
    b = torch.tensor([1.0, 2.0, 3.0])
    e = b.expand(4, 3)
    print("  쓰기 전 b =", b.tolist())
    e[0, 0] = 99.0                  # 통과한다
    print("  e[0,0] = 99 한 뒤")
    print("    b =", b.tolist(), "<- 원본까지 바뀌었다")
    for row in e.tolist():
        print("    e 행 =", row)

    e2 = torch.tensor([1.0, 2.0, 3.0]).expand(4, 3)
    try:
        e2.add_(1.0)
    except RuntimeError as ex:
        print("  e2.add_(1.0) :", str(ex).splitlines()[0][:78])
    print("  색인 대입은 지나가고 제자리 연산은 막힌다")
    safe = torch.tensor([1.0, 2.0, 3.0]).expand(4, 3).clone()
    safe[0, 0] = 99.0
    print("  clone() 뒤에는 한 줄만 바뀐다:", safe[:, 0].tolist())


# ========================================================================
# 4. 늘릴 수 있는 축은 길이 1 인 축뿐이다
# ========================================================================

def which_axes():
    header("4. 늘릴 수 있는 축")
    try:
        torch.ones(2, 3).expand(4, 3)
    except RuntimeError as ex:
        print("  (2,3) -> (4,3) :", str(ex).splitlines()[0][:84])
    print("  (1,3) -> (4,3) :", tuple(torch.ones(1, 3).expand(4, 3).shape))
    print("  (3,)  -> (4,3) :", tuple(torch.ones(3).expand(4, 3).shape),
          "— 모자라는 축은 앞에 붙여 준다")
    print("  repeat 는 아무 축이나 된다:",
          tuple(torch.ones(2, 3).repeat(2, 1).shape))


# ========================================================================
# 5. 모양이 같고 보폭만 다른 또 한 가지 — 메모리 형식
# ========================================================================

def memory_format():
    header("5. 메모리 형식 (channels_last)")
    img = torch.randn(8, 3, 32, 32)
    cl = img.contiguous(memory_format=torch.channels_last)
    print("  모양          :", tuple(img.shape), "둘이 같다:", tuple(cl.shape))
    print("  값이 같은가    :", torch.equal(img, cl))
    print("  NCHW 보폭      :", img.stride())
    print("  channels_last :", cl.stride())
    print("  img.is_contiguous()                           :", img.is_contiguous())
    print("  cl.is_contiguous()                            :", cl.is_contiguous())
    print("  cl.is_contiguous(memory_format=channels_last) :",
          cl.is_contiguous(memory_format=torch.channels_last))
    print("  '연속' 은 한 가지가 아니라 어느 순서로 연속인지의 문제다")


def main():
    same_values()
    cost()
    writing()
    which_axes()
    memory_format()


if __name__ == "__main__":
    main()
```

**출력:**

```
=== 1. 같은 값인데 쓰인 메모리가 다르다 ===
  b (원본)       shape=(3,) stride=(1,) 저장소 원소=3
  b.expand(4,3)  shape=(4, 3) stride=(0, 1) 저장소 원소=3
  b.repeat(4,1)  shape=(4, 3) stride=(3, 1) 저장소 원소=12
  두 결과의 값이 같은가: True
  expand 의 두 행이 같은 자리: True
  repeat 의 두 행이 같은 자리: False
  보폭 0 은 '이 축으로는 움직이지 말라' 는 뜻이다

=== 2. 1000 줄로 늘리는 값 ===
  expand 저장소 1000개 = 0.004 MB
  expand 걸린 시간 0.7 ~ 1.9 us
  repeat 저장소 1000000개 = 4.000 MB
  repeat 걸린 시간 157.6 ~ 330.4 us
  expand 는 모양만 적어 두므로 크기와 거의 무관하다

=== 3. 늘린 텐서에 쓰기 ===
  쓰기 전 b = [1.0, 2.0, 3.0]
  e[0,0] = 99 한 뒤
    b = [99.0, 2.0, 3.0] <- 원본까지 바뀌었다
    e 행 = [99.0, 2.0, 3.0]
    e 행 = [99.0, 2.0, 3.0]
    e 행 = [99.0, 2.0, 3.0]
    e 행 = [99.0, 2.0, 3.0]
  e2.add_(1.0) : unsupported operation: more than one element of the written-to tensor refers t
  색인 대입은 지나가고 제자리 연산은 막힌다
  clone() 뒤에는 한 줄만 바뀐다: [99.0, 1.0, 1.0, 1.0]

=== 4. 늘릴 수 있는 축 ===
  (2,3) -> (4,3) : The expanded size of the tensor (4) must match the existing size (2) at non-singleto
  (1,3) -> (4,3) : (4, 3)
  (3,)  -> (4,3) : (4, 3) — 모자라는 축은 앞에 붙여 준다
  repeat 는 아무 축이나 된다: (4, 3)

=== 5. 메모리 형식 (channels_last) ===
  모양          : (8, 3, 32, 32) 둘이 같다: (8, 3, 32, 32)
  값이 같은가    : True
  NCHW 보폭      : (3072, 1024, 32, 1)
  channels_last : (3072, 1, 96, 3)
  img.is_contiguous()                           : True
  cl.is_contiguous()                            : False
  cl.is_contiguous(memory_format=channels_last) : True
  '연속' 은 한 가지가 아니라 어느 순서로 연속인지의 문제다
```

시간 줄은 돌릴 때마다 흔들린다. 그래서 다섯 번 재어 가장 빠른 값과 가장 느린
값을 함께 적었고, 메모리는 흔들리지 않으므로 다른 줄에 떼어 두었다.

## 2. 같은 값, 다른 메모리

`expand(4, 3)`과 `repeat(4, 1)`은 **똑같은 수를 담은 똑같은 모양**의 텐서를
돌려준다. `torch.equal`이 `True`다. 다른 것은 그 뒤다.

| | 모양 | 보폭 | 저장소 원소 |
|---|---|---|---|
| `b` | `(3,)` | `(1,)` | 3 |
| `b.expand(4,3)` | `(4,3)` | **`(0, 1)`** | **3** |
| `b.repeat(4,1)` | `(4,3)` | `(3, 1)` | 12 |

`expand`의 첫 축 보폭이 0이다. 행을 하나 내려가라고 하면 메모리에서 **0칸**
움직인다. 그러니 네 행이 모두 같은 세 칸을 읽는다.

```
expand 의 두 행이 같은 자리: True
repeat 의 두 행이 같은 자리: False
```

`repeat`는 12칸을 새로 잡아 세 수를 네 번 적어 넣었다. 그래서 행마다 자리가
다르다.

## 3. 값이 얼마나 되는가

길이 1,000짜리를 1,000×1,000으로 늘려 보면 둘의 차이가 보인다.

| | 저장소 | 걸린 시간 |
|---|---|---|
| `expand` | 1,000개 = **0.004 MB** | 0.7 ~ 1.9 µs |
| `repeat` | 1,000,000개 = **4.000 MB** | 157.6 ~ 330.4 µs |

메모리는 **1,000배**다. 원본만큼만 쓰느냐 결과만큼 쓰느냐의 차이다.

시간 쪽은 퍼짐이 커서 한 수로 적을 수 없지만(`repeat`가 두 배 넘게 흔들린다)
두 범위가 겹치지 않으므로 말할 수는 있다. 가장 느린 `expand`가 가장 빠른
`repeat`보다 80배 넘게 빠르다. `expand`는 보폭 표만 고쳐 쓰므로 크기와 거의
무관하고, `repeat`는 100만 개를 실제로 적어야 한다.

그러니 **읽기만 할 작정이면 `expand`를 쓴다.** 브로드캐스팅이 메모리를 먹지
않는 것도 같은 까닭이다 — 연습문제 2에서 확인한다.

## 4. 늘린 텐서에 쓰면 — 막는 쓰기와 막지 않는 쓰기

보폭 0에는 값이 따른다. 한 칸을 네 자리가 같이 보고 있으니, 한 번 쓰면 네 군데가
같이 바뀐다.

```
쓰기 전 b = [1.0, 2.0, 3.0]
e[0,0] = 99 한 뒤
  b = [99.0, 2.0, 3.0] <- 원본까지 바뀌었다
  e 행 = [99.0, 2.0, 3.0]
  e 행 = [99.0, 2.0, 3.0]
  e 행 = [99.0, 2.0, 3.0]
  e 행 = [99.0, 2.0, 3.0]
```

첫 행 첫 칸에만 99를 넣었는데 **네 행이 모두 바뀌고 원본 `b`까지 바뀌었다.**
그리고 아무 말도 없었다.

그런데 제자리 연산으로 같은 일을 하려 하면 막는다.

```
e2.add_(1.0) : unsupported operation: more than one element of the
written-to tensor refers to a single memory location
```

**색인 대입은 지나가고 제자리 연산은 막힌다.** 메시지는 정확히 맞는 말을 하고
있는데 — 쓰려는 텐서의 여러 원소가 한 자리를 가리킨다 — 그 검사가 `e[0,0] = 99`
에는 걸리지 않는다. 같은 위험을 한쪽에서는 거절하고 한쪽에서는 통과시킨다.

그러니 믿을 것은 검사가 아니라 규칙이다. **늘린 텐서는 읽기 전용으로 쓴다.**
고칠 작정이면 `clone()`을 먼저 한다. 그러면 12칸을 새로 잡으므로 한 줄만 바뀐다.

```
clone() 뒤에는 한 줄만 바뀐다: [99.0, 1.0, 1.0, 1.0]
```

## 5. 늘릴 수 있는 축은 길이 1인 축뿐이다

```
(2,3) -> (4,3) : The expanded size of the tensor (4) must match
                 the existing size (2) at non-singleton dimension
```

보폭 0으로 늘린다는 것은 "원래 한 칸이던 것을 여러 번 읽는다"는 뜻이므로, 원래
길이가 1이어야 한다. 길이 2를 4로 늘리는 방법은 없다 — 수를 지어내야 하니까.
`(3,)`에서 `(4,3)`으로는 되는데, 모자라는 축을 앞에 1로 붙여 `(1,3)`으로 본
다음 늘리는 것이다. 브로드캐스팅의 규칙과 같다.

`repeat`는 실제로 베끼므로 이런 제약이 없다. `(2,3)`을 `(4,3)`으로 만들 수 있다.
둘은 바꿔 쓸 수 있는 함수가 아니라 **쓸 수 있는 자리가 다른 함수**다.

## 6. 모양이 같고 보폭만 다른 또 한 가지

보폭이 모양과 따로 움직인다는 생각은 `expand`보다 훨씬 넓게 쓰인다. 영상 배치의
**메모리 형식**이 그 보기다.

```
모양          : (8, 3, 32, 32)   둘이 같다
값이 같은가    : True
NCHW 보폭      : (3072, 1024, 32, 1)
channels_last : (3072, 1,    96, 3)
```

둘은 모양도 값도 같다. 다른 것은 **메모리에 눕는 순서**뿐이다. NCHW는 한 채널의
32×32를 다 적고 다음 채널로 넘어가고(채널 보폭 1024), channels_last는 한 화소의
세 채널을 붙여 적는다(채널 보폭 1).

그래서 `is_contiguous()`를 물으면 재미있는 답이 나온다.

| 묻는 것 | 답 |
|---|---|
| `img.is_contiguous()` | True |
| `cl.is_contiguous()` | **False** |
| `cl.is_contiguous(memory_format=channels_last)` | **True** |

`cl`은 메모리에 빈틈 없이 누워 있다. 그런데 기본으로 묻는 `is_contiguous()`는
"NCHW 순서로 연속인가"를 묻기 때문에 `False`다. **연속이냐 아니냐는 한 가지
물음이 아니라, 어느 순서로 연속이냐는 물음이다.**

이것이 실전에서 쓰이는 까닭은 합성곱 커널이 채널을 가장 안쪽에서 읽을 때 더
빠른 경우가 있어서다. 같은 수를 담은 같은 모양의 텐서를 보폭만 바꿔 넘기는 것이
`model.to(memory_format=torch.channels_last)`가 하는 일이다.

## 7. 정리

| | 저장소 | 쓸 수 있나 | 늘릴 수 있는 축 |
|---|---|---|---|
| `expand` | 안 늘어난다 (보폭 0) | **안 된다** (`clone()` 먼저) | 길이 1인 축만 |
| `repeat` | 결과만큼 늘어난다 | 된다 | 아무 축이나 |

보폭·저장소 모델 자체는 이 절 첫 쪽 「메모리 배치와 스트라이드」에 있다. 이 쪽에서
더한 것은 그 모델이 보폭 0까지 밀렸을 때 생기는 두 가지다 — **공짜에 가까운
확대**와 **조용히 번지는 쓰기**.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
길이 3인 벡터를 `(1000, 3)`으로 늘려라. `expand`와 `repeat` 두 가지로 만들고,
각각 저장소에 원소가 몇 개 들어 있는지 세어라. 모양과 값은 같은가?

</div>

??? success "풀이"
    ```python
    def elems(t):
        return t.untyped_storage().nbytes() // t.element_size()

    b = torch.tensor([1.0, 2.0, 3.0])
    e, r = b.expand(1000, 3), b.repeat(1000, 1)
    print("  모양이 같은가:", e.shape == r.shape, tuple(e.shape))
    print("  값이 같은가  :", torch.equal(e, r))
    print(f"  expand 저장소 {elems(e):>7}개")
    print(f"  repeat 저장소 {elems(r):>7}개")
    print(f"  배수 {elems(r) / elems(e):.0f}")
    ```

    ```
      모양이 같은가: True (1000, 3)
      값이 같은가  : True
      expand 저장소       3개
      repeat 저장소    3000개
      배수 1000
    ```

    모양도 값도 같은데 쓰인 메모리가 1,000배 다르다. `expand`는 원본 세 칸을
    그대로 쓰고, `repeat`는 3,000칸을 새로 잡아 세 수를 1,000번 적었다.

    늘리는 배수가 곧 메모리 배수다. 그래서 큰 배치에 작은 벡터를 더하려고
    `repeat`로 미리 펴 두는 코드는 배치 크기만큼 메모리를 더 쓴다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
`a + b`에서 `a`가 `(4,3)`, `b`가 `(3,)`일 때 파이토치는 `b`를 늘려 더한다. 이때
`repeat`처럼 메모리를 쓰는가, `expand`처럼 안 쓰는가? `(2000,2000) + (2000,)`로
확인하고, `repeat`로 미리 펴서 더한 쪽과 시간을 견주어라.

</div>

??? success "풀이"
    ```python
    import time

    a = torch.ones(4, 3)
    b = torch.tensor([1.0, 2.0, 3.0])
    print("  a + b             :", tuple((a + b).shape))
    print("  a + b.expand(4,3) :", tuple((a + b.expand(4, 3)).shape))
    print("  두 결과가 같은가   :", torch.equal(a + b, a + b.expand(4, 3)))

    big_a, big_b = torch.ones(2000, 2000), torch.ones(2000)
    for label, fn in (("브로드캐스팅", lambda: big_a + big_b),
                      ("repeat 로 펴고", lambda: big_a + big_b.repeat(2000, 1))):
        got = []
        for _ in range(5):
            t0 = time.perf_counter(); fn(); got.append(time.perf_counter() - t0)
        print(f"  {label:14s} {min(got) * 1e3:.1f} ~ {max(got) * 1e3:.1f} ms")
    ```

    ```
      a + b             : (4, 3)
      a + b.expand(4,3) : (4, 3)
      두 결과가 같은가   : True
      브로드캐스팅       0.3 ~ 0.7 ms
      repeat 로 펴고     1.0 ~ 1.2 ms
    ```

    `a + b`와 `a + b.expand(4,3)`이 같은 값을 준다. **브로드캐스팅은 `expand`로
    적어 둔 것과 같은 일**이고, 그래서 메모리를 새로 잡지 않는다.

    `repeat`로 미리 펴면 400만 칸(16 MB)을 적은 다음 더하므로 두세 배 느리다.
    범위가 겹치지 않으니 느리다고 말할 수 있다.

    그래서 브로드캐스팅으로 되는 일을 `repeat`로 손수 펴는 코드는 느리고 메모리도
    더 쓴다. 모양을 맞추려고 `repeat`를 쓰고 있다면, 대개 `unsqueeze`로 길이 1인
    축을 넣어 주는 것으로 충분하다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
`requires_grad=True`인 길이 3 벡터를 `(4,3)`으로 늘려 `sum()`을 역전파하면
경사는 얼마가 되는가? 늘리지 않고 그냥 `sum()`한 경우와 견주고, 보폭 0으로
설명하라. `repeat`로 늘린 쪽은 어떻게 되는가?

</div>

??? success "풀이"
    ```python
    for label, fn in (("늘리지 않고   ", lambda t: t),
                      ("expand(4,3)  ", lambda t: t.expand(4, 3)),
                      ("repeat(4,1)  ", lambda t: t.repeat(4, 1))):
        b = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        fn(b).sum().backward()
        print(f"  {label} b.grad = {b.grad.tolist()}")
    ```

    ```
      늘리지 않고    b.grad = [1.0, 1.0, 1.0]
      expand(4,3)   b.grad = [4.0, 4.0, 4.0]
      repeat(4,1)   b.grad = [4.0, 4.0, 4.0]
    ```

    늘리지 않으면 1이고, 넷으로 늘리면 **4**다. `expand`와 `repeat`가 같은 값을
    준다.

    까닭은 순전파를 거꾸로 읽으면 나온다. `b[0]`은 결과의 네 자리에
    **쓰였다**(보폭 0이니 네 행이 같은 칸을 읽는다). 한 입력이 여러 출력에
    쓰였으면 경사는 그 출력들의 경사를 **모두 더해서** 돌아온다. 네 자리에서 각
    1씩 왔으니 4다.

    `repeat`도 같은 수가 나오는 것이 중요하다. 메모리를 복사했든 보폭으로
    흉내 냈든 **수학적으로 같은 함수**이므로 미분도 같아야 한다. 보폭 0은 메모리를
    아끼는 구현일 뿐, 계산하는 함수를 바꾸지 않는다.

    여기서 실전에 걸리는 것이 하나 있다. 편향 하나를 배치 전체에 더하면
    (`y = x @ W.T + bias`, 브로드캐스팅) `bias`의 경사는 **배치 크기만큼 더해진
    값**으로 온다. 그래서 로스를 배치 평균으로 잡는 관행이 편향의 경사까지
    같이 맞춰 주는 것이고, 합으로 잡으면 배치 크기를 바꿀 때 편향이 받는 걸음도
    같이 커진다. 「최적화 기초」에서 러닝레이트가 배치 크기와 맞물리는 까닭의
    한 조각이 이것이다.
