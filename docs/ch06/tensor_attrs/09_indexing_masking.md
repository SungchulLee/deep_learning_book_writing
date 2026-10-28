# 불리언 마스킹과 where — 뷰로 오는 것과 베껴 오는 것

텐서에서 일부를 꺼내는 길이 여럿인데, 크게 두 갈래로 갈린다.

| | 결과 |
|---|---|
| 자르기 `a[1:3, 2:4]`, `a[::2]` | **뷰** — 고치면 원본이 바뀐다 |
| 참거짓 가리기 `a[a > 0]` | **베낌** — 고쳐도 원본은 그대로다 |
| 정수 배열 `a[[0, 2]]` | **베낌** |

읽을 때는 둘이 똑같아 보인다. 고치려 할 때에야 갈린다. 그런데 **대입은 또
다르다** — `a[mask] = 0`은 베낀 것에 쓰는 것이 아니라 원본을 고친다. 같은 꼴을
오른쪽에 쓰느냐 왼쪽에 쓰느냐로 뜻이 바뀐다.

마지막 두 절이 이 쪽에서 가장 비싼 값이다. **`torch.where`는 고르지 않은 가지도
셈한다.** 그래서 0으로 나눈 다음 가리는 것만으로는 역전파의 NaN을 막지 못한다.

## 1. 코드

```python
"""자리 잡기와 가리기 — 뷰로 오는 것과 베껴 오는 것.

텐서에서 일부를 꺼내는 길이 여럿이고, 크게 두 갈래로 갈린다.

- **자르기**(`a[1:3, 2:4]`, `a[::2]`)는 **뷰**다. 고치면 원본이 바뀐다
- **참거짓 가리기**와 **정수 배열 자리 잡기**는 **베낀다**. 고쳐도 원본은 그대로다

읽을 때는 둘이 같아 보이므로, 고치려 할 때에야 갈린다. 그런데 대입은
또 다르다 — `a[mask] = 0` 은 베낀 것에 쓰는 것이 아니라 원본을 고친다.

마지막 절이 이 쪽에서 가장 비싼 값이다. `torch.where` 는 **고르지 않은
가지도 셈한다.** 그래서 0 으로 나눈 다음 가리는 것만으로는 역전파의
NaN 을 막지 못한다.

실행: python 09_indexing_masking.py
"""
import torch


def header(title):
    print(f"\n=== {title} ===")


def shares(a, b):
    return a.untyped_storage().data_ptr() == b.untyped_storage().data_ptr()


# ========================================================================
# 1. 자르기는 뷰다
# ========================================================================

def slicing_is_a_view():
    header("1. 자르기는 뷰다")
    a = torch.arange(20).reshape(4, 5)
    print("a =");  print(a)
    v = a[1:3, 2:4]
    print("  a[1:3, 2:4] =");  print(v)
    print("  저장소를 나눠 쓰는가:", shares(a, v))
    v[0, 0] = 999
    print("  뷰를 999 로 고친 뒤 a[1,2] =", a[1, 2].item())

    step = a[::2, ::2]
    print("  건너뛰며 자른 a[::2, ::2] 도 뷰인가:", shares(a, step),
          " 보폭", step.stride())


# ========================================================================
# 2. 가리기는 베낀다 — 읽을 때는
# ========================================================================

def masking_copies():
    header("2. 참거짓으로 가려 읽으면 베낀다")
    a = torch.tensor([1, -2, 3, -4, 5, -6])
    mask = a > 0
    picked = a[mask]
    print("  a      =", a.tolist())
    print("  a > 0  =", mask.tolist())
    print("  a[mask]=", picked.tolist(), " 모양", tuple(picked.shape))
    print("  저장소를 나눠 쓰는가:", shares(a, picked))
    picked[0] = 100
    print("  꺼낸 것을 100 으로 고친 뒤 a =", a.tolist(), "(그대로다)")

    idx = torch.tensor([0, 2, 4])
    print("  정수 배열로 고른 a[[0,2,4]] 도 베끼는가:",
          not shares(a, a[idx]))
    print("  가려 읽으면 몇 개가 나올지 모르므로 뷰로 만들 수 없다")


# ========================================================================
# 3. 대입은 다르다
# ========================================================================

def masked_assignment():
    header("3. 가려서 대입하면 원본을 고친다")
    a = torch.tensor([1, -2, 3, -4, 5, -6])
    before = a.data_ptr()
    a[a < 0] = 0
    print("  a[a < 0] = 0  ->", a.tolist())
    print("  같은 메모리인가:", a.data_ptr() == before)
    print("  읽기는 베끼고 쓰기는 제자리다 — 같은 꼴인데 갈린다")

    b = torch.arange(10).float()
    b[(b > 3) & (b < 7)] = -1
    print("  & 와 | 로 조건을 묶는다:", b.tolist())
    print("  and / or 가 아니라 & / | 다. 파이썬 연산자는 텐서에 못 쓴다")


# ========================================================================
# 4. 고르는 길들
# ========================================================================

def selecting():
    header("4. 고르는 길들")
    a = torch.arange(12).reshape(3, 4)
    print("a =");  print(a)

    r = torch.tensor([0, 2, 1])
    c = torch.tensor([1, 3, 2])
    print("  a[r, c] 짝으로 골라 (0,1),(2,3),(1,2):", a[r, c].tolist())
    print("  a[[0,2]] 행 통째로:", a[torch.tensor([0, 2])].tolist())
    print("  a[:, [1,3]] 열 통째로:", a[:, torch.tensor([1, 3])].tolist())

    rr = torch.tensor([[0], [1], [2]])      # (3,1)
    cc = torch.tensor([[0, 2]])             # (1,2)
    print("  브로드캐스팅으로 격자:", a[rr, cc].tolist())

    print("  masked_select :", a.masked_select(a > 7).tolist())
    print("  nonzero       :", a.nonzero().shape, "(자리의 좌표)")
    ri, ci = torch.where(a > 9)
    print("  where(조건) 하나만 주면 좌표:", ri.tolist(), ci.tolist())


# ========================================================================
# 5. 자리 더하기와 줄임표
# ========================================================================

def shapes_and_ellipsis():
    header("5. None 과 줄임표")
    a = torch.randn(3, 4)
    print("  a.shape          :", tuple(a.shape))
    print("  a[None].shape    :", tuple(a[None].shape), "= unsqueeze(0)")
    print("  a[:, None].shape :", tuple(a[:, None].shape))
    print("  a[..., None].shape:", tuple(a[..., None].shape), "= unsqueeze(-1)")

    big = torch.randn(2, 3, 4, 5, 6)
    print("  big.shape        :", tuple(big.shape))
    print("  big[..., 0]      :", tuple(big[..., 0].shape))
    print("  big[0, ..., 2]   :", tuple(big[0, ..., 2].shape))
    print("  줄임표는 '남은 축 전부' 라서 차원 수가 바뀌어도 안 고쳐도 된다")


# ========================================================================
# 6. where 는 고르지 않은 가지도 셈한다
# ========================================================================

def where_computes_both():
    header("6. where 는 양쪽을 다 셈한다")
    x = torch.tensor([0.0, 1.0, 2.0], requires_grad=True)
    y = torch.where(x > 0, 1.0 / x, torch.zeros_like(x))
    print("  torch.where(x > 0, 1/x, 0)")
    print("    순전파:", y.tolist(), " <- 0 자리가 잘 가려졌다")
    y.sum().backward()
    print("    x.grad:", x.grad.tolist(), " <- NaN 이 새어 나온다")
    print("  1/x 는 가려지기 전에 이미 계산되었고, 그 미분도 그래프에 남았다")

    x2 = torch.tensor([0.0, 1.0, 2.0], requires_grad=True)
    denom = torch.where(x2 > 0, x2, torch.ones_like(x2))   # 분모를 먼저 고친다
    y2 = torch.where(x2 > 0, 1.0 / denom, torch.zeros_like(x2))
    y2.sum().backward()
    print("  분모를 먼저 안전하게 바꾸면")
    print("    순전파:", y2.tolist())
    print("    x.grad:", x2.grad.tolist(), " <- 깨끗하다")


# ========================================================================
# 7. 어텐션 마스크 — 쓸모와 함정
# ========================================================================

def attention_mask():
    header("7. 어텐션 마스크")
    torch.manual_seed(0)
    s = torch.randn(1, 4, requires_grad=True)
    m = torch.tensor([[False, False, True, True]])
    p = torch.softmax(s.masked_fill(m, float("-inf")), dim=-1)
    print("  일부만 가리면 softmax:", [round(v, 4) for v in p[0].tolist()])
    print("    가린 자리가 정확히 0 이 된다 — exp(-inf) = 0")

    s2 = torch.randn(1, 4, requires_grad=True)
    m2 = torch.ones(1, 4, dtype=torch.bool)
    p2 = torch.softmax(s2.masked_fill(m2, float("-inf")), dim=-1)
    print("  한 줄을 통째로 가리면:", p2[0].tolist())
    print("    전부 -inf 라 분모가 0 이 되어 NaN 이다")
    print("    길이가 0 인 문장이 배치에 섞이면 여기서 터진다")

    causal = torch.triu(torch.ones(4, 4), diagonal=1).bool()
    print("  인과 마스크(위쪽 삼각):")
    for row in causal.tolist():
        print("   ", " ".join("X" if v else "." for v in row))


def main():
    slicing_is_a_view()
    masking_copies()
    masked_assignment()
    selecting()
    shapes_and_ellipsis()
    where_computes_both()
    attention_mask()


if __name__ == "__main__":
    main()
```

**출력:**

```

=== 1. 자르기는 뷰다 ===
a =
tensor([[ 0,  1,  2,  3,  4],
        [ 5,  6,  7,  8,  9],
        [10, 11, 12, 13, 14],
        [15, 16, 17, 18, 19]])
  a[1:3, 2:4] =
tensor([[ 7,  8],
        [12, 13]])
  저장소를 나눠 쓰는가: True
  뷰를 999 로 고친 뒤 a[1,2] = 999
  건너뛰며 자른 a[::2, ::2] 도 뷰인가: True  보폭 (10, 2)

=== 2. 참거짓으로 가려 읽으면 베낀다 ===
  a      = [1, -2, 3, -4, 5, -6]
  a > 0  = [True, False, True, False, True, False]
  a[mask]= [1, 3, 5]  모양 (3,)
  저장소를 나눠 쓰는가: False
  꺼낸 것을 100 으로 고친 뒤 a = [1, -2, 3, -4, 5, -6] (그대로다)
  정수 배열로 고른 a[[0,2,4]] 도 베끼는가: True
  가려 읽으면 몇 개가 나올지 모르므로 뷰로 만들 수 없다

=== 3. 가려서 대입하면 원본을 고친다 ===
  a[a < 0] = 0  -> [1, 0, 3, 0, 5, 0]
  같은 메모리인가: True
  읽기는 베끼고 쓰기는 제자리다 — 같은 꼴인데 갈린다
  & 와 | 로 조건을 묶는다: [0.0, 1.0, 2.0, 3.0, -1.0, -1.0, -1.0, 7.0, 8.0, 9.0]
  and / or 가 아니라 & / | 다. 파이썬 연산자는 텐서에 못 쓴다

=== 4. 고르는 길들 ===
a =
tensor([[ 0,  1,  2,  3],
        [ 4,  5,  6,  7],
        [ 8,  9, 10, 11]])
  a[r, c] 짝으로 골라 (0,1),(2,3),(1,2): [1, 11, 6]
  a[[0,2]] 행 통째로: [[0, 1, 2, 3], [8, 9, 10, 11]]
  a[:, [1,3]] 열 통째로: [[1, 3], [5, 7], [9, 11]]
  브로드캐스팅으로 격자: [[0, 2], [4, 6], [8, 10]]
  masked_select : [8, 9, 10, 11]
  nonzero       : torch.Size([11, 2]) (자리의 좌표)
  where(조건) 하나만 주면 좌표: [2, 2] [2, 3]

=== 5. None 과 줄임표 ===
  a.shape          : (3, 4)
  a[None].shape    : (1, 3, 4) = unsqueeze(0)
  a[:, None].shape : (3, 1, 4)
  a[..., None].shape: (3, 4, 1) = unsqueeze(-1)
  big.shape        : (2, 3, 4, 5, 6)
  big[..., 0]      : (2, 3, 4, 5)
  big[0, ..., 2]   : (3, 4, 5)
  줄임표는 '남은 축 전부' 라서 차원 수가 바뀌어도 안 고쳐도 된다

=== 6. where 는 양쪽을 다 셈한다 ===
  torch.where(x > 0, 1/x, 0)
    순전파: [0.0, 1.0, 0.5]  <- 0 자리가 잘 가려졌다
    x.grad: [nan, -1.0, -0.25]  <- NaN 이 새어 나온다
  1/x 는 가려지기 전에 이미 계산되었고, 그 미분도 그래프에 남았다
  분모를 먼저 안전하게 바꾸면
    순전파: [0.0, 1.0, 0.5]
    x.grad: [0.0, -1.0, -0.25]  <- 깨끗하다

=== 7. 어텐션 마스크 ===
  일부만 가리면 softmax: [0.8623, 0.1377, 0.0, 0.0]
    가린 자리가 정확히 0 이 된다 — exp(-inf) = 0
  한 줄을 통째로 가리면: [nan, nan, nan, nan]
    전부 -inf 라 분모가 0 이 되어 NaN 이다
    길이가 0 인 문장이 배치에 섞이면 여기서 터진다
  인과 마스크(위쪽 삼각):
    . X X X
    . . X X
    . . . X
    . . . .
```

## 2. 자르기는 뷰, 가리기는 베낌

```
a[1:3, 2:4]  저장소를 나눠 쓰는가: True
뷰를 999 로 고친 뒤 a[1,2] = 999

a[a > 0]     저장소를 나눠 쓰는가: False
꺼낸 것을 100 으로 고친 뒤 a = [1, -2, 3, -4, 5, -6] (그대로다)
```

건너뛰며 자르는 `a[::2, ::2]`도 뷰다. 보폭이 `(10, 2)`로 바뀔 뿐 메모리는 그대로
쓴다. 「메모리 배치와 스트라이드」에서 본 그대로다.

가리기가 뷰가 될 수 없는 까닭은 분명하다. **몇 개가 나올지 미리 모르고, 나오는
자리도 고르지 않다.** 보폭 하나로는 "3번, 7번, 11번 칸"을 가리킬 수 없다. 그러니
새 자리를 잡아 모아 담는 수밖에 없다.

정수 배열로 고르는 `a[[0, 2]]`도 같은 까닭으로 베낀다. 행 두 개가 메모리에 떨어져
있어도 상관없이 골라야 하기 때문이다.

## 3. 읽기는 베끼고 쓰기는 제자리다

```
a[a < 0] = 0  ->  [1, 0, 3, 0, 5, 0]
같은 메모리인가: True
```

`a[a < 0]`을 **읽으면** 새 텐서가 나온다. 그런데 거기에 **대입하면** `a`가 제자리에서
바뀐다. 파이썬이 `a[mask] = 0`을 `a.__setitem__(mask, 0)`으로 바꿔 부르기 때문이다 —
꺼낸 사본에 쓰는 것이 아니라 처음부터 다른 일이다.

그래서 이렇게는 안 된다.

```python
sub = a[a < 0]      # 사본이 나온다
sub[:] = 0          # 사본만 바뀐다. a 는 그대로다
```

조건을 묶을 때 쓰는 것은 `and`/`or`가 아니라 `&`/`|`다. 파이썬의 `and`는 양쪽을
참거짓 하나로 보려 하는데 텐서는 그렇게 줄여지지 않아 터진다. 그리고 `&`는 우선
순위가 비교 연산자보다 높으므로 **괄호가 필요하다** — `(b > 3) & (b < 7)`.

## 4. 고르는 길들

| 쓰는 꼴 | 하는 일 |
|---|---|
| `a[r, c]` | 좌표 **짝**으로 고른다 — `(0,1), (2,3), (1,2)` → `[1, 11, 6]` |
| `a[[0, 2]]` | 0번과 2번 **행 통째로** |
| `a[:, [1, 3]]` | 1번과 3번 **열 통째로** |
| `a[rr, cc]` (브로드캐스팅) | 행 벡터와 열 벡터가 만나 **격자**를 고른다 |
| `a.masked_select(mask)` | `a[mask]`와 같다 |
| `a.nonzero()` | 값이 아니라 **자리의 좌표**를 돌려준다 |
| `torch.where(조건)` | 인자 하나만 주면 축마다 좌표를 묶어 돌려준다 |

첫 줄이 헷갈리는 자리다. `a[r, c]`에서 `r`과 `c`가 둘 다 길이 3인 벡터면 **행 3개
× 열 3개 = 9개를 고르는 것이 아니라 짝지어 3개**를 고른다. 격자가 필요하면
브로드캐스팅이 되도록 모양을 `(3,1)`과 `(1,2)`로 맞춰 주어야 한다.

## 5. `None`과 줄임표

```
a[None].shape     : (1, 3, 4)   = unsqueeze(0)
a[:, None].shape  : (3, 1, 4)
a[..., None].shape: (3, 4, 1)   = unsqueeze(-1)
```

`None`은 그 자리에 길이 1인 축을 끼운다. `unsqueeze`와 같은 일인데, 여러 자리를
한 번에 다룰 때 읽기 쉽다.

줄임표 `...`는 **남은 축 전부**를 뜻한다.

```
big.shape      : (2, 3, 4, 5, 6)
big[..., 0]    : (2, 3, 4, 5)
big[0, ..., 2] : (3, 4, 5)
```

`big[:, :, :, :, 0]`이라고 적어도 같지만, 차원 수가 바뀌면 고쳐야 한다. `...`로
적어 두면 안 고쳐도 된다. 마지막 축을 다루는 코드가 `[..., -1]`로 적히는 것이
이 까닭이다.

## 6. `where`는 양쪽을 다 셈한다

이것이 이 쪽에서 가장 조심할 자리다.

```python
y = torch.where(x > 0, 1.0 / x, torch.zeros_like(x))
```

$x = 0$인 자리를 가려 두었으니 안전해 보인다. 순전파는 실제로 멀쩡하다.

```
순전파: [0.0, 1.0, 0.5]     <- 0 자리가 잘 가려졌다
x.grad: [nan, -1.0, -0.25]  <- NaN 이 새어 나온다
```

**역전파에서 NaN이 나온다.** `torch.where`는 파이썬의 `if`가 아니다. 조건을 보고
한쪽만 계산하는 것이 아니라, **`1.0 / x`를 세 자리 모두에 대해 먼저 계산해 놓고**
그중에서 고른다. $1/0 = \infty$가 이미 만들어졌고 그 미분 $-1/x^2$도 그래프에
남아 있으므로, 고르지 않은 자리의 미분이 그대로 올라온다.

고치는 법은 **가리기 전에 입력을 안전하게 만드는 것**이다.

```python
denom = torch.where(x > 0, x, torch.ones_like(x))   # 분모를 먼저 고친다
y = torch.where(x > 0, 1.0 / denom, torch.zeros_like(x))
```

```
순전파: [0.0, 1.0, 0.5]
x.grad: [0.0, -1.0, -0.25]  <- 깨끗하다
```

순전파 결과는 똑같다. 달라진 것은 **$1/0$이 애초에 만들어지지 않았다**는 것뿐이다.
연습문제 3에서 이 규칙이 `log`와 `sqrt`에도 그대로 적용되는지 확인한다.

## 7. 어텐션 마스크 — 쓸모와 함정

트랜스포머에서 가리기를 쓰는 꼴이 이것이다. 가릴 자리에 $-\infty$를 넣고
소프트맥스를 통과시키면 $e^{-\infty} = 0$이라 그 자리의 확률이 정확히 0이 된다.

```
일부만 가리면 softmax: [0.8623, 0.1377, 0.0, 0.0]
```

깔끔하다. 그런데 **한 줄을 통째로 가리면** 무너진다.

```
한 줄을 통째로 가리면: [nan, nan, nan, nan]
```

전부 $-\infty$이므로 소프트맥스의 분모 $\sum e^{s_i}$가 0이 되고, $0/0$이 된다.
길이가 0인 문장이 배치에 하나라도 섞이면 여기서 NaN이 나고, 그 NaN이 역전파를
타고 모델 전체로 번진다.

6절과 같은 모양의 함정이다. **가리는 것은 이미 계산된 값을 고르는 일이지, 계산을
안 하게 만드는 일이 아니다.**

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
다음 여덟 가지가 뷰인지 베낀 것인지 가려라. `a = torch.arange(12).reshape(3, 4)`로
놓고 `a[1:3]`, `a[:, 1:3]`, `a[::2]`, `a[1]`, `a[a > 5]`, `a[[0, 2]]`, `a.t()`,
`a[..., None]`.

</div>

??? success "풀이"
    ```python
    a = torch.arange(12).reshape(3, 4)

    def shares(x, y):
        return x.untyped_storage().data_ptr() == y.untyped_storage().data_ptr()

    cases = (("a[1:3]", lambda t: t[1:3]),
             ("a[:, 1:3]", lambda t: t[:, 1:3]),
             ("a[::2]", lambda t: t[::2]),
             ("a[1]", lambda t: t[1]),
             ("a[a > 5]", lambda t: t[t > 5]),
             ("a[[0, 2]]", lambda t: t[torch.tensor([0, 2])]),
             ("a.t()", lambda t: t.t()),
             ("a[..., None]", lambda t: t[..., None]))
    for name, fn in cases:
        r = fn(a)
        print(f"  {name:14s} {'뷰  ' if shares(a, r) else '베낌'}  모양 {tuple(r.shape)}")
    ```

    ```
      a[1:3]         뷰    모양 (2, 4)
      a[:, 1:3]      뷰    모양 (3, 2)
      a[::2]         뷰    모양 (2, 4)
      a[1]           뷰    모양 (4,)
      a[a > 5]       베낌  모양 (6,)
      a[[0, 2]]      베낌  모양 (2, 4)
      a.t()          뷰    모양 (4, 3)
      a[..., None]   뷰    모양 (3, 4, 1)
    ```

    **여덟 중 여섯이 뷰다.** 베끼는 것은 참거짓 가리기와 정수 배열 자리 잡기
    둘뿐이다.

    가르는 기준이 보인다. 꺼내는 자리가 **일정한 간격으로 놓여 있으면** 보폭으로
    적을 수 있으니 뷰가 된다. `a[1]`은 한 행이니 잇달아 있고, `a[::2]`는 두 칸씩
    건너뛰고, `a.t()`는 보폭을 바꿔 쓴 것이다.

    `a[[0, 2]]`가 베끼는 것이 조금 아쉽다. 0번과 2번 행은 사실 간격이 고른데도
    베낀다. 인덱스가 `[0, 2]`인지 `[0, 2, 1]`인지 미리 알 수 없으므로 일반적인
    길로 처리하기 때문이다. 간격이 고른 것이 분명하면 `a[::2]`로 적어야 공짜다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
길이가 제각각인 문장 넷을 길이 4로 맞춰 배치에 담았다. 길이가 `[3, 1, 4, 0]`일 때
패딩 자리를 가리는 마스크를 만들고 소프트맥스를 통과시켜라. 7절에서 본 일이
일어날 것이다. 어떻게 피하겠는가?

</div>

??? success "풀이"
    ```python
    lengths = torch.tensor([3, 1, 4, 0])
    L = 4
    pad = torch.arange(L)[None, :] >= lengths[:, None]   # 브로드캐스팅으로 한 번에
    torch.manual_seed(0)
    scores = torch.randn(4, L)
    p = torch.softmax(scores.masked_fill(pad, float("-inf")), dim=-1)
    for row in p.tolist():
        print("   ", [round(v, 4) for v in row])

    # 빈 줄은 첫 칸만 열어 준다
    safe = pad.clone()
    safe[pad.all(dim=-1), 0] = False
    p2 = torch.softmax(scores.masked_fill(safe, float("-inf")), dim=-1)
    for row in p2.tolist():
        print("   ", [round(v, 4) for v in row])
    ```

    ```
      패딩 마스크(X 가 가릴 자리)
        . . . X
        . X X X
        . . . .
        X X X X

      [0.2287, 0.2227, 0.5487, 0.0]
      [1.0, 0.0, 0.0, 0.0]
      [0.3107, 0.0636, 0.3194, 0.3063]
      [nan, nan, nan, nan]      <- 길이 0

      빈 줄의 첫 칸만 열어 주면
      [0.2287, 0.2227, 0.5487, 0.0]
      [1.0, 0.0, 0.0, 0.0]
      [0.3107, 0.0636, 0.3194, 0.3063]
      [1.0, 0.0, 0.0, 0.0]
    ```

    마스크를 만드는 한 줄이 먼저 볼 만하다. `arange(L)[None, :] >= lengths[:, None]`은
    `(1,4)`와 `(4,1)`을 브로드캐스팅해 `(4,4)`를 만든다. 루프 없이 "자리 번호가 그
    줄의 길이 이상이면 패딩"을 그대로 적은 것이다.

    길이 0인 넷째 줄이 전부 NaN이다. 고치는 길은 **빈 줄의 아무 칸이나 하나를
    열어 두는 것**이다. 그러면 소프트맥스가 그 칸에 확률 1을 주고 NaN이 사라진다.
    어차피 그 줄의 결과는 로스에서 빼야 하므로 어느 칸을 열든 상관없다.

    더 나은 길은 애초에 빈 문장을 배치에 안 넣는 것이다. 다만 데이터가 큰 곳에서는
    빈 줄 하나가 섞여 들어 학습 전체를 NaN으로 만드는 일이 실제로 생기므로,
    방어를 한 줄 넣어 두는 쪽이 싸다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
6절은 `1/x`에서 NaN이 새는 것을 보였다. $\log x$와 $\sqrt{x}$에서도 같은 일이
나는가? 셋을 한 틀로 확인하고, 고치는 **일반적인 꼴**을 적어라.

</div>

??? success "풀이"
    ```python
    for name, fn in (("1/x", lambda t: 1.0 / t),
                     ("log(x)", lambda t: t.log()),
                     ("sqrt(x)", lambda t: t.sqrt())):
        x = torch.tensor([0.0, 1.0, 4.0], requires_grad=True)
        y = torch.where(x > 0, fn(x), torch.zeros_like(x))
        y.sum().backward()
        print(f"  where(x>0, {name:8s}, 0)   x.grad "
              f"{[round(v, 3) if v == v else 'nan' for v in x.grad.tolist()]}")

    for name, fn in (("1/x", lambda t: 1.0 / t),
                     ("log(x)", lambda t: t.log()),
                     ("sqrt(x)", lambda t: t.sqrt())):
        x = torch.tensor([0.0, 1.0, 4.0], requires_grad=True)
        xs = torch.where(x > 0, x, torch.ones_like(x))   # 입력을 먼저 고친다
        y = torch.where(x > 0, fn(xs), torch.zeros_like(x))
        y.sum().backward()
        print(f"  입력을 먼저 고치면 {name:8s} x.grad "
              f"{[round(v, 3) for v in x.grad.tolist()]}")
    ```

    ```
      where(x>0, 1/x     , 0)   x.grad ['nan', -1.0, -0.062]
      where(x>0, log(x)  , 0)   x.grad ['nan', 1.0, 0.25]
      where(x>0, sqrt(x) , 0)   x.grad ['nan', 0.5, 0.25]

      입력을 먼저 고치면 1/x      x.grad [0.0, -1.0, -0.062]
      입력을 먼저 고치면 log(x)   x.grad [0.0, 1.0, 0.25]
      입력을 먼저 고치면 sqrt(x)  x.grad [0.0, 0.5, 0.25]
    ```

    **셋 다 같다.** 세 함수 모두 $x = 0$에서 도함수가 발산하므로
    ($-1/x^2$, $1/x$, $1/(2\sqrt{x})$), 가려 두었어도 그 자리의 미분이 NaN으로
    올라온다.

    그러니 일반적인 꼴은 이렇게 적힌다.

    ```python
    # 틀림 — 출력을 가린다
    y = torch.where(cond, f(x), fallback)

    # 맞음 — 입력을 먼저 안전하게 만든다
    x_safe = torch.where(cond, x, safe_value)
    y = torch.where(cond, f(x_safe), fallback)
    ```

    `where`를 **두 번** 쓰는 것이 핵심이다. 안쪽 것이 $f$가 위험한 값을 아예
    보지 못하게 막고, 바깥 것이 결과를 고른다. 흔히 "double where 트릭"이라
    부른다.

    더 간단한 길이 있을 때도 많다. $\log$라면 `torch.log(x + eps)`, 나눗셈이라면
    `x / (d + eps)`처럼 작은 수를 더하는 것이다. 다만 `eps`를 더하면 **순전파
    값도 조금 달라지므로**, 결과가 정확히 0이어야 하는 자리에서는 `where`를 두 번
    쓰는 쪽이 맞다.

    이 함정이 알아채기 어려운 까닭도 적어 둘 만하다. 순전파 값은 세 경우 모두
    멀쩡했다. 로스를 찍어 보아도 정상이고, 학습이 몇 걸음 돌아가다 파라미터가
    통째로 NaN이 되고 나서야 알게 된다. 그래서 NaN을 쫓을 때는 로스가 아니라
    **경사**를 먼저 보아야 한다.
