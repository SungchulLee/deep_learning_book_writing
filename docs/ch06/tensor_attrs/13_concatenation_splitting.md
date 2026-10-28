# 이어 붙이기와 나누기 — 합치면 베끼고 나누면 뷰다

이 절은 한 쌍의 대칭에서 시작한다.

| | 메모리 |
|---|---|
| `cat` · `stack` | 결과를 담을 자리를 새로 잡고 **베낀다** |
| `split` · `chunk` · `unbind` | **뷰**를 돌려준다 |

생각해 보면 그럴 수밖에 없다. 흩어져 있던 텐서 둘을 한 덩이로 만들려면 어딘가에
나란히 적어야 하고, 한 덩이를 조각내는 것은 **어디부터 어디까지인지만 적어 두면**
되기 때문이다.

그리고 나누는 쪽에 함정이 하나 있다. **`chunk(k)`가 언제나 $k$덩이를 주지는
않는다.**

## 1. 코드

```python
"""이어 붙이기와 나누기 — 합치면 베끼고 나누면 뷰다.

이 절은 그 대칭에서 시작한다.

- `cat` · `stack` 은 결과를 담을 자리를 새로 잡고 **베낀다**
- `split` · `chunk` · `unbind` 는 **뷰**를 돌려준다. 고치면 원본이 바뀐다

나누는 쪽에서 조심할 것이 하나 더 있다. **`chunk(k)` 가 언제나 k 덩이를
주지는 않는다.** 요청보다 적게 나오는 자리가 있다.

실행: 이 쪽의 코드를 파일로 갈무리해 돌린다
"""
import torch


def header(title):
    print(f"\n=== {title} ===")


def shares(a, b):
    return a.untyped_storage().data_ptr() == b.untyped_storage().data_ptr()


# ========================================================================
# 1. 합치기 — cat 과 stack
# ========================================================================

def joining():
    header("1. cat 과 stack")
    a = torch.tensor([[1, 2], [3, 4]])
    b = torch.tensor([[5, 6], [7, 8]])
    print("  cat(dim=0)  :", tuple(torch.cat([a, b], dim=0).shape), "축 2개 그대로")
    print("  cat(dim=1)  :", tuple(torch.cat([a, b], dim=1).shape))
    print("  stack(dim=0):", tuple(torch.stack([a, b], dim=0).shape), "축 3개로")
    print("  stack(dim=1):", tuple(torch.stack([a, b], dim=1).shape))
    print("  cat 결과가 원본과 저장소를 나눠 쓰는가:",
          shares(a, torch.cat([a, b], dim=0)))
    print("  합치는 쪽은 언제나 새 자리를 잡는다")

    c = torch.tensor([[9, 10]])
    print("  셋 이상도 한 번에:", tuple(torch.cat([a, b, c], dim=0).shape))


# ========================================================================
# 2. hstack / vstack 은 차원에 따라 다르다
# ========================================================================

def stack_helpers():
    header("2. hstack / vstack")
    u, v = torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6])
    print("  1차원 hstack:", torch.hstack([u, v]).tolist(), " <- dim=0 처럼")
    print("  1차원 vstack:", torch.vstack([u, v]).tolist(), " <- 행으로 쌓는다")
    m = torch.ones(2, 2, dtype=torch.long)
    n = torch.zeros(2, 2, dtype=torch.long)
    print("  2차원 hstack 모양:", tuple(torch.hstack([m, n]).shape), " <- dim=1")
    print("  2차원 vstack 모양:", tuple(torch.vstack([m, n]).shape), " <- dim=0")
    print("  hstack 은 1차원에서 dim=0, 2차원에서 dim=1 이다")
    print("  축을 손으로 적는 cat/stack 이 헷갈릴 일이 없다")


# ========================================================================
# 3. 나누기 — split 은 크기, chunk 는 개수
# ========================================================================

def splitting():
    header("3. split 과 chunk")
    x = torch.arange(10)
    print("  split(3)       크기:", [len(c) for c in x.split(3)],
          " <- 3씩 자르고 남는 건 남는 대로")
    print("  split([2,3,5]) 크기:", [len(c) for c in x.split([2, 3, 5])])
    try:
        x.split([2, 3, 4])
    except RuntimeError as e:
        print("  split([2,3,4]) :", str(e).splitlines()[0][:70])
    print("  split 에 리스트를 주면 합이 딱 맞아야 한다")


def chunk_surprise():
    header("4. chunk(k) 가 k 덩이를 주지 않을 때")
    for n, k in ((10, 4), (5, 3), (4, 3), (7, 3), (9, 4), (12, 5)):
        cs = torch.arange(n).chunk(k)
        flag = "" if len(cs) == k else "   <- 요청은 %d" % k
        print(f"  arange({n:2d}).chunk({k}) -> {len(cs)}덩이  "
              f"크기 {[len(c) for c in cs]}{flag}")
    print("  chunk 는 먼저 덩이 크기를 ceil(n/k) 로 정하고 그 크기로 자른다")
    print("  9 를 4 로 나누면 크기 3 이 되고, 3 씩 자르면 세 덩이면 끝난다")


# ========================================================================
# 5. 나눈 조각은 뷰다
# ========================================================================

def pieces_are_views():
    header("5. 나눈 조각은 뷰다")
    a = torch.arange(12).reshape(3, 4)
    for name, parts in (("chunk(3)", a.chunk(3)),
                        ("split(2)", a.split(2)),
                        ("unbind(0)", a.unbind(0))):
        print(f"  {name:10s} 첫 조각이 뷰인가: {shares(a, parts[0])}")
    first = a.chunk(3)[0]
    first[0, 0] = 99
    print("  조각을 99 로 고치니 원본 a[0,0] =", a[0, 0].item())
    print("  나누는 쪽은 베끼지 않는다. 읽기만 할 때는 공짜다")
    print("  고칠 작정이면 조각마다 clone() 을 해야 한다")


# ========================================================================
# 6. 되돌리기
# ========================================================================

def round_trip():
    header("6. 되돌리기")
    x = torch.arange(10)
    print("  split 하고 cat 하면 돌아오는가:",
          torch.equal(x, torch.cat(x.split(3))))
    y = torch.randn(3, 4)
    print("  unbind 하고 stack 하면 돌아오는가:",
          torch.equal(y, torch.stack(y.unbind(0))))
    print("  unbind(0) 은 for 문으로 도는 것과 같다:",
          all(torch.equal(p, q) for p, q in zip(y.unbind(0), y)))
    print("  다만 돌아온 것은 새 텐서다:",
          shares(y, torch.stack(y.unbind(0))))


def main():
    joining()
    stack_helpers()
    splitting()
    chunk_surprise()
    pieces_are_views()
    round_trip()


if __name__ == "__main__":
    main()
```

**출력:**

```
=== 1. cat 과 stack ===
  cat(dim=0)  : (4, 2) 축 2개 그대로
  cat(dim=1)  : (2, 4)
  stack(dim=0): (2, 2, 2) 축 3개로
  stack(dim=1): (2, 2, 2)
  cat 결과가 원본과 저장소를 나눠 쓰는가: False
  합치는 쪽은 언제나 새 자리를 잡는다
  셋 이상도 한 번에: (5, 2)

=== 2. hstack / vstack ===
  1차원 hstack: [1, 2, 3, 4, 5, 6]  <- dim=0 처럼
  1차원 vstack: [[1, 2, 3], [4, 5, 6]]  <- 행으로 쌓는다
  2차원 hstack 모양: (2, 4)  <- dim=1
  2차원 vstack 모양: (4, 2)  <- dim=0
  hstack 은 1차원에서 dim=0, 2차원에서 dim=1 이다
  축을 손으로 적는 cat/stack 이 헷갈릴 일이 없다

=== 3. split 과 chunk ===
  split(3)       크기: [3, 3, 3, 1]  <- 3씩 자르고 남는 건 남는 대로
  split([2,3,5]) 크기: [2, 3, 5]
  split([2,3,4]) : split_with_sizes expects split_sizes to sum exactly to 10 (input tenso
  split 에 리스트를 주면 합이 딱 맞아야 한다

=== 4. chunk(k) 가 k 덩이를 주지 않을 때 ===
  arange(10).chunk(4) -> 4덩이  크기 [3, 3, 3, 1]
  arange( 5).chunk(3) -> 3덩이  크기 [2, 2, 1]
  arange( 4).chunk(3) -> 2덩이  크기 [2, 2]   <- 요청은 3
  arange( 7).chunk(3) -> 3덩이  크기 [3, 3, 1]
  arange( 9).chunk(4) -> 3덩이  크기 [3, 3, 3]   <- 요청은 4
  arange(12).chunk(5) -> 4덩이  크기 [3, 3, 3, 3]   <- 요청은 5
  chunk 는 먼저 덩이 크기를 ceil(n/k) 로 정하고 그 크기로 자른다
  9 를 4 로 나누면 크기 3 이 되고, 3 씩 자르면 세 덩이면 끝난다

=== 5. 나눈 조각은 뷰다 ===
  chunk(3)   첫 조각이 뷰인가: True
  split(2)   첫 조각이 뷰인가: True
  unbind(0)  첫 조각이 뷰인가: True
  조각을 99 로 고치니 원본 a[0,0] = 99
  나누는 쪽은 베끼지 않는다. 읽기만 할 때는 공짜다
  고칠 작정이면 조각마다 clone() 을 해야 한다

=== 6. 되돌리기 ===
  split 하고 cat 하면 돌아오는가: True
  unbind 하고 stack 하면 돌아오는가: True
  unbind(0) 은 for 문으로 도는 것과 같다: True
  다만 돌아온 것은 새 텐서다: False
```

## 2. `cat`과 `stack`

```
cat(dim=0)  : (4, 2)      축 2개 그대로
stack(dim=0): (2, 2, 2)   축 3개로
```

「cat 시각화」에서 본 그대로다. `cat`은 있는 축을 늘리고 `stack`은 없던 축을
만든다. 그리고 **둘 다 베낀다** — 결과가 원본과 저장소를 나눠 쓰지 않는다.

## 3. `hstack`/`vstack`은 차원에 따라 다른 축을 쓴다

| | 1차원 입력 | 2차원 입력 |
|---|---|---|
| `hstack` | 이어 붙인다 (`dim=0`처럼) | `dim=1` |
| `vstack` | 행으로 쌓는다 (`(2,3)`) | `dim=0` |

`hstack`이 1차원에서는 `dim=0`이고 2차원에서는 `dim=1`이다. 이름이 "가로"를
뜻하는데 1차원 텐서에는 가로밖에 없으니 그렇게 정해 둔 것이다.

편한 이름이지만 **입력 차원을 보고 생각해야** 하므로, 코드에서는 축을 손으로
적는 `cat(..., dim=1)`이 헷갈릴 일이 없다. numpy에서 넘어온 코드에나 쓸 일이다.

## 4. `split`은 크기, `chunk`는 개수

```
split(3)        크기: [3, 3, 3, 1]
split([2,3,5])  크기: [2, 3, 5]
split([2,3,4])  : split_sizes to sum exactly to 10
```

`split(3)`은 "3씩 자르라"는 뜻이라 10개를 자르면 마지막이 1개로 남는다. 리스트를
주면 **합이 딱 맞아야** 하고, 안 맞으면 터진다.

## 5. `chunk(k)`가 $k$덩이를 주지 않을 때

```
arange(10).chunk(4) -> 4덩이  [3, 3, 3, 1]
arange( 4).chunk(3) -> 2덩이  [2, 2]        <- 요청은 3
arange( 9).chunk(4) -> 3덩이  [3, 3, 3]     <- 요청은 4
arange(12).chunk(5) -> 4덩이  [3, 3, 3, 3]  <- 요청은 5
```

`chunk`는 **개수를 보장하지 않는다.** 하는 일을 적으면 이렇다.

1. 덩이 크기를 $\lceil n/k \rceil$로 정한다
2. 그 크기로 끝까지 자른다

9를 4로 나누면 크기가 $\lceil 9/4 \rceil = 3$이고, 3씩 자르면 **세 덩이로 끝난다.**
넷째 덩이를 만들 것이 남지 않는다.

이것이 문제가 되는 자리는 분명하다. 여러 장치에 나눠 보내려고 `chunk(num_gpus)`를
불렀는데 조각이 모자라면, 마지막 장치가 일을 못 받는다. 개수가 꼭 맞아야 하면
`chunk`가 아니라 `split`에 크기 리스트를 주거나, `tensor_split`을 쓴다 — 그쪽은
개수를 지킨다.

## 6. 나눈 조각은 뷰다

```
chunk(3)   첫 조각이 뷰인가: True
split(2)   첫 조각이 뷰인가: True
unbind(0)  첫 조각이 뷰인가: True

조각을 99 로 고치니 원본 a[0,0] = 99
```

셋 다 뷰다. 읽기만 할 때는 공짜이고, **고칠 작정이면 조각마다 `clone()`을**
해야 한다. 조각을 고쳐 원본이 바뀌는 것은 「메모리와 저장소」의 `expand`와 같은
종류의 조용한 사고다.

## 7. 되돌리기

```
split 하고 cat 하면 돌아오는가: True
unbind 하고 stack 하면 돌아오는가: True
다만 돌아온 것은 새 텐서다: False
```

`unbind(0)`은 그냥 `for row in y`와 같다. 쓸모는 **명시적으로 축을 고를 수 있다**는
데 있다 — `unbind(1)`이면 열로 푼다.

되돌아온 것이 새 텐서라는 점은 중요하다. 나누기는 공짜였는데 다시 합치는 데서
값을 치른다. 「cat 시각화」에서 루프 `cat`이 비쌌던 까닭이 여기 있다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
`(6, 4)` 텐서를 `cat`과 `stack`으로 각각 둘씩 합치면 결과 모양이 어떻게 되는가?
`dim`을 0과 1로 바꿔 가며 네 가지를 **먼저 적고** 확인하라.

</div>

??? success "풀이"
    ```python
    a = b = torch.zeros(6, 4)
    for fn_name, fn in (("cat", torch.cat), ("stack", torch.stack)):
        for d in (0, 1):
            print(f"  {fn_name}(dim={d}) -> {tuple(fn([a, b], dim=d).shape)}")
    ```

    ```
      cat(dim=0) -> (12, 4)
      cat(dim=1) -> (6, 8)
      stack(dim=0) -> (2, 6, 4)
      stack(dim=1) -> (6, 2, 4)
    ```

    `cat`은 그 축의 길이를 더하고, `stack`은 그 자리에 길이 2인 축을 끼운다.

    `stack(dim=1)`이 `(6, 2, 4)`인 것을 보아 두면 좋다. `stack`의 `dim`은 **새 축이
    들어갈 자리**이지 기존 축이 아니다. 그래서 `dim`을 0부터 `ndim`까지 줄 수
    있고, `(6,4)`에 쌓으면 `dim=2`도 된다 — `(6, 4, 2)`가 나온다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
`chunk(4)`가 몇 덩이를 주는지 $n = 1$부터 12까지 적어라. 덩이 수가 **$n$이 늘어도
줄어드는** 자리가 있는가?

</div>

??? success "풀이"
    ```python
    for n in range(1, 13):
        cs = torch.arange(n).chunk(4)
        print(f"  n={n:2d}  chunk(4) -> {len(cs)}덩이 {[len(c) for c in cs]}")
    ```

    ```
      n= 1  chunk(4) -> 1덩이 [1]
      n= 2  chunk(4) -> 2덩이 [1, 1]
      n= 3  chunk(4) -> 3덩이 [1, 1, 1]
      n= 4  chunk(4) -> 4덩이 [1, 1, 1, 1]
      n= 5  chunk(4) -> 3덩이 [2, 2, 1]
      n= 6  chunk(4) -> 3덩이 [2, 2, 2]
      n= 7  chunk(4) -> 4덩이 [2, 2, 2, 1]
      n= 8  chunk(4) -> 4덩이 [2, 2, 2, 2]
      n= 9  chunk(4) -> 3덩이 [3, 3, 3]
      n=10  chunk(4) -> 4덩이 [3, 3, 3, 1]
      n=11  chunk(4) -> 4덩이 [3, 3, 3, 2]
      n=12  chunk(4) -> 4덩이 [3, 3, 3, 3]
    ```

    **있다.** $n=4$에서 4덩이였다가 $n=5$에서 **3덩이로 줄어든다.** $n=8$에서 4,
    $n=9$에서 다시 3이다.

    덩이 크기가 $\lceil n/4 \rceil$로 한 칸 올라가는 순간(4→5에서 1→2, 8→9에서
    2→3) 같은 개수를 덮는 데 덩이가 덜 들기 때문이다.

    이것이 왜 위험한지는 코드가 `len(chunks)`를 안 보고 `4`를 가정할 때 드러난다.
    데이터 수가 하필 8에서 9로 늘면 조각이 하나 사라지고, 네 번째 장치나 네 번째
    일꾼이 조용히 아무 일도 안 하게 된다. 개수가 중요하면 `torch.tensor_split`을
    쓴다 — 그쪽은 요청한 개수를 지키고 크기를 고르게 맞춘다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
나누고 합치는 것이 **미분을 지나가는지** 확인하라. (가) 텐서를 `chunk`로 나눠
조각마다 로스를 셈해 더한 뒤 역전파하면 경사가 맞는가? (나) 같은 텐서를 두 번
`cat`해서 역전파하면 경사가 어떻게 되는가?

</div>

??? success "풀이"
    ```python
    torch.manual_seed(0)
    x = torch.randn(1000, 8, requires_grad=True)
    loss = sum((p ** 2).sum() for p in x.chunk(4))
    loss.backward()
    print("  chunk 조각마다 셈하고 더해 역전파:",
          torch.allclose(x.grad, 2 * x.detach()))

    y = torch.randn(4, 3, requires_grad=True)
    torch.cat([y, y], dim=0).sum().backward()
    print("  같은 텐서를 두 번 cat 하면 경사:", y.grad[0].tolist())
    ```

    ```
      chunk 조각마다 셈하고 더해 역전파: True
      같은 텐서를 두 번 cat 하면 경사: [2.0, 2.0, 2.0]
    ```

    **(가)** 조각마다 따로 셈해 더한 것이 통째로 셈한 것과 똑같다 —
    $\partial(\sum x^2)/\partial x = 2x$가 그대로 나왔다. 나누기는 뷰이므로 그래프가
    끊기지 않고, 역전파가 조각들의 경사를 제자리에 다시 꽂아 넣는다.

    이것이 「메모리 관리」의 경사 누적이 성립하는 바탕이다. 배치를 조각내 따로
    `backward()`하고 한 번 `step()`하면 큰 배치와 같아지는 것은 나누기가 미분을
    그대로 지나가기 때문이다.

    **(나)** 경사가 1이 아니라 **2**다. `y`가 결과의 두 자리에 쓰였으므로 양쪽에서
    온 경사가 더해진다. 「메모리와 저장소」의 `expand`와 같은 규칙이고, 「cat
    시각화」에서 **서로 다른** 두 텐서를 이었을 때 각자 제 몫만 받은 것과 대비된다.

    가르는 것은 `cat`이 아니라 **같은 텐서를 두 번 넣었는가**다. 한 입력이 여러
    출력에 쓰이면 언제나 더해진다.
