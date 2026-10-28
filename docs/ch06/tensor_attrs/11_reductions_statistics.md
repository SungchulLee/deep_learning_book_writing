# 축약과 통계 — 어느 축이 접히는가

`sum`·`mean`·`std`는 축 하나를 접어 없앤다. 쓰기는 쉬운데 조용히 틀리는 자리가
셋 있다.

- **`keepdim`을 빼면** 뒤의 브로드캐스팅이 어긋난다. 모양이 안 맞으면 터져 주지만
  **정사각이면 터지지 않고 엉뚱한 축에서 뺀다**
- **`std`의 기본값**은 `correction=1`, 곧 불편 추정량이다. numpy는 반대다
- **`median`**은 짝수 개일 때 가운데 둘을 평균내지 않는다

## 1. 코드

```python
"""축약과 통계 — 어느 축이 접히는가, 그리고 약속이 다른 자리들.

`sum`·`mean`·`std` 는 축 하나를 접어 없앤다. 쓰기는 쉬운데 두 군데서
조용히 틀린다.

- `keepdim` 을 빼면 뒤의 브로드캐스팅이 어긋난다. 모양이 안 맞으면
  터져 주지만 **정사각이면 터지지 않고 엉뚱한 축에서 뺀다**
- `std` 의 기본값은 `correction=1`, 곧 **불편 추정량**이다
- `median` 은 짝수 개일 때 가운데 둘을 평균내지 **않는다**

실행: 이 쪽의 코드를 파일로 갈무리해 돌린다
"""
import torch


def header(title):
    print(f"\n=== {title} ===")


# ========================================================================
# 1. dim 은 '접을 축' 이다
# ========================================================================

def which_axis():
    header("1. dim 은 접을 축이다")
    x = torch.tensor([[1.0, 2.0, 3.0],
                      [4.0, 5.0, 6.0],
                      [7.0, 8.0, 9.0]])
    print("x =");  print(x)
    print("  x.sum()        =", x.sum().item(), " (다 접는다)")
    print("  x.sum(dim=0)   =", x.sum(dim=0).tolist(), " 모양",
          tuple(x.sum(dim=0).shape), " 0번 축이 사라졌다")
    print("  x.sum(dim=1)   =", x.sum(dim=1).tolist(), " 모양",
          tuple(x.sum(dim=1).shape), " 1번 축이 사라졌다")
    y = torch.randn(2, 3, 4)
    print("  (2,3,4).sum(dim=(0,2)) 모양:",
          tuple(y.sum(dim=(0, 2)).shape), " 두 축을 한꺼번에")
    print("  헷갈리면 '남는 축' 을 보면 된다 — dim 에 적은 축이 없어진다")


# ========================================================================
# 2. keepdim 을 빼면
# ========================================================================

def keepdim_matters():
    header("2. keepdim")
    torch.manual_seed(0)
    x = torch.randn(3, 4)
    print("  (3,4) 에서")
    print("    mean(dim=1).shape         :", tuple(x.mean(dim=1).shape))
    print("    mean(dim=1, keepdim).shape:",
          tuple(x.mean(dim=1, keepdim=True).shape))
    try:
        x - x.mean(dim=1)
    except RuntimeError as e:
        print("    x - mean(dim=1) :", str(e).splitlines()[0][:62])
    print("    여기서는 터져 주어 바로 안다")

    sq = torch.randn(4, 4)                      # 정사각이면
    wrong = sq - sq.mean(dim=1)                 # 터지지 않는다
    right = sq - sq.mean(dim=1, keepdim=True)
    print("  (4,4) 정사각에서는")
    print("    터지지 않는다. 모양:", tuple(wrong.shape))
    print("    두 결과가 같은가:", torch.allclose(wrong, right))
    print("    바른 쪽 행 평균:", [round(v, 6) for v in right.mean(dim=1).tolist()])
    print("    틀린 쪽 행 평균:", [round(v, 6) for v in wrong.mean(dim=1).tolist()])
    print("    행을 가운데로 옮기려 했는데 열에서 뺀 셈이 되었다")


# ========================================================================
# 3. std 의 correction
# ========================================================================

def correction():
    header("3. std 의 기본값은 불편 추정량이다")
    x = torch.tensor([1.0, 2.0, 3.0, 4.0])
    n = x.numel()
    ss = ((x - x.mean()) ** 2).sum()
    print("  손으로 sqrt(SS/n)     (편향) =", round((ss / n).sqrt().item(), 6))
    print("  손으로 sqrt(SS/(n-1)) (불편) =", round((ss / (n - 1)).sqrt().item(), 6))
    print("  x.std()             =", round(x.std().item(), 6), " <- 기본값")
    print("  x.std(correction=1) =", round(x.std(correction=1).item(), 6))
    print("  x.std(correction=0) =", round(x.std(correction=0).item(), 6))
    print("  correction 은 분모에서 빼는 수다. 1 이 베셀 보정이고 기본값이다")
    print("  numpy 의 np.std 는 기본이 ddof=0 — 반대다")
    print("  var 도 같다:", round(x.var().item(), 6), "=",
          round((x.std() ** 2).item(), 6))


# ========================================================================
# 4. median 과 비김
# ========================================================================

def median_and_ties():
    header("4. median 과 비김")
    e = torch.tensor([1.0, 2.0, 3.0, 4.0])
    print("  [1,2,3,4] 의")
    print("    torch.median()   =", e.median().item(), " <- 가운데 둘 중 아래쪽")
    print("    quantile(0.5)    =", e.quantile(0.5).item(), " <- 둘의 평균")
    print("  numpy 와 파이썬 statistics 는 2.5 를 준다")
    print("  median 은 실제로 있는 값을 돌려주겠다는 약속이다")

    t = torch.tensor([3, 1, 4, 1, 5, 9, 2, 6, 5])
    v, i = t.mode()
    print("  [3,1,4,1,5,9,2,6,5] 의 mode =", v.item(), " 자리", i.item())
    print("    1 과 5 가 둘 다 두 번 나오는데 하나만 돌려준다")
    a = torch.tensor([1.0, 3.0, 3.0, 2.0])
    print("  [1,3,3,2].argmax() =", a.argmax().item(), " <- 비기면 첫 자리")


# ========================================================================
# 5. 자주 쓰는 축약들
# ========================================================================

def the_rest():
    header("5. 나머지")
    torch.manual_seed(0)
    x = torch.randn(3, 4)
    mx, idx = x.max(dim=1)
    print("  max(dim=1) 는 값과 자리를 같이 준다:",
          [round(v, 4) for v in mx.tolist()], idx.tolist())
    print("  min/max 를 dim 없이 부르면 값만:", round(x.max().item(), 4))
    lo, hi = x.aminmax()
    print("  aminmax() 한 번에:", round(lo.item(), 4), round(hi.item(), 4))

    b = torch.tensor([True, True, False])
    print("  all / any            :", b.all().item(), b.any().item())
    print("  numel / count_nonzero:", x.numel(),
          torch.count_nonzero(x > 0).item())
    print("  norm(2) 과 손으로     :", round(x.norm().item(), 6),
          round((x ** 2).sum().sqrt().item(), 6))
    c = torch.tensor([1, 2, 3, 4])
    print("  cumsum               :", c.cumsum(0).tolist())
    print("  bincount             :",
          torch.bincount(torch.tensor([1, 2, 3, 2, 1, 2, 3])).tolist())
    print("    자리가 곧 값이라 나타나지 않은 0 도 칸을 차지한다")


# ========================================================================
# 6. 배치 정규화처럼 접기
# ========================================================================

def batch_stats():
    header("6. 채널별 통계")
    torch.manual_seed(0)
    batch = torch.randn(4, 3, 8, 8)          # (N, C, H, W)
    mean = batch.mean(dim=(0, 2, 3), keepdim=True)
    std = batch.std(dim=(0, 2, 3), keepdim=True)
    print("  batch.shape:", tuple(batch.shape))
    print("  mean.shape :", tuple(mean.shape), " 채널만 남는다")
    normed = (batch - mean) / (std + 1e-5)
    print("  정규화 뒤 채널 평균:",
          [round(v, 6) for v in normed.mean(dim=(0, 2, 3)).tolist()])
    print("  정규화 뒤 채널 표준편차:",
          [round(v, 6) for v in normed.std(dim=(0, 2, 3)).tolist()])
    print("  1 에서 조금 모자란다 — 분모에 더한 1e-5 때문이다")


def main():
    which_axis()
    keepdim_matters()
    correction()
    median_and_ties()
    the_rest()
    batch_stats()


if __name__ == "__main__":
    main()
```

**출력:**

```
=== 1. dim 은 접을 축이다 ===
x =
tensor([[1., 2., 3.],
        [4., 5., 6.],
        [7., 8., 9.]])
  x.sum()        = 45.0  (다 접는다)
  x.sum(dim=0)   = [12.0, 15.0, 18.0]  모양 (3,)  0번 축이 사라졌다
  x.sum(dim=1)   = [6.0, 15.0, 24.0]  모양 (3,)  1번 축이 사라졌다
  (2,3,4).sum(dim=(0,2)) 모양: (3,)  두 축을 한꺼번에
  헷갈리면 '남는 축' 을 보면 된다 — dim 에 적은 축이 없어진다

=== 2. keepdim ===
  (3,4) 에서
    mean(dim=1).shape         : (3,)
    mean(dim=1, keepdim).shape: (3, 1)
    x - mean(dim=1) : The size of tensor a (4) must match the size of tensor b (3) a
    여기서는 터져 주어 바로 안다
  (4,4) 정사각에서는
    터지지 않는다. 모양: (4, 4)
    두 결과가 같은가: False
    바른 쪽 행 평균: [0.0, 0.0, 0.0, 0.0]
    틀린 쪽 행 평균: [0.248552, -0.060033, 0.805917, -0.994435]
    행을 가운데로 옮기려 했는데 열에서 뺀 셈이 되었다

=== 3. std 의 기본값은 불편 추정량이다 ===
  손으로 sqrt(SS/n)     (편향) = 1.118034
  손으로 sqrt(SS/(n-1)) (불편) = 1.290994
  x.std()             = 1.290994  <- 기본값
  x.std(correction=1) = 1.290994
  x.std(correction=0) = 1.118034
  correction 은 분모에서 빼는 수다. 1 이 베셀 보정이고 기본값이다
  numpy 의 np.std 는 기본이 ddof=0 — 반대다
  var 도 같다: 1.666667 = 1.666667

=== 4. median 과 비김 ===
  [1,2,3,4] 의
    torch.median()   = 2.0  <- 가운데 둘 중 아래쪽
    quantile(0.5)    = 2.5  <- 둘의 평균
  numpy 와 파이썬 statistics 는 2.5 를 준다
  median 은 실제로 있는 값을 돌려주겠다는 약속이다
  [3,1,4,1,5,9,2,6,5] 의 mode = 1  자리 3
    1 과 5 가 둘 다 두 번 나오는데 하나만 돌려준다
  [1,3,3,2].argmax() = 1  <- 비기면 첫 자리

=== 5. 나머지 ===
  max(dim=1) 는 값과 자리를 같이 준다: [1.541, 0.838, 0.182] [0, 3, 3]
  min/max 를 dim 없이 부르면 값만: 1.541
  aminmax() 한 번에: -2.1788 1.541
  all / any            : False True
  numel / count_nonzero: 12 5
  norm(2) 과 손으로     : 3.549277 3.549277
  cumsum               : [1, 3, 6, 10]
  bincount             : [0, 2, 3, 2]
    자리가 곧 값이라 나타나지 않은 0 도 칸을 차지한다

=== 6. 채널별 통계 ===
  batch.shape: (4, 3, 8, 8)
  mean.shape : (1, 3, 1, 1)  채널만 남는다
  정규화 뒤 채널 평균: [-0.0, -0.0, 0.0]
  정규화 뒤 채널 표준편차: [0.99999, 0.99999, 0.99999]
  1 에서 조금 모자란다 — 분모에 더한 1e-5 때문이다
```

## 2. `dim`은 "접을 축"이다

```
x.sum(dim=0) = [12, 15, 18]   모양 (3,)   0번 축이 사라졌다
x.sum(dim=1) = [6, 15, 24]    모양 (3,)   1번 축이 사라졌다
```

둘 다 결과가 `(3,)`이라 헷갈리기 쉬운데, **`dim`에 적은 축이 없어진다**고 읽으면
된다. `(2,3,4)`에서 `dim=(0,2)`로 접으면 1번 축만 남아 `(3,)`이 된다.

외우기보다 **남는 축을 보는** 편이 빠르다. 채널별 통계를 내고 싶으면 채널 축만
남겨야 하므로 나머지를 다 적는다 — `dim=(0, 2, 3)`.

## 3. `keepdim`을 빼면 정사각에서 조용히 틀린다

행마다 평균을 빼 가운데로 옮기는 흔한 코드를 생각해 보자.

```python
x - x.mean(dim=1)              # keepdim 없이
```

`(3, 4)`에서는 터진다. `mean(dim=1)`이 `(3,)`인데 `x`의 마지막 축은 4라서 맞지
않는다.

```
The size of tensor a (4) must match the size of tensor b (3)
```

그런데 **정사각이면 터지지 않는다.**

```
(4,4) 정사각에서는
  터지지 않는다. 모양: (4, 4)
  두 결과가 같은가: False
  바른 쪽 행 평균: [0.0, 0.0, 0.0, 0.0]
  틀린 쪽 행 평균: [-0.621345, -0.103309, 0.048584, 0.676069]
```

행을 가운데로 옮기려 했으니 행 평균이 0이어야 하는데, 0이 아니다. `(4,)`짜리
평균이 **마지막 축에 맞춰져** 열에서 빼 버렸다.

「형상, 자료형, 장치」의 `A @ B.T`와 똑같은 모양의 함정이다. **정사각에서만
조용하다.** 그래서 `keepdim=True`는 모양을 맞추려는 장식이 아니라, 접은 축이
어디였는지를 **잊지 않기 위한 표시**다.

## 4. `std`의 기본값은 불편 추정량이다

$[1, 2, 3, 4]$로 확인하면 이렇다.

| | 식 | 값 |
|---|---|---|
| 편향 | $\sqrt{SS/n}$ | 1.118034 |
| 불편 | $\sqrt{SS/(n-1)}$ | 1.290994 |
| `x.std()` | 기본값 | **1.290994** |
| `x.std(correction=1)` | | 1.290994 |
| `x.std(correction=0)` | | 1.118034 |

`correction`은 **분모에서 빼는 수**다. 1이 베셀 보정이고 그것이 기본값이므로,
파이토치의 `std()`는 **불편 추정량**이다.

여기서 걸리는 것은 numpy와 반대라는 점이다. `np.std`의 기본값은 `ddof=0`, 곧
편향 추정량이다. 같은 데이터를 두 라이브러리로 재면 다른 수가 나오고, $n$이
작을수록 차이가 크다. 논문의 표준편차를 재현하려는데 안 맞을 때 먼저 볼 자리다.

## 5. `median`은 평균내지 않는다

```
[1,2,3,4] 의
  torch.median() = 2.0    <- 가운데 둘 중 아래쪽
  quantile(0.5)  = 2.5    <- 둘의 평균
```

numpy와 파이썬 `statistics`는 2.5를 준다. 파이토치만 2.0이다.

틀린 것이 아니라 **다른 약속**이다. `torch.median`은 "데이터에 실제로 있는 값을
돌려준다"를 지킨다. 그래서 정수 텐서에서도 정수가 나오고, 자료형이 바뀌지
않는다. 평균을 원하면 `quantile(0.5)`를 쓴다.

비기는 자리도 약속이 필요하다.

| | 비기면 |
|---|---|
| `mode()` | 하나만 돌려준다 (위 예에서 1과 5가 둘 다 두 번인데 1) |
| `argmax()` | **첫 자리**를 돌려준다 |

`argmax`가 첫 자리라는 것은 분류에서 쓸모가 있다. 로짓이 완전히 같은 두 클래스가
있으면 언제나 번호가 작은 쪽이 뽑히므로, 적어도 **재현은 된다.**

## 6. 채널별 통계

```
batch.shape: (4, 3, 8, 8)
mean.shape : (1, 3, 1, 1)   채널만 남는다
정규화 뒤 채널 평균:     [-0.0, -0.0, 0.0]
정규화 뒤 채널 표준편차: [0.99999, 0.99999, 0.99999]
```

배치 정규화가 하는 일이 이 두 줄이다. 표준편차가 1이 아니라 0.99999인 것은
분모에 더한 `1e-5` 때문이고, 0으로 나누는 것을 막으려고 일부러 넣은 값이다.

`keepdim=True`가 여기서 필수다. `(1,3,1,1)`이라야 `(4,3,8,8)`과 브로드캐스팅되어
채널마다 제 평균이 빠진다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
`torch.arange(24).reshape(2, 3, 4)`를 `dim`을 바꿔 가며 더하면 결과 모양이 어떻게
되는가? `None`, `0`, `1`, `2`, `(0,1)`, `(1,2)`에 대해 **먼저 적고** 확인하라.

</div>

??? success "풀이"
    ```python
    x = torch.arange(24).reshape(2, 3, 4).float()
    for d in (None, 0, 1, 2, (0, 1), (1, 2)):
        r = x.sum() if d is None else x.sum(dim=d)
        print(f"  dim={str(d):8s} -> 모양 {tuple(r.shape)}")
    ```

    ```
      dim=None     -> 모양 ()
      dim=0        -> 모양 (3, 4)
      dim=1        -> 모양 (2, 4)
      dim=2        -> 모양 (2, 3)
      dim=(0, 1)   -> 모양 (4,)
      dim=(1, 2)   -> 모양 (2,)
    ```

    규칙이 하나다. **적은 축을 모양에서 지우면 된다.** `(2,3,4)`에서 0을 지우면
    `(3,4)`, `(0,1)`을 지우면 `(4,)`. `dim=None`이면 다 지워 0차원이 남는다.

    0차원 `()`은 스칼라다. 모양이 비어 있지 `None`이 아니며, `.item()`으로 파이썬
    수를 꺼낼 수 있다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
`(N, C, H, W)` 배치에서 **채널별**·**표본별**·**자리별** 평균을 각각 구하려면
`dim`에 무엇을 적어야 하는가? 셋 다 `keepdim=True`로 구해 `(b - m)`이 되는지
확인하라.

</div>

??? success "풀이"
    ```python
    b = torch.randn(4, 3, 8, 8)
    for name, dims in (("채널별 (0,2,3)", (0, 2, 3)),
                       ("표본별 (1,2,3)", (1, 2, 3)),
                       ("자리별 (0,1)", (0, 1))):
        m = b.mean(dim=dims, keepdim=True)
        print(f"  {name:16s} mean.shape {tuple(m.shape)}  "
              f"(b-m).shape {tuple((b - m).shape)}")
    ```

    ```
      채널별 (0,2,3)      mean.shape (1, 3, 1, 1)  (b-m).shape (4, 3, 8, 8)
      표본별 (1,2,3)      mean.shape (4, 1, 1, 1)  (b-m).shape (4, 3, 8, 8)
      자리별 (0,1)        mean.shape (1, 1, 8, 8)  (b-m).shape (4, 3, 8, 8)
    ```

    **남기고 싶은 축만 빼고 전부 적는다.** 채널을 남기려면 `(0,2,3)`, 표본을
    남기려면 `(1,2,3)`이다.

    셋이 서로 다른 정규화에 해당한다.

    | 남긴 축 | 무엇인가 |
    |---|---|
    | 채널 | **배치 정규화** — 채널마다 통계를 낸다 |
    | 표본 | **층 정규화**에 가깝다 — 표본마다 제 통계로 맞춘다 |
    | 자리 | 화소 자리마다 — 영상에서는 잘 안 쓴다 |

    `keepdim=True` 덕분에 셋 다 `(4,3,8,8)`로 브로드캐스팅된다. 빼면 모양이
    돌아오는지 확인하는 것이 `dim`을 제대로 적었는지 보는 가장 빠른 길이다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
float32는 유효숫자가 24비트뿐이다. 1000 근처의 수 20만 개를 더하면 오차가 쌓일
것 같다. **차례대로 더한 것**과 `torch.sum`을 float64 참값과 견주어라. 결과를
어떻게 설명하겠는가?

</div>

??? success "풀이"
    ```python
    import numpy as np

    torch.manual_seed(0)
    v = torch.rand(200000, dtype=torch.float32) + 1000.0
    exact = float(v.numpy().astype(np.float64).sum())

    s = np.float32(0.0)                       # float32 누산기 하나로 차례대로
    for x in v.numpy():
        s = np.float32(s + x)

    print(f"  차례대로 (float32 누산기) {float(s):16.3f}")
    print(f"  torch.sum               {float(v.sum()):16.3f}")
    print(f"  float64 참값             {exact:16.3f}")
    print(f"  차례대로 상대오차 {abs(float(s) - exact) / exact:.2e}")
    print(f"  torch.sum 상대오차 {abs(float(v.sum()) - exact) / exact:.2e}")
    ```

    ```
      차례대로 (float32 누산기)    200534704.000
      torch.sum                  200100160.000
      float64 참값                200100163.508
      차례대로 상대오차 2.17e-03
      torch.sum 상대오차 1.75e-08
    ```

    **짐작한 오차는 실제로 생긴다 — 차례대로 더하면.** 참값이 2억 100만인데
    차례대로 더한 값은 2억 53만이다. **43만이나 크다.**

    까닭은 누산기가 커질수록 더하는 수가 상대적으로 작아지기 때문이다. 누산기가
    $2 \times 10^8$쯤 되면 float32의 간격이 16 정도인데 거기에 1000을 더하면
    반올림으로 밀려 올라간다. 2억 번 가까이 그러면 쌓인다.

    그런데 **`torch.sum`은 다섯 자릿수나 정확하다**(1.75e-08). 차례대로 더하지
    않기 때문이다. 파이토치는 **짝지어 더하기**(pairwise summation)를 쓴다 —
    덩어리로 나누어 각각 더한 다음 그 결과들을 다시 더하는 식이라, 누산기가
    한없이 커지지 않는다. 오차가 $O(n)$ 대신 $O(\log n)$으로 는다.

    여기서 가져갈 것이 둘이다.

    첫째, **걱정은 맞았는데 이미 해결되어 있었다.** 라이브러리 함수를 손으로 짠
    루프로 바꾸면 이런 것을 잃는다. `for`로 합을 구하지 말아야 할 까닭이 속도만은
    아니다.

    둘째, 그래도 안전한 쪽을 원하면 **누산 자료형을 올리면 된다.**
    `v.sum(dtype=torch.float64)`는 입력이 float32여도 float64로 더한다. 로스를
    모으거나 평가 지표를 쌓을 때 쓸 만하다.
