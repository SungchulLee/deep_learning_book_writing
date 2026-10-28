# 비교, 범위 제한, 정렬 — NaN이 섞이면 달라지는 것들

비교 연산자는 참거짓 텐서를 주고, 정렬은 값과 자리를 같이 준다. 거기까지는
짐작대로다. 조심할 자리가 둘이다.

- **NaN은 어느 비교에도 참이 아니다.** `x > 0`도 `x < 0`도 거짓이라, 거르는
  조건을 **어떻게 적느냐에 따라** NaN이 남기도 하고 빠지기도 한다
- **`clamp`의 경사는 범위 밖에서 0이다.** 자른 자리는 학습이 멈춘다

## 1. 코드

```python
"""비교, 범위 제한, 정렬 — NaN 이 섞이면 달라지는 것들.

비교 연산자는 참거짓 텐서를 돌려주고, 정렬은 값과 자리를 같이 준다.
여기까지는 짐작대로다. 조심할 자리가 둘이다.

- **NaN 은 어느 비교에도 참이 아니다.** `x > 0` 도 `x < 0` 도 거짓이라,
  거르는 조건을 어떻게 적느냐에 따라 NaN 이 남기도 하고 빠지기도 한다
- **`clamp` 의 경사는 범위 밖에서 0 이다.** 자른 자리는 학습이 멈춘다

실행: python 15_comparison_clamping_sorting.py
"""
import torch

NAN = float("nan")
INF = float("inf")


def header(title):
    print(f"\n=== {title} ===")


# ========================================================================
# 1. 비교는 참거짓 텐서를 준다
# ========================================================================

def comparisons():
    header("1. 비교 연산자")
    a = torch.tensor([1, 5, 3, 5])
    b = torch.tensor([2, 4, 3, 5])
    print("  a =", a.tolist(), "  b =", b.tolist())
    print("  a > b  :", (a > b).tolist(), " dtype", (a > b).dtype)
    print("  a == b :", (a == b).tolist())
    print("  a >= 3 :", (a >= 3).tolist(), " 스칼라와도 된다")
    print("  (a > b).sum() =", (a > b).sum().item(), " <- 참을 센다")
    print("  & | ~ 로 묶는다:", ((a > 2) & (a < 5)).tolist())
    print("  and/or/not 는 못 쓴다. 텐서를 참거짓 하나로 줄일 수 없어서다")
    print("  torch.equal(a, b) 는 통째로 비교해 하나를 준다:",
          torch.equal(a, b))


# ========================================================================
# 2. NaN 은 어느 비교에도 참이 아니다
# ========================================================================

def nan_comparisons():
    header("2. NaN 과 비교")
    print("  nan == nan :", (torch.tensor(NAN) == torch.tensor(NAN)).item())
    x = torch.tensor([1.0, NAN, 3.0])
    print("  x =", x.tolist())
    print("  x > 0 :", (x > 0).tolist())
    print("  x < 0 :", (x < 0).tolist(), " <- 둘 다 거짓이다")
    print("  x[x > 0]    :", x[x > 0].tolist(), " <- NaN 이 조용히 빠진다")
    print("  x[~(x <= 0)]:", x[~(x <= 0)].tolist(), " <- 부정하면 남는다")
    print("  같은 뜻으로 적은 두 조건이 다른 답을 준다")
    print("  NaN 을 찾으려면 isnan 을 써야 한다:", torch.isnan(x).tolist())

    s = torch.tensor([1.0, NAN, INF, -INF])
    print("  isfinite:", torch.isfinite(s).tolist())
    print("  isinf   :", torch.isinf(s).tolist())


# ========================================================================
# 3. 정렬과 극값에서의 NaN
# ========================================================================

def nan_in_sorting():
    header("3. 정렬과 극값에서의 NaN")
    y = torch.tensor([3.0, NAN, 1.0, 2.0])
    print("  y =", y.tolist())
    print("  sort().values :", y.sort().values.tolist(), " <- 맨 뒤로 간다")
    print("  max()         :", y.max().item(), " <- 번져 나온다")
    print("  argmax()      :", y.argmax().item(), " <- NaN 자리를 가리킨다")
    print("  maximum(y, 0) :", torch.maximum(y, torch.zeros(4)).tolist())
    print("  nanmean / nansum 은 건너뛴다:",
          torch.tensor([1.0, NAN, 3.0]).nanmean().item(),
          torch.tensor([1.0, NAN, 3.0]).nansum().item())
    print("  max 와 argmax 가 어긋나 보이지만 둘 다 NaN 자리를 말하고 있다")


# ========================================================================
# 4. clamp 의 경사는 밖에서 0 이다
# ========================================================================

def clamping():
    header("4. clamp")
    x = torch.tensor([-3.0, -0.5, 0.5, 3.0])
    print("  x            :", x.tolist())
    print("  clamp(-1, 1) :", x.clamp(-1, 1).tolist())
    print("  clamp(min=0) :", x.clamp(min=0).tolist(), " = relu")
    print("  clip 은 clamp 의 다른 이름:", torch.equal(x.clip(-1, 1),
                                                 x.clamp(-1, 1)))

    v = torch.tensor([-3.0, -0.5, 0.5, 3.0], requires_grad=True)
    v.clamp(-1, 1).sum().backward()
    print("  clamp 의 경사:", v.grad.tolist())
    print("  잘린 자리는 0 이다 — 그 원소는 더 이상 배우지 않는다")
    print("  활성값을 clamp 로 묶으면 ReLU 의 죽은 뉴런과 같은 일이 난다")


# ========================================================================
# 5. 정렬 · topk · kthvalue
# ========================================================================

def sorting():
    header("5. 정렬과 고르기")
    t = torch.tensor([3.0, 1.0, 4.0, 1.0, 5.0])
    v, i = t.sort()
    print("  sort        값:", v.tolist(), " 자리:", i.tolist())
    print("  sort(내림차순):", t.sort(descending=True).values.tolist())
    print("  argsort     :", t.argsort().tolist(), " = sort 의 자리만")
    print("  정렬한 값은 t[argsort] 와 같은가:",
          torch.equal(t[t.argsort()], t.sort().values))

    tv, ti = t.topk(2)
    print("  topk(2)     값:", tv.tolist(), " 자리:", ti.tolist())
    print("  topk(2, largest=False):", t.topk(2, largest=False).values.tolist())
    print("  kthvalue(2) 두 번째로 작은 값:", t.kthvalue(2).values.item())

    m = torch.tensor([[3.0, 1.0], [2.0, 4.0]])
    print("  2차원은 축을 고른다 sort(dim=0):",
          m.sort(dim=0).values.flatten().tolist())


# ========================================================================
# 6. 가까운가 — allclose 의 식
# ========================================================================

def closeness():
    header("6. allclose")
    a = torch.tensor([1.0, 2.0, 3.0])
    print("  기본값은 rtol=1e-5, atol=1e-8")
    print("  판정식은 |a - b| <= atol + rtol * |b| 다")
    for d in (1e-4, 1e-5, 1e-6):
        b = a + d
        bound = 1e-8 + 1e-5 * b.abs()
        print(f"  차이 {d:.0e}  |a-b| {d:.1e} vs 한계 {bound.min():.1e}~"
              f"{bound.max():.1e}  allclose {torch.allclose(a, b)}")
    print("  rtol 이 상대 오차라 큰 값일수록 너그러워진다")
    print("  0 근처를 견줄 때는 atol 을 손으로 올려야 한다")
    z = torch.zeros(3)
    print("  allclose(0, 1e-7) :", torch.allclose(z, z + 1e-7))
    print("  allclose(0, 1e-7, atol=1e-6):",
          torch.allclose(z, z + 1e-7, atol=1e-6))
    print("  NaN 은 기본으로 같지 않다:",
          torch.allclose(torch.tensor([NAN]), torch.tensor([NAN])))
    print("  equal_nan=True 로 바꿀 수 있다:",
          torch.allclose(torch.tensor([NAN]), torch.tensor([NAN]),
                         equal_nan=True))


def main():
    comparisons()
    nan_comparisons()
    nan_in_sorting()
    clamping()
    sorting()
    closeness()


if __name__ == "__main__":
    main()
```

**출력:**

```
=== 1. 비교 연산자 ===
  a = [1, 5, 3, 5]   b = [2, 4, 3, 5]
  a > b  : [False, True, False, False]  dtype torch.bool
  a == b : [False, False, True, True]
  a >= 3 : [False, True, True, True]  스칼라와도 된다
  (a > b).sum() = 1  <- 참을 센다
  & | ~ 로 묶는다: [False, False, True, False]
  and/or/not 는 못 쓴다. 텐서를 참거짓 하나로 줄일 수 없어서다
  torch.equal(a, b) 는 통째로 비교해 하나를 준다: False

=== 2. NaN 과 비교 ===
  nan == nan : False
  x = [1.0, nan, 3.0]
  x > 0 : [True, False, True]
  x < 0 : [False, False, False]  <- 둘 다 거짓이다
  x[x > 0]    : [1.0, 3.0]  <- NaN 이 조용히 빠진다
  x[~(x <= 0)]: [1.0, nan, 3.0]  <- 부정하면 남는다
  같은 뜻으로 적은 두 조건이 다른 답을 준다
  NaN 을 찾으려면 isnan 을 써야 한다: [False, True, False]
  isfinite: [True, False, False, False]
  isinf   : [False, False, True, True]

=== 3. 정렬과 극값에서의 NaN ===
  y = [3.0, nan, 1.0, 2.0]
  sort().values : [1.0, 2.0, 3.0, nan]  <- 맨 뒤로 간다
  max()         : nan  <- 번져 나온다
  argmax()      : 1  <- NaN 자리를 가리킨다
  maximum(y, 0) : [3.0, nan, 1.0, 2.0]
  nanmean / nansum 은 건너뛴다: 2.0 4.0
  max 와 argmax 가 어긋나 보이지만 둘 다 NaN 자리를 말하고 있다

=== 4. clamp ===
  x            : [-3.0, -0.5, 0.5, 3.0]
  clamp(-1, 1) : [-1.0, -0.5, 0.5, 1.0]
  clamp(min=0) : [0.0, 0.0, 0.5, 3.0]  = relu
  clip 은 clamp 의 다른 이름: True
  clamp 의 경사: [0.0, 1.0, 1.0, 0.0]
  잘린 자리는 0 이다 — 그 원소는 더 이상 배우지 않는다
  활성값을 clamp 로 묶으면 ReLU 의 죽은 뉴런과 같은 일이 난다

=== 5. 정렬과 고르기 ===
  sort        값: [1.0, 1.0, 3.0, 4.0, 5.0]  자리: [1, 3, 0, 2, 4]
  sort(내림차순): [5.0, 4.0, 3.0, 1.0, 1.0]
  argsort     : [1, 3, 0, 2, 4]  = sort 의 자리만
  정렬한 값은 t[argsort] 와 같은가: True
  topk(2)     값: [5.0, 4.0]  자리: [4, 2]
  topk(2, largest=False): [1.0, 1.0]
  kthvalue(2) 두 번째로 작은 값: 1.0
  2차원은 축을 고른다 sort(dim=0): [2.0, 1.0, 3.0, 4.0]

=== 6. allclose ===
  기본값은 rtol=1e-5, atol=1e-8
  판정식은 |a - b| <= atol + rtol * |b| 다
  차이 1e-04  |a-b| 1.0e-04 vs 한계 1.0e-05~3.0e-05  allclose False
  차이 1e-05  |a-b| 1.0e-05 vs 한계 1.0e-05~3.0e-05  allclose False
  차이 1e-06  |a-b| 1.0e-06 vs 한계 1.0e-05~3.0e-05  allclose True
  rtol 이 상대 오차라 큰 값일수록 너그러워진다
  0 근처를 견줄 때는 atol 을 손으로 올려야 한다
  allclose(0, 1e-7) : False
  allclose(0, 1e-7, atol=1e-6): True
  NaN 은 기본으로 같지 않다: False
  equal_nan=True 로 바꿀 수 있다: True
```

## 2. 비교는 참거짓 텐서를 준다

```
a > b  : [False, True, False, False]  dtype torch.bool
(a > b).sum() = 1
```

결과가 `torch.bool`이므로 `.sum()`으로 세거나 색인에 그대로 쓸 수 있다. 조건을
묶을 때는 **`&`·`|`·`~`**이고 `and`·`or`·`not`은 못 쓴다 — 파이썬 연산자는 피연산자를
참거짓 하나로 줄이려 하는데 텐서는 줄여지지 않는다.

`torch.equal`은 다르다. 원소별이 아니라 **통째로** 비교해 파이썬 `bool` 하나를
준다. 모양이 다르면 그냥 `False`이고 터지지 않는다.

## 3. NaN은 어느 비교에도 참이 아니다

```
nan == nan : False
x > 0 : [True, False, True]
x < 0 : [False, False, False]
```

둘 다 거짓이다. IEEE 754가 그렇게 정해 두었다 — NaN은 "수가 아니"므로 크다고도
작다고도 할 수 없다.

여기서 조용한 사고가 난다.

```
x[x > 0]     : [1.0, 3.0]        <- NaN 이 빠졌다
x[~(x <= 0)] : [1.0, nan, 3.0]   <- NaN 이 남았다
```

**"0보다 큰 것만 고른다"를 두 가지로 적었는데 답이 다르다.** 수학적으로는 같은
조건인데, NaN에서 둘 다 거짓이므로 긍정형은 NaN을 떨구고 부정형은 NaN을 살린다.

그래서 데이터에 NaN이 있을 수 있다면 **거르는 조건에 NaN 처리를 명시적으로**
적어야 한다. `torch.isnan`이 그것을 위해 있다.

## 4. 정렬과 극값에서의 NaN

| | 결과 |
|---|---|
| `sort().values` | `[1.0, 2.0, 3.0, nan]` — 맨 뒤로 |
| `max()` | `nan` — 번져 나온다 |
| `argmax()` | `1` — NaN 자리를 가리킨다 |
| `maximum(y, 0)` | `[3.0, nan, 1.0, 2.0]` — 번진다 |
| `nanmean` / `nansum` | 건너뛴다 |

`max()`가 `nan`을 주고 `argmax()`가 NaN의 자리를 주는 것은 어긋난 것이 아니라
**둘 다 같은 말**을 하고 있는 것이다. 다만 분류에서 `argmax`로 예측 클래스를
뽑는 코드라면, 로짓에 NaN이 하나 생기는 순간 **언제나 그 클래스를 예측하게**
된다. 정확도가 갑자기 한 클래스로 쏠리면 의심할 자리다.

## 5. `clamp`의 경사는 밖에서 0이다

```
x            : [-3.0, -0.5, 0.5, 3.0]
clamp(-1, 1) : [-1.0, -0.5, 0.5, 1.0]
clamp 의 경사: [0.0, 1.0, 1.0, 0.0]
```

잘린 두 자리의 경사가 **0**이다. 당연하다 — 그 구간에서 출력이 상수이므로
미분이 0이다.

그런데 이것이 뜻하는 바가 크다. 활성값을 `clamp`로 묶어 두면 **범위를 벗어난
원소는 더 이상 배우지 않는다.** 벗어난 채로 굳어 있으면 영영 돌아오지 못한다.
`clamp(min=0)`이 곧 ReLU이고, ReLU의 "죽은 뉴런"이 바로 이 현상이다.

`clamp`를 써도 되는 자리와 안 되는 자리가 갈린다.

| | |
|---|---|
| 출력을 범위에 맞출 때 (확률, 픽셀값) | 괜찮다 — 거기서 더 배울 것이 없다 |
| 중간 활성값을 안정시키려 할 때 | 위험하다 — 경사가 끊긴다 |
| 경사 자체를 자를 때 | **쓰면 안 된다** — 연습문제 3 |

## 6. `allclose`의 판정식

```
|a - b| <= atol + rtol * |b|          (기본 rtol=1e-5, atol=1e-8)
```

```
차이 1e-04  한계 1.0e-05~3.0e-05  allclose False
차이 1e-05  한계 1.0e-05~3.0e-05  allclose False
차이 1e-06  한계 1.0e-05~3.0e-05  allclose True
```

한계가 원소마다 다른 것을 보아 두자. `rtol`이 **상대** 오차라 `|b|`가 클수록
너그러워진다. 그래서 1e-5 차이가 값 1에서는 걸리고 값 3에서는 통과할 수 있다.

0 근처에서는 `rtol`이 아무 일도 못 하므로 `atol`만 남는다.

```
allclose(0, 1e-7)              : False
allclose(0, 1e-7, atol=1e-6)   : True
```

기본 `atol`이 1e-8이라 1e-7도 거른다. **0에 가까운 값을 견줄 때는 `atol`을
손으로 올려야** 한다.

NaN은 기본으로 같지 않다고 본다. `equal_nan=True`를 주면 NaN끼리 같다고 친다 —
저장했다 불러온 텐서를 견줄 때 쓸 일이 있다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
점수 텐서에서 **80점 이상인 사람 수**와 **그 사람들의 평균**을 구하라.
`scores = torch.tensor([85., 92., 78., 95., 88., 70.])`.

</div>

??? success "풀이"
    ```python
    scores = torch.tensor([85., 92., 78., 95., 88., 70.])
    mask = scores >= 80
    print("  마스크 :", mask.tolist(), " dtype", mask.dtype)
    print("  몇 명  :", mask.sum().item())
    print("  평균   :", scores[mask].mean().item())
    print("  비율   :", mask.float().mean().item())
    ```

    ```
      마스크 : [True, True, False, True, True, False]  dtype torch.bool
      몇 명  : 4
      평균   : 90.0
      비율   : 0.6666666865348816
    ```

    `mask.sum()`이 개수이고 **`mask.float().mean()`이 비율**이다. 참거짓을 0과 1로
    바꿔 평균내면 비율이 되므로, 정확도를 구하는 코드가 거의 언제나 이 꼴이다 —
    `(pred == target).float().mean()`.

    `.float()`를 빼먹으면 어떻게 되는가. `bool` 텐서에 `.mean()`을 부르면 「축약과
    통계」에서 본 것처럼 `mean(): could not infer output dtype`로 거절한다.
    다행히 터져 주는 쪽이다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
NaN이 섞인 데이터에서 "양수만" 고르려 한다. `x[x > 0]`과 `x[~(x <= 0)]`이 다른
답을 준다. 어느 쪽이 맞는가? **NaN을 명시적으로 다루는** 세 번째 방법을 적고,
평균을 구하는 세 가지 길을 견주어라.

</div>

??? success "풀이"
    ```python
    x = torch.tensor([1.0, float("nan"), 3.0, -2.0])
    print("  x[x > 0]      :", x[x > 0].tolist())
    print("  x[~(x <= 0)]  :", x[~(x <= 0)].tolist())
    print("  x[(x > 0) & ~torch.isnan(x)] :",
          x[(x > 0) & ~torch.isnan(x)].tolist())
    print("  평균 세 가지")
    print("    x.mean()           :", x.mean().item())
    print("    x.nanmean()        :", x.nanmean().item())
    print("    x[~torch.isnan(x)].mean():", x[~torch.isnan(x)].mean().item())
    ```

    ```
      x[x > 0]      : [1.0, 3.0]
      x[~(x <= 0)]  : [1.0, nan, 3.0]
      x[(x > 0) & ~torch.isnan(x)] : [1.0, 3.0]
      평균 세 가지
        x.mean()           : nan
        x.nanmean()        : 0.6666666865348816
        x[~torch.isnan(x)].mean(): 0.6666666865348816
    ```

    **둘 중 어느 쪽도 "맞다"고 할 수 없다.** 묻지 않은 것을 각자 다르게
    정했을 뿐이다. `x > 0`은 NaN을 버리고 `~(x <= 0)`은 남긴다. 어느 쪽을 원하는지
    코드에 적혀 있지 않은 것이 문제다.

    세 번째 꼴 `(x > 0) & ~torch.isnan(x)`은 **뜻이 적혀 있다.** 길지만 읽는
    사람이 의도를 묻지 않아도 된다.

    평균 쪽도 같다. `mean()`은 NaN 하나에 전부 오염되고, `nanmean()`은 건너뛴다.
    `nanmean`이 "NaN을 빼고 센다"는 것을 **이름으로** 말해 주므로, 손으로
    마스킹하는 것보다 낫다. 다만 전부 NaN인 줄에서는 `nanmean`도 NaN을 준다 —
    나눌 것이 없기 때문이다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
경사가 너무 클 때 `grad.clamp_(-1, 1)`로 자르면 안 될까? 자르기 전후의 **방향**을
견주고, `clip_grad_norm_`과 무엇이 다른지 보여라.

</div>

??? success "풀이"
    ```python
    def cos(a, b):
        return (a @ b / (a.norm() * b.norm())).item()

    g = torch.tensor([10.0, 0.1, -0.05, 3.0])
    byclamp = g.clamp(-1, 1)
    bynorm = g / g.norm()
    print("  원래 경사  :", [round(v, 4) for v in g.tolist()])
    print("  clamp(-1,1):", [round(v, 4) for v in byclamp.tolist()])
    print("  노름 1 로  :", [round(v, 4) for v in bynorm.tolist()])
    print("  원래 방향과의 코사인  clamp:", round(cos(g, byclamp), 4),
          "  노름 자르기:", round(cos(g, bynorm), 4))
    ```

    ```
      원래 경사  : [10.0, 0.1, -0.05, 3.0]
      clamp(-1,1): [1.0, 0.1, -0.05, 1.0]
      노름 1 로  : [0.9578, 0.0096, -0.0048, 0.2873]
      원래 방향과의 코사인  clamp: 0.8785   노름 자르기: 1.0
    ```

    **`clamp`는 방향을 바꾼다.** 코사인이 0.8785이니 원래 경사와 28도쯤
    틀어졌다. 10과 3이 둘 다 1로 잘리면서 **원래 3배 차이였던 두 성분이 같아졌기**
    때문이다. 크기만 줄이려 했는데 "어느 쪽으로 갈지"까지 바꿔 버렸다.

    노름으로 나누면 모든 성분이 같은 비율로 줄어 **방향이 정확히 보존된다**
    (코사인 1.0). 이것이 `torch.nn.utils.clip_grad_norm_`이 하는 일이다 — 전체
    노름이 한계를 넘으면 그 비율만큼 전부를 줄인다.

    실제 모델에서도 같다.

    ```python
    m = nn.Linear(10, 10)
    m(torch.randn(4, 10) * 50).pow(2).sum().backward()
    before = torch.cat([p.grad.flatten() for p in m.parameters()]).clone()
    torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
    after = torch.cat([p.grad.flatten() for p in m.parameters()])
    # 노름 61859.09 -> 1.0000   방향 코사인 1.000000
    ```

    노름이 6만에서 1로 줄었는데 방향 코사인은 1.000000이다.

    5절의 표에 "경사 자체를 자를 때는 `clamp`를 쓰면 안 된다"고 적은 까닭이
    이것이다. 경사 하강은 **방향을 따라가는** 방법이고, 크기는 러닝레이트가
    맡는다. 크기를 조절하려다 방향을 흔들면 러닝레이트를 아무리 맞춰도 소용이
    없다.
