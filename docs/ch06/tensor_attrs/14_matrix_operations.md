# 행렬 연산 — 어느 것을 쓰고 무엇을 믿을 수 있는가

행렬곱 하나를 적는 길이 여럿이다. `*`, `@`, `mm`, `bmm`, `matmul`, `einsum`.
선형대수 함수도 스무 가지가 넘는다. 그런데 **쓰다가 틀리는 자리**는 몇 군데로
좁혀진다.

- `*`는 원소별, `@`는 행렬곱이다. **정사각에서는 둘 다 터지지 않고** 다른 답을 준다
- `matmul`은 브로드캐스팅하고 `mm`·`bmm`은 모양을 꼭 맞춰야 한다
- **역행렬을 구해 곱하지 말고 `solve`를 쓴다**
- float32에서 믿을 수 있는 자릿수는 **조건수가 정한다**

## 1. 코드

```python
"""행렬 연산 — 어느 것을 쓰고 무엇을 믿을 수 있는가.

행렬곱 하나를 적는 길이 여럿이고(`*`, `@`, `mm`, `bmm`, `matmul`,
`einsum`) 선형대수 함수도 여럿이다. 외울 것은 세 가지다.

- `*` 는 원소별, `@` 는 행렬곱이다. 둘 다 터지지 않고 다른 답을 줄 수 있다
- `matmul` 은 브로드캐스팅하고 `mm`·`bmm` 은 모양을 꼭 맞춰야 한다
- **역행렬을 구해 곱하지 말고 `solve` 를 쓴다.** 그리고 float32 에서
  믿을 수 있는 자릿수는 **조건수가 정한다**

실행: python 14_matrix_operations.py
"""
import math
import time

import torch
import torch.nn.functional as F


def header(title):
    print(f"\n=== {title} ===")


def spread(fn, rep=7):
    ts = []
    for _ in range(rep):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return min(ts) * 1e3, max(ts) * 1e3


# ========================================================================
# 1. * 와 @
# ========================================================================

def elementwise_vs_matmul():
    header("1. * 와 @")
    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    b = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
    print("  a * b (원소별):", (a * b).flatten().tolist())
    print("  a @ b (행렬곱):", (a @ b).flatten().tolist())
    print("  정사각이라 모양이 같아 터지지 않는다. 값만 다르다")
    print("  a.mul(b) 와 a.matmul(b) 로도 적는다")

    v = torch.tensor([1.0, 2.0])
    print("  행렬 @ 벡터:", (a @ v).tolist(), " 모양", tuple((a @ v).shape))
    print("  벡터 @ 벡터 (내적):", torch.dot(v, v).item(), "=", (v @ v).item())
    print("  바깥곱 outer:", torch.outer(v, v).flatten().tolist())


# ========================================================================
# 2. mm / bmm / matmul 이 받는 모양
# ========================================================================

def which_matmul():
    header("2. mm / bmm / matmul")
    A2, B2 = torch.randn(3, 4), torch.randn(4, 5)
    A3, B3 = torch.randn(2, 3, 4), torch.randn(2, 4, 5)
    cases = (("mm    (2D,2D)", torch.mm, (A2, B2)),
             ("mm    (3D,3D)", torch.mm, (A3, B3)),
             ("bmm   (3D,3D)", torch.bmm, (A3, B3)),
             ("bmm   (2D,2D)", torch.bmm, (A2, B2)),
             ("matmul(2D,2D)", torch.matmul, (A2, B2)),
             ("matmul(3D,3D)", torch.matmul, (A3, B3)),
             ("matmul(3D,2D)", torch.matmul, (A3, B2)))
    for name, fn, args in cases:
        try:
            print(f"  {name} -> {tuple(fn(*args).shape)}")
        except RuntimeError as e:
            print(f"  {name} -> {str(e).splitlines()[0][:44]}")
    print("  @ 는 matmul 이다. 브로드캐스팅까지 해 주므로 보통 이것만 쓰면 된다")


# ========================================================================
# 3. 어텐션에 1/sqrt(d) 가 붙는 까닭
# ========================================================================

def why_scale():
    header("3. 어텐션의 1/sqrt(d)")
    torch.manual_seed(0)
    for d in (8, 64, 512):
        Q, K = torch.randn(1, 10, d), torch.randn(1, 10, d)
        s = Q @ K.transpose(-2, -1)
        for label, sc in (("조정 없이  ", s), ("1/sqrt(d) 로", s / math.sqrt(d))):
            p = F.softmax(sc, dim=-1)
            ent = -(p * p.clamp_min(1e-12).log()).sum(-1).mean()
            print(f"  d={d:4d} {label} 점수 표준편차 {sc.std():6.3f}  "
                  f"최대 확률 {p.max(dim=-1).values.mean():.4f}  "
                  f"엔트로피 {ent:.4f}")
    print("  조정 없이 d 가 크면 softmax 가 한 자리에 쏠린다")
    print("  Q·K 는 d 개를 더한 값이라 표준편차가 sqrt(d) 로 자란다")
    print("  그래서 sqrt(d) 로 나누면 d 와 무관하게 1 쯤에서 머문다")


# ========================================================================
# 4. 역행렬을 구해 곱하지 않는다
# ========================================================================

def solve_not_inverse():
    header("4. inv 로 풀기 vs solve")
    torch.manual_seed(0)
    n = 500
    A = torch.randn(n, n)
    A = A @ A.t() + n * torch.eye(n)          # 대칭이고 잘 조건지어진 것
    b = torch.randn(n, 1)
    exact = torch.linalg.solve(A.double(), b.double())
    for name, x in (("inv(A) @ b ", torch.linalg.inv(A) @ b),
                    ("solve(A, b)", torch.linalg.solve(A, b))):
        err = (x.double() - exact).abs().max().item()
        print(f"  {name}  참값과의 차 {err:.3e}")
    t1 = spread(lambda: torch.linalg.inv(A) @ b)
    t2 = spread(lambda: torch.linalg.solve(A, b))
    print(f"  inv(A) @ b   {t1[0]:6.2f} ~ {t1[1]:6.2f} ms")
    print(f"  solve(A, b)  {t2[0]:6.2f} ~ {t2[1]:6.2f} ms")
    print("  잘 조건지어진 행렬에서는 정확도가 비슷하고 solve 가 빠르다")
    print("  나쁘게 조건지어지면 정확도도 갈린다 — 다음 절")


# ========================================================================
# 5. 조건수가 믿을 자릿수를 정한다
# ========================================================================

def conditioning():
    header("5. 조건수")
    torch.manual_seed(0)
    Q, _ = torch.linalg.qr(torch.randn(50, 50))
    for c in (1e1, 1e4, 1e7):
        d = torch.logspace(0, math.log10(c), 50)
        M = Q @ torch.diag(d) @ Q.t()
        err = (M @ torch.linalg.inv(M) - torch.eye(50)).abs().max().item()
        print(f"  조건수 {c:7.0e}  A @ inv(A) 가 I 에서 벗어난 최대 {err:.3e}")
    print("  float32 는 유효숫자가 약 7 자리다")
    print("  조건수가 10^k 면 그중 k 자리를 잃는다고 보면 맞는다")
    print("  조건수 10^7 이면 남는 자리가 없어 4% 가까이 어긋난다")


# ========================================================================
# 6. einsum — 하나로 다 적는다
# ========================================================================

def einsum_demo():
    header("6. einsum")
    torch.manual_seed(0)
    A, B = torch.randn(3, 4), torch.randn(4, 5)
    v, w = torch.randn(3), torch.randn(3)
    T = torch.randn(2, 3, 4)
    U = torch.randn(2, 4, 5)
    checks = (
        ("내적      i,i->  ", torch.einsum("i,i->", v, w), torch.dot(v, w)),
        ("바깥곱    i,j->ij", torch.einsum("i,j->ij", v, v), torch.outer(v, v)),
        ("행렬곱    ik,kj->ij", torch.einsum("ik,kj->ij", A, B), A @ B),
        ("전치      ij->ji", torch.einsum("ij->ji", A), A.t()),
        ("대각합    ii->", torch.einsum("ii->", torch.eye(4)), torch.tensor(4.0)),
        ("배치곱    bik,bkj->bij", torch.einsum("bik,bkj->bij", T, U), T @ U),
    )
    for name, got, want in checks:
        print(f"  {name:22s} 같은가: {torch.allclose(got, want, atol=1e-5)}")
    print("  첨자에 없는 글자는 더해서 없앤다 — 그것이 규칙 전부다")


# ========================================================================
# 7. 분해와 나머지
# ========================================================================

def decompositions():
    header("7. 분해들")
    torch.manual_seed(0)
    A = torch.randn(4, 3)
    U, S, Vh = torch.linalg.svd(A, full_matrices=False)
    print("  SVD   A = U S Vh 로 되돌아오는가:",
          torch.allclose(U @ torch.diag(S) @ Vh, A, atol=1e-5))
    print("    특이값:", [round(v, 4) for v in S.tolist()])
    Q, R = torch.linalg.qr(A)
    print("  QR    A = Q R 로 되돌아오는가:",
          torch.allclose(Q @ R, A, atol=1e-5))
    P = torch.randn(3, 3); P = P @ P.t() + 3 * torch.eye(3)
    L = torch.linalg.cholesky(P)
    print("  촐레스키 L @ L.T 로 되돌아오는가:",
          torch.allclose(L @ L.t(), P, atol=1e-4))
    print("  rank:", torch.linalg.matrix_rank(A).item(),
          " det(3x3):", round(torch.linalg.det(P).item(), 4),
          " trace:", round(torch.trace(P).item(), 4))
    print("  trace 는 대각의 합이다:",
          round(torch.diag(P).sum().item(), 4))


def main():
    elementwise_vs_matmul()
    which_matmul()
    why_scale()
    solve_not_inverse()
    conditioning()
    einsum_demo()
    decompositions()


if __name__ == "__main__":
    main()
```

**출력:**

```
=== 1. * 와 @ ===
  a * b (원소별): [5.0, 12.0, 21.0, 32.0]
  a @ b (행렬곱): [19.0, 22.0, 43.0, 50.0]
  정사각이라 모양이 같아 터지지 않는다. 값만 다르다
  a.mul(b) 와 a.matmul(b) 로도 적는다
  행렬 @ 벡터: [5.0, 11.0]  모양 (2,)
  벡터 @ 벡터 (내적): 5.0 = 5.0
  바깥곱 outer: [1.0, 2.0, 2.0, 4.0]

=== 2. mm / bmm / matmul ===
  mm    (2D,2D) -> (3, 5)
  mm    (3D,3D) -> self must be a matrix
  bmm   (3D,3D) -> (2, 3, 5)
  bmm   (2D,2D) -> batch1 must be a 3D tensor
  matmul(2D,2D) -> (3, 5)
  matmul(3D,3D) -> (2, 3, 5)
  matmul(3D,2D) -> (2, 3, 5)
  @ 는 matmul 이다. 브로드캐스팅까지 해 주므로 보통 이것만 쓰면 된다

=== 3. 어텐션의 1/sqrt(d) ===
  d=   8 조정 없이   점수 표준편차  2.462  최대 확률 0.5630  엔트로피 1.2362
  d=   8 1/sqrt(d) 로 점수 표준편차  0.870  최대 확률 0.2694  엔트로피 2.0312
  d=  64 조정 없이   점수 표준편차  8.453  최대 확률 0.9043  엔트로피 0.2353
  d=  64 1/sqrt(d) 로 점수 표준편차  1.057  최대 확률 0.3526  엔트로피 1.8558
  d= 512 조정 없이   점수 표준편차 19.955  최대 확률 0.9722  엔트로피 0.0651
  d= 512 1/sqrt(d) 로 점수 표준편차  0.882  최대 확률 0.2584  엔트로피 2.0592
  조정 없이 d 가 크면 softmax 가 한 자리에 쏠린다
  Q·K 는 d 개를 더한 값이라 표준편차가 sqrt(d) 로 자란다
  그래서 sqrt(d) 로 나누면 d 와 무관하게 1 쯤에서 머문다

=== 4. inv 로 풀기 vs solve ===
  inv(A) @ b   참값과의 차 3.862e-09
  solve(A, b)  참값과의 차 3.754e-09
  inv(A) @ b     0.86 ~   0.89 ms
  solve(A, b)    0.45 ~   0.46 ms
  잘 조건지어진 행렬에서는 정확도가 비슷하고 solve 가 빠르다
  나쁘게 조건지어지면 정확도도 갈린다 — 다음 절

=== 5. 조건수 ===
  조건수   1e+01  A @ inv(A) 가 I 에서 벗어난 최대 1.073e-06
  조건수   1e+04  A @ inv(A) 가 I 에서 벗어난 최대 6.044e-05
  조건수   1e+07  A @ inv(A) 가 I 에서 벗어난 최대 3.854e-02
  float32 는 유효숫자가 약 7 자리다
  조건수가 10^k 면 그중 k 자리를 잃는다고 보면 맞는다
  조건수 10^7 이면 남는 자리가 없어 4% 가까이 어긋난다

=== 6. einsum ===
  내적      i,i->          같은가: True
  바깥곱    i,j->ij         같은가: True
  행렬곱    ik,kj->ij       같은가: True
  전치      ij->ji         같은가: True
  대각합    ii->            같은가: True
  배치곱    bik,bkj->bij    같은가: True
  첨자에 없는 글자는 더해서 없앤다 — 그것이 규칙 전부다

=== 7. 분해들 ===
  SVD   A = U S Vh 로 되돌아오는가: True
    특이값: [3.2245, 1.4537, 0.2943]
  QR    A = Q R 로 되돌아오는가: True
  촐레스키 L @ L.T 로 되돌아오는가: True
  rank: 3  det(3x3): 115.589  trace: 15.7779
  trace 는 대각의 합이다: 15.7779
```

## 2. `*`와 `@`는 정사각에서 둘 다 통과한다

```
a * b (원소별): [5.0, 12.0, 21.0, 32.0]
a @ b (행렬곱): [19.0, 22.0, 43.0, 50.0]
```

$2 \times 2$ 둘이라 **모양이 같으므로 어느 쪽도 터지지 않는다.** 값만 다르다.
이것이 오타 하나가 조용히 지나가는 자리다 — 직사각형이었다면 `*`가 바로 터졌을
것이다.

벡터가 섞이면 `@`가 알아서 맞춘다. `행렬 @ 벡터`는 행렬-벡터 곱이고
`벡터 @ 벡터`는 내적이다. 바깥곱을 원하면 `torch.outer`를 따로 불러야 한다.

## 3. `mm` / `bmm` / `matmul`

| | 2D, 2D | 3D, 3D | 3D, 2D |
|---|---|---|---|
| `mm` | `(3,5)` | `self must be a matrix` | — |
| `bmm` | `batch1 must be a 3D tensor` | `(2,3,5)` | — |
| `matmul` (= `@`) | `(3,5)` | `(2,3,5)` | **`(2,3,5)`** |

`matmul`만 **브로드캐스팅**한다. `(2,3,4) @ (4,5)`에서 두 번째 것을 배치마다 다시
쓴다. 그래서 보통은 `@`만 쓰면 되고, `mm`·`bmm`은 "여기서는 반드시 2차원(또는
3차원)이어야 한다"고 못 박고 싶을 때 쓴다.

## 4. 어텐션의 $1/\sqrt{d}$

| $d$ | | 점수 표준편차 | 최대 확률 | 엔트로피 |
|---|---|---|---|---|
| 8 | 조정 없이 | 2.462 | 0.5630 | 1.2362 |
| 8 | $1/\sqrt{d}$ | 0.870 | 0.2694 | 2.0312 |
| 64 | 조정 없이 | 8.453 | 0.9043 | 0.2353 |
| 64 | $1/\sqrt{d}$ | 1.057 | 0.3526 | 1.8558 |
| 512 | 조정 없이 | **19.955** | **0.9722** | **0.0651** |
| 512 | $1/\sqrt{d}$ | 0.882 | 0.2584 | 2.0592 |

$d = 512$에서 조정을 빼면 한 자리가 확률 0.97을 가져가고 엔트로피가 0.065로
주저앉는다. **어텐션이 사실상 한 곳만 보게 된다.**

까닭은 간단한 셈이다. $q \cdot k = \sum_{i=1}^{d} q_i k_i$는 평균 0, 분산 1짜리
항 $d$개의 합이므로 표준편차가 $\sqrt{d}$로 자란다. 측정값이 그대로다 —
$\sqrt{512} \approx 22.6$이고 잰 값이 19.955다.

소프트맥스는 입력 차이가 몇만 되어도 포화하므로, 표준편차 20짜리 점수를 넣으면
가장 큰 것만 남는다. $\sqrt{d}$로 나누면 $d$와 무관하게 1 근처에 머문다 —
표의 0.870 / 1.057 / 0.882가 그것이다.

> 포화가 왜 나쁜가. 소프트맥스가 한 자리에 쏠리면 **경사도 거의 0**이 된다.
> 「제자리 연산」에서 본 것처럼 $\sigma'$가 출력으로 적히기 때문이다. 학습 초반에
> 어텐션이 굳어 버리면 거기서 못 빠져나온다.

## 5. 역행렬을 구해 곱하지 않는다

```
inv(A) @ b    참값과의 차 3.862e-09     0.86 ~ 0.88 ms
solve(A, b)   참값과의 차 3.754e-09     0.45 ~ 0.46 ms
```

잘 조건지어진 $500 \times 500$에서는 **정확도가 거의 같고 `solve`가 두 배
빠르다.** 역행렬을 통째로 구하는 것은 $n$개의 연립방정식을 푸는 일이라, 하나만
풀면 될 때 $n$배의 일을 하는 셈이다.

정확도까지 갈리는 것은 조건이 나빠질 때다. 연습문제 3에서 잰다.

## 6. 조건수가 믿을 자릿수를 정한다

```
조건수 1e+01   A @ inv(A) 가 I 에서 벗어난 최대 1.073e-06
조건수 1e+04                                   6.044e-05
조건수 1e+07                                   3.854e-02
```

**float32의 유효숫자는 약 7자리**($2^{-24} \approx 6 \times 10^{-8}$)다. 조건수가
$10^k$이면 그중 $k$자리를 잃는다. 그러니 조건수 $10^7$이면 남는 자리가 없고,
실제로 $A A^{-1}$이 단위행렬에서 **4% 가까이** 벗어난다.

이것은 파이토치의 흠이 아니라 **그 행렬로 할 수 있는 일의 한계**다. 고치려면
자료형을 올리거나(연습문제 3), 애초에 조건을 좋게 만들어야 한다 — 정규화를
넣거나, $A^\top A$ 대신 QR·SVD로 푸는 식이다.

> 최소제곱을 $(X^\top X)^{-1} X^\top y$로 푸는 코드를 자주 보는데,
> $X^\top X$의 조건수는 $X$의 조건수의 **제곱**이다. 잃는 자릿수가 두 배가 된다.
> `torch.linalg.lstsq`는 그렇게 풀지 않는다.

## 7. `einsum`은 규칙이 하나다

```
내적      i,i->          True
바깥곱    i,j->ij        True
행렬곱    ik,kj->ij      True
전치      ij->ji         True
대각합    ii->           True
배치곱    bik,bkj->bij   True
```

여섯 가지가 전부 같은 표기로 적히고, 전부 전용 함수와 같은 답을 준다.

규칙은 하나다. **화살표 오른쪽에 없는 글자는 더해서 없앤다.** `ik,kj->ij`에서
`k`가 사라졌으니 `k`에 대해 더한다 — 그것이 행렬곱이다. `ii->`는 같은 글자를 두
번 써서 대각을 고른 다음 더한 것이니 대각합이다.

축이 셋 넘게 얽히는 곳(어텐션, 텐서 수축)에서는 `permute`와 `reshape`를 늘어놓는
것보다 `einsum` 한 줄이 읽기 쉽다. 다만 **빠르다는 보장은 없다** — 전용 커널이
있는 연산은 그쪽이 빠른 경우가 많다.

## 8. 분해들

```
SVD   A = U S Vh 로 되돌아오는가: True    특이값: [3.2245, 1.4537, 0.2943]
QR    A = Q R 로 되돌아오는가: True
촐레스키 L @ L.T 로 되돌아오는가: True
```

세 분해 모두 되돌려 원래 행렬이 나오는지 확인했다. **분해를 쓸 때 가장 먼저
할 일이 이 확인**이다 — 인자 순서나 `full_matrices` 같은 옵션을 잘못 주면 모양은
맞는데 값이 다른 것이 나온다.

특이값은 조건수와 바로 이어진다. 조건수가 가장 큰 특이값 나누기 가장 작은
특이값이므로, 위 행렬은 $3.2245 / 0.2943 \approx 11$로 아주 좋다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
`a`가 `(2,3)`, `b`가 `(3,2)`일 때 `a * b`와 `a @ b` 중 무엇이 되는가? `a * a`와
`a @ a`는 어떤가? 넷을 **먼저 적고** 확인하라.

</div>

??? success "풀이"
    ```python
    a, b = torch.ones(2, 3), torch.ones(3, 2)
    for name, f in (("a * b", lambda: a * b), ("a @ b", lambda: a @ b),
                    ("a * a", lambda: a * a), ("a @ a", lambda: a @ a)):
        try:
            print(f"  {name} -> {tuple(f().shape)}")
        except RuntimeError as e:
            print(f"  {name} -> {str(e).splitlines()[0][:60]}")
    ```

    ```
      a * b -> The size of tensor a (3) must match the size of tensor b (2)
      a @ b -> (2, 2)
      a * a -> (2, 3)
      a @ a -> mat1 and mat2 shapes cannot be multiplied (2x3 and 2x3)
    ```

    **정확히 엇갈린다.** `*`가 되는 자리에서 `@`가 안 되고, `@`가 되는 자리에서
    `*`가 안 된다.

    `*`는 모양이 같아야 하고(또는 브로드캐스팅되어야 하고), `@`는 안쪽 차원이
    맞물려야 한다. 요구가 반대이므로 직사각형에서는 **오타가 반드시 터진다.**

    그래서 2절의 정사각 경우가 특별한 것이다. $n \times n$에서는 둘 다 통과하고
    값만 달라지므로, 오타가 끝까지 살아남는다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
멀티헤드 어텐션의 점수를 `einsum`으로 적어라. `Q`와 `K`가
`(배치, 헤드, 길이, 차원)`일 때 점수는 `(배치, 헤드, 길이, 길이)`다.
`@`로 적은 것과 같은지 확인하라.

</div>

??? success "풀이"
    ```python
    torch.manual_seed(0)
    B, H, L, D = 2, 4, 6, 16
    Q, K = torch.randn(B, H, L, D), torch.randn(B, H, L, D)

    by_matmul = Q @ K.transpose(-2, -1)
    by_einsum = torch.einsum("bhid,bhjd->bhij", Q, K)
    print("  모양:", tuple(by_einsum.shape))
    print("  같은가:", torch.allclose(by_matmul, by_einsum, atol=1e-5))
    ```

    ```
      모양: (2, 4, 6, 6)
      같은가: True
    ```

    첨자를 읽으면 식이 그대로 보인다. `bhid,bhjd->bhij`에서 **`d`가 오른쪽에
    없으므로 `d`에 대해 더한다.** 그것이 $q_i \cdot k_j$다. `b`와 `h`는 양쪽에
    그대로 있으니 배치 축으로 남는다.

    `@`로 적으려면 `K.transpose(-2, -1)`을 기억해야 하는데, `einsum`은 `i`와 `j`를
    다른 글자로 쓰는 것만으로 전치가 표현된다. **축이 넷인 곳에서 `einsum`이
    읽기 쉬운 까닭**이 이것이다.

    다만 「형상, 자료형, 장치」에서 본 `.T`와 `.mT` 혼동은 `einsum`에서는 아예
    일어나지 않는다. 축을 번호가 아니라 이름으로 가리키기 때문이다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
6절은 조건수가 $10^7$이면 float32가 4%쯤 틀린다고 했다. $10^{10}$이면 어떻게
되는가? 같은 문제를 float64로 풀면 얼마나 나아지는가? 둘을 재고, **언제 자료형을
올려야 하는지** 적어라.

</div>

??? success "풀이"
    ```python
    import math

    torch.manual_seed(0)
    Q, _ = torch.linalg.qr(torch.randn(50, 50, dtype=torch.float64))
    for c in (1e4, 1e7, 1e10):
        d = torch.logspace(0, math.log10(c), 50, dtype=torch.float64)
        M = Q @ torch.diag(d) @ Q.t()
        b = torch.randn(50, 1, dtype=torch.float64)
        exact = torch.linalg.solve(M, b)
        x32 = torch.linalg.solve(M.float(), b.float()).double()
        r32 = ((x32 - exact).norm() / exact.norm()).item()
        print(f"  조건수 {c:7.0e}  float32 상대오차 {r32:.3e}")
    ```

    ```
      조건수   1e+04  float32 상대오차 3.483e-05
      조건수   1e+07  float32 상대오차 1.560e-02
      조건수   1e+10  float32 상대오차 1.051e+00
    ```

    조건수 $10^{10}$에서 상대오차가 **1.051**이다. 100%가 넘는다 — 답이 참값과
    같은 크기만큼 틀렸다는 뜻이고, **아무 뜻도 없는 수**가 나온 것이다.

    규칙이 그대로 들어맞는다. float32의 7자리에서 조건수의 지수만큼 빼면
    $7 - 4 = 3$자리(오차 $10^{-5}$), $7 - 7 = 0$자리(오차 $10^{-2}$),
    $7 - 10 < 0$자리(오차 $10^{0}$)다.

    float64는 유효숫자가 약 16자리이므로 조건수 $10^{10}$에서도 6자리가 남는다.
    같은 문제를 float64로 풀면 답이 멀쩡하다.

    **언제 올려야 하는가.** 딥러닝의 본체(합성곱, 행렬곱, 경사)는 올릴 일이 거의
    없다. 조건수가 그렇게 나쁘지 않고, 경사 하강 자체가 근사이므로 7자리면
    넉넉하다. 「혼합 정밀도」에서 본 것처럼 오히려 내리는 쪽이 이득이다.

    올려야 하는 자리는 **선형대수를 직접 푸는 곳**이다. 공분산 행렬을 뒤집거나,
    최소제곱을 정규방정식으로 풀거나, 가우스 과정의 커널 행렬을 촐레스키
    분해하는 곳 — 모두 조건수가 쉽게 $10^8$을 넘는다. 그런 곳은 float64로
    풀고 결과만 float32로 내리는 것이 보통이다.
