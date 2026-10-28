# 벡터 출력의 역전파 — `backward(v)`는 야코비를 주지 않는다

출력이 홑값이면 `backward()`로 끝난다. 모양이 있으면 거절한다.

```
grad can be implicitly created only for scalar outputs
```

까닭은 **미분의 결과가 야코비 행렬**이어서 `.grad`에 담을 수 없기 때문이다.
`.grad`는 언제나 그 텐서와 같은 모양이다.

그래서 `backward(v)`는 야코비를 만들지 않고 **$v^\top J$ 하나**만 계산한다.
벡터-야코비 곱(VJP)이라 부르는 것이 이것이고, 역전파 전체가 이 연산의
연쇄다.

## 1. 코드

```python
"""backward(v) 는 야코비를 주지 않는다 — v^T J 를 준다.

출력이 홑값이면 `backward()` 로 끝난다. 모양이 있으면 거절한다.

    grad can be implicitly created only for scalar outputs

까닭은 미분의 결과가 **야코비 행렬**이어서 `.grad` 에 담을 수 없기
때문이다. `.grad` 는 언제나 그 텐서와 같은 모양이다.

그래서 `backward(v)` 는 야코비를 만들지 않고 **v 와의 곱 하나**만
계산한다. 야코비 전체가 필요하면 v 를 바꿔 가며 여러 번 불러야 하고,
그 값을 이 쪽에서 잰다.

실행: 이 쪽의 코드를 파일로 갈무리해 돌린다
"""
import time

import torch


def header(title):
    print(f"\n=== {title} ===")


# ========================================================================
# 1. 홑값이 아니면 거절한다
# ========================================================================

def refuses():
    header("1. 거절")
    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    y = x ** 2                      # 모양이 있는 출력
    try:
        y.backward()
    except RuntimeError as e:
        print("  y.backward():", str(e).splitlines()[0])
    print("  y 가 (3,) 이고 x 가 (3,) 이면 야코비는 3x3 이다")
    print("  x.grad 는 (3,) 이므로 거기 담을 수 없다")


# ========================================================================
# 2. v 는 '출력을 얼마씩 중히 볼 것인가' 다
# ========================================================================

def what_v_means():
    header("2. v 의 뜻")
    for v in ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 1.0],
              [10.0, 1.0, 0.1]):
        x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        (x ** 2).backward(torch.tensor(v))
        print(f"  v={str(v):22s} x.grad={x.grad.tolist()}")
    print("  y_i = x_i^2 이라 야코비가 대각이고 대각 원소가 2x_i 다")
    print("  그래서 v^T J = v * 2x 가 그대로 보인다")
    print("  v 를 전부 1 로 주는 것은 y.sum().backward() 와 같다")
    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    (x ** 2).sum().backward()
    print("  확인:", x.grad.tolist())


# ========================================================================
# 3. 야코비 전체를 얻으려면 m 번 불러야 한다
# ========================================================================

def building_jacobian():
    header("3. 야코비 모으기")
    torch.manual_seed(0)
    A = torch.randn(4, 3)
    x = torch.randn(3, requires_grad=True)
    y = A @ x                        # 야코비는 A 자신이다

    rows = []
    for i in range(4):
        v = torch.zeros(4)
        v[i] = 1.0
        x.grad = None
        y.backward(v, retain_graph=True)
        rows.append(x.grad.clone())
    J = torch.stack(rows)
    print("  단위벡터 네 개로 네 줄을 모았다")
    print("  J 가 A 와 같은가:", torch.allclose(J, A))
    print("  출력이 m 개면 backward 를 m 번 불러야 한다")
    print("  retain_graph=True 가 필요하고, 매번 x.grad 를 비워야 한다")


# ========================================================================
# 4. 값이 얼마나 드는가
# ========================================================================

def the_cost():
    header("4. VJP 한 번 vs 야코비 전체")
    torch.manual_seed(0)
    n, m = 512, 512
    A = torch.randn(m, n)

    def one_vjp():
        x = torch.randn(n, requires_grad=True)
        (A @ x).backward(torch.randn(m))

    def full_jacobian():
        x = torch.randn(n, requires_grad=True)
        y = A @ x
        for i in range(m):
            v = torch.zeros(m)
            v[i] = 1.0
            x.grad = None
            y.backward(v, retain_graph=True)

    got = {}
    for label, fn in (("VJP 한 번", one_vjp), ("야코비 전체", full_jacobian)):
        ts = []
        for _ in range(3):
            t0 = time.perf_counter()
            fn()
            ts.append(time.perf_counter() - t0)
        got[label] = min(ts) * 1e3
        print(f"  {label:12s} {got[label]:8.2f} ms")
    ratio = got["야코비 전체"] / got["VJP 한 번"]
    print(f"  {ratio:.0f} 배 든다. 출력 개수 {m} 보다는 작다")
    print("  backward 를 m 번 부르는 값이 m 배까지는 안 가기 때문이다")
    print("  그래도 자릿수가 같다. 로스를 홑값으로 만드는 까닭이 이것이다")


# ========================================================================
# 5. 로스가 홑값인 까닭
# ========================================================================

def why_scalar_loss():
    header("5. 로스는 왜 홑값인가")
    torch.manual_seed(0)
    W = torch.randn(3, 4, requires_grad=True)
    x = torch.randn(8, 4)
    out = x @ W.t()                   # (8, 3)
    print("  출력이 (8,3) 이면 야코비는 (8,3) x (3,4) 짜리다")
    print("  원소 수:", 8 * 3 * 3 * 4)
    loss = out.pow(2).mean()          # 홑값으로 줄인다
    loss.backward()
    print("  홑값으로 줄이면 backward 한 번으로 끝난다")
    print("  W.grad 모양:", tuple(W.grad.shape), "= W 와 같다")
    print("  학습에 필요한 것은 야코비가 아니라 이 하나뿐이다")


def main():
    refuses()
    what_v_means()
    building_jacobian()
    the_cost()
    why_scalar_loss()


if __name__ == "__main__":
    main()
```

**출력:**

```
=== 1. 거절 ===
  y.backward(): grad can be implicitly created only for scalar outputs
  y 가 (3,) 이고 x 가 (3,) 이면 야코비는 3x3 이다
  x.grad 는 (3,) 이므로 거기 담을 수 없다

=== 2. v 의 뜻 ===
  v=[1.0, 0.0, 0.0]        x.grad=[2.0, 0.0, 0.0]
  v=[0.0, 1.0, 0.0]        x.grad=[0.0, 4.0, 0.0]
  v=[1.0, 1.0, 1.0]        x.grad=[2.0, 4.0, 6.0]
  v=[10.0, 1.0, 0.1]       x.grad=[20.0, 4.0, 0.6000000238418579]
  y_i = x_i^2 이라 야코비가 대각이고 대각 원소가 2x_i 다
  그래서 v^T J = v * 2x 가 그대로 보인다
  v 를 전부 1 로 주는 것은 y.sum().backward() 와 같다
  확인: [2.0, 4.0, 6.0]

=== 3. 야코비 모으기 ===
  단위벡터 네 개로 네 줄을 모았다
  J 가 A 와 같은가: True
  출력이 m 개면 backward 를 m 번 불러야 한다
  retain_graph=True 가 필요하고, 매번 x.grad 를 비워야 한다

=== 4. VJP 한 번 vs 야코비 전체 ===
  VJP 한 번          0.03 ms
  야코비 전체           6.32 ms
  210 배 든다. 출력 개수 512 보다는 작다
  backward 를 m 번 부르는 값이 m 배까지는 안 가기 때문이다
  그래도 자릿수가 같다. 로스를 홑값으로 만드는 까닭이 이것이다

=== 5. 로스는 왜 홑값인가 ===
  출력이 (8,3) 이면 야코비는 (8,3) x (3,4) 짜리다
  원소 수: 288
  홑값으로 줄이면 backward 한 번으로 끝난다
  W.grad 모양: (3, 4) = W 와 같다
  학습에 필요한 것은 야코비가 아니라 이 하나뿐이다
```

## 2. `v`는 "출력을 얼마씩 중히 볼 것인가"다

$y_i = x_i^2$이면 야코비가 대각이고 대각 원소가 $2x_i$다. 그래서 $v^\top J$가
눈에 보인다.

| `v` | `x.grad` |
|---|---|
| `[1, 0, 0]` | `[2, 0, 0]` |
| `[0, 1, 0]` | `[0, 4, 0]` |
| `[1, 1, 1]` | `[2, 4, 6]` |
| `[10, 1, 0.1]` | `[20, 4, 0.6]` |

$v$의 자리마다 그 출력의 몫이 그만큼 섞여 들어온다. **$v$를 전부 1로 주는 것은
`y.sum().backward()`와 같다** — 확인해 보면 `[2, 4, 6]`으로 같다.

그러니 홑값 로스의 `backward()`는 특별한 것이 아니다. **$v = 1$을 넣어 주는
것**뿐이다.

## 3. 야코비 전체를 얻으려면

```python
for i in range(m):
    v = torch.zeros(m); v[i] = 1.0
    x.grad = None                      # 안 비우면 쌓인다
    y.backward(v, retain_graph=True)   # 안 남기면 두 번째가 터진다
    rows.append(x.grad.clone())
```

$v$를 $i$번째 단위벡터로 두면 $v^\top J$가 $J$의 $i$번째 **줄**이다. $m$번
모으면 야코비가 완성된다. $y = Ax$의 야코비는 $A$ 자신이므로 확인도 쉽다 —
`True`가 나왔다.

두 군데를 챙겨야 한다. `retain_graph=True`가 없으면 두 번째 호출이 터지고
(「계산 그래프 재사용」), `x.grad`를 안 비우면 줄들이 더해진다(「경사 누적」).
**뒤쪽이 더 위험하다** — 터지지 않고 틀린 야코비가 나온다.

## 4. 값이 얼마나 드는가

```
VJP 한 번          0.03 ms
야코비 전체          6.32 ms
210 배 든다. 출력 개수 512 보다는 작다
```

출력이 512개이니 512번 걸어야 하는데, 실제로는 210배다. `backward` 한 번의
값에는 그래프를 훑는 고정 비용이 섞여 있어 $m$배까지는 안 간다.

그래도 **자릿수가 같다.** 출력이 많으면 야코비는 못 구한다고 보아야 한다.

## 5. 로스를 홑값으로 만드는 까닭

```
출력이 (8,3) 이면 야코비는 (8,3) x (3,4) 짜리다
원소 수: 288
홑값으로 줄이면 backward 한 번으로 끝난다
W.grad 모양: (3, 4) = W 와 같다
```

학습에 필요한 것은 **야코비가 아니다.** $\partial L/\partial W$ 하나이고, 그것은
$W$와 같은 모양이다.

그래서 로스를 홑값으로 만드는 것이 관행이 아니라 **필요**다. 홑값이 아니면
$v$를 손으로 정해야 하고, 무엇을 넣든 그것이 곧 "어느 출력을 얼마나 줄일
것인가"를 정하는 일이 된다. 로스 함수가 하는 일이 바로 그 결정이다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
`y.sum().backward()`와 `y.backward(torch.ones_like(y))`가 같은가? 그리고
`y.mean().backward()`는 어떤 `v`에 해당하는가?

</div>

??? success "풀이"
    ```python
    for label, run in (
        ("y.sum().backward()", lambda x: (x ** 2).sum().backward()),
        ("y.backward(ones)", lambda x: (x ** 2).backward(torch.ones(3))),
        ("y.mean().backward()", lambda x: (x ** 2).mean().backward()),
        ("y.backward(ones/3)", lambda x: (x ** 2).backward(torch.ones(3) / 3)),
    ):
        x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        run(x)
        print(f"  {label:22s} x.grad = {[round(v, 6) for v in x.grad.tolist()]}")
    ```

    ```
      y.sum().backward()     x.grad = [2.0, 4.0, 6.0]
      y.backward(ones)       x.grad = [2.0, 4.0, 6.0]
      y.mean().backward()    x.grad = [0.666667, 1.333333, 2.0]
      y.backward(ones/3)     x.grad = [0.666667, 1.333333, 2.0]
    ```

    **`sum`은 $v = \mathbf{1}$이고 `mean`은 $v = \mathbf{1}/n$이다.**

    줄이는 함수가 곧 $v$를 고르는 일이라는 것이 여기서 분명해진다.
    「`nn.Module`로 옮기기」에서 `reduction="sum"`과 `"mean"`이 경사를 $N$배
    가르던 것이 같은 이야기다 — $v$가 $N$배 다르기 때문이다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
3절에서 `x.grad = None`을 빼먹으면 어떻게 되는가? 야코비가 어떻게 틀리는지
보여라.

</div>

??? success "풀이"
    ```python
    torch.manual_seed(0)
    A = torch.randn(3, 3)
    for clear in (True, False):
        x = torch.randn(3, requires_grad=True)
        y = A @ x
        rows = []
        for i in range(3):
            v = torch.zeros(3); v[i] = 1.0
            if clear:
                x.grad = None
            y.backward(v, retain_graph=True)
            rows.append(x.grad.clone())
        J = torch.stack(rows)
        print(f"  비움={clear!s:5s} J == A 인가: {torch.allclose(J, A)}")
        if not clear:
            print("    2번째 줄 =", [round(v, 4) for v in J[1].tolist()])
            print("    A[0]+A[1] =", [round(v, 4) for v in (A[0] + A[1]).tolist()])
    ```

    ```
      비움=True  J == A 인가: True
      비움=False J == A 인가: False
        2번째 줄 = [2.1094, -1.378, -3.5774]
        A[0]+A[1] = [2.1094, -1.378, -3.5774]
    ```

    **둘째 줄이 $A$의 0번과 1번 줄의 합**이다. 셋째 줄은 세 줄 전부의 합이 된다.

    오류는 안 난다. 모양도 맞다. **아래로 갈수록 누적된 야코비**가 나올 뿐이다.

    첫 줄은 맞으므로 "첫 줄만 보고 맞다"고 넘어가기 쉽다. 야코비를 손으로
    모을 때는 **마지막 줄을 검산하는** 편이 낫다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
4절처럼 손으로 모으는 대신 `torch.func.jacrev`를 쓰면 얼마나 빠른가? 그리고
입력이 출력보다 많을 때와 적을 때 **어느 방향이 유리한지** 재어라.

</div>

??? success "풀이"
    ```python
    import time
    from torch.func import jacrev, jacfwd

    def timeit(fn, rep=5):
        for _ in range(2):
            fn()                        # 예열
        ts = []
        for _ in range(rep):
            t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
        return min(ts) * 1e3

    for n, m in ((4096, 8), (8, 4096), (2048, 32), (32, 2048)):
        torch.manual_seed(0)
        A, x = torch.randn(m, n), torch.randn(n)
        f = lambda t: (A @ t).tanh()
        rev, fwd = timeit(lambda: jacrev(f)(x)), timeit(lambda: jacfwd(f)(x))
        print(f"  입력 {n:5d} -> 출력 {m:5d}   jacrev {rev:7.2f} ms   "
              f"jacfwd {fwd:7.2f} ms   "
              f"{'rev' if rev < fwd else 'fwd'} 가 {max(rev, fwd) / min(rev, fwd):.1f}배 빠르다")
    ```

    ```
      입력  4096 -> 출력     8   jacrev    0.10 ms   jacfwd    5.39 ms   rev 가 52.1배 빠르다
      입력     8 -> 출력  4096   jacrev    9.23 ms   jacfwd    0.19 ms   fwd 가 48.0배 빠르다
      입력  2048 -> 출력    32   jacrev    0.09 ms   jacfwd    0.86 ms   rev 가  9.9배 빠르다
      입력    32 -> 출력  2048   jacrev    2.08 ms   jacfwd    0.20 ms   fwd 가 10.2배 빠르다
    ```

    **방향이 정확히 뒤집히고, 치우친 정도만큼 벌어진다.** 4096대 8에서 52배,
    2048대 32에서 10배다.

    `jacrev`는 VJP를 출력 수만큼 부르고, `jacfwd`는 JVP(야코비-벡터 곱)를
    **입력 수만큼** 부른다. 그래서

    - **출력이 적으면**(로스 하나) 역방향이 싸다 — 딥러닝이 여기 있다
    - **입력이 적으면** 순방향이 싸다 — 하이퍼파라미터 몇 개에 대한 민감도 같은 것

    역전파가 딥러닝의 표준이 된 까닭이 이 비대칭이다. 파라미터는 수백만 개인데
    로스는 하나이므로, 역방향 한 번이면 끝난다. 순방향으로 하면 파라미터 수만큼
    불러야 한다.
