# 계산 그래프 재사용 — 무엇이 해제되고 무엇이 남는가

`backward()`를 두 번 부르면 터진다고 흔히 말한다. 정확하지 않다. 터지는 그래프가
있고 **안 터지는 그래프가 있다.**

터지는 까닭은 그래프가 사라져서가 아니라 **역전파가 저장해 둔 중간값이
해제되어서**다. 저장할 것이 없는 연산은 두 번 걸어도 멀쩡하다.

그리고 이 쪽에서 가장 자주 섞이는 것이 하나 더 있다. `retain_graph`는 경사를
지우는 일과 **아무 상관이 없다.**

| | 무엇을 맡는가 |
|---|---|
| `retain_graph=True` | 순전파가 세운 길과 **저장된 중간값** |
| `zero_grad()` · `.grad.zero_()` | 잎에 쌓인 **경사 값** |

## 1. 코드

```python
"""계산 그래프 재사용 — 무엇이 해제되고 무엇이 남는가.

`backward()` 를 두 번 부르면 터진다고 흔히 말한다. 정확히는
**역전파가 저장해 둔 중간값이 해제되어** 터지는 것이고, 저장할 것이
없는 그래프는 두 번 불러도 멀쩡하다.

`retain_graph=True` 는 그 중간값을 남겨 달라는 부탁이다. 경사를
지우는 일과는 아무 상관이 없다 — 흔히 섞이는 자리다.

- 그래프: 순전파가 세운 길과 거기 저장된 중간값. `retain_graph` 가 맡는다
- 경사: 잎에 쌓이는 값. `zero_grad()` 가 맡는다

실행: 이 쪽의 코드를 파일로 갈무리해 돌린다
"""
import torch


def header(title):
    print(f"\n=== {title} ===")


# ========================================================================
# 1. 두 번 부르면 터지는 그래프와 안 터지는 그래프
# ========================================================================

def twice():
    header("1. 두 번 부르면")
    builders = (("x.sum()", lambda t: t.sum(), "저장할 것이 없다"),
                ("(x**2).sum()", lambda t: (t ** 2).sum(), "x 를 저장한다"),
                ("x.exp().sum()", lambda t: t.exp().sum(), "exp(x) 를 저장한다"))
    for expr, build, why in builders:
        x = torch.tensor([1.0, 2.0], requires_grad=True)
        y = build(x)
        y.backward()
        try:
            y.backward()
            print(f"  {expr:14s} 두 번째도 된다   ({why})")
        except RuntimeError as e:
            print(f"  {expr:14s} {str(e).splitlines()[0][:40]}...  ({why})")
    print("  '그래프가 사라져서' 가 아니라 '저장해 둔 값이 해제되어서' 다")


# ========================================================================
# 2. retain_graph 가 하는 일
# ========================================================================

def retaining():
    header("2. retain_graph=True")
    torch.manual_seed(0)
    x = torch.randn(3, requires_grad=True)
    loss = (x ** 2).sum()
    loss.backward(retain_graph=True)
    print("  1번째 뒤 x.grad :", [round(v, 4) for v in x.grad.tolist()])
    loss.backward()                       # 남겨 둔 그래프를 다시 쓴다
    print("  2번째 뒤 x.grad :", [round(v, 4) for v in x.grad.tolist()])
    print("  2x 가 두 번 더해져 4x 가 되었다")
    print("  손으로 4x        :", [round(v, 4) for v in (4 * x.detach()).tolist()])
    try:
        loss.backward()                   # 이번에는 안 남겼으므로
    except RuntimeError as e:
        print("  3번째:", str(e).splitlines()[0][:56])
    print("  마지막 역전파에는 retain_graph 를 주지 않는 것이 보통이다")


# ========================================================================
# 3. 그래프와 경사는 다른 것이다
# ========================================================================

def graph_is_not_grad():
    header("3. 섞이는 자리")
    torch.manual_seed(0)
    x = torch.randn(3, requires_grad=True)

    loss = (x ** 2).sum()
    loss.backward(retain_graph=True)
    first = x.grad.clone()
    x.grad.zero_()                        # 경사만 지운다
    loss.backward()                       # 그래프는 아직 살아 있다
    print("  경사를 지워도 그래프는 그대로:",
          torch.allclose(first, x.grad))
    print("  retain_graph 는 그래프를, zero_() 는 경사를 맡는다")

    x.grad.zero_()
    fresh = (x ** 2).sum()                # 순전파를 다시 하면
    fresh.backward()                      # 새 그래프라 retain 이 필요 없다
    print("  순전파를 다시 하면 새 그래프가 생긴다:",
          torch.allclose(first, x.grad))
    print("  학습 루프가 이렇게 돈다. 그래서 retain_graph 가 거의 필요 없다")


# ========================================================================
# 4. 정말 필요한 두 자리
# ========================================================================

def when_needed():
    header("4. 정말 필요한 두 자리")

    # (가) 한 번의 순전파에서 나온 로스 둘을 따로 역전파할 때
    x = torch.tensor([1.0, 2.0], requires_grad=True)
    h = x * 3                              # 둘이 함께 쓰는 중간값
    a = (h ** 2).sum()
    b = h.abs().sum()
    a.backward(retain_graph=True)          # h 를 b 도 써야 한다
    b.backward()
    print("  (가) 로스 둘을 따로:", [round(v, 4) for v in x.grad.tolist()])
    y = torch.tensor([1.0, 2.0], requires_grad=True)
    g = y * 3
    ((g ** 2).sum() + g.abs().sum()).backward()
    print("       합쳐서 한 번  :", [round(v, 4) for v in y.grad.tolist()])
    print("       같다. 다만 따로 부르려면 retain_graph 가 필요하다")

    # (나) 경사의 경사 — create_graph 가 retain_graph 를 켠다
    z = torch.tensor([2.0], requires_grad=True)
    first, = torch.autograd.grad((z ** 3).sum(), z, create_graph=True)
    second, = torch.autograd.grad(first.sum(), z)
    print("  (나) f = z^3, z = 2")
    print("       1차 도함수 3z^2 =", first.item(), " 손으로", 3 * 2 ** 2)
    print("       2차 도함수 6z   =", second.item(), " 손으로", 6 * 2)
    print("       create_graph=True 는 retain_graph 까지 켜 준다")


def main():
    twice()
    retaining()
    graph_is_not_grad()
    when_needed()


if __name__ == "__main__":
    main()
```

**출력:**

```
=== 1. 두 번 부르면 ===
  x.sum()        두 번째도 된다   (저장할 것이 없다)
  (x**2).sum()   Trying to backward through the graph a s...  (x 를 저장한다)
  x.exp().sum()  Trying to backward through the graph a s...  (exp(x) 를 저장한다)
  '그래프가 사라져서' 가 아니라 '저장해 둔 값이 해제되어서' 다

=== 2. retain_graph=True ===
  1번째 뒤 x.grad : [3.082, -0.5869, -4.3576]
  2번째 뒤 x.grad : [6.164, -1.1737, -8.7152]
  2x 가 두 번 더해져 4x 가 되었다
  손으로 4x        : [6.164, -1.1737, -8.7152]
  3번째: Trying to backward through the graph a second time (or d
  마지막 역전파에는 retain_graph 를 주지 않는 것이 보통이다

=== 3. 섞이는 자리 ===
  경사를 지워도 그래프는 그대로: True
  retain_graph 는 그래프를, zero_() 는 경사를 맡는다
  순전파를 다시 하면 새 그래프가 생긴다: True
  학습 루프가 이렇게 돈다. 그래서 retain_graph 가 거의 필요 없다

=== 4. 정말 필요한 두 자리 ===
  (가) 로스 둘을 따로: [21.0, 39.0]
       합쳐서 한 번  : [21.0, 39.0]
       같다. 다만 따로 부르려면 retain_graph 가 필요하다
  (나) f = z^3, z = 2
       1차 도함수 3z^2 = 12.0  손으로 12
       2차 도함수 6z   = 12.0  손으로 12
       create_graph=True 는 retain_graph 까지 켜 준다
```

## 2. 터지는 그래프와 안 터지는 그래프

| 식 | 두 번째 `backward()` | 까닭 |
|---|---|---|
| `x.sum()` | **된다** | 저장할 것이 없다 |
| `(x**2).sum()` | `RuntimeError` | `x`를 저장해 둔다 |
| `x.exp().sum()` | `RuntimeError` | `exp(x)`를 저장해 둔다 |

$\frac{d}{dx}x^2 = 2x$를 계산하려면 $x$가 있어야 하고, $\frac{d}{dx}e^x = e^x$를
계산하려면 $e^x$가 있어야 한다. 첫 역전파가 끝나면 파이토치는 그 버퍼를 놓아
준다 — 보통은 다시 쓸 일이 없으니 메모리를 돌려주는 것이 맞다.

반면 `sum`의 미분은 1이라 아무것도 필요 없다. 그래서 두 번째도 그냥 된다.

**"같은 코드가 어떤 식에서는 터지고 어떤 식에서는 안 터진다"**는 것이 여기서
나온다. 오류 메시지만 보고 `retain_graph=True`를 붙이기 전에, 왜 그 값이
필요한지를 보는 편이 낫다.

## 3. `retain_graph=True`

```
1번째 뒤 x.grad : [3.082, -0.5869, -4.3576]
2번째 뒤 x.grad : [6.164, -1.1737, -8.7152]
손으로 4x        : [6.164, -1.1737, -8.7152]
```

같은 그래프를 두 번 걸었으므로 $2x$가 두 번 더해져 $4x$가 되었다. **경사가
쌓이는 것은 그래프를 남긴 것과 별개**이고, 지우고 싶으면 따로 지워야 한다.

세 번째는 터진다. 두 번째 호출에 `retain_graph`를 주지 않았기 때문이다. 그래서
여러 번 걸을 때는 **마지막 호출에만 빼는** 것이 보통이다 — 그래야 버퍼가 제때
풀린다.

## 4. 섞이는 자리

```
경사를 지워도 그래프는 그대로: True
순전파를 다시 하면 새 그래프가 생긴다: True
```

`x.grad.zero_()`를 불러도 그래프는 살아 있다. 둘은 다른 것을 가리킨다.

그리고 둘째 줄이 **학습 루프에서 `retain_graph`가 거의 필요 없는 까닭**이다.
루프는 걸음마다 순전파를 다시 하고, 그때마다 새 그래프가 생긴다. 지난 그래프는
쓸 일이 없으니 놓아 주는 것이 맞다.

> `backward()`가 두 번째 호출에서 터진다고 `retain_graph=True`를 붙이는 것은
> 대개 **잘못된 고침**이다. 진짜 문제는 순전파를 다시 해야 할 자리에서 안 하고
> 있는 것이고, 붙여 두면 터지지는 않으면서 메모리를 걸음마다 더 쥔다.

## 5. 정말 필요한 두 자리

```
(가) 로스 둘을 따로: [21.0, 39.0]
     합쳐서 한 번  : [21.0, 39.0]

(나) f = z^3, z = 2
     1차 도함수 3z^2 = 12.0   손으로 12
     2차 도함수 6z   = 12.0   손으로 12
```

**(가) 한 번의 순전파에서 나온 로스 둘을 따로 역전파할 때.** 둘이 같은 중간값
`h`를 쓰므로 첫 역전파가 그것을 놓아 주면 안 된다. 다만 합쳐서 한 번 부르면
같은 답이 나오고 더 빠르다 — 따로 불러야 하는 것은 로스마다 다른 파라미터에
경사를 보낼 때다.

**(나) 경사의 경사를 구할 때.** $f = z^3$에서 1차가 $3z^2 = 12$, 2차가 $6z = 12$로
둘 다 손으로 센 값과 맞는다. 여기서는 `create_graph=True`를 썼는데, 그것이
`retain_graph`까지 켜 준다 — 경사 자체를 미분하려면 경사를 만든 그래프가
살아 있어야 하기 때문이다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
`y = x.sin().sum()`과 `y = x.sum().sin()`은 둘 다 홑값이다. 어느 쪽이 두 번
`backward()`할 수 있는가? **먼저 짐작하고** 확인하라.

</div>

??? success "풀이"
    짐작의 근거는 2절이다. 역전파가 **무엇을 저장해야 하는지** 보면 된다.

    - `x.sin().sum()`: $\frac{d}{dx}\sin x = \cos x$이므로 **$x$를 저장**해야 한다
    - `x.sum().sin()`: 먼저 더해 홑값을 만든 다음 $\sin$을 취한다. 저장되는 것은
      **0차원 스칼라** 하나뿐이다

    ```python
    for expr, build in (("x.sin().sum()", lambda t: t.sin().sum()),
                        ("x.sum().sin()", lambda t: t.sum().sin())):
        x = torch.tensor([1.0, 2.0], requires_grad=True)
        y = build(x)
        y.backward()
        try:
            y.backward()
            print(f"  {expr:14s} 두 번째도 된다")
        except RuntimeError as e:
            print(f"  {expr:14s} {str(e).splitlines()[0][:38]}...")
    ```

    ```
      x.sin().sum()  Trying to backward through the graph a...
      x.sum().sin()  Trying to backward through the graph a...
    ```

    **둘 다 터진다.** 짐작이 반만 맞았다.

    `x.sum().sin()`도 $\cos(\sum x)$를 계산하려면 $\sum x$가 있어야 하므로 저장할
    것이 있다. 크기가 작을 뿐(스칼라 하나) 저장은 저장이다.

    그래서 2절의 `x.sum()`이 통과한 것은 **저장할 것이 전혀 없는 아주 드문
    경우**였다. 실제 모델에서 두 번 `backward()`가 그냥 되는 일은 거의 없다고
    보아야 한다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
`retain_graph=True`를 쓰지 않고 같은 경사를 두 번 얻으려면 어떻게 하는가? 두
방법의 결과가 같은지 확인하고, 어느 쪽이 나은지 적어라.

</div>

??? success "풀이"
    ```python
    torch.manual_seed(0)
    x = torch.randn(3, requires_grad=True)

    loss = (x ** 2).sum()                     # 방법 1: 그래프를 남긴다
    loss.backward(retain_graph=True)
    loss.backward()
    way1 = x.grad.clone()

    x.grad = None
    for _ in range(2):                        # 방법 2: 순전파를 다시 한다
        (x ** 2).sum().backward()
    way2 = x.grad.clone()

    print("  retain_graph :", [round(v, 4) for v in way1.tolist()])
    print("  다시 순전파   :", [round(v, 4) for v in way2.tolist()])
    print("  같은가        :", torch.allclose(way1, way2))
    ```

    ```
      retain_graph : [6.164, -1.1737, -8.7152]
      다시 순전파   : [6.164, -1.1737, -8.7152]
      같은가        : True
    ```

    결과가 같다. 가르는 것은 **무엇을 아끼느냐**다.

    | | 아끼는 것 | 치르는 것 |
    |---|---|---|
    | `retain_graph` | 순전파를 다시 안 한다 | 중간값을 계속 들고 있다 |
    | 다시 순전파 | 메모리를 놓아 준다 | 순전파를 한 번 더 한다 |

    이것은 「메모리 관리」의 **그래디언트 체크포인팅과 정확히 같은 거래**다.
    메모리와 계산을 맞바꾸는 것이고, 어느 쪽이 나은지는 어느 쪽이 모자라느냐에
    달렸다.

    파라미터가 바뀌지 않았다면 두 방법의 결과가 같지만, **바뀌었다면 다르다.**
    `retain_graph`는 옛 순전파의 값으로 계산하고 다시 순전파하는 쪽은 새 값으로
    계산한다. 학습 루프 한가운데서는 거의 언제나 뒤쪽이 맞다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
$f(x) = x^3$의 2차 도함수를 `create_graph=True`로 구했다. **3차 도함수**도 구할
수 있는가? 그리고 `create_graph` 없이 2차를 구하려 하면 어떻게 되는가?

</div>

??? success "풀이"
    ```python
    z = torch.tensor([2.0], requires_grad=True)
    g1, = torch.autograd.grad((z ** 3).sum(), z, create_graph=True)
    g2, = torch.autograd.grad(g1.sum(), z, create_graph=True)
    g3, = torch.autograd.grad(g2.sum(), z)
    print(f"  f=z^3, z=2     1차 {g1.item():5.1f} (3z^2={3*4})")
    print(f"                 2차 {g2.item():5.1f} (6z  ={6*2})")
    print(f"                 3차 {g3.item():5.1f} (6   =6)")

    w = torch.tensor([2.0], requires_grad=True)
    h1, = torch.autograd.grad((w ** 3).sum(), w)      # create_graph 없이
    print("  create_graph 없이 1차:", h1.item(),
          " requires_grad:", h1.requires_grad)
    try:
        torch.autograd.grad(h1.sum(), w)
    except RuntimeError as e:
        print("  그 1차를 다시 미분하면:", str(e).splitlines()[0][:58])
    ```

    ```
      f=z^3, z=2     1차  12.0 (3z^2=12)
                     2차  12.0 (6z  =12)
                     3차   6.0 (6   =6)
      create_graph 없이 1차: 12.0  requires_grad: False
      그 1차를 다시 미분하면: element 0 of tensors does not require grad and does not ha
    ```

    **3차까지 간다.** 몇 번이든 된다 — `create_graph=True`를 계속 켜 두면
    경사를 만드는 과정 자체가 다시 그래프로 기록되기 때문이다.

    `create_graph` 없이 구한 1차는 `requires_grad=False`인 **그냥 숫자**다.
    값은 12로 맞지만 그것이 어디서 왔는지에 대한 기록이 없으므로, 다시 미분하려
    하면 「스칼라 로스의 역전파」 연습문제 2에서 본 것과 같은 거절이 나온다.

    고차 미분이 실제로 쓰이는 자리가 있다. 메타 학습(MAML)에서 "한 걸음 학습한
    뒤의 성능"을 미분할 때, WGAN-GP 같은 경사 벌칙에서 경사의 크기를 로스에 넣을
    때가 그렇다. 둘 다 **경사를 다시 미분**하므로 `create_graph=True`가 필수다.

    값이 싸지는 않다. 경사를 만드는 그래프까지 들고 있어야 하므로 메모리가 대략
    두 배가 되고, 역전파가 한 겹 더 깊어진다.
