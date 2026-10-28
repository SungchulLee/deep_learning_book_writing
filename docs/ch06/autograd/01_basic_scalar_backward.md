# 스칼라 로스의 역전파 — 누가 `.grad`를 받는가

$L = \sum_i x_i^2$ 하나를 끝까지 따라간다. 식은 단순하지만 그 과정에서 네 가지
속성이 각자 다른 일을 하고, 그 넷을 섞으면 뒤에 나오는 모든 것이 헷갈린다.

| | 무엇을 말하는가 |
|---|---|
| `requires_grad` | 이 텐서를 **따라갈** 것인가 |
| `is_leaf` | 이 텐서가 `.grad`를 **받을** 것인가 |
| `grad_fn` | 무엇이 이 텐서를 **만들었나** |
| `.grad` | 받은 경사가 **쌓이는** 자리 |

가장 자주 걸리는 것은 둘째 줄이다. 잎이 아닌 텐서는 **역전파에 참여하면서도
`.grad`를 받지 않는다.** 지나가기만 한다.

## 1. 코드

```python
"""홑값 로스의 역전파 — 누가 .grad 를 받는가.

$L = \\sum_i x_i^2$ 하나를 끝까지 따라가며 네 가지를 가른다.

- `requires_grad` — 이 텐서를 따라갈 것인가
- `is_leaf`      — 이 텐서가 `.grad` 를 받을 것인가
- `grad_fn`      — 무엇이 이 텐서를 만들었나
- `.grad`        — 받은 경사가 쌓이는 자리

잎이 아닌 텐서는 역전파에 참여하면서도 `.grad` 를 받지 않는다.
받고 싶으면 `retain_grad()` 로 부탁해야 한다.

실행: python 01_basic_scalar_backward.py
"""
import warnings

import torch


def header(title):
    print(f"\n=== {title} ===")


def fn_name(t):
    """grad_fn 은 이름만 적는다. 객체를 그대로 찍으면 메모리 주소가
    따라 나와 돌릴 때마다 달라진다."""
    return type(t.grad_fn).__name__ if t.grad_fn is not None else None


# ========================================================================
# 1. 잎 텐서를 만든다
# ========================================================================

def make_leaf():
    header("1. 잎 텐서")
    torch.manual_seed(0)
    x = torch.randn(3, requires_grad=True)
    print("  x              :", [round(v, 4) for v in x.tolist()])
    print("  x.requires_grad:", x.requires_grad)
    print("  x.is_leaf      :", x.is_leaf, " <- 내가 직접 만들었다")
    print("  x.grad_fn      :", fn_name(x), " <- 만든 연산이 없다")
    print("  x.grad         :", x.grad, " <- 아직 역전파를 안 했다")
    return x


# ========================================================================
# 2. 순전파가 그래프를 세운다
# ========================================================================

def forward(x):
    header("2. 순전파")
    loss = (x ** 2).sum()
    print("  loss = (x**2).sum() =", round(loss.item(), 6))
    print("  loss.shape   :", tuple(loss.shape), " <- 0차원, 홑값이다")
    print("  loss.is_leaf :", loss.is_leaf)
    print("  loss.grad_fn :", fn_name(loss))
    print("  x -> pow -> sum -> loss 로 이어진 그래프가 세워졌다")
    return loss


# ========================================================================
# 3. 역전파
# ========================================================================

def backward(x, loss):
    header("3. 역전파")
    loss.backward()
    print("  x.grad     :", [round(v, 4) for v in x.grad.tolist()])
    print("  손으로 2x  :", [round(v, 4) for v in (2 * x.detach()).tolist()])
    print("  같은가     :", torch.allclose(x.grad, 2 * x.detach()))
    print("  홑값이라 backward() 에 아무것도 안 주어도 된다")
    print("  위에서 내려온 경사를 1 로 놓고 시작한다는 뜻이다")


# ========================================================================
# 4. 잎이 아닌 텐서는 .grad 를 받지 않는다
# ========================================================================

def non_leaf():
    header("4. 잎이 아닌 텐서")
    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    y = 2 * x
    z = (y ** 2).sum()
    print(f"  x  is_leaf={x.is_leaf!s:5s} grad_fn={fn_name(x)}")
    print(f"  y  is_leaf={y.is_leaf!s:5s} grad_fn={fn_name(y)}")
    print(f"  z  is_leaf={z.is_leaf!s:5s} grad_fn={fn_name(z)}")
    z.backward()
    print("  x.grad :", x.grad.tolist(), " <- 받았다")
    with warnings.catch_warnings():       # 경고문에 빌드 기계 경로가 붙어 나온다
        warnings.simplefilter("ignore")
        print("  y.grad :", y.grad, " <- 잎이 아니라 못 받는다")
    print("  읽으려 하면 파이토치가 경고를 띄운다 (여기서는 꺼 두었다)")
    print("  y 가 역전파에 참여하지 않은 것이 아니다. 지나가기만 한 것이다")


# ========================================================================
# 5. retain_grad() 로 부탁한다
# ========================================================================

def retaining():
    header("5. retain_grad()")
    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    y = 2 * x
    y.retain_grad()                    # 역전파 전에 불러야 한다
    (y ** 2).sum().backward()
    print("  y.grad      :", y.grad.tolist())
    print("  손으로 2y   :", (2 * y.detach()).tolist())
    print("  x.grad      :", x.grad.tolist())
    print("  손으로 8x   :", (8 * x.detach()).tolist(), " (dz/dx = 2y*2 = 8x)")
    print("  retain_grad 는 값을 남기고 retain_graph 는 그래프를 남긴다")
    print("  이름이 한 글자 차이라 섞이기 쉽다")

    late = torch.tensor([1.0], requires_grad=True)
    w = 2 * late
    (w ** 2).sum().backward()
    w.retain_grad()                    # 역전파 뒤에 부르면
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        print("  역전파 뒤에 부르면 w.grad :", w.grad, " <- 늦었다")


def main():
    x = make_leaf()
    loss = forward(x)
    backward(x, loss)
    non_leaf()
    retaining()


if __name__ == "__main__":
    main()
```

**출력:**

```
=== 1. 잎 텐서 ===
  x              : [1.541, -0.2934, -2.1788]
  x.requires_grad: True
  x.is_leaf      : True  <- 내가 직접 만들었다
  x.grad_fn      : None  <- 만든 연산이 없다
  x.grad         : None  <- 아직 역전파를 안 했다

=== 2. 순전파 ===
  loss = (x**2).sum() = 7.207892
  loss.shape   : ()  <- 0차원, 홑값이다
  loss.is_leaf : False
  loss.grad_fn : SumBackward0
  x -> pow -> sum -> loss 로 이어진 그래프가 세워졌다

=== 3. 역전파 ===
  x.grad     : [3.082, -0.5869, -4.3576]
  손으로 2x  : [3.082, -0.5869, -4.3576]
  같은가     : True
  홑값이라 backward() 에 아무것도 안 주어도 된다
  위에서 내려온 경사를 1 로 놓고 시작한다는 뜻이다

=== 4. 잎이 아닌 텐서 ===
  x  is_leaf=True  grad_fn=None
  y  is_leaf=False grad_fn=MulBackward0
  z  is_leaf=False grad_fn=SumBackward0
  x.grad : [8.0, 16.0, 24.0]  <- 받았다
  y.grad : None  <- 잎이 아니라 못 받는다
  읽으려 하면 파이토치가 경고를 띄운다 (여기서는 꺼 두었다)
  y 가 역전파에 참여하지 않은 것이 아니다. 지나가기만 한 것이다

=== 5. retain_grad() ===
  y.grad      : [4.0, 8.0, 12.0]
  손으로 2y   : [4.0, 8.0, 12.0]
  x.grad      : [8.0, 16.0, 24.0]
  손으로 8x   : [8.0, 16.0, 24.0]  (dz/dx = 2y*2 = 8x)
  retain_grad 는 값을 남기고 retain_graph 는 그래프를 남긴다
  이름이 한 글자 차이라 섞이기 쉽다
  역전파 뒤에 부르면 w.grad : None  <- 늦었다
```

## 2. 잎이란 내가 만든 것이다

```
x.requires_grad: True
x.is_leaf      : True    <- 내가 직접 만들었다
x.grad_fn      : None    <- 만든 연산이 없다
x.grad         : None    <- 아직 역전파를 안 했다
```

`is_leaf`가 참이라는 것은 **그래프를 거꾸로 거슬러 올라가 더 갈 곳이 없다**는
뜻이다. 내가 `torch.randn(3, requires_grad=True)`으로 만들었으니 그 위에 아무것도
없다.

그래서 `grad_fn`이 `None`이다. 둘은 같은 사실을 두 방향에서 말한다 — 만든 연산이
없으면 잎이고, 잎이면 만든 연산이 없다.

`.grad`가 `None`인 것은 아직 아무것도 안 했기 때문이다. 0이 아니라 `None`인
것에 뜻이 있다. 「Autograd 관련 속성」에서 보았듯 옵티마이저는 이 둘을 다르게
취급한다.

## 3. 순전파가 그래프를 세운다

```
loss.shape   : ()          <- 0차원, 홑값이다
loss.is_leaf : False
loss.grad_fn : SumBackward0
```

`loss = (x ** 2).sum()` 한 줄을 적었을 뿐인데 `x → pow → sum → loss`라는 그래프가
세워졌다. `grad_fn`이 `SumBackward0`인 것은 마지막 연산이 `sum`이었다는 뜻이고,
그 `SumBackward0`이 다시 `PowBackward0`을 가리키고 있다.

## 4. 홑값이면 `backward()`에 아무것도 안 준다

```
x.grad     : [3.082, -0.5869, -4.3576]
손으로 2x  : [3.082, -0.5869, -4.3576]
```

$\partial L/\partial x_i = 2x_i$가 그대로 나왔다.

`loss.backward()`에 인자를 안 준 것은 **위에서 내려온 경사를 1로 놓는다**는
뜻이다. 로스가 0차원이므로 "로스 자신에 대한 로스의 미분"이 1이고, 거기서
연쇄 법칙이 시작된다. 홑값이 아니면 그 1을 손으로 주어야 하고, 그것이 뒤에 나오는
벡터-야코비 곱이다.

## 5. 잎이 아니면 `.grad`를 받지 않는다

```
x  is_leaf=True  grad_fn=None
y  is_leaf=False grad_fn=MulBackward0
z  is_leaf=False grad_fn=SumBackward0

x.grad : [8.0, 16.0, 24.0]   <- 받았다
y.grad : None                <- 잎이 아니라 못 받는다
```

`y = 2 * x`는 분명히 역전파에 참여했다. $x$가 경사를 받으려면 $y$를 거쳐야 하기
때문이다. 그런데도 `y.grad`가 `None`이다.

**참여하는 것과 보관하는 것이 다르다.** 중간 텐서의 경사는 계산되어 다음 칸으로
넘겨진 다음 버려진다. 모델 하나에 중간 텐서가 수백만 개일 수 있으니, 전부 들고
있으면 메모리가 남아나지 않는다.

읽으려 하면 파이토치가 경고를 띄운다. 그냥 "없다"가 아니라 **"받고 싶으면
`retain_grad()`를 부르라"**고 알려 주는 경고다.

## 6. `retain_grad()`는 역전파 **전에** 불러야 한다

```
y.grad      : [4.0, 8.0, 12.0]
손으로 2y   : [4.0, 8.0, 12.0]
x.grad      : [8.0, 16.0, 24.0]
손으로 8x   : [8.0, 16.0, 24.0]   (dz/dx = 2y*2 = 8x)
```

둘 다 손으로 센 값과 맞는다. 연쇄 법칙이 그대로 보인다 —
$\frac{dz}{dx} = \frac{dz}{dy}\frac{dy}{dx} = 2y \cdot 2 = 8x$.

순서가 중요하다.

```
역전파 뒤에 부르면 w.grad : None   <- 늦었다
```

`retain_grad()`는 "이번 역전파에서 네 경사를 남겨 달라"는 부탁이므로, 역전파가
끝난 뒤에 불러 봐야 소용이 없다. 다시 순전파부터 해야 한다.

그리고 이름이 비슷한 둘을 갈라 두자.

| | 무엇을 남기는가 |
|---|---|
| `retain_grad()` | 그 텐서의 **경사 값** |
| `retain_graph=True` | 역전파가 쓰려고 저장해 둔 **중간값** |

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
$f(x) = x^3 - 2x^2 + x$에 대해 autograd로 $f'(3)$을 구하고 손으로 미분한 값과
견주어라. `f` 자체의 `.grad`는 왜 `None`인가?

</div>

??? success "풀이"
    ```python
    x = torch.tensor(3.0, requires_grad=True)
    f = x ** 3 - 2 * x ** 2 + x
    f.backward()
    print("  f(3)   :", f.item())
    print("  x.grad :", x.grad.item())
    print("  손으로 3x^2-4x+1 =", 3 * 3 ** 2 - 4 * 3 + 1)
    print("  f.is_leaf :", f.is_leaf)
    ```

    ```
      f(3)   : 12.0
      x.grad : 16.0
      손으로 3x^2-4x+1 = 16
      f.is_leaf : False
    ```

    $f'(x) = 3x^2 - 4x + 1$이므로 $f'(3) = 27 - 12 + 1 = 16$이다.

    `f.grad`가 `None`인 까닭은 5절 그대로다. `f`는 계산으로 생긴 텐서라 잎이
    아니고, 잎이 아니면 `.grad`를 받지 않는다. 게다가 여기서 `f`는 역전파의
    **출발점**이므로 "위에서 내려온 경사"가 곧 1인데, 그 1을 어디에 저장할
    까닭도 없다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
`x`가 `requires_grad=True`일 때, `y = x.detach() * 2`로 만든 `y`에
`retain_grad()`를 부르면 어떻게 되는가? 그리고 그 `y`로 로스를 만들어
`backward()`하면?

</div>

??? success "풀이"
    ```python
    x = torch.tensor([1.0, 2.0], requires_grad=True)
    y = x.detach() * 2
    print("  y.requires_grad :", y.requires_grad)
    print("  y.is_leaf       :", y.is_leaf)
    try:
        y.retain_grad()
        print("  retain_grad() 통과")
    except RuntimeError as e:
        print("  retain_grad() :", str(e).splitlines()[0][:60])
    try:
        (y ** 2).sum().backward()
    except RuntimeError as e:
        print("  backward() :", str(e).splitlines()[0][:66])
    ```

    ```
      y.requires_grad : False
      y.is_leaf       : True
      retain_grad() : can't retain_grad on Tensor that has requires_grad=False
      backward() : element 0 of tensors does not require grad and does not have
    ```

    세 줄이 다 뜻밖일 수 있다.

    **`y`는 잎이다.** `detach()`가 그래프를 끊었으므로 `y`는 그래프 바깥에 있고,
    그래프 바깥에 있다는 말과 잎이라는 말은 같다. `requires_grad`가 `False`인
    텐서는 언제나 잎이다.

    **`retain_grad()`가 거절한다.** 반가운 일이다 — `requires_grad=False`인
    텐서에는 애초에 남길 경사가 없으므로, 조용히 통과시켜 두었다가 나중에
    `.grad`가 `None`인 것을 보고 헤매는 대신 그 자리에서 알려 준다.

    **`backward()`도 터진다.** 뒤로 걸어갈 길이 없기 때문이다. `detach()`가 끊은
    자리에서 그래프가 끝났으므로 `x`까지 갈 수 없고, 엔진이 `element 0 of tensors
    does not require grad`라고 거절한다.

    셋을 합치면 이렇다. **`detach()` 뒤로는 아무 경사도 흐르지 않고, 파이토치가
    그것을 세 번 다 알려 준다.** 조용히 틀리는 쪽이 아니라 터져 주는 쪽이다.

    여기서 가져갈 것은 **`requires_grad=False`와 `is_leaf=True`가 같이 다닌다**는
    점이다. 「Autograd 관련 속성」의 표에서 네 조합 중 셋만 있던 까닭이 이것이다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
같은 잎 텐서로 로스를 **세 번** 만들어 `backward()`를 세 번 부르되, 사이에
경사를 지우지 않는다. `x.grad`가 어떻게 변하는가? 그리고 **중간에
`retain_grad()`한 텐서**는 어떻게 되는가?

</div>

??? success "풀이"
    ```python
    x = torch.tensor([1.0, 2.0], requires_grad=True)
    for i in range(1, 4):
        y = 2 * x
        y.retain_grad()
        (y ** 2).sum().backward()
        print(f"  {i}번째  x.grad = {x.grad.tolist()}   y.grad = {y.grad.tolist()}")
    ```

    ```
      1번째  x.grad = [8.0, 16.0]   y.grad = [4.0, 8.0]
      2번째  x.grad = [16.0, 32.0]   y.grad = [4.0, 8.0]
      3번째  x.grad = [24.0, 48.0]   y.grad = [4.0, 8.0]
    ```

    **`x.grad`는 쌓이는데 `y.grad`는 안 쌓인다.**

    까닭은 둘의 수명이 다르기 때문이다. `x`는 바깥에 있는 **같은 텐서**라서 세
    번의 경사가 전부 그 하나에 더해진다. 반면 `y`는 반복마다 `2 * x`로 **새로
    만들어지는** 텐서다. 매번 다른 객체이므로 쌓일 자리가 없고, 지난 바퀴의 `y`는
    버려진다.

    이것이 학습 루프에서 `opt.zero_grad()`를 매 걸음 불러야 하는 까닭 전부다.
    파라미터는 루프 바깥에 있는 잎이라 지우지 않으면 계속 쌓이고, 중간 활성값은
    매 걸음 새로 만들어지므로 신경 쓸 일이 없다.

    쌓이는 것이 기능이라는 점도 기억해 둘 만하다. 「메모리 관리」의 경사 누적은
    이 성질을 일부러 쓰는 것이다 — 배치를 쪼개 여러 번 `backward()`하고 한 번
    `step()`한다.
