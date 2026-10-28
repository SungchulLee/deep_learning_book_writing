# PyTorch Autograd 기초

[앞 쪽](01_manual_gradient_numpy.md)에서 경사 식을 손으로 밀었다. 이 쪽에서는 그 한 줄을 `loss.backward()`로 바꾼다.

바뀌는 것은 **그 한 줄뿐이다.** 나머지는 그대로 남고, 대신 새 줄 두 개가 생긴다. 이 쪽은 바뀌는 것과 남는 것을 또렷이 갈라 놓는 데 목적이 있다. autograd가 무엇을 해 주고 무엇은 여전히 내가 해야 하는지가 분명해야, 뒤에 나올 옵티마이저와 학습 루프가 어디까지 자동인지 알 수 있다.

---

## 1. `requires_grad`가 켜는 것

```
  보통 텐서       requires_grad=False  grad_fn=None
  켜 둔 텐서      requires_grad=True  grad_fn=None
  a*3 의 grad_fn  None
  b*3 의 grad_fn  MulBackward0
```

두 텐서 모두 `grad_fn`이 `None`이다. **직접 만든 텐서는 어느 쪽이든 잎(leaf)** 이고, 잎은 자기를 만든 연산이 없다.

갈리는 것은 그 다음이다. 보통 텐서에 3을 곱하면 결과도 `grad_fn=None`이다. 켜 둔 텐서에 3을 곱하면 결과가 `MulBackward0`을 달고 나온다. **"나를 만든 것은 곱셈이다"** 라는 쪽지다.

연산마다 이 쪽지가 붙어 사슬을 이루고, `backward()`는 그 사슬을 거꾸로 따라가며 연쇄 법칙을 적용한다. 그게 전부다.

---

## 2. 손으로 민 식과 같은 수가 나오는가

autograd를 믿기 전에 한 번 견주어 본다.

```
  f(x) = 3x^2 + 2x + 1,  f'(x) = 6x + 2
  x       손으로 6x+2      autograd       차이
  -1.0        -4.000000      -4.000000  0.00e+00
  0.0         +2.000000      +2.000000  0.00e+00
  2.0        +14.000000     +14.000000  0.00e+00
  3.5        +23.000000     +23.000000  0.00e+00

  앞 쪽의 로스에서도 같은지 본다
  L(w) = mean((w*x - y)^2),  손 식은 (2/N) sum x(wx - y)
  w       손으로            autograd        차이
  0.0          -44.000000      -44.000000  0.00e+00
  1.0          -22.000000      -22.000000  0.00e+00
  2.0           +0.000000       +0.000000  0.00e+00
```

차이가 **정확히 0** 이다. 근사가 아니다.

이 점을 분명히 해 두는 것이 좋다. autograd는 수치 미분이 아니다. 차분을 쓰지 않으므로 $h$를 고를 일도 없고 절단 오차도 없다. 연쇄 법칙을 기계가 대신 적용해 **정확한 도함수의 값** 을 계산한다. [앞 쪽](01_manual_gradient_numpy.md)의 경사 검사에서 중앙 차분이 $10^{-9}$만큼 어긋났던 것과 견주어 보라.

---

## 3. 학습 루프에서 바뀌는 줄

앞 쪽의 넘파이 루프와 autograd 루프를 나란히 둔다.

```python
# 넘파이                                  # autograd
y_pred = w * X                           loss = ((w * X - Y) ** 2).mean()
g = np.mean(2 * X * (y_pred - Y))        w.grad = None
w = w - 0.01 * g                         loss.backward()
                                         with torch.no_grad():
                                             w -= 0.01 * w.grad
```

```
  넘파이 판 20 걸음 뒤 w = 1.986103
  autograd 판            w = 1.986103
  차이 8.96e-08
```

같은 답에 닿는다. 차이 $9 \times 10^{-8}$은 float64와 float32의 차이다.

**사라진 줄**은 손으로 민 경사 식 하나다. **새로 생긴 줄**은 둘이다.

- `w.grad = None` — 지우기. 4절에서 왜 필요한지 본다
- `with torch.no_grad():` — 갱신을 기록하지 않기. 5절에서 본다

**남은 줄**은 순전파, 로스, 갱신이다. 전부 그대로다.

!!! note "autograd는 미분만 대신한다"
    경사를 구하는 일과 그 경사로 무엇을 할지 정하는 일은 다른 일이다. autograd는 앞의 것만 해 준다. 러닝레이트를 고르고, 지우고, 갱신하고, 멈출 때를 정하는 것은 여전히 사람 몫이다. 옵티마이저가 그중 갱신을 가져가지만(6절), 그래도 나머지는 남는다.

---

## 4. 지우지 않으면

`.grad`는 **덮어쓰지 않고 더한다.**

```
  같은 로스를 세 번 역전파한다 (참 경사는 매번 2w = 4)
    1번째 뒤  w.grad = 4.0
    2번째 뒤  w.grad = 8.0
    3번째 뒤  w.grad = 12.0
```

학습 루프에서 지우지 않으면 지난 걸음의 경사가 그대로 남아 쌓인다.

```
  지운다    20 걸음 뒤 w = +1.986103
  안 지운다  20 걸음 뒤 w = +3.976701
```

목표가 2.0인데 3.98까지 갔다. **지나쳐 버렸다.** 걸음마다 경사가 누적되어 실효 걸음 폭이 계속 커지기 때문이다.

이 결과가 고약한 까닭은 **터지지 않았다** 는 데 있다. 발산해서 `nan`이 나왔으면 바로 알아챘을 텐데, 그럴듯하게 틀린 수로 끝났다. 러닝레이트가 조금만 컸어도 터졌을 것이다.

!!! note "이것은 버그가 아니라 기능이다"
    더하는 것이 기본값인 까닭이 있다. 배치를 쪼개 여러 번 역전파하고 한 번만 걸으면 큰 배치와 같은 경사가 된다. 그것이 [경사 누적](../autograd/02_gradient_accumulation.md)이다.

---

## 5. `no_grad`와 `detach`

```
  그냥           grad_fn=MulBackward0 requires_grad=True
  no_grad 안에서 grad_fn=None         requires_grad=False
  detach() 하면  grad_fn=None         requires_grad=False
```

결과는 같아 보이지만 쓰는 자리가 다르다.

- **`no_grad`는 범위다.** "이 블록 안에서 일어나는 일은 기록하지 마라"
- **`detach`는 한 텐서다.** "이 텐서에서 줄을 끊어라"

### `detach`는 저장 공간을 함께 쓴다

```
  d[0] 를 99 로 바꾸면 t 는 [99.0, 2.0, 3.0]
```

`detach()`는 **새 텐서를 만들지 않는다.** 같은 메모리를 가리키는 다른 시야일 뿐이다. 떼어 낸 값을 고치면 원본도 고쳐진다. 따로 두려면 `.detach().clone()`이라야 한다.

### 갱신에 `no_grad`를 빼면

```
   a leaf Variable that requires grad is being used in an in-place operatio
```

막힌다. $w$는 잎이고 미분의 **대상** 인데, 그것을 제자리에서 고치면 기록해 둔 그래프가 어긋나기 때문이다. PyTorch가 아예 예외로 끊는다.

그래서 갱신은 셋 중 하나로 한다 — `no_grad` 블록 안에서 하거나, `w.data`를 쓰거나, 옵티마이저에 맡긴다. 세 번째가 보통의 길이다.

---

## 6. 옵티마이저에 맡기면

```
  손으로 갱신      w = 1.98610306
  SGD 옵티마이저   w = 1.98610306
  차이 0.00e+00
```

**한 비트도 다르지 않다.** `lr`만 준 `torch.optim.SGD`는 손으로 적은 두 줄과 글자 그대로 같은 일을 한다.

이것을 확인해 두는 것이 중요하다. 옵티마이저가 마법을 부린다고 생각하면 모멘텀이나 Adam이 무엇을 바꾸는지 이해할 수 없다. 기본 SGD는 아무것도 바꾸지 않는다. **모멘텀이나 웨이트 감쇠를 켜야 비로소 다른 일이 시작된다**(연습문제 4에서 확인한다).

바뀌는 것은 코드의 모양뿐이다.

```python
w.grad = None                    opt.zero_grad()
loss.backward()         →        loss.backward()
with torch.no_grad():            opt.step()
    w -= 0.01 * w.grad
```

파라메터가 하나일 때는 차이가 없지만, 수백 개가 되면 `opt.step()` 한 줄이 전부를 돌아 준다.

---

## 7. 전체 코드와 출력

```python
"""PyTorch autograd 기초 — 손으로 민 식이 무엇으로 바뀌는가.

앞 쪽에서 손으로 민 경사 식을 `loss.backward()` 한 줄로 바꾼다.
그런데 바뀌는 것은 **그 한 줄뿐** 이다. 나머지는 그대로 남는다.

이 쪽은 바뀌는 것과 남는 것을 또렷이 가른다.

- autograd 가 내놓는 경사가 앞 쪽의 손 식과 같은 수인가
- 학습 루프에서 사라지는 줄과 새로 생기는 줄
- 왜 지워야 하는가 (지우지 않으면 무엇이 일어나는가)
- no_grad 와 detach 는 각각 무엇을 하는가

실행: python 02_pytorch_autograd_basics.py
"""
import numpy as np
import torch

X = torch.tensor([1., 2., 3., 4., 5.])
Y = 2 * X


def header(title):
    print(f"\n=== {title} ===")


# ========================================================================
# 1. requires_grad 가 켜는 것
# ========================================================================

def tracking():
    header("1. requires_grad 가 켜는 것")
    a = torch.tensor(2.0)
    b = torch.tensor(2.0, requires_grad=True)
    print(f"  보통 텐서       requires_grad={a.requires_grad}  "
          f"grad_fn={a.grad_fn}")
    print(f"  켜 둔 텐서      requires_grad={b.requires_grad}  "
          f"grad_fn={b.grad_fn}")

    ya, yb = a * 3, b * 3
    print(f"  a*3 의 grad_fn  {ya.grad_fn}")
    print(f"  b*3 의 grad_fn  {type(yb.grad_fn).__name__}")
    print("  켜 두면 연산마다 '무엇이 나를 만들었나' 가 함께 기록된다")
    print("  그 기록을 거꾸로 따라가는 것이 backward() 다")


# ========================================================================
# 2. 손으로 민 식과 같은 수가 나오는가
# ========================================================================

def same_number():
    header("2. 손 식 vs autograd")
    print("  f(x) = 3x^2 + 2x + 1,  f'(x) = 6x + 2")
    print("  x       손으로 6x+2      autograd       차이")
    for v in (-1.0, 0.0, 2.0, 3.5):
        x = torch.tensor(v, requires_grad=True)
        (3 * x ** 2 + 2 * x + 1).backward()
        hand = 6 * v + 2
        print(f"  {v:<7.1f} {hand:+13.6f}  {x.grad.item():+13.6f}  "
              f"{abs(hand - x.grad.item()):.2e}")

    print("\n  앞 쪽의 로스에서도 같은지 본다")
    print("  L(w) = mean((w*x - y)^2),  손 식은 (2/N) sum x(wx - y)")
    print("  w       손으로            autograd        차이")
    for v in (0.0, 1.0, 2.0):
        w = torch.tensor(v, requires_grad=True)
        ((w * X - Y) ** 2).mean().backward()
        hand = float((2 * X * (v * X - Y)).mean())
        print(f"  {v:<7.1f} {hand:+15.6f}  {w.grad.item():+14.6f}  "
              f"{abs(hand - w.grad.item()):.2e}")
    print("  같은 수다. autograd 가 새로운 수학을 하는 것이 아니다")
    print("  연쇄 법칙을 기계가 대신 적용해 줄 뿐이다")


# ========================================================================
# 3. 루프에서 무엇이 바뀌는가
# ========================================================================

def loop_diff():
    header("3. 학습 루프 — 바뀌는 줄과 남는 줄")

    # (가) 앞 쪽의 넘파이 판
    Xn = np.array([1., 2., 3., 4., 5.])
    Yn = 2 * Xn
    w_np = 0.0
    for _ in range(20):
        y_pred = w_np * Xn
        g = np.mean(2 * Xn * (y_pred - Yn))     # <- 손으로 민 식
        w_np = w_np - 0.01 * g

    # (나) autograd 판
    w = torch.tensor(0.0, requires_grad=True)
    for _ in range(20):
        loss = ((w * X - Y) ** 2).mean()
        w.grad = None                            # <- 지운다 (새로 생긴 줄)
        loss.backward()                          # <- 손 식을 대신한다
        with torch.no_grad():                    # <- 갱신은 기록하지 않는다
            w -= 0.01 * w.grad

    print(f"  넘파이 판 20 걸음 뒤 w = {w_np:.6f}")
    print(f"  autograd 판            w = {w.item():.6f}")
    print(f"  차이 {abs(w_np - w.item()):.2e}")
    print("  사라진 줄: 손으로 민 경사 식 한 줄")
    print("  새로 생긴 줄: 지우기(w.grad = None) 와 no_grad 블록")
    print("  남은 줄: 순전파, 로스, 갱신 — 전부 그대로다")
    print("  autograd 는 미분만 대신한다. 최적화는 여전히 손으로 적는다")


# ========================================================================
# 4. 지우지 않으면
# ========================================================================

def accumulation():
    header("4. 지우지 않으면 무엇이 일어나는가")
    w = torch.tensor(2.0, requires_grad=True)
    print("  같은 로스를 세 번 역전파한다 (참 경사는 매번 2w = 4)")
    for i in range(3):
        (w ** 2).backward()
        print(f"    {i+1}번째 뒤  w.grad = {w.grad.item():.1f}")
    print("  더해진다. 덮어쓰지 않는다")

    print("\n  지우지 않은 채로 학습하면")
    for clear in (True, False):
        w = torch.tensor(0.0, requires_grad=True)
        for _ in range(20):
            loss = ((w * X - Y) ** 2).mean()
            if clear:
                w.grad = None
            loss.backward()
            with torch.no_grad():
                w -= 0.01 * w.grad
        lab = "지운다  " if clear else "안 지운다"
        print(f"    {lab}  20 걸음 뒤 w = {w.item():+.6f}")
    print("  안 지우면 지난 경사가 쌓여 걸음이 점점 커지고 지나쳐 버린다")
    print("  (이 성질을 일부러 쓰는 것이 경사 누적이다)")


# ========================================================================
# 5. no_grad 와 detach
# ========================================================================

def no_grad_and_detach():
    header("5. no_grad 와 detach")
    w = torch.tensor(1.0, requires_grad=True)

    a = w * 2
    with torch.no_grad():
        b = w * 2
    c = (w * 2).detach()
    print(f"  그냥           grad_fn={type(a.grad_fn).__name__:<12} "
          f"requires_grad={a.requires_grad}")
    print(f"  no_grad 안에서 grad_fn={str(b.grad_fn):<12} "
          f"requires_grad={b.requires_grad}")
    print(f"  detach() 하면  grad_fn={str(c.grad_fn):<12} "
          f"requires_grad={c.requires_grad}")
    print("  no_grad 는 '이 블록 안을 기록하지 마라' 다 (범위)")
    print("  detach 는 '이 텐서에서 줄을 끊어라' 다 (한 텐서)")

    print("\n  detach 는 저장 공간을 함께 쓴다")
    t = torch.tensor([1., 2., 3.], requires_grad=True)
    d = t.detach()
    d[0] = 99.0
    print(f"  d[0] 를 99 로 바꾸면 t 는 {t.detach().tolist()}")
    print("  따로 두려면 .detach().clone() 이라야 한다")

    print("\n  갱신에 no_grad 를 빼면")
    u = torch.tensor(0.0, requires_grad=True)
    ((u * X - Y) ** 2).mean().backward()
    try:
        u -= 0.01 * u.grad
    except RuntimeError as e:
        print("  ", str(e).splitlines()[0][:72])
    print("  잎 텐서를 제자리에서 고치려 하면 막는다. 갱신은 미분 대상이")
    print("  아니므로 no_grad 안에서 하거나 u.data 를 쓰거나 옵티마이저에")
    print("  맡긴다")


# ========================================================================
# 6. 같은 루프를 옵티마이저로
# ========================================================================

def with_optimizer():
    header("6. 옵티마이저에 맡기면")
    w1 = torch.tensor(0.0, requires_grad=True)
    for _ in range(20):
        loss = ((w1 * X - Y) ** 2).mean()
        w1.grad = None
        loss.backward()
        with torch.no_grad():
            w1 -= 0.01 * w1.grad

    w2 = torch.tensor(0.0, requires_grad=True)
    opt = torch.optim.SGD([w2], lr=0.01)
    for _ in range(20):
        loss = ((w2 * X - Y) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

    print(f"  손으로 갱신      w = {w1.item():.8f}")
    print(f"  SGD 옵티마이저   w = {w2.item():.8f}")
    print(f"  차이 {abs(w1.item() - w2.item()):.2e}")
    print("  lr 만 준 SGD 는 손으로 적은 두 줄과 글자 그대로 같은 일을 한다")
    print("  모멘텀이나 웨이트 감쇠를 켜야 비로소 다른 일을 시작한다")


def main():
    tracking()
    same_number()
    loop_diff()
    accumulation()
    no_grad_and_detach()
    with_optimizer()


if __name__ == "__main__":
    main()
```

**출력:**

```
=== 1. requires_grad 가 켜는 것 ===
  보통 텐서       requires_grad=False  grad_fn=None
  켜 둔 텐서      requires_grad=True  grad_fn=None
  a*3 의 grad_fn  None
  b*3 의 grad_fn  MulBackward0
  켜 두면 연산마다 '무엇이 나를 만들었나' 가 함께 기록된다
  그 기록을 거꾸로 따라가는 것이 backward() 다

=== 2. 손 식 vs autograd ===
  f(x) = 3x^2 + 2x + 1,  f'(x) = 6x + 2
  x       손으로 6x+2      autograd       차이
  -1.0        -4.000000      -4.000000  0.00e+00
  0.0         +2.000000      +2.000000  0.00e+00
  2.0        +14.000000     +14.000000  0.00e+00
  3.5        +23.000000     +23.000000  0.00e+00

  앞 쪽의 로스에서도 같은지 본다
  L(w) = mean((w*x - y)^2),  손 식은 (2/N) sum x(wx - y)
  w       손으로            autograd        차이
  0.0          -44.000000      -44.000000  0.00e+00
  1.0          -22.000000      -22.000000  0.00e+00
  2.0           +0.000000       +0.000000  0.00e+00
  같은 수다. autograd 가 새로운 수학을 하는 것이 아니다
  연쇄 법칙을 기계가 대신 적용해 줄 뿐이다

=== 3. 학습 루프 — 바뀌는 줄과 남는 줄 ===
  넘파이 판 20 걸음 뒤 w = 1.986103
  autograd 판            w = 1.986103
  차이 8.96e-08
  사라진 줄: 손으로 민 경사 식 한 줄
  새로 생긴 줄: 지우기(w.grad = None) 와 no_grad 블록
  남은 줄: 순전파, 로스, 갱신 — 전부 그대로다
  autograd 는 미분만 대신한다. 최적화는 여전히 손으로 적는다

=== 4. 지우지 않으면 무엇이 일어나는가 ===
  같은 로스를 세 번 역전파한다 (참 경사는 매번 2w = 4)
    1번째 뒤  w.grad = 4.0
    2번째 뒤  w.grad = 8.0
    3번째 뒤  w.grad = 12.0
  더해진다. 덮어쓰지 않는다

  지우지 않은 채로 학습하면
    지운다    20 걸음 뒤 w = +1.986103
    안 지운다  20 걸음 뒤 w = +3.976701
  안 지우면 지난 경사가 쌓여 걸음이 점점 커지고 지나쳐 버린다
  (이 성질을 일부러 쓰는 것이 경사 누적이다)

=== 5. no_grad 와 detach ===
  그냥           grad_fn=MulBackward0 requires_grad=True
  no_grad 안에서 grad_fn=None         requires_grad=False
  detach() 하면  grad_fn=None         requires_grad=False
  no_grad 는 '이 블록 안을 기록하지 마라' 다 (범위)
  detach 는 '이 텐서에서 줄을 끊어라' 다 (한 텐서)

  detach 는 저장 공간을 함께 쓴다
  d[0] 를 99 로 바꾸면 t 는 [99.0, 2.0, 3.0]
  따로 두려면 .detach().clone() 이라야 한다

  갱신에 no_grad 를 빼면
   a leaf Variable that requires grad is being used in an in-place operatio
  잎 텐서를 제자리에서 고치려 하면 막는다. 갱신은 미분 대상이
  아니므로 no_grad 안에서 하거나 u.data 를 쓰거나 옵티마이저에
  맡긴다

=== 6. 옵티마이저에 맡기면 ===
  손으로 갱신      w = 1.98610306
  SGD 옵티마이저   w = 1.98610306
  차이 0.00e+00
  lr 만 준 SGD 는 손으로 적은 두 줄과 글자 그대로 같은 일을 한다
  모멘텀이나 웨이트 감쇠를 켜야 비로소 다른 일을 시작한다
```

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
$f(x) = x^3 - 2x^2 + x$에 대해 autograd로 $f'(3)$을 구하고, 손으로 민 도함수와 맞는지 확인하라.

</div>

??? success "풀이"
    ```python
    import torch

    x = torch.tensor(3.0, requires_grad=True)
    (x ** 3 - 2 * x ** 2 + x).backward()
    print(f"  autograd f'(3) = {x.grad.item():.6f}")
    print(f"  손으로 3x^2-4x+1 = {3 * 9 - 4 * 3 + 1}")
    ```

    ```
      autograd f'(3) = 16.000000
      손으로 3x^2-4x+1 = 16
    ```

    $f'(x) = 3x^2 - 4x + 1$이므로 $f'(3) = 27 - 12 + 1 = 16$이다.

    `x`를 `torch.tensor(3.0)`으로 만들면서 `requires_grad=True`를 빼면 `.backward()`가 "element 0 of tensors does not require grad"로 막는다. 기록이 없으니 거꾸로 따라갈 것도 없기 때문이다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
잎 텐서 `w`를 만들고 같은 식의 로스를 세 번 역전파하되 경사를 지우지 말라. 매번 `w.grad`를 찍고 관찰한 값을 설명하라.

</div>

??? success "풀이"
    ```python
    import torch

    w = torch.tensor(2.0, requires_grad=True)
    for i in range(3):
        loss = (w ** 2).sum()        # 매번 새로 만든다
        loss.backward()
        print(f"  {i+1}번째 뒤 w.grad = {w.grad.item()}")
    ```

    ```
      1번째 뒤 w.grad = 4.0
      2번째 뒤 w.grad = 8.0
      3번째 뒤 w.grad = 12.0
    ```

    참 경사는 매번 $2w = 4$인데 `.grad`는 4, 8, 12로 는다. **더해지기 때문이다.**

    루프 안에서 `loss`를 **매번 새로 만드는 것** 이 중요하다. 밖에서 한 번 만들어 두고 세 번 `backward()`를 부르면 연습문제 3의 오류가 난다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
같은 그래프에 `.backward()`를 `retain_graph=True` 없이 두 번 부르면 왜 오류가 나는가? **저장 텐서가 필요 없는 단순한 연산**(`w * 3` 같은)이라면 두 번 불러도 되는지 재어 보라.

</div>

??? success "풀이"
    ```python
    import torch

    w = torch.tensor(2.0, requires_grad=True)
    loss = (w ** 2).sum()
    loss.backward()
    print(f"  첫 backward 뒤 w.grad = {w.grad.item()}")
    try:
        loss.backward()
    except RuntimeError as e:
        print("  같은 그래프를 다시 부르면:", str(e).splitlines()[0][:66])

    w2 = torch.tensor(2.0, requires_grad=True)
    l2 = (w2 ** 2).sum()
    l2.backward(retain_graph=True)
    l2.backward()
    print(f"  retain_graph=True 로 두 번 하면 w.grad = {w2.grad.item()}")

    w3 = torch.tensor(2.0, requires_grad=True)   # 저장 텐서가 필요 없는 연산
    l3 = w3 * 3
    l3.backward()
    try:
        l3.backward()
        print(f"  w*3 은 두 번 불러도 된다: w.grad = {w3.grad.item()}")
    except RuntimeError as e:
        print("  w*3 도 막힌다:", str(e).splitlines()[0][:60])
    ```

    ```
      첫 backward 뒤 w.grad = 4.0
      같은 그래프를 다시 부르면: Trying to backward through the graph a second time (or directly ac
      retain_graph=True 로 두 번 하면 w.grad = 8.0
      w*3 도 막힌다: Trying to backward through the graph a second time (or direc
    ```

    **마지막 줄이 짐작을 뒤집는다.** $w \times 3$의 도함수는 3으로 상수이고 중간값을 저장할 것이 없는데도 두 번째 호출이 막힌다.

    까닭은 PyTorch가 저장 텐서만 푸는 것이 아니라 **그래프 구조 자체를 푼다** 는 데 있다. 첫 `backward()`가 끝나면 `grad_fn` 사이의 연결이 해제되므로, 저장할 중간값이 있었는지와 무관하게 두 번째로 거슬러 올라갈 길이 없다.

    그러니 "저장 텐서가 필요 없으면 괜찮다"는 기억은 쓸 수 없다. 같은 그래프를 두 번 걸으려면 **언제나** `retain_graph=True`가 필요하다.

    `retain_graph=True`는 그래프와 저장 텐서를 다음 호출까지 붙들어 두므로 메모리를 그만큼 더 오래 쓴다. 그리고 두 번 역전파하면 `.grad`가 4에서 8로 **더해진다.** 그래프를 남기는 일과 경사를 지우는 일은 서로 다른 일이다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
본문 6절은 `lr`만 준 `torch.optim.SGD`가 손으로 적은 갱신과 한 비트도 다르지 않음을 보였다. 그렇다면 **모멘텀** 은 무엇을 더하는가? PyTorch의 모멘텀 갱신식을 손으로 구현해 `torch.optim.SGD(momentum=...)`와 맞추고, 모멘텀이 이 문제에서 이득인지 판정하라.

</div>

??? success "풀이"
    PyTorch의 모멘텀은 속도 $v$를 두고 이렇게 간다.

    $$v_{t} = \mu v_{t-1} + g_t, \qquad w_{t+1} = w_t - \eta\, v_t$$

    경사에 $(1-\mu)$를 곱하지 **않는다는** 점을 눈여겨보라. 문헌에 따라 $v_t = \mu v_{t-1} + (1-\mu)g_t$로 적는 곳도 있고, 그 경우 실효 러닝레이트가 달라진다.

    ```python
    import torch

    X = torch.tensor([1., 2., 3., 4., 5.])
    Y = 2 * X

    def hand(mom, steps=20, lr=0.01):
        w = torch.tensor(0.0, requires_grad=True)
        v = torch.zeros(())
        for _ in range(steps):
            w.grad = None
            ((w * X - Y) ** 2).mean().backward()
            with torch.no_grad():
                v = mom * v + w.grad          # PyTorch 의 꼴
                w -= lr * v
        return w.item()

    def torch_sgd(mom, steps=20, lr=0.01):
        w = torch.tensor(0.0, requires_grad=True)
        o = torch.optim.SGD([w], lr=lr, momentum=mom)
        for _ in range(steps):
            o.zero_grad()
            ((w * X - Y) ** 2).mean().backward()
            o.step()
        return w.item()

    print("  모멘텀   손으로         torch.optim.SGD   차이")
    for m in (0.0, 0.5, 0.9):
        a, b = hand(m), torch_sgd(m)
        print(f"  {m:<8} {a:.8f}    {b:.8f}      {abs(a - b):.2e}")
    ```

    ```
      모멘텀   손으로         torch.optim.SGD   차이
      0.0      1.98610306    1.98610306      0.00e+00
      0.5      2.00102162    2.00102162      0.00e+00
      0.9      2.65417814    2.65417838      2.38e-07
    ```

    세 경우 모두 맞는다. 모멘텀 0.9에서 $2 \times 10^{-7}$ 차이는 float32의 누적 반올림이다.

    **이득인가.** 20걸음 뒤의 값을 목표 2.0과 견주면 이렇다.

    | 모멘텀 | 20걸음 뒤 $w$ | 오차 |
    |---|---|---|
    | 0.0 | 1.986103 | 0.0139 |
    | 0.5 | 2.001022 | 0.0010 |
    | 0.9 | 2.654178 | 0.6542 |

    0.5는 열네 배 가까이 좋아졌고 **0.9는 마흔일곱 배 나빠졌다.**

    까닭은 [러닝레이트 쪽](learning_rate.md) 1절의 셈과 같다. 모멘텀은 실효 걸음 폭을 대략 $1/(1-\mu)$배로 키운다. $\mu = 0.9$면 10배이고, 이 문제의 수렴 경계가 $\eta < 0.0909$인데 $\eta = 0.01$의 10배는 0.1로 **경계를 넘는다.** 그래서 지나치고 진동한다.

    **모멘텀을 켜는 것은 러닝레이트를 키우는 것과 비슷한 일이다.** 둘을 따로 고르는 것이 아니라 함께 고른다. $\mu = 0.9$가 흔한 기본값인 까닭은 그 값에 맞춰 러닝레이트를 10분의 1쯤 낮춰 잡기 때문이다.

---

## 정리하며

1. **`grad_fn`이 autograd의 전부다**: 연산마다 "무엇이 나를 만들었나"를 적어 두고 `backward()`가 그 사슬을 거꾸로 간다
2. **autograd는 수치 미분이 아니다**: 손으로 민 식과 차이가 정확히 0이다. $h$도 절단 오차도 없다
3. **루프에서 사라지는 줄은 하나, 생기는 줄은 둘이다**: 경사 식이 빠지고 지우기와 `no_grad`가 들어온다. 순전파·로스·갱신은 그대로다
4. **지우지 않으면 조용히 틀린다**: 목표 2.0에 3.98로 끝났고 예외는 나지 않았다
5. **`no_grad`는 범위, `detach`는 텐서**: 그리고 `detach`는 저장 공간을 함께 쓰므로 고치면 원본이 바뀐다
6. **그래프는 저장 텐서와 함께 풀린다**: $w \times 3$처럼 저장할 것이 없는 연산도 두 번째 `backward()`가 막힌다
7. **`lr`만 준 SGD는 손으로 적은 두 줄과 한 비트도 다르지 않다**: 모멘텀을 켜야 다른 일이 시작되고, 그때는 러닝레이트를 다시 골라야 한다

**참고 문헌**

- Paszke, A. et al. (2017). Automatic differentiation in PyTorch. *NeurIPS Autodiff Workshop*.
- Baydin, A. G. et al. (2018). Automatic Differentiation in Machine Learning: a Survey. *JMLR*, 18(153).
