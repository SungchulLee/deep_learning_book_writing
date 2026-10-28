# 최소 학습 단계 — 다섯 걸음이 왜 그 순서인가

점 세 개로 직선 하나를 맞춘다. 데이터가 작아 걸음마다 무슨 일이 일어나는지 전부
찍어 볼 수 있다.

학습 루프 한 바퀴는 언제나 이 다섯이다.

1. **순전파** — 예측을 만든다
2. **로스** — 얼마나 틀렸는지 홑값으로 적는다
3. **경사 지우기** — 지난 걸음의 경사를 없앤다
4. **역전파** — 경사를 채운다
5. **갱신** — `no_grad` 안에서 파라미터를 고친다

3번이 4번 **앞**에 오는 것이 중요하다. 뒤에 두면 방금 구한 경사를 지우게 되고,
그러면 모델이 **한 걸음도 움직이지 않는다.** 3절에서 그 장면을 본다.

## 1. 코드

```python
"""가장 작은 학습 루프 — 다섯 걸음이 왜 그 순서인가.

점 세 개로 직선 하나를 맞춘다. 데이터가 작아 걸음마다 무슨 일이
일어나는지 전부 찍어 볼 수 있다.

루프 한 바퀴는 언제나 이 다섯이다.

1. 순전파 — 예측을 만든다
2. 로스 — 얼마나 틀렸는지 홑값으로 적는다
3. 경사 지우기 — 지난 걸음의 경사를 없앤다
4. 역전파 — 경사를 채운다
5. 갱신 — no_grad 안에서 파라미터를 고친다

3번이 4번 **앞**에 오는 것이 중요하다. 뒤에 두면 방금 구한 경사를
지우게 된다.

실행: 이 쪽의 코드를 파일로 갈무리해 돌린다
"""
import torch


def header(title):
    print(f"\n=== {title} ===")


X = torch.tensor([[-1.0], [0.0], [1.0]])
Y = torch.tensor([[1.0], [0.0], [-1.0]])      # 참값은 w=-1, b=0


# ========================================================================
# 1. 한 바퀴를 하나씩 찍어 본다
# ========================================================================

def one_step():
    header("1. 한 바퀴")
    torch.manual_seed(0)
    w = torch.randn(1, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    print(f"  처음  w={w.item():+.4f}  b={b.item():+.4f}")

    y_hat = X * w + b                                  # 1. 순전파
    loss = ((y_hat - Y) ** 2).mean()                   # 2. 로스
    print(f"  예측  {[round(v, 4) for v in y_hat.flatten().tolist()]}")
    print(f"  참값  {[round(v, 4) for v in Y.flatten().tolist()]}")
    print(f"  로스  {loss.item():.6f}")

    print(f"  지우기 전 w.grad = {w.grad}")            # 3. 지우기
    if w.grad is not None:
        w.grad.zero_()
    loss.backward()                                    # 4. 역전파
    print(f"  역전파 뒤 w.grad = {w.grad.item():+.6f}  b.grad = {b.grad.item():+.6f}")

    # 손으로: dL/dw = (2/n) * sum (y_hat - y) * x
    hand_w = (2 * (y_hat.detach() - Y) * X).mean()
    hand_b = (2 * (y_hat.detach() - Y)).mean()
    print(f"  손으로    w.grad = {hand_w.item():+.6f}  b.grad = {hand_b.item():+.6f}")

    with torch.no_grad():                              # 5. 갱신
        w -= 0.1 * w.grad
        b -= 0.1 * b.grad
    print(f"  갱신 뒤  w={w.item():+.4f}  b={b.item():+.4f}")


# ========================================================================
# 2. 세 걸음 돌려 본다
# ========================================================================

def three_steps():
    header("2. 세 걸음")
    torch.manual_seed(0)
    w = torch.randn(1, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    for step in range(1, 4):
        loss = ((X * w + b - Y) ** 2).mean()
        if w.grad is not None:
            w.grad.zero_()
            b.grad.zero_()
        loss.backward()
        with torch.no_grad():
            w -= 0.1 * w.grad
            b -= 0.1 * b.grad
        print(f"  {step}걸음  로스 {loss.item():.6f}  "
              f"w {w.item():+.4f}  b {b.item():+.4f}")
    print("  참값은 w=-1, b=0 이다. 로스가 줄고 w 가 -1 쪽으로 간다")


# ========================================================================
# 3. 지우기를 역전파 뒤에 두면
# ========================================================================

def wrong_order():
    header("3. 지우기를 역전파 뒤에 두면")
    for label, clear_first in (("지우고 역전파", True), ("역전파하고 지우기", False)):
        torch.manual_seed(0)
        w = torch.randn(1, requires_grad=True)
        for _ in range(3):
            loss = ((X * w - Y) ** 2).mean()
            if clear_first and w.grad is not None:
                w.grad.zero_()
            loss.backward()
            if not clear_first:
                w.grad.zero_()          # 방금 구한 경사를 지운다
            with torch.no_grad():
                w -= 0.1 * w.grad
        print(f"  {label:18s} 세 걸음 뒤 w = {w.item():+.6f}")
    print("  뒤에 두면 w 가 처음 값에서 꼼짝도 하지 않는다")


# ========================================================================
# 4. no_grad 를 빼면
# ========================================================================

def without_no_grad():
    header("4. no_grad 를 빼면")
    torch.manual_seed(0)
    w = torch.randn(1, requires_grad=True)
    ((X * w - Y) ** 2).mean().backward()
    try:
        w -= 0.1 * w.grad
    except RuntimeError as e:
        print("  w -= ... 를 그냥 쓰면:", str(e).splitlines()[0])
    with torch.no_grad():
        w -= 0.1 * w.grad
    print("  no_grad 안에서는 된다. w =", round(w.item(), 6))
    print("  갱신은 학습의 일부가 아니라 학습 바깥의 일이다")
    print("  그래서 그래프에 기록하지 않는다")


# ========================================================================
# 5. 옵티마이저가 대신해 준다
# ========================================================================

def with_optimizer():
    header("5. 옵티마이저로 적으면")
    torch.manual_seed(0)
    w = torch.randn(1, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.SGD([w, b], lr=0.1)
    for _ in range(3):
        opt.zero_grad()
        ((X * w + b - Y) ** 2).mean().backward()
        opt.step()
    print(f"  옵티마이저로  w {w.item():+.6f}  b {b.item():+.6f}")

    torch.manual_seed(0)
    w2 = torch.randn(1, requires_grad=True)
    b2 = torch.zeros(1, requires_grad=True)
    for _ in range(3):
        if w2.grad is not None:
            w2.grad.zero_()
            b2.grad.zero_()
        ((X * w2 + b2 - Y) ** 2).mean().backward()
        with torch.no_grad():
            w2 -= 0.1 * w2.grad
            b2 -= 0.1 * b2.grad
    print(f"  손으로        w {w2.item():+.6f}  b {b2.item():+.6f}")
    print("  같은가:", torch.allclose(w, w2) and torch.allclose(b, b2))
    print("  opt.zero_grad() 가 3번, opt.step() 이 5번을 맡는다")


def main():
    one_step()
    three_steps()
    wrong_order()
    without_no_grad()
    with_optimizer()


if __name__ == "__main__":
    main()
```

**출력:**

```
=== 1. 한 바퀴 ===
  처음  w=+1.5410  b=+0.0000
  예측  [-1.541, 0.0, 1.541]
  참값  [1.0, 0.0, -1.0]
  로스  4.304441
  지우기 전 w.grad = None
  역전파 뒤 w.grad = +3.387995  b.grad = +0.000000
  손으로    w.grad = +3.387995  b.grad = +0.000000
  갱신 뒤  w=+1.2022  b=+0.0000

=== 2. 세 걸음 ===
  1걸음  로스 4.304441  w +1.2022  b +0.0000
  2걸음  로스 3.233113  w +0.9086  b +0.0000
  3걸음  로스 2.428427  w +0.6541  b +0.0000
  참값은 w=-1, b=0 이다. 로스가 줄고 w 가 -1 쪽으로 간다

=== 3. 지우기를 역전파 뒤에 두면 ===
  지우고 역전파            세 걸음 뒤 w = +0.654094
  역전파하고 지우기          세 걸음 뒤 w = +1.540996
  뒤에 두면 w 가 처음 값에서 꼼짝도 하지 않는다

=== 4. no_grad 를 빼면 ===
  w -= ... 를 그냥 쓰면: a leaf Variable that requires grad is being used in an in-place operation.
  no_grad 안에서는 된다. w = 1.202197
  갱신은 학습의 일부가 아니라 학습 바깥의 일이다
  그래서 그래프에 기록하지 않는다

=== 5. 옵티마이저로 적으면 ===
  옵티마이저로  w +0.654094  b +0.000000
  손으로        w +0.654094  b +0.000000
  같은가: True
  opt.zero_grad() 가 3번, opt.step() 이 5번을 맡는다
```

## 2. 한 바퀴를 손으로 따라가기

```
처음  w=+1.5410  b=+0.0000
예측  [-1.541, 0.0, 1.541]
참값  [1.0, 0.0, -1.0]
로스  4.304441
```

$x \in \{-1, 0, 1\}$에 대해 $y \in \{1, 0, -1\}$이므로 참값은 $w = -1$, $b = 0$이다.
처음 $w$가 $+1.54$이니 예측이 부호부터 반대다.

```
역전파 뒤 w.grad = +3.387995   b.grad = +0.000000
손으로    w.grad = +3.387995   b.grad = +0.000000
```

손으로 센 값과 맞는다. $L = \frac{1}{n}\sum(\hat{y} - y)^2$이므로

$$
\frac{\partial L}{\partial w} = \frac{2}{n}\sum (\hat{y}_i - y_i)\, x_i, \qquad
\frac{\partial L}{\partial b} = \frac{2}{n}\sum (\hat{y}_i - y_i)
$$

$b$의 경사가 정확히 0인 것은 우연이 아니다. 데이터가 원점을 기준으로 대칭이라
오차의 합이 0이기 때문이다. 그래서 `b`는 세 걸음 내내 0에 머문다.

## 3. 경사를 지우는 자리가 역전파 **앞**이다

```
지우고 역전파        세 걸음 뒤 w = +0.654094
역전파하고 지우기     세 걸음 뒤 w = +1.540996
```

둘째 줄의 `+1.540996`은 **처음 값 그대로**다. 세 걸음을 돌았는데 조금도 움직이지
않았다.

까닭은 뻔하다. 역전파가 `w.grad`를 채워 놓았는데 그것을 지우고 나서 걸으니,
걸음이 언제나 0이다. 로스는 계속 계산되고 오류도 안 나므로, **학습이 안 되는데
돌아가기는 한다.**

> 이 버그는 로스 곡선이 평평한 직선으로 나오는 것으로 드러난다. 학습이 안 될 때
> 러닝레이트나 모델을 의심하기 전에 **순서부터** 볼 자리다.

## 4. 갱신에 `no_grad`가 필요한 까닭

```
w -= ... 를 그냥 쓰면: a leaf Variable that requires grad is being used
                      in an in-place operation.
```

`w`는 잎이고 미분이 따라붙어 있다. 그 값을 제자리에서 바꾸면 이미 쌓아 둔 그래프가
가리키는 값과 달라지므로 파이토치가 막는다.

`torch.no_grad()` 안에서는 이 검사를 넘긴다. 그래도 되는 까닭은 **갱신이 학습의
일부가 아니라 학습 바깥의 일**이기 때문이다. $w \leftarrow w - \eta \nabla w$를
다시 미분할 일은 없다. 그러니 기록할 까닭도 없다.

`no_grad`를 안 쓰고 새 텐서를 만드는 길도 있다 — `w = w - 0.1 * w.grad`. 그런데
이러면 `w`가 **잎이 아니게 되어** 다음 걸음에서 `.grad`를 못 받는다. 그래서
제자리 갱신이 맞고, 제자리 갱신이려면 `no_grad`가 필요하다.

## 5. 옵티마이저는 3번과 5번을 맡는다

```
옵티마이저로  w +0.654094  b +0.000000
손으로        w +0.654094  b +0.000000
같은가: True
```

소수점 여섯째 자리까지 같다. `torch.optim.SGD`가 하는 일은 손으로 적은 그
두 줄이 전부다.

```python
opt.zero_grad()     # 3. 경사 지우기
loss.backward()     # 4. 역전파
opt.step()          # 5. no_grad 안에서 갱신
```

옵티마이저를 쓰는 값은 **파라미터가 많아질 때** 나온다. 모델 하나에 텐서가
수백 개인데 일일이 `if p.grad is not None: p.grad.zero_()`를 적을 수는 없다.
그리고 모멘텀이나 Adam처럼 **상태를 들고 있어야 하는** 갱신은 손으로 적기
어렵다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
2절에서 `b.grad`가 정확히 0이었다. 데이터를 $y \in \{2, 1, 0\}$으로 바꾸면
어떻게 되는가? **먼저 짐작하고** 확인하라.

</div>

??? success "풀이"
    참값이 달라진다. $x \in \{-1, 0, 1\}$에 $y \in \{2, 1, 0\}$이면 $w = -1$,
    $b = 1$인 직선이다. 이제 $b$가 0이 아니므로 경사도 0이 아닐 것이다.

    ```python
    X = torch.tensor([[-1.0], [0.0], [1.0]])
    for name, Y in (("y = [1, 0, -1]", torch.tensor([[1.0], [0.0], [-1.0]])),
                    ("y = [2, 1,  0]", torch.tensor([[2.0], [1.0], [0.0]]))):
        torch.manual_seed(0)
        w = torch.randn(1, requires_grad=True)
        b = torch.zeros(1, requires_grad=True)
        ((X * w + b - Y) ** 2).mean().backward()
        print(f"  {name}  w.grad {w.grad.item():+.6f}  b.grad {b.grad.item():+.6f}")
    ```

    ```
      y = [1, 0, -1]  w.grad +3.387995  b.grad +0.000000
      y = [2, 1,  0]  w.grad +3.387995  b.grad -2.000000
    ```

    **`w.grad`는 그대로이고 `b.grad`만 달라졌다.**

    $y$ 전체를 1씩 올린 것은 직선을 위로 평행이동한 것이므로 **기울기에 대한
    요구는 바뀌지 않는다.** 그래서 $\partial L/\partial w$가 같다.

    $b$ 쪽은 $\frac{2}{n}\sum(\hat y_i - y_i)$인데, 예측이 그대로인 채 목표만
    1씩 올라갔으니 오차 평균이 $-1$이 되어 경사가 $-2$다. 음수이므로 $b$를
    **올리는** 방향으로 움직인다 — 맞는 방향이다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
4절에서 `w = w - 0.1 * w.grad`로 적으면 `w`가 잎이 아니게 된다고 했다. 확인하고,
**두 번째 걸음에서 무슨 일이 나는지** 보라.

</div>

??? success "풀이"
    ```python
    X = torch.tensor([[-1.0], [0.0], [1.0]])
    Y = torch.tensor([[1.0], [0.0], [-1.0]])
    torch.manual_seed(0)
    w = torch.randn(1, requires_grad=True)
    print(f"  처음   is_leaf={w.is_leaf}  requires_grad={w.requires_grad}")

    ((X * w - Y) ** 2).mean().backward()
    w = w - 0.1 * w.grad                    # 새 텐서를 만든다
    print(f"  갱신 뒤 is_leaf={w.is_leaf}  requires_grad={w.requires_grad}")

    ((X * w - Y) ** 2).mean().backward()
    print(f"  두 번째 걸음의 w.grad = {w.grad}")
    ```

    ```
      처음   is_leaf=True  requires_grad=True
      갱신 뒤 is_leaf=False  requires_grad=True
      두 번째 걸음의 w.grad = None
    ```

    **`w.grad`가 `None`이다.** 그런데 오류는 나지 않는다.

    `w = w - 0.1 * w.grad`가 새 텐서를 만들었고, 그것은 연산의 결과이므로 잎이
    아니다. 「스칼라 로스의 역전파」에서 본 대로 잎이 아니면 `.grad`를 받지
    않는다.

    그러니 두 번째 걸음에서 `w -= 0.1 * w.grad`를 하려 하면 `None`에 곱하기를
    시도해 `TypeError`로 터지거나, 옵티마이저를 쓴다면 **그 파라미터를 조용히
    건너뛴다.** 옵티마이저는 처음에 넘겨받은 텐서 객체를 들고 있는데, 이제 `w`라는
    이름은 다른 객체를 가리키고 있기 때문이다.

    이것이 파라미터 갱신을 **반드시 제자리로** 해야 하는 까닭이다. 그리고 제자리
    갱신은 잎에 막혀 있으므로 `no_grad`가 필요하다. 둘이 맞물려 있다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
3절에서 순서를 틀리면 모델이 안 움직였다. 그런데 **옵티마이저를 쓰면서** 같은
실수를 하면 어떻게 되는가? `opt.zero_grad()`를 `opt.step()` **뒤에** 두는 경우와
`loss.backward()` 뒤에 두는 경우를 각각 보라.

</div>

??? success "풀이"
    ```python
    X = torch.tensor([[-1.0], [0.0], [1.0]])
    Y = torch.tensor([[1.0], [0.0], [-1.0]])

    def run(where):
        torch.manual_seed(0)
        w = torch.randn(1, requires_grad=True)
        opt = torch.optim.SGD([w], lr=0.1)
        for _ in range(3):
            if where == "앞":
                opt.zero_grad()
            ((X * w - Y) ** 2).mean().backward()
            if where == "역전파 뒤":
                opt.zero_grad()
            opt.step()
            if where == "step 뒤":
                opt.zero_grad()
        return w.item()

    for where in ("앞", "역전파 뒤", "step 뒤"):
        print(f"  zero_grad 를 {where:10s} 에 두면  w = {run(where):+.6f}")
    ```

    ```
      zero_grad 를 앞          에 두면  w = +0.654094
      zero_grad 를 역전파 뒤      에 두면  w = +1.540996
      zero_grad 를 step 뒤     에 두면  w = +0.654094
    ```

    세 가지 가운데 **둘은 맞고 하나는 틀린다.**

    - **앞에 두기** — 맞다. 가장 흔한 꼴이다
    - **`step()` 뒤에 두기** — 맞다. 다음 바퀴가 시작될 때는 이미 비어 있으니
      "앞에 두기"와 같은 일이다
    - **역전파 뒤에 두기** — 틀린다. 방금 구한 경사를 지우고 걸으므로 `step()`이
      아무것도 안 한다

    `step()` 뒤에 두는 꼴도 자주 보이고, 결과가 같다. 다만 **루프 바깥에서
    `backward()`를 한 번 더 부르는 코드**가 섞이면 앞에 두는 쪽이 안전하다 —
    "걷기 직전에 비어 있음을 보장한다"가 더 지키기 쉬운 약속이기 때문이다.

    그리고 틀린 경우가 **오류 없이 조용히** 지나간다는 점이 이 문제의 요점이다.
    `w`가 처음 값 그대로인 것을 눈으로 확인하지 않으면 모른다.
