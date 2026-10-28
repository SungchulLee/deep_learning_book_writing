# 사용자 정의 Autograd 함수 — 역전파를 직접 적는다

PyTorch가 아는 연산을 쓰는 한 미분은 공짜다. 그러나 새 연산을 짜거나, 수치적으로 더 안정한 미분을 손으로 넣거나, 바깥 라이브러리를 그래프에 이어야 할 때가 있다. 그때 쓰는 것이 `torch.autograd.Function`이고, **순전파와 역전파를 둘 다 내가 적는다.**

적는다는 말은 곧 **틀릴 수 있다**는 말이다. 이 쪽은 제곱과 ReLU 둘을 직접 적어 보고, 적은 것이 맞는지 확인하는 방법과 그래도 남는 어긋남 하나를 본다.

## 1. 코드

```python
"""튜토리얼 20: 맞춤 자동 미분 함수 — forward 와 backward 를 직접 적는다."""
import torch

torch.manual_seed(0)


def header(title):
    print(f"\n{'='*70}\n{title}\n{'='*70}")


# ========================================================================
# 두 개의 맞춤 함수
# ========================================================================
class SquareFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)          # backward 가 x 를 쓴다
        return input ** 2

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return grad_output * 2 * input        # VJP: 상류 그래디언트 x 2x


class ReLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0             # x = 0 은 그대로 흘려보낸다
        return grad_input


def main():
    header("1. SquareFunction")
    x = torch.tensor([2.0, 3.0], requires_grad=True)
    y = SquareFunction.apply(x)
    print(f"x      = {x.data.tolist()}")
    print(f"y = x^2 = {y.data.tolist()}   grad_fn = {type(y.grad_fn).__name__}")
    y.sum().backward()
    print(f"x.grad = {x.grad.tolist()}   (2x = {[2*v for v in x.data.tolist()]})")

    header("2. ReLUFunction")
    x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], requires_grad=True)
    y = ReLUFunction.apply(x)
    print(f"x        = {x.data.tolist()}")
    print(f"ReLU(x)  = {y.data.tolist()}")
    y.sum().backward()
    print(f"x.grad   = {x.grad.tolist()}")

    header("3. 맞춘 것이 정말 맞는가 — gradcheck")
    # gradcheck 는 수치 미분과 맞대어 본다. float64 와 작은 입력을 쓴다.
    a = torch.randn(4, dtype=torch.double, requires_grad=True)
    print(f"SquareFunction: {torch.autograd.gradcheck(SquareFunction.apply, (a,))}")
    # ReLU 는 0 에서 꺾이므로 0 근처를 피해 뽑는다
    b = torch.randn(4, dtype=torch.double).sign() * (torch.rand(4, dtype=torch.double) + 0.5)
    b.requires_grad_(True)
    print(f"ReLUFunction:   {torch.autograd.gradcheck(ReLUFunction.apply, (b,))}")

    header("4. 내장과 견주기 — forward 는 같고 backward 는 다르다")
    pts = torch.tensor([-1.0, 0.0, 1.0])

    xa = pts.clone().requires_grad_(True)
    ya = torch.relu(xa)
    ya.sum().backward()

    xb = pts.clone().requires_grad_(True)
    yb = ReLUFunction.apply(xb)
    yb.sum().backward()

    print(f"입력                {pts.tolist()}")
    print(f"forward  내장       {ya.data.tolist()}")
    print(f"forward  맞춤       {yb.data.tolist()}   같은가? {torch.equal(ya.data, yb.data)}")
    print(f"backward 내장       {xa.grad.tolist()}")
    print(f"backward 맞춤       {xb.grad.tolist()}   같은가? {torch.equal(xa.grad, xb.grad)}")
    print(f"\n갈리는 자리는 x = 0 한 점뿐이다: 내장 {xa.grad[1].item()}, 맞춤 {xb.grad[1].item()}")


if __name__ == "__main__":
    main()
```

**출력:**

```
======================================================================
1. SquareFunction
======================================================================
x      = [2.0, 3.0]
y = x^2 = [4.0, 9.0]   grad_fn = SquareFunctionBackward
x.grad = [4.0, 6.0]   (2x = [4.0, 6.0])

======================================================================
2. ReLUFunction
======================================================================
x        = [-2.0, -1.0, 0.0, 1.0, 2.0]
ReLU(x)  = [0.0, 0.0, 0.0, 1.0, 2.0]
x.grad   = [0.0, 0.0, 1.0, 1.0, 1.0]

======================================================================
3. 맞춘 것이 정말 맞는가 — gradcheck
======================================================================
SquareFunction: True
ReLUFunction:   True

======================================================================
4. 내장과 견주기 — forward 는 같고 backward 는 다르다
======================================================================
입력                [-1.0, 0.0, 1.0]
forward  내장       [0.0, 0.0, 1.0]
forward  맞춤       [0.0, 0.0, 1.0]   같은가? True
backward 내장       [0.0, 0.0, 1.0]
backward 맞춤       [0.0, 1.0, 1.0]   같은가? False

갈리는 자리는 x = 0 한 점뿐이다: 내장 0.0, 맞춤 1.0
```

## 2. 무엇을 적어야 하는가

`forward`는 출력을 셈하고, `backward`는 **VJP**를 셈한다 — 출력 쪽에서 내려온 그래디언트 `grad_output`을 받아 입력 쪽 그래디언트를 돌려준다. 제곱이면 $\partial y/\partial x = 2x$이므로 `grad_output * 2 * input`이고, ReLU면 양수 자리는 그대로 흘리고 음수 자리는 0으로 막는다.

`backward`가 쓸 값은 `ctx.save_for_backward`로 맡겨 둔다. 제곱의 미분에는 $x$가 필요하므로 입력을 맡기고, 꼭 필요한 것만 맡긴다 — 안 쓸 값을 맡기면 그만큼 메모리를 오래 붙들고, 쓸 값을 빠뜨리면 미분을 셈할 수 없다.

## 3. 적은 것이 맞는지 어떻게 아는가

손으로 적은 미분은 **수치 미분과 맞대어** 확인한다. `torch.autograd.gradcheck`가 그 일을 한다 — 입력을 아주 조금씩 흔들어 보고 그 차이가 내가 적은 `backward`와 맞는지 본다.

위 출력의 3절이 둘 다 `True`다. 다만 쓰는 데 조건이 둘 있다.

- **`float64`로 넘긴다.** 아주 작은 흔들림의 차이를 보는 일이라 `float32`로는 반올림에 묻힌다.
- **꺾인 자리를 피해 뽑는다.** ReLU는 $x = 0$에서 미분이 없으므로, 그 언저리의 점이 섞이면 수치 미분과 맞을 까닭이 없다. 위 코드가 `b`를 뽑을 때 부호를 정해 두고 크기를 0.5 이상으로 둔 것이 그 때문이다.

## 4. 그래도 남는 어긋남 — $x = 0$

4절이 내장 `torch.relu`와 방금 적은 것을 견준다. **순전파는 똑같고 역전파는 다르다.**

| 입력 | 순전파 (내장 = 맞춤) | 역전파 내장 | 역전파 맞춤 |
|---|---|---|---|
| $-1$ | 0 | 0 | 0 |
| **$0$** | **0** | **0** | **1** |
| $1$ | 1 | 1 | 1 |

갈리는 것은 **한 점뿐**이다. 까닭은 간단하다 — ReLU는 $x = 0$에서 미분이 없고, 왼쪽에서 오면 0이고 오른쪽에서 오면 1이다. 그러니 거기에 무엇을 둘지는 **정하는 사람의 몫**이고, 위 코드는 `grad_input[input < 0] = 0`이라 적어 $0$을 "음수가 아닌 쪽"으로 보냈다. PyTorch는 $0$을 고른다.

!!! warning "forward 가 같다고 같은 함수가 아니다"
    맞춤 함수를 확인할 때 `torch.allclose(built_in, custom)` 한 줄로 "같다"고 적기 쉽다. 그러나 그것은 **순전파만 견준 것이다.** 역전파를 직접 적는 쪽에서 역전파를 견주지 않으면, 맞는지 틀리는지를 정작 묻지 않은 셈이 된다. 위의 표가 그 한 줄로는 보이지 않는다.

    실제로 이 어긋남이 학습을 망가뜨릴 일은 거의 없다. 입력이 정확히 $0.0$인 자리는 드물고, 있어도 한 점이다. 그러나 **"거의 없다"와 "없다"는 다른 말**이고, 맞춤 함수를 쓸 때 어디를 의심해야 하는지는 이런 자리다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
`SquareFunction.forward`에서 `ctx.save_for_backward(input)` 한 줄을 지우면 무슨 일이 생기는가? 돌려 보고, 오류 메시지가 가리키는 것이 무엇인지 적어라.

</div>

??? success "풀이"
    `backward`의 `input, = ctx.saved_tensors`에서 터진다. 맡긴 것이 없으니 꺼낼 것도 없다.

    ```python
    class NoSave(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input):
            return input ** 2            # 맡기지 않는다
        @staticmethod
        def backward(ctx, grad_output):
            input, = ctx.saved_tensors   # 여기서 터진다
            return grad_output * 2 * input

    x = torch.tensor([2.0], requires_grad=True)
    try:
        NoSave.apply(x).sum().backward()
    except ValueError as e:
        print(type(e).__name__, e)
    ```

    ```
    ValueError not enough values to unpack (expected 1, got 0)
    ```

    오류가 가리키는 것은 **`ctx`가 비어 있다**는 사실뿐이다. "미분이 틀렸다"고 말해 주지는 않는다. 그래서 더 고약한 경우는 따로 있다 — 맡기기는 했는데 **엉뚱한 것**을 맡기면 오류 없이 **틀린 그래디언트**가 흐른다. 3절의 `gradcheck`가 그런 자리를 잡으라고 있는 것이다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
`SquareFunction.backward`의 `2 * input`을 일부러 `3 * input`으로 바꾸어 보라. 평소처럼 학습을 돌려도 터지지 않는다. `gradcheck`는 이것을 잡아내는가? 무엇이라 말하는가?

</div>

??? success "풀이"
    잡아낸다. 터지지 않는 까닭은 틀린 그래디언트도 **모양과 자료형이 맞는 텐서**이기 때문이다. autograd는 그것을 그대로 흘려보낼 뿐 옳은지 따지지 않는다.

    ```python
    class WrongSquare(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input):
            ctx.save_for_backward(input)
            return input ** 2
        @staticmethod
        def backward(ctx, grad_output):
            input, = ctx.saved_tensors
            return grad_output * 3 * input     # 2 여야 한다

    a = torch.randn(4, dtype=torch.double, requires_grad=True)
    try:
        torch.autograd.gradcheck(WrongSquare.apply, (a,))
    except Exception as e:
        print(type(e).__name__)
        print(str(e).splitlines()[0])
    ```

    ```
    GradcheckError
    Jacobian mismatch for output 0 with respect to input 0,
    ```

    `gradcheck`가 하는 말은 **"손으로 적은 야코비와 수치로 잰 야코비가 다르다"**이다. 이것이 맞춤 함수에서 가장 중요한 습관이다 — 새로 적은 `backward`는 **학습을 돌리기 전에** 한 번 맞대어 본다. 학습 곡선은 틀린 그래디언트로도 그럴듯하게 내려갈 수 있어서, 거기서는 드러나지 않는다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
4절은 $x = 0$에서 내장이 $0$, 이 쪽의 맞춤이 $1$을 준다고 했다. $0$·$1$·$0.5$ 세 가지로 두는 ReLU를 각각 적고, 세 가지 모두 `gradcheck`를 통과하는지 확인하라. 통과한다면 그것은 무엇을 뜻하는가?

</div>

??? success "풀이"
    세 가지는 `backward`의 비교 한 글자씩만 다르다.

    ```python
    def make_relu(at_zero):
        class R(torch.autograd.Function):
            @staticmethod
            def forward(ctx, input):
                ctx.save_for_backward(input)
                return input.clamp(min=0)
            @staticmethod
            def backward(ctx, grad_output):
                input, = ctx.saved_tensors
                mask = (input > 0).to(grad_output.dtype)
                mask = mask + at_zero * (input == 0).to(grad_output.dtype)
                return grad_output * mask
        return R.apply

    for v in (0.0, 1.0, 0.5):
        b = torch.randn(4, dtype=torch.double).sign() * (torch.rand(4, dtype=torch.double) + 0.5)
        b.requires_grad_(True)
        print(f"at_zero = {v}:  gradcheck {torch.autograd.gradcheck(make_relu(v), (b,))}")
    ```

    ```
    at_zero = 0.0:  gradcheck True
    at_zero = 1.0:  gradcheck True
    at_zero = 0.5:  gradcheck True
    ```

    **셋 다 통과한다.** 위 코드가 $0$을 피해 뽑으므로 `gradcheck`는 $x = 0$에서 무엇을 두든 아예 묻지 않는다.

    그러면 $0$을 **넣어서** 뽑으면 어떻게 되는가. 셋 다 실패할 것 같지만 그렇지 않다.

    ```python
    for v in (0.0, 1.0, 0.5):
        c = torch.tensor([0.0, 1.0, -1.0, 2.0], dtype=torch.double, requires_grad=True)
        try:
            torch.autograd.gradcheck(make_relu(v), (c,))
            print(f"at_zero = {v}:  통과")
        except Exception as e:
            print(f"at_zero = {v}:  {type(e).__name__}")
    ```

    ```
    at_zero = 0.0:  GradcheckError
    at_zero = 1.0:  GradcheckError
    at_zero = 0.5:  통과
    ```

    **0.5만 통과한다.** 까닭은 ReLU에 있지 않고 **재는 방법**에 있다. `gradcheck`는 가운뎃차분으로 미분을 어림하는데, 꺾인 점에서는

    $$
    \frac{f(0+h) - f(0-h)}{2h} = \frac{h - 0}{2h} = \frac{1}{2}
    $$

    이 되어 $h$를 아무리 줄여도 **양쪽 기울기의 평균인 0.5**가 나온다. 그러니 0.5로 적은 쪽만 "수치와 맞는" 것이 된다.

    요점은 이것이다. **`gradcheck`가 통과했다는 말은 내 미분이 옳다는 뜻이 아니라 그 도구가 재는 방식과 맞는다는 뜻이다.** 여기서는 그 도구가 꺾인 점에 제 나름의 관례(평균)를 쥐고 있고, 정작 PyTorch가 고른 $0$은 그 검사에서 **떨어진다.** 셋 다 쓸 수 있는 답이며, 어느 것도 증명으로 고른 것이 아니다.

## 정리하며

맞춤 함수를 쓸 때 손이 가는 곳은 `forward`지만, **틀리는 곳은 거의 언제나 `backward`**다. 그리고 틀려도 터지지 않는다 — 모양만 맞으면 autograd는 그대로 흘려보내고, 학습 곡선도 그럴듯하게 내려간다.

그래서 순서가 정해져 있다. `float64`와 꺾이지 않은 점으로 **`gradcheck`를 먼저 돌리고**, 그다음에 학습을 돌린다. 그러고도 남는 것이 $x = 0$ 같은 자리인데, 그것은 버그가 아니라 **내가 고른 관례**이고 고른 줄 알고 있어야 하는 것이다.
