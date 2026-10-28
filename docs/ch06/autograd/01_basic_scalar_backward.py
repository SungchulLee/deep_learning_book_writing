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
