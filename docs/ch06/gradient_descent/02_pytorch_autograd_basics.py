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
