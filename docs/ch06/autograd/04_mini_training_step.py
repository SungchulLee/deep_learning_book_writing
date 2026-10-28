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

실행: python 04_mini_training_step.py
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
