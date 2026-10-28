"""nn.Module 로 옮기기 — 추상화를 걷어 내면 같은 수가 나오는가.

「완전한 선형 회귀」에서 손으로 관리한 세 가지를 라이브러리에 넘긴다.

| 손으로 | 라이브러리 |
|---|---|
| `w`, `b` 를 직접 만들고 들고 다님 | `nn.Linear(1, 1)` |
| `((pred - y) ** 2).mean()` | `nn.MSELoss()` |
| `with no_grad(): w -= lr * w.grad` | `torch.optim.SGD` |

편해지는 것은 분명한데, **셈이 그대로인지**는 따로 물어야 한다.
이 쪽은 두 길을 나란히 돌려 소수점까지 견준다.

실행: python 06_linear_regression_with_module.py
"""
import torch
import torch.nn as nn


def header(title):
    print(f"\n=== {title} ===")


TRUE_W, TRUE_B, N = 2.0, 1.0, 100


def make_data(seed=42):
    torch.manual_seed(seed)
    x = torch.randn(N, 1)
    return x, TRUE_W * x + TRUE_B + 0.2 * torch.randn(N, 1)


# ========================================================================
# 1. 손으로 짠 것
# ========================================================================

def by_hand(x, y, lr=0.1, epochs=200, w0=None, b0=None):
    w = w0.clone().requires_grad_(True)
    b = b0.clone().requires_grad_(True)
    for _ in range(epochs):
        loss = ((x * w + b - y) ** 2).mean()
        if w.grad is not None:
            w.grad.zero_()
            b.grad.zero_()
        loss.backward()
        with torch.no_grad():
            w -= lr * w.grad
            b -= lr * b.grad
    return w.item(), b.item(), loss.item()


# ========================================================================
# 2. 라이브러리로 짠 것
# ========================================================================

def by_module(x, y, lr=0.1, epochs=200, w0=None, b0=None):
    model = nn.Linear(1, 1)
    with torch.no_grad():                 # 같은 자리에서 시작하게 맞춘다
        model.weight.copy_(w0.view(1, 1))
        model.bias.copy_(b0)
    criterion = nn.MSELoss()
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    for _ in range(epochs):
        loss = criterion(model(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model.weight.item(), model.bias.item(), loss.item()


def compare():
    header("1. 두 길이 같은 수를 주는가")
    x, y = make_data()
    torch.manual_seed(0)
    w0, b0 = torch.randn(1), torch.zeros(1)
    hw, hb, hl = by_hand(x, y, w0=w0, b0=b0)
    mw, mb, ml = by_module(x, y, w0=w0, b0=b0)
    print(f"  손으로      w {hw:.8f}  b {hb:.8f}  로스 {hl:.8f}")
    print(f"  nn.Module   w {mw:.8f}  b {mb:.8f}  로스 {ml:.8f}")
    print(f"  차이        w {abs(hw - mw):.2e}  b {abs(hb - mb):.2e}")
    print("  같은 자리에서 시작하면 소수점 끝까지 같다")


# ========================================================================
# 3. 무엇이 무엇을 대신하는가
# ========================================================================

def what_replaces_what():
    header("2. 바꿔치기")
    x, y = make_data()
    model = nn.Linear(1, 1)

    print("  파라미터는 모델이 들고 있다")
    for name, p in model.named_parameters():
        print(f"    {name:8s} 모양 {tuple(p.shape)}  requires_grad {p.requires_grad}")

    pred = model(x)
    hand_loss = ((pred - y) ** 2).mean()
    mse_loss = nn.MSELoss()(pred, y)
    print(f"  로스  손으로 {hand_loss.item():.8f}  MSELoss {mse_loss.item():.8f}"
          f"  같은가 {torch.allclose(hand_loss, mse_loss)}")

    print("  모델의 순전파도 손으로 적은 것과 같다:",
          torch.allclose(model(x), x @ model.weight.t() + model.bias))


# ========================================================================
# 4. 모델이 커지면 손으로는 못 한다
# ========================================================================

def why_bother():
    header("3. 모델이 커지면")
    small = nn.Linear(1, 1)
    big = nn.Sequential(nn.Linear(784, 256), nn.ReLU(),
                        nn.Linear(256, 128), nn.ReLU(),
                        nn.Linear(128, 10))
    for name, m in (("nn.Linear(1,1)", small), ("3층 그물", big)):
        n_t = sum(1 for _ in m.parameters())
        n_p = sum(p.numel() for p in m.parameters())
        print(f"  {name:16s} 텐서 {n_t}개   수 {n_p:,}개")
    print("  텐서 여섯 개를 손으로 zero_() 하고 갱신하는 코드는 쓸 수 없다")
    print("  opt.zero_grad() 와 opt.step() 이 그것을 한 줄로 만든다")

    opt = torch.optim.SGD(big.parameters(), lr=0.1, momentum=0.9)
    x = torch.randn(4, 784)
    big(x).sum().backward()
    opt.step()
    state = sum(len(s) for s in opt.state.values())
    print(f"  게다가 모멘텀은 상태를 들고 있어야 한다. 버퍼 {state}개")


# ========================================================================
# 5. 추상화가 숨기는 것
# ========================================================================

def what_it_hides():
    header("4. 숨겨지는 것")
    model = nn.Linear(1, 1)
    print("  model.weight 의 모양:", tuple(model.weight.shape), "= (out, in)")
    print("  그래서 순전파가 x @ W.t() 다 —「Linear 웨이트 관례」의 그 전치")

    x, y = make_data()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    loss = nn.MSELoss()(model(x), y)
    loss.backward()
    before = model.weight.item()
    opt.step()
    after = model.weight.item()
    print(f"  opt.step() 한 걸음: {before:.6f} -> {after:.6f}")
    print(f"  손으로 w - lr*grad : "
          f"{before - 0.1 * model.weight.grad.item():.6f}")
    print("  step() 이 하는 일은 no_grad 안의 뺄셈 한 줄이다")

    print("  reduction 을 sum 으로 바꾸면 로스도 경사도 N 배가 된다")
    for red in ("mean", "sum"):
        m2 = nn.Linear(1, 1)
        with torch.no_grad():
            m2.weight.copy_(model.weight)
            m2.bias.copy_(model.bias)
        nn.MSELoss(reduction=red)(m2(x), y).backward()
        print(f"    reduction={red:5s} weight.grad {m2.weight.grad.item():12.4f}")


def main():
    compare()
    what_replaces_what()
    why_bother()
    what_it_hides()


if __name__ == "__main__":
    main()
