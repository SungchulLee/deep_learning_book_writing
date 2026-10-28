"""밑바닥부터 짠 선형 회귀 — 경사 하강은 어디로 가는가.

`y = 2x + 1` 에 잡음을 섞어 점 백 개를 만들고, 경사 하강으로 `w` 와
`b` 를 찾는다. 「최소 학습 단계」의 루프를 그대로 쓰되 이번에는
**어디에 닿는지**를 묻는다.

흔히 "참값을 찾는다" 고 말하는데 정확하지 않다. 경사 하강이 닿는
곳은 **최소제곱해**이고, 그것은 잡음 때문에 참값과 다르다. 이 쪽은
셋을 나란히 놓고 그 차이를 잰다.

실행: python 05_complete_linear_regression.py
"""
import torch


def header(title):
    print(f"\n=== {title} ===")


TRUE_W, TRUE_B, N, NOISE = 2.0, 1.0, 100, 0.2


def make_data(noise=NOISE, seed=42):
    torch.manual_seed(seed)
    x = torch.randn(N, 1)
    y = TRUE_W * x + TRUE_B + noise * torch.randn(N, 1)
    return x, y


def closed_form(x, y):
    """최소제곱해를 바로 구한다. 경사 하강이 가야 할 곳이다."""
    X = torch.cat([x, torch.ones_like(x)], dim=1)
    beta = torch.linalg.lstsq(X, y).solution
    return beta[0].item(), beta[1].item()


# ========================================================================
# 1. 데이터
# ========================================================================

def show_data():
    header("1. 데이터")
    x, y = make_data()
    print(f"  참 모델  y = {TRUE_W}x + {TRUE_B} + N(0, {NOISE}^2)")
    print(f"  표본 {N}개   x 범위 [{x.min():.3f}, {x.max():.3f}]")
    print(f"               y 범위 [{y.min():.3f}, {y.max():.3f}]")
    return x, y


# ========================================================================
# 2. 학습
# ========================================================================

def train(x, y, lr=0.1, epochs=1000, log=()):
    torch.manual_seed(0)
    w = torch.randn(1, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    history = []
    for epoch in range(1, epochs + 1):
        loss = ((x * w + b - y) ** 2).mean()
        if w.grad is not None:
            w.grad.zero_()
            b.grad.zero_()
        loss.backward()
        with torch.no_grad():
            w -= lr * w.grad
            b -= lr * b.grad
        history.append(loss.item())
        if epoch in log:
            print(f"  {epoch:5d} 걸음  로스 {loss.item():.6f}  "
                  f"w {w.item():+.6f}  b {b.item():+.6f}")
    return w.item(), b.item(), history


def run_training(x, y):
    header("2. 학습")
    w, b, hist = train(x, y, log=(1, 10, 100, 500, 1000))
    print(f"  로스가 {hist[0]:.4f} 에서 {hist[-1]:.6f} 로 내려갔다")
    return w, b


# ========================================================================
# 3. 어디에 닿았는가
# ========================================================================

def where_did_it_land(x, y, w, b):
    header("3. 참값 · 해석해 · 경사 하강")
    cw, cb = closed_form(x, y)
    print(f"  참값      w={TRUE_W:.6f}  b={TRUE_B:.6f}")
    print(f"  최소제곱해 w={cw:.6f}  b={cb:.6f}")
    print(f"  경사 하강  w={w:.6f}  b={b:.6f}")
    print(f"  해석해와의 차  w {abs(w - cw):.2e}   b {abs(b - cb):.2e}")
    print(f"  참값과의 차    w {abs(w - TRUE_W):.6f}   b {abs(b - TRUE_B):.6f}")
    print("  경사 하강은 참값이 아니라 해석해로 간다 — 소수점 일곱 자리까지")
    print("  남은 차이는 알고리즘의 잘못이 아니라 잡음이 만든 것이다")


# ========================================================================
# 4. 잡음을 줄이면 해석해가 참값으로 간다
# ========================================================================

def noise_sweep():
    header("4. 잡음과 해석해")
    for noise in (0.4, 0.2, 0.05, 0.0):
        x, y = make_data(noise=noise)
        cw, cb = closed_form(x, y)
        print(f"  잡음 {noise:.2f}  해석해 w={cw:.6f} b={cb:.6f}  "
              f"참값과의 차 {abs(cw - TRUE_W):.6f}")
    print("  잡음이 0 이면 해석해가 참값과 정확히 같다")
    print("  그러니 '학습이 덜 되었나' 와 '데이터가 그렇게 생겼나' 는 다른 물음이다")


# ========================================================================
# 5. 러닝레이트가 정하는 것
# ========================================================================

def learning_rates():
    header("5. 러닝레이트")
    x, y = make_data()
    cw, cb = closed_form(x, y)
    for lr in (0.001, 0.01, 0.1, 0.5, 1.1):
        w, b, hist = train(x, y, lr=lr, epochs=200)
        gap = abs(w - cw) + abs(b - cb)
        state = "발산" if not (gap == gap) or gap > 1e3 else f"해석해와 {gap:.2e}"
        print(f"  lr={lr:<6} 200 걸음 뒤 로스 {hist[-1]:12.6f}   {state}")
    print("  너무 작으면 못 닿고 너무 크면 튄다. 가운데가 넓다")


def main():
    x, y = show_data()
    w, b = run_training(x, y)
    where_did_it_land(x, y, w, b)
    noise_sweep()
    learning_rates()


if __name__ == "__main__":
    main()
