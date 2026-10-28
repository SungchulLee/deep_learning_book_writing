"""경사 누적 — 기본값이 더하기인 까닭.

`.grad` 는 덮어쓰지 않고 **더한다**. 그래서 학습 루프마다 지워야 하고,
지우지 않으면 지난 걸음의 경사가 섞인다.

버그처럼 보이지만 기능이다. 배치를 쪼개 여러 번 역전파하고 한 번만
걸으면 큰 배치와 **같은 경사**가 나온다. 이 쪽은 그것이 정말 같은지
재어 본다. 다만 `mean` 로스에서는 쪼갠 수로 나누어 주어야 한다.

실행: python 02_gradient_accumulation.py
"""
import torch


def header(title):
    print(f"\n=== {title} ===")


# ========================================================================
# 1. 지우지 않으면 쌓인다
# ========================================================================

def accumulates():
    header("1. 지우지 않으면 쌓인다")
    x = torch.tensor(2.0, requires_grad=True)
    print("  처음 x.grad :", x.grad)
    (x ** 2).backward()
    print("  loss1 = x^2 뒤 :", x.grad.item(), " (2x = 4)")
    (3 * x).backward()
    print("  loss2 = 3x 뒤  :", x.grad.item(), " (4 + 3 = 7)")
    print("  두 로스의 경사가 더해졌다. 덮어쓰지 않는다")


# ========================================================================
# 2. 지우는 두 길이 걸음을 다르게 만든다
# ========================================================================

def two_ways_to_clear():
    header("2. zero_() 와 None")
    x = torch.tensor(2.0, requires_grad=True)
    (x ** 2).backward()
    x.grad.zero_()
    print("  zero_() 뒤 :", x.grad, " 자료형", type(x.grad).__name__)
    x.grad = None
    print("  None 으로   :", x.grad, " 자료형", type(x.grad).__name__)

    # 모멘텀이 있으면 이 둘이 다른 걸음을 만든다
    import torch.nn as nn
    got = []
    for set_to_none in (True, False):
        torch.manual_seed(0)
        m = nn.Linear(3, 1)
        opt = torch.optim.SGD(m.parameters(), lr=0.1, momentum=0.9)
        xb, yb = torch.randn(5, 3), torch.randn(5, 1)
        nn.functional.mse_loss(m(xb), yb).backward()
        opt.step()
        opt.zero_grad(set_to_none=set_to_none)
        opt.step()                      # 경사가 빈 채로 한 걸음 더
        got.append(m.weight.detach().flatten().clone())
    print("  모멘텀을 켜고 빈 걸음을 한 번 더 걸으면")
    print("    set_to_none=True :", [f"{v:+.6f}" for v in got[0].tolist()])
    print("    set_to_none=False:", [f"{v:+.6f}" for v in got[1].tolist()])
    print("    차이", f"{(got[0] - got[1]).abs().max():.6f}",
          "— None 은 건너뛰고 0 은 모멘텀이 민다")


# ========================================================================
# 3. 일부러 쌓기 — 큰 배치와 같은가
# ========================================================================

def micro_batching():
    header("3. 쪼개서 쌓으면 큰 배치와 같은가")
    torch.manual_seed(0)
    X, Y = torch.randn(40, 3), torch.randn(40, 1)
    W0 = torch.randn(3, 1)

    def whole(reduction):
        w = W0.clone().requires_grad_(True)
        sq = (X @ w - Y) ** 2
        (sq.mean() if reduction == "mean" else sq.sum()).backward()
        return w.grad.clone()

    def split(reduction, k=4, rescale=True):
        w = W0.clone().requires_grad_(True)
        n = len(X) // k
        for i in range(k):
            sq = (X[i * n:(i + 1) * n] @ w - Y[i * n:(i + 1) * n]) ** 2
            part = sq.mean() if reduction == "mean" else sq.sum()
            if rescale and reduction == "mean":
                part = part / k          # 평균의 평균이 되지 않게 맞춘다
            part.backward()
        return w.grad.clone()

    for red in ("sum", "mean"):
        a, b = whole(red), split(red)
        print(f"  {red:5s} 통째로 {[round(v, 6) for v in a.flatten().tolist()]}")
        print(f"        쪼개서 {[round(v, 6) for v in b.flatten().tolist()]}")
        print(f"        같은가 {torch.allclose(a, b, atol=1e-5)}")
    bad = split("mean", rescale=False)
    print("  mean 인데 k 로 안 나누면 같은가:",
          torch.allclose(whole("mean"), bad, atol=1e-5))
    print("    배수:", round((bad / whole("mean")).flatten()[0].item(), 4),
          "= 쪼갠 수 k")


# ========================================================================
# 4. 로스가 여럿일 때
# ========================================================================

def several_losses():
    header("4. 로스가 여럿일 때")
    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    ((x - 2) ** 2).mean().add(0.1 * x.abs().mean()).backward()
    combined = x.grad.clone()
    print("  합쳐서 한 번 :", [round(v, 6) for v in combined.tolist()])

    y = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    mse = ((y - 2) ** 2).mean()
    l1 = y.abs().mean()
    mse.backward(retain_graph=True)        # 그래프를 남겨 두어야 한다
    print("  MSE 만 하고   :", [round(v, 6) for v in y.grad.tolist()])
    (0.1 * l1).backward()
    print("  L1 까지 더하면:", [round(v, 6) for v in y.grad.tolist()])
    print("  합쳐서 한 번과 같은가:", torch.allclose(combined, y.grad))
    print("  쌓이는 성질 덕분에 로스를 따로 역전파해도 된다")
    print("  다만 같은 그래프를 두 번 걸으니 retain_graph=True 가 필요하다")


def main():
    accumulates()
    two_ways_to_clear()
    micro_batching()
    several_losses()


if __name__ == "__main__":
    main()
