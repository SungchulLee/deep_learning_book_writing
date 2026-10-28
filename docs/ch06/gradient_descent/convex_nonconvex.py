"""볼록과 비볼록 — 무엇이 실제로 달라지는가.

"볼록하면 국소 최솟값이 전역 최솟값" 은 정의에서 바로 나오는 말이라
외우기 쉽지만, 그래서 무엇이 달라지는지는 잘 와닿지 않는다. 이 쪽은
**출발점을 바꿔 보는 것** 으로 그 차이를 드러낸다.

- 볼록한 문제는 어디서 출발해도 같은 곳에 닿는가
- 비볼록한 문제는 얼마나 달라지는가
- 뉴럴 넷트웍의 비볼록성은 어디서 오는가 (대칭을 수로 확인한다)
- 헤세 행렬의 고윳값으로 볼록성을 가려낼 수 있는가
- 안장점에서 경사 하강법은 정말 멈추는가

실행: python convex_nonconvex.py
"""
import numpy as np
import torch
import torch.nn as nn


def header(title):
    print(f"\n=== {title} ===")


# ========================================================================
# 1. 볼록성의 정의를 수로 확인한다
# ========================================================================

def jensen():
    header("1. 정의를 수로 — f(tx+(1-t)y) <= t f(x) + (1-t) f(y)")

    fns = {
        "x^2        ": lambda x: x ** 2,
        "|x|        ": lambda x: np.abs(x),
        "exp(x)     ": lambda x: np.exp(x),
        "log(1+e^x) ": lambda x: np.log1p(np.exp(x)),
        "x^3        ": lambda x: x ** 3,
        "sin(x)     ": lambda x: np.sin(x),
        "x^4-3x^2   ": lambda x: x ** 4 - 3 * x ** 2,
    }
    rng = np.random.default_rng(0)
    xs = rng.uniform(-3, 3, 20000)
    ys = rng.uniform(-3, 3, 20000)
    ts = rng.uniform(0, 1, 20000)

    print("  함수            어긴 비율    가장 크게 어긴 양")
    for name, f in fns.items():
        lhs = f(ts * xs + (1 - ts) * ys)
        rhs = ts * f(xs) + (1 - ts) * f(ys)
        gap = lhs - rhs                      # 양수면 어긴 것
        bad = gap > 1e-12
        print(f"  {name}  {bad.mean()*100:6.2f}%      {gap.max():+.3e}")
    print("  위 넷은 한 번도 어기지 않는다. 아래 셋은 어긴다")
    print("  주의: 어기는 쌍을 못 찾았다는 것이 증명은 아니다. 반례를 찾는")
    print("  데는 쓸 수 있어도 볼록임을 보이는 데는 쓸 수 없다")


# ========================================================================
# 2. 출발점을 바꿔 보면 — 볼록한 쪽
# ========================================================================

def convex_starts():
    header("2. 볼록한 문제: 출발점을 바꿔도 같은 곳에 닿는가")
    torch.manual_seed(0)
    X = torch.randn(200, 5)
    w_true = torch.randn(5, 1)
    Y = X @ w_true + 0.1 * torch.randn(200, 1)

    # 정규방정식으로 참 최솟값을 구해 둔다
    Xb = torch.cat([X, torch.ones(200, 1)], 1)
    opt_w = torch.linalg.lstsq(Xb, Y).solution
    opt_loss = ((Xb @ opt_w - Y) ** 2).mean().item()
    print(f"  정규방정식이 주는 최소 로스 {opt_loss:.8f}")

    print("  출발 씨앗   2000 걸음 뒤 로스   최솟값과의 차이   해와의 거리")
    sols = []
    for seed in range(5):
        torch.manual_seed(seed)
        m = nn.Linear(5, 1)
        with torch.no_grad():               # 일부러 멀리 떨어뜨려 출발
            m.weight.mul_(10).add_(seed * 3.0)
            m.bias.mul_(10).add_(seed * 3.0)
        o = torch.optim.SGD(m.parameters(), lr=0.05)
        for _ in range(2000):
            o.zero_grad()
            nn.functional.mse_loss(m(X), Y).backward()
            o.step()
        loss = nn.functional.mse_loss(m(X), Y).item()
        got = torch.cat([m.weight.detach().flatten(), m.bias.detach()])
        sols.append(got)
        print(f"  {seed:<11} {loss:.8f}        {loss-opt_loss:+.2e}"
              f"        {(got - opt_w.flatten()).norm():.6f}")
    stack = torch.stack(sols)
    print(f"  다섯 해 사이의 최대 거리 {torch.cdist(stack, stack).max():.8f}")
    print("  출발점이 달라도 같은 점에 닿는다. 볼록한 문제의 뜻이 이것이다")


# ========================================================================
# 3. 출발점을 바꿔 보면 — 비볼록한 쪽
# ========================================================================

def nonconvex_starts():
    header("3. 비볼록한 문제: 같은 데이터, 같은 설정, 출발만 다르게")
    torch.manual_seed(0)
    X = torch.randn(200, 5)
    Y = torch.sin(X.sum(1, keepdim=True) * 2)

    losses, sols = [], []
    print("  출발 씨앗   2000 걸음 뒤 로스")
    for seed in range(5):
        torch.manual_seed(seed)
        m = nn.Sequential(nn.Linear(5, 16), nn.Tanh(), nn.Linear(16, 1))
        o = torch.optim.SGD(m.parameters(), lr=0.05)
        for _ in range(2000):
            o.zero_grad()
            nn.functional.mse_loss(m(X), Y).backward()
            o.step()
        loss = nn.functional.mse_loss(m(X), Y).item()
        losses.append(loss)
        sols.append(m(X).detach().flatten())
        print(f"  {seed:<11} {loss:.8f}")
    print(f"  가장 좋은 것 {min(losses):.6f}, 가장 나쁜 것 {max(losses):.6f}, "
          f"비 {max(losses)/min(losses):.2f}")

    stack = torch.stack(sols)
    print(f"  다섯 해의 예측값 사이 최대 거리 "
          f"{torch.cdist(stack, stack).max():.6f}")
    print("  로스도 다르고 함수 자체도 다르다")
    print("  2절에서는 이 거리가 1e-6 자리였다. 그것이 두 세계의 차이다")


# ========================================================================
# 4. 비볼록성은 어디서 오는가 — 대칭을 수로 확인한다
# ========================================================================

def symmetries():
    header("4. 대칭 — 같은 함수를 내는 다른 파라메터")
    torch.manual_seed(0)
    X = torch.randn(50, 4)

    m = nn.Sequential(nn.Linear(4, 6), nn.ReLU(), nn.Linear(6, 1))
    base = m(X).detach()

    # (가) 숨은 유닛의 차례를 바꾼다
    perm = torch.tensor([3, 1, 5, 0, 4, 2])
    m2 = nn.Sequential(nn.Linear(4, 6), nn.ReLU(), nn.Linear(6, 1))
    with torch.no_grad():
        m2[0].weight.copy_(m[0].weight[perm])
        m2[0].bias.copy_(m[0].bias[perm])
        m2[2].weight.copy_(m[2].weight[:, perm])
        m2[2].bias.copy_(m[2].bias)
    print(f"  (가) 유닛 차례 바꾸기: 출력 최대 차이 "
          f"{(m2(X).detach() - base).abs().max():.3e}")
    print(f"       파라메터는 달라졌는가 "
          f"{not torch.equal(m2[0].weight, m[0].weight)}")
    import math
    print(f"       6개 유닛이면 같은 함수를 내는 자리가 6! = "
          f"{math.factorial(6)}개다")

    # (나) ReLU 의 양의 동차성 — 앞을 c 배, 뒤를 1/c 배
    c = 3.0
    m3 = nn.Sequential(nn.Linear(4, 6), nn.ReLU(), nn.Linear(6, 1))
    with torch.no_grad():
        m3[0].weight.copy_(m[0].weight * c)
        m3[0].bias.copy_(m[0].bias * c)
        m3[2].weight.copy_(m[2].weight / c)
        m3[2].bias.copy_(m[2].bias)
    print(f"  (나) 앞을 {c}배 뒤를 1/{c}배: 출력 최대 차이 "
          f"{(m3(X).detach() - base).abs().max():.3e}")
    print(f"       웨이트 노름은 {m[0].weight.norm():.4f} -> "
          f"{m3[0].weight.norm():.4f} 로 바뀐다")
    print("       c 는 어떤 양수여도 되므로 같은 함수를 내는 자리가 무한히 많다")

    # (다) tanh 는 양의 동차가 아니다
    t1 = nn.Sequential(nn.Linear(4, 6), nn.Tanh(), nn.Linear(6, 1))
    t2 = nn.Sequential(nn.Linear(4, 6), nn.Tanh(), nn.Linear(6, 1))
    with torch.no_grad():
        t2[0].weight.copy_(t1[0].weight * c)
        t2[0].bias.copy_(t1[0].bias * c)
        t2[2].weight.copy_(t1[2].weight / c)
        t2[2].bias.copy_(t1[2].bias)
    print(f"  (다) 같은 조작을 tanh 에 하면: 출력 최대 차이 "
          f"{(t2(X).detach() - t1(X).detach()).abs().max():.4f}")
    print("       전혀 다른 함수가 된다. 척도 대칭은 ReLU 류에만 있다")

    print("\n  최솟값이 여럿 있다는 것은 이런 뜻이다. 서로 떨어진 자리에")
    print("  똑같이 좋은 해가 있으므로 함수가 볼록할 수가 없다")
    print("  (볼록하면 최솟값의 모임이 볼록집합이라 둘 사이를 이은 선분도")
    print("  모두 최솟값이어야 하는데, 섞어 놓은 넷트웍은 그렇지 않다)")

    # 정말 그런지 — 두 해의 중간점을 보자
    with torch.no_grad():
        mid = nn.Sequential(nn.Linear(4, 6), nn.ReLU(), nn.Linear(6, 1))
        for a, b, c2 in zip(m.parameters(), m2.parameters(), mid.parameters()):
            c2.copy_((a + b) / 2)
    Y = base                                    # m 과 m2 가 내는 바로 그 값
    print(f"\n  m 의 로스        {nn.functional.mse_loss(m(X), Y).item():.3e}")
    print(f"  m2(섞은 것) 로스 {nn.functional.mse_loss(m2(X), Y).item():.3e}")
    print(f"  둘의 중간점 로스 {nn.functional.mse_loss(mid(X), Y).item():.6f}")
    print("  최솟값 둘을 이은 선분의 가운데가 최솟값이 아니다 — 비볼록이다")


# ========================================================================
# 5. 헤세 행렬로 가려낼 수 있는가
# ========================================================================

def hessians():
    header("5. 헤세 행렬의 고윳값")
    from torch.func import functional_call, hessian

    torch.manual_seed(0)
    X = torch.randn(40, 3)
    Y = torch.randn(40, 1)

    def eig_of(model, x, y):
        names = [n for n, _ in model.named_parameters()]
        flat = {n: p.detach().clone() for n, p in model.named_parameters()}

        def f(*vals):
            d = dict(zip(names, vals))
            return nn.functional.mse_loss(functional_call(model, d, (x,)), y)

        H = hessian(f, argnums=tuple(range(len(names))))(*flat.values())
        # 블록을 한 장의 행렬로 이어 붙인다
        sizes = [flat[n].numel() for n in names]
        rows = []
        for i in range(len(names)):
            rows.append(torch.cat([H[i][j].reshape(sizes[i], sizes[j])
                                   for j in range(len(names))], 1))
        full = torch.cat(rows, 0)
        return torch.linalg.eigvalsh((full + full.T) / 2)

    lin = nn.Linear(3, 1)
    ev = eig_of(lin, X, Y)
    print(f"  선형 회귀 (볼록)   고윳값 {ev.min():.6f} ~ {ev.max():.6f}")
    print(f"                     음수 고윳값 개수 {(ev < -1e-6).sum().item()}"
          f" / {len(ev)}")

    torch.manual_seed(0)
    mlp = nn.Sequential(nn.Linear(3, 4), nn.Tanh(), nn.Linear(4, 1))
    ev2 = eig_of(mlp, X, Y)
    print(f"  작은 MLP (비볼록)  고윳값 {ev2.min():.6f} ~ {ev2.max():.6f}")
    print(f"                     음수 고윳값 개수 "
          f"{(ev2 < -1e-6).sum().item()} / {len(ev2)}")
    print("  음수 고윳값이 하나라도 있으면 그 점에서 아래로 휜 방향이 있다")
    print("  주의: 한 점에서 음수가 없다고 볼록인 것은 아니다. 볼록하려면")
    print("  모든 점에서 그래야 한다. 아래 줄이 그 보기다")
    torch.manual_seed(5)
    ev3 = eig_of(mlp, torch.randn(40, 3), torch.randn(40, 1))
    print(f"  같은 MLP, 다른 자료  음수 고윳값 "
          f"{(ev3 < -1e-6).sum().item()} / {len(ev3)}")


# ========================================================================
# 6. 안장점에서 정말 멈추는가
# ========================================================================

def saddle():
    header("6. 안장점")
    # f(x, y) = x^2 - y^2, 원점이 안장점
    print("  f(x, y) = x^2 - y^2,  grad = (2x, -2y),  원점에서 grad = 0")
    print("  헤세 = diag(2, -2) 이므로 고윳값 +2 와 -2 — 안장점이다")
    print("  x 쪽은 0.98 배씩 줄고 y 쪽은 1.02 배씩 는다 (eta = 0.01)")

    print("\n  출발점              200 걸음 뒤 (x, y)          로스")
    for p0 in ((1.0, 0.0), (1.0, 1e-8), (1.0, 1e-4), (1.0, 0.5)):
        x, y = p0
        for _ in range(200):
            x, y = x - 0.01 * 2 * x, y - 0.01 * (-2 * y)
        print(f"  {str(p0):<19} ({x:.3e}, {y:.3e})     {x**2 - y**2:+.4e}")
    print("  200 걸음으로는 y = 0.5 만 빠져나왔다. 1e-8 은 5.2e-7 이 되었을")
    print("  뿐이고 로스는 y = 0 일 때와 자릿수까지 같다 — 아직 안장점이다")

    # 빠져나오는 데 몇 걸음이 드는가
    print("\n  |y| > 1 이 될 때까지 걸리는 걸음 수")
    print("  처음 y       걸음 수    log(1/y0)/log(1.02) 로 센 값")
    import math
    for y0 in (0.5, 1e-2, 1e-4, 1e-8, 1e-12):
        y, n = y0, 0
        while abs(y) < 1.0 and n < 10 ** 6:
            y = y * 1.02
            n += 1
        pred = math.log(1 / y0) / math.log(1.02)
        print(f"  {y0:<12.0e} {n:>7}    {pred:>12.1f}")
    print("  걸음 수가 y0 에 반비례하지 않고 **로그** 로 는다. y0 를 1e-2")
    print("  에서 1e-8 로, 곧 백만분의 1로 줄여도 걸음은 233 에서 931 로")
    print("  네 배가 될 뿐이다. y 가 지수적으로 커지기 때문이다")
    print("  그래서 안장점은 '갇히는 곳' 이 아니라 '느려지는 곳' 이다")
    print("  다만 y 가 정확히 0 이면 영원히 못 나온다. 경사가 그 방향으로")
    print("  정확히 0 이라 아무리 걸어도 0 에 0 을 곱할 뿐이다")
    print("  실제 학습에서는 경사 잡음이 이 정확한 0 을 깨뜨려 준다")


def main():
    jensen()
    convex_starts()
    nonconvex_starts()
    symmetries()
    hessians()
    saddle()


if __name__ == "__main__":
    main()
