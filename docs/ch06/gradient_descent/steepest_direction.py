"""경사가 가장 가파른 방향인가 — 재어 본다.

최급상승 정리는 증명이 세 줄이라 믿기 쉽지만, 정작 쓸 때 걸리는 자리는
증명 안에 없다. 이 쪽은 다섯 가지를 재어 본다.

- 방향을 한 바퀴 돌려 보면 정말 경사 쪽이 가장 큰가
- 경사는 등위곡선에 수직인가
- 1차 테일러가 말하는 감소와 실제 감소가 언제 갈라지는가
- "가장 가파름" 은 길이를 어떻게 재느냐에 딸려 있다 (L2 / L1 / L-inf)
- 경사 클리핑은 방향을 지키는가

실행: python steepest_direction.py
"""
import numpy as np
import torch

torch.manual_seed(0)
np.random.seed(0)


def header(title):
    print(f"\n=== {title} ===")


# ========================================================================
# 1. 방향을 한 바퀴 돌려 본다
# ========================================================================

def sweep_directions():
    header("1. 방향을 한 바퀴 돌려 본다")
    # f(x, y) = x^2 + 3y^2, 점 (1, 2) 에서 grad = (2, 12)
    def f(p):
        return p[0] ** 2 + 3 * p[1] ** 2

    p = np.array([1.0, 2.0])
    g = np.array([2 * p[0], 6 * p[1]])
    print(f"  grad = {g.tolist()}   ||grad|| = {np.linalg.norm(g):.6f}")

    h = 1e-6
    rows = []
    for deg in range(0, 360, 30):
        t = np.deg2rad(deg)
        u = np.array([np.cos(t), np.sin(t)])
        numeric = (f(p + h * u) - f(p)) / h     # 차분으로 잰 방향도함수
        rows.append((deg, float(g @ u), float(numeric)))

    print("  각도   grad.u      차분으로 잰 값")
    for deg, dot, num in rows:
        print(f"  {deg:3d}   {dot:+9.6f}   {num:+9.6f}")

    best = max(rows, key=lambda r: r[1])
    print(f"  30도 간격 중 가장 큰 각도: {best[0]}도  ({best[1]:+.6f})")
    ang = np.rad2deg(np.arctan2(g[1], g[0]))
    print(f"  경사 자신의 각도: {ang:.4f}도 — 격자에 없어 30도가 가장 가깝다")

    # 격자를 촘촘히 하면 경사 각도로 수렴한다
    fine = max(np.linspace(0, 360, 36001)[:-1],
               key=lambda d: g @ np.array([np.cos(np.deg2rad(d)),
                                           np.sin(np.deg2rad(d))]))
    print(f"  0.01도 간격으로 다시 고르면: {fine:.2f}도")
    u_star = g / np.linalg.norm(g)
    print(f"  최대 증가율 = ||grad|| = {g @ u_star:.6f}")


# ========================================================================
# 2. 등위곡선에 수직인가
# ========================================================================

def perpendicular():
    header("2. 등위곡선에 수직인가")
    # f = x^2 + 4y^2, 점 (2, 1), 등위곡선 x^2 + 4y^2 = 8
    p = np.array([2.0, 1.0])
    g = np.array([2 * p[0], 8 * p[1]])
    print(f"  f(2,1) = {p[0]**2 + 4*p[1]**2:.1f},  grad = {g.tolist()}")

    # 등위곡선을 매개화해 접벡터를 수로 구한다
    def on_level(s):
        # x = sqrt(8) cos s, y = sqrt(2) sin s 이면 x^2 + 4y^2 = 8
        return np.array([np.sqrt(8) * np.cos(s), np.sqrt(2) * np.sin(s)])

    s0 = np.arctan2(p[1] / np.sqrt(2), p[0] / np.sqrt(8))
    print(f"  되찾은 점: {on_level(s0).round(6).tolist()}")
    h = 1e-6
    tangent = (on_level(s0 + h) - on_level(s0 - h)) / (2 * h)
    tangent = tangent / np.linalg.norm(tangent)
    print(f"  단위 접벡터: {tangent.round(6).tolist()}")
    print(f"  grad . tangent = {g @ tangent:+.3e}  <- 0 이다")

    # 접선 방향으로 움직이면 f 가 정말 그대로인가
    for step in (1e-3, 1e-2, 1e-1):
        q = p + step * tangent
        val = q[0] ** 2 + 4 * q[1] ** 2
        print(f"  접선으로 {step:5.0e} 만큼 가면 f = {val:.8f} "
              f"(차이 {val - 8:+.2e})")
    print("  1차로는 변화가 없고, 남는 것은 2차 항이다 (걸음의 제곱으로 준다)")


# ========================================================================
# 3. 1차 테일러가 말하는 감소와 실제 감소
# ========================================================================

def taylor_vs_actual():
    header("3. 1차 테일러 vs 실제 감소")
    # f = x^2 + 3y^2 에서 x <- x - eta * grad
    def f(p):
        return p[0] ** 2 + 3 * p[1] ** 2

    p = np.array([1.0, 2.0])
    g = np.array([2 * p[0], 6 * p[1]])
    f0 = f(p)
    print(f"  시작 f = {f0:.6f},  ||grad||^2 = {g @ g:.6f}")
    print("  eta      예측 감소     실제 감소     줄었는가")
    for eta in (1e-4, 1e-3, 1e-2, 0.1, 1 / 6, 0.2, 0.34, 0.4):
        pred = eta * (g @ g)                 # eta * ||grad||^2
        actual = f0 - f(p - eta * g)
        print(f"  {eta:<7.4f} {pred:11.6f}  {actual:+12.6f}   "
              f"{'예' if actual > 0 else '아니오'}")
    print("  작은 eta 에서는 둘이 맞고, 커지면 실제가 예측에서 떨어진다")
    print("  y 축 곡률이 6 이라 eta >= 2/6 = 0.3333 에서 y 쪽이 발산한다")


# ========================================================================
# 4. '가장 가파름' 은 길이를 재는 법에 딸려 있다
# ========================================================================

def which_norm():
    header("4. 노름을 바꾸면 가장 가파른 방향도 바뀐다")
    g = np.array([2.0, 12.0])
    print(f"  grad = {g.tolist()}")

    # 각 노름 아래 ||d|| <= 1 에서 grad.d 를 가장 작게 하는 d
    l2 = -g / np.linalg.norm(g)
    l1 = np.zeros(2)
    l1[np.argmax(np.abs(g))] = -np.sign(g[np.argmax(np.abs(g))])
    linf = -np.sign(g)

    for name, d in (("L2  ", l2), ("L1  ", l1), ("Linf", linf)):
        ang = np.rad2deg(np.arccos(d @ l2 / (np.linalg.norm(d))))
        print(f"  {name} 아래 최급강하 {d.round(6).tolist()}   "
              f"grad.d = {g @ d:+9.6f}   L2 해와 {ang:5.1f}도 차이")
    print("  L1 아래에서는 좌표 하나만 움직인다 — 경사의 방향이 아니다")
    print("  '경사가 가장 가파르다' 는 유클리드 길이로 잴 때의 이야기다")

    # 척도를 바꾸는 것도 같은 일이다
    print("\n  변수를 y' = y/10 으로 바꿔 재면")
    # f = x^2 + 3y^2 = x^2 + 300 y'^2
    g2 = np.array([2.0, 600.0 * 0.2])        # y' = 0.2 일 때
    print(f"    새 좌표의 grad = {g2.tolist()}")
    print(f"    예전 방향 {( -g/np.linalg.norm(g)).round(4).tolist()}, "
          f"새 방향 {(-g2/np.linalg.norm(g2)).round(4).tolist()}")
    print("    같은 점, 같은 함수인데 '가장 가파른 방향' 이 달라졌다")


# ========================================================================
# 5. 클리핑은 방향을 지키는가
# ========================================================================

def clipping():
    header("5. 노름 클리핑과 원소별 클리핑")
    import torch.nn as nn
    torch.manual_seed(0)
    m = nn.Linear(4, 3)
    nn.functional.mse_loss(m(torch.randn(8, 4)), torch.randn(8, 3)).backward()
    flat = torch.cat([p.grad.flatten() for p in m.parameters()])
    before = flat.clone()
    print(f"  경사 노름 {before.norm():.6f},  원소 최대 "
          f"{before.abs().max():.6f}")

    def cos(a, b):
        return (a @ b / (a.norm() * b.norm())).item()

    nn.utils.clip_grad_norm_(m.parameters(), max_norm=0.1)
    after_norm = torch.cat([p.grad.flatten() for p in m.parameters()]).clone()
    print(f"  노름 클리핑 뒤 노름 {after_norm.norm():.6f}   "
          f"원래와의 코사인 {cos(before, after_norm):.10f}")

    for p, chunk in zip(m.parameters(),
                        before.split([p.numel() for p in m.parameters()])):
        p.grad = chunk.view_as(p).clone()     # 되돌린다
    nn.utils.clip_grad_value_(m.parameters(), clip_value=0.02)
    after_val = torch.cat([p.grad.flatten() for p in m.parameters()])
    print(f"  원소별 클리핑 뒤 노름 {after_val.norm():.6f}   "
          f"원래와의 코사인 {cos(before, after_val):.10f}")
    print("  노름 클리핑은 방향을 그대로 둔다 (코사인 1). 원소별은 방향을 바꾼다")

    zero = nn.Linear(2, 1)
    for p in zero.parameters():
        p.grad = torch.zeros_like(p)
    n = nn.utils.clip_grad_norm_(zero.parameters(), max_norm=1.0)
    print(f"  경사가 모두 0 일 때 clip_grad_norm_ 은 {n.item():.1f} 을 주고 "
          "나누지 않는다")
    g0 = torch.zeros(3)
    print("  손으로 g/||g|| 를 적으면:", (g0 / g0.norm()).tolist(),
          "<- 0 으로 나누어 NaN 이 된다")


# ========================================================================
# 6. 손으로 민 식과 autograd 가 같은가
# ========================================================================

def by_hand_and_autograd():
    header("6. 손으로 민 식 vs autograd")
    x = torch.tensor([1., 2., 3., 4., 5.])
    y = torch.tensor([2., 4., 6., 8., 10.])
    w = torch.tensor(0.5, requires_grad=True)

    loss = torch.mean((w * x - y) ** 2)
    loss.backward()
    print(f"  loss      = {loss.item():.6f}")
    print(f"  w.grad    = {w.grad.item():.6f}")

    # dL/dw = (2/N) * sum x_i (w x_i - y_i)
    hand = (2 / len(x) * (x * (0.5 * x - y)).sum()).item()
    print(f"  손으로 민 식 = {hand:.6f}")
    print(f"  같은가      = {abs(hand - w.grad.item()) < 1e-6}")


def main():
    sweep_directions()
    perpendicular()
    taylor_vs_actual()
    which_norm()
    clipping()
    by_hand_and_autograd()


if __name__ == "__main__":
    main()
