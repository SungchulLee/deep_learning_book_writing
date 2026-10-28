"""전진 모드와 후진 모드 — 셈으로 말고 재어 본다.

본문의 표는 후진 모드가 $O(T)$, 전진 모드가 $O(n \\cdot T)$ 라고
적는다. 파라미터가 4,225 개면 4,225 배 차이가 나야 한다는 말이다.

재어 보면 55 배다. 셈이 틀린 것이 아니라 **상수가 숨어 있다.**
전진 모드도 한 번에 여러 방향을 묶어 처리하므로 통과 횟수만큼
곱해지지 않는다.

그래도 자릿수가 같고, 방향을 뒤집으면 **정확히 거꾸로** 뒤집힌다.

실행: python gradient_computation.py
"""
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call, jacfwd, jacrev


def header(title):
    print(f"\n=== {title} ===")


def timeit(fn, rep=3):
    fn()                                   # 예열
    ts = []
    for _ in range(rep):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return min(ts) * 1e3


# ========================================================================
# 1. 딥러닝의 꼴 — 입력 많고 출력 하나
# ========================================================================

def deep_learning_shape():
    header("1. 파라미터는 많고 로스는 하나")
    torch.manual_seed(0)
    net = nn.Sequential(nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 1))
    n_param = sum(p.numel() for p in net.parameters())
    x, y = torch.randn(8, 64), torch.randn(8, 1)
    params = {k: v.detach() for k, v in net.named_parameters()}

    def loss_fn(p):
        return F.mse_loss(functional_call(net, p, (x,)), y)

    rev = timeit(lambda: jacrev(loss_fn)(params))
    fwd = timeit(lambda: jacfwd(loss_fn)(params))
    print(f"  파라미터 {n_param:,}개, 로스는 홑값 하나")
    print(f"  jacrev (후진) {rev:8.2f} ms")
    print(f"  jacfwd (전진) {fwd:8.2f} ms")
    print(f"  전진이 {fwd / rev:.0f} 배 느리다")
    print(f"  표의 셈으로는 {n_param:,} 배여야 하는데 {fwd / rev:.0f} 배다")
    print("  전진 모드도 여러 방향을 묶어 처리하므로 통과 횟수만큼 안 곱해진다")


# ========================================================================
# 2. 거꾸로 — 입력 하나에 출력 많기
# ========================================================================

def the_other_shape():
    header("2. 거꾸로 뒤집으면")
    torch.manual_seed(0)
    A = torch.randn(2048, 1)

    def h(t):
        return (A @ t.view(1, 1)).squeeze(1).tanh()

    xin = torch.randn(1)
    rev = timeit(lambda: jacrev(h)(xin))
    fwd = timeit(lambda: jacfwd(h)(xin))
    print(f"  입력 1 개 -> 출력 2048 개")
    print(f"  jacrev (후진) {rev:8.2f} ms")
    print(f"  jacfwd (전진) {fwd:8.2f} ms")
    print(f"  이번에는 전진이 {rev / fwd:.0f} 배 빠르다")
    print("  고르는 기준은 라이브러리가 아니라 함수의 모양이다")


# ========================================================================
# 3. 후진 모드가 치르는 값 — 활성값을 들고 있어야 한다
# ========================================================================

def the_memory_cost():
    header("3. 후진 모드가 치르는 값")
    torch.manual_seed(0)
    net = nn.Sequential(*[nn.Sequential(nn.Linear(256, 256), nn.ReLU())
                          for _ in range(8)])
    x = torch.randn(64, 256)
    with torch.no_grad():
        no_graph = net(x)
    with_graph = net(x)
    print("  no_grad 로 돌린 결과 grad_fn :", no_graph.grad_fn)
    print("  그냥 돌린 결과 grad_fn       :",
          type(with_graph.grad_fn).__name__)
    act = 64 * 256 * 4 * 8 * 2 / 1e6
    par = sum(p.numel() * 4 for p in net.parameters()) / 1e6
    print(f"  파라미터 {par:.2f} MB   층마다 저장되는 활성값 대략 {act:.2f} MB")
    print("  전진 모드는 활성값을 안 들고 있어도 된다 — 앞으로만 가니까")
    print("  후진 모드의 속도는 공짜가 아니라 메모리와 바꾼 것이다")


# ========================================================================
# 4. 야코비를 모으는 두 방향
# ========================================================================

def two_directions():
    header("4. 야코비를 줄로 모으기 vs 열로 모으기")

    def f(t):
        return torch.stack([t[0] ** 2, t[0] * t[1], t[1] ** 2])

    x = torch.tensor([1.0, 2.0])
    print("  f: R^2 -> R^3,  야코비는 3x2")
    print("  jacrev (줄 단위, 출력 3 번):")
    for r in jacrev(f)(x).tolist():
        print("    ", [round(v, 4) for v in r])
    print("  jacfwd (열 단위, 입력 2 번):")
    for r in jacfwd(f)(x).tolist():
        print("    ", [round(v, 4) for v in r])
    print("  같은가:", torch.allclose(jacrev(f)(x), jacfwd(f)(x)))
    print("  손으로: [[2x, 0], [y, x], [0, 2y]] = [[2,0],[2,1],[0,4]]")
    print("  출력 3 > 입력 2 이므로 여기서는 전진이 한 번 덜 돈다")


def main():
    deep_learning_shape()
    the_other_shape()
    the_memory_cost()
    two_directions()


if __name__ == "__main__":
    main()
