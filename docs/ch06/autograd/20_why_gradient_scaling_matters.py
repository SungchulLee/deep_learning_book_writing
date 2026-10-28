"""배치를 키우면 러닝레이트를 어떻게 해야 하는가.

「경사 누적」에서 `mean` 로스를 조각 수로 나누어야 한다는 것을 보았다.
그것은 **같은 경사를 만들기 위한** 조정이었다.

이 쪽은 다른 물음이다. 실효 배치를 **정말로 키웠을 때** 러닝레이트를
어떻게 할 것인가. 경사의 크기는 그대로인데(평균이니까) 두 가지가
달라진다.

- **에폭당 걸음 수가 줄어든다** — 배치가 두 배면 걸음이 절반
- **경사의 잡음이 줄어든다** — 표본이 많으면 추정이 정확해진다

앞의 것만 보면 러닝레이트를 k 배 해야 하고(선형 규칙), 뒤의 것까지
보면 sqrt(k) 배가 맞다는 주장이 나온다. 재어 본다.

실행: python 20_why_gradient_scaling_matters.py
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def header(title):
    print(f"\n=== {title} ===")


N, D = 2048, 16
BASE_B, BASE_LR = 32, 0.05


def make_data(seed=0):
    torch.manual_seed(seed)
    X = torch.randn(N, D)
    W = torch.randn(D, 1)
    return X, X @ W + 0.1 * torch.randn(N, 1)


def train(X, Y, batch, lr, epochs=4, seed=0):
    torch.manual_seed(seed)
    m = nn.Linear(D, 1)
    opt = torch.optim.SGD(m.parameters(), lr=lr)
    for _ in range(epochs):
        perm = torch.randperm(N)
        for i in range(0, N, batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            F.mse_loss(m(X[idx]), Y[idx]).backward()
            opt.step()
    return F.mse_loss(m(X), Y).item()


# ========================================================================
# 1. 배치가 크면 걸음이 준다
# ========================================================================

def fewer_steps():
    header("1. 에폭당 걸음 수")
    for b in (32, 64, 128, 256, 512):
        print(f"  배치 {b:4d}  에폭당 걸음 {N // b:3d}번  "
              f"4 에폭이면 {4 * (N // b):4d}번")
    print("  같은 데이터를 보는데 걷는 횟수가 열여섯 배 차이 난다")


# ========================================================================
# 2. 경사의 잡음은 줄어든다
# ========================================================================

def less_noise():
    header("2. 배치와 경사의 잡음")
    X, Y = make_data()
    torch.manual_seed(0)
    m = nn.Linear(D, 1)
    for b in (8, 32, 128, 512, N):
        gs = []
        for t in range(16):
            torch.manual_seed(100 + t)
            idx = torch.randperm(N)[:b]
            m.zero_grad()
            F.mse_loss(m(X[idx]), Y[idx]).backward()
            gs.append(m.weight.grad.flatten().clone())
        G = torch.stack(gs)
        spread = G.std(dim=0).mean().item()
        size = G.mean(dim=0).norm().item()
        print(f"  배치 {b:4d}  경사 크기 {size:7.4f}  표본마다 흔들림 {spread:7.4f}"
              f"  비 {spread / size:6.3f}")
    print("  크기는 그대로고 흔들림만 준다 — 평균이라 그렇다")
    print("  흔들림이 1/sqrt(배치) 로 줄어드는 것이 sqrt 규칙의 근거다")


# ========================================================================
# 3. 세 가지 조정을 견준다
# ========================================================================

def three_rules():
    header("3. 러닝레이트를 어떻게 할 것인가")
    X, Y = make_data()
    print(f"  기준: 배치 {BASE_B}, lr {BASE_LR}  ->"
          f" {train(X, Y, BASE_B, BASE_LR):.6f}")
    print(f"  {'배치':>5s}  {'그대로':>22s}  {'선형 k배':>22s}  {'sqrt k배':>22s}")
    for b in (32, 64, 128, 256, 512):
        k = b / BASE_B
        cells = []
        for lr in (BASE_LR, BASE_LR * k, BASE_LR * math.sqrt(k)):
            cells.append(f"lr={lr:5.3f} {train(X, Y, b, lr):9.6f}")
        print(f"  {b:5d}  " + "  ".join(f"{c:>22s}" for c in cells))


# ========================================================================
# 4. 읽기
# ========================================================================

def reading():
    header("4. 무엇이 보이는가")
    print("  그대로 두면 배치 256 부터 무너진다 — 걸음이 모자라서다")
    print("  선형 k배는 256 까지 버티고 512 에서 나빠진다 — 걸음이 너무 커서다")
    print("  sqrt k배는 끝까지 같은 자리에 닿는다")
    print("  이 문제에서는 sqrt 쪽이 이겼다. 다만 조건이 붙는다")
    print("    - 이 로스는 볼록하고 잡음이 작다")
    print("    - 큰 딥러닝 모델에서는 선형 규칙이 더 잘 듣는다고 알려져 있다")
    print("    - 어느 쪽이든 '안 바꾸는 것' 보다는 낫다")


def main():
    fewer_steps()
    less_noise()
    three_rules()
    reading()


if __name__ == "__main__":
    main()
