"""배치에서의 VJP — 경사는 표본별로 오지 않는다.

배치로 로스를 내고 역전파하면 파라미터의 경사가 **하나** 나온다.
표본이 256 개여도 하나다. 그것은 표본별 경사를 **더한** 것이고,
어느 표본이 얼마나 밀었는지는 사라진다.

평소에는 그것이 원하는 바다 — 걸음은 하나니까. 그런데 표본별
경사가 필요한 자리가 있다.

- 차등 프라이버시 — 표본마다 경사를 잘라야 한다
- 영향 함수 — 어느 표본이 예측을 바꾸는가
- 경사 잡음 재기 — 표본 사이의 퍼짐

실행: python 27_batch_vjp.py
"""
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call, grad, vmap


def header(title):
    print(f"\n=== {title} ===")


# ========================================================================
# 1. 배치 경사는 표본별 경사의 합이다
# ========================================================================

def sum_over_batch():
    header("1. 더해서 온다")
    torch.manual_seed(0)
    m = nn.Linear(3, 1)
    X, Y = torch.randn(4, 3), torch.randn(4, 1)

    m.zero_grad()
    F.mse_loss(m(X), Y, reduction="sum").backward()
    batch_g = m.weight.grad.clone()

    per = []
    for i in range(4):
        m.zero_grad()
        F.mse_loss(m(X[i:i + 1]), Y[i:i + 1], reduction="sum").backward()
        per.append(m.weight.grad.clone())
    stacked = torch.stack(per)

    print("  배치 한 번:", [round(t, 5) for t in batch_g.flatten().tolist()])
    print("  표본별 합 :", [round(t, 5) for t in stacked.sum(0).flatten().tolist()])
    print("  같은가    :", torch.allclose(batch_g, stacked.sum(0), atol=1e-5))
    print("  표본마다:")
    for i, g in enumerate(per):
        print(f"    {i}: {[round(t, 5) for t in g.flatten().tolist()]}")
    print("  0번 표본이 혼자 -5.87 을 밀고 있다. 합에서는 안 보인다")
    return stacked


# ========================================================================
# 2. vmap 으로 한 번에
# ========================================================================

def with_vmap(reference):
    header("2. vmap 으로 모으기")
    torch.manual_seed(0)
    m = nn.Linear(3, 1)
    X, Y = torch.randn(4, 3), torch.randn(4, 1)
    params = {k: v.detach() for k, v in m.named_parameters()}

    def loss_one(p, x, y):
        pred = functional_call(m, p, (x.unsqueeze(0),))
        return F.mse_loss(pred, y.unsqueeze(0), reduction="sum")

    per_sample = vmap(grad(loss_one), in_dims=(None, 0, 0))(params, X, Y)
    print("  결과 모양:", {k: tuple(v.shape) for k, v in per_sample.items()})
    print("  첫 축이 배치다 — 표본마다 한 벌씩")
    print("  루프로 모은 것과 같은가:",
          torch.allclose(per_sample["weight"], reference, atol=1e-5))


# ========================================================================
# 3. 값
# ========================================================================

def costs():
    header("3. 값")
    torch.manual_seed(0)
    net = nn.Linear(128, 64)
    X, Y = torch.randn(256, 128), torch.randn(256, 64)
    params = {k: v.detach() for k, v in net.named_parameters()}

    def loss_one(p, x, y):
        pred = functional_call(net, p, (x.unsqueeze(0),))
        return F.mse_loss(pred, y.unsqueeze(0), reduction="sum")

    def spread(fn, rep=3):
        fn()
        ts = []
        for _ in range(rep):
            t0 = time.perf_counter()
            fn()
            ts.append(time.perf_counter() - t0)
        return min(ts) * 1e3

    def loop():
        out = []
        for i in range(256):
            net.zero_grad()
            F.mse_loss(net(X[i:i + 1]), Y[i:i + 1], reduction="sum").backward()
            out.append(net.weight.grad.clone())
        return torch.stack(out)

    def vm():
        return vmap(grad(loss_one), in_dims=(None, 0, 0))(params, X, Y)

    def batched():
        net.zero_grad()
        F.mse_loss(net(X), Y, reduction="sum").backward()
        return net.weight.grad

    print(f"  배치 한 번 (합만) {spread(batched):8.2f} ms")
    print(f"  vmap  (표본별)   {spread(vm):8.2f} ms")
    print(f"  루프  (표본별)   {spread(loop):8.2f} ms")
    print("  표본별이 필요하면 vmap 이 루프보다 훨씬 낫다")
    print("  다만 합만 필요하면 배치 한 번이 비교가 안 되게 싸다")


# ========================================================================
# 4. 쓸모 — 경사의 잡음을 재기
# ========================================================================

def measuring_noise():
    header("4. 표본별 경사로 잡음 재기")
    torch.manual_seed(0)
    net = nn.Linear(16, 1)
    params = {k: v.detach() for k, v in net.named_parameters()}

    def loss_one(p, x, y):
        pred = functional_call(net, p, (x.unsqueeze(0),))
        return F.mse_loss(pred, y.unsqueeze(0), reduction="sum")

    torch.manual_seed(7)
    BIG = 4096
    Xa, Ya = torch.randn(BIG, 16), torch.randn(BIG, 1)
    ga = vmap(grad(loss_one), in_dims=(None, 0, 0))(params, Xa, Ya)["weight"]
    ga = ga.reshape(BIG, -1)
    per_sample_spread = ga.std(0).mean()
    print(f"  표본 하나짜리 경사의 흩어짐 {per_sample_spread:.4f}"
          f"  (표본 {BIG}개로 잰 값)")
    print("  이것은 데이터의 성질이라 배치 크기와 무관하다")
    print("  배치 평균의 흩어짐은 거기서 sqrt(n) 으로 줄어든다")
    for n in (16, 64, 256, 1024):
        means = []
        for t in range(24):                 # 배치를 여러 번 뽑아 평균의 퍼짐을 본다
            idx = torch.randperm(BIG)[:n]
            means.append(ga[idx].mean(0))
        M = torch.stack(means)
        print(f"  배치 {n:5d}  평균의 흩어짐 {M.std(0).mean():8.5f}"
              f"   예측 {per_sample_spread / n ** 0.5:8.5f}")
    print("  잰 값이 예측과 맞는다 — 「배치를 키우면」의 1/sqrt(n) 이 이것이다")


# ========================================================================
# 5. 차등 프라이버시가 이것을 쓴다
# ========================================================================

def dp_clipping():
    header("5. 표본마다 자르기")
    torch.manual_seed(0)
    net = nn.Linear(8, 1)
    X, Y = torch.randn(32, 8), torch.randn(32, 1)
    params = {k: v.detach() for k, v in net.named_parameters()}

    def loss_one(p, x, y):
        pred = functional_call(net, p, (x.unsqueeze(0),))
        return F.mse_loss(pred, y.unsqueeze(0), reduction="sum")

    g = vmap(grad(loss_one), in_dims=(None, 0, 0))(params, X, Y)["weight"]
    g = g.reshape(32, -1)
    norms = g.norm(dim=1)
    print(f"  표본별 경사 노름  최소 {norms.min():.4f}  최대 {norms.max():.4f}")
    C = 1.0
    scale = (C / norms.clamp(min=C)).unsqueeze(1)
    clipped = g * scale
    print(f"  C={C} 로 자른 뒤   최소 {clipped.norm(dim=1).min():.4f}  "
          f"최대 {clipped.norm(dim=1).max():.4f}")
    print("  자르기 전 합의 노름:", round(g.sum(0).norm().item(), 4))
    print("  자른 뒤 합의 노름  :", round(clipped.sum(0).norm().item(), 4))
    print("  한 표본이 걸음을 좌우하지 못하게 막는 것이 요점이다")
    print("  배치 경사만 있으면 이 일을 할 수 없다 — 이미 더해졌으니까")


def main():
    ref = sum_over_batch()
    with_vmap(ref)
    costs()
    measuring_noise()
    dp_clipping()


if __name__ == "__main__":
    main()
