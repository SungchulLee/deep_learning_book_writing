"""경사 누적의 값 — 메모리를 아끼고 시간을 치른다.

「경사 누적」에서 쪼개서 쌓은 것이 통째로 한 것과 **같은 경사**를
준다는 것을 확인했다. 이 쪽은 그 다음 물음을 본다 — **값이 얼마인가.**

쪼개면 활성값 메모리가 조각 크기에 비례해 줄어든다. 대신 순전파와
역전파를 여러 번 해야 하므로 시간이 는다. 두 축을 재어, 조각 수를
어떻게 고를지 가늠해 본다.

실행: python 19_gradient_accumulation_for_large_batches.py
"""
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


def header(title):
    print(f"\n=== {title} ===")


def spread(fn, rep=5):
    for _ in range(2):          # 예열
        fn()
    ts = []
    for _ in range(rep):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return min(ts) * 1e3, max(ts) * 1e3


BATCH, DIM = 256, 512


def setup(seed=0):
    torch.manual_seed(seed)
    net = nn.Sequential(nn.Linear(DIM, DIM), nn.ReLU(),
                        nn.Linear(DIM, DIM), nn.ReLU(),
                        nn.Linear(DIM, 10))
    opt = torch.optim.SGD(net.parameters(), lr=0.01)
    X = torch.randn(BATCH, DIM)
    Y = torch.randint(0, 10, (BATCH,))
    return net, opt, X, Y


def step_fn(net, opt, X, Y, k):
    def run():
        opt.zero_grad()
        n = BATCH // k
        for i in range(k):
            part = F.cross_entropy(net(X[i * n:(i + 1) * n]),
                                   Y[i * n:(i + 1) * n]) / k
            part.backward()
        opt.step()
    return run


# ========================================================================
# 1. 시간이 얼마나 느는가
# ========================================================================

def time_cost():
    header("1. 조각 수와 시간")
    net, opt, X, Y = setup()
    times = {}
    for k in (1, 2, 4, 8, 16, 32):
        lo, _ = spread(step_fn(net, opt, X, Y, k))
        times[k] = lo
        print(f"  {k:2d} 조각  조각 크기 {BATCH // k:3d}  "
              f"{lo:6.2f} ms   통째 대비 {lo / times[1]:5.2f}배   "
              f"조각당 {lo / k:6.3f} ms")
    print("  조각 수에 꼭 비례하지는 않는다 — 조각당 시간이 줄어든다")
    print("  작은 배치는 행렬곱이 덜 효율적이지만 일 자체도 적기 때문이다")
    return times


# ========================================================================
# 2. 메모리는 조각 크기가 정한다
# ========================================================================

def memory_cost():
    header("2. 조각 수와 활성값 메모리")
    print("  활성값은 '조각 크기 x 폭 x 층수 x 4바이트' 쯤이다")
    for k in (1, 4, 16, 32):
        n = BATCH // k
        approx = n * DIM * 4 * 4 / 1e6      # 네 군데쯤 저장된다고 보고
        print(f"  {k:2d} 조각  조각 크기 {n:3d}  활성값 대략 {approx:6.3f} MB")
    print("  파라미터와 경사는 조각 수와 무관하다 — 모델 크기가 정한다")
    net, _, _, _ = setup()
    pbytes = sum(p.numel() * p.element_size() for p in net.parameters())
    print(f"  이 모델의 파라미터 {pbytes / 1e6:.2f} MB (경사도 같은 크기)")
    print("  그래서 누적으로 아끼는 것은 활성값뿐이다")


# ========================================================================
# 3. 어디까지 쪼개는 것이 쓸모 있는가
# ========================================================================

def diminishing(times):
    header("3. 쪼갤수록 이득이 줄어든다")
    ks = sorted(times)
    ratios = [times[b] / times[a] for a, b in zip(ks, ks[1:])]
    print("  조각 크기를 반으로 줄일 때마다")
    print("    활성값 메모리: 반으로 — 언제나 반")
    print("    시간:", " -> ".join(f"{r:.2f}" for r in ratios), "배")
    print("  메모리 이득은 일정하고 시간 손해는 쌓인다")
    print("  그래서 '모델이 올라가는 가장 큰 조각' 을 고르는 것이 맞다")
    print("  더 쪼개도 경사는 같으므로 학습에는 아무 이득이 없다")


# ========================================================================
# 4. 모델이 깊으면 체크포인팅과 견줄 만하다
# ========================================================================

def versus_checkpointing():
    header("4. 체크포인팅과 견주기")
    from torch.utils.checkpoint import checkpoint
    torch.manual_seed(0)
    blocks = nn.ModuleList([nn.Sequential(nn.Linear(DIM, DIM), nn.ReLU())
                            for _ in range(8)])
    head = nn.Linear(DIM, 10)
    params = list(blocks.parameters()) + list(head.parameters())
    opt = torch.optim.SGD(params, lr=0.01)
    X = torch.randn(BATCH, DIM)
    Y = torch.randint(0, 10, (BATCH,))

    def forward(x, use_ckpt):
        for b in blocks:
            x = checkpoint(b, x, use_reentrant=False) if use_ckpt else b(x)
        return head(x)

    def run(k, use_ckpt):
        def f():
            opt.zero_grad()
            n = BATCH // k
            for i in range(k):
                part = F.cross_entropy(forward(X[i * n:(i + 1) * n], use_ckpt),
                                       Y[i * n:(i + 1) * n]) / k
                part.backward()
            opt.step()
        return f

    for label, k, ck in (("통째로", 1, False),
                         ("8조각 누적", 8, False),
                         ("체크포인팅", 1, True)):
        lo, hi = spread(run(k, ck))
        print(f"  {label:12s} {lo:6.2f} ~ {hi:6.2f} ms")
    print("  둘 다 메모리를 시간과 바꾸는데 체크포인팅이 덜 비쌌다")
    print("  다만 아끼는 축이 다르다 — 누적은 배치 축, 체크포인팅은 깊이 축")
    print("  둘을 같이 쓰면 곱으로 아낀다")


def main():
    times = time_cost()
    memory_cost()
    diminishing(times)
    versus_checkpointing()


if __name__ == "__main__":
    main()
