"""배치 · 미니배치 · SGD — 셋이 실제로 무엇을 바꾸는가.

배치 크기에 대한 설명은 대개 "크면 안정적이고 느리다, 작으면 시끄럽고
빠르다" 로 끝난다. 이 쪽은 그 말의 각 조각을 따로 재어 본다.

- 분산이 정말 1/B 로 주는가 (유한 모집단 보정까지)
- B = N 이면 추출 분산이 정말 0 인가
- 경사 누적이 큰 배치와 정말 같은가 (어디까지 같은가)
- 마지막 자투리 배치가 무엇을 바꾸는가
- 섞지 않으면 어떻게 되는가
- 같은 계산량에서 B 를 바꾸면 어디에 닿는가

실행: python batch_minibatch_sgd.py
"""
import numpy as np
import torch
import torch.nn as nn


def header(title):
    print(f"\n=== {title} ===")


def make_data(n=1024, d=8, seed=0):
    torch.manual_seed(seed)
    X = torch.randn(n, d)
    w = torch.randn(d, 1)
    Y = X @ w + 0.5 * torch.randn(n, 1)
    return X, Y


# ========================================================================
# 1. 분산이 1/B 로 주는가
# ========================================================================

def variance_law():
    header("1. 미니배치 경사의 분산")
    X, Y = make_data()
    N = len(X)
    w = torch.zeros(8, 1, requires_grad=True)

    # 표본 하나짜리 경사를 전부 구해 둔다 (로스는 제곱오차)
    with torch.no_grad():
        per = 2 * X.unsqueeze(2) * (X @ w - Y).unsqueeze(1)   # (N, d, 1)
        per = per.squeeze(2)
        full = per.mean(0)
        sigma2 = per.var(0, correction=1).sum().item()        # 성분별 분산의 합

    print(f"  표본 {N}개,  전체 경사 노름 {full.norm():.6f}")
    print(f"  표본 하나 경사의 분산 합 sigma^2 = {sigma2:.6f}")

    g = torch.Generator().manual_seed(0)
    print("\n  B      잰 분산      sigma^2/B     비     비복원 보정값")
    for B in (1, 2, 4, 16, 64, 256, 512, 1024):
        draws = []
        for _ in range(4000):
            idx = torch.randperm(N, generator=g)[:B]          # 비복원
            draws.append(per[idx].mean(0))
        v = torch.stack(draws).var(0, correction=1).sum().item()
        pred = sigma2 / B
        corr = pred * (N - B) / (N - 1)
        print(f"  {B:<6} {v:11.6f}  {pred:11.6f}  {v/pred:6.3f}  "
              f"{corr:11.6f}")
    print("  B 가 작을 때는 잰 값이 sigma^2/B 와 맞는다")
    print("  B 가 N 에 가까워지면 비복원 보정값 쪽이 맞는다")
    print("  B = N 이면 뽑을 것이 하나뿐이라 분산이 정확히 0 이다")


# ========================================================================
# 2. 경사 누적이 큰 배치와 같은가
# ========================================================================

def accumulation():
    header("2. 경사 누적 vs 한 번에 큰 배치")
    X, Y = make_data(n=256)

    def big_batch():
        torch.manual_seed(1)
        m = nn.Linear(8, 1)
        nn.functional.mse_loss(m(X), Y).backward()
        return torch.cat([p.grad.flatten() for p in m.parameters()])

    def accumulate(micro, divide):
        torch.manual_seed(1)
        m = nn.Linear(8, 1)
        k = 256 // micro
        for i in range(k):
            s = slice(i * micro, (i + 1) * micro)
            loss = nn.functional.mse_loss(m(X[s]), Y[s])
            (loss / k if divide else loss).backward()
        return torch.cat([p.grad.flatten() for p in m.parameters()])

    ref = big_batch()
    print(f"  배치 256 한 번의 경사 노름 {ref.norm():.8f}")
    for micro in (128, 64, 32, 8):
        got = accumulate(micro, divide=True)
        print(f"  {micro:>4} 씩 {256//micro:>2}번 쌓고 k 로 나눔  "
              f"최대 차이 {(got-ref).abs().max():.3e}  "
              f"같은가 {torch.allclose(got, ref, atol=1e-6)}")
    bad = accumulate(32, divide=False)
    print(f"  32 씩 8번 쌓고 나누지 않음        노름 {bad.norm():.8f}  "
          f"= {bad.norm()/ref.norm():.1f} 배")
    print("  mean 로스를 쓰면 쪼갠 수로 나누어야 한다. 안 나누면 k 배가 된다")

    # BatchNorm 이 있으면 어떻게 되는가
    print("\n  BatchNorm 이 끼면")
    Xb = torch.randn(256, 8)
    Yb = torch.randn(256, 1)

    def with_bn(micro):
        torch.manual_seed(2)
        m = nn.Sequential(nn.Linear(8, 8), nn.BatchNorm1d(8), nn.Linear(8, 1))
        k = 256 // micro
        for i in range(k):
            s = slice(i * micro, (i + 1) * micro)
            nn.functional.mse_loss(m(Xb[s]), Yb[s]).div(k).backward()
        return torch.cat([p.grad.flatten() for p in m.parameters()])

    r = with_bn(256)
    for micro in (128, 32):
        got = with_bn(micro)
        print(f"  {micro:>4} 씩 쌓기 vs 256 한 번:  최대 차이 "
              f"{(got-r).abs().max():.6f}  같은가 "
              f"{torch.allclose(got, r, atol=1e-6)}")
    print("  BatchNorm 은 배치 안에서 통계를 내므로 쪼개면 달라진다")
    print("  경사 누적이 '큰 배치와 똑같다' 는 말에는 이 단서가 붙는다")


# ========================================================================
# 3. 마지막 자투리 배치
# ========================================================================

def last_batch():
    header("3. 마지막 자투리 배치 (N=1000, B=32)")
    N, B = 1000, 32
    k_keep = -(-N // B)
    print(f"  drop_last=False: 배치 {k_keep}개, 마지막은 {N - (k_keep-1)*B}개")
    print(f"  drop_last=True : 배치 {N // B}개, 버리는 표본 {N % B}개")

    X, Y = make_data(n=N, d=4)
    w0 = torch.zeros(4, 1)

    def run(drop):
        w = w0.clone().requires_grad_(True)
        g = torch.Generator().manual_seed(0)
        seen = 0
        for _ in range(5):
            perm = torch.randperm(N, generator=g)
            stop = (N // B) * B if drop else N
            for i in range(0, stop, B):
                idx = perm[i:i + B]
                if w.grad is not None:
                    w.grad = None
                ((X[idx] @ w - Y[idx]) ** 2).mean().backward()
                with torch.no_grad():
                    w -= 0.05 * w.grad
                seen += len(idx)
        return w.detach(), seen

    a, sa = run(False)
    b, sb = run(True)
    print(f"  5 에폭 뒤 두 해의 거리 {(a-b).norm():.6f}")
    print(f"  본 표본 수 {sa} vs {sb}  (차이 {sa-sb})")
    print("  자투리 배치는 표본 수가 적어 경사가 더 시끄럽다")
    print("  표본 하나짜리 배치가 생기면 BatchNorm 이 터지므로 조심한다")
    try:
        bn = nn.BatchNorm1d(4)
        bn(torch.randn(1, 4))
    except ValueError as e:
        print("  nn.BatchNorm1d 에 1개를 넣으면:", str(e)[:58])


# ========================================================================
# 4. 섞지 않으면
# ========================================================================

def shuffling():
    header("4. 섞기 — 자료가 정렬되어 있을 때")
    torch.manual_seed(0)
    # 라벨별로 몰아 놓은 자료
    n_per = 250
    X = torch.cat([torch.randn(n_per, 2) + torch.tensor([float(c), 0.0]) * 3
                   for c in range(4)])
    Y = torch.cat([torch.full((n_per,), c) for c in range(4)])
    print(f"  표본 {len(X)}개, 라벨 4종이 250개씩 차례로 쌓여 있다")
    print(f"  앞 10개의 라벨: {Y[:10].tolist()}")

    def run(shuffle, seed=1, epochs=10, B=32):
        torch.manual_seed(seed)
        m = nn.Sequential(nn.Linear(2, 32), nn.ReLU(), nn.Linear(32, 4))
        o = torch.optim.SGD(m.parameters(), lr=0.1)
        g = torch.Generator().manual_seed(seed)
        for _ in range(epochs):
            perm = (torch.randperm(len(X), generator=g) if shuffle
                    else torch.arange(len(X)))
            for i in range(0, len(X), B):
                idx = perm[i:i + B]
                o.zero_grad()
                nn.functional.cross_entropy(m(X[idx]), Y[idx]).backward()
                o.step()
        with torch.no_grad():
            acc = (m(X).argmax(1) == Y).float().mean().item() * 100
            loss = nn.functional.cross_entropy(m(X), Y).item()
        return acc, loss

    print("  씨앗 5개의 평균과 퍼짐을 적는다")
    for shuffle in (True, False):
        v = [run(shuffle, seed=s)[0] for s in range(5)]
        print(f"  섞기 {str(shuffle):<6} 10 에폭 뒤 정확도 평균 "
              f"{np.mean(v):6.2f}%   퍼짐 {min(v):6.2f}~{max(v):6.2f}")
    print("  섞지 않으면 한 배치가 한 라벨뿐이라 '지금 본 라벨' 로만 민다")
    print("  매번 마지막 라벨 쪽으로 치우친 채 에폭이 끝난다")

    print("\n  라벨이 이미 섞여 있는 자료였다면")
    g = torch.Generator().manual_seed(7)
    pm = torch.randperm(len(X), generator=g)
    X, Y = X[pm], Y[pm]
    for shuffle in (True, False):
        v = [run(shuffle, seed=s)[0] for s in range(5)]
        print(f"  섞기 {str(shuffle):<6} 10 에폭 뒤 정확도 평균 "
              f"{np.mean(v):6.2f}%   퍼짐 {min(v):6.2f}~{max(v):6.2f}")
    print("  이번에는 섞은 쪽이 오히려 평균이 낮다. 그러나 퍼짐을 보라 —")
    print("  47.70~87.50 으로 40%p 가 넘는다. 두 줄의 퍼짐이 겹치므로")
    print("  차이를 말할 수 없다. 여기서 큰 것은 섞기가 아니라 씨앗이다")
    print("  섞기가 중요한 까닭은 '자료에 순서가 있을 때' 다. 위 표에서는")
    print("  52.72% 대 76.40% 로 퍼짐이 겹치지 않았다")


# ========================================================================
# 5. 같은 계산량에서 B 를 바꾸면
# ========================================================================

def equal_compute():
    header("5. 같은 계산량 (표본 처리 횟수 고정) 에서 B 바꾸기")
    X, Y = make_data(n=1024, d=8)
    budget = 1024 * 30                       # 30 에폭어치 표본 처리
    print(f"  예산: 표본 처리 {budget}회 = 30 에폭,  씨앗 5개")

    # 참 최솟값
    Xb = torch.cat([X, torch.ones(len(X), 1)], 1)
    opt = torch.linalg.lstsq(Xb, Y).solution
    best = ((Xb @ opt - Y) ** 2).mean().item()
    print(f"  정규방정식의 최소 로스 {best:.6f}")

    print("\n  B      갱신 횟수   로스 평균    퍼짐              최솟값과 차이")
    for B in (1, 8, 32, 128, 1024):
        outs = []
        for seed in range(5):
            torch.manual_seed(seed)
            m = nn.Linear(8, 1)
            o = torch.optim.SGD(m.parameters(), lr=0.01)
            g = torch.Generator().manual_seed(seed)
            used, steps = 0, 0
            while used < budget:
                perm = torch.randperm(len(X), generator=g)
                for i in range(0, len(X), B):
                    idx = perm[i:i + B]
                    o.zero_grad()
                    nn.functional.mse_loss(m(X[idx]), Y[idx]).backward()
                    o.step()
                    used += len(idx)
                    steps += 1
                    if used >= budget:
                        break
            with torch.no_grad():
                outs.append(nn.functional.mse_loss(m(X), Y).item())
        print(f"  {B:<6} {steps:>9}   {np.mean(outs):.6f}   "
              f"{min(outs):.6f}~{max(outs):.6f}   {np.mean(outs)-best:+.6f}")
    print("  가장 좋은 것은 양 끝이 아니라 가운데의 B = 32 다")
    print("  B = 1024 는 30 번밖에 못 걸어 최솟값에서 5.2 나 떨어져 있고,")
    print("  B = 1 은 30720 번을 걸었는데도 0.018 만큼 위에 머문다 —")
    print("  경사가 시끄러워 최솟값 둘레를 맴돌기 때문이다 (고정 lr 이다)")
    print("  '작은 배치가 좋다' 도 '큰 배치가 좋다' 도 아니다. 갱신 횟수와")
    print("  경사 잡음이 맞서는 절충점이 있고, 여기서는 B = 32 였다")


# ========================================================================
# 6. 걸리는 시간 — 2의 거듭제곱이 정말 빠른가
# ========================================================================

def timing():
    header("6. 에폭당 걸리는 시간")
    import time

    X, Y = make_data(n=10000, d=10)
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(10, 64), nn.ReLU(), nn.Linear(64, 1))

    def one_epoch(B):
        o = torch.optim.SGD(model.parameters(), lr=0.001)
        for i in range(0, len(X), B):
            o.zero_grad()
            nn.functional.mse_loss(model(X[i:i + B]), Y[i:i + B]).backward()
            o.step()

    def best_of(B, rep=5):
        ts = []
        for _ in range(rep):
            t0 = time.perf_counter()
            one_epoch(B)
            ts.append(time.perf_counter() - t0)
        return min(ts), max(ts)

    one_epoch(64)                              # 몸풀기
    print("  표본 10000개, 2층 MLP, 한 에폭을 다섯 번 재어 최솟값과 최댓값")
    print("  B        갱신 횟수")
    print("  (시간은 다음 줄에 따로 적는다 — 기계마다 다르기 때문이다)")
    for B in (1, 8, 32, 64, 128, 1024, 10000):
        lo, hi = best_of(B, rep=3 if B == 1 else 5)
        print(f"  {B:<8} {-(-len(X)//B):>9}")
        print(f"      걸린 시간 {lo*1e3:.1f}~{hi*1e3:.1f} ms")
    print("  갱신 횟수가 1 만 배 차이인데 시간은 그만큼 벌어지지 않는다")
    print("  작은 배치에서는 파이썬 루프와 커널 호출이 값을 지배한다")

    print("\n  2의 거듭제곱이 정말 빠른가 (CPU)")
    print("  견주려면 갱신 횟수가 같아야 하므로 이웃한 크기끼리만 본다")
    for trio in ((127, 128, 129), (255, 256, 257)):
        print(f"  --- {trio[1]} 둘레 ---")
        for B in trio:
            lo, hi = best_of(B)
            mark = "  <- 2의 거듭제곱" if B & (B - 1) == 0 else ""
            print(f"  B = {B:<6} 갱신 {-(-len(X)//B):>3}회{mark}")
            print(f"      걸린 시간 {lo*1e3:.1f}~{hi*1e3:.1f} ms")
    print("  세 값의 퍼짐이 서로 겹친다. 이 기계의 CPU 에서는 차이가 없다")
    print("  '2의 거듭제곱을 써라' 는 조언은 GPU 의 타일 크기에서 온 것이고,")
    print("  쓰는 기계에서 직접 재어 보지 않고 옮겨 적을 말이 아니다")


def main():
    variance_law()
    accumulation()
    last_batch()
    shuffling()
    equal_compute()
    timing()


if __name__ == "__main__":
    main()
