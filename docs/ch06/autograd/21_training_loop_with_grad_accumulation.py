"""경사를 쌓는 학습 루프 — 나머지가 생길 때.

「경사 누적」에서 `mean` 로스를 조각 수로 나누라고 했다. 그 규칙에
단서가 둘 붙는다. 데이터 수가 딱 안 떨어질 때 생기는 일이다.

- **남는 마이크로배치** — 누적 수로 안 나눠떨어지면 마지막 몇 개의
  경사가 조용히 버려진다
- **고르지 않은 조각** — 마지막 조각이 짧은데 같은 1/k 를 곱하면
  전체 배치와 다른 경사가 된다

둘 다 오류를 내지 않는다. 그래서 이 쪽에서 재어 본다.

실행: python 21_training_loop_with_grad_accumulation.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def header(title):
    print(f"\n=== {title} ===")


N, D, BATCH, ACC = 14, 4, 3, 2      # 14 는 3 으로도 6 으로도 안 떨어진다


def setup():
    torch.manual_seed(0)
    return torch.randn(N, D), torch.randn(N, 1), torch.randn(D, 1)


def fresh(W0):
    torch.manual_seed(0)
    m = nn.Linear(D, 1)
    with torch.no_grad():
        m.weight.copy_(W0.view(1, D))
    return m


# ========================================================================
# 1. 루프의 뼈대
# ========================================================================

def skeleton():
    header("1. 누적 루프의 뼈대")
    print("  for 마이크로배치:")
    print("      로스 = criterion(...) / ACC")
    print("      로스.backward()")
    print("      if 마이크로배치 번호 % ACC == 0:")
    print("          opt.step(); opt.zero_grad()")
    print(f"  N={N}, 배치 {BATCH} -> 마이크로배치 {-(-N // BATCH)}개")
    print(f"  누적 {ACC} 로 묶으면 {-(-N // BATCH) // ACC}번 걷고 "
          f"{-(-N // BATCH) % ACC}개가 남는다")


# ========================================================================
# 2. 남는 마이크로배치
# ========================================================================

def leftover():
    header("2. 남는 마이크로배치")
    X, Y, W0 = setup()
    for label, handle_tail in (("마지막을 버린다", False),
                               ("마지막도 걷는다", True)):
        m = fresh(W0)
        opt = torch.optim.SGD(m.parameters(), lr=0.1)
        micro = steps = 0
        for i in range(0, N, BATCH):
            (F.mse_loss(m(X[i:i + BATCH]), Y[i:i + BATCH]) / ACC).backward()
            micro += 1
            if micro % ACC == 0:
                opt.step()
                opt.zero_grad()
                steps += 1
        if micro % ACC and handle_tail:
            opt.step()
            opt.zero_grad()
            steps += 1
        print(f"  {label:16s} 마이크로배치 {micro}개  step {steps}번  "
              f"w[:2] {[round(v, 6) for v in m.weight.detach().flatten()[:2].tolist()]}")
    print("  버리는 쪽은 마지막 조각의 경사가 통째로 사라진다")
    print("  에폭마다 같은 자리가 빠지므로, 그 표본들은 영영 덜 배운다")


# ========================================================================
# 3. 고르지 않은 조각에 1/k 를 곱하면
# ========================================================================

def uneven():
    header("3. 조각 크기가 고르지 않을 때")
    X, Y, W0 = setup()
    sizes = [len(X[i:i + BATCH]) for i in range(0, N, BATCH)]
    print("  조각 크기:", sizes)
    print(f"  마지막이 {sizes[-1]}개뿐인데 1/{len(sizes)} 를 똑같이 곱한다면?")

    m = fresh(W0)
    F.mse_loss(m(X), Y).backward()
    exact = m.weight.grad.flatten().clone()

    for label, mode in (("1/조각수", "count"), ("표본수 비례", "size")):
        m2 = fresh(W0)
        for i in range(0, N, BATCH):
            xb, yb = X[i:i + BATCH], Y[i:i + BATCH]
            w = 1 / len(sizes) if mode == "count" else len(xb) / N
            (F.mse_loss(m2(xb), yb) * w).backward()
        g = m2.weight.grad.flatten()
        print(f"  {label:12s} 전체 배치와 같은가 {torch.allclose(g, exact, atol=1e-6)}"
              f"   가장 큰 차 {(g - exact).abs().max():.6f}")
    print("  조각이 고르면 둘이 같다. 고르지 않으면 표본수로 가중해야 한다")


# ========================================================================
# 4. 제대로 적은 루프
# ========================================================================

def correct_loop():
    header("4. 제대로 적은 루프")
    X, Y, W0 = setup()
    m = fresh(W0)
    opt = torch.optim.SGD(m.parameters(), lr=0.1)

    micro = steps = seen = 0
    pending = 0
    for i in range(0, N, BATCH):
        xb, yb = X[i:i + BATCH], Y[i:i + BATCH]
        (F.mse_loss(m(xb), yb) * (len(xb) / N)).backward()   # 표본수로 가중
        micro += 1
        pending += len(xb)
        seen += len(xb)
        if micro % ACC == 0 or seen == N:                    # 마지막도 걷는다
            opt.step()
            opt.zero_grad()
            steps += 1
            pending = 0
    print(f"  마이크로배치 {micro}개  step {steps}번  남은 것 {pending}개")
    print(f"  w[:2] {[round(v, 6) for v in m.weight.detach().flatten()[:2].tolist()]}")
    print("  seen == N 조건 하나로 나머지를 챙긴다")
    print("  가중치를 len(xb)/N 로 두면 한 에폭이 전체 배치 한 걸음과 같아진다")


def main():
    skeleton()
    leftover()
    uneven()
    correct_loop()


if __name__ == "__main__":
    main()
