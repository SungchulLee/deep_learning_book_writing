"""배치 · 미니배치 · SGD 를 공정하게 견주기.

셋을 견주는 코드는 흔하지만, 공정하게 견주는 코드는 드물다. 흔히
세 군데서 어긋난다.

1. 모델을 따로 만들어 **출발점이 다르다**
2. 에폭 수를 맞춰 **갱신 횟수가 다르다**
3. 에폭 도중의 로스를 평균 내어 **서로 다른 파라메터의 값을 섞는다**

이 쪽은 셋을 차례로 고치고, 고칠 때마다 결론이 어떻게 바뀌는지 본다.
개념은 「배치, 미니배치, 확률적 경사 하강법」 쪽에서 다룬다.

실행: python 01_batch_vs_minibatch_sgd.py
"""
import numpy as np
import torch
import torch.nn as nn

N, D = 1000, 10
LR = 0.01


def header(title):
    print(f"\n=== {title} ===")


def make_data(seed=0):
    torch.manual_seed(seed)
    X = torch.randn(N, D)
    w = torch.randn(D, 1)
    Y = X @ w + 2.0 + 0.5 * torch.randn(N, 1)
    return X, Y


X, Y = make_data()


def fresh_model(seed=7):
    """모든 견줌이 **같은 출발점** 에서 시작하게 한다."""
    torch.manual_seed(seed)
    return nn.Linear(D, 1)


def full_loss(model):
    """에폭이 끝난 뒤 같은 전체 자료에서 다시 잰다."""
    with torch.no_grad():
        return nn.functional.mse_loss(model(X), Y).item()


def run(batch, epochs, lr=LR, seed=7, data_seed=0):
    model = fresh_model(seed)
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    g = torch.Generator().manual_seed(data_seed)
    mixed, honest, steps = [], [], 0
    for _ in range(epochs):
        perm = torch.randperm(N, generator=g)
        seen = []
        for i in range(0, N, batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            loss = nn.functional.mse_loss(model(X[idx]), Y[idx])
            loss.backward()
            opt.step()
            seen.append(loss.item())       # 갱신 **전** 파라메터에서 잰 값
            steps += 1
        mixed.append(float(np.mean(seen)))  # 서로 다른 파라메터의 값을 섞는다
        honest.append(full_loss(model))     # 에폭 끝 파라메터에서 다시 잰 값
    return model, mixed, honest, steps


# ========================================================================
# 1. 흔한 꼴 — 세 군데가 어긋난 견줌
# ========================================================================

def unfair():
    header("1. 흔한 꼴: 에폭을 맞추고 에폭 평균 로스를 적는다")
    print("  (모델도 따로 만들어 출발점이 다르다)")
    print("  방식           갱신 횟수   마지막 에폭 평균 로스")
    out = {}
    for batch, lab in ((N, "배치 GD   "), (32, "미니배치 32"), (1, "SGD       ")):
        torch.manual_seed(batch)                   # 일부러 씨앗을 다르게
        m = nn.Linear(D, 1)
        opt = torch.optim.SGD(m.parameters(), lr=LR)
        g = torch.Generator().manual_seed(0)
        steps = 0
        for _ in range(30):
            perm = torch.randperm(N, generator=g)
            seen = []
            for i in range(0, N, batch):
                idx = perm[i:i + batch]
                opt.zero_grad()
                loss = nn.functional.mse_loss(m(X[idx]), Y[idx])
                loss.backward()
                opt.step()
                seen.append(loss.item())
                steps += 1
            last = float(np.mean(seen))
        out[lab] = last
        print(f"  {lab}   {steps:>8}   {last:>20.6f}")
    print("  이 표만 보면 SGD 가 압도적으로 좋아 보인다")
    print("  그런데 세 줄이 서로 다른 세 가지를 재고 있다")


# ========================================================================
# 2. 첫째를 고친다 — 같은 출발점
# ========================================================================

def same_start():
    header("2. 고침 (1): 같은 출발점에서 시작한다")
    m0 = fresh_model()
    print(f"  세 방식 모두 같은 초기 모델에서 출발한다")
    print(f"  초기 전체 로스 {full_loss(m0):.6f}")
    print("  방식           갱신 횟수   에폭 평균     에폭 끝 전체 로스")
    res = {}
    for batch, lab in ((N, "배치 GD   "), (32, "미니배치 32"), (1, "SGD       ")):
        _, mixed, honest, steps = run(batch, 30)
        res[lab] = (steps, mixed[-1], honest[-1])
        print(f"  {lab}   {steps:>8}   {mixed[-1]:10.6f}   {honest[-1]:14.6f}")
    print("  에폭 평균과 전체 로스가 다르다. 특히 SGD 에서 크게 벌어진다")
    for lab, (s, mx, hn) in res.items():
        print(f"    {lab} 차이 {abs(mx-hn):.6f}  ({abs(mx-hn)/hn*100:.1f}%)")
    return res


# ========================================================================
# 3. 둘째를 고친다 — 섞인 평균 대신 정직한 로스
# ========================================================================

def why_mixed_lies():
    header("3. 고침 (2): 에폭 평균이 왜 거짓말을 하는가")
    _, mixed, honest, _ = run(1, 30)
    print("  SGD 의 처음 다섯 에폭")
    print("  에폭   에폭 평균      에폭 끝 전체 로스    차이")
    for e in range(5):
        print(f"  {e:>4}   {mixed[e]:10.6f}   {honest[e]:16.6f}   "
              f"{mixed[e]-honest[e]:+.6f}")
    print("  첫 에폭에서 +0.45 로 크게 벌어진다. 에폭 **도중** 의 파라메터로")
    print("  잰 값들이 섞여 있고, 학습 초기에는 그 자리들이 에폭 끝보다")
    print("  한참 나쁘기 때문이다")
    print("  수렴한 뒤에는 부호가 뒤섞인다 (에폭 2 는 -0.003 이다) — 더 이상")
    print("  한 방향으로 움직이지 않으므로 평균이 위도 아래도 될 수 있다")

    print("\n  배치별로 그 어긋남을 잰다 (30 에폭째)")
    print("  배치    에폭 평균     전체 로스    어긋남")
    for batch in (1, 8, 32, 128, N):
        _, mx, hn, _ = run(batch, 30)
        lab = "전체" if batch == N else str(batch)
        print(f"  {lab:<6}  {mx[-1]:10.6f}  {hn[-1]:10.6f}  "
              f"{mx[-1]-hn[-1]:+.6f}")
    print("  배치 1 에서 8, 32, 128 로 갈수록 어긋남이 준다 (0.011 -> 0.0004)")
    print("  에폭 안에서 덜 움직이기 때문이다")
    print("  그런데 배치 GD 가 +0.22 로 가장 크다. 거꾸로 보이지만 까닭이")
    print("  다르다 — 에폭에 갱신이 한 번뿐이라 '평균' 이 곧 **에폭 처음**")
    print("  파라메터의 로스다. 정확히 한 걸음 뒤처진 값이고, 아직 멀리")
    print("  있어 한 걸음의 이득이 크다. 셋 다 같은 병이다: 에폭 평균은")
    print("  에폭 끝 모델의 성능이 아니다")


# ========================================================================
# 4. 셋째를 고친다 — 갱신 횟수를 맞춘다
# ========================================================================

def same_updates():
    header("4. 고침 (3): 갱신 횟수를 맞춘다")
    target = 960                      # 미니배치 32 로 30 에폭 할 때의 걸음 수
    print(f"  모두 약 {target} 번 갱신하도록 에폭을 조절한다")
    print("  배치   에폭    갱신 횟수   전체 로스     본 표본 수")
    for batch in (1, 8, 32, 128, N):
        per_epoch = -(-N // batch)
        epochs = max(1, round(target / per_epoch))
        _, _, honest, steps = run(batch, epochs)
        lab = "전체" if batch == N else str(batch)
        print(f"  {lab:<5}  {epochs:>5}  {steps:>9}   {honest[-1]:10.6f}   "
              f"{epochs*N:>10}")
    print("  이제 '갱신 횟수' 열이 거의 같다. 대신 '본 표본 수' 가 어긋난다")
    print("  배치 GD 는 960 번 갱신하려고 자료를 960 번 훑어야 한다")
    print("  둘을 동시에 맞출 수는 없다 — 이것이 배치 크기 견줌의 본질이다")


# ========================================================================
# 5. 계산량을 맞추면
# ========================================================================

def same_compute():
    header("5. 계산량(본 표본 수)을 맞추면")
    print("  모두 30 에폭 = 표본 처리 30000 회로 고정하고 씨앗 5개")
    print("  배치    전체 로스 평균   퍼짐                  갱신 횟수")
    for batch in (1, 8, 32, 128, N):
        vals, steps = [], 0
        for s in range(5):
            _, _, honest, steps = run(batch, 30, seed=s, data_seed=s)
            vals.append(honest[-1])
        lab = "전체" if batch == N else str(batch)
        print(f"  {lab:<6}  {np.mean(vals):12.6f}   "
              f"{min(vals):.6f}~{max(vals):.6f}   {steps:>8}")
    A = torch.cat([X, torch.ones(N, 1)], 1)
    sol = torch.linalg.lstsq(A, Y).solution
    print(f"  정규방정식의 최소 로스 {((A @ sol - Y) ** 2).mean().item():.6f}")
    print("  가장 좋은 것은 양 끝이 아니라 가운데의 배치 32 다 (0.246200)")
    print("  — 최소 로스 0.246098 에 1e-4 까지 닿았다")
    print("  배치 전체는 30 걸음뿐이라 5.80 에 머물고, 배치 1 은 30000 번을")
    print("  걷고도 0.2688 이다. 경사가 시끄러워 최솟값 둘레를 맴돈다")
    print("  퍼짐을 보라 — 배치 1 은 0.2613~0.2790 으로 배치 32 의")
    print("  0.246144~0.246250 보다 200 배 넓다. 고정 lr 의 대가다")


# ========================================================================
# 6. 옵티마이저를 바꾸면
# ========================================================================

def optimizers():
    header("6. SGD 대신 Adam 을 쓰면")
    print("  배치 32, 30 에폭, 씨앗 5개")
    print("  옵티마이저         lr       전체 로스 평균   퍼짐")
    for name, make, lr in (
            ("SGD              ", torch.optim.SGD, 0.01),
            ("SGD(momentum=0.9)", lambda p, lr: torch.optim.SGD(
                p, lr=lr, momentum=0.9), 0.01),
            ("Adam             ", torch.optim.Adam, 0.01),
            ("Adam             ", torch.optim.Adam, 0.001)):
        vals = []
        for s in range(5):
            m = fresh_model(s)
            opt = make(m.parameters(), lr)
            g = torch.Generator().manual_seed(s)
            for _ in range(30):
                perm = torch.randperm(N, generator=g)
                for i in range(0, N, 32):
                    idx = perm[i:i + 32]
                    opt.zero_grad()
                    nn.functional.mse_loss(m(X[idx]), Y[idx]).backward()
                    opt.step()
            vals.append(full_loss(m))
        print(f"  {name}  {lr:<7}  {np.mean(vals):12.6f}   "
              f"{min(vals):.6f}~{max(vals):.6f}")
    print("  'Adam 이 SGD 보다 빠르다' 는 러닝레이트에 딸린 말이다")
    print("  같은 lr = 0.01 에서는 셋이 사실상 같은 자리에 닿는다")
    print("  (0.246200 / 0.252163 / 0.246264 — 모멘텀 쪽이 오히려 조금 나쁘다)")
    print("  Adam 의 흔한 기본값 lr = 0.001 은 6.76 으로 한참 못 미친다 —")
    print("  960 번의 갱신으로는 그 걸음 폭이 모자란다")
    print("  옵티마이저를 견주려면 각자의 러닝레이트를 따로 맞춰 주어야 하고,")
    print("  그 전의 비교는 옵티마이저가 아니라 lr 을 견준 것이다")


def main():
    unfair()
    same_start()
    why_mixed_lies()
    same_updates()
    same_compute()
    optimizers()


if __name__ == "__main__":
    main()
