"""러닝레이트 — 얼마나 걸을 것인가.

러닝레이트에 대한 조언은 대개 "너무 크면 발산하고 너무 작으면 느리다" 로
끝난다. 맞는 말이지만 쓸 수 있는 말은 아니다. 이 쪽은 수로 바꾼다.

- 수렴 경계 eta < 2/a 가 정말 경계인가
- "최적 러닝레이트는 1/L" 은 어디까지 참인가 (조건수가 1이 아니면 틀리다)
- 러닝레이트 탐색기가 그리는 곡선은 어떻게 생겼나
- 흔한 스케줄들이 실제로 내놓는 값
- 배치를 k 배 키우면 러닝레이트도 k 배인가

MNIST 를 쓰는 부분은 저장소 뿌리에서 돌리는 것을 전제로 한다 (./data).

실행: python learning_rate.py
"""
import math

import numpy as np
import torch
import torch.nn as nn

DEV = "mps" if torch.backends.mps.is_available() else "cpu"


def header(title):
    print(f"\n=== {title} ===")


# ========================================================================
# 1. 1차원 이차함수 — 경계가 정말 2/a 인가
# ========================================================================

def one_dim_boundary():
    header("1. L = a(x-x*)^2 / 2 에서 수렴 경계")
    a, x0 = 4.0, 1.0           # 2/a = 0.5, 1/a = 0.25
    print(f"  a = {a},  1/a = {1/a},  2/a = {2/a}")
    print("  eta      50 걸음 뒤 |x - x*|     1-eta*a")
    for eta in (0.05, 0.25, 0.45, 0.49, 0.5, 0.51, 0.6):
        x = x0
        for _ in range(50):
            x = x - eta * a * x
        r = 1 - eta * a
        shown = f"{abs(x):.6e}" if abs(x) < 1e30 else "넘침"
        print(f"  {eta:<7} {shown:>18}   {r:+.4f}")
    print("  eta = 0.25 에서 한 걸음 만에 정확히 0 이 된다 (1-eta*a = 0)")
    print("  eta = 0.5 에서 |1-eta*a| = 1 — 줄지도 늘지도 않고 튕긴다")
    print("  eta = 0.51 부터 늘어난다. 경계는 식이 말한 그대로 2/a 다")

    print("\n  eta = 0.5 에서 x 가 어떻게 되는지 다섯 걸음만:")
    x = x0
    for t in range(5):
        x = x - 0.5 * a * x
        print(f"    걸음 {t+1}: x = {x:+.4f}")
    print("    부호만 뒤집으며 제자리다. 로스도 그대로다")


# ========================================================================
# 2. 조건수가 1이 아니면 1/L 은 최적이 아니다
# ========================================================================

def condition_number():
    header("2. 2차원 — 최적 러닝레이트는 1/L 이 아니다")
    # L(x) = (mu x1^2 + L x2^2) / 2,  고윳값 mu 와 L
    mu, LL = 1.0, 20.0
    kappa = LL / mu
    print(f"  고윳값 {mu} 와 {LL},  조건수 kappa = {kappa}")
    print(f"  책에 흔한 1/L = {1/LL:.6f}")
    print(f"  참 최적 2/(L+mu) = {2/(LL+mu):.6f}")

    def run(eta, steps=200):
        x = np.array([1.0, 1.0])
        for _ in range(steps):
            x = x - eta * np.array([mu * x[0], LL * x[1]])
            if not np.isfinite(x).all():
                return float("inf")
        return float(np.linalg.norm(x))

    print("\n  eta         200 걸음 뒤 ||x||      수렴률 max|1-eta*lam|")
    # 1/L = 0.05 이고 2/L = 0.1 이라 겹치는 값은 한 번만 적는다
    for eta in (0.01, 1 / LL, 0.0952, 2 / (LL + mu), 0.0953, 2 / LL, 0.11):
        rate = max(abs(1 - eta * mu), abs(1 - eta * LL))
        v = run(eta)
        shown = f"{v:.6e}" if math.isfinite(v) else "넘침"
        print(f"  {eta:<11.6f} {shown:>18}   {rate:.6f}")

    best = (kappa - 1) / (kappa + 1)
    print(f"\n  이론이 말하는 최적 수렴률 (kappa-1)/(kappa+1) = {best:.6f}")
    print(f"  2/(L+mu) 에서의 수렴률 = "
          f"{max(abs(1-2/(LL+mu)*mu), abs(1-2/(LL+mu)*LL)):.6f}  — 같다")
    print(f"  1/L 에서의 수렴률       = "
          f"{max(abs(1-1/LL*mu), abs(1-1/LL*LL)):.6f}  — 더 나쁘다")
    print("  1/L 은 '발산하지 않음' 을 보장하는 값이지 가장 빠른 값이 아니다")
    print("  두 값의 차이는 조건수가 클수록 벌어진다")

    print("\n  조건수를 바꿔 가며 1/L 과 2/(L+mu) 가 걸리는 걸음 수:")
    print("  kappa   1/L 로 1e-6 까지   2/(L+mu) 로 1e-6 까지")
    for k in (2, 10, 100, 1000):
        m2, l2 = 1.0, float(k)
        out = []
        for eta in (1 / l2, 2 / (l2 + m2)):
            rate = max(abs(1 - eta * m2), abs(1 - eta * l2))
            out.append(math.ceil(math.log(1e-6) / math.log(rate)))
        print(f"  {k:<7} {out[0]:>13}   {out[1]:>18}")


# ========================================================================
# 3. 러닝레이트 탐색기
# ========================================================================

def lr_finder():
    header("3. 러닝레이트 탐색기 (MNIST 1000장, MLP)")
    from torchvision import datasets, transforms

    tf = transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize((0.1307,), (0.3081,))])
    train = datasets.MNIST("./data", train=True, download=False, transform=tf)
    idx = torch.arange(1000)
    X = torch.stack([train[i][0] for i in idx]).view(1000, -1).to(DEV)
    Y = torch.tensor([train[i][1] for i in idx]).to(DEV)

    def fresh():
        torch.manual_seed(0)
        return nn.Sequential(nn.Linear(784, 128), nn.ReLU(),
                             nn.Linear(128, 10)).to(DEV)

    lrs = np.logspace(-6, 1, 40)
    losses = []
    for lr in lrs:
        m = fresh()
        opt = torch.optim.SGD(m.parameters(), lr=float(lr))
        for _ in range(20):                    # 같은 20 걸음씩만
            opt.zero_grad()
            loss = nn.functional.cross_entropy(m(X), Y)
            loss.backward()
            opt.step()
        losses.append(float(loss.item()))

    start = float(nn.functional.cross_entropy(fresh()(X), Y).item())
    print(f"  시작 로스 {start:.4f}  (20 걸음 전)")
    print("  lr          20 걸음 뒤 로스")
    for lr, l in zip(lrs, losses):
        if not math.isfinite(l):
            mark, shown = "발산", "nan"
        elif l > start:
            mark, shown = "나빠짐", f"{l:.4f}"
        else:
            mark, shown = "", f"{l:.4f}"
        print(f"  {lr:<11.3e} {shown:>10}  {mark}")

    fin = [(lr, l) for lr, l in zip(lrs, losses) if math.isfinite(l)]
    best_lr, best_l = min(fin, key=lambda p: p[1])
    print(f"\n  가장 낮은 로스 {best_l:.4f} 가 lr = {best_lr:.3e} 에서 나왔다")
    diverged = [lr for lr, l in zip(lrs, losses)
                if not math.isfinite(l) or l > start]
    if diverged:
        print(f"  로스가 시작값보다 나빠지기 시작한 가장 작은 lr = "
              f"{min(diverged):.3e}")
    print("  흔한 조언은 '발산 지점보다 한 자릿수 아래' 를 고르라는 것이다")
    print("  이 곡선에서 그 값은 대략", f"{min(diverged)/10:.3e}" if diverged
          else "(발산 지점 없음)")
    return lrs, losses, start      # figures/make_figures.py 가 이 값을 쓴다


# ========================================================================
# 4. 스케줄이 실제로 내놓는 값
# ========================================================================

def schedules():
    header("4. 스케줄이 내놓는 값 (lr0 = 0.1, 20 에폭)")
    p = nn.Parameter(torch.zeros(1))

    def trace(make, n=20):
        opt = torch.optim.SGD([p], lr=0.1)
        sch = make(opt)
        out = []
        for _ in range(n):
            out.append(opt.param_groups[0]["lr"])
            opt.step()
            sch.step()
        return out

    rows = {
        "StepLR(5, 0.5)": trace(
            lambda o: torch.optim.lr_scheduler.StepLR(o, 5, 0.5)),
        "Exponential(0.9)": trace(
            lambda o: torch.optim.lr_scheduler.ExponentialLR(o, 0.9)),
        "Cosine(T=20)": trace(
            lambda o: torch.optim.lr_scheduler.CosineAnnealingLR(o, 20)),
        "Linear warmup 5": trace(
            lambda o: torch.optim.lr_scheduler.LambdaLR(
                o, lambda e: min(1.0, (e + 1) / 5))),
    }
    print("  에폭  " + "  ".join(f"{k:>16}" for k in rows))
    for e in range(20):
        print(f"  {e:>4}  " + "  ".join(f"{v[e]:>16.6f}"
                                        for v in rows.values()))
    for k, v in rows.items():
        print(f"  {k:<18} 마지막 {v[-1]:.6f}  처음 대비 "
              f"{v[-1]/v[0]:.4f}")
    print("  코사인은 마지막에 0 으로 간다. 워밍업은 5 에폭에 제값에 닿는다")


# ========================================================================
# 5. 배치를 k 배 키우면 러닝레이트도 k 배인가
# ========================================================================

def linear_scaling():
    header("5. 선형 비례 규칙 (MNIST 6000장, 3 에폭, 씨앗 3개)")
    from torchvision import datasets, transforms

    tf = transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize((0.1307,), (0.3081,))])
    train = datasets.MNIST("./data", train=True, download=False, transform=tf)
    test = datasets.MNIST("./data", train=False, download=False, transform=tf)
    Xtr = torch.stack([train[i][0] for i in range(6000)]).view(6000, -1).to(DEV)
    Ytr = torch.tensor([train[i][1] for i in range(6000)]).to(DEV)
    Xte = torch.stack([test[i][0] for i in range(2000)]).view(2000, -1).to(DEV)
    Yte = torch.tensor([test[i][1] for i in range(2000)]).to(DEV)

    def run(batch, lr, seed, epochs=3):
        torch.manual_seed(seed)
        m = nn.Sequential(nn.Linear(784, 128), nn.ReLU(),
                          nn.Linear(128, 10)).to(DEV)
        opt = torch.optim.SGD(m.parameters(), lr=lr)
        g = torch.Generator().manual_seed(seed)
        steps = 0
        for _ in range(epochs):
            perm = torch.randperm(6000, generator=g).to(DEV)
            for i in range(0, 6000, batch):
                b = perm[i:i + batch]
                opt.zero_grad()
                nn.functional.cross_entropy(m(Xtr[b]), Ytr[b]).backward()
                opt.step()
                steps += 1
        with torch.no_grad():
            acc = (m(Xte).argmax(1) == Yte).float().mean().item() * 100
        return acc, steps

    base_b, base_lr = 32, 0.05
    print(f"  기준: 배치 {base_b}, lr {base_lr}")
    print("  배치   걸음   lr(k배)   정확도 평균  퍼짐         lr 그대로일 때")
    for k in (1, 2, 4, 8, 16):
        b = base_b * k
        scaled = [run(b, base_lr * k, s) for s in (0, 1, 2)]
        same = [run(b, base_lr, s) for s in (0, 1, 2)]
        acc_s = [a for a, _ in scaled]
        acc_m = [a for a, _ in same]
        print(f"  {b:<6} {scaled[0][1]:<6} {base_lr*k:<9.3f} "
              f"{np.mean(acc_s):6.2f}%  {min(acc_s):5.2f}~{max(acc_s):5.2f}   "
              f"{np.mean(acc_m):6.2f}% ({min(acc_m):5.2f}~{max(acc_m):5.2f})")
    print("  k = 8 까지는 lr 을 함께 키운 쪽이 낫다. k = 16 에서 뒤집힌다 —")
    print("  비례한 lr 0.8 의 퍼짐이 20%p 를 넘고 평균도 한참 낮다")

    # 그런데 걸음 수가 함께 줄었다. 둘을 갈라 놓고 다시 본다.
    print("\n  배치 512 에서 걸음 수를 맞춰 주면 (씨앗 3개)")
    for ep in (3, 48):
        for lr in (0.8, 0.05):
            v = [run(512, lr, s, epochs=ep) for s in (0, 1, 2)]
            acc = [a for a, _ in v]
            print(f"    {ep:>2} 에폭 ({v[0][1]:>3} 걸음)  lr {lr:<5} "
                  f"{np.mean(acc):6.2f}%  퍼짐 {min(acc):5.2f}~{max(acc):5.2f}")
    print("  576 걸음으로 (배치 32 의 564 걸음과 거의 같게) 맞추면")
    print("  lr 0.8 도 89.02% 까지 올라온다")
    print("  k = 16 의 참사는 비례 규칙이 깨져서가 아니라 36 걸음밖에")
    print("  돌지 않아서였다. 배치를 16배 키우면 에폭당 걸음이 16배 준다")
    print("  다만 걸음을 맞춘 뒤에도 lr 을 키운 쪽이 더 낫지는 않다")
    print("  (89.02% 대 89.67%). 이 문제 크기에서 비례 규칙의 이득은 없다")


def main():
    one_dim_boundary()
    condition_number()
    lr_finder()
    schedules()
    linear_scaling()


if __name__ == "__main__":
    main()
