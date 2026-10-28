# 경사 누적 학습 루프 — 나머지가 생길 때

「경사 누적」에서 `mean` 로스를 조각 수로 나누라고 했다. 그 규칙에 단서가 둘
붙는다. 둘 다 **데이터 수가 딱 안 떨어질 때** 생기고, 둘 다 오류를 내지 않는다.

- **남는 마이크로배치** — 누적 수로 안 나눠떨어지면 마지막 몇 개의 경사가
  조용히 버려진다
- **고르지 않은 조각** — 마지막 조각이 짧은데 같은 $1/k$를 곱하면 전체 배치와
  **다른 경사**가 나온다

$N = 14$, 배치 3, 누적 2로 재어 본다. 14는 3으로도 6으로도 안 떨어지므로 둘 다
일어난다.

## 1. 코드

```python
"""경사를 쌓는 학습 루프 — 나머지가 생길 때.

「경사 누적」에서 `mean` 로스를 조각 수로 나누라고 했다. 그 규칙에
단서가 둘 붙는다. 데이터 수가 딱 안 떨어질 때 생기는 일이다.

- **남는 마이크로배치** — 누적 수로 안 나눠떨어지면 마지막 몇 개의
  경사가 조용히 버려진다
- **고르지 않은 조각** — 마지막 조각이 짧은데 같은 1/k 를 곱하면
  전체 배치와 다른 경사가 된다

둘 다 오류를 내지 않는다. 그래서 이 쪽에서 재어 본다.

실행: 이 쪽의 코드를 파일로 갈무리해 돌린다
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
```

**출력:**

```
=== 1. 누적 루프의 뼈대 ===
  for 마이크로배치:
      로스 = criterion(...) / ACC
      로스.backward()
      if 마이크로배치 번호 % ACC == 0:
          opt.step(); opt.zero_grad()
  N=14, 배치 3 -> 마이크로배치 5개
  누적 2 로 묶으면 2번 걷고 1개가 남는다

=== 2. 남는 마이크로배치 ===
  마지막을 버린다         마이크로배치 5개  step 2번  w[:2] [0.296894, -0.065658]
  마지막도 걷는다         마이크로배치 5개  step 3번  w[:2] [0.199234, -0.010484]
  버리는 쪽은 마지막 조각의 경사가 통째로 사라진다
  에폭마다 같은 자리가 빠지므로, 그 표본들은 영영 덜 배운다

=== 3. 조각 크기가 고르지 않을 때 ===
  조각 크기: [3, 3, 3, 3, 2]
  마지막이 2개뿐인데 1/5 를 똑같이 곱한다면?
  1/조각수        전체 배치와 같은가 False   가장 큰 차 0.167252
  표본수 비례       전체 배치와 같은가 True   가장 큰 차 0.000000
  조각이 고르면 둘이 같다. 고르지 않으면 표본수로 가중해야 한다

=== 4. 제대로 적은 루프 ===
  마이크로배치 5개  step 3번  남은 것 0개
  w[:2] [0.183782, -0.400689]
  seen == N 조건 하나로 나머지를 챙긴다
  가중치를 len(xb)/N 로 두면 한 에폭이 전체 배치 한 걸음과 같아진다
```

## 2. 남는 마이크로배치

```
마지막을 버린다   마이크로배치 5개  step 2번  w[:2] [0.296894, -0.065658]
마지막도 걷는다   마이크로배치 5개  step 3번  w[:2] [0.199234, -0.010484]
```

마이크로배치가 5개인데 2개씩 묶으면 **1개가 남는다.** 흔한 꼴의 루프는

```python
if micro % ACC == 0:
    opt.step(); opt.zero_grad()
```

이므로 남은 1개는 `step()`을 못 만나고, 그 경사는 다음 에폭의 첫 묶음에
섞이거나(지우지 않았다면) 그냥 사라진다.

가중치가 0.2969와 0.1992로 **눈에 띄게 다르다.** 버린 쪽이 틀린 것이고,
오류는 안 난다.

그리고 섞지 않으면 **에폭마다 같은 표본**이 빠진다. 연습문제 2에서 그 자리를
본다.

## 3. 고르지 않은 조각에 $1/k$를 곱하면

```
조각 크기: [3, 3, 3, 3, 2]

1/조각수      전체 배치와 같은가 False   가장 큰 차 0.167252
표본수 비례    전체 배치와 같은가 True    가장 큰 차 0.000000
```

마지막 조각이 2개뿐인데 다른 조각과 **같은 $1/5$**를 곱하면, 그 두 표본이
제 몫보다 1.5배 무겁게 들어간다. 전체 배치의 경사와 0.167 어긋난다.

$\dfrac{|\text{조각}|}{N}$로 가중하면 **정확히 0** 차이로 맞는다.

$$
\sum_j \frac{n_j}{N} \cdot \frac{1}{n_j}\sum_{i \in j} \ell_i
= \frac{1}{N}\sum_i \ell_i
$$

조각 평균에 조각 크기를 곱하면 조각 합이 되고, 그것들을 더해 $N$으로 나누면
전체 평균이다. **$1/k$는 조각이 고를 때만 이것과 같아진다.**

> 조각이 고르면(즉 `drop_last=True`이면) $n_j = N/k$이므로
> $n_j/N = 1/k$다. 그래서 흔한 조언 "로스를 누적 수로 나눠라"가 대개 맞는
> 것이고, 안 맞는 자리가 정확히 마지막 조각이다.

## 4. 제대로 적은 루프

```python
for xb, yb in loader:
    (criterion(model(xb), yb) * (len(xb) / N)).backward()
    micro += 1
    seen += len(xb)
    if micro % ACC == 0 or seen == N:        # 마지막도 챙긴다
        opt.step()
        opt.zero_grad()
```

```
마이크로배치 5개  step 3번  남은 것 0개
```

두 군데를 고쳤다. **가중치를 `len(xb) / N`로** 두고, **`seen == N` 조건을
`or`로 더해** 마지막 묶음도 걷게 했다.

이렇게 하면 한 에폭이 **전체 배치 한 걸음과 정확히 같은 방향**으로 간다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
`DataLoader`의 `drop_last`가 이 문제를 어떻게 바꾸는가? 켜고 끄고 조각 크기와
본 표본 수를 세어라.

</div>

??? success "풀이"
    ```python
    from torch.utils.data import TensorDataset, DataLoader

    ds = TensorDataset(X, Y)
    for last in (False, True):
        dl = DataLoader(ds, batch_size=BATCH, drop_last=last)
        sizes = [len(xb) for xb, _ in dl]
        print(f"  drop_last={last!s:5s} 조각 {len(sizes)}개 크기 {sizes}  "
              f"본 표본 {sum(sizes)}/{N}")
    ```

    ```
      drop_last=False 조각 5개 크기 [3, 3, 3, 3, 2]  본 표본 14/14
      drop_last=True  조각 4개 크기 [3, 3, 3, 3]  본 표본 12/14
    ```

    **`drop_last=True`가 3절의 문제를 없앤다.** 조각이 전부 3개로 고르니
    $1/k$와 `len(xb)/N`이 같아진다.

    대신 **표본 두 개를 아예 안 본다.** 14개 중 12개만 학습에 쓴다.

    그래서 둘의 거래가 이렇게 된다.

    | | 얻는 것 | 잃는 것 |
    |---|---|---|
    | `drop_last=True` | 조각이 고르다, 코드가 단순하다 | 매 에폭 일부 표본 |
    | `drop_last=False` | 전부 본다 | 가중치와 나머지를 챙겨야 한다 |

    데이터가 많으면 앞쪽이 흔하다 — 1%도 안 되는 표본을 잃을 뿐이고,
    배치 정규화도 조각 크기가 고른 쪽을 좋아한다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
`drop_last=True`로 버려지는 표본이 **에폭마다 같은가?** 셔플을 켜고 끄고
확인하라.

</div>

??? success "풀이"
    ```python
    X = torch.arange(N).float().unsqueeze(1)
    ds = TensorDataset(X, torch.zeros(N, 1))

    print("셔플 없이")
    dl = DataLoader(ds, batch_size=3, drop_last=True, shuffle=False)
    for e in range(3):
        seen = torch.cat([xb[:, 0] for xb, _ in dl])
        print(f"  에폭 {e + 1}  못 본 표본",
              sorted(set(range(N)) - {int(v) for v in seen}))

    print("셔플을 켜면")
    g = torch.Generator().manual_seed(0)
    dl2 = DataLoader(ds, batch_size=3, drop_last=True, shuffle=True, generator=g)
    for e in range(3):
        seen = torch.cat([xb[:, 0] for xb, _ in dl2])
        print(f"  에폭 {e + 1}  못 본 표본",
              sorted(set(range(N)) - {int(v) for v in seen}))
    ```

    ```
    셔플 없이
      에폭 1  못 본 표본 [12, 13]
      에폭 2  못 본 표본 [12, 13]
      에폭 3  못 본 표본 [12, 13]
    셔플을 켜면
      에폭 1  못 본 표본 [5, 8]
      에폭 2  못 본 표본 [0, 10]
      에폭 3  못 본 표본 [1, 6]
    ```

    **셔플이 없으면 12번과 13번이 영영 학습에 안 들어간다.** 백 에폭을 돌려도
    그 둘은 한 번도 안 본다.

    셔플을 켜면 매 에폭 다른 둘이 빠지므로, 길게 보면 모든 표본이 거의 같은
    횟수만큼 들어간다. $14/14 \times (12/14)$의 비율로 줄어들 뿐이다.

    그래서 **`drop_last=True`는 `shuffle=True`와 한 짝**이다. 학습 로더에
    셔플을 켜는 까닭이 경사의 잡음을 바꾸는 것만이 아니라 여기에도 있다.

    평가 로더에서는 `drop_last=False`여야 한다. 평가는 **모든 표본을 봐야**
    하고, 조각 크기가 고르지 않아도 가중 평균을 내면 된다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
스케줄러를 `micro % ACC == 0` 안이 아니라 **마이크로배치마다** 부르면 어떻게
되는가? `StepLR`로 세 에폭을 돌려 러닝레이트를 찍어 보라.

</div>

??? success "풀이"
    ```python
    for label, inside in (("step 마다", True), ("마이크로배치마다", False)):
        m = fresh(W0)
        opt = torch.optim.SGD(m.parameters(), lr=0.1)
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=2, gamma=0.5)
        micro, lrs = 0, []
        for _ in range(3):
            for i in range(0, N, BATCH):
                xb, yb = X[i:i + BATCH], Y[i:i + BATCH]
                (F.mse_loss(m(xb), yb) * (len(xb) / N)).backward()
                micro += 1
                if micro % ACC == 0:
                    opt.step(); opt.zero_grad()
                    if inside:
                        sched.step(); lrs.append(opt.param_groups[0]["lr"])
                if not inside:
                    sched.step(); lrs.append(opt.param_groups[0]["lr"])
        print(f"  {label:16s}", [round(v, 5) for v in lrs])
    ```

    ```
      step 마다         [0.1, 0.05, 0.05, 0.025, 0.025, 0.0125, 0.0125]
      마이크로배치마다    [0.1, 0.05, 0.05, 0.025, 0.025, 0.0125, 0.0125,
                        0.00625, 0.00625, 0.00313, 0.00313, 0.00156,
                        0.00156, 0.00078, 0.00078]
    ```

    **두 배 빨리 줄어든다.** 세 에폭 끝에 0.0125여야 할 러닝레이트가
    0.00078이다 — 열여섯 배 작다.

    까닭은 셈이 그대로다. 마이크로배치가 15번, 걸음이 7번이므로 스케줄러를
    마이크로배치마다 부르면 **두 배 넘게** 진행한다. 누적 수가 8이라면 여덟
    배가 된다.

    그래서 규칙이 하나 더 붙는다. **스케줄러는 `opt.step()`과 같은 자리에
    둔다.** 둘은 한 짝이고, 그 짝을 떼어 놓으면 스케줄이 설정과 다른 속도로
    간다.

    같은 이야기가 로그와 조기 종료에도 걸린다. "1000 스텝마다 저장"이라고 적은
    코드가 누적을 켠 순간 **$k$배 자주** 저장하게 된다.
