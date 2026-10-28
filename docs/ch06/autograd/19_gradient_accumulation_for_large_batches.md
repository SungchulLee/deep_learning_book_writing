# 큰 배치를 위한 경사 누적 — 값이 얼마인가

「경사 누적」에서 쪼개서 쌓은 것이 통째로 한 것과 **같은 경사**를 준다는 것을
확인했다(`mean` 로스면 조각 수로 나눈다는 단서와 함께). 이 쪽은 그 다음 물음을
본다.

**값이 얼마인가.** 쪼개면 활성값 메모리가 조각 크기에 비례해 줄어들고, 대신
순전파와 역전파를 여러 번 해야 하니 시간이 는다. 두 축을 재면 조각 수를 어떻게
고를지 가늠이 선다.

## 1. 코드

```python
"""경사 누적의 값 — 메모리를 아끼고 시간을 치른다.

「경사 누적」에서 쪼개서 쌓은 것이 통째로 한 것과 **같은 경사**를
준다는 것을 확인했다. 이 쪽은 그 다음 물음을 본다 — **값이 얼마인가.**

쪼개면 활성값 메모리가 조각 크기에 비례해 줄어든다. 대신 순전파와
역전파를 여러 번 해야 하므로 시간이 는다. 두 축을 재어, 조각 수를
어떻게 고를지 가늠해 본다.

실행: 이 쪽의 코드를 파일로 갈무리해 돌린다
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
```

**출력:**

```
=== 1. 조각 수와 시간 ===
   1 조각  조각 크기 256    0.91 ms   통째 대비  1.00배   조각당  0.907 ms
   2 조각  조각 크기 128    1.13 ms   통째 대비  1.25배   조각당  0.566 ms
   4 조각  조각 크기  64    1.60 ms   통째 대비  1.76배   조각당  0.400 ms
   8 조각  조각 크기  32    2.01 ms   통째 대비  2.22배   조각당  0.252 ms
  16 조각  조각 크기  16    4.58 ms   통째 대비  5.05배   조각당  0.287 ms
  32 조각  조각 크기   8    8.53 ms   통째 대비  9.40배   조각당  0.267 ms
  조각 수에 꼭 비례하지는 않는다 — 조각당 시간이 줄어든다
  작은 배치는 행렬곱이 덜 효율적이지만 일 자체도 적기 때문이다

=== 2. 조각 수와 활성값 메모리 ===
  활성값은 '조각 크기 x 폭 x 층수 x 4바이트' 쯤이다
   1 조각  조각 크기 256  활성값 대략  2.097 MB
   4 조각  조각 크기  64  활성값 대략  0.524 MB
  16 조각  조각 크기  16  활성값 대략  0.131 MB
  32 조각  조각 크기   8  활성값 대략  0.066 MB
  파라미터와 경사는 조각 수와 무관하다 — 모델 크기가 정한다
  이 모델의 파라미터 2.12 MB (경사도 같은 크기)
  그래서 누적으로 아끼는 것은 활성값뿐이다

=== 3. 쪼갤수록 이득이 줄어든다 ===
  조각 크기를 반으로 줄일 때마다
    활성값 메모리: 반으로 — 언제나 반
    시간: 1.25 -> 1.41 -> 1.26 -> 2.28 -> 1.86 배
  메모리 이득은 일정하고 시간 손해는 쌓인다
  그래서 '모델이 올라가는 가장 큰 조각' 을 고르는 것이 맞다
  더 쪼개도 경사는 같으므로 학습에는 아무 이득이 없다

=== 4. 체크포인팅과 견주기 ===
  통째로            3.93 ~   5.59 ms
  8조각 누적         9.20 ~  17.42 ms
  체크포인팅          7.07 ~  12.22 ms
  둘 다 메모리를 시간과 바꾸는데 체크포인팅이 덜 비쌌다
  다만 아끼는 축이 다르다 — 누적은 배치 축, 체크포인팅은 깊이 축
  둘을 같이 쓰면 곱으로 아낀다
```

이 측정은 시간이라 돌릴 때마다 흔들린다. 아래 수들도 그 폭 안에서 읽어야 한다.

## 2. 시간은 조각 수에 비례하지 않는다

| 조각 수 | 조각 크기 | 시간 | 통째 대비 | 조각당 |
|---|---|---|---|---|
| 1 | 256 | 0.95 ms | 1.00 | 0.947 |
| 4 | 64 | 1.49 ms | 1.57 | 0.371 |
| 16 | 16 | 4.65 ms | 4.91 | 0.291 |
| 32 | 8 | 8.27 ms | 8.72 | 0.258 |

**32조각이 32배가 아니라 8.7배다.** 조각당 시간이 0.947에서 0.258로 줄기
때문이다.

까닭이 둘 겹쳐 있다. 조각이 작으면 **행렬곱이 덜 효율적**이지만(GPU든 CPU든
큰 행렬을 좋아한다), 동시에 **일 자체가 적다**(8개 표본의 순전파는 256개의
1/32이다). 뒤쪽이 더 크게 작용해서 전체로는 32배까지 안 간다.

그래도 **느려지기는 한다.** 공짜가 아니다.

## 3. 메모리는 조각 크기가 정한다

```
 1 조각  조각 크기 256  활성값 대략  2.097 MB
16 조각  조각 크기  16  활성값 대략  0.131 MB
이 모델의 파라미터 2.12 MB (경사도 같은 크기)
```

**아끼는 것은 활성값뿐이다.** 파라미터와 경사는 모델 크기가 정하므로 조각 수와
무관하다.

이 비율이 중요하다. 이 작은 모델에서는 활성값(2.1 MB)과 파라미터(2.1 MB)가
비슷해서 누적으로 아낄 수 있는 몫이 절반뿐이다. 반면 **활성값이 파라미터보다
훨씬 큰** 경우 — 긴 시퀀스, 큰 영상, 깊은 그물 — 에는 누적이 크게 이긴다.

그래서 누적을 쓰기 전에 **어느 쪽이 메모리를 먹고 있는지** 보아야 한다.
파라미터가 안 올라가는 것이면 누적으로는 못 고친다.

## 4. 쪼갤수록 이득이 줄어든다

```
조각 크기를 반으로 줄일 때마다
  활성값 메모리: 반으로 — 언제나 반
  시간: 1.25 -> 1.41 -> 1.26 -> 2.28 -> 1.86 배
```

메모리 이득은 **일정하게 절반**이고 시간 손해는 **곱해서 쌓인다.**

그래서 고르는 법이 분명해진다. **모델이 올라가는 가장 큰 조각**을 쓴다. 더
쪼개도 경사는 「경사 누적」에서 본 대로 똑같으므로 **학습에는 아무 이득이
없고**, 시간만 더 든다.

## 5. 체크포인팅과 견주기

```
통째로        4.35 ~  5.31 ms
8조각 누적    11.85 ~ 13.10 ms
체크포인팅     6.57 ~ 10.76 ms
```

둘 다 메모리를 시간과 바꾸는 기법인데, **아끼는 축이 다르다.**

| | 줄이는 것 | 치르는 것 |
|---|---|---|
| 경사 누적 | **배치 축** — 한 번에 처리하는 표본 수 | 순전파·역전파를 여러 번 |
| 체크포인팅 | **깊이 축** — 층마다 저장하는 활성값 | 역전파에서 순전파를 다시 |

여기서는 체크포인팅이 덜 비쌌지만, 모델과 배치 모양에 따라 뒤집힌다. 그리고
**둘을 같이 쓸 수 있다** — 배치를 쪼개고 각 조각에서 체크포인팅하면 아끼는
양이 곱해진다. 큰 언어 모델을 작은 GPU에서 학습하는 코드가 거의 둘 다 쓴다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
조각 수를 바꿔도 **옵티마이저가 걷는 횟수**는 같다. 몇 번 걷는지 세어
확인하라.

</div>

??? success "풀이"
    ```python
    net, opt, X, Y = setup()
    for k in (1, 4, 16):
        steps = {"backward": 0, "step": 0}
        opt.zero_grad()
        n = BATCH // k
        for i in range(k):
            (F.cross_entropy(net(X[i * n:(i + 1) * n]),
                             Y[i * n:(i + 1) * n]) / k).backward()
            steps["backward"] += 1
        opt.step()
        steps["step"] += 1
        print(f"  {k:2d} 조각  backward {steps['backward']:2d}번  "
              f"step {steps['step']}번")
    ```

    ```
       1 조각  backward  1번  step 1번
       4 조각  backward  4번  step 1번
      16 조각  backward 16번  step 1번
    ```

    **`backward`는 조각 수만큼, `step`은 한 번.** 이것이 경사 누적의 정의
    그대로다.

    그래서 "실효 배치 크기"라는 말이 성립한다. 옵티마이저 쪽에서 보면 256개를
    한 번에 본 것과 구별되지 않는다.

    학습률 스케줄러나 로그를 걸 때 **어느 쪽을 세는지** 조심해야 한다.
    `step()` 횟수를 세야 하는데 `backward()` 횟수를 세면 스케줄이 $k$배 빨리
    간다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
3절은 "파라미터가 안 올라가면 누적으로 못 고친다"고 했다. 모델 크기와 배치
크기를 바꿔 가며 **활성값과 파라미터의 비**를 재어, 누적이 언제 쓸모 있는지
가늠하라.

</div>

??? success "풀이"
    ```python
    for dim, depth, batch in ((512, 3, 256), (512, 3, 4096),
                              (4096, 3, 256), (512, 32, 256)):
        torch.manual_seed(0)
        layers = []
        for _ in range(depth):
            layers += [nn.Linear(dim, dim), nn.ReLU()]
        net = nn.Sequential(*layers)
        pbytes = sum(p.numel() * p.element_size() for p in net.parameters())
        act = batch * dim * depth * 2 * 4        # 층마다 두 군데쯤
        print(f"  폭 {dim:5d} 깊이 {depth:2d} 배치 {batch:5d}  "
              f"파라미터 {pbytes / 1e6:7.1f} MB  활성값 {act / 1e6:7.1f} MB  "
              f"비 {act / pbytes:6.2f}")
    ```

    ```
      폭   512 깊이  3 배치   256  파라미터     3.2 MB  활성값     3.1 MB  비   1.00
      폭   512 깊이  3 배치  4096  파라미터     3.2 MB  활성값    50.3 MB  비  15.97
      폭  4096 깊이  3 배치   256  파라미터   201.4 MB  활성값    25.2 MB  비   0.12
      폭   512 깊이 32 배치   256  파라미터    33.6 MB  활성값    33.6 MB  비   1.00
    ```

    비가 클수록 누적이 이긴다.

    - **배치가 크면** 비가 그대로 커진다(16.00). 누적이 가장 잘 듣는 자리다
    - **모델이 넓으면** 비가 작아진다(0.12). 파라미터가 $d^2$로 자라는데
      활성값은 $d$로만 자라기 때문이다. 누적해도 거의 못 아낀다
    - **깊으면** 둘이 같이 자라 비가 그대로다

    그래서 **"배치를 줄여서 될 일인가"**를 먼저 물어야 한다. 큰 언어 모델에서
    파라미터만 수십 GB인 경우, 누적은 아무 도움이 안 되고 모델 병렬화나
    ZeRO 같은 다른 길이 필요하다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
5절에서 누적과 체크포인팅을 **같이** 쓰면 곱으로 아낀다고 했다. 네 조합을
재어 시간이 어떻게 되는지 보라.

</div>

??? success "풀이"
    ```python
    from torch.utils.checkpoint import checkpoint

    torch.manual_seed(0)
    blocks = nn.ModuleList([nn.Sequential(nn.Linear(DIM, DIM), nn.ReLU())
                            for _ in range(8)])
    head = nn.Linear(DIM, 10)
    opt = torch.optim.SGD(list(blocks.parameters()) + list(head.parameters()),
                          lr=0.01)
    X, Y = torch.randn(BATCH, DIM), torch.randint(0, 10, (BATCH,))

    def forward(x, ck):
        for b in blocks:
            x = checkpoint(b, x, use_reentrant=False) if ck else b(x)
        return head(x)

    def run(k, ck):
        def f():
            opt.zero_grad()
            n = BATCH // k
            for i in range(k):
                (F.cross_entropy(forward(X[i * n:(i + 1) * n], ck),
                                 Y[i * n:(i + 1) * n]) / k).backward()
            opt.step()
        return f

    for k in (1, 8):
        for ck in (False, True):
            lo, hi = spread(run(k, ck))
            print(f"  {k} 조각, 체크포인팅 {str(ck):5s}  {lo:6.2f} ~ {hi:6.2f} ms")
    ```

    ```
      1 조각, 체크포인팅 False    6.16 ~   8.45 ms
      1 조각, 체크포인팅 True     8.77 ~  10.40 ms
      8 조각, 체크포인팅 False   11.36 ~  16.79 ms
      8 조각, 체크포인팅 True    19.65 ~  27.13 ms
    ```

    시간이 **대략 곱해진다.** 체크포인팅 혼자 1.42배, 누적 혼자 1.84배,
    둘을 곱하면 2.6배인데 잰 값이 6.16 → 19.65로 3.2배다. 어림이 맞는
    자릿수다 — 시간 측정이라 폭이 넓으니 그 이상을 말할 수는 없다.

    메모리도 곱으로 줄어든다. 배치를 8분의 1로, 깊이 쪽 활성값을 또 그만큼
    줄이므로, 둘을 합치면 혼자서는 못 올라가는 설정이 올라간다.

    치르는 값이 네 배라는 것은 가볍지 않다. 학습이 하루 걸리던 것이 나흘이
    된다. 그래서 순서가 있다.

    1. **배치를 줄여 본다** — 공짜다. 다만 배치 정규화가 있으면 결과가 달라진다
    2. **혼합 정밀도를 켠다** — 활성값이 절반이 되고 보통 더 빨라진다
    3. **누적을 쓴다** — 실효 배치를 되돌린다
    4. **체크포인팅을 켠다** — 그래도 안 올라갈 때

    「혼합 정밀도」가 2번이고 그것이 가장 수지가 맞는다. 메모리를 아끼면서
    느려지지 않는 유일한 수단이기 때문이다.
