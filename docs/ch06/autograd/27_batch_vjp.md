# 배치에서의 VJP — 경사는 표본별로 오지 않는다

배치로 로스를 내고 역전파하면 파라미터의 경사가 **하나** 나온다. 표본이 256개여도
하나다. 그것은 표본별 경사를 **더한** 것이고, 어느 표본이 얼마나 밀었는지는
사라진다.

평소에는 그것이 원하는 바다 — 걸음은 하나니까. 그런데 표본별 경사가 필요한
자리가 있다.

| | 무엇에 쓰는가 |
|---|---|
| 차등 프라이버시 | 표본마다 경사를 잘라야 한다 |
| 영향 함수 | 어느 표본이 예측을 바꾸는가 |
| 경사 잡음 재기 | 표본 사이의 퍼짐 |

## 1. 코드

```python
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
```

**출력:**

```
=== 1. 더해서 온다 ===
  배치 한 번: [-4.60474, 3.3489, -8.14294]
  표본별 합 : [-4.60474, 3.3489, -8.14294]
  같은가    : True
  표본마다:
    0: [-5.87015, 1.53148, -2.92195]
    1: [2.27928, -0.65733, -1.36572]
    2: [-0.43605, -0.24452, -0.36171]
    3: [-0.57782, 2.71928, -3.49357]
  0번 표본이 혼자 -5.87 을 밀고 있다. 합에서는 안 보인다

=== 2. vmap 으로 모으기 ===
  결과 모양: {'weight': (4, 1, 3), 'bias': (4, 1)}
  첫 축이 배치다 — 표본마다 한 벌씩
  루프로 모은 것과 같은가: True

=== 3. 값 ===
  배치 한 번 (합만)     0.05 ms
  vmap  (표본별)       0.84 ms
  루프  (표본별)       8.89 ms
  표본별이 필요하면 vmap 이 루프보다 훨씬 낫다
  다만 합만 필요하면 배치 한 번이 비교가 안 되게 싸다

=== 4. 표본별 경사로 잡음 재기 ===
  표본 하나짜리 경사의 흩어짐 2.2537  (표본 4096개로 잰 값)
  이것은 데이터의 성질이라 배치 크기와 무관하다
  배치 평균의 흩어짐은 거기서 sqrt(n) 으로 줄어든다
  배치    16  평균의 흩어짐  0.51366   예측  0.56343
  배치    64  평균의 흩어짐  0.25161   예측  0.28172
  배치   256  평균의 흩어짐  0.13367   예측  0.14086
  배치  1024  평균의 흩어짐  0.05855   예측  0.07043
  잰 값이 예측과 맞는다 — 「배치를 키우면」의 1/sqrt(n) 이 이것이다

=== 5. 표본마다 자르기 ===
  표본별 경사 노름  최소 0.0360  최대 22.0155
  C=1.0 로 자른 뒤   최소 0.0360  최대 1.0000
  자르기 전 합의 노름: 51.302
  자른 뒤 합의 노름  : 5.5121
  한 표본이 걸음을 좌우하지 못하게 막는 것이 요점이다
  배치 경사만 있으면 이 일을 할 수 없다 — 이미 더해졌으니까
```

## 2. 더해서 온다

```
배치 한 번: [-4.60474, 3.3489, -8.14294]
표본별 합 : [-4.60474, 3.3489, -8.14294]

표본마다:
  0: [-5.87015, 1.53148, -2.92195]
  1: [2.27928, -0.65733, -1.36572]
  2: [-0.43605, -0.24452, -0.36171]
  3: [-0.57782, 2.71928, -3.49357]
```

합이 정확히 맞는다. 그리고 **0번 표본이 혼자 $-5.87$을 밀고 있다** — 합계
$-4.60$보다 크다. 다른 표본들이 일부 되밀어 상쇄한 것이다.

합에서는 그 사정이 안 보인다. 한 표본이 걸음을 좌우하고 있어도 알 수 없다.

## 3. `vmap`으로 한 번에

```python
params = {k: v.detach() for k, v in m.named_parameters()}

def loss_one(p, x, y):
    pred = functional_call(m, p, (x.unsqueeze(0),))
    return F.mse_loss(pred, y.unsqueeze(0), reduction="sum")

per_sample = vmap(grad(loss_one), in_dims=(None, 0, 0))(params, X, Y)
```

```
결과 모양: {'weight': (4, 1, 3), 'bias': (4, 1)}
루프로 모은 것과 같은가: True
```

세 조각이 맞물린다.

- **`functional_call`** — 모듈의 파라미터를 바깥에서 넣어 준다. 모듈이 들고 있는
  것을 쓰지 않으므로 표본마다 다른 파라미터를 넣을 수도 있다
- **`grad`** — 함수를 받아 경사를 돌려주는 함수를 만든다. `.grad`를 안 건드린다
- **`vmap`** — 배치 축을 따라 한꺼번에 돌린다. `in_dims=(None, 0, 0)`은
  "파라미터는 공유하고 $x$와 $y$는 0번 축으로 나눠라"는 뜻이다

## 4. 값

```
배치 한 번 (합만)     0.06 ms
vmap  (표본별)       0.90 ms
루프  (표본별)       8.63 ms
```

표본별이 필요하면 **`vmap`이 루프보다 아홉 배 빠르다.** 루프는 256번
`backward`를 부르고, `vmap`은 한 번에 처리한다.

다만 **합만 필요하면 배치 한 번이 비교가 안 되게 싸다**(0.06 ms). 표본별 경사를
들고 있으려면 파라미터 크기 × 배치 크기만큼의 메모리도 든다. 꼭 필요한 자리에만
쓴다.

## 5. 경사의 잡음 재기

```
표본 하나짜리 경사의 흩어짐 2.2537  (표본 4096개로 잰 값)

배치    16  평균의 흩어짐  0.51366   예측  0.56343
배치    64  평균의 흩어짐  0.25161   예측  0.28172
배치   256  평균의 흩어짐  0.13367   예측  0.14086
배치  1024  평균의 흩어짐  0.05855   예측  0.07043
```

표본 하나짜리 경사의 흩어짐은 **데이터의 성질**이라 배치 크기와 무관하다.
배치 평균의 흩어짐은 거기서 $\sqrt{n}$으로 줄어들고, 잰 값이 예측과 맞는다.

「배치를 키우면 러닝레이트를」에서 $1/\sqrt{k}$ 규칙의 근거로 든 것이 바로 이
수다. 거기서는 배치를 여러 번 뽑아 간접적으로 쟀는데, 여기서는 표본별 경사를
직접 들고 있으므로 **한 배치에서 바로 추정**할 수 있다.

실전에서 쓸모가 있다. 학습 중에 이 비를 보면 **배치를 키울 때가 되었는지**
알 수 있다 — 잡음이 신호보다 크면 키우는 것이 이득이다.

## 6. 차등 프라이버시의 표본별 자르기

```
표본별 경사 노름  최소 0.0360  최대 22.0155
C=1.0 로 자른 뒤   최소 0.0360  최대 1.0000
자르기 전 합의 노름: 51.302
자른 뒤 합의 노름  : 5.5121
```

표본 하나의 경사 노름이 22.0인데 다른 것은 0.036이다. **600배 차이**다. 그
한 표본이 배치 경사를 거의 혼자 정하고 있다.

차등 프라이버시는 그것을 막는다. 표본마다 노름을 $C$ 이하로 자르면 **어느
표본도 걸음을 $C$ 이상 바꿀 수 없다.** 한 사람의 데이터가 모델에 남기는 자취에
상한이 생기고, 거기에 잡음을 더해 보장을 완성한다.

핵심은 **배치 경사만 있으면 이 일을 할 수 없다**는 것이다. 이미 더해졌으므로
누가 얼마를 밀었는지 되돌릴 길이 없다. 「비교, 범위 제한, 정렬」에서 본
`clip_grad_norm_`은 **배치 전체**의 노름을 자르는 것이라 다른 일이다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
`reduction="mean"`으로 바꾸면 2절의 관계가 어떻게 되는가? 배치 경사가 표본별
경사의 **무엇**이 되는가?

</div>

??? success "풀이"
    ```python
    torch.manual_seed(0)
    m = nn.Linear(3, 1)
    X, Y = torch.randn(4, 3), torch.randn(4, 1)
    per = []
    for i in range(4):
        m.zero_grad()
        F.mse_loss(m(X[i:i + 1]), Y[i:i + 1], reduction="mean").backward()
        per.append(m.weight.grad.clone())
    stacked = torch.stack(per)

    m.zero_grad()
    F.mse_loss(m(X), Y, reduction="mean").backward()
    print("  배치 (mean)      :", [round(t, 5) for t in m.weight.grad.flatten().tolist()])
    print("  표본별 평균      :", [round(t, 5) for t in stacked.mean(0).flatten().tolist()])
    print("  표본별 합        :", [round(t, 5) for t in stacked.sum(0).flatten().tolist()])
    ```

    ```
      배치 (mean)      : [-1.15118, 0.83723, -2.03574]
      표본별 평균      : [-1.15118, 0.83723, -2.03574]
      표본별 합        : [-4.60474, 3.3489, -8.14294]
    ```

    **`mean`이면 평균, `sum`이면 합**이다. 당연한데 적어 둘 값이 있다.

    표본별 경사를 모아 `.sum(0)`할지 `.mean(0)`할지가 로스의 `reduction`과
    맞아야 한다. 「`nn.Module`로 옮기기」에서 `sum`과 `mean`이 경사를 $N$배
    가르던 것과 같은 자리다.

    차등 프라이버시 구현에서는 **표본별로 자른 뒤 더하고 그다음 $N$으로
    나누는** 순서가 중요하다. 순서를 바꾸면 자르기가 뜻을 잃는다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
표본별 경사를 들고 있으면 메모리가 얼마나 드는가? 파라미터 수와 배치 크기로
셈하고 재어라.

</div>

??? success "풀이"
    ```python
    for n, din, dout in ((32, 128, 64), (256, 128, 64), (256, 512, 512)):
        net = nn.Linear(din, dout)
        params = {k: v.detach() for k, v in net.named_parameters()}

        def loss_one(p, x, y):
            pred = functional_call(net, p, (x.unsqueeze(0),))
            return F.mse_loss(pred, y.unsqueeze(0), reduction="sum")

        X, Y = torch.randn(n, din), torch.randn(n, dout)
        g = vmap(grad(loss_one), in_dims=(None, 0, 0))(params, X, Y)
        pbytes = sum(p.numel() * 4 for p in net.parameters())
        gbytes = sum(v.numel() * 4 for v in g.values())
        print(f"  배치 {n:4d} {din}->{dout}  파라미터 {pbytes / 1e6:6.2f} MB  "
              f"표본별 경사 {gbytes / 1e6:7.2f} MB  {gbytes / pbytes:5.0f}배")
    ```

    ```
      배치   32 128->64  파라미터   0.03 MB  표본별 경사    1.06 MB     32배
      배치  256 128->64  파라미터   0.03 MB  표본별 경사    8.45 MB    256배
      배치  256 512->512  파라미터   1.05 MB  표본별 경사  268.96 MB    256배
    ```

    **정확히 배치 크기만큼** 든다. 파라미터 한 벌을 표본 수만큼 들고 있으니
    당연하다.

    마지막 줄이 무섭다. 파라미터가 1 MB인 작은 층 하나에 배치 256이면
    **269 MB**다. 실제 모델에서 전 층의 표본별 경사를 들고 있는 것은 대개
    불가능하다.

    그래서 차등 프라이버시 라이브러리(Opacus 같은 것)는 **층마다 즉시 노름을
    재고 잘라서 더한 다음 버린다.** 전부를 동시에 들고 있지 않는다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
6절의 자르기가 **학습을 어떻게 바꾸는가?** 자르기 있는 쪽과 없는 쪽으로 열
걸음씩 돌려, 이상값 하나가 있을 때 무슨 일이 나는지 보라.

</div>

??? success "풀이"
    ```python
    torch.manual_seed(0)
    X, Y = torch.randn(32, 8), torch.randn(32, 1)
    Y[0] = 100.0                                   # 이상값 하나

    def run(clip):
        torch.manual_seed(1)
        net = nn.Linear(8, 1)
        opt = torch.optim.SGD(net.parameters(), lr=0.01)
        params = {k: v.detach() for k, v in net.named_parameters()}

        def loss_one(p, x, y):
            pred = functional_call(net, p, (x.unsqueeze(0),))
            return F.mse_loss(pred, y.unsqueeze(0), reduction="sum")

        for _ in range(10):
            params = {k: v.detach() for k, v in net.named_parameters()}
            g = vmap(grad(loss_one), in_dims=(None, 0, 0))(params, X, Y)
            flat = g["weight"].reshape(32, -1)
            if clip:
                scale = (1.0 / flat.norm(dim=1).clamp(min=1.0)).unsqueeze(1)
                flat = flat * scale
            opt.zero_grad()
            net.weight.grad = flat.sum(0).reshape(net.weight.shape) / 32
            net.bias.grad = torch.zeros_like(net.bias)
            opt.step()
        return F.mse_loss(net(X[1:]), Y[1:]).item()

    print(f"  자르기 없이  이상값 뺀 로스 {run(False):10.4f}")
    print(f"  자르기 있게  이상값 뺀 로스 {run(True):10.4f}")
    ```

    ```
      자르기 없이  이상값 뺀 로스     2.6300
      자르기 있게  이상값 뺀 로스     0.9634
    ```

    **자른 쪽이 나머지 31개 표본에 훨씬 잘 맞는다.** 2.63 대 0.96으로 세 배 가깝다.

    이상값 하나가 100이라 그 표본의 경사가 나머지를 압도한다. 자르지 않으면
    모델이 그 하나를 좇다가 나머지를 놓친다.

    이것이 차등 프라이버시의 덤이다. 자르기는 **프라이버시를 위한 장치인데
    이상값에 대한 robustness도 따라온다.** 거꾸로 말하면, 한 표본이 모델을
    크게 바꿀 수 있다는 것이 프라이버시 문제이자 안정성 문제다 — 같은 성질의
    두 얼굴이다.
