# 경사 누적 — 기본값이 더하기인 까닭

`.grad`는 덮어쓰지 않고 **더한다.** 그래서 학습 루프마다 지워야 하고, 지우는
줄을 빼먹으면 지난 걸음의 경사가 섞인다.

처음 보면 불편한 기본값이다. 왜 덮어쓰지 않을까. **기능이기 때문이다.** 배치를
쪼개 여러 번 역전파하고 한 번만 걸으면 큰 배치와 같은 경사가 나오고, 로스가
여럿일 때 따로 역전파해 더할 수도 있다.

이 쪽은 그 "같다"가 정말 같은지 재어 본다. 한 군데에 함정이 있다 — **`mean`
로스에서는 쪼갠 수로 나누어 주어야** 한다.

## 1. 코드

```python
"""경사 누적 — 기본값이 더하기인 까닭.

`.grad` 는 덮어쓰지 않고 **더한다**. 그래서 학습 루프마다 지워야 하고,
지우지 않으면 지난 걸음의 경사가 섞인다.

버그처럼 보이지만 기능이다. 배치를 쪼개 여러 번 역전파하고 한 번만
걸으면 큰 배치와 **같은 경사**가 나온다. 이 쪽은 그것이 정말 같은지
재어 본다. 다만 `mean` 로스에서는 쪼갠 수로 나누어 주어야 한다.

실행: python 02_gradient_accumulation.py
"""
import torch


def header(title):
    print(f"\n=== {title} ===")


# ========================================================================
# 1. 지우지 않으면 쌓인다
# ========================================================================

def accumulates():
    header("1. 지우지 않으면 쌓인다")
    x = torch.tensor(2.0, requires_grad=True)
    print("  처음 x.grad :", x.grad)
    (x ** 2).backward()
    print("  loss1 = x^2 뒤 :", x.grad.item(), " (2x = 4)")
    (3 * x).backward()
    print("  loss2 = 3x 뒤  :", x.grad.item(), " (4 + 3 = 7)")
    print("  두 로스의 경사가 더해졌다. 덮어쓰지 않는다")


# ========================================================================
# 2. 지우는 두 길이 걸음을 다르게 만든다
# ========================================================================

def two_ways_to_clear():
    header("2. zero_() 와 None")
    x = torch.tensor(2.0, requires_grad=True)
    (x ** 2).backward()
    x.grad.zero_()
    print("  zero_() 뒤 :", x.grad, " 자료형", type(x.grad).__name__)
    x.grad = None
    print("  None 으로   :", x.grad, " 자료형", type(x.grad).__name__)

    # 모멘텀이 있으면 이 둘이 다른 걸음을 만든다
    import torch.nn as nn
    got = []
    for set_to_none in (True, False):
        torch.manual_seed(0)
        m = nn.Linear(3, 1)
        opt = torch.optim.SGD(m.parameters(), lr=0.1, momentum=0.9)
        xb, yb = torch.randn(5, 3), torch.randn(5, 1)
        nn.functional.mse_loss(m(xb), yb).backward()
        opt.step()
        opt.zero_grad(set_to_none=set_to_none)
        opt.step()                      # 경사가 빈 채로 한 걸음 더
        got.append(m.weight.detach().flatten().clone())
    print("  모멘텀을 켜고 빈 걸음을 한 번 더 걸으면")
    print("    set_to_none=True :", [f"{v:+.6f}" for v in got[0].tolist()])
    print("    set_to_none=False:", [f"{v:+.6f}" for v in got[1].tolist()])
    print("    차이", f"{(got[0] - got[1]).abs().max():.6f}",
          "— None 은 건너뛰고 0 은 모멘텀이 민다")


# ========================================================================
# 3. 일부러 쌓기 — 큰 배치와 같은가
# ========================================================================

def micro_batching():
    header("3. 쪼개서 쌓으면 큰 배치와 같은가")
    torch.manual_seed(0)
    X, Y = torch.randn(40, 3), torch.randn(40, 1)
    W0 = torch.randn(3, 1)

    def whole(reduction):
        w = W0.clone().requires_grad_(True)
        sq = (X @ w - Y) ** 2
        (sq.mean() if reduction == "mean" else sq.sum()).backward()
        return w.grad.clone()

    def split(reduction, k=4, rescale=True):
        w = W0.clone().requires_grad_(True)
        n = len(X) // k
        for i in range(k):
            sq = (X[i * n:(i + 1) * n] @ w - Y[i * n:(i + 1) * n]) ** 2
            part = sq.mean() if reduction == "mean" else sq.sum()
            if rescale and reduction == "mean":
                part = part / k          # 평균의 평균이 되지 않게 맞춘다
            part.backward()
        return w.grad.clone()

    for red in ("sum", "mean"):
        a, b = whole(red), split(red)
        print(f"  {red:5s} 통째로 {[round(v, 6) for v in a.flatten().tolist()]}")
        print(f"        쪼개서 {[round(v, 6) for v in b.flatten().tolist()]}")
        print(f"        같은가 {torch.allclose(a, b, atol=1e-5)}")
    bad = split("mean", rescale=False)
    print("  mean 인데 k 로 안 나누면 같은가:",
          torch.allclose(whole("mean"), bad, atol=1e-5))
    print("    배수:", round((bad / whole("mean")).flatten()[0].item(), 4),
          "= 쪼갠 수 k")


# ========================================================================
# 4. 로스가 여럿일 때
# ========================================================================

def several_losses():
    header("4. 로스가 여럿일 때")
    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    ((x - 2) ** 2).mean().add(0.1 * x.abs().mean()).backward()
    combined = x.grad.clone()
    print("  합쳐서 한 번 :", [round(v, 6) for v in combined.tolist()])

    y = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    mse = ((y - 2) ** 2).mean()
    l1 = y.abs().mean()
    mse.backward(retain_graph=True)        # 그래프를 남겨 두어야 한다
    print("  MSE 만 하고   :", [round(v, 6) for v in y.grad.tolist()])
    (0.1 * l1).backward()
    print("  L1 까지 더하면:", [round(v, 6) for v in y.grad.tolist()])
    print("  합쳐서 한 번과 같은가:", torch.allclose(combined, y.grad))
    print("  쌓이는 성질 덕분에 로스를 따로 역전파해도 된다")
    print("  다만 같은 그래프를 두 번 걸으니 retain_graph=True 가 필요하다")


def main():
    accumulates()
    two_ways_to_clear()
    micro_batching()
    several_losses()


if __name__ == "__main__":
    main()
```

**출력:**

```
=== 1. 지우지 않으면 쌓인다 ===
  처음 x.grad : None
  loss1 = x^2 뒤 : 4.0  (2x = 4)
  loss2 = 3x 뒤  : 7.0  (4 + 3 = 7)
  두 로스의 경사가 더해졌다. 덮어쓰지 않는다

=== 2. zero_() 와 None ===
  zero_() 뒤 : tensor(0.)  자료형 Tensor
  None 으로   : None  자료형 NoneType
  모멘텀을 켜고 빈 걸음을 한 번 더 걸으면
    set_to_none=True : ['-0.021486', '+0.282214', '-0.422172']
    set_to_none=False: ['-0.036934', '+0.257463', '-0.374460']
    차이 0.047712 — None 은 건너뛰고 0 은 모멘텀이 민다

=== 3. 쪼개서 쌓으면 큰 배치와 같은가 ===
  sum   통째로 [23.311787, 19.769474, -57.027729]
        쪼개서 [23.311785, 19.769472, -57.027725]
        같은가 True
  mean  통째로 [0.582795, 0.494237, -1.425693]
        쪼개서 [0.582795, 0.494237, -1.425693]
        같은가 True
  mean 인데 k 로 안 나누면 같은가: False
    배수: 4.0 = 쪼갠 수 k

=== 4. 로스가 여럿일 때 ===
  합쳐서 한 번 : [-0.633333, 0.033333, 0.7]
  MSE 만 하고   : [-0.666667, 0.0, 0.666667]
  L1 까지 더하면: [-0.633333, 0.033333, 0.7]
  합쳐서 한 번과 같은가: True
  쌓이는 성질 덕분에 로스를 따로 역전파해도 된다
  다만 같은 그래프를 두 번 걸으니 retain_graph=True 가 필요하다
```

## 2. 더해진다

```
loss1 = x^2 뒤 : 4.0   (2x = 4)
loss2 = 3x 뒤  : 7.0   (4 + 3 = 7)
```

두 번째 역전파가 첫 번째 결과를 지우지 않았다. $4 + 3 = 7$이다.

학습 루프에서 이것을 안 지우면 어떻게 되는가. 걸음마다 지금까지의 경사가 전부
더해진 값으로 걷게 되므로, **걸음이 점점 커지고 방향도 옛 경사에 끌린다.**
발산하기도 하고, 운이 나쁘면 발산하지 않은 채 조금 나쁜 자리로 수렴해서
알아채지 못한다.

## 3. 지우는 두 길

```
zero_() 뒤 : tensor(0.)   자료형 Tensor
None 으로   : None         자료형 NoneType
```

메모리를 아끼는 차이로만 설명되는 일이 많은데, **걸음이 달라진다.**

```
set_to_none=True  : -0.021486  +0.282214  -0.422172
set_to_none=False : -0.036934  +0.257463  -0.374460
차이 0.047712
```

모멘텀을 켜 두고 경사가 빈 채로 한 걸음 더 걸어 보았다. `None`이면 `step()`이
그 파라미터를 **건너뛰고**, 0이면 "0만큼 움직여라"로 읽혀 **모멘텀이 계속
민다.**

파이토치 2.0부터 `zero_grad()`의 기본값이 `set_to_none=True`인 까닭이 이것이다.
경사를 지웠다는 말은 "이번에는 할 말이 없다"는 뜻이어야 한다.

## 4. 쪼개서 쌓으면 큰 배치와 같은가

```
sum   통째로 [23.311787, 19.769474, -57.027729]
      쪼개서 [23.311785, 19.769472, -57.027725]   같은가 True

mean  통째로 [0.582795, 0.494237, -1.425693]
      쪼개서 [0.582795, 0.494237, -1.425693]      같은가 True
```

40개를 한 번에 넣은 것과 10개씩 네 번 넣어 쌓은 것이 **같다.** (소수점 여섯째
자리에서 보이는 차이는 더하는 순서가 달라 생긴 부동소수점 오차다.)

그래서 GPU 메모리가 모자라 배치 256을 못 올릴 때, 64씩 네 번 돌려 쌓고 한 번
`step()`하면 배치 256과 같은 학습이 된다. 「메모리 관리」에서 본 그 기법이다.

### `mean`에서는 나누어야 한다

```
mean 인데 k 로 안 나누면 같은가: False
  배수: 4.0 = 쪼갠 수 k
```

`sum` 로스는 그냥 쌓으면 맞는다. 쪼갠 조각들의 합이 전체의 합이니까.

`mean`은 다르다. 조각마다 **그 조각의 평균**을 내므로, 넷을 더하면 전체 평균의
**4배**가 된다. 정확히 쪼갠 수만큼이다. 그래서 조각마다 $k$로 나누어 주어야
한다.

```python
loss = criterion(pred, y) / accumulation_steps    # 이 한 줄
loss.backward()
```

이 줄을 빼먹으면 **실효 러닝레이트가 $k$배**가 된다. 학습이 터지면 금방 알지만,
터지지 않을 만큼만 크면 "왜 이 설정이 더 안 좋지"로 끝난다.

## 5. 로스가 여럿일 때

```
합쳐서 한 번 : [-0.633333, 0.033333, 0.7]
MSE 만 하고   : [-0.666667, 0.0, 0.666667]
L1 까지 더하면: [-0.633333, 0.033333, 0.7]
합쳐서 한 번과 같은가: True
```

`(mse + 0.1 * l1).backward()`와 `mse.backward()` 다음 `(0.1 * l1).backward()`가
같은 결과를 준다. 쌓이는 성질 덕분이다.

둘째 길에는 `retain_graph=True`가 필요하다. 같은 순전파에서 나온 두 로스가
**같은 그래프**를 쓰므로, 첫 역전파가 버퍼를 놓아 주면 둘째가 걸을 길이 없다.

실전에서는 대개 합쳐서 한 번 부르는 쪽이 낫다 — 그래프를 한 번만 걸으므로 빠르고
`retain_graph`도 필요 없다. 따로 부르는 것이 쓸모 있는 자리는 **로스마다 다른
파라미터 묶음에 경사를 보내야 할 때**(GAN의 생성자·판별자 같은 경우)다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
`zero_grad()`를 빼먹은 학습 루프를 다섯 걸음 돌려 `w.grad`가 어떻게 변하는지
보라. 제대로 지운 쪽과 견주어라.

</div>

??? success "풀이"
    ```python
    for label, clear in (("지운다  ", True), ("안 지운다", False)):
        w = torch.tensor(5.0, requires_grad=True)
        for step in range(5):
            if clear and w.grad is not None:
                w.grad.zero_()
            ((w - 1) ** 2).backward()
            with torch.no_grad():
                w -= 0.1 * w.grad
        print(f"  {label}  w = {w.item():8.4f}   마지막 grad = {w.grad.item():8.4f}")
    ```

    ```
      지운다    w =   2.3107   마지막 grad =   3.2768
      안 지운다  w =  -2.2397   마지막 grad =  14.2208
    ```

    참값은 $w = 1$이다. 지운 쪽은 5에서 2.31까지 내려왔고, 안 지운 쪽은 **1을
    지나쳐 −2.24까지 넘어갔다.**

    걸음마다 지난 경사가 더해지니 걸음이 점점 커지고, 최솟값에 닿은 뒤에도 그
    관성으로 지나쳐 버린다. 여기서는 발산까지 가지 않았지만, 러닝레이트가 조금만
    컸어도 터졌을 것이다.

    **터지면 차라리 다행이다.** 이 보기처럼 "조금 나쁜 답"으로 끝나면 버그가
    숨는다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
배치 32를 한 번에 쓰는 것과 8씩 네 번 쌓는 것이 같아지려면 로스를 어떻게
적어야 하는가? `reduction="mean"`과 `"sum"` 두 경우를 모두 확인하라.

</div>

??? success "풀이"
    ```python
    import torch.nn.functional as F

    torch.manual_seed(0)
    X, Y = torch.randn(32, 4), torch.randn(32, 1)
    W0 = torch.randn(4, 1)

    def run(reduction, k, divide):
        w = W0.clone().requires_grad_(True)
        if k == 1:
            F.mse_loss(X @ w, Y, reduction=reduction).backward()
        else:
            n = 32 // k
            for i in range(k):
                l = F.mse_loss(X[i*n:(i+1)*n] @ w, Y[i*n:(i+1)*n],
                               reduction=reduction)
                (l / k if divide else l).backward()
        return w.grad.clone()

    for red in ("mean", "sum"):
        base = run(red, 1, False)
        for divide in (False, True):
            g = run(red, 4, divide)
            r = (g / base).flatten()[0].item()
            print(f"  {red:5s} k=4 나누기={divide!s:5s}  "
                  f"전체 대비 {r:.4f}배  같은가 {torch.allclose(base, g, atol=1e-5)}")
    ```

    ```
      mean  k=4 나누기=False  전체 대비 4.0000배  같은가 False
      mean  k=4 나누기=True   전체 대비 1.0000배  같은가 True
      sum   k=4 나누기=False  전체 대비 1.0000배  같은가 True
      sum   k=4 나누기=True   전체 대비 0.2500배  같은가 False
    ```

    **정확히 반대다.**

    | | 나누어야 하는가 |
    |---|---|
    | `reduction="mean"` | **그렇다** — 안 나누면 $k$배 |
    | `reduction="sum"` | **아니다** — 나누면 $1/k$배 |

    까닭은 한 줄로 적힌다. `sum`은 쪼개도 합이 보존되지만 `mean`은 보존되지
    않는다. 조각 평균 넷을 더하면 전체 평균의 네 배다.

    파이토치의 로스 함수가 거의 다 `reduction="mean"`을 기본값으로 두고 있으므로,
    **실전에서는 거의 언제나 나누어야 한다**고 기억해 두면 맞는다.

    조각 크기가 고르지 않으면(마지막 조각이 짧을 때) `/k`도 정확하지 않다.
    엄밀히 하려면 조각마다 `len(batch) / total`을 곱해야 한다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
경사 누적이 큰 배치와 "같다"고 했다. **같지 않은 것**은 무엇인가? 배치 정규화가
들어 있는 모델에서 확인하라.

</div>

??? success "풀이"
    ```python
    import torch.nn as nn

    torch.manual_seed(0)
    X = torch.randn(32, 4)

    def grads(k):
        torch.manual_seed(0)
        net = nn.Sequential(nn.Linear(4, 8), nn.BatchNorm1d(8), nn.Linear(8, 1))
        net.zero_grad()
        n = 32 // k
        for i in range(k):
            (net(X[i * n:(i + 1) * n]).pow(2).sum() / k).backward()
        g = torch.cat([p.grad.flatten() for p in net.parameters()])
        rm = net[1].running_mean.clone()
        return g, rm

    g1, rm1 = grads(1)
    g4, rm4 = grads(4)
    print("  경사가 같은가        :", torch.allclose(g1, g4, atol=1e-5))
    print("  가장 큰 차이         :", f"{(g1 - g4).abs().max():.6f}")
    print("  running_mean 이 같은가:", torch.allclose(rm1, rm4, atol=1e-6))
    print("  running_mean[:3] 통째:", [round(v, 5) for v in rm1[:3].tolist()])
    print("  running_mean[:3] 쪼갬:", [round(v, 5) for v in rm4[:3].tolist()])
    ```

    ```
      경사가 같은가        : False
      가장 큰 차이         : 16.036142
      running_mean 이 같은가: False
      running_mean[:3] 통째: [-0.04642, -0.01221, -0.03849]
      running_mean[:3] 쪼갬: [-0.16165, -0.04048, -0.13295]
    ```

    **경사도 다르고 `running_mean`도 다르다.**

    배치 정규화는 **그 배치 안에서** 평균과 표준편차를 낸다. 32개로 낸 통계와
    8개로 낸 통계는 다르므로, 정규화된 값이 달라지고 따라서 경사도 달라진다.
    4절에서 같았던 것은 모델에 배치 간 상호작용이 없었기 때문이다.

    차이가 작지도 않다. 가장 큰 경사 성분이 **16이나 벌어진다.**

    `running_mean`은 더 어긋난다. 쪼개면 갱신이 네 번 일어나므로 이동평균이 네
    걸음 가고, 값이 서너 배로 벌어졌다. 통계가 다른 속도로 쌓인다는 뜻이다.

    그래서 경사 누적이 큰 배치와 같다는 말에는 단서가 붙는다 —
    **표본끼리 섞이지 않는 모델에서만** 그렇다. 배치 정규화가 있으면 다르고,
    대조 학습처럼 배치 안에서 짝을 짓는 로스도 다르다.

    실전의 해법은 배치 정규화를 **그룹 정규화나 층 정규화로 바꾸는 것**이다.
    둘 다 표본마다 따로 통계를 내므로 배치 크기에 휘둘리지 않고, 그래서 경사
    누적과도 잘 맞는다. 큰 모델을 작은 GPU에서 학습하는 코드가 거의 배치
    정규화를 쓰지 않는 까닭이 이것이다.
