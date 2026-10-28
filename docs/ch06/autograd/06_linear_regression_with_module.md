# `nn.Module`로 옮기기 — 추상화를 걷어 내면 같은 수가 나오는가

「완전한 선형 회귀」에서 손으로 관리한 세 가지를 라이브러리에 넘긴다.

| 손으로 | 라이브러리 |
|---|---|
| `w`, `b`를 직접 만들고 들고 다님 | `nn.Linear(1, 1)` |
| `((pred - y) ** 2).mean()` | `nn.MSELoss()` |
| `with no_grad(): w -= lr * w.grad` | `torch.optim.SGD` |

편해지는 것은 분명하다. 그런데 **셈이 그대로인지**는 따로 물어야 한다. 추상화를
믿고 쓰려면 한 번은 걷어 내고 확인해 두는 편이 낫다.

## 1. 코드

```python
"""nn.Module 로 옮기기 — 추상화를 걷어 내면 같은 수가 나오는가.

「완전한 선형 회귀」에서 손으로 관리한 세 가지를 라이브러리에 넘긴다.

| 손으로 | 라이브러리 |
|---|---|
| `w`, `b` 를 직접 만들고 들고 다님 | `nn.Linear(1, 1)` |
| `((pred - y) ** 2).mean()` | `nn.MSELoss()` |
| `with no_grad(): w -= lr * w.grad` | `torch.optim.SGD` |

편해지는 것은 분명한데, **셈이 그대로인지**는 따로 물어야 한다.
이 쪽은 두 길을 나란히 돌려 소수점까지 견준다.

실행: 이 쪽의 코드를 파일로 갈무리해 돌린다
"""
import torch
import torch.nn as nn


def header(title):
    print(f"\n=== {title} ===")


TRUE_W, TRUE_B, N = 2.0, 1.0, 100


def make_data(seed=42):
    torch.manual_seed(seed)
    x = torch.randn(N, 1)
    return x, TRUE_W * x + TRUE_B + 0.2 * torch.randn(N, 1)


# ========================================================================
# 1. 손으로 짠 것
# ========================================================================

def by_hand(x, y, lr=0.1, epochs=200, w0=None, b0=None):
    w = w0.clone().requires_grad_(True)
    b = b0.clone().requires_grad_(True)
    for _ in range(epochs):
        loss = ((x * w + b - y) ** 2).mean()
        if w.grad is not None:
            w.grad.zero_()
            b.grad.zero_()
        loss.backward()
        with torch.no_grad():
            w -= lr * w.grad
            b -= lr * b.grad
    return w.item(), b.item(), loss.item()


# ========================================================================
# 2. 라이브러리로 짠 것
# ========================================================================

def by_module(x, y, lr=0.1, epochs=200, w0=None, b0=None):
    model = nn.Linear(1, 1)
    with torch.no_grad():                 # 같은 자리에서 시작하게 맞춘다
        model.weight.copy_(w0.view(1, 1))
        model.bias.copy_(b0)
    criterion = nn.MSELoss()
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    for _ in range(epochs):
        loss = criterion(model(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model.weight.item(), model.bias.item(), loss.item()


def compare():
    header("1. 두 길이 같은 수를 주는가")
    x, y = make_data()
    torch.manual_seed(0)
    w0, b0 = torch.randn(1), torch.zeros(1)
    hw, hb, hl = by_hand(x, y, w0=w0, b0=b0)
    mw, mb, ml = by_module(x, y, w0=w0, b0=b0)
    print(f"  손으로      w {hw:.8f}  b {hb:.8f}  로스 {hl:.8f}")
    print(f"  nn.Module   w {mw:.8f}  b {mb:.8f}  로스 {ml:.8f}")
    print(f"  차이        w {abs(hw - mw):.2e}  b {abs(hb - mb):.2e}")
    print("  같은 자리에서 시작하면 소수점 끝까지 같다")


# ========================================================================
# 3. 무엇이 무엇을 대신하는가
# ========================================================================

def what_replaces_what():
    header("2. 바꿔치기")
    x, y = make_data()
    model = nn.Linear(1, 1)

    print("  파라미터는 모델이 들고 있다")
    for name, p in model.named_parameters():
        print(f"    {name:8s} 모양 {tuple(p.shape)}  requires_grad {p.requires_grad}")

    pred = model(x)
    hand_loss = ((pred - y) ** 2).mean()
    mse_loss = nn.MSELoss()(pred, y)
    print(f"  로스  손으로 {hand_loss.item():.8f}  MSELoss {mse_loss.item():.8f}"
          f"  같은가 {torch.allclose(hand_loss, mse_loss)}")

    print("  모델의 순전파도 손으로 적은 것과 같다:",
          torch.allclose(model(x), x @ model.weight.t() + model.bias))


# ========================================================================
# 4. 모델이 커지면 손으로는 못 한다
# ========================================================================

def why_bother():
    header("3. 모델이 커지면")
    small = nn.Linear(1, 1)
    big = nn.Sequential(nn.Linear(784, 256), nn.ReLU(),
                        nn.Linear(256, 128), nn.ReLU(),
                        nn.Linear(128, 10))
    for name, m in (("nn.Linear(1,1)", small), ("3층 그물", big)):
        n_t = sum(1 for _ in m.parameters())
        n_p = sum(p.numel() for p in m.parameters())
        print(f"  {name:16s} 텐서 {n_t}개   수 {n_p:,}개")
    print("  텐서 여섯 개를 손으로 zero_() 하고 갱신하는 코드는 쓸 수 없다")
    print("  opt.zero_grad() 와 opt.step() 이 그것을 한 줄로 만든다")

    opt = torch.optim.SGD(big.parameters(), lr=0.1, momentum=0.9)
    x = torch.randn(4, 784)
    big(x).sum().backward()
    opt.step()
    state = sum(len(s) for s in opt.state.values())
    print(f"  게다가 모멘텀은 상태를 들고 있어야 한다. 버퍼 {state}개")


# ========================================================================
# 5. 추상화가 숨기는 것
# ========================================================================

def what_it_hides():
    header("4. 숨겨지는 것")
    model = nn.Linear(1, 1)
    print("  model.weight 의 모양:", tuple(model.weight.shape), "= (out, in)")
    print("  그래서 순전파가 x @ W.t() 다 —「Linear 웨이트 관례」의 그 전치")

    x, y = make_data()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    loss = nn.MSELoss()(model(x), y)
    loss.backward()
    before = model.weight.item()
    opt.step()
    after = model.weight.item()
    print(f"  opt.step() 한 걸음: {before:.6f} -> {after:.6f}")
    print(f"  손으로 w - lr*grad : "
          f"{before - 0.1 * model.weight.grad.item():.6f}")
    print("  step() 이 하는 일은 no_grad 안의 뺄셈 한 줄이다")

    print("  reduction 을 sum 으로 바꾸면 로스도 경사도 N 배가 된다")
    for red in ("mean", "sum"):
        m2 = nn.Linear(1, 1)
        with torch.no_grad():
            m2.weight.copy_(model.weight)
            m2.bias.copy_(model.bias)
        nn.MSELoss(reduction=red)(m2(x), y).backward()
        print(f"    reduction={red:5s} weight.grad {m2.weight.grad.item():12.4f}")


def main():
    compare()
    what_replaces_what()
    why_bother()
    what_it_hides()


if __name__ == "__main__":
    main()
```

**출력:**

```
=== 1. 두 길이 같은 수를 주는가 ===
  손으로      w 2.00236273  b 1.00713444  로스 0.03121442
  nn.Module   w 2.00236273  b 1.00713444  로스 0.03121442
  차이        w 0.00e+00  b 0.00e+00
  같은 자리에서 시작하면 소수점 끝까지 같다

=== 2. 바꿔치기 ===
  파라미터는 모델이 들고 있다
    weight   모양 (1, 1)  requires_grad True
    bias     모양 (1,)  requires_grad True
  로스  손으로 2.32904792  MSELoss 2.32904792  같은가 True
  모델의 순전파도 손으로 적은 것과 같다: True

=== 3. 모델이 커지면 ===
  nn.Linear(1,1)   텐서 2개   수 2개
  3층 그물            텐서 6개   수 235,146개
  텐서 여섯 개를 손으로 zero_() 하고 갱신하는 코드는 쓸 수 없다
  opt.zero_grad() 와 opt.step() 이 그것을 한 줄로 만든다
  게다가 모멘텀은 상태를 들고 있어야 한다. 버퍼 6개

=== 4. 숨겨지는 것 ===
  model.weight 의 모양: (1, 1) = (out, in)
  그래서 순전파가 x @ W.t() 다 —「Linear 웨이트 관례」의 그 전치
  opt.step() 한 걸음: -0.775056 -> -0.215242
  손으로 w - lr*grad : -0.215242
  step() 이 하는 일은 no_grad 안의 뺄셈 한 줄이다
  reduction 을 sum 으로 바꾸면 로스도 경사도 N 배가 된다
    reduction=mean  weight.grad      -4.4662
    reduction=sum   weight.grad    -446.6163
```

## 2. 같은 자리에서 출발하면 끝까지 같다

```
손으로      w 2.00236273  b 1.00713444  로스 0.03121442
nn.Module   w 2.00236273  b 1.00713444  로스 0.03121442
차이        w 0.00e+00  b 0.00e+00
```

**비트 단위로 같다.** 차이가 `0.00e+00`이지 "아주 작다"가 아니다.

같게 만들려고 한 일은 하나뿐이다 — `nn.Linear`의 초기값을 손으로 쓴 쪽과 맞춘
것. `model.weight.copy_(w0)`가 그것이다. 초기값만 맞추면 그 뒤로는 완전히 같은
계산이 돈다.

그러니 `nn.Linear`·`MSELoss`·`SGD`는 **다른 알고리즘이 아니라 같은 알고리즘의
다른 표기**다. 둘을 견주어 결과가 다르다면 초기값이 다르거나, 어딘가에서
`reduction`이나 러닝레이트가 어긋난 것이다.

## 3. 무엇이 무엇을 대신하는가

```
weight   모양 (1, 1)  requires_grad True
bias     모양 (1,)    requires_grad True

로스  손으로 2.32904792   MSELoss 2.32904792   같은가 True
모델의 순전파도 손으로 적은 것과 같다: True
```

`nn.Linear`가 하는 일은 **텐서 둘을 들고 있다가 `x @ W.t() + b`를 하는 것**이
전부다. 「Linear 웨이트 관례」에서 그 전치가 왜 붙는지 보았다.

`MSELoss()(pred, y)`도 `((pred - y) ** 2).mean()`과 소수점까지 같다.

## 4. 그런데 왜 쓰는가

```
nn.Linear(1,1)   텐서 2개   수 2개
3층 그물          텐서 6개   수 235,146개
게다가 모멘텀은 상태를 들고 있어야 한다. 버퍼 6개
```

모델이 작을 때는 손으로 써도 된다. 커지면 못 쓴다.

- **파라미터가 여섯 개**만 되어도 `if p.grad is not None: p.grad.zero_()`를
  여섯 번 적어야 한다. 하나 빠뜨리면 「최소 학습 단계」 3절처럼 조용히 안 배운다
- **모멘텀이나 Adam은 상태를 들고 있어야 한다.** 파라미터마다 버퍼를 하나씩
  만들어 걸음마다 갱신하는 일을 손으로 하기는 어렵다

그러니 옵티마이저의 값은 "다른 알고리즘"이 아니라 **빠뜨릴 데가 없어지는 것**에
있다.

## 5. 추상화가 숨기는 것

```
opt.step() 한 걸음: -0.775056 -> -0.215242
손으로 w - lr*grad : -0.215242
```

`step()`이 하는 일은 `no_grad` 안의 뺄셈 한 줄이다. 숨겨져 있을 뿐 마법이 아니다.

숨겨져서 걸리는 자리가 하나 있다.

```
reduction=mean  weight.grad      -4.4662
reduction=sum   weight.grad    -446.6163
```

**정확히 100배**, 곧 표본 수 $N$만큼이다. `MSELoss`의 기본값이 `"mean"`이라
평소에는 생각할 일이 없는데, 어딘가에서 `"sum"`으로 바꾸면 **경사가 $N$배가
되고 실효 러닝레이트도 $N$배**가 된다. 「경사 누적」에서 마이크로배치를 나눌
때와 같은 함정이다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
`nn.Linear`의 초기 가중치는 어디서 오는가? 입력 차원을 바꿔 가며 범위를 재고,
규칙을 찾아라.

</div>

??? success "풀이"
    ```python
    import math

    torch.manual_seed(0)
    for d in (1, 10, 100, 1000):
        m = nn.Linear(d, 1)
        bound = 1 / math.sqrt(d)
        w = m.weight.detach()
        print(f"  in={d:5d}  weight 범위 [{w.min():+.4f}, {w.max():+.4f}]  "
              f"1/sqrt(in) = {bound:.4f}  넘는가 {bool((w.abs() > bound).any())}")
    ```

    ```
      in=    1  weight 범위 [-0.0075, -0.0075]  1/sqrt(in) = 1.0000  넘는가 False
      in=   10  weight 범위 [-0.2603, +0.2507]  1/sqrt(in) = 0.3162  넘는가 False
      in=  100  weight 범위 [-0.0998, +0.0994]  1/sqrt(in) = 0.1000  넘는가 False
      in= 1000  weight 범위 [-0.0316, +0.0316]  1/sqrt(in) = 0.0316  넘는가 False
    ```

    **$U(-1/\sqrt{d_{in}},\, +1/\sqrt{d_{in}})$ 균등분포**다. 입력이 100개면
    ±0.1, 1000개면 ±0.0316 안에 들어간다.

    $1/\sqrt{d_{in}}$으로 줄이는 까닭은 「행렬 연산」의 어텐션 scaling과 같다.
    출력 하나는 $d_{in}$개를 더한 값이므로, 가중치의 크기를 그대로 두면 출력의
    표준편차가 $\sqrt{d_{in}}$으로 자란다. 층을 쌓으면 그것이 거듭 곱해져
    터지거나 사라진다.

    그래서 **초기화는 모델 구조가 정한다.** 손으로 `torch.randn`으로 파라미터를
    만들면 이 조정이 빠지고, 깊은 그물에서는 그것만으로 학습이 안 되기도 한다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
옵티마이저가 **무엇을 들고 있는지** 들여다보라. `step()` 전후로 `opt.state`와
`opt.param_groups`가 어떻게 달라지는가?

</div>

??? success "풀이"
    ```python
    m = nn.Linear(3, 2)
    opt = torch.optim.SGD(m.parameters(), lr=0.1, momentum=0.9)
    print("  step 전 state:", dict(opt.state))
    m(torch.randn(4, 3)).sum().backward()
    opt.step()
    for p, st in opt.state.items():
        print(f"  파라미터 모양 {tuple(p.shape)} -> 상태 키 {list(st.keys())}  "
              f"버퍼 모양 {tuple(st['momentum_buffer'].shape)}")
    print("  param_groups:",
          [{k: v for k, v in g.items() if k != 'params'} for g in opt.param_groups])
    ```

    ```
      step 전 state: {}
      파라미터 모양 (2, 3) -> 상태 키 ['momentum_buffer']  버퍼 모양 (2, 3)
      파라미터 모양 (2,) -> 상태 키 ['momentum_buffer']  버퍼 모양 (2,)
      param_groups: [{'lr': 0.1, 'momentum': 0.9, 'dampening': 0, 'weight_decay': 0,
                      'nesterov': False, 'maximize': False, ...}]
    ```

    **`state`가 처음에는 비어 있다.** 첫 `step()`에서야 만들어진다 — 미리 잡아
    두면 쓰지도 않을 메모리를 차지하니까.

    만들어진 뒤에는 **파라미터마다 그 모양 그대로의 버퍼**가 하나씩 붙는다.
    모멘텀을 쓰면 모델 크기만큼 메모리가 더 든다는 뜻이고, Adam은 버퍼가 둘이라
    두 배가 든다. 「모델 체크포인트」에서 옵티마이저 상태를 같이 저장해야 했던
    것이 이것이다.

    `param_groups`는 하이퍼파라미터를 담는다. 여기에 손을 대면 학습 도중에
    러닝레이트를 바꿀 수 있고, 스케줄러가 하는 일이 바로 그것이다 —
    `opt.param_groups[0]["lr"] = 새값`.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
5절에서 `reduction="sum"`이 경사를 $N$배로 만들었다. 그렇다면 `sum`으로 바꾸고도
**같은 학습**을 하려면 러닝레이트를 어떻게 해야 하는가? 확인하라.

</div>

??? success "풀이"
    ```python
    def run(red, lr, epochs=200):
        m = nn.Linear(1, 1)
        with torch.no_grad():
            m.weight.copy_(w0.view(1, 1))
            m.bias.copy_(b0)
        opt = torch.optim.SGD(m.parameters(), lr=lr)
        crit = nn.MSELoss(reduction=red)
        for _ in range(epochs):
            opt.zero_grad()
            crit(m(x), y).backward()
            opt.step()
        return m.weight.item(), m.bias.item()

    a = run("mean", 0.1)
    b = run("sum", 0.1)
    c = run("sum", 0.1 / N)
    print(f"  mean, lr=0.1      w {a[0]:.6f} b {a[1]:.6f}")
    print(f"  sum,  lr=0.1      w {b[0]:.6f} b {b[1]:.6f}")
    print(f"  sum,  lr=0.1/N    w {c[0]:.6f} b {c[1]:.6f}")
    ```

    ```
      mean, lr=0.1      w 2.002363 b 1.007134
      sum,  lr=0.1      w nan b nan
      sum,  lr=0.1/N    w 2.002363 b 1.007134
    ```

    **러닝레이트를 $N$으로 나누면 소수점까지 같아진다.**

    그냥 `sum`으로 바꾸면 `nan`이다. 경사가 100배이므로 걸음도 100배가 되고,
    「완전한 선형 회귀」 연습문제 3에서 잰 안정 경계 0.9566을 한참 넘어선다.

    여기서 가져갈 것은 **`reduction`과 러닝레이트가 한 쌍**이라는 점이다. 둘 중
    하나만 바꾸면 학습이 달라진다. 그리고 바뀌는 방향이 나쁜 쪽이면 `nan`으로
    바로 알지만, 반대 방향(예: `mean`인데 배치만 키우기)이면 **조용히 덜
    배운다.**

    파이토치가 `reduction="mean"`을 기본값으로 둔 까닭이 이것이다. 평균은 배치
    크기에 휘둘리지 않으므로, 러닝레이트 하나를 정해 두면 배치를 바꿔도 대체로
    쓸 수 있다.
