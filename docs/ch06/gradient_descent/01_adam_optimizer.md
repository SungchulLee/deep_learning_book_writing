# Adam 옵티마이저 — 손으로 짜서 맞춰 보기

Adam을 설명하는 글은 많고 대개 식을 적고 끝난다. 그런데 식을 적을 수 있다고 아는 것이 아니다.

이 쪽은 Adam을 **손으로 짜서 `torch.optim.Adam`과 한 비트까지 맞춰** 본다. 맞으면 이해한 것이고, 안 맞으면 어디가 다른지가 드러난다. 그 다음 식만 보아서는 알 수 없는 네 가지를 잰다.

- 바이어스 보정이 첫 걸음을 키우는가 줄이는가
- Adam의 서명은 무엇인가 — "빠르다"가 아니다
- 러닝레이트를 각자 맞춘 비교에서도 Adam이 이기는가
- `Adam(weight_decay=...)`과 `AdamW`가 얼마나 다른가

식의 유도와 수렴 이론은 [Adam 옵티마이저](../../ch09/optimizers/adam.md)에서 다룬다.

---

## 1. 손으로 짠 Adam

Adam은 두 가지 지수이동평균을 들고 다닌다. 경사의 평균 $m_t$와 경사 제곱의 평균 $v_t$다.

$$m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t, \qquad v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2$$

둘 다 0에서 출발하므로 초기에 0 쪽으로 치우쳐 있다. 그것을 보정하고

$$\hat{m}_t = \frac{m_t}{1-\beta_1^t}, \qquad \hat{v}_t = \frac{v_t}{1-\beta_2^t}$$

이렇게 걷는다.

$$\theta_{t+1} = \theta_t - \eta\,\frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

이것을 그대로 코드로 옮기고 PyTorch와 나란히 돌린다.

```
  걸음   손으로 짠 w[0]      PyTorch w[0]       차이
     1   +0.229432702065   +0.229432702065   0.00e+00
     2   +0.228432789445   +0.228432789445   0.00e+00
     5   +0.225434079766   +0.225434079766   0.00e+00
    10   +0.220442920923   +0.220442920923   0.00e+00
  10 걸음 뒤 모든 파라메터의 최대 차이 2.98e-08
```

소수점 아래 열두 자리까지 같다. 위 다섯 줄이 Adam의 전부다.

---

## 2. 바이어스 보정 — 흔히 거꾸로 설명되는 것

$m$과 $v$가 0에서 시작하므로 초기에는 참값보다 작다. 그래서 "보정이 작아진 값을 키워 준다"고 설명하는 글이 많다.

**분자와 분모가 함께 작아진다는 점이 빠져 있다.** 몫에서는 어떻게 되는지 재어 본다.

```
  걸음   보정 있음 걸음폭   보정 없음 걸음폭   비
     1   0.00100000       0.00316228       3.1623
     2   0.00100000       0.00424959       4.2496
     3   0.00100000       0.00495023       4.9502
     5   0.00100000       0.00579714       5.7971
    10   0.00100000       0.00652788       6.5279
```

**보정을 빼면 첫 걸음이 3.16배 커진다.** 줄어드는 것이 아니다.

손으로도 나온다. 첫 걸음에서 $m_1 = (1-\beta_1)g$, $v_1 = (1-\beta_2)g^2$이므로 보정 없는 걸음폭은

$$\eta\,\frac{(1-\beta_1)g}{\sqrt{(1-\beta_2)g^2}} = \eta\,\frac{0.1}{\sqrt{0.001}} = 3.1623\,\eta$$

다. 보정하면 $\hat{m}_1 = g$, $\hat{v}_1 = g^2$이라 걸음폭이 **정확히 $\eta$** 다.

곧 바이어스 보정은 **초기의 과도한 걸음을 막는** 장치다. 보정 없이 두면 처음 수십 걸음 동안 걸음폭이 제멋대로 커지고, 그 구간이 하필 파라메터가 무작위라 경사를 믿을 수 없는 구간이다.

!!! note "워밍업과의 관계"
    바이어스 보정을 빼면 초기에 걸음이 커진다 — 워밍업이 고치려는 바로 그 증상이다. 실제로 "Adam에 워밍업이 필요한 까닭"을 바이어스 보정이 불완전하기 때문으로 설명하는 연구가 있다(RAdam이 그 선을 따른다).

---

## 3. Adam의 서명 — 빠름이 아니라 고름

Adam이 좋은 까닭을 "빠르다"로 요약하는 일이 흔하다. 곡률이 좌표마다 1,000배 차이 나는 문제에서 재어 본다.

$$f(x) = \tfrac{1}{2}\sum_i \lambda_i x_i^2, \qquad \lambda = [1,\ 10,\ 100,\ 1000]$$

```
  2000 걸음 뒤 좌표별 |x| (0 에 가까울수록 좋다)
  옵티마이저        lr        x1        x2        x3        x4
  SGD               0.001    1.35e-01  1.86e-09  5.61e-45  0.00e+00
  SGD(mom=0.9)      0.001    2.16e-10  5.61e-45  0.00e+00  2.80e-45
  Adam              0.001    2.07e-02  2.07e-02  2.07e-02  2.07e-02
  Adam              0.01     6.59e-44  9.11e-44  8.97e-44  9.11e-44
```

**Adam이 이기지 않는다.** $\eta = 0.001$에서 Adam의 네 좌표가 모두 $2.07\times10^{-2}$인데, 모멘텀을 켠 SGD는 $x_1$을 $2.16\times10^{-10}$까지 내렸다. 이 문제에서는 모멘텀만으로 충분하다.

**읽어야 할 것은 Adam 줄의 네 값이 전부 같다는 사실이다.** 곡률이 1,000배 차이 나는 좌표들이 같은 속도로 줄었다. $\sqrt{v}$로 나누는 것이 좌표마다 걸음을 맞춰 주기 때문이다.

SGD 줄과 견주면 차이가 분명하다. SGD는 $x_4$($\lambda=1000$)를 완전히 0으로 만들어 놓고 $x_1$($\lambda=1$)은 0.135에 남겨 둔다. 러닝레이트를 키워 $x_1$을 구하려 하면 $x_4$가 터진다. **하나의 러닝레이트로 두 요구를 동시에 만족시킬 수 없다.**

Adam의 실질적인 이점이 여기 있다. **어느 좌표가 느린지 몰라도 된다.** 모델이 크고 레이어마다 경사 크기가 제각각일 때, 그것들을 일일이 조사해 러닝레이트를 맞출 필요가 없어진다.

---

## 4. 러닝레이트를 각자 맞춰도 이기는가

옵티마이저 비교에서 가장 흔한 어긋남은 **러닝레이트를 맞추지 않는 것** 이다. SGD에 0.01, Adam에 0.001을 주고 비교하면 옵티마이저가 아니라 그 두 숫자를 견준 것이 된다.

같은 초기 모델에서 출발시키고, 각자의 러닝레이트를 쓸어 가장 좋은 값을 고른 뒤, 그 값으로 씨앗 다섯 개를 돌린다.

```
  옵티마이저         0.0003     0.001     0.003      0.01      0.03       0.1       0.3
  SGD               0.4890    0.4484    0.4208    0.3032    0.0815    0.0465       inf
  SGD(mom=0.9)      0.4224    0.3159    0.0876    0.0505    0.0268    0.0213       inf
  Adam              0.2393    0.0528    0.0196    0.0088    0.0087    0.0082    0.0244
  고른 lr: SGD 0.1,  SGD(mom=0.9) 0.1,  Adam 0.1

  옵티마이저     lr       로스 평균     퍼짐
  SGD           0.1      0.050810   0.046529~0.058011
  SGD(mom=0.9)  0.1      0.015427   0.011327~0.021344
  Adam          0.1      0.008795   0.008155~0.009836
```

**맞춰 주어도 Adam이 이긴다.** 0.008795 대 0.050810으로 5.8배다. 퍼짐도 겹치지 않는다.

다만 "얼마나"는 크게 달라진다. 쓸어 보기 전 $\eta = 0.001$에서는 0.4484 대 0.0528로 **8.5배** 였다. 비교를 어떻게 하느냐에 따라 같은 사실이 5.8배로도 8.5배로도 보고된다.

**그리고 Adam 행의 평평함을 보라.** 0.003부터 0.3까지 0.0082~0.0196 사이에 있다. 백 배 범위에서 아무 값이나 집어도 쓸 만하다. SGD 행은 0.0003에서 0.1까지 0.489에서 0.047로 열 배 움직이고 0.3에서는 발산한다.

이것이 실무에서 Adam을 먼저 집는 진짜 까닭이다. 최종 성능이 더 좋아서라기보다 **고를 범위가 넓어서** 다.

---

## 5. `Adam(weight_decay)`과 `AdamW`

둘 다 웨이트 감쇠를 하는데 방식이 다르다. `Adam`은 경사에 $\lambda\theta$를 더하고, `AdamW`는 갱신이 끝난 뒤 $\theta$를 따로 줄인다.

```
  wd = 0.0:  Adam [-0.0048174429684877396, -0.004818218760192394]
             AdamW [-0.0048174429684877396, -0.004818218760192394]
             차이 0.000000

  wd = 0.1:  Adam [-0.004818068817257881, -0.0048181721940636635]
             AdamW [-0.01744718849658966, -0.017447788268327713]
             차이 0.012630
```

$\lambda = 0$이면 둘이 비트까지 똑같다. 그런데 $\lambda = 0.1$을 주면 **갈리는 모양이 뜻밖이다.**

`Adam` 쪽은 $-0.0048174$에서 $-0.0048181$로 **거의 변하지 않았다.** 웨이트 감쇠를 켰는데 아무 일도 일어나지 않은 것과 다름없다. `AdamW` 쪽은 $-0.0174$로 3.6배 커졌다.

까닭은 3절의 $\sqrt{v}$ 나누기에 있다. `Adam`은 $\lambda\theta$를 경사에 더하는데, **그 합이 다시 $\sqrt{v}$로 나뉜다.** 로스 경사가 큰 자리에서는 $v$가 커서 감쇠 항이 묻혀 버린다.

곧 **`Adam`의 `weight_decay`는 경사 크기에 따라 세기가 달라진다.** 경사가 작은 파라메터는 세게 줄고 큰 파라메터는 거의 안 준다. 의도한 정규화가 아니다.

`AdamW`는 갱신 뒤에 따로 빼므로 언제나 적은 만큼 줄인다. **이것이 `AdamW`가 따로 있는 까닭이다.** 같은 `weight_decay=0.1`을 주어도 두 옵티마이저가 전혀 다른 양만큼 줄인다는 점을 기억해야 한다 — 하나에서 다른 하나로 바꿀 때 값을 다시 맞춰야 한다.

---

## 6. 학습을 이어 하려면

```
  옵티마이저 상태까지 저장  40 걸음 통째와의 차이 0.000000e+00
  모델만 저장        40 걸음 통째와의 차이 4.266685e-02
```

20걸음에서 저장하고 20걸음을 더 돌린 결과를, 40걸음 통째로 돌린 결과와 견주었다. 옵티마이저 상태까지 저장하면 **정확히 같다.** 모델만 저장하면 0.043만큼 어긋난다.

$m$, $v$, $t$가 0으로 돌아가 **다시 1걸음째가 되기 때문이다.** 2절에서 본 대로 1걸음째의 걸음폭은 특별하므로, 이어 하는 자리에서 갑자기 다른 걸음을 떼게 된다.

!!! warning "SGD에 익숙하면 잊기 쉽다"
    모멘텀 없는 SGD는 상태가 없어 모델만 저장해도 그만이다. 그래서 `optimizer.state_dict()`를 저장하는 습관이 들지 않고, Adam으로 바꾼 뒤에도 그대로 둔다. 체크포인트에서 이어 학습할 때 로스가 살짝 튀는 흔한 원인이 이것이다.

---

## 7. 전체 코드와 출력

```python
"""Adam — 손으로 짜서 torch.optim.Adam 과 맞춰 본다.

Adam 을 설명하는 글은 많지만 대개 식을 적고 끝난다. 식을 적을 수 있다고
아는 것이 아니므로, 이 쪽은 손으로 짜서 PyTorch 와 **한 비트까지**
맞춰 본다. 맞으면 이해한 것이고, 안 맞으면 어디가 다른지가 드러난다.

이어서 네 가지를 잰다.

- 바이어스 보정이 하는 일 (첫 걸음을 키우는가 줄이는가)
- Adam 의 서명은 무엇인가 (곡률이 다른 좌표를 같은 속도로 줄인다)
- 같은 러닝레이트에서 견주지 않은 비교는 무엇을 재는가
- Adam 과 AdamW 의 웨이트 감쇠가 어떻게 다른가

개념과 유도는 ch09 의 Adam 쪽에서 다룬다.

실행: 이 쪽의 코드를 파일로 갈무리해 돌린다
"""
import math

import numpy as np
import torch
import torch.nn as nn


def header(title):
    print(f"\n=== {title} ===")


# ========================================================================
# 1. 손으로 짠 Adam
# ========================================================================

def hand_adam_step(p, g, state, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
    """Adam 한 걸음. PyTorch 의 구현과 같은 꼴로 적는다."""
    state["t"] += 1
    t = state["t"]
    state["m"] = b1 * state["m"] + (1 - b1) * g
    state["v"] = b2 * state["v"] + (1 - b2) * g * g
    m_hat = state["m"] / (1 - b1 ** t)          # 바이어스 보정
    v_hat = state["v"] / (1 - b2 ** t)
    return p - lr * m_hat / (v_hat.sqrt() + eps)


def match_pytorch():
    header("1. 손으로 짠 Adam vs torch.optim.Adam")
    torch.manual_seed(0)
    X = torch.randn(64, 5)
    Y = torch.randn(64, 1)

    torch.manual_seed(1)
    m1 = nn.Linear(5, 1)
    torch.manual_seed(1)
    m2 = nn.Linear(5, 1)

    opt = torch.optim.Adam(m2.parameters(), lr=1e-3)
    states = [{"m": torch.zeros_like(p), "v": torch.zeros_like(p), "t": 0}
              for p in m1.parameters()]

    print("  걸음   손으로 짠 w[0]      PyTorch w[0]       차이")
    for step in range(1, 11):
        # 손으로
        for p in m1.parameters():
            if p.grad is not None:
                p.grad = None
        nn.functional.mse_loss(m1(X), Y).backward()
        with torch.no_grad():
            for p, st in zip(m1.parameters(), states):
                p.copy_(hand_adam_step(p, p.grad, st))
        # PyTorch
        opt.zero_grad()
        nn.functional.mse_loss(m2(X), Y).backward()
        opt.step()
        if step in (1, 2, 5, 10):
            a = m1.weight.detach().flatten()[0].item()
            b = m2.weight.detach().flatten()[0].item()
            print(f"  {step:>4}   {a:+.12f}   {b:+.12f}   {abs(a-b):.2e}")
    gap = max((p1 - p2).abs().max().item()
              for p1, p2 in zip(m1.parameters(), m2.parameters()))
    print(f"  10 걸음 뒤 모든 파라메터의 최대 차이 {gap:.2e}")
    print("  float32 의 반올림 자리다. 같은 알고리즘이라는 뜻이다")


# ========================================================================
# 2. 바이어스 보정이 하는 일
# ========================================================================

def bias_correction():
    header("2. 바이어스 보정 — 첫 걸음을 키우는가 줄이는가")
    b1, b2, eps, lr = 0.9, 0.999, 1e-8, 1e-3
    g = torch.tensor(2.0)                      # 아무 경사나

    print("  m 과 v 를 0 에서 시작하므로 처음 몇 걸음은 0 쪽으로 치우친다")
    print("  걸음   보정 있음 걸음폭   보정 없음 걸음폭   비")
    m = v = torch.zeros(())
    m2 = v2 = torch.zeros(())
    for t in range(1, 11):
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        with_c = lr * (m / (1 - b1 ** t)) / ((v / (1 - b2 ** t)).sqrt() + eps)
        m2 = b1 * m2 + (1 - b1) * g
        v2 = b2 * v2 + (1 - b2) * g * g
        without = lr * m2 / (v2.sqrt() + eps)
        if t in (1, 2, 3, 5, 10):
            print(f"  {t:>4}   {with_c.item():.8f}       "
                  f"{without.item():.8f}       {without.item()/with_c.item():.4f}")
    print("  보정을 **빼면** 첫 걸음이 3.16 배 커진다. 줄어드는 것이 아니다")
    print(f"  까닭: 보정 없는 첫 걸음은 (1-b1)g / sqrt((1-b2)g^2)")
    print(f"       = {1-b1} / sqrt({1-b2}) = {(1-b1)/math.sqrt(1-b2):.4f} 배")
    print("  보정이 있으면 첫 걸음이 정확히 lr 이다 (m_hat/sqrt(v_hat) = 1)")
    print("  곧 바이어스 보정은 **초기의 과도한 걸음을 막는** 장치다")
    print("  흔히 '0 쪽으로 치우친 것을 키워 준다' 고 적지만, 분자와 분모가")
    print("  함께 치우쳐 있어 몫에서는 반대로 나온다")


# ========================================================================
# 3. Adam 이 이기는 자리 — 조건수가 나쁜 문제
# ========================================================================

def where_adam_wins():
    header("3. Adam 의 서명 — 좌표마다 같은 속도")
    print("  좌표마다 곡률이 크게 다른 이차함수를 만든다")
    print("  f(x) = sum_i lam_i x_i^2 / 2,  lam = [1, 10, 100, 1000]")
    lam = torch.tensor([1.0, 10.0, 100.0, 1000.0])
    print(f"  조건수 {lam.max()/lam.min():.0f}")

    def run(make, lr, steps=2000):
        x = torch.ones(4, requires_grad=True)
        o = make([x], lr)
        for _ in range(steps):
            o.zero_grad()
            (0.5 * (lam * x ** 2).sum()).backward()
            o.step()
        return x.detach().abs()

    print("\n  2000 걸음 뒤 좌표별 |x| (0 에 가까울수록 좋다)")
    print("  옵티마이저        lr        x1        x2        x3        x4")
    for name, make, lr in (
            ("SGD             ", lambda p, l: torch.optim.SGD(p, lr=l), 0.001),
            ("SGD(mom=0.9)    ", lambda p, l: torch.optim.SGD(
                p, lr=l, momentum=0.9), 0.001),
            ("Adam            ", lambda p, l: torch.optim.Adam(p, lr=l), 0.001),
            ("Adam            ", lambda p, l: torch.optim.Adam(p, lr=l), 0.01)):
        r = run(make, lr)
        print(f"  {name}  {lr:<8} " + "  ".join(f"{v:.2e}" for v in r))
    print("  SGD 는 lam=1 인 좌표(x1)를 1.35e-01 에 남겨 둔 채 끝난다.")
    print("  lr 을 키우면 lam=1000 인 좌표가 터지므로 키울 수도 없다")
    print("  모멘텀을 켜면 x1 도 2.16e-10 까지 내려간다 — 이 문제에서는")
    print("  모멘텀만으로 충분하다. Adam 이 꼭 이기는 것은 아니다")
    print("  Adam 의 특징은 이김이 아니라 **고름** 이다. lr=0.001 줄의 네")
    print("  값이 2.07e-02 로 전부 같다. 곡률이 1000 배 차이 나는 좌표들을")
    print("  같은 속도로 줄인다 — sqrt(v) 로 나누어 걸음을 맞추기 때문이다")
    print("  그래서 어느 좌표가 느린지 몰라도 되고, 그 점이 실무의 이점이다")


# ========================================================================
# 4. 공정하게 견주기 — 각자의 러닝레이트를 쓸어 본다
# ========================================================================

def fair_compare():
    header("4. 공정하게: 같은 출발점 + 각자 최적 lr + 씨앗 5개")
    torch.manual_seed(0)
    X = torch.linspace(-3, 3, 200).unsqueeze(1)
    Y = torch.sin(X * 2) + 0.1 * torch.randn(200, 1)

    def run(make, lr, seed, epochs=300):
        torch.manual_seed(seed)
        m = nn.Sequential(nn.Linear(1, 32), nn.Tanh(),
                          nn.Linear(32, 32), nn.Tanh(), nn.Linear(32, 1))
        o = make(m.parameters(), lr)
        for _ in range(epochs):
            o.zero_grad()
            nn.functional.mse_loss(m(X), Y).backward()
            o.step()
        with torch.no_grad():
            return nn.functional.mse_loss(m(X), Y).item()

    opts = (("SGD         ", lambda p, l: torch.optim.SGD(p, lr=l)),
            ("SGD(mom=0.9)", lambda p, l: torch.optim.SGD(
                p, lr=l, momentum=0.9)),
            ("Adam        ", lambda p, l: torch.optim.Adam(p, lr=l)))
    grid = (0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3)

    print("  먼저 각자의 lr 을 씨앗 하나로 쓸어 가장 좋은 값을 고른다")
    print("  옵티마이저     " + "".join(f"{l:>10}" for l in grid))
    best = {}
    for name, make in opts:
        row = []
        for lr in grid:
            v = run(make, lr, 0)
            row.append(v if math.isfinite(v) else float("inf"))
        best[name] = grid[int(np.argmin(row))]
        print(f"  {name}  " + "".join(
            f"{(v if v < 1e4 else float('inf')):>10.4f}" for v in row))
    print("  고른 lr: " + ",  ".join(f"{k.strip()} {v}" for k, v in best.items()))

    print("\n  그 lr 로 씨앗 5개를 돌린다")
    print("  옵티마이저     lr       로스 평균     퍼짐")
    for name, make in opts:
        vals = [run(make, best[name], s) for s in range(5)]
        print(f"  {name}  {best[name]:<7}  {np.mean(vals):.6f}   "
              f"{min(vals):.6f}~{max(vals):.6f}")
    print("  셋 다 최적 lr 이 0.1 로 같게 나왔다. 그 lr 에서도 Adam 이")
    print("  0.008795 로 SGD 의 0.050810 보다 5.8 배 낮다 — 이 문제에서는")
    print("  러닝레이트를 맞춰 주어도 Adam 이 이긴다")
    print("  다만 쓸어 보기 전의 표를 보라. lr=0.001 에서는 0.4484 대")
    print("  0.0528 로 8.5 배였다. 어느 쪽이든 Adam 이 낫지만 '얼마나'")
    print("  는 lr 을 맞췄는지에 따라 크게 달라진다")
    print("  그리고 Adam 의 lr 열은 0.003 부터 0.3 까지 평평하다 — 아무")
    print("  값이나 집어도 쓸 만하다는 뜻이고, 이것이 실무의 큰 이점이다")


# ========================================================================
# 5. Adam 과 AdamW 의 웨이트 감쇠
# ========================================================================

def decay():
    header("5. Adam(weight_decay) vs AdamW")
    print("  Adam 의 weight_decay 는 경사에 wd*p 를 더한다 — 그러면 그 항도")
    print("  sqrt(v) 로 나뉜다. AdamW 는 갱신 뒤에 따로 p 를 줄인다")

    for wd in (0.0, 0.1):
        outs = []
        for make in (torch.optim.Adam, torch.optim.AdamW):
            torch.manual_seed(0)
            p = nn.Parameter(torch.tensor([1.0, 1.0]))
            o = make([p], lr=0.1, weight_decay=wd)
            for _ in range(50):
                o.zero_grad()
                # 좌표마다 경사 크기가 다른 로스
                (0.5 * (torch.tensor([0.01, 100.0]) * p ** 2).sum()).backward()
                o.step()
            outs.append(p.detach().clone())
        print(f"  wd = {wd}:  Adam {outs[0].tolist()}")
        print(f"             AdamW {outs[1].tolist()}")
        print(f"             차이 {(outs[0]-outs[1]).abs().max():.6f}")
    print("  wd = 0 이면 둘이 비트까지 똑같다. wd 를 켜야 갈린다")
    print("  그런데 갈리는 모양이 뜻밖이다. Adam 쪽은 wd 를 0 에서 0.1 로")
    print("  올려도 값이 -0.0048174 에서 -0.0048181 로 **거의 안 변한다**")
    print("  AdamW 쪽은 -0.0174 로 3.6 배 커진다")
    print("  까닭: Adam 은 wd*p 를 경사에 더하는데, 그 합이 다시 sqrt(v) 로")
    print("  나뉜다. 로스 경사가 큰 자리에서는 v 가 커서 감쇠 항이 묻힌다")
    print("  곧 Adam 의 weight_decay 는 **경사 크기에 따라 세기가 달라진다**")
    print("  AdamW 는 갱신 뒤에 따로 빼므로 언제나 적은 만큼 줄인다")
    print("  이것이 AdamW 가 따로 있는 까닭이다. 같은 wd 값을 주어도 두")
    print("  옵티마이저가 전혀 다른 양만큼 줄인다")


# ========================================================================
# 6. 이어서 학습하려면 옵티마이저 상태도 저장해야 한다
# ========================================================================

def resume():
    header("6. 학습을 이어 하려면")
    torch.manual_seed(0)
    X, Y = torch.randn(32, 4), torch.randn(32, 1)

    def train(m, o, steps):
        for _ in range(steps):
            o.zero_grad()
            nn.functional.mse_loss(m(X), Y).backward()
            o.step()

    torch.manual_seed(1)
    ref = nn.Linear(4, 1)
    o_ref = torch.optim.Adam(ref.parameters(), lr=0.01)
    train(ref, o_ref, 40)

    for save_opt in (True, False):
        torch.manual_seed(1)
        m = nn.Linear(4, 1)
        o = torch.optim.Adam(m.parameters(), lr=0.01)
        train(m, o, 20)
        ckpt = {"model": m.state_dict(),
                "opt": o.state_dict() if save_opt else None}

        torch.manual_seed(1)
        m2 = nn.Linear(4, 1)
        m2.load_state_dict(ckpt["model"])
        o2 = torch.optim.Adam(m2.parameters(), lr=0.01)
        if ckpt["opt"] is not None:
            o2.load_state_dict(ckpt["opt"])
        train(m2, o2, 20)

        d = max((a - b).abs().max().item()
                for a, b in zip(m2.parameters(), ref.parameters()))
        lab = "옵티마이저 상태까지 저장" if save_opt else "모델만 저장      "
        print(f"  {lab}  40 걸음 통째와의 차이 {d:.6e}")
    print("  모델만 저장하면 m, v, t 가 0 으로 돌아가 다시 1 걸음째가 된다")
    print("  2 절에서 본 대로 1 걸음째의 걸음폭은 특별하므로, 이어 하는")
    print("  자리에서 갑자기 큰 걸음을 떼게 된다")
    print("  SGD(모멘텀 없음)는 상태가 없어 이 문제가 없다. 그래서 잊기 쉽다")


def main():
    match_pytorch()
    bias_correction()
    where_adam_wins()
    fair_compare()
    decay()
    resume()


if __name__ == "__main__":
    main()
```

??? note "전체 출력 (94줄)"

    ```
    === 1. 손으로 짠 Adam vs torch.optim.Adam ===
      걸음   손으로 짠 w[0]      PyTorch w[0]       차이
         1   +0.229432702065   +0.229432702065   0.00e+00
         2   +0.228432789445   +0.228432789445   0.00e+00
         5   +0.225434079766   +0.225434079766   0.00e+00
        10   +0.220442920923   +0.220442920923   0.00e+00
      10 걸음 뒤 모든 파라메터의 최대 차이 2.98e-08
      float32 의 반올림 자리다. 같은 알고리즘이라는 뜻이다

    === 2. 바이어스 보정 — 첫 걸음을 키우는가 줄이는가 ===
      m 과 v 를 0 에서 시작하므로 처음 몇 걸음은 0 쪽으로 치우친다
      걸음   보정 있음 걸음폭   보정 없음 걸음폭   비
         1   0.00100000       0.00316228       3.1623
         2   0.00100000       0.00424959       4.2496
         3   0.00100000       0.00495023       4.9502
         5   0.00100000       0.00579714       5.7971
        10   0.00100000       0.00652788       6.5279
      보정을 **빼면** 첫 걸음이 3.16 배 커진다. 줄어드는 것이 아니다
      까닭: 보정 없는 첫 걸음은 (1-b1)g / sqrt((1-b2)g^2)
           = 0.09999999999999998 / sqrt(0.0010000000000000009) = 3.1623 배
      보정이 있으면 첫 걸음이 정확히 lr 이다 (m_hat/sqrt(v_hat) = 1)
      곧 바이어스 보정은 **초기의 과도한 걸음을 막는** 장치다
      흔히 '0 쪽으로 치우친 것을 키워 준다' 고 적지만, 분자와 분모가
      함께 치우쳐 있어 몫에서는 반대로 나온다

    === 3. Adam 의 서명 — 좌표마다 같은 속도 ===
      좌표마다 곡률이 크게 다른 이차함수를 만든다
      f(x) = sum_i lam_i x_i^2 / 2,  lam = [1, 10, 100, 1000]
      조건수 1000

      2000 걸음 뒤 좌표별 |x| (0 에 가까울수록 좋다)
      옵티마이저        lr        x1        x2        x3        x4
      SGD               0.001    1.35e-01  1.86e-09  5.61e-45  0.00e+00
      SGD(mom=0.9)      0.001    2.16e-10  5.61e-45  0.00e+00  2.80e-45
      Adam              0.001    2.07e-02  2.07e-02  2.07e-02  2.07e-02
      Adam              0.01     6.59e-44  9.11e-44  8.97e-44  9.11e-44
      SGD 는 lam=1 인 좌표(x1)를 1.35e-01 에 남겨 둔 채 끝난다.
      lr 을 키우면 lam=1000 인 좌표가 터지므로 키울 수도 없다
      모멘텀을 켜면 x1 도 2.16e-10 까지 내려간다 — 이 문제에서는
      모멘텀만으로 충분하다. Adam 이 꼭 이기는 것은 아니다
      Adam 의 특징은 이김이 아니라 **고름** 이다. lr=0.001 줄의 네
      값이 2.07e-02 로 전부 같다. 곡률이 1000 배 차이 나는 좌표들을
      같은 속도로 줄인다 — sqrt(v) 로 나누어 걸음을 맞추기 때문이다
      그래서 어느 좌표가 느린지 몰라도 되고, 그 점이 실무의 이점이다

    === 4. 공정하게: 같은 출발점 + 각자 최적 lr + 씨앗 5개 ===
      먼저 각자의 lr 을 씨앗 하나로 쓸어 가장 좋은 값을 고른다
      옵티마이저         0.0003     0.001     0.003      0.01      0.03       0.1       0.3
      SGD               0.4890    0.4484    0.4208    0.3032    0.0815    0.0465       inf
      SGD(mom=0.9)      0.4224    0.3159    0.0876    0.0505    0.0268    0.0213       inf
      Adam              0.2393    0.0528    0.0196    0.0088    0.0087    0.0082    0.0244
      고른 lr: SGD 0.1,  SGD(mom=0.9) 0.1,  Adam 0.1

      그 lr 로 씨앗 5개를 돌린다
      옵티마이저     lr       로스 평균     퍼짐
      SGD           0.1      0.050810   0.046529~0.058011
      SGD(mom=0.9)  0.1      0.015427   0.011327~0.021344
      Adam          0.1      0.008795   0.008155~0.009836
      셋 다 최적 lr 이 0.1 로 같게 나왔다. 그 lr 에서도 Adam 이
      0.008795 로 SGD 의 0.050810 보다 5.8 배 낮다 — 이 문제에서는
      러닝레이트를 맞춰 주어도 Adam 이 이긴다
      다만 쓸어 보기 전의 표를 보라. lr=0.001 에서는 0.4484 대
      0.0528 로 8.5 배였다. 어느 쪽이든 Adam 이 낫지만 '얼마나'
      는 lr 을 맞췄는지에 따라 크게 달라진다
      그리고 Adam 의 lr 열은 0.003 부터 0.3 까지 평평하다 — 아무
      값이나 집어도 쓸 만하다는 뜻이고, 이것이 실무의 큰 이점이다

    === 5. Adam(weight_decay) vs AdamW ===
      Adam 의 weight_decay 는 경사에 wd*p 를 더한다 — 그러면 그 항도
      sqrt(v) 로 나뉜다. AdamW 는 갱신 뒤에 따로 p 를 줄인다
      wd = 0.0:  Adam [-0.0048174429684877396, -0.004818218760192394]
                 AdamW [-0.0048174429684877396, -0.004818218760192394]
                 차이 0.000000
      wd = 0.1:  Adam [-0.004818068817257881, -0.0048181721940636635]
                 AdamW [-0.01744718849658966, -0.017447788268327713]
                 차이 0.012630
      wd = 0 이면 둘이 비트까지 똑같다. wd 를 켜야 갈린다
      그런데 갈리는 모양이 뜻밖이다. Adam 쪽은 wd 를 0 에서 0.1 로
      올려도 값이 -0.0048174 에서 -0.0048181 로 **거의 안 변한다**
      AdamW 쪽은 -0.0174 로 3.6 배 커진다
      까닭: Adam 은 wd*p 를 경사에 더하는데, 그 합이 다시 sqrt(v) 로
      나뉜다. 로스 경사가 큰 자리에서는 v 가 커서 감쇠 항이 묻힌다
      곧 Adam 의 weight_decay 는 **경사 크기에 따라 세기가 달라진다**
      AdamW 는 갱신 뒤에 따로 빼므로 언제나 적은 만큼 줄인다
      이것이 AdamW 가 따로 있는 까닭이다. 같은 wd 값을 주어도 두
      옵티마이저가 전혀 다른 양만큼 줄인다

    === 6. 학습을 이어 하려면 ===
      옵티마이저 상태까지 저장  40 걸음 통째와의 차이 0.000000e+00
      모델만 저장        40 걸음 통째와의 차이 4.266685e-02
      모델만 저장하면 m, v, t 가 0 으로 돌아가 다시 1 걸음째가 된다
      2 절에서 본 대로 1 걸음째의 걸음폭은 특별하므로, 이어 하는
      자리에서 갑자기 큰 걸음을 떼게 된다
      SGD(모멘텀 없음)는 상태가 없어 이 문제가 없다. 그래서 잊기 쉽다
    ```

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
Adam의 첫 걸음 폭이 경사의 **크기** 와 무관함을 보여라. 경사를 $10^{-6}$, $1$, $10^{6}$으로 주고 한 걸음 뗀 뒤의 변화량을 각각 재어라.

</div>

??? success "풀이"
    ```python
    import torch
    import torch.nn as nn

    print("  경사 g       1 걸음 뒤 |p 변화|     lr")
    for g in (1e-6, 1.0, 1e6):
        p = nn.Parameter(torch.tensor([0.0]))
        o = torch.optim.Adam([p], lr=0.01)
        p.grad = torch.tensor([float(g)])
        o.step()
        print(f"  {g:<12.0e} {abs(p.item()):.10f}      0.01")
    ```

    ```
      경사 g       1 걸음 뒤 |p 변화|     lr
      1e-06        0.0099009890      0.01
      1e+00        0.0099999998      0.01
      1e+06        0.0099999998      0.01
    ```

    경사가 $10^{12}$배 차이 나는데 걸음폭이 모두 0.01, 곧 **정확히 $\eta$** 다.

    본문 2절의 식에서 바로 나온다. 첫 걸음에 $\hat{m}_1 = g$, $\hat{v}_1 = g^2$이므로

    $$\eta\,\frac{g}{\sqrt{g^2}+\epsilon} \approx \eta\,\text{sign}(g)$$

    다. Adam의 첫 걸음은 사실상 **부호만 쓴다.**

    $g = 10^{-6}$ 줄만 0.00990으로 1% 작은 것이 눈에 띈다. $\epsilon = 10^{-8}$이 분모에서 더 이상 무시되지 않기 때문이다($10^{-6}$에 대해 $10^{-8}$은 1%다). 연습문제 2가 이 자리를 더 본다.

    이 성질이 양날이다. 경사가 아무리 작아도 걸음을 떼므로 평평한 구간을 잘 빠져나오고, 경사가 아무리 커도 걸음이 커지지 않으므로 터지지 않는다. 대신 **경사의 크기 정보를 버린다** — 최솟값에 가까워져도 걸음이 저절로 작아지지 않는다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
`eps`는 "0으로 나누기를 막는 작은 수"로만 설명되는 일이 많다. 경사가 작을 때 `eps`가 실제로 하는 일을 재어라.

</div>

??? success "풀이"
    ```python
    print("  eps        경사 1e-4 로 100 걸음 뒤 |p|     경사 1.0 로 100 걸음 뒤 |p|")
    for eps in (1e-8, 1e-4, 1e-2):
        out = []
        for g in (1e-4, 1.0):
            p = nn.Parameter(torch.tensor([0.0]))
            o = torch.optim.Adam([p], lr=0.01, eps=eps)
            for _ in range(100):
                p.grad = torch.tensor([g])
                o.step()
            out.append(abs(p.item()))
        print(f"  {eps:<10.0e} {out[0]:.6f}                      {out[1]:.6f}")
    ```

    ```
      eps        경사 1e-4 로 100 걸음 뒤 |p|     경사 1.0 로 100 걸음 뒤 |p|
      1e-08      0.999899                      0.999999
      1e-04      0.500000                      0.999899
      1e-02      0.009901                      0.990099
    ```

    기본값 $\epsilon = 10^{-8}$에서는 두 경사 모두 100걸음에 1.0을 간다. 걸음마다 $\eta = 0.01$씩이다.

    $\epsilon = 10^{-2}$에서는 경사 $10^{-4}$ 쪽이 0.0099에 그친다. **100분의 1이다.** 분모 $\sqrt{\hat{v}} + \epsilon$에서 $\sqrt{\hat{v}} = 10^{-4}$이 $\epsilon = 10^{-2}$에 묻혀 버리기 때문이다.

    $\epsilon = 10^{-4}$, 경사 $10^{-4}$인 줄이 정확히 0.5인 것도 같은 까닭이다. 분모가 $10^{-4} + 10^{-4}$이라 걸음폭이 딱 절반이다.

    **그러니 `eps`는 안전장치만이 아니라 "이보다 작은 경사는 무시한다"는 문턱이다.** 경사가 매우 작은 파라메터가 있는 모델에서 `eps`를 올리면 그 파라메터들이 조용히 학습을 멈춘다. 거꾸로, 혼합 정밀도 학습에서 `eps`가 너무 작으면 $\sqrt{\hat{v}}$가 float16의 바닥 아래로 내려가 분모가 0이 되기도 한다. 그래서 그런 설정에서는 $10^{-8}$ 대신 $10^{-6}$ 같은 값을 권하는 일이 있다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
경사가 한동안 1.0으로 안정적이다가 갑자기 10으로 튀는 상황을 만들고, `beta2`를 바꿔 가며 걸음폭이 어떻게 반응하는지 재어라.

</div>

??? success "풀이"
    ```python
    print("  beta2     튄 직후 걸음폭      10 걸음 뒤 걸음폭")
    for b2 in (0.9, 0.99, 0.999):
        p = nn.Parameter(torch.tensor([0.0]))
        o = torch.optim.Adam([p], lr=0.01, betas=(0.9, b2))
        for _ in range(200):
            p.grad = torch.tensor([1.0])
            o.step()
        before = p.item()
        p.grad = torch.tensor([10.0])
        o.step()
        jump = abs(p.item() - before)
        for _ in range(9):
            prev = p.item()
            p.grad = torch.tensor([10.0])
            o.step()
        print(f"  {b2:<9} {jump:.8f}        {abs(p.item() - prev):.8f}")
    ```

    ```
      beta2     튄 직후 걸음폭      10 걸음 뒤 걸음폭
      0.9       0.00575483        0.00847983
      0.99      0.01298380        0.01999998
      0.999     0.01529348        0.02755618
    ```

    **`beta2`가 작을수록 튄 직후의 걸음이 작다.** 0.9에서 0.00575, 0.999에서 0.01529로 2.7배 차이다.

    $v$가 얼마나 빨리 따라오느냐의 문제다. $\beta_2 = 0.9$면 새 경사가 한 걸음에 10%씩 반영되므로 10으로 튄 값이 곧바로 $v$에 들어가고, 분모가 커져 걸음이 억제된다. $\beta_2 = 0.999$는 0.1%씩만 반영하므로 **아직 예전의 작은 경사를 기억하고 있고**, 분모가 작은 채로 큰 $\hat{m}$을 나누어 큰 걸음을 뗀다.

    기본값이 0.999인 까닭은 미니배치 경사가 본래 시끄럽기 때문이다. 한 배치에서 우연히 큰 경사가 나왔다고 걸음을 바꾸면 그 잡음을 따라가게 된다. 긴 기억이 잡음을 평균 내 준다.

    대신 **진짜 변화에도 늦게 반응한다.** 경사의 척도가 학습 도중 실제로 바뀌는 상황(스케줄을 크게 바꾸거나 로스 항을 추가하는 등)에서는 수백 걸음 동안 걸음폭이 맞지 않는다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
`AdamW`를 손으로 구현해 `torch.optim.AdamW`와 맞춰라. 본문 5절에서 `Adam`의 `weight_decay`가 거의 아무 일도 하지 않았는데, 그 까닭을 식으로 설명하라.

</div>

??? success "풀이"
    `AdamW`는 감쇠를 갱신에서 **떼어 낸다.**

    $$\theta \leftarrow \theta - \eta\lambda\theta, \qquad \theta \leftarrow \theta - \eta\frac{\hat{m}}{\sqrt{\hat{v}}+\epsilon}$$

    ```python
    import torch
    import torch.nn as nn

    lam = torch.tensor([0.01, 100.0])

    def hand_adamw(steps=30, lr=0.1, wd=0.1, b1=0.9, b2=0.999, eps=1e-8):
        p = torch.tensor([1.0, 1.0])
        m = torch.zeros(2)
        v = torch.zeros(2)
        for t in range(1, steps + 1):
            g = lam * p                       # f = sum lam_i p_i^2 / 2
            m = b1 * m + (1 - b1) * g
            v = b2 * v + (1 - b2) * g * g
            mh, vh = m / (1 - b1 ** t), v / (1 - b2 ** t)
            p = p - lr * wd * p               # 분리된 감쇠
            p = p - lr * mh / (vh.sqrt() + eps)
        return p

    pw = nn.Parameter(torch.tensor([1.0, 1.0]))
    o = torch.optim.AdamW([pw], lr=0.1, weight_decay=0.1)
    for _ in range(30):
        o.zero_grad()
        (0.5 * (lam * pw ** 2).sum()).backward()
        o.step()

    h = hand_adamw()
    print(f"  손으로  {h.tolist()}")
    print(f"  PyTorch {pw.detach().tolist()}")
    print(f"  최대 차이 {(h - pw.detach()).abs().max():.2e}")
    ```

    ```
      손으로  [-0.0027174632996320724, -0.002716507762670517]
      PyTorch [-0.0027174614369869232, -0.002716490998864174]
      최대 차이 1.68e-08
    ```

    맞는다. 감쇠를 **갱신 앞에** 적용하는 것까지 PyTorch와 같아야 자릿수가 맞는다.

    **`Adam`의 감쇠가 묻히는 까닭.** `Adam`은 감쇠를 경사에 섞는다.

    $$g_t \leftarrow g_t + \lambda\theta_t$$

    이 $g_t$가 $m$과 $v$에 모두 들어가고, 최종 걸음은

    $$\eta\,\frac{\widehat{m(g+\lambda\theta)}}{\sqrt{\widehat{v(g+\lambda\theta)}}+\epsilon}$$

    이다. 분자와 분모에 **같은 양이 들어간다.** 1절의 식이 본질적으로 $\text{sign}$에 가깝게 동작하므로, $\lambda\theta$를 더해도 그 몫은 크게 변하지 않는다. $|g| \gg \lambda|\theta|$인 자리에서는 특히 그렇다.

    본문 5절의 측정이 그 극단이다. $\lambda$를 0에서 0.1로 올렸는데 결과가 $-0.0048174$에서 $-0.0048181$로 $7\times10^{-7}$ 움직였다.

    **실무에서 뜻하는 바 둘.**

    - `Adam(weight_decay=0.1)`을 쓰고 "정규화를 켰다"고 생각하면 안 된다. 켜졌는지 재어 보아야 한다
    - `Adam`에서 `AdamW`로 바꿀 때 같은 `weight_decay` 값을 그대로 쓰면 **정규화가 갑자기 세진다.** 둘은 같은 이름의 다른 하이퍼파라메터다

    이것이 Loshchilov와 Hutter가 `AdamW`를 따로 제안한 요지다. "Adam은 일반화가 나쁘다"고 보고되던 결과의 상당 부분이 알고리즘이 아니라 **감쇠가 작동하지 않았던 것** 이었다.

---

## 정리하며

1. **Adam은 다섯 줄이다**: 손으로 짠 것이 `torch.optim.Adam`과 열두 자리까지 같다
2. **바이어스 보정은 첫 걸음을 줄인다**: 빼면 3.16배 커진다. "작아진 것을 키워 준다"는 설명은 분모가 함께 작아진다는 점을 빠뜨린 것이다
3. **첫 걸음 폭은 경사 크기와 무관하게 정확히 $\eta$다**: 경사가 $10^{-6}$이든 $10^{6}$이든 같다
4. **Adam의 서명은 빠름이 아니라 고름이다**: 곡률이 1,000배 다른 네 좌표를 $2.07\times10^{-2}$로 똑같이 줄인다. 이 문제에서 더 낮게 내려간 것은 모멘텀 SGD다
5. **러닝레이트를 각자 맞춰도 Adam이 이겼다**: 0.0088 대 0.0508. 다만 맞추기 전에는 8.5배였고 맞춘 뒤에는 5.8배다
6. **Adam의 진짜 이점은 lr 범위가 넓다는 것이다**: 0.003부터 0.3까지 평평하다
7. **`eps`는 안전장치가 아니라 문턱이다**: $10^{-2}$로 올리면 작은 경사의 파라메터가 100분의 1 속도로 배운다
8. **`Adam(weight_decay)`는 거의 작동하지 않았다**: $\lambda$를 0.1로 올려도 결과가 $7\times10^{-7}$ 움직였다. `AdamW`는 3.6배 움직였다. 같은 이름의 다른 하이퍼파라메터다
9. **Adam은 체크포인트에 옵티마이저 상태가 필요하다**: 모델만 저장하면 $m, v, t$가 0으로 돌아가 다시 1걸음째가 된다

**참고 문헌**

- Kingma, D. P., & Ba, J. (2015). Adam: A Method for Stochastic Optimization. *ICLR*. arXiv:1412.6980.
- Loshchilov, I., & Hutter, F. (2019). Decoupled Weight Decay Regularization. *ICLR*. arXiv:1711.05101.
- Liu, L., et al. (2020). On the Variance of the Adaptive Learning Rate and Beyond. *ICLR*. arXiv:1908.03265.
- Reddi, S., Kale, S., & Kumar, S. (2018). On the Convergence of Adam and Beyond. *ICLR*.
