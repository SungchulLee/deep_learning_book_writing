# 옵티마이저 없이 갱신하기 — `step()`이 숨기고 있는 세 줄

`opt.step()` 한 줄이 하는 일을 손으로 적어 본다. SGD·모멘텀·Adam 셋을 차례로
짜서 파이토치 것과 견주고, 숫자가 맞는지 확인한다.

맞으면 두 가지를 얻는다. **옵티마이저가 마법이 아니라는 것**, 그리고 **어디를
건드리면 무엇이 달라지는지**를 코드로 알게 되는 것.

그 과정에서 「최소 학습 단계」에서 본 구별이 다시 나온다. 경사를
**구하는**(`backward`) 일과 그 경사로 값을 **바꾸는**(갱신) 일은 다른 단계이고,
둘 사이에 `no_grad`가 들어간다.

## 1. 코드

```python
"""옵티마이저 없이 걷기 — step() 이 숨기고 있는 세 줄.

`opt.step()` 한 줄이 하는 일을 손으로 적어 본다. 경사를 구하는
일(`backward`)과 그 경사로 값을 바꾸는 일(갱신)은 **다른 단계**이고,
둘 사이에 `no_grad` 가 들어간다.

SGD · 모멘텀 · Adam 을 차례로 손으로 짜서 파이토치 것과 견준다.
셋 다 소수점 끝까지 맞으면, 옵티마이저가 마법이 아니라는 것이
확인된 셈이다.

실행: 이 쪽의 코드를 파일로 갈무리해 돌린다
"""
import warnings

import torch


def header(title):
    print(f"\n=== {title} ===")


def problem(seed=0):
    """작은 이차함수 하나. 최솟값은 w = [1, -2] 다."""
    torch.manual_seed(seed)
    target = torch.tensor([1.0, -2.0])
    w = torch.tensor([3.0, 3.0], requires_grad=True)
    return w, target


def loss_of(w, target):
    return ((w - target) ** 2).sum()


# ========================================================================
# 1. SGD
# ========================================================================

def plain_sgd():
    header("1. SGD")
    lr, steps = 0.1, 20

    w, t = problem()
    opt = torch.optim.SGD([w], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        loss_of(w, t).backward()
        opt.step()
    by_torch = w.detach().clone()

    w2, t2 = problem()
    for _ in range(steps):
        if w2.grad is not None:
            w2.grad.zero_()
        loss_of(w2, t2).backward()
        with torch.no_grad():
            w2 -= lr * w2.grad               # 이 한 줄이 step() 이다
    print("  torch.optim.SGD :", [round(v, 8) for v in by_torch.tolist()])
    print("  손으로          :", [round(v, 8) for v in w2.detach().tolist()])
    print("  같은가          :", torch.allclose(by_torch, w2.detach(), atol=0))


# ========================================================================
# 2. 모멘텀
# ========================================================================

def momentum():
    header("2. 모멘텀")
    lr, mu, steps = 0.1, 0.9, 20

    w, t = problem()
    opt = torch.optim.SGD([w], lr=lr, momentum=mu)
    for _ in range(steps):
        opt.zero_grad()
        loss_of(w, t).backward()
        opt.step()
    by_torch = w.detach().clone()

    w2, t2 = problem()
    buf = None
    for _ in range(steps):
        if w2.grad is not None:
            w2.grad.zero_()
        loss_of(w2, t2).backward()
        with torch.no_grad():
            g = w2.grad
            buf = g.clone() if buf is None else mu * buf + g
            w2 -= lr * buf
    print("  torch 모멘텀 :", [round(v, 8) for v in by_torch.tolist()])
    print("  손으로       :", [round(v, 8) for v in w2.detach().tolist()])
    print("  가장 큰 차이 :", f"{(by_torch - w2.detach()).abs().max():.2e}")
    print("  첫 걸음의 버퍼를 g 로 두는 것이 파이토치의 약속이다")
    print("  0 으로 두고 mu*0 + g 를 해도 같은 값이지만, 구현에 따라 다르다")


# ========================================================================
# 3. Adam
# ========================================================================

def adam():
    header("3. Adam")
    lr, b1, b2, eps, steps = 0.1, 0.9, 0.999, 1e-8, 20

    w, t = problem()
    opt = torch.optim.Adam([w], lr=lr, betas=(b1, b2), eps=eps)
    for _ in range(steps):
        opt.zero_grad()
        loss_of(w, t).backward()
        opt.step()
    by_torch = w.detach().clone()

    w2, t2 = problem()
    m = torch.zeros_like(w2)
    v = torch.zeros_like(w2)
    for step in range(1, steps + 1):
        if w2.grad is not None:
            w2.grad.zero_()
        loss_of(w2, t2).backward()
        with torch.no_grad():
            g = w2.grad
            m = b1 * m + (1 - b1) * g
            v = b2 * v + (1 - b2) * g * g
            m_hat = m / (1 - b1 ** step)       # 치우침 보정
            v_hat = v / (1 - b2 ** step)
            w2 -= lr * m_hat / (v_hat.sqrt() + eps)
    print("  torch.optim.Adam:", [round(v, 8) for v in by_torch.tolist()])
    print("  손으로          :", [round(v, 8) for v in w2.detach().tolist()])
    print("  가장 큰 차이     :", f"{(by_torch - w2.detach()).abs().max():.2e}")


# ========================================================================
# 4. no_grad 를 빼면
# ========================================================================

def why_no_grad():
    header("4. no_grad 를 빼면")
    w, t = problem()
    loss_of(w, t).backward()
    try:
        w -= 0.1 * w.grad
    except RuntimeError as e:
        print("  제자리로 고치면:", str(e).splitlines()[0][:62])

    w2, t2 = problem()
    loss_of(w2, t2).backward()
    w3 = w2 - 0.1 * w2.grad                  # 새 텐서를 만들면 통과한다
    print("  새 텐서를 만들면 통과한다. 그런데")
    print("    w3.is_leaf       :", w3.is_leaf)
    print("    w3.requires_grad :", w3.requires_grad)
    loss_of(w3, t2).backward()
    with warnings.catch_warnings():        # 경고문에 빌드 기계 경로가 붙는다
        warnings.simplefilter("ignore")
        print("    두 번째 걸음의 w3.grad:", w3.grad)
    print("  잎이 아니라 경사를 못 받는다 — 다음 걸음이 없다")


def main():
    plain_sgd()
    momentum()
    adam()
    why_no_grad()


if __name__ == "__main__":
    main()
```

**출력:**

```
=== 1. SGD ===
  torch.optim.SGD : [1.02305853, -1.94235396]
  손으로          : [1.02305853, -1.94235396]
  같은가          : True

=== 2. 모멘텀 ===
  torch 모멘텀 : [0.30149513, -3.74626255]
  손으로       : [0.30149496, -3.74626255]
  가장 큰 차이 : 1.79e-07
  첫 걸음의 버퍼를 g 로 두는 것이 파이토치의 약속이다
  0 으로 두고 mu*0 + g 를 해도 같은 값이지만, 구현에 따라 다르다

=== 3. Adam ===
  torch.optim.Adam: [1.22245252, 1.06033909]
  손으로          : [1.22245252, 1.06033909]
  가장 큰 차이     : 0.00e+00

=== 4. no_grad 를 빼면 ===
  제자리로 고치면: a leaf Variable that requires grad is being used in an in-plac
  새 텐서를 만들면 통과한다. 그런데
    w3.is_leaf       : False
    w3.requires_grad : True
    두 번째 걸음의 w3.grad: None
  잎이 아니라 경사를 못 받는다 — 다음 걸음이 없다
```

## 2. SGD는 뺄셈 한 줄이다

```
torch.optim.SGD : [1.02305853, -1.94235396]
손으로          : [1.02305853, -1.94235396]
같은가          : True
```

```python
with torch.no_grad():
    w -= lr * w.grad
```

$w \leftarrow w - \eta \nabla w$가 식 그대로다. 스무 걸음을 돌려도 어긋나지
않는다.

## 3. 모멘텀은 버퍼 하나를 더 든다

```python
buf = g.clone() if buf is None else mu * buf + g
w -= lr * buf
```

```
torch 모멘텀 : [0.30149513, -3.74626255]
손으로       : [0.30149496, -3.74626255]
가장 큰 차이 : 1.79e-07
```

$1.79 \times 10^{-7}$ 차이는 float32의 반올림이다 — 더하는 순서가 조금 달라
생긴다.

**첫 걸음을 어떻게 시작하느냐**가 구현마다 갈리는 자리다. 파이토치는 버퍼를
0으로 두고 $\mu \cdot 0 + g$를 하는 대신 **첫 경사를 그대로** 버퍼에 넣는다.
여기서는 결과가 같지만, 다른 라이브러리의 수를 재현하려 할 때 이런 약속 차이가
안 맞는 원인이 된다.

목표가 $[1, -2]$인데 모멘텀 쪽이 $[0.30, -3.75]$로 **지나쳐 있다.** 관성이
붙어 최솟값을 넘어간 것이고, 스무 걸음으로는 돌아오지 못했다. 모멘텀이 언제나
좋은 것은 아니라는 뜻이다.

## 4. Adam은 버퍼 둘과 보정

```python
m = b1 * m + (1 - b1) * g              # 1차 모멘트
v = b2 * v + (1 - b2) * g * g          # 2차 모멘트
m_hat = m / (1 - b1 ** step)           # 치우침 보정
v_hat = v / (1 - b2 ** step)
w -= lr * m_hat / (v_hat.sqrt() + eps)
```

```
torch.optim.Adam: [1.22245252, 1.06033909]
손으로          : [1.22245252, 1.06033909]
가장 큰 차이     : 0.00e+00
```

**차이가 정확히 0이다.** 다섯 줄이 파이토치 구현과 같은 수를 낸다.

`step`으로 나누는 **치우침 보정**이 Adam에서 빠뜨리기 쉬운 부분이다. $m$과 $v$를
0에서 시작하므로 처음 몇 걸음은 실제보다 작게 추정되는데, $1 - \beta^t$로 나누어
그것을 되돌린다. 「모델 체크포인트」에서 옵티마이저 상태를 안 되살리면 Adam의
첫 걸음이 1.64배 커졌던 것이 이 보정이 다시 적용되기 때문이다.

Adam도 스무 걸음으로는 목표에 못 닿았다($[1.22, 1.06]$ 대 $[1, -2]$). 걸음
크기를 경사의 크기로 나누므로 **멀리 있어도 성큼 가지 않는다** — SGD와 다른
성격이다.

## 5. `no_grad`를 빼면

```
제자리로 고치면: a leaf Variable that requires grad is being used in an in-place
새 텐서를 만들면 통과한다. 그런데
  w3.is_leaf       : False
  w3.requires_grad : True
  두 번째 걸음의 w3.grad: None
```

빠져나갈 길이 있어 보인다 — 제자리 대신 `w3 = w - 0.1 * w.grad`로 새 텐서를
만들면 오류가 안 난다.

그런데 그 `w3`는 **잎이 아니다.** 「최소 학습 단계」 연습문제 2에서 본 그대로,
두 번째 걸음에서 `.grad`가 `None`이라 거기서 멈춘다. 게다가 옵티마이저를
쓴다면 그것은 **처음 넘겨받은 객체**를 계속 들고 있으므로, 이름만 바뀐 새
텐서는 영영 갱신되지 않는다.

그러니 갱신은 **제자리여야 하고**, 제자리이려면 `no_grad`가 필요하다. 둘이
맞물려 있다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
`w.grad`를 지우지 않고 손으로 갱신하면 어떻게 되는가? SGD 다섯 걸음을 지우는
쪽과 안 지우는 쪽으로 돌려 보라.

</div>

??? success "풀이"
    ```python
    for label, clear in (("지운다  ", True), ("안 지운다", False)):
        w, t = problem()
        for _ in range(5):
            if clear and w.grad is not None:
                w.grad.zero_()
            loss_of(w, t).backward()
            with torch.no_grad():
                w -= 0.1 * w.grad
        print(f"  {label}  w = {[round(v, 6) for v in w.detach().tolist()]}")
    print("  목표는 [1.0, -2.0]")
    ```

    ```
      지운다    w = [1.65536, -0.3616]
      안 지운다  w = [-0.61984, -6.0496]
    ```

    지운 쪽은 $[3, 3]$에서 목표 $[1, -2]$ 쪽으로 다가가고 있다. 안 지운 쪽은
    **목표를 한참 지나쳐 반대편으로 날아갔다.**

    걸음마다 지난 경사가 더해지므로 걸음이 기하급수적으로 커진다. 「경사 누적」
    연습문제 1에서 본 것과 같은 꼴인데, 여기서는 두 성분이 서로 다른 속도로
    발산하는 것까지 보인다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
3절에서 모멘텀이 목표를 지나쳤다. $\mu$를 바꿔 가며 **지나치는 정도**를 재고,
어느 값이 가장 빨리 닿는지 찾아라.

</div>

??? success "풀이"
    ```python
    print(f"  {'mu':>5}  {'20걸음 뒤 w':>26}  {'목표와의 거리':>12}")
    for mu in (0.0, 0.3, 0.5, 0.7, 0.9, 0.95):
        w, t = problem()
        opt = torch.optim.SGD([w], lr=0.1, momentum=mu)
        for _ in range(20):
            opt.zero_grad()
            loss_of(w, t).backward()
            opt.step()
        d = (w.detach() - t).norm().item()
        print(f"  {mu:>5}  {str([round(v, 4) for v in w.detach().tolist()]):>26}"
              f"  {d:12.6f}")
    ```

    ```
         mu                    20걸음 뒤 w       목표와의 거리
        0.0           [1.0231, -1.9424]      0.062087
        0.3           [1.0002, -1.9995]      0.000580
        0.5           [1.0006, -1.9986]      0.001508
        0.7            [0.947, -2.1326]      0.142803
        0.9           [0.3015, -3.7463]      1.880782
       0.95          [-0.2057, -5.0142]      3.246432
    ```

    **0.3이 가장 좋다.** 거리가 0.00058로, 모멘텀 없는 쪽(0.0621)보다 **백 배**
    가깝다.

    그리고 0.7부터 나빠지기 시작해 0.9에서 망가진다. 거리가 1.88이니 모멘텀
    없는 것보다 **서른 배 나쁘다.** 관성이 너무 커서 최솟값을 지나치고
    되돌아오기를 반복한다.

    흔히 쓰는 0.9가 여기서 최악인 것이 눈에 걸린다. 0.9가 표준이 된 것은 **큰
    모델에서 경사가 잡음투성이일 때** 그 잡음을 평균내 주기 때문인데, 이
    보기처럼 경사가 깨끗한 이차함수에서는 평균낼 잡음이 없어 관성만 남는다.

    「최적화 기초」에서 옵티마이저 선택이 1.0~3.3배만 바꾼 데 비해 러닝레이트가
    28.7~92.4배를 바꾼 것과 같은 이야기다. **하이퍼파라미터의 좋은 값은 문제에
    달려 있고**, 표준값은 어떤 문제들에서 좋았던 값일 뿐이다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
4절의 Adam에서 **치우침 보정을 빼면** 어떻게 되는가? 첫 몇 걸음을 파이토치와
견주어, 보정이 무엇을 고치는지 보여라.

</div>

??? success "풀이"
    ```python
    lr, b1, b2, eps = 0.1, 0.9, 0.999, 1e-8

    def hand_adam(correct, steps=5):
        w, t = problem()
        m = torch.zeros_like(w)
        v = torch.zeros_like(w)
        out = []
        for step in range(1, steps + 1):
            if w.grad is not None:
                w.grad.zero_()
            loss_of(w, t).backward()
            with torch.no_grad():
                g = w.grad
                m = b1 * m + (1 - b1) * g
                v = b2 * v + (1 - b2) * g * g
                if correct:
                    mh, vh = m / (1 - b1 ** step), v / (1 - b2 ** step)
                else:
                    mh, vh = m, v
                w -= lr * mh / (vh.sqrt() + eps)
                out.append(w[0].item())
        return out

    w, t = problem()
    opt = torch.optim.Adam([w], lr=lr, betas=(b1, b2), eps=eps)
    ref = []
    for _ in range(5):
        opt.zero_grad()
        loss_of(w, t).backward()
        opt.step()
        ref.append(w[0].item())

    print(f"  {'걸음':>4} {'파이토치':>12} {'보정 있음':>12} {'보정 없음':>12}")
    for i, (a, b, c) in enumerate(zip(ref, hand_adam(True), hand_adam(False)), 1):
        print(f"  {i:>4} {a:12.6f} {b:12.6f} {c:12.6f}")
    ```

    ```
        걸음         파이토치        보정 있음        보정 없음
           1     2.900000     2.900000     2.683772
           2     2.800167     2.800167     2.262265
           3     2.700624     2.700624     1.782943
           4     2.601505     2.601505     1.283828
           5     2.502956     2.502956     0.806633
    ```

    **보정 있는 쪽이 파이토치와 똑같고, 없는 쪽은 훨씬 크게 걷는다.**

    첫 걸음을 보자. 보정 있는 쪽은 3.000에서 2.900으로 정확히 0.1 —
    러닝레이트만큼 — 갔다. 보정 없는 쪽은 2.684로 **0.316**을 갔다.

    셈이 그 수를 설명한다. $m$과 $v$를 0에서 시작하므로 첫 걸음에서
    $m_1 = 0.1g$, $v_1 = 0.001g^2$이고

    $$
    \frac{m_1}{\sqrt{v_1}} = \frac{0.1\,g}{0.0316\,|g|} \approx 3.16\,\text{sign}(g)
    $$

    이다. 여기에 $\eta = 0.1$을 곱하면 0.316이다. 잰 값과 맞는다.

    보정은 $m$을 $1 - \beta_1^t = 0.1$로, $v$를 $1 - \beta_2^t = 0.001$로 나눈다.
    그러면 $\hat{m}_1 = g$, $\hat{v}_1 = g^2$이 되어 비가 정확히 1이 되고,
    걸음이 $\eta$가 된다.

    **그래서 Adam의 첫 걸음은 언제나 러닝레이트 크기다.** 경사가 크든 작든
    상관없다 — 그것이 Adam이 러닝레이트를 고르기 쉬운 까닭이다.

    보정을 빼면 초반 몇 걸음이 세 배씩 커진다. 여기서는 목표 쪽으로 가는
    방향이라 우연히 빨라 보이지만, 방향이 나쁘면 그만큼 크게 어긋난다. 보정은
    **빠르게 하려는 것이 아니라 걸음 크기를 예측 가능하게** 만드는 장치다.
