# 경사를 0으로 만드는 세 가지 길 — 그리고 "어디까지" 지우는가

`opt.zero_grad()` · `model.zero_grad()` · `p.grad = None`. 셋 다 경사를
치운다. 그런데 **둘은 범위가 다르고 하나는 남기는 것이 다르다.**

| 길 | 범위 | 남기는 것 |
|---|---|---|
| `opt.zero_grad()` | **옵티마이저가 든 것만** | `None` |
| `model.zero_grad()` | 모델 전체 | `None` |
| `p.grad = None` | 그 텐서 하나 | `None` |
| `p.grad.zero_()` | 그 텐서 하나 | **0으로 채운 텐서** |

첫 줄이 함정이다. 옵티마이저에 **일부만** 넘겼다면 `opt.zero_grad()`는 나머지를
안 지우고, 안 지워진 경사는 걸음마다 쌓인다.

## 1. 코드

```python
"""경사를 지우는 세 가지 길 — 그리고 '어디까지' 지우는가.

`opt.zero_grad()` · `model.zero_grad()` · `p.grad = None`.
셋 다 경사를 치우는데, 둘은 **범위가 다르고** 하나는 **남기는 것이
다르다.**

| 길 | 범위 | 남기는 것 |
|---|---|---|
| `opt.zero_grad()` | 옵티마이저가 든 것만 | `None` (기본값) |
| `model.zero_grad()` | 모델 전체 | `None` (기본값) |
| `p.grad = None` | 그 텐서 하나 | `None` |
| `p.grad.zero_()` | 그 텐서 하나 | **0 으로 채운 텐서** |

범위가 갈리는 자리가 함정이다. 옵티마이저에 일부만 넘겼다면
`opt.zero_grad()` 는 나머지를 **안 지운다.**

실행: 이 쪽의 코드를 파일로 갈무리해 돌린다
"""
import inspect

import torch
import torch.nn as nn
import torch.nn.functional as F


def header(title):
    print(f"\n=== {title} ===")


def has_grad(net):
    return [p.grad is not None for p in net.parameters()]


# ========================================================================
# 1. 범위가 다르다
# ========================================================================

def scope():
    header("1. 범위")
    torch.manual_seed(0)
    net = nn.Sequential(nn.Linear(4, 8), nn.Linear(8, 2))
    opt = torch.optim.SGD(net[1].parameters(), lr=0.1)   # 뒤쪽만 넘겼다
    x, y = torch.randn(8, 4), torch.randn(8, 2)

    F.mse_loss(net(x), y).backward()
    print("  역전파 뒤 경사 있는가 :", has_grad(net))
    opt.zero_grad()
    print("  opt.zero_grad() 뒤    :", has_grad(net))
    print("    앞 둘이 True 로 남았다 — 옵티마이저가 모르는 텐서다")
    net.zero_grad()
    print("  model.zero_grad() 뒤  :", has_grad(net))
    print("  모델 전체를 지우려면 모델에게 시켜야 한다")


# ========================================================================
# 2. 안 지우면 쌓인다
# ========================================================================

def leftover_accumulates():
    header("2. 안 지운 쪽은 쌓인다")
    torch.manual_seed(0)
    net = nn.Sequential(nn.Linear(4, 8), nn.Linear(8, 2))
    opt = torch.optim.SGD(net[1].parameters(), lr=0.1)
    x, y = torch.randn(8, 4), torch.randn(8, 2)

    sizes = []
    for step in range(3):
        opt.zero_grad()                       # 뒤쪽만 지운다
        F.mse_loss(net(x), y).backward()
        sizes.append(net[0].weight.grad.abs().max().item())
    print("  0층 경사의 크기가 걸음마다:",
          [round(v, 6) for v in sizes])
    print("  지워지지 않아 1배, 2배, 3배로 자란다")
    print("  0층을 학습하지 않더라도 이 경사는 메모리를 차지한다")


# ========================================================================
# 3. None 과 0 으로 채운 텐서
# ========================================================================

def none_or_zeros():
    header("3. None 과 0")
    torch.manual_seed(1)
    w = torch.randn(3, requires_grad=True)
    opt = torch.optim.SGD([w], lr=0.1)

    (w ** 2).sum().backward()
    print("  역전파 뒤          :", w.grad)
    opt.zero_grad()
    print("  opt.zero_grad()    :", w.grad, " <- 기본이 None")
    (w ** 2).sum().backward()
    opt.zero_grad(set_to_none=False)
    print("  set_to_none=False  :", w.grad)
    (w ** 2).sum().backward()
    w.grad = None
    print("  w.grad = None      :", w.grad)
    (w ** 2).sum().backward()
    w.grad.zero_()
    print("  w.grad.zero_()     :", w.grad)

    sig = inspect.signature(opt.zero_grad)
    print("  이 판의 기본값     :", sig.parameters["set_to_none"].default,
          f"(torch {torch.__version__})")
    print("  예전 판은 0 으로 채우는 것이 기본이었다")


# ========================================================================
# 4. 그 차이가 걸음을 바꾼다
# ========================================================================

def it_changes_the_step():
    header("4. 모멘텀이 있으면 걸음이 달라진다")
    got = []
    for set_to_none in (True, False):
        torch.manual_seed(0)
        m = nn.Linear(3, 1)
        opt = torch.optim.SGD(m.parameters(), lr=0.1, momentum=0.9)
        x, y = torch.randn(5, 3), torch.randn(5, 1)
        F.mse_loss(m(x), y).backward()
        opt.step()
        opt.zero_grad(set_to_none=set_to_none)
        opt.step()                              # 경사가 빈 채로 한 걸음
        got.append(m.weight.detach().flatten().clone())
    print("  set_to_none=True  :", "  ".join(f"{v:+.6f}" for v in got[0].tolist()))
    print("  set_to_none=False :", "  ".join(f"{v:+.6f}" for v in got[1].tolist()))
    print(f"  가장 큰 차이      : {(got[0] - got[1]).abs().max():.6f}")
    print("  None 이면 step 이 건너뛰고, 0 이면 모멘텀이 민다")


# ========================================================================
# 5. 그래서 무엇을 쓰는가
# ========================================================================

def which():
    header("5. 고르기")
    torch.manual_seed(0)
    net = nn.Sequential(nn.Linear(4, 8), nn.Linear(8, 2))
    opt = torch.optim.SGD(net.parameters(), lr=0.1)
    x, y = torch.randn(8, 4), torch.randn(8, 2)

    for _ in range(2):
        opt.zero_grad()
        F.mse_loss(net(x), y).backward()
        opt.step()
    print("  보통은 opt.zero_grad() 다 — 걷는 것과 지우는 것이 한 짝이다")
    print("  옵티마이저가 모델 전체를 들고 있으면 범위도 같다")
    print("  들고 있는 파라미터 수:",
          sum(len(g["params"]) for g in opt.param_groups),
          " 모델의 파라미터 수:", len(list(net.parameters())))
    print("  둘이 다르면 model.zero_grad() 를 쓰는 편이 안전하다")


def main():
    scope()
    leftover_accumulates()
    none_or_zeros()
    it_changes_the_step()
    which()


if __name__ == "__main__":
    main()
```

**출력:**

```
=== 1. 범위 ===
  역전파 뒤 경사 있는가 : [True, True, True, True]
  opt.zero_grad() 뒤    : [True, True, False, False]
    앞 둘이 True 로 남았다 — 옵티마이저가 모르는 텐서다
  model.zero_grad() 뒤  : [False, False, False, False]
  모델 전체를 지우려면 모델에게 시켜야 한다

=== 2. 안 지운 쪽은 쌓인다 ===
  0층 경사의 크기가 걸음마다: [0.218426, 0.436852, 0.655278]
  지워지지 않아 1배, 2배, 3배로 자란다
  0층을 학습하지 않더라도 이 경사는 메모리를 차지한다

=== 3. None 과 0 ===
  역전파 뒤          : tensor([1.3227, 0.5338, 0.1234])
  opt.zero_grad()    : None  <- 기본이 None
  set_to_none=False  : tensor([0., 0., 0.])
  w.grad = None      : None
  w.grad.zero_()     : tensor([0., 0., 0.])
  이 판의 기본값     : True (torch 2.5.1)
  예전 판은 0 으로 채우는 것이 기본이었다

=== 4. 모멘텀이 있으면 걸음이 달라진다 ===
  set_to_none=True  : -0.021486  +0.282214  -0.422172
  set_to_none=False : -0.036934  +0.257463  -0.374460
  가장 큰 차이      : 0.047712
  None 이면 step 이 건너뛰고, 0 이면 모멘텀이 민다

=== 5. 고르기 ===
  보통은 opt.zero_grad() 다 — 걷는 것과 지우는 것이 한 짝이다
  옵티마이저가 모델 전체를 들고 있으면 범위도 같다
  들고 있는 파라미터 수: 4  모델의 파라미터 수: 4
  둘이 다르면 model.zero_grad() 를 쓰는 편이 안전하다
```

## 2. 범위가 다르다

```
역전파 뒤 경사 있는가 : [True, True, True, True]
opt.zero_grad() 뒤    : [True, True, False, False]
model.zero_grad() 뒤  : [False, False, False, False]
```

옵티마이저에 뒤쪽 층만 넘겼더니 **앞쪽 둘이 `True`로 남았다.** 옵티마이저는
제가 들고 있는 것만 지운다.

전이학습에서 이 꼴이 흔하다. 앞쪽을 얼리고 거른 목록으로 옵티마이저를 만들면,
얼린 층에도 경사는 계속 오는데(「경사 추적 끄기」에서 본 대로 경사는 얼린 층을
지나간다) 아무도 안 지운다.

## 3. 안 지운 쪽은 쌓인다

```
0층 경사의 크기가 걸음마다: [0.218426, 0.436852, 0.655278]
```

**정확히 1배, 2배, 3배다.** 걸음마다 더해지기 때문이다.

그 층을 학습하지 않으니 결과는 안 틀린다. 다만

- **메모리를 차지한다** — 경사 텐서가 파라미터 크기만큼 계속 살아 있다
- **나중에 녹이면 터진다** — 그때까지 쌓인 경사로 첫 걸음을 걷는다

그래서 일부만 학습할 때는 `model.zero_grad()`를 쓰거나, 얼린 층의 경사가 아예
안 오게 만드는 편이 낫다.

## 4. `None`과 0으로 채운 텐서

```
역전파 뒤          : tensor([1.3227, 0.5338, 0.1234])
opt.zero_grad()    : None              <- 기본이 None
set_to_none=False  : tensor([0., 0., 0.])
w.grad = None      : None
w.grad.zero_()     : tensor([0., 0., 0.])
이 판의 기본값     : True (torch 2.5.1)
```

파이토치 2.0부터 기본값이 `set_to_none=True`다. 예전 판은 0으로 채웠으므로,
옛 코드를 그대로 돌리면 **동작이 달라진다.**

달라지는 것이 메모리만이 아니다.

```
set_to_none=True  : -0.021486  +0.282214  -0.422172
set_to_none=False : -0.036934  +0.257463  -0.374460
가장 큰 차이      : 0.047712
```

모멘텀을 켜고 경사가 빈 채로 한 걸음 더 걸으면 **0.0477만큼 다른 자리**에 간다.
`None`이면 `step()`이 그 파라미터를 건너뛰고, 0이면 "0만큼 움직여라"로 읽혀
모멘텀이 계속 민다.

## 5. 무엇을 쓰는가

**보통은 `opt.zero_grad()`**다. 걷는 것과 지우는 것이 한 짝이라 코드가
읽히고, 옵티마이저가 모델 전체를 들고 있으면 범위도 같다.

범위가 다른 경우에만 `model.zero_grad()`가 필요하다. 확인은 한 줄이면 된다.

```python
sum(len(g["params"]) for g in opt.param_groups) == len(list(net.parameters()))
```

`p.grad = None`을 손으로 쓰는 자리는 드물다. 특정 파라미터의 경사만 버리고
싶을 때 — 이를테면 「파라미터 얼리고 녹이기」에서 녹일 때 쌓인 것을 버릴
때 — 쯤이다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
3절의 "메모리를 차지한다"를 재어라. 옵티마이저가 안 든 층의 경사가 몇 KB를
쥐고 있는가?

</div>

??? success "풀이"
    ```python
    net = nn.Sequential(nn.Linear(256, 256), nn.Linear(256, 2))
    opt = torch.optim.SGD(net[1].parameters(), lr=0.1)
    x, y = torch.randn(32, 256), torch.randn(32, 2)

    def grad_bytes(m):
        return sum(p.grad.numel() * p.grad.element_size()
                   for p in m.parameters() if p.grad is not None)

    for i in range(3):
        opt.zero_grad()
        F.mse_loss(net(x), y).backward()
        print(f"  {i + 1}걸음 뒤 경사가 쓰는 메모리 {grad_bytes(net) / 1e3:.1f} KB")
    net.zero_grad()
    print("  model.zero_grad() 뒤:", grad_bytes(net) / 1e3, "KB")
    ```

    ```
      1걸음 뒤 경사가 쓰는 메모리 265.2 KB
      2걸음 뒤 경사가 쓰는 메모리 265.2 KB
      3걸음 뒤 경사가 쓰는 메모리 265.2 KB
      model.zero_grad() 뒤: 0.0 KB
    ```

    **메모리는 안 늘어난다.** 값만 쌓일 뿐 텐서는 하나이기 때문이다. 265 KB에서
    그대로 머문다.

    그러니 3절에서 말한 비용은 "걸음마다 늘어난다"가 아니라 **"안 쓰는 것을
    계속 들고 있다"**가 맞다. `Linear(256,256)` 하나가 265 KB이니, 사전학습된
    큰 모델에서 앞쪽을 전부 얼리면 모델 크기만큼을 쓸모없이 쥐고 있게 된다.

    진짜 문제는 값 쪽이다. 쌓인 경사로 녹이는 순간 첫 걸음이 그만큼 커진다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
`set_to_none`의 차이가 Adam에서는 어떻게 나타나는가? SGD 모멘텀과 같은 모양인가?

</div>

??? success "풀이"
    ```python
    for label, stn in (("None", True), ("0 채우기", False)):
        torch.manual_seed(0)
        mm = nn.Linear(4, 2)
        o = torch.optim.Adam(mm.parameters(), lr=0.1)
        F.mse_loss(mm(x), y).backward()
        o.step()
        o.zero_grad(set_to_none=stn)
        o.step(); o.step()              # 경사가 빈 채로 두 걸음
        print(f"  Adam {label:8s} weight {mm.weight.detach().flatten()[:2].tolist()}")
    ```

    ```
      Adam None     weight [-0.10374340415000916, 0.16822180151939392]
      Adam 0 채우기  weight [-0.22254490852355957, 0.049420297145843506]
    ```

    **차이가 SGD보다 크다.** 0.12쯤 벌어졌다.

    Adam은 1차 모멘트 $m$을 들고 있는데, 경사를 0으로 채우면
    $m \leftarrow 0.9m + 0.1 \times 0$으로 **줄어들기는 해도 0이 되지 않는다.**
    그 남은 $m$으로 계속 걷는다. 게다가 2차 모멘트 $v$도 같이 줄어드는데
    $m/\sqrt{v}$에서 분모가 더 빨리 줄면 **걸음이 오히려 커질** 수도 있다.

    `None`이면 그 파라미터를 통째로 건너뛰므로 $m$도 $v$도 그대로 멈춰 있다.

    그래서 기본값이 `True`로 바뀐 것이 Adam에서 더 중요하다. 걸음을 안 걷는
    것과 "경사가 0인 걸음을 걷는 것"은 상태를 가진 옵티마이저에서 전혀 다른
    일이다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
`zero_grad()`를 넣을 자리가 넷이다. `backward()` 앞, `backward()` 뒤, `step()`
뒤, 아예 안 하기. 다섯 걸음씩 돌려 로스를 견주어라.

</div>

??? success "풀이"
    ```python
    for where in ("backward 앞", "backward 뒤", "step 뒤", "아예 안 함"):
        torch.manual_seed(0)
        n2 = nn.Sequential(nn.Linear(4, 8), nn.Linear(8, 2))
        o = torch.optim.SGD(n2.parameters(), lr=0.1)
        for _ in range(5):
            if where == "backward 앞":
                o.zero_grad()
            F.mse_loss(n2(x), y).backward()
            if where == "backward 뒤":
                o.zero_grad()
            o.step()
            if where == "step 뒤":
                o.zero_grad()
        print(f"  {where:12s} 마지막 로스 {F.mse_loss(n2(x), y).item():.6f}")
    ```

    ```
      backward 앞   마지막 로스 0.676400
      backward 뒤   마지막 로스 1.254248
      step 뒤       마지막 로스 0.676400
      아예 안 함     마지막 로스 0.942723
    ```

    **넷이 세 가지 결과로 갈린다.**

    - **`backward` 앞**과 **`step` 뒤**는 같다(0.676400). 바퀴를 가로질러 보면
      같은 자리이기 때문이다. 둘 다 맞다
    - **`backward` 뒤**가 가장 나쁘다(1.254248). 방금 구한 경사를 지우므로
      `step()`이 아무것도 안 한다. **로스가 처음 값 그대로**다
    - **아예 안 함**은 그 중간이다(0.942723). 경사가 쌓여 걸음이 커지는데,
      이 보기에서는 터지지 않고 조금 지나쳤다

    넷 중 **둘이 맞고 둘이 틀리는데, 틀린 둘이 서로 다르게 틀린다.** 안 지우는
    쪽은 적어도 학습이 일어나므로 로스가 줄기는 한다. 잘못된 자리에 둔 쪽은
    아예 안 배운다.

    그래서 증상으로 구별할 수 있다. **로스가 완전히 평평하면 지우는 자리를,
    로스가 요동치면 안 지웠는지를** 보면 된다.
