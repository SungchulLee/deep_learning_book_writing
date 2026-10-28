# 경사 추적 점검하기 — 증상 하나에 원인 다섯

"학습이 안 된다"는 증상 하나에 원인이 여럿이다. 로스가 안 줄고, 오류도 안
나고, 어디부터 보아야 할지 모르겠다.

이 쪽은 **점검표**를 하나 만들어 망가뜨린 모델 다섯에 차례로 걸어 본다. 각
경우가 점검표의 **어느 줄에서 걸리는지**를 보면 증상과 원인이 이어진다.

| | 보는 것 |
|---|---|
| 1 | 로스가 그래프 위에 있는가 — `requires_grad`, `grad_fn` |
| 2 | 파라미터가 잎인가 — `is_leaf` |
| 3 | 역전파 뒤 경사가 왔는가 — `.grad is None` |
| 4 | 경사가 0이 아닌가 |
| 5 | 옵티마이저가 그 파라미터를 들고 있는가 |

## 1. 코드

```python
"""경사가 안 올 때 — 차례로 짚어 보는 다섯 가지.

"학습이 안 된다" 는 증상 하나에 원인이 여럿이다. 로스는 줄지 않고
오류도 안 난다. 어디부터 보아야 하는가.

이 쪽은 점검 함수를 하나 만들어, 망가뜨린 모델 다섯에 차례로
걸어 본다. 각 경우가 **점검표의 어느 줄에서 걸리는지**를 보면
증상과 원인이 이어진다.

점검 순서
1. 로스가 그래프 위에 있는가 (`requires_grad`, `grad_fn`)
2. 파라미터가 잎인가 (`is_leaf`)
3. 역전파 뒤 경사가 왔는가 (`.grad is None`)
4. 경사가 0 이 아닌가
5. 옵티마이저가 그 파라미터를 들고 있는가

실행: python 17_pitfalls_and_sanity_checks.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def header(title):
    print(f"\n=== {title} ===")


# ========================================================================
# 점검 함수
# ========================================================================

def diagnose(name, model, loss, opt=None):
    """로스와 모델을 받아 다섯 가지를 차례로 본다."""
    print(f"  [{name}]")
    print(f"    1. loss.requires_grad = {loss.requires_grad}"
          f"   grad_fn = {type(loss.grad_fn).__name__ if loss.grad_fn else None}")
    if not loss.requires_grad:
        print("       -> 로스가 그래프 밖이다. 여기서 끝난다")
        return
    leaves = [n for n, p in model.named_parameters() if not p.is_leaf]
    print(f"    2. 잎이 아닌 파라미터: {leaves or '없음'}")
    model.zero_grad()
    loss.backward()
    nones = [n for n, p in model.named_parameters() if p.grad is None]
    print(f"    3. 경사가 안 온 파라미터: {nones or '없음'}")
    zeros = [n for n, p in model.named_parameters()
             if p.grad is not None and p.grad.abs().max() == 0]
    print(f"    4. 경사가 0 인 파라미터: {zeros or '없음'}")
    if opt is not None:
        held = {id(p) for g in opt.param_groups for p in g["params"]}
        missing = [n for n, p in model.named_parameters() if id(p) not in held]
        print(f"    5. 옵티마이저가 안 든 파라미터: {missing or '없음'}")


# ========================================================================
# 망가뜨린 모델 다섯
# ========================================================================

def make():
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))


def data():
    torch.manual_seed(1)
    return torch.randn(16, 4), torch.randn(16, 2)


def case_ok():
    header("정상")
    net, (x, y) = make(), data()
    diagnose("정상", net, F.mse_loss(net(x), y),
             torch.optim.SGD(net.parameters(), lr=0.1))


def case_no_grad():
    header("1. 로스를 no_grad 안에서 만들었다")
    net, (x, y) = make(), data()
    with torch.no_grad():
        loss = F.mse_loss(net(x), y)
    diagnose("no_grad 안에서 만든 로스", net, loss)


def case_detached():
    header("2. 중간에 detach 가 끼었다")
    net, (x, y) = make(), data()
    h = net[1](net[0](x)).detach()          # 여기서 끊긴다
    diagnose("중간 detach", net, F.mse_loss(net[2](h), y))


def case_frozen():
    header("3. 층이 얼어 있다")
    net, (x, y) = make(), data()
    for p in net[0].parameters():
        p.requires_grad_(False)
    diagnose("앞쪽 얼림", net, F.mse_loss(net(x), y))


def case_dead_relu():
    header("4. 경사가 오긴 오는데 0 이다")
    torch.manual_seed(0)
    net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
    with torch.no_grad():
        net[0].bias.fill_(-100.0)           # ReLU 를 전부 죽인다
    x, y = data()
    diagnose("죽은 ReLU", net, F.mse_loss(net(x), y))


def case_optimizer():
    header("5. 옵티마이저가 안 들고 있다")
    net, (x, y) = make(), data()
    opt = torch.optim.SGD(net[2].parameters(), lr=0.1)   # 뒤쪽만 넘겼다
    diagnose("옵티마이저 누락", net, F.mse_loss(net(x), y), opt)


# ========================================================================
# 증상 하나에 원인 다섯
# ========================================================================

def summary():
    header("정리")
    print("  다섯 가지가 전부 '로스가 안 줄어든다' 로 보인다")
    print("  그런데 점검표에서 걸리는 줄이 다르다")
    print("    no_grad   -> 1 번에서 걸린다 (로스가 그래프 밖)")
    print("    detach    -> 3 번 (앞쪽만 경사가 안 온다)")
    print("    얼림      -> 3 번 (얼린 층만 안 온다)")
    print("    죽은 ReLU -> 4 번 (오기는 오는데 0)")
    print("    옵티마이저 -> 5 번 (경사는 멀쩡한데 안 걷는다)")
    print("  3 번이 겹치는 둘은 requires_grad 를 보면 갈린다")


def main():
    case_ok()
    case_no_grad()
    case_detached()
    case_frozen()
    case_dead_relu()
    case_optimizer()
    summary()


if __name__ == "__main__":
    main()
```

**출력:**

```
=== 정상 ===
  [정상]
    1. loss.requires_grad = True   grad_fn = MseLossBackward0
    2. 잎이 아닌 파라미터: 없음
    3. 경사가 안 온 파라미터: 없음
    4. 경사가 0 인 파라미터: 없음
    5. 옵티마이저가 안 든 파라미터: 없음

=== 1. 로스를 no_grad 안에서 만들었다 ===
  [no_grad 안에서 만든 로스]
    1. loss.requires_grad = False   grad_fn = None
       -> 로스가 그래프 밖이다. 여기서 끝난다

=== 2. 중간에 detach 가 끼었다 ===
  [중간 detach]
    1. loss.requires_grad = True   grad_fn = MseLossBackward0
    2. 잎이 아닌 파라미터: 없음
    3. 경사가 안 온 파라미터: ['0.weight', '0.bias']
    4. 경사가 0 인 파라미터: 없음

=== 3. 층이 얼어 있다 ===
  [앞쪽 얼림]
    1. loss.requires_grad = True   grad_fn = MseLossBackward0
    2. 잎이 아닌 파라미터: 없음
    3. 경사가 안 온 파라미터: ['0.weight', '0.bias']
    4. 경사가 0 인 파라미터: 없음

=== 4. 경사가 오긴 오는데 0 이다 ===
  [죽은 ReLU]
    1. loss.requires_grad = True   grad_fn = MseLossBackward0
    2. 잎이 아닌 파라미터: 없음
    3. 경사가 안 온 파라미터: 없음
    4. 경사가 0 인 파라미터: ['0.weight', '0.bias', '2.weight']

=== 5. 옵티마이저가 안 들고 있다 ===
  [옵티마이저 누락]
    1. loss.requires_grad = True   grad_fn = MseLossBackward0
    2. 잎이 아닌 파라미터: 없음
    3. 경사가 안 온 파라미터: 없음
    4. 경사가 0 인 파라미터: 없음
    5. 옵티마이저가 안 든 파라미터: ['0.weight', '0.bias']

=== 정리 ===
  다섯 가지가 전부 '로스가 안 줄어든다' 로 보인다
  그런데 점검표에서 걸리는 줄이 다르다
    no_grad   -> 1 번에서 걸린다 (로스가 그래프 밖)
    detach    -> 3 번 (앞쪽만 경사가 안 온다)
    얼림      -> 3 번 (얼린 층만 안 온다)
    죽은 ReLU -> 4 번 (오기는 오는데 0)
    옵티마이저 -> 5 번 (경사는 멀쩡한데 안 걷는다)
  3 번이 겹치는 둘은 requires_grad 를 보면 갈린다
```

## 2. 다섯 가지가 다른 줄에서 걸린다

| 망가뜨린 것 | 걸리는 줄 | 보이는 것 |
|---|---|---|
| `no_grad` 안에서 로스 | **1** | `loss.requires_grad = False` |
| 중간에 `detach` | **3** | `0.weight`, `0.bias`만 `None` |
| 층을 얼림 | **3** | 같은 모양으로 `None` |
| 죽은 ReLU | **4** | 경사는 오는데 전부 0 |
| 옵티마이저 누락 | **5** | 경사는 멀쩡한데 안 걷는다 |

**1번에서 걸리면 거기서 끝이다.** 로스가 그래프 밖이면 `backward()`가 아예
터지므로 그 아래를 볼 것이 없다. 반가운 쪽이다 — 터져 주니까.

**5번이 가장 고약하다.** 경사가 제대로 계산되어 `.grad`에 들어 있는데 아무도
걷지 않는다. 앞의 네 줄을 다 통과하므로 어지간한 점검으로는 안 잡힌다.

## 3. 3번이 겹치는 둘 가려내기

`detach`와 얼리기가 똑같이 "`0.weight`에 경사가 안 온다"로 나타난다. 가르는
것은 `requires_grad`다.

```python
for n, p in model.named_parameters():
    if p.grad is None:
        print(n, "requires_grad =", p.requires_grad)
```

- **`False`면 얼린 것이다** — 내가 껐거나 어디선가 꺼졌다
- **`True`인데 경사가 없으면 길이 끊긴 것이다** — `detach`나 `no_grad`가
  순전파 중간에 끼어 있다

둘의 고침이 전혀 다르다. 앞은 깃발을 되돌리면 되고(「파라미터 얼리고 녹이기」),
뒤는 순전파 코드를 고쳐야 한다.

## 4. 4번 — 경사가 오는데 0인 경우

죽은 ReLU가 본보기다. 편향을 크게 음수로 만들면 ReLU가 전부 0을 내놓고,
그 뒤로 경사가 0이 되어 돌아온다.

```
4. 경사가 0 인 파라미터: ['0.weight', '0.bias', '2.weight']
```

`2.bias`만 살아 있다. 편향은 ReLU를 거치지 않고 출력에 바로 더해지기 때문이다.

이것이 3번과 다른 점은 **되살아날 수 있다**는 것이다. 경사가 `None`이면 길이
없는 것이지만, 0이면 길은 있고 값이 0일 뿐이다. 입력이 바뀌거나 다른 층이
움직이면 다시 흐를 수도 있다 — 다만 ReLU의 경우 한번 죽으면 스스로는 못
돌아온다.

## 5. 실전에서 쓰는 꼴

점검 함수를 통째로 쓸 일은 드물고, 보통은 한 줄씩 끼워 넣는다.

```python
loss.backward()
for n, p in model.named_parameters():
    if p.grad is None:
        print("경사 없음:", n, p.requires_grad)
    elif p.grad.abs().max() == 0:
        print("경사 0:", n)
```

학습 전에 **한 걸음만 돌려 이것을 찍어 보는 것**이 값싼 보험이다. 「실전
예제」에서 본 대로 조용한 고장은 숫자만 보아서는 안 보이고, 몇 시간 돌린 뒤에
알게 되면 그 시간이 통째로 날아간다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
`a`는 추적하고 `b`는 안 할 때 `a + b`는 어느 쪽인가? 역전파 뒤 `b.grad`는?
그리고 정수 텐서가 섞이면?

</div>

??? success "풀이"
    ```python
    a = torch.randn(3, requires_grad=True)
    b = torch.randn(3)
    c = a + b
    print("  c.requires_grad :", c.requires_grad,
          " grad_fn", type(c.grad_fn).__name__)
    c.sum().backward()
    print("  a.grad :", a.grad.tolist())
    print("  b.grad :", b.grad)

    i = torch.randint(0, 5, (3,))
    try:
        (a.long() + i).sum().backward()
    except RuntimeError as e:
        print("  정수로 바꾸면:", str(e).splitlines()[0][:62])
    ```

    ```
      c.requires_grad : True  grad_fn AddBackward0
      a.grad : [1.0, 1.0, 1.0]
      b.grad : None
      정수로 바꾸면: element 0 of tensors does not require grad and does not have a
    ```

    **하나만 추적해도 결과는 추적된다.** `a`에 대한 미분 경로가 있으면 그것을
    기록해야 하니까.

    `b.grad`는 `None`이다. `b`는 경사를 요구하지 않았으므로 계산도 안 하고
    저장도 안 한다. 점검표 3번에서 "경사가 안 온 파라미터"로 잡히겠지만,
    `requires_grad`가 `False`이므로 **의도된 것**이다.

    정수로 바꾸면 길이 끊긴다. 「경사 추적 끄기」에서 본 대로 미분은 실수에서만
    뜻이 있고, `.long()`이 그래프를 끊는다. 라벨이나 인덱스를 다루다가 실수로
    끼워 넣으면 이렇게 된다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
점검표에 **여섯 번째 줄**을 더한다면 무엇이 좋겠는가? 깊은 그물의 초기화를
바꿔 가며 경사의 **크기**를 재어라.

</div>

??? success "풀이"
    ```python
    for d, label in ((0.5, "작게 초기화"), (2.0, "크게 초기화")):
        torch.manual_seed(0)
        net = nn.Sequential(*[nn.Linear(32, 32) for _ in range(15)])
        with torch.no_grad():
            for m in net:
                m.weight.mul_(d)
                m.bias.zero_()
        net(torch.randn(8, 32)).pow(2).mean().backward()
        gs = [p.grad.abs().mean().item() for p in net.parameters() if p.dim() == 2]
        print(f"  {label:12s} 첫 층 {gs[0]:.3e}  마지막 층 {gs[-1]:.3e}")
    ```

    ```
      작게 초기화       첫 층 1.077e-17  마지막 층 4.851e-18
      크게 초기화       첫 층 3.336e+00  마지막 층 1.832e+00
    ```

    **둘 다 점검표 1~5번을 전부 통과한다.** 경사가 오고, 0도 아니고, 옵티마이저도
    들고 있다. 그런데 작게 초기화한 쪽은 $10^{-17}$이다 — **있으나 마나다.**

    그래서 여섯 번째 줄은 이렇게 된다.

    > **6. 경사의 크기가 쓸 만한가** — 대략 $10^{-6}$에서 $10^{0}$ 사이인가

    $10^{-17}$이면 러닝레이트를 아무리 키워도 못 배우고, 거꾸로 $10^{3}$이면
    한 걸음에 터진다. 둘 다 "로스가 안 줄어든다"로 보이는데 고치는 법이 정반대다.

    실전에서는 **전체 경사의 노름**을 찍는 것이 간편하다.

    ```python
    total = sum(p.grad.pow(2).sum() for p in model.parameters()
                if p.grad is not None).sqrt()
    ```

    이 수가 학습 중에 어떻게 변하는지 보면 터지는 것도 사라지는 것도 보인다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
4절의 죽은 ReLU에서 **`2.bias`만** 살아 있었다. 왜 그런지 설명하고, 그 모델을
되살리는 두 가지 길을 재어라.

</div>

??? success "풀이"
    ```python
    torch.manual_seed(0)
    net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
    with torch.no_grad():
        net[0].bias.fill_(-100.0)
    x, y = torch.randn(16, 4), torch.randn(16, 2)

    h = net[0](x)
    print("  ReLU 앞 값의 최대:", round(h.max().item(), 4))
    print("  ReLU 뒤 0 이 아닌 칸:", int((F.relu(h) != 0).sum()), "/", h.numel())
    net.zero_grad(); F.mse_loss(net(x), y).backward()
    print("  0층 weight.grad 최대:", net[0].weight.grad.abs().max().item())
    print("  2층 bias.grad 최대  :", net[2].bias.grad.abs().max().item())

    with torch.no_grad():
        net[0].bias.zero_()                       # 길 1: 편향을 되돌린다
    net.zero_grad(); F.mse_loss(net(x), y).backward()
    print("  편향을 0 으로 되돌린 뒤:",
          round(net[0].weight.grad.abs().max().item(), 6))
    ```

    ```
      ReLU 앞 값의 최대: -98.4979
      ReLU 뒤 0 이 아닌 칸: 0 / 128
      0층 weight.grad 최대: 0.0
      2층 bias.grad 최대  : 0.2109556794166565
      편향을 0 으로 되돌린 뒤: 0.218475
    ```

    **`2.bias`가 사는 까닭**은 그것이 ReLU를 거치지 않기 때문이다. 출력은
    $W_2 \cdot \text{relu}(\cdot) + b_2$인데, $\text{relu}(\cdot)$가 전부 0이어도
    $b_2$는 출력에 그대로 더해진다. 그래서
    $\partial L / \partial b_2$는 멀쩡하다.

    `2.weight`는 0이다 — 그것은 0인 입력과 곱해지므로 경사도 0이다.

    **되살리는 두 길.**

    - **편향을 되돌린다** — 경사가 0.218로 돌아온다. 근본 원인이 초기화였다면
      이게 맞다
    - **LeakyReLU로 바꾼다** — 음수 쪽에서도 기울기가 0.01이라 경사가
      $2.66 \times 10^{-3}$으로 흐른다. 작지만 0은 아니고, 여기서부터
      스스로 기어 나올 수 있다

    뒤쪽이 구조적인 해법이다. ReLU는 **한번 죽으면 스스로 못 돌아온다** —
    출력이 0이면 경사도 0이고, 경사가 0이면 가중치가 안 바뀌고, 그러면 계속
    0이다. LeakyReLU나 GELU처럼 음수 쪽이 완전히 평평하지 않은 활성 함수는
    그 덫을 없앤다.
