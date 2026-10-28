# 간단한 뉴럴 넷트웍 — 텐서에서 학습 한 걸음까지

앞의 쪽들이 텐서와 자동 미분을 따로따로 보았다. 이 쪽은 그 둘을 이어 **학습 한 걸음**을 끝까지 따라간다 — 레이어를 쌓고, 파라메터를 세고, 로스를 재고, 그래디언트를 구하고, 웨이트를 옮긴다.

마지막 두 절은 처음 쓰는 사람이 거의 반드시 헷갈리는 자리다. `model.eval()`이 무엇을 하고 무엇을 하지 않는지, 그리고 클래스 속성에 넣은 텐서가 왜 학습되지 않는지다.

## 1. 코드

```python
"""튜토리얼 21: 단순한 뉴럴 넷트웍 — 텐서에서 학습 한 걸음까지."""
import torch
import torch.nn as nn

torch.manual_seed(0)


def header(title):
    print(f"\n{'='*70}\n{title}\n{'='*70}")


class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 10)
        self.fc3 = nn.Linear(10, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


def main():
    header("1. 레이어 하나의 모양")
    layer = nn.Linear(3, 2)
    print(f"{layer}")
    print(f"weight {tuple(layer.weight.shape)}   bias {tuple(layer.bias.shape)}")
    print(f"(3,) 하나를 넣으면 -> {tuple(layer(torch.randn(1, 3)).shape)}")

    header("2. 파라메터를 레이어별로 센다")
    model = SimpleNet()
    total = 0
    for name, p in model.named_parameters():
        total += p.numel()
        print(f"  {name:12s} {str(tuple(p.shape)):10s} {p.numel():>5d}")
    print(f"  {'합':12s} {'':10s} {total:>5d}")

    header("3. 로스와 그래디언트")
    pred = torch.tensor([[2.5], [3.0], [1.5]], requires_grad=True)
    targ = torch.tensor([[2.0], [3.0], [2.0]])
    loss = nn.MSELoss()(pred, targ)
    loss.backward()
    print(f"예측 - 목표 = {(pred - targ).detach().flatten().tolist()}")
    print(f"MSE        = {loss.item():.4f}   (제곱의 평균)")
    print(f"d로스/d예측 = {pred.grad.flatten().tolist()}")
    print(f"            = 2(예측 - 목표)/n, n = 3")

    header("4. 한 걸음이 실제로 무엇을 바꾸는가")
    model = SimpleNet()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    X, y = torch.randn(32, 10), torch.randn(32, 1)
    w_before = model.fc3.weight.detach().clone()
    opt.zero_grad()
    nn.MSELoss()(model(X), y).backward()
    g = model.fc3.weight.grad.detach().clone()
    opt.step()
    w_after = model.fc3.weight.detach()
    print(f"fc3.weight 첫 세 개  이전 {w_before[0, :3].tolist()}")
    print(f"                     이후 {w_after[0, :3].tolist()}")
    print(f"바뀐 양               {(w_after - w_before)[0, :3].tolist()}")
    print(f"-lr * grad            {(-0.1 * g)[0, :3].tolist()}")
    print(f"둘이 같은가? {torch.allclose(w_after - w_before, -0.1 * g)}")

    header("5. eval() 과 no_grad() 는 다른 일을 한다")
    model = SimpleNet()
    x = torch.randn(5, 10)

    model.eval()
    out = model(x)
    print(f"eval() 뒤       requires_grad = {out.requires_grad}, grad_fn = {type(out.grad_fn).__name__}")
    with torch.no_grad():
        out2 = model(x)
    print(f"no_grad() 안    requires_grad = {out2.requires_grad}, grad_fn = {out2.grad_fn}")
    print("-> 그래디언트를 끄는 것은 no_grad() 다. eval() 은 끄지 않는다.")

    # 그러면 eval() 은 무엇을 바꾸는가 — 이 모델에서는 아무것도 바꾸지 않는다
    model.train(); a = model(x)
    model.eval();  b = model(x)
    print(f"\n이 모델에서 train 과 eval 의 출력이 같은가? {torch.equal(a, b)}")

    class WithDropout(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(10, 1)
            self.drop = nn.Dropout(0.5)

        def forward(self, x):
            return self.drop(self.fc(x))

    d = WithDropout()
    d.train(); a = d(x)
    d.eval();  b = d(x)
    print(f"드롭아웃을 넣으면 같은가?                 {torch.equal(a, b)}")
    print("-> eval() 이 바꾸는 것은 드롭아웃·배치 정규화처럼 모드를 타는 레이어뿐이다.")

    header("6. 속성으로 둔 텐서는 파라메터가 아니다")
    class Two(nn.Module):
        def __init__(self):
            super().__init__()
            self.plain = torch.ones(3)                 # 그냥 속성
            self.param = nn.Parameter(torch.ones(3))   # 등록된 파라메터

    t = Two()
    print(f"parameters() 가 세는 수 : {sum(p.numel() for p in t.parameters())}")
    print(f"state_dict() 의 열쇠     : {list(t.state_dict().keys())}")
    print(f"plain.requires_grad      : {t.plain.requires_grad}")
    print("-> 클래스 속성에 넣었다고 학습되지 않는다. 옵티마이저도 저장도 지나친다.")


if __name__ == "__main__":
    main()
```

**출력:**

```
======================================================================
1. 레이어 하나의 모양
======================================================================
Linear(in_features=3, out_features=2, bias=True)
weight (2, 3)   bias (2,)
(3,) 하나를 넣으면 -> (1, 2)

======================================================================
2. 파라메터를 레이어별로 센다
======================================================================
  fc1.weight   (20, 10)     200
  fc1.bias     (20,)         20
  fc2.weight   (10, 20)     200
  fc2.bias     (10,)         10
  fc3.weight   (1, 10)       10
  fc3.bias     (1,)           1
  합                         441

======================================================================
3. 로스와 그래디언트
======================================================================
예측 - 목표 = [0.5, 0.0, -0.5]
MSE        = 0.1667   (제곱의 평균)
d로스/d예측 = [0.3333333432674408, 0.0, -0.3333333432674408]
            = 2(예측 - 목표)/n, n = 3

======================================================================
4. 한 걸음이 실제로 무엇을 바꾸는가
======================================================================
fc3.weight 첫 세 개  이전 [-0.25123631954193115, 0.033209990710020065, 0.2943178117275238]
                     이후 [-0.23099485039710999, 0.0324883833527565, 0.29684343934059143]
바뀐 양               [0.020241469144821167, -0.0007216073572635651, 0.002525627613067627]
-lr * grad            [0.020241469144821167, -0.0007216068333946168, 0.002525618299841881]
둘이 같은가? True

======================================================================
5. eval() 과 no_grad() 는 다른 일을 한다
======================================================================
eval() 뒤       requires_grad = True, grad_fn = AddmmBackward0
no_grad() 안    requires_grad = False, grad_fn = None
-> 그래디언트를 끄는 것은 no_grad() 다. eval() 은 끄지 않는다.

이 모델에서 train 과 eval 의 출력이 같은가? True
드롭아웃을 넣으면 같은가?                 False
-> eval() 이 바꾸는 것은 드롭아웃·배치 정규화처럼 모드를 타는 레이어뿐이다.

======================================================================
6. 속성으로 둔 텐서는 파라메터가 아니다
======================================================================
parameters() 가 세는 수 : 3
state_dict() 의 열쇠     : ['param']
plain.requires_grad      : False
-> 클래스 속성에 넣었다고 학습되지 않는다. 옵티마이저도 저장도 지나친다.
```

## 2. 파라메터는 모양에서 바로 나온다

`nn.Linear(in, out)`은 웨이트 $(out, in)$과 바이어스 $(out,)$를 쥔다. 수식으로는 $y = xA^{\top} + b$이므로 행과 열이 뒤집혀 보이는데, 담긴 수는 $in \times out + out$개다.

출력의 2절이 그것을 레이어별로 센다. `fc1`과 `fc2`가 웨이트 200개로 같은 것이 눈에 띈다 — $10 \times 20$과 $20 \times 10$이니 당연하지만, **파라메터 수는 방향을 모른다**는 것을 보여 주는 자리이기도 하다. 바이어스는 출력 쪽 크기를 따르므로 20과 10으로 갈린다.

## 3. 한 걸음은 뺄셈 하나다

4절이 재는 것이 학습의 전부다. 옵티마이저가 `step()`에서 하는 일은

$$
w \leftarrow w - \eta \cdot \frac{\partial L}{\partial w}
$$

이고, 출력이 그것을 그대로 보인다 — 바뀐 양과 $-\eta \cdot \text{grad}$가 같다(SGD에 모멘텀을 두지 않았을 때다). 찍힌 수의 끝자리가 조금 어긋나는 것은 `float32`의 셈 차례 탓이고, `allclose`가 `True`다.

## 4. `eval()`은 그래디언트를 끄지 않는다

이것이 이 쪽에서 가장 자주 틀리는 자리다. 5절을 보라.

| | `requires_grad` | 드롭아웃이 도는가 |
|---|---|---|
| `model.train()` | True | 돈다 |
| `model.eval()` | **True** | 멈춘다 |
| `with torch.no_grad():` | **False** | (모드와 무관) |

**둘은 서로 다른 스위치다.**

- `eval()`은 **모드를 타는 레이어**에게 알린다. 드롭아웃은 끄고, 배치 정규화는 배치 통계 대신 쌓아 둔 값을 쓰게 한다. 그뿐이다.
- `no_grad()`는 **그래프를 쌓지 않게** 한다. 메모리를 아끼고 속도를 올린다.

출력이 둘을 갈라 보인다. `eval()` 뒤에도 `requires_grad`가 `True`이고 `grad_fn`이 붙어 있다. 그리고 이 모델에는 드롭아웃도 배치 정규화도 없으므로 **`eval()`이 바꾸는 것이 하나도 없다** — `train()`과 `eval()`의 출력이 똑같다. 드롭아웃을 하나 넣으면 비로소 갈린다.

!!! warning "둘을 한 덩어리로 외우기 쉽다"
    추론할 때는 거의 언제나 `model.eval()`과 `with torch.no_grad():`를 **나란히** 쓴다. 그래서 "평가 모드로 두면 그래디언트가 꺼진다"고 외워도 한동안 탈이 나지 않는다. 둘 다 켜 두었으니 결과는 맞다.

    탈은 하나만 쓰는 날 난다. 드롭아웃이 있는 모델을 `no_grad()`만으로 평가하면 드롭아웃이 **여전히 돌아** 같은 입력에 매번 다른 점수가 나오고, 그래디언트가 필요한 자리에서 `eval()`만 믿으면 그래프가 그대로 쌓인다. 어느 쪽도 오류를 내지 않는다.

    이 모델이 특히 속이기 쉬운 까닭이 하나 더 있다. 모드를 타는 레이어가 하나도 없어 **`eval()`이 바꾸는 것이 아무것도 없다** — 위 출력이 `train`과 `eval`의 결과가 같다고 적는 그대로다. 아무 일도 하지 않는 호출은 무슨 일을 한다고 믿어도 반증되지 않는다.

## 5. 속성에 넣었다고 파라메터가 아니다

6절이 그것을 보인다. `self.plain = torch.ones(3)`은 그냥 속성이다 — `parameters()`가 세지 않고, `state_dict()`에 들어가지 않고, `requires_grad`도 `False`다. 옵티마이저에게 넘어가지 않으니 **학습되지 않고**, 저장에도 빠지고, `.to(device)`도 비껴간다.

`nn.Parameter`로 감싸면 셋 다 해결된다. 학습하지는 않되 모델과 함께 다녀야 하는 값이라면 `register_buffer`를 쓴다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
출력의 2절은 `SimpleNet`의 파라메터가 441개라고 한다. 레이어별로 손으로 세어 그 수가 나오는지 확인하라. `fc1`과 `fc2`의 **웨이트가 둘 다 200개**인데 **바이어스는 20개와 10개로 다른** 까닭은 무엇인가?

</div>

??? success "풀이"
    `nn.Linear(in, out)`은 $in \times out$개의 웨이트와 $out$개의 바이어스를 쥔다.

    | | 웨이트 | 바이어스 | 합 |
    |---|---|---|---|
    | `fc1` (10 → 20) | $10 \times 20 = 200$ | 20 | 220 |
    | `fc2` (20 → 10) | $20 \times 10 = 200$ | 10 | 210 |
    | `fc3` (10 → 1) | $10 \times 1 = 10$ | 1 | 11 |
    | | | | **441** |

    웨이트가 같은 까닭은 곱셈이 **순서를 가리지 않기** 때문이다 — $10 \times 20$과 $20 \times 10$은 같은 수다. 바이어스는 다르다. 바이어스는 **나오는 쪽마다 하나씩** 붙으므로 출력 크기를 그대로 따르고, 그래서 20과 10으로 갈린다.

    곧 파라메터 수만 보아서는 레이어가 넓히는 중인지 좁히는 중인지 알 수 없고, **바이어스를 보면 알 수 있다.**

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
드롭아웃이 있는 모델을 네 가지로 돌려 보라 — `train()`/`eval()` × 쓰거나 안 쓰거나 `no_grad()`. 각 칸에서 **(1) 출력이 매번 달라지는가 (2) `requires_grad`가 무엇인가**를 재어 표를 채워라.

</div>

??? success "풀이"
    ```python
    import torch, torch.nn as nn
    torch.manual_seed(0)

    m = nn.Sequential(nn.Linear(4, 4), nn.Dropout(0.5))
    x = torch.ones(1, 4)

    for mode in ("train", "eval"):
        getattr(m, mode)()
        a, b = m(x), m(x)
        print(f"{mode:5s} 그냥      : 두 번이 같은가 {torch.equal(a, b)}, "
              f"requires_grad {a.requires_grad}")
        with torch.no_grad():
            a, b = m(x), m(x)
            print(f"{mode:5s} no_grad   : 두 번이 같은가 {torch.equal(a, b)}, "
                  f"requires_grad {a.requires_grad}")
    ```

    ```
    train 그냥      : 두 번이 같은가 False, requires_grad True
    train no_grad   : 두 번이 같은가 False, requires_grad False
    eval  그냥      : 두 번이 같은가 True, requires_grad True
    eval  no_grad   : 두 번이 같은가 True, requires_grad False
    ```

    **두 스위치가 서로 섞이지 않는다.** 세로로 읽으면 `no_grad()`만 `requires_grad`를 바꾸고, 가로로 읽으면 `eval()`만 드롭아웃을 멈춘다.

    위험한 칸은 **`train` + `no_grad`**다. 그래디언트를 끈 채로 평가하는 듯하지만 드롭아웃이 여전히 돌아, 같은 입력에 매번 다른 답이 나온다. 평가 점수가 까닭 없이 흔들린다면 가장 먼저 볼 자리다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
학습하지는 않되 모델과 함께 다녀야 하는 값이 있다고 하자(이를테면 정규화 상수). 속성으로 두는 것, `nn.Parameter`로 감싸는 것, `register_buffer`로 등록하는 것 **셋이 어떻게 다른지** 네 가지로 재어라 — `parameters()`에 잡히는가, `state_dict()`에 들어가는가, `requires_grad`가 무엇인가, `.to(device)`를 따라가는가.

</div>

??? success "풀이"
    ```python
    import torch, torch.nn as nn

    class Three(nn.Module):
        def __init__(self):
            super().__init__()
            self.plain = torch.ones(3)                  # 그냥 속성
            self.param = nn.Parameter(torch.ones(3))    # 학습한다
            self.register_buffer("buf", torch.ones(3))  # 함께 다니되 학습하지 않는다

    m = Three()
    print("parameters():", [n for n, _ in m.named_parameters()])
    print("buffers()   :", [n for n, _ in m.named_buffers()])
    print("state_dict():", list(m.state_dict().keys()))

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    m.to(dev)
    for n, v in [("plain", m.plain), ("param", m.param), ("buf", m.buf)]:
        print(f"  {n:6s} {v.device}")
    ```

    ```
    parameters(): ['param']
    buffers()   : ['buf']
    state_dict(): ['param', 'buf']
      plain  cpu
      param  mps:0
      buf    mps:0
    ```

    | | `parameters()` | `state_dict()` | `requires_grad` | `.to(device)` |
    |---|---|---|---|---|
    | 그냥 속성 | ✗ | ✗ | False | **따라가지 않는다** |
    | `nn.Parameter` | ✓ | ✓ | True | 따라간다 |
    | `register_buffer` | ✗ | ✓ | False | 따라간다 |

    정규화 상수라면 **`register_buffer`가 답이다.** 학습되지 않으면서 저장에 들어가고 장치를 따라간다.

    그냥 속성이 가장 고약하다. 넷 가운데 셋을 빠뜨리는데 **오류는 나지 않는다.** CPU에서 짠 코드가 GPU에서 "expected all tensors to be on the same device"로 터지거나, 저장했다 불러온 모델이 그 값만 초깃값으로 돌아가 있거나 한다. 배치 정규화가 `running_mean`을 버퍼로 두는 까닭이 이것이다.

## 정리하며

학습 한 걸음은 네 줄이다 — 지우고(`zero_grad`), 재고(`forward`, 로스), 미분하고(`backward`), 옮긴다(`step`). 4절이 그 마지막 줄이 정말 $w - \eta \cdot \partial L/\partial w$인지를 수로 확인했다.

그 네 줄보다 자주 탈이 나는 것은 **스위치와 등록**이다. `eval()`과 `no_grad()`는 서로 다른 일을 하고 추론할 때는 둘 다 필요하며, 클래스 속성에 넣은 텐서는 옵티마이저도 저장도 장치 이동도 지나친다. 둘 다 **오류 없이** 틀리는 쪽이라 알고 찾지 않으면 드러나지 않는다.
