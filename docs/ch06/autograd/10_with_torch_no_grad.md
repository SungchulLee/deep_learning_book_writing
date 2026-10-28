# `torch.no_grad` — 끄는 것은 역전파가 아니라 기록이다

`with torch.no_grad():` 안에서는 새로 만드는 텐서에 `grad_fn`이 붙지 않는다.
그래프가 안 세워지니 역전파가 쓰려고 저장해 둘 중간값도 없고, 그래서 추론에서
메모리를 아낀다.

여기까지는 널리 알려져 있는데, 그 위에 얹히는 짐작 둘이 어긋난다.

- **"`no_grad` 안에서는 `backward()`를 못 부른다"** — 부를 수 있다. 그래프를
  블록 **밖에서** 세웠다면 멀쩡히 돈다
- **"`no_grad`가 텐서의 `requires_grad`를 끈다"** — 안 끈다. 블록 안에서
  **만드는** 것에만 영향을 준다

## 1. 코드

```python
"""torch.no_grad — 끄는 것은 역전파가 아니라 기록이다.

`with torch.no_grad():` 안에서는 새로 만드는 텐서에 `grad_fn` 이
붙지 않는다. 그래서 그래프가 안 세워지고, 역전파가 쓰려고 저장해 둘
중간값도 없다.

자주 어긋나는 짐작이 둘 있다.

- "no_grad 안에서는 backward 를 못 부른다" — 부를 수 있다. 그래프를
  **블록 밖에서 세웠다면** 멀쩡히 돈다
- "no_grad 가 텐서의 requires_grad 를 끈다" — 안 끈다. 블록 안에서
  **만드는** 것에만 영향을 준다

실행: 이 쪽의 코드를 파일로 갈무리해 돌린다
"""
import torch
import torch.nn as nn


def header(title):
    print(f"\n=== {title} ===")


# ========================================================================
# 1. 블록 안에서 만든 것만 달라진다
# ========================================================================

def scope():
    header("1. 범위는 코드 블록이다")
    w = torch.randn(3, requires_grad=True)
    print("  들어가기 전 w.requires_grad :", w.requires_grad)
    with torch.no_grad():
        print("  블록 안 w.requires_grad     :", w.requires_grad, " <- 그대로다")
        inside = w * 2
        print("  블록 안에서 만든 것          :", inside.requires_grad)
    outside = w * 2
    print("  블록 밖에서 만든 것          :", outside.requires_grad)
    print("  no_grad 는 텐서가 아니라 '여기서 하는 일' 에 건다")


# ========================================================================
# 2. 막는 것은 그래프 세우기지 역전파가 아니다
# ========================================================================

def what_is_blocked():
    header("2. 무엇이 막히는가")
    w = torch.randn(2, requires_grad=True)
    loss = (w * 2).sum()                 # 그래프를 블록 밖에서 세운다
    with torch.no_grad():
        loss.backward()                  # 역전파는 블록 안에서
    print("  밖에서 세운 그래프를 안에서 역전파:", w.grad.tolist(), " 된다")

    w2 = torch.randn(2, requires_grad=True)
    with torch.no_grad():
        loss2 = (w2 * 2).sum()           # 그래프를 블록 안에서 세우면
    print("  안에서 만든 loss2.grad_fn        :", loss2.grad_fn)
    try:
        loss2.backward()
    except RuntimeError as e:
        print("  그것을 역전파하면               :",
              str(e).splitlines()[0][:52])
    print("  그러니 막히는 것은 세우기다. 걷기가 아니다")


# ========================================================================
# 3. 안쪽에서 다시 켤 수 있다
# ========================================================================

def nesting():
    header("3. 겹쳐 쓰기")
    w = torch.randn(2, requires_grad=True)
    print("  지금 추적 켜져 있나:", torch.is_grad_enabled())
    with torch.no_grad():
        print("    no_grad 안       :", torch.is_grad_enabled())
        with torch.enable_grad():
            print("    enable_grad 안   :", torch.is_grad_enabled())
            a = w * 2
        b = w * 3
    print("  enable_grad 안에서 만든 것:", a.requires_grad)
    print("  그 바깥에서 만든 것       :", b.requires_grad)
    print("  평가 루프 한가운데서 한 조각만 미분해야 할 때 쓴다")


# ========================================================================
# 4. 꾸밈씨로도 쓴다
# ========================================================================

def decorator():
    header("4. 함수 전체에 걸기")

    @torch.no_grad()
    def predict(model, x):
        return model(x)

    torch.manual_seed(0)
    net = nn.Linear(4, 2)
    out = predict(net, torch.randn(3, 4))
    print("  @torch.no_grad() 를 붙인 함수의 결과:", out.requires_grad)
    print("  평가 함수에 한 번 붙여 두면 부를 때마다 감쌀 필요가 없다")


# ========================================================================
# 5. 평가 루프에서 eval() 과 함께
# ========================================================================

def eval_loop():
    header("5. eval() 과 no_grad 는 다른 일이다")
    torch.manual_seed(0)
    net = nn.Sequential(nn.Linear(8, 16), nn.Dropout(0.5), nn.Linear(16, 1))
    x = torch.randn(4, 8)

    net.train()
    with torch.no_grad():
        a, b = net(x), net(x)
    print("  train() + no_grad: 두 번이 같은가", torch.allclose(a, b),
          " <- 드롭아웃이 켜져 있다")
    net.eval()
    with torch.no_grad():
        c, d = net(x), net(x)
    print("  eval()  + no_grad: 두 번이 같은가", torch.allclose(c, d))
    net.eval()
    e = net(x)
    print("  eval() 만          : grad_fn =",
          type(e.grad_fn).__name__, " <- 그래프를 들고 있다")
    print("  eval() 은 '무엇을 셈할지', no_grad 는 '기록할지' 를 정한다")


def main():
    scope()
    what_is_blocked()
    nesting()
    decorator()
    eval_loop()


if __name__ == "__main__":
    main()
```

**출력:**

```
=== 1. 범위는 코드 블록이다 ===
  들어가기 전 w.requires_grad : True
  블록 안 w.requires_grad     : True  <- 그대로다
  블록 안에서 만든 것          : False
  블록 밖에서 만든 것          : True
  no_grad 는 텐서가 아니라 '여기서 하는 일' 에 건다

=== 2. 무엇이 막히는가 ===
  밖에서 세운 그래프를 안에서 역전파: [2.0, 2.0]  된다
  안에서 만든 loss2.grad_fn        : None
  그것을 역전파하면               : element 0 of tensors does not require grad and does 
  그러니 막히는 것은 세우기다. 걷기가 아니다

=== 3. 겹쳐 쓰기 ===
  지금 추적 켜져 있나: True
    no_grad 안       : False
    enable_grad 안   : True
  enable_grad 안에서 만든 것: True
  그 바깥에서 만든 것       : False
  평가 루프 한가운데서 한 조각만 미분해야 할 때 쓴다

=== 4. 함수 전체에 걸기 ===
  @torch.no_grad() 를 붙인 함수의 결과: False
  평가 함수에 한 번 붙여 두면 부를 때마다 감쌀 필요가 없다

=== 5. eval() 과 no_grad 는 다른 일이다 ===
  train() + no_grad: 두 번이 같은가 False  <- 드롭아웃이 켜져 있다
  eval()  + no_grad: 두 번이 같은가 True
  eval() 만          : grad_fn = AddmmBackward0  <- 그래프를 들고 있다
  eval() 은 '무엇을 셈할지', no_grad 는 '기록할지' 를 정한다
```

## 2. 범위는 텐서가 아니라 코드 블록이다

```
들어가기 전 w.requires_grad : True
블록 안 w.requires_grad     : True    <- 그대로다
블록 안에서 만든 것          : False
블록 밖에서 만든 것          : True
```

`w`의 깃발은 블록 안에서도 `True`다. **`no_grad`는 텐서를 건드리지 않는다.**

바뀌는 것은 그 안에서 **새로 만들어지는** 것뿐이고, 블록을 나오면 곧바로
원상태다. 「경사 추적 끄기」의 `requires_grad_(False)`와 정반대 성격이다 —
그쪽은 텐서에 영구적으로 걸고, 이쪽은 코드 구간에 일시적으로 건다.

## 3. 막히는 것은 그래프 세우기다

```
밖에서 세운 그래프를 안에서 역전파: [2.0, 2.0]   된다
안에서 만든 loss2.grad_fn        : None
그것을 역전파하면               : element 0 of tensors does not require grad
```

**블록 안에서 `backward()`를 부르는 것은 아무 문제가 없다.** 그래프가 이미
세워져 있으면 걸어 내려가는 데 아무 지장이 없다.

막히는 것은 **그 안에서 그래프를 세우는 일**이다. `loss2`를 블록 안에서
만들었으니 `grad_fn`이 `None`이고, 걸어 내려갈 길이 없어 거절당한다.

이 구별이 쓸모 있는 자리가 있다. 긴 평가 루프를 `no_grad`로 감싸 두고, 그 안에서
미리 만들어 둔 로스를 역전파해도 된다. 반대로 블록 안에서 모델을 돌려 로스를
만들었다면 그것으로는 학습할 수 없다 — **그 로스는 숫자일 뿐 그래프가 아니다.**

## 4. 안쪽에서 다시 켤 수 있다

```
지금 추적 켜져 있나: True
  no_grad 안       : False
  enable_grad 안   : True
enable_grad 안에서 만든 것: True
그 바깥에서 만든 것       : False
```

`torch.enable_grad()`가 `no_grad`를 되돌린다. `torch.is_grad_enabled()`로 지금
상태를 물어볼 수도 있다.

쓰이는 자리는 **평가 루프 한가운데서 한 조각만 미분해야 할 때**다. 적대적
예제를 만들 때 입력에 대한 경사가 필요하거나, 설명 기법(Grad-CAM 같은 것)이
평가 중에 경사를 보는 경우가 그렇다.

## 5. 꾸밈씨로도 쓴다

```python
@torch.no_grad()
def predict(model, x):
    return model(x)
```

평가 함수에 한 번 붙여 두면 부를 때마다 감쌀 필요가 없다. 빠뜨릴 일이
없어진다는 것이 `with`보다 나은 점이다.

## 6. `eval()`과 `no_grad`는 다른 일이다

```
train() + no_grad: 두 번이 같은가 False   <- 드롭아웃이 켜져 있다
eval()  + no_grad: 두 번이 같은가 True
eval() 만          : grad_fn = AddmmBackward0   <- 그래프를 들고 있다
```

둘 다 해야 한다.

| | 무엇을 정하는가 | 빼먹으면 |
|---|---|---|
| `model.eval()` | 모델이 **무엇을 계산할지** | 답이 **틀리고** 매번 달라진다 |
| `torch.no_grad()` | 그 계산을 **기록할지** | 답은 맞고 **느리고 메모리를 쓴다** |

빼먹었을 때의 결과가 전혀 다르다. `no_grad`를 빼면 성능 문제이고, `eval()`을
빼면 **정확도 문제**다.

배치 정규화가 있으면 더 나쁘다. `train()` 모드에서는 **평가 배치의 통계로**
정규화하므로 한 표본의 예측이 같이 묶인 표본에 따라 달라지고, 배치 크기 1로
평가하면 아예 터지기도 한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
`no_grad` 블록 안에서 파라미터를 제자리에서 고칠 수 있는가? 「최소 학습 단계」에서
본 것을 확인하라.

</div>

??? success "풀이"
    ```python
    w = torch.tensor([1.0, 2.0], requires_grad=True)
    (w ** 2).sum().backward()
    try:
        w -= 0.1 * w.grad
    except RuntimeError as e:
        print("  그냥 고치면   :", str(e).splitlines()[0][:58])
    with torch.no_grad():
        w -= 0.1 * w.grad
    print("  no_grad 안에서:", [round(v, 4) for v in w.tolist()])
    print("  깃발은 그대로 :", w.requires_grad, " 잎인가:", w.is_leaf)
    ```

    ```
      그냥 고치면   : a leaf Variable that requires grad is being used in an in-
      no_grad 안에서: [0.8, 1.6]
      깃발은 그대로 : True  잎인가: True
    ```

    `no_grad` 안에서는 잎에 대한 제자리 연산 검사를 넘긴다. 그러고도 `w`는
    **여전히 잎이고 여전히 `requires_grad=True`**다.

    이것이 옵티마이저가 하는 일이다. 3절의 말로 하면, 갱신은 **기록할 일이 아닌**
    계산이다 — $w \leftarrow w - \eta \nabla w$를 다시 미분할 까닭이 없다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
적대적 예제를 만들려면 평가 중에 **입력에 대한 경사**가 필요하다. `no_grad`로
감싼 평가 루프 안에서 그것을 얻어라.

</div>

??? success "풀이"
    ```python
    torch.manual_seed(0)
    net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
    net.eval()
    x = torch.randn(1, 4)
    target = torch.tensor([1])

    with torch.no_grad():                      # 평가 루프 전체
        plain = net(x)
        print("  보통 평가 grad_fn:", plain.grad_fn)

        with torch.enable_grad():              # 이 조각만 다시 켠다
            xa = x.clone().requires_grad_(True)
            loss = nn.functional.cross_entropy(net(xa), target)
            loss.backward()
        print("  입력에 대한 경사 :",
              [round(v, 4) for v in xa.grad[0].tolist()])

        adv = x + 0.1 * xa.grad.sign()         # FGSM 한 걸음
        print("  원래 예측        :", net(x).argmax(1).item())
        print("  흔든 뒤 예측     :", net(adv).argmax(1).item())
    ```

    ```
      보통 평가 grad_fn: None
      입력에 대한 경사 : [-0.0584, -0.0821, 0.0388, -0.0283]
      원래 예측        : 1
      흔든 뒤 예측     : 1
    ```

    `enable_grad` 안에서만 그래프가 세워져 `xa.grad`를 얻었다. 바깥은 여전히
    `no_grad`라 `plain`의 `grad_fn`이 `None`이다.

    `x.clone().requires_grad_(True)`로 새 잎을 만든 것이 핵심이다. 원래 `x`는
    깃발이 꺼져 있으므로 경사를 못 받고, 그렇다고 `x` 자체에 깃발을 켜면 평가
    루프의 다른 부분까지 영향을 받는다.

    이 예에서는 한 걸음으로 예측이 안 바뀌었다 — 학습하지 않은 그물이라 결정
    경계가 어디 있는지 모르는 상태다. 제대로 학습한 모델에서는 이 정도 흔들림으로
    예측이 뒤집히는 것이 적대적 예제의 요점이다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
`no_grad`로 감싼 평가에서 얻은 출력을 **다음 학습의 목표값으로** 쓰려 한다
(지식 증류나 강화학습의 타깃망이 이렇게 한다). 되는가? `inference_mode`로
감쌌다면?

</div>

??? success "풀이"
    ```python
    torch.manual_seed(0)
    teacher = nn.Linear(4, 2)
    student = nn.Linear(4, 2)
    x = torch.randn(8, 4)

    with torch.no_grad():                       # 교사의 출력
        soft = teacher(x)
    loss = nn.functional.mse_loss(student(x), soft)
    loss.backward()
    print("  no_grad 로 만든 목표값으로 학습:",
          "된다" if student.weight.grad is not None else "안 된다")
    print("    목표값에는 경사가 안 간다:", soft.requires_grad)

    student.zero_grad()
    with torch.inference_mode():
        soft2 = teacher(x)
    try:
        nn.functional.mse_loss(student(x), soft2).backward()
        print("  inference_mode 로 만든 목표값으로 학습: 된다")
    except RuntimeError as e:
        print("  inference_mode 로 만든 목표값:",
              str(e).splitlines()[0][:60])
    ```

    ```
      no_grad 로 만든 목표값으로 학습: 된다
        목표값에는 경사가 안 간다: False
      inference_mode 로 만든 목표값: Inference tensors cannot be saved for backward.
    ```

    **`no_grad`는 되고 `inference_mode`는 막힌다.**

    `no_grad`로 만든 목표값은 평범한 텐서다. 로스가 그것을 상수처럼 읽고 경사는
    학생 쪽으로만 흐른다 — `soft.requires_grad`가 `False`인 것이 그 뜻이다.

    `inference_mode`로 만든 것은 **추론 텐서**라 역전파를 위해 저장될 수조차
    없다. `mse_loss`가 그 목표값을 저장해 두어야 하는데 거절당한다.

    「경사 추적을 멈추는 시점」에서 적은 기준이 그대로 들어맞는다. **그 결과를
    뒤에 다시 쓸 일이 있으면 `no_grad`**다. 지식 증류의 교사, 강화학습의 타깃망,
    준지도 학습의 가짜 라벨이 전부 여기 해당한다.

    오류 메시지가 "To work around" 로 이어지며 `.clone()` 을 권한다. 추론
    텐서를 복제하면 평범한 텐서가 되기 때문이다. 다만 그럴 바에는 처음부터
    `no_grad` 를 쓰는 편이 간단하다.

    교사 모델을 `no_grad`로 감싸는 것 자체는 반드시 해야 한다. 안 감싸면 교사의
    그래프가 통째로 살아 있어 메모리를 두 배로 쓰고, 실수로 교사까지 학습시킬
    수도 있다.
