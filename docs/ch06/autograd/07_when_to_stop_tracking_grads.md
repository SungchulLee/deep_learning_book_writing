# 경사 추적을 멈추는 시점 — 네 가지 길과 쓰는 자리

파이토치가 추적을 끄는 길을 넷 준다. 셋은 자주 섞이고, 넷째는 아는 사람이 적다.

| 길 | 무엇을 건드리는가 | 되돌릴 수 있는가 |
|---|---|---|
| `t.requires_grad_(False)` | **그 텐서의 깃발** | 된다 |
| `with torch.no_grad()` | 블록 안의 **기록** | 블록을 나가면 끝 |
| `t.detach()` | **새 텐서**를 만든다 | 원본은 그대로 |
| `with torch.inference_mode()` | 기록 + 버전 관리까지 | **못 되돌린다** |

고르는 기준은 "무엇을 끄고 싶은가"다. 텐서 하나를 영영 학습에서 빼고 싶은가,
이 블록 동안만 기록을 멈추고 싶은가, 값만 꺼내 가고 싶은가.

## 1. 코드

```python
"""경사 추적을 멈추는 네 가지 길 — 어느 것을 언제 쓰는가.

파이토치가 추적을 끄는 길을 넷 준다. 셋은 자주 섞이고, 넷째는
아는 사람이 적다.

| 길 | 무엇을 건드리는가 | 되돌릴 수 있는가 |
|---|---|---|
| `requires_grad_(False)` | 그 텐서의 깃발 | 된다 |
| `with torch.no_grad()`  | 블록 안의 **기록** | 블록을 나가면 끝 |
| `t.detach()`            | 새 텐서를 만든다 | 원본은 그대로 |
| `with torch.inference_mode()` | 기록 + 버전 관리까지 | **못 되돌린다** |

실행: python 07_when_to_stop_tracking_grads.py
"""
import torch
import torch.nn as nn


def header(title):
    print(f"\n=== {title} ===")


def describe(name, t):
    fn = type(t.grad_fn).__name__ if t.grad_fn is not None else None
    print(f"  {name:28s} requires_grad={t.requires_grad!s:5s} grad_fn={fn}")


# ========================================================================
# 1. 넷이 무엇을 건드리는가
# ========================================================================

def what_each_touches():
    header("1. 무엇을 건드리는가")
    w = torch.randn(3, requires_grad=True)

    w.requires_grad_(False)
    print("  requires_grad_(False) 뒤 w:", w.requires_grad, " <- 텐서 자체가 바뀐다")
    w.requires_grad_(True)

    with torch.no_grad():
        y = w * 2
    describe("no_grad 안에서 만든 y", y)
    print("  블록 밖 w 는 그대로:", w.requires_grad)

    d = w.detach()
    describe("w.detach()", d)
    print("  원본 w 는 그대로:", w.requires_grad,
          "  저장소 공유:", d.data_ptr() == w.data_ptr())

    h = w * 2
    h.detach_()
    describe("h.detach_() (제자리)", h)


# ========================================================================
# 2. 그래프를 세우느냐 마느냐
# ========================================================================

def graph_or_not():
    header("2. 그래프가 세워지는가")
    torch.manual_seed(0)
    net = nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 10))
    x = torch.randn(64, 256)

    describe("기본", net(x))
    with torch.no_grad():
        describe("no_grad", net(x))
    with torch.inference_mode():
        describe("inference_mode", net(x))
    print("  뒤의 둘은 grad_fn 이 None 이다 — 그래프를 아예 안 세운다")
    print("  그래서 역전파가 쓰려고 저장해 둘 중간값도 없다")
    print("  긴 그물에서 아끼는 메모리가 바로 그 중간값들이다")


# ========================================================================
# 3. no_grad 와 inference_mode 가 갈리는 자리
# ========================================================================

def nograd_vs_inference():
    header("3. no_grad 와 inference_mode")
    with torch.no_grad():
        a = torch.randn(3) * 2
    a.requires_grad_(True)
    print("  no_grad 로 만든 것은 나중에 켤 수 있다:", a.requires_grad)
    (a * 3).sum().backward()
    print("    역전파도 된다. a.grad =", a.grad.tolist())

    with torch.inference_mode():
        b = torch.randn(3) * 2
    try:
        b.requires_grad_(True)
    except RuntimeError as e:
        print("  inference_mode 로 만든 것은:", str(e).splitlines()[0][:66])
    print("  추론 텐서에는 표가 박혀 있어 학습 그래프에 다시 들어갈 수 없다")
    print("  그래서 더 많이 아끼는 대신, 결과를 학습에 쓰려면 못 쓴다")


# ========================================================================
# 4. 어느 자리에 어느 것을 쓰는가
# ========================================================================

def which_where():
    header("4. 자리마다 쓰는 것")
    torch.manual_seed(0)
    net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
    x, y = torch.randn(16, 4), torch.randn(16, 1)
    opt = torch.optim.SGD(net.parameters(), lr=0.1)

    # (가) 파라미터 갱신 — 옵티마이저 안에서 no_grad 가 쓰인다
    opt.zero_grad()
    nn.functional.mse_loss(net(x), y).backward()
    opt.step()
    print("  (가) 갱신: opt.step() 이 속에서 no_grad 를 쓴다")

    # (나) 평가 — no_grad 로 감싼다
    net.eval()
    with torch.no_grad():
        val = nn.functional.mse_loss(net(x), y)
    print(f"  (나) 평가: no_grad 로 감싼다. 로스 {val.item():.6f}")
    print("       eval() 과 no_grad 는 다른 일이다 — 둘 다 필요하다")
    net.train()

    # (다) 기록 — detach 로 값만 꺼낸다
    loss = nn.functional.mse_loss(net(x), y)
    history = [loss.detach()]
    print("  (다) 기록: loss.detach() 로 값만 들고 간다")
    print("       detach 없이 리스트에 쌓으면 그래프가 통째로 따라온다")

    # (라) 얼리기 — requires_grad_(False)
    for p in net[0].parameters():
        p.requires_grad_(False)
    frozen = [p.requires_grad for p in net.parameters()]
    print("  (라) 얼리기: requires_grad_(False). 층마다 깃발:", frozen)


def main():
    what_each_touches()
    graph_or_not()
    nograd_vs_inference()
    which_where()


if __name__ == "__main__":
    main()
```

**출력:**

```
=== 1. 무엇을 건드리는가 ===
  requires_grad_(False) 뒤 w: False  <- 텐서 자체가 바뀐다
  no_grad 안에서 만든 y             requires_grad=False grad_fn=None
  블록 밖 w 는 그대로: True
  w.detach()                   requires_grad=False grad_fn=None
  원본 w 는 그대로: True   저장소 공유: True
  h.detach_() (제자리)            requires_grad=False grad_fn=None

=== 2. 그래프가 세워지는가 ===
  기본                           requires_grad=True  grad_fn=AddmmBackward0
  no_grad                      requires_grad=False grad_fn=None
  inference_mode               requires_grad=False grad_fn=None
  뒤의 둘은 grad_fn 이 None 이다 — 그래프를 아예 안 세운다
  그래서 역전파가 쓰려고 저장해 둘 중간값도 없다
  긴 그물에서 아끼는 메모리가 바로 그 중간값들이다

=== 3. no_grad 와 inference_mode ===
  no_grad 로 만든 것은 나중에 켤 수 있다: True
    역전파도 된다. a.grad = [3.0, 3.0, 3.0]
  inference_mode 로 만든 것은: Setting requires_grad=True on inference tensor outside InferenceMo
  추론 텐서에는 표가 박혀 있어 학습 그래프에 다시 들어갈 수 없다
  그래서 더 많이 아끼는 대신, 결과를 학습에 쓰려면 못 쓴다

=== 4. 자리마다 쓰는 것 ===
  (가) 갱신: opt.step() 이 속에서 no_grad 를 쓴다
  (나) 평가: no_grad 로 감싼다. 로스 1.046843
       eval() 과 no_grad 는 다른 일이다 — 둘 다 필요하다
  (다) 기록: loss.detach() 로 값만 들고 간다
       detach 없이 리스트에 쌓으면 그래프가 통째로 따라온다
  (라) 얼리기: requires_grad_(False). 층마다 깃발: [False, False, True, True]
```

## 2. 넷이 건드리는 것이 다르다

```
requires_grad_(False) 뒤 w: False      <- 텐서 자체가 바뀐다
no_grad 안에서 만든 y: requires_grad=False
  블록 밖 w 는 그대로: True
w.detach()           : requires_grad=False   저장소 공유: True
  원본 w 는 그대로    : True
```

**`requires_grad_(False)`만 원본을 바꾼다.** 나머지 둘은 원본을 건드리지 않고
각자 다른 방식으로 피해 간다 — `no_grad`는 그 안에서 **만드는** 텐서에 깃발을
안 달고, `detach`는 깃발 없는 **새 손잡이**를 내준다.

`detach()`가 저장소를 공유한다는 점은 「데이터 관련 속성」에서 본 대로다. 미분만
끊은 같은 데이터이므로, 고칠 작정이면 `detach().clone()`이라야 한다.

## 3. 그래프를 세우느냐 마느냐

```
기본           : requires_grad=True   grad_fn=AddmmBackward0
no_grad        : requires_grad=False  grad_fn=None
inference_mode : requires_grad=False  grad_fn=None
```

`grad_fn`이 `None`이라는 것은 **그래프가 아예 안 세워졌다**는 뜻이다. 그래프가
없으면 역전파가 쓰려고 저장해 둘 중간값도 없다.

이것이 추론에서 `no_grad`로 감싸는 진짜 이유다. 깊은 그물의 순전파는 층마다
활성값을 남기는데, 역전파를 할 생각이 없다면 그것을 들고 있을 까닭이 없다.
「메모리 관리」에서 그 활성값이 모델 가중치보다 훨씬 클 수 있음을 보았다.

## 4. `no_grad`와 `inference_mode`가 갈리는 자리

```
no_grad 로 만든 것은 나중에 켤 수 있다: True
  역전파도 된다. a.grad = [3.0, 3.0, 3.0]

inference_mode 로 만든 것은:
  Setting requires_grad=True on inference tensor outside InferenceMode is not allowed
```

`no_grad` 안에서 나온 텐서는 **평범한 텐서**다. 나중에 `requires_grad_(True)`를
켜서 학습에 쓸 수 있다.

`inference_mode`에서 나온 텐서는 다르다. **추론 텐서라는 표가 박혀 있어** 학습
그래프에 다시 들어갈 수 없다. 더 공격적으로 아끼는 대신(버전 관리까지 꺼서
「데이터 관련 속성」의 `_version` 추적도 하지 않는다) 되돌릴 길을 막은 것이다.

그래서 고르는 기준은 **그 결과를 다시 학습에 쓸 일이 있는가**다.

| | 쓸 자리 |
|---|---|
| `inference_mode` | 서비스에 띄운 모델, 평가 전용 루프 |
| `no_grad` | 그 결과를 뒤에 다시 쓸 수도 있는 곳 (강화학습의 타깃값, 준지도 학습의 가짜 라벨) |

헷갈리면 `no_grad`를 쓰는 편이 안전하다. 아끼는 차이는 크지 않고, 막혔을 때
드러나는 오류 메시지가 한참 뒤에 나타나기 때문이다.

## 5. 자리마다 쓰는 것

```
(가) 갱신: opt.step() 이 속에서 no_grad 를 쓴다
(나) 평가: no_grad 로 감싼다. 로스 1.046843
(다) 기록: loss.detach() 로 값만 들고 간다
(라) 얼리기: requires_grad_(False). 층마다 깃발: [False, False, True, True]
```

**(가) 파라미터 갱신** — 내가 쓸 일은 거의 없다. 옵티마이저가 속에서 한다.
손으로 갱신한다면 「최소 학습 단계」에서 본 대로 `no_grad`가 필요하다.

**(나) 평가** — `model.eval()`과 `torch.no_grad()`는 **다른 일**이고 둘 다
필요하다. `eval()`은 드롭아웃과 배치 정규화의 동작을 바꾸고, `no_grad`는 그래프를
안 세운다. 하나만 하면 반만 맞는다.

**(다) 기록** — 로스를 리스트에 쌓을 때 `detach()`를 빼먹으면 **그래프가 통째로
따라온다.** 에폭마다 하나씩 쌓이면 메모리가 학습 내내 늘어난다. 「메모리 관리」와
「cat 시각화」에서 같은 자리를 짚었다.

**(라) 얼리기** — 층마다 `requires_grad_(False)`를 걸면 그 층은 경사를 받지
않는다. 전이학습에서 앞쪽을 얼리는 것이 이것이고, 영구적인 변경이라 다시 켜려면
명시적으로 `True`로 돌려야 한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
`no_grad` 블록 안에서 **이미 `requires_grad=True`인 텐서**를 읽으면 그 깃발이
바뀌는가? 그리고 블록 안에서 그 텐서로 만든 결과는 어떤가?

</div>

??? success "풀이"
    ```python
    w = torch.randn(3, requires_grad=True)
    print("  블록 들어가기 전 w.requires_grad :", w.requires_grad)
    with torch.no_grad():
        print("  블록 안 w.requires_grad        :", w.requires_grad)
        y = w * 2
        print("  블록 안에서 만든 y             :", y.requires_grad)
    print("  블록 나온 뒤 w.requires_grad     :", w.requires_grad)
    z = w * 2
    print("  블록 밖에서 만든 z               :", z.requires_grad)
    ```

    ```
      블록 들어가기 전 w.requires_grad : True
      블록 안 w.requires_grad        : True
      블록 안에서 만든 y             : False
      블록 나온 뒤 w.requires_grad     : True
      블록 밖에서 만든 z               : True
    ```

    **`w`의 깃발은 블록 안에서도 `True`다.** `no_grad`는 텐서를 건드리지 않는다.

    바뀌는 것은 **블록 안에서 새로 만들어지는 것**뿐이다. `y`는 `requires_grad=True`인
    `w`에서 나왔는데도 `False`다. 블록을 나오면 다시 정상으로 돌아가 `z`는 `True`다.

    그래서 `no_grad`는 "이 텐서를 학습에서 빼라"가 아니라 **"여기서 하는 일은
    기록하지 말라"**는 뜻이다. 범위가 텐서가 아니라 코드 블록이다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
로스를 리스트에 쌓을 때 `detach()`를 빼먹으면 메모리가 는다고 했다. 백 번
쌓아 보고, `detach()` 한 쪽과 안 한 쪽에서 **살아 있는 그래프의 수**를 세어라.

</div>

??? success "풀이"
    ```python
    import gc

    torch.manual_seed(0)
    net = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1))
    x = torch.randn(32, 64)

    for label, keep in (("detach 함  ", True), ("detach 안 함", False)):
        gc.collect()
        hist = []
        for _ in range(100):
            loss = net(x).pow(2).mean()
            hist.append(loss.detach() if keep else loss)
        alive = sum(1 for o in gc.get_objects()
                    if isinstance(o, torch.Tensor) and o.grad_fn is not None)
        print(f"  {label}  grad_fn 이 살아 있는 텐서 {alive:4d}개")
        del hist
        gc.collect()
    ```

    ```
      detach 함    grad_fn 이 살아 있는 텐서    3개
      detach 안 함  grad_fn 이 살아 있는 텐서  300개
    ```

    **백 배 차이다.** `detach()` 한 쪽에 남은 3개는 마지막 바퀴의 중간값들이고
    다음 바퀴에서 버려진다. 안 한 쪽의 300개는 **백 바퀴 치가 전부 살아 있는
    것**이다 — 바퀴마다 로스 하나와 그 그래프의 중간값 둘이 리스트에 묶여 있다. 리스트가 로스
    텐서를 들고 있고, 로스 텐서가 `grad_fn`을 들고 있고, `grad_fn`이 그 순전파의
    중간값들을 들고 있기 때문이다.

    숫자 하나를 기록하려고 모델 백 번 치의 활성값을 쥐고 있는 셈이다. 학습이
    길어질수록 늘기만 하므로, **에폭 중반에 메모리가 터지는 전형적인 꼴**이
    여기서 나온다.

    `.item()`으로 파이썬 수를 꺼내는 것도 같은 일을 한다. 기록용이라면 그쪽이
    더 분명하다 — 텐서가 아니라 수라는 것이 코드에 드러나니까.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
평가 루프에서 `model.eval()`만 하고 `no_grad`를 빼면 **결과가 틀리는가, 느려질
뿐인가?** 그리고 반대로 `no_grad`만 하고 `eval()`을 빼면?

</div>

??? success "풀이"
    ```python
    torch.manual_seed(0)
    net = nn.Sequential(nn.Linear(16, 32), nn.Dropout(0.5), nn.Linear(32, 1))
    x = torch.randn(8, 16)

    net.eval()
    with torch.no_grad():
        a = net(x)
    net.eval()
    b = net(x)                               # no_grad 만 뺐다
    net.train()
    with torch.no_grad():
        c1, c2 = net(x), net(x)              # eval 만 뺐다

    print("  eval + no_grad  와 eval 만:", torch.allclose(a, b), " 값은 같다")
    print("    다만 b 는 그래프를 들고 있다. grad_fn =",
          type(b.grad_fn).__name__)
    print("  no_grad 만 했을 때 두 번 부르면 같은가:", torch.allclose(c1, c2))
    print("    드롭아웃이 켜져 있어 부를 때마다 다르다")
    ```

    ```
      eval + no_grad  와 eval 만: True  값은 같다
        다만 b 는 그래프를 들고 있다. grad_fn = AddmmBackward0
      no_grad 만 했을 때 두 번 부르면 같은가: False
        드롭아웃이 켜져 있어 부를 때마다 다르다
    ```

    **둘이 전혀 다른 종류의 실수다.**

    `no_grad`를 빼면 **값은 맞다.** 느려지고 메모리를 더 쓸 뿐이다. 나쁘지만
    결과를 틀리게 하지는 않는다.

    `eval()`을 빼면 **값이 틀린다.** 드롭아웃이 켜진 채로 평가하므로 유닛 절반을
    끄고 답을 낸 것이고, 게다가 부를 때마다 다른 값이 나와 **재현되지도 않는다.**
    「제자리 연산」의 연습문제에서 같은 것을 재었다 — 0.23점씩 흔들렸다.

    배치 정규화가 있으면 더 나쁘다. `train()` 모드에서는 **평가 배치의 통계로**
    정규화하므로, 배치를 어떻게 묶느냐에 따라 한 표본의 예측이 달라진다. 배치
    크기 1로 평가하면 아예 터지기도 한다.

    그래서 순서를 외울 것이 아니라 역할로 기억하는 편이 낫다.
    **`eval()`은 모델이 무엇을 계산할지, `no_grad`는 그 계산을 기록할지**를
    정한다. 묻는 것이 다르므로 둘 다 필요하다.
