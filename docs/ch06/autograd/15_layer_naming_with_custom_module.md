# 맞춤 모듈로 레이어에 이름 붙이기 — `Sequential`로는 못 적는 것들

속성 이름이 그대로 `state_dict`의 키가 되는 것은 「`OrderedDict`으로 레이어에
이름 붙이기」와 같다. 다른 것은 **그 사이에 무엇을 끼워 넣을 수 있느냐**다.

`nn.Sequential`은 "한 줄로 지나가는" 모델만 적을 수 있다. 앞 층의 출력이 다음
층의 입력이고, 그게 전부다. 그 틀을 벗어나는 것 넷을 본다.

- **건너뛰어 더하기** — 입력을 한 번 더 쓴다
- **갈래 치기** — 출력이 둘 이상
- **중간값 꺼내기** — `forward`가 인자를 더 받는다
- **파라미터 없는 층을 안 만들기** — `F.relu`로 끝

## 1. 코드

```python
"""nn.Module 을 상속해 쓰기 — Sequential 로는 못 하는 것들.

`nn.Sequential` 은 "한 줄로 지나가는" 모델만 적을 수 있다. 입력을
두 갈래로 보내거나, 건너뛰어 더하거나, 중간값을 꺼내 쓰려면
`forward` 를 손으로 적어야 한다.

속성 이름이 그대로 `state_dict` 의 키가 되는 것은 「OrderedDict 으로
레이어에 이름 붙이기」와 같다. 다른 것은 **그 사이에 무엇을 끼워 넣을
수 있느냐**다.

실행: 이 쪽의 코드를 파일로 갈무리해 돌린다
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def header(title):
    print(f"\n=== {title} ===")


# ========================================================================
# 1. 속성 이름이 키가 된다
# ========================================================================

class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(5, 3)
        self.head = nn.Linear(3, 1)

    def forward(self, x):
        return self.head(F.relu(self.encoder(x)))


def names():
    header("1. 속성 이름이 키가 된다")
    net = TinyNet()
    print("  state_dict 키:", list(net.state_dict().keys()))
    print("  ReLU 를 F.relu 로 적어 모듈이 아예 없다")
    print("  모듈 목록:", [n for n, _ in net.named_children()])
    print("  Sequential 과 달리 파라미터 없는 층을 안 만들어도 된다")


# ========================================================================
# 2. Sequential 로는 못 적는 것 — 건너뛰기
# ========================================================================

class Residual(nn.Module):
    def __init__(self, d, skip=True):
        super().__init__()
        self.fc1 = nn.Linear(d, d)
        self.fc2 = nn.Linear(d, d)
        self.skip = skip

    def forward(self, x):
        h = self.fc2(F.relu(self.fc1(x)))
        return x + h if self.skip else h          # 입력을 더한다


def skip():
    header("2. 건너뛰어 더하기")
    torch.manual_seed(0)
    block = Residual(4)
    x = torch.randn(2, 4)
    print("  입력 모양", tuple(x.shape), "-> 출력", tuple(block(x).shape))
    print("  x 를 한 번 더 쓰는 식은 Sequential 로 적을 수 없다")
    print("  Sequential 은 앞 층의 출력만 다음 층에 넘기기 때문이다")

    print("  블록 스무 개를 쌓고 첫 블록에 닿는 경사를 재어 본다")
    for use_skip in (True, False):
        torch.manual_seed(0)
        deep = nn.Sequential(*[Residual(16, use_skip) for _ in range(20)])
        deep(torch.randn(8, 16)).pow(2).mean().backward()
        first = deep[0].fc1.weight.grad.abs().mean().item()
        last = deep[-1].fc1.weight.grad.abs().mean().item()
        print(f"    지름길 {str(use_skip):5s} 첫 블록 {first:.3e}  "
              f"마지막 블록 {last:.3e}  비 {first / last:.2e}")
    print("  지름길이 없으면 첫 블록의 경사가 사라진다")
    print("  더하기의 미분이 1 이라 지름길이 경사를 그대로 흘려 보낸다")


# ========================================================================
# 3. 두 갈래로 보내기
# ========================================================================

class TwoHeads(nn.Module):
    def __init__(self):
        super().__init__()
        self.trunk = nn.Linear(4, 8)
        self.cls = nn.Linear(8, 3)
        self.reg = nn.Linear(8, 1)

    def forward(self, x):
        h = F.relu(self.trunk(x))
        return self.cls(h), self.reg(h)          # 둘을 돌려준다


def branching():
    header("3. 출력이 둘")
    torch.manual_seed(0)
    net = TwoHeads()
    a, b = net(torch.randn(2, 4))
    print("  분류 머리", tuple(a.shape), "  회귀 머리", tuple(b.shape))
    print("  Sequential 은 텐서 하나만 돌려줄 수 있다")

    loss = F.cross_entropy(a, torch.tensor([0, 1])) + b.pow(2).mean()
    loss.backward()
    print("  두 로스를 더해 한 번 역전파하면 trunk 가 둘 다에게서 받는다")
    print("  trunk.weight.grad 가 있는가:", net.trunk.weight.grad is not None)


# ========================================================================
# 4. 중간값을 꺼내 쓰기
# ========================================================================

class WithTap(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(4, 8)
        self.b = nn.Linear(8, 8)
        self.c = nn.Linear(8, 2)

    def forward(self, x, return_hidden=False):
        h1 = F.relu(self.a(x))
        h2 = F.relu(self.b(h1))
        out = self.c(h2)
        return (out, h1, h2) if return_hidden else out


def tapping():
    header("4. 중간값 꺼내기")
    torch.manual_seed(0)
    net = WithTap()
    x = torch.randn(2, 4)
    print("  보통 부르면 :", tuple(net(x).shape))
    out, h1, h2 = net(x, return_hidden=True)
    print("  중간값까지  :", tuple(out.shape), tuple(h1.shape), tuple(h2.shape))
    print("  forward 에 인자를 더 받을 수 있는 것도 Sequential 과 다르다")
    print("  특징 뽑기, 증류, 시각화가 전부 이 자리에서 나온다")


# ========================================================================
# 5. 흔한 실수 — super().__init__() 을 빼먹으면
# ========================================================================

def forgot_super():
    header("5. super().__init__() 을 빼먹으면")

    class Broken(nn.Module):
        def __init__(self):
            self.fc = nn.Linear(2, 2)          # super() 를 안 불렀다

    try:
        Broken()
    except AttributeError as e:
        print("  만들 때:", str(e).splitlines()[0][:72])
    print("  nn.Module 이 속성을 가로채려면 먼저 제 살림을 차려야 한다")
    print("  그 살림이 _parameters, _modules 같은 사전들이다")


def main():
    names()
    skip()
    branching()
    tapping()
    forgot_super()


if __name__ == "__main__":
    main()
```

**출력:**

```
=== 1. 속성 이름이 키가 된다 ===
  state_dict 키: ['encoder.weight', 'encoder.bias', 'head.weight', 'head.bias']
  ReLU 를 F.relu 로 적어 모듈이 아예 없다
  모듈 목록: ['encoder', 'head']
  Sequential 과 달리 파라미터 없는 층을 안 만들어도 된다

=== 2. 건너뛰어 더하기 ===
  입력 모양 (2, 4) -> 출력 (2, 4)
  x 를 한 번 더 쓰는 식은 Sequential 로 적을 수 없다
  Sequential 은 앞 층의 출력만 다음 층에 넘기기 때문이다
  블록 스무 개를 쌓고 첫 블록에 닿는 경사를 재어 본다
    지름길 True  첫 블록 1.177e-01  마지막 블록 5.382e-02  비 2.19e+00
    지름길 False 첫 블록 2.311e-15  마지막 블록 9.248e-04  비 2.50e-12
  지름길이 없으면 첫 블록의 경사가 사라진다
  더하기의 미분이 1 이라 지름길이 경사를 그대로 흘려 보낸다

=== 3. 출력이 둘 ===
  분류 머리 (2, 3)   회귀 머리 (2, 1)
  Sequential 은 텐서 하나만 돌려줄 수 있다
  두 로스를 더해 한 번 역전파하면 trunk 가 둘 다에게서 받는다
  trunk.weight.grad 가 있는가: True

=== 4. 중간값 꺼내기 ===
  보통 부르면 : (2, 2)
  중간값까지  : (2, 2) (2, 8) (2, 8)
  forward 에 인자를 더 받을 수 있는 것도 Sequential 과 다르다
  특징 뽑기, 증류, 시각화가 전부 이 자리에서 나온다

=== 5. super().__init__() 을 빼먹으면 ===
  만들 때: cannot assign module before Module.__init__() call
  nn.Module 이 속성을 가로채려면 먼저 제 살림을 차려야 한다
  그 살림이 _parameters, _modules 같은 사전들이다
```

## 2. `F.relu`로 적으면 모듈이 안 생긴다

```
state_dict 키: ['encoder.weight', 'encoder.bias', 'head.weight', 'head.bias']
모듈 목록: ['encoder', 'head']
```

`nn.ReLU()`를 속성으로 두지 않고 `forward` 안에서 `F.relu`를 부르면 **모듈
목록에 안 나타난다.** 파라미터가 없으니 `state_dict`에도 차이가 없다.

둘 중 어느 쪽이든 결과는 같다(연습문제 1). 가르는 것은 **모델을 들여다보거나
고칠 때**다 — `nn.ReLU`로 두면 `print(model)`에 보이고 훅을 걸 수 있다.

## 3. 건너뛰어 더하기 — 경사가 살아남는다

```python
def forward(self, x):
    return x + self.fc2(F.relu(self.fc1(x)))
```

`x`를 한 번 더 쓴다. `Sequential`로는 적을 수 없다 — 앞 층의 출력만 넘어가니
원래 `x`에 닿을 길이 없다.

블록 스무 개를 쌓고 **첫 블록에 닿는 경사**를 재어 보면 무엇이 달라지는지
보인다.

| | 첫 블록 | 마지막 블록 | 비 |
|---|---|---|---|
| 지름길 있음 | 1.177e-01 | 5.382e-02 | 2.19 |
| 지름길 없음 | **2.311e-15** | 9.248e-04 | 2.50e-12 |

지름길이 없으면 첫 블록의 경사가 $10^{-15}$다. **사라졌다.** 앞쪽 층들이
사실상 학습되지 않는다.

까닭은 연쇄 법칙이다. 스무 층을 거치며 각 층의 야코비가 곱해지는데, 그 크기가
1보다 조금만 작아도 스무 번 곱하면 0으로 간다. 지름길이 있으면
$\frac{\partial}{\partial x}(x + f(x)) = 1 + f'(x)$라 **1이 더해져** 곱해도 안
줄어든다.

이것이 ResNet이 깊은 그물을 학습 가능하게 만든 방법이고, 그것을 적으려면
`forward`를 손으로 써야 한다.

## 4. 출력이 둘

```python
def forward(self, x):
    h = F.relu(self.trunk(x))
    return self.cls(h), self.reg(h)
```

```
분류 머리 (2, 3)   회귀 머리 (2, 1)
두 로스를 더해 한 번 역전파하면 trunk 가 둘 다에게서 받는다
trunk.weight.grad 가 있는가: True
```

`Sequential`은 텐서 하나만 돌려줄 수 있다. 객체 탐지(상자와 클래스), 다중
작업 학습, VAE(평균과 로그분산)가 전부 이 꼴이다.

`trunk`가 두 머리 양쪽에서 경사를 받는다 — 「경사 누적」에서 본 대로 한 입력이
여러 곳에 쓰이면 **더해져서** 돌아온다.

## 5. 중간값 꺼내기

```python
def forward(self, x, return_hidden=False):
    ...
    return (out, h1, h2) if return_hidden else out
```

```
보통 부르면 : (2, 2)
중간값까지  : (2, 2) (2, 8) (2, 8)
```

`forward`가 **인자를 더 받는 것**도 `Sequential`과 다르다. 특징 뽑기, 지식
증류, 어텐션 시각화가 전부 중간값을 꺼내는 일이다.

(모델을 안 고치고 꺼내는 길도 있다 — `register_forward_hook`. 남의 모델에서
꺼낼 때 그쪽을 쓴다.)

## 6. `super().__init__()`을 빼먹으면

```
만들 때: cannot assign module before Module.__init__() call
```

`nn.Module`이 속성 대입을 **가로채서** 등록하려면 먼저 제 살림(`_parameters`,
`_modules` 같은 사전들)을 차려야 한다. 그래서 `super().__init__()`이 첫 줄에
와야 하고, 빼먹으면 모델을 만드는 그 자리에서 터진다.

다행히 터져 준다. 그런데 **리스트에 담으면 안 터진다** — 연습문제 3이 그
자리다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
`F.relu`로 적은 모델과 `nn.ReLU()`를 속성으로 둔 모델은 무엇이 다른가?
같은 입력에 같은 답을 주는가?

</div>

??? success "풀이"
    ```python
    class A(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(2, 2)
        def forward(self, x):
            return F.relu(self.fc(x))

    class B(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(2, 2)
            self.act = nn.ReLU()
        def forward(self, x):
            return self.act(self.fc(x))

    for n, m in (("F.relu", A()), ("nn.ReLU", B())):
        print(f"  {n:8s} state_dict {list(m.state_dict().keys())}  "
              f"모듈 {[k for k, _ in m.named_children()]}")
    torch.manual_seed(0); a = A()
    torch.manual_seed(0); b = B()
    print("  출력이 같은가:", torch.allclose(a(torch.randn(3, 2)),
                                        b(torch.randn(3, 2))))
    ```

    ```
      F.relu   state_dict ['fc.weight', 'fc.bias']  모듈 ['fc']
      nn.ReLU  state_dict ['fc.weight', 'fc.bias']  모듈 ['fc', 'act']
      출력이 같은가: True
    ```

    **`state_dict`는 같고 모듈 목록만 다르다.** ReLU에 파라미터가 없으니 저장할
    것도 없기 때문이다. 출력도 당연히 같다.

    그래서 고르는 기준이 저장이나 성능이 아니라 **보이느냐**다.

    | | |
    |---|---|
    | `nn.ReLU` | `print(model)`에 보이고, 훅을 걸 수 있고, `inplace=True`를 줄 수 있다 |
    | `F.relu` | 코드가 짧고, 같은 활성값을 여러 번 쓰기 좋다 |

    `Sequential`을 쓸 때는 선택지가 없다 — 모듈이어야 한다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
`model(x)`와 `model.forward(x)`는 같은 값을 준다. 그러면 왜 `model(x)`로
부르라고 하는가?

</div>

??? success "풀이"
    ```python
    m = B()
    x = torch.randn(3, 2)
    print("  값이 같은가:", torch.allclose(m(x), m.forward(x)))

    calls = []
    h = m.register_forward_hook(lambda mod, i, o: calls.append(1))
    m(x)
    m.forward(x)
    h.remove()
    print("  훅이 불린 횟수:", len(calls), "<- m(x) 에서만 불린다")
    ```

    ```
      값이 같은가: True
      훅이 불린 횟수: 1 <- m(x) 에서만 불린다
    ```

    **값은 같은데 훅이 안 불린다.** 두 번 불렀는데 훅은 한 번만 작동했다.

    `model(x)`는 `__call__`을 거치고, 그 안에서 `forward`를 부르기 **전후로**
    등록된 훅들을 실행한다. `forward`를 직접 부르면 그 과정을 통째로 건너뛴다.

    훅에 기대는 기능이 많다.

    - 중간 활성값을 꺼내는 코드(Grad-CAM, 특징 추출)
    - `torch.jit.trace`나 프로파일러
    - 양자화·가지치기 라이브러리가 끼워 넣는 훅

    평소에는 차이가 없다가 **그런 도구를 붙이는 순간 조용히 안 먹는다.**
    그래서 언제나 `model(x)`로 부른다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
`super().__init__()`을 빼먹으면 그 자리에서 터진다. 그런데 **층을 파이썬
리스트에 담으면** 안 터진다. 무엇이 조용히 깨지는지 전부 찾아라.

</div>

??? success "풀이"
    ```python
    class Bad(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = [nn.Linear(2, 2) for _ in range(2)]   # 그냥 리스트

        def forward(self, x):
            for l in self.layers:
                x = l(x)
            return x

    bad = Bad()
    print("  순전파 되는가:", tuple(bad(torch.randn(3, 2)).shape))
    print("  파라미터 수  :", sum(p.numel() for p in bad.parameters()))
    print("  state_dict   :", list(bad.state_dict().keys()))
    bad.to(torch.float64)
    print("  to(float64) 뒤 층의 dtype:", bad.layers[0].weight.dtype)
    ```

    ```
      순전파 되는가: (3, 2)
      파라미터 수  : 0
      state_dict   : []
      to(float64) 뒤 층의 dtype: torch.float32
    ```

    **순전파는 멀쩡히 돈다.** 그리고 나머지가 전부 깨진다.

    | 깨지는 것 | 드러나는 때 |
    |---|---|
    | 파라미터 수가 **0** | 옵티마이저에 안 들어가 **학습이 안 된다** |
    | `state_dict`가 **비어 있다** | 저장했는데 가중치가 없다 |
    | `.to()`가 안 옮긴다 | GPU에서 **device mismatch**로 터진다 |

    셋 다 **한참 뒤에** 드러난다. 학습을 한참 돌리고 "왜 로스가 안 줄지"
    하거나, 저장한 모델을 불러와 보니 무작위이거나, GPU로 옮긴 뒤에야 터진다.

    고치는 법은 `nn.ModuleList`로 감싸는 것 한 줄이다.

    ```python
    self.layers = nn.ModuleList([nn.Linear(2, 2) for _ in range(2)])
    ```

    까닭은 6절과 같다. `nn.Module.__setattr__`이 가로채 등록하는 대상이
    **`nn.Module`과 `nn.Parameter`뿐**이고, 리스트는 그 둘 다 아니다.
    `nn.ModuleList`는 모듈이므로 등록되고, 그 안의 층들을 제 자식으로
    알려 준다.

    사전이면 `nn.ModuleDict`, 파라미터 여럿이면 `nn.ParameterList`가 있다.
    **담는 그릇이 필요하면 `nn.` 이 붙은 것을 쓴다**고 기억하면 된다.
