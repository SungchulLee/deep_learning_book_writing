# 레이어 출력하기 — 무엇이 세어지고 무엇이 저장되는가

`print(model)` 한 줄이 꽤 많은 것을 보여 준다. 그런데 **거기 안 나오는 것**이
있고, 모델을 고치거나 저장할 때 걸리는 것은 대개 그쪽이다.

| | 무엇인가 | 학습되는가 | 저장되는가 |
|---|---|---|---|
| 파라미터 | `nn.Parameter` | **예** | 예 |
| 버퍼 | `register_buffer` 로 등록한 텐서 | 아니오 | **예** |

세 번째 열이 잊히는 자리다. 파라미터 수를 세면 버퍼는 빠지는데, **저장 파일에는
들어간다.** 배치 정규화의 `running_mean`이 거기 있다.

## 1. 코드

```python
"""모델 속 들여다보기 — 무엇이 세어지고 무엇이 저장되는가.

`print(model)` 한 줄이 꽤 많은 것을 보여 주지만, 거기 안 나오는 것이
있다. 모델을 고치거나 저장할 때 알아야 하는 것은 셋이다.

- **모듈 나무** — `named_children()` 은 바로 아래만, `named_modules()`
  는 전부
- **파라미터** — 학습되는 텐서. `parameters()` 가 준다
- **버퍼** — 학습되지는 않는데 **저장은 되는** 텐서. 배치 정규화의
  `running_mean` 이 여기 있다

세 번째가 자주 잊힌다. 파라미터 수를 세면 버퍼는 빠지는데, 저장
파일에는 들어간다.

실행: python 13_print_layers.py
"""
import torch
import torch.nn as nn


def header(title):
    print(f"\n=== {title} ===")


class Block(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.fc = nn.Linear(d, d)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.fc(x))


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Linear(4, 8)
        self.blocks = nn.Sequential(Block(8), Block(8))
        self.head = nn.Linear(8, 2)
        self.register_buffer("steps", torch.zeros(1))

    def forward(self, x):
        return self.head(self.blocks(self.stem(x)))


# ========================================================================
# 1. print(model)
# ========================================================================

def show_repr(net):
    header("1. print(model)")
    print(net)
    print("  들여쓰기가 중첩을 보여 준다")
    print("  그런데 파라미터 수도, 출력 모양도, 버퍼도 안 나온다")


# ========================================================================
# 2. 바로 아래만 vs 전부
# ========================================================================

def tree(net):
    header("2. children 과 modules")
    print("  named_children (바로 아래만):",
          [n for n, _ in net.named_children()])
    print(f"  named_modules  (전부 {len(list(net.modules()))}개):")
    for name, m in net.named_modules():
        depth = name.count(".") + (0 if name == "" else 1)
        print(f"    {'  ' * depth}{name or '(root)':20s} {type(m).__name__}")
    print("  modules() 는 자기 자신도 센다 — 그래서 하나 더 많다")
    print("  이름이 점으로 이어진 꼴이 그대로 state_dict 의 키가 된다")


# ========================================================================
# 3. 파라미터와 버퍼
# ========================================================================

def params_and_buffers(net):
    header("3. 파라미터와 버퍼")
    for name, p in net.named_parameters():
        print(f"  param  {name:22s} {str(tuple(p.shape)):10s} "
              f"requires_grad={p.requires_grad}")
    for name, b in net.named_buffers():
        print(f"  buffer {name:22s} {str(tuple(b.shape)):10s} "
              f"requires_grad={b.requires_grad}")
    print("  파라미터 수:", sum(p.numel() for p in net.parameters()))
    print("  버퍼  수   :", sum(b.numel() for b in net.buffers()),
          "<- 위 셈에 안 들어간다")
    print("  state_dict 키:")
    for k in net.state_dict():
        print("   ", k)
    print("  버퍼가 맨 앞에 끼어 있다. 저장은 되고 학습은 안 된다")


# ========================================================================
# 4. 실제로 버퍼가 쓰이는 자리
# ========================================================================

def real_buffer():
    header("4. 배치 정규화의 버퍼")
    bn = nn.BatchNorm1d(4)
    print("  파라미터:", [n for n, _ in bn.named_parameters()])
    print("  버퍼    :", [n for n, _ in bn.named_buffers()])
    before = bn.running_mean.clone()
    bn.train()
    bn(torch.randn(8, 4))
    print("  순전파 한 번 뒤 running_mean 이 바뀌었나:",
          not torch.equal(before, bn.running_mean))
    print("  경사는 없다:", bn.running_mean.grad_fn, bn.running_mean.requires_grad)
    print("  학습되지 않는데 순전파가 고친다 — 그래서 버퍼다")


# ========================================================================
# 5. 쓸모 있는 요약 한 줄
# ========================================================================

def summary(net):
    header("5. 층마다 수를 세어 본다")
    print(f"  {'이름':24s} {'종류':12s} {'파라미터':>10s}")
    total = 0
    for name, m in net.named_modules():
        own = sum(p.numel() for p in m.parameters(recurse=False))
        if own or not list(m.children()):
            print(f"  {name or '(root)':24s} {type(m).__name__:12s} {own:>10,}")
            total += own
    print(f"  {'합':24s} {'':12s} {total:>10,}")
    print("  recurse=False 로 그 층이 '제 몫으로' 가진 것만 센다")
    print("  안 그러면 부모가 자식 것을 다시 세어 합이 부풀어 오른다")


def main():
    torch.manual_seed(0)
    net = Net()
    show_repr(net)
    tree(net)
    params_and_buffers(net)
    real_buffer()
    summary(net)


if __name__ == "__main__":
    main()
```

**출력:**

```
=== 1. print(model) ===
Net(
  (stem): Linear(in_features=4, out_features=8, bias=True)
  (blocks): Sequential(
    (0): Block(
      (fc): Linear(in_features=8, out_features=8, bias=True)
      (act): ReLU()
    )
    (1): Block(
      (fc): Linear(in_features=8, out_features=8, bias=True)
      (act): ReLU()
    )
  )
  (head): Linear(in_features=8, out_features=2, bias=True)
)
  들여쓰기가 중첩을 보여 준다
  그런데 파라미터 수도, 출력 모양도, 버퍼도 안 나온다

=== 2. children 과 modules ===
  named_children (바로 아래만): ['stem', 'blocks', 'head']
  named_modules  (전부 10개):
    (root)               Net
      stem                 Linear
      blocks               Sequential
        blocks.0             Block
          blocks.0.fc          Linear
          blocks.0.act         ReLU
        blocks.1             Block
          blocks.1.fc          Linear
          blocks.1.act         ReLU
      head                 Linear
  modules() 는 자기 자신도 센다 — 그래서 하나 더 많다
  이름이 점으로 이어진 꼴이 그대로 state_dict 의 키가 된다

=== 3. 파라미터와 버퍼 ===
  param  stem.weight            (8, 4)     requires_grad=True
  param  stem.bias              (8,)       requires_grad=True
  param  blocks.0.fc.weight     (8, 8)     requires_grad=True
  param  blocks.0.fc.bias       (8,)       requires_grad=True
  param  blocks.1.fc.weight     (8, 8)     requires_grad=True
  param  blocks.1.fc.bias       (8,)       requires_grad=True
  param  head.weight            (2, 8)     requires_grad=True
  param  head.bias              (2,)       requires_grad=True
  buffer steps                  (1,)       requires_grad=False
  파라미터 수: 202
  버퍼  수   : 1 <- 위 셈에 안 들어간다
  state_dict 키:
    steps
    stem.weight
    stem.bias
    blocks.0.fc.weight
    blocks.0.fc.bias
    blocks.1.fc.weight
    blocks.1.fc.bias
    head.weight
    head.bias
  버퍼가 맨 앞에 끼어 있다. 저장은 되고 학습은 안 된다

=== 4. 배치 정규화의 버퍼 ===
  파라미터: ['weight', 'bias']
  버퍼    : ['running_mean', 'running_var', 'num_batches_tracked']
  순전파 한 번 뒤 running_mean 이 바뀌었나: True
  경사는 없다: None False
  학습되지 않는데 순전파가 고친다 — 그래서 버퍼다

=== 5. 층마다 수를 세어 본다 ===
  이름                       종류                 파라미터
  stem                     Linear               40
  blocks.0.fc              Linear               72
  blocks.0.act             ReLU                  0
  blocks.1.fc              Linear               72
  blocks.1.act             ReLU                  0
  head                     Linear               18
  합                                            202
  recurse=False 로 그 층이 '제 몫으로' 가진 것만 센다
  안 그러면 부모가 자식 것을 다시 세어 합이 부풀어 오른다
```

## 2. `print(model)`이 보여 주는 것과 안 보여 주는 것

```
Net(
  (stem): Linear(in_features=4, out_features=8, bias=True)
  (blocks): Sequential(
    (0): Block(
      (fc): Linear(in_features=8, out_features=8, bias=True)
      (act): ReLU()
    )
    ...
```

들여쓰기가 중첩을 그대로 보여 주고, 각 층의 설정도 보인다. 모델을 처음 볼 때
가장 빠른 길이다.

안 나오는 것이 셋이다. **파라미터 수**, **출력 모양**, **버퍼**. 앞의 둘은 따로
세어야 하고(5절), 셋째는 아예 다른 목록에 있다(4절).

## 3. 바로 아래만 vs 전부

```
named_children (바로 아래만): ['stem', 'blocks', 'head']
named_modules  (전부 10개):
  (root)               Net
    stem               Linear
    blocks             Sequential
      blocks.0         Block
        blocks.0.fc    Linear
        ...
```

`children()`은 **한 겹만**, `modules()`는 **나무 전체**를 준다. 그리고
`modules()`는 **자기 자신도 센다** — 그래서 10개 중 하나가 루트다.

이름이 점으로 이어진 꼴에 주목하자. `blocks.0.fc`가 그대로 `state_dict`의 키
앞부분이 된다. 그래서 **모델의 속성 이름을 바꾸면 옛 체크포인트를 못 읽는다.**
「모델 체크포인트」에서 구조가 안 저장된다고 한 것의 다른 얼굴이다.

고르는 기준은 간단하다. 층마다 무언가를 하려면 `modules()`, 큰 덩이 단위로
다루려면 `children()`.

## 4. 버퍼 — 학습되지 않는데 순전파가 고친다

```
파라미터: ['weight', 'bias']
버퍼    : ['running_mean', 'running_var', 'num_batches_tracked']
순전파 한 번 뒤 running_mean 이 바뀌었나: True
경사는 없다: None False
```

배치 정규화가 버퍼의 본보기다. `running_mean`은

- `requires_grad=False`이고 `grad_fn`도 없다 — **학습되지 않는다**
- 그런데 **순전파가 고친다** — 배치 통계를 이동평균으로 쌓는다
- `state_dict`에 들어간다 — **저장된다**

세 성질이 다 필요해서 버퍼라는 갈래가 따로 있다. 파라미터로 두면 옵티마이저가
건드리려 들고, 그냥 속성으로 두면 저장도 장치 이동도 안 된다.

```
state_dict 키:
  steps                 <- 버퍼가 맨 앞에
  stem.weight
  ...
```

그래서 **파라미터 수와 저장 파일 크기가 안 맞는다.** 수를 셀 때는 빠지고 저장할
때는 들어가므로, 둘을 견주려면 버퍼도 더해야 한다.

## 5. 층마다 세기 — `recurse=False`가 필요하다

```
이름                       종류                 파라미터
stem                     Linear               40
blocks.0.fc              Linear               72
blocks.0.act             ReLU                  0
blocks.1.fc              Linear               72
blocks.1.act             ReLU                  0
head                     Linear               18
합                                            202
```

`m.parameters()`는 **자식 것까지** 준다. 그래서 모든 모듈을 돌며 더하면 같은
파라미터를 여러 번 센다 — 루트가 전부를, `blocks`가 블록 둘을, 각 블록이 제
것을 다시.

`recurse=False`를 주면 **그 층이 제 몫으로 가진 것**만 센다. 그러면 합이
맞는다. 연습문제 1에서 틀리면 얼마나 부풀어 오르는지 본다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
`recurse=False`를 빼면 합이 얼마나 부풀어 오르는가? 세 층짜리 그물로 확인하라.

</div>

??? success "풀이"
    ```python
    net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
    print("  recurse=True  로 더한 합:",
          sum(sum(p.numel() for p in m.parameters()) for m in net.modules()))
    print("  recurse=False 로 더한 합:",
          sum(sum(p.numel() for p in m.parameters(recurse=False))
              for m in net.modules()))
    print("  올바른 값             :", sum(p.numel() for p in net.parameters()))
    ```

    ```
      recurse=True  로 더한 합: 116
      recurse=False 로 더한 합: 58
      올바른 값             : 58
    ```

    **정확히 두 배**가 되었다. `Sequential` 자신이 58개를 전부 세고, 그 아래
    `Linear` 둘이 각자 제 것을 다시 세었기 때문이다.

    중첩이 깊으면 배수가 커진다. 블록 안에 블록이 있는 구조라면 세 배, 네 배가
    된다.

    가장 간단한 길은 **층마다 세지 않는 것**이다. 전체 수는
    `sum(p.numel() for p in net.parameters())` 한 줄이면 맞는다. 층별 내역이
    필요할 때만 `recurse=False`가 필요하다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
층을 리스트에 담아 두면 모델이 알아보는가? 파이썬 리스트와 `nn.ModuleList`를
둘 다 가진 모듈을 만들어 확인하라.

</div>

??? success "풀이"
    ```python
    class WithList(nn.Module):
        def __init__(self):
            super().__init__()
            self.good = nn.ModuleList([nn.Linear(2, 2) for _ in range(3)])
            self.bad = [nn.Linear(2, 2) for _ in range(3)]

    m = WithList()
    print("  ModuleList 의 파라미터:", len(list(m.good.parameters())))
    print("  전체 파라미터 수      :", len(list(m.parameters())))
    print("  state_dict 키:", list(m.state_dict().keys()))
    ```

    ```
      ModuleList 의 파라미터: 6
      전체 파라미터 수      : 6
      state_dict 키: ['good.0.weight', 'good.0.bias', 'good.1.weight',
                      'good.1.bias', 'good.2.weight', 'good.2.bias']
    ```

    **`bad`는 아예 없는 것처럼 취급된다.** 전체 파라미터가 6개뿐이고
    `state_dict`에 `bad`가 한 줄도 없다.

    그냥 파이썬 리스트에 담으면 `nn.Module`이 그것을 등록하지 못한다.
    `__setattr__`이 가로채 등록하는 대상이 `nn.Module`과 `nn.Parameter`뿐이기
    때문이다.

    결과가 고약하다. **순전파는 멀쩡히 돌아간다** — 리스트에서 꺼내 쓰면 되니까.
    그런데

    - 옵티마이저에 안 들어가므로 **학습되지 않는다**
    - `model.to(device)`가 안 옮기므로 **장치가 어긋나 터진다**
    - `state_dict`에 없으므로 **저장도 복원도 안 된다**

    셋 다 한참 뒤에 드러난다. 층을 여럿 담을 때는 `nn.ModuleList`나
    `nn.Sequential`, 사전이면 `nn.ModuleDict`를 써야 한다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
배치 정규화의 버퍼가 **언제** 갱신되는가? `train()`과 `eval()`에서 각각 세 번씩
순전파하고 `running_mean`과 `num_batches_tracked`를 보라.

</div>

??? success "풀이"
    ```python
    bn = nn.BatchNorm1d(3)
    print("  처음           :", [round(v, 4) for v in bn.running_mean.tolist()],
          " num_batches:", bn.num_batches_tracked.item())
    bn.train()
    for _ in range(3):
        bn(torch.randn(8, 3))
    print("  train 3번 뒤   :", [round(v, 4) for v in bn.running_mean.tolist()],
          " num_batches:", bn.num_batches_tracked.item())
    bn.eval()
    for _ in range(3):
        bn(torch.randn(8, 3))
    print("  eval 3번 뒤    :", [round(v, 4) for v in bn.running_mean.tolist()],
          " num_batches:", bn.num_batches_tracked.item())
    ```

    ```
      처음           : [0.0, 0.0, 0.0]  num_batches: 0
      train 3번 뒤   : [0.0658, 0.1429, -0.027]  num_batches: 3
      eval 3번 뒤    : [0.0658, 0.1429, -0.027]  num_batches: 3
      eval 에서는 안 바뀐다
    ```

    **`train()`에서만 갱신된다.** `eval()`에서는 쌓아 둔 통계를 **읽기만** 한다.

    그래서 `model.eval()`이 하는 일이 둘이다 — 드롭아웃을 끄고, 배치 정규화를
    "쌓기"에서 "읽기"로 바꾼다. 「torch.no_grad 사용하기」에서 `eval()`을
    빼먹으면 답이 틀린다고 한 까닭이 여기까지 이어진다. 평가 중에 `train()`
    모드면 **평가 데이터의 통계가 모델에 스며든다** — 시험 문제를 보고 모델을
    고치는 셈이다.

    `num_batches_tracked`가 따로 있는 것도 뜻이 있다. `momentum=None`으로 두면
    이동평균 대신 **누적 평균**을 쓰는데, 그때 몇 번째인지 알아야 하기
    때문이다.

    > 평가 전에 `model.eval()`을 부르고 **평가가 끝나면 `model.train()`으로
    > 되돌리는** 것을 잊기 쉽다. 되돌리지 않으면 그다음 학습이 통째로 `eval()`
    > 모드에서 돌아가고, 드롭아웃이 꺼진 채 배치 정규화 통계도 안 쌓인다.
    > 터지지 않고 결과만 나빠지는 종류다.
