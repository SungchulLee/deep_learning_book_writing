# 파라미터 얼리고 녹이기 — 미세조정의 실제 꼴

「경사 추적 끄기」에서 `requires_grad_(False)`가 무엇을 하는지 보았다. 이 쪽은
그것을 **학습 과정에 어떻게 엮는지**를 본다.

| 묻는 것 | |
|---|---|
| 얼린 채로 돌리다가 **중간에 녹이려면** | 옵티마이저를 어떻게 하는가 |
| 아예 얼리지 않고 | **층마다 다른 러닝레이트**를 주는 길 |

둘째 쪽이 실전에서 더 자주 쓰인다. 얼리기는 러닝레이트를 0으로 두는 것과
같고, 0과 평소 값 사이에 쓸 만한 자리가 넓기 때문이다.

## 1. 코드

```python
"""얼리고 녹이기 — 미세조정의 실제 꼴.

「경사 추적 끄기」에서 `requires_grad_(False)` 가 무엇을 하는지 보았다.
이 쪽은 그것을 **학습 과정에 어떻게 엮는지**를 본다.

- 얼린 채로 돌리다가 **중간에 녹이려면** 옵티마이저를 어떻게 하는가
- 아예 얼리지 않고 **층마다 다른 러닝레이트**를 주는 길
- 둘을 견주면 어느 쪽이 무엇에 쓸모 있는가

실행: python 16_freeze_and_unfreeze_parameters.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def header(title):
    print(f"\n=== {title} ===")


def make(seed=0):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))


def data(seed=1):
    torch.manual_seed(seed)
    return torch.randn(16, 4), torch.randn(16, 2)


def moved(net, before):
    return [round((p.detach() - b).abs().max().item(), 6)
            for p, b in zip(net.parameters(), before)]


# ========================================================================
# 1. 얼리기 — 깃발과 옵티마이저
# ========================================================================

def freeze_basics():
    header("1. 얼린 층은 안 움직인다")
    net, (x, y) = make(), data()
    for p in net[0].parameters():
        p.requires_grad_(False)
    opt = torch.optim.SGD(net.parameters(), lr=0.1)
    before = [p.detach().clone() for p in net.parameters()]
    opt.zero_grad()
    F.mse_loss(net(x), y).backward()
    opt.step()
    print("  움직인 거리:", moved(net, before))
    print("  앞 두 개가 0 이다 — 얼린 층의 weight 와 bias")
    print("  opt 에 전부 넘겼는데도 안 움직였다. 경사가 None 이라 건너뛴다")


# ========================================================================
# 2. 녹이기 — 옵티마이저를 어떻게 하는가
# ========================================================================

def unfreeze():
    header("2. 녹일 때 옵티마이저")
    x, y = data()

    for label, build_opt in (
        ("거르고 만든 옵티마이저", lambda n: torch.optim.SGD(
            [p for p in n.parameters() if p.requires_grad], lr=0.1)),
        ("전부 넘긴 옵티마이저", lambda n: torch.optim.SGD(
            n.parameters(), lr=0.1)),
    ):
        net = make()
        for p in net[0].parameters():
            p.requires_grad_(False)
        opt = build_opt(net)
        for p in net[0].parameters():        # 녹인다
            p.requires_grad_(True)
        before = [p.detach().clone() for p in net.parameters()]
        opt.zero_grad()
        F.mse_loss(net(x), y).backward()
        opt.step()
        print(f"  {label:22s} 움직인 거리 {moved(net, before)}")
    print("  거르고 만든 쪽은 녹여도 앞 두 개가 0 이다")
    print("  옵티마이저가 그 텐서를 애초에 안 들고 있기 때문이다")


# ========================================================================
# 3. 고치는 두 길
# ========================================================================

def two_fixes():
    header("3. 고치는 두 길")
    x, y = data()

    # (가) 옵티마이저를 다시 만든다
    net = make()
    for p in net[0].parameters():
        p.requires_grad_(False)
    opt = torch.optim.SGD([p for p in net.parameters() if p.requires_grad],
                          lr=0.1, momentum=0.9)
    opt.zero_grad(); F.mse_loss(net(x), y).backward(); opt.step()
    carried = len(opt.state)
    for p in net[0].parameters():
        p.requires_grad_(True)
    opt = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9)
    print(f"  (가) 다시 만들기: 모멘텀 버퍼 {carried}개 -> {len(opt.state)}개")
    print("       쌓아 둔 모멘텀을 잃는다")

    # (나) 집어넣는다
    net2 = make()
    for p in net2[0].parameters():
        p.requires_grad_(False)
    opt2 = torch.optim.SGD([p for p in net2.parameters() if p.requires_grad],
                           lr=0.1, momentum=0.9)
    opt2.zero_grad(); F.mse_loss(net2(x), y).backward(); opt2.step()
    kept = len(opt2.state)
    for p in net2[0].parameters():
        p.requires_grad_(True)
    opt2.add_param_group({"params": list(net2[0].parameters()), "lr": 0.01})
    print(f"  (나) add_param_group: 모멘텀 버퍼 {kept}개 그대로")
    print(f"       그룹 {len(opt2.param_groups)}개, lr {[g['lr'] for g in opt2.param_groups]}")
    print("       녹인 층에 작은 lr 을 따로 줄 수도 있다")


# ========================================================================
# 4. 얼리지 않고 러닝레이트만 낮추기
# ========================================================================

def layerwise_lr():
    header("4. 층마다 다른 러닝레이트")
    net, (x, y) = make(), data()
    opt = torch.optim.SGD([
        {"params": net[0].parameters(), "lr": 0.001},
        {"params": net[2].parameters(), "lr": 0.1},
    ])
    before = [p.detach().clone() for p in net.parameters()]
    opt.zero_grad()
    F.mse_loss(net(x), y).backward()
    opt.step()
    print("  움직인 거리:", moved(net, before))
    print("  앞쪽이 0 은 아닌데 백 배 작다")
    print("  얼리기가 lr=0 이라면, 이것은 아주 작은 lr 이다")
    print("  사전학습 특징을 지키면서도 조금은 맞춰 간다")


# ========================================================================
# 5. 셋을 한자리에
# ========================================================================

def compare_all():
    header("5. 세 가지를 나란히")
    x, y = data()
    rows = []

    net = make()
    for p in net[0].parameters():
        p.requires_grad_(False)
    opt = torch.optim.SGD(net.parameters(), lr=0.1)
    before = [p.detach().clone() for p in net.parameters()]
    opt.zero_grad(); F.mse_loss(net(x), y).backward(); opt.step()
    rows.append(("얼리기", moved(net, before)))

    net = make()
    opt = torch.optim.SGD([{"params": net[0].parameters(), "lr": 0.001},
                           {"params": net[2].parameters(), "lr": 0.1}])
    before = [p.detach().clone() for p in net.parameters()]
    opt.zero_grad(); F.mse_loss(net(x), y).backward(); opt.step()
    rows.append(("작은 lr", moved(net, before)))

    net = make()
    opt = torch.optim.SGD(net.parameters(), lr=0.1)
    before = [p.detach().clone() for p in net.parameters()]
    opt.zero_grad(); F.mse_loss(net(x), y).backward(); opt.step()
    rows.append(("전부 학습", moved(net, before)))

    print(f"  {'':10s} {'0.weight':>10s} {'0.bias':>10s} "
          f"{'2.weight':>10s} {'2.bias':>10s}")
    for name, m in rows:
        print(f"  {name:10s} {m[0]:>10.6f} {m[1]:>10.6f} "
              f"{m[2]:>10.6f} {m[3]:>10.6f}")
    print("  뒤쪽 층은 셋 다 같다 — 앞쪽을 어떻게 하든 영향이 없다")


def main():
    freeze_basics()
    unfreeze()
    two_fixes()
    layerwise_lr()
    compare_all()


if __name__ == "__main__":
    main()
```

**출력:**

```
=== 1. 얼린 층은 안 움직인다 ===
  움직인 거리: [0.0, 0.0, 0.012712, 0.03497]
  앞 두 개가 0 이다 — 얼린 층의 weight 와 bias
  opt 에 전부 넘겼는데도 안 움직였다. 경사가 None 이라 건너뛴다

=== 2. 녹일 때 옵티마이저 ===
  거르고 만든 옵티마이저           움직인 거리 [0.0, 0.0, 0.012712, 0.03497]
  전부 넘긴 옵티마이저            움직인 거리 [0.010357, 0.007619, 0.012712, 0.03497]
  거르고 만든 쪽은 녹여도 앞 두 개가 0 이다
  옵티마이저가 그 텐서를 애초에 안 들고 있기 때문이다

=== 3. 고치는 두 길 ===
  (가) 다시 만들기: 모멘텀 버퍼 2개 -> 0개
       쌓아 둔 모멘텀을 잃는다
  (나) add_param_group: 모멘텀 버퍼 2개 그대로
       그룹 2개, lr [0.1, 0.01]
       녹인 층에 작은 lr 을 따로 줄 수도 있다

=== 4. 층마다 다른 러닝레이트 ===
  움직인 거리: [0.000104, 7.6e-05, 0.012712, 0.03497]
  앞쪽이 0 은 아닌데 백 배 작다
  얼리기가 lr=0 이라면, 이것은 아주 작은 lr 이다
  사전학습 특징을 지키면서도 조금은 맞춰 간다

=== 5. 세 가지를 나란히 ===
               0.weight     0.bias   2.weight     2.bias
  얼리기          0.000000   0.000000   0.012712   0.034970
  작은 lr        0.000104   0.000076   0.012712   0.034970
  전부 학습        0.010357   0.007619   0.012712   0.034970
  뒤쪽 층은 셋 다 같다 — 앞쪽을 어떻게 하든 영향이 없다
```

## 2. 얼린 층은 옵티마이저에 넘겨도 안 움직인다

```
움직인 거리: [0.0, 0.0, 0.012712, 0.03497]
```

`net.parameters()`를 통째로 넘겼는데 앞 두 개가 정확히 0이다. 경사가 `None`이라
`step()`이 건너뛰기 때문이다.

그래서 **거르지 않아도 된다.** 거르는 쪽이 메모리를 조금 아끼지만, 다음 절을
보면 안 거르는 쪽이 편한 자리가 있다.

## 3. 녹일 때 걸리는 자리

```
거르고 만든 옵티마이저   움직인 거리 [0.0, 0.0, 0.012712, 0.03497]
전부 넘긴 옵티마이저    움직인 거리 [0.010357, 0.007619, 0.012712, 0.03497]
```

**깃발을 녹였는데 거른 쪽은 여전히 0이다.**

옵티마이저는 만들 때 받은 텐서 목록을 들고 있고, 그 목록에 없는 것은 영영 안
걷는다. `requires_grad`를 되돌려 경사는 오는데 걷는 사람이 없다.

오류가 안 난다는 점이 나쁘다. 단계적으로 녹이는 학습에서 **녹인 층이 안
배우는 채로 끝까지 간다.**

## 4. 고치는 두 길

```
(가) 다시 만들기: 모멘텀 버퍼 2개 -> 0개
     쌓아 둔 모멘텀을 잃는다
(나) add_param_group: 모멘텀 버퍼 2개 그대로
     그룹 2개, lr [0.1, 0.01]
```

**(가) 옵티마이저를 다시 만든다.** 간단하지만 쌓아 둔 상태를 전부 버린다.
모멘텀 버퍼가 2개에서 0개가 되었다. Adam이라면 1·2차 모멘트가 모두 날아가고,
「모델 체크포인트」에서 본 대로 **첫 걸음이 1.64배 커진다.**

**(나) `add_param_group`으로 집어넣는다.** 기존 상태가 그대로 남고, 덤으로 녹인
층에 **다른 러닝레이트**를 줄 수 있다. 사전학습된 층을 녹일 때는 작은 값을
주는 것이 보통이므로 이쪽이 맞다.

```python
opt.add_param_group({"params": net[0].parameters(), "lr": 0.01})
```

> 애초에 `net.parameters()`를 통째로 넘겨 두었다면 이 문제가 없다. 얼린 동안은
> 경사가 `None`이라 건너뛰고, 녹이면 저절로 걷기 시작한다. 러닝레이트를 따로
> 줄 필요가 없다면 그쪽이 가장 간단하다.

## 5. 얼리지 않고 러닝레이트만 낮추기

```
움직인 거리: [0.000104, 7.6e-05, 0.012712, 0.03497]
```

앞쪽이 0은 아닌데 **백 배 작다.** `lr=0.001`과 `lr=0.1`의 비 그대로다.

```python
opt = torch.optim.SGD([
    {"params": net[0].parameters(), "lr": 0.001},
    {"params": net[2].parameters(), "lr": 0.1},
])
```

**얼리기가 $\eta = 0$이라면 이것은 아주 작은 $\eta$다.** 사전학습된 특징을 거의
지키면서도 새 데이터에 조금씩 맞춰 간다.

## 6. 셋을 나란히

```
             0.weight     0.bias   2.weight     2.bias
얼리기        0.000000   0.000000   0.012712   0.034970
작은 lr      0.000104   0.000076   0.012712   0.034970
전부 학습      0.010357   0.007619   0.012712   0.034970
```

앞쪽 열이 **0 → 0.0001 → 0.0104**로 벌어진다. 연속된 눈금 위의 세 점이다.

뒤쪽 두 열은 **셋 다 똑같다.** 앞쪽을 어떻게 하든 뒤쪽 층이 받는 경사는 그대로다
— 「경사 추적 끄기」에서 본 대로 얼린 층도 경사를 지나보내기 때문이다.

실전의 쓰임을 정리하면 이렇다.

| | 쓸 자리 |
|---|---|
| **얼리기** | 데이터가 아주 적을 때, 특징 추출기로만 쓸 때 |
| **작은 lr** | 미세조정의 기본. 데이터가 어느 정도 있을 때 |
| **전부 같은 lr** | 데이터가 많고 영역이 많이 다를 때 |

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
`Linear(784,256) → Linear(256,128) → Linear(128,10)` 그물에서 앞 두 층을 얼리면
학습되는 파라미터가 몇 퍼센트 남는가?

</div>

??? success "풀이"
    ```python
    net = nn.Sequential(nn.Linear(784, 256), nn.ReLU(),
                        nn.Linear(256, 128), nn.ReLU(),
                        nn.Linear(128, 10))

    def summary(m):
        t = sum(p.numel() for p in m.parameters())
        tr = sum(p.numel() for p in m.parameters() if p.requires_grad)
        return t, tr, 100 * tr / t

    print("  처음        : 전체 {:,}  학습 {:,}  ({:.1f}%)".format(*summary(net)))
    for layer in (net[0], net[2]):
        for p in layer.parameters():
            p.requires_grad_(False)
    print("  앞 둘 얼림  : 전체 {:,}  학습 {:,}  ({:.1f}%)".format(*summary(net)))

    opt = torch.optim.SGD([p for p in net.parameters() if p.requires_grad],
                          lr=0.1, momentum=0.9)
    net(torch.randn(8, 784)).sum().backward()
    opt.step()
    print("  옵티마이저 상태 버퍼 수:", len(opt.state))
    ```

    ```
      처음        : 전체 235,146  학습 235,146  (100.0%)
      앞 둘 얼림  : 전체 235,146  학습 1,290  (0.5%)
      옵티마이저 상태 버퍼 수: 2
    ```

    **0.5%만 남는다.** 23만 개 중 1,290개다.

    모델의 무게가 앞쪽에 몰려 있기 때문이다 — `Linear(784, 256)` 하나가 20만
    개를 넘는다. 「실전 예제」에서 `fc1`이 전체의 91%를 차지한 것과 같은 모양이다.

    옵티마이저 상태가 **2개**인 것도 보아 둘 만하다. 거르고 넘겼으므로 학습되는
    텐서 수만큼만 버퍼를 만든다. 전부 넘겼다면 6개가 되어 모멘텀 버퍼로 모델
    크기만큼을 더 썼을 것이다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
층이 셋인 그물에 층마다 다른 러닝레이트(0.001 / 0.01 / 0.1)를 주고, 움직인
거리가 그 비를 따르는지 확인하라. 그리고 **학습 도중에 전부 열 배로** 올려라.

</div>

??? success "풀이"
    ```python
    net = nn.Sequential(nn.Linear(4, 8), nn.Linear(8, 8), nn.Linear(8, 2))
    opt = torch.optim.SGD([{"params": l.parameters(), "lr": lr}
                           for l, lr in zip(net, (0.001, 0.01, 0.1))])
    before = [p.detach().clone() for p in net.parameters()]
    opt.zero_grad()
    F.mse_loss(net(x), y).backward()
    opt.step()
    for i, (p, b) in enumerate(zip(net.parameters(), before)):
        print(f"  파라미터 {i} (층 {i // 2}, lr {[0.001, 0.01, 0.1][i // 2]})  "
              f"움직임 {(p.detach() - b).abs().max():.6f}")

    for g in opt.param_groups:
        g["lr"] *= 10
    print("  lr 을 열 배로:", [g["lr"] for g in opt.param_groups])
    ```

    ```
      파라미터 0 (층 0, lr 0.001)  움직임 0.000051
      파라미터 1 (층 0, lr 0.001)  움직임 0.000019
      파라미터 2 (층 1, lr 0.01)  움직임 0.000698
      파라미터 3 (층 1, lr 0.01)  움직임 0.000329
      파라미터 4 (층 2, lr 0.1)  움직임 0.009783
      파라미터 5 (층 2, lr 0.1)  움직임 0.009042
      lr 을 열 배로: [0.01, 0.1, 1.0]
    ```

    **열 배씩 벌어진다** — 0.00005, 0.0007, 0.0098. 다만 정확히 열 배는 아닌데,
    층마다 경사의 크기 자체가 다르기 때문이다. 움직인 거리는 $\eta \times
    |\nabla|$이므로 둘 다 작용한다.

    `param_groups`의 `lr`을 직접 고치면 학습 도중에 바꿀 수 있다. **스케줄러가
    하는 일이 정확히 이것**이고, 그래서 스케줄러도 그룹마다 다른 값을 유지한다.

    층이 깊어질수록 러닝레이트를 키우는 이 꼴을 **구별 학습률**(discriminative
    learning rates)이라 부른다. 사전학습된 특징은 조금만, 새로 단 머리는 크게
    움직이게 한다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
학습 **도중에** 층을 얼리면 옵티마이저에 쌓인 모멘텀은 어떻게 되는가? 얼린
층이 그 모멘텀으로 계속 움직이지는 않는가?

</div>

??? success "풀이"
    ```python
    net = nn.Sequential(nn.Linear(4, 8), nn.Linear(8, 2))
    opt = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9)
    for _ in range(3):
        opt.zero_grad()
        F.mse_loss(net(x), y).backward()
        opt.step()
    buf = opt.state[net[0].weight]["momentum_buffer"].abs().max().item()
    print(f"  얼리기 전 0층 모멘텀 버퍼 최대 {buf:.6f}")

    for p in net[0].parameters():
        p.requires_grad_(False)
    w_before = net[0].weight.detach().clone()
    for _ in range(3):
        opt.zero_grad()
        F.mse_loss(net(x), y).backward()
        opt.step()
    print("  얼린 뒤 0층이 움직였나:",
          not torch.equal(net[0].weight.detach(), w_before))
    print("  버퍼가 남아 있나:",
          "momentum_buffer" in opt.state.get(net[0].weight, {}))
    ```

    ```
      얼리기 전 0층 모멘텀 버퍼 최대 0.433993
      얼린 뒤 0층이 움직였나: False
      버퍼가 남아 있나: True
    ```

    **움직이지 않는다.** 모멘텀 버퍼에 0.43이 쌓여 있는데도 그렇다.

    까닭은 `step()`이 **경사가 `None`인 파라미터를 아예 건너뛰기** 때문이다.
    버퍼를 읽지도 쓰지도 않고 지나간다. 「경사 누적」에서 `set_to_none=True`와
    0으로 채우기가 갈렸던 것이 바로 이 분기다 — 0으로 채워 두었다면 `step()`이
    들어가서 모멘텀으로 밀었을 것이다.

    버퍼는 **남아 있다.** 나중에 녹이면 그 모멘텀이 그대로 되살아난다. 좋을
    수도 나쁠 수도 있다 — 얼려 둔 동안 다른 층이 많이 바뀌었다면, 옛 모멘텀은
    이제 맞지 않는 방향일 수 있다.

    깔끔하게 하려면 녹일 때 그 파라미터의 상태를 비우면 된다.

    ```python
    for p in net[0].parameters():
        opt.state.pop(p, None)       # 쌓인 모멘텀을 버린다
        p.requires_grad_(True)
    ```
