"""requires_grad 깃발 — 얼리기와, 얼려도 흐르는 것.

`requires_grad` 는 **그 텐서의 속성**이다. 켜고 끄면 영구적이고,
블록을 나온다고 돌아오지 않는다. 전이학습에서 층을 얼리는 일이
이 깃발 하나로 끝난다.

조심할 것이 두 가지다.

- **잎에만 바꿀 수 있다.** 계산으로 생긴 텐서에는 못 건다
- **얼린 층도 경사를 지나보낸다.** 쌓지 않을 뿐이다. 그래서 중간 층을
  얼려도 그 앞 층은 멀쩡히 배운다

실행: python 08_turn_off_gradient_tracking.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def header(title):
    print(f"\n=== {title} ===")


# ========================================================================
# 1. 어디에 걸 수 있는가
# ========================================================================

def where_allowed():
    header("1. 깃발을 걸 수 있는 텐서")
    a = torch.randn(3)
    print("  잎에 켜기:", a.requires_grad_(True).requires_grad)
    print("  만들 때 켜기:", torch.randn(3, requires_grad=True).requires_grad)

    b = torch.randn(3, requires_grad=True) * 2        # 계산으로 생긴 것
    try:
        b.requires_grad_(False)
    except RuntimeError as e:
        print("  잎이 아닌 것에:", str(e).splitlines()[0][:66])

    c = torch.randint(0, 5, (3,))
    try:
        c.requires_grad_(True)
    except RuntimeError as e:
        print("  정수 텐서에  :", str(e).splitlines()[0][:60])
    print("  미분은 실수에서만 뜻이 있으므로 정수에는 못 건다")


# ========================================================================
# 2. 층을 얼린다
# ========================================================================

def freezing():
    header("2. 층 얼리기")
    torch.manual_seed(0)
    net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
    for p in net[0].parameters():
        p.requires_grad_(False)                       # 앞쪽 층을 언다

    x, y = torch.randn(16, 4), torch.randn(16, 1)
    opt = torch.optim.SGD(net.parameters(), lr=0.1)   # 전부 넘겨도 된다
    before = [p.detach().clone() for p in net.parameters()]
    opt.zero_grad()
    F.mse_loss(net(x), y).backward()
    opt.step()
    for i, (p, b0) in enumerate(zip(net.parameters(), before)):
        print(f"  파라미터 {i}  requires_grad={p.requires_grad!s:5s} "
              f"grad={'None' if p.grad is None else '있음'}  "
              f"바뀌었나 {not torch.equal(p.detach(), b0)}")
    print("  얼린 둘은 경사도 안 받고 값도 안 바뀌었다")
    print("  옵티마이저에 전부 넘겨도 되지만, 거르는 편이 깔끔하다")


# ========================================================================
# 3. 얼려도 경사는 지나간다
# ========================================================================

def gradient_flows_through():
    header("3. 얼린 층을 경사가 지나간다")
    torch.manual_seed(0)
    net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
    for p in net[2].parameters():                     # 이번엔 뒤쪽을 언다
        p.requires_grad_(False)
    x, y = torch.randn(16, 4), torch.randn(16, 1)
    F.mse_loss(net(x), y).backward()
    print("  뒤쪽을 얼렸을 때")
    print("    앞쪽 층 grad :",
          "있음" if net[0].weight.grad is not None else "None")
    print("    얼린 층 grad :", net[2].weight.grad)
    print("  경사가 얼린 층을 통과해 앞쪽까지 갔다")
    print("  얼린다는 것은 '안 쌓는다' 이지 '길을 막는다' 가 아니다")


# ========================================================================
# 4. 얼리기와 detach 와 no_grad 를 한 자리에서
# ========================================================================

def three_tools_on_a_net():
    header("4. 가운데 층에 세 가지를 걸어 본다")
    x, y = torch.randn(16, 4), torch.randn(16, 1)

    def fresh():
        torch.manual_seed(0)
        return nn.Sequential(nn.Linear(4, 8), nn.Linear(8, 8), nn.Linear(8, 1))

    def report(label, net):
        got = ["있음" if p.grad is not None else "None"
               for p in (net[0].weight, net[1].weight, net[2].weight)]
        print(f"  {label:26s} 0층 {got[0]:4s}  1층 {got[1]:4s}  2층 {got[2]:4s}")

    # (가) 가운데 층의 깃발만 끈다
    net = fresh()
    for p in net[1].parameters():
        p.requires_grad_(False)
    F.mse_loss(net(x), y).backward()
    report("(가) requires_grad_(False)", net)

    # (나) 가운데 층의 출력을 detach 한다
    net = fresh()
    h = net[1](net[0](x)).detach()
    F.mse_loss(net[2](h), y).backward()
    report("(나) 출력을 detach", net)

    # (다) 가운데 층을 no_grad 로 돌린다
    net = fresh()
    h0 = net[0](x)
    with torch.no_grad():
        h = net[1](h0)
    F.mse_loss(net[2](h), y).backward()
    report("(다) no_grad 로 감싸기", net)

    print("  (가) 만 0층까지 경사가 간다. 1층을 지나보내기 때문이다")
    print("  (나)(다) 는 1층에서 길이 끊겨 0층이 아무것도 못 받는다")
    print("  '가운데 층만 얼리고 싶다' 면 (가) 라야 한다")


def main():
    where_allowed()
    freezing()
    gradient_flows_through()
    three_tools_on_a_net()


if __name__ == "__main__":
    main()
