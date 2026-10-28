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
