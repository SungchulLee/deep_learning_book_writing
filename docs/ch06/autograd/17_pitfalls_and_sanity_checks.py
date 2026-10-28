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
