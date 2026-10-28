"""경사를 지우는 세 가지 길 — 그리고 '어디까지' 지우는가.

`opt.zero_grad()` · `model.zero_grad()` · `p.grad = None`.
셋 다 경사를 치우는데, 둘은 **범위가 다르고** 하나는 **남기는 것이
다르다.**

| 길 | 범위 | 남기는 것 |
|---|---|---|
| `opt.zero_grad()` | 옵티마이저가 든 것만 | `None` (기본값) |
| `model.zero_grad()` | 모델 전체 | `None` (기본값) |
| `p.grad = None` | 그 텐서 하나 | `None` |
| `p.grad.zero_()` | 그 텐서 하나 | **0 으로 채운 텐서** |

범위가 갈리는 자리가 함정이다. 옵티마이저에 일부만 넘겼다면
`opt.zero_grad()` 는 나머지를 **안 지운다.**

실행: python 18_three_ways_to_zero_gradients.py
"""
import inspect

import torch
import torch.nn as nn
import torch.nn.functional as F


def header(title):
    print(f"\n=== {title} ===")


def has_grad(net):
    return [p.grad is not None for p in net.parameters()]


# ========================================================================
# 1. 범위가 다르다
# ========================================================================

def scope():
    header("1. 범위")
    torch.manual_seed(0)
    net = nn.Sequential(nn.Linear(4, 8), nn.Linear(8, 2))
    opt = torch.optim.SGD(net[1].parameters(), lr=0.1)   # 뒤쪽만 넘겼다
    x, y = torch.randn(8, 4), torch.randn(8, 2)

    F.mse_loss(net(x), y).backward()
    print("  역전파 뒤 경사 있는가 :", has_grad(net))
    opt.zero_grad()
    print("  opt.zero_grad() 뒤    :", has_grad(net))
    print("    앞 둘이 True 로 남았다 — 옵티마이저가 모르는 텐서다")
    net.zero_grad()
    print("  model.zero_grad() 뒤  :", has_grad(net))
    print("  모델 전체를 지우려면 모델에게 시켜야 한다")


# ========================================================================
# 2. 안 지우면 쌓인다
# ========================================================================

def leftover_accumulates():
    header("2. 안 지운 쪽은 쌓인다")
    torch.manual_seed(0)
    net = nn.Sequential(nn.Linear(4, 8), nn.Linear(8, 2))
    opt = torch.optim.SGD(net[1].parameters(), lr=0.1)
    x, y = torch.randn(8, 4), torch.randn(8, 2)

    sizes = []
    for step in range(3):
        opt.zero_grad()                       # 뒤쪽만 지운다
        F.mse_loss(net(x), y).backward()
        sizes.append(net[0].weight.grad.abs().max().item())
    print("  0층 경사의 크기가 걸음마다:",
          [round(v, 6) for v in sizes])
    print("  지워지지 않아 1배, 2배, 3배로 자란다")
    print("  0층을 학습하지 않더라도 이 경사는 메모리를 차지한다")


# ========================================================================
# 3. None 과 0 으로 채운 텐서
# ========================================================================

def none_or_zeros():
    header("3. None 과 0")
    torch.manual_seed(1)
    w = torch.randn(3, requires_grad=True)
    opt = torch.optim.SGD([w], lr=0.1)

    (w ** 2).sum().backward()
    print("  역전파 뒤          :", w.grad)
    opt.zero_grad()
    print("  opt.zero_grad()    :", w.grad, " <- 기본이 None")
    (w ** 2).sum().backward()
    opt.zero_grad(set_to_none=False)
    print("  set_to_none=False  :", w.grad)
    (w ** 2).sum().backward()
    w.grad = None
    print("  w.grad = None      :", w.grad)
    (w ** 2).sum().backward()
    w.grad.zero_()
    print("  w.grad.zero_()     :", w.grad)

    sig = inspect.signature(opt.zero_grad)
    print("  이 판의 기본값     :", sig.parameters["set_to_none"].default,
          f"(torch {torch.__version__})")
    print("  예전 판은 0 으로 채우는 것이 기본이었다")


# ========================================================================
# 4. 그 차이가 걸음을 바꾼다
# ========================================================================

def it_changes_the_step():
    header("4. 모멘텀이 있으면 걸음이 달라진다")
    got = []
    for set_to_none in (True, False):
        torch.manual_seed(0)
        m = nn.Linear(3, 1)
        opt = torch.optim.SGD(m.parameters(), lr=0.1, momentum=0.9)
        x, y = torch.randn(5, 3), torch.randn(5, 1)
        F.mse_loss(m(x), y).backward()
        opt.step()
        opt.zero_grad(set_to_none=set_to_none)
        opt.step()                              # 경사가 빈 채로 한 걸음
        got.append(m.weight.detach().flatten().clone())
    print("  set_to_none=True  :", "  ".join(f"{v:+.6f}" for v in got[0].tolist()))
    print("  set_to_none=False :", "  ".join(f"{v:+.6f}" for v in got[1].tolist()))
    print(f"  가장 큰 차이      : {(got[0] - got[1]).abs().max():.6f}")
    print("  None 이면 step 이 건너뛰고, 0 이면 모멘텀이 민다")


# ========================================================================
# 5. 그래서 무엇을 쓰는가
# ========================================================================

def which():
    header("5. 고르기")
    torch.manual_seed(0)
    net = nn.Sequential(nn.Linear(4, 8), nn.Linear(8, 2))
    opt = torch.optim.SGD(net.parameters(), lr=0.1)
    x, y = torch.randn(8, 4), torch.randn(8, 2)

    for _ in range(2):
        opt.zero_grad()
        F.mse_loss(net(x), y).backward()
        opt.step()
    print("  보통은 opt.zero_grad() 다 — 걷는 것과 지우는 것이 한 짝이다")
    print("  옵티마이저가 모델 전체를 들고 있으면 범위도 같다")
    print("  들고 있는 파라미터 수:",
          sum(len(g["params"]) for g in opt.param_groups),
          " 모델의 파라미터 수:", len(list(net.parameters())))
    print("  둘이 다르면 model.zero_grad() 를 쓰는 편이 안전하다")


def main():
    scope()
    leftover_accumulates()
    none_or_zeros()
    it_changes_the_step()
    which()


if __name__ == "__main__":
    main()
