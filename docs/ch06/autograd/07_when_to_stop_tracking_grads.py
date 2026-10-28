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
