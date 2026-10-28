"""튜토리얼 22: 최적화 기초 — 그래디언트를 받아 파라메터를 옮기는 일."""
import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(0)


def header(title):
    print(f"\n{'='*70}\n{title}\n{'='*70}")


def main():
    header("1. SGD 한 걸음은 손으로 따라갈 수 있다")
    p = torch.tensor([2.0], requires_grad=True)
    opt = optim.SGD([p], lr=0.1)
    print(f"{'':4s}{'p':>8s}{'grad':>9s}{'loss':>9s}{'다음 p':>10s}")
    for i in range(3):
        opt.zero_grad()
        loss = (p - 1.0) ** 2
        loss.backward()
        g, cur = p.grad.item(), p.item()
        opt.step()
        print(f"{i+1:<4d}{cur:8.4f}{g:9.4f}{loss.item():9.4f}{p.item():10.4f}")
    print("grad = 2(p - 1) 이고, 다음 p = p - 0.1 * grad 다.")

    header("2. 그래디언트 자르기 — 넘을 때만 줄인다")
    def clip_demo(vec, max_norm):
        q = nn.Parameter(torch.zeros(len(vec)))
        q.grad = torch.tensor(vec)
        before = q.grad.norm().item()
        direction_before = q.grad / q.grad.norm()
        torch.nn.utils.clip_grad_norm_([q], max_norm=max_norm)
        after = q.grad.norm().item()
        same_dir = torch.allclose(q.grad / q.grad.norm(), direction_before, atol=1e-6)
        print(f"  {str(vec):22s} 노름 {before:6.3f} -> {after:6.3f}   방향 그대로? {same_dir}")

    print(f"max_norm = 1.0")
    clip_demo([3.0, 4.0], 1.0)        # 노름 5 — 줄어든다
    clip_demo([0.3, 0.4], 1.0)        # 노름 0.5 — 그대로다
    print("자르기는 **줄이기만** 한다. 작은 그래디언트를 키워 주지 않는다.")

    header("3. 러닝레이트 스케줄러")
    opt = optim.SGD([torch.zeros(1, requires_grad=True)], lr=1.0)
    sched = optim.lr_scheduler.StepLR(opt, step_size=3, gamma=0.1)
    lrs = []
    for _ in range(10):
        opt.step()
        sched.step()
        lrs.append(opt.param_groups[0]["lr"])
    print("에포크:", list(range(1, 11)))
    print("러닝레이트:", [f"{v:g}" for v in lrs])
    print("step_size=3 마다 gamma=0.1 을 곱한다.")

    header("4. SGD 와 Adam 을 같은 문제에서 견준다")
    # 배울 것이 있는 문제를 만든다 — y = Xw* + 잡음
    torch.manual_seed(0)
    X = torch.randn(200, 10)
    w_true = torch.randn(10, 1)
    y = X @ w_true + 0.1 * torch.randn(200, 1)

    def train(make_opt, steps=100):
        torch.manual_seed(1)                      # 같은 초깃값에서 출발한다
        m = nn.Linear(10, 1)
        o = make_opt(m.parameters())
        hist = []
        for _ in range(steps):
            o.zero_grad()
            loss = nn.MSELoss()(m(X), y)
            loss.backward()
            o.step()
            hist.append(loss.item())
        return hist

    runs = {
        "SGD  lr=0.01": train(lambda p: optim.SGD(p, lr=0.01)),
        "SGD  lr=0.1 ": train(lambda p: optim.SGD(p, lr=0.1)),
        "Adam lr=0.01": train(lambda p: optim.Adam(p, lr=0.01)),
        "Adam lr=0.1 ": train(lambda p: optim.Adam(p, lr=0.1)),
    }
    print(f"{'':14s}{'1 스텝':>10s}{'10':>10s}{'50':>10s}{'100':>10s}")
    for name, h in runs.items():
        print(f"{name:14s}{h[0]:10.4f}{h[9]:10.4f}{h[49]:10.4f}{h[99]:10.4f}")
    s01, a01 = runs["SGD  lr=0.01"][-1], runs["Adam lr=0.01"][-1]
    s10, a10 = runs["SGD  lr=0.1 "][-1], runs["Adam lr=0.1 "][-1]
    print(f"\nlr 을 고정하고 옵티마이저만 바꾸면   {a01/s01:.1f}배 (0.01), {a10/s10:.1f}배 (0.1)")
    print(f"옵티마이저를 고정하고 lr 만 바꾸면  {s01/s10:.1f}배 (SGD), {a01/a10:.1f}배 (Adam)")
    print("이 문제에서는 옵티마이저보다 lr 이 훨씬 크게 가른다.")


if __name__ == "__main__":
    main()
