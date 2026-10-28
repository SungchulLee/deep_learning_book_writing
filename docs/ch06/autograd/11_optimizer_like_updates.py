"""옵티마이저 없이 걷기 — step() 이 숨기고 있는 세 줄.

`opt.step()` 한 줄이 하는 일을 손으로 적어 본다. 경사를 구하는
일(`backward`)과 그 경사로 값을 바꾸는 일(갱신)은 **다른 단계**이고,
둘 사이에 `no_grad` 가 들어간다.

SGD · 모멘텀 · Adam 을 차례로 손으로 짜서 파이토치 것과 견준다.
셋 다 소수점 끝까지 맞으면, 옵티마이저가 마법이 아니라는 것이
확인된 셈이다.

실행: python 11_optimizer_like_updates.py
"""
import warnings

import torch


def header(title):
    print(f"\n=== {title} ===")


def problem(seed=0):
    """작은 이차함수 하나. 최솟값은 w = [1, -2] 다."""
    torch.manual_seed(seed)
    target = torch.tensor([1.0, -2.0])
    w = torch.tensor([3.0, 3.0], requires_grad=True)
    return w, target


def loss_of(w, target):
    return ((w - target) ** 2).sum()


# ========================================================================
# 1. SGD
# ========================================================================

def plain_sgd():
    header("1. SGD")
    lr, steps = 0.1, 20

    w, t = problem()
    opt = torch.optim.SGD([w], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        loss_of(w, t).backward()
        opt.step()
    by_torch = w.detach().clone()

    w2, t2 = problem()
    for _ in range(steps):
        if w2.grad is not None:
            w2.grad.zero_()
        loss_of(w2, t2).backward()
        with torch.no_grad():
            w2 -= lr * w2.grad               # 이 한 줄이 step() 이다
    print("  torch.optim.SGD :", [round(v, 8) for v in by_torch.tolist()])
    print("  손으로          :", [round(v, 8) for v in w2.detach().tolist()])
    print("  같은가          :", torch.allclose(by_torch, w2.detach(), atol=0))


# ========================================================================
# 2. 모멘텀
# ========================================================================

def momentum():
    header("2. 모멘텀")
    lr, mu, steps = 0.1, 0.9, 20

    w, t = problem()
    opt = torch.optim.SGD([w], lr=lr, momentum=mu)
    for _ in range(steps):
        opt.zero_grad()
        loss_of(w, t).backward()
        opt.step()
    by_torch = w.detach().clone()

    w2, t2 = problem()
    buf = None
    for _ in range(steps):
        if w2.grad is not None:
            w2.grad.zero_()
        loss_of(w2, t2).backward()
        with torch.no_grad():
            g = w2.grad
            buf = g.clone() if buf is None else mu * buf + g
            w2 -= lr * buf
    print("  torch 모멘텀 :", [round(v, 8) for v in by_torch.tolist()])
    print("  손으로       :", [round(v, 8) for v in w2.detach().tolist()])
    print("  가장 큰 차이 :", f"{(by_torch - w2.detach()).abs().max():.2e}")
    print("  첫 걸음의 버퍼를 g 로 두는 것이 파이토치의 약속이다")
    print("  0 으로 두고 mu*0 + g 를 해도 같은 값이지만, 구현에 따라 다르다")


# ========================================================================
# 3. Adam
# ========================================================================

def adam():
    header("3. Adam")
    lr, b1, b2, eps, steps = 0.1, 0.9, 0.999, 1e-8, 20

    w, t = problem()
    opt = torch.optim.Adam([w], lr=lr, betas=(b1, b2), eps=eps)
    for _ in range(steps):
        opt.zero_grad()
        loss_of(w, t).backward()
        opt.step()
    by_torch = w.detach().clone()

    w2, t2 = problem()
    m = torch.zeros_like(w2)
    v = torch.zeros_like(w2)
    for step in range(1, steps + 1):
        if w2.grad is not None:
            w2.grad.zero_()
        loss_of(w2, t2).backward()
        with torch.no_grad():
            g = w2.grad
            m = b1 * m + (1 - b1) * g
            v = b2 * v + (1 - b2) * g * g
            m_hat = m / (1 - b1 ** step)       # 치우침 보정
            v_hat = v / (1 - b2 ** step)
            w2 -= lr * m_hat / (v_hat.sqrt() + eps)
    print("  torch.optim.Adam:", [round(v, 8) for v in by_torch.tolist()])
    print("  손으로          :", [round(v, 8) for v in w2.detach().tolist()])
    print("  가장 큰 차이     :", f"{(by_torch - w2.detach()).abs().max():.2e}")


# ========================================================================
# 4. no_grad 를 빼면
# ========================================================================

def why_no_grad():
    header("4. no_grad 를 빼면")
    w, t = problem()
    loss_of(w, t).backward()
    try:
        w -= 0.1 * w.grad
    except RuntimeError as e:
        print("  제자리로 고치면:", str(e).splitlines()[0][:62])

    w2, t2 = problem()
    loss_of(w2, t2).backward()
    w3 = w2 - 0.1 * w2.grad                  # 새 텐서를 만들면 통과한다
    print("  새 텐서를 만들면 통과한다. 그런데")
    print("    w3.is_leaf       :", w3.is_leaf)
    print("    w3.requires_grad :", w3.requires_grad)
    loss_of(w3, t2).backward()
    with warnings.catch_warnings():        # 경고문에 빌드 기계 경로가 붙는다
        warnings.simplefilter("ignore")
        print("    두 번째 걸음의 w3.grad:", w3.grad)
    print("  잎이 아니라 경사를 못 받는다 — 다음 걸음이 없다")


def main():
    plain_sgd()
    momentum()
    adam()
    why_no_grad()


if __name__ == "__main__":
    main()
