"""자동 미분에 쓰이는 텐서 속성 — 누구에게 .grad 가 생기는가.

`requires_grad=True` 로 만들어도 `.grad` 가 채워지는 텐서는 그 가운데
일부뿐이다. 잎(leaf)만 받는다. 이 쪽은 그 경계를 재어서 보인다.

- requires_grad / is_leaf / grad_fn 의 조합 — 넷 가운데 셋만 있다
- 잎이 아닌 텐서의 .grad 는 None 이고, 파이토치가 경고를 띄운다
- 누적이 기본값이다. set_to_none=True 는 0 으로 채우기와 걸음이 다르다
- 홑값이 아닌 출력에는 v 를 주어야 한다(벡터-야코비 곱)
- backward 를 두 번 부르면 터지는 그래프와 안 터지는 그래프가 있다
- no_grad 없이 잎을 제자리에서 고치면 막힌다

실행: python 01_autograd_related.py
"""
import textwrap
import unicodedata
import warnings

import torch
import torch.nn as nn


def header(title):
    print(f"\n=== {title} ===")


def pad(s, width):
    """한글은 두 칸을 차지하므로 글자 수로 세면 표가 어긋난다."""
    w = sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in str(s))
    return str(s) + " " * max(0, width - w)


def fn_name(t):
    """grad_fn 을 이름으로만 적는다. 객체를 그대로 찍으면 메모리
    주소가 따라 나와 돌릴 때마다 달라진다."""
    return type(t.grad_fn).__name__ if t.grad_fn is not None else None


# ========================================================================
# 1. requires_grad 와 is_leaf — 넷 가운데 셋
# ========================================================================

def combinations():
    header("1. requires_grad x is_leaf")
    a = torch.tensor([1.0])
    b = torch.tensor([1.0], requires_grad=True)
    cases = (
        ("상수", a),
        ("파라미터", b),
        ("파라미터 x 2", b * 2),
        ("상수 x 2", a * 2),
        ("detach()", (b * 2).detach()),
    )
    print(pad("", 14) + f"{'requires_grad':>13s} {'is_leaf':>8s} {'grad_fn':>14s}")
    for name, t in cases:
        print(pad(name, 14) + f"{str(t.requires_grad):>13s} "
              f"{str(t.is_leaf):>8s} {str(fn_name(t)):>14s}")
    print("requires_grad=False 이면서 is_leaf=False 인 텐서는 없다")


# ========================================================================
# 2. .grad 는 잎에만 채워진다
# ========================================================================

def only_leaves_get_grad():
    header("2. 잎이 아니면 .grad 가 None 이다")
    x = torch.tensor([1.0, 2.0], requires_grad=True)
    h = x * 3           # 잎이 아니다
    y = (h ** 2).sum()
    y.backward()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        g = h.grad
        print("h.grad :", g)
        # 경고문 뒤쪽에는 파이토치를 빌드한 기계의 경로가 붙어 나오므로
        # (돌리는 기계마다 다르다) 앞 두 문장만 적는다.
        msg = str(caught[0].message) if caught else "경고 없음"
        msg = msg[:msg.index("If you indeed")].strip()
        print("경고   :", textwrap.fill(msg, 66, subsequent_indent=" " * 9))
    print("x.grad :", x.grad)

    # retain_grad() 로 부탁하면 중간 텐서도 받는다
    x2 = torch.tensor([1.0, 2.0], requires_grad=True)
    h2 = x2 * 3
    h2.retain_grad()
    (h2 ** 2).sum().backward()
    print("retain_grad 뒤 h2.grad :", h2.grad)
    print("손으로  dy/dh = 2h = 2*[3,6] =", (2 * (x2 * 3)).tolist())


# ========================================================================
# 3. 누적이 기본값이다
# ========================================================================

def accumulation():
    header("3. 지우지 않으면 쌓인다")
    w = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    for i in range(1, 4):
        (w ** 2).sum().backward()
        print(f"  {i}번째 backward 뒤 w.grad = {w.grad.tolist()}")
    w.grad.zero_()
    (w ** 2).sum().backward()
    print(f"  zero_() 하고 한 번 더  w.grad = {w.grad.tolist()}  (2w)")


def none_vs_zeros():
    header("4. set_to_none=True 는 0 으로 채우기와 다르다")
    got = []
    for set_to_none in (True, False):
        torch.manual_seed(0)
        m = nn.Linear(3, 1)
        opt = torch.optim.SGD(m.parameters(), lr=0.1, momentum=0.9)
        x, y = torch.randn(5, 3), torch.randn(5, 1)
        nn.functional.mse_loss(m(x), y).backward()
        opt.step()                       # 모멘텀이 쌓인 한 걸음
        opt.zero_grad(set_to_none=set_to_none)
        opt.step()                       # 경사가 비었는데 또 걷는다
        got.append(m.weight.detach().flatten().clone())
    print("  set_to_none=True   :", "  ".join(f"{v:+.6f}" for v in got[0].tolist()))
    print("  set_to_none=False  :", "  ".join(f"{v:+.6f}" for v in got[1].tolist()))
    print(f"  가장 큰 차이        : {(got[0] - got[1]).abs().max():.6f}")
    print("  None 이면 step 이 그 파라미터를 건너뛴다.")
    print("  0 이면 모멘텀이 남아 계속 밀고 간다.")


# ========================================================================
# 5. 홑값이 아닌 출력 — 벡터-야코비 곱
# ========================================================================

def vjp():
    header("5. 홑값이 아니면 v 를 주어야 한다")
    torch.manual_seed(0)
    x = torch.randn(4, 3, requires_grad=True)
    A = torch.randn(2, 3)
    out = x @ A.t()                       # (4, 3) @ (3, 2) = (4, 2)
    v = torch.tensor([[1.0, 0.0], [0.5, -1.0], [0.0, 0.0], [2.0, 3.0]])

    try:
        out.backward()
    except RuntimeError as e:
        print("  v 없이 부르면:", str(e).splitlines()[0])

    out.backward(v)
    print("  x.grad.shape :", tuple(x.grad.shape))
    print("  손으로 v @ A 와 같은가:", torch.allclose(x.grad, v @ A))
    print("  x.grad[1]    :", [round(t, 4) for t in x.grad[1].tolist()])
    print("  (v @ A)[1]   :", [round(t, 4) for t in (v @ A)[1].tolist()])


# ========================================================================
# 6. backward 두 번 — 터지는 그래프와 안 터지는 그래프
# ========================================================================

def twice():
    header("6. 두 번 부르면 터지는가")
    builders = (
        ("x.sum()", lambda t: t.sum(), "저장할 것이 없다"),
        ("(x**2).sum()", lambda t: (t ** 2).sum(), "x 를 저장해 둔다"),
        ("x.exp().sum()", lambda t: t.exp().sum(), "exp(x) 를 저장해 둔다"),
    )
    for expr, build, why in builders:
        x = torch.tensor([1.0, 2.0], requires_grad=True)
        y = build(x)
        y.backward()
        try:
            y.backward()
            print(f"  {expr:14s} 두 번째도 된다   grad={x.grad.tolist()}  ({why})")
        except RuntimeError as e:
            print(f"  {expr:14s} {str(e).splitlines()[0][:46]}...  ({why})")

    x = torch.tensor([1.0, 2.0], requires_grad=True)
    y = (x ** 2).sum()
    y.backward(retain_graph=True)
    y.backward()
    print(f"  retain_graph=True 로는 두 번   grad={x.grad.tolist()}  (2*2x)")


# ========================================================================
# 7. no_grad 없이 잎을 고치면
# ========================================================================

def updating():
    header("7. 파라미터를 제자리에서 고치려면")
    p = torch.tensor([1.0, 2.0], requires_grad=True)
    (p ** 2).sum().backward()
    try:
        p -= 0.1 * p.grad
    except RuntimeError as e:
        print("  no_grad 없이:", str(e).splitlines()[0])

    with torch.no_grad():
        p -= 0.1 * p.grad
    print("  no_grad 안에서:", [round(v, 4) for v in p.tolist()])
    print("  p.requires_grad 는 그대로:", p.requires_grad)
    print("  p.is_leaf 도 그대로:", p.is_leaf)


def main():
    combinations()
    only_leaves_get_grad()
    accumulation()
    none_vs_zeros()
    vjp()
    twice()
    updating()


if __name__ == "__main__":
    main()
