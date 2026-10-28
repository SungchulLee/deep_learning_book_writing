"""제자리 연산 — 막히는 자리와 막히지 않는 자리.

밑줄로 끝나는 연산(`add_`, `mul_`, `zero_`)은 새 텐서를 만들지 않고
가리키는 메모리를 직접 고친다. 메모리를 아끼지만 미분이 걸린 텐서를
고치면 셈이 어긋날 수 있다.

파이토치가 그것을 어떻게 막는지가 이 쪽의 이야기인데, 한 가지로
막지 않는다.

- 잎에는 **그 자리에서** 막는다
- 중간 텐서에는 막지 않는다. 통과시켜 두었다가, 역전파가 그 값을
  쓰려 할 때만 터진다
- 그래서 같은 제자리 연산이 어떤 식에서는 멀쩡하고 어떤 식에서는
  터진다. 그 연산이 역전파에 무엇을 저장해 두었느냐가 가른다

실행: python 07_inplace_operations.py
"""
import time

import torch
import torch.nn as nn


def header(title):
    print(f"\n=== {title} ===")


def spread(fn, rep=7):
    """한 번 잰 시간은 측정이 아니다."""
    ts = []
    for _ in range(rep):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return min(ts) * 1e3, max(ts) * 1e3


# ========================================================================
# 1. 밑줄 관례와 별칭
# ========================================================================

def basics():
    header("1. 밑줄은 '제자리에서' 라는 표시다")
    x = torch.tensor([1.0, 2.0, 3.0])
    before = x.data_ptr()
    y = x.add(10)            # 밑줄 없음 — 새 텐서
    print("  x.add(10)  -> 새 텐서, x 는", x.tolist())
    print("    같은 메모리인가:", y.data_ptr() == before)
    x.add_(10)               # 밑줄 있음 — 제자리
    print("  x.add_(10) -> x 가", x.tolist())
    print("    같은 메모리인가:", x.data_ptr() == before)

    a = torch.zeros(3)
    b = a                    # 같은 텐서를 가리키는 다른 이름
    a.add_(5)
    print("  a 를 고치면 b 도:", b.tolist(), " 같은 객체인가:", a is b)


# ========================================================================
# 2. 잎에는 그 자리에서 막는다
# ========================================================================

def leaves_are_blocked():
    header("2. 잎에는 막는다")
    leaf = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    try:
        leaf.add_(1)
    except RuntimeError as e:
        print("  leaf.add_(1):", str(e).splitlines()[0])

    # 길 1: 제자리가 아닌 연산을 쓴다
    ok = leaf + 1
    print("  leaf + 1 은 된다:", ok.tolist(), " grad_fn",
          type(ok.grad_fn).__name__)

    # 길 2: no_grad 안에서 고친다 (옵티마이저가 하는 일)
    with torch.no_grad():
        leaf.add_(1)
    print("  no_grad 안에서는 된다:", leaf.tolist())
    print("  requires_grad 는 그대로:", leaf.requires_grad)


# ========================================================================
# 3. 중간 텐서에는 막지 않는다
# ========================================================================

def non_leaves_pass():
    header("3. 중간 텐서에는 막지 않는다")
    print("  y 를 만들고 y.add_(1) 한 다음 역전파해 본다")
    cases = (
        ("y = x * 2", lambda t: t * 2, "아무것도"),
        ("y = x ** 2", lambda t: t ** 2, "입력 x"),
        ("y = x.exp()", lambda t: t.exp(), "출력 y"),
        ("y = x.relu()", lambda t: t.relu(), "출력 y"),
    )
    for label, build, saves in cases:
        x = torch.tensor([1.0, 2.0], requires_grad=True)
        y = build(x)
        y.add_(1.0)                      # 넷 다 여기서는 통과한다
        try:
            y.sum().backward()
            result = f"역전파 통과  x.grad={[round(v, 3) for v in x.grad.tolist()]}"
        except RuntimeError:
            result = "역전파에서 RuntimeError"
        print(f"  {label:13s} 역전파가 쓰는 것 {saves:8s} -> {result}")
    print("  제자리 연산 자체는 넷 다 통과했다. 갈린 것은 역전파다")


# ========================================================================
# 4. 그러면 nn.ReLU(inplace=True) 는 왜 안전한가
# ========================================================================

def inplace_relu():
    header("4. nn.ReLU(inplace=True)")
    for flag in (False, True):
        torch.manual_seed(0)
        net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(inplace=flag),
                            nn.Linear(8, 1))
        x = torch.randn(3, 4)
        net.zero_grad()
        net(x).sum().backward()
        g = sum(p.grad.abs().sum().item() for p in net.parameters())
        print(f"  inplace={flag!s:5s} 역전파 통과, 경사 절댓값 합 {g:.6f}")
    print("  두 값이 같다. relu 를 제자리에서 해도 셈이 바뀌지 않는다")
    print("  파이토치가 제 연산의 제자리 꼴은 역전파까지 맞춰 두었기 때문이다")

    x = torch.tensor([-1.0, 2.0], requires_grad=True)
    y = torch.relu(x)
    y.add_(1.0)                          # 내가 손으로 끼워 넣은 제자리 연산
    try:
        y.sum().backward()
        print("  손으로 add_ 를 넣어도 통과")
    except RuntimeError as e:
        print("  손으로 add_ 를 넣으면:", str(e).splitlines()[0][:66])
    print("  파이토치가 아는 제자리와 내가 끼워 넣은 제자리는 다르다")


# ========================================================================
# 5. 무엇을 아끼는가
# ========================================================================

def savings():
    header("5. 아끼는 것")
    for shape in ((100, 100), (1000, 1000), (4000, 4000)):
        big = torch.randn(*shape)
        mb = big.numel() * 4 / 1e6
        o = spread(lambda: big + 1.0)
        i = spread(lambda: big.add_(1.0))
        print(f"  {str(shape):14s} 한 장 {mb:6.2f} MB")
        print(f"    새로 만들기 {o[0]:7.3f} ~ {o[1]:7.3f} ms")
        print(f"    제자리     {i[0]:7.3f} ~ {i[1]:7.3f} ms")
    print("  새로 만드는 쪽은 결과를 담을 한 장을 더 잡고 거기 적는다")


def main():
    basics()
    leaves_are_blocked()
    non_leaves_pass()
    inplace_relu()
    savings()


if __name__ == "__main__":
    main()
