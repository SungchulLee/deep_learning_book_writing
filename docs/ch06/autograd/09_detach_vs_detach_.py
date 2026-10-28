"""detach() 와 detach_() — 새 손잡이냐 제자리냐.

둘 다 미분 추적을 끊는다. 다른 것은 **무엇에 대고 하느냐**다.

- `t.detach()`  — 추적 없는 **새 텐서**를 내준다. `t` 는 그대로
- `t.detach_()` — **`t` 자신**을 끊는다. 되돌릴 수 없다

그리고 `detach_()` 에는 함정이 하나 있다. 이미 역전파가 저장해 둔
텐서를 제자리에서 끊어 버리면, 뒤에 `backward()` 가 터진다.

실행: python 09_detach_vs_detach_.py
"""
import torch


def header(title):
    print(f"\n=== {title} ===")


def describe(name, t):
    fn = type(t.grad_fn).__name__ if t.grad_fn is not None else None
    print(f"  {name:22s} requires_grad={t.requires_grad!s:5s} grad_fn={fn}")


# ========================================================================
# 1. 새 손잡이냐 제자리냐
# ========================================================================

def new_or_in_place():
    header("1. 무엇이 바뀌는가")
    x = torch.randn(3, requires_grad=True)

    y = x * 2
    d = y.detach()
    print("  y.detach()")
    describe("    돌려받은 d", d)
    describe("    원본 y", y)
    print("    같은 객체인가:", d is y,
          "  저장소 공유:", d.data_ptr() == y.data_ptr())

    y2 = x * 2
    r = y2.detach_()
    print("  y2.detach_()")
    describe("    돌려받은 r", r)
    describe("    원본 y2", y2)
    print("    같은 객체인가:", r is y2)
    print("  detach_ 는 y2 자신을 바꾸었다. 되돌릴 길이 없다")


# ========================================================================
# 2. 잎이 아닌 것에도 걸린다
# ========================================================================

def works_on_non_leaf():
    header("2. requires_grad_ 와 달리 잎이 아니어도 된다")
    x = torch.randn(3, requires_grad=True)
    h = x * 2
    try:
        h.requires_grad_(False)
    except RuntimeError as e:
        print("  h.requires_grad_(False):", str(e).splitlines()[0][:58])
    h.detach_()
    print("  h.detach_()            : 된다.", h.requires_grad)
    print("  오류 메시지가 가리키는 길이 바로 이것이다")


# ========================================================================
# 3. detach_ 의 함정 — 저장된 텐서를 끊으면
# ========================================================================

def the_trap():
    header("3. 저장된 텐서를 제자리에서 끊으면")
    x = torch.randn(3, requires_grad=True)
    y = x * 2
    loss = (y ** 2).sum()        # 역전파가 y 를 저장해 둔다
    y.detach_()                  # 그 y 를 제자리에서 끊는다
    try:
        loss.backward()
    except RuntimeError as e:
        print("  backward:", str(e).splitlines()[0][:72])
    print("  detach() 였다면 새 텐서를 만들 뿐이라 아무 일도 없다")

    x2 = torch.randn(3, requires_grad=True)
    y2 = x2 * 2
    loss2 = (y2 ** 2).sum()
    _ = y2.detach()              # 새 손잡이만 하나 더 만든다
    loss2.backward()
    print("  detach() 뒤 backward:", torch.allclose(x2.grad, 8 * x2.detach()),
          " (dloss/dx = 8x)")


# ========================================================================
# 4. 어느 쪽을 쓰는가
# ========================================================================

def which_one():
    header("4. 쓰는 자리")
    x = torch.randn(4, requires_grad=True)

    # (가) 값만 꺼내 기록한다 — detach()
    loss = (x ** 2).sum()
    history = [loss.detach()]
    print("  (가) 기록용으로 꺼내기 -> detach()")
    print("       원본 그래프를 건드리지 않는다:",
          type(loss.grad_fn).__name__)

    # (나) 순환 신경망에서 숨은 상태를 끊는다 — 둘 다 쓰인다
    h = torch.zeros(2, requires_grad=True)
    for _ in range(3):
        h = (h + x[:2]).tanh()
    print("  (나) RNN 숨은 상태: 바퀴마다 h = h.detach() 로 끊는다")
    print("       안 끊으면 그래프가 바퀴 수만큼 길어진다")
    print("       h 를 다시 대입하므로 detach() 로 충분하다")
    print("  detach_() 가 꼭 필요한 자리는 드물다")
    print("  같은 이름을 계속 쓰면서 그래프만 끊고 싶을 때뿐이다")


def main():
    new_or_in_place()
    works_on_non_leaf()
    the_trap()
    which_one()


if __name__ == "__main__":
    main()
