"""보폭이 성능에 미치는 영향 — 재어 본다.

「메모리 배치와 스트라이드」 쪽의 10절에 쓰이는 측정이다. 흔히
"연속이 빠르다" 고 하는데, 어디까지 참인지 재어 본다.

한 번 잰 시간은 측정이 아니므로 일곱 번 재어 가장 빠른 값과 가장
느린 값을 함께 적는다. 두 범위가 겹치면 차이를 말할 수 없다.

실행: python memory_layout_strides.py
"""
import time

import torch


def header(title):
    print(f"\n=== {title} ===")


def spread(fn, rep=7):
    ts = []
    for _ in range(rep):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return min(ts) * 1e3, max(ts) * 1e3


def show(name, a, b, label_a="연속", label_b="비연속"):
    x, y = spread(a), spread(b)
    gap = "범위가 겹친다" if not (x[1] < y[0] or y[1] < x[0]) else \
          f"{y[0] / x[0]:.1f}배"
    print(f"  {name:12s} {label_a} {x[0]:6.2f}~{x[1]:6.2f}   "
          f"{label_b} {y[0]:6.2f}~{y[1]:6.2f} ms   {gap}")


# ========================================================================
# 1. 전치 뷰가 정말 느린가
# ========================================================================

def transposed_view():
    header("1. 전치 뷰 (4000x4000)")
    torch.manual_seed(0)
    x = torch.randn(4000, 4000)
    xt = x.t()
    show("sum()", lambda: x.sum(), lambda: xt.sum())
    show("sum(dim=0)", lambda: x.sum(dim=0), lambda: xt.sum(dim=0))
    show("x + 1", lambda: x + 1, lambda: xt + 1)
    show("clone()", lambda: x.clone(), lambda: xt.clone())
    print("  대부분 범위가 겹친다. sum(dim=0) 은 오히려 전치 쪽이 빠르다")
    print("  전치하면 0번 축이 메모리에서 잇달아 있게 되기 때문이다")


# ========================================================================
# 2. 보폭이 큰 뷰에서는
# ========================================================================

def strided_view():
    header("2. 네 칸마다 고른 뷰 (2000x2000 -> 2000x500)")
    torch.manual_seed(0)
    big = torch.randn(2000, 2000)
    sl = big[:, ::4]
    slc = sl.contiguous()
    show("sum()", lambda: slc.sum(), lambda: sl.sum())
    show("exp()", lambda: slc.exp(), lambda: sl.exp())
    show("matmul", lambda: slc @ slc.t(), lambda: sl @ sl.t())
    print("  sum 에서만 연속 쪽이 뚜렷이 빠르다. 나머지는 범위가 겹친다")
    print("  연산마다 다르므로 '연속이 빠르다' 로 뭉뚱그릴 수 없다")


# ========================================================================
# 3. contiguous() 를 미리 불러 두면 이득인가
# ========================================================================

def prefetch():
    header("3. contiguous() 를 미리 불러 두면")
    torch.manual_seed(0)
    x = torch.randn(4000, 4000)
    xt = x.t()

    def as_is():
        return sum((xt * 2).sum() for _ in range(3))

    def made_contiguous():
        c = xt.contiguous()
        return sum((c * 2).sum() for _ in range(3))

    a, b = spread(as_is), spread(made_contiguous)
    print(f"  전치 뷰 그대로 세 번      {a[0]:6.2f}~{a[1]:6.2f} ms")
    print(f"  contiguous 하고 세 번     {b[0]:6.2f}~{b[1]:6.2f} ms")
    print("  베끼는 값이 아끼는 값보다 크다. 세 번으로는 본전을 못 뽑는다")


# ========================================================================
# 4. 연속이 '반드시' 필요한 자리
# ========================================================================

def when_required():
    header("4. 연속이 필요한 자리")
    x = torch.randn(100, 100)
    try:
        x.t().view(-1)
    except RuntimeError as e:
        print("  x.t().view(-1):", str(e).splitlines()[0][:70])
    flat = x.t().reshape(-1)
    print("  x.t().reshape(-1) 은 된다. 모양:", tuple(flat.shape))
    print("  reshape 가 베꼈는가:",
          flat.untyped_storage().data_ptr() != x.untyped_storage().data_ptr())
    print("  reshape 는 '필요하면 베낀다'. view 는 '못 하면 거절한다'")
    print("  성능이 아니라 이 차이가 contiguous() 를 부르는 진짜 까닭이다")


def main():
    transposed_view()
    strided_view()
    prefetch()
    when_required()


if __name__ == "__main__":
    main()
