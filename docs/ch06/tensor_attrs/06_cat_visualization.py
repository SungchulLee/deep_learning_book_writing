"""torch.cat 이 어느 축을 늘리는지 — 그리고 루프에서 쓰면 생기는 일.

이어 붙이는 함수들을 두루 훑는 것은 「이어 붙이기와 나누기」 쪽이고,
여기서는 `cat` 한 가지를 눈으로 본다. 값만 보고도 어디서 왔는지
알도록 A 는 10번대, B 는 20번대로 채운다.

마지막 절이 이 쪽의 값이다. `cat` 은 언제나 새 메모리에 베끼므로,
루프를 돌며 결과에 하나씩 이어 붙이면 이미 모은 것을 매번 다시
베낀다. 얼마나 되는지 재어 본다.

실행: python 06_cat_visualization.py
"""
import time

import torch


def header(title):
    print(f"\n=== {title} ===")


A = torch.tensor([[11, 12, 13],
                  [14, 15, 16]])
B = torch.tensor([[21, 22, 23],
                  [24, 25, 26]])


# ========================================================================
# 1. 어느 축이 길어지는가
# ========================================================================

def axes():
    header("1. dim 이 정하는 것")
    print("A =");  print(A);  print("  A.shape =", tuple(A.shape))
    print("B =");  print(B);  print("  B.shape =", tuple(B.shape))

    cat0 = torch.cat([A, B], dim=0)
    print("\ntorch.cat([A, B], dim=0) — 세로로 길어진다")
    print(cat0)
    print(f"  {tuple(A.shape)} + {tuple(B.shape)} -> {tuple(cat0.shape)}")

    cat1 = torch.cat([A, B], dim=1)
    print("\ntorch.cat([A, B], dim=1) — 가로로 길어진다")
    print(cat1)
    print(f"  {tuple(A.shape)} + {tuple(B.shape)} -> {tuple(cat1.shape)}")
    print(f"  원소 수는 둘 다 {cat0.numel()}개로 같다")


# ========================================================================
# 2. cat 과 stack
# ========================================================================

def cat_vs_stack():
    header("2. cat 은 늘리고 stack 은 만든다")
    cat0 = torch.cat([A, B], dim=0)
    st = torch.stack([A, B], dim=0)
    print("  cat  (dim=0):", tuple(cat0.shape), " 축 2개 그대로")
    print("  stack(dim=0):", tuple(st.shape), " 축 3개로 늘었다")
    same = torch.cat([A.unsqueeze(0), B.unsqueeze(0)], dim=0)
    print("  stack 은 unsqueeze 하고 cat 한 것과 같은가:",
          torch.equal(st, same))
    try:
        torch.stack([A, torch.zeros(2, 4, dtype=torch.long)], dim=0)
    except RuntimeError as e:
        print("  모양이 다르면 stack:", str(e).splitlines()[0][:66])


# ========================================================================
# 3. 이어 붙일 수 없는 경우
# ========================================================================

def refusals():
    header("3. 이어 붙일 수 없는 경우")
    C = torch.zeros(2, 4, dtype=torch.long)
    try:
        torch.cat([A, C], dim=0)
    except RuntimeError as e:
        print("  cat([A(2,3), C(2,4)], dim=0):")
        print("   ", str(e).splitlines()[0])
    print("  같은 둘을 dim=1 로는 이을 수 있다:",
          tuple(torch.cat([A, C], dim=1).shape))
    print("  이어 붙이는 축만 더해지고 나머지는 같아야 한다")


# ========================================================================
# 4. cat 은 베낀다 — 루프에서 쓰면
# ========================================================================

def copies_every_time():
    header("4. cat 은 언제나 베낀다")
    cat0 = torch.cat([A, B], dim=0)
    print("  A 와 결과가 저장소를 나눠 쓰는가:",
          A.untyped_storage().data_ptr() == cat0.untyped_storage().data_ptr())
    cat0[0, 0] = 999
    print("  결과를 999 로 고친 뒤 A[0,0] =", A[0, 0].item(), "(그대로다)")


def loop_cost():
    header("5. 루프마다 이어 붙이면")

    def loop_cat(n, width=256):
        out = torch.zeros(0, width)
        for _ in range(n):
            out = torch.cat([out, torch.randn(1, width)], dim=0)
        return out

    def collect_then_cat(n, width=256):
        parts = [torch.randn(1, width) for _ in range(n)]
        return torch.cat(parts, dim=0)

    print("  256 짜리 줄을 n 번 모은다")
    for n in (500, 1000, 2000, 4000):
        times = {}
        for name, fn in (("loop", loop_cat), ("once", collect_then_cat)):
            ts = []
            for _ in range(3):
                t0 = time.perf_counter()
                r = fn(n)
                ts.append(time.perf_counter() - t0)
            times[name] = min(ts) * 1e3
        print(f"  n={n:5d}  루프마다 cat {times['loop']:8.1f} ms   "
              f"모아서 한 번 {times['once']:6.1f} ms   "
              f"배수 {times['loop'] / times['once']:5.1f}")
    print("  n 을 두 배 하면 루프 쪽은 네 배에 가깝게 는다")
    print("  이미 모은 것을 매번 다시 베끼기 때문이다")


# ========================================================================
# 6. 경사는 제 몫만 간다
# ========================================================================

def gradients():
    header("6. cat 의 경사")
    a = torch.tensor([[1.0, 2.0]], requires_grad=True)
    b = torch.tensor([[3.0, 4.0]], requires_grad=True)
    g = torch.tensor([[10.0, 20.0], [30.0, 40.0]])
    (torch.cat([a, b], dim=0) * g).sum().backward()
    print("  위에서 내려온 경사 =", g.tolist())
    print("  a.grad =", a.grad.tolist())
    print("  b.grad =", b.grad.tolist())
    print("  이어 붙인 자리대로 잘라 나눠 준다")


def main():
    axes()
    cat_vs_stack()
    refusals()
    copies_every_time()
    loop_cost()
    gradients()


if __name__ == "__main__":
    main()
