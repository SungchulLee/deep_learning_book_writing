"""비교, 범위 제한, 정렬 — NaN 이 섞이면 달라지는 것들.

비교 연산자는 참거짓 텐서를 돌려주고, 정렬은 값과 자리를 같이 준다.
여기까지는 짐작대로다. 조심할 자리가 둘이다.

- **NaN 은 어느 비교에도 참이 아니다.** `x > 0` 도 `x < 0` 도 거짓이라,
  거르는 조건을 어떻게 적느냐에 따라 NaN 이 남기도 하고 빠지기도 한다
- **`clamp` 의 경사는 범위 밖에서 0 이다.** 자른 자리는 학습이 멈춘다

실행: python 15_comparison_clamping_sorting.py
"""
import torch

NAN = float("nan")
INF = float("inf")


def header(title):
    print(f"\n=== {title} ===")


# ========================================================================
# 1. 비교는 참거짓 텐서를 준다
# ========================================================================

def comparisons():
    header("1. 비교 연산자")
    a = torch.tensor([1, 5, 3, 5])
    b = torch.tensor([2, 4, 3, 5])
    print("  a =", a.tolist(), "  b =", b.tolist())
    print("  a > b  :", (a > b).tolist(), " dtype", (a > b).dtype)
    print("  a == b :", (a == b).tolist())
    print("  a >= 3 :", (a >= 3).tolist(), " 스칼라와도 된다")
    print("  (a > b).sum() =", (a > b).sum().item(), " <- 참을 센다")
    print("  & | ~ 로 묶는다:", ((a > 2) & (a < 5)).tolist())
    print("  and/or/not 는 못 쓴다. 텐서를 참거짓 하나로 줄일 수 없어서다")
    print("  torch.equal(a, b) 는 통째로 비교해 하나를 준다:",
          torch.equal(a, b))


# ========================================================================
# 2. NaN 은 어느 비교에도 참이 아니다
# ========================================================================

def nan_comparisons():
    header("2. NaN 과 비교")
    print("  nan == nan :", (torch.tensor(NAN) == torch.tensor(NAN)).item())
    x = torch.tensor([1.0, NAN, 3.0])
    print("  x =", x.tolist())
    print("  x > 0 :", (x > 0).tolist())
    print("  x < 0 :", (x < 0).tolist(), " <- 둘 다 거짓이다")
    print("  x[x > 0]    :", x[x > 0].tolist(), " <- NaN 이 조용히 빠진다")
    print("  x[~(x <= 0)]:", x[~(x <= 0)].tolist(), " <- 부정하면 남는다")
    print("  같은 뜻으로 적은 두 조건이 다른 답을 준다")
    print("  NaN 을 찾으려면 isnan 을 써야 한다:", torch.isnan(x).tolist())

    s = torch.tensor([1.0, NAN, INF, -INF])
    print("  isfinite:", torch.isfinite(s).tolist())
    print("  isinf   :", torch.isinf(s).tolist())


# ========================================================================
# 3. 정렬과 극값에서의 NaN
# ========================================================================

def nan_in_sorting():
    header("3. 정렬과 극값에서의 NaN")
    y = torch.tensor([3.0, NAN, 1.0, 2.0])
    print("  y =", y.tolist())
    print("  sort().values :", y.sort().values.tolist(), " <- 맨 뒤로 간다")
    print("  max()         :", y.max().item(), " <- 번져 나온다")
    print("  argmax()      :", y.argmax().item(), " <- NaN 자리를 가리킨다")
    print("  maximum(y, 0) :", torch.maximum(y, torch.zeros(4)).tolist())
    print("  nanmean / nansum 은 건너뛴다:",
          torch.tensor([1.0, NAN, 3.0]).nanmean().item(),
          torch.tensor([1.0, NAN, 3.0]).nansum().item())
    print("  max 와 argmax 가 어긋나 보이지만 둘 다 NaN 자리를 말하고 있다")


# ========================================================================
# 4. clamp 의 경사는 밖에서 0 이다
# ========================================================================

def clamping():
    header("4. clamp")
    x = torch.tensor([-3.0, -0.5, 0.5, 3.0])
    print("  x            :", x.tolist())
    print("  clamp(-1, 1) :", x.clamp(-1, 1).tolist())
    print("  clamp(min=0) :", x.clamp(min=0).tolist(), " = relu")
    print("  clip 은 clamp 의 다른 이름:", torch.equal(x.clip(-1, 1),
                                                 x.clamp(-1, 1)))

    v = torch.tensor([-3.0, -0.5, 0.5, 3.0], requires_grad=True)
    v.clamp(-1, 1).sum().backward()
    print("  clamp 의 경사:", v.grad.tolist())
    print("  잘린 자리는 0 이다 — 그 원소는 더 이상 배우지 않는다")
    print("  활성값을 clamp 로 묶으면 ReLU 의 죽은 뉴런과 같은 일이 난다")


# ========================================================================
# 5. 정렬 · topk · kthvalue
# ========================================================================

def sorting():
    header("5. 정렬과 고르기")
    t = torch.tensor([3.0, 1.0, 4.0, 1.0, 5.0])
    v, i = t.sort()
    print("  sort        값:", v.tolist(), " 자리:", i.tolist())
    print("  sort(내림차순):", t.sort(descending=True).values.tolist())
    print("  argsort     :", t.argsort().tolist(), " = sort 의 자리만")
    print("  정렬한 값은 t[argsort] 와 같은가:",
          torch.equal(t[t.argsort()], t.sort().values))

    tv, ti = t.topk(2)
    print("  topk(2)     값:", tv.tolist(), " 자리:", ti.tolist())
    print("  topk(2, largest=False):", t.topk(2, largest=False).values.tolist())
    print("  kthvalue(2) 두 번째로 작은 값:", t.kthvalue(2).values.item())

    m = torch.tensor([[3.0, 1.0], [2.0, 4.0]])
    print("  2차원은 축을 고른다 sort(dim=0):",
          m.sort(dim=0).values.flatten().tolist())


# ========================================================================
# 6. 가까운가 — allclose 의 식
# ========================================================================

def closeness():
    header("6. allclose")
    a = torch.tensor([1.0, 2.0, 3.0])
    print("  기본값은 rtol=1e-5, atol=1e-8")
    print("  판정식은 |a - b| <= atol + rtol * |b| 다")
    for d in (1e-4, 1e-5, 1e-6):
        b = a + d
        bound = 1e-8 + 1e-5 * b.abs()
        print(f"  차이 {d:.0e}  |a-b| {d:.1e} vs 한계 {bound.min():.1e}~"
              f"{bound.max():.1e}  allclose {torch.allclose(a, b)}")
    print("  rtol 이 상대 오차라 큰 값일수록 너그러워진다")
    print("  0 근처를 견줄 때는 atol 을 손으로 올려야 한다")
    z = torch.zeros(3)
    print("  allclose(0, 1e-7) :", torch.allclose(z, z + 1e-7))
    print("  allclose(0, 1e-7, atol=1e-6):",
          torch.allclose(z, z + 1e-7, atol=1e-6))
    print("  NaN 은 기본으로 같지 않다:",
          torch.allclose(torch.tensor([NAN]), torch.tensor([NAN])))
    print("  equal_nan=True 로 바꿀 수 있다:",
          torch.allclose(torch.tensor([NAN]), torch.tensor([NAN]),
                         equal_nan=True))


def main():
    comparisons()
    nan_comparisons()
    nan_in_sorting()
    clamping()
    sorting()
    closeness()


if __name__ == "__main__":
    main()
