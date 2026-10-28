"""축약과 통계 — 어느 축이 접히는가, 그리고 약속이 다른 자리들.

`sum`·`mean`·`std` 는 축 하나를 접어 없앤다. 쓰기는 쉬운데 두 군데서
조용히 틀린다.

- `keepdim` 을 빼면 뒤의 브로드캐스팅이 어긋난다. 모양이 안 맞으면
  터져 주지만 **정사각이면 터지지 않고 엉뚱한 축에서 뺀다**
- `std` 의 기본값은 `correction=1`, 곧 **불편 추정량**이다
- `median` 은 짝수 개일 때 가운데 둘을 평균내지 **않는다**

실행: python 11_reductions_statistics.py
"""
import torch


def header(title):
    print(f"\n=== {title} ===")


# ========================================================================
# 1. dim 은 '접을 축' 이다
# ========================================================================

def which_axis():
    header("1. dim 은 접을 축이다")
    x = torch.tensor([[1.0, 2.0, 3.0],
                      [4.0, 5.0, 6.0],
                      [7.0, 8.0, 9.0]])
    print("x =");  print(x)
    print("  x.sum()        =", x.sum().item(), " (다 접는다)")
    print("  x.sum(dim=0)   =", x.sum(dim=0).tolist(), " 모양",
          tuple(x.sum(dim=0).shape), " 0번 축이 사라졌다")
    print("  x.sum(dim=1)   =", x.sum(dim=1).tolist(), " 모양",
          tuple(x.sum(dim=1).shape), " 1번 축이 사라졌다")
    y = torch.randn(2, 3, 4)
    print("  (2,3,4).sum(dim=(0,2)) 모양:",
          tuple(y.sum(dim=(0, 2)).shape), " 두 축을 한꺼번에")
    print("  헷갈리면 '남는 축' 을 보면 된다 — dim 에 적은 축이 없어진다")


# ========================================================================
# 2. keepdim 을 빼면
# ========================================================================

def keepdim_matters():
    header("2. keepdim")
    torch.manual_seed(0)
    x = torch.randn(3, 4)
    print("  (3,4) 에서")
    print("    mean(dim=1).shape         :", tuple(x.mean(dim=1).shape))
    print("    mean(dim=1, keepdim).shape:",
          tuple(x.mean(dim=1, keepdim=True).shape))
    try:
        x - x.mean(dim=1)
    except RuntimeError as e:
        print("    x - mean(dim=1) :", str(e).splitlines()[0][:62])
    print("    여기서는 터져 주어 바로 안다")

    sq = torch.randn(4, 4)                      # 정사각이면
    wrong = sq - sq.mean(dim=1)                 # 터지지 않는다
    right = sq - sq.mean(dim=1, keepdim=True)
    print("  (4,4) 정사각에서는")
    print("    터지지 않는다. 모양:", tuple(wrong.shape))
    print("    두 결과가 같은가:", torch.allclose(wrong, right))
    print("    바른 쪽 행 평균:", [round(v, 6) for v in right.mean(dim=1).tolist()])
    print("    틀린 쪽 행 평균:", [round(v, 6) for v in wrong.mean(dim=1).tolist()])
    print("    행을 가운데로 옮기려 했는데 열에서 뺀 셈이 되었다")


# ========================================================================
# 3. std 의 correction
# ========================================================================

def correction():
    header("3. std 의 기본값은 불편 추정량이다")
    x = torch.tensor([1.0, 2.0, 3.0, 4.0])
    n = x.numel()
    ss = ((x - x.mean()) ** 2).sum()
    print("  손으로 sqrt(SS/n)     (편향) =", round((ss / n).sqrt().item(), 6))
    print("  손으로 sqrt(SS/(n-1)) (불편) =", round((ss / (n - 1)).sqrt().item(), 6))
    print("  x.std()             =", round(x.std().item(), 6), " <- 기본값")
    print("  x.std(correction=1) =", round(x.std(correction=1).item(), 6))
    print("  x.std(correction=0) =", round(x.std(correction=0).item(), 6))
    print("  correction 은 분모에서 빼는 수다. 1 이 베셀 보정이고 기본값이다")
    print("  numpy 의 np.std 는 기본이 ddof=0 — 반대다")
    print("  var 도 같다:", round(x.var().item(), 6), "=",
          round((x.std() ** 2).item(), 6))


# ========================================================================
# 4. median 과 비김
# ========================================================================

def median_and_ties():
    header("4. median 과 비김")
    e = torch.tensor([1.0, 2.0, 3.0, 4.0])
    print("  [1,2,3,4] 의")
    print("    torch.median()   =", e.median().item(), " <- 가운데 둘 중 아래쪽")
    print("    quantile(0.5)    =", e.quantile(0.5).item(), " <- 둘의 평균")
    print("  numpy 와 파이썬 statistics 는 2.5 를 준다")
    print("  median 은 실제로 있는 값을 돌려주겠다는 약속이다")

    t = torch.tensor([3, 1, 4, 1, 5, 9, 2, 6, 5])
    v, i = t.mode()
    print("  [3,1,4,1,5,9,2,6,5] 의 mode =", v.item(), " 자리", i.item())
    print("    1 과 5 가 둘 다 두 번 나오는데 하나만 돌려준다")
    a = torch.tensor([1.0, 3.0, 3.0, 2.0])
    print("  [1,3,3,2].argmax() =", a.argmax().item(), " <- 비기면 첫 자리")


# ========================================================================
# 5. 자주 쓰는 축약들
# ========================================================================

def the_rest():
    header("5. 나머지")
    torch.manual_seed(0)
    x = torch.randn(3, 4)
    mx, idx = x.max(dim=1)
    print("  max(dim=1) 는 값과 자리를 같이 준다:",
          [round(v, 4) for v in mx.tolist()], idx.tolist())
    print("  min/max 를 dim 없이 부르면 값만:", round(x.max().item(), 4))
    lo, hi = x.aminmax()
    print("  aminmax() 한 번에:", round(lo.item(), 4), round(hi.item(), 4))

    b = torch.tensor([True, True, False])
    print("  all / any            :", b.all().item(), b.any().item())
    print("  numel / count_nonzero:", x.numel(),
          torch.count_nonzero(x > 0).item())
    print("  norm(2) 과 손으로     :", round(x.norm().item(), 6),
          round((x ** 2).sum().sqrt().item(), 6))
    c = torch.tensor([1, 2, 3, 4])
    print("  cumsum               :", c.cumsum(0).tolist())
    print("  bincount             :",
          torch.bincount(torch.tensor([1, 2, 3, 2, 1, 2, 3])).tolist())
    print("    자리가 곧 값이라 나타나지 않은 0 도 칸을 차지한다")


# ========================================================================
# 6. 배치 정규화처럼 접기
# ========================================================================

def batch_stats():
    header("6. 채널별 통계")
    torch.manual_seed(0)
    batch = torch.randn(4, 3, 8, 8)          # (N, C, H, W)
    mean = batch.mean(dim=(0, 2, 3), keepdim=True)
    std = batch.std(dim=(0, 2, 3), keepdim=True)
    print("  batch.shape:", tuple(batch.shape))
    print("  mean.shape :", tuple(mean.shape), " 채널만 남는다")
    normed = (batch - mean) / (std + 1e-5)
    print("  정규화 뒤 채널 평균:",
          [round(v, 6) for v in normed.mean(dim=(0, 2, 3)).tolist()])
    print("  정규화 뒤 채널 표준편차:",
          [round(v, 6) for v in normed.std(dim=(0, 2, 3)).tolist()])
    print("  1 에서 조금 모자란다 — 분모에 더한 1e-5 때문이다")


def main():
    which_axis()
    keepdim_matters()
    correction()
    median_and_ties()
    the_rest()
    batch_stats()


if __name__ == "__main__":
    main()
