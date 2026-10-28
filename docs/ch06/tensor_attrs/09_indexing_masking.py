"""자리 잡기와 가리기 — 뷰로 오는 것과 베껴 오는 것.

텐서에서 일부를 꺼내는 길이 여럿이고, 크게 두 갈래로 갈린다.

- **자르기**(`a[1:3, 2:4]`, `a[::2]`)는 **뷰**다. 고치면 원본이 바뀐다
- **참거짓 가리기**와 **정수 배열 자리 잡기**는 **베낀다**. 고쳐도 원본은 그대로다

읽을 때는 둘이 같아 보이므로, 고치려 할 때에야 갈린다. 그런데 대입은
또 다르다 — `a[mask] = 0` 은 베낀 것에 쓰는 것이 아니라 원본을 고친다.

마지막 절이 이 쪽에서 가장 비싼 값이다. `torch.where` 는 **고르지 않은
가지도 셈한다.** 그래서 0 으로 나눈 다음 가리는 것만으로는 역전파의
NaN 을 막지 못한다.

실행: python 09_indexing_masking.py
"""
import torch


def header(title):
    print(f"\n=== {title} ===")


def shares(a, b):
    return a.untyped_storage().data_ptr() == b.untyped_storage().data_ptr()


# ========================================================================
# 1. 자르기는 뷰다
# ========================================================================

def slicing_is_a_view():
    header("1. 자르기는 뷰다")
    a = torch.arange(20).reshape(4, 5)
    print("a =");  print(a)
    v = a[1:3, 2:4]
    print("  a[1:3, 2:4] =");  print(v)
    print("  저장소를 나눠 쓰는가:", shares(a, v))
    v[0, 0] = 999
    print("  뷰를 999 로 고친 뒤 a[1,2] =", a[1, 2].item())

    step = a[::2, ::2]
    print("  건너뛰며 자른 a[::2, ::2] 도 뷰인가:", shares(a, step),
          " 보폭", step.stride())


# ========================================================================
# 2. 가리기는 베낀다 — 읽을 때는
# ========================================================================

def masking_copies():
    header("2. 참거짓으로 가려 읽으면 베낀다")
    a = torch.tensor([1, -2, 3, -4, 5, -6])
    mask = a > 0
    picked = a[mask]
    print("  a      =", a.tolist())
    print("  a > 0  =", mask.tolist())
    print("  a[mask]=", picked.tolist(), " 모양", tuple(picked.shape))
    print("  저장소를 나눠 쓰는가:", shares(a, picked))
    picked[0] = 100
    print("  꺼낸 것을 100 으로 고친 뒤 a =", a.tolist(), "(그대로다)")

    idx = torch.tensor([0, 2, 4])
    print("  정수 배열로 고른 a[[0,2,4]] 도 베끼는가:",
          not shares(a, a[idx]))
    print("  가려 읽으면 몇 개가 나올지 모르므로 뷰로 만들 수 없다")


# ========================================================================
# 3. 대입은 다르다
# ========================================================================

def masked_assignment():
    header("3. 가려서 대입하면 원본을 고친다")
    a = torch.tensor([1, -2, 3, -4, 5, -6])
    before = a.data_ptr()
    a[a < 0] = 0
    print("  a[a < 0] = 0  ->", a.tolist())
    print("  같은 메모리인가:", a.data_ptr() == before)
    print("  읽기는 베끼고 쓰기는 제자리다 — 같은 꼴인데 갈린다")

    b = torch.arange(10).float()
    b[(b > 3) & (b < 7)] = -1
    print("  & 와 | 로 조건을 묶는다:", b.tolist())
    print("  and / or 가 아니라 & / | 다. 파이썬 연산자는 텐서에 못 쓴다")


# ========================================================================
# 4. 고르는 길들
# ========================================================================

def selecting():
    header("4. 고르는 길들")
    a = torch.arange(12).reshape(3, 4)
    print("a =");  print(a)

    r = torch.tensor([0, 2, 1])
    c = torch.tensor([1, 3, 2])
    print("  a[r, c] 짝으로 골라 (0,1),(2,3),(1,2):", a[r, c].tolist())
    print("  a[[0,2]] 행 통째로:", a[torch.tensor([0, 2])].tolist())
    print("  a[:, [1,3]] 열 통째로:", a[:, torch.tensor([1, 3])].tolist())

    rr = torch.tensor([[0], [1], [2]])      # (3,1)
    cc = torch.tensor([[0, 2]])             # (1,2)
    print("  브로드캐스팅으로 격자:", a[rr, cc].tolist())

    print("  masked_select :", a.masked_select(a > 7).tolist())
    print("  nonzero       :", a.nonzero().shape, "(자리의 좌표)")
    ri, ci = torch.where(a > 9)
    print("  where(조건) 하나만 주면 좌표:", ri.tolist(), ci.tolist())


# ========================================================================
# 5. 자리 더하기와 줄임표
# ========================================================================

def shapes_and_ellipsis():
    header("5. None 과 줄임표")
    a = torch.randn(3, 4)
    print("  a.shape          :", tuple(a.shape))
    print("  a[None].shape    :", tuple(a[None].shape), "= unsqueeze(0)")
    print("  a[:, None].shape :", tuple(a[:, None].shape))
    print("  a[..., None].shape:", tuple(a[..., None].shape), "= unsqueeze(-1)")

    big = torch.randn(2, 3, 4, 5, 6)
    print("  big.shape        :", tuple(big.shape))
    print("  big[..., 0]      :", tuple(big[..., 0].shape))
    print("  big[0, ..., 2]   :", tuple(big[0, ..., 2].shape))
    print("  줄임표는 '남은 축 전부' 라서 차원 수가 바뀌어도 안 고쳐도 된다")


# ========================================================================
# 6. where 는 고르지 않은 가지도 셈한다
# ========================================================================

def where_computes_both():
    header("6. where 는 양쪽을 다 셈한다")
    x = torch.tensor([0.0, 1.0, 2.0], requires_grad=True)
    y = torch.where(x > 0, 1.0 / x, torch.zeros_like(x))
    print("  torch.where(x > 0, 1/x, 0)")
    print("    순전파:", y.tolist(), " <- 0 자리가 잘 가려졌다")
    y.sum().backward()
    print("    x.grad:", x.grad.tolist(), " <- NaN 이 새어 나온다")
    print("  1/x 는 가려지기 전에 이미 계산되었고, 그 미분도 그래프에 남았다")

    x2 = torch.tensor([0.0, 1.0, 2.0], requires_grad=True)
    denom = torch.where(x2 > 0, x2, torch.ones_like(x2))   # 분모를 먼저 고친다
    y2 = torch.where(x2 > 0, 1.0 / denom, torch.zeros_like(x2))
    y2.sum().backward()
    print("  분모를 먼저 안전하게 바꾸면")
    print("    순전파:", y2.tolist())
    print("    x.grad:", x2.grad.tolist(), " <- 깨끗하다")


# ========================================================================
# 7. 어텐션 마스크 — 쓸모와 함정
# ========================================================================

def attention_mask():
    header("7. 어텐션 마스크")
    torch.manual_seed(0)
    s = torch.randn(1, 4, requires_grad=True)
    m = torch.tensor([[False, False, True, True]])
    p = torch.softmax(s.masked_fill(m, float("-inf")), dim=-1)
    print("  일부만 가리면 softmax:", [round(v, 4) for v in p[0].tolist()])
    print("    가린 자리가 정확히 0 이 된다 — exp(-inf) = 0")

    s2 = torch.randn(1, 4, requires_grad=True)
    m2 = torch.ones(1, 4, dtype=torch.bool)
    p2 = torch.softmax(s2.masked_fill(m2, float("-inf")), dim=-1)
    print("  한 줄을 통째로 가리면:", p2[0].tolist())
    print("    전부 -inf 라 분모가 0 이 되어 NaN 이다")
    print("    길이가 0 인 문장이 배치에 섞이면 여기서 터진다")

    causal = torch.triu(torch.ones(4, 4), diagonal=1).bool()
    print("  인과 마스크(위쪽 삼각):")
    for row in causal.tolist():
        print("   ", " ".join("X" if v else "." for v in row))


def main():
    slicing_is_a_view()
    masking_copies()
    masked_assignment()
    selecting()
    shapes_and_ellipsis()
    where_computes_both()
    attention_mask()


if __name__ == "__main__":
    main()
