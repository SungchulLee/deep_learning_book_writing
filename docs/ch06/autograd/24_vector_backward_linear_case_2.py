"""같은 수학, 다른 모양 — v 의 모양은 출력과 꼭 맞아야 한다.

「선형 사상의 VJP」에서 $x$ 를 $(3,)$ 으로 두었다. 수학책처럼
$(3,1)$ 열벡터로 두면 어떻게 되는가. **답은 같고 모양만 달라진다.**

그런데 `backward(v)` 의 `v` 는 출력과 **꼭 같은 모양**이어야 한다.
브로드캐스팅으로 넘어가지 않는다 — 이것이 반가운 소식이다.

반가운 소식에는 짝이 있다. 모양이 **조용히** 틀리는 자리는 따로
있고, 그쪽이 훨씬 자주 문다. 5절이 그 자리다.

실행: python 24_vector_backward_linear_case_2.py
"""
import warnings

import torch
import torch.nn.functional as F


def header(title):
    print(f"\n=== {title} ===")


# ========================================================================
# 1. 같은 수학, 다른 모양
# ========================================================================

def same_math():
    header("1. (3,) 와 (3,1)")
    torch.manual_seed(0)
    A = torch.randn(4, 3)
    base = torch.randn(3)
    x1 = base.clone().requires_grad_(True)              # (3,)
    x2 = base.view(3, 1).clone().requires_grad_(True)   # (3,1)
    print("  x (3,)  -> y", tuple((A @ x1).shape))
    print("  x (3,1) -> y", tuple((A @ x2).shape))

    v = torch.tensor([1.0, 0.5, -1.0, 2.0])
    (A @ x1).backward(v)
    (A @ x2).backward(v.view(4, 1))
    print("  x1.grad 모양", tuple(x1.grad.shape),
          " x2.grad 모양", tuple(x2.grad.shape))
    print("  x1.grad", [round(t, 6) for t in x1.grad.tolist()])
    print("  x2.grad", [round(t, 6) for t in x2.grad.flatten().tolist()])
    print("  값이 같은가:", torch.allclose(x1.grad, x2.grad.flatten()))
    print("  경사는 언제나 그 텐서와 같은 모양으로 온다")


# ========================================================================
# 2. v 의 모양은 엄격하다
# ========================================================================

def strict_v():
    header("2. v 의 모양 검사")
    torch.manual_seed(0)
    A = torch.randn(4, 3)
    for shape in ((4, 1), (4,), (1, 1), (1,)):
        x = torch.randn(3, 1, requires_grad=True)
        try:
            (A @ x).backward(torch.ones(shape))
            print(f"  v 모양 {str(shape):8s} -> 통과. x.grad "
                  f"{[round(t, 4) for t in x.grad.flatten().tolist()]}")
        except RuntimeError as e:
            print(f"  v 모양 {str(shape):8s} -> {str(e).splitlines()[0][:52]}")
    print("  출력이 (4,1) 이므로 v 도 (4,1) 이라야 한다")
    print("  브로드캐스팅으로 넘어가지 않는다 — 터져 주는 쪽이다")


# ========================================================================
# 3. 열벡터로 적으면 식이 수학책과 같아진다
# ========================================================================

def column_form():
    header("3. 열벡터 꼴")
    torch.manual_seed(0)
    A = torch.randn(4, 3)
    x = torch.randn(3, 1, requires_grad=True)
    v = torch.randn(4, 1)
    (A @ x).backward(v)
    print("  x.grad    :", [round(t, 6) for t in x.grad.flatten().tolist()])
    print("  A^T v     :", [round(t, 6) for t in (A.t() @ v).flatten().tolist()])
    print("  모양도 같은가:", x.grad.shape == (A.t() @ v).shape)
    print("  (3,1) 로 두면 A^T v 의 모양이 그대로 x.grad 다")
    print("  (3,) 로 두면 값은 같고 모양만 (3,) 로 눌린다")


# ========================================================================
# 4. 어느 쪽으로 적을 것인가
# ========================================================================

def which_form():
    header("4. 고르기")
    print("  (3,) 꼴")
    print("    장점: 코드가 짧다. 파이토치의 거의 모든 함수가 이쪽을 받는다")
    print("    단점: 수학책의 열벡터 식과 모양이 다르다")
    print("  (3,1) 꼴")
    print("    장점: A^T v 의 모양이 그대로 나온다")
    print("    단점: 어디선가 (3,) 와 섞이면 브로드캐스팅이 일어난다")
    print("  파이토치는 (배치, 피처) 를 쓰므로 표본 하나도 (1, 3) 이 보통이다")


# ========================================================================
# 5. 조용히 틀리는 자리
# ========================================================================

def the_silent_one():
    header("5. 모양이 조용히 무는 자리")
    torch.manual_seed(0)
    pred = torch.randn(8, 1)
    targ_col = torch.randn(8, 1)
    targ_flat = targ_col.flatten()
    print("  pred", tuple(pred.shape),
          " target 열", tuple(targ_col.shape),
          " target 평평", tuple(targ_flat.shape))

    a = F.mse_loss(pred, targ_col)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        b = F.mse_loss(pred, targ_flat)
    print(f"  (8,1) vs (8,1) 로스 {a.item():.6f}")
    print(f"  (8,1) vs (8,)  로스 {b.item():.6f}   경고 {len(caught)}개")
    print("  (8,1) 과 (8,) 이 브로드캐스팅되어 (8,8) 이 된다")
    print("  표본 여덟 개를 서로 엇갈려 견준 셈이다")

    p1 = pred.clone().requires_grad_(True)
    F.mse_loss(p1, targ_col).backward()
    p2 = pred.clone().requires_grad_(True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        F.mse_loss(p2, targ_flat).backward()
    print("  경사가 같은가:", torch.allclose(p1.grad, p2.grad))
    print("  바른 경사:", [round(t, 5) for t in p1.grad.flatten()[:3].tolist()])
    print("  틀린 경사:", [round(t, 5) for t in p2.grad.flatten()[:3].tolist()])
    print("  경고는 뜨지만 터지지는 않는다. 경고를 끄면 아무 표도 없다")


def main():
    same_math()
    strict_v()
    column_form()
    which_form()
    the_silent_one()


if __name__ == "__main__":
    main()
