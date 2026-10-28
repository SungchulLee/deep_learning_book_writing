"""nn.Linear 의 웨이트는 왜 (out, in) 인가.

`nn.Linear(5, 3)` 의 `weight.shape` 는 `(3, 5)` 다. 입력이 5 인데 5 가
뒤에 온다. 수학책의 $y = Wx$ 와 같은 방향이면서, 코드는
$Y = XW^\\top$ 로 적는다.

이 쪽은 그 방향이 무엇을 쉽게 만드는지 본다. 빠르기 때문이라는 설명이
그럴듯해 보이므로 그것부터 재어 본다 — 재어 보면 아니다.

- 모양 (out, in) 과 한 행의 뜻
- 뒤집어 저장하면 느려지는가 (재어 보니 그렇지 않다)
- 수학식 y = Wx 와 코드 Y = XW^T 가 같은 것
- 경사 셋의 값: dW = g^T x, db = sum g, dx = g W
- 앞쪽 축은 몇 개여도 된다

실행: python 05_linear_weight_convention.py
"""
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


def header(title):
    print(f"\n=== {title} ===")


def spread(fn, n=9):
    """한 번 잰 시간은 측정이 아니다. 가장 빠른 값과 가장 느린 값."""
    got = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        got.append(time.perf_counter() - t0)
    return min(got) * 1e3, max(got) * 1e3


# ========================================================================
# 1. 모양 — 한 행이 출력 뉴런 하나
# ========================================================================

def shapes():
    header("1. 모양과 한 행의 뜻")
    torch.manual_seed(0)
    lin = nn.Linear(5, 3)
    W, b = lin.weight, lin.bias
    print("  nn.Linear(5, 3)")
    print("    weight.shape :", tuple(W.shape), "= (out, in)")
    print("    bias.shape   :", tuple(b.shape), "= (out,)")
    print("    파라미터 수   :", sum(p.numel() for p in lin.parameters()),
          "= 3*5 + 3")

    print("  W[1] = 출력 뉴런 1 로 들어오는 웨이트 다섯 개")
    print("    연속인가 :", W[1].is_contiguous(), " 보폭", W[1].stride())
    print("    메모리 간격(바이트):",
          [W[1][i].data_ptr() - W[1][0].data_ptr() for i in range(5)])
    print("  W[:,1] = 입력 1 이 나가는 웨이트 세 개")
    print("    연속인가 :", W[:, 1].is_contiguous(), " 보폭", W[:, 1].stride())
    print("    메모리 간격(바이트):",
          [W[:, 1][i].data_ptr() - W[:, 1][0].data_ptr() for i in range(3)])


# ========================================================================
# 2. 빠르기 때문인가 — 재어 본다
# ========================================================================

def is_it_speed():
    header("2. 뒤집어 저장하면 느려지는가")
    torch.manual_seed(0)
    print("  W 를 (out,in) 로 두고 x @ W.t() 하는 것과")
    print("  V 를 (in,out) 로 두고 x @ V    하는 것을 견준다")
    # 어느 쪽이 빠른지는 돌릴 때마다 뒤바뀐다. 그래서 '겹치는가' 같은
    # 참·거짓을 찍어 두면 그 줄이 쪽마다 달라진다. 범위만 적고, 판단은
    # 본문에서 한다.
    for B, din, dout in ((512, 2048, 2048), (4096, 512, 512), (64, 4096, 1024)):
        x = torch.randn(B, din)
        W = torch.randn(dout, din)
        V = W.t().contiguous()
        print(f"  B={B:5d} in={din:5d} out={dout:5d}")
        print("    두 결과가 같은가:", torch.allclose(x @ W.t(), x @ V, atol=1e-3))
        a = spread(lambda: x @ W.t())
        c = spread(lambda: x @ V)
        print(f"    x @ W.t() {a[0]:5.2f} ~ {a[1]:5.2f} ms")
        print(f"    x @ V     {c[0]:5.2f} ~ {c[1]:5.2f} ms")


# ========================================================================
# 3. y = Wx 와 Y = XW^T 는 같은 것
# ========================================================================

def same_math():
    header("3. 수학식과 코드")
    torch.manual_seed(0)
    lin = nn.Linear(5, 3)
    W, b = lin.weight, lin.bias
    x = torch.randn(4, 5)

    print("  배치로:  lin(x).shape =", tuple(lin(x).shape),
          " x @ W.t() + b 와 같은가:",
          torch.allclose(lin(x), x @ W.t() + b, atol=1e-6))

    # 한 표본을 열벡터로 보면 W x 다
    one = x[0]
    print("  한 표본:  W @ x[0] + b 와 같은가:",
          torch.allclose(lin(one), W @ one + b, atol=1e-6))
    print("  둘 다 맞다. 표본을 행으로 쌓으면 전치가 붙는 것뿐이다")

    # 한 뉴런은 내적 하나
    k = 1
    print(f"  뉴런 {k} 하나:  lin(x)[0,{k}] =",
          round(lin(x)[0, k].item(), 6))
    print(f"              x[0] @ W[{k}] + b[{k}] =",
          round((one @ W[k] + b[k]).item(), 6))
    print("  W[k] 가 연속이라 이 내적은 메모리를 한 줄로 읽는다")


# ========================================================================
# 4. 경사 셋
# ========================================================================

def gradients():
    header("4. 경사 셋의 모양과 값")
    torch.manual_seed(0)
    lin = nn.Linear(5, 3)
    W, b = lin.weight, lin.bias
    x = torch.randn(4, 5, requires_grad=True)
    g = torch.randn(4, 3)        # 위에서 내려온 경사
    lin.zero_grad(set_to_none=True)
    F.linear(x, W, b).backward(g)

    print("  W.grad == g.T @ x  :",
          torch.allclose(W.grad, g.t() @ x.detach(), atol=1e-6),
          " 모양", tuple(W.grad.shape))
    print("  b.grad == g.sum(0) :",
          torch.allclose(b.grad, g.sum(0), atol=1e-6),
          " 모양", tuple(b.grad.shape))
    print("  x.grad == g @ W    :",
          torch.allclose(x.grad, g @ W.detach(), atol=1e-6),
          " 모양", tuple(x.grad.shape))
    print("  경사의 모양은 언제나 그 파라미터의 모양과 같다")


# ========================================================================
# 5. 앞쪽 축은 몇 개여도 된다
# ========================================================================

def leading_axes():
    header("5. 앞쪽 축")
    torch.manual_seed(0)
    lin = nn.Linear(5, 3)
    for shape in ((5,), (4, 5), (2, 7, 5), (2, 3, 7, 5)):
        print(f"  입력 {str(shape):14s} -> 출력 {tuple(lin(torch.randn(shape)).shape)}")
    x3 = torch.randn(2, 7, 5)
    flat = lin(x3.reshape(-1, 5)).reshape(2, 7, 3)
    print("  (2,7,5) 를 (14,5) 로 펴서 넣은 것과 같은가:",
          torch.allclose(lin(x3), flat, atol=1e-6))
    print("  마지막 축만 in_features 와 맞으면 된다")


# ========================================================================
# 6. 전치는 복사하지 않는다
# ========================================================================

def transpose_is_free():
    header("6. W.t() 는 복사가 아니다")
    torch.manual_seed(0)
    W = nn.Linear(5, 3).weight
    print("  W.is_contiguous()     :", W.is_contiguous(), " 보폭", W.stride())
    print("  W.t().is_contiguous() :", W.t().is_contiguous(),
          " 보폭", W.t().stride())
    print("  같은 저장소인가        :",
          W.untyped_storage().data_ptr() == W.t().untyped_storage().data_ptr())
    print("  그래서 x @ W.t() 는 전치하느라 메모리를 쓰지 않는다")


def main():
    shapes()
    is_it_speed()
    same_math()
    gradients()
    leading_axes()
    transpose_is_free()


if __name__ == "__main__":
    main()
