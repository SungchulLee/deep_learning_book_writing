"""선형 사상의 VJP — 왜 A 가 아니라 A^T v 인가.

$y = Ax$ 에서 `y.backward(v)` 를 부르면 `x.grad` 에 무엇이 들어오는가.
손으로 유도해 보면 $A^\\top v$ 이고, 그것은 **$A$ 를 곱하는 것이 아니라
$A^\\top$ 을 곱하는 것**이다.

전치가 붙는 까닭을 모양만으로도 알 수 있다. $A$ 가 $(m, n)$ 이고
$v$ 가 $(m,)$ 이면, 결과는 $x$ 와 같은 $(n,)$ 이어야 한다.
$(m, n)$ 과 $(m,)$ 으로 $(n,)$ 을 만드는 길은 $A^\\top v$ 뿐이다.

실행: python 23_vector_backward_linear_case_1.py
"""
import torch


def header(title):
    print(f"\n=== {title} ===")


M, N = 4, 3


def setup(seed=0):
    torch.manual_seed(seed)
    return torch.randn(M, N), torch.randn(N, requires_grad=True)


# ========================================================================
# 1. autograd 와 손계산
# ========================================================================

def compare():
    header("1. A^T v 와 맞는가")
    A, x = setup()
    v = torch.tensor([1.0, 0.5, -1.0, 2.0])
    y = A @ x
    print("  A 모양", tuple(A.shape), "  x 모양", tuple(x.shape),
          "  y 모양", tuple(y.shape))
    y.backward(v)
    hand = A.t() @ v
    print("  x.grad :", [round(t, 6) for t in x.grad.tolist()])
    print("  A^T v  :", [round(t, 6) for t in hand.tolist()])
    print("  같은가 :", torch.allclose(x.grad, hand))


# ========================================================================
# 2. 모양만 보아도 전치가 나온다
# ========================================================================

def shapes_force_it():
    header("2. 모양이 전치를 강요한다")
    A, x = setup()
    v = torch.randn(M)
    print(f"  A ({M},{N})   v ({M},)   결과는 x 와 같은 ({N},) 여야 한다")
    for name, expr in (("A @ v  ", lambda: A @ v),
                       ("A.t() @ v", lambda: A.t() @ v),
                       ("v @ A  ", lambda: v @ A)):
        try:
            r = expr()
            print(f"  {name} -> 모양 {tuple(r.shape)}")
        except RuntimeError as e:
            print(f"  {name} -> {str(e).splitlines()[0][:50]}")
    print("  A.t() @ v 와 v @ A 가 둘 다 (3,) 을 준다")
    print("  그리고 둘은 같은 값이다:", torch.allclose(A.t() @ v, v @ A))


# ========================================================================
# 3. 유도
# ========================================================================

def derivation():
    header("3. 유도")
    A, x = setup()
    v = torch.randn(M)
    print("  y_i = sum_j A_ij x_j 이므로  dy_i/dx_j = A_ij")
    print("  스칼라 s = sum_i v_i y_i 를 만들면")
    print("    ds/dx_j = sum_i v_i * dy_i/dx_j = sum_i v_i A_ij = (A^T v)_j")
    s = (v * (A @ x)).sum()
    s.backward()
    print("  s = (v*y).sum() 을 역전파한 x.grad:",
          [round(t, 6) for t in x.grad.tolist()])
    print("  A^T v                            :",
          [round(t, 6) for t in (A.t() @ v).tolist()])
    print("  backward(v) 는 이 s 를 만들어 역전파한 것과 같다")


# ========================================================================
# 4. 야코비는 A 자신이다
# ========================================================================

def jacobian_is_A():
    header("4. 야코비 = A")
    A, x = setup()
    y = A @ x
    rows = []
    for i in range(M):
        e = torch.zeros(M)
        e[i] = 1.0
        x.grad = None
        y.backward(e, retain_graph=True)
        rows.append(x.grad.clone())
    J = torch.stack(rows)
    print("  단위벡터로 모은 J 가 A 와 같은가:", torch.allclose(J, A))
    print("  그래서 v^T J = v^T A = A^T v 다 (벡터이므로 같다)")


# ========================================================================
# 5. A 에 대한 경사도 같이 본다
# ========================================================================

def grad_wrt_A():
    header("5. A 에 대한 경사")
    torch.manual_seed(0)
    A = torch.randn(M, N, requires_grad=True)
    x = torch.randn(N, requires_grad=True)
    v = torch.randn(M)
    (A @ x).backward(v)
    outer = torch.outer(v, x.detach())
    print("  A.grad 모양:", tuple(A.grad.shape), " = A 와 같다")
    print("  A.grad == outer(v, x) 인가:", torch.allclose(A.grad, outer))
    print("  y_i = sum_j A_ij x_j 이므로 dy_i/dA_ij = x_j")
    print("  ds/dA_ij = v_i x_j — 바깥곱이다")


def main():
    compare()
    shapes_force_it()
    derivation()
    jacobian_is_A()
    grad_wrt_A()


if __name__ == "__main__":
    main()
