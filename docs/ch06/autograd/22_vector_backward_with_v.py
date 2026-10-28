"""backward(v) 는 야코비를 주지 않는다 — v^T J 를 준다.

출력이 홑값이면 `backward()` 로 끝난다. 모양이 있으면 거절한다.

    grad can be implicitly created only for scalar outputs

까닭은 미분의 결과가 **야코비 행렬**이어서 `.grad` 에 담을 수 없기
때문이다. `.grad` 는 언제나 그 텐서와 같은 모양이다.

그래서 `backward(v)` 는 야코비를 만들지 않고 **v 와의 곱 하나**만
계산한다. 야코비 전체가 필요하면 v 를 바꿔 가며 여러 번 불러야 하고,
그 값을 이 쪽에서 잰다.

실행: python 22_vector_backward_with_v.py
"""
import time

import torch


def header(title):
    print(f"\n=== {title} ===")


# ========================================================================
# 1. 홑값이 아니면 거절한다
# ========================================================================

def refuses():
    header("1. 거절")
    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    y = x ** 2                      # 모양이 있는 출력
    try:
        y.backward()
    except RuntimeError as e:
        print("  y.backward():", str(e).splitlines()[0])
    print("  y 가 (3,) 이고 x 가 (3,) 이면 야코비는 3x3 이다")
    print("  x.grad 는 (3,) 이므로 거기 담을 수 없다")


# ========================================================================
# 2. v 는 '출력을 얼마씩 중히 볼 것인가' 다
# ========================================================================

def what_v_means():
    header("2. v 의 뜻")
    for v in ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 1.0],
              [10.0, 1.0, 0.1]):
        x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        (x ** 2).backward(torch.tensor(v))
        print(f"  v={str(v):22s} x.grad={x.grad.tolist()}")
    print("  y_i = x_i^2 이라 야코비가 대각이고 대각 원소가 2x_i 다")
    print("  그래서 v^T J = v * 2x 가 그대로 보인다")
    print("  v 를 전부 1 로 주는 것은 y.sum().backward() 와 같다")
    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    (x ** 2).sum().backward()
    print("  확인:", x.grad.tolist())


# ========================================================================
# 3. 야코비 전체를 얻으려면 m 번 불러야 한다
# ========================================================================

def building_jacobian():
    header("3. 야코비 모으기")
    torch.manual_seed(0)
    A = torch.randn(4, 3)
    x = torch.randn(3, requires_grad=True)
    y = A @ x                        # 야코비는 A 자신이다

    rows = []
    for i in range(4):
        v = torch.zeros(4)
        v[i] = 1.0
        x.grad = None
        y.backward(v, retain_graph=True)
        rows.append(x.grad.clone())
    J = torch.stack(rows)
    print("  단위벡터 네 개로 네 줄을 모았다")
    print("  J 가 A 와 같은가:", torch.allclose(J, A))
    print("  출력이 m 개면 backward 를 m 번 불러야 한다")
    print("  retain_graph=True 가 필요하고, 매번 x.grad 를 비워야 한다")


# ========================================================================
# 4. 값이 얼마나 드는가
# ========================================================================

def the_cost():
    header("4. VJP 한 번 vs 야코비 전체")
    torch.manual_seed(0)
    n, m = 512, 512
    A = torch.randn(m, n)

    def one_vjp():
        x = torch.randn(n, requires_grad=True)
        (A @ x).backward(torch.randn(m))

    def full_jacobian():
        x = torch.randn(n, requires_grad=True)
        y = A @ x
        for i in range(m):
            v = torch.zeros(m)
            v[i] = 1.0
            x.grad = None
            y.backward(v, retain_graph=True)

    got = {}
    for label, fn in (("VJP 한 번", one_vjp), ("야코비 전체", full_jacobian)):
        ts = []
        for _ in range(3):
            t0 = time.perf_counter()
            fn()
            ts.append(time.perf_counter() - t0)
        got[label] = min(ts) * 1e3
        print(f"  {label:12s} {got[label]:8.2f} ms")
    ratio = got["야코비 전체"] / got["VJP 한 번"]
    print(f"  {ratio:.0f} 배 든다. 출력 개수 {m} 보다는 작다")
    print("  backward 를 m 번 부르는 값이 m 배까지는 안 가기 때문이다")
    print("  그래도 자릿수가 같다. 로스를 홑값으로 만드는 까닭이 이것이다")


# ========================================================================
# 5. 로스가 홑값인 까닭
# ========================================================================

def why_scalar_loss():
    header("5. 로스는 왜 홑값인가")
    torch.manual_seed(0)
    W = torch.randn(3, 4, requires_grad=True)
    x = torch.randn(8, 4)
    out = x @ W.t()                   # (8, 3)
    print("  출력이 (8,3) 이면 야코비는 (8,3) x (3,4) 짜리다")
    print("  원소 수:", 8 * 3 * 3 * 4)
    loss = out.pow(2).mean()          # 홑값으로 줄인다
    loss.backward()
    print("  홑값으로 줄이면 backward 한 번으로 끝난다")
    print("  W.grad 모양:", tuple(W.grad.shape), "= W 와 같다")
    print("  학습에 필요한 것은 야코비가 아니라 이 하나뿐이다")


def main():
    refuses()
    what_v_means()
    building_jacobian()
    the_cost()
    why_scalar_loss()


if __name__ == "__main__":
    main()
