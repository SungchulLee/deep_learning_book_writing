"""비선형 함수의 전체 야코비안 — x 에 따라 달라진다.

「선형 사상의 VJP」에서 야코비가 $A$ 자신이었다. **상수**였다는 뜻이다.
비선형 함수에서는 야코비가 **자리마다 달라진다.**

$$
f(x_1, x_2) = \\begin{pmatrix} x_1^2 x_2 \\\\ \\sin x_1 + x_2^3 \\end{pmatrix}
$$

의 야코비를 세 가지로 구해 서로 견준다 — VJP 를 모아서, `torch.func`
으로, 그리고 수치 미분으로.

실행: python 25_vector_backward_nonlinear_full_jacobian.py
"""
import math

import torch
from torch.func import jacrev


def header(title):
    print(f"\n=== {title} ===")


def f(t):
    return torch.stack([t[0] ** 2 * t[1],
                        torch.sin(t[0]) + t[1] ** 3])


def hand_jacobian(a, b):
    """손으로 미분한 야코비."""
    return [[2 * a * b, a ** 2],
            [math.cos(a), 3 * b ** 2]]


# ========================================================================
# 1. 야코비가 자리마다 달라진다
# ========================================================================

def varies():
    header("1. 자리마다 다르다")
    for pt in ([1.0, 2.0], [2.0, 1.0]):
        J = jacrev(f)(torch.tensor(pt))
        print(f"  x={pt}")
        for r, h in zip(J.tolist(), hand_jacobian(*pt)):
            print(f"    자동 {[round(v, 6) for v in r]}"
                  f"   손으로 {[round(v, 6) for v in h]}")
    print("  선형에서는 A 하나였는데 여기서는 x 를 넣어야 정해진다")


# ========================================================================
# 2. VJP 를 모아서
# ========================================================================

def by_vjp():
    header("2. VJP 를 모아서")
    x = torch.tensor([1.0, 2.0], requires_grad=True)
    y = f(x)
    rows = []
    for i in range(2):
        e = torch.zeros(2)
        e[i] = 1.0
        x.grad = None
        y.backward(e, retain_graph=True)
        rows.append(x.grad.clone())
    J = torch.stack(rows)
    print("  모은 J:", [[round(v, 6) for v in r] for r in J.tolist()])
    print("  jacrev 와 같은가:",
          torch.allclose(J, jacrev(f)(torch.tensor([1.0, 2.0]))))
    print("  출력이 2 개라 backward 를 2 번 불렀다")


# ========================================================================
# 3. 수치 미분으로 검산
# ========================================================================

def numeric_check():
    header("3. 수치 미분")

    def f64(t):
        return torch.stack([t[0] ** 2 * t[1],
                            torch.sin(t[0]) + t[1] ** 3])

    def numeric(fn, x, h=1e-4):
        cols = []
        for j in range(len(x)):
            xp, xm = x.clone(), x.clone()
            xp[j] += h
            xm[j] -= h
            cols.append((fn(xp) - fn(xm)) / (2 * h))
        return torch.stack(cols, dim=1)

    x = torch.tensor([1.0, 2.0], dtype=torch.float64)
    Jn = numeric(f64, x)
    Ja = jacrev(f64)(x)
    print("  수치:", [[round(v, 6) for v in r] for r in Jn.tolist()])
    print("  자동:", [[round(v, 6) for v in r] for r in Ja.tolist()])
    print(f"  가장 큰 차: {(Jn - Ja).abs().max():.2e}")
    print("  float64 로 해야 이만큼 맞는다. float32 면 한참 거칠다")


# ========================================================================
# 4. 세 길의 값
# ========================================================================

def costs():
    header("4. 셋 중 무엇을 쓰는가")
    print("  VJP 모으기  : 출력 수만큼 backward. retain_graph 와 grad 비우기를 챙겨야")
    print("  jacrev      : 같은 일을 라이브러리가 한다. 짧고 안전하다")
    print("  수치 미분    : 입력 수의 두 배만큼 순전파. 느리고 거칠다")
    print("  수치 미분의 쓸모는 속도가 아니라 **검산**이다")
    print("  손으로 짠 backward 가 맞는지 보려면 이것 말고 길이 없다")


# ========================================================================
# 5. 그런데 보통은 야코비가 필요 없다
# ========================================================================

def usually_not_needed():
    header("5. 보통은 안 쓴다")
    x = torch.tensor([1.0, 2.0], requires_grad=True)
    loss = f(x).pow(2).sum()          # 홑값으로 줄인다
    loss.backward()
    print("  로스를 홑값으로 줄이면 backward 한 번")
    print("  x.grad:", [round(v, 6) for v in x.grad.tolist()])
    J = jacrev(f)(torch.tensor([1.0, 2.0]))
    y = f(torch.tensor([1.0, 2.0]))
    print("  야코비로 손계산 (2y)^T J:",
          [round(v, 6) for v in (J.t() @ (2 * y)).tolist()])
    print("  같은 값이다. 야코비를 거치지 않고 바로 얻은 것뿐이다")


def main():
    varies()
    by_vjp()
    numeric_check()
    costs()
    usually_not_needed()


if __name__ == "__main__":
    main()
