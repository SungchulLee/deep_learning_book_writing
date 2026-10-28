"""행렬 연산 — 어느 것을 쓰고 무엇을 믿을 수 있는가.

행렬곱 하나를 적는 길이 여럿이고(`*`, `@`, `mm`, `bmm`, `matmul`,
`einsum`) 선형대수 함수도 여럿이다. 외울 것은 세 가지다.

- `*` 는 원소별, `@` 는 행렬곱이다. 둘 다 터지지 않고 다른 답을 줄 수 있다
- `matmul` 은 브로드캐스팅하고 `mm`·`bmm` 은 모양을 꼭 맞춰야 한다
- **역행렬을 구해 곱하지 말고 `solve` 를 쓴다.** 그리고 float32 에서
  믿을 수 있는 자릿수는 **조건수가 정한다**

실행: python 14_matrix_operations.py
"""
import math
import time

import torch
import torch.nn.functional as F


def header(title):
    print(f"\n=== {title} ===")


def spread(fn, rep=7):
    ts = []
    for _ in range(rep):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return min(ts) * 1e3, max(ts) * 1e3


# ========================================================================
# 1. * 와 @
# ========================================================================

def elementwise_vs_matmul():
    header("1. * 와 @")
    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    b = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
    print("  a * b (원소별):", (a * b).flatten().tolist())
    print("  a @ b (행렬곱):", (a @ b).flatten().tolist())
    print("  정사각이라 모양이 같아 터지지 않는다. 값만 다르다")
    print("  a.mul(b) 와 a.matmul(b) 로도 적는다")

    v = torch.tensor([1.0, 2.0])
    print("  행렬 @ 벡터:", (a @ v).tolist(), " 모양", tuple((a @ v).shape))
    print("  벡터 @ 벡터 (내적):", torch.dot(v, v).item(), "=", (v @ v).item())
    print("  바깥곱 outer:", torch.outer(v, v).flatten().tolist())


# ========================================================================
# 2. mm / bmm / matmul 이 받는 모양
# ========================================================================

def which_matmul():
    header("2. mm / bmm / matmul")
    A2, B2 = torch.randn(3, 4), torch.randn(4, 5)
    A3, B3 = torch.randn(2, 3, 4), torch.randn(2, 4, 5)
    cases = (("mm    (2D,2D)", torch.mm, (A2, B2)),
             ("mm    (3D,3D)", torch.mm, (A3, B3)),
             ("bmm   (3D,3D)", torch.bmm, (A3, B3)),
             ("bmm   (2D,2D)", torch.bmm, (A2, B2)),
             ("matmul(2D,2D)", torch.matmul, (A2, B2)),
             ("matmul(3D,3D)", torch.matmul, (A3, B3)),
             ("matmul(3D,2D)", torch.matmul, (A3, B2)))
    for name, fn, args in cases:
        try:
            print(f"  {name} -> {tuple(fn(*args).shape)}")
        except RuntimeError as e:
            print(f"  {name} -> {str(e).splitlines()[0][:44]}")
    print("  @ 는 matmul 이다. 브로드캐스팅까지 해 주므로 보통 이것만 쓰면 된다")


# ========================================================================
# 3. 어텐션에 1/sqrt(d) 가 붙는 까닭
# ========================================================================

def why_scale():
    header("3. 어텐션의 1/sqrt(d)")
    torch.manual_seed(0)
    for d in (8, 64, 512):
        Q, K = torch.randn(1, 10, d), torch.randn(1, 10, d)
        s = Q @ K.transpose(-2, -1)
        for label, sc in (("조정 없이  ", s), ("1/sqrt(d) 로", s / math.sqrt(d))):
            p = F.softmax(sc, dim=-1)
            ent = -(p * p.clamp_min(1e-12).log()).sum(-1).mean()
            print(f"  d={d:4d} {label} 점수 표준편차 {sc.std():6.3f}  "
                  f"최대 확률 {p.max(dim=-1).values.mean():.4f}  "
                  f"엔트로피 {ent:.4f}")
    print("  조정 없이 d 가 크면 softmax 가 한 자리에 쏠린다")
    print("  Q·K 는 d 개를 더한 값이라 표준편차가 sqrt(d) 로 자란다")
    print("  그래서 sqrt(d) 로 나누면 d 와 무관하게 1 쯤에서 머문다")


# ========================================================================
# 4. 역행렬을 구해 곱하지 않는다
# ========================================================================

def solve_not_inverse():
    header("4. inv 로 풀기 vs solve")
    torch.manual_seed(0)
    n = 500
    A = torch.randn(n, n)
    A = A @ A.t() + n * torch.eye(n)          # 대칭이고 잘 조건지어진 것
    b = torch.randn(n, 1)
    exact = torch.linalg.solve(A.double(), b.double())
    for name, x in (("inv(A) @ b ", torch.linalg.inv(A) @ b),
                    ("solve(A, b)", torch.linalg.solve(A, b))):
        err = (x.double() - exact).abs().max().item()
        print(f"  {name}  참값과의 차 {err:.3e}")
    t1 = spread(lambda: torch.linalg.inv(A) @ b)
    t2 = spread(lambda: torch.linalg.solve(A, b))
    print(f"  inv(A) @ b   {t1[0]:6.2f} ~ {t1[1]:6.2f} ms")
    print(f"  solve(A, b)  {t2[0]:6.2f} ~ {t2[1]:6.2f} ms")
    print("  잘 조건지어진 행렬에서는 정확도가 비슷하고 solve 가 빠르다")
    print("  나쁘게 조건지어지면 정확도도 갈린다 — 다음 절")


# ========================================================================
# 5. 조건수가 믿을 자릿수를 정한다
# ========================================================================

def conditioning():
    header("5. 조건수")
    torch.manual_seed(0)
    Q, _ = torch.linalg.qr(torch.randn(50, 50))
    for c in (1e1, 1e4, 1e7):
        d = torch.logspace(0, math.log10(c), 50)
        M = Q @ torch.diag(d) @ Q.t()
        err = (M @ torch.linalg.inv(M) - torch.eye(50)).abs().max().item()
        print(f"  조건수 {c:7.0e}  A @ inv(A) 가 I 에서 벗어난 최대 {err:.3e}")
    print("  float32 는 유효숫자가 약 7 자리다")
    print("  조건수가 10^k 면 그중 k 자리를 잃는다고 보면 맞는다")
    print("  조건수 10^7 이면 남는 자리가 없어 4% 가까이 어긋난다")


# ========================================================================
# 6. einsum — 하나로 다 적는다
# ========================================================================

def einsum_demo():
    header("6. einsum")
    torch.manual_seed(0)
    A, B = torch.randn(3, 4), torch.randn(4, 5)
    v, w = torch.randn(3), torch.randn(3)
    T = torch.randn(2, 3, 4)
    U = torch.randn(2, 4, 5)
    checks = (
        ("내적      i,i->  ", torch.einsum("i,i->", v, w), torch.dot(v, w)),
        ("바깥곱    i,j->ij", torch.einsum("i,j->ij", v, v), torch.outer(v, v)),
        ("행렬곱    ik,kj->ij", torch.einsum("ik,kj->ij", A, B), A @ B),
        ("전치      ij->ji", torch.einsum("ij->ji", A), A.t()),
        ("대각합    ii->", torch.einsum("ii->", torch.eye(4)), torch.tensor(4.0)),
        ("배치곱    bik,bkj->bij", torch.einsum("bik,bkj->bij", T, U), T @ U),
    )
    for name, got, want in checks:
        print(f"  {name:22s} 같은가: {torch.allclose(got, want, atol=1e-5)}")
    print("  첨자에 없는 글자는 더해서 없앤다 — 그것이 규칙 전부다")


# ========================================================================
# 7. 분해와 나머지
# ========================================================================

def decompositions():
    header("7. 분해들")
    torch.manual_seed(0)
    A = torch.randn(4, 3)
    U, S, Vh = torch.linalg.svd(A, full_matrices=False)
    print("  SVD   A = U S Vh 로 되돌아오는가:",
          torch.allclose(U @ torch.diag(S) @ Vh, A, atol=1e-5))
    print("    특이값:", [round(v, 4) for v in S.tolist()])
    Q, R = torch.linalg.qr(A)
    print("  QR    A = Q R 로 되돌아오는가:",
          torch.allclose(Q @ R, A, atol=1e-5))
    P = torch.randn(3, 3); P = P @ P.t() + 3 * torch.eye(3)
    L = torch.linalg.cholesky(P)
    print("  촐레스키 L @ L.T 로 되돌아오는가:",
          torch.allclose(L @ L.t(), P, atol=1e-4))
    print("  rank:", torch.linalg.matrix_rank(A).item(),
          " det(3x3):", round(torch.linalg.det(P).item(), 4),
          " trace:", round(torch.trace(P).item(), 4))
    print("  trace 는 대각의 합이다:",
          round(torch.diag(P).sum().item(), 4))


def main():
    elementwise_vs_matmul()
    which_matmul()
    why_scale()
    solve_not_inverse()
    conditioning()
    einsum_demo()
    decompositions()


if __name__ == "__main__":
    main()
