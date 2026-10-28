"""벡터 역전파의 함정 — 터지는 것과 조용한 것.

「벡터 출력의 역전파」부터 여기까지 `backward(v)` 를 여러 각도로
보았다. 이 쪽은 **틀렸을 때 어떻게 드러나는지**로 모아 정리한다.

터져 주는 쪽
- v 를 아예 안 주면
- v 의 모양이 출력과 다르면

조용한 쪽
- v 를 비우지 않고 여러 번 부르면 (경사가 쌓인다)
- v 에 requires_grad 를 켜면 (아무 일도 안 일어난다)
- v 의 자료형이 다르면 (말없이 맞춰 준다)

실행: python 26_pitfalls_of_vector_backward.py
"""
import torch


def header(title):
    print(f"\n=== {title} ===")


# ========================================================================
# 1. 터져 주는 둘
# ========================================================================

def the_loud_ones():
    header("1. 터져 주는 둘")
    x = torch.randn(3, requires_grad=True)
    try:
        (x ** 2).backward()
    except RuntimeError as e:
        print("  v 를 안 주면 :", str(e).splitlines()[0])
    for shape in ((4,), (3, 1), ()):
        xx = torch.randn(3, requires_grad=True)
        try:
            (xx ** 2).backward(torch.ones(shape))
            print(f"  v 모양 {str(shape):6s} -> 통과")
        except RuntimeError as e:
            print(f"  v 모양 {str(shape):6s} -> {str(e).splitlines()[0][:50]}")
    print("  둘 다 고치기 쉽다. 메시지가 무엇이 어긋났는지 적어 준다")


# ========================================================================
# 2. 조용한 것 하나 — 쌓인다
# ========================================================================

def accumulates():
    header("2. 조용한 것 — 쌓인다")
    torch.manual_seed(0)
    A = torch.randn(3, 3)
    for clear in (True, False):
        x = torch.randn(3, requires_grad=True)
        y = A @ x
        rows = []
        for i in range(3):
            e = torch.zeros(3)
            e[i] = 1.0
            if clear:
                x.grad = None
            y.backward(e, retain_graph=True)
            rows.append(x.grad.clone())
        J = torch.stack(rows)
        print(f"  비움={clear!s:5s} J == A 인가: {torch.allclose(J, A)}")
        if not clear:
            print("    첫 줄은 맞다:", torch.allclose(J[0], A[0]))
            print("    둘째 줄 = A[0]+A[1] 인가:",
                  torch.allclose(J[1], A[0] + A[1]))
    print("  모양도 맞고 첫 줄도 맞아서 검산이 쉽게 속는다")


# ========================================================================
# 3. 조용한 것 둘 — v 의 requires_grad
# ========================================================================

def v_requires_grad():
    header("3. 조용한 것 — v 에 깃발을 켜면")
    x = torch.randn(3, requires_grad=True)
    v = torch.ones(3, requires_grad=True)
    (x ** 2).backward(v)
    print("  통과한다. x.grad :", [round(t, 4) for t in x.grad.tolist()])
    print("  그런데 v.grad    :", v.grad)
    print("  v 는 '가중치' 일 뿐 미분 대상이 아니다. 깃발은 무시된다")
    print("  v 에 대한 민감도가 필요하면 식을 다시 세워야 한다")
    x2 = torch.randn(3, requires_grad=True)
    v2 = torch.ones(3, requires_grad=True)
    (v2 * x2 ** 2).sum().backward()
    print("  (v*y).sum() 으로 적으면 v.grad 도 온다:",
          [round(t, 4) for t in v2.grad.tolist()])


# ========================================================================
# 4. 조용한 것 셋 — 자료형
# ========================================================================

def dtype_of_v():
    header("4. 조용한 것 — v 의 자료형")
    for dt in (torch.float32, torch.float64, torch.float16, torch.int64):
        x = torch.randn(3, requires_grad=True)
        (x ** 2).backward(torch.ones(3, dtype=dt))
        print(f"  v dtype {str(dt):16s} -> 통과, x.grad dtype {x.grad.dtype}")
    print("  말없이 맞춰 준다. 정수 v 도 받는다")
    print("  0.5 같은 가중치를 정수 텐서에 담으면 0 으로 잘린다")
    x = torch.randn(3, requires_grad=True)
    (x ** 2).backward(torch.tensor([1, 0, 1]))          # 정수
    print("  정수 [1,0,1] 로 준 x.grad:", [round(t, 4) for t in x.grad.tolist()])


# ========================================================================
# 5. 쌓임을 아예 피하는 길
# ========================================================================

def use_autograd_grad():
    header("5. autograd.grad 로 피하기")
    torch.manual_seed(0)
    A = torch.randn(3, 3)
    x = torch.randn(3, requires_grad=True)
    y = A @ x
    rows = []
    for i in range(3):
        e = torch.zeros(3)
        e[i] = 1.0
        g, = torch.autograd.grad(y, x, e, retain_graph=True)
        rows.append(g)
    J = torch.stack(rows)
    print("  autograd.grad 로 모은 J == A :", torch.allclose(J, A))
    print("  x.grad 는 그대로 :", x.grad)
    print("  .grad 를 건드리지 않으니 비울 일도, 쌓일 일도 없다")
    print("  야코비를 모을 때는 이쪽이 안전하다")


# ========================================================================
# 6. v 의 부호
# ========================================================================

def sign_of_v():
    header("6. v 의 부호")
    for sign, label in ((1.0, "v=+1 (경사 하강)"), (-1.0, "v=-1 (경사 상승)")):
        x = torch.tensor([3.0], requires_grad=True)
        opt = torch.optim.SGD([x], lr=0.1)
        for _ in range(5):
            opt.zero_grad()
            (x ** 2).backward(torch.tensor([sign]))
            opt.step()
        print(f"  {label}  3.0 -> {x.item():.4f}")
    print("  부호를 뒤집으면 최솟값에서 멀어진다")
    print("  GAN 처럼 올리고 싶은 로스가 있을 때 쓰는 길이기도 하다")


def main():
    the_loud_ones()
    accumulates()
    v_requires_grad()
    dtype_of_v()
    use_autograd_grad()
    sign_of_v()


if __name__ == "__main__":
    main()
