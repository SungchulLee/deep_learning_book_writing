"""계산 그래프 재사용 — 무엇이 해제되고 무엇이 남는가.

`backward()` 를 두 번 부르면 터진다고 흔히 말한다. 정확히는
**역전파가 저장해 둔 중간값이 해제되어** 터지는 것이고, 저장할 것이
없는 그래프는 두 번 불러도 멀쩡하다.

`retain_graph=True` 는 그 중간값을 남겨 달라는 부탁이다. 경사를
지우는 일과는 아무 상관이 없다 — 흔히 섞이는 자리다.

- 그래프: 순전파가 세운 길과 거기 저장된 중간값. `retain_graph` 가 맡는다
- 경사: 잎에 쌓이는 값. `zero_grad()` 가 맡는다

실행: python 03_retain_graph.py
"""
import torch


def header(title):
    print(f"\n=== {title} ===")


# ========================================================================
# 1. 두 번 부르면 터지는 그래프와 안 터지는 그래프
# ========================================================================

def twice():
    header("1. 두 번 부르면")
    builders = (("x.sum()", lambda t: t.sum(), "저장할 것이 없다"),
                ("(x**2).sum()", lambda t: (t ** 2).sum(), "x 를 저장한다"),
                ("x.exp().sum()", lambda t: t.exp().sum(), "exp(x) 를 저장한다"))
    for expr, build, why in builders:
        x = torch.tensor([1.0, 2.0], requires_grad=True)
        y = build(x)
        y.backward()
        try:
            y.backward()
            print(f"  {expr:14s} 두 번째도 된다   ({why})")
        except RuntimeError as e:
            print(f"  {expr:14s} {str(e).splitlines()[0][:40]}...  ({why})")
    print("  '그래프가 사라져서' 가 아니라 '저장해 둔 값이 해제되어서' 다")


# ========================================================================
# 2. retain_graph 가 하는 일
# ========================================================================

def retaining():
    header("2. retain_graph=True")
    torch.manual_seed(0)
    x = torch.randn(3, requires_grad=True)
    loss = (x ** 2).sum()
    loss.backward(retain_graph=True)
    print("  1번째 뒤 x.grad :", [round(v, 4) for v in x.grad.tolist()])
    loss.backward()                       # 남겨 둔 그래프를 다시 쓴다
    print("  2번째 뒤 x.grad :", [round(v, 4) for v in x.grad.tolist()])
    print("  2x 가 두 번 더해져 4x 가 되었다")
    print("  손으로 4x        :", [round(v, 4) for v in (4 * x.detach()).tolist()])
    try:
        loss.backward()                   # 이번에는 안 남겼으므로
    except RuntimeError as e:
        print("  3번째:", str(e).splitlines()[0][:56])
    print("  마지막 역전파에는 retain_graph 를 주지 않는 것이 보통이다")


# ========================================================================
# 3. 그래프와 경사는 다른 것이다
# ========================================================================

def graph_is_not_grad():
    header("3. 섞이는 자리")
    torch.manual_seed(0)
    x = torch.randn(3, requires_grad=True)

    loss = (x ** 2).sum()
    loss.backward(retain_graph=True)
    first = x.grad.clone()
    x.grad.zero_()                        # 경사만 지운다
    loss.backward()                       # 그래프는 아직 살아 있다
    print("  경사를 지워도 그래프는 그대로:",
          torch.allclose(first, x.grad))
    print("  retain_graph 는 그래프를, zero_() 는 경사를 맡는다")

    x.grad.zero_()
    fresh = (x ** 2).sum()                # 순전파를 다시 하면
    fresh.backward()                      # 새 그래프라 retain 이 필요 없다
    print("  순전파를 다시 하면 새 그래프가 생긴다:",
          torch.allclose(first, x.grad))
    print("  학습 루프가 이렇게 돈다. 그래서 retain_graph 가 거의 필요 없다")


# ========================================================================
# 4. 정말 필요한 두 자리
# ========================================================================

def when_needed():
    header("4. 정말 필요한 두 자리")

    # (가) 한 번의 순전파에서 나온 로스 둘을 따로 역전파할 때
    x = torch.tensor([1.0, 2.0], requires_grad=True)
    h = x * 3                              # 둘이 함께 쓰는 중간값
    a = (h ** 2).sum()
    b = h.abs().sum()
    a.backward(retain_graph=True)          # h 를 b 도 써야 한다
    b.backward()
    print("  (가) 로스 둘을 따로:", [round(v, 4) for v in x.grad.tolist()])
    y = torch.tensor([1.0, 2.0], requires_grad=True)
    g = y * 3
    ((g ** 2).sum() + g.abs().sum()).backward()
    print("       합쳐서 한 번  :", [round(v, 4) for v in y.grad.tolist()])
    print("       같다. 다만 따로 부르려면 retain_graph 가 필요하다")

    # (나) 경사의 경사 — create_graph 가 retain_graph 를 켠다
    z = torch.tensor([2.0], requires_grad=True)
    first, = torch.autograd.grad((z ** 3).sum(), z, create_graph=True)
    second, = torch.autograd.grad(first.sum(), z)
    print("  (나) f = z^3, z = 2")
    print("       1차 도함수 3z^2 =", first.item(), " 손으로", 3 * 2 ** 2)
    print("       2차 도함수 6z   =", second.item(), " 손으로", 6 * 2)
    print("       create_graph=True 는 retain_graph 까지 켜 준다")


def main():
    twice()
    retaining()
    graph_is_not_grad()
    when_needed()


if __name__ == "__main__":
    main()
