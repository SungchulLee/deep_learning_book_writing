"""모양·자료형·장치 — 연산이 되는지를 정하는 세 가지.

텐서 둘을 더할 수 있는지는 속에 든 수가 아니라 이 세 속성이 정한다.
그런데 셋이 어긋났을 때 하는 일이 같지 않다.

- 모양이 어긋나면 터진다
- 장치가 어긋나면 터진다
- 자료형이 어긋나면 **조용히 한쪽으로 올라간다**

그래서 앞의 둘은 고치면 끝이고, 자료형은 터지지 않기 때문에 찾아야
한다. 뒤이어 `.to()` 가 언제 같은 객체를 돌려주는지와, `T`·`mT`·`H`
가 2차원에서만 셋이 같은 것을 본다.

실행: python 03_shape_dtype_device.py
"""
import warnings

import torch


def header(title):
    print(f"\n=== {title} ===")


# ========================================================================
# 1. 세 속성을 읽는다
# ========================================================================

def reading():
    header("1. 속성 읽기")
    x = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    print("x =")
    print(x)
    print("  x.shape       :", x.shape, " (x.size() 와 같다)")
    print("  x.ndim        :", x.ndim, " = len(x.shape)")
    print("  x.numel()     :", x.numel())
    print("  x.dtype       :", x.dtype)
    print("  x.device      :", x.device)
    print("  x.layout      :", x.layout)
    print("  x.element_size:", x.element_size(), "바이트")
    print("  차지하는 메모리:", x.numel() * x.element_size(), "바이트")


# ========================================================================
# 2. 모양이 어긋나면 터진다 — 다만 브로드캐스트가 먼저 손을 쓴다
# ========================================================================

def shapes():
    header("2. 모양이 어긋나면")
    a = torch.ones(3, 4)
    try:
        a + torch.ones(4, 3)
    except RuntimeError as e:
        print("  (3,4) + (4,3) :", str(e).splitlines()[0])
    print("  (3,4) + (4,)  :", tuple((a + torch.ones(4)).shape),
          "— 브로드캐스트가 받아 준다")
    print("  (3,4) + (3,1) :", tuple((a + torch.ones(3, 1)).shape))
    try:
        a @ torch.ones(3, 4)
    except RuntimeError as e:
        print("  (3,4) @ (3,4) :", str(e).splitlines()[0])
    print(f"  (3,4) 와 (4,3) 은 원소가 둘 다 {a.numel()}개인데도 더할 수 없다")


# ========================================================================
# 3. 장치가 어긋나면 터진다
# ========================================================================

def devices():
    header("3. 장치가 어긋나면")
    dev = ("cuda" if torch.cuda.is_available()
           else "mps" if torch.backends.mps.is_available() else None)
    if dev is None:
        print("  가속기가 없어 cpu 뿐이다 — 이 절은 건너뛴다")
        return
    try:
        torch.ones(3) + torch.ones(3, device=dev)
    except RuntimeError as e:
        print(f"  cpu + {dev} :", str(e).splitlines()[0])
    print("  장치는 자료형과 달라서, 한쪽으로 자동으로 옮겨 주지 않는다")


# ========================================================================
# 4. 자료형이 어긋나면 조용히 올라간다
# ========================================================================

def dtypes():
    header("4. 자료형이 어긋나면")
    f32 = torch.ones(3, dtype=torch.float32)
    cases = (
        ("0 차원 float64", torch.tensor(1.0, dtype=torch.float64)),
        ("1 개짜리 float64", torch.ones(1, dtype=torch.float64)),
        ("3 개짜리 float64", torch.ones(3, dtype=torch.float64)),
        ("파이썬 float 1.0", 1.0),
    )
    for name, other in cases:
        pad = " " * max(0, 18 - sum(2 if "가" <= c <= "힣" else 1 for c in name))
        print(f"  float32 + {name}{pad}-> {(f32 + other).dtype}")
    print("  모양이 있는 float64 만 이긴다. 0 차원은 못 이긴다")

    i = torch.tensor([1, 2, 3])
    print("  int / 2       :", (i / 2).dtype, (i / 2).tolist(), "— 나누기는 실수로")
    print("  int // 2      :", (i // 2).dtype, (i // 2).tolist())
    try:
        i.mean()
    except RuntimeError as e:
        print("  int.mean()    :", str(e).splitlines()[0][:72])


# ========================================================================
# 5. .to() 는 바꿀 것이 없으면 그 텐서를 그대로 돌려준다
# ========================================================================

def conversion():
    header("5. .to() 가 언제 새것을 만드는가")
    x = torch.ones(3)
    print("  x.to(torch.float32) is x :", x.to(torch.float32) is x)
    print("  x.to('cpu')         is x :", x.to("cpu") is x)
    print("  x.to(torch.float64) is x :", x.to(torch.float64) is x)
    dev = ("cuda" if torch.cuda.is_available()
           else "mps" if torch.backends.mps.is_available() else None)
    if dev:
        y = x.to(dev)
        print(f"  x.to('{dev}')         is x :", y is x)
        print(f"  y.to('{dev}')         is y :", y.to(dev) is y)
    print("  그래서 x.to(...) 를 부르고 x 를 계속 쓰면 아무 일도 안 한 것이 된다")
    x.to(torch.float64)
    print("  x.to(torch.float64) 만 부른 뒤 x.dtype :", x.dtype)


# ========================================================================
# 6. T / mT / H — 2차원에서만 셋이 같다
# ========================================================================

def transposes():
    header("6. T / mT / H")
    m = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    print("  2차원에서는 셋이 같다:",
          torch.equal(m.T, m.mT) and torch.equal(m.mT, m.H))

    r3 = torch.arange(24).reshape(2, 3, 4)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        shape_T = tuple(r3.T.shape)
        note = str(caught[0].message).split(" and it will")[0] if caught else ""
    print("  3차원 .T  :", shape_T, "— 축을 전부 거꾸로 돌린다")
    print("    경고    :", note)
    print("  3차원 .mT :", tuple(r3.mT.shape), "— 마지막 두 축만 바꾼다")
    try:
        r3.H
    except RuntimeError as e:
        print("  3차원 .H  :", str(e).splitlines()[0][:66])

    c = torch.tensor([[1 + 2j, 3 - 1j]])
    print("  복소수 mT :", c.mT.flatten().tolist())
    print("  복소수 H  :", c.H.flatten().tolist(), "— 켤레까지 취한다")


def main():
    reading()
    shapes()
    devices()
    dtypes()
    conversion()
    transposes()


if __name__ == "__main__":
    main()
