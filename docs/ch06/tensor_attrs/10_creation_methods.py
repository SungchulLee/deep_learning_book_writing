"""텐서를 만드는 길들 — 베끼는 것, 나눠 쓰는 것, 아무것도 안 하는 것.

만드는 함수가 많지만 외울 것은 몇 가지뿐이다.

- `tensor` 는 베끼고 `as_tensor`·`from_numpy` 는 나눠 쓴다
- `empty` 는 **아무것도 하지 않는다**. 그 자리에 있던 값이 그대로 나온다
- `arange` 는 끝을 뺀다고 적혀 있지만 부동소수점 때문에 들어오는 자리가 있다.
  개수를 내가 정하려면 `linspace` 다
- `*_like` 는 dtype 과 device 는 따라가고 `requires_grad` 는 안 따라간다

실행: python 10_creation_methods.py
"""
import numpy as np
import torch


def header(title):
    print(f"\n=== {title} ===")


# ========================================================================
# 1. 베끼는가 나눠 쓰는가
# ========================================================================

def copy_or_share():
    header("1. tensor / as_tensor / from_numpy")
    arr = np.array([[1, 2], [3, 4]])
    for name, fn in (("torch.tensor", torch.tensor),
                     ("torch.as_tensor", torch.as_tensor),
                     ("torch.from_numpy", torch.from_numpy)):
        t = fn(arr)
        print(f"  {name:18s} dtype {str(t.dtype):12s} "
              f"메모리 공유 {np.shares_memory(arr, t.numpy())}")
    arr[0, 0] = 999
    print("  배열을 999 로 고친 뒤")
    print("    torch.tensor 로 만든 것  :", torch.tensor([[1, 2], [3, 4]])[0, 0].item(),
          "(미리 베꼈다)")
    print("    torch.as_tensor 로 만든 것:", torch.as_tensor(arr)[0, 0].item())

    f32 = np.array([[1.0, 2.0]], dtype=np.float32)
    shared = np.shares_memory(f32, torch.as_tensor(f32, dtype=torch.float64).numpy())
    print("  as_tensor 에 dtype 을 바꿔 달라면 공유:", shared,
          "— 바꾸려면 베낄 수밖에 없다")


# ========================================================================
# 2. empty 는 아무것도 하지 않는다
# ========================================================================

def empty_is_garbage():
    header("2. empty 가 주는 것")
    a = torch.empty(5)
    print("  갓 잡은 empty(5):", a.tolist())
    print("  0 처럼 보인다. 운영체제가 새 쪽을 0 으로 비워 주기 때문이다")

    dirty = torch.empty(1000)
    dirty.fill_(12345.0)       # 그 자리를 더럽혀 놓고
    del dirty                  # 놓아 준다
    b = torch.empty(1000)      # 같은 자리를 다시 받는다
    print("  더럽힌 뒤 다시 잡은 empty(1000) 의 앞 다섯:",
          [round(v, 1) for v in b[:5].tolist()])
    print("  0 이 아닌 칸:", int((b != 0).sum()), "/ 1000")
    print("  방금 놓아 준 내 데이터가 그대로 나온다")
    print("  그러니 empty 로 만든 텐서는 반드시 채우고 나서 읽어야 한다")


# ========================================================================
# 3. 값을 정해 만들기
# ========================================================================

def constants():
    header("3. 값을 정해 만들기")
    print("  zeros(2,3)      :", torch.zeros(2, 3).flatten().tolist())
    print("  ones(2,3)       :", torch.ones(2, 3).flatten().tolist())
    print("  full((2,3), 7.5):", torch.full((2, 3), 7.5).flatten().tolist())
    print("  eye(3)          :", torch.eye(3).tolist())
    print("  eye(2,4) 정사각이 아니어도:", torch.eye(2, 4).tolist())

    v = torch.tensor([1.0, 2.0, 3.0])
    print("  diag(벡터) 는 행렬을 만든다:", torch.diag(v).shape)
    m = torch.arange(9).reshape(3, 3)
    print("  diag(행렬) 은 대각을 뽑는다:", torch.diag(m).tolist())
    print("  diag(행렬, 1) 한 칸 위  :", torch.diag(m, 1).tolist())
    print("  같은 이름인데 입력 차원에 따라 하는 일이 반대다")


# ========================================================================
# 4. 수열 — arange 와 linspace
# ========================================================================

def sequences():
    header("4. arange 와 linspace")
    for a, b, s in ((0, 1, 0.1), (0, 0.3, 0.1), (1, 1.3, 0.1)):
        r = torch.arange(a, b, s)
        inc = abs(r[-1].item() - b) < 1e-6
        print(f"  arange({a}, {b}, {s}) -> {r.numel()}개  "
              f"끝값 {b} 가 들어갔는가: {inc}")
    print("  셋 다 폭이 같은데 마지막만 끝값을 넣었다")
    print("  (1.3-1)/0.1 이 부동소수점에서 3.0000000000000004 가 되기 때문이다")
    print("  linspace(1, 1.2, 3):", torch.linspace(1, 1.2, 3).tolist())
    print("  linspace 는 개수를 내가 적으므로 이런 일이 없다")

    r64 = torch.arange(0, 1, 0.1, dtype=torch.float64)
    print("  그리고 값 자체도 정확하지 않다")
    print("    arange(0,1,0.1)[3] =", repr(r64[3].item()))
    print("    == 0.3 인가:", r64[3].item() == 0.3)


# ========================================================================
# 5. 난수 — 그리고 씨앗
# ========================================================================

def randoms():
    header("5. 난수")
    torch.manual_seed(42)
    print("  rand(2,3)   [0,1) 고른분포:",
          [round(v, 4) for v in torch.rand(2, 3).flatten().tolist()])
    print("  randn(2,3)  N(0,1)       :",
          [round(v, 4) for v in torch.randn(2, 3).flatten().tolist()])
    print("  randint(0,10,(5,))       :", torch.randint(0, 10, (5,)).tolist())
    print("  randperm(8)              :", torch.randperm(8).tolist())

    torch.manual_seed(42)
    first = torch.rand(3)
    torch.manual_seed(42)
    print("  같은 씨앗으로 두 번:", torch.equal(first, torch.rand(3)))

    g = torch.Generator().manual_seed(7)
    print("  Generator 를 따로 두면:",
          [round(v, 4) for v in torch.rand(3, generator=g).tolist()])
    print("  전역 씨앗을 건드리지 않고 재현할 수 있다")


# ========================================================================
# 6. _like 는 어디까지 따라가는가
# ========================================================================

def like_family():
    header("6. *_like")
    tpl = torch.randn(3, 4, dtype=torch.float64, requires_grad=True)
    z = torch.zeros_like(tpl)
    print(f"  본   dtype {tpl.dtype} device {tpl.device} "
          f"requires_grad {tpl.requires_grad}")
    print(f"  like dtype {z.dtype} device {z.device} "
          f"requires_grad {z.requires_grad}")
    print("  dtype 과 device 는 따라가고 requires_grad 는 안 따라간다")
    print("  dtype 만 바꿔 달라고 할 수도 있다:",
          torch.zeros_like(tpl, dtype=torch.int32).dtype)


# ========================================================================
# 7. 장치와 requires_grad
# ========================================================================

def device_and_grad():
    header("7. 만들 때 정하는 것")
    dev = ("cuda" if torch.cuda.is_available()
           else "mps" if torch.backends.mps.is_available() else "cpu")
    t = torch.randn(2, 2, device=dev)
    print(f"  device='{dev}' 로 바로 만들기:", t.device)
    print("  만들고 나서 옮기면 한 번 더 베낀다 — 처음부터 거기서 만드는 편이 낫다")
    x = torch.randn(2, 2, requires_grad=True)
    print("  requires_grad=True 로 만들기:", x.requires_grad, " 잎인가:", x.is_leaf)


def main():
    copy_or_share()
    empty_is_garbage()
    constants()
    sequences()
    randoms()
    like_family()
    device_and_grad()


if __name__ == "__main__":
    main()
