"""텐서를 NumPy 로 넘기기 — 거절당하는 두 자리와 공유되는 메모리.

`.numpy()` 는 세 가지를 한꺼번에 묻는다.

- **그래프에 매여 있는가** — `requires_grad=True` 면 거절한다
- **어느 장치에 있는가** — CPU 가 아니면 거절한다
- **메모리를 나눠 쓸 것인가** — 거절하지 않는다. 그냥 나눠 쓴다

앞의 둘은 터져 주니 고치면 끝이다. 셋째가 조용하다.

실행: python 12_converting_to_numpy.py
"""
import numpy as np
import torch


def header(title):
    print(f"\n=== {title} ===")


# ========================================================================
# 1. 거절당하는 두 자리
# ========================================================================

def refusals():
    header("1. 거절")
    g = torch.randn(3, requires_grad=True)
    try:
        g.numpy()
    except RuntimeError as e:
        print("  requires_grad=True :", str(e).splitlines()[0])

    dev = ("cuda" if torch.cuda.is_available()
           else "mps" if torch.backends.mps.is_available() else None)
    if dev:
        m = torch.randn(3, device=dev)
        try:
            m.numpy()
        except TypeError as e:
            print(f"  device={dev:5s}      :", str(e).splitlines()[0])
        print("  .cpu().numpy() 로는 된다:", m.cpu().numpy().shape)
    else:
        print("  가속기가 없어 장치 쪽은 건너뛴다")
    print("  두 메시지 모두 고칠 방법을 함께 적어 준다")
    print("  그래서 x.detach().cpu().numpy() 세 토막이 관용구가 되었다")


# ========================================================================
# 2. 메모리는 조용히 공유된다
# ========================================================================

def sharing():
    header("2. 공유")
    t = torch.tensor([1.0, 2.0, 3.0])
    a = t.numpy()
    print("  t      :", t.tolist())
    a[0] = 99.0
    print("  a[0]=99 뒤 t:", t.tolist(), " <- 텐서가 바뀌었다")
    t[1] = 77.0
    print("  t[1]=77 뒤 a:", a.tolist(), " <- 배열도 바뀐다")
    print("  np.shares_memory:", np.shares_memory(t.numpy(), a))

    d = torch.randn(3, requires_grad=True).detach().numpy()
    print("  detach().numpy() 도 공유한다 — detach 가 저장소를 안 끊으니까")

    safe = torch.tensor([1.0, 2.0, 3.0]).numpy().copy()
    print("  끊으려면 .copy() 나 텐서 쪽 .clone() 이 필요하다:", safe.tolist())


# ========================================================================
# 3. 거꾸로 — numpy 에서 텐서로
# ========================================================================

def back():
    header("3. 거꾸로")
    arr = np.array([1.0, 2.0, 3.0])
    shared = torch.from_numpy(arr)
    copied = torch.tensor(arr)
    arr[0] = 55.0
    print("  배열을 55 로 고친 뒤")
    print("    from_numpy  :", shared.tolist(), " dtype", shared.dtype)
    print("    torch.tensor:", copied.tolist(), " dtype", copied.dtype)
    print("  numpy 의 기본 실수형이 float64 라 dtype 도 따라온다")

    f32 = torch.from_numpy(np.array([1.0, 2.0], dtype=np.float32))
    print("  np.float32 로 만들면:", f32.dtype)


# ========================================================================
# 4. 공유가 물어뜯는 자리
# ========================================================================

def the_bite():
    header("4. 공유가 물어뜯는 자리")
    w = torch.tensor([1.0, 2.0], requires_grad=True)
    y = w * 3
    loss = (y ** 2).sum()
    arr = y.detach().numpy()         # 역전파가 저장해 둔 값을 꺼냈다
    arr[0] = 99.0                    # 그것을 numpy 쪽에서 고친다
    try:
        loss.backward()
        print("  numpy 로 고친 뒤 backward: 통과")
        print("    w.grad =", [round(v, 1) for v in w.grad.tolist()],
              " 참값 18w =", [18.0, 36.0])
    except RuntimeError as e:
        print("  numpy 로 고친 뒤 backward:", str(e).splitlines()[0][:62])

    w2 = torch.tensor([1.0, 2.0], requires_grad=True)
    y2 = w2 * 3
    loss2 = (y2 ** 2).sum()
    arr2 = y2.detach().numpy().copy()    # 베껴서 꺼낸다
    arr2[0] = 99.0
    loss2.backward()
    print("  .copy() 로 꺼낸 뒤 backward: w.grad =",
          [round(v, 1) for v in w2.grad.tolist()])
    print("  넘겨준 배열을 고칠 작정이면 반드시 베껴야 한다")


def main():
    refusals()
    sharing()
    back()
    the_bite()


if __name__ == "__main__":
    main()
