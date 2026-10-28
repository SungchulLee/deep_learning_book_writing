"""데이터를 꺼내는 길 — 꺼낸 것이 원본과 아직 이어져 있는가.

텐서에서 값을 꺼내는 길은 여럿이다. `item()`·`tolist()`·`clone()`·
`detach()`·`numpy()`. 이 길들은 두 가지에서 갈린다.

- 저장소를 나눠 쓰는가 (고치면 원본도 바뀌는가)
- 그래프에 이어져 있는가 (경사가 거슬러 올라가는가)

둘은 따로 움직인다. `clone()` 은 저장소를 끊고 그래프는 남기며,
`detach()` 는 그래프를 끊고 저장소는 남긴다. 그리고 `.data` 는 둘 다
남긴 채로 autograd 의 검사만 지나간다 — 그래서 가장 위험하다.

실행: python 02_data_related.py
"""
import numpy as np
import torch
import torch.nn as nn


def header(title):
    print(f"\n=== {title} ===")


# ========================================================================
# 1. 홑값과 리스트로 꺼내기
# ========================================================================

def extracting():
    header("1. item() 과 tolist()")
    s = torch.tensor(3.14159)
    print("  s          :", s)
    print("  s.item()   :", repr(s.item()))
    print("  s.item() == 3.14159 ?", s.item() == 3.14159)
    print("  float64 로 담으면:",
          repr(torch.tensor(3.14159, dtype=torch.float64).item()))

    v = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    print("  v.tolist() :", v.tolist())
    print("  꺼낸 리스트는 파이썬 객체라 저장소를 나눠 쓰지 않는다")


# ========================================================================
# 2. 복사 셋 — 무엇을 끊고 무엇을 남기는가
# ========================================================================

def copies():
    header("2. clone / detach / detach().clone()")
    torch.manual_seed(0)
    a = torch.randn(4, requires_grad=True)
    cases = (
        ("a (원본)", a),
        ("a.clone()", a.clone()),
        ("a.detach()", a.detach()),
        ("a.detach().clone()", a.detach().clone()),
    )
    print(f"  {'':20s}{'공유':>6s}{'requires_grad':>16s}{'grad_fn':>18s}")
    for name, t in cases:
        fn = type(t.grad_fn).__name__ if t.grad_fn is not None else "None"
        label = name + " " * max(0, 20 - sum(2 if "가" <= ch <= "힣"
                                             else 1 for ch in name))
        print(f"  {label}{str(t.data_ptr() == a.data_ptr()):>6s}"
              f"{str(t.requires_grad):>16s}{fn:>18s}")

    # clone() 은 그래프를 남기므로 경사가 원본까지 거슬러 간다
    b = torch.tensor([1.0, 2.0], requires_grad=True)
    (b.clone() ** 2).sum().backward()
    print("  clone() 을 지나 온 경사  b.grad =", b.grad.tolist(), "(2b)")

    # detach() 는 끊으므로 뒤로 걸을 길이 없다
    c = torch.tensor([1.0, 2.0], requires_grad=True)
    try:
        (c.detach() ** 2).sum().backward()
    except RuntimeError as e:
        print("  detach() 를 지나 뒤로 걸으면:", str(e).splitlines()[0])


# ========================================================================
# 3. numpy 는 메모리를 나눠 쓴다
# ========================================================================

def numpy_shares():
    header("3. numpy 와 메모리를 나눠 쓴다")
    t = torch.tensor([1.0, 2.0, 3.0])
    arr = t.numpy()
    arr[0] = 99.0
    print("  arr[0]=99 로 고치니 텐서 =", t.tolist())
    t[1] = 77.0
    print("  텐서를 고치니 배열       =", arr.tolist())
    print("  양쪽이 같은 메모리를 가리킨다. 복사가 아니다")

    b = np.array([1.0, 2.0])
    tb = torch.from_numpy(b)
    b[0] = 5.0
    print("  from_numpy 뒤 배열을 고치니 텐서 =", tb.tolist())
    print("  그리고 dtype 이", tb.dtype, "다 — numpy 의 기본값이 float64 라서다")

    lin = nn.Linear(2, 1)
    try:
        lin(torch.from_numpy(np.array([[1.0, 2.0]])))
    except RuntimeError as e:
        print("  그대로 모델에 넣으면:", str(e).splitlines()[0])
    ok = lin(torch.from_numpy(np.array([[1.0, 2.0]], dtype=np.float32)))
    print("  np.float32 로 만들면 통과:", tuple(ok.shape))


# ========================================================================
# 4. 꺼낼 수 없는 자리
# ========================================================================

def cannot_extract():
    header("4. 바로 꺼낼 수 없는 두 자리")
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
            print(f"  device={dev:4s}       :", str(e).splitlines()[0])
        print("  .cpu().numpy() 로는 된다:", m.cpu().numpy().shape)
    else:
        print("  가속기가 없어 장치 쪽은 건너뛴다 (cpu 뿐)")
    print("  두 메시지 모두 고칠 방법을 함께 적어 준다")


# ========================================================================
# 5. .data — 검사만 지나간다
# ========================================================================

def data_attribute():
    header("5. .data 는 버전 카운터를 지나간다")
    # 파이토치는 텐서마다 _version 을 세어 둔다. 역전파가 저장해 둔 값이
    # 그 뒤에 고쳐졌으면 숫자가 어긋나므로, 엔진이 거절한다.
    for label, use_data in (("y.add_(1)", False), ("y.data.add_(1)", True)):
        x = torch.tensor([1.0, 2.0], requires_grad=True)
        y = x.exp()               # exp 의 역전파는 결과 y 를 저장해 둔다
        (y.data if use_data else y).add_(1.0)
        print(f"  {label:16s} _version={y._version}", end="  ")
        try:
            y.sum().backward()
            print("경사 =", [round(v, 4) for v in x.grad.tolist()])
        except RuntimeError as e:
            print("RuntimeError:", str(e).splitlines()[0][:62])
    print("  참값 d(sum exp x)/dx = exp(x) =",
          [round(v, 4) for v in torch.tensor([1.0, 2.0]).exp().tolist()])
    print("  .data 쪽은 1 씩 크다 — 고쳐진 y 를 그대로 읽었다")
    print("  막아 주던 검사를 끈 셈이라, 터지는 대신 틀린 수가 나온다")


def main():
    extracting()
    copies()
    numpy_shares()
    cannot_extract()
    data_attribute()


if __name__ == "__main__":
    main()
