"""복사 없이 늘리기 — expand 와 repeat, 그리고 메모리 형식.

이 절 첫 쪽 「메모리 배치와 스트라이드」에서 저장소·보폭·오프셋 모델과
뷰·복사의 갈림을 다루었다. 이 쪽은 그 모델이 가장 극단으로 쓰이는
자리를 본다 — 보폭 0.

- `expand` 는 보폭을 0 으로 두어 저장소를 한 줄도 늘리지 않는다
- `repeat` 는 정말로 베낀다
- 늘린 텐서에 쓰면 한 번 쓴 값이 여러 자리에 나타난다. 그런데 파이토치가
  막는 쓰기와 막지 않는 쓰기가 갈린다
- 모양이 같고 보폭만 다른 또 한 가지 — 메모리 형식(channels_last)

실행: python 04_memory_and_storage.py
"""
import time

import torch


def header(title):
    print(f"\n=== {title} ===")


def elems(t):
    """저장소에 실제로 들어 있는 원소 수. 모양이 아니라 쓰인 메모리다."""
    return t.untyped_storage().nbytes() // t.element_size()


# ========================================================================
# 1. 같은 값, 다른 저장소
# ========================================================================

def same_values():
    header("1. 같은 값인데 쓰인 메모리가 다르다")
    b = torch.tensor([1.0, 2.0, 3.0])
    e = b.expand(4, 3)
    r = b.repeat(4, 1)
    for name, t in (("b (원본)", b), ("b.expand(4,3)", e), ("b.repeat(4,1)", r)):
        pad = " " * max(0, 15 - sum(2 if "가" <= c <= "힣" else 1 for c in name))
        print(f"  {name}{pad}shape={tuple(t.shape)} stride={t.stride()} "
              f"저장소 원소={elems(t)}")
    print("  두 결과의 값이 같은가:", torch.equal(e, r))
    print("  expand 의 두 행이 같은 자리:", e[0].data_ptr() == e[1].data_ptr())
    print("  repeat 의 두 행이 같은 자리:", r[0].data_ptr() == r[1].data_ptr())
    print("  보폭 0 은 '이 축으로는 움직이지 말라' 는 뜻이다")


# ========================================================================
# 2. 값이 얼마나 되는가
# ========================================================================

def cost():
    header("2. 1000 줄로 늘리는 값")
    big = torch.randn(1000)
    # 시간과 메모리를 다른 줄에 찍는다. 한 줄에 같이 두면 시간이 흔들릴
    # 때마다 그 줄 전체가 못 믿을 줄이 되어, 메모리 수까지 확인할 수 없다.
    for name, fn in (("expand", lambda: big.expand(1000, 1000)),
                     ("repeat", lambda: big.repeat(1000, 1))):
        got = []
        for _ in range(5):          # 한 번 잰 시간은 측정이 아니다
            t0 = time.perf_counter()
            out = fn()
            got.append(time.perf_counter() - t0)
        print(f"  {name} 저장소 {elems(out)}개 = {elems(out) * 4 / 1e6:.3f} MB")
        print(f"  {name} 걸린 시간 {min(got) * 1e6:.1f} ~ "
              f"{max(got) * 1e6:.1f} us")
    print("  expand 는 모양만 적어 두므로 크기와 거의 무관하다")


# ========================================================================
# 3. 늘린 텐서에 쓰면 — 막는 쓰기와 막지 않는 쓰기
# ========================================================================

def writing():
    header("3. 늘린 텐서에 쓰기")
    b = torch.tensor([1.0, 2.0, 3.0])
    e = b.expand(4, 3)
    print("  쓰기 전 b =", b.tolist())
    e[0, 0] = 99.0                  # 통과한다
    print("  e[0,0] = 99 한 뒤")
    print("    b =", b.tolist(), "<- 원본까지 바뀌었다")
    for row in e.tolist():
        print("    e 행 =", row)

    e2 = torch.tensor([1.0, 2.0, 3.0]).expand(4, 3)
    try:
        e2.add_(1.0)
    except RuntimeError as ex:
        print("  e2.add_(1.0) :", str(ex).splitlines()[0][:78])
    print("  색인 대입은 지나가고 제자리 연산은 막힌다")
    safe = torch.tensor([1.0, 2.0, 3.0]).expand(4, 3).clone()
    safe[0, 0] = 99.0
    print("  clone() 뒤에는 한 줄만 바뀐다:", safe[:, 0].tolist())


# ========================================================================
# 4. 늘릴 수 있는 축은 길이 1 인 축뿐이다
# ========================================================================

def which_axes():
    header("4. 늘릴 수 있는 축")
    try:
        torch.ones(2, 3).expand(4, 3)
    except RuntimeError as ex:
        print("  (2,3) -> (4,3) :", str(ex).splitlines()[0][:84])
    print("  (1,3) -> (4,3) :", tuple(torch.ones(1, 3).expand(4, 3).shape))
    print("  (3,)  -> (4,3) :", tuple(torch.ones(3).expand(4, 3).shape),
          "— 모자라는 축은 앞에 붙여 준다")
    print("  repeat 는 아무 축이나 된다:",
          tuple(torch.ones(2, 3).repeat(2, 1).shape))


# ========================================================================
# 5. 모양이 같고 보폭만 다른 또 한 가지 — 메모리 형식
# ========================================================================

def memory_format():
    header("5. 메모리 형식 (channels_last)")
    img = torch.randn(8, 3, 32, 32)
    cl = img.contiguous(memory_format=torch.channels_last)
    print("  모양          :", tuple(img.shape), "둘이 같다:", tuple(cl.shape))
    print("  값이 같은가    :", torch.equal(img, cl))
    print("  NCHW 보폭      :", img.stride())
    print("  channels_last :", cl.stride())
    print("  img.is_contiguous()                           :", img.is_contiguous())
    print("  cl.is_contiguous()                            :", cl.is_contiguous())
    print("  cl.is_contiguous(memory_format=channels_last) :",
          cl.is_contiguous(memory_format=torch.channels_last))
    print("  '연속' 은 한 가지가 아니라 어느 순서로 연속인지의 문제다")


def main():
    same_values()
    cost()
    writing()
    which_axes()
    memory_format()


if __name__ == "__main__":
    main()
