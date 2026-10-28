"""이어 붙이기와 나누기 — 합치면 베끼고 나누면 뷰다.

이 절은 그 대칭에서 시작한다.

- `cat` · `stack` 은 결과를 담을 자리를 새로 잡고 **베낀다**
- `split` · `chunk` · `unbind` 는 **뷰**를 돌려준다. 고치면 원본이 바뀐다

나누는 쪽에서 조심할 것이 하나 더 있다. **`chunk(k)` 가 언제나 k 덩이를
주지는 않는다.** 요청보다 적게 나오는 자리가 있다.

실행: python 13_concatenation_splitting.py
"""
import torch


def header(title):
    print(f"\n=== {title} ===")


def shares(a, b):
    return a.untyped_storage().data_ptr() == b.untyped_storage().data_ptr()


# ========================================================================
# 1. 합치기 — cat 과 stack
# ========================================================================

def joining():
    header("1. cat 과 stack")
    a = torch.tensor([[1, 2], [3, 4]])
    b = torch.tensor([[5, 6], [7, 8]])
    print("  cat(dim=0)  :", tuple(torch.cat([a, b], dim=0).shape), "축 2개 그대로")
    print("  cat(dim=1)  :", tuple(torch.cat([a, b], dim=1).shape))
    print("  stack(dim=0):", tuple(torch.stack([a, b], dim=0).shape), "축 3개로")
    print("  stack(dim=1):", tuple(torch.stack([a, b], dim=1).shape))
    print("  cat 결과가 원본과 저장소를 나눠 쓰는가:",
          shares(a, torch.cat([a, b], dim=0)))
    print("  합치는 쪽은 언제나 새 자리를 잡는다")

    c = torch.tensor([[9, 10]])
    print("  셋 이상도 한 번에:", tuple(torch.cat([a, b, c], dim=0).shape))


# ========================================================================
# 2. hstack / vstack 은 차원에 따라 다르다
# ========================================================================

def stack_helpers():
    header("2. hstack / vstack")
    u, v = torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6])
    print("  1차원 hstack:", torch.hstack([u, v]).tolist(), " <- dim=0 처럼")
    print("  1차원 vstack:", torch.vstack([u, v]).tolist(), " <- 행으로 쌓는다")
    m = torch.ones(2, 2, dtype=torch.long)
    n = torch.zeros(2, 2, dtype=torch.long)
    print("  2차원 hstack 모양:", tuple(torch.hstack([m, n]).shape), " <- dim=1")
    print("  2차원 vstack 모양:", tuple(torch.vstack([m, n]).shape), " <- dim=0")
    print("  hstack 은 1차원에서 dim=0, 2차원에서 dim=1 이다")
    print("  축을 손으로 적는 cat/stack 이 헷갈릴 일이 없다")


# ========================================================================
# 3. 나누기 — split 은 크기, chunk 는 개수
# ========================================================================

def splitting():
    header("3. split 과 chunk")
    x = torch.arange(10)
    print("  split(3)       크기:", [len(c) for c in x.split(3)],
          " <- 3씩 자르고 남는 건 남는 대로")
    print("  split([2,3,5]) 크기:", [len(c) for c in x.split([2, 3, 5])])
    try:
        x.split([2, 3, 4])
    except RuntimeError as e:
        print("  split([2,3,4]) :", str(e).splitlines()[0][:70])
    print("  split 에 리스트를 주면 합이 딱 맞아야 한다")


def chunk_surprise():
    header("4. chunk(k) 가 k 덩이를 주지 않을 때")
    for n, k in ((10, 4), (5, 3), (4, 3), (7, 3), (9, 4), (12, 5)):
        cs = torch.arange(n).chunk(k)
        flag = "" if len(cs) == k else "   <- 요청은 %d" % k
        print(f"  arange({n:2d}).chunk({k}) -> {len(cs)}덩이  "
              f"크기 {[len(c) for c in cs]}{flag}")
    print("  chunk 는 먼저 덩이 크기를 ceil(n/k) 로 정하고 그 크기로 자른다")
    print("  9 를 4 로 나누면 크기 3 이 되고, 3 씩 자르면 세 덩이면 끝난다")


# ========================================================================
# 5. 나눈 조각은 뷰다
# ========================================================================

def pieces_are_views():
    header("5. 나눈 조각은 뷰다")
    a = torch.arange(12).reshape(3, 4)
    for name, parts in (("chunk(3)", a.chunk(3)),
                        ("split(2)", a.split(2)),
                        ("unbind(0)", a.unbind(0))):
        print(f"  {name:10s} 첫 조각이 뷰인가: {shares(a, parts[0])}")
    first = a.chunk(3)[0]
    first[0, 0] = 99
    print("  조각을 99 로 고치니 원본 a[0,0] =", a[0, 0].item())
    print("  나누는 쪽은 베끼지 않는다. 읽기만 할 때는 공짜다")
    print("  고칠 작정이면 조각마다 clone() 을 해야 한다")


# ========================================================================
# 6. 되돌리기
# ========================================================================

def round_trip():
    header("6. 되돌리기")
    x = torch.arange(10)
    print("  split 하고 cat 하면 돌아오는가:",
          torch.equal(x, torch.cat(x.split(3))))
    y = torch.randn(3, 4)
    print("  unbind 하고 stack 하면 돌아오는가:",
          torch.equal(y, torch.stack(y.unbind(0))))
    print("  unbind(0) 은 for 문으로 도는 것과 같다:",
          all(torch.equal(p, q) for p, q in zip(y.unbind(0), y)))
    print("  다만 돌아온 것은 새 텐서다:",
          shares(y, torch.stack(y.unbind(0))))


def main():
    joining()
    stack_helpers()
    splitting()
    chunk_surprise()
    pieces_are_views()
    round_trip()


if __name__ == "__main__":
    main()
