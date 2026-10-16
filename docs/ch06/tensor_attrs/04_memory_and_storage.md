# 메모리와 저장소

텐서는 숫자들의 저장소와 그 저장소를 읽는 모양·스트라이드·오프셋으로 구성된다. 뷰를 바꿀 때와 데이터를 복사할 때를 구별하면 메모리 공유를 이해할 수 있다.

## 1. 코드

```python
#!/usr/bin/env python3
"""
텐서의 기억 자리 구조와 보기/베끼기 액션.

Covers:
- transpose()/permute()의 보기, 걸음 바뀜, .contiguous()
- view()와 reshape()과 clone() 견주기
- expand()과 repeat() 견주기
- 간단한 gather()/scatter_() 맛보기(자리 번호로 옮기기)
"""

import torch

# ========================================================================
# 메인
# ========================================================================

def header(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

def main():
    torch.manual_seed(0)

    # -------------------------------------------------------------------------
    header("transpose()/permute() return VIEWS (stride changes)")
    t = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    T = t.transpose(0, 1)     # (4, 3), view
    P = t.permute(1, 0)       # (4, 3), view
    print("t:\n", t)
    print("T = t.transpose(0,1):\n", T, "| is_contiguous:", T.is_contiguous(), "| stride:", T.stride())
    print("P = t.permute(1,0):\n", P, "| is_contiguous:", P.is_contiguous(), "| stride:", P.stride())

    # -------------------------------------------------------------------------
    header("Make contiguous copy: .contiguous()")
    Tc = T.contiguous()
    print("Tc.is_contiguous:", Tc.is_contiguous(), "| Tc.stride:", Tc.stride())

    # -------------------------------------------------------------------------
    header("view() vs reshape() vs clone()")
    v = t.view(12)            # view (when possible) → shares storage
    r = t.reshape(6, 2)       # may return view or copy
    c = t.clone()             # always copy
    print("view shares storage:", t.untyped_storage().data_ptr() == v.untyped_storage().data_ptr())
    print("clone shares storage:", t.untyped_storage().data_ptr() == c.untyped_storage().data_ptr())
    print("Before in-place, v[:5]:", v[:5])
    t[0, 0] = -999
    print("After  in-place, v[:5]:", v[:5])
    print("Clone first row (unchanged):", c[0])

    # -------------------------------------------------------------------------
    header("expand() vs repeat()")
    b = torch.tensor([1., 2., 3.])   # (3,)
    print("b.expand(2,3):\n", b.expand(2, 3))  # view (no alloc): stride tricks
    print("b.repeat(2,1):\n", b.repeat(2, 1))  # real data replication

    # -------------------------------------------------------------------------
    header("gather() / scatter_() mini-demo")
    A = torch.arange(1, 13).reshape(3, 4)  # [[1..4],[5..8],[9..12]]
    idx = torch.tensor([[0, 2], [1, 3], [0, 0]])  # per-row indices
    picked = A.gather(dim=1, index=idx)
    tgt = torch.zeros_like(A)
    tgt.scatter_(dim=1, index=idx, src=torch.tensor([[9, 9], [8, 8], [7, 7]]))
    print("A:\n", A)
    print("gather(dim=1, idx):\n", picked)
    print("scatter_ into zeros:\n", tgt)

    header("Done")

if __name__ == "__main__":
    main()
```

**출력:**

```

================================================================================
transpose()/permute() return VIEWS (stride changes)
================================================================================
t:
 tensor([[ 0.,  1.,  2.,  3.],
        [ 4.,  5.,  6.,  7.],
        [ 8.,  9., 10., 11.]])
T = t.transpose(0,1):
 tensor([[ 0.,  4.,  8.],
        [ 1.,  5.,  9.],
        [ 2.,  6., 10.],
        [ 3.,  7., 11.]]) | is_contiguous: False | stride: (1, 4)
P = t.permute(1,0):
 tensor([[ 0.,  4.,  8.],
        [ 1.,  5.,  9.],
        [ 2.,  6., 10.],
        [ 3.,  7., 11.]]) | is_contiguous: False | stride: (1, 4)

================================================================================
Make contiguous copy: .contiguous()
================================================================================
Tc.is_contiguous: True | Tc.stride: (3, 1)

================================================================================
view() vs reshape() vs clone()
================================================================================
view shares storage: True
clone shares storage: False
Before in-place, v[:5]: tensor([0., 1., 2., 3., 4.])
After  in-place, v[:5]: tensor([-999.,    1.,    2.,    3.,    4.])
Clone first row (unchanged): tensor([0., 1., 2., 3.])

================================================================================
expand() vs repeat()
================================================================================
b.expand(2,3):
 tensor([[1., 2., 3.],
        [1., 2., 3.]])
b.repeat(2,1):
 tensor([[1., 2., 3.],
        [1., 2., 3.]])

================================================================================
gather() / scatter_() mini-demo
================================================================================
A:
 tensor([[ 1,  2,  3,  4],
        [ 5,  6,  7,  8],
        [ 9, 10, 11, 12]])
gather(dim=1, idx):
 tensor([[1, 3],
        [6, 8],
        [9, 9]])
scatter_ into zeros:
 tensor([[9, 0, 9, 0],
        [0, 8, 0, 8],
        [7, 0, 0, 0]])

================================================================================
Done
================================================================================
```

## 2. 논의

기본 슬라이스와 전치는 저장소를 공유하는 뷰를 만든다. 같은 저장소여도 첫 원소 위치와 스트라이드가 달라 `data_ptr()`가 다를 수 있다. 공유 여부를 볼 때는 저장소 포인터와 `storage_offset()`도 함께 확인한다.

`view`는 요청한 모양을 현재 스트라이드로 표현할 수 있을 때만 성공한다. 비연속 입력도 가능한 뷰가 있으며, `reshape`는 필요하면 복사한다. `contiguous()`는 요청한 메모리 형식으로 이미 연속이면 그대로 반환하고 아니면 복사한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
모양이 $(24,)$인 텐서를 만들어 $(2, 3, 4)$로 재구성하라. 그런 다음 차원을 $(4, 2, 3)$으로 바꾸고 전체 원소 개수가 변하지 않았음을 확인하라.

</div>

??? success "연습문제 1 풀이"
    ```python
    x = torch.arange(24)
    x_3d = x.reshape(2, 3, 4)
    x_perm = x_3d.permute(2, 0, 1)
    assert x_perm.shape == (4, 2, 3)
    assert x_perm.numel() == 24
    ```

---


<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
`transpose()` 후에 `view()`는 실패하지만 `reshape()`는 성공하는 이유를 설명하라. `contiguous()`는 무엇을 하는가?

</div>

??? success "연습문제 2 풀이"
    전치는 저장소를 공유하면서 축과 스트라이드를 바꾼다. 전치 후에도 요청한 모양이 스트라이드와 호환되면 `view`가 가능하지만, 예시의 평탄화는 불가능하다. `reshape`는 필요한 경우 복사하며 `contiguous()`는 요청한 메모리 형식으로 연속된 결과를 준다.

---


<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
NCHW 형식의 모양 $(32, 3, 224, 224)$인 이미지 배치가 주어졌을 때 `permute()`로 NHWC 형식으로 바꾸고, 공간 차원을 펼쳐 모양 $(32, 3, 50176)$을 얻어라.

</div>

??? success "연습문제 3 풀이"
    ```python
    images = torch.randn(32, 3, 224, 224)
    nhwc = images.permute(0, 2, 3, 1)  # (32, 224, 224, 3)
    # (32, 3, 50176)을 얻으려면 원본에서 H와 W를 펼친다:
    flat = images.flatten(start_dim=2)  # (32, 3, 50176)
    print(flat.shape)
    ```

## 정리하며

텐서는 숫자들의 저장소와 그 저장소를 읽는 모양·스트라이드·오프셋으로 구성된다. 뷰를 바꿀 때와 데이터를 복사할 때를 구별하면 메모리 공유를 이해할 수 있다.
