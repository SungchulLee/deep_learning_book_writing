# 메모리와 저장소

이 스크립트는 텐서의 메모리와 저장소을 보여준다. 이 개념들을 이해하는 것은 효과적인 PyTorch 프로그래밍과 딥러닝 모델 개발에 필수적이다.

## 코드

```python
#!/usr/bin/env python3
"""
텐서의 기억 자리 구조와 보기/베끼기 움직임.

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
    print("view shares storage:", id(t.storage()) == id(v.storage()))
    print("clone shares storage:", id(t.storage()) == id(c.storage()))
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
    main()```

## 논의

재구성 연산은 데이터를 반드시 복사하지 않으면서 텐서의 논리적 배치를 바꾼다. `view()` 메서드는 연속된 메모리를 요구하며 항상 뷰를 반환하고, `reshape()`는 (필요하면 복사하여) 어떤 텐서에서도 동작한다. `transpose()`나 `permute()` 같은 연산은 데이터 배치가 아니라 스트라이드를 바꾸므로 결과가 연속적이지 않을 수 있다.

PyTorch는 (저장소를 공유하는 뷰를 반환하는) 기본 슬라이싱과 (복사본을 반환하는) 불리언 마스크나 정수 배열을 이용한 고급 인덱싱을 모두 지원한다. 이 구분을 이해하는 것은 메모리 효율을 위해서도, 인덱싱한 결과를 수정할 때 의도치 않은 부작용을 피하기 위해서도 중요하다.

텐서 생성 함수는 데이터를 초기화하는 유연한 방법을 제공한다. `torch.zeros`, `torch.randn`, `torch.arange` 같은 팩토리 함수는 `dtype`, `device`, `requires_grad` 매개변수를 받으므로 불필요한 복사 없이 목표 장치에 곧바로 할당할 수 있다.

## 연습문제

**연습문제 1.**
모양이 $(24,)$인 텐서를 만들어 $(2, 3, 4)$로 재구성하라. 그런 다음 차원을 $(4, 2, 3)$으로 바꾸고 전체 원소 개수가 변하지 않았음을 확인하라.

??? success "연습문제 1 풀이"
    ```python
    x = torch.arange(24)
    x_3d = x.reshape(2, 3, 4)
    x_perm = x_3d.permute(2, 0, 1)
    assert x_perm.shape == (4, 2, 3)
    assert x_perm.numel() == 24
    ```

---


**연습문제 2.**
`transpose()` 후에 `view()`는 실패하지만 `reshape()`는 성공하는 이유를 설명하라. `contiguous()`는 무엇을 하는가?

??? success "연습문제 2 풀이"
    `transpose()`는 스트라이드는 다르지만 바탕 저장소는 같은 뷰를 반환한다. 데이터가 더 이상 행 우선 순서가 아니므로 (연속된 메모리를 요구하는) `view()`가 실패한다. `reshape()`는 비연속성을 감지하여 복사본을 만든다. `.contiguous()`를 호출하면 행 우선 배치를 가진 새 텐서가 명시적으로 만들어지며, 그 뒤에는 `view()`가 동작한다.

---


**연습문제 3.**
NCHW 형식의 모양 $(32, 3, 224, 224)$인 이미지 배치가 주어졌을 때 `permute()`로 NHWC 형식으로 바꾸고, 공간 차원을 펼쳐 모양 $(32, 3, 50176)$을 얻어라.

??? success "연습문제 3 풀이"
    ```python
    images = torch.randn(32, 3, 224, 224)
    nhwc = images.permute(0, 2, 3, 1)  # (32, 224, 224, 3)
    # (32, 3, 50176)을 얻으려면 원본에서 H와 W를 펼친다:
    flat = images.flatten(start_dim=2)  # (32, 3, 50176)
    print(flat.shape)
    ```
