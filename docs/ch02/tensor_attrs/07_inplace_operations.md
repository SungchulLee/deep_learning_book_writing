# 제자리 연산

이 스크립트는 텐서의 제자리 연산을 보여준다. 이 개념들을 이해하는 것은 효과적인 PyTorch 프로그래밍과 딥러닝 모델 개발에 필수적이다.

## 1. 코드

```python
#!/usr/bin/env python3
"""
제자리 셈: 잘 듦과 함정.

Covers:
- 이름 관례: 밑줄(_)로 끝나는 셈
- 성능의 이점과 기억 자리를 나눠 쓰는 데 따른 결과
- 제자리 셈에 대한 자동 미분의 매임
- 흔한 제자리 셈: add_, mul_, clamp_ 따위
- 제자리 셈을 언제 쓰고 언제 피할까
"""

import torch
import torch.nn as nn

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
    header("Basic in-place operations")
    x = torch.tensor([1., 2., 3., 4., 5.])
    print("Original x:", x)
    
    x.add_(10)  # x = x + 10, in-place
    print("After x.add_(10):", x)
    
    x.mul_(2)   # x = x * 2, in-place
    print("After x.mul_(2):", x)
    
    x.clamp_(0, 30)  # Clamp values in-place
    print("After x.clamp_(0, 30):", x)

    # -------------------------------------------------------------------------
    header("Memory sharing with in-place ops")
    a = torch.randn(3, 4)
    b = a  # b is just another reference to the same tensor
    a_id = id(a)
    
    a.add_(1)  # Modifies the underlying data
    print("a and b share memory:", id(a) == id(b) == a_id)
    print("b also changed:", b[0, :3])

    # -------------------------------------------------------------------------
    header("Out-of-place vs in-place comparison")
    import time
    
    # 시간 측정을 위한 큰 텐서
    big = torch.randn(1000, 1000)
    
    # 제자리가 아님(새 텐서를 만든다)
    start = time.time()
    for _ in range(100):
        result = big + 1.0
    out_of_place_time = time.time() - start
    
    # 제자리(기존 텐서를 수정한다)
    start = time.time()
    for _ in range(100):
        big.add_(1.0)
    in_place_time = time.time() - start
    
    print(f"Out-of-place: {out_of_place_time:.4f}s")
    print(f"In-place:     {in_place_time:.4f}s")
    print(f"Speedup:      {out_of_place_time/in_place_time:.2f}x")

    # -------------------------------------------------------------------------
    header("Autograd restriction: in-place on leaf tensors with requires_grad")
    leaf = torch.tensor([1., 2., 3.], requires_grad=True)
    
    try:
        leaf.add_(1)  # This will fail!
    except RuntimeError as e:
        print("ERROR (expected):", str(e)[:80] + "...")
    
    # 해결 1: 제자리가 아닌 연산을 쓴다
    leaf2 = torch.tensor([1., 2., 3.], requires_grad=True)
    result = leaf2 + 1  # This works fine
    print("Out-of-place works:", result)
    
    # 해결 2: 매개변수 갱신에 torch.no_grad()를 쓴다
    leaf3 = torch.tensor([1., 2., 3.], requires_grad=True)
    with torch.no_grad():
        leaf3.add_(1)  # OK inside no_grad
    print("In-place with no_grad:", leaf3)

    # -------------------------------------------------------------------------
    header("In-place ops on non-leaf tensors (intermediate results)")
    x = torch.tensor([1., 2., 3.], requires_grad=True)
    y = x * 2  # Non-leaf tensor
    
    try:
        y.add_(1)  # Also problematic for autograd
    except RuntimeError as e:
        print("ERROR (expected):", str(e)[:80] + "...")
    print("In-place on non-leaf can break autograd graph")

    # -------------------------------------------------------------------------
    header("Common in-place operations showcase")
    t = torch.randn(5)
    print("Original t:", t)
    
    t_copy = t.clone()
    t_copy.abs_()
    print("abs_():", t_copy)
    
    t_copy = t.clone()
    t_copy.neg_()
    print("neg_():", t_copy)
    
    t_copy = t.clone()
    t_copy.sqrt_().abs_()  # Chain in-place ops
    print("sqrt_().abs_():", t_copy)
    
    t_copy = t.clone()
    t_copy.pow_(2)
    print("pow_(2):", t_copy)
    
    t_copy = torch.randn(5)
    t_copy.uniform_(-1, 1)  # Fill with uniform random
    print("uniform_(-1, 1):", t_copy)
    
    t_copy = torch.zeros(5)
    t_copy.normal_(mean=0, std=1)  # Fill with normal random
    print("normal_(0, 1):", t_copy)

    # -------------------------------------------------------------------------
    header("fill_ and zero_ operations")
    m = torch.randn(3, 3)
    print("Before fill_:\n", m)
    
    m.fill_(7.0)
    print("After fill_(7.0):\n", m)
    
    m.zero_()
    print("After zero_():\n", m)

    # -------------------------------------------------------------------------
    header("Indexed in-place assignment")
    arr = torch.zeros(5)
    arr[1:4] = torch.tensor([10., 20., 30.])
    print("After indexed assignment:", arr)
    
    arr[arr > 15] = -1  # Boolean masking with assignment
    print("After boolean mask assignment:", arr)

    # -------------------------------------------------------------------------
    header("copy_ for in-place copying")
    src = torch.randn(3, 3)
    dst = torch.zeros(3, 3)
    print("dst before copy_:\n", dst)
    
    dst.copy_(src)  # Copy src data into dst
    print("dst after copy_(src):\n", dst)

    # -------------------------------------------------------------------------
    header("Best practices: when to use in-place")
    print("✓ Use in-place ops for:")
    print("  - Parameter updates inside torch.no_grad()")
    print("  - Memory-critical situations")
    print("  - Explicit tensor initialization (fill_, zero_, normal_)")
    print("  - When you KNOW autograd won't be needed")
    print("\n✗ Avoid in-place ops for:")
    print("  - Leaf tensors with requires_grad=True")
    print("  - Intermediate computation results in autograd")
    print("  - When code clarity is more important than speed")
    print("  - When tensors might be aliased unexpectedly")

if __name__ == "__main__":
    main()```

## 2. 논의

경사 추적을 제어하는 것은 정확성과 성능 모두에 필수적이다. `torch.no_grad()` 컨텍스트 관리자는 매개변수 갱신이나 추론처럼 계산 그래프에 포함되어서는 안 되는 연산에 대해 autograd를 끈다. `.detach()` 메서드는 저장소는 공유하지만 그래프와는 분리된 텐서를 만들며, 값을 기록하거나 NumPy로 변환할 때 유용하다.

PyTorch는 (저장소를 공유하는 뷰를 반환하는) 기본 슬라이싱과 (복사본을 반환하는) 불리언 마스크나 정수 배열을 이용한 고급 인덱싱을 모두 지원한다. 이 구분을 이해하는 것은 메모리 효율을 위해서도, 인덱싱한 결과를 수정할 때 의도치 않은 부작용을 피하기 위해서도 중요하다.

텐서 생성 함수는 데이터를 초기화하는 유연한 방법을 제공한다. `torch.zeros`, `torch.randn`, `torch.arange` 같은 팩토리 함수는 `dtype`, `device`, `requires_grad` 매개변수를 받으므로 불필요한 복사 없이 목표 장치에 곧바로 할당할 수 있다.

## 연습문제

**연습문제 1.**
`torch.eye`로 $4 \times 4$ 단위 행렬을 만들고, `torch.diag`로 대각 성분이 $[1, 2, 3, 4]$인 대각 행렬을 만들어라. 둘이 다름을 확인하라.

??? success "연습문제 1 풀이"
    ```python
    I = torch.eye(4)
    D = torch.diag(torch.tensor([1., 2., 3., 4.]))
    print(torch.equal(I, D))  # False -- D has non-unit diagonal
    ```

---


**연습문제 2.**
`torch.empty()`와 `torch.zeros()`의 차이를 설명하라. `empty()`는 언제 쓰는 것이 적절한가?

??? success "연습문제 2 풀이"
    `torch.zeros()`는 모든 원소를 0으로 초기화하지만 `torch.empty()`는 초기화 없이 메모리만 할당한다(값은 그 메모리에 이전에 있던 것이 무엇이든 그대로이다). 초기화되지 않은 내용은 예측할 수 없으므로 `empty()`는 모든 값을 곧바로 덮어쓸 계획일 때만 써야 한다. 0으로 채우는 단계를 건너뛰므로 조금 더 빠르다.

---


**연습문제 3.**
`torch.linspace`로 $-\pi$와 $\pi$ 사이에 균등 간격의 점 100개를 만들어라. 각 점에서 $\sin(x)$를 계산하고 절댓값의 최댓값이 대략 1임을 확인하라.

??? success "연습문제 3 풀이"
    ```python
    import torch, math
    x = torch.linspace(-math.pi, math.pi, 100)
    y = torch.sin(x)
    print(y.abs().max())  # Approximately 1.0
    ```

## 정리하며

**다룬 것** — 제자리 연산

경사 추적을 제어하는 것은 정확성과 성능 모두에 필수적이다.

앞의 연습문제 3개로 직접 확인할 수 있다.
