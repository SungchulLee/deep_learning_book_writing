# 제자리 연산

제자리 연산은 텐서가 가리키는 데이터를 직접 바꾼다. 새 결과를 만드는 연산과 달리 뷰·별칭·역전파에 필요한 값까지 영향을 받을 수 있다.

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
    
    # 해결 2: 파라메터 갱신에 torch.no_grad()를 쓴다
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
    main()
```

??? note "전체 출력 (93줄)"

    ```

    ================================================================================
    Basic in-place operations
    ================================================================================
    Original x: tensor([1., 2., 3., 4., 5.])
    After x.add_(10): tensor([11., 12., 13., 14., 15.])
    After x.mul_(2): tensor([22., 24., 26., 28., 30.])
    After x.clamp_(0, 30): tensor([22., 24., 26., 28., 30.])

    ================================================================================
    Memory sharing with in-place ops
    ================================================================================
    a and b share memory: True
    b also changed: tensor([ 2.5410,  0.7066, -1.1788])

    ================================================================================
    Out-of-place vs in-place comparison
    ================================================================================
    Out-of-place: 0.0739s
    In-place:     0.0211s
    Speedup:      3.51x

    ================================================================================
    Autograd restriction: in-place on leaf tensors with requires_grad
    ================================================================================
    ERROR (expected): a leaf Variable that requires grad is being used in an in-place operation....
    Out-of-place works: tensor([2., 3., 4.], grad_fn=<AddBackward0>)
    In-place with no_grad: tensor([2., 3., 4.], requires_grad=True)

    ================================================================================
    In-place ops on non-leaf tensors (intermediate results)
    ================================================================================
    In-place on non-leaf can break autograd graph

    ================================================================================
    Common in-place operations showcase
    ================================================================================
    Original t: tensor([-0.5707, -2.4878,  0.7807, -0.3018,  0.1119])
    abs_(): tensor([0.5707, 2.4878, 0.7807, 0.3018, 0.1119])
    neg_(): tensor([ 0.5707,  2.4878, -0.7807,  0.3018, -0.1119])
    sqrt_().abs_(): tensor([   nan,    nan, 0.8836,    nan, 0.3346])
    pow_(2): tensor([0.3257, 6.1892, 0.6095, 0.0911, 0.0125])
    uniform_(-1, 1): tensor([ 0.8857,  0.0620, -0.2872,  0.9226, -0.6273])
    normal_(0, 1): tensor([-1.2833,  0.5187,  0.4093,  1.3558,  0.4099])

    ================================================================================
    fill_ and zero_ operations
    ================================================================================
    Before fill_:
     tensor([[-1.4160,  0.5773,  2.2979],
            [-0.2392, -0.0332,  0.1740],
            [-0.5576, -0.3373,  1.0424]])
    After fill_(7.0):
     tensor([[7., 7., 7.],
            [7., 7., 7.],
            [7., 7., 7.]])
    After zero_():
     tensor([[0., 0., 0.],
            [0., 0., 0.],
            [0., 0., 0.]])

    ================================================================================
    Indexed in-place assignment
    ================================================================================
    After indexed assignment: tensor([ 0., 10., 20., 30.,  0.])
    After boolean mask assignment: tensor([ 0., 10., -1., -1.,  0.])

    ================================================================================
    copy_ for in-place copying
    ================================================================================
    dst before copy_:
     tensor([[0., 0., 0.],
            [0., 0., 0.],
            [0., 0., 0.]])
    dst after copy_(src):
     tensor([[ 1.3839,  1.5731, -1.4241],
            [-0.1446, -0.1387, -0.9443],
            [-0.6245, -0.2811,  1.2099]])

    ================================================================================
    Best practices: when to use in-place
    ================================================================================
    ✓ Use in-place ops for:
      - Parameter updates inside torch.no_grad()
      - Memory-critical situations
      - Explicit tensor initialization (fill_, zero_, normal_)
      - When you KNOW autograd won't be needed

    ✗ Avoid in-place ops for:
      - Leaf tensors with requires_grad=True
      - Intermediate computation results in autograd
      - When code clarity is more important than speed
      - When tensors might be aliased unexpectedly
    ```


## 2. 논의

`add_`처럼 밑줄로 끝나는 메서드와 인덱싱 대입은 제자리 수정이다. 같은 저장소의 수정 위치를 참조하는 뷰도 영향을 받는다. 단순히 메모리를 아끼는 문법으로만 해석하지 않는다.

그래디언트를 요구하는 잎 텐서의 직접 수정과, 역전파에 저장된 값의 수정은 오류가 날 수 있다. `no_grad`는 갱신 기록을 끄지만 이미 저장된 값을 바꿔도 안전하다고 보장하지는 않는다. 일부 제자리 연산은 허용되며 그 경우에도 미분하려는 함수가 달라질 수 있다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
`torch.eye`로 $4 \times 4$ 단위 행렬을 만들고, `torch.diag`로 대각 성분이 $[1, 2, 3, 4]$인 대각 행렬을 만들어라. 둘이 다름을 확인하라.

</div>

??? success "연습문제 1 풀이"
    ```python
    I = torch.eye(4)
    D = torch.diag(torch.tensor([1., 2., 3., 4.]))
    print(torch.equal(I, D))  # False -- D has non-unit diagonal
    ```

---


<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
`torch.empty()`와 `torch.zeros()`의 차이를 설명하라. `empty()`는 언제 쓰는 것이 적절한가?

</div>

??? success "연습문제 2 풀이"
    `torch.zeros()`는 0으로 초기화한다. `torch.empty()`의 값은 일반적으로 초기화되지 않으므로 읽기 전에 사용할 모든 원소를 채워야 한다. 초기화 비용을 줄일 수 있지만 실제 성능 차이는 작업에 따라 다르며, 결정론적 실행의 설정에 따라 미정의 값 사용을 막기 위한 채움이 수행될 수도 있다.

---


<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
`torch.linspace`로 $-\pi$와 $\pi$ 사이에 균등 간격의 점 100개를 만들어라. 각 점에서 $\sin(x)$를 계산하고 절댓값의 최댓값이 대략 1임을 확인하라.

</div>

??? success "연습문제 3 풀이"
    ```python
    import torch, math
    x = torch.linspace(-math.pi, math.pi, 100)
    y = torch.sin(x)
    print(y.abs().max())  # Approximately 1.0
    ```

## 정리하며

제자리 연산은 텐서가 가리키는 데이터를 직접 바꾼다. 새 결과를 만드는 연산과 달리 뷰·별칭·역전파에 필요한 값까지 영향을 받을 수 있다.
