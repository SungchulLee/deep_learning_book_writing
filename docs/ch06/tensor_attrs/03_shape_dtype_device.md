# 형상, 자료형, 장치

같은 숫자가 들어 있어도 모양·자료형·장치가 다르면 허용되는 연산이 달라진다. 계산 전에 이 세 속성을 확인하는 습관을 익힌다.

## 1. 코드

```python
#!/usr/bin/env python3
"""
꼴 / 데이터 클래스 / 기기 / 자리 구조 속성과 도우미.

Covers:
- shape / size() / ndim
- dtype / device / requires_grad
- layout
- is_cuda / is_contiguous() / stride() / storage_offset()
- 뒤집기 도우미: T / mT / H
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
    header("Base tensor setup")
    x = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    print("x:\n", x)

    # -------------------------------------------------------------------------
    header("Shape / ndim")
    print("x.shape     :", x.shape)
    print("x.size()    :", x.size())
    print("x.ndim      :", x.ndim)

    # -------------------------------------------------------------------------
    header("dtype / device / requires_grad / layout")
    print("x.dtype       :", x.dtype)
    print("x.device      :", x.device)
    print("x.requires_grad:", x.requires_grad)
    print("x.layout      :", x.layout)

    # -------------------------------------------------------------------------
    header("Memory layout attributes")
    print("x.is_cuda       :", x.is_cuda)
    print("x.is_contiguous :", x.is_contiguous())
    print("x.stride()      :", x.stride())
    print("x.storage_offset:", x.storage_offset())

    # -------------------------------------------------------------------------
    header("Transpose helpers (T / mT / H)")
    print("x.T  (simple transpose):\n", x.T)
    print("x.mT (matrix transpose):\n", x.mT)
    print("x.H  (Hermitian transpose):\n", x.H)

    # -------------------------------------------------------------------------
    header("Autograd quick peek")
    y = x.clone().requires_grad_(True)
    z = (y * y).sum()
    print("y.is_leaf :", y.is_leaf)  # True
    print("z.grad_fn :", z.grad_fn)  # e.g., <SumBackward0>
    z.backward()
    print("y.grad:\n", y.grad)

if __name__ == "__main__":
    main()
```

**출력:**

```

================================================================================
Base tensor setup
================================================================================
x:
 tensor([[ 0.,  1.,  2.,  3.],
        [ 4.,  5.,  6.,  7.],
        [ 8.,  9., 10., 11.]])

================================================================================
Shape / ndim
================================================================================
x.shape     : torch.Size([3, 4])
x.size()    : torch.Size([3, 4])
x.ndim      : 2

================================================================================
dtype / device / requires_grad / layout
================================================================================
x.dtype       : torch.float32
x.device      : cpu
x.requires_grad: False
x.layout      : torch.strided

================================================================================
Memory layout attributes
================================================================================
x.is_cuda       : False
x.is_contiguous : True
x.stride()      : (4, 1)
x.storage_offset: 0

================================================================================
Transpose helpers (T / mT / H)
================================================================================
x.T  (simple transpose):
 tensor([[ 0.,  4.,  8.],
        [ 1.,  5.,  9.],
        [ 2.,  6., 10.],
        [ 3.,  7., 11.]])
x.mT (matrix transpose):
 tensor([[ 0.,  4.,  8.],
        [ 1.,  5.,  9.],
        [ 2.,  6., 10.],
        [ 3.,  7., 11.]])
x.H  (Hermitian transpose):
 tensor([[ 0.,  4.,  8.],
        [ 1.,  5.,  9.],
        [ 2.,  6., 10.],
        [ 3.,  7., 11.]])

================================================================================
Autograd quick peek
================================================================================
y.is_leaf : True
z.grad_fn : <SumBackward0 object at 0x112687d90>
y.grad:
 tensor([[ 0.,  2.,  4.,  6.],
        [ 8., 10., 12., 14.],
        [16., 18., 20., 22.]])
```

## 2. 논의

`shape`는 각 축의 길이, `dtype`은 원소의 표현 방식, `device`는 계산과 저장이 이루어지는 장치를 나타낸다. 원소 개수가 같다고 모양이 같지는 않으며, 같은 모양이라고 자료형과 장치가 맞는 것도 아니다.

`.to(...)`는 필요한 자료형·장치 변환을 수행하며 이미 조건이 같으면 원래 텐서를 반환할 수 있다. 텐서에서는 반환값을 사용해야 한다. 모양 변화는 `reshape` 등의 별도 연산으로 수행한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
함수 $f(x) = x^3 - 2x^2 + x$를 생각하자. PyTorch autograd를 사용하여 $f'(3)$을 계산하라.

</div>

??? success "연습문제 1 풀이"
    ```python
    import torch

    x = torch.tensor(3.0, requires_grad=True)
    f = x**3 - 2*x**2 + x
    f.backward()
    print(x.grad)  # f'(x) = 3x^2 - 4x + 1 = 27 - 12 + 1 = 16.0
    ```

---


<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
제곱합처럼 저장된 중간값이 필요한 로스에서 `retain_graph=True` 없이 같은 그래프에 `.backward()`를 두 번 호출하면 오류가 나는 이유를 설명하라. `retain_graph=True`는 메모리 사용량에 어떤 영향을 주는가?

</div>

??? success "연습문제 2 풀이"
    기본 역전파는 사용을 마친 저장 텐서를 해제한다. 같은 그래프의 두 번째 역전파에 이 값들이 필요하면 `RuntimeError`가 발생한다. 저장 텐서가 필요 없는 단순 연산은 반복 호출이 가능할 수도 있으므로 모든 그래프에서 반드시 오류가 나는 것은 아니다. 첫 호출의 `retain_graph=True`는 필요한 중간 결과를 보존해 재사용을 가능하게 하지만 그만큼 메모리를 더 오래 사용한다. `.grad` 초기화는 이 보존 여부와 별개의 작업이다.

---


<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
잎 텐서 `w`를 만들고, 반복마다 같은 식으로 로스를 새로 계산하라. 경사를 초기화하지 않고 `.backward()`를 세 번 호출하며 매번 `w.grad`를 출력하라. 관찰된 값을 설명하라.

</div>

??? success "연습문제 3 풀이"
    ```python
    import torch

    w = torch.tensor(2.0, requires_grad=True)
    for i in range(3):
        loss = (w ** 2).sum()
        loss.backward()
        print(f'After backward {i+1}: w.grad = {w.grad}')
    # 출력: 4.0, 8.0, 12.0
    # 경사가 누적된다. 매 backward가 기존 경사에 2*w = 4.0을 더한다.
    ```

## 정리하며

같은 숫자가 들어 있어도 모양·자료형·장치가 다르면 허용되는 연산이 달라진다. 계산 전에 이 세 속성을 확인하는 습관을 익힌다.
