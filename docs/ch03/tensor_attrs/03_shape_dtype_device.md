# 모양, 자료형, 장치

이 스크립트는 텐서의 모양, 자료형, 장치을 보여준다. 이 개념들을 이해하는 것은 효과적인 PyTorch 프로그래밍과 딥러닝 모델 개발에 필수적이다.

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

## 2. 논의

이 코드는 `requires_grad=True`인 텐서에 대한 연산을 자동으로 추적하는 PyTorch의 autograd 체계를 보여준다. 스칼라 손실에 `.backward()`를 호출하면 autograd가 계산 그래프를 역방향으로 훑으며 연쇄 법칙을 적용해 모든 잎 텐서의 경사를 계산한다. 이 구조가 PyTorch의 모든 신경망 학습을 떠받친다.

여기서 보여준 패턴들은 실무적인 PyTorch 개발의 토대이다. 각 개념은 데이터 표현, 자동 미분, 하드웨어 가속을 하나의 일관된 API로 통합하는 텐서 추상화 위에 세워진다.

## 연습문제

**연습문제 1.**
함수 $f(x) = x^3 - 2x^2 + x$를 생각하자. PyTorch autograd를 사용하여 $f'(3)$을 계산하라.

??? success "연습문제 1 풀이"
    ```python
    import torch

    x = torch.tensor(3.0, requires_grad=True)
    f = x**3 - 2*x**2 + x
    f.backward()
    print(x.grad)  # f'(x) = 3x^2 - 4x + 1 = 27 - 12 + 1 = 16.0
    ```

---


**연습문제 2.**
`retain_graph=True` 없이 같은 계산 그래프에 `.backward()`를 두 번 호출하면 오류가 나는 이유를 설명하라. `retain_graph=True`는 메모리 사용량에 어떤 영향을 주는가?

??? success "연습문제 2 풀이"
    기본적으로 PyTorch는 메모리를 아끼기 위해 `.backward()` 후에 계산 그래프를 해제한다. `.backward()`를 두 번째로 호출하면 더 이상 존재하지 않는 그래프를 훑으려 하므로 `RuntimeError`가 발생한다. `retain_graph=True`로 두면 그래프가 메모리에 남아 재사용할 수 있지만, 모든 중간 텐서가 할당된 채로 남으므로 메모리 소비가 늘어난다.

---


**연습문제 3.**
잎 텐서 `w`를 만들고 손실을 계산한 뒤, 경사를 초기화하지 않고 `.backward()`를 세 번 호출하며 매번 `w.grad`를 출력하는 코드를 작성하라. 관찰된 값을 설명하라.

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

**다룬 것** — 모양, 자료형, 장치

이 코드는 `requires_grad=True`인 텐서에 대한 연산을 자동으로 추적하는 PyTorch의 autograd 체계를 보여준다.

앞의 연습문제 3개로 직접 확인할 수 있다.
