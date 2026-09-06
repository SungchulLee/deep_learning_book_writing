# 논리 연산 - 불리언 연산과 마스킹

이 스크립트는 논리 연산, 즉 불리언 연산과 마스킹을 보여준다. 이 개념들을 이해하는 것은 효과적인 PyTorch 프로그래밍과 딥러닝 모델 개발에 필수적이다.

## 코드

```python
"""Tutorial 13: Logical Operations - Boolean operations and masking"""
import torch

# ========================================================================
# 메인
# ========================================================================

def header(title): print(f"\n{'='*70}\n{title}\n{'='*70}")

def main():
    header("1. Comparison Operations")
    a = torch.tensor([1, 2, 3, 4, 5])
    b = torch.tensor([5, 4, 3, 2, 1])
    print(f"a = {a}\nb = {b}\n")
    print(f"a > b: {a > b}")
    print(f"a == b: {a == b}")
    print(f"torch.eq(a, b): {torch.eq(a, b)}")
    print(f"torch.gt(a, b): {torch.gt(a, b)}")
    
    header("2. Logical Operations - AND, OR, NOT")
    x = torch.tensor([True, True, False, False])
    y = torch.tensor([True, False, True, False])
    print(f"x = {x}\ny = {y}\n")
    print(f"x & y (AND): {x & y}")
    print(f"x | y (OR): {x | y}")
    print(f"~x (NOT): {~x}")
    print(f"x ^ y (XOR): {x ^ y}")
    print(f"torch.logical_and(x, y): {torch.logical_and(x, y)}")
    
    header("3. Boolean Masking")
    data = torch.tensor([10, 20, 5, 30, 15])
    print(f"Data: {data}")
    mask = data > 15
    print(f"Mask (data > 15): {mask}")
    filtered = data[mask]
    print(f"Filtered data: {filtered}")
    complex_mask = (data > 10) & (data < 25)
    print(f"Complex mask: {complex_mask}")
    print(f"Filtered: {data[complex_mask]}")
    
    header("4. Conditional Selection")
    x = torch.tensor([-2, -1, 0, 1, 2])
    print(f"x = {x}")
    result = torch.where(x > 0, x, torch.zeros_like(x))  # ReLU
    print(f"ReLU (where x>0, x, 0): {result}")
    a = torch.tensor([1, 2, 3])
    b = torch.tensor([10, 20, 30])
    condition = torch.tensor([True, False, True])
    selected = torch.where(condition, a, b)
    print(f"\nSelect from a or b: {selected}")
    
    header("5. Element-wise Comparison")
    x = torch.tensor([[1, 2], [3, 4]])
    y = torch.tensor([[2, 2], [2, 4]])
    print(f"x:\n{x}\ny:\n{y}\n")
    print(f"torch.eq(x, y):\n{torch.eq(x, y)}")
    print(f"torch.allclose(x, y): {torch.allclose(x.float(), y.float())}")
    z = torch.tensor([[1.0001, 2.0], [3.0, 4.0]])
    w = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    print(f"\nClose values? {torch.allclose(z, w, atol=1e-3)}")
    
    header("6. Practical Example: Data Cleaning")
    data = torch.tensor([1.0, 2.0, float('nan'), 4.0, float('inf')])
    print(f"Raw data: {data}")
    is_finite = torch.isfinite(data)
    print(f"is_finite: {is_finite}")
    clean_data = data[is_finite]
    print(f"Clean data: {clean_data}")
    data_with_outliers = torch.tensor([1, 2, 100, 3, 4, 200])
    mask = (data_with_outliers > 0) & (data_with_outliers < 50)
    cleaned = data_with_outliers[mask]
    print(f"\nOutlier removal: {cleaned}")

if __name__ == "__main__":
    main()```

## 논의

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
